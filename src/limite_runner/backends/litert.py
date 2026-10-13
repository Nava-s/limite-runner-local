"""LiteRT-LM backend (CPU fallback, no compilation required).

Uses the real ``litert-lm-api`` 0.17.0 Python API (Engine / Session /
SamplerConfig) against the community ``.litertlm`` bundles.

Hard constraints (from the conversion release, enforced here):
- CPU only: the tested GPU delegate rejects this graph. Never enable it.
- Synchronous decode has a fixed 10-minute deadline in LiteRT-LM 0.17.0:
  *all* generation goes through the async/streaming path.
- KV cache alone is 1.5 / 3.0 / 6.0 GiB for 16K / 32K / 64K contexts: we
  estimate RAM before loading and refuse explicitly when it does not fit.

The caller renders the chat template (model-embedded Jinja), so sessions run
with ``apply_prompt_template=False`` -- the equivalent of ``--no-template``
and stop tokens are stripped here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator

from ..artifacts import litert_bundle_path
from ..config import LITERT_KV_GIB, TOKEN_ENDOFTEXT, TOKEN_IM_END, RunConfig
from ..hw import Hardware, detect_hardware, estimate_ram_needed
from .base import Backend, BackendInfo, GenerationResult, SamplingParams, StreamChunk

log = logging.getLogger(__name__)

STOP_TOKENS: tuple[str, ...] = (TOKEN_ENDOFTEXT, TOKEN_IM_END)


class _StopFilter:
    """Incremental stop-string filter for streamed decode.

    Emits only text not yet emitted and holds back any trailing suffix that
    could still turn out to be the beginning of a stop string, so a stop
    marker split across chunks never leaks partially and the accumulated
    text is never re-emitted at the stop."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self.stops = tuple(s for s in stops if s)
        self._acc = ""
        self._emitted = 0

    def feed(self, piece: str) -> tuple[str, bool]:
        """Consume a raw chunk; return (delta_to_emit, finished)."""
        self._acc += piece
        cut, hit = len(self._acc), False
        for s in self.stops:
            i = self._acc.find(s)
            if i != -1 and i < cut:
                cut, hit = i, True
        if hit:
            # Everything before the earliest stop that was not emitted yet;
            # the stop itself and whatever follows are dropped.
            return self._acc[self._emitted:cut], True
        # No stop yet: hold back the longest suffix that is a proper prefix
        # of some stop string (it may become one once more chunks arrive).
        safe = len(self._acc)
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self._acc)), 0, -1):
                if s.startswith(self._acc[-k:]):
                    safe = min(safe, len(self._acc) - k)
                    break
        delta = self._acc[self._emitted:safe]
        self._emitted = safe
        return delta, False

    @property
    def accumulated(self) -> str:
        """Full text seen so far (emitted + held back)."""
        return self._acc

    def flush(self) -> str:
        """Remaining held-back text (generation ended without a stop string)."""
        delta = self._acc[self._emitted:]
        self._emitted = len(self._acc)
        return delta


class LiteRTBackend(Backend):
    name = "litert"

    def __init__(
        self,
        bundle_path: str | Path | None = None,
        ctx: int = 16384,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.bundle_path = Path(bundle_path) if bundle_path else None
        self.ctx = ctx
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._engine = None

    def load(self, **kwargs) -> None:
        if self._engine is not None:
            return
        if self.bundle_path is None:
            self.bundle_path = Path(
                kwargs.get("bundle_path", "")
            ) if kwargs.get("bundle_path") else None
        if kwargs.get("ctx"):
            self.ctx = int(kwargs["ctx"])
        if not self.bundle_path or not self.bundle_path.exists():
            raise RuntimeError(
                "LiteRT bundle not found. Run: limite setup --backend litert "
                f"(expected a .litertlm bundle for ctx={self.ctx})"
            )
        hw: Hardware = detect_hardware()
        need = estimate_ram_needed("litert", self.ctx)
        avail = hw.ram_available_gib
        if avail and avail < need:
            raise RuntimeError(
                f"insufficient RAM for LiteRT ctx={self.ctx}: need ~{need:.1f} GiB "
                f"(KV cache alone {LITERT_KV_GIB.get(self.ctx, '?')} GiB), "
                f"available {avail:.1f} GiB."
            )
        try:
            from litert_lm import Engine, LogSeverity, interfaces
            from litert_lm import set_min_log_severity as _set_min_log_severity

            # Benign startup noise (NPU accelerator not registered, optional
            # bundle sections missing) would otherwise spam stderr at every
            # load. Errors stay visible.
            try:
                _set_min_log_severity(LogSeverity.ERROR)
            except Exception:
                pass
        except ImportError as exc:
            raise RuntimeError(
                "litert backend requires: pip install 'limite-runner[litert]' "
                "(pins litert-lm-api==0.17.0)"
            ) from exc
        log.info(
            "loading LiteRT bundle %s on CPU (max_num_tokens=%d)",
            self.bundle_path,
            self.ctx,
        )
        self._engine = Engine(
            model_path=str(self.bundle_path),
            backend=interfaces.Backend.CPU(thread_count=os.cpu_count()),
            max_num_tokens=self.ctx,
            cache_dir=self.cache_dir,
            # Exact per-session decode counters: stream_full reads them to
            # tell a token-cap end ("length") from an EOS end ("stop").
            # Without this flag get_benchmark_info fails on 0.17.0.
            enable_benchmark=True,
        )

    def _session(self, params: SamplingParams):
        from litert_lm import interfaces

        sampler = interfaces.SamplerConfig(
            temperature=params.temperature,
            top_p=params.top_p,
        )
        return self._engine.create_session(
            apply_prompt_template=False,
            sampler_config=sampler,
            max_output_tokens=params.max_tokens,
        )

    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        self.load()
        # Always via the async path: sync decode has a fixed 10-minute
        # deadline in LiteRT-LM 0.17.0 (conversion README).
        parts: list[str] = []
        reason = "stop"
        for chunk in self.stream_full(prompt, params):
            if chunk.text:
                parts.append(chunk.text)
            if chunk.finish_reason:
                reason = chunk.finish_reason
        return GenerationResult(
            text="".join(parts),
            finish_reason=reason,
            completion_tokens=0,
        )

    def _session_end_reason(self, session, filt: _StopFilter, params: SamplingParams) -> str:
        """Why did the decode loop end without a stop-string hit?

        The public iterator ends silently both on EOS and on the
        max_output_tokens cap (run_decode_async swallows the internal
        "Max number of tokens reached" error). The session's benchmark
        counters expose the exact number of decoded tokens: hitting the cap
        means "length", ending earlier means the model's own stop condition
        fired. Fallback when the counters are unavailable: re-tokenize the
        emitted text (EOS contributes no text; the cap yields ~max tokens)."""
        limit = params.max_tokens if params.max_tokens and params.max_tokens > 0 else 0
        if limit:
            try:
                n = int(session.get_benchmark_info().last_decode_token_count)
                return "length" if n >= limit else "stop"
            except Exception:
                pass
            try:
                n = len(self._engine.tokenize(filt.accumulated))
                return "length" if n >= limit else "stop"
            except Exception:
                pass
            # Neither the session counters nor the tokenizer could tell: do
            # not invent "stop" for an end nobody observed.
            return "unverified"
        # No generation cap: run_decode_async can only end on its own stop
        # condition, so the silent end is an observed stop.
        return "stop"

    def stream_full(self, prompt: str, params: SamplingParams) -> Iterator[StreamChunk]:
        self.load()
        stops = tuple(params.stop) if params.stop else STOP_TOKENS
        filt = _StopFilter(stops)
        session = self._session(params)
        try:
            session.run_prefill([prompt])
            for responses in session.run_decode_async():
                for piece in responses.texts:
                    delta, done = filt.feed(piece)
                    if delta:
                        yield StreamChunk(text=delta)
                    if done:
                        yield StreamChunk(text="", finish_reason="stop")
                        return
            # Decoding ended without any stop string (EOS or token cap):
            # release whatever the filter was still holding, then report the
            # observed end condition instead of a hardcoded "stop".
            tail = filt.flush()
            if tail:
                yield StreamChunk(text=tail)
            yield StreamChunk(
                text="", finish_reason=self._session_end_reason(session, filt, params)
            )
        finally:
            try:
                session.close()
            except Exception:
                pass

    def stream(self, prompt: str, params: SamplingParams) -> Iterator[str]:
        for chunk in self.stream_full(prompt, params):
            if chunk.text:
                yield chunk.text

    def tokenize(self, text: str) -> list[int]:
        self.load()
        return list(self._engine.tokenize(text))

    def eos_ids(self) -> list[list[int]]:
        self.load()
        try:
            return [list(e) for e in self._engine.eos_token_ids()]
        except Exception:
            return []

    def is_ready(self) -> bool:
        # The bundle being present (available) is not readiness: the CPU
        # engine must be loaded in-process.
        return self._engine is not None

    def info(self) -> BackendInfo:
        try:
            import litert_lm

            versions = {"litert-lm-api": getattr(litert_lm, "__version__", "0.17.0")}
        except ImportError as exc:
            return BackendInfo(name=self.name, available=False, error=str(exc))
        # Resolve the effective bundle path: explicit override, else the
        # default data dir (so `doctor` shows what would really be loaded).
        bundle = self.bundle_path
        if bundle is None:
            bundle = litert_bundle_path(RunConfig().resolved_data_dir(), self.ctx)
        return BackendInfo(
            name=self.name,
            available=True,
            detail=(
                f"bundle={bundle} present={bundle.exists()} ctx={self.ctx} "
                f"backend=CPU loaded={self._engine is not None}"
            ),
            versions=versions,
        )

    def close(self) -> None:
        eng, self._engine = self._engine, None
        if eng is not None:
            try:
                eng.close()
            except Exception:
                pass
