"""llama.cpp backend (primary): patched llama-server managed as a subprocess.

The Limite architecture exists only in the community-patched build
(llama.cpp @ 5836771 + runtime/limite.patch). This backend:

- locates ``llama-server`` produced by ``scripts/build_llamacpp.sh``
  (or ``LIMITE_LLAMA_SERVER`` env override);
- always passes ``--no-context-shift`` (context shifting is not a validated
  feature of the adapter);
- sets ``-ngl`` automatically: full offload when a CUDA/Metal GPU is present,
  0 on CPU-only machines (the conversion certifies both paths);
- speaks the raw ``/completion`` endpoint with the already-rendered prompt,
  so the model's chat template is applied exactly once by the caller.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import httpx

from ..config import TOKEN_ENDOFTEXT, TOKEN_IM_END
from ..hw import Hardware, detect_hardware
from .base import Backend, BackendInfo, GenerationResult, SamplingParams, StreamChunk

log = logging.getLogger(__name__)

STOP_TOKENS: tuple[str, ...] = (TOKEN_ENDOFTEXT, TOKEN_IM_END)

_CANDIDATE_BINS = (
    "tools/llama.cpp/build/bin/llama-server",
    "build/bin/llama-server",
)


def find_llama_server(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get("LIMITE_LLAMA_SERVER")
    if env and Path(env).exists():
        return Path(env)
    for cand in _CANDIDATE_BINS:
        p = Path(cand)
        if p.exists():
            return p
    found = shutil.which("llama-server")
    return Path(found) if found else None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _finish_reason(evt: dict) -> str:
    """Map llama.cpp's end condition to the OpenAI finish reason.

    Field semantics verified against the pinned revision (``LLAMACPP_COMMIT``,
    ``tools/server/server-task.cpp`` builds the final event, and
    ``server-context.cpp`` sets the flags):

    - the generation cap (``n_predict``, our ``max_tokens``) ends with
      ``stop_type: "limit"`` and ``truncated: false``;
    - running out of context with ``--no-context-shift`` sets BOTH
      ``stop_type: "limit"`` and ``truncated: true`` (there ``truncated``
      means context overflow, not the generation cap);
    - EOS ends with ``stop_type: "eos"``, a stop word with ``"word"``;
      this revision emits no ``stopped_*`` booleans on the wire.

    Both limit conditions map to ``length``; everything else to ``stop``."""
    if evt.get("stop_type") == "limit" or evt.get("truncated"):
        return "length"
    return "stop"


#: Lines of llama-server output kept in memory for error reports. The full
#: stream is consumed continuously (see _drain_logs); only the tail is stored.
LOG_TAIL_LINES = 200


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(
        self,
        model_path: str | Path | None = None,
        ctx: int = 16384,
        ngl: int | None = None,
        server_bin: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.ctx = ctx
        self.ngl_arg = ngl
        self.server_bin = Path(server_bin) if server_bin else None
        self.host = host
        self.port = port or _free_port()
        self._proc: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._log_lines: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self._log_lock = threading.Lock()
        self._log_thread: threading.Thread | None = None

    # -- subprocess log consumption ---------------------------------------
    def _drain_logs(self, pipe) -> None:
        """Read llama-server's combined stdout/stderr until EOF.

        The pipe must be drained for as long as the server runs: a full pipe
        buffer blocks the subprocess (its logging writes are blocking)."""
        try:
            for line in iter(pipe.readline, ""):
                line = line.rstrip("\n")
                with self._log_lock:
                    self._log_lines.append(line)
                log.debug("llama-server: %s", line)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _log_tail(self, limit: int = 4000) -> str:
        with self._log_lock:
            return "\n".join(self._log_lines)[-limit:]

    # -- lifecycle ------------------------------------------------------
    def load(self, **kwargs) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if kwargs.get("model_path"):
            self.model_path = Path(kwargs["model_path"])
        if kwargs.get("ctx"):
            self.ctx = int(kwargs["ctx"])
        if kwargs.get("ngl") is not None:
            self.ngl_arg = int(kwargs["ngl"])
        if not self.model_path or not self.model_path.exists():
            raise RuntimeError(
                f"GGUF model not found: {self.model_path}. "
                "Run: limite setup --backend llamacpp (downloads the verified GGUF "
                "into the data dir) or set LIMITE_MODEL to your .gguf file."
            )
        server = find_llama_server(self.server_bin)
        if server is None:
            raise RuntimeError(
                "patched llama-server binary not found. Run: limite setup "
                "--backend llamacpp (builds it via scripts/build_llamacpp.sh, "
                "Linux/macOS or WSL2) or set LIMITE_LLAMA_SERVER to the binary."
            )
        hw: Hardware = detect_hardware()
        ngl = self.ngl_arg
        if ngl is None:
            ngl = 99 if (hw.gpu_name or hw.metal_available) else 0
        cmd = [
            str(server),
            "-m", str(self.model_path),
            "-c", str(self.ctx),
            "-ngl", str(ngl),
            "--no-context-shift",
            "--host", self.host,
            "--port", str(self.port),
        ]
        log.info("starting llama-server: %s", " ".join(cmd))
        self._log_lines.clear()
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Keep draining the pipe for the whole server lifetime, otherwise the
        # pipe buffer fills up and llama-server blocks on its own logging.
        self._log_thread = threading.Thread(
            target=self._drain_logs,
            args=(self._proc.stdout,),
            daemon=True,
            name="llama-server-log",
        )
        self._log_thread.start()
        base = f"http://{self.host}:{self.port}"
        self._client = httpx.Client(base_url=base, timeout=600.0)
        deadline = time.time() + 300
        last_err = ""
        try:
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited during load:\n{self._log_tail()}"
                    )
                try:
                    r = self._client.get("/health")
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("status") in ("ok", "loading model"):
                            if data.get("status") == "ok":
                                log.info(
                                    "llama-server ready (ngl=%d ctx=%d)", ngl, self.ctx
                                )
                                return
                except Exception as exc:
                    last_err = str(exc)
                time.sleep(1.0)
            raise RuntimeError(
                f"llama-server did not become ready: {last_err}\n"
                f"last output:\n{self._log_tail()}"
            )
        except Exception:
            # Every startup failure must leave no subprocess/pipe behind.
            self.close()
            raise

    def _payload(self, prompt: str, params: SamplingParams, stream: bool) -> dict:
        stops = list(params.stop) if params.stop else list(STOP_TOKENS)
        return {
            "prompt": prompt,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n_predict": params.max_tokens,
            "stop": stops,
            "stream": stream,
            "cache_prompt": True,
        }

    # -- generation -----------------------------------------------------
    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        self.load()
        assert self._client is not None
        r = self._client.post("/completion", json=self._payload(prompt, params, False))
        r.raise_for_status()
        data = r.json()
        return GenerationResult(
            text=data.get("content", ""),
            finish_reason=_finish_reason(data),
            prompt_tokens=int(data.get("tokens_evaluated", 0) or 0),
            completion_tokens=int(data.get("tokens_predicted", 0) or 0),
        )

    def stream_full(self, prompt: str, params: SamplingParams) -> Iterator[StreamChunk]:
        self.load()
        assert self._client is not None
        with self._client.stream(
            "POST", "/completion", json=self._payload(prompt, params, True)
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body:
                    continue
                try:
                    evt = json.loads(body)
                except ValueError:
                    continue
                piece = evt.get("content", "")
                if piece:
                    yield StreamChunk(text=piece)
                if evt.get("stop", False):
                    # End condition read from the pinned revision's event
                    # schema: stop_type "limit" (or context truncation) is a
                    # length end, EOS/stop-word are a stop end.
                    yield StreamChunk(text="", finish_reason=_finish_reason(evt))
                    return
            # The stream ended without the final `stop` event the pinned
            # revision always emits (connection dropped / server killed):
            # the generation is incomplete and its end condition unknown.
            # Never fabricate "stop": report an error instead.
            raise RuntimeError(
                "llama-server stream ended without a final event: the "
                "completion is incomplete and its end condition is unknown "
                "(not concluded, not a verified length end)."
            )

    def stream(self, prompt: str, params: SamplingParams) -> Iterator[str]:
        for chunk in self.stream_full(prompt, params):
            if chunk.text:
                yield chunk.text

    def is_ready(self) -> bool:
        # "available" only says the binary exists: readiness means the
        # llama-server subprocess is actually alive.
        return self._proc is not None and self._proc.poll() is None

    def info(self) -> BackendInfo:
        server = find_llama_server(self.server_bin)
        running = self._proc is not None and self._proc.poll() is None
        hw = detect_hardware()
        auto_ngl = 99 if (hw.gpu_name or hw.metal_available) else 0
        return BackendInfo(
            name=self.name,
            available=server is not None,
            detail=(
                f"server={server} model={self.model_path} ctx={self.ctx} "
                f"ngl={self.ngl_arg if self.ngl_arg is not None else f'auto({auto_ngl})'} "
                f"running={running}"
            ),
            versions={},
            error=None if server is not None else "llama-server not built",
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
        self._proc = None
        # Process death (or an already-exited process) closes the pipe: the
        # reader thread hits EOF and returns. Give it a moment, never block.
        t = self._log_thread
        self._log_thread = None
        if t is not None and t.is_alive():
            t.join(timeout=5)
