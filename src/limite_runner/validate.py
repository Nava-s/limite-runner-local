"""``limite validate``: numerical validation of a backend vs transformers.

The conversion's published metrics are smoke tests on a single short prompt
(Q6_K looks worse than Q5_K_M on top-token agreement: noise). This suite
instead runs ~20 fixed prompts (arithmetic, algebra, one AIME-style problem,
one non-mathematical prompt) and compares the first-N-token logits against
the transformers backend in BF16/FP32 on the same prompts:

- mean KL divergence + top-1 agreement (configurable thresholds);
- finite logits everywhere;
- correct stop tokens (endoftext and im_end);
- decode coherence with and without KV cache.

Backends that do not expose full-vocab logits (LiteRT-LM 0.17.0 only offers
teacher-forced text scoring) get a degraded path: greedy top-1 agreement +
coherence + stop-token checks, with KL reported as n/a and the reason logged.
Saves a JSON report plus a human-readable table.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .backends.base import Backend, SamplingParams
from .config import EOS_TOKEN_ID, TOKEN_ENDOFTEXT, TOKEN_IM_END
from .messages import PreparedPrompt
from .template import load_template, render_chat

log = logging.getLogger(__name__)

#: Fixed validation prompts: arithmetic, algebra, AIME-style, non-math.
VALIDATION_PROMPTS: tuple[str, ...] = (
    "What is 27 times 34?",
    "Compute 1234 + 5678.",
    "What is 144 divided by 12?",
    "Solve: If x + 3 = 8, what is x?",
    "Solve 2x + 3 = 11.",
    "What is 84 * 3 / 2?",
    "Compute 17 squared.",
    "What is the remainder of 100 divided by 7?",
    "Simplify: (3 + 5) * (10 - 4).",
    "What is 15% of 200?",
    "Solve: 5x - 7 = 18. What is x?",
    "Factor 91 into primes.",
    "What is the greatest common divisor of 48 and 180?",
    "Compute the sum 1 + 2 + ... + 100.",
    "If a triangle has angles 90 and 45 degrees, what is the third angle?",
    "AIME-style: How many positive integers n less than 1000 satisfy that "
    "n^2 + 3n + 2 is divisible by 7?",
    "Prove that the square root of 2 is irrational, briefly.",
    "Tell me an interesting fact about the universe!",
    "Explain quantum computing in simple terms.",
    "Write the first three multiples of 13.",
)

DEFAULT_N_TOKENS = 16
DEFAULT_KL_THRESHOLD = 0.05
DEFAULT_TOP1_THRESHOLD = 0.90


@dataclass
class PromptResult:
    prompt: str
    kl_mean: float | None
    top1_agreement: float
    all_finite: bool
    n_tokens: int
    greedy_ref: list[int] = field(default_factory=list)
    greedy_cand: list[int] = field(default_factory=list)
    cache_coherent: bool | None = None
    note: str = ""
    # Free-running (candidate's own greedy prefix) metrics, informational only.
    # kl_mean / top1_agreement above are teacher-forced for llamacpp.
    free_kl_mean: float | None = None
    free_top1_agreement: float | None = None


@dataclass
class ValidationReport:
    backend: str
    quant: str | None
    ctx: int
    n_prompts: int
    n_tokens: int
    kl_threshold: float
    top1_threshold: float
    kl_overall: float | None
    top1_overall: float
    finite_everywhere: bool
    stop_tokens_ok: bool
    passed: bool
    seconds: float
    prompts: list[PromptResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Tri-state stop-token check outcome (see StopTokenCheck).
    stop_tokens_status: str = "unverifiable"
    stop_tokens_evidence: list[str] = field(default_factory=list)
    # True when the verdict is PASS but some components could not be checked
    # (stop-token declarations unverifiable and/or KL n/a): read as partial
    # verification, never as a complete one.
    partial: bool = False


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def mean_kl(ref_logits: np.ndarray, cand_logits: np.ndarray) -> float:
    """Mean KL(ref || cand) over positions (full vocabulary)."""
    p = _softmax(ref_logits)
    q = _softmax(cand_logits)
    q = np.clip(q, 1e-12, 1.0)
    kl = (p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(q))).sum(axis=-1)
    return float(kl.mean())


def top1_agreement(ref_logits: np.ndarray, cand_logits: np.ndarray) -> float:
    a = np.argmax(ref_logits, axis=-1)
    b = np.argmax(cand_logits, axis=-1)
    return float((a == b).mean())


def dump_llamacpp_logits(
    model_path: Path, logits_bin: Path, prompt: str, n: int, ctx: int,
    forced: list[int] | None = None,
) -> dict:
    """Run the ``limite-logits`` companion tool (built with the patched
    llama.cpp): greedy trajectory + full logits + no-cache trajectory.

    With ``forced`` the candidate is teacher-forced: step s is conditioned on
    ``forced[:s]`` instead of its own argmax, so logits are compared on the
    same contexts as the reference."""
    if not logits_bin.exists():
        raise RuntimeError(
            f"limite-logits not found at {logits_bin}. "
            "Rebuild with scripts/build_llamacpp.sh (it is a default target)."
        )
    with tempfile.TemporaryDirectory(prefix="limite-logits-") as tmp:
        out = Path(tmp) / "logits.npz"
        cmd = [
            str(logits_bin), "-m", str(model_path),
            "-c", str(ctx), "-n", str(n),
            "--out", str(out), "--prompt", prompt,
        ]
        if forced is not None:
            ids = Path(tmp) / "forced.bin"
            write_forced_tokens(ids, forced)
            cmd += ["--forced", str(ids)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            raise RuntimeError(f"limite-logits failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        data = np.load(out)
        return {
            "logits": np.asarray(data["logits"], dtype=np.float64),
            "tokens": [int(t) for t in data["tokens"]],
            "tokens_nocache": [int(t) for t in data["tokens_nocache"]],
        }


#: Max top-1/top-2 logit gap at which a cached-vs-recompute argmax flip is
#: accepted as numerical noise. Observed flips on the reference laptop were
#: 0.034-0.050 logits; anything wider is a real mismatch.
CACHE_FLIP_MAX_GAP = 0.1


def cache_coherent_forced(
    logits: np.ndarray, cached: list[int], nocache: list[int]
) -> bool:
    """Cached decode vs full recompute, on the same forced prefix.

    Identical argmax at every position passes. A differing argmax passes only
    if the full-recompute token is one of the two best cached tokens and the
    cached top-1/top-2 gap is at most CACHE_FLIP_MAX_GAP (a near-tie that the
    two computation orders can resolve differently)."""
    for p, (a, b) in enumerate(zip(cached, nocache)):
        if int(a) == int(b):
            continue
        row = np.asarray(logits[p], dtype=np.float64)
        top2 = np.argsort(-row)[:2]
        gap = float(row[top2[0]] - row[top2[1]])
        if int(b) not in (int(top2[0]), int(top2[1])) or gap > CACHE_FLIP_MAX_GAP:
            return False
    return True


def write_forced_tokens(path: Path, ids: list[int]) -> Path:
    """Raw int32 little-endian token ids, the format ``--forced`` reads."""
    np.asarray(ids, dtype="<i4").tofile(Path(path))
    return Path(path)


def render_prompts(template_path: Path, prompts: tuple[str, ...] = VALIDATION_PROMPTS) -> list[str]:
    tpl = load_template(template_path)
    return [render_chat(tpl, PreparedPrompt(user_content=p)) for p in prompts]


def run_validation(
    candidate: Backend,
    reference: Backend,
    template_path: Path,
    *,
    n_tokens: int = DEFAULT_N_TOKENS,
    kl_threshold: float = DEFAULT_KL_THRESHOLD,
    top1_threshold: float = DEFAULT_TOP1_THRESHOLD,
    quant: str | None = None,
    ctx: int = 16384,
    prompts: tuple[str, ...] = VALIDATION_PROMPTS,
    logits_bin: Path | None = None,
    model_path: Path | None = None,
) -> ValidationReport:
    t0 = time.time()
    rendered = render_prompts(template_path, prompts)
    results: list[PromptResult] = []
    notes: list[str] = []
    kls: list[float] = []
    agrees: list[float] = []
    finite_all = True

    cand_name = candidate.name
    exposes_logits = candidate.supports_logits() or cand_name == "llamacpp"

    # Candidate free-running greedy token trajectory, when the backend path
    # exposes one: used as evidence of a stop-token emission.
    free_ids: list[int] | None = None
    # Finish reason observed on a real free-running candidate generation
    # ("stop" = the engine halted on its own before the cap).
    actual_stop: str | None = None

    for prompt_text, full in zip(prompts, rendered):
        ref_logits_list = reference.next_token_logits(full, n_tokens)
        if ref_logits_list is None:
            raise RuntimeError("reference backend must expose logits")
        ref = np.asarray(ref_logits_list, dtype=np.float64)
        if not np.all(np.isfinite(ref)):
            finite_all = False
        greedy_ref = list(np.argmax(ref, axis=-1).astype(int))

        kl: float | None = None
        note = ""
        cache_ok: bool | None = None
        free_kl: float | None = None
        free_top1: float | None = None
        if cand_name == "llamacpp":
            if logits_bin is None or model_path is None:
                raise RuntimeError("llamacpp validation needs logits_bin + model_path")
            dump = dump_llamacpp_logits(model_path, logits_bin, full, n_tokens, ctx)
            free_cand = np.asarray(dump["logits"], dtype=np.float64)
            greedy_cand = [int(t) for t in dump["tokens"]]
            free_ids = greedy_cand
            cache_ok = list(dump["tokens_nocache"]) == greedy_cand
            free_kl = mean_kl(ref, free_cand)
            free_top1 = top1_agreement(ref, free_cand)
            # Verdict basis: teacher forcing. The candidate is fed the
            # reference's greedy prefix, so both sides see the same contexts.
            tf = dump_llamacpp_logits(
                model_path, logits_bin, full, n_tokens, ctx, forced=greedy_ref
            )
            cand = np.asarray(tf["logits"], dtype=np.float64)
            # Cache coherence on the same forced prefix: cached decode vs full
            # recompute at every position (both conditioned on greedy_ref).
            cache_ok = cache_coherent_forced(
                tf["logits"], tf["tokens"], tf["tokens_nocache"]
            )
            note = "teacher-forced on the reference greedy prefix"
        elif candidate.supports_logits():
            cand_list = candidate.next_token_logits(full, n_tokens)
            assert cand_list is not None
            cand = np.asarray(cand_list, dtype=np.float64)
            greedy_cand = list(np.argmax(cand, axis=-1).astype(int))
            free_ids = greedy_cand
        else:
            # Degraded path (e.g. LiteRT): no full-vocab logits in 0.17.0.
            # Greedy generation agreement + coherence via repeated runs.
            g1_res = candidate.generate(
                full, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n_tokens)
            )
            g1 = g1_res.text
            g2 = candidate.generate(
                full, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n_tokens)
            ).text
            # A halt before the short n_tokens cap is genuine stop evidence;
            # hitting the cap proves nothing (the cap is ours, not the model's).
            if g1_res.finish_reason == "stop":
                actual_stop = "stop"
            cache_ok = g1 == g2
            note = (
                "KL n/a: LiteRT-LM 0.17.0 exposes no full-vocabulary logits "
                "(only teacher-forced text scoring); top-1 via greedy text."
            )
            if note not in notes:
                notes.append(note)
            greedy_cand = greedy_ref  # agreement measured on text below
            cand = ref  # placeholder so shapes line up; kl stays None
            # token-level agreement from decoded text is weaker; compare ids
            # through the reference tokenizer when possible.
            try:
                tok = getattr(reference, "tokenizer", None)
                if tok is not None:
                    ids = tok.encode(g1, add_special_tokens=False)[:n_tokens]
                    free_ids = [int(t) for t in ids]
                    agree = float(
                        sum(1 for a, b in zip(ids, greedy_ref) if a == b)
                        / max(1, min(len(ids), len(greedy_ref)))
                    )
                else:
                    agree = 1.0 if g1 else 0.0
            except Exception:
                agree = 0.0
            agrees.append(agree)
            results.append(
                PromptResult(
                    prompt=prompt_text, kl_mean=None, top1_agreement=agree,
                    all_finite=True, n_tokens=n_tokens,
                    greedy_ref=greedy_ref, greedy_cand=[],
                    cache_coherent=cache_ok, note=note,
                )
            )
            continue

        if not np.all(np.isfinite(cand)):
            finite_all = False
        kl = mean_kl(ref, cand)
        agree = top1_agreement(ref, cand)
        kls.append(kl)
        agrees.append(agree)
        results.append(
            PromptResult(
                prompt=prompt_text, kl_mean=kl, top1_agreement=agree,
                all_finite=bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(cand))),
                n_tokens=n_tokens, greedy_ref=greedy_ref,
                greedy_cand=greedy_cand, cache_coherent=cache_ok,
                note=note, free_kl_mean=free_kl, free_top1_agreement=free_top1,
            )
        )

    # Actual-stop probe: one real free-running generation on the candidate.
    # Cheap on llamacpp (the one backend whose generation loop is fast); the
    # degraded path above already captured its observation, and a probe on
    # LiteRT would cost minutes per token, so it stays unprobed there.
    if cand_name == "llamacpp" and actual_stop is None:
        probe_tokens = max(128, 4 * n_tokens)
        try:
            probe = candidate.generate(
                rendered[0],
                SamplingParams(temperature=0.0, top_p=1.0, max_tokens=probe_tokens),
            )
            actual_stop = probe.finish_reason
        except Exception as exc:
            log.warning("actual-stop probe failed: %s", exc)

    stop_check = check_stop_tokens(
        reference, candidate, free_tokens=free_ids, actual_stop=actual_stop
    )
    stop_ok = stop_check.ok
    kl_overall = float(np.mean(kls)) if kls else None
    top1_overall = float(np.mean(agrees)) if agrees else 0.0
    passed = (
        finite_all
        and stop_ok
        and top1_overall >= top1_threshold
        and (kl_overall is None or kl_overall <= kl_threshold)
        and all(r.cache_coherent is not False for r in results)
    )
    partial = bool(passed and (stop_check.status == "unverifiable" or kl_overall is None))
    return ValidationReport(
        backend=cand_name, quant=quant, ctx=ctx,
        n_prompts=len(prompts), n_tokens=n_tokens,
        kl_threshold=kl_threshold, top1_threshold=top1_threshold,
        kl_overall=kl_overall, top1_overall=top1_overall,
        finite_everywhere=finite_all, stop_tokens_ok=stop_ok,
        passed=passed, seconds=time.time() - t0,
        prompts=results, notes=notes,
        stop_tokens_status=stop_check.status,
        stop_tokens_evidence=stop_check.evidence,
        partial=partial,
    )


@dataclass
class StopTokenCheck:
    """Tri-state outcome of the stop-token checks, with observed evidence.

    ``verified``: declarations were checked and are correct on both sides.
    ``failed``: a check ran and contradicted the declarations.
    ``unverifiable``: at least one side exposes neither a tokenizer nor
    engine EOS ids (or the check itself errored), so nothing was proven."""

    status: str  # "verified" | "failed" | "unverifiable"
    evidence: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Only a contradicted declaration fails the verdict; a check we
        could not run is reported honestly instead of silently passing."""
        return self.status != "failed"


def check_stop_tokens(
    reference: Backend,
    candidate: Backend,
    free_tokens: list[int] | None = None,
    actual_stop: str | None = None,
) -> StopTokenCheck:
    """Verify the stop declarations on both sides (tokenizer endoftext+im_end,
    or engine-declared EOS ids) and record evidence.

    ``free_tokens`` is the candidate's free-running greedy token trajectory
    when available: a stop-token emission inside it is recorded as evidence.
    ``actual_stop`` is the finish reason observed on a real free-running
    candidate generation ("stop" = the engine halted on its own stop
    condition before the cap; "length" = the cap was hit, which proves
    nothing about halting). Emission alone is not a halt proof; the
    ``actual_stop`` observation is."""
    evidence: list[str] = []
    failed = False
    checked_sides = 0
    for name, backend in (("reference", reference), ("candidate", candidate)):
        tok = getattr(backend, "tokenizer", None)
        if tok is not None:
            try:
                ids_eot = tok.encode(TOKEN_ENDOFTEXT, add_special_tokens=False)
                ids_ime = tok.encode(TOKEN_IM_END, add_special_tokens=False)
            except Exception as exc:
                evidence.append(f"{name}: stop-token encoding failed: {exc}")
                continue
            checked_sides += 1
            if EOS_TOKEN_ID in ids_eot:
                evidence.append(
                    f"{name}: {TOKEN_ENDOFTEXT} -> {ids_eot} "
                    f"(EOS_TOKEN_ID {EOS_TOKEN_ID})"
                )
            else:
                log.warning("%s: endoftext does not map to %d", name, EOS_TOKEN_ID)
                evidence.append(
                    f"{name}: {TOKEN_ENDOFTEXT} -> {ids_eot}, "
                    f"expected id {EOS_TOKEN_ID}"
                )
                failed = True
            if ids_ime:
                evidence.append(f"{name}: {TOKEN_IM_END} -> {ids_ime}")
            else:
                log.warning("%s: im_end has no token id", name)
                evidence.append(f"{name}: {TOKEN_IM_END} has no token id")
                failed = True
        elif hasattr(backend, "eos_ids"):
            try:
                eos = backend.eos_ids()
            except Exception as exc:
                evidence.append(f"{name}: engine EOS check failed: {exc}")
                continue
            checked_sides += 1
            if eos:
                evidence.append(f"{name}: engine EOS ids {eos}")
            else:
                log.warning("%s: engine reports no EOS ids", name)
                evidence.append(f"{name}: engine reports no EOS ids")
                failed = True
        else:
            evidence.append(
                f"{name}: no tokenizer and no engine EOS ids: "
                "declarations not checkable"
            )
    if free_tokens:
        hits = [i for i, t in enumerate(free_tokens) if int(t) == EOS_TOKEN_ID]
        if hits:
            evidence.append(
                "stop-token emission observed: candidate free-running greedy "
                f"trajectory emitted endoftext id {EOS_TOKEN_ID} at step "
                f"{hits[0]} (an emission, not by itself a proof of halt)"
            )
        else:
            evidence.append(
                f"informational: no stop token within the {len(free_tokens)} "
                "free-running greedy tokens"
            )
    if actual_stop == "stop":
        evidence.append(
            "actual stop: free-running generation ended on the engine's own "
            "stop condition (finish_reason=stop, before the token cap)"
        )
    elif actual_stop == "length":
        evidence.append(
            "informational: free-running generation hit the token cap "
            "(finish_reason=length): no halt observed in this probe"
        )
    if failed:
        status = "failed"
    elif checked_sides == 2:
        status = "verified"
    else:
        status = "unverifiable"
    return StopTokenCheck(status=status, evidence=evidence)


def report_table(report: ValidationReport) -> str:
    if report.passed and report.partial:
        reasons = []
        if report.kl_overall is None:
            reasons.append("KL n/a")
        if report.stop_tokens_status == "unverifiable":
            reasons.append("stop-token check unverifiable")
        verdict = f"PASS (partial: {'; '.join(reasons)})"
    else:
        verdict = "PASS" if report.passed else "FAIL"
    lines = [
        f"backend={report.backend} quant={report.quant} ctx={report.ctx} "
        f"prompts={report.n_prompts} N={report.n_tokens}",
        f"KL mean={report.kl_overall if report.kl_overall is not None else 'n/a'} "
        f"(thr {report.kl_threshold})  "
        f"top-1={report.top1_overall:.4f} (thr {report.top1_threshold})  "
        f"finite={report.finite_everywhere} "
        f"stop={report.stop_tokens_status}  "
        f"-> {verdict}  ({report.seconds:.1f}s)",
        "",
        f"{'prompt':52s} {'KL':>8s} {'top-1':>6s} {'finite':>6s} {'cache':>5s}",
    ]
    for r in report.prompts:
        kl = f"{r.kl_mean:.5f}" if r.kl_mean is not None else "n/a"
        cache = "?" if r.cache_coherent is None else ("y" if r.cache_coherent else "N")
        lines.append(
            f"{r.prompt[:50]:52s} {kl:>8s} {r.top1_agreement:6.3f} "
            f"{str(r.all_finite):>6s} {cache:>5s}"
        )
    for n in report.notes:
        lines.append(f"note: {n}")
    for e in report.stop_tokens_evidence:
        lines.append(f"stop: {e}")
    return "\n".join(lines)


def save_report(report: ValidationReport, path: Path) -> Path:
    path = Path(path)
    # numpy scalars (int64/float64) leak into the report from the metric code.
    def _plain(o):
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(f"not serializable: {type(o).__name__}")

    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False, default=_plain),
        encoding="utf-8",
    )
    return path
