"""Typer CLI: ``limite setup | run | serve | doctor | validate``.

Configuration precedence (highest wins): CLI flag > environment variable
(``LIMITE_DATA_DIR``, ``LIMITE_MODEL``, ``LIMITE_LLAMA_SERVER``,
``LIMITE_HOST``, ``LIMITE_PORT``, ``LIMITE_MAX_TOKENS``, ``LIMITE_NGL``,
``LIMITE_BACKEND``, ``LIMITE_QUANT``, ``LIMITE_CTX``) > built-in default
(see ``config.py``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .artifacts import (
    chat_template_path,
    gguf_path,
    litert_bundle_path,
    setup_artifacts,
    show_patch_diff,
)
from .backends import make_backend
from .backends.base import SamplingParams
from .config import (
    CTX_CHOICES,
    DEFAULT_CTX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_QUANT,
    DEFAULT_TOP_P,
    DEFAULT_TEMPERATURE,
    QUANT_WARN_BELOW,
    RunConfig,
)
from .hw import (
    check_memory_budget,
    choose_backend,
    choose_quant,
    detect_hardware,
)
from .messages import PreparedPrompt, prepare_completion_prompt, prepare_messages
from .template import load_template, render_chat

app = typer.Typer(
    name="limite",
    help="Local runner for Limite 1B Violetto (llamacpp / litert / transformers).",
    no_args_is_help=True,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("limite")


def _env_str(name: str, fallback: str) -> str:
    return os.environ.get(name) or fallback


def _env_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return fallback
    try:
        return int(raw)
    except ValueError:
        raise typer.BadParameter(f"{name} must be an integer, got {raw!r}")


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise typer.BadParameter(f"{name} must be an integer, got {raw!r}")


def _err_text(exc: Exception) -> str:
    """A non-empty, distinguishable description for any exception."""
    return str(exc) or type(exc).__name__


def _finish_note(finish_reason: str) -> str:
    """Human-readable end-of-generation status.

    A truncated answer must never read as a complete one: ``length`` is
    reported explicitly, distinct from a concluded (``stop``) generation, and
    an end nobody observed is reported as ``unverified`` — never as
    concluded. Interruptions/errors are reported separately by callers."""
    if finish_reason == "length":
        return (
            "[finish_reason=length] token limit reached: the answer is "
            "truncated and may be incomplete."
        )
    if finish_reason == "unverified":
        return (
            "[finish_reason=unverified] no completion status was observed: "
            "the answer is NOT verified as complete."
        )
    return "[finish_reason=stop] answer concluded."


def _build_script() -> Path:
    """Locate ``build_llamacpp.sh`` for both install layouts.

    A wheel install ships it as package data (with ``limite-logits.cpp``
    next to it); a source checkout has it under ``scripts/``. No download
    is ever attempted: the script and its companion source are part of the
    distribution."""
    candidates = [
        Path(__file__).resolve().parent / "data" / "build_llamacpp.sh",
        Path(__file__).resolve().parent.parent.parent / "scripts" / "build_llamacpp.sh",
        Path("scripts/build_llamacpp.sh"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise typer.BadParameter(
        "build_llamacpp.sh not found in this installation. Expected it as "
        "package data (limite_runner/data/) or in scripts/ of the source "
        "checkout. Reinstall limite-runner from the repository."
    )


def _cfg(
    backend: Optional[str],
    quant: Optional[str],
    ctx: Optional[int],
    data_dir: Optional[Path],
    ngl: Optional[int],
) -> RunConfig:
    # Flag > env > auto/default (see the module docstring for the full table).
    backend = backend or os.environ.get("LIMITE_BACKEND") or None
    quant = quant or os.environ.get("LIMITE_QUANT") or None
    ctx = ctx if ctx is not None else _env_int("LIMITE_CTX", DEFAULT_CTX)
    if ctx not in CTX_CHOICES:
        raise typer.BadParameter(f"--ctx must be one of {CTX_CHOICES}")
    ngl = ngl if ngl is not None else _env_optional_int("LIMITE_NGL")
    return RunConfig(
        backend=backend, quant=quant, ctx=ctx, data_dir=data_dir, ngl=ngl
    )


@app.command()
def doctor() -> None:
    """Print detected hardware, backend availability and versions."""
    hw = detect_hardware()
    typer.echo(json.dumps(hw.summary(), indent=2))
    cfg = RunConfig()
    ddir = cfg.resolved_data_dir()
    typer.echo(f"data_dir: {ddir}")
    model_env = os.environ.get("LIMITE_MODEL")
    if model_env:
        typer.echo(
            f"LIMITE_MODEL: {model_env} "
            f"(exists={Path(model_env).exists()})"
        )
    typer.echo(
        f"chat template: {chat_template_path(ddir)} "
        f"(exists={chat_template_path(ddir).exists()}; run `limite setup` to fetch)"
    )
    for name in ("llamacpp", "litert", "transformers"):
        try:
            b = make_backend(name, cfg)
            info = b.info()
            status = "available" if info.available else f"UNAVAILABLE: {info.error}"
            typer.echo(f"- {name}: {status} | {info.detail} | {info.versions}")
        except Exception as exc:
            typer.echo(f"- {name}: ERROR: {_err_text(exc)}")


@app.command()
def setup(
    backend: str = typer.Option(
        None, "--backend", help="llamacpp | litert | transformers (default: auto)"
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", help=f"GGUF quantization (default: LIMITE_QUANT or {DEFAULT_QUANT})"
    ),
    ctx: Optional[int] = typer.Option(
        None, "--ctx", help=f"context size (default: LIMITE_CTX or {DEFAULT_CTX})"
    ),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    llama_cpp_dir: Optional[Path] = typer.Option(
        None, "--llama-cpp-dir", help="checkout dir for patched llama.cpp build"
    ),
    build: bool = typer.Option(True, "--build/--no-build", help="build llama.cpp"),
) -> None:
    """Download artifacts (SHA256-verified) and build patched llama.cpp."""
    cfg = _cfg(backend, quant, ctx, data_dir, None)
    hw = detect_hardware()
    name = choose_backend(hw, cfg)
    q = choose_quant(hw, cfg) if name == "llamacpp" else (cfg.quant or DEFAULT_QUANT)
    if q in QUANT_WARN_BELOW:
        typer.echo(
            f"WARNING: {q} is below Q4: on a 1B model quality collapses. "
            "Prefer Q8_0 (default) or at least Q5_K_M.",
            err=True,
        )
    problem = check_memory_budget(hw, name, cfg.ctx)
    if problem:
        typer.echo(f"ERROR: {problem}", err=True)
        raise typer.Exit(1)
    ddir = cfg.resolved_data_dir()
    typer.echo(f"backend={name} quant={q} ctx={cfg.ctx} data_dir={ddir}")
    done = setup_artifacts(ddir, name, q, cfg.ctx)
    typer.echo(f"artifacts verified, lockfile: {ddir / 'limite.lock.json'}")
    if name == "llamacpp":
        if not build:
            # --no-build: nothing is applied or compiled, so there is nothing
            # to confirm and no patch diff to review (it also avoids needing
            # git in minimal runtime environments, e.g. the container).
            typer.echo(
                "--no-build: patch application and build skipped. "
                "Re-run without --no-build (or scripts/build_llamacpp.sh) to build."
            )
        else:
            diff = show_patch_diff(ddir, llama_cpp_dir or Path("tools/llama.cpp"))
            typer.echo(diff)
            if not typer.confirm(
                "Apply the third-party Limite patch shown above and build llama.cpp?",
                default=True,
            ):
                typer.echo("Aborted: patch not applied.")
                raise typer.Exit(1)
            import subprocess

            script = _build_script()
            cmd = ["bash", str(script)]
            if llama_cpp_dir:
                cmd += ["--dir", str(llama_cpp_dir)]
            # The patch lives where THIS setup downloaded it: pass it (and the
            # data dir) explicitly so the build never looks in another place,
            # whatever LIMITE_DATA_DIR the parent environment carries.
            patch = ddir / "conv" / "runtime" / "limite.patch"
            if patch.exists():
                cmd += ["--patch", str(patch)]
            build_env = dict(os.environ, LIMITE_DATA_DIR=str(ddir))
            typer.echo(f"building: {' '.join(cmd)}")
            r = subprocess.run(cmd, env=build_env)
            if r.returncode != 0:
                typer.echo("Build failed. See output above.", err=True)
                raise typer.Exit(1)
            typer.echo("llama.cpp build complete.")
    typer.echo(json.dumps({k: v for k, v in done.items() if k != "files"}, indent=2))


def _load_backend(name: str, cfg: RunConfig):
    ddir = cfg.resolved_data_dir()
    if name == "llamacpp":
        # LIMITE_MODEL overrides the data-dir-derived GGUF path entirely
        # (e.g. a model stored outside the data directory).
        model_env = os.environ.get("LIMITE_MODEL")
        if model_env:
            model = Path(model_env)
        else:
            model = gguf_path(ddir, cfg.quant or DEFAULT_QUANT)
        b = make_backend(name, cfg, model_path=model)
    elif name == "litert":
        bundle = litert_bundle_path(ddir, cfg.ctx)
        b = make_backend(name, cfg, bundle_path=bundle)
    else:
        b = make_backend(name, cfg)
    b.load()
    return b


def _render_or_raw(
    cfg: RunConfig, prompt: Optional[str], no_template: bool
) -> tuple[str, bool]:
    """Return (prompt_text, is_chat). Chat mode renders the model template."""
    ddir = cfg.resolved_data_dir()
    if prompt is None:
        raise typer.BadParameter("a prompt is required")
    if no_template:
        return prepare_completion_prompt(prompt), False
    prepared = prepare_messages([{"role": "user", "content": prompt}])
    for note in prepared.discarded:
        log.warning("%s", note)
    tpl_path = chat_template_path(ddir)
    if not tpl_path.exists():
        raise RuntimeError(
            f"chat template not found: {tpl_path}. "
            f"Run: limite setup --data-dir {ddir} (downloads the verified artifacts)"
        )
    tpl = load_template(tpl_path)
    return render_chat(tpl, prepared), True


@app.command()
def run(
    prompt: str = typer.Argument(..., help="single-turn math prompt"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    quant: Optional[str] = typer.Option(None, "--quant"),
    ctx: Optional[int] = typer.Option(
        None, "--ctx", help=f"default: LIMITE_CTX or {DEFAULT_CTX}"
    ),
    stream: bool = typer.Option(False, "--stream/--no-stream"),
    no_template: bool = typer.Option(
        False, "--no-template", help="raw completion (base mode)"
    ),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, "--temperature"),
    top_p: float = typer.Option(DEFAULT_TOP_P, "--top-p"),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help=f"default: LIMITE_MAX_TOKENS or {DEFAULT_MAX_TOKENS}"
    ),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    ngl: Optional[int] = typer.Option(None, "--ngl"),
) -> None:
    """Run one prompt (single-turn) on the selected backend."""
    cfg = _cfg(backend, quant, ctx, data_dir, ngl)
    limit = (
        max_tokens
        if max_tokens is not None
        else _env_int("LIMITE_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    )
    if limit < 1:
        raise typer.BadParameter("--max-tokens must be >= 1")
    hw = detect_hardware()
    name = choose_backend(hw, cfg)
    if cfg.quant is None:
        cfg.quant = choose_quant(hw, cfg)
    if cfg.quant in QUANT_WARN_BELOW:
        typer.echo(f"WARNING: {cfg.quant} is below Q4: expect degraded quality.", err=True)
    problem = check_memory_budget(hw, name, cfg.ctx)
    if problem:
        typer.echo(f"ERROR: {problem}", err=True)
        raise typer.Exit(1)
    try:
        text, _ = _render_or_raw(cfg, prompt, no_template)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"ERROR: {_err_text(exc)}", err=True)
        raise typer.Exit(1)
    params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=limit
    )
    try:
        b = _load_backend(name, cfg)
    except Exception as exc:
        typer.echo(f"ERROR: {_err_text(exc)}", err=True)
        raise typer.Exit(1)
    try:
        if stream or name == "litert":
            # litert always streams internally (10-minute sync-decode limit)
            finish = None
            try:
                for chunk in b.stream_full(text, params):
                    if chunk.text:
                        typer.echo(chunk.text, nl=False)
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
                typer.echo()
            except KeyboardInterrupt:
                typer.echo()
                typer.echo(
                    "[interrupted] stopped by user: the output above is partial.",
                    err=True,
                )
                raise typer.Exit(130)
            typer.echo(_finish_note(finish or "unverified"), err=True)
            if not finish or finish == "unverified":
                # No observed end condition: never present it as concluded.
                raise typer.Exit(1)
        else:
            try:
                res = b.generate(text, params)
            except KeyboardInterrupt:
                typer.echo("[interrupted] stopped by user.", err=True)
                raise typer.Exit(130)
            typer.echo(res.text)
            typer.echo(_finish_note(res.finish_reason), err=True)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"[error] {_err_text(exc)}", err=True)
        raise typer.Exit(1)
    finally:
        b.close()


@app.command()
def serve(
    backend: Optional[str] = typer.Option(None, "--backend"),
    quant: Optional[str] = typer.Option(None, "--quant"),
    ctx: Optional[int] = typer.Option(
        None, "--ctx", help=f"default: LIMITE_CTX or {DEFAULT_CTX}"
    ),
    host: Optional[str] = typer.Option(
        None, "--host", help="default: LIMITE_HOST or 127.0.0.1"
    ),
    port: Optional[int] = typer.Option(
        None, "--port", help="default: LIMITE_PORT or 8000"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help=f"default: LIMITE_MAX_TOKENS or {DEFAULT_MAX_TOKENS}"
    ),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    ngl: Optional[int] = typer.Option(None, "--ngl"),
) -> None:
    """Start the OpenAI-compatible server (/v1/chat/completions, SSE, chat UI)."""
    cfg = _cfg(backend, quant, ctx, data_dir, ngl)
    host = host or _env_str("LIMITE_HOST", "127.0.0.1")
    port = port if port is not None else _env_int("LIMITE_PORT", 8000)
    limit = (
        max_tokens
        if max_tokens is not None
        else _env_int("LIMITE_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    )
    if limit < 1:
        raise typer.BadParameter("--max-tokens must be >= 1")
    hw = detect_hardware()
    name = choose_backend(hw, cfg)
    if cfg.quant is None:
        cfg.quant = choose_quant(hw, cfg)
    problem = check_memory_budget(hw, name, cfg.ctx)
    if problem:
        typer.echo(f"ERROR: {problem}", err=True)
        raise typer.Exit(1)
    try:
        b = _load_backend(name, cfg)
    except Exception as exc:
        typer.echo(f"ERROR: {_err_text(exc)}", err=True)
        raise typer.Exit(1)
    from .server import serve as _serve

    tpl = chat_template_path(cfg.resolved_data_dir())
    typer.echo(f"serving backend={name} on http://{host}:{port}/v1")
    typer.echo(f"chat UI: http://{host}:{port}/ (default max tokens: {limit})")
    try:
        _serve(b, str(tpl) if tpl.exists() else None, host, port, limit)
    finally:
        b.close()


@app.command()
def validate(
    backend: Optional[str] = typer.Option(None, "--backend"),
    quant: Optional[str] = typer.Option(None, "--quant"),
    ctx: Optional[int] = typer.Option(
        None, "--ctx", help=f"default: LIMITE_CTX or {DEFAULT_CTX}"
    ),
    n_tokens: int = typer.Option(16, "--n-tokens"),
    kl_threshold: float = typer.Option(0.05, "--kl-threshold"),
    top1_threshold: float = typer.Option(0.90, "--top1-threshold"),
    out: Path = typer.Option(Path("validation-report.json"), "--out"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    ngl: Optional[int] = typer.Option(None, "--ngl"),
) -> None:
    """Validate a backend vs the transformers BF16 reference (20 prompts)."""
    from .validate import report_table, run_validation, save_report

    cfg = _cfg(backend, quant, ctx, data_dir, ngl)
    hw = detect_hardware()
    name = choose_backend(hw, cfg)
    if cfg.quant is None:
        cfg.quant = choose_quant(hw, cfg)
    problem = check_memory_budget(hw, name, cfg.ctx)
    if problem:
        typer.echo(f"ERROR: {problem}", err=True)
        raise typer.Exit(1)
    typer.echo(f"loading candidate backend={name} ...")
    cand = _load_backend(name, cfg)
    typer.echo("loading transformers reference (BF16, pinned revision) ...")
    ref = make_backend("transformers", cfg)
    ref.load()
    logits_bin = None
    model_p = None
    if name == "llamacpp":
        from .backends.llamacpp import find_llama_server

        server = find_llama_server()
        if server is not None:
            logits_bin = server.parent / "limite-logits"
        model_p = gguf_path(cfg.resolved_data_dir(), cfg.quant or DEFAULT_QUANT)
    try:
        report = run_validation(
            cand, ref,
            chat_template_path(cfg.resolved_data_dir()),
            n_tokens=n_tokens,
            kl_threshold=kl_threshold,
            top1_threshold=top1_threshold,
            quant=cfg.quant,
            ctx=cfg.ctx,
            logits_bin=logits_bin,
            model_path=model_p,
        )
    finally:
        cand.close()
        ref.close()
    save_report(report, out)
    typer.echo(report_table(report))
    typer.echo(f"report: {out}")
    if not report.passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
