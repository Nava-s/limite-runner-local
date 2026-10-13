"""Artifact download + SHA256 verification + lockfile.

Files come from the pinned Hugging Face revisions only (see config.py).
Every byte is hash-checked against ``artifact_catalog()``; any mismatch is
fatal. The third-party llama.cpp patch is shown as a readable diff during
``limite setup`` and never applied silently.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx
from tqdm import tqdm

from .config import (
    CONV_REPO,
    CONV_REVISION,
    HF_MODEL_REPO,
    HF_MODEL_REVISION,
    HF_WEIGHTS,
    artifact_catalog,
    load_lockfile,
    write_lockfile,
)

log = logging.getLogger(__name__)


def hf_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(
    repo: str,
    revision: str,
    remote_path: str,
    local_path: Path,
    expected_sha256: str,
    expected_bytes: int = 0,
) -> Path:
    local_path = Path(local_path)
    if local_path.exists():
        got = sha256_file(local_path)
        if got == expected_sha256:
            log.info("cached %s (sha256 ok)", local_path)
            return local_path
        log.warning("cached %s has wrong hash, re-downloading", local_path)
        local_path.unlink()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = hf_url(repo, revision, remote_path)
    log.info("downloading %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", expected_bytes or 0))
        with open(local_path, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=local_path.name
        ) as bar:
            for chunk in r.iter_bytes(4 * 1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    got = sha256_file(local_path)
    if got != expected_sha256:
        local_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {remote_path}:\n"
            f"  expected {expected_sha256}\n"
            f"  got      {got}\n"
            "Refusing to continue."
        )
    log.info("verified %s", local_path)
    return local_path


def setup_artifacts(
    data_dir: Path,
    backend: str,
    quant: str,
    ctx: int,
) -> dict:
    """Download + verify everything ``backend`` needs. Returns lockfile data."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    catalog = artifact_catalog()
    lock = load_lockfile(data_dir / "limite.lock.json")
    done: dict = {
        "hf_model_repo": HF_MODEL_REPO,
        # Always the pin this tool is bound to, whatever backend is being set
        # up (the lockfile records the ground truth, not the setup choice).
        "hf_model_revision": HF_MODEL_REVISION,
        "conv_repo": CONV_REPO,
        "conv_revision": CONV_REVISION,
        "files": dict(lock.get("files", {})),
    }

    def need(remote: str) -> Path:
        sha, size = catalog[remote]
        local = data_dir / "conv" / remote
        download_verified(CONV_REPO, CONV_REVISION, remote, local, sha, size)
        done["files"][remote] = {"sha256": sha, "bytes": size, "local": str(local)}
        return local

    # Chat template + generation defaults are needed by every backend.
    need("upstream/chat_template.jinja")
    need("upstream/generation_config.json")

    if backend == "llamacpp":
        from .config import CONV_REVISION as _cr  # noqa: F401 (pin documented)

        need(f"gguf/limite-1b-violetto-{quant}.gguf")
        need("runtime/limite.patch")
        need("runtime/README.md")
        need("validation.json")
    elif backend == "litert":
        need(f"litert/limite-1b-violetto-int4-ctx{ctx}.litertlm")
    elif backend == "transformers":
        weights_dir = data_dir / "hf" / HF_MODEL_REPO.replace("/", "--")
        weights_dir.mkdir(parents=True, exist_ok=True)
        for fname, sha in HF_WEIGHTS.items():
            local = weights_dir / fname
            if fname == "model.safetensors":
                # 2 GB: stream + verify.
                download_verified(
                    HF_MODEL_REPO, HF_MODEL_REVISION, fname, local, sha,
                    expected_bytes=2070811520,
                )
            else:
                # Small text/tokenizer files: same verified path (no SDK needed).
                if local.exists() and sha256_file(local) == sha:
                    log.info("cached %s (sha256 ok)", local)
                else:
                    download_verified(HF_MODEL_REPO, HF_MODEL_REVISION, fname, local, sha)
            done["files"][f"hf:{fname}"] = {"sha256": sha, "local": str(local)}
    else:
        raise ValueError(f"unknown backend: {backend}")

    write_lockfile(data_dir / "limite.lock.json", done)
    return done


def show_patch_diff(data_dir: Path, llama_cpp_dir: Path) -> str:
    """Render the third-party patch as a readable diff (never applied silently)."""
    patch = Path(data_dir) / "conv" / "runtime" / "limite.patch"
    if not patch.exists():
        return "(patch not downloaded yet)"
    header = [
        "THIRD-PARTY CODE: community llama.cpp Limite adapter",
        f"patch file : {patch}",
        "upstream   : ggml-org/llama.cpp @ 5836771 (pinned)",
        "license    : adapter derived from Paradigma's Apache-2.0 reference graph",
        "",
        "The patch adds src/models/limite.cpp (LLM_ARCH_LIMITE) plus small",
        "hooks in llama-arch / llama-model / llama-context / llama-quant.",
        "Review the diff below before it is applied by scripts/build_llamacpp.sh.",
        "",
    ]
    # Run from outside any repository: inside one (the workspace is a git
    # repo), git apply --stat reported "0 files changed" for this patch.
    cwd = str(llama_cpp_dir) if Path(llama_cpp_dir).exists() else tempfile.gettempdir()
    try:
        files = subprocess.run(
            ["git", "apply", "--stat", str(patch)],
            capture_output=True, text=True, cwd=cwd,
        )
    except FileNotFoundError:
        # The runtime image ships no git (only the builder stage does):
        # the diff is informational, it must not abort `setup`.
        files = None
    stat = files.stdout if files is not None and files.returncode == 0 else "(git not available for --stat)"
    return "\n".join(header) + stat


def gguf_path(data_dir: Path, quant: str) -> Path:
    return Path(data_dir) / "conv" / "gguf" / f"limite-1b-violetto-{quant}.gguf"


def litert_bundle_path(data_dir: Path, ctx: int) -> Path:
    return (
        Path(data_dir) / "conv" / "litert"
        / f"limite-1b-violetto-int4-ctx{ctx}.litertlm"
    )


def chat_template_path(data_dir: Path) -> Path:
    return Path(data_dir) / "conv" / "upstream" / "chat_template.jinja"
