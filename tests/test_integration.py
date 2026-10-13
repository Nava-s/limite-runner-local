"""Integration tests against real backends (skipped without artifacts).

Run after ``limite setup``: e.g. ``limite setup --backend litert`` then
``pytest tests/test_integration.py``. Selects the backend via LIMITE_BACKEND
(defaults to litert, the CPU path).
"""

import os
from pathlib import Path

import pytest

from limite_runner.artifacts import (
    chat_template_path,
    gguf_path,
    litert_bundle_path,
)
from limite_runner.backends import make_backend
from limite_runner.backends.base import SamplingParams
from limite_runner.config import DEFAULT_QUANT, RunConfig
from limite_runner.hw import detect_hardware
from limite_runner.messages import PreparedPrompt
from limite_runner.template import load_template, render_chat


def _data_dir() -> Path:
    from platformdirs import user_data_dir

    return Path(os.environ.get("LIMITE_DATA_DIR", user_data_dir("limite-runner", "limite")))


@pytest.fixture(scope="module")
def live_backend():
    name = os.environ.get("LIMITE_BACKEND", "litert")
    ddir = _data_dir()
    cfg = RunConfig(backend=name)
    if name == "litert" and not litert_bundle_path(ddir, 16384).exists():
        pytest.skip("litert bundle not downloaded (run: limite setup --backend litert)")
    if name == "llamacpp" and not gguf_path(ddir, DEFAULT_QUANT).exists():
        pytest.skip("GGUF not downloaded (run: limite setup --backend llamacpp)")
    tpl = chat_template_path(ddir)
    if not tpl.exists():
        pytest.skip("chat template missing (run: limite setup)")
    kwargs = {}
    if name == "litert":
        kwargs["bundle_path"] = litert_bundle_path(ddir, 16384)
    if name == "llamacpp":
        kwargs["model_path"] = gguf_path(ddir, DEFAULT_QUANT)
    b = make_backend(name, cfg, **kwargs)
    b.load()
    yield b, tpl
    b.close()


def test_live_single_turn_math(live_backend):
    backend, tpl_path = live_backend
    prompt = render_chat(
        load_template(tpl_path), PreparedPrompt(user_content="What is 27 times 34?")
    )
    res = backend.generate(prompt, SamplingParams(max_tokens=64))
    assert isinstance(res.text, str) and len(res.text) > 0


def test_live_streaming_yields_chunks(live_backend):
    backend, tpl_path = live_backend
    prompt = render_chat(
        load_template(tpl_path), PreparedPrompt(user_content="What is 2 + 2?")
    )
    chunks = list(backend.stream(prompt, SamplingParams(max_tokens=32)))
    assert chunks, "streaming produced no chunks"
    assert "".join(chunks).strip() != ""
