"""Unit tests for llama.cpp finish-reason mapping and readiness (no model).

The finish-reason field semantics are verified against the pinned revision
(``LLAMACPP_COMMIT``): ``truncated`` signals context overflow, the
generation limit arrives as ``stop_type: "limit"``. Reading ``truncated``
alone mislabeled a capped generation as "stop" -- including in the
validation actual-stop probe.
"""

import subprocess
import sys

import httpx
import pytest

from limite_runner.backends.base import SamplingParams
from limite_runner.backends.llamacpp import LlamaCppBackend, _finish_reason


def test_stop_type_limit_is_length_even_with_truncated_false():
    # Reproduction: generation cap reached; at the pinned revision the event
    # carries stop_type "limit" with truncated=false (truncated means
    # context overflow).
    assert (
        _finish_reason({"stop": True, "truncated": False, "stop_type": "limit"})
        == "length"
    )


def test_context_overflow_is_length():
    # --no-context-shift context exhaustion sets both flags.
    assert (
        _finish_reason({"stop": True, "truncated": True, "stop_type": "limit"})
        == "length"
    )
    assert _finish_reason({"stop": True, "truncated": True}) == "length"


def test_eos_and_word_are_stop():
    assert (
        _finish_reason({"stop": True, "truncated": False, "stop_type": "eos"})
        == "stop"
    )
    assert (
        _finish_reason({"stop": True, "truncated": False, "stop_type": "word"})
        == "stop"
    )


def test_missing_fields_default_to_stop():
    assert _finish_reason({"stop": True}) == "stop"


def _client_with(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )


def test_generate_maps_limit_event_to_length():
    b = LlamaCppBackend(model_path="unused")
    b.load = lambda: None  # type: ignore[method-assign]

    def handler(request):
        return httpx.Response(
            200,
            json={
                "content": "text",
                "stop": True,
                "truncated": False,
                "stop_type": "limit",
                "tokens_evaluated": 5,
                "tokens_predicted": 3,
            },
        )

    b._client = _client_with(handler)
    res = b.generate("p", SamplingParams(max_tokens=3))
    assert res.finish_reason == "length"
    assert res.text == "text"


def test_generate_maps_eos_event_to_stop():
    b = LlamaCppBackend(model_path="unused")
    b.load = lambda: None  # type: ignore[method-assign]

    def handler(request):
        return httpx.Response(
            200,
            json={
                "content": "done",
                "stop": True,
                "truncated": False,
                "stop_type": "eos",
                "tokens_evaluated": 5,
                "tokens_predicted": 2,
            },
        )

    b._client = _client_with(handler)
    assert b.generate("p", SamplingParams()).finish_reason == "stop"


def test_stream_full_maps_limit_event_to_length():
    b = LlamaCppBackend(model_path="unused")
    b.load = lambda: None  # type: ignore[method-assign]
    body = (
        'data: {"content": "te", "stop": false}\n\n'
        'data: {"content": "xt", "stop": false}\n\n'
        'data: {"content": "", "stop": true, "truncated": false, '
        '"stop_type": "limit"}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    b._client = _client_with(handler)
    chunks = list(b.stream_full("p", SamplingParams(max_tokens=3)))
    assert "".join(c.text for c in chunks) == "text"
    assert chunks[-1].finish_reason == "length"


def test_is_ready_tracks_subprocess_lifecycle():
    b = LlamaCppBackend()
    assert b.is_ready() is False  # binary may exist, but nothing is running
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    b._proc = proc
    assert b.is_ready() is True
    b.close()
    assert b.is_ready() is False


def test_stream_without_final_event_is_not_concluded():
    # Regression: a stream that ended after some text but WITHOUT the final
    # `stop` event used to fall back to finish_reason="stop", presenting an
    # incomplete/unknown-end generation as concluded. It must raise instead.
    b = LlamaCppBackend(model_path="unused")
    b.load = lambda: None  # type: ignore[method-assign]
    body = (
        'data: {"content": "partial answer", "stop": false}\n\n'
        'data: {"content": " more", "stop": false}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    b._client = _client_with(handler)
    chunks = []
    with pytest.raises(RuntimeError) as exc:
        for c in b.stream_full("p", SamplingParams(max_tokens=3)):
            chunks.append(c)
    assert "without a final event" in str(exc.value)
    assert "incomplete" in str(exc.value)
    # The text produced before the break is preserved, but no chunk ever
    # claims a finish reason: nothing concluded is fabricated.
    assert "".join(c.text for c in chunks) == "partial answer more"
    assert all(c.finish_reason is None for c in chunks)
