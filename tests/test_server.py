"""Server tests: OpenAI-compatible API against a fake backend.

Covers chat + streaming + raw completions + models + health, and the
mandatory 400 on tools/function-calling. The ``openai`` SDK test is
skipped when the dev extra is not installed.
"""

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from limite_runner.backends.base import (
    Backend,
    BackendInfo,
    GenerationResult,
    SamplingParams,
)
from limite_runner.config import DEFAULT_MAX_TOKENS
from limite_runner.server import create_app

from .test_template import TEMPLATE


class FakeBackend(Backend):
    name = "fake"

    def __init__(self) -> None:
        self.last_params: SamplingParams | None = None
        self.prompts: list[str] = []

    def load(self, **kwargs) -> None:
        return None

    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        assert "Please reason step by step" in prompt or "2 + 2" in prompt
        self.last_params = params
        self.prompts.append(prompt)
        return GenerationResult(
            text="\\boxed{4}", prompt_tokens=10, completion_tokens=3
        )

    def stream(self, prompt: str, params: SamplingParams) -> Iterator[str]:
        self.last_params = params
        self.prompts.append(prompt)
        yield "\\boxed"
        yield "{4}"

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, detail="fake")


@pytest.fixture()
def backend():
    return FakeBackend()


@pytest.fixture()
def client(backend):
    app = create_app(backend, template_path=str(TEMPLATE))
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_identifies_the_runner(client):
    # The Windows launcher reuses a server on the port ONLY if /health
    # identifies this runner: a raw llama-server also answers {"status":"ok"}
    # and must never be mistaken for it (it has no single-turn preprocessing).
    body = client.get("/health").json()
    assert body["service"] == "limite-runner"
    assert "version" in body


def test_health_unavailable_backend_is_503():
    class BrokenBackend(FakeBackend):
        def info(self) -> BackendInfo:
            return BackendInfo(
                name="fake", available=False, detail="fake", error="not built"
            )

    app = create_app(BrokenBackend(), template_path=str(TEMPLATE))
    r = TestClient(app).get("/health")
    # Regression: /health used to answer 200 "ok" with an unusable backend.
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable"
    assert body["error"] == "not built"


def test_health_available_but_not_ready_is_503():
    # Regression: "available" (binary/libraries present) was treated as
    # readiness; llama-server reported ok while its subprocess was down.
    class NotLoadedBackend(FakeBackend):
        def is_ready(self) -> bool:
            return False

    app = create_app(NotLoadedBackend(), template_path=str(TEMPLATE))
    r = TestClient(app).get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["ready"] is False


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "limite-1b-violetto"


def test_chat_completions_truncates_history(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [
                {"role": "system", "content": "custom"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "\\boxed{4}"
    assert body["usage"]["completion_tokens"] == 3


def test_chat_tools_rejected_with_400(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "calc"}}],
        },
    )
    assert r.status_code == 400
    assert "text-only" in r.json()["detail"]


def test_chat_tool_choice_rejected_with_400(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "auto",
        },
    )
    assert r.status_code == 400


def test_chat_streaming_sse(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    text = r.text
    assert "data: [DONE]" in text
    assert "\\boxed" in text


def test_raw_completions_no_template(client):
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 ="},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"] == "\\boxed{4}"


# -- sampling validation -------------------------------------------------

def test_completions_max_tokens_zero_rejected_with_400(client):
    # Regression: max_tokens=0 used to fall through the `or`-chain into an
    # AttributeError (CompletionRequest has no max_completion_tokens) -> 500.
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "max_tokens": 0},
    )
    assert r.status_code == 400
    assert "max_tokens" in r.json()["detail"]


def test_completions_negative_max_tokens_rejected_with_400(client):
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "max_tokens": -5},
    )
    assert r.status_code == 400


def test_chat_zero_max_completion_tokens_rejected_with_400(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "max_completion_tokens": 0,
        },
    )
    assert r.status_code == 400
    assert "max_completion_tokens" in r.json()["detail"]


def test_explicit_max_tokens_is_respected_not_fallback(client, backend):
    # 0/negative must be rejected, and any valid explicit value must win over
    # the server default instead of being swallowed by `or` fallbacks.
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "max_tokens": 9},
    )
    assert r.status_code == 200
    assert backend.last_params is not None
    assert backend.last_params.max_tokens == 9


def test_default_max_tokens_when_absent(client, backend):
    # The interface default is a real generation cap (2048), never unlimited:
    # a cap hit must surface as finish_reason "length".
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 ="},
    )
    assert r.status_code == 200
    assert backend.last_params.max_tokens == DEFAULT_MAX_TOKENS
    assert DEFAULT_MAX_TOKENS == 2048


def test_max_completion_tokens_wins_over_max_tokens(client, backend):
    # OpenAI semantics: max_completion_tokens is the current field, max_tokens
    # is deprecated. When both are sent, the current one must take precedence.
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "max_tokens": 5,
            "max_completion_tokens": 7,
        },
    )
    assert r.status_code == 200
    assert backend.last_params.max_tokens == 7


def test_max_tokens_alone_is_used(client, backend):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "max_tokens": 11,
        },
    )
    assert r.status_code == 200
    assert backend.last_params.max_tokens == 11


def test_invalid_temperature_rejected_with_400(client):
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "temperature": 2.5},
    )
    assert r.status_code == 400
    assert "temperature" in r.json()["detail"]


def test_invalid_top_p_rejected_with_400(client):
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "top_p": 0},
    )
    assert r.status_code == 400
    assert "top_p" in r.json()["detail"]


# -- finish reason --------------------------------------------------------

class _LengthBackend(FakeBackend):
    """Backend that reaches the token cap instead of emitting EOS."""

    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        self.last_params = params
        return GenerationResult(
            text="partial answer", finish_reason="length",
            prompt_tokens=10, completion_tokens=params.max_tokens,
        )

    def stream_full(self, prompt: str, params: SamplingParams):
        from limite_runner.backends.base import StreamChunk

        yield StreamChunk(text="partial ")
        yield StreamChunk(text="answer")
        yield StreamChunk(text="", finish_reason="length")


@pytest.fixture()
def length_client():
    app = create_app(_LengthBackend(), template_path=str(TEMPLATE))
    return TestClient(app)


def test_streaming_finish_reason_length(length_client):
    # Regression: the final SSE chunk always declared "stop", even when the
    # generation ended on the token cap.
    r = length_client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert '"finish_reason": "length"' in r.text
    assert '"finish_reason": "stop"' not in r.text


def test_streaming_completion_finish_reason_length(length_client):
    r = length_client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "stream": True},
    )
    assert r.status_code == 200
    assert '"finish_reason": "length"' in r.text


def test_non_streaming_finish_reason_passed_through(length_client):
    r = length_client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_streaming_default_wrapper_reports_unverified(client):
    # Regression: the fallback for backends that implement only text
    # streaming fabricated finish_reason="stop". The plain-text contract
    # cannot tell how the generation ended, so it must report "unverified" —
    # never present an unverified answer as concluded.
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "stream": True},
    )
    assert r.status_code == 200
    assert '"finish_reason": "unverified"' in r.text
    assert '"finish_reason": "stop"' not in r.text


def test_streaming_without_terminal_status_is_unverified(client, backend):
    # A backend stream that ends without any finish status must not be
    # concluded by the server's fallback either.
    from limite_runner.backends.base import StreamChunk

    def no_terminal(prompt, params):
        yield StreamChunk(text="partial")

    backend.stream_full = no_terminal  # type: ignore[method-assign]
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert '"finish_reason": "unverified"' in r.text
    assert '"finish_reason": "stop"' not in r.text
    assert '"error"' not in r.text


# -- single-turn prompt contract ------------------------------------------

def test_two_consecutive_questions_second_prompt_only(client, backend):
    # Two independent questions in a row: the prompt of the second generation
    # must contain ONLY the second question (no history, no previous turn).
    q1 = "What is 27 times 34?"
    q2 = "What is the derivative of x^2?"
    for q in (q1, q2):
        r = client.post(
            "/v1/chat/completions",
            json={"model": "limite-1b-violetto", "messages": [{"role": "user", "content": q}]},
        )
        assert r.status_code == 200
    assert len(backend.prompts) == 2
    second = backend.prompts[1]
    assert q2 in second
    assert q1 not in second


def test_history_sent_by_client_is_not_resent_to_the_model(client, backend):
    # Even if a client re-sends the whole transcript (the native UI bug this
    # runner exists to prevent), the model only ever sees the last question.
    q1 = "What is 27 times 34?"
    q2 = "Compute 17 squared."
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [
                {"role": "user", "content": q1},
                {"role": "assistant", "content": "The previous answer was \\boxed{918}."},
                {"role": "user", "content": q2},
            ],
        },
    )
    assert r.status_code == 200
    prompt = backend.prompts[-1]
    assert q2 in prompt
    assert q1 not in prompt
    assert "918" not in prompt


# -- errors are distinct from a finished answer ---------------------------

def test_error_with_empty_message_is_still_distinguishable(client, backend):
    # Bare asserts (and some protocol errors) carry an empty message: an
    # "" error event would be indistinguishable from success payloads.
    from limite_runner.backends.base import StreamChunk

    def broken(prompt, params):
        yield StreamChunk(text="partial")
        raise AssertionError()

    backend.stream_full = broken  # type: ignore[method-assign]
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert '"error": "AssertionError"' in r.text
    assert '"finish_reason": "stop"' not in r.text


def test_streaming_error_is_not_reported_as_a_finish(client, backend):
    from limite_runner.backends.base import StreamChunk

    def broken(prompt, params):
        yield StreamChunk(text="partial")
        raise RuntimeError("engine exploded")

    backend.stream_full = broken  # type: ignore[method-assign]
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert '"error"' in r.text
    assert "engine exploded" in r.text
    # An errored stream must not claim a conclusion.
    assert '"finish_reason": "stop"' not in r.text


def test_error_during_streaming_completion_is_distinct(client, backend):
    from limite_runner.backends.base import StreamChunk

    def broken(prompt, params):
        yield StreamChunk(text="x")
        raise RuntimeError("boom")

    backend.stream_full = broken  # type: ignore[method-assign]
    r = client.post(
        "/v1/completions",
        json={"model": "limite-1b-violetto", "prompt": "2 + 2 =", "stream": True},
    )
    assert '"error"' in r.text
    assert '"finish_reason": "stop"' not in r.text


# -- paths with spaces -----------------------------------------------------

def test_template_path_with_spaces_works(tmp_path, backend):
    # The reference repo lives in a directory with spaces; every resource
    # path must survive quoting.
    import shutil

    spaced = tmp_path / "cartella con spazi"
    spaced.mkdir()
    shutil.copy(TEMPLATE, spaced / "chat_template.jinja")
    app = create_app(
        FakeBackend(), template_path=str(spaced / "chat_template.jinja")
    )
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "limite-1b-violetto",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "\\boxed{4}"


async def test_openai_sdk_compat():
    openai = pytest.importorskip("openai")
    import httpx

    app = create_app(FakeBackend(), template_path=str(TEMPLATE))
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    sdk = openai.AsyncOpenAI(
        base_url="http://test/v1", api_key="x", http_client=http_client
    )
    resp = await sdk.chat.completions.create(
        model="limite-1b-violetto",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    assert resp.choices[0].message.content == "\\boxed{4}"
    stream = await sdk.chat.completions.create(
        model="limite-1b-violetto",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        stream=True,
    )
    chunks = [c.choices[0].delta.content or "" async for c in stream]
    assert "".join(chunks) == "\\boxed{4}"
    await http_client.aclose()
