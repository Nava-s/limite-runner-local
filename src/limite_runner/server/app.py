"""OpenAI-compatible HTTP server (FastAPI).

Endpoints: ``/v1/chat/completions`` and ``/v1/completions`` (SSE streaming),
``/v1/models``, ``/health``. Works with the stock ``openai`` Python SDK.
``/`` serves the adapted llama.cpp UI; ``/simple`` serves ``static/chat.html``:
it sends only the latest user question and shows the real end state of
every answer (concluded / token cap / interrupted / error).

Single-turn enforcement (see messages.py): history and custom system prompts
are discarded with a warning; any tools/function-calling request gets a clear
400 (the model's own chat template rejects them too -- we fail fast).
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..backends.base import Backend, SamplingParams
from ..config import (
    DEFAULT_CTX,
    DEFAULT_MAX_TOKENS,
    DEFAULT_QUANT,
    DEFAULT_TOP_P,
    DEFAULT_TEMPERATURE,
)
from ..messages import (
    InvalidMessagesError,
    ToolsNotSupportedError,
    prepare_completion_prompt,
    prepare_messages,
)
from ..template import load_template, render_chat

log = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: Any
    tool_calls: Any | None = None
    function_call: Any | None = None


class ChatRequest(BaseModel):
    model: str = "limite-1b-violetto"
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    tools: list[Any] | None = None
    functions: list[Any] | None = None
    function_call: Any | None = None
    tool_choice: Any | None = None
    stop: list[str] | str | None = None


class CompletionRequest(BaseModel):
    model: str = "limite-1b-violetto"
    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None


def _resolve_max_tokens(req: ChatRequest | CompletionRequest, default_max: int) -> int:
    """Explicit precedence between the two OpenAI token limits:
    ``max_completion_tokens`` (current field) > ``max_tokens`` (deprecated) >
    server default. Either explicit value must be a positive integer."""
    if isinstance(req, ChatRequest) and req.max_completion_tokens is not None:
        chosen, source = req.max_completion_tokens, "max_completion_tokens"
    elif req.max_tokens is not None:
        chosen, source = req.max_tokens, "max_tokens"
    else:
        return default_max
    if chosen < 1:
        # 0 (or negative) used to slip through the old `or`-chain and crash
        # with a 500 on /v1/completions; reject it as a client error.
        raise HTTPException(
            status_code=400,
            detail=f"{source} must be a positive integer, got {chosen}.",
        )
    return chosen


def _sampling(req: ChatRequest | CompletionRequest, default_max: int) -> SamplingParams:
    stop = req.stop
    stops = [stop] if isinstance(stop, str) else (stop or [])
    temperature = DEFAULT_TEMPERATURE if req.temperature is None else req.temperature
    top_p = DEFAULT_TOP_P if req.top_p is None else req.top_p
    if not 0.0 <= temperature <= 2.0:
        raise HTTPException(
            status_code=400,
            detail=f"temperature must be within [0, 2], got {temperature}.",
        )
    if not 0.0 < top_p <= 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"top_p must be within (0, 1], got {top_p}.",
        )
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=_resolve_max_tokens(req, default_max),
        stop=stops,
    )


def _chat_html(default_max_tokens: int) -> str:
    """Lightweight fallback chat page (single file, no external assets).

    The page is the Limite-coherent interface: it sends only the latest user
    question to ``/v1/chat/completions`` (no history, no system prompt, no
    tools/MCP), keeps the conversation visible client-side and reports the
    real end state (concluded / token cap / interrupted / error). Served
    from package data so it works from a wheel install too."""
    path = Path(__file__).resolve().parent / "static" / "chat.html"
    text = path.read_text(encoding="utf-8")
    return text.replace("__DEFAULT_MAX_TOKENS__", str(int(default_max_tokens)))


def create_app(
    backend: Backend,
    template_path: str | None = None,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    model_name: str = "limite-1b-violetto",
) -> FastAPI:
    template = load_template(template_path) if template_path else None
    app = FastAPI(title="limite-runner", version=__version__)
    chat_html = _chat_html(default_max_tokens)

    def render_user(user_content: str) -> str:
        if template is None:
            raise HTTPException(
                500, "chat template not configured (run: limite setup first)"
            )
        from ..messages import PreparedPrompt

        return render_chat(template, PreparedPrompt(user_content=user_content))

    @app.get("/", include_in_schema=False)
    def chat_ui() -> HTMLResponse:
        # Local single-turn chat page: requests go through this server's
        # preprocessing (last user message only, tools rejected).
        path = Path(__file__).parent / "static" / "llama-ui" / "index.html"
        return HTMLResponse(path.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/simple", include_in_schema=False)
    def simple_ui():
        return HTMLResponse(chat_html, headers={"Cache-Control": "no-store"})

    @app.get("/ui/props", include_in_schema=False)
    def ui_props():
        # UI configuration, not a claim of backend readiness (see /health).
        return {
            "role": "model", "model_path": model_name, "model_alias": model_name,
            "total_slots": 1, "chat_template": "", "build_info": __version__,
            "modalities": {"vision": False, "audio": False, "video": False},
            "default_generation_settings": {"n_ctx": getattr(backend, "ctx", DEFAULT_CTX), "params": {
                "temperature": DEFAULT_TEMPERATURE, "top_p": DEFAULT_TOP_P,
                "n_predict": default_max_tokens, "max_tokens": default_max_tokens,
            }},
        }

    @app.get("/props", include_in_schema=False)
    def legacy_native_ui_hint():
        # The native llama-server web UI polls this endpoint from leftover
        # tabs. It is not supported here: answer 404 with a hint instead of
        # a silent miss (the access-log line is filtered out, see
        # _LegacyUiNoiseFilter).
        raise HTTPException(
            status_code=404,
            detail=(
                "This is limite-runner: the native llama-server UI is not "
                "supported (it re-sends history and ships tools). Open / for "
                "the supported single-turn chat."
            ),
        )

    @app.get("/health")
    def health():
        info = backend.info()
        # "service" identifies this server: clients (e.g. the Windows
        # launcher) must not mistake any other service on the same port
        # -- e.g. a raw llama-server, whose /health answers {"status":"ok"}
        # with no runner preprocessing -- for this runner.
        body = {
            "status": "ok",
            "ready": True,
            "service": "limite-runner",
            "version": __version__,
            "backend": info.name,
            "detail": info.detail,
            "versions": info.versions,
        }
        if not info.available:
            # The backend cannot work at all (binaries/libraries missing).
            body.update(status="unavailable", ready=False, error=info.error)
            return JSONResponse(status_code=503, content=body)
        if not backend.is_ready():
            # Present but the engine is not loaded/running yet (e.g.
            # llama-server binary found while its subprocess is down).
            body.update(
                status="not_ready",
                ready=False,
                error="backend present but the engine is not loaded/running",
            )
            return JSONResponse(status_code=503, content=body)
        return body

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "paradigma-inc",
                }
            ],
        }

    # -- chat completions ------------------------------------------------
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        try:
            if req.tools or req.functions or req.function_call or req.tool_choice:
                raise ToolsNotSupportedError(
                    "Limite Violetto supports text-only chat; tools are not supported."
                )
            prepared = prepare_messages(
                [m.model_dump() for m in req.messages],
                tools=req.tools,
                function_call=req.function_call,
            )
        except ToolsNotSupportedError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except InvalidMessagesError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        for note in prepared.discarded:
            log.warning("chat: discarded: %s", note)
        prompt = render_user(prepared.user_content)
        params = _sampling(req, default_max_tokens)
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:
            def gen():
                yield f"data: {json_dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
                finish = None
                try:
                    for chunk in backend.stream_full(prompt, params):
                        if chunk.text:
                            yield f"data: {json_dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': chunk.text}, 'finish_reason': None}]})}\n\n"
                        if chunk.finish_reason:
                            finish = chunk.finish_reason
                except Exception as exc:
                    yield f"data: {json_dumps({'error': _err_text(exc)})}\n\n"
                    return
                # No observed end condition -> never claim "stop": report the
                # completion as unverified instead of concluded.
                yield f"data: {json_dumps({'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish or 'unverified'}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        try:
            res = backend.generate(prompt, params)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=_err_text(exc))
        return {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": res.text},
                    "finish_reason": res.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "total_tokens": res.prompt_tokens + res.completion_tokens,
            },
        }

    # -- raw completions (base mode: no template) -------------------------
    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        try:
            prompt = prepare_completion_prompt(req.prompt)
        except InvalidMessagesError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        params = _sampling(req, default_max_tokens)
        rid = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:
            def gen():
                finish = None
                try:
                    for chunk in backend.stream_full(prompt, params):
                        if chunk.text:
                            yield f"data: {json_dumps({'id': rid, 'object': 'text_completion', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'text': chunk.text, 'finish_reason': None}]})}\n\n"
                        if chunk.finish_reason:
                            finish = chunk.finish_reason
                except Exception as exc:
                    yield f"data: {json_dumps({'error': _err_text(exc)})}\n\n"
                    return
                # Same contract as chat: unobserved end -> "unverified".
                yield f"data: {json_dumps({'id': rid, 'object': 'text_completion', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'text': '', 'finish_reason': finish or 'unverified'}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        try:
            res = backend.generate(prompt, params)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=_err_text(exc))
        return {
            "id": rid,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {"index": 0, "text": res.text, "finish_reason": res.finish_reason}
            ],
            "usage": {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "total_tokens": res.prompt_tokens + res.completion_tokens,
            },
        }

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static" / "llama-ui", check_dir=False), name="llama-ui")
    return app


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _err_text(exc: Exception) -> str:
    """A non-empty, distinguishable description for any exception.

    Some exceptions carry an empty message (bare ``assert``, some httpx
    protocol errors): reporting "" would make the error indistinguishable."""
    return str(exc) or type(exc).__name__


class _LegacyUiNoiseFilter(logging.Filter):
    """Keep the serve/launcher terminal readable: drop access-log lines for
    polling probes, not for real traffic.

    Observed noise (a leftover native llama-server UI tab and similar foreign
    clients poll these forever, burying everything under 404 lines):
    ``/props`` (tight retry loop), ``/sw.js``, ``/build.json``, ``/tools``,
    ``/v1/streams/lookup``. ``/health`` is silenced too because the chat page
    polls it every few seconds for its readiness badge. Every request is
    still answered; only the access-log line is dropped (raise the log level
    to DEBUG to see them). Real API traffic -- ``/``, ``/v1/...`` completions
    and models -- stays visible."""

    #: Poll/probe paths of unsupported interfaces (answered, not logged).
    NOISY = (
        "/props",
        "/sw.js",
        "/build.json",
        "/tools",
        "/v1/streams/lookup",
        "/health",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        return not any(path in msg for path in self.NOISY)


def serve(
    backend: Backend,
    template_path: str | None,
    host: str,
    port: int,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> None:
    import uvicorn

    app = create_app(backend, template_path, default_max_tokens)
    # Keep the serve/launcher terminal readable: a leftover native-UI tab
    # polls /props in a loop and would bury everything under 404 lines.
    logging.getLogger("uvicorn.access").addFilter(_LegacyUiNoiseFilter())
    uvicorn.run(app, host=host, port=port, log_level="info")
