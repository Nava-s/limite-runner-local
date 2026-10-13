"""Single-turn message preprocessing (Limite constraints).

Limite 1B Violetto is single-turn: the canonical mathematical system prompt
plus one user message. The chat template embedded in the model raises on tools,
tool calls, non-string content, custom system prompts and unknown roles. This
module enforces the same contract *before* the template runs, with clear
errors for API clients and warnings whenever something is discarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import CANONICAL_SYSTEM_PROMPT

log = logging.getLogger(__name__)


class ToolsNotSupportedError(ValueError):
    """Raised when a request uses tools / function calling."""


class InvalidMessagesError(ValueError):
    """Raised when messages violate the single-turn contract."""


@dataclass
class PreparedPrompt:
    """Result of preprocessing: exactly one user message, history discarded."""

    user_content: str
    discarded: list[str] = field(default_factory=list)
    raw_messages: list[dict] = field(default_factory=list)

    def as_single_message(self) -> list[dict]:
        """Messages in the shape expected by the model's chat template."""
        return [{"role": "user", "content": self.user_content}]


def _has_tools(request: dict) -> bool:
    return bool(request.get("tools") or request.get("functions"))


def prepare_messages(
    messages: list[dict],
    *,
    tools: list | None = None,
    function_call: object = None,
) -> PreparedPrompt:
    """Reduce a chat transcript to the canonical single-turn prompt.

    - tools / functions / tool_calls / function_call -> ToolsNotSupportedError
    - custom system prompts -> discarded (warning), canonical one is implied
    - multi-turn history -> discarded (warning), only the last user turn kept
    - assistant / tool roles in the final turn -> rejected
    """
    if tools or function_call:
        raise ToolsNotSupportedError(
            "Limite Violetto supports text-only chat; tools are not supported."
        )

    if not messages:
        raise InvalidMessagesError("messages must contain at least one user message")

    discarded: list[str] = []
    last_user: str | None = None

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if msg.get("tool_calls") or msg.get("function_call"):
            raise ToolsNotSupportedError(
                "Limite Violetto supports text-only chat; tool calls are not supported."
            )
        if not isinstance(content, str):
            raise InvalidMessagesError(
                "Limite Violetto supports text-only chat; message content must be a string."
            )
        if role == "system":
            if content != CANONICAL_SYSTEM_PROMPT:
                discarded.append(f"custom system prompt at index {i} replaced by canonical")
            continue
        if role == "user":
            if i < len(messages) - 1:
                discarded.append(f"historical user message at index {i} dropped")
            last_user = content
        elif role == "assistant":
            discarded.append(f"assistant message at index {i} dropped (single-turn)")
        elif role == "tool":
            raise ToolsNotSupportedError(
                "Limite Violetto supports text-only chat; tools are not supported."
            )
        else:
            raise InvalidMessagesError(
                "Limite Violetto supports only user, assistant, and the exact "
                "canonical system message; other roles are not supported."
            )

    if last_user is None:
        raise InvalidMessagesError("no user message found")

    # Anything after the last user turn is dropped by construction (loop above
    # only records the last user). Keep the canonical system prompt implied.
    for note in discarded:
        log.warning("prepare_messages: %s", note)

    return PreparedPrompt(
        user_content=last_user, discarded=discarded, raw_messages=list(messages)
    )


def prepare_completion_prompt(prompt: str) -> str:
    """Raw /v1/completions passthrough (base mode): no template, no filtering."""
    if not isinstance(prompt, str):
        raise InvalidMessagesError("prompt must be a string")
    return prompt
