"""Unit tests: single-turn preprocessing and tools rejection."""

import pytest

from limite_runner.config import CANONICAL_SYSTEM_PROMPT
from limite_runner.messages import (
    InvalidMessagesError,
    ToolsNotSupportedError,
    prepare_completion_prompt,
    prepare_messages,
)


def test_single_user_message_passes_through():
    p = prepare_messages([{"role": "user", "content": "What is 27 times 34?"}])
    assert p.user_content == "What is 27 times 34?"
    assert p.discarded == []


def test_history_truncated_to_last_user_message():
    p = prepare_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
    )
    assert p.user_content == "What is 2 + 2?"
    assert len(p.discarded) == 2  # historical user + assistant


def test_custom_system_prompt_discarded():
    p = prepare_messages(
        [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
    )
    assert p.user_content == "What is 2 + 2?"
    assert any("system prompt" in d for d in p.discarded)


def test_canonical_system_prompt_accepted_silently():
    p = prepare_messages(
        [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": "Solve 2x + 3 = 11."},
        ]
    )
    assert p.user_content == "Solve 2x + 3 = 11."
    assert p.discarded == []


def test_tools_rejected():
    with pytest.raises(ToolsNotSupportedError, match="text-only chat"):
        prepare_messages(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "calc"}}],
        )


def test_tool_calls_in_message_rejected():
    with pytest.raises(ToolsNotSupportedError):
        prepare_messages(
            [{"role": "user", "content": "hi", "tool_calls": [{"id": "1"}]}]
        )


def test_tool_role_rejected():
    with pytest.raises(ToolsNotSupportedError):
        prepare_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "42"},
            ]
        )


def test_unknown_role_rejected():
    with pytest.raises(InvalidMessagesError, match="only user"):
        prepare_messages([{"role": "developer", "content": "hi"}])


def test_non_string_content_rejected():
    with pytest.raises(InvalidMessagesError, match="must be a string"):
        prepare_messages([{"role": "user", "content": [{"type": "text"}]}])


def test_no_user_message_rejected():
    with pytest.raises(InvalidMessagesError):
        prepare_messages([{"role": "system", "content": CANONICAL_SYSTEM_PROMPT}])


def test_completion_passthrough():
    assert prepare_completion_prompt("The capital of France is") == (
        "The capital of France is"
    )
    with pytest.raises(InvalidMessagesError):
        prepare_completion_prompt(123)  # type: ignore[arg-type]
