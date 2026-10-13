"""Unit tests: the model's own chat template renders the canonical prompt.

Uses the exact Jinja template shipped by the checkpoint at the pinned
revision (SHA256 0c4e2c29..., same file the conversion embeds in the
GGUF/LiteRT artifacts). We never rewrite it -- these tests execute it.
"""

from pathlib import Path

import pytest

from limite_runner.messages import PreparedPrompt, prepare_messages
from limite_runner.template import load_template, render_chat

TEMPLATE = Path(__file__).parent / "data" / "chat_template.jinja"


def _render(user: str) -> str:
    tpl = load_template(TEMPLATE)
    return render_chat(tpl, PreparedPrompt(user_content=user))


def test_renders_canonical_system_prompt():
    out = _render("What is 2 + 2?")
    assert "Please reason step by step" in out
    assert "boxed" in out
    assert "What is 2 + 2?" in out
    assert out.endswith("<|im_start|>assistant\n")
    assert out.startswith("<|endoftext|><|im_start|>system\n")


def test_end_to_end_prepare_then_render():
    prepared = prepare_messages(
        [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "What is 27 times 34?"},
        ]
    )
    out = render_chat(load_template(TEMPLATE), prepared)
    assert "pirate" not in out
    assert "old question" not in out
    assert "What is 27 times 34?" in out


def test_template_itself_rejects_tools():
    import jinja2

    tpl = load_template(TEMPLATE)
    with pytest.raises(Exception, match="tools are not supported"):
        tpl.render(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function"}],
            add_generation_prompt=True,
        )


def test_template_itself_rejects_custom_system():
    tpl = load_template(TEMPLATE)
    with pytest.raises(Exception, match="fixed mathematical system prompt"):
        tpl.render(
            messages=[
                {"role": "system", "content": "custom"},
                {"role": "user", "content": "hi"},
            ],
            tools=None,
            add_generation_prompt=True,
        )
