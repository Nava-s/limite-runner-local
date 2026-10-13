"""Rendering of the model's embedded chat template.

The Jinja template is taken verbatim from the model artifacts
(``chat_template.jinja`` / GGUF-embedded metadata / LiteRT bundle). We never
rewrite it; this module only executes the model's own template with Jinja2,
exactly like ``tokenizer.apply_chat_template`` does on the HF side.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from .config import CANONICAL_SYSTEM_PROMPT
from .messages import PreparedPrompt

#: Keep the template's own raise_exception helpers available.
_TEMPLATE_ENV = jinja2.Environment(  # noqa: S701 - model-provided template
    autoescape=False,
    undefined=jinja2.StrictUndefined,
)


def _raise_exception(message: str):  # pragma: no cover - mirrors HF behaviour
    raise ValueError(message)


_TEMPLATE_ENV.globals["raise_exception"] = _raise_exception


def load_template(template_path: Path) -> jinja2.Template:
    text = Path(template_path).read_text(encoding="utf-8")
    return _TEMPLATE_ENV.from_string(text)


def render_chat(template: jinja2.Template, prepared: PreparedPrompt) -> str:
    """Render the canonical single-turn prompt with the model's template.

    We pass only the user message (plus ``add_generation_prompt=True``); the
    canonical system prompt is baked into the template itself.
    """
    return template.render(
        messages=prepared.as_single_message(),
        tools=None,
        add_generation_prompt=True,
    )


def render_chat_text(template_text: str, user_content: str) -> str:
    """Convenience: render from raw template text (used by tests)."""
    tpl = _TEMPLATE_ENV.from_string(template_text)
    return render_chat(
        tpl, PreparedPrompt(user_content=user_content)
    )


CANONICAL_TEMPLATE_SNIPPET = CANONICAL_SYSTEM_PROMPT
