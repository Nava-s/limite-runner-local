"""Backend interface: one abstraction over llamacpp / litert / transformers.

All backends consume an already-rendered prompt string (chat template applied
by the caller using the model's own template, or raw text in base mode) and
sampling parameters. Token-level access (for ``limite validate``) is optional
and implemented where the runtime exposes it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass
class GenerationResult:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamChunk:
    """One streamed piece; the final chunk carries ``finish_reason``.

    ``finish_reason`` is only ever an *observed* end condition:
    ``"stop"`` (the engine halted on its own), ``"length"`` (the generation
    cap or the context limit was reached), ``"unverified"`` (the backend
    cannot tell — the answer must NOT be presented as concluded)."""

    text: str
    finish_reason: str | None = None  # "stop" | "length" | "unverified"


@dataclass
class BackendInfo:
    name: str
    available: bool
    detail: str = ""
    versions: dict = field(default_factory=dict)
    error: str | None = None


class Backend(abc.ABC):
    """Common interface: load(), generate(), stream(), info()."""

    name: str = "base"

    @abc.abstractmethod
    def load(self, **kwargs) -> None:
        """Load the model (idempotent). Raises on failure."""

    @abc.abstractmethod
    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        """Synchronous full generation."""

    @abc.abstractmethod
    def stream(self, prompt: str, params: SamplingParams) -> Iterator[str]:
        """Yield text deltas as they are produced."""

    def stream_full(
        self, prompt: str, params: SamplingParams
    ) -> Iterator[StreamChunk]:
        """Stream text deltas plus the real finish reason on the last chunk.

        The default wraps ``stream()`` and can only report ``"unverified"``:
        the plain-text contract cannot tell a token-cap end from an EOS end,
        and inventing ``"stop"`` would present an unverified answer as
        concluded. Backends that observe the end condition override this."""
        for piece in self.stream(prompt, params):
            yield StreamChunk(text=piece)
        yield StreamChunk(text="", finish_reason="unverified")

    @abc.abstractmethod
    def info(self) -> BackendInfo:
        """Describe the backend state (versions, availability, errors)."""

    def is_ready(self) -> bool:
        """True when the engine is actually loaded/running and can serve.

        Distinct from ``info().available`` (binaries/libraries present): a
        backend can be available yet not ready -- e.g. llama-server built
        but its subprocess not started. The default assumes a backend used
        in-process is ready; backends with a load lifecycle override it."""
        return True

    def close(self) -> None:  # pragma: no cover - optional
        """Release resources (subprocesses, GPU memory)."""

    # -- optional, used by limite validate ---------------------------------
    def next_token_logits(self, prompt: str, n: int) -> list[list[float]] | None:
        """Logits of the first n predicted tokens, if the runtime exposes them."""
        return None

    def supports_logits(self) -> bool:
        return False
