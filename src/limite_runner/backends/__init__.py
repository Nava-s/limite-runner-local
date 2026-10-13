"""Backend factory."""

from __future__ import annotations

from ..config import RunConfig
from .base import Backend


def make_backend(name: str, cfg: RunConfig, **kwargs) -> Backend:
    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend

        return LlamaCppBackend(ctx=cfg.ctx, ngl=cfg.ngl, **kwargs)
    if name == "litert":
        from .litert import LiteRTBackend

        return LiteRTBackend(ctx=cfg.ctx, **kwargs)
    if name == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(
            model_id=cfg.model_id,
            revision=cfg.model_revision,
            device=cfg.device,
        )
    raise ValueError(f"unknown backend: {name!r} (llamacpp | litert | transformers)")


__all__ = ["Backend", "make_backend"]
