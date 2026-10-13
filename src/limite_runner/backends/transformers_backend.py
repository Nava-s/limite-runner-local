"""Transformers reference backend (numeric ground truth).

Official checkpoint via ``trust_remote_code=True`` pinned to the exact
revision the community conversion was built from, ``attn_implementation="sdpa"``,
device auto-selected (cuda > mps > cpu). Used mainly to validate the other
backends (``limite validate``).
"""

from __future__ import annotations

import logging
from typing import Iterator

from ..config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    HF_MODEL_REPO,
    HF_MODEL_REVISION,
)
from .base import Backend, BackendInfo, GenerationResult, SamplingParams, StreamChunk

log = logging.getLogger(__name__)


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(
        self,
        model_id: str = HF_MODEL_REPO,
        revision: str = HF_MODEL_REVISION,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device_arg = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.device = None

    # -- lifecycle ---------------------------------------------------------
    def load(self, **kwargs) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers backend requires: pip install 'limite-runner[transformers]'"
            ) from exc

        device = self.device_arg
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        log.info(
            "loading %s@%s on %s (trust_remote_code=True, attn=sdpa)",
            self.model_id,
            self.revision,
            device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True
        )
        # The remote Limite config class does not define use_cache under
        # transformers 5.x, but its modeling code reads it: set it explicitly.
        config = AutoConfig.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True
        )
        if getattr(config, "use_cache", None) is None:
            config.use_cache = True
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            dtype=self.dtype,
            config=config,
            attn_implementation="sdpa",
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    def _generate_ids(self, prompt: str, params: SamplingParams):
        import torch

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=params.max_tokens,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-5),
            top_p=params.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0, inputs.input_ids.shape[1] :]
        return inputs.input_ids[0], new_ids

    # -- generation --------------------------------------------------------
    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        self.load()
        prompt_ids, new_ids = self._generate_ids(prompt, params)
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            finish_reason="length" if len(new_ids) >= params.max_tokens else "stop",
            prompt_tokens=int(prompt_ids.shape[0]),
            completion_tokens=int(new_ids.shape[0]),
        )

    def stream_full(self, prompt: str, params: SamplingParams) -> Iterator[StreamChunk]:
        # Transformers has no incremental stream here; emulate with token-wise
        # decoding of one full generation (adequate for the reference backend).
        self.load()
        _, new_ids = self._generate_ids(prompt, params)
        n = len(new_ids)
        for i, tid in enumerate(new_ids.tolist()):
            piece = self.tokenizer.decode([tid], skip_special_tokens=True)
            if piece:
                yield StreamChunk(text=piece)
        reason = "length" if n >= params.max_tokens else "stop"
        yield StreamChunk(text="", finish_reason=reason)

    def stream(self, prompt: str, params: SamplingParams) -> Iterator[str]:
        for chunk in self.stream_full(prompt, params):
            if chunk.text:
                yield chunk.text

    # -- validation support -------------------------------------------------
    def supports_logits(self) -> bool:
        return True

    def next_token_logits(self, prompt: str, n: int) -> list[list[float]] | None:
        """Teacher-forced logits of the first n greedy/reference tokens.

        Returns an (n x vocab) list: logit vector at each of the first n
        prediction positions, computed from the logits of the prompt + the
        tokens greedily chosen at each step (so the comparison against other
        backends is done on the same token trajectory).
        """
        self.load()
        import torch

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        ids = inputs.input_ids[0].tolist()
        out_logits: list[list[float]] = []
        with torch.inference_mode():
            cur = list(ids)
            for _ in range(n):
                t = torch.tensor([cur], device=self.model.device)
                logits = self.model(input_ids=t).logits[0, -1]
                out_logits.append(logits.float().cpu().tolist())
                nxt = int(torch.argmax(logits))
                cur.append(nxt)
        return out_logits

    def logprobs_of_continuation(self, prompt: str, continuation: str) -> list[float]:
        """Per-token log-probability of a fixed continuation (teacher forcing)."""
        self.load()
        import torch

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        cont_ids = self.tokenizer.encode(continuation, return_tensors="pt").to(
            self.model.device
        )
        full = torch.cat([prompt_ids, cont_ids], dim=1)
        with torch.inference_mode():
            logits = self.model(input_ids=full).logits
        logprobs = []
        for i in range(cont_ids.shape[1]):
            pos = prompt_ids.shape[1] + i - 1
            lp = torch.log_softmax(logits[0, pos].float(), dim=-1)
            logprobs.append(float(lp[int(cont_ids[0, i])]))
        return logprobs

    def is_ready(self) -> bool:
        return self.model is not None

    def info(self) -> BackendInfo:
        try:
            import torch
            import transformers

            versions = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
        except ImportError as exc:
            return BackendInfo(
                name=self.name,
                available=False,
                error=f"not installed: {exc}",
            )
        loaded = self.model is not None
        return BackendInfo(
            name=self.name,
            available=True,
            detail=f"device={self.device or 'auto'} loaded={loaded}",
            versions=versions,
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
