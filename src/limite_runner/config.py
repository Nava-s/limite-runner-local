"""Pinned sources, artifact catalogue and runtime defaults.

Everything here comes from Phase-0 source inspection and is mirrored in
``limite.lock.json`` (written by ``limite setup``). Do not invent values:
hashes are from Tdamre's ``artifact_manifest.json`` / ``SHA256SUMS``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Pinned upstream revisions (reproducibility)
# ---------------------------------------------------------------------------

#: HF checkpoint. Moved from e078b537 (weights only, no modeling code, so
#: trust_remote_code cannot load it) to main 9f4d1567 (adds auto_map and the
#: modeling_limite.py code). model.safetensors and tokenizer.json are
#: byte-identical between the two (same LFS etag/size). See CHANGELOG 0.1.1.
HF_MODEL_REPO = "paradigma-inc/limite-1b-violetto"
HF_MODEL_REVISION = "9f4d1567975a44c35127503deadfe52d434d734f"

#: Community conversion (GGUF + LiteRT + llama.cpp patch).
CONV_REPO = "Tdamre/limite-1b-violetto-GGUF-LiteRT"
CONV_REVISION = "24cae8a67a3d4186c679678be9240507064689d2"

#: llama.cpp upstream commit the Limite patch applies to (verified: applies
#: cleanly, see CHANGELOG.md).
LLAMACPP_COMMIT = "58367713a6935c0810103378144008df32e3d5db"
LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp.git"

#: SHA256 of runtime/limite.patch (matches artifact_manifest.json).
LIMITE_PATCH_SHA256 = "0626d2e23c8eef43bee2e6e31acf574aa4757e4556fad42f146a3314cc12f418"

#: LiteRT-LM version the bundles were validated with (CPU only).
LITERT_LM_VERSION = "0.17.0"

# ---------------------------------------------------------------------------
# Generation defaults (from the checkpoint's generation_config.json)
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_CTX = 16384
CTX_CHOICES = (16384, 32768, 65536)
MAX_POSITION_EMBEDDINGS = 131072

#: Default generation cap (finish_reason "length" when reached). Never
#: unlimited: the interface must be able to tell a concluded answer from a
#: truncated one. Configurable per request (max_completion_tokens/max_tokens),
#: per server (``limite serve --max-tokens`` / ``LIMITE_MAX_TOKENS``) and in
#: the chat page.
DEFAULT_MAX_TOKENS = 2048

#: Canonical system prompt: kept verbatim inside the model's chat template;
#: the server never accepts a replacement.
CANONICAL_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)

#: Special token ids (tokenizer_config.json / generation_config.json).
TOKEN_ENDOFTEXT = "<|endoftext|>"
TOKEN_IM_END = "<|im_end|>"
TOKEN_IM_START = "<|im_start|>"
EOS_TOKEN_ID = 151643

#: Quantization ladder shipped by the conversion.
QUANTIZATIONS = ("Q2_K_L", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
DEFAULT_QUANT = "Q8_0"
#: Below this, quality on a 1B model collapses: warn loudly.
QUANT_WARN_BELOW = ("Q2_K_L", "Q3_K_M")

#: LiteRT KV-cache footprint alone, per compiled context (README of conversion).
LITERT_KV_GIB = {16384: 1.5, 32768: 3.0, 65536: 6.0}

# ---------------------------------------------------------------------------
# Artifact catalogue (Tdamre conversion; hashes from artifact_manifest.json)
# ---------------------------------------------------------------------------

_ARTIFACTS: dict[str, tuple[str, int]] = {
    # path in CONV_REPO: (sha256, bytes)
    "gguf/limite-1b-violetto-Q2_K_L.gguf": (
        "4734300b2cf6434a0d46b07f5137927feb3a274e46a55b39fcd0b8dacd3181c6",
        491085056,
    ),
    "gguf/limite-1b-violetto-Q3_K_M.gguf": (
        "8852421fc815af242e4d9aed6f121c89b133902544fa7bb04e6a4c26986283c0",
        635481856,
    ),
    "gguf/limite-1b-violetto-Q4_K_M.gguf": (
        "e306eafbaa49e03b05adb1fa21211296833f8aa6cf2f7b717ca340a5cebd5fe7",
        734191616,
    ),
    "gguf/limite-1b-violetto-Q5_K_M.gguf": (
        "f446df7bc13f1baab53d5fce6a7c45d4b79883d4685ed0b0b8ac87918f5771eb",
        825552896,
    ),
    "gguf/limite-1b-violetto-Q6_K.gguf": (
        "22187d47a59099ada2a190e574529d0ff34dd87739d5142067d3d4536304b9d6",
        922624256,
    ),
    "gguf/limite-1b-violetto-Q8_0.gguf": (
        "9b8db0d57120ab43c9e6028fd74f74d5e3d5d32aa2f80aee367f1154c9eee436",
        1173322496,
    ),
    "litert/limite-1b-violetto-int4-ctx16384.litertlm": (
        "30b991b031af0837afa3d5480f1b6fec6ecf8f78727287a627282820ba0521ae",
        840288240,
    ),
    "litert/limite-1b-violetto-int4-ctx32768.litertlm": (
        "f4b592747acf35920524303bf67f15d4eaae35305c6f8c7cf7ec28d17d7c6d24",
        857130992,
    ),
    "litert/limite-1b-violetto-int4-ctx65536.litertlm": (
        "457df4c944b4c349b89db3593f102e3c0d692d4a7893d006276f3fe7db58ecd0",
        890816496,
    ),
    "runtime/limite.patch": (
        "0626d2e23c8eef43bee2e6e31acf574aa4757e4556fad42f146a3314cc12f418",
        16843,
    ),
    "runtime/README.md": (
        "0aa4125a8a2df978acd941aed5353009c3efa7e9714f01a16a1227c16f6e5f7c",
        2231,
    ),
    "upstream/chat_template.jinja": (
        "0c4e2c299b0bd80a20f53ec55f9b7f8db418e037c378289ce707a038cd34e2de",
        1504,
    ),
    "upstream/generation_config.json": (
        "6eb00e0a6af54ce4be361ee44a4c470fca091a7006855de9571cc035d4d686a7",
        273,
    ),
    "upstream/config.json": (
        "52a6a80ffd369638f4cc522ad728735d29bcda7cce1ecb5096947e10c1ee8bc0",
        2938,
    ),
    "validation.json": (
        "38e457781b8e11e373a017d0f7f6f656f61a797e9e9eec300fc9b53c9c012a1c",
        45815,
    ),
    "SHA256SUMS": ("", 0),  # hash not self-listed; integrity via artifact_manifest
}

HF_WEIGHTS = {
    # files of HF_MODEL_REPO @ HF_MODEL_REVISION needed for the transformers
    # reference backend.
      # Hashes at HF_MODEL_REVISION (main 9f4d1567), computed from the files as
      # downloaded. config.json, generation_config.json, chat_template.jinja and
      # tokenizer_config.json differ from the conversion's copies; the
      # transformers reference does not use the conversion's template.
      "config.json": "d080b1f7a04a44c44cb0deb1d3fcb7ce99212fce5573f5e15b74b7f84b74463d",
      "generation_config.json": "f512b06d847fbceacedadb3414fa9b99f7568c98ecbe3039841ff6cdb32d5bc5",
      "chat_template.jinja": "f6bd2b6bf08be63507e9da57900f5b4572be94605074eff65423da5374c48011",
      "special_tokens_map.json": "6676f091c8bc4d1b50146427cfde92073402866b87b6e39223227931b70083e9",
      "tokenizer.json": "296e081e2f5ecf9d87814aa9b0f4b12d670ed2b2e2be6c84e01a9466c953afb7",
      "tokenizer_config.json": "7c9b02a0378ecebbf99549aea37259bc8183df58a70c3d3f564f8ea7d4144681",
    "model.safetensors": "9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5",
}


def artifact_catalog() -> dict[str, tuple[str, int]]:
    """Return {repo_path: (sha256, bytes)} for every verified artifact."""
    return {k: v for k, v in _ARTIFACTS.items() if v[0]}


@dataclass
class RunConfig:
    """Effective runtime configuration (CLI flags / config file override)."""

    backend: str | None = None  # llamacpp | litert | transformers | None=auto
    quant: str | None = None  # GGUF quantization (llamacpp); None=auto
    ctx: int = DEFAULT_CTX
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    host: str = "127.0.0.1"
    port: int = 8000
    ngl: int | None = None  # GPU layers for llamacpp; None=auto
    device: str | None = None  # transformers device; None=auto
    data_dir: Path | None = None  # artifact cache; None=platformdirs default
    model_id: str = HF_MODEL_REPO
    model_revision: str = HF_MODEL_REVISION
    extra: dict = field(default_factory=dict)

    def resolved_data_dir(self) -> Path:
        if self.data_dir is not None:
            return Path(self.data_dir)
        env = os.environ.get("LIMITE_DATA_DIR")
        if env:
            return Path(env)
        from platformdirs import user_data_dir

        return Path(user_data_dir("limite-runner", "limite"))


def load_lockfile(path: Path) -> dict:
    """Load limite.lock.json written by ``limite setup`` (empty dict if absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_lockfile(path: Path, data: dict) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
