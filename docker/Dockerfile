# limite-runner CPU image (Debian slim + Python 3.12).
# Patched llama.cpp is built CPU-only; LiteRT runs on CPU; transformers
# reference uses the CPU torch wheel. No AWS dependency anywhere.
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake g++ make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/limite-runner
COPY scripts/ scripts/
COPY tools/ tools/

# Fetch the pinned third-party patch straight from HF and verify its SHA256.
ARG CONV_REV=24cae8a67a3d4186c679678be9240507064689d2
ARG PATCH_SHA256=0626d2e23c8eef43bee2e6e31acf574aa4757e4556fad42f146a3314cc12f418
RUN mkdir -p patches \
 && curl -fsSL -o patches/limite.patch \
      https://huggingface.co/Tdamre/limite-1b-violetto-GGUF-LiteRT/resolve/${CONV_REV}/runtime/limite.patch \
 && echo "${PATCH_SHA256}  patches/limite.patch" | sha256sum -c -

RUN JOBS=$(nproc) bash scripts/build_llamacpp.sh --dir /opt/llama.cpp --patch /opt/limite-runner/patches/limite.patch

FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LIMITE_LLAMA_SERVER=/opt/llama.cpp/build/bin/llama-server \
    LIMITE_LOGITS_BIN=/opt/llama.cpp/build/bin/limite-logits \
    LIMITE_DATA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
COPY --from=builder /opt/llama.cpp/build/bin/ /opt/llama.cpp/build/bin/
WORKDIR /opt/limite-runner
COPY pyproject.toml README.md ./
COPY src/ src/
RUN uv pip install --system --no-cache ".[litert,transformers]"
COPY docker/entrypoint.sh /entrypoint.sh
# COPY does not carry the executable bit reliably (e.g. checkouts on
# case-insensitive/filesystem-mode-stripping hosts): force it.
RUN chmod 0755 /entrypoint.sh
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
