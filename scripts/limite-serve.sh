#!/usr/bin/env bash
# Start the Limite runner (llamacpp backend) for the local chat / API.
#
# Used by Avvia-Limite-GPU.cmd under WSL and directly on Linux. It only
# CHECKS requirements and starts `limite serve`: no download, no compile.
#
# Configuration (environment variables; all optional):
#   LIMITE_DATA_DIR        artifact directory
#                          default: $HOME/.local/share/limite-runner
#   LIMITE_LLAMA_SERVER    patched llama-server binary
#                          default: auto (repo build dir, then PATH)
#   LIMITE_MODEL           GGUF file
#                          default: $LIMITE_DATA_DIR/conv/gguf/limite-1b-violetto-$LIMITE_QUANT.gguf
#   LIMITE_QUANT           GGUF quantization            default: Q8_0
#   LIMITE_CTX             context size                 default: 16384
#   LIMITE_HOST            bind address                 default: 127.0.0.1
#   LIMITE_PORT            bind port                    default: 8080
#   LIMITE_GPU             1 = GPU offload (-ngl 99), 0 = CPU (-ngl 0)
#                          default: 1
#   LIMITE_MAX_TOKENS      generation cap (finish_reason "length")
#                          default: 2048
#   LIMITE_BIN             explicit path to the `limite` executable (optional)
#   LIMITE_WSL_REPO        checkout to run `uv run limite ...` from (optional;
#                          <repo>/.venv/bin/limite is used when present)
#   LIMITE_LD_LIBRARY_PATH extra library path for the CUDA build (optional;
#                          WSL2 typically needs /usr/lib/wsl/lib:/usr/local/cuda/lib64)
#
# Precedence: existing environment variable > built-in default. The Windows
# launcher adds one more level below the environment (limite-launcher.env).
set -euo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

DATA_DIR="${LIMITE_DATA_DIR:-$HOME/.local/share/limite-runner}"
QUANT="${LIMITE_QUANT:-Q8_0}"
CTX="${LIMITE_CTX:-16384}"
HOST="${LIMITE_HOST:-127.0.0.1}"
PORT="${LIMITE_PORT:-8080}"
MAXTOK="${LIMITE_MAX_TOKENS:-2048}"
GPU="${LIMITE_GPU:-1}"
MODEL="${LIMITE_MODEL:-$DATA_DIR/conv/gguf/limite-1b-violetto-$QUANT.gguf}"
TEMPLATE="$DATA_DIR/conv/upstream/chat_template.jinja"

err() { echo "ERROR: $*" >&2; }

# -- GPU switch ------------------------------------------------------------
case "$GPU" in
  1|true|yes|on)  NGL=99 ;;
  0|false|no|off) NGL=0 ;;
  *) err "LIMITE_GPU must be 1 (GPU) or 0 (CPU), got '$GPU'"; exit 1 ;;
esac

if [[ "$NGL" != "0" && -n "${LIMITE_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LIMITE_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# -- requirements (each failure explains how to fix it; nothing auto-fixes) --
if [[ ! -d "$DATA_DIR" ]]; then
  err "data directory not found: $DATA_DIR"
  echo "Run: limite setup --backend llamacpp --quant $QUANT --data-dir $DATA_DIR" >&2
  echo "(or set LIMITE_DATA_DIR to the directory holding conv/)" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  err "model not found: $MODEL"
  echo "Run: limite setup --backend llamacpp --quant $QUANT --data-dir $DATA_DIR" >&2
  echo "(or set LIMITE_MODEL to your .gguf file)" >&2
  exit 1
fi

if [[ ! -f "$TEMPLATE" ]]; then
  err "chat template not found: $TEMPLATE"
  echo "Run: limite setup --backend llamacpp --data-dir $DATA_DIR (it downloads the verified template)" >&2
  exit 1
fi

if [[ -n "${LIMITE_LLAMA_SERVER:-}" ]]; then
  if [[ ! -x "$LIMITE_LLAMA_SERVER" ]]; then
    err "LIMITE_LLAMA_SERVER is not an executable file: $LIMITE_LLAMA_SERVER"
    echo "Build the patched server with: CUDA=ON bash scripts/build_llamacpp.sh" >&2
    exit 1
  fi
  SERVER="$LIMITE_LLAMA_SERVER"
else
  SERVER=""
  for cand in \
    "${LIMITE_WSL_REPO:-}/tools/llama.cpp/build/bin/llama-server" \
    "tools/llama.cpp/build/bin/llama-server" \
    "build/bin/llama-server"; do
    if [[ -n "$cand" && -x "$cand" ]]; then SERVER="$cand"; break; fi
  done
  if [[ -z "$SERVER" ]] && command -v llama-server >/dev/null 2>&1; then
    SERVER="$(command -v llama-server)"
  fi
  if [[ -z "$SERVER" ]]; then
    err "patched llama-server binary not found"
    echo "Build it with: CUDA=ON bash scripts/build_llamacpp.sh   (CPU: without CUDA=ON)" >&2
    echo "or set LIMITE_LLAMA_SERVER to the binary path." >&2
    exit 1
  fi
fi
export LIMITE_LLAMA_SERVER="$SERVER"
export LIMITE_MODEL="$MODEL"
export LIMITE_DATA_DIR="$DATA_DIR"

# -- how to run `limite` ---------------------------------------------------
if [[ -n "${LIMITE_BIN:-}" ]]; then
  if [[ ! -x "$LIMITE_BIN" ]]; then
    err "LIMITE_BIN is not an executable file: $LIMITE_BIN"
    exit 1
  fi
  RUN=("$LIMITE_BIN")
elif [[ -n "${LIMITE_WSL_REPO:-}" ]]; then
  if [[ ! -d "$LIMITE_WSL_REPO" ]]; then
    err "LIMITE_WSL_REPO is not a directory: $LIMITE_WSL_REPO"
    exit 1
  fi
  if [[ -x "$LIMITE_WSL_REPO/.venv/bin/limite" ]]; then
    # `uv sync` layout: the project venv already has the console script.
    RUN=("$LIMITE_WSL_REPO/.venv/bin/limite")
  elif command -v uv >/dev/null 2>&1; then
    RUN=(uv run --project "$LIMITE_WSL_REPO" limite)
  else
    err "no $LIMITE_WSL_REPO/.venv/bin/limite and no uv: cannot run limite from LIMITE_WSL_REPO"
    echo "Run 'uv sync --extra litert' in $LIMITE_WSL_REPO, or install uv, or set LIMITE_BIN." >&2
    exit 1
  fi
elif command -v limite >/dev/null 2>&1; then
  RUN=(limite)
elif [[ -x "$HOME/.local/bin/limite" ]]; then
  RUN=("$HOME/.local/bin/limite")
else
  err "the 'limite' command is not available in this environment"
  echo "Install the package (from the repository checkout):" >&2
  echo "  uv sync --extra litert        # or: pip install -e ." >&2
  echo "or set LIMITE_BIN / LIMITE_WSL_REPO to the installation." >&2
  exit 1
fi

echo "== limite-runner =="
echo "data dir : $DATA_DIR"
echo "model    : $MODEL"
echo "server   : $SERVER"
echo "gpu      : $([[ "$NGL" == "0" ]] && echo "no (-ngl 0)" || echo "yes (-ngl 99)")"
echo "endpoint : http://$HOST:$PORT/  (chat UI)  -  http://$HOST:$PORT/v1 (OpenAI API)"
echo "max tok  : $MAXTOK"

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "OK: all requirements are satisfied."
  exit 0
fi

echo "Stop with Ctrl+C."

exec "${RUN[@]}" serve \
  --backend llamacpp \
  --quant "$QUANT" \
  --ctx "$CTX" \
  --host "$HOST" \
  --port "$PORT" \
  --max-tokens "$MAXTOK" \
  --ngl "$NGL" \
  --data-dir "$DATA_DIR"
