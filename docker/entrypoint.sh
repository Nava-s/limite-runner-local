#!/usr/bin/env bash
# Container entrypoint: ensure artifacts exist (verified download), then exec.
set -euo pipefail
DATA_DIR="${LIMITE_DATA_DIR:-/data}"
BACKEND="${LIMITE_BACKEND:-litert}"
QUANT="${LIMITE_QUANT:-Q8_0}"
CTX="${LIMITE_CTX:-16384}"

if [[ "${1:-serve}" == "serve" || "${1:-}" == "run" || "${1:-}" == "validate" ]]; then
  need_setup=0
  case "$BACKEND" in
    llamacpp) [[ -f "$DATA_DIR/conv/gguf/limite-1b-violetto-${QUANT}.gguf" ]] || need_setup=1 ;;
    litert)   [[ -f "$DATA_DIR/conv/litert/limite-1b-violetto-int4-ctx${CTX}.litertlm" ]] || need_setup=1 ;;
    transformers) [[ -f "$DATA_DIR/hf/paradigma-inc--limite-1b-violetto/model.safetensors" ]] || need_setup=1 ;;
  esac
  if [[ "$need_setup" == "1" ]]; then
    echo "artifacts missing in $DATA_DIR: running verified setup (backend=$BACKEND)..."
    limite setup --backend "$BACKEND" --quant "$QUANT" --ctx "$CTX" --data-dir "$DATA_DIR" --no-build
  fi
  extra=()
  [[ "$*" == *"--data-dir"* ]] || extra+=(--data-dir "$DATA_DIR")
  [[ "$*" == *"--backend"* ]]  || extra+=(--backend "$BACKEND")
  [[ "$*" == *"--quant"* ]]    || extra+=(--quant "$QUANT")
  [[ "$*" == *"--ctx"* ]]      || extra+=(--ctx "$CTX")
  exec limite "$@" "${extra[@]}"
fi

exec limite "$@"
