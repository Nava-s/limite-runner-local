#!/usr/bin/env bash
# Build patched llama.cpp (Limite adapter) on Linux and macOS.
#
# - Clones ggml-org/llama.cpp at the pinned commit (unless --dir exists).
# - Verifies the third-party patch SHA256, shows what it touches, then applies.
# - Builds llama-server, llama-completion, llama-imatrix, llama-quantize and
#   our companion logit-dump tool `limite-logits` (used by `limite validate`).
#
# Usage:
#   bash scripts/build_llamacpp.sh [--dir tools/llama.cpp] [--patch <file>]
#   CUDA=ON JOBS=8 bash scripts/build_llamacpp.sh   # CUDA route (Linux+NVIDIA)
#
# Windows is not a target of this script: use WSL2 or the Docker images.
set -euo pipefail

DIR="tools/llama.cpp"
PATCH=""
REVISION="58367713a6935c0810103378144008df32e3d5db"
PATCH_SHA256="0626d2e23c8eef43bee2e6e31acf574aa4757e4556fad42f146a3314cc12f418"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="$2"; shift 2 ;;
    --patch) PATCH="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PATCH" ]]; then
  # Default: patch downloaded by `limite setup` into the data dir (honoring
  # LIMITE_DATA_DIR like the CLI), else repo copy. No download from here.
  for cand in \
    "${LIMITE_DATA_DIR:-$HOME/.local/share/limite-runner}/conv/runtime/limite.patch" \
    "$(dirname "$0")/../patches/limite.patch"; do
    if [[ -f "$cand" ]]; then PATCH="$cand"; break; fi
  done
fi
if [[ -z "$PATCH" || ! -f "$PATCH" ]]; then
  echo "limite.patch not found. Run: limite setup --backend llamacpp" >&2
  exit 1
fi

echo "== verifying third-party patch sha256 =="
GOT="$(sha256sum "$PATCH" | cut -d' ' -f1)"
if [[ "$GOT" != "$PATCH_SHA256" ]]; then
  echo "SHA256 mismatch for $PATCH" >&2
  echo "  expected $PATCH_SHA256" >&2
  echo "  got      $GOT" >&2
  exit 1
fi
echo "patch sha256 OK: $GOT"

if [[ ! -d "$DIR/.git" ]]; then
  mkdir -p "$(dirname "$DIR")"
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$DIR"
  git -C "$DIR" checkout --detach "$REVISION"
fi
if [[ "$(git -C "$DIR" rev-parse HEAD)" != "$REVISION" ]]; then
  echo "The existing llama.cpp checkout is not at the pinned revision." >&2
  echo "  expected $REVISION" >&2
  echo "  got      $(git -C "$DIR" rev-parse HEAD)" >&2
  exit 1
fi

echo "== patch contents (review before apply) =="
git -C "$DIR" apply --stat "$PATCH" || true
if git -C "$DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "Limite runtime patch is already applied."
else
  git -C "$DIR" apply --check "$PATCH"
  git -C "$DIR" apply "$PATCH"
  echo "patch applied."
fi

echo "== installing limite-logits companion tool =="
# The companion source ships both in the source tree (tools/) and as package
# data next to this script (wheel installs). No download: pick the first hit.
LOGITS_CPP=""
for cand in \
  "$(dirname "$0")/../tools/limite-logits.cpp" \
  "$(dirname "$0")/limite-logits.cpp"; do
  if [[ -f "$cand" ]]; then LOGITS_CPP="$cand"; break; fi
done
if [[ -z "$LOGITS_CPP" ]]; then
  echo "limite-logits.cpp not found next to the script or in tools/." >&2
  echo "Reinstall limite-runner (the file ships with the package)." >&2
  exit 1
fi
cp "$LOGITS_CPP" "$DIR/tools/limite-logits.cpp"
# Register the extra target without touching upstream build files permanently:
# append an include-guarded block to tools/CMakeLists.txt once.
if ! grep -q "limite-logits" "$DIR/tools/CMakeLists.txt"; then
  cat >> "$DIR/tools/CMakeLists.txt" <<'EOF'

# limite-runner: companion logit-dump tool (not upstream)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/limite-logits.cpp")
    add_executable(limite-logits limite-logits.cpp)
    # At the pinned revision the common library target is `llama-common`
    # (shared), not `common` (renamed upstream in mid-2025).
    target_link_libraries(limite-logits PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS limite-logits RUNTIME)
endif()
EOF
fi

echo "== cmake configure (GGML_CUDA=${CUDA:-OFF}) =="
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="${CUDA:-OFF}" -DLLAMA_BUILD_TESTS=OFF
cmake --build "$DIR/build" --parallel "${JOBS:-8}" \
  --target llama-completion llama-server llama-imatrix llama-quantize limite-logits

echo "== done. binaries in $DIR/build/bin/ =="
ls "$DIR/build/bin/" | grep -E "llama-(server|completion|imatrix|quantize)|limite-logits"
