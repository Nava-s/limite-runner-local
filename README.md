# limite-runner

**A community toolkit for running Limite 1B Violetto locally**, with a tested
Windows + WSL2 + NVIDIA GPU setup. It brings together verified model downloads,
a patched llama.cpp build, a command-line interface, a local single-turn chat,
and an OpenAI-compatible API.

**Status: 0.1.4 beta.** Start with the [GPU quickstart](#quickstart).
See [tested environments](#tested-environments) for the scope of verification.

## Purpose and contributions

Limite 1B Violetto is a compact mathematical reasoning model developed by
[Paradigma](https://huggingface.co/paradigma-inc/limite-1b-violetto).
Its custom architecture requires a compatible inference runtime. This project
integrates [Tdamre's GGUF conversion and llama.cpp adapter patch](https://huggingface.co/Tdamre/limite-1b-violetto-GGUF-LiteRT)
into a repeatable local setup, so users can spend less time assembling and
debugging the pieces themselves.

The contributions in this repository are:

- **Installation and configuration:** pinned revisions, SHA256-verified
  downloads, build tooling, requirement checks and a configurable launcher.
- **A model-appropriate interface:** each question is sent independently,
  tools are rejected, and completed, truncated and interrupted responses
  have distinct states.
- **Integration and verification:** a common CLI/API for llama.cpp, LiteRT
  and Transformers, regression tests, and numerical comparison tooling.

The model belongs to Paradigma; the conversion and adapter patch come from
Tdamre; the inference engine is [llama.cpp](https://github.com/ggml-org/llama.cpp).
This repository contributes the runner and integration tooling. It is a
community project and does not imply endorsement by those authors.

Paradigma also provides [official vLLM and Transformers inference paths](https://huggingface.co/paradigma-inc/limite-1b-violetto#run-with-vllm).
This toolkit offers an additional local workflow. It does not modify the
model weights or claim improved mathematical accuracy. The browser chat is
served by the runner; this project does not install a runtime into LM Studio
or Ollama.

## Quickstart

### A. Windows + WSL2 + NVIDIA GPU (tested interactive route)

Requirements: Windows 11 with WSL2 (Ubuntu 24.04 tested), an NVIDIA driver,
the CUDA toolkit installed **inside WSL** (see Troubleshooting for sm_120),
Python 3.12 and [uv](https://docs.astral.sh/uv/) in WSL. Building also needs
Git, CMake and a C/C++ build toolchain. Download or clone this repository,
then open its directory inside WSL. The commands below run there.

```bash
# 1. inside WSL, from the repository checkout
uv sync

# 2. download the verified artifacts (GGUF Q8_0 + chat template + patch)
uv run limite setup --backend llamacpp --no-build

# 3. build the patched llama.cpp once (needs the CUDA toolkit in WSL;
#    the patch comes from step 2 and its SHA256 is verified before applying)
CUDA=ON JOBS=8 bash scripts/build_llamacpp.sh

# 4. print the Linux path needed by the Windows launcher
pwd -P
```

Copy `limite-launcher.env.example` to `limite-launcher.env` next to
`Avvia-Limite-GPU.cmd`. Add the following line, replacing the example path
with the output of `pwd -P` above:

```dotenv
LIMITE_WSL_REPO=/absolute/Linux/path/to/limite-runner
```

Use a Linux/WSL path, including `/mnt/c/...` if the checkout is on a Windows
drive. Write spaces literally, without surrounding quotes. This setting
lets the launcher find both `.venv/bin/limite` and the binary built under
`tools/llama.cpp/build/bin/`. Set `LIMITE_WSL_DISTRO` too if your distribution
is not named `Ubuntu`. The local configuration file is git-ignored.

For an existing installation elsewhere, set `LIMITE_BIN` to its Linux
executable path and `LIMITE_LLAMA_SERVER` to the patched server binary.

On Windows, from a terminal in the repository, run:

```bat
Avvia-Limite-GPU.cmd --check
Avvia-Limite-GPU.cmd
```

You can also double-click the launcher. It starts the runner inside WSL and
opens [the local chat](http://127.0.0.1:8080/). Ctrl+C in the launcher window
stops the server. With the absolute paths configured above, it can be started
from another working directory too. If the old native llama-server is using
8080, stop it first or set a different `LIMITE_PORT` in the configuration.

The launcher adds a fresh `?limite-runner=...` query to the chat URL each
time it opens the browser, including when reusing a running server. This
avoids reusing a cached native llama-ui document from an earlier server on
the same port. The port and API endpoints stay unchanged.

Check the state of
the configured port with `Avvia-Limite-GPU.cmd --probe` (exit codes: `0` runner
ready, `2` runner not ready, `3` another service, `1` explicit connection
refused, `4` DNS failure/timeout/uncertain). The launcher reuses the port only
for `0`/`2` and starts only on `1`: a DNS failure, timeout or generic
connection failure never counts as a free port.

The chat page is served by the Python runner: every request goes through the
runner's single-turn preprocessing (see [Chat behavior](#chat-behavior)).
The bundled chat adapts the original llama.cpp UI: sidebar history, Markdown/LaTeX,
collapsible reasoning, copy and regenerate. Its transport is specific to this
runner: only the latest question is sent, with no tools, MCP or custom system
prompt. The lightweight page remains available at `/simple`.

### B. LiteRT fallback (no C++ compilation; slow on the tested CPU)

Use this route for smoke tests. On the reference Windows CPU setup, decoding
took about 40–50 seconds **per token**; use route A for interactive chat.
Package availability and behavior on other platforms have not been qualified
by this project. From a checkout with Python 3.12 and uv installed:

```bash
uv sync --extra litert              # Python 3.12, locked deps + LiteRT backend
uv run limite setup --backend litert
uv run limite run "What is 27 times 34?" --stream --backend litert
```

Then serve the chat and API (`--backend litert` keeps this route:
without it a GPU/Metal machine would auto-pick `llamacpp`):

```bash
uv run limite serve --port 8000 --backend litert
```

The chat page is at `http://127.0.0.1:8000/`, the API at `/v1`.
For the Python example, install the client with `uv pip install openai`
and run it with `uv run python` in another terminal:

```python
from openai import OpenAI
sdk = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
print(sdk.chat.completions.create(
    model="limite-1b-violetto",
    messages=[{"role": "user", "content": "Solve 2x + 3 = 11."}],
    temperature=0.6, top_p=0.95,
).choices[0].message.content)
```

Installing from a built wheel outside the source tree works too
(`pip install limite_runner-<version>-py3-none-any.whl`): the llama.cpp
build script and the `limite-logits` companion source ship inside the wheel,
these resources do not have to be retrieved separately. Model setup and
llama.cpp compilation still require their own downloads and prerequisites.

## Chat behavior

Limite is single-turn, tool-less and refuses custom system prompts. The
bundled chat page (`/`) is built around that contract:

- **only the last question** is sent to the model; the conversation stays
  visible in the page but is never re-sent (server-side enforcement too:
  history and custom system prompts are dropped with a warning, tools get a
  clear 400);
- **tools, MCP and system-prompt customization are disabled** — the model's
  own template rejects them;
- **a real generation cap** (default **2048 tokens**, configurable in the
  page, per request with `max_completion_tokens`/`max_tokens`, or per server
  with `--max-tokens` / `LIMITE_MAX_TOKENS`) — never unlimited;
- **honest end states**: every answer is labelled `concluded` (stop),
  `token limit reached — truncated and possibly incomplete` (length),
  `interrupted` (you pressed Stop), `unverified` (the stream ended without
  any completion status — never presented as concluded) or `error` (with the
  reason; an empty message is replaced by the exception type). A truncated
  answer is never presented as complete;
- `finish_reason` is propagated end to end (SSE and non-streaming): `stop`
  when the model's own stop condition fired (EOS or stop word), `length` when
  the generation cap or the context limit was reached. A stream that ends
  without the engine's final event is an error (llamacpp) or `unverified`
  (fallbacks) — never a fabricated `stop`.

## Configuration

Precedence, highest wins: **CLI flag > environment variable >
`limite-launcher.env` (Windows launcher only) > built-in default**.

| what | CLI | environment | default |
|---|---|---|---|
| data directory (artifacts, model, template) | `--data-dir` | `LIMITE_DATA_DIR` | platformdirs user data dir (`~/.local/share/limite-runner` on Linux/WSL) |
| llama-server binary | no dedicated flag | `LIMITE_LLAMA_SERVER` | repo build dir, then PATH |
| GGUF model file | `--quant` (name inside data dir) | `LIMITE_MODEL` | `$LIMITE_DATA_DIR/conv/gguf/limite-1b-violetto-<QUANT>.gguf` |
| quantization / context | `--quant`, `--ctx` | `LIMITE_QUANT`, `LIMITE_CTX` | `Q8_0` / `16384` |
| host / port | `--host`, `--port` | `LIMITE_HOST`, `LIMITE_PORT` | `127.0.0.1` / `8000` (`8080` in the Windows launcher) |
| GPU offload (llama.cpp `-ngl`) | `--ngl` | `LIMITE_NGL`, `LIMITE_GPU` (launcher: `1`/`0`) | auto (99 with GPU/Metal, 0 otherwise) |
| generation cap (tokens) | `--max-tokens` | `LIMITE_MAX_TOKENS` | `2048` |
| backend | `--backend` | `LIMITE_BACKEND` | auto (llamacpp on GPU/Metal, else litert) |
| WSL distribution (launcher) | — | `LIMITE_WSL_DISTRO` | `Ubuntu` |
| `limite` executable (launcher) | — | `LIMITE_BIN`, `LIMITE_WSL_REPO` | PATH / `<repo>/.venv/bin/limite` |
| CUDA libraries in WSL (launcher) | — | `LIMITE_LD_LIBRARY_PATH` | `/usr/lib/wsl/lib:/usr/local/cuda/lib64` |

Use absolute paths for launcher configuration; paths with spaces are supported.
The Windows launcher reads
`limite-launcher.env` (KEY=VALUE lines, `#` comments — see the `.example`);
that file is personal and git-ignored. `scripts/limite-serve.sh` is the
Linux/WSL entry point used by the launcher and can be run directly.

## Backends

| backend      | what it is | needs |
|---|---|---|
| `llamacpp` (primary) | patched `llama-server` as a subprocess (`--no-context-shift` always) | `limite setup --backend llamacpp` builds llama.cpp @ `5836771` + community patch (Linux/macOS/WSL2; on Windows use WSL2 or Docker) |
| `litert` (no-build fallback) | LiteRT-LM 0.17.0 on CPU with the `.litertlm` bundles | from checkout: `uv sync --extra litert`, then `uv run limite setup --backend litert` |
| `transformers` (numeric reference) | official checkpoint, `trust_remote_code=True`, `attn_implementation="sdpa"`, pinned revision | from checkout: `uv sync --extra transformers` |

## Tested environments

These are results from the project's verification runs, not a claim that every
hardware and operating-system combination is supported.

| route | verification | intended use |
|---|---|---|
| llama.cpp, Windows + WSL2 + NVIDIA | GPU build, runner and chat/API tested on the reference laptop | primary interactive route |
| llama.cpp, Docker CPU | clean build and model run verified in the 0.1.3 campaign | CPU container route; not a fresh 0.1.4 Docker qualification |
| LiteRT, native Windows CPU | real bundle loaded and decoded; about 40–50 s/token | smoke tests, not interactive chat |
| Transformers BF16 reference | numerical comparison workflow exercised at N = 16 | validation; GPU runner route not qualified |
| Docker CUDA, macOS/Metal, bare-metal Linux, native Windows llama.cpp | not verified by this release | additional platform testing needed |

**Recorded checks:**

- Windows 11 + WSL2 Ubuntu 24.04 + CUDA 13.0 in WSL (RTX 5090 Laptop,
  sm_120): `scripts/build_llamacpp.sh` with `CUDA=ON` (including with a
  custom data directory containing spaces), `limite setup --backend
  llamacpp`, the Windows launcher (`--check` and full start), chat/API
  through the runner on GPU (~150 tok/s Q8_0): two consecutive questions,
  history sent by a client and dropped, tools rejected (400), token cap →
  `finish_reason=length`, concluded answer → `finish_reason=stop`, mid-stream
  interruption leaves the server healthy.
- Docker CPU image (0.1.3 verification): clean `docker build --no-cache` + fresh-volume run with
  `LIMITE_BACKEND=llamacpp` (verified download, `setup --no-build` without
  prompts, capped generation → `length`, EOS → `stop`, tools/invalid params
  → 400, engine restart after a crash, `/health` readiness states).
- Clean-wheel install outside the source tree (venv in a path with spaces):
  `limite doctor`, `limite run`/`serve` missing-requirement errors, build
  script resolvable from package data.
- Regression suite: 154 tests passed with
  `pytest tests -q --ignore=tests/test_integration.py`. The excluded file
  contains live integration tests; this count is not a new full model run.

**Not verified for this release** (declared, not claimed): macOS (Metal),
bare-metal Linux without WSL, Windows-native llama.cpp CPU/GPU without WSL2, the CUDA
Docker image, `litert` at interactive speed, `transformers` on GPU, `limite
validate` at longer N than 16 tokens.

## Model constraints (enforced, not left to the user)

- **Single-turn**: canonical math system prompt + last user message only.
  History and custom system prompts are discarded with a warning.
- **Tools/function-calling rejected** with a clear 400 (the model's own
  template rejects them too).
- **Template**: the model's embedded Jinja (never rewritten by hand).
- **Sampling defaults**: `temperature=0.6`, `top_p=0.95` (from
  `generation_config.json`; LiteRT example's `top_p=1.0/top_k=50` ignored).
  Requests may override them within validated ranges: `temperature` in
  `[0, 2]`, `top_p` in `(0, 1]`, token limits ≥ 1 — anything else is a 400.
  Token-limit precedence is explicit: `max_completion_tokens` (current OpenAI
  field) > `max_tokens` (deprecated) > server default (2048).
- **Context**: 16384 / 32768 / 65536 (default 16384). RAM is estimated before
  load (LiteRT KV alone: 1.5 / 3.0 / 6.0 GiB); refusal is explicit.
- **Quantization**: default `Q8_0`, recommended minimum `Q5_K_M`, loud warning
  below Q4.
- **Base vs chat**: `/v1/completions` and `--no-template` give raw completion.
- **Honest status**: `/health` answers 200 `ok` only when the backend is
  available **and** its engine is actually loaded/running — otherwise 503
  with `status="unavailable"` (binaries/libraries missing) or
  `status="not_ready"` (present but the engine is down, e.g. llama-server
  not started). It also identifies the service (`service: limite-runner`):
  the launcher reuses a port only when this identity is present, and refuses
  any other server there (e.g. a raw llama-server, which has no single-turn
  preprocessing). The `finish_reason` in every completion — streamed or not —
  reflects the observed end: `stop` when the model's own stop condition
  fired (EOS or stop word), `length` when the generation cap was reached
  (llama.cpp `stop_type: "limit"`) or the context ran out. The field
  semantics are pinned to the pinned llama.cpp revision, not guessed.

## CLI

```bash
uv run limite doctor
uv run limite run "Solve 2x + 3 = 11." --backend llamacpp --stream --max-tokens 2048
uv run limite serve --backend llamacpp --host 127.0.0.1 --port 8000
uv run limite validate --help
```

These examples assume a prepared backend. Use `uv run limite <command> --help`
for options. With an installed console command on PATH, the `uv run` prefix
can be omitted. For direct llama.cpp use, set `LIMITE_LLAMA_SERVER` to the
absolute binary path if it cannot be discovered from your working directory.

`limite run` prints the end status on stderr: `[finish_reason=stop] answer
concluded.` / `[finish_reason=length] token limit reached: the answer is
truncated and may be incomplete.` / `[finish_reason=unverified] no completion
status was observed: the answer is NOT verified as complete.` / `[interrupted]`
on Ctrl+C (exit 130) / `[error]` on failures (exit 1). Unverified and errored
runs exit 1: a truncated or unobserved answer is never a concluded one.

## Validation

`limite validate` checks numerical agreement with the Transformers BF16
reference using 20 fixed prompts. It produces a JSON report and a readable
table. This checks runtime behavior, not mathematical benchmark accuracy.

- **Logits:** mean KL divergence and top-1 agreement over the first N tokens,
  with configurable thresholds. The llama.cpp comparison uses the reference
  token prefix on both sides (teacher forcing). Free-running agreement is
  reported separately.
- **Cache and numerical checks:** finite logits and cache/no-cache coherence;
  an argmax change is accepted only at a near-tie (gap at most 0.1 logit).
- **Stopping:** token declarations are marked `verified`, `failed` or
  `unverifiable`. A separate generation probe records the observed end state;
  emitting an EOS token alone does not prove that the engine stopped.

Missing evidence is explicit: a PASS with unavailable KL or unverifiable stop
declarations is labelled **partial**. LiteRT 0.17.0 does not expose full-vocabulary
logits, so its KL is unavailable. Verification beyond N = 16 is not claimed.
See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details.

## Docker

Two separate images, with different support status:

**CPU image (`docker/Dockerfile`) — llama.cpp verified in the 0.1.3 campaign.**

```bash
docker build -f docker/Dockerfile -t limite-runner:cpu .
docker run -p 127.0.0.1:8000:8000 -v limite-data:/data -e LIMITE_BACKEND=llamacpp limite-runner:cpu
```

The entrypoint runs a verified `limite setup --no-build` when `/data` is
empty (no confirmation prompt) and serves on 8000 (chat page at `/`).

**CUDA image (`docker/Dockerfile.cuda`) — NOT verified for this release.**
It builds the CUDA 13.0 route (`nvidia/cuda:13.0.0`, `CUDA=ON`, cu130 torch)
but no end-to-end run was performed; the verified GPU route is Windows + WSL2
with `scripts/build_llamacpp.sh` (see above).

```bash
docker build -f docker/Dockerfile.cuda -t limite-runner:cuda .   # needs NVIDIA runtime
docker run --gpus all -p 127.0.0.1:8000:8000 -v limite-data:/data -e LIMITE_BACKEND=llamacpp limite-runner:cuda
```

## Requirements

- **Python** 3.12 (the package pins `>=3.12`).
- **GPU route**: NVIDIA driver on the host + CUDA toolkit inside WSL2 (or the
  CUDA Docker base). The distro package is too old for sm_120 (RTX 50xx):
  use NVIDIA's repository (`cuda-toolkit-13-0`). The runtime needs
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64` (the launcher sets
  it via `LIMITE_LD_LIBRARY_PATH`).
- **Disk**: ~3 GB after a LiteRT setup (840 MB bundle plus a ~2.2 GB XNNPACK
  weight cache generated next to it on first run); the Q8_0 GGUF adds 1.1 GB.
- **Network**: needed for dependency installation, verified model downloads,
  and the initial llama.cpp source/build dependencies. Prepared local
  inference does not require a cloud inference service.

## Stopping the server

- Windows launcher: Ctrl+C in the launcher window (the server runs in the
  foreground; the window can then be closed).
- `limite serve` in a terminal: Ctrl+C.
- Docker: `docker stop <container>`.
- If a port is still busy, identify the owning process
  (`netstat -ano | findstr :8080` on Windows) and stop the relevant server,
  or select another `LIMITE_PORT`.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `ERROR: model not found: ...gguf` | run `limite setup --backend llamacpp --data-dir <dir>`, or set `LIMITE_MODEL` to your `.gguf` |
| `ERROR: patched llama-server binary not found` | build with `CUDA=ON bash scripts/build_llamacpp.sh`, or set `LIMITE_LLAMA_SERVER` |
| `ERROR: chat template not found: ...` | run `limite setup` (downloads the verified template into the data dir) |
| `ERRORE: WSL2 non trovato` | install WSL2 (`wsl --install`), reboot |
| `ERRORE: distribuzione WSL "..." non disponibile` | check `wsl -l -v`, set `LIMITE_WSL_DISTRO` (no spaces in the name: this `wsl.exe` keeps quotes as part of the name) |
| `the 'limite' command is not available` | `uv sync --extra litert` in the checkout and set `LIMITE_WSL_REPO`, or set `LIMITE_BIN` |
| `/health` → 503 `unavailable` | binaries/libraries missing: read `error` in the body |
| `/health` → 503 `not_ready` | engine present but not loaded (e.g. llama-server crashed): the next request reloads it |
| launcher: `la porta ... NON e' limite-runner` | another service occupies the port (e.g. the old native llama-server chat): change `LIMITE_PORT` or stop it - the launcher never reuses it |
| launcher: `non ha dato una risposta certa` | DNS, timeout or another uncertain transport failure: check `LIMITE_HOST` and the service/port; the launcher does not start automatically |
| CUDA build fails on RTX 50xx | Ubuntu's toolkit is too old for sm_120: install `cuda-toolkit-13-0` from NVIDIA's WSL repository |
| LiteRT answers take minutes | known limit: ~40-50 s/token on CPU; use the `llamacpp` backend |
| answers cut mid-sentence | that is `finish_reason=length` (token cap): raise `max_completion_tokens` / `--max-tokens` / `LIMITE_MAX_TOKENS` |

## Attribution and license

- **Model** Limite 1B Violetto: [Paradigma](https://huggingface.co/paradigma-inc)
  (`paradigma-inc/limite-1b-violetto`), Apache-2.0.
- **Conversion** (GGUF quants, LiteRT bundles, llama.cpp `limite` adapter
  patch, validation artifacts): [Tdamre](https://huggingface.co/Tdamre)
  (`Tdamre/limite-1b-violetto-GGUF-LiteRT`). The patch is third-party code:
  `limite setup` shows the diff and never applies it silently.
- **llama.cpp**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
  at the pinned commit `5836771` (its own license).
- **Runner** (this repository: backends, CLI, server, chat page, validation
  tooling): limite-runner contributors.

This repository is licensed under Apache-2.0 (see [LICENSE](LICENSE)); the
copyright line matches `pyproject.toml` authors.

## Known limits

- `llamacpp` build script targets Linux/macOS/WSL2 only (upstream constraint
  of the conversion's `build.sh`); Windows users run it in WSL2 or Docker.
  Verified on WSL2 (Ubuntu 24.04, CUDA toolkit 13.0 from NVIDIA's WSL
  repository, RTX 5090 Laptop): GPU decode ~150 tok/s vs ~34 tok/s CPU for
  the Q8_0 GGUF.
- LiteRT: CPU only; sync decode has a 10-minute deadline → we always stream.
  **Measured speed (reference laptop, 32 cores, Windows, CPU route):** engine
  load ~7 s, prefill ~50-60 s for a short prompt, then **~40-50 s per token**
  (~1.5 tok/min) — the same with or without the XNNPACK weight cache.
  "Runs" does not mean "runs at interactive speed": `limite run`/`serve` on
  LiteRT are usable only for smoke tests; `limite validate` (20 prompts) would
  take many hours. Use the llamacpp backend for anything real.
- Disk: the data dir needs ~3 GB after a LiteRT setup (840 MB bundle plus a
  ~2.2 GB XNNPACK weight cache generated next to it on first run); GGUF
  quants add their own size on top.
- No training, no multi-turn, no tool calling, no GPU for LiteRT, no
  sharding/tensor-parallel (unsupported by the model), no LM Studio.
- `trust_remote_code=True` only ever on the pinned revision `9f4d1567...`
  (main at the time of the change; the weights are identical to e078b537,
  which has no modeling code).
- The Windows launcher needs a WSL distribution whose name has no spaces
  (`wsl.exe` on this channel keeps quotes as part of the name); run
  `scripts/limite-serve.sh` directly inside WSL otherwise.
- Use the adapted UI served by the runner at `/`, not a separate native
  llama-server UI. The latter bypasses this integration. Browser conversations
  are saved locally per origin; generation cannot resume across page reloads.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current design,
[CHANGELOG.md](CHANGELOG.md) for the change history and
[RELEASE.md](RELEASE.md) for the beta release notes.

### Rebuilding the bundled UI

The adapted Svelte UI lives in `webui/` (upstream MIT license included).
End users do not need Node: its compiled assets are included in the wheel.
After changing UI sources, run `npm ci`, `npm run check`,
`node --test runner-chat.test.mjs`, and `npm run build` inside `webui/`,
sequentially, before building the Python wheel. See [webui/README.md](webui/README.md).
