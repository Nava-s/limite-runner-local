# Release notes â€” limite-runner 0.1.4 (beta)

First public beta. Scope: a reproducible local runtime for Limite 1B
Violetto with one declared supported route and an interface that respects the
model's constraints.

## Highlights

- **Portable Windows launcher** (`Avvia-Limite-GPU.cmd`): no personal paths,
  configuration via environment / `limite-launcher.env` / defaults, works from
  any directory and with paths containing spaces, requirement checks with
  actionable messages (`--check`), never downloads or rebuilds on its own.
- **Limite-coherent chat**: the bundled page (`http://host:port/`) sends only
  the latest question through the runner's single-turn preprocessing; the
  history stays visible in the browser but is never re-sent; tools/MCP/system
  prompt customization are disabled; every answer is labelled concluded /
  token limit (truncated) / interrupted / unverified / error, and
  `finish_reason` (`stop`/`length`) is propagated end to end — a stream that
  ends without an observed completion status is never presented as concluded.
  Default generation cap: 2048 tokens, configurable.
- **Reproducible install**: verified artifact downloads (SHA-256, pinned
  revisions), llama.cpp build script and `limite-logits` companion shipped
  inside the wheel, custom data directories (including paths with spaces)
  honored by setup, build and serve.
- **Honest status**: `/health` distinguishes `ok` / `unavailable` /
  `not_ready`; missing requirements produce single actionable errors.

## Verified in this release

Environment: Windows 11 + WSL2 Ubuntu 24.04, CUDA toolkit 13.0 in WSL,
RTX 5090 Laptop (sm_120), Python 3.12.

| verification | result |
|---|---|
| unit suite (151 tests, live LiteRT decode excluded by design) | green |
| clean-wheel install outside the source tree (venv in a path with spaces) | `limite doctor`/`run`/`serve` behave as documented; build script resolves from package data |
| `Avvia-Limite-GPU.cmd --check` (repo path with spaces) | requirements checked, exit 0 |
| launcher start, chat page at `/` | served, default cap 2048, payload = one user message |
| two consecutive questions | second prompt carries only the second question (unit test + live output) |
| history re-sent by a client | dropped with warnings; no leak into the answer |
| tools / function calling | 400 with a clear message |
| token cap 64/128/256 | `finish_reason=length` (truncated, never shown as complete) |
| concluded answer (405 tokens) | `finish_reason=stop`, correct `\boxed{918}` |
| mid-stream interruption | server stays healthy (`/health` ok), next request works |
| llama-server killed mid-stream | SSE `{"error": ...}` event, no `finish_reason` fabricated; runner stays up (503 `not_ready`, honest) |
| stream ending without the engine's final event | error (llamacpp) / `finish_reason=unverified` (fallbacks) — never a fabricated `stop` |
| port already serving another service | launcher refuses it ("la porta ... NON e' limite-runner", exit 1): verified against a real native llama-server on 8080 |
| runner identity on `/health` | `service: "limite-runner"` + `version`; launcher reuse probe returns 0 only for it |
| probe behaviours (real `--probe` path, Windows PowerShell) | 200 runner → 0, 503 runner → 2 (body+identity read on errors), 200 without identity → 3, 404 → 3, timeout/DNS failure → 4 (no auto-start), refused → 1 |
| transport classification (controlled exceptions, Windows PowerShell) | generic ConnectFailure, HostNotFound and TimedOut → 4; nested SocketError.ConnectionRefused → 1 |
| `LIMITE_QUANT` / `LIMITE_CTX` on the direct CLI | resolved (`Q5_K_M` / `32768` observed live) |
| `setup --data-dir` build hand-off | build receives `--patch <data-dir>/...` and `LIMITE_DATA_DIR` (unit-verified) |
| launcher wired to the existing CUDA install (`limite-launcher.env`) | `--check` ok; start -> probe 0, chat UI, real answer `\boxed{918}`; GPU decode ~98 tok/s end-to-end (405 tokens / 4.1 s) after the `CUDA=ON` restore |
| terminal noise from foreign UI probes (`/props` loop & co.) | answered but not logged: 60+ probes leave zero lines while real traffic stays visible |
| `scripts/build_llamacpp.sh` with custom `LIMITE_DATA_DIR` (spaces) | patch SHA verified from that dir, all binaries rebuilt |
| start with the same custom data dir | model/template resolved from it, generation concluded `\boxed{144}` |
| Docker CPU image (from 0.1.3 campaign) | clean build + fresh volume + real model end to end |

Runtime correctness is what is claimed here; answer *quality* is the model's
own (reasoning style, verbosity) and is out of the runner's scope.

## Verification commands (executed on the environment above)

```bash
# unit suite (live LiteRT decode tests excluded: ~40-50 s/token, by design)
.venv/Scripts/python -m pytest tests -q --ignore=tests/test_integration.py

# wheel build + clean install outside the source tree (venv path with spaces)
uv build
python -m venv "<tmp>/clean env test/venv"
"<tmp>/clean env test/venv/Scripts/python" -m pip install dist/limite_runner-0.1.4-py3-none-any.whl
"<tmp>/clean env test/venv/Scripts/limite" doctor
"<tmp>/clean env test/venv/Scripts/limite" run "What is 2+2?" --backend llamacpp --data-dir "<tmp>/data dir vuota"
# -> actionable ERROR lines (missing template / missing GGUF), exit 1

# README-driven install in a fresh repo copy (path with spaces)
uv sync --extra litert
uv run limite doctor

# launcher requirement checks (Windows, repo path with spaces)
Avvia-Limite-GPU.cmd --check

# live GPU run through the same entry point the launcher uses
bash scripts/limite-serve.sh            # (with LIMITE_BIN/LIMITE_LLAMA_SERVER set)
curl http://127.0.0.1:8099/health       # -> {"status":"ok","ready":true,...}
curl http://127.0.0.1:8099/             # -> chat page, cap 2048, one-message payload
# two questions (each request carries a single user message):
curl -d '{"model":"limite-1b-violetto","messages":[{"role":"user","content":"What is 27 times 34?"}],"max_completion_tokens":2048,"temperature":0}' ...
curl -d '{"model":"limite-1b-violetto","messages":[{"role":"user","content":"What is 15 plus 27?"}],"max_completion_tokens":256,"temperature":0}' ...
# history re-sent by a client -> dropped (server logs "historical user message ... dropped")
curl -d '{"model":"...","messages":[{"role":"user","content":"What is 27 times 34?"},{"role":"assistant","content":"The answer is 918."},{"role":"user","content":"What is 6 times 7?"}],...}' ...
curl -d '{"model":"...","messages":[{"role":"user","content":"hi"}],"tools":[{...}]}' ...   # -> 400
# caps -> finish_reason=length; concluded answer -> finish_reason=stop (405 tokens)
# mid-stream abort -> /health stays ok, next request works

# custom data dir with spaces: build + start
LIMITE_DATA_DIR="<tmp>/data dir custom" JOBS=32 bash scripts/build_llamacpp.sh --dir ~/limite-build/tools/llama.cpp
LIMITE_DATA_DIR="<tmp>/data dir custom" bash scripts/limite-serve.sh   # health shows the spaced model path
```

## Known limits

- Verified route is Windows + WSL2 + CUDA only. **Not verified**: the CUDA
  Docker image, macOS (Metal), bare-metal Linux, Windows-native CPU/GPU,
  LiteRT at interactive speed (~40-50 s/token on CPU â€” smoke tests only),
  `transformers` on GPU, `limite validate` beyond N = 16 tokens.
- `limite validate` verdict on the Q8_0 candidate is PASS (teacher-forced,
  20 prompts); components that cannot be checked are marked partial.
- Single-turn only: no multi-turn, no tools, no training, no sharding.
- The launcher needs a WSL distribution name without spaces (`wsl.exe` quirk,
  documented in the README Troubleshooting).
- The native llama-server web UI is not supported (history re-sending,
  built-in browser tools, no honest finish states): use the bundled page.

## Install

See the README quickstart: `uv sync --extra litert` (or `pip install` the
wheel), `limite setup --backend llamacpp --no-build` (verified artifacts),
then `CUDA=ON bash scripts/build_llamacpp.sh` once, then `Avvia-Limite-GPU.cmd`
(or `limite serve`). Configuration table, requirements, how to stop the
server and troubleshooting are in the README.

## Attribution

Model: Paradigma (`paradigma-inc/limite-1b-violetto`, Apache-2.0).
Conversion (GGUF/LiteRT/llama.cpp patch): Tdamre
(`Tdamre/limite-1b-violetto-GGUF-LiteRT`). llama.cpp: ggml-org @ pinned
commit `5836771`. Runner: limite-runner contributors (Apache-2.0, see
LICENSE).


### Local UI adaptation follow-up (2026-10-12, uncommitted)

The runner now serves an adaptation of the original llama.cpp UI at `/`.
This supersedes the earlier lightweight-only UI statement. `/simple` remains
available. Build/check instructions and upstream attribution are in
`webui/README.md`. Verified: 154 Python tests, 8 transport tests, Svelte check,
CUDA generation for two different questions in one conversation, rendered
formulas, collapsible reasoning and explicit interruption. Existing browser
conversations were retained on the same origin. No claim of renewed Docker,
macOS or other-platform qualification is made by this UI change.
