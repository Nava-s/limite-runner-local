# ARCHITECTURE — limite-runner (current state)

One package, three interchangeable inference backends, one CLI, one
OpenAI-compatible server. No cloud dependency; the same image that runs on a
laptop runs on EC2/ECS/SageMaker.

## Pipeline

`CLI / HTTP request` → `messages.prepare_*` (single-turn enforcement) →
`template.render_chat` (model's own Jinja) → `Backend.generate / .stream_full`
(`StreamChunk` pieces + real `finish_reason`: `stop` or `length`) →
`SamplingParams(temperature=0.6, top_p=0.95)` → text (+SSE).

Base mode (`/v1/completions`, `--no-template`) skips the template and feeds
the raw prompt straight to the backend.

## Components (one section per deployed component)

### `scripts/compare_prompt_modes.py` — optional prompt-format experiment

Uses an already-running llama-server to compare the original chat prompt,
chat with its system turn removed, and raw text, without altering model
artifacts or server configuration. Flow: `/props` + `/apply-template` →
construct three prompt formats → sequential `/completion` calls with equal
settings and no prompt-cache reuse → JSON report and readable text.

| function | purpose |
|---|---|
| `main` | question/settings/output arguments, prompt construction and report |
| nested `request` | HTTP JSON requests using the Python standard library |

Example: `python scripts/compare_prompt_modes.py --max-tokens 2048`.
The noncanonical formats are experimental; one greedy example is not a
quality benchmark. Reports preserve the exact submitted prompts and the
observed stop type so truncation cannot be mistaken for a completed answer.

### `Avvia-Limite-GPU.cmd` + `scripts/limite-serve.sh` — Windows/WSL2 launcher

Portable entry point for the supported GPU route. The `.cmd` resolves every
path from its own location (`%~dp0`), loads configuration (precedence:
environment > `limite-launcher.env` in its folder > built-in defaults),
preflights Windows side (WSL2 present, distro reachable — note: this
`wsl.exe` keeps quotes as part of the `-d` name, so distro names with spaces
are refused with an explanation), reuses a healthy server on the target port
or starts one, and opens the browser when `/health` reports `ok`
(`--open-when-ready` internal mode). `--check` runs the requirement checks
without starting anything. No download, no compile: missing pieces produce
actionable errors and exit 1.

Every browser-opening branch uses `CHAT_URL`, with a fresh
`?limite-runner=...` query per launcher invocation. This forces a distinct
document URL when a browser has cached native llama-ui at the same origin;
startup, ready-server reuse and not-ready reuse all follow this path.

`scripts/limite-serve.sh` runs inside WSL/Linux and is the single place that
knows the Linux defaults (`$HOME/.local/share/limite-runner`, auto-detected
llama-server, `-ngl 99`/`0` from `LIMITE_GPU`, host/port/cap from env). It
checks model, template, llama-server and the `limite` launcher
(`LIMITE_BIN` > `<repo>/.venv/bin/limite` > `uv run --project` > PATH >
`~/.local/bin`) and then `exec`s `limite serve --backend llamacpp` — so the
browser chat goes through the runner's preprocessing, never through the raw
llama-server web UI. Reference table:

| parte | cosa fa |
|---|---|
| env file + defaults | `limite-launcher.env` (non versionato) e variabili d'ambiente |
| preflight | WSL2, distribuzione (no spazi), script presente; messaggi azionabili |
| `--check` | solo controlli requisiti (usato anche dai test) |
| `limite-serve.sh` | verifica modello/template/binario/limite, poi `limite serve` |

### `scripts/health-probe.ps1` — Windows launcher service probe

Runs under Windows PowerShell 5.1 and identifies the service at the configured
URL. The launcher reuses an identified runner (0 ready, 2 not ready), rejects
another HTTP service (3), starts on explicit socket connection refusal (1),
and blocks automatic startup for DNS, timeout or indeterminate failures (4).

Flow: the launcher passes the health URL to `Invoke-WebRequest`; HTTP error
bodies go through `Read-ErrorBody`, then `Get-ServiceName` identifies the
runner. Transport errors traverse the full inner-exception chain and require
`SocketError.ConnectionRefused` for exit 1. `ConnectFailure` alone is
insufficient; DNS and timeout errors remain uncertain.

| function / block | purpose |
|---|---|
| `Read-ErrorBody` | reads ErrorDetails or the HTTP response stream on 4xx/5xx |
| `Get-ServiceName` | extracts service identity and readiness from JSON |
| transport catch block | classifies typed nested exceptions without localized message matching |
| final exit selection | maps HTTP status and identity to 0, 2 or 3 |

### `src/limite_runner/config.py` — shared pins, catalogue, defaults

Holds every pinned value (HF revision `9f4d1567…`, conversion revision,
llama.cpp commit `5836771`, patch SHA256, LiteRT-LM 0.17.0), the verified
artifact catalogue (path → sha256/size from `artifact_manifest.json`), and the
runtime defaults (sampling, contexts, quants, `DEFAULT_MAX_TOKENS = 2048`).
Flow: CLI/backends read the
catalogue and defaults from here; `artifacts.py` verifies downloads against
it. Reference table:

| symbol | meaning |
|---|---|
| `HF_MODEL_REVISION` | checkpoint pin for the transformers reference |
| `artifact_catalog()` | verified `{repo_path: (sha256, bytes)}` |
| `RunConfig` | effective runtime config (flags override auto-choice) |
| `load/write_lockfile` | `limite.lock.json` persistence |

### `src/limite_runner/hw.py` — shared dev+prod hardware probe

Detects OS/arch/RAM/GPU-VRAM/CUDA/Metal/WSL at startup and auto-selects
backend (`llamacpp` with GPU/Metal, else `litert`) and quantization (`Q8_0`),
unless overridden. Estimates RAM before load and refuses explicitly.
Flow: `detect_hardware()` → `choose_backend/choose_quant` →
`check_memory_budget()` gate in `run/serve/validate`. Reference table:

| funzione | cosa fa |
|---|---|
| `detect_hardware` | OS, arch, CPU, RAM, nvidia-smi GPU/VRAM/driver, nvcc, Metal, WSL |
| `choose_backend` | GPU/Metal → llamacpp, altrimenti litert (override da config) |
| `choose_quant` | default Q8_0 (override da config) |
| `estimate_ram_needed` / `check_memory_budget` | stima pre-load, rifiuto esplicito |

### `src/limite_runner/messages.py` — shared single-turn contract

Enforces Limite's single-turn use before the template runs: drops history and
custom system prompts (warning each time), rejects tools/tool-calls/unknown
roles/non-string content with explicit errors. Flow: request messages enter,
`prepare_messages` reduces them to exactly one user turn
(`PreparedPrompt`), or raises `ToolsNotSupportedError`/`InvalidMessagesError`
(the server maps these to HTTP 400). Reference table:

| funzione | cosa fa |
|---|---|
| `prepare_messages` | riduce a single-turn canonico, scarta con warning |
| `prepare_completion_prompt` | passthrough grezzo per la modalità base |
| `ToolsNotSupportedError` | tools/function-calling rifiutati con messaggio chiaro |

### `src/limite_runner/template.py` — shared template execution

Executes the model's own Jinja chat template (downloaded SHA256-verified from
the pinned revision; identical to the GGUF/LiteRT-embedded copy) with only
the user message + `add_generation_prompt=True`. Never rewritten by hand.
Flow: verified `chat_template.jinja` → Jinja2 env con `raise_exception` →
`render_chat` produce il prompt finale. Reference table:

| funzione | cosa fa |
|---|---|
| `load_template` | carica il Jinja verificato dagli artefatti |
| `render_chat` / `render_chat_text` | esegue il template del modello sul single-turn |

### `src/limite_runner/backends/llamacpp.py` — deployed as patched llama-server

Primary backend. Manages a patched `llama-server` subprocess (found via
`LIMITE_LLAMA_SERVER`, setup build dir, or PATH), always with
`--no-context-shift`; `-ngl 99` on CUDA/Metal, `0` on CPU-only. Speaks the
raw `/completion` endpoint (streaming SSE included) with the caller-rendered
prompt so the template is applied exactly once. The subprocess pipe is
drained continuously by a reader thread (a full pipe buffer would block
llama-server on its own logging); a bounded tail is kept for error reports,
and every startup failure (exit or readiness timeout) tears the process,
client and reader down before raising. Flow: `load()` avvia, drena i log e
attende `/health` → `generate()` POST `/completion` → `stream()` SSE →
`close()` termina il processo. Reference table:

| funzione | cosa fa |
|---|---|
| `find_llama_server` | localizza il binario patchato |
| `_drain_logs` / `_log_tail` | consuma i log del server per tutto il runtime (niente pipe piena), coda limitata per i report |
| `load` | avvia llama-server, attende readiness, fallisce con l'output reale ripulendo subprocess+pipe |
| `is_ready` | motore realmente attivo (subprocess vivo) — distinto da `available` (binario presente) |
| `generate` / `stream_full` | completion grezza + SSE sul prompt già renderizzato; reason dallo schema dell'evento finale alla revisione fissata (`stop_type: "limit"` o `truncated` → `length`, `"eos"`/`"word"` → `stop`); stream chiuso senza evento finale → errore esplicito (mai `stop` inventato) |
| `info` / `close` | stato build+runtime / shutdown pulito (processo, client, reader) |

### `src/limite_runner/backends/litert.py` — deployed as litert-lm CPU engine

No-compilation fallback. Real `litert-lm-api` 0.17.0 `Engine(CPU)` +
`Session(apply_prompt_template=False)`; **all** generation via
`run_decode_async` (sync decode has the 10-minute deadline); stop tokens
filtered incrementally (`_StopFilter`: emette solo testo nuovo e trattiene i
suffissi che potrebbero essere l'inizio di uno stop — nessuna riemissione
dell'accumulato, nessun marcatore spezzato tra chunk che trapela); the
finish reason is observed, not declared: `run_decode_async` ends silently
both on EOS and on the token cap (the internal "Max number of tokens
reached" error is swallowed), so `stream_full` reads the session's
`get_benchmark_info().last_decode_token_count` (engine created with
`enable_benchmark=True`: cap reached → `length`, earlier end → `stop`;
fallback: re-tokenize the emitted text); RAM gate
against the 1.5/3/6 GiB KV footprints. Flow: `load()` verifica bundle+RAM →
`run_prefill` → `run_decode_async` → `_StopFilter` → reason osservata →
flush del testo trattenuto se lo stream termina senza stop. Reference table:

| funzione | cosa fa |
|---|---|
| `load` | bundle presente, budget RAM, `Engine(CPU)` (mai GPU delegate) |
| `generate` / `stream_full` | solo path async/streaming, filtro stop incrementale, finish reason osservata |
| `_StopFilter` | emette solo delta nuovi, trattiene i possibili prefissi di stop, flush a fine stream |
| `_session_end_reason` | cap raggiunto → `length` (contatori sessione, fallback tokenize), altrimenti `stop` |
| `tokenize` / `eos_ids` | utilità per validazione e debug |

### `src/limite_runner/backends/transformers_backend.py` — deployed as reference process

Numeric ground truth. Official checkpoint at the pinned revision,
`trust_remote_code=True`, `attn_implementation="sdpa"`, device auto
(cuda/mps/cpu). Exposes full logits for `validate`. Flow: `load()` scarica
dal lockdir HF → `generate()` greedy/sampled → `next_token_logits()` traiettoria
greedy con logit full-vocab. Reference table:

| funzione | cosa fa |
|---|---|
| `load` | checkpoint pinnato, sdpa, device automatico |
| `generate` / `stream` | generazione (stream emulato token-wise) |
| `next_token_logits` / `logprobs_of_continuation` | logit full-vocab per KL/top-1 |

### `src/limite_runner/server/static/chat.html` — deployed at `/simple`

Single-file local chat page (no external assets, shipped in the wheel). The
browser keeps the conversation visible but sends **only the latest user
question** to `/v1/chat/completions` (`stream: true`, one message, no tools,
no system prompt); the generation cap defaults to the server default (2048)
and is adjustable per message. Every answer carries an explicit end state —
concluded (`stop`), token cap → "truncated, possibly incomplete" (`length`),
unverified (`unverified`), interrupted (AbortController on the Stop button),
error (distinct colour and text) — so a truncated or unobserved answer never
reads as complete. Reference table:

| parte | cosa fa |
|---|---|
| payload builder | un solo messaggio utente: la cronologia resta in pagina, mai nel payload |
| SSE reader | delta + `finish_reason` finale; eventi `{"error": ...}` distinti dai finish |
| statusText | mapping finish/abort/error → etichetta onesta (mai "conclusa" su troncata) |
| Stop button | abort della fetch → stato "interrotta" |

### `src/limite_runner/server/app.py` — deployed as uvicorn service

OpenAI-compatible frontend: `/v1/chat/completions` + `/v1/completions` (SSE),
`/v1/models`, `/health`, `/` (the chat page above, with the server's default
cap injected). Stock `openai` SDK compatible. `/health` is an
honest readiness probe with three states: 200 `ok` only when the backend is
available **and** the engine is actually loaded/running (`Backend.is_ready`,
e.g. the llama-server subprocess alive); 503 `unavailable` when the backend
cannot work at all (binaries/libraries missing); 503 `not_ready` when it is
present but the engine is not up. Streaming
propagates the backend's observed finish reason (`stop` / `length`) into the
final SSE chunk instead of a hardcoded `stop`. Maps tools→400,
preprocessing errors→400, invalid sampling parameters→400 (temperature in
[0, 2], top_p in (0, 1], token limits ≥ 1; `max_completion_tokens` takes
precedence over the deprecated `max_tokens`, then the server default),
backend failures→500. Flow: request → preprocessing → validazione sampling →
render → backend (`stream_full`) → OpenAI JSON/SSE. Reference table:

| funzione | cosa fa |
|---|---|
| `create_app` | costruisce FastAPI su un backend già caricato; `default_max_tokens` (2048) iniettato nella chat page |
| `chat_ui` / `_chat_html` | `/` serves the adapted llama-ui bundle; `/simple` serves the lightweight fallback |
| `health` | readiness onesta: 200 solo con backend disponibile e motore pronto; 503 `unavailable`/`not_ready` altrimenti; identifica il servizio (`service: "limite-runner"`, `version`) — il launcher riutilizza la porta solo con questa identità |
| `chat_completions` / `completions` | endpoint chat (template) e base (grezzo), con streaming e finish reason reale; senza fine osservata → `unverified`, senza mai dichiarare `stop` |
| `_sampling` / `_resolve_max_tokens` | validazione parametri + precedenza esplicita tra i due limiti di token |
| `_err_text` | messaggio errore mai vuoto (fallback: nome dell'eccezione) |
| `serve` | avvio uvicorn |

### `src/limite_runner/cli.py` — deployed as `limite` console script

Typer commands `setup/run/serve/doctor/validate` wiring hw→artifacts→backend→
output. `setup` verifies SHA256, writes the lockfile, shows the third-party
patch and asks before applying/building; with `--no-build` nothing is applied
or compiled, so no confirmation is requested (non-interactive callers — e.g.
the Docker entrypoint — have no stdin). Configuration precedence is explicit:
CLI flag > env (`LIMITE_DATA_DIR`, `LIMITE_MODEL`, `LIMITE_LLAMA_SERVER`,
`LIMITE_HOST`, `LIMITE_PORT`, `LIMITE_MAX_TOKENS`, `LIMITE_NGL`,
`LIMITE_BACKEND`) > built-in default. `run`/`serve` report the generation end
state (`_finish_note`: concluded / truncated / interrupted / error); missing
requirements are single actionable ERROR lines, never tracebacks. The build
script is resolved by `_build_script` from package data (wheel installs) or
`scripts/` (source checkouts) — never downloaded. Flow per comando: detect →
budget → artefatti → backend → azione. Reference table:

| comando | cosa fa |
|---|---|
| `doctor` | hardware, data dir effettiva, presenza modello/template, disponibilità backend, versioni, cause di fallimento |
| `setup` | download verificato + lockfile + patch visibile + build opzionale (conferma solo se builda; script da package data) |
| `run` | un prompt single-turn, anche in streaming, con stato finale su stderr |
| `serve` | server OpenAI-compatibile + chat page (`/`) |
| `validate` | suite numerica contro il riferimento transformers |

### `src/limite_runner/validate.py` — deployed via `limite validate`

20 fixed prompts; per-prompt mean-KL + top-1 over the first N teacher-forced positions (the reference greedy prefix fed to the candidate); free-running metrics are informational
logits vs transformers BF16; finite-logit, cache/no-cache coherence and
tri-state stop-token checks (`verified` / `failed` / `unverifiable`, with the
observed declarations as evidence). Stop-token emission in the candidate's
free-running trajectory is recorded as an emission, not a halt proof; the
halt proof is a dedicated probe: one real free-running candidate generation
whose observed finish reason (`stop` = engine halted on its own before the
cap; `length` = cap hit, informational) lands in the evidence. A PASS with
components that could not be checked (stop declarations unverifiable and/or
KL n/a) is marked `partial=True` and rendered as "PASS (partial: …)": read
as partial verification, never as a complete one. JSON report + table.
llamacpp logits via
the `limite-logits` companion (C API); LiteRT degrades honestly (KL n/a).
Flow: render prompts → reference logits → candidate logits (o path degrada)
→ metriche → soglie → probe arresto → report. Reference table:

| funzione | cosa fa |
|---|---|
| `run_validation` | orchestra i 20 prompt e decide PASS/FAIL (+ flag `partial`) |
| `mean_kl` / `top1_agreement` | metriche full-vocab |
| `dump_llamacpp_logits` | invoca il companion tool |
| `check_stop_tokens` | tri-stato endoftext + im_end con prove (dichiarazioni, emissione, arresto effettivo osservato) |
| `report_table` / `save_report` | tabella leggibile + JSON |

### `tools/limite-logits.cpp` — deployed next to llama-server

Companion logit-dump built with the patched tree (public C API only):
greedy trajectory + full float32 logits + no-cache re-decode trajectory; the --forced option feeds a reference prefix to both (teacher forcing),
written as `.npz`. Flow: tokenize → chunked prefill → N greedy steps (logit
per step) → memory-clear full recompute per step → npz. Reference table:

| parte | cosa fa |
|---|---|
| arg parsing / load | `-m -c -n --prompt/--prompt-file --out` |
| cached trajectory | prefill + decode incrementale con KV cache |
| no-cache trajectory | ricalcolo completo a ogni step |
| npz writer | zip stored + header npy + CRC32, senza dipendenze |

### `scripts/build_llamacpp.sh` — deployed at setup/build time (Linux/macOS)

Reproducible build: verifies patch SHA256, pins checkout to `5836771`,
shows `apply --stat`, applies, registers+builds `llama-server`,
`llama-completion`, `llama-imatrix`, `llama-quantize`, `limite-logits`
(`CUDA=ON` for the NVIDIA route; CPU otherwise).

### `docker/Dockerfile` + `docker/Dockerfile.cuda` — deployed as images

CPU (Debian slim, CPU llama.cpp + litert + torch CPU) and CUDA
(`nvidia/cuda:13.0.0`, `CUDA=ON` build + cu130 torch) images; `entrypoint.sh`
(registered executable in git, `chmod 0755` forced in both Dockerfiles) runs
verified `setup --no-build` when `/data` is empty — no confirmation prompt,
the CLI skips it with `--no-build` — then serves on 8000 (chat page at `/`).
No service dependency beyond the volume and optional `--gpus`.
Verification status: the **CPU image is verified** end to end (clean build,
fresh volume, real model); the **CUDA image is built but not verified** —
the verified GPU route is Linux/WSL2 via `scripts/build_llamacpp.sh`.

### `scripts/build_llamacpp.sh` — deployment notes

Reproducible build: verifies patch SHA256 (from `LIMITE_DATA_DIR` or the repo
copy), pins checkout to `5836771`, shows `apply --stat`, applies, copies the
`limite-logits` companion source (found next to the script in wheel installs
or in `tools/` in source checkouts), registers+builds `llama-server`,
`llama-completion`, `llama-imatrix`, `llama-quantize`, `limite-logits`
(`CUDA=ON` for the NVIDIA route; CPU otherwise).


### `webui/` and `server/static/llama-ui/` — browser UI deployed at `/`

The original llama.cpp UI is adapted at a pinned upstream revision (see
`webui/README.md` and its MIT license). Svelte keeps the sidebar, IndexedDB
history, Markdown/KaTeX, reasoning disclosure and message actions. The dedicated
transport accepts only the newest user text and the three supported sampling
parameters. UI capabilities never imply additional model capabilities. Flow:
composer/history -> `runnerRequest` -> `/v1/chat/completions` -> SSE parser ->
reasoning/text callbacks -> local database. Tools/MCP/agentic execution,
replay, pre-encoding and automatic title inference are disabled. No service
worker is registered. Root HTML uses `Cache-Control: no-store`.

| Function/component | Responsibility |
|---|---|
| `runnerRequest` | Allowlist request fields; last user text only, independent of saved settings/history |
| `sendRunnerMessage` | UTF-8/SSE buffering, split reasoning tags, errors and observed finish states; abort labels interruption |
| `RunnerStatus` | Poll `/health`; distinguish ready, not ready, wrong service and unreachable |
| `/ui/props` | UI configuration and generation defaults, separate from readiness |
| `package-runner-ui.mjs` | Ship upstream license and provenance beside compiled assets |

The terminal status note is persisted with assistant text. Reload does not
resume generation. Upstream components not exposed by the adapted UI remain
in the source snapshot for traceability; upstream documentation is reference
material, not a claim of supported runner features.
