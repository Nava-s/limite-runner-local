# CHANGELOG â€” limite-runner

Chronological record (one entry per change: what, why, verification).
The architecture document describes the current state; history lives here.

## 0.1.0 â€” initial implementation

- Phase 0 source inspection (verified, nothing assumed):
  what â€” read HF `paradigma-inc/limite-1b-violetto` card/API (`config.json`,
  `generation_config.json`, embedded chat template), conversion repo
  `Tdamre/limite-1b-violetto-GGUF-LiteRT` (README, `runtime/README.md`,
  `build.sh`, `artifact_manifest.json`, `validation.json`, `REPRODUCE.md`),
  `yamura4/limite-1b-violetto-GGUF` README, `litert-lm-api` 0.17.0 wheel API
  (`engine.py`/`session.py`/`interfaces.py`), llama.cpp @ `5836771` headers
  (`llama.h`, `llama-batch.cpp`, `common.cpp`, `perplexity.cpp`).
  why â€” the brief forbids inventing LiteRT/patch/custom-code APIs.
  verification â€” patch `git apply --check` passes on the pinned commit;
  patch SHA256 `0626d2e2â€¦` and template SHA256 `0c4e2c29â€¦` reproduced locally.
- Package skeleton + pins (`config.py`, `pyproject.toml` with
  `limite-runner[llamacpp|litert|transformers]` extras).
  why â€” minimal install stays light; backend deps are opt-in.
- Core: `hw.py` (detection + auto backend/quant + RAM gate), `messages.py`
  (single-turn + tools rejection), `template.py` (model's own Jinja).
- Backends: `llamacpp.py` (llama-server subprocess, `--no-context-shift`,
  auto `-ngl`), `litert.py` (CPU-only, async-only decode, RAM gate),
  `transformers_backend.py` (pinned revision, sdpa, auto device, logits).
- Server (`server/app.py`): `/v1/chat/completions`, `/v1/completions` (SSE),
  `/v1/models`, `/health`; tools â†’ 400; openai-SDK compatible.
- CLI (`cli.py`): `doctor/setup/run/serve/validate`; setup verifies SHA256,
  writes `limite.lock.json`, shows the third-party patch diff and asks before
  applying/building.
- Validation (`validate.py` + `tools/limite-logits.cpp`): 20 fixed prompts,
  mean-KL + top-1 vs transformers BF16, finite/stop-token/cache checks, JSON +
  table; LiteRT degrades honestly (KL n/a, reason logged).
- Build (`scripts/build_llamacpp.sh`, Linux/macOS) + `Dockerfile` (CPU) +
  `Dockerfile.cuda` (CUDA 13.0 base) + `entrypoint.sh`.
- Tests: messages/template/server/hardware/numerics unit tests, openai-SDK
  compat test, live-backend integration tests (skip without artifacts).
- Docs: README (3-command quickstart, backendÃ—OSÃ—hardware matrix, limits),
  ARCHITECTURE.md (current state), this CHANGELOG.

## 0.1.1 â€” audit + measured LiteRT caveats (2026-10-09)

- what â€” LICENSE (Apache-2.0 full text) added at the repo root; copyright
  line "limite-runner contributors" matches pyproject authors.
  why â€” README and pyproject declared Apache-2.0 but no license file shipped.
  verification â€” file present at root, standard unmodified license text.
- what â€” measured the live LiteRT path end-to-end on the reference laptop
  (Windows 11, 32 cores, RTX 5090 Laptop, CPU route): engine load 6.6-7.2 s,
  create_session instant, prefill 49.8-58.7 s for a 13-token prompt, decode
  ~40-50 s per token (~1.5 tok/min), identical with and without the existing
  2.2 GB XNNPACK weight cache. Documented in README (Known limits + matrix
  speed note). why â€” README claimed a plain "âœ… CPU" for LiteRT with no speed
  caveat, which made "runs" look like "usable"; the measured numbers explain
  why live integration tests (64+32 tokens) and `limite validate` are
  impractical on this backend. verification â€” three timed probe runs against
  the downloaded ctx16384 bundle (timestamps per stage and per chunk).
- what â€” documented the data-dir disk footprint (~3 GB: 840 MB bundle +
  ~2.2 GB generated XNNPACK cache + small upstream files).
  why â€” the generated cache was neither expected nor documented.
  verification â€” file sizes on disk after a real setup + first decode.
- what â€” investigated the LiteRT slowness before accepting it: tried
  `Backend.CPU(thread_count=32)`, `activation_data_type=FLOAT32` and the
  existing 2.2 GB XNNPACK weight cache. None changed the timings (4 tokens
  always 175-183 s). LiteRT's own benchmark counters place the cost inside
  the engine: prefill 0.167 tok/s, decode 0.0218 tok/s.
  why â€” rule out every knob we control before declaring the CPU fallback
  unusable. verification â€” three timed probe runs with per-chunk timestamps
  and a `BenchmarkInfo` dump (probe scripts, run on the reference laptop).
- what â€” `limite.lock.json` now always records `hf_model_revision` (the pin),
  instead of `null` when setup runs with a non-transformers backend.
  why â€” the lockfile must record ground truth; `null` read as "no pin".
  verification â€” unit tests green; the pin is a constant taken from config.
- what â€” `limite doctor` now shows the effective LiteRT bundle path with
  `present=True/False` instead of `bundle=None`.
  why â€” the old field showed only the explicit override and read as "nothing
  downloaded" while the bundle was in place. verification â€” doctor output on
  this machine shows the resolved path `present=True`.
- what â€” `RunConfig.resolved_data_dir` honors `LIMITE_DATA_DIR` (explicit
  `--data-dir` still wins). why â€” the Docker images set the variable and the
  integration tests honored it, but the runtime ignored it.
  verification â€” new `tests/test_config.py` (3 tests); 38 unit tests green.
- what â€” LiteRT `load()` raises the library log threshold to ERROR (guarded)
  and passes `thread_count=os.cpu_count()` to the CPU backend.
  why â€” every load printed benign noise (NPU accelerator not registered,
  optional bundle sections missing); threads are the documented knob even
  though measured flat on this machine. verification â€” startup warnings gone
  in probe logs; unit tests green.
- what â€” llama.cpp built in WSL2 (Ubuntu 24.04, gcc 13.3, cmake 3.28, CPU
  route, `JOBS=32`), via `scripts/build_llamacpp.sh`. The script's
  `limite-logits` target linked against `common`, which does not exist at
  the pinned revision 5836771 (the library is `llama-common`, shared).
  Fixed the link line in the script (`llama-common llama`), same as the
  imatrix target upstream. why â€” `limite validate` needs that binary.
  verification â€” `llama-server`, `llama-completion`, `limite-logits` built;
  `limite run` returns tokens from the Q8_0 GGUF through the llamacpp route.
- what â€” HF reference pin moved from e078b537 to main 9f4d1567 (config.py,
  README, ARCHITECTURE). why â€” e078b537 has weights and tokenizer only (no
  `configuration_limite.py` / `modeling_limite.py`, no `auto_map`), so
  `trust_remote_code=True` could not load the architecture (error: model
  type not recognized). Checked at the LFS level: `model.safetensors`
  (2070811520 bytes, same etag) and `tokenizer.json` are identical across the
  two revisions. `config.json` differs (adds `auto_map`, and the end-of-text
  id field differs); `chat_template.jinja` differs too. Our prompt rendering
  uses the conversion's template, so it does not depend on the HF one.
  verification â€” remote code downloads and loads; reference model loads on CPU.
- what â€” `transformers_backend.load()` reads the remote config and sets
  `use_cache=True` when it is missing. why â€” with transformers 5.19 the
  remote modeling code reads `config.use_cache`, which the remote config
  class does not define. verification â€” the forward pass starts (the error
  moved on to the next line).
- what â€” `save_report` converts numpy scalars. why â€” `json.dumps` failed on
  `int64` from the metric code after a full run had finished.
  verification â€” report written, `validation-report.json`.
- what â€” first end-to-end `limite validate` (llamacpp Q8_0 candidate vs
  transformers BF16 reference, 20 prompts, 16 tokens, CPU reference). Result:
  FAIL. Mean KL 0.995 (threshold 0.05), top-1 0.9375 (threshold 0.90),
  finite and cache checks pass. 17 of 20 prompts agree with the reference
  (KL about 1e-4 to 4e-3, top-1 1.0). Three prompts diverge at greedy
  positions 6, 10 and 12: "Solve: If x + 3 = 8, what is x?" (KL 10.71),
  "What is 144 divided by 12?" (KL 5.70), "Compute 17 squared." (KL 3.48).
  why â€” documented as found, not as expected. The cause is NOT determined:
  a near-tie flip in Q8_0 would not give KL this large, so a candidate-side
  difference (patch, tokenization, or numerics on specific contexts) is the
  working hypothesis. Next step: compare the logits at the first divergent
  position between the two runs and try an F16 GGUF as a control.
- what â€” CUDA build NOT done. The WSL distro has no CUDA toolkit (nvcc
  absent), and the Ubuntu package is too old for sm_120 (RTX 5090); the
  NVIDIA repo route is a multi-GB install not yet attempted. Status: the
  GPU route is open; the CPU llamacpp route works.
- open â€” `HF_WEIGHTS` hashes in `config.py` still describe e078b537
  (`config.json`, `chat_template.jinja`). `limite setup --backend transformers`
  will fail verification until they are recomputed at 9f4d1567. The
  transformers reference itself loads from the hub and is not affected.
- open â€” `limite setup` shows "0 files changed" for the patch diff before the
  checkout exists (`show_patch_diff` runs `git apply --stat` against a
  missing directory). Cosmetic; the build script shows the real stat.
- open â€” the LiteRT route is measured at about 40-50 s per token on the
  reference laptop (see 0.1.1 above). Unchanged.
- what â€” CUDA build of the patched llama.cpp in WSL2 (Ubuntu 24.04). Installed
  `cuda-toolkit-13-0` (13.0.88) from NVIDIA's WSL repository (same major as
  `docker/Dockerfile.cuda`), then ran `CUDA=ON JOBS=32 scripts/build_llamacpp.sh`
  on the existing patched checkout. why â€” the CPU build could not use the
  RTX 5090. Driver libcuda is taken from `/usr/lib/wsl/lib`; the runtime needs
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64`. verification â€”
  build contains `ARCHS = 1200` (sm_120); `--list-devices` shows
  `CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU`; same Q8_0 model, 128 tokens,
  temp 0: 149.7 tok/s on GPU vs 33.5 tok/s with `CUDA_VISIBLE_DEVICES=-1`.
- what â€” validation rerun on the CUDA build: mean KL 1.447, top-1 0.903, FAIL.
  Same three prompts diverge (the fourth, "Solve: 5x - 7 = 18", now diverges
  too). Cache-coherence check now fails on 3 prompts. why â€” to test whether
  the CPU-side numerics were the cause. Result: the failure is not
  CPU/CUDA-specific.
- what â€” diagnosis of the divergence on "Solve: If x + 3 = 8, what is x?"
  (script: HF reference in float32 path with `use_cache=True`, candidate via
  llama-server `/completion` with `n_probs=5`). Prompt tokens identical (51/51).
  Greedy steps 0-5 agree. At step 6 the reference picks token 13482 ("linear")
  and the candidate picks 23606 ("equation"); the TOP-5 SETS ARE IDENTICAL,
  only the order of the first two differs. So this is a near-tie flip, not a
  logit error. Once one token differs, the free-running trajectories diverge,
  and the validator compares logits of the two DIFFERENT prefixes: that is
  where KL 3-10 comes from. The metric is wrong for near-tie prompts; the
  candidate is not shown to be broken. Upstream reports Q8_0 KL 0.0072 and
  top-1 0.977 against the F32 source on its own prompts (validation.json), and
  our 17 agreeing prompts are consistent with that.
  Consequence: the current `limite validate` verdict (FAIL) is not a reliable
  quality signal. Needed fix (not yet implemented): teacher-force the
  reference greedy tokens into the candidate and compare logits position by
  position, and report free-running agreement separately. Cache-coherence
  failures probably have the same cause (no-cache recompute flips near-ties)
  but this is not verified.
- open â€” `limite validate` must not be read as a verdict until teacher forcing
  is implemented. The 0.1.1 FAIL entry above stands as a historical record.

## 0.1.2 â€” teacher-forced validation (2026-10-09)

- what â€” `limite-logits` gets `--forced IDS.bin` (raw int32 little-endian
  token ids). Step s is conditioned on the forced prefix instead of the
  candidate's own argmax, both in the cached and in the no-cache trajectory.
  `limite validate` now runs the candidate twice: free-running (informational,
  stored in `free_kl_mean` / `free_top1_agreement`) and teacher-forced on the
  reference greedy prefix (verdict). why â€” free-running comparison compares
  logits of different prefixes after the first near-tie flip, so KL 3-10 on
  three prompts was a metric artefact (see 0.1.1 divergence diagnosis).
  verification â€” tests `test_teacher_forcing.py` (file format round trip);
  live run below.
- what â€” cache-coherence rule under teacher forcing. A cached-vs-recompute
  argmax difference is accepted only if the recompute token is one of the two
  best cached tokens and the top-1/top-2 gap is at most
  `CACHE_FLIP_MAX_GAP = 0.1` logits. Strict argmax equality was rejected
  because the three flips observed were 0.034, 0.044 and 0.050 logits wide
  (same near-tie tokens as the reference), i.e. numerical noise from two
  computation orders. why â€” strict equality flagged numerical near-ties as
  incoherent. verification â€” `test_cache_rule.py` (4 cases, incl. a wide-gap
  flip that is still rejected and a flip outside the top 2).
- what â€” live result, llamacpp Q8_0 (CUDA build, RTX 5090) vs transformers
  BF16 reference, 20 prompts, N = 16, teacher-forced: mean KL 0.000557
  (threshold 0.05), top-1 0.9875 (threshold 0.90), finite, stop tokens ok,
  cache coherent on all 20 -> PASS. Per-prompt KL 0.00008 to 0.00356; the
  three former outliers are 0.00031-0.00072 with top-1 0.938.
  Not comparable to the upstream figure (KL 0.0072 vs the F32 source on the
  upstream prompt set): different reference and prompts.
  why â€” record the outcome of the corrected method, with the criteria change
  stated, not replacing the 0.1.1 FAIL.
- what â€” `HF_WEIGHTS` recomputed at HF_MODEL_REVISION 9f4d1567. Four files
  differ from the conversion's copies: config.json, generation_config.json,
  chat_template.jinja, tokenizer_config.json. model.safetensors and
  tokenizer.json, special_tokens_map.json are unchanged. The old
  model.safetensors hash (9dc7e89e...) is the one main serves, which confirms
  the weights match the pinned conversion source. why â€” `limite setup
  --backend transformers` could not verify the new pin. verification â€”
  `limite setup --backend transformers` on a clean data dir: every file
  verified, exit 0.
- what â€” `show_patch_diff` runs `git apply --stat` outside any repository
  when the llama.cpp checkout does not exist. why â€” inside the workspace git
  repo it printed "0 files changed"; the patch itself has 7 files, 222
  insertions. verification â€” output for the real patch in WSL lists all 7.
- what â€” `transformers` reference now uses `AutoConfig` and sets
  `use_cache=True` when missing (see 0.1.1). unchanged here.
- open â€” teacher forcing is implemented for the llamacpp candidate only. The
  transformers candidate (used only in tests) and LiteRT keep their previous
  paths. LiteRT has no full-vocabulary logits (see 0.1.1), so its KL stays n/a.
- open â€” 20 prompts x 16 tokens is short. A longer N and a second reference
  quantization (e.g. F16 GGUF) would separate quantization error from graph
  error; neither was run.

## 0.1.3 â€” robustness fixes (2026-10-12)

- Prompt-format experiment on the running WSL CUDA Q8_0 server: compared
  the original rendered template, the same chat delimiters without its
  system turn, and raw text for "Give me an explanation of Taylor Series."
  Added `scripts/compare_prompt_modes.py`; results and exact prompts are in
  `experiments/template-comparison-2026-10-12.json` and readable `.txt`.
  All calls used the raw `/completion` endpoint, no history, cache_prompt
  false, temperature 0, top_p 0.95, seed 42, and a 2048-token cap. The
  original and no-system runs both reached the cap (23.846 s and 23.411 s);
  the no-system output still reproduced the step-by-step/boxed instruction
  although it was absent from its submitted prompt. Raw text also reached
  the cap (25.026 s) and degenerated into repeated Taylor-series sentences.
  Removing the system did not resolve verbosity in this single greedy
  example. This does not establish a general causal explanation or an
  optimal prompt configuration; the supported template and live browser
  configuration were not changed by the experiment.

- Browser chat compatibility: added `--skip-chat-parsing` to the reference
  GPU launcher. The native UI triggered automatic parser generation with
  tool-call probes, which the original Limite template rejects. This flag
  leaves prompt rendering intact and uses a pure content output parser;
  tools remain unsupported and reasoning tags remain in the text.
  Browser inspection also found two built-in Browser tools selected in the
  UI defaults. These send real tool definitions and still fail rendering;
  disabled Browser under Settings > Tools and documented starting a new
  chat after saving, because old conversations keep their own tool picks.
  Verification: submitted "What is the derivative of x^2 with respect to
  x?" from the actual browser UI; generation completed with boxed 2x,
  561 tokens in 5.0 s (UI-reported 112.86 tok/s), without a tool/parser
  exception. An Italian probe generated excessive ambiguous reasoning and
  was interrupted; this verifies UI compatibility, not language quality.

- Added `Avvia-Limite-GPU.cmd` for the reference Windows/WSL2 environment:
  starts the existing CUDA llama-server with Q8_0 on localhost:8080, or
  opens the chat when a healthy local server already exists. Documented
  the built-in browser chat and the single-turn usage. Verification:
  CUDA CLI returned `\\boxed{918}` for 27 times 34; the GPU server started
  with `-ngl 99`, and GET `/` returned HTML (200), `/health` returned `ok`.
  The launcher paths are specific to this PC; browser chat uses the native
  server rather than the Python frontend.

- what â€” `docker/entrypoint.sh` is now registered executable: the git index
  carries mode `100755` (`git update-index --chmod=+x`) and both Dockerfiles
  force `RUN chmod 0755 /entrypoint.sh` after the COPY. Also `limite setup
  --no-build` no longer asks for confirmation: the "apply patch and build?"
  prompt is shown only when a build was actually requested; with
  `--no-build` it prints that nothing was applied or built and exits 0.
  why â€” Docker started `entrypoint.sh` directly, which failed on checkouts
  where the exec bit was lost (Windows working tree); and the entrypoint
  runs `setup --no-build` non-interactively, so the unconditional
  confirmation aborted container startup without stdin.
  verification â€” `git ls-files -s docker/entrypoint.sh` â†’ `100755`; new
  `tests/test_cli.py`: `--no-build` exits 0 without prompting, plain
  `setup` still aborts without stdin; CLI smoke via `typer.testing`.
- what â€” `/v1/*` sampling parameters are now validated, and the token-limit
  precedence is explicit: `max_completion_tokens` (current OpenAI field) >
  `max_tokens` (deprecated) > server default; explicit values must be â‰¥ 1,
  `temperature` within [0, 2], `top_p` within (0, 1]. Violations return 400
  with the offending parameter named.
  why â€” `/v1/completions` with `max_tokens=0` returned 500 (the old
  `or`-chain fell through to `req.max_completion_tokens`, which
  `CompletionRequest` does not even have); negative limits and out-of-range
  sampling were silently accepted.
  verification â€” new server tests: max_tokens=0 / negative / zero
  max_completion_tokens â†’ 400; explicit 9 and 11 passed through (no
  `or`-fallback to the default 512); both limits sent â†’ 7 wins; invalid
  temperature/top_p â†’ 400. Suite green.
- what â€” LiteRT streaming stop handling rewritten as an incremental filter
  (`_StopFilter`): it emits only text not yet emitted, holds back any
  trailing suffix that could be the beginning of a stop string, cuts at the
  earliest stop and flushes the held tail when the stream ends without a
  stop.
  why â€” the old loop re-emitted the whole accumulator when a stop arrived
  (reproduced as `abcdef` â†’ `abcdefabcdef`), and a stop marker split across
  chunks leaked its prefix into the output.
  verification â€” new `tests/test_litert_stop.py` (10 cases: no
  re-emission, split marker, hold-and-release, earliest stop wins, flush,
  empty/no stops); live probe on the reference laptop against the real
  engine: both the custom-stop and default-stop cases produce
  non-self-duplicated output.
- what â€” llama-server's combined stdout/stderr pipe is now drained
  continuously by a daemon reader thread (`_drain_logs`); a bounded
  ring buffer (last 200 lines) feeds error reports and debug logging.
  Every startup failure path â€” server exit and readiness timeout â€” calls
  `close()` before raising, tearing down the subprocess, the HTTP client
  and the reader; `close()` also joins the reader thread.
  why â€” nothing read the pipe while the server ran: a full pipe buffer
  blocks llama-server on its own logging; the readiness timeout raised
  without closing the process or the pipe.
  verification â€” new `tests/test_llamacpp_logs.py`: 500-line output fully
  drained with the tail kept; a fake immediately-dying server raises
  "exited during load" and leaves `_proc`/`_client` clean with the real
  output (`boom`) in the log tail; `close()` joins the reader.
- what â€” `check_stop_tokens` now returns a tri-state `StopTokenCheck`
  (`verified` / `failed` / `unverifiable`) with an evidence list, and takes
  the candidate's free-running token trajectory to record an actual
  stop-token emission (or note its absence as informational). The report
  gains `stop_tokens_status` + `stop_tokens_evidence`; the table prints
  `stop=verified|failed|unverifiable` and lists the evidence. The verdict
  (`passed`) still fails only on `failed` â€” `unverifiable` is reported
  honestly instead of silently counting as a pass.
  why â€” the old bool could return True without any EOS declared anywhere
  (e.g. a backend with neither tokenizer nor engine EOS ids): nothing was
  verified but the check "passed".
  verification â€” new `tests/test_stop_check.py` (8 cases: verified/failed/
  unverifiable per side, EOS-check error, actual-stop evidence,
  informational absence, both sides listed). Full unit suite green
  (75 passed, `tests/test_integration.py` excluded: live LiteRT decode
  ~40-50 s/token, impractical by design â€” see Known limits).
- what â€” README quickstart rewritten: `uv sync --extra litert` (the LiteRT
  extra was never installed by it), all commands through `uv run` (the
  `limite` script is inside the unactivated `.venv`), and explicit
  `--backend litert` on `run`/`serve` (auto-selection would pick
  `llamacpp` on GPU/Metal machines right after a LiteRT-only setup).
  Documented the validated sampling ranges and the token-limit precedence
  in "Model constraints"; documented the tri-state stop check in
  "Validation". ARCHITECTURE.md updated accordingly.
  why â€” the quickstart produced a working install but a failing first run
  on a stock machine (`limite: command not found` / missing litert extra /
  wrong backend choice on GPUs).
  verification â€” commands match the CLI surface (`--backend`, `--stream`)
  and the extras declared in pyproject.
- what â€” follow-ups on the robustness batch above (same release):
  `/health` now answers 200 `ok` only when `backend.info().available` is
  true; otherwise 503 with `status="unavailable"` and the backend's own
  error. All streaming endpoints consume the new
  `Backend.stream_full()` (yields `StreamChunk` pieces; the last carries
  the observed finish reason): the final SSE chunk propagates `stop` or
  `length` instead of a hardcoded `"stop"`. llamacpp derives it from the
  final SSE event's `truncated` flag; transformers from the generated-id
  count; LiteRT observes the session's benchmark counter
  (`get_benchmark_info().last_decode_token_count`, fallback: re-tokenize
  the emitted text) because `run_decode_async` ends silently both on EOS
  and on the token cap (the internal "Max number of tokens reached" error
  is swallowed by the public iterator); `LiteRTBackend.generate` no longer
  hardcodes `finish_reason="stop"`. The plain `stream()` text contract is
  unchanged and now wraps `stream_full` where the backend knows more.
  why â€” readiness reported ok with an unusable backend (reproduced), and
  every streamed completion declared "stop" even when the generation hit
  the token cap, so downstream consumers could not tell the two apart.
  verification â€” new tests: health 503 (unavailable backend); streamed
  chat/completions finish_reason "length" end-to-end; non-streaming
  reason passthrough; LiteRT stream_full unit tests on fake engine/session
  (length via counters, stop on early halt, stop-string hit, tokenize
  fallback, session closed); suite green (91 passed).
- what â€” follow-up on the tri-state stop check: the free-running
  trajectory's stop-token emission is now recorded explicitly as "an
  emission, not by itself a proof of halt", and `check_stop_tokens` takes
  `actual_stop` â€” the finish reason observed on a real free-running
  candidate generation (dedicated 128-token greedy probe on llamacpp;
  captured from the already-run degraded-path generation when it halted
  before the cap). `stop` = the engine halted on its own stop condition
  (actual-stop evidence); `length` = cap hit, informational only. A PASS
  whose components were not fully checkable (stop declarations
  unverifiable and/or KL n/a) sets `ValidationReport.partial=True` and the
  table renders "PASS (partial: KL n/a; stop-token check unverifiable)".
  why â€” emitting EOS within N tokens does not demonstrate that the engine
  halted, and an unverifiable stop check still yielding a plain PASS read
  as a complete verification.
  verification â€” new tests: emission wording (no halt-proof claim),
  actual_stop=stop / length / absent evidence lines, partial-pass table
  rendering listing only the missing components; suite green (91 passed).
- what â€” follow-ups on the follow-ups (still 0.1.3):
  (1) `/health` now distinguishes backend availability from engine
  readiness via the new `Backend.is_ready()` (default: ready; llamacpp =
  subprocess alive, litert = engine loaded, transformers = model loaded).
  Statuses: 200 `ok` (+ `ready: true`) only when available AND ready;
  503 `unavailable` (binaries/libraries missing) and 503 `not_ready`
  (present but the engine is down, e.g. llama-server built yet not
  started). (2) The llama.cpp finish-reason mapping was read from the
  pinned revision's server source (fetched at `LLAMACPP_COMMIT`,
  `tools/server/server-task.cpp` + `server-context.cpp`) instead of
  guessed: the generation cap (n_predict) ends with
  `stop_type: "limit"` and `truncated: false` â€” `truncated` alone means
  context overflow (with `--no-context-shift` it accompanies
  `stop_type: "limit"`); EOS/stop-word end with `"eos"`/`"word"`; this
  revision emits no `stopped_*` booleans. `_finish_reason` implements
  exactly that schema and both `generate()` and `stream_full()` use it.
  why â€” `/health` still answered 200 "ok" with llama-server not running
  (available only said the binary exists), and a capped generation was
  mislabeled "stop" (truncated=false), which would also have produced a
  false actual-stop line in the validation probe.
  verification â€” new `tests/test_llamacpp_finish.py`: stop_type
  limit/truncated=false â†’ length (the reproduced case), context overflow â†’
  length, eos/word â†’ stop, missing fields â†’ stop; `generate()` and
  `stream_full()` against a mocked server returning a limit event â†’
  length; `is_ready` across the subprocess lifecycle. Server test:
  available-but-not-ready â†’ 503 `not_ready`. Suite green (100 passed).
- what â€” clean-environment Docker release verification (CPU image):
  `docker build --no-cache` (the log shows the `RUN chmod 0755
  /entrypoint.sh` step), in-image checks (`/entrypoint.sh` is
  `-rwxr-xr-x`, `llama-server --version` reports the pinned
  `commit 58367713a`), then a fresh-volume run with
  `LIMITE_BACKEND=llamacpp`: verified download of the Q8_0 GGUF
  (1.1 GB, SHA256 ok), `--no-build` without any prompt, uvicorn serving.
  Real-model end-to-end results through the new code paths: capped
  8-token `/v1/completions` â†’ real `stop_type: "limit"` event â†’
  `finish_reason: "length"`; chat completion â†’ EOS at 490/512 tokens â†’
  `finish_reason: "stop"` with a correct answer (`\boxed{918}` for
  27Ã—34); SSE stream ends with the observed reason; tools /
  max_tokens=0 / temperature=2.5 all rejected with 400; killing the
  llama-server process inside the container flips `/health` to 503
  `not_ready` (running=False) and the next generation reloads the engine
  automatically (health back to 200 ok).
  The clean run also caught a real bug: `setup --no-build` crashed in the
  runtime container (`FileNotFoundError: 'git'` â€” the final image stage
  ships no git) because `show_patch_diff` ran unconditionally. Fixed: the
  patch diff (and its confirm prompt) is shown only when a build was
  requested, and `show_patch_diff` tolerates a missing git binary
  ("git not available for --stat").
  why â€” the release claim "runs from a clean container" had never been
  exercised end-to-end against the real model.
  verification â€” this run; 2 new CLI tests (no-build does not call
  `show_patch_diff`; missing-git diff degrades to a note). Suite green
  (102 passed).

## 0.1.4 â€” public beta preparation (2026-10-12)

Release goal: a stranger can install and use the declared supported route
(Windows + WSL2 + CUDA) without editing code or copying personal paths.

- what â€” `Avvia-Limite-GPU.cmd` rewritten as a portable launcher: no personal
  paths anywhere, configuration via environment variable > `limite-launcher.env`
  (git-ignored, `.example` shipped) > built-in default, every path resolved
  from the script location (`%~dp0`), paths with spaces supported, `--check`
  mode runs only the requirement checks. Preflight errors are actionable
  (WSL2 missing, distribution missing â€” with `wsl -l -v` â€”, script missing)
  and never trigger downloads or rebuilds. When a healthy server already
  answers on the configured port it is reused; otherwise the launcher starts
  the server in the foreground and a helper opens the browser when `/health`
  turns `ok`. why â€” the previous launcher hard-coded `/home/<user>/...` paths
  of the reference PC and only worked from that machine. verification â€”
  `tests/test_launcher.py` (no personal paths, defaults, `%~dp0`, env example
  fully commented, gitignore entry) + live runs of `--check` from the repo
  (path with spaces) on Windows 11/WSL2.
- what â€” `scripts/limite-serve.sh`, the Linux/WSL side of the launcher: checks
  data dir, GGUF model, chat template, patched llama-server and the `limite`
  entry point (resolution order: `LIMITE_BIN` > `<LIMITE_WSL_REPO>/.venv/bin/limite`
  > `uv run --project` > PATH > `~/.local/bin`), each failure printing how to
  fix it, then `exec`s `limite serve --backend llamacpp` with `LIMITE_HOST/
  PORT/GPU/MAX_TOKENS/...`. `--check` prints the effective configuration and
  stops. why â€” the browser chat must go through the runner's preprocessing,
  and missing requirements must be explained before anything starts.
  verification â€” live `--check` and full start in WSL2 (GPU, `-ngl 99`);
  `tests/test_launcher.py::test_check_mode_*` exercise missing distribution
  and missing data dir with spaces through the real `.cmd`.
- what â€” bundled single-turn chat page (`src/limite_runner/server/static/chat.html`)
  served by the runner at `/`: it keeps the conversation visible in the
  browser but sends **only the latest user question** (`messages: [{role:
  "user", content: question}]`), no tools/MCP/system-prompt controls (the page
  states they are disabled), generation cap defaulted to the server default
  and adjustable per message, Stop button (AbortController), and an explicit
  end state per answer: concluded (`stop`), "token limit reached: truncated
  and possibly incomplete" (`length`), interrupted, error â€” a truncated answer
  is never shown as complete. why â€” the native llama-server web UI re-sends
  history, ships Browser tools and generates without a cap; those were real
  problems observed on the reference PC (see 0.1.3). verification â€” new
  `tests/test_chat_ui.py` (7 tests: payload carries one message and no
  tools/system, no tool controls, 2048 default + override, all four end
  states, stop button) and live requests against the GPU server (below).
- what â€” default generation cap is now `DEFAULT_MAX_TOKENS = 2048`
  (`config.py`) everywhere: `RunConfig.max_tokens`, `create_app`/`serve`
  defaults, `limite run`/`serve --max-tokens` (was 512/256). Configurable per
  request (`max_completion_tokens` > `max_tokens` > server default), per
  server (`--max-tokens`), via `LIMITE_MAX_TOKENS`, and in the chat page.
  New env defaults for `limite serve`: `LIMITE_HOST`, `LIMITE_PORT`;
  `LIMITE_NGL` for `--ngl`; `LIMITE_MODEL` overrides the data-dir-derived
  GGUF path in `_load_backend`. Precedence (flag > env > default) documented
  in the CLI docstring and README. why â€” the interface must always have a
  real cap and the launcher must not need code edits to configure itself.
  verification â€” `tests/test_server.py::test_default_max_tokens_when_absent`
  (2048), `tests/test_cli.py` env helper + `LIMITE_MODEL` with spaces, live
  `max_completion_tokens=64/128/256` runs returning `finish_reason=length`.
- what â€” `limite run` reports the generation end on stderr
  (`[finish_reason=stop] answer concluded.` / `[finish_reason=length] token
  limit reached: the answer is truncated and may be incomplete.` /
  `[interrupted]` with exit 130 on Ctrl+C / `[error]` with exit 1), streaming
  and non-streaming. Missing requirements (template, model, llama-server)
  print one actionable ERROR line and exit 1 instead of a traceback â€”
  reproduced from a clean wheel install with an empty data dir. `limite doctor`
  now shows the effective data dir, `LIMITE_MODEL` presence and the expected
  template path. why â€” clean-environment testing crashed with raw
  `FileNotFoundError` before any hint. verification â€” new tests
  (`test_missing_template_error_is_actionable`,
  `test_missing_model_error_surfaces_as_cli_error`,
  `test_env_int_and_finish_note`, `test_missing_server_error_is_actionable`)
  and the clean-venv run below.
- what â€” `limite setup` build script resolution (`_build_script`) checks the
  package-data copy first (`limite_runner/data/build_llamacpp.sh`, shipped in
  the wheel via hatch `force-include` together with `tools/limite-logits.cpp`),
  then the source checkout; `build_llamacpp.sh` finds `limite-logits.cpp` next
  to itself or in `tools/`, and takes the patch from `LIMITE_DATA_DIR` before
  the default. No download is ever attempted for these files. why â€” an
  install outside the source tree had no way to reach `scripts/`+`tools/`.
  verification â€” wheel contents listing (both files present), `_build_script`
  resolves inside `site-packages` from a clean venv, and a live
  `scripts/build_llamacpp.sh` run against the WSL checkout with a custom
  `LIMITE_DATA_DIR` (path with spaces) rebuilt all binaries including
  `limite-logits`.
- what â€” package version aligned to `0.1.4` (pyproject `0.1.0` and
  `__version__` lagged behind the CHANGELOG's 0.1.3), FastAPI app version now
  reads `__version__`; README/ARCHITECTURE/CHANGELOG/RELEASE.md updated.
  why â€” a public release must not ship contradictory version numbers.
  verification â€” installed wheel reports `0.1.4` (`limite_runner.__version__`).
- what â€” verification campaign for this release (commands in RELEASE.md):
  (1) full unit suite green (128 tests, live LiteRT decode tests excluded); (2) live GPU run through the new launcher path:
  chat page served at `/` (2048 default, one-message payload), two consecutive
  questions where the second answer mentions only the second question, a
  client re-sending history dropped with server warnings and no leak, tools
  â†’ 400, caps at 64/128/256 tokens â†’ `finish_reason=length`, a concluded
  answer â†’ `finish_reason=stop` (405 tokens, `\boxed{918}`), a mid-stream
  abort leaves `/health` ok and the next request works; (3) build + start
  with a custom data dir containing spaces (model/template symlinked, patch
  copied): build verified the patch SHA from that dir, server health reports
  the spaced model path, generation concluded `\boxed{144}`; (4) clean-wheel
  install in a venv outside the source tree in a path with spaces: `limite
  doctor`/`run`/`serve` behave as documented and the build script resolves
  from package data; (5) `Avvia-Limite-GPU.cmd --check` end to end.
  why â€” the release criterion is "another user can install and use the
  declared route without modifying code or copying personal paths".
  verification â€” see RELEASE.md for the exact commands and environment.
- open â€” the CUDA Docker image is built but not verified (the verified GPU
  route is WSL2); macOS/bare-Linux/Windows-native routes remain unverified;
  LiteRT stays ~40-50 s/token on CPU; `limite validate` at N > 16 not run;
  WSL distribution names with spaces cannot be passed through this `wsl.exe`
  (documented in Troubleshooting).
- what — pre-release review fixes (same 0.1.4 batch):
  (1) **Launcher port identity.** `/health` now returns
  `service: "limite-runner"` + `version`; `Avvia-Limite-GPU.cmd` reuses the
  port only when that identity is present (probe exit codes 0/2/3/1) and
  exits with a clear error when another service answers there (the old
  native llama-server chat answers `{"status":"ok"}` and must never be
  mistaken for the runner — it has no single-turn preprocessing).
  why — the old check accepted any `status: "ok"`, so the launcher would
  reopen the wrong chat. verification — live: the pre-existing llama-server
  on 8080 is now refused with "la porta 8080 risponde ma NON e' limite-runner"
  and exit 1; a real runner on 8089 answers `service=limite-runner` and the
  probe returns 0. New tests: `test_health_identifies_the_runner`.
  (2) **No fabricated "stop" on incomplete streams.** A llama-server stream
  that ends without the final `stop` event now raises ("stream ended without
  a final event: the completion is incomplete") instead of yielding
  `finish_reason="stop"`; the server/CLI fallbacks report `finish_reason:
  "unverified"` (chat page: "NON è verificata"; CLI: exit 1) instead of the
  hardcoded "stop"; the base `stream_full` wrapper (text-only backends) and
  LiteRT's unobserved end also report "unverified". Error events with an
  empty message are replaced by the exception type (`{"error":
  "AssertionError"}`), so errors stay distinguishable. why — a stream that
  ends after some text with no end event used to read as concluded.
  verification — live mid-stream kill of llama-server (precise PID) delivers
  the SSE `{"error": ...}` event with no finish_reason and the runner stays
  up (503 `not_ready` on the next `/health`, honest); new tests
  `test_stream_without_final_event_is_not_concluded`,
  `test_stream_full_reports_unverified_when_nothing_was_observed`,
  `test_streaming_default_wrapper_reports_unverified`,
  `test_streaming_without_terminal_status_is_unverified`,
  `test_error_with_empty_message_is_still_distinguishable`.
  (3) **`setup --data-dir` reaches the build.** The build subprocess now
  receives `--patch <data-dir>/conv/runtime/limite.patch` (when present) and
  `LIMITE_DATA_DIR=<data-dir>` in its environment, so a clean install with
  `--data-dir` never searches for the patch elsewhere. why — before, only
  the download used the requested dir; the build inherited the caller's env.
  verification — `test_setup_build_receives_patch_and_data_dir` captures the
  exact cmd and env.
  (4) **`LIMITE_QUANT` / `LIMITE_CTX` honored by the direct CLI.** `_cfg`
  resolves quant and ctx from the environment when the flags are absent
  (flag > env > default), for `run`/`serve`/`setup`/`validate`; `--ctx`
  options became optional so the env value is visible. why — the README
  declared these variables but only the launcher translated them into flags;
  direct `limite run/serve` showed quant=None ctx=16384. verification —
  live: `LIMITE_QUANT=Q5_K_M LIMITE_CTX=32768` now resolves `quant=Q5_K_M
  ctx=32768`; tests `test_cfg_honors_limite_quant_and_limite_ctx`,
  `test_cfg_flags_beat_env`, `test_cfg_rejects_invalid_ctx`.
  (5) `experiments/template-comparison-2026-10-12.json` anonimized
  (`/home/antonio_proietti/` → `/home/<user>/`); the personal-path test now
  also scans `scripts/` and `experiments/` so nothing personal can return.
- what — launcher probe rewritten as `scripts/health-probe.ps1` (Windows
  PowerShell 5.1, same `powershell.exe` the launcher uses) with a 5-way
  contract instead of "any `status: ok`": `0` runner ready, `2` runner
  present but not ready (identity also read on HTTP 503/4xx), `3` another
  service answers (any HTTP response without `service: "limite-runner"`,
  including 404 and a bare `{"status":"ok"}`), `1` unreachable (connection
  refused: the port looks free), `4` timeout/indeterminate. The launcher
  reuses only on 0/2, refuses on 3, starts only on 1 and refuses to start on
  4 ("un timeout NON dimostra che la porta sia libera"). Transport failures
  are classified from `WebExceptionStatus`/`SocketErrorCode` enums, never
  from localized message text (a refused connection on an Italian Windows
  reads "rifiuto della connessione" and was being misread as uncertain).
  New `--probe` mode runs the probe alone and returns its code.
  why — the previous probe threw on every HTTP error status (so a 503 runner
  and a 404 from another server were conflated with "no server") and any
  timeout would have triggered an automatic start on a possibly busy port.
  verification — new behavioural tests in `tests/test_launcher.py` run the
  real launcher probe path (`Avvia-Limite-GPU.cmd --probe` -> `powershell.exe
  scripts/health-probe.ps1`) against local servers reproducing 200 runner,
  503 runner, 200 without identity (the old llama-server case), 404, a
  never-answering timeout and a refused connection (7 tests, including the
  powershell.exe/5.1-compatibility contract). Live: the real native
  llama-server on 8080 -> exit 3; a free port -> exit 1. Suite green
  (146 passed).
- what — health-probe exit 1 now requires an explicit
  `SocketError.ConnectionRefused` anywhere in the inner-exception chain.
  DNS failures, generic `ConnectFailure`, timeout and other socket errors
  return 4 and prevent automatic startup. why — an unresolved hostname
  previously returned 1 even though it gave no evidence of a free port.
  verification — five regression cases cover the real launcher with an
  unresolved `.invalid` hostname and controlled Windows PowerShell transport
  exceptions (generic failure, HostNotFound, nested refusal, TimedOut).
  Full suite: 151 passed (`pytest tests -q --ignore=tests/test_integration.py`).
- what — README prepared for a public community beta: practical purpose and
  upstream credits now lead the document, with links to the official serving
  alternatives. Repaired corrupted UTF-8 text, documented the required
  `LIMITE_WSL_REPO` launcher configuration, distinguished recorded platform
  checks from unverified routes, and aligned the Docker CPU example with the
  tested llama.cpp backend. Simplified the validation explanation and removed
  speculative cloud deployment framing. why — a new reader should understand
  who contributed each component and how to reproduce the supported setup.
  verification — documentation checked against launcher/backend behavior,
  local links, section anchors, Markdown fences and encoding. Documentation
  only; no runtime changes or new model/build qualification.
- what — expanded `.gitignore` to cover the downloaded llama.cpp checkout,
  model weights, local artifact stores, logs, environment files and Python
  tooling caches. Kept `uv.lock`, configuration examples, `tests/data` and
  reviewed experiment reports eligible for version control. why — preparing
  a public repository should not accidentally include local runtime artifacts
  or private configuration. verification — checked representative paths with
  `git check-ignore` and reviewed the tracked file list; no files removed.
- what — the Windows launcher now opens a fresh chat document URL using a
  `limite-runner` query parameter in every browser-opening branch, including
  reuse of a running server. why — native llama-ui cached at localhost:8080
  could reappear and call incompatible endpoints, reporting 404 / Server
  unavailable even while the runner was healthy. verification — executed
  the actual launcher against the runner on port 8080, opened its generated
  URL in the in-app browser, and completed "What is 2 + 2?" with answer 4
  and the UI status "Risposta conclusa". No port change or manual cache
  clearing is needed for subsequent launcher openings.
- what — launcher wired to the existing installation via the local
  `limite-launcher.env` (git-ignored): `LIMITE_LLAMA_SERVER` ->
  `~/limite-build/tools/llama.cpp/build/bin/llama-server`, `LIMITE_BIN` ->
  the WSL venv install, `LIMITE_DATA_DIR` -> the downloaded artifacts.
  `Avvia-Limite-GPU.cmd --check` and a full launcher start were verified
  against that installation with no download and no build.
  why — the launcher had no local configuration for this machine yet.
  verification — `--check` exit 0; launcher start -> probe 0 on 8080
  (service=limite-runner, ready), chat UI served, real generation concluded
  (`\boxed{918}`, finish=stop).
- what — restored the CUDA build that an earlier verification step had
  silently disabled: the custom-data-dir build check (0.1.4 item "build +
  start with a custom data dir") ran `build_llamacpp.sh` WITHOUT `CUDA=ON`,
  and the script's cmake line flips the existing tree to
  `GGML_CUDA=BOOL=OFF`, overwriting the CUDA `llama-server` executable.
  Detected via `ldd` (no CUDA libs), `llama-server --list-devices` (empty)
  and a measured ~26 tok/s generation. Restored with
  `CUDA=ON JOBS=32 bash scripts/build_llamacpp.sh` on the existing checkout
  (incremental, no download): `libggml-cuda/libcudart/libcublas` linked,
  `--list-devices` shows `CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU`,
  `GGML_CUDA:BOOL=ON`, and the same launcher start now decodes at
  ~98 tok/s end-to-end (405 tokens in 4.1 s) instead of ~26.
  why — the 0.1.4 verification note claimed a plain "rebuild"; the side
  effect on the GPU configuration was real and must not be left undocumented.
  verification — before/after evidence above; launcher start re-verified
  after the restore (probe 0, finish=stop, `\boxed{918}`).
- what — the serve/launcher terminal no longer floods with probe noise.
  A leftover tab of the native llama-server UI (and similar foreign clients)
  polls `/props` in a tight retry loop, and other probes hit `/sw.js`,
  `/build.json`, `/tools`, `/v1/streams/lookup`; the chat page itself polls
  `/health` every 5 s for its readiness badge. Every one of these landed in
  the launcher window as an access-log line and buried the real output
  (reported as "launcher prints errors"). `serve` now attaches
  `_LegacyUiNoiseFilter` to the `uvicorn.access` logger: those poll/probe
  paths are still answered (and `/props` gets an explicit 404 hint pointing
  at `/`, the supported chat) but no longer logged; real traffic (`/`,
  `/v1/models`, `/v1/chat/completions`, `/v1/completions`) stays visible,
  and DEBUG still shows everything. why — a terminal full of 404 lines reads
  as a broken server even when every request is served correctly.
  verification — new tests (`/props` 404 with hint; filter drops the six
  probe paths and keeps real traffic); live: 60+ probe requests (10 per
  path, plus POST variants) against the running server leave ZERO lines in
  the serve log, while `GET /` and `POST /v1/chat/completions` log normally.


## Unreleased — adapted original chat UI (2026-10-12)

- Restore the original llama.cpp/Svelte interface through a runner-specific
  transport: local history, Markdown/KaTeX and collapsible reasoning; only the
  newest text question reaches the API. Tools, MCP, system customization,
  pre-encoding and native replay/control are disabled.
- Serve the compiled bundle at `/`, configuration at `/ui/props`, and preserve
  the lightweight UI at `/simple`. Ship upstream MIT license and provenance.
- Persist observed stop/length/interruption notices; missing terminal events
  and SSE errors are failures, never invented successful completions.
- Verification: 154 Python tests, 8 dedicated transport tests, Svelte check;
  live CUDA browser checks for distinct consecutive answers and Stop.
