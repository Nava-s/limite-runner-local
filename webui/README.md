# Limite Runner chat UI

Adapted from `tools/ui` in ggml-org/llama.cpp at
`58367713a6935c0810103378144008df32e3d5db`. Upstream copyright and MIT
license: [LICENSE.upstream](LICENSE.upstream). The original developer
reference is preserved in [README.upstream.md](README.upstream.md); its
full feature list does **not** describe this adaptation.

The original layout, local IndexedDB conversations, Markdown/KaTeX,
collapsible reasoning, copying and regeneration are retained.
The transport sends only the latest text question to the Python runner.
Tools, MCP, custom system prompts, native replay/control, automatic title
inference and conversation pre-encoding are disabled. No service worker is
registered. No remote CDN is needed at runtime.

## Build and verification

Use Node 22.18+ (verified here with Node 26) and the committed package lock:

```sh
npm ci
npm run check
node --test runner-chat.test.mjs
npm run build
```

Run checks and build sequentially. Build output belongs in
`../src/limite_runner/server/static/llama-ui` and is shipped in the Python
wheel; end users do not need Node. `LLAMA_UI_OUT_DIR` can override the path.
Source changes require rebuilding the bundle before packaging Python.
Upstream tests are retained as reference; the runner-specific transport tests
above are the tests executed for this adaptation.

The visible end-state note is saved with each assistant message, so reloading
history does not turn truncated/interrupted output into a complete answer.
Browser history is local to the origin (host and port); changing port does not
move that history. Generation does not resume after a reload.
