# Running Morph on CPI

[Morph](../../agentic-app-builder) is a generative-UI runtime whose generation is a
series of **forced tool calls** that must return JSON matching a schema. CPI now
*speaks* that protocol over its OpenAI-compatible endpoint, so Morph can connect to
CPI as a "Local (OpenAI-compatible)" provider.

## What shipped (Node API layer)

1. **Tool / function calling emulation** (`web/server/openai_compat.mjs`)
   - `/v1/chat/completions` previously rejected `tools` / `tool_choice` outright.
     It now accepts them. When a tool is forced (`tool_choice:{type:"function",…}`,
     or any single tool with `auto`), the tool's JSON-Schema `parameters` are
     injected as a developer instruction ("respond with ONLY a JSON object matching
     this schema"), the model generates, and the output is returned as a proper
     `tool_calls` response with `finish_reason:"tool_calls"`.
   - `extractJsonObject()` tolerantly pulls the JSON out of the model text (strips
     ```` ```json ```` fences, takes the first balanced `{…}`).
   - Non-tool requests are completely unchanged (zero regression).

2. **Single-engine request queuing** (`web/server/index.mjs`, `guardOpenAiIdle`)
   - Concurrent `/v1` requests used to fail immediately with `engine_busy`. They now
     **wait** (up to `waitForIdleMs`, default 120 s) for the in-flight generation to
     finish. This lets Morph's parallel section agents (`Promise.all` of N calls)
     serialize through the single engine instead of failing.

Both changes are covered by a transform unit test (request→instruction, response→
tool_calls, JSON extraction, no-regression) — 13 assertions, run with:
`node web/server/_tooltest.mjs` (see git history for the test body).

## How to point Morph at CPI

1. Start CPI with a capable model: `start_local` (API on `http://localhost:3001`).
2. In Morph's **Add key** modal → **Local** tab:
   - Base URL: `http://localhost:3001/v1`
   - Model: your CPI model id (from `GET /api/models`)
   - Key: anything (ignored locally)
3. Morph's key-checker (`/api/live/check-key`) will immediately tell you whether the
   model returns a valid tool/JSON response.

## What still limits quality (engine-side, NOT shipped here)

The API now *accepts* Morph and returns tool calls, but the **JSON quality is entirely
the model's job**. For Morph to render real canvases you still need, in priority order:

1. **Constrained / grammar-guided decoding.** Today `response_format`/tool schemas are
   only a *prompt hint*. Without grammar-constrained sampling, small models emit
   malformed JSON and `tool_calls.arguments` won't parse. This is the single biggest
   reliability lever.
2. **A capable model.** CPI is a Llama-2-architecture engine; its bundled tiny models
   (TinyLlama, CPT-tiny) cannot produce strict nested JSON. Use a strong Llama-2-7B+
   instruct / function-calling fine-tune, or add Llama-3 / Qwen2 architecture support.
3. **Throughput (CUDA).** Morph fans out section agents; queued CPU inference of a 7B
   model will feel very slow. A fast CUDA path keeps it interactive.
4. **Incremental tool streaming.** Tool-mode streaming currently emits the JSON as a
   single `tool_calls` delta at the end (not token-by-token), so Morph's progressive
   section reveal won't animate — the final result is still correct.

### Realistic expectation today
With the shipped changes + a decent Llama-2-7B instruct, expect the **simplest** Morph
schemas (`shape_data`, a single `stat-grid`) to sometimes succeed. Multi-section
parallel generation needs items 1–2 above. Morph's CSV-import path needs no model and
works regardless.
