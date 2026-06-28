# Developer instructions: make CPI run Morph

Start here. This is the task, the read order, the verification harness, and the
acceptance bar. The detailed reference is in the other two docs.

## Mission

Make CPI emit **schema-valid JSON instances** for forced tool calls, so Morph
(a generative-UI runtime that drives the model entirely through forced tool calls)
renders real canvases against a local model instead of a paid cloud API.

The Node/API half is already done (tool-calling protocol + request queuing, shipped
and unit-tested). **Your work is the C++ engine: grammar-constrained JSON decoding.**

## Read order

1. [MORPH_SUPPORT.md](MORPH_SUPPORT.md) — what already shipped on the Node side + how
   Morph connects to CPI. Context.
2. [MORPH_ENGINE_TODO.md](MORPH_ENGINE_TODO.md) — the prioritized engine work + the
   decided model target (Qwen2.5-7B primary, Llama-3.1-8B validation).
3. [MORPH_GRAMMAR_DESIGN.md](MORPH_GRAMMAR_DESIGN.md) — **the actual P0 design**: module
   layout, signatures, the five-engine wiring, tokenizer change, decode-loop masking,
   build order, and GPU-free test strategy. This is your spec.

## Verified baseline (what CPI does *today*)

A forced-tool request to a running CPI (`/v1/chat/completions`, Llama-3.1-8B, no
grammar) returns a correct envelope but the **wrong content** — the model echoes the
schema instead of producing an instance:

```
request tool schema:  { ok: boolean (required), note: string }
finish_reason:        tool_calls          ✅ protocol correct (Node layer)
arguments returned:   {"type":"object","properties":{"ok":...},"required":["ok"]}
                                          ❌ that's the SCHEMA, not an instance
latency:              ~23 s on CPU         ⚠️ P2 perf
```

**This is the problem you're solving.** It parses as JSON but is not a valid instance,
so Morph cannot use it. (Reproduce with the smoke test below.)

## Acceptance bar (definition of done)

The **same** forced-tool request returns a valid *instance*:
```
arguments returned:   {"ok": true}        (or {"ok": true, "note": "..."}) — matches schema
```
…for Morph's real schemas (`plan_canvas`, `fill_section` with nested kanban
columns/cards, `shape_data`), at `temperature=0`, terminating cleanly at the closing
brace, on the **primary target Qwen2.5-7B-Instruct** (Llama-3.1-8B passing too =
portability proof).

## Build & verify loop

Follow the build order in [MORPH_GRAMMAR_DESIGN.md](MORPH_GRAMMAR_DESIGN.md#build-order).
Each milestone has a concrete check:

| # | Milestone | How to verify |
|---|-----------|---------------|
| 1 | Grammar module (schema→GBNF, parser, state, mask) | **Unit tests, GPU-free.** Golden schema→GBNF + acceptance tests (see design doc test strategy). Must be green before touching engines. |
| 2 | `Tokenizer::token_pieces()` byte table | Unit test: leading-space / byte-fallback tokens map to the right bytes. |
| 3 | `GenerationConstraints` plumbing + `json_schema` NDJSON parse | Compiles; existing callers unaffected (trailing optional arg). A request with `json_schema` reaches the sampler (log it). |
| 4 | Mask wired into the decode loop (Qwen35 first, then Llama) | **Re-run the smoke test below** → `arguments` is now a valid *instance* (`{"ok":true}`), not the schema. |
| 5 | Grammar-driven clean EOS | Output stops at the closing brace; no trailing prose/repeats. |
| 6 | **Morph end-to-end** | Point Morph's Local provider at CPI; checks below pass. |

### Smoke test (milestones 1–5)

CPI running on `:3001` with a model loaded. From `web/`:
```js
// node _smoke.mjs  (Node 18+)
const m = (await (await fetch('http://localhost:3001/api/models')).json()).models[0].id;
const body = { model: m, max_tokens: 80, stream: false,
  messages: [{ role: 'user', content: 'Confirm you are reachable.' }],
  tools: [{ type:'function', function:{ name:'pong', parameters:{
    type:'object', properties:{ ok:{type:'boolean'}, note:{type:'string'} }, required:['ok'] }}}],
  tool_choice: { type:'function', function:{ name:'pong' } } };
const d = await (await fetch('http://localhost:3001/v1/chat/completions',
  { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) })).json();
const a = d.choices[0].message.tool_calls[0].function.arguments;
console.log(a, '→', JSON.parse(a));   // PASS when this is {ok:true}, not the schema
```

### Morph end-to-end (milestone 6)

Morph dev server on `:3000`. In Morph's **Add key** modal → **Local** tab:
base URL `http://localhost:3001/v1`, model = the id from `/api/models`. Then:
1. Morph's key-checker (`POST :3000/api/live/check-key`, header `x-llm-provider: openai`,
   `x-llm-base-url: http://localhost:3001/v1`) returns `status: "ok"`.
2. Paste a CSV → the async "shape" step (simplest schema) produces a sane layout.
3. A short prompt ("a 3-column kanban") renders a canvas.

## Guardrails

- **Grammar module is pure C++ / no CUDA** — it must build and unit-test without a GPU.
- Disabling the greedy CUDA-graph fast path and speculative decoding **while a grammar
  is active** is expected (design doc §Interactions); document the latency hit, don't
  try to combine them in P0.
- Don't regress non-grammar requests: every change is behind "grammar supplied".
- Target precision **int8/fp16, not int4**; raise `--max-context` to 8192 for the Morph
  profile.

## Questions for the Morph side

The Node↔C++ contract field is agreed as `json_schema` (the tool's `parameters`) on the
NDJSON request line, prose instruction as fallback. If you need the schema shaped
differently (e.g. pre-compiled GBNF instead), say so and the Node side will adapt.
