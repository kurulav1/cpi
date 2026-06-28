# CPI C++ engine work to run Morph well

> **Current priority — see [MORPH_PERF_TODO.md](./MORPH_PERF_TODO.md):** Morph local generation
> now works and is 100% correct on its benchmark; the open issue is SPEED. Measured ~24s/gen on
> Qwen2.5-7B fp16 ⇒ ~20 tok/s decode, ~3–5× below what a 7B should do on a 5090. Ordered, measured
> tasks (decode throughput, prefix KV caching, int4-path slowness, streaming, spec-decoding) are in
> that doc. (The earlier decoder-halt blocker in [MORPH_DECODER_BUG.md](./MORPH_DECODER_BUG.md) is RESOLVED.)

Context: the Node API layer (`web/server/`) now emulates OpenAI tool-calling by
injecting the tool's JSON schema as a prompt instruction and wrapping the model's
text output back into a `tool_calls` response. **That makes CPI *accept* Morph, but the
output JSON is only as good as the model's free-form generation.** The items below are
the C++ engine work needed to make that JSON *reliable and fast*. They're ordered by
impact.

The engine already has: multi-arch support (`llama`, `llama4`, `qwen35` engines),
KV-cache (paged + int4), quantization (int4/int8/turboquant), speculative decoding,
tensor parallel, per-request sampling (`top_k`/`top_p`/`repetition_penalty`,
per-request `temperature` with a greedy fast-path at `temp<=0`). What it does **not**
have is any form of constrained/grammar decoding — that's P0.

> **P0 implementation design:** see [MORPH_GRAMMAR_DESIGN.md](MORPH_GRAMMAR_DESIGN.md)
> for the grounded, reviewable design (module layout, signatures, per-engine wiring,
> test strategy) plus three corrections to the integration sites named below.

## Recommended model target (decided)

- **Primary:** Qwen2.5-7B-Instruct (Qwen2 family, `qwen2` chat template) → optimize and
  tune the grammar pipeline against this one (`Qwen35CudaEngine`).
- **Secondary (validation only):** Llama-3.1-8B-Instruct → run it to prove the
  grammar/template pipeline isn't Qwen-specific; do **not** split primary tuning effort.
- **Precision:** int8 or fp16, **not** int4 — int4 measurably degrades strict-JSON
  adherence at 7B.
- **Context:** raise `--max-context` to ≥4096–8192 for the Morph profile (both models do
  128k natively; CPI's default is 2048). Not "both engines as co-equal targets" — one
  primary to optimize, one secondary to validate portability.

---

## P0 — Constrained / grammar-guided JSON decoding  *(the one that matters)*

**What:** At each decode step, restrict sampling to only tokens that keep the output a
valid continuation of a JSON value matching a supplied JSON Schema. Without this, small
models emit malformed/oversimplified JSON and Morph's `tool_calls.arguments` won't
`JSON.parse`.

**Implementation outline:**
1. **JSON-Schema → grammar compiler.** Add a component that converts a JSON Schema (the
   tool's `parameters`) into a token-level grammar. Reference implementation:
   llama.cpp's `json-schema-to-grammar` + GBNF grammar + `llama_grammar` sampler. Porting
   that approach is the cleanest path.
2. **Grammar state + logit mask.** Maintain a grammar/parser state per generation; at
   each step compute the set of grammar-valid next tokens and mask all other logits to
   `-inf` before sampling.
   - Integration site: **`src/engine/llama_engine_sampling.cpp`** — the
     `sample_from_logits` path (it already branches on `temperature` and dispatches via
     `detail::dispatch_sample_from_logits`). Apply the mask there, for both the greedy
     fast-path and the sampling path.
   - Advance grammar state with each accepted token; terminate when the grammar reaches
     a complete value (drives clean EOS — see P1).
3. **Plumb the schema in.** Thread a grammar/schema param from the request down to the
   sampler:
   - **`src/app/main_modes.cpp`** interactive NDJSON loop (`std::getline(std::cin,…)`)
     already parses `prompt`/`max_new`/`temp`/`stop` via `json_get_*`. Add parsing of a
     new `json_schema` (or `grammar`) field and pass it into `generate_stream(...)`.
   - **`web/server/`** (Node, already tool-aware) should forward the tool's JSON schema
     on the NDJSON request line — this is the contract; coordinate the field name. See
     "Node↔C++ contract" below.
4. **Token-boundary correctness.** JSON grammars operate on characters but sampling is
   on tokens; handle multi-char tokens and partial-token grammar advancement (llama.cpp's
   approach handles this — reuse it).

**Why for Morph:** every Morph generation is a forced tool call that must return
schema-valid JSON (`plan_canvas`, `fill_section` with nested kanban columns/cards,
`shape_data`). This is the difference between "connects but fails to parse" and "renders
a canvas."

---

## P1 — Quality & correctness

1. **Deterministic structured sampling.** For JSON, default to `temperature=0` (greedy)
   — already supported per-request (`req_temp`, greedy fast-path). Ensure the grammar mask
   composes with the greedy/argmax path, and consider adding a `seed` to `EngineOptions`
   (`include/engine/engine_types.hpp`) for reproducibility.
2. **Clean EOS on JSON completion.** The model should stop immediately when the JSON value
   is complete and not ramble past the closing brace. With P0 the grammar can force EOS at
   a complete value; without it, rely on `stop` sequences (`find_first_stop_pos` exists in
   `main_helpers`). Verify EOS (`eos_token_id`) fires promptly.
3. **Context window.** `EngineOptions.max_context` defaults to **2048**; Morph section
   prompts embed schemas + instructions and can exceed that. Validate ≥4096–8192 works
   (KV-cache sizing in `src/runtime/kv_cache.cpp`, RoPE scaling via `rope_theta`) and raise
   the default for the Morph profile.
4. **Confirm a capable model + its template.** A `qwen35` engine exists — a Qwen2.5-7B-
   Instruct-class model is a much better JSON/tool model than the Llama-2 examples. Verify:
   tokenizer (`hf_bpe_tokenizer`) loads it, the chat template is correct, and `.ll2c`
   packing preserves it. Document the recommended model in `web/config.json`.

---

## P2 — Performance & scale

1. **Latency for serialized agents.** Morph fans out section agents; the Node layer now
   queues them through the single engine, so each canvas = several sequential generations.
   Short-JSON generation must be fast (~1–2 s) — lean on the CUDA path, `speculative_decoder`,
   and quantized matvec kernels (`src/kernels/`). Profile decode (`profile_decode_phases`).
2. **True concurrency (optional, larger).** For real parallelism instead of queuing, add
   continuous/in-flight batching (multiple sequences per forward pass) in the decode loop
   (`llama_engine_generation.cpp` / `_forward_decode.cpp`) and lift `max_batch` past 1.
   Otherwise a worker pool of N engine processes (memory cost = N model copies) is simpler.
3. **Quantization vs JSON adherence.** int4 weight/KV quant can degrade strict-JSON
   following. Offer/recommend int8 or fp16 for the Morph profile; note the tradeoff
   (`kv_cache_int4`, `streaming_quant_bits`, `tq_mode`).

---

## Node ↔ C++ contract (so the layer already built lines up)

The Node side (`web/server/openai_compat.mjs` → `buildInternalBodyFromChatRequest`)
detects Morph's forced tool and currently injects the schema **as prose**. Once the engine
supports grammars, switch to passing the schema **structurally**:

- Node: add the tool's JSON Schema as a field on the internal request (e.g. `json_schema`)
  and include it on the NDJSON line sent to `llama_infer`.
- C++ (`main_modes.cpp`): parse that field → compile to grammar (P0.1) → pass to the
  sampler (P0.2). Keep the prose instruction as a fallback when no grammar is supplied.

Agree the exact field name between the Node and C++ changes.

---

### TL;DR for the dev
**P0 = port llama.cpp-style JSON-Schema→GBNF grammar + logit masking into
`llama_engine_sampling.cpp`, fed by a new `json_schema` field parsed in `main_modes.cpp`.**
Everything else is quality/perf around that. With P0 + a Qwen2.5-7B-Instruct-class model,
Morph should generate real canvases; without P0 it will mostly fail to parse.
