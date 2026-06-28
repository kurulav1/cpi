# P0 design: grammar-constrained JSON decoding

Implementation design for the P0 item in [MORPH_ENGINE_TODO.md](MORPH_ENGINE_TODO.md):
constrain decoding so every Morph tool call emits schema-valid JSON. This document
is the design to review **before** code changes; it is grounded in the current code,
not the reference implementation.

## Target (decided)

- **Primary:** Qwen2.5-7B-Instruct (`qwen2` chat template) → `Qwen35CudaEngine`
  ([src/engine/qwen35_cuda_engine.cu](../src/engine/qwen35_cuda_engine.cu)).
  Optimize and tune against this one.
- **Validation only:** Llama-3.1-8B-Instruct → `LlamaEngine`
  ([src/engine/llama_engine_sampling.cpp](../src/engine/llama_engine_sampling.cpp)).
  Run it to prove the grammar/template pipeline is not Qwen-specific; do **not** split
  primary tuning effort.
- **Precision:** int8 or fp16 — **not** int4 (int4 measurably degrades strict-JSON
  adherence at 7B).
- **Context:** Morph profile runs `--max-context` ≥ 4096–8192 (CPI default is 2048;
  both models do 128k natively).

## Three corrections to the original assessment (verified in code)

1. **It is not one sampler — there are five engines.** The TODO says "integrate in
   `llama_engine_sampling.cpp`," but that file only governs `LlamaEngine`. Each of
   `LlamaEngine`, `Qwen35CudaEngine`, `Qwen35CpuEngine`, `Llama4CudaEngine`,
   `Llama4CpuEngine`, `CpuLlamaEngine` has its own `generate_stream` + sampling loop.
   → The grammar machinery must live in **one engine-agnostic module**; only the
   "pull host logits → mask → pick → advance" wiring is per-engine.
2. **The greedy fast-path computes argmax on-device and never returns logits**
   ([llama_engine_sampling.cpp:12-19](../src/engine/llama_engine_sampling.cpp#L12-L19)).
   A grammar mask needs host logits, so grammar mode must force the host-logits path
   and apply the mask before argmax. Enabling a grammar therefore **disables the
   CUDA-graph greedy fast path** (a latency cost — see P2).
3. **The schema cannot reach the sampler today, and per-token bytes are not exposed.**
   `GenerateStreamFn` is fixed at 4 args ([main_modes.hpp:35](../include/app/main_modes.hpp#L35))
   and `decode_next_token(token,pos,temp,history)` has no schema channel. Grammar
   advancement needs the **bytes each token emits**, which the `Tokenizer` facade does
   not expose (`HfBpeTokenizer` keeps `id_to_piece_` private and byte-level-encoded).

## Module layout

New, **pure C++ (no CUDA)** so it builds and unit-tests without a GPU:

```
include/grammar/json_schema_to_grammar.hpp   src/grammar/json_schema_to_grammar.cpp
include/grammar/grammar.hpp                  src/grammar/grammar.cpp
include/grammar/grammar_sampler.hpp          src/grammar/grammar_sampler.cpp
```

- **`json_schema_to_grammar`** — JSON Schema → GBNF string. Port the subset of
  llama.cpp's `json-schema-to-grammar` that Morph's schemas use: `type`
  (object/array/string/number/integer/boolean/null), `properties`, `required`,
  `enum`, `items`, nested objects, `additionalProperties`. Whitespace rules permissive.
- **`grammar`** — GBNF parser → rules; stack-based UTF-8 matcher (port of llama.cpp
  `grammar_parser` + `llama_grammar_accept`). Operates on **codepoints/bytes**, not
  tokens.
- **`grammar_sampler`** — bridges grammar ↔ tokens. Owns the compiled grammar, current
  stack(s), and a pointer to the token→bytes table. API:
  - `void apply_mask(std::vector<float>& logits) const;` — sets every token whose bytes
    cannot extend the current grammar state to `-inf`.
  - `bool accept(int token_id);` — advances grammar state by that token's bytes;
    returns `is_complete()`.
  - `bool is_complete() const;` — grammar has matched a full top-level value.

## Tokenizer change

Add to `model::Tokenizer` and `HfBpeTokenizer`:

```cpp
const std::vector<std::string>& token_pieces() const;  // index = token id, value = raw bytes
```

Built once at load. Each entry is the literal bytes that token contributes
(byte-level GPT-2 decode / byte-fallback resolved; special/added tokens → empty bytes
so the mask never blocks on them but the grammar does not consume them either).

**Known edge to handle:** a single token decoded in isolation can differ from its
in-context contribution (leading word-boundary `▁`→space). JSON grammars tolerate
whitespace between tokens, so map a leading boundary marker to a space; treat this as
a correctness risk to cover with a unit test, mirroring llama.cpp's `token_to_piece`.

## Plumbing the schema in

1. **Request struct.** Add `struct GenerationConstraints { grammar::GrammarSampler* grammar = nullptr; int seed = -1; };`
2. **Engine signature.** Add a trailing `const GenerationConstraints* constraints = nullptr`
   to `generate_stream` on each engine (default keeps every existing caller compiling).
   `generate_stream` stashes `constraints->grammar` in a member `active_grammar_` for
   the duration; `decode_next_token` reads it. (Avoids threading a new arg through every
   internal call site.)
3. **`GenerateStreamFn` typedef** ([main_modes.hpp](../include/app/main_modes.hpp)) gains
   the same trailing optional param; the binding lambdas in
   [main.cpp:261-266](../src/main.cpp#L261-L266) forward it.
4. **NDJSON parse** ([main_modes.cpp:119-150](../src/app/main_modes.cpp#L119-L150)):
   read a new `json_schema` field (the tool's `parameters`) and/or a raw `grammar`
   field. Compile once per request → `GrammarSampler` → pass via `GenerationConstraints`.
   Keep the prose instruction as fallback when no schema is supplied.

## Masking in the decode loop (per engine)

In `decode_next_token`, when `active_grammar_ != nullptr`:
1. Skip the on-device argmax fast path; take the host-logits path
   (`forward_token_logits(...,&h_logits)` / `decode_next_token_logits_graph`).
2. `active_grammar_->apply_mask(h_logits);`
3. Sample with the existing `detail::dispatch_sample_from_logits` — masked entries are
   `-inf`, so both the greedy (`temp<=0` argmax) and sampling paths naturally exclude
   them. No change to the sampler internals.

In the `generate_stream` loop, after `next = decode_next_token(...)`:
- `active_grammar_->accept(next);` to advance grammar state.
- **Stopping is EOS-driven, not break-on-complete.** A value like `integer` can
  "terminate" after one digit, so breaking the instant the grammar *can* stop would
  truncate `42` to `4`. Instead: the mask permits EOS only once the value is complete
  (`can_terminate()`), and the generated grammar has **no trailing-whitespace rule**, so
  once the top-level value closes the only unmasked token is EOS — the model is forced
  to emit it, and the existing `next == eos_token_id` check breaks the loop (clean EOS,
  P1.2). `accept()` returns `can_terminate()` for diagnostics only.

## Interactions / out of scope for P0

- **Greedy CUDA-graph fast path:** disabled while a grammar is active (item 2 above).
  Document the latency hit; revisit in P2.
- **Speculative decoding:** the verify path computes on-device argmax across K tokens
  ([llama_engine_generation.cpp:706](../src/engine/llama_engine_generation.cpp#L706)),
  which masking can't reach cheaply. P0 = **disable grammar when a draft model is set**
  (or fall back to non-speculative). Combining them is a later item.
- **Token-streaming of tool JSON:** unchanged; Node still emits one `tool_calls` delta.

## P1 items that ride along

- **Seed** ([engine_types.hpp](../include/engine/engine_types.hpp)): the samplers use
  `thread_local std::mt19937 rng(42)`. Add `seed` to `GenerationConstraints`; only
  matters for `temp>0` (JSON should run `temp=0`, already deterministic).
- **Context window:** keep the global default at 2048 (VRAM cost); ship the Morph
  profile with `--max-context 8192`. Verify the RoPE tables (`d_rope_cos_`/`d_rope_sin_`,
  sized `max_seq_len x head_dim`) and KV-cache `layer_stride` (uses `options_.max_context`)
  are allocated for the raised value.
- **Model packing/template:** verify a Qwen2.5-7B-Instruct `.ll2c` pack loads via
  `HfBpeTokenizer`, the `qwen2` chat template is correct, and special tokens survive.
  Document the recommended model + Morph profile in `web/config.json`.

## Node ↔ C++ contract

Node sends the tool's JSON Schema structurally as `json_schema` on the NDJSON request
line (field name agreed = `json_schema`). Prose injection stays as the fallback path
for when grammars are unavailable.

## Build order

1. Grammar module (schema→GBNF, GBNF parser, grammar state, mask) — pure C++ **+ unit
   tests**. Largest piece; build/test without a GPU.
2. Tokenizer `token_pieces()` table.
3. `GenerationConstraints` plumbing: engine signatures → `GenerateStreamFn` →
   `main.cpp` lambdas → `main_modes.cpp` `json_schema` parse.
4. Wire masking into `Qwen35CudaEngine` (primary), then `LlamaEngine` (validation).
5. Grammar-driven clean EOS; disable greedy graph + speculative under grammar.
6. P1: seed; Morph profile `max_context=8192`; Qwen2.5-7B pack/template verification.
7. Node: send `json_schema` structurally, keep prose fallback.
8. Docs: `web/config.json` Morph profile.

## Test strategy (most is GPU-free)

- **Schema→GBNF golden tests:** Morph's real tool schemas (`plan_canvas`,
  `fill_section` with nested kanban columns/cards, `shape_data`) → expected GBNF.
- **Grammar acceptance tests:** grammar + a valid JSON string → assert each step's mask
  admits the correct next token and rejects an invalid one (letter where digit required,
  missing-comma, etc.).
- **Token-piece edge test:** leading-space / byte-fallback tokens map to the right bytes.
- **Integration (CPU engine, CI-friendly):** run `CpuLlamaEngine` with a grammar on a
  tiny model; assert the output parses and matches the schema. The host-side grammar
  logic is fully testable without CUDA — that's why the module is CUDA-free.
- Reuse the `tools/sampling_parity_test.cpp` harness pattern already in the tree.
