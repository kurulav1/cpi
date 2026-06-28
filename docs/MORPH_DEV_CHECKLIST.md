# Precise task list: wire grammar-constrained decoding into the engine

**State of play (verified in code):** the grammar *core* is already built and CMake-wired
— `src/grammar/{grammar,json_schema_to_grammar,grammar_sampler}.cpp`, the `grammar_test`
target, and a working `GrammarSampler` (`apply_mask` / `accept` / `can_terminate`). The
algorithmic work is done. **Everything below is integration plumbing.** Design rationale:
[MORPH_GRAMMAR_DESIGN.md](MORPH_GRAMMAR_DESIGN.md).

Do them in order. Each task = file(s) · exact change · done-check.

---

## Status (verified)

Tasks **0–8, 9a, 9b, and 10 are implemented and verified**. The full engine builds with
CUDA (`-DCMAKE_CUDA_ARCHITECTURES=120`, `nvcc 13.2` + MSVC) with no new `/W4` warnings,
and the **end-to-end smoke test passes on Llama-3.1-8B-Instruct (fp16, GPU-cached)**:

| | output | tokens |
|---|---|---|
| no grammar | `"To return JSON, you typically need to be in a context where you can send HTTP responses..."` | 40 (ran to limit, prose) |
| with `json_schema` | `{"ok": true}` (and `{"ok": true, "note": "..."}` at ctx 8192) | 5–22 (valid instance, clean EOS) |

`[grammar] active for request id=…` is logged; the `pong({ok,note},required:[ok])`
request returns a schema-valid **instance**, not the schema echo.

- **Task 8** — speculative path now *falls back* to non-speculative single-token decode
  on the target engine when a grammar is supplied (`src/main.cpp`), so a `json_schema`
  request still honours the grammar even with `--draft-model`.
- **Task 9a (seed)** — shared sampling RNG made seedable (`detail::dispatch_seed_sampler_rng`
  in `llama_engine_sampling_utils.cpp`); `LlamaEngine::generate_stream` reseeds it from
  `GenerationConstraints::seed`. Affects temperature>0 only (JSON runs temp 0). The other
  engines' per-function RNGs are not yet seedable (deferred; no Morph impact at temp 0).
- **Task 9b (context)** — `web/config.json` `maxContext` is 8192; verified an 8192-context
  grammar run works on the 5090. A `_morph_note` documents the recommended profile.
- **Task 10 (Node)** — `web/server/openai_compat.mjs` sets `jsonSchema` on the internal
  body (forced tool's `parameters`, or a `response_format` json_schema); `index.mjs`
  carries it through `buildCliArgs` → the NDJSON `json_schema` field. Prose instruction
  kept as fallback. Transform verified by an inline check.

- **Task 9c (primary model)** — DONE. Fetched **Qwen2.5-7B-Instruct** (`tools/dl_qwen7b.py`),
  converted + packed to fp16 `.ll2c` (`convert_hf_to_bins.py` + `pack_ll2c.py`, no streaming
  flags = fp16; 14 GB, auto-discovered under `artifacts/`). Grammar smoke test passes on it:
  `{"ok": true, "note": "Connection successful"}` and the nested kanban
  (`{title, columns:[{name,cards}...]}`) both return valid instances with clean EOS. The
  Qwen2.5 family was also verified on the already-present Qwen2.5-Coder-3B `.ll2c`. Note: the
  intermediate `bins/` dir (14 GB) is reclaimable after packing.

**All P0 checklist tasks (0–10) are implemented and verified.**

**P2 performance — DONE.** The mask originally re-walked the grammar for every token
(~152k) every step → ~9–13 tok/s grammar-on (an ~8x slowdown). `GrammarSampler::apply_mask`
now (a) precomputes each token's codepoints once and (b) memoizes `(grammar-state, codepoint)`
transitions per step (a lazy DFA — the JSON grammar has very few states, so the expensive
walk runs a few thousand times instead of 152k×length). Measured on Qwen2.5-7B fp16:
**pong 9.5 → 50.5 tok/s, kanban 13.9 → 56.0 tok/s (4–5.3x)** vs a 90 tok/s grammar-off
ceiling — overhead cut from ~8x to ~1.6x; a 53-token canvas section decodes in ~0.95 s.
The optimized mask is proven byte-identical to a brute-force reference by the
`fastpath parity` test. Remaining optional polish: seed wiring on the non-Llama engines.
GPU-free unit tests: `grammar_test` (48), `tokenizer_pieces_test` (6).

---

### 0. Prove the core (checkpoint, ~5 min)
- **Build & run** the existing test: `cmake --build build --target grammar_test` then run it.
- **Done when:** green. If green, parse/mask/accept/schema→GBNF are proven and the rest is
  wiring. If red, fix the core first (it's pure C++, GPU-free).

### 1. Expose the tokenizer byte table
- **Files:** `include/model/tokenizer.hpp`, `src/model/tokenizer.cpp`,
  `include/model/hf_bpe_tokenizer.hpp` (`id_to_piece_` already exists, built at load,
  ~`hf_bpe_tokenizer.cpp:551`).
- **Add:** `const std::vector<std::string>& token_pieces() const;` on `Tokenizer` (facade,
  delegates to backend) and `HfBpeTokenizer` (returns `id_to_piece_`).
- **Correctness:** each entry must be the **raw bytes the token emits** — byte-level GPT-2
  decode resolved, leading word-boundary marker (`Ġ`/`▁`) → a space; special/added tokens →
  **empty string** (the sampler masks empty-byte tokens by contract). SentencePiece backend
  may return empty (targets use HfBpe).
- **Done when:** unit test asserts the bytes for a leading-space token, a digit token, and a
  special token (empty) are correct.

### 2. `GenerationConstraints` struct + per-engine member
- **New file:** `include/engine/generation_constraints.hpp`:
  `struct GenerationConstraints { grammar::GrammarSampler* grammar = nullptr; int seed = -1; };`
- **Add member** to each engine (`LlamaEngine`, `Qwen35CudaEngine`, `Qwen35CpuEngine`,
  `Llama4CudaEngine`, `Llama4CpuEngine`, `CpuLlamaEngine`):
  `grammar::GrammarSampler* active_grammar_ = nullptr;`
- **Done when:** compiles.

### 3. Thread constraints through `generate_stream` (all engines)
- **Change signature** (current `include/engine/llama_engine.hpp:90` + each engine):
  add trailing `, const GenerationConstraints* constraints = nullptr` — the default keeps
  every existing caller compiling.
- **In each impl:** at entry `active_grammar_ = constraints ? constraints->grammar : nullptr;`
  and reset to `nullptr` on exit (scope guard).
- **Done when:** compiles, existing callers untouched.

### 4. Mask in `decode_next_token` (Qwen35 first, then Llama)
- **File:** `src/engine/llama_engine_sampling.cpp` (and the equivalent in each engine).
- **Current** (`:10-19`): when `temp<=0` it takes an **on-device argmax fast path that never
  returns host logits**. That path can't be masked.
- **Change:** `if (active_grammar_) { /* force host-logits path */ }` — skip the greedy
  fast-path **and** `can_use_greedy_decode_graph()`; materialize `h_logits` via
  `forward_token_logits(token,pos,&h_logits,nullptr)`; then
  `active_grammar_->apply_mask(h_logits);` then the existing
  `detail::dispatch_sample_from_logits(...)` (masked entries are `-inf`, so both the
  `temp<=0` argmax and the sampling path exclude them — **no sampler-internal change**).
- **In the `generate_stream` loop**, after `next = decode_next_token(...)`:
  `if (active_grammar_) active_grammar_->accept(next);`
- **Done when:** the smoke test (below) returns a valid **instance** (`{"ok":true}`), not the
  schema echo it returns today.

### 5. `GenerateStreamFn` typedef + `main.cpp` lambdas
- **`include/app/main_modes.hpp:35`** — add a trailing
  `const engine::GenerationConstraints* = nullptr` to the `std::function` signature.
- **`src/main.cpp` (~261-266)** — the lambdas that call `engine->generate_stream(...)` forward
  the new arg.
- **Done when:** compiles.

### 6. Parse `json_schema` from the NDJSON request
- **File:** `src/app/main_modes.cpp:119-150` (interactive loop).
- **Add:** read a `json_schema` field. ⚠️ **Gotcha:** `json_get_string` returns *scalar* values;
  `json_schema` is a nested object, so extract its **raw JSON substring** from the request
  line (brace-matched), don't use `json_get_string`. Then:
  ```cpp
  grammar::Grammar g = grammar::Grammar::parse(grammar::json_schema_to_grammar(schema_str));
  grammar::GrammarSampler sampler(std::move(g), tokenizer->token_pieces(), tokenizer->eos_id());
  engine::GenerationConstraints c{&sampler, req_seed};
  // pass &c into generate_stream(...)
  ```
  Compile once per request. Keep the prose-instruction fallback when `json_schema` is absent.
- **Done when:** a request with `json_schema` logs "grammar active" and reaches task 4.

### 7. Clean EOS (mostly free)
- **Verify** `json_schema_to_grammar` emits **no trailing-whitespace rule** after the root
  value, so once the value closes the only unmasked token is EOS (`apply_mask` already permits
  EOS only when `can_terminate()`). Confirm the generate loop breaks on `next == eos_id`.
- **Done when:** output stops at the closing brace — no trailing prose/repeats.

### 8. Disable incompatible fast paths while a grammar is active
- Greedy CUDA-graph: already skipped by task 4.
- **Speculative decoding** (`src/engine/llama_engine_generation.cpp:706`, on-device argmax over
  K drafts): when `active_grammar_` **and** a draft model is set, fall back to non-speculative
  single-token decode.
- **Done when:** grammar runs correctly even with a draft model configured.

### 9. P1 (after end-to-end works)
- **Seed:** wire `GenerationConstraints::seed` into the samplers' `thread_local std::mt19937`
  (only matters at `temp>0`; JSON runs `temp=0`).
- **Context:** Morph profile `--max-context 8192`; verify RoPE tables (`d_rope_cos_`/`d_rope_sin_`)
  and KV `layer_stride` (uses `options_.max_context`) are allocated for the raised value.
- **Model:** verify a **Qwen2.5-7B-Instruct** `.ll2c` loads via `HfBpeTokenizer`, `qwen2`
  template correct, special tokens survive. Document model + profile in `web/config.json`.

### 10. Node side (Morph team — coordinate, not C++)
- `web/server/openai_compat.mjs` → send the tool's `parameters` as a structural `json_schema`
  field on the NDJSON line (keep the prose instruction as fallback). Field name agreed:
  `json_schema`.

---

## Verification harness

**Smoke test (after task 4)** — CPI on `:3001` with a model. A forced-tool request must return
a valid *instance*. Today (no grammar) it returns the schema echoed back:
```
today:   {"type":"object","properties":{"ok":...},"required":["ok"]}   ❌
target:  {"ok": true}                                                  ✅
```
(See [MORPH_DEV_INSTRUCTIONS.md](MORPH_DEV_INSTRUCTIONS.md) for the exact smoke-test script.)

**End-to-end (definition of done)** — Morph's Local provider → `http://localhost:3001/v1`:
key-checker returns `ok`; a CSV import's shape step produces a sane layout; "a 3-column kanban"
renders a canvas — on **Qwen2.5-7B-Instruct**, `temp=0`, int8/fp16, clean EOS.
