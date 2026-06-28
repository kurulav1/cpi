# MORPH — Decoder halts early on specific token pieces (RESOLVED)

> **RESOLVED — root cause was NOT the decoder.** The C++ engine generates fine
> *past* `?`/`."`/em-dash (verified: a raw `llama_infer` run on R2 produced
> `a=="0"?b:then DONE…` for the full token budget). The truncation was in the
> **Node layer**: `normalizeGeneratedChatText` in `web/server/index.mjs` ran an
> **ungated "drop trailing incomplete tail"** heuristic that cut output at the last
> `.`/`?`/`!` whenever the trailing text had no terminator — so every ternary `?`,
> quoted `."`, decimal, or URL truncated. (The vocab/EOS/special-token hypotheses
> below were not the cause.) The C++ sibling `trim_incomplete_trailing_tail` was
> correctly gated behind `looks_degenerate_repetition`; the JS copy was not — the
> two had drifted out of sync.
>
> **Fix:** gate the JS trim behind `looksDegenerateRepetition` (matching C++), and
> extract `normalizeGeneratedChatText` + `looksDegenerateRepetition` to
> `web/server/text_normalize.mjs` with a regression test (`text_normalize.test.mjs`,
> R1–R4 + Morph DSL ternary + em-dash + degenerate-salvage; `node
> web/server/text_normalize.test.mjs` → 9/9). Verified end-to-end on a live server:
> R1–R4 and the em-dash repro all now run to `DONE`. **No engine/build change.**

---

## Original report (diagnosis superseded by the resolution above)

**One line:** CPI stops generation (`finish_reason: "stop"`) immediately after emitting
certain token pieces — notably the ternary `?` in context and `."` (period before a closing
quote) — even though the model has much more to say. This makes CPI unusable for structured /
code-like output (Morph's UI DSL relies on `?:` ternaries and quoted strings everywhere).

This is an **engine bug, not a model bug** (see "Why it's the engine" below).

---

## Impact on Morph

Morph generates a small UI as a compact DSL whose handlers use ternaries, e.g.
`display=display=="0"?$item:display~$item`. Every generation dies at the first `?`. Result:
truncated, non-functional output (e.g. a calculator with no `=` button). Blocks the entire
local-generation path. Hosted models (Anthropic/OpenAI) are unaffected because they don't run
through CPI's decoder — which localizes the bug to CPI.

---

## Reproductions (server already running on :3001)

Run these against a live CPI. They use plain `/v1/chat/completions`, greedy (`temperature:0`),
and a trivial "repeat this" task so the *intended* continuation is unambiguous.

**R1 — `."` halts (period before closing quote):**
```bash
curl -s http://localhost:3001/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"Qwen2.5-Coder-32B-Instruct-streaming-int4","max_tokens":80,"temperature":0,
 "messages":[{"role":"system","content":"Repeat EXACTLY, then write DONE:"},
             {"role":"user","content":"X then \".\" then DONE"}]}'
```
Observed: content = `X then ".`  · `finish_reason:"stop"`  (halts ON the period; never emits the
closing `"` or `then DONE`). Expected: the full string ending in `DONE`.

**R2 — ternary `?` halts (the fatal one for Morph):**
```bash
curl -s http://localhost:3001/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"Qwen2.5-Coder-32B-Instruct-streaming-int4","max_tokens":80,"temperature":0,
 "messages":[{"role":"system","content":"Repeat EXACTLY, then write DONE:"},
             {"role":"user","content":"a==\"0\"?b then DONE"}]}'
```
Observed: content = `a=="0"?`  · `finish_reason:"stop"`  (halts right after the `?`).

**R3 / R4 — controls that COMPLETE correctly (prove it's specific tokens, not length/format):**
- `"0"` alone (quoted char, no following `?`/`.`) → completes, emits `DONE`.
- plain text with no quotes (`alpha bravo charlie then DONE`) → completes.

Also reproduced with em-dash `—` (multibyte) truncating mid-string.

---

## Why it's the engine, not the model

1. **Model-independent.** Identical halts on **Llama-3.1-8B-Instruct** and
   **Qwen2.5-Coder-32B-Instruct** at the *same* trigger tokens. Two different weight sets do not
   coincidentally both emit EOS after `."`/`?` — the engine is the common factor.
2. **Path-independent.** Happens both on the forced tool-call (grammar-constrained JSON) path
   and on plain `/v1/chat/completions` with `stop: []` explicitly set (so it is not a stop-string
   match in the OpenAI-compat layer).
3. **Greedy + trivial task.** At `temperature:0` repeating a given string, the correct next token
   is obvious and high-probability; the model "choosing" EOS there is not plausible.
4. **`finish_reason:"stop"`**, not `"length"` — so the loop believes the sequence *ended*, i.e.
   an EOS/end-of-text token was selected (or the stop check fired) right after these pieces.

---

## Hypothesis (where to look — "token-piece byte mapping")

The trigger tokens are byte-level-BPE pieces that need piece→byte decoding (closing-quote merges
like `."`, `?`, `"` with byte-fallback/space markers). Likely one of:

- **Vocab index / piece→byte misalignment:** the logit mass for the intended next token lands on
  a slot whose token-id is actually EOS (or a special token) in CPI's vocab, because the
  piece→id or id→bytes table is off for these specific pieces. Classic off-by-one / wrong
  byte-fallback mapping. Content-specific (only after these pieces) rather than global.
- **Streaming detokenizer corrupting bytes** for these pieces, and the **stop/EOS check running
  on the corrupted bytes** falsely matching end-of-text.
- **Special-token detection** treating a normal piece whose bytes overlap a special-token marker
  (`<|...|>`, `</s>`, `<|eot_id|>`) as that special token.

---

## Work list (ordered)

1. **Instrument the halt.** At the step generation stops in R2, log: the sampled **token id**,
   its **decoded bytes/string**, and the **top-5 logits with their token ids + decoded pieces**.
   This immediately distinguishes the two failure classes:
   - If a normal-looking high-logit token's **id maps to EOS** → vocab/piece→id misalignment.
   - If **EOS itself has an abnormally high logit** → logit processing / special-token handling.
2. **Verify vocab alignment** for the pieces around the triggers (`."`, `?`, `?"`, `0"`, `"`)
   against the reference HF `tokenizer.json`: confirm CPI's id↔piece↔bytes agree exactly,
   including byte-fallback tokens (`<0xXX>`) and the space/▁/Ġ markers.
3. **Check EOS / special-token ids.** Confirm CPI's configured EOS id(s) match the model
   (Llama3 uses `<|eot_id|>` 128009 / `<|end_of_text|>` 128001; Qwen uses `<|im_end|>` etc.).
   A wrong/duplicated EOS id can collide with ordinary pieces.
4. **Audit the streaming detokenizer's piece→byte path** for these merges; ensure the stop/EOS
   comparison runs on token **ids**, not on decoded text bytes.
5. **Add regression tests** from R1–R4 below as fixtures.

---

## Acceptance criteria

- R1 emits the full `X then "." then DONE`; R2 emits the full `a=="0"?b then DONE`; em-dash test
  completes. `finish_reason` becomes `"stop"` only at the *real* end (or `"length"` at the cap).
- A Morph calculator prompt generates a complete DSL containing the `=` and `C` buttons and a
  ternary handler, on both Llama-3.1-8B and Qwen2.5-Coder-32B.
- R3/R4 controls still pass (no regression on currently-working content).

See also [MORPH_ENGINE_TODO.md](./MORPH_ENGINE_TODO.md), [MORPH_SUPPORT.md](./MORPH_SUPPORT.md).
