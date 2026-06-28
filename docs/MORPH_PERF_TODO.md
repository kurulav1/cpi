# MORPH — CPI performance work list (measured)

Morph's local generation now works end-to-end and is **100% correct** on its benchmark
(`apps/builder/src/lib/runtime/bench.mjs`, 6–12 diverse prompts: valid surface, interactive,
all expressions parse, right primitive). Quality is solved. The remaining gap is **speed** —
"instant" is the product's whole UX, and the numbers say there's a lot of headroom on the CPI
side. Ordered by impact.

## Benchmark snapshot (clean, contention-free, RTX 5090)
| model | path | quality | avg latency / gen |
|---|---|---|---|
| Qwen2.5-7B-Instruct | fp16, gpu-cached | 100% | **~24s** ← current default |
| Qwen2.5-Coder-32B | int4-streaming | 100% | ~73s |
| Qwen2.5-Coder-3B | int4-streaming | 83% | ~29s |

A generation emits ~400–600 DSL tokens. The system prompt is a fixed ~2,300-token guide; only
the short user message varies per request.

---

## P0 — autoMaxTokens ignored client max_tokens — ✅ DONE

**Bug:** `autoMaxTokens` defaulted ON and `computeDynamicMaxNewTokens` (a keyword heuristic)
overrode the client's `max_tokens` entirely — picking ~96 tokens for many prompts and silently
truncating output mid-generation (`finish_reason:"stop"`, completion on 32-multiples). Hit any
OpenAI-compatible client relying on `max_tokens`.
**Fix (shipped, `web/server/index.mjs`):** an explicit client `max_tokens`/`maxNewTokens` is now
authoritative (the OpenAI contract); the auto heuristic applies only when the client supplied no
budget. **Verified:** `max_tokens:400` → 400 completion tokens (was ~96); omitting it still uses
the heuristic (22 tokens for "Say hi").

## P0 — Decode throughput (biggest lever) — ✅ DONE (13 → 83 tok/s)

**Measured root cause (not generic decode):** at Morph's ~2,300-token context, decode was
**13 tok/s and attention was 81% of decode time** (62 ms/token); prefill was already fine
(667 tok/s) and MLP at the memory-bandwidth floor (7.5 ms/token). The bottleneck was the
**GQA-fused decode attention kernel**: `launch_attention_step` / `_device_pos` in
`src/kernels/kernels_attention_decode.cu` dispatched it for GQA models (Qwen 28/4 heads,
head_dim 128) *before* the split-K path — it launches only `num_kv_heads` (4) blocks and scans
the sequence serially per block (~2% occupancy on a 170-SM 5090 at 2k+ tokens).

**Fix (shipped):** keep the GQA-fused kernel for short context, route long context to the
existing split-K path (parallel over KV chunks, `num_heads × ⌈seq/32⌉` blocks). Host launcher
gates on `seq_len <= kGqaFusedMaxSeq` (256); the graph/`_device_pos` launcher (can't read
runtime seq) gives split-K priority for GQA, with a fixed `num_heads × scratch_chunks` grid +
per-chunk seq guard so it stays CUDA-graph-safe as the sequence grows. Scratch is already sized
`(max_context+31)/32`, so no KV is dropped.

**Result (Qwen2.5-7B fp16, RTX 5090, ~2,200-token ctx):**
- decode **13 → 82.8 tok/s** (graph/temp-0 path); attention 9310 ms → **265 ms** (81% → 12% of
  decode); MLP (weight-read floor) is now the dominant phase, as it should be.
- a 200-token gen: **22.2s → 7.0s end-to-end (3.2×)**; Morph's ~400–600-token gens now ~6–10s.
- short-context decode **unchanged (~95 tok/s, no regression)**; greedy output **byte-identical**
  to pre-fix (split-K computes exact attention).

## P0 — `autoMaxTokens` default silently truncates output (CORRECTNESS, not perf)
**Observation:** `/v1/chat/completions` defaults `autoMaxTokens=true` (`web/server/index.mjs` ~L747),
which runs `computeDynamicMaxNewTokens` — a heuristic that picks an output budget from prompt
**keywords** and **ignores the client's `max_tokens`**. For Morph's DSL prompts it chose ~96 tokens
and **truncated every non-trivial app mid-handler** (the calculator lost its bottom keypad rows). It
masquerades as a clean finish: `finish_reason:"stop"`, and `completion_tokens` lands on a 32-multiple
(96, 320…), so it looks natural — it is not (temperature-independent; deterministic; server-side).
**Impact:** a standard OpenAI-compatible client never sends `autoMaxTokens`, so the default truncates
ANY client that relies on `max_tokens`, not just Morph.
**Morph workaround (shipped):** the OpenAI client now sends `autoMaxTokens:false`, so CPI honors
`max_tokens`. This unblocked Morph — but the default still bites every other client.
**Task:** default `autoMaxTokens` OFF, OR never let the heuristic return a budget BELOW an explicit
client `max_tokens` (treat client `max_tokens` as a floor/authority when present). **Verify:** a
request with `max_tokens:2600` and no `autoMaxTokens` flag generates a full ~140-token DSL app
without truncation.

## P1 — Greedy decode emits EOS early on repeated-token runs (CORRECTNESS) — ✅ DONE (min_new_tokens)

**Fix (shipped):** added a **`min_new_tokens` knob** that masks the EOS logit to -inf until N tokens
are generated, so greedy (temp 0) cannot terminate early on a collapsed/repeated run. Plumbed
end-to-end: OpenAI request `min_tokens` (or `min_new_tokens`) → Node `min_new` on the NDJSON line →
`GenerationConstraints.min_new_tokens` → `LlamaEngine` masks EOS in `decode_next_token` for the first
N steps (forces the host-logits path so it works under the greedy CUDA graph too). `main_modes` also
holds off its stop-id/sentence-stop checks until N. Grammar takes precedence (a completed grammar may
permit only EOS, so min_new yields to it — no deadlock). **Verified end-to-end** via the server:
`"Reply with only: ok"` stops at **1** token normally, but at **41** with `min_tokens:40` (EOS
suppressed until 40, then natural stop). Default 0 = off, so non-Morph clients are unaffected.
Morph can set e.g. `min_tokens: ~200` and drop the temp-0.6 retry — back to pure greedy.

<details><summary>Original report</summary>

**Symptom:** at temperature 0 (greedy), generation truncates MID-OUTPUT — `finish_reason:"stop"`,
deterministic, and `completion_tokens` tends to land on a multiple of 32. It is NOT a length cap
(autoMaxTokens is already fixed) and NOT the Node trailing-tail trim. The model emits the EOS token
prematurely the moment its next-token distribution collapses during a **run of identical/repeated
tokens** — e.g. a big round number (\`"revenue":4000000\`, \`"employees":500000\`) or a repeated array
(\`[0,0,0,0,0]\`). Real Morph repro: "compare accenture and capgemini" → the model seeds
\`"employees":500000,"revenue":4000000\` and stops dead at ~61 tokens, before any visible node →
Morph shows empty_surface.

**Evidence it's a sampling/decoding issue, not a cap** (same prompt + system, max_tokens 2600,
autoMaxTokens:false):
| temperature | completion_tokens | result |
|---|---|---|
| 0.0 | 61  | truncated mid-number, no UI |
| 0.3 | 126 | complete (but still truncates ~3/4 runs — borderline) |
| 0.6 | 196 | complete |
So a little sampling escapes it; pure greedy gets stuck choosing EOS after the repeated run.

**Morph workaround (shipped):** generate greedily first (temp 0 — best quality for token-sensitive
apps like a calculator keypad, which *degrades* under blanket sampling), and ONLY on an
empty/truncated result retry once at temp 0.6. Works, but costs a full second generation on the
unlucky prompts and is a band-aid over a decode bug.

**Suggested CPI fix (the clean one):** a `min_new_tokens` / `min-tokens` knob that suppresses the
EOS logit until N tokens are generated, OR a repetition-aware EOS guard (down-weight EOS while the
recent window is low-entropy/repeating). Either lets greedy (temp 0) stay the default — best quality
+ reproducible + prefix-cache-friendly — without the early-EOS trap. **Verify:** the accenture prompt
(or any prompt seeding `4000000`/`[0,0,0,0]`) completes a full app at temperature 0.

</details>

## P1 — Prefix KV caching (make the static prompt ~free) — ✅ DONE (prefill 4.6s → ~0.13s)

**Fix (shipped, `LlamaEngine`):** the engine tracks `resident_prefix_` (the tokens whose KV is
currently resident). On each `generate_stream`, it computes the longest common token prefix
between the new prompt and the resident one, **skips re-prefilling that shared prefix**, and
prefills only the divergent tail (`prefill_prompt`/`_sequential` gained a `start_pos`). KV for an
identical prefix at identical positions is bit-exact (causal attention + position-based RoPE), so
output is unchanged; `reset_kv_cache` invalidates the tracked prefix whenever KV is wiped. Gated
to the simple contiguous fp16 KV layout (paged / int4-KV / TQ3 / MoE / sliding-window fall back).
**Result (Qwen2.5-7B, same ~2,300-token system prompt, varying user message):** prefill
**4579 ms → ~125 ms (37×)** on the 2nd+ request; a 60-token gen **5.3s → 0.85s**. Output
**byte-identical** between cold (full prefill) and warm (reuse) for the same prompt. Morph keeps
the system prefix byte-stable, so repeat gens are now decode-bound (~3–4s for 400–600 tokens).

## P1 — int4-streaming is *slower* than fp16 (it shouldn't be)
**Observation:** the int4-streaming models (3B, 32B) decode **slower per token** than the fp16 7B —
backwards; quantized should be ≥ fp16 throughput. This is why the 32B costs 73s.

**Assessed (not the GQA attention bug):** int4-streaming uses int4 *weights* with an **fp16 KV
cache**, so it goes through the same `launch_attention_step` that P0 already fixed — its
long-context attention is now split-K/parallel too (so the 32B almost certainly already dropped
from 73s; not yet re-measured). The int4-KV launcher (`launch_attention_step_int4`) was already
split-K. So the residual slowness is **not** attention — it's the **int4 weight dequant** in the
projection/MLP matvecs (per-token dequant without a fused int4 matmul), as suspected.
**Task (larger, deferred):** a fused dequant+matmul int4 kernel. Not a quick one-line fix; lower
Morph value than P1 prefix-cache since the fp16 7B is the default and the 32B isn't currently
needed (both score 100%).

## P2 — SSE token streaming (any path) — ✅ DONE

**Root cause:** every streaming path threw `ReferenceError: openAiSystemFingerprint is not defined`
on the **first** SSE write — the function was defined (not exported) in `openai_compat.mjs` but
used unimported in `index.mjs` (~18 sites across chat/responses/completions streaming). Headers
were already sent, so the handler couldn't recover and the socket dropped with **zero bytes**
(curl exit 18). Non-streaming worked because it builds responses inside `openai_compat.mjs` where
the function was in scope.
**Fix (shipped):** `export` it + add to the `index.mjs` import. **Verified:** `curl -N ...
'{"stream":true,...}'` now emits 23 incremental `data:` chunks (role delta → token-by-token
content → `finish_reason:"stop"` → `data: [DONE]`) and closes cleanly (curl exit 0). Morph
progressive reveal is now a one-line `stream:false`→`true` flip on its side.

## P2 — SSE token streaming (any path) — original notes
**Update (Morph side):** Morph now generates the app as **plain text** (not a forced tool call) and
is ~27% faster for it. It also switched its client to a streaming-shaped interface and is fully wired
to consume SSE deltas — the `/live` UI will render the app **progressively as tokens arrive** the
moment CPI streams. So this is now purely a CPI-side gap; no Morph work remains.

**Observation:** `stream:true` against CPI drops the socket on **both** the tool-call path AND the
plain `/v1/chat/completions` path — a direct `curl -N ... -d '{"stream":true,...}'` returns no chunks
then terminates. Until that's fixed, `OpenAIClient.streamComplete` requests `stream:false` and emits
the whole completion once (one big SSE delta), so generation works but there's no progressive reveal.
**Task:** Support reliable SSE token streaming on the plain chat-completions path (grammar/tool path
is no longer required). **Verify:** `curl -N -X POST .../v1/chat/completions -d '{"stream":true,...}'`
emits incremental `data:` chunks and closes cleanly. Then Morph progressive reveal is a one-line flip
(`stream:false`→`true` + forward deltas) on its side.

## P2 — Speculative decoding for the 7B
CPI already has speculative decoding; it isn't used for the 7B default. Wire a small draft model
(e.g. Qwen2.5-0.5B) to speculate for the 7B — structured/low-entropy output like the DSL speculates
well, often 2–3× decode. **Verify:** tok/s up, greedy output byte-identical.

## P3 — DSL grammar (robustness insurance, not urgent)
Quality is already 100%, so this is insurance, not a fix. Optionally accept a Morph-DSL grammar (or
expose the grammar interface) so DSL output is valid-by-construction. Lower priority than speed.

## FEATURE — `/v1/embeddings` endpoint (unblocks Morph local RAG) — ✅ DONE + hardened

**Shipped (CPI-native, zero external deps — matches the hand-rolled tokenizer / safetensors / CUDA
philosophy):** a config-driven embedding "core" — not bge-hardcoded:
- WordPiece tokenizer (`src/model/wordpiece_tokenizer.cpp`, BertNormalizer + greedy `##`).
- `EmbeddingConfig` (`src/engine/embedding_config.cpp`) — pooling/normalize/prefixes from
  `config.json` + `cpi_embed.json` + `1_Pooling/config.json`.
- Hand-rolled CUDA BERT encoder (`src/engine/bert_embedder.cu`) — embed/LN/bidirectional-attn/GELU,
  CLS or mean pool, L2 normalize.
- `cpi_embed` worker (`src/cpi_embed.cpp`) — persistent NDJSON process, batches the whole `input[]`
  in one pass, order-preserving.
- `POST /v1/embeddings` (`web/server/index.mjs`) — OpenAI shape; `input_type:"query"|"document"`
  (default document) applies the model's retrieval prefix **server-side** so Morph stays
  model-agnostic. Model: **bge-small-en-v1.5 (384d)**.

**Verified:** batch of N → N equal-length 384-d vectors in order; identical text → identical vector;
L2 norm = 1.0000; `cos(king,queen)=0.7349 > cos(king,banana)=0.5806`;
`cos(query "capital of France", "Paris is the capital of France")=0.8220 >
cos(same query, "Bananas are yellow")=0.3182`.

### Hand-off P0 — works out-of-the-box & survives autostart — ✅ DONE
The original break: `/v1/embeddings` 500'd (`embed binary not found … set EMBED_BIN`) because
`cpi_embed` is a CUDA-only target that lands in `build-run/Release` (or `build-cuda/`), never the
plain CPU `build/`, and the resolver guessed a single wrong dir. Fixed (`web/server/embed_worker.mjs`):
- **Path resolution scans candidates in order** — `EMBED_BIN`, the dir holding the resolved
  `llama_infer`, then `build-run` / `build-cuda` / `build` — and returns the first that exists. No
  `EMBED_BIN` needed; the autostarted service finds the binary on its own. (Also fixed the earlier
  non-ASCII-home-dir bug: `fileURLToPath` instead of `new URL().pathname`, which URL-encoded `Väinö`.)
- **Fail loud, not mid-use:** startup logs `embeddings: enabled (bge-small) bin=…` or
  `DISABLED (cpi_embed not found at … — RAG/folder-search will 503)`; `GET /v1/models` carries a
  non-standard `embeddings:{available,model}` readiness hint so Morph can preflight; and the route
  itself returns a clear **503 `embeddings_unavailable`** (not an opaque 500) when the binary/model
  is missing. **Verified with `EMBED_BIN` unset**: startup log = enabled, `/v1/models` →
  `{available:true,model:"bge-small-en-v1.5"}`, live batch call works.

### Hand-off P1 — embed throughput at folder scale — ✅ already satisfied
Not a spawn-per-chunk path. `embed_worker.mjs` keeps **one persistent `cpi_embed` process**
(`ensureWorker`: `if (proc) return`) and sends the entire `input[]` array as a single batched
request through a serialized queue — indexing a 300-file folder is one worker, N tokenized+encoded
items, not N process spawns. No change required.

### Hand-off P1 — embed worker warmup — ⬜ OPEN (deferred)
First embed call still pays cold start (process spawn + weight upload; bge-small is tiny so it's
sub-second, not the multi-second CUDA-graph capture the chat path pays). Optional: spawn + one
throwaway embed at server start, tied into the existing chat warmup, so the first folder index has
zero stall. Low urgency given bge-small's size.

### Hand-off P2 — build/packaging — ⬜ OPEN (deferred)
`cpi_embed` needs CUDA and isn't in the default build flow. Decide whether `start_local`/install
should build + place it automatically when CUDA is present, so a fresh checkout has working
embeddings with no manual step. (Runtime resolution already handles *finding* it across build dirs;
this is purely about *producing* it.)

<details><summary>Original feature spec</summary>

**Why:** the next Morph direction is private, OFFLINE RAG over the user's own docs/notes — the killer
local-first feature (their info never leaves the machine; CPI retrieves + answers locally). It needs
vector embeddings, and CPI has none today: `POST /v1/embeddings` → 404 (only chat models exist).
Morph is going semantic (not keyword), so this is the blocking item.

**Model — retrieval-tuned, 384-dim.** Recommend **bge-small-en-v1.5 (384d)** or gte-small (384d);
bge-base-en-v1.5 (768d) if you want a quality bump. Must be a RETRIEVAL-trained sentence model
(bge/gte/e5/nomic family) — NOT the chat LLM's hidden states (weak without mean-pooling + contrastive
training), NOT a generic STS model. At personal-corpus scale (hundreds–~10k chunks) 384d is the sweet
spot: 10k × 384 floats ≈ 15 MB in IndexedDB, sub-50ms JS cosine. Skip 1024d+. 512-token input is
plenty (Morph chunks to ~250–400 tokens). English-first is fine; multilingual (bge-m3) only if needed.

**Endpoint — OpenAI-compatible:**
- `POST /v1/embeddings`, body `{ model, input: string | string[], input_type?: "query" | "document" }` →
  `{ object:"list", model, data:[{ object:"embedding", index, embedding:number[] }], usage:{prompt_tokens,total_tokens} }`.
- **`input_type` (the one design decision):** retrieval models embed queries vs passages differently
  (bge/e5 prepend an instruction to the QUERY only). Preferred: the endpoint accepts `input_type` and
  applies the model's prefix internally, so Morph stays model-agnostic (sends `"document"` on ingest,
  `"query"` on search). Alternatively Morph prepends the prefix itself — say which you'd rather.
- **Batch:** accept `input` arrays, return `data[i]` ↔ `input[i]` in order (Morph embeds many chunks/ingest).
- **L2-normalize** outputs (cosine == dot product; Morph does similarity in JS — can normalize itself if not).
- **Plain float arrays** (`encoding_format: float`, not base64). **Deterministic** (same text → same vector).
  Fixed dim per model; Morph reads it from the first response. Truncate over-length inputs gracefully.

**Verify:** a batch of N inputs → N equal-length vectors in order; identical text → identical vector;
cos(query "capital of France", doc "Paris is the capital of France") > cos(same query, "Bananas are yellow").

**Morph side (parallel, not blocked):** ingest + chunk docs → embed via this endpoint → store vectors
locally → cosine top-k retrieve → an `ask` runtime-LLM atom synthesizes a cited answer from the chunks.
The `ask` atom + ingestion/chunking are embedding-independent, so Morph can build them while this lands.

---

## Done (no action)
- Trailing-tail truncation (ternary `?`, `."`, decimals, URLs) — RESOLVED by gating the Node-layer
  `normalizeGeneratedChatText` trim behind `looksDegenerateRepetition`. See [MORPH_DECODER_BUG.md](./MORPH_DECODER_BUG.md).

See also [MORPH_ENGINE_TODO.md](./MORPH_ENGINE_TODO.md), [MORPH_SUPPORT.md](./MORPH_SUPPORT.md).
