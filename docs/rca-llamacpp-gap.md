# RCA: why CPI decodes Gemma E2B at 59-74% of llama.cpp on the same GPU

Date: 2026-07-22. Scope: single-request greedy decode, Gemma 4 E2B, RTX 5090, Windows.
Same weights, same machine, measured the same day:

| precision        | llama.cpp | CPI | ratio |
|------------------|-----------|-----|-------|
| BF16 / fp16      | 226.8     | 168 | 74%   |
| Q8_0 / int8      | 310.5     | 210 | 68%   |
| Q4_0 / int4      | 396.4     | 235 | 59%   |

llama-bench tg400 vs `llama_infer` 400-token runs; both GPU-resident, both graph-executed.
This document explains the gap from this week's measurements — every claim below has a
number or a falsification receipt behind it, and the ledger of *disproven* explanations is
at the end, because half of the RCA is knowing which intuitive causes are NOT it.

## The one-sentence version

The gap is not one defect: it is the compounding of per-op fixed costs in a small-model
latency regime (largest share), kernel interiors that are days old against kernels that
are five years old (second), quant formats retrofitted to kernels rather than co-designed
with them (third), all held in place by a development loop that until this week priced its
kernels with tools that lie under graph execution (the meta-cause), and paid for
deliberately with a set of architectural values that trade single-model speed for
portability and verifiability (the chosen tax).

## Cause 1 — the latency regime: fixed per-op cost x op count, on a model too small to hide it

E2B decode is ~500 kernel launches per token after this week's consolidation (~870 before).
The model's *mandatory* work per token is tiny: int4 weight bytes are ~1.1 GB ≈ 0.7 ms at
achievable bandwidth. Everything else in our 4.3 ms token is per-op overhead of some kind:
dependency latency between kernels, grid ramp-up on small launches, partial-wave execution.

The graph-mode ablation levers price the classes directly (int4, depth ~400):

| class                          | cost/token | note |
|--------------------------------|-----------|------|
| attention (stats+reduce, 35L)  | 0.74 ms   | after the chunk-16 occupancy fix |
| norms / rope / kv-store        | 0.55 ms   | ~350 tiny launches |
| elementwise (add/gelu/scale)   | 0.09 ms   | effectively free — see falsifications |
| quant gemvs + act-quant + head | ~2.9 ms   | vs ~0.9 ms of bytes |

Two structural facts make this a *small-model* phenomenon rather than a CPI-wide one:

- On 8B-class models CPI reaches **79% of the bandwidth roofline** (CUDA) and its Metal
  decode is at 80-91% of llama.cpp *with the gap smallest on the largest model*. The same
  ordering appears on two backends and two GPUs: overhead is fixed, work scales with model.
- llama.cpp pays the regime too — their Q4 token is ~2.5 ms of which ~1.6 ms is also not
  mandatory bytes. They don't escape per-op cost; they've ground it down ~2x finer.

## Cause 2 — kernel interior maturity: every kernel we profiled deeply yielded a win

The strongest empirical pattern of the whole effort: **each time a kernel was actually
measured at its interior (not benchmarked end-to-end), it was found leaving 2-10x on the
table.** This week alone:

- weight-only int4/int8 matvecs read packed rows a *byte at a time* (~190 GB/s effective);
  int4 decoded slower than fp16 until the wide 16-byte-load rewrite (134 → 179).
- decode attention ran 1-block-per-head on head_dim 256 until the split-K rewrite
  (earlier: +60%), then at half occupancy until the chunk-16 change (+10%).
- the tied LM head re-read 800 MB of fp16 embedding every token (~0.5 ms) on quant runs.
- the fp16 gemv had been fixed similarly in an earlier round, as had Metal's GEMM
  (running on half its threads), Metal's reduce, the argmax (149 µs single-block)...

Extrapolation, not speculation: the classes we have *not* yet interior-profiled (the
norm-class 0.55 ms, the attention stats inner loop, gemv ILP under partial waves) are
statistically likely to hold the same kind of findings. llama.cpp's kernels have had five
years and hundreds of contributors of exactly this grind — per-architecture tile tuning,
mmvq per-quant specializations, a flash-decode attention. There is no evidence of a
structural wall; there is evidence of an hours-invested differential.

The attention kernel illustrates the remaining distance concretely: with kv_heads = 1
(GQA 8:1), our per-(head, chunk) stats blocks read the same K/V chunk **eight times**;
a restructured kernel would read it once. That alone bounds a large share of the 0.74 ms.

## Cause 3 — format/kernel co-design: their quant layouts were built for their kernels

llama.cpp's Q4_0 stores scale-with-block (fp16 scale inline in the 18-byte block, one load
stream, no second pointer chase) and quantizes activations to q8_1 *with per-block sums*
precisely so the int4 bias correction costs one multiply per block. Q4_K goes further
(superblocks, 6-bit scales). The formats and kernels co-evolved.

CPI's int4 is a retrofit: two's-complement nibbles packed at load, scales in a separate
array (a second read stream), activations fp16 until this week. The dp4a + perm8 work
closed most of this (int4 189 → 235), but the layout still costs: separate scale reads,
and the perm8 activation buffer exists only because the nibble packing wasn't designed
for integer dot alignment. A format designed for the kernel would fold both away.

The same asymmetry, smaller: rope tables, sandwich-norm ops, and PLE all execute as
separate generic ops in CPI's IR where a model-specific runtime would fuse them at design
time (see Cause 5 — this is partly a choice).

## Cause 4 (meta) — the measurement loop: our tools priced the wrong things for most of the effort

This is the cause that *kept the other causes alive*. The receipts:

- the eager profiler prices every op at ~11-14 µs of which most is launch cost the graph
  amortizes — it pointed at fusion, which measured **neutral** under graph, twice.
- the persistent-kernel experiment (whole token in one cooperative launch) was built on
  boundary-cost arithmetic that the graph had already invalidated — 6% slower.
- the dead-attention-block hypothesis (grids sized for max_ctx) was plausible, cheap to
  believe, and **false** at shallow depth (ablation: neutral).
- ctest passed 15/15 with a kernel in the tree that generated PAD-0 from the first token
  — quant decode had no golden coverage, so green meant nothing there.
- thermal drift reached 3-5% between adjacent runs after hours of benchmarking — every
  non-interleaved comparison that day was noise.

llama.cpp's contributors work with ncu/nsight per-kernel truth as a matter of course. CPI
on this Windows box had, until this week: an eager profiler (lies under graph), end-to-end
tok/s (too coarse), and golden streams (right answers, no prices). The ablation levers
(`CPI_CUDA_ABLATE_ATTENTION/_ELEMENTWISE/_NORMS`) were built mid-effort and immediately
re-ranked the work. **A team optimizing with coarse instruments spends its hours on
falsifications; the falsification ledger below is the invoice.** The single highest-value
infrastructure investment for closing the rest of the gap is per-kernel graph-mode
profiling (Nsight Compute on the replayed graph), not another kernel hypothesis.

## Cause 5 — the chosen tax: CPI's values price some of llama.cpp's speed off the menu

These are not defects; they are the project's stated philosophy, with a known cost:

- **One IR, two executors.** The op plan must stay backend-neutral and complete, so ops
  are granular (norm, rope, kv-store separate) and model-specific mega-ops are resisted
  by design. llama.cpp specializes freely per model and per architecture.
- **Bit-identity gates.** Graph==eager, cat==split, fused==unfused — every optimization
  this week was required to preserve the token stream exactly (or, for lossy quant, be
  verified against a reference first). This discipline caught real bugs repeatedly, and
  it also rejects whole classes of reduction-reordering fusions llama.cpp ships without
  a second thought.
- **Hand-rolled everything, no deps.** The 5-year-tuned kernel library that llama.cpp
  *is* cannot be imported, only re-derived — that is explicitly the point of CPI, and it
  means the maturity differential of Cause 2 closes at the rate of our own profiling
  hours, no faster.

The tax is real but bounded: prefill on Metal reached 93-108% of llama.cpp under the same
constraints, and 8B decode reaches 79% of roofline. The values cost most exactly where
Cause 1 bites — tiny models, latency regime, many small ops.

## What it is NOT — the falsification ledger

Explanations that sound right, were tried or measured this week, and are dead:

1. **"Too many kernels; fuse them."** Elementwise fusion: neutral under graph, twice.
   Whole-class ablation prices all elementwise at 0.09 ms. Dead.
2. **"Kernel launches cost ~4.5 µs each; remove launches."** True eager, false under
   graph — the persistent cooperative kernel (zero launches, grid.sync between ops) was
   6% *slower*. Graphed boundaries pipeline. Dead as stated; survives only as "latency
   between *dependent* kernels", which is Cause 1.
3. **"The attention grids launch mostly dead blocks."** Bucketed grids + window-relative
   mapping: neutral at depth 400. The cost is inside live blocks. Dead (kept for deep
   context, where it is real).
4. **"Quantization will make decode proportionally faster."** Only after the kernels were
   rewritten; with byte-wise loads int4 was *slower* than fp16. Bytes only win when the
   kernel can stream them.
5. **"240+ tok/s may exceed what this GPU does for this model."** llama.cpp does 396.
   Dead the day the local benchmark ran — and note no public benchmark existed; only the
   same-box measurement settled it.

## Ranked residual (what the remaining ~1.8 ms against llama.cpp's Q4 consists of)

1. ~1.5 ms: gemv-chain latency excess (Cause 1 + 2) — needs interior profiling, then
   either mmq-style weights-through-shared, multi-stream graph forks for the independent
   PLE chain, or both.
2. ~0.35 ms: attention interior (Cause 2) — GQA restructure reads K once instead of 8x;
   start from an ncu trace, not a hypothesis (six attention hypotheses died on Metal).
3. ~0.3 ms: norm-class interior + int4 head (Causes 2, 3).

Nothing on this list is structural. The two systemic actions that outrank any single
kernel: (a) stand up per-kernel graph-mode profiling on this box; (b) add golden coverage
for quant decode so green means something there.
