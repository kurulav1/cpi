# CPI - Cross-Platform Inference

CPI is a local LLM inference engine with a CLI, REST API, and web UI. It supports CPU inference everywhere and CUDA acceleration when a CUDA toolchain is available.

## Benchmarks

Greedy single-request decode throughput on the reference machine below (`--benchmark`, temperature 0,
2048-token context). GPU weights are held resident with `--gpu-cache-all`. Once a prompt's prefix is
cached, warm-request latency is decode-bound, so decode tokens/s is the number that sets end-to-end
latency.

### Reference hardware

- **GPU:** NVIDIA GeForce RTX 5090 — 32 GB, `sm_120` (Blackwell), driver 591.86
- **CPU:** AMD Ryzen 9 9950X3D (16-core)
- **RAM:** 32 GB · **OS:** Windows 11 Pro · **CUDA:** 13.2

| Model | Params | Weights | Peak VRAM | GPU decode (tok/s) | CPU decode (tok/s) |
| ----- | ------ | ------- | --------- | ------------------ | ------------------ |
| Qwen2.5-Coder-3B-Instruct | 3B | fp16 | 10.7 GB | ~132 | ~7.3 |
| Qwen2.5-7B-Instruct | 7B | fp16 | 19.3 GB | ~86 | ~3.4 |
| Llama-3.1-8B-Instruct | 8B | fp16 | 20.2 GB | ~81 | ~3.2 |
| Qwen2.5-Coder-32B-Instruct | 32B | int4 (streaming) | 26.8 GB | ~43 | — |

Peak VRAM is total GPU memory during the run (includes ~1 GB desktop compositor). The 32B on CPU is
omitted: its int4 weights dequantize past the 32 GB of system RAM (out-of-memory). See
[docs/benchmarks.md](docs/benchmarks.md) for methodology and the full context × quant sweep.

### vs llama.cpp

Same fp16 weights on both sides (llama.cpp gets a GGUF converted from the identical checkpoint),
same GPU, greedy, and the two engines are run **interleaved** — the GPU throttles over a long
session, so numbers taken minutes apart are not comparable.

| Model | Test | llama.cpp | CPI | CPI / llama.cpp |
| ----- | ---- | --------- | --- | --------------- |
| Qwen2.5-0.5B | decode 256 | 820 tok/s | 851 tok/s | **104%** |
| Qwen2.5-0.5B | prefill 1024 | 53,252 tok/s | 117,884 tok/s | **221%** |
| Llama-3.1-8B | decode 256 | ~97 tok/s | ~99 tok/s | **~101%** (parity) |
| Llama-3.1-8B | prefill 1024 | 12,768 tok/s | 16,148 tok/s | **126%** |

At 8B, decode is bandwidth-bound — the weights alone are 16 GB per token — so both engines sit near
the same memory roofline and parity is the ceiling for either of them. The small-model decode gap
comes from per-kernel and per-call overhead, which is a large share of a 0.5B token and negligible
on an 8B one.

Reproduce with `llama-bench` (`-p 0 -n 256` for decode, `-p 1024 -n 0` for prefill) against
`--benchmark --temp 0 --eos-token -1 --no-prefix-reuse`. `--no-prefix-reuse` is required for the
prefill numbers: without it a repeated prompt hits CPI's prefix cache and skips prefill entirely.

### Concurrent throughput (continuous batching)

For multi-user serving, continuous batching is CPI's default serving path (paged KV cache + batched
decode): many requests are prefilled into their own paged blocks and decoded together one step at a time.
Aggregate decode throughput below — total tokens/s summed over all concurrent sequences —
measured with `--batch-bench` (greedy, fp16 resident, prompt 8 / 64 new tokens, short context) on the
same RTX 5090. The "1 req" column is the identical engine at batch 1.

| Model | Params | Vocab | 1 req | batch 16 | batch 32 | batch 64 | peak |
| ----- | ------ | ----- | ----- | -------- | -------- | -------- | ---- |
| Qwen2.5-0.5B-Instruct | 0.5B | 152k | ~243 tok/s | 1662 (6.0×) | 1709 (6.3×) | 1974 (7.1×) | 7.1× @64 |
| TinyLlama-1.1B-Chat | 1.1B | 32k | ~263 tok/s | 2896 (10.3×) | 3879 (13.9×) | 4652 (17.2×) | 17.2× |
| Qwen2.5-Coder-3B | 3B | 152k | ~126 tok/s | 999 (7.8×) | 1198 (9.5×) | 1763 (13.8×) | 13.8× |
| Llama-2-7b-chat | 7B | 32k | ~85 tok/s | 1000 (11.7×) | 1532 (17.9×) | 2118 (24.1×) | 24.1× |
| Qwen2.5-7B-Instruct | 7B | 152k | ~85 tok/s | 807 (9.5×) | 1132 (13.3×) | 1475 (17.6×) | 17.6× |
| Llama-3.1-8B-Instruct | 8B | 128k | ~80 tok/s | 775 (9.7×) | 1105 (14.2×) | 1457 (18.2×) | 18.2× |

Notes:

- **Larger, compute-bound models batch better** — the 8B reaches ~18× aggregate and is still climbing
  at batch 64, while the tiny 0.5B plateaus near ~7× (its slim compute can't hide per-step overhead).
- **Smaller vocabularies batch better** — each step transfers a `[batch × vocab]` logit block, so the
  32k-vocab models (Llama-2, TinyLlama) outscale the 128–152k-vocab ones at the same batch size.
- **No single-request penalty on real models** — the batch-1 slowdown (batched GEMM vs the tuned
  single-token kernel) is a small-model artifact and vanishes by 7–8B (~1.0×).
- **Sampling (temperature > 0) currently runs ~half the greedy throughput at high batch** (e.g.
  Llama-3.1-8B batch 64: 747 vs 1457 tok/s) because top-k/top-p sampling runs per row on the host;
  on-device sampling is planned.
- **Shared-prefix reuse** — concurrent requests that share a leading prefix (a common system prompt, a
  multi-turn chat) adopt each other's cached KV blocks instead of re-prefilling. A small per-worker LRU
  keeps several distinct prefixes live at once, so interleaved requests don't evict each other; on a long
  shared prefix this cuts time-to-first-token by up to ~7× (measured 1.18 s → 0.16 s, Qwen2.5-0.5B,
  988-token prefix).
- **VRAM-sized KV pool** — the paged block pool auto-sizes to free VRAM rather than a single context
  window, so how many sequences run concurrently scales with the card. Under genuine over-subscription
  the newest sequences are preempted (and the client told) as a safety net, rather than the server
  crashing. Override the pool size with `LLAMA_INFER_KV_POOL_TOKENS`.
- **Default on the web server** for supported models (opt out with `CPI_BATCH_WORKER=0`); requires
  fp16-resident weights (`--gpu-cache-all`) + paged KV (`--paged-blocks`) and full-attention models.
  Quantized / MoE / streaming models (e.g. the 32B int4) fall back to single-request serving.

### Decode throughput vs context length

Single-request greedy decode (tok/s) as the prefilled context grows. Each decode step's
attention scans the whole KV cache, so throughput falls as the context lengthens (`--benchmark`
with `--tokens-file`, fp16 resident, RTX 5090).

| Model | 512 | 2048 | 4096 | 8192 | 16384 | 32768 |
| ----- | --- | ---- | ---- | ---- | ----- | ----- |
| Qwen2.5-7B-Instruct | ~91 | ~86 | ~80 | ~71 | ~67 | ~65 |
| Qwen2.5-Coder-7B-Instruct | ~90 | ~86 | ~81 | ~72 | ~71 | ~63 |
| Llama-3.1-8B-Instruct | ~86 | ~81 | ~75 | ~65 | ~60 | ~49 |

Decode stays flat to a few thousand tokens, then falls as attention over the KV cache dominates each
step — at 32K it's ~1.4–1.75× slower than at 512. Decode attention is GQA-aware (each K/V entry is
read from HBM once per KV head and shared across its query-head group, not re-read per query head) and
splits the KV sequence across the SMs FlashDecoding-style: at long context each grid block streams
several KV blocks under a running online softmax, so the memory system stays saturated instead of
running thousands of tiny latency-bound blocks. That coarsened split roughly doubles long-context
decode (Llama-3.1-8B at 32K: ~30 → ~49 tok/s, ~34% → ~52% of the weight+KV bandwidth roofline; same
before/after binary, `LLAMA_INFER_ATTN_BPC=1` vs default). Llama-3.1-8B still falls off fastest: it
has 8 KV heads to Qwen2.5-7B's 4, so ~2× the KV to scan per token and a smaller (4× vs 7×) query group
to amortize it over. (Contexts past ~4K need
`--tokens-file`, since a token list that long exceeds the OS command-line limit.)

## Highlights

- CPU and CUDA inference paths
- Continuous batching with a paged KV cache for concurrent multi-user serving (default; VRAM-sized pool, shared-prefix reuse)
- Streaming weights for models larger than VRAM; runtime int8/int4 quantization
- Native `tokenizer.json` and SentencePiece tokenizer support
- React web UI plus Node API bridge in `web/`
- Model auto-discovery, one-command Hugging Face download + `.ll2c` conversion

## Research & Benchmarking

| Document | Contents |
| -------- | -------- |
| [docs/research.md](docs/research.md) | System architecture, design rationale, related work |
| [docs/benchmarks.md](docs/benchmarks.md) | Benchmark methodology, metric definitions, reproduction guide |
| [docs/results/](docs/results/) | Machine-generated JSON and Markdown result files |

### Benchmark tools

```powershell
# Throughput / latency / memory sweep (context × quant × CPU/CUDA)
python tools/bench_sweep.py --model artifacts/model.ll2c --tokenizer path/to/tokenizer.json `
    --context-lengths 128 512 2048 4096 --quant-modes fp16 int8 int4

# WikiText-2 perplexity (fp16 / int8 / int4)
python tools/perplexity.py --model-dir artifacts/model/hf --all-modes

# Aggregate results into Markdown report (updates docs/benchmarks.md)
python tools/bench_report.py --patch-benchmarks
```

## Quick Start

### Prerequisites

| Tool | Version | Needed for |
| ---- | ------- | ---------- |
| C++ compiler | C++20 (MSVC 2022, GCC 11+, or Clang 14+) | building the engine |
| CMake | ≥ 3.24 | build system |
| Python | ≥ 3.10 + pip | model download/conversion (`tools/`) |
| Node.js | ≥ 18 + npm | web UI + REST API (`web/`) |
| CUDA Toolkit | ≥ 12 (optional) | GPU acceleration — auto-detected; CPU-only build otherwise |

No CUDA GPU is required: without a CUDA toolkit CMake falls back to a CPU-only
build automatically (see [Build Modes](#build-modes)). GitHub-hosted CI builds
and tests this CPU-only path on Linux and Windows on every push.

### First-run setup

Linux/macOS:

```bash
./install.sh
```

Windows:

```powershell
install.bat
```

These scripts:

- install Python dependencies from `requirements.txt`
- install web dependencies with `npm ci`
- create `web/.env` from `web/.env.example` if needed
- create `web/config.json` from `web/config.example.json` if needed
- build `llama_infer` if it is missing

### Run the packaged local app

Linux/macOS:

```bash
./start_local.sh
```

Windows:

```powershell
start_local.bat
```

This starts the API and static web app on `http://localhost:3001`.

### Run the dev web stack

Linux/macOS:

```bash
./start_web.sh
```

Windows:

```powershell
start_web.bat
```

This starts:

- API on `http://localhost:3001`
- Vite UI on `http://localhost:5173`

## Apple Silicon (Metal)

CPI has a second GPU backend. Both execute the same op-plan IR (`include/engine/op_plan.hpp`)
and the same plan built by `build_llama_plan()` — Metal is a backend, not a fork. Adding Qwen3
(per-head QK-norm) and Gemma (GeGLU, embedding scale) needed **zero new kernels**: they are
capability flags on the geometry, not forks.

```bash
cmake -S . -B build -DLLAMA_ENGINE_ENABLE_CUDA=OFF -DLLAMA_ENGINE_ENABLE_METAL=ON       -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

export CPI_METAL_SOURCE=$PWD/src/kernels/metal/cpi_kernels.metal
./build/metal_infer model.ll2c --tokenizer tokenizer.json     --prompt "Explain in two sentences why the sky is blue."     --max-new 80 --temp 0.8 --top-p 0.95 --quant 4
```

**No Xcode required.** The offline `metal` compiler ships with Xcode, not with the Command
Line Tools, so CPI compiles its shaders at runtime via `newLibraryWithSource` — that needs only
the Metal framework, which every Mac has. A stock Mac with the CLT is enough. If Xcode *is*
present, CMake precompiles a `.metallib` and skips the runtime step.

**Supported.** fp16 and weight-only int4/int8, dense uniform-geometry decoders (Llama 2/3,
Mistral, Qwen2.5, Qwen3, Gemma), batched prefill, single-token decode, and the full sampler
(temperature, top-k, top-p, repetition penalty, n-gram blocking) via CPI's shared implementation.

**Single-request serving runs on Metal.** The main `cpi` binary — which the REST/web bridge spawns
— dispatches to the Metal engine on Apple Silicon (it used to fall back to the scalar CPU engine).
The interactive JSON streaming protocol the web UI uses works end to end (token deltas, temperature
sampling, metrics), and `--weight-quant int4/int8` serves a quantized model, so a large model fits
on a small Mac. On an M4, Qwen2.5-0.5B streams at ~54 tok/s (fp16) / ~87 tok/s (int4), versus
~2 tok/s on the CPU fallback.

**Still not on Metal**, and rejected loudly rather than half-done: MoE, the vision tower, and
linear attention (Qwen3.5). Those need work above the kernels.

Continuous batching now runs on Metal — `--interactive-batch`, the same flag and the same
multi-user worker the CUDA server uses. Verified on an M4 with two concurrent requests: the
second is admitted while the first is mid-generation, both then advance one token per step
interleaved, and the first retires without disturbing the second.

It is the *same scheduler*, not a Metal port of one. `engine::BatchScheduler` owns admission,
newest-first preemption, block growth and the shared-prefix LRU; a backend supplies only the two
operations that need a GPU (a suffix prefill and a batched decode step). Metal cannot drift from
CUDA's serving policy because it never states one. That scheduler used to live inside
`LlamaEngine`, which is the only reason batching was ever CUDA-only — of its ~200 lines, two
touched a GPU.

**Verifying it.** Every kernel family has a CPU-reference check (`metal_smoke`, 37 checks),
plus golden token streams that must reproduce the CUDA backend exactly. `tools/metal_verify.sh`
runs the lot in one command on any Mac. This is not ceremony: GitHub's macOS runners have no
GPU, so CI can only compile Metal, never execute it — and a kernel that no gate executed was
silently corrupting every fp16 prompt of >=16 tokens for weeks (see the GEMM note below).
To make that check automatic you need an Apple Silicon runner —
[docs/metal-ci-runner.md](docs/metal-ci-runner.md) is the recipe.

**Measured** (Apple M4, 10-core GPU, 16 GB):

| Model | | GPU weights | decode @64 | decode @256 | prefill |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | fp16 | 1.17 GB | 88.2 tok/s | 82.4 tok/s | 2999 tok/s |
| Qwen2.5-0.5B | int4 | 0.51 GB | 178.3 tok/s | 156.0 tok/s | 2800 tok/s |
| **Llama-3.1-8B** | **int4** | **4.91 GB** | **20.5 tok/s** | — | **161 tok/s** |

Prefill is a ~540-token prompt; decode is `--max-new N` from a short one. Decode is quoted at two
lengths on purpose: it falls as the KV cache grows, so a single figure only means something next to
the token count that produced it (below, that omission cost us a whole bogus comparison). The 8B row
is carried from an earlier machine and was not re-measured here.

> **The fp16 prefill figure used to read 3730 tok/s, and that number was an artifact.** It is
> worth keeping the story attached to the table it corrupted.
>
> `cpi_gemm_f16` gives each simdgroup a 32×32 sub-tile, so its thread count has to follow its
> row tile. When that tile was raised 64 → 128, the shader changed and the host's thread count
> did not — so it ran 4 simdgroups against a 128-row tile and **wrote only half the rows of
> each one**. Every fp16 prompt of ≥16 tokens (where the GEMM replaces the GEMV) decoded from a
> corrupted prefill.
>
> It survived because **a kernel that skips half its writes is not slower, it is faster**. The
> benchmark rewarded it: re-running the same measurement on the broken kernel still reproduces
> 3563–3720 tok/s, which is where 3730 came from. And no correctness gate ever executed it —
> `metal_smoke` did not cover the GEMM at all, and every golden prompt is 10–12 tokens, below
> the threshold. The engine's fastest fp16 path was benchmarked every session and verified
> never.
>
> Two things fell out once it was fixed. The "+37% from a taller tile over 8 simdgroups" was
> **the entire bug** — with the geometry consistent, 128 rows over 8 simdgroups is ~3% *slower*
> than 64 over 4, so the tile is back to 64 and its rationale (more simdgroups hide fragment-load
> latency) is unsupported. And the int4 GEMM, whose row tile really is 64, was never affected:
> int4 prefill was long thought to lag fp16 badly, and a long hunt for the missing speed found
> nothing because there was nothing to find. It was being compared against a number that did not
> exist.

The 8B is the point of quantization: at fp16 its weights are ~15 GB and it does **not** fit in a
16 GB Mac — attempting it drives the machine into swap. At int4 it runs in 4.91 GB with zero
swaps. (The 3.05x saving beats Qwen2.5's 2.3x because the 8B's LM head is untied and gets
quantized too; Qwen2.5 ties its head to the embedding table, which must stay fp16 for the lookup.)

### vs llama.cpp, same machine, same weights

llama.cpp's GGUF is rebuilt from CPI's own `.ll2c` checkpoint (`tools/ll2c_to_hf_safetensors.py`
→ `convert_hf_to_gguf.py` → `llama-quantize Q4_0`), so both sides run the same 4-bit
Llama-3.1-8B on the same M4.

| | llama.cpp | CPI | CPI / llama.cpp |
| --- | --- | --- | --- |
| decode (64 tok) | 22.9 tok/s | 20.5 tok/s | 90% |
| prefill (551 tok) | 241 tok/s | 161 tok/s | 67% |

Decode is measured from a short prompt on both sides; it falls to ~17.8 tok/s once the KV cache
holds 551 tokens, which is the shape of attention, not of the backend.

Not parity, and worth being precise about where the remaining gap is rather than rounding it
off. Both figures above are from `llama-bench -p 512 -n 64` and CPI's `--benchmark` on the same
machine, same weights, minutes apart.

Prefill is 75% one kernel — the quantized blocked GEMM — and it runs at **~3.5 TFLOP/s, about
85% of this chip's rated 4.26 TFLOPS FP32 peak.** llama.cpp sustains ~3.9 TFLOP/s across its
*whole* prefill, so its GEMM is faster than the FP32 peak: it is reaching fp16 rate somewhere we
are not. Where, we don't know yet — and the honest version of that sentence is more useful than a
guess.

What is known, because it was measured and not assumed:

| tried | result |
| --- | --- |
| double-buffered K-blocks | +1 tok/s — reverted |
| 4×4 register tiling (16 matrix ops per 8 loads, was 8 per 9) | +4 |
| widened the token tile 64 → 128 (halves weight traffic) | 0 |
| K-major weight tile, dropping the transposed fragment load | +1 |
| deepening the K-block 32 → 64 | **−13** (costs occupancy) |
| activation fragments straight from device (frees threadgroup memory) | **−7** |
| padding the tile stride to break 8-way bank conflicts | **−4** |
| **half accumulators** (the only path past the FP32 peak) | **−107** — Apple's matrix units have no fast half-accumulate path |

Eight attempts at that inner loop, none worth more than 5%, and the two that *should* have been
free both lost. That is what a kernel sitting on a hardware wall looks like. The wins that did
land came from structure, not from the loop:

- **The token tail was falling to the scalar GEMV.** A 551-token prompt chunks into 512 + 39, and
  the 39-token tail could not fill a GEMM tile, so it went to the GEMV — which re-streams the
  whole model once per 8-token tile. Those 39 tokens swept 4.9 GB of weights five times and cost
  **a third of the prefill**. Letting the GEMM take the tail as a masked tile: **104 → 142 tok/s**.
- **Prefill attention was O(T²) in device traffic.** Each (token, head) threadgroup walked the
  whole KV cache alone — ~80 GB at ~73 GB/s. Blocking over 8 queries, so a key block serves eight
  of them: **142 → 161**.

Both were found by per-kernel profiling (`CPI_METAL_PROFILE=1`), in one shot, after four
inner-loop optimizations had each moved nothing. The op-level profile said "GEMM = 75%" and was
technically true and completely misleading: `OpKind::Gemv` dispatches two different kernels, and
the cheap-looking one was eating a third of the run.

### Small models, grounded against llama.cpp built on the same Mac

The 8B numbers above compare against a llama.cpp figure taken separately. On a later machine
llama.cpp was **built on the box with Metal** and a Qwen2.5-0.5B GGUF (same checkpoint as CPI's
`.ll2c`, quantized to Q4_0 with `llama-quantize`), for a same-weights head-to-head. The 0.5B gap
is *wider* than the 8B's, and the honest table is worth keeping:

Both sides run the same protocol per row -- **the same token count on both**, which sounds too
obvious to state and is exactly what went wrong twice below: `llama-bench -p N -n 0` / `-p 0 -n N`
against CPI's `metal_infer` with an N-token prompt / `--max-new N`.

| | llama.cpp | CPI | CPI / llama.cpp |
| --- | --- | --- | --- |
| fp16 prefill, 541 tok | 3943 | 2999 | 76% |
| fp16 prefill, 640 tok | 4063 | 3128 | 77% |
| fp16 prefill, 1024 tok | 4045 | 3057 | 76% |
| int4 prefill, 541 tok | 3883 | 2800 | 72% |
| int4 prefill, 640 tok | 3959 | 3055 | 77% |
| int4 prefill, 1024 tok | 3921 | 2985 | 76% |
| fp16 decode @ depth 256 | 96.1 | 87.1 | 91% |
| fp16 decode @ depth 512 | 95.5 | 86.2 | 90% |
| fp16 decode @ depth 1024 | 94.6 | 84.9 | 90% |
| fp16 decode @ depth 2048 | 89.3 | 79.4 | 89% |

> **The int4 decode row used to read 198 / 176 / 89%, and the 89% was not real.** Both numbers were
> honestly measured and the comparison between them still wasn't: CPI's 176 came from a 64-token
> decode, llama.cpp's 198 from `llama-bench`'s 256-token one. A decode's speed falls as the KV
> cache grows, so that row raced our short run against their long one. Matched, it is 81% at 64
> tokens and 72% at 256. Nothing regressed -- re-measuring at 64 still reproduces 87.3 / 179.5
> against the old 86.5 / 176. The lesson is that two correct measurements do not make a correct
> ratio, and a table has to state the protocol or it will eventually compare different ones.

Splitting decode by length is what exposed the second gap, and it turned out to be a bug rather
than a gap. Decode against context depth, fp16, 32 tokens generated at each:

| depth | llama.cpp | CPI before | CPI now |
| --- | --- | --- | --- |
| ~16 | 96.0 | 88.4 | 89.2 |
| 256 | 96.1 | 74.6 | 85.5 |
| 512 | 95.5 | 63.6 | 80.8 |
| 1024 | 94.6 | 49.2 | 76.1 |
| 2048 | 89.3 | 34.0 | **67.3 (+98%)** |

**Decode used to collapse 2.6x over 2048 tokens where llama.cpp stayed flat**, and the giveaway was
that it could not possibly be bandwidth: at 2048 keys the KV cache is ~25 MB against 1.17 GB of
weights, so 2% of the traffic was somehow costing 160% of the time. It was parallelism.
`cpi_attention_decode` gives one threadgroup to each (token, head), which for a PREFILL is
thousands of them -- but decode has one token, so the grid was `heads`: **14 threadgroups on a
10-core GPU**, each serially walking the whole cache. Splitting the keys across threadgroups and
merging the partial softmaxes (log-sum-exp, so the merge is exact and the tokens are unchanged)
doubles decode at depth. The right split is measured, not reasoned: it keeps improving to ~512
threadgroups, far past one-per-core, because oversubscription is what hides each one's latency.

This is the same bug CUDA had and fixed (a31758c) -- a decode path that quietly degrades to one
block per head. Worth knowing it is a *class* of bug, not an incident.

> **The prefill rows read 63% until the token counts were matched, and that 63% was the same
> mistake as the decode row above.** CPI's figure came from a 541-token prompt; llama.cpp's came
> from `pp512`. Prefill is chunked at a slot's width, so 541 tokens is a full chunk *plus a stub* --
> the worst case -- while 512 is one clean chunk, the best. llama.cpp pays this too (`pp541` is
> 3943 against `pp512`'s 4159, -5.2%), so the honest comparison was never ours-at-worst against
> theirs-at-best. Matched at every length it is a flat ~76%, and the four points above agree to
> within 1% -- which is itself the tell that the earlier number was an artifact of one prompt.

Chasing that artifact is what found the real bug. A GEMM's efficiency climbs with its token count
(0.69 TFLOP/s over 28 tokens, 2.96 over 512), so the stub chunk that greedy chunking leaves behind
is disproportionately expensive: **crossing 512 tokens by two cost 19 ms**, and a 540-token prefill
spent 19% of itself on the 5% of tokens that spilled. The slot width was 512 because that was "a
few MB", not because of anything about the hardware. Pricing it off a memory budget instead lets a
1-2k prompt prefill in one chunk. Back-to-back against the pre-change binary, a 541-token prefill
goes **199 ms → 180 ms (+10%)** and a 640-token one **223 → 203 (+10%)**; run-to-run spread is a
couple of percent, so quote the A/B and not the best pair. Chunk
boundaries are invisible to the arithmetic -- the GEMM reduces over `in_dim`, which no split
touches -- so the token stream is bit-identical, which is what the goldens and a before/after diff
at 541 tokens both confirm.

**None of the goldens could have caught a chunking bug**: every one of them is under 128 tokens, so
they never split at all. The gate went green on a change it did not execute, and the diff against
the pre-change binary is what actually tested it.

The split-KV decode above had the identical problem -- it waits for 256 keys and no golden reaches
that depth -- which is why `CPI_METAL_ATTN_SPLIT_MIN` exists: setting it to 1 splits at any depth,
so the existing CUDA-referenced goldens run straight through the new kernels. `metal_verify.sh`
re-runs two of them that way, and a mutation (deleting the merge's `exp(m_c - m)` rescale) confirms
the check fails when it should.

> **That mutation also caught the gate lying.** `metal_decode_test` allowed a "tie": if the Metal
> and CUDA streams first diverge where the top-2 logits are within 0.05, that is genuinely
> explainable, because fp32 summation order differs between backends and a near-equal choice can
> fall either way. But it set the whole verdict from that one token and stopped looking. The broken
> merge forked at a tie, collapsed into repeated EOS for the next hundred tokens, agreed with the
> CPU engine on **21 of 128** -- and the gate printed PASS. It printed that 21/128 too, and did not
> gate on it.
>
> A tie explains the *fork*, not what follows: after it the two are different sequences and
> comparing them to the golden is meaningless. The check now teacher-forces the golden prefix and
> judges every position on its own, so a tie excuses only the token it occurs at. All 10 checks
> still pass under the stricter rule -- nothing was hiding behind it -- but that was luck, and the
> old rule would have passed an arbitrarily broken engine whose first fork happened to be close.

A counter (`MetalContext::gpu_busy_ms()`, GPU wall-clock via command-buffer timestamps) settled
the first question: prefill is **97% GPU-busy**, so the gap is kernel efficiency, not dispatch
overhead. From a starting point of ~47%, three kernel wins closed most of the gap — all transfer
to the 8B:

- **Flash attention on the matrix units** (biggest win). Attention scored QK^T and did P·V with
  per-thread scalar dot products, leaving the matrix units idle through ~21% of prefill. Computing
  both products with `simdgroup_matrix` (fp32 accumulate, the fast path), keeping the per-query
  online softmax, cut attention from 130 ms to 45 ms: **fp16 prefill 2156 → 2764 tok/s**. Gated to
  head_dim ≤ 128; Gemma's 256 keeps the scalar kernel.
- **Parallelized the online softmax.** It ran on 8 threads (one per query) while 248 idled at the
  barrier; `KEY_BLOCK == 32 == simd width`, so one simdgroup per query does the max/weight/sum as
  lane-parallel reductions: **fp16 2022 → 2156**.
- **Attention scored QK^T with a 32-lane reduction** (superseded by the flash kernel, but it was
  the first step): one thread per pair, full dot product, no reduction — **1745 → 2023**.
- **The two GEMMs want different token-tile widths.** The quantized GEMM reads activation fragments
  from device, so a wider tile (128) adds weight reuse at no threadgroup-memory cost: **int4
  1527 → 1903**. The fp16 GEMM stages its activations, so the same widening drops occupancy and
  loses — it keeps 64.
- **THE fp16 GEMM IS F32-ALU BOUND AT ~91% OF ITS LIMITER. It is finished.** Xcode's Metal
  Debugger, on a `.gputrace` from `metal_gemm_bench` (`CPI_METAL_GPUTRACE=<path>`):

  | | |
  | --- | --- |
  | F32 limiter | **90.86%** (utilization 80.57%) |
  | F16 limiter | **0.00%** (utilization 0.00%) |
  | Occupancy target | 37.07% |
  | MMU / last-level cache | 6.58% / 11.98% |

  **F16 utilization is exactly zero**: the inputs are half, but every multiply-accumulate issues
  on the F32 pipe, because the kernel accumulates into `simdgroup_float8x8`. That pipe is the
  limiter. The kernel is not stalling — it is saturating the unit it runs on, and there is no
  headroom in this design.

  This also corrects the number quoted here for a long time. **"53% of peak" was measured against
  the wrong peak** — a theoretical fp16 rate the matrix unit does not deliver when accumulating in
  fp32. Against the F32 matrix rate it is ~91%, and the kernel was never leaving 20% on the table.

  It retro-explains the entire lever sweep below: K-depth flat, taller tile slower, more
  simdgroups slower, bigger accumulator tile spilling. **None could help, because none add F32
  issue capacity.** Every one was aimed at a bottleneck that was not there.

  The only route to the F16 pipe is half accumulators, and that measured **3x slower** on two
  unrelated kernels (Apple's matrix unit accumulates in fp32 natively, so half-accumulate appears
  to be emulated). So further tuning of the *tile* here is a dead end with a receipt.

  > **The "dead end" and the 37% occupancy were both the BENCH, and clean captures of the real
  > pass say something different.** Every reading above came from `metal_gemm_bench` -- one shape
  > run hot, back to back. Genuinely GEMM-only captures of a real prefill (every other op ablated,
  > which only became possible once `CPI_METAL_ABLATE` stopped silently ignoring misspelled names
  > this session) read, at two lengths:
  >
  > | | T=257 | T=511 | bench, one shape |
  > | --- | --- | --- | --- |
  > | occupancy | 94% | (n/a) | 37% |
  > | F32 limiter | 79% | 85% | 91% |
  > | F32 utilization | 70% | 76% | 81% |
  > | instruction-throughput limiter | 69% | 75% | -- |
  >
  > So the real GEMM runs at high occupancy, not 37%, and F32 utilization **climbs with prompt
  > length** -- 70 -> 76 -> 81% -- because the big MLP GEMMs (gate/up/down, 87% of prefill FLOPs)
  > tile better as T grows. The T=257 capture overstated the headroom for that reason. At a real
  > length it is ~15%, not 20%, and the co-limiter is the tell: **instruction throughput at 75%,
  > nearly level with F32's 85%.** The matmul phase is already F32-dense; what idles the F32 pipe
  > is the FILL phase between barriers -- staging each K-block's operands into threadgroup memory
  > (integer address math, uint4 loads) while the matrix units wait.
  >
  > So the phase that idles the F32 pipe is the fill, and the obvious fix -- **double-buffer the
  > K-block staging so block n+1 loads while block n multiplies** -- was built and MEASURED, and it
  > is slower. Two staging slots, one barrier per block instead of two, the fill's device loads in
  > flight through the matmul: correct (541-token stream byte-identical, bench spot-checks pass) and
  > a direct back-to-back A/B read **T=511 168 -> 177 ms and T=2041 769 -> 810 ms, ~5% WORSE**.
  >
  > The reason is the occupancy the same capture reported: **94%**. A threadgroup that stalls in its
  > fill is already covered by the ~dozen other resident threadgroups running their matmuls -- the
  > fill latency was hidden across threadgroups, never actually exposed, so there was nothing to
  > overlap. Double-buffering only doubled the threadgroup memory (8 -> 16 KB), which lets fewer
  > threadgroups be resident, LOWERING the occupancy that was doing the hiding. The per-encoder "F32
  > idles during fill" reading was true and the inference from it was wrong: at high occupancy an
  > idle in one threadgroup is not an idle in the machine. Reverted.
  >
  > So the fill/compute-overlap lever is spent, and with it the last cheap prefill idea. What is
  > left is a genuine formulation gap -- llama.cpp's `mul_mm` sustains more on the same F32 hardware
  > -- plus dispatch-boundary overhead that fusing gate+up and q+k+v would trim by a few percent.
  > Neither is a today-sized win, and the ~15% F32 headroom is now understood to be occupancy-hidden
  > rather than recoverable by scheduling. The prefill number stands at ~76%.

  What that gap IS was narrowed by ruling out the structural explanation. llama.cpp's backend
  prints `MTL,BLAS` and its binary links Accelerate (whose BLAS runs on Apple's AMX coprocessor),
  so the obvious theory was that its prefill GEMMs go through a vendor library we neither use nor
  are allowed to. They do not: **pp512 is flat at 4142 / 4163 / 4163 / 4163 tok/s across 1 / 2 / 4 /
  8 CPU threads**, and an AMX/Accelerate path would scale with threads. The BLAS backend is linked
  but idle; prefill runs on the Metal GPU via ggml's own `mul_mm` simdgroup kernels. So the gap is
  **kernel versus kernel** -- two hand-rolled Metal GEMMs -- not hand-rolled versus vendor, which
  is the answer to whether it is worth chasing at all: it is at least investigable. It does not
  contradict the receipt above for the *tile*, but it means if llama.cpp's is faster on the same
  shapes it has found F32 issue density we have not. That capture has now been read (see the note
  above): the real GEMM is F32-limited at 79%, not 91%, with ~20% of the pipe lost to
  instruction-throughput overhead -- so the density is there to be recovered, and the gap is
  investigable rather than structural.

- **The fp16 GEMM "simdgroup starvation" finding was not real, and its retraction is the more
  useful result.** For two sessions the GEMM sat at ~53% of peak and resisted every lever (wider
  tiles, half accumulators, register tiling). It was then diagnosed as too few simdgroups to hide
  inner-loop `simdgroup_load` latency, and running the 4×4 tile over a taller 128-row block with 8
  simdgroups appeared to confirm it: **2764 → 3730 tok/s**. It was not a fix. The shader's tile
  grew; the host's thread count stayed at 128, so the kernel ran 4 simdgroups over a 128-row tile
  and wrote **half the rows**. The measurement was rewarding skipped work, and no gate caught it
  (see the GEMM note above the benchmark table). Corrected, 128-over-8 is ~3% *slower* than
  64-over-4; the tile is back to 64 and the latency-hiding premise has no evidence behind it. The
  ~53%-of-peak question is **reopened and still unexplained.**

- **The prefill gap is a ~1.3x between our own kernels and our own pass, and it is unexplained.**
  The GEMM bench's per-shape times sum to ~153 ms of GEMM for a 541-token prefill. The pass
  spends ~190 ms. Same kernels, same shapes.

  This entry read *1.5x* and *137 ms* for a while, and the 137 was measuring a shape the engine
  cannot run: the bench defaults to one T=541 dispatch, where a 541-token prefill was really
  512+28 against a 512-wide slot. Benching the chunks the pass actually issues (T=512 at 2.96
  TFLOP/s **plus T=28 at 0.69**) predicts 153 ms, and the stub's cost is what led to the chunking
  fix above. A benchmark has to run the shape the program runs, or it measures a program that
  does not exist.

  Eliminated by measurement, not argument: cache residency (rotating 24 distinct weight
  matrices: 2.85 TFLOP/s, flat), shape mix (all 7 real projections: 2.83; k/v are half-rate at
  1.38 but far too small to matter), GPU clock (Maximum throughout the timed loops), fp16
  accumulators (4.4x slower — the F16 pipe is emulated), and serial dispatch (a concurrent
  encoder bought ~3%).

  Xcode's counters on a real prefill say **nothing is saturated**: F32 limiter 39.95%, ALU
  utilization 24.59%, occupancy target 100%, MMU 4%, LLC 6%, launch 8%. Threads are resident
  and waiting. On the *bench* the same kernel reads F32 90.86% — which is why a limiter read
  off a bench cannot be quoted as the kernel's.

  > **Two entries were struck from that list because the experiments behind them never ran.**
  > `CPI_METAL_ABLATE` took a comma-separated list of op-kind names and silently ignored any name
  > that matched no op -- while still announcing `ABLATION ACTIVE`. The op is called `Attention`;
  > the runs said `Attn`. So "non-GEMM ops cost ~3.5%" was measured with attention still running,
  > and the "GEMM-only" `.gputrace` whose counters retired the dilution theory was never GEMM-only
  > -- it contained every attention dispatch. Both conclusions are withdrawn. The switch now
  > refuses to start on a name it does not know and prints the valid ones, because a profiling aid
  > that quietly measures something other than what was asked is worse than not having one: it
  > produces confident numbers, and confident numbers get written down. (Ablating `Attention` for
  > real is what then found the decode collapse above, worth 2x.)
  >
  > The prefill-chunking entry is struck too, for a different reason: it was true and became
  > obsolete. A 481-token prompt was one chunk against a 512-wide slot, so "one chunk vs chunked"
  > compared 481-in-one against 541-as-512+28 and read the tail's cost as noise.

- **Prefill attention costs 30 ms of a 193 ms prefill (15.5%) and runs at 0.42 TFLOP/s — a
  SEVENTH of our own GEMM's 2.86.** It is the largest identified lever left, and the mechanism is
  known: `cpi_attention_prefill_mm` blocks 8 queries per threadgroup, so each head re-reads the
  whole K/V cache once per 8 queries — 68 passes at T=541. Widening the block to 16 measured
  **193 → 177 ms with attention's share falling 30 → 12 ms** — and produced WRONG TOKENS. The
  fragment plan scored exactly one `simdgroup_float8x8` from row 0, so the second half of a
  16-query block was never written: the fp16 GEMM's bug precisely, a tile widened past the threads
  that serve it, and **the 9% "win" was just doing half the work.** Third time in one day that a
  speedup turned out to be a correctness bug.

  Making the matrix ops loop over `QMM_BLOCK/8` fragments fixes the correctness, and then the win
  is real but not for the reason it was tried. It is **not** cutting K/V re-reads — those are
  cache-served (a layer's K+V is ~277 KB at T=541, held in the LLC without trying), and the "3 GB
  of redundant traffic" that motivated it was DRAM-naive arithmetic. What 16 does is fill the
  scoring matmul: it runs `qfs·n_kg` simdgroup work items, and with 4 key groups one query fragment
  leaves 4 of 8 simdgroups idle where two fill them all. At 2041 tokens, where attention is 39% of
  the pass, **attention goes 330 → 301 ms (−9%)** from 8 to 16 and back to 330 at 32.

  This first read as a dead end because it was measured at 541 tokens, where attention is 15% of
  prefill and 9% of that is under the run-to-run noise. Both measurements were right; the
  conclusion was drawn at the length where the effect is smallest. **Measure a small term where it
  is small and you will call it zero** — the same mistake, in the same session, as testing the
  decode split before a prompt was deep enough to need it. `QMM_BLOCK` is 16 (its own constant: the
  scalar kernel's `Q_BLOCK` is stuck at 8 by Gemma's head_dim 256, whose golden *skips* without a
  checkpoint, so a shared bump would have broken it invisibly).

  So attention's cost is **not** redundant KV traffic. What it is, on the decode side, is now
  measured rather than guessed.

- **Decode attention costs 496 us fixed + 1.838 us per key, per token.** Ablating `Attention`
  across depth gives a decode time of **345 / 346 / 346 / 346 ms at depths 257 / 512 / 1022 /
  2042** -- flat to a millisecond over an 8x range, which is the control every earlier attempt
  lacked: it proves attention is the ONLY depth-dependent cost, and that the ablation is really
  removing it. Attention itself reads 968 / 1375 / 2375 / 4250 us, and the linear fit predicts the
  held-out points to within 0.1% (depth 1022: 2374 predicted, 2375 measured).

  The fixed 496 us is 48 dispatches per token (split + merge, 24 layers) of launch overhead, ~4% of
  a token. The per-key 1.838 us is the other 88% at depth 2048, and it is the whole gap:
  llama.cpp's entire attention there is ~800 us against our 4250.

  Each key-head is a 64-element dot product, and it was running at **~23 GFLOP/s, 0.5% of this
  chip's peak**, because the scoring loop was reduction-bound rather than compute-bound:

  ```metal
  for (uint i = lane; i < p.head_dim; i += 32u) d += q_sh[i] * float(kt[i]);
  d = simd_sum(d);   // 2 MACs per lane, then a 5-step shuffle reduction -- PER KEY
  ```

  Two MACs of work per lane and then a full cross-lane reduction, for every key. **Giving each
  THREAD its own key removes the reduction entirely**, and the fit says exactly what that was
  worth: the per-key term went **1.838 -> 0.595 us/key, 3.1x**, while the 496 us fixed term did not
  move (503 us) -- as it should not, since it is dispatch overhead and the dispatch count is
  unchanged. A model that predicts which half of a number a change will move, and is right, is
  worth more than the change.

  One key per thread needs as many keys in flight as the threadgroup is wide, which is why the
  decode kernel runs a **64-thread** threadgroup against the 256 the rest of the file uses: chunks
  sized to fill the GPU are only ~56 keys, so 256 threads would idle 200 of them and swap one
  bottleneck for another. `DEC_KEY_BLOCK`/`DEC_TG` are its own constants -- `KEY_BLOCK` is shared
  with the prefill kernels, the same lesson as `QMM_BLOCK` above.

  Decode is now **87.1 / 86.2 / 84.9 / 79.4 tok/s at depths 257 / 512 / 1022 / 2042** -- 89-91% of
  llama.cpp at every depth, and nearly flat, against 38% at depth 2048 before any of this.

  The same redundancy is the whole decode gap: ablating `Attention` at depth 2048 gives 66.9 →
  **92.5 tok/s**, so attention is 4.16 ms of a 14.9 ms token where llama.cpp spends ~0.8 ms, and
  decode without it is flat in context. **One fix, both gaps.**

The lesson generalizes past this kernel: **an optimization validated only by the metric it
improves is not validated at all.** Every lever in this section was accepted or rejected on
wall-clock alone, which cannot distinguish "faster" from "doing less." The GEMM now has a CPU
reference check in `metal_smoke` across T = 1…100 that derives its dispatch from the tile
constants instead of restating them, so the shader and host cannot drift apart again silently.
`cpi_gemm_quant` and the attention kernels still have no such check.

Both prefill paths sit at ~63% of llama.cpp, so the remaining gap is common to both rather than
something specific to the quantized GEMM. And it is the GEMM: **every other op in prefill costs
~3.5% combined**, measured by deleting them (`CPI_METAL_ABLATE`) rather than by profiling them.

That number matters because the profiler says otherwise. `CPI_METAL_PROFILE` reported the
non-GEMM ops at ~34%, which made fusing them look like the obvious next move — but it serialises
the pass, giving every tiny dispatch its own command buffer, so it inflates precisely the small
ops one is asking about (it charges 551 ms of GPU work to a pass that really takes ~190). Removing
each op and re-timing puts SiluMul, KvStore and attention at ~0 apiece and all of them together
at ~7 ms of 200. **Fusion is not the lever.** Prefill is 95% GPU-busy, so it is not dispatch
overhead either.

**Correctness.** Qwen2.5-0.5B, Qwen3-0.6B and gemma-2b each reproduce the CUDA backend's greedy
token stream exactly on an Apple M4 (`src/tests/golden/`). `metal_decode_test` gates on two
oracles: the fp32 CPU engine for the first forward (argmax + a bounded logit diff), and a CUDA
golden for the whole stream. Quantized runs keep the CPU-oracle bound (looser, but still a bound)
and skip the golden — a quantized model is a *different* model and cannot reproduce an fp16 stream.

**Two traps worth knowing, both found the hard way:**

- **Metal's `tanh` overflows to NaN.** It evaluates as `(exp(2z)-1)/(exp(2z)+1)`, and `exp(2z)`
  exceeds fp32 past `z ≈ 44`, where CUDA's `tanhf` saturates. Gemma reaches `z ≈ 62` on ordinary
  activations, so the GeGLU produced NaN on real inputs. Use the sigmoid form:
  `0.5*(1+tanh(z))` **is** `sigmoid(2z)`, which is safe at both ends.
- **When a new backend disagrees with a reference, the reference is a suspect too.** Bringing this
  backend up surfaced three long-standing bugs in `CpuLlamaEngine` — a hardcoded `rope_theta`,
  missing QK-norm, and no Gemma support at all — each producing a fluent, plausible, *wrong* model.
  All three were briefly mistaken for "expected numerical drift" in the new backend. That
  explanation is exactly what makes a real bug invisible.

## Build Modes

### Default configure

A normal configure now works on machines without CUDA. If CMake cannot find a usable CUDA compiler or toolkit, CPI automatically falls back to a CPU-only build.

```bash
cmake -S . -B build
cmake --build build
```

### Explicit CPU-only build

Use this when you want a guaranteed no-CUDA build:

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release
```

Equivalent manual configure:

```bash
cmake -S . -B build/cpu-release -DLLAMA_ENGINE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-release
```

### Distributable CUDA build

For binaries that should run across multiple NVIDIA GPU generations, use the preset with explicit fatbin targets instead of `native`:

```bash
cmake --preset cuda-distributable-release
cmake --build --preset cuda-distributable-release
```

That preset builds for:

- `75` - Turing / RTX 20xx
- `80` - Ampere A100
- `86` - Ampere RTX 30xx
- `89` - Ada RTX 40xx
- `90` - Hopper H100

For local machine-specific builds, the default CUDA path still uses `CMAKE_CUDA_ARCHITECTURES=native`.

## Web Configuration

The web server reads `web/config.json`, with environment variables overriding it.

Create the config manually if needed:

```bash
cd web
cp config.example.json config.json
```

Set at least:

```json
{
  "modelPath": "/path/to/your/model.ll2c",
  "tokenizerPath": "/path/to/your/tokenizer.json"
}
```

Useful environment variables:

- `LLAMA_INFER_BIN`
- `LLAMA_MODEL_DIRS`
- `LLAMA_MODEL_PATH`
- `LLAMA_TOKENIZER_PATH`
- `LLAMA_CHAT_TEMPLATE`

`web/.env.example` now uses cross-platform defaults:

- `LLAMA_INFER_BIN` is blank by default so the server auto-detects the right binary path
- `LLAMA_MODEL_DIRS` defaults to `../artifacts`
- model-specific paths are blank until you choose a model

## Scripts

- `install.bat` / `install.sh`: install dependencies, create default web config, build `llama_infer` if missing
- `start_local.bat` / `start_local.sh`: packaged local app flow
- `start_web.bat` / `start_web.sh`: API + Vite dev flow
- `start_docker.bat` / `start_docker.sh`: Docker-based flow

All repo-managed Node install paths use `npm ci`.

## API

The server runs on port `3001` by default.

### `GET /api/health`

Returns server status, busy flag, and active runtime configuration.

### `GET /api/models`

Returns discovered model profiles.

```bash
curl http://localhost:3001/api/models
```

### `POST /api/generate`

Blocking inference.

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is CUDA?"}]}'
```

### `POST /api/chat/stream`

Streaming inference with newline-delimited JSON events.

```bash
curl -X POST http://localhost:3001/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain RoPE embeddings."}]}'
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

```bash
curl -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

## Preparing Models

Easiest — download from Hugging Face and convert to `.ll2c` in one step (auto-discovered afterwards):

```bash
python tools/hf_download.py download Qwen/Qwen2.5-Coder-7B-Instruct
```

Or convert an existing local Hugging Face checkpoint manually:

```bash
python tools/convert_hf_to_bins.py \
  --hf-dir /path/to/model \
  --out-dir /path/to/model_bins
```

```bash
python tools/pack_ll2c.py \
  --input-dir /path/to/model_bins \
  --output /path/to/model.ll2c
```

Validate the result:

```bash
python tools/validate_ll2c.py /path/to/model.ll2c
```

TinyLlama note:

- prefer `tokenizer.json` over `tokenizer.model`
- use `tinyllama-chatml` only when you specifically want the role-marker format

## Useful Commands

CLI run:

```bash
./build/llama_infer /path/to/model.ll2c \
  --prompt "Hello" \
  --tokenizer /path/to/tokenizer.json \
  --max-new 64
```

Windows Release binary:

```powershell
.\build\Release\llama_infer.exe C:\path\to\model.ll2c --prompt "Hello" --tokenizer C:\path\to\tokenizer.json --max-new 64
```

## Notes

- CPU-only builds skip CUDA tools such as `cuda_bandwidth_bench` and `moe_kernel_parity_test`.
- The web server can auto-detect several model settings, but `web/config.json` or env vars should still point at a real model/tokenizer pair.
- This repo is still performance-oriented and evolving; the portable first-run path is much better now, but some model-specific tuning remains manual.
