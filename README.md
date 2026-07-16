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

Continuous batching is partly there. The kernels and the engine step exist — a paged KV pool,
batched paged decode, and paged prefill, all sharing `engine::SequenceBlockTable`, the same
backend-free allocator CUDA uses, so a block table means the same thing on both. What is still
`LlamaEngine`-only is the layer above: the scheduler (admission, preemption, block growth) lives
inside that class rather than behind an interface, and the batch worker is compiled out without
CUDA. So single-request serving runs on a Mac today; the multi-user batched server does not yet.

**Measured** (Apple M4, 10-core GPU, 16 GB):

| Model | | GPU weights | decode | prefill |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | fp16 | 1.17 GB | 86.5 tok/s | 2700 tok/s |
| Qwen2.5-0.5B | int4 | 0.51 GB | 146.7 tok/s | 2510 tok/s |
| **Llama-3.1-8B** | **int4** | **4.91 GB** | **20.5 tok/s** | **161 tok/s** |

Prefill is a ~540-token prompt; decode is from a short one (it falls with KV length — see below).

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

| | llama.cpp | CPI | CPI / llama.cpp |
| --- | --- | --- | --- |
| fp16 prefill (~540) | 4163 | 2700 | 65% |
| int4 prefill (~540) | 4066 | 2510 | 62% |
| int4 decode (short ctx) | 198 | 176 | 89% |

The fp16 prefill row read 3730 / 90% until the GEMM bug above was found; the corrected figure is
2700 / 65%. That correction is what makes the table coherent: **both prefill paths sit at ~63%**,
where the story used to be "fp16 is nearly there and int4 is mysteriously behind." There is one
prefill gap, not an int4-specific one — which also means the long, fruitless hunt for int4's
missing speed was chasing a difference that was never real.

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

The lesson generalizes past this kernel: **an optimization validated only by the metric it
improves is not validated at all.** Every lever in this section was accepted or rejected on
wall-clock alone, which cannot distinguish "faster" from "doing less." The GEMM now has a CPU
reference check in `metal_smoke` across T = 1…100 that derives its dispatch from the tile
constants instead of restating them, so the shader and host cannot drift apart again silently.
`cpi_gemm_quant` and the attention kernels still have no such check.

Both prefill paths now sit at ~63% of llama.cpp, so the remaining gap is common to both rather
than something specific to the quantized GEMM.

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
