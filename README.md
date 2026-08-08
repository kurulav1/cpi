# CPI - Cross-Platform Inference

CPI is a local LLM inference engine with a CLI, REST API, and web UI. It runs on CPU everywhere,
on NVIDIA GPUs through CUDA, and on Apple Silicon through Metal. The two GPU backends execute
the same model plans and are held to token-identical output by cross-backend gates. No external
runtime dependencies beyond the vendor toolkits: every decode kernel, attention kernel, quant
format, tokenizer, and container reader is in this repo. The one exception worth naming is that
large prefill/batch GEMMs on CUDA go through cuBLAS/cuBLASLt (part of the CUDA toolkit the build
already requires) -- hand-rolling a competitive fp16 tensor-core GEMM buys no user-visible win.
Everything decode-side, and everything on Metal, is hand-rolled.

## Benchmarks

Greedy single-request decode throughput on the reference machine below (`--benchmark`, temperature 0,
2048-token context). GPU weights are held resident with `--gpu-cache-all`. Once a prompt's prefix is
cached, warm-request latency is decode-bound, so decode tokens/s is the number that sets end-to-end
latency.

### Reference hardware

- **GPU:** NVIDIA GeForce RTX 5090, 32 GB, `sm_120` (Blackwell), driver 591.86
- **CPU:** AMD Ryzen 9 9950X3D (16-core)
- **RAM:** 32 GB · **OS:** Windows 11 Pro · **CUDA:** 13.2

| Model | Params | Weights | Peak VRAM | GPU decode (tok/s) | CPU decode (tok/s) |
| ----- | ------ | ------- | --------- | ------------------ | ------------------ |
| Qwen2.5-Coder-3B-Instruct | 3B | fp16 | 10.7 GB | ~132 | ~7.3 |
| Qwen2.5-7B-Instruct | 7B | fp16 | 19.3 GB | ~86 | ~3.4 |
| Llama-3.1-8B-Instruct | 8B | fp16 | 20.2 GB | ~81 | ~3.2 |
| Qwen2.5-Coder-32B-Instruct | 32B | int4 (streaming) | 26.8 GB | ~43 | n/a |

Peak VRAM is total GPU memory during the run (includes ~1 GB desktop compositor). The 32B on CPU is
omitted: its int4 weights dequantize past the 32 GB of system RAM (out-of-memory). See
[docs/benchmarks.md](docs/benchmarks.md) for methodology and the full context × quant sweep.

CPU prefill is batched (chunked cache-blocked GEMM -- weights stream from RAM once per 256-token
chunk instead of once per token, with an AVX-512 tier behind a runtime CPUID check): ~92 tok/s on
the 8B and ~1180 tok/s on the 0.5B at a ~640-token prompt, a ~30× win over the old token-by-token
prefill. Against llama.cpp's CPU backend (build `b4aa7dd47`, zen4 kernels, GPU hidden -- note
`-ngl 0` alone is not a CPU benchmark: with a CUDA build present, llama.cpp still runs prompt
processing on the GPU) on the same box and weights, CPI CPU decode is within
~90% (3.15 vs 3.49 tok/s, both bandwidth-bound) and CPU prefill reaches ~50% (llama.cpp's mature
fp16 GEMM kernels keep the rest); the CPU engine's first job is being the verification oracle,
and `CPI_CPU_PREFILL_TOKENWISE=1` restores the legacy path (`CPI_CPU_PREFILL_CHUNK` tunes the
chunk; larger helps on big-L3 parts).

### vs llama.cpp

Same fp16 weights on both sides (llama.cpp gets a GGUF converted from the identical checkpoint),
same GPU, greedy, and the two engines are run **interleaved**: the GPU throttles over a long
session, so numbers taken minutes apart are not comparable.

Provenance: all numbers are self-reported from the single reference machine above; each table
states when its rows were measured and any row carried from an earlier pass says so. Timer scopes
differ between the tools -- `llama-bench` excludes sampling, CPI's `--benchmark` includes it -- which
flatters llama.cpp by a small margin on decode; the ratios below leave that in rather than adjust
for it. A fresh 2026-08-05 re-run against llama.cpp build `1269cb1` (`-fa 1`, warm, interleaved)
put the 8B at decode **99%** (82.2 vs 83.2 tok/s), prefill@512 **133%**, prefill@2048 **106%** --
consistent with the table's earlier rows.

**Same file, both engines.** Since CPI reads GGUF directly, the two engines can now be pointed at
the byte-identical file rather than at two conversions of one checkpoint. Llama-3.1-8B, same RTX
5090, interleaved, decode of 128 tokens after a 133-token prompt (`llama-bench -n 128` against
CPI's `[bench] decode_tok_per_s`, both decode-only):

| File | llama.cpp | CPI | CPI / llama.cpp |
| ---- | --------- | --- | --------------- |
| `llama31-8b-F16.gguf` | 94.3 tok/s | 94.3 tok/s | **100%** |
| `llama31-8b-Q4_K_M.gguf` | 248.3 tok/s | 210.3 tok/s | **85%** |

The F16 row is the honest apples-to-apples case: identical bytes, identical arithmetic, parity. The
Q4_K_M row is not like-for-like -- llama.cpp runs k-quant blocks natively, while CPI dequantizes and
re-packs into its own int4 (`--weight-quant int4`), which costs a second quantization and leaves it
at 4.6 GB on disk but 12.5 GB resident. Loading those blocks natively is the open work; the gap is
the size of the prize.

One weakness the same measurement exposes: CPI's **int4 prefill is much slower than its fp16
prefill** (375 vs 2145 tok/s on a 133-token prompt), so quantized runs pay at prompt ingestion what
they win at decode. Long-prompt workloads should measure both phases before choosing.

| Model | Test | llama.cpp | CPI | CPI / llama.cpp |
| ----- | ---- | --------- | --- | --------------- |
| Qwen2.5-0.5B | decode 256 | 820 tok/s | 851 tok/s | **104%** |
| Qwen2.5-0.5B | prefill 1024 | 53,252 tok/s | 117,884 tok/s | **221%** |
| Llama-3.1-8B | decode 256 | ~97 tok/s | ~99 tok/s | **~101%** (parity) |
| Llama-3.1-8B | prefill 1024 | 12,768 tok/s | 16,148 tok/s | **126%** |

At 8B, decode is bandwidth-bound (the weights alone are 16 GB per token), so both engines sit near
the same memory roofline and parity is the ceiling for either of them. The small-model decode gap
comes from per-kernel and per-call overhead, which is a large share of a 0.5B token and negligible
on an 8B one.

Reproduce with `llama-bench` (`-p 0 -n 256` for decode, `-p 1024 -n 0` for prefill) against
`--benchmark --temp 0 --eos-token -1 --no-prefix-reuse`. `--no-prefix-reuse` is required for the
prefill numbers: without it a repeated prompt hits CPI's prefix cache and skips prefill entirely.

### Concurrent throughput (continuous batching)

For multi-user serving, continuous batching is CPI's default serving path (paged KV cache + batched
decode): many requests are prefilled into their own paged blocks and decoded together one step at a time.
Aggregate decode throughput below, total tokens/s summed over all concurrent sequences,
measured with `--batch-bench` (greedy, fp16 resident, prompt 8 / 64 new tokens, short context) on the
same RTX 5090. The "1 req" column is the identical engine at batch 1.

| Model | Params | Vocab | 1 req | batch 16 | batch 32 | batch 64 | peak |
| ----- | ------ | ----- | ----- | -------- | -------- | -------- | ---- |
| Qwen2.5-0.5B-Instruct | 0.5B | 152k | ~243 tok/s | 1662 (6.0×) | 1709 (6.3×) | 1974 (7.1×) | 7.1× @64 |
| TinyLlama-1.1B-Chat | 1.1B | 32k | ~263 tok/s | 2896 (10.3×) | 3879 (13.9×) | 4652 (17.2×) | 17.2× |
| Qwen2.5-Coder-3B | 3B | 152k | ~126 tok/s | 999 (7.8×) | 1198 (9.5×) | 1763 (13.8×) | 13.8× |
| Llama-2-7b-chat | 7B | 32k | ~85 tok/s | 1000 (11.7×) | 1532 (17.9×) | 2118 (24.1×) | 24.1× |
| Qwen2.5-7B-Instruct | 7B | 152k | ~85 tok/s | 807 (9.5×) | 1132 (13.3×) | 1475 (17.6×) | 17.6× |
| Llama-3.1-8B-Instruct | 8B | 128k | ~80 tok/s | 951 (11.9×) | 1733 (21.7×) | 2469 (30.9×) | 30.9× @64 |

The 8B row is re-measured on the current **device-argmax** greedy path (the default). Before the
device argmax, the same greedy path shipped the full logit block and reached 1795 tok/s at batch 64
(still selectable with `CPI_BATCH_ARGMAX=0`); the older 1457 predates the v2 pinned-staging + scratch
reuse. The other rows predate the device-argmax path and were not re-measured on this pass -- they
would rise similarly.

Notes:

- **Larger, compute-bound models batch better**: the 8B reaches ~31× aggregate and is still climbing
  at batch 64, while the tiny 0.5B plateaus near ~7× (its slim compute can't hide per-step overhead).
- **Smaller vocabularies batch better** (on the full-logits path): shipping a `[batch × vocab]` logit
  block each step is why the 32k-vocab models (Llama-2, TinyLlama) outscale the 128–152k-vocab ones in
  the rows above. The device top-k / argmax paths keep that block off the bus, so the effect now shows
  up only in the LM-head GEMV compute (which still scales with vocab), not the transfer.
- **No single-request penalty on real models**: the batch-1 slowdown (batched GEMM vs the tuned
  single-token kernel) is a small-model artifact and vanishes by 7–8B (~1.0×).
- **Sampling (temperature > 0) decides on device too.** Top-k / top-p and the repetition penalty
  are applied to the batched logits on the GPU, so only ~k candidates per row cross to the host
  instead of the full `[batch × vocab]` block. Two baselines at Llama-3.1-8B batch 64 (top-k 40),
  because the honest comparison depends on whether the host still has its own fast path:
  - **vs the host's own top-k shortcut** (no penalty): 815 → 1918 tok/s, **2.35×**.
  - **at repetition penalty 1.05** (the served default): 453 → 1973 tok/s, **4.36×** -- but part of
    that ratio is the host *losing* its shortcut, not the device winning: `sample_from_logits`
    disables its top-k fast path for penalty > 1 and falls back to a full-vocabulary softmax + sort,
    while the device applies the penalty *before* the top-k and keeps the fast path.

  Greedy device argmax (next bullet) still leads sampling at this batch (2469 vs 1973). The host
  path is selectable with `CPI_BATCH_TOPK=0` (A/B lever + safety switch); grammar-constrained and
  n-gram-blocked requests still sample on the host, since both need the full vocabulary.
- **Greedy (temperature ≤ 0) also decides on device.** A batched argmax returns one winner id per
  row (B ints) instead of the full logit block; the repetition penalty and the min-tokens EOS
  suppression are applied on the GPU first, so the winner matches the host token-for-token. This is
  a smaller win than sampling -- greedy's host work was already a single argmax scan, so only the
  bus transfer is saved -- but still lifts Llama-3.1-8B batch 64 from **1795 to 2469 tok/s** (1.38×).
  Selectable with `CPI_BATCH_ARGMAX=0`.
- **Shared-prefix reuse**: concurrent requests that share a leading prefix (a common system prompt, a
  multi-turn chat) adopt each other's cached KV blocks instead of re-prefilling. A small per-worker LRU
  keeps several distinct prefixes live at once, so interleaved requests don't evict each other; on a long
  shared prefix this cuts time-to-first-token by up to ~7× (measured 1.18 s → 0.16 s, Qwen2.5-0.5B,
  988-token prefix).
- **VRAM-sized KV pool**: the paged block pool auto-sizes to free VRAM rather than a single context
  window, so how many sequences run concurrently scales with the card. Under genuine over-subscription
  the newest sequences are preempted (and the client told) as a safety net, rather than the server
  crashing. Override the pool size with `CPI_KV_POOL_TOKENS`.
- **Quantized KV pool** (`CPI_KV_QUANT=84` conservative, `=4` aggressive): the same paged pool holds
  2.61× / 3.88× more tokens in the same VRAM budget (measured on the 8B: 97k → 254k / 377k pool
  tokens), with an fp16 sink + recent-window tier keeping retrieval quality (needle-exact at 8k and
  30k). Throughput matches fp16 batching at shallow depth and passes it where aggregate KV reads
  dominate (measured crossover: 1024-deep sequences at batch 64). Prefix reuse is disabled under
  the quality tier -- each sequence's fp16 window is its own.
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
step; at 32K it's ~1.4–1.75× slower than at 512. Decode attention is GQA-aware (each K/V entry is
read from HBM once per KV head and shared across its query-head group, not re-read per query head) and
splits the KV sequence across the SMs FlashDecoding-style: at long context each grid block streams
several KV blocks under a running online softmax, so the memory system stays saturated instead of
running thousands of tiny latency-bound blocks. That coarsened split roughly doubles long-context
decode (Llama-3.1-8B at 32K: ~30 → ~49 tok/s, ~34% → ~52% of the weight+KV bandwidth roofline; same
before/after binary, `CPI_ATTN_BPC=1` vs default). Llama-3.1-8B still falls off fastest: it
has 8 KV heads to Qwen2.5-7B's 4, so ~2× the KV to scan per token and a smaller (4× vs 7×) query group
to amortize it over. (Contexts past ~4K need
`--tokens-file` or `--prompt-file`, since a token list that long exceeds the OS command-line limit.)

Correctness note for long contexts: Llama-3.1's usable window past its 8192 base relies on
llama3-style rope scaling, which is **opt-in**: set `CPI_ROPE_SCALING=llama3` (parameters
overridable via `CPI_ROPE_SCALING_PARAMS=factor,low,high,orig_max`). Without it, retrieval
degrades beyond ~8K even though decode runs fine -- the throughput rows above are unaffected,
but quality at 16K/32K needs the flag.

## Highlights

- Three backends (CPU, CUDA, Metal) executing one shared op-plan IR; the GPU backends are gated
  token-identical against each other
- Model families: Llama 2/3/3.1, Mistral, Mixtral, Qwen2/2.5/3, Qwen3.5 (linear attention),
  Gemma, Gemma 4 (per-layer geometry, per-layer embeddings, KV sharing; the 26B-A4B MoE runs
  int4-resident), DeepSeek-V2 (native MLA attention + fine-grained MoE; CUDA), plus BERT-style
  embedding models
- Multimodal: image input (`--image` and web upload) for Qwen3.5 and Gemma 4, on both GPU
  backends, gated against HuggingFace and cross-backend references
- Continuous batching with a paged KV cache for concurrent multi-user serving (default;
  VRAM-sized pool, shared-prefix reuse)
- KV-cache quantization (`CPI_KV_QUANT`): K8V4 conservative or KV4 with a QuaRot-style
  rotation, both with an fp16 sink + recent-window quality tier -- 2.6–3.9× more KV pool in the
  same VRAM, needle-retrieval-gated at 8k–30k context, and at/above fp16 throughput in the
  deep-and-concurrent regime it exists for
- Speculative decoding (`--draft-model`), lossless with respect to the target's own output;
  prompt-lookup speculation needs no draft model at all (`CPI_CUDA_SPEC` / `CPI_METAL_SPEC`)
- Streaming weights for models larger than VRAM; runtime int8/int4 quantization on load
- Grammar and JSON-schema constrained decoding
- OpenAI-compatible API served **from the binary itself** (`--serve`: hand-rolled HTTP + SSE, no
  interpreter and no third-party dependency), driving the same batching scheduler as the web stack;
  plus the embeddings endpoint, React web UI, and Node API bridge in `web/`
- Native `tokenizer.json` and SentencePiece tokenizer support
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

### Download a prebuilt binary

Tagged releases attach prebuilt `cpi` binaries (see the repository's Releases page):

| Asset | Platform | Backend |
| ----- | -------- | ------- |
| `cpi-linux-x64-cpu` | Linux x64 | CPU |
| `cpi-windows-x64-cpu` | Windows x64 | CPU |
| `cpi-macos-arm64-metal` | macOS (Apple Silicon) | Metal GPU + CPU |

Unpack and check the build with `cpi --version`. The macOS archive bundles the Metal
shader sources next to the binary; launch it with the included `./run.sh` (which sets
`CPI_METAL_SOURCE` for you) or set that variable by hand. CUDA is not shipped as a binary (it is
GPU-architecture and driver specific) -- build it from source per [Build Modes](#build-modes).

### Prerequisites

| Tool | Version | Needed for |
| ---- | ------- | ---------- |
| C++ compiler | C++20 (MSVC 2022, GCC 11+, or Clang 14+) | building the engine |
| CMake | ≥ 3.24 | build system |
| Python | ≥ 3.10 + pip | model download/conversion (`tools/`) |
| Node.js | ≥ 18 + npm | web UI + REST API (`web/`) |
| CUDA Toolkit | ≥ 12.5 (optional) | GPU acceleration; auto-detected; CPU-only build otherwise. 12.5 is a hard floor: the MoE grouped-expert path calls `cublasGemmGroupedBatchedEx`, which older toolkits don't have (found the hard way when the Docker image pinned 12.4) |

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
- build `cpi` if it is missing

### Run a GGUF

GGUF files load directly, no conversion step:

```bash
cpi model.gguf --tokenizer tokenizer.json --prompt "Hello" --gpu-cache-all
```

Tensor types: F32, F16, BF16, the flat quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) and the k-quants
(Q2_K/Q3_K/Q4_K/Q5_K/Q6_K). fp16 tensors are served straight from the mapping, so an F16 GGUF
starts as fast as CPI's own container and decodes at the same speed (measured 91 tok/s either way on
an 8B, with generation **token-identical** to the same checkpoint's `.ll2c`).

One thing to know before pointing a quantized file at it: CPI **dequantizes to fp16 at load**, so
by default a 4.6 GB Q4_K_M is resident at the fp16 footprint -- the download shrinks, the model in
VRAM does not. Adding `--weight-quant int4` re-packs it on the GPU and is worth it on both axes,
measured on an 8B Q4_K_M:

| | peak VRAM | decode |
| --- | --- | --- |
| default (fp16 resident) | 20.7 GB | 74 tok/s |
| `--weight-quant int4` | 12.5 GB | **134 tok/s** |

The speedup is the same reason quantization helps anywhere: decode is bandwidth-bound, so fewer
weight bytes per token is directly faster. The cost is a second quantization on top of the file's
own (output stays coherent, but it is not the GGUF's exact numerics). Loading the file's blocks
straight into CPI's packed format -- no fp16 round trip, no double quantization -- is the next step
on this path. Dequantization is multi-threaded meanwhile: an 8B Q4_K_M starts in ~13 s against
~7 s for F16.

Architectures are mapped conservatively: `llama` (verified against a `.ll2c` of the same checkpoint,
bit-exact weights and identical output) and `qwen2`. Anything else is **refused with an
explanation** rather than loaded under the wrong conventions -- an unmapped architecture still has
recognisable tensor names, so the failure mode being avoided is confident nonsense, not a crash.
Gemma, Gemma 4 and DeepSeek-V2 GGUFs need their norm and config conventions mapped first; convert
those to `.ll2c`/`.cpi` meanwhile.

### Serve an OpenAI-compatible API (no Node, no dependencies)

The binary is its own HTTP server, so headless serving needs nothing but `cpi`:

```bash
cpi model.ll2c --tokenizer tokenizer.json --serve --port 8080 --paged-blocks --chat-template qwen2
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions` (both support `"stream": true`
for SSE), `POST /v1/embeddings`, `GET /v1/models`, `GET /health`. `response_format` with a
`json_schema` compiles to the grammar sampler, and a client that disconnects mid-stream has its KV
blocks reclaimed immediately.

**Vision** works over the API too, in the OpenAI content-array shape (a `data:` URL is decoded
in-process):

```json
{"messages":[{"role":"user","content":[
  {"type":"text","text":"What is in this image?"},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0..."}}]}]}
```

**Embeddings** need a BERT-family model: add `--embed-model <dir>`. Without one the route answers
503 with the reason. The vectors are identical to the standalone `cpi_embed` worker's.

**Auth and exposure.** `--api-key <token>` (or `CPI_API_KEY`) requires a bearer token on every
`/v1/*` route; `/health` stays open for load balancers. The server binds `127.0.0.1` by default and
**refuses to start** on a non-loopback `--host` without a key, rather than silently publishing an
unauthenticated model to the network.

**Two serving modes**, reported in the startup line. Models on the LlamaEngine (Llama, Qwen2.5,
Mistral, ...) and everything on Metal serve `mode=batched`: the same continuous-batching scheduler
as the web stack, so concurrent clients interleave rather than queue (4 concurrent chats measured at
2.23x the serial time on a 0.5B). Models on the op-plan CUDA engine (Gemma 4, Qwen3.5, DeepSeek-V2)
serve `mode=serial`: that engine has no batch scheduler yet, so requests queue behind one lock. They
are still fully served, vision included; they just do not batch.

The Node stack below remains the way to get the web UI, the model hub, and the admin routes.

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

CPI has a second GPU backend. Both backends execute the same op-plan IR
(`include/engine/op_plan.hpp`) and the same plans built by the shared builders in
`src/engine/op_plan_builder.cpp`. Metal is a backend, not a fork: adding a model means a new
plan or a capability flag, not new engine code, and the two backends are held together by
token-identity gates rather than by intent.

### What runs on Metal

Verified by running on an Apple M4 (16 GB), each against a CUDA or HuggingFace reference:

- **Model families**: Llama 2/3 (Llama-3.2-1B fp16 is token-identical to the CUDA backend;
  Llama-3.1-8B runs int8/int4), Mistral, Qwen2.5, Qwen3, Qwen3.5 (linear attention plus full
  attention), Gemma, Gemma 4 (per-layer geometry, PLE, KV sharing).
- **Multimodal**: both vision towers. Qwen3.5's (`--image`) reproduces the HuggingFace token
  stream 8/8; Gemma 4's matches the CUDA tower's soft tokens at per-token cosine 1.0000. One
  templated splice (`include/app/image_prompt.hpp`) serves both backends.
- **Quantization**: on-load weight-only int8/int4 (`--weight-quant`). int8 reproduces the fp16
  token stream on every model gated so far. The 8B-class models need it: fp16 weights of ~15 GB
  do not fit a 16 GB Mac, int4 runs in ~5 GB.
- **Serving**: the REST/web bridge, the OpenAI-compatible endpoint, embeddings
  (`MetalBertEmbedder`, gated against its oracle), grammar/JSON-schema constrained decoding,
  and continuous batching with the same `BatchScheduler` the CUDA server uses. The scheduler is
  shared code; Metal supplies only the two GPU operations, so serving policy cannot drift
  between backends.
- **Speculative decoding**: `--draft-model`, lossless by construction and verified
  token-identical. Qwen2.5-3B with a 0.5B draft measures 1.50x on the M4. (An 8B target with a
  1B draft is currently slower than plain decode; profiling that is an open item.)

### Measured

Apple M4, 10-core GPU, 16 GB. Decode falls as the KV cache grows, so decode figures only mean
something next to the token count that produced them.

| Model | Weights | Decode (tok/s) | Notes |
| ----- | ------- | -------------- | ----- |
| Qwen2.5-0.5B | fp16 | 88 @ depth 16, 80 @ depth 2048 | 85-92% of llama.cpp at every depth |
| Qwen2.5-0.5B | int4 | 172 @ depth 16, 142 @ depth 2048 | 77-86% of llama.cpp |
| Gemma 4 E2B | fp16 (BF16 HF dir, no conversion) | 13.9 | at the bandwidth roofline; batched prefill ~3.4 s for a 1250-token prompt |
| Llama-3.1-8B | int8 | 8.8 | token-identical to the CUDA fp16 reference |
| Llama-3.1-8B | int4 | 13.7 | |

Prefill at 2048-token prompts is at parity with llama.cpp (93–108% across the gated models,
2026-07-22 sweep); short prompts remain behind (down to ~50% at ~130 tokens), because the GEMM
carries a fixed per-pass weight-read cost that long prompts amortize and short ones don't. The
GEMM itself measures at its hardware ceiling; the full investigation (including the dead ends
and the retractions) is in [docs/metal-optimization-log.md](docs/metal-optimization-log.md).

### Build and run

```bash
cmake -S . -B build -DCPI_ENABLE_CUDA=OFF -DCPI_ENABLE_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

export CPI_METAL_SOURCE=$PWD/src/kernels/metal
./build/cpi model.ll2c --tokenizer tokenizer.json \
    --prompt "Explain in two sentences why the sky is blue." --max-new 80 --temp 0
```

**No Xcode required.** The offline `metal` compiler ships with Xcode, not the Command Line
Tools, so CPI compiles its shaders at runtime from `CPI_METAL_SOURCE`; that needs only the
Metal framework, which every Mac has. If Xcode is present, CMake precompiles a `.metallib` and
the runtime step is skipped.

A Gemma 4 HuggingFace checkpoint directory runs directly (BF16 converts at load); other
families convert to `.ll2c` first with the tools in `tools/` (see Preparing Models below).

### Verification

Every kernel family has a CPU-reference check (`metal_smoke`), the engine has golden token
streams that must reproduce the CUDA backend exactly (`metal_decode_test`), the vision towers
gate against HuggingFace and CUDA references (`metal_vision_test`), and `tools/metal_verify.sh`
runs the lot on any Mac. This is not ceremony: GitHub's macOS runners have no GPU, so CI can
only compile Metal, never execute it, and the project has a documented history of kernels that
were benchmarked constantly and verified never. The log linked above records those cases;
[docs/metal-ci-runner.md](docs/metal-ci-runner.md) is the recipe for an Apple Silicon runner
that closes the hole permanently.

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
cmake -S . -B build/cpu-release -DCPI_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
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

### Binary footprint

The engine itself is small; the deployment weight of the CUDA build is almost entirely the vendor
GEMM library it links (the one stated exception to hand-rolled kernels):

| Build | Binary | Runtime dependencies |
| ----- | ------ | -------------------- |
| CUDA (single-arch `native`) | `cpi.exe` ≈ 7.9 MB | `cublasLt64_13.dll` ≈ 432 MB + `cublas64_13.dll` ≈ 49 MB (CUDA toolkit), NVIDIA driver, MSVC runtime |
| CPU-only | < 1 MB | MSVC / libstdc++ runtime only |

A multi-architecture fatbin build (`cuda-distributable-release`) grows the binary with each added
`sm_*` target but leaves the dependency picture unchanged.

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

- `CPI_BIN`
- `CPI_MODEL_DIRS`
- `CPI_MODEL_PATH`
- `CPI_TOKENIZER_PATH`
- `CPI_CHAT_TEMPLATE`

`web/.env.example` now uses cross-platform defaults:

- `CPI_BIN` is blank by default so the server auto-detects the right binary path
- `CPI_MODEL_DIRS` defaults to `../artifacts`
- model-specific paths are blank until you choose a model

## Scripts

- `install.bat` / `install.sh`: install dependencies, create default web config, build `cpi` if missing
- `start_local.bat` / `start_local.sh`: packaged local app flow
- `start_web.bat` / `start_web.sh`: API + Vite dev flow
- `start_docker.bat` / `start_docker.sh`: Docker-based flow

All repo-managed Node install paths use `npm ci`.

## API

The server runs on port `3001` by default. The full contract is defined in an **OpenAPI 3.1**
document at [`docs/openapi.yaml`](docs/openapi.yaml), also served live at
`GET /openapi.yaml` -- point any OpenAPI viewer or client generator at it.

There are two namespaces: **`/api/*`** (CPI-native, error envelope `{ "error": "..." }`) and
**`/v1/*`** (OpenAI-compatible, so existing OpenAI clients work by changing the base URL).

**Exposure model.** The server binds `127.0.0.1` by default (`host` in `web/config.json` or
`CPI_HOST`). The admin surface -- model hub downloads, quant jobs, and the filesystem pickers --
additionally refuses any non-localhost client unless you set a bearer token (`adminToken` /
`CPI_ADMIN_TOKEN`) and send `Authorization: Bearer <token>`; when a token is set it is required
from localhost too. So exposing the server (`host: 0.0.0.0`, or `npm run dev:lan` for the dev UI)
shares inference only, never disk access, unless you deliberately configure a token.

In **Docker** this bites by design: the image sets `CPI_HOST=0.0.0.0` (the port mapping needs it),
and even your own browser on the host reaches the container through the docker bridge -- a
non-loopback peer -- so the hub/quant/picker UI returns 403 until you set `CPI_ADMIN_TOKEN`
(the `start_docker` scripts pass it through if exported). Inference works without it. All of this
is verified by running the container: health open, admin 403 bare / 401 wrong-token / 200 with
the token, and OpenAI-compatible chat served end-to-end. The same image runs CPU-only on hosts
without a GPU (automatic fallback -- but note Docker Desktop's WSL2 backend injects the GPU even
without `--gpus`; force CPU with `-e CUDA_VISIBLE_DEVICES=-1`, and set `OMP_NUM_THREADS` to the
physical core count, since SMT oversubscription under WSL2 measured 2.4× slower).

By default inference routes carry no auth (`authScope: "admin"`), so a non-loopback bind still
exposes unauthenticated compute. Set `authScope: "all"` (`CPI_AUTH_SCOPE=all`) to require the
bearer token on every `/api` and `/v1` route except `/api/health`; `/healthz/*` and `/metrics`
stay open for orchestrator probes and scrapers, and the server refuses to start in this mode
without a token. `demoMode` additionally disables the admin surface for everyone and rate-limits
generation per IP.

| Group | Endpoints |
| ----- | --------- |
| Health | `GET /api/health` · `GET /metrics` · `GET /healthz/live` · `GET /healthz/ready` |
| Inference | `POST /api/generate` (blocking) · `POST /api/chat/stream` (NDJSON) · `POST /api/warmup` |
| Models | `GET /api/models` · `POST /api/quant/select` · `GET /api/quant/state` |
| Model hub | `GET /api/hub/search` · `POST /api/hub/download` · `GET /api/hub/status/{jobId}` (SSE) · `GET /api/hub/jobs` · `DELETE /api/hub/jobs/{jobId}` |
| Quantize | `POST /api/quant/convert` · `GET /api/quant/status/{jobId}` (SSE) · `GET /api/quant/jobs` · `DELETE /api/quant/jobs/{jobId}` |
| System | `POST /api/system/model-dir` · `POST /api/system/pick-folder` |
| OpenAI | `POST /v1/chat/completions` · `POST /v1/completions` · `POST /v1/embeddings` · `POST /v1/responses` · `GET /v1/models` · `GET /v1/models/{model}` |

### Common examples

Blocking inference:

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is CUDA?"}]}'
```

OpenAI-compatible chat (drop-in for OpenAI SDKs -- set the base URL to `http://localhost:3001/v1`):

```bash
curl -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

Download and convert a model over the API (the same pipeline as `tools/hf_download.py`, async with
a job id -- poll progress on `GET /api/hub/status/{jobId}`):

```bash
curl -X POST http://localhost:3001/api/hub/download \
  -H "Content-Type: application/json" \
  -d '{"repoId":"Qwen/Qwen2.5-0.5B-Instruct"}'
# → {"jobId":"..."}
```

## Preparing Models

Easiest: download from Hugging Face and convert to `.ll2c` in one step (auto-discovered afterwards):

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
./build/cpi /path/to/model.ll2c \
  --prompt "Hello" \
  --tokenizer /path/to/tokenizer.json \
  --max-new 64
```

Windows Release binary:

```powershell
.\build\Release\cpi.exe C:\path\to\model.ll2c --prompt "Hello" --tokenizer C:\path\to\tokenizer.json --max-new 64
```

## Notes

- CPU-only builds skip CUDA tools such as `cuda_bandwidth_bench` and `moe_kernel_parity_test`.
- The web server can auto-detect several model settings, but `web/config.json` or env vars should still point at a real model/tokenizer pair.
- This repo is still performance-oriented and evolving; the portable first-run path is much better now, but some model-specific tuning remains manual.
