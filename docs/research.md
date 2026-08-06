# CPI: Cross-Platform Inference

## Abstract

CPI is a local LLM inference engine designed around two goals: portability and
performance. It provides a single binary (`cpi`) that runs on any x86
machine — falling back to a multithreaded CPU path when no CUDA device is
present — and a CUDA path that leverages custom kernels and post-training
quantization to push decode throughput on consumer and datacenter GPUs. A
Node.js API layer exposes an OpenAI-compatible REST interface, enabling
drop-in integration with standard tooling. A React web UI and Python-based
conversion/evaluation tools complete the pipeline.

---

## Motivation

Most open-source LLM inference stacks require either a GPU (vLLM, TensorRT-LLM)
or accept substantial CPU performance penalties as a side-effect of portability
(llama.cpp). CPI takes a different approach: it maintains **two separate,
optimized code paths** — one targeting AVX2/AVX-512 CPU execution and one
targeting NVIDIA CUDA — selected automatically at runtime based on available
hardware. This means the same model artifact and binary can be deployed on a
laptop or a datacenter node without reconfiguration.

A secondary goal is **production-grade serving semantics** at local scale: a
single-GPU machine should be able to host a chat API with warm-worker latency
(no cold-start per request), streaming token output, resource-limit safety
valves, and an OpenAI-compatible endpoint that existing clients can reach
without modification.

---

## System Architecture

### Inference Engine (C++)

The core engine is a C++ library compiled with CMake. The build system detects
CUDA availability and compiles either a CUDA-enabled binary or a CPU-only
binary; both expose the same interface to the application layer.

**Engine inventory and roles.** There are two executor tiers plus a set of
deliberately independent reference engines; each class earns its existence in a
specific way, and the asymmetries between them are stated here rather than
discovered by reading 10k lines.

| Engine class | Role | Architecture | Compute | Weight format |
|---|---|---|---|---|
| `PlanCudaEngine` | op-plan executor (the extensibility core) | Gemma / Gemma 4 (+MoE), Qwen3.5 (dense, MoE, vision), DeepSeek-V2 (MLA + MoE) | CUDA | `.ll2c`, safetensors |
| `PlanMetalEngine` | op-plan executor, same plans as CUDA | same families, gated token-identical vs CUDA | Metal | `.ll2c`, safetensors |
| `LlamaEngine` | uniform-geometry serving fast path | Llama 2/3/3.1, Mistral, Qwen2/2.5/3-dense, Phi-3 MoE | CUDA | `.ll2c` |
| `Llama4CudaEngine` | expert-streaming for larger-than-VRAM MoE | Llama 4 Scout/Maverick | CUDA | safetensors |
| `CpuLlamaEngine` | fp32 reference oracle | Llama-family | CPU | `.ll2c` |
| `Llama4CpuEngine` / `Qwen35CpuEngine` | per-family reference oracles | Llama 4 / Qwen 3.5 | CPU | safetensors |

Why the split, honestly:

- **The op-plan IR is where new model families land.** Gemma 4, Qwen3.5 and
  DeepSeek-V2 were added as capability flags and plan stages, not forks; both
  GPU backends execute the same plans and are gated token-identical.
- **`LlamaEngine` is a deliberate second core, and that is real debt.** It
  predates the IR and holds every serving-critical capability: CUDA graphs,
  continuous batching with the paged KV pool, KV-cache quantization,
  speculative decoding, EAGLE. Models that run on the plan executor get none
  of those today. Porting the serving stack onto the plan executor is the
  known large item; until then CPI has one fast serving core and one general
  executor, and pretending otherwise would be spin.
- **The CPU engines are oracles, not products.** They are simple, independently
  derived implementations used to gate the GPU paths. Folding them into the IR
  would make them share the executor's bugs, which defeats their purpose — a
  parity oracle that inherits the engine's mistake once made a real GPU fix
  look like a regression here.
- **`Llama4CudaEngine` predates op-plan MoE** and exists for per-layer expert
  streaming of models larger than VRAM. Now that the plan executor runs MoE
  (DeepSeek, Gemma 4), it is a port-and-retire candidate, not a design
  statement.
- Known type-level asymmetry: `PlanCudaEngine` implements
  `runtime::SequenceModel` (the shared decode driver); `PlanMetalEngine` does
  not yet, so the backends are interchangeable at the gate level but not at
  the interface level.

The main binary auto-detects the model format (`.ll2c` vs. a safetensors
directory) and the model family (via `config.json` or header inspection), then
instantiates the appropriate engine class.

### Weight Format: `.ll2c`

`.ll2c` is a compact binary container for model tensors. Unlike GGUF it carries
all quantization variants in a single file: an fp16 base plus optional appended
int8 and int4 weight tables. The header (V1–V5) encodes architecture
hyperparameters (hidden size, number of layers, RoPE theta, MoE topology) so
the runtime does not need a separate config file.

**Conversion pipeline:**

```
HuggingFace safetensors → tools/convert_hf_to_bins.py → per-tensor .bin files
                       → tools/pack_ll2c.py             → model.ll2c (fp16)
                       → tools/quantize_ll2c_streaming.py → model.ll2c (+ int8/int4)
```

### Quantization: TurboQuant (int8/int4)

CPI implements a weight-only quantization scheme ("TurboQuant") applied to MLP
projection matrices. Activations remain in fp16; only the stored weights are
quantized. At decode time, weight rows are dequantized on-the-fly before the
GEMM, which keeps the arithmetic in fp16 while halving (int8) or quartering
(int4) the weight transfer bandwidth. This is the same approach used by
[GPTQ](https://arxiv.org/abs/2210.17323) and
[AWQ](https://arxiv.org/abs/2306.00978), though CPI's implementation is
hand-rolled against its own kernel infrastructure.

**Custom CUDA kernels** (`src/kernels/`):

- `kernels_weight_only_matvec.cu` — weight-dequant + fp16 GEMM for MLP layers
- `kernels_attention_decode.cu` — fused multi-head attention for decode (cached KV)
- `kernels_attention_decode_int4.cu` — int4 KV-cache variant
- `kernels_turboquant.cu` — packing/unpacking helpers
- `kernels_ops_matvec.cu` — general-purpose fp16 matmul

### KV Cache

A block-allocated KV cache (`src/runtime/kv_cache.cpp`) stores past key/value
tensors for efficient autoregressive decode. The CUDA engine optionally
compresses the cache to int4 to allow longer contexts within the same VRAM
budget.

### Tensor Parallelism

Multi-GPU tensor parallelism (`src/engine/tensor_parallel.cpp`) distributes
MLP weight shards across available devices and reduces results with
cuBLAS + NCCL-free all-reduce. This is primarily used for large MoE models
(Phi-MoE, Mixtral) that exceed single-GPU VRAM.

### Serving Layer (Node.js)

The API server (`web/server/index.mjs`) maintains a **warm interactive worker**
— a persistent `cpi` subprocess that receives inference requests over
stdin as NDJSON lines and streams token deltas back on stdout. This eliminates
the model-load latency that would otherwise occur on every request. The server:

- Enforces single-generation serialization (one GPU at a time)
- Monitors CPU and memory usage and throttles generation if limits are exceeded
- Applies per-model-family prompt budgets and template normalization
- Exposes `/v1/chat/completions` (OpenAI-compatible), `/api/chat/stream`
  (NDJSON streaming), and `/api/generate` (blocking)

### CPT Backend

A thin Python shim (`tools/cpt_gpt_worker.py`,  `tools/hf_chat_worker.py`)
allows the server to route requests to HuggingFace-format model directories
(e.g. CPT's `cpt_gpt` architecture) without compiling a custom C++ engine.
This is the recommended path for architecture experimentation.

---

## Key Technical Contributions

1. **Unified CPU/CUDA binary** — single build artifact, hardware-adaptive at
   runtime, no separate CPU and GPU distributions.

2. **`.ll2c` multi-quantization container** — fp16 + int8 + int4 in one file
   with a versioned header; eliminates the need to manage separate model files
   per quantization level.

3. **Warm interactive worker protocol** — NDJSON over stdin/stdout turns a
   CLI inference binary into a stateful streaming server without any IPC
   framework dependency.

4. **Resource-limit safety valves** — the server monitors host CPU and RAM
   during generation and aborts runaway runs, making it safe to deploy on
   shared hardware.

5. **CPT architecture support** — a Python-level plugin path lets new
   architectures be served immediately through the REST API before a native
   C++ engine is written.

---

## Supported Models

| Family | Architecture | Engine | Status |
|--------|-------------|--------|--------|
| Llama 2 7B/13B/70B | Dense transformer | CUDA / CPU | Production |
| Llama 3 8B/70B | Dense transformer | CUDA / CPU | Production |
| Llama 4 Scout / Maverick | MoE transformer | CUDA / CPU | Production |
| TinyLlama 1.1B | Dense transformer | CUDA / CPU | Production |
| Mistral 7B | Dense transformer (sliding window) | CUDA / CPU | Production |
| Phi-3 Mini | Dense transformer | CUDA / CPU | Production |
| Phi-tiny/mini MoE | MoE transformer | CUDA | Beta |
| Mixtral 8×7B | MoE transformer | CUDA | Production |
| Qwen 3.5 | Dense + MoE | CUDA / CPU | Production |
| CPT GPT | Custom (via Python backend) | CPU | Experimental |

---

## Related Work

- **llama.cpp** ([Gerganov 2023](https://github.com/ggerganov/llama.cpp)) —
  the dominant CPU-focused inference stack. CPI targets a different tradeoff:
  native CUDA kernels over cross-platform GGML, and a first-class HTTP server
  over a CLI-centric workflow.

- **vLLM** ([Kwon et al. 2023](https://arxiv.org/abs/2309.06180)) — paged KV
  cache and continuous batching for high-throughput multi-user serving. CPI
  targets single-user local deployment rather than multi-tenant throughput.

- **TensorRT-LLM** (NVIDIA 2023) — production GPU inference with extensive
  quantization support. Requires NVIDIA hardware and a complex build environment;
  CPI prioritizes portability and simplicity.

- **GPTQ** ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323)) — the
  weight quantization algorithm CPI's int4 path is conceptually related to.
  CPI uses a simpler MSE-based calibration without the full GPTQ Hessian
  computation.

- **AWQ** ([Lin et al. 2023](https://arxiv.org/abs/2306.00978)) — activation-
  aware weight quantization. CPI's quantization does not yet incorporate
  activation-aware scaling.

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [README.md](../README.md) | Quick start, build instructions, API reference |
| [docs/benchmarks.md](benchmarks.md) | Benchmark methodology and latest results |
| [docs/results/](results/) | Machine-generated JSON and Markdown result files |
| [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md) | Release process |

---

## Citing This Work

If you use CPI in academic work, please cite the repository:

```
@software{cpi2025,
  title  = {{CPI}: Cross-Platform Inference},
  author = {Kurula, Väinö},
  year   = {2025},
  url    = {https://github.com/your-org/cpi}
}
```
