# CPI — Cross-Platform Inference

Memory-aware, cross-platform CUDA inference engine for LLaMA-family decoder models.

## Highlights

- C++20 + CUDA + cuBLAS.
- Cross-platform weight memory mapping (`mmap`/`MapViewOfFile`).
- Full decoder block execution per layer:
  - RMSNorm
  - Q/K/V projections
  - RoPE (configurable base frequency via `--rope-theta`)
  - causal attention with KV cache
  - MLP (SwiGLU)
  - residual connections
- Runtime-safe design for 12 GB GPUs using streamed per-layer weights.
- Default fp16 decode path uses fused device-side `QKV` / `w1+w3` projections,
  cached cuBLASLt matmul plans, a warp-tiled exact online-softmax decode attention kernel, and a true chunked prompt-prefill path to reduce prompt overhead
  without changing model outputs.
- When all transformer layers fit on GPU, the runtime switches to a dedicated fully resident fast path that skips
  weight-streaming barriers and transfer bookkeeping.
- For greedy fully resident decode, the runtime can replay the steady-state token loop through a CUDA Graph to cut
  launch overhead without changing outputs.
- Native tokenizer backends for SentencePiece `.model` and HF `tokenizer.json` BPE files.
- Supports both standard MHA (`num_kv_heads == num_heads`) and GQA models
  (`num_kv_heads < num_heads`, e.g. TinyLlama) through unified runtime paths.

## Layout

- `include/platform/*`, `src/platform/*`: file mapping.
- `include/runtime/*`, `src/runtime/*`, `src/kernels/*`: CUDA helpers and kernels.
- `include/model/*`, `src/model/*`: model loading and tokenizer integration.
- `include/engine/*`, `src/engine/*`: inference orchestration.
- `tools/convert_hf_to_bins.py`: extract all needed tensors from HF safetensors.
- `tools/pack_ll2c.py`: pack tensors into `.ll2c` format.

## Release Hygiene

- Keep secrets in local `.env` only (ignored by git).
- Use `.env.example` as the committed template.
- Run the pre-release checks in `RELEASE_CHECKLIST.md` before tagging.

## Build

### Windows (MSVC + CUDA)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

To target a specific GPU architecture, pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>`.
Common values: `75` (Turing/RTX 20xx), `80` (Ampere/A100), `86` (RTX 30xx), `89` (RTX 40xx).

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Web Chat UI

The repo includes a React + Tailwind chat client, a REST API server, and a small Node bridge in `web/`.
It formats multi-turn chat prompts, streams `llama_infer` output to browsers or API callers, and keeps the native runtime as the source of truth.

### Configuration

The server reads config from `web/config.json` (user-edited) with environment variables as overrides.

```bash
cd web
cp config.example.json config.json   # Linux/macOS
# or: copy config.example.json config.json  (Windows)
```

Edit `web/config.json` and set at minimum:

```json
{
  "modelPath": "/path/to/your/model.ll2c",
  "tokenizerPath": "/path/to/your/tokenizer.json"
}
```

All other fields have sensible defaults. See `web/config.example.json` for the full list.

Alternatively, set environment variables (useful for Docker/CI). Env vars override `config.json`:
`LLAMA_MODEL_PATH`, `LLAMA_TOKENIZER_PATH`, `LLAMA_MODEL_DIRS`, `LLAMA_INFER_BIN`, etc.

### Auto-configuration

The server auto-detects per-model settings at startup:

- **RoPE theta** — reads `rope_theta` from the model's `config.json` (HuggingFace format) if present, otherwise infers from filename (`llama3`/`llama4` → 500000, others → 10000). The correct `--rope-theta` flag is injected automatically.
- **Chat template** — inferred from model family name and tokenizer type (`.model` vs `.json`).
- **Tokenizer pairing** — ranked by proximity to the model file and family name matching.
- **streaming quant** — for `*streaming*` / `*packed*` models, `--int8-streaming` is auto-added by default (you can override with `--weight-quant int4` in extra args).

### Local Dev

1. Build `llama_infer` first.
2. Copy and edit `web/config.json` as above.
3. Start the server:

```bash
cd web
npm install
npm run dev
```

This starts:

- the API server on `http://localhost:3001`
- the Vite UI on `http://localhost:5173`

Quick start scripts (handle build + serve in one step):

- Windows: `start_local.bat`
- Linux/macOS: `./start_local.sh`
- Windows web dev only: `start_web.bat`
- Linux/macOS web dev only: `./start_web.sh`
- Windows Docker: `start_docker.bat C:\path\to\models`
- Linux/macOS Docker: `./start_docker.sh /path/to/models`

### API

All endpoints accept and return JSON. The server runs on port 3001 by default.

#### `GET /api/health`
Returns server status, busy flag, and the active runtime configuration.

#### `GET /api/models`
Returns all discovered model profiles.

```bash
curl http://localhost:3001/api/models
```

#### `POST /api/generate`
Blocking inference — waits for the full response and returns it as JSON.

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is CUDA?"}]}'
```

Response: `{ text, elapsedMs, profileId, modelLabel, template, maxNewTokens, temperature }`

#### `POST /api/chat/stream`
Streaming inference — returns newline-delimited JSON events.

```bash
curl -X POST http://localhost:3001/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain RoPE embeddings."}]}'
```

Events: `{ type: "start" }` → `{ type: "delta", delta: "..." }` × N → `{ type: "done", message, elapsedMs }`

#### `POST /v1/chat/completions` (OpenAI-compatible)
Drop-in compatible with the OpenAI chat completions API. Works with any OpenAI client library.

Non-streaming:
```bash
curl -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

Streaming (SSE):
```bash
curl -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

Python example using the `openai` SDK:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3001/v1", api_key="unused")

response = client.chat.completions.create(
    model="llama2",
    messages=[{"role": "user", "content": "Explain attention in one paragraph."}]
)
print(response.choices[0].message.content)
```

**Common request fields** (all endpoints):
| Field | Description |
|---|---|
| `messages` | Array of `{ role, content }` — `user`, `assistant`, `system` |
| `profileId` / `model` | Model profile id or label (defaults to server's selected model) |
| `maxNewTokens` / `max_tokens` | Max tokens to generate (32–4096) |
| `temperature` | Sampling temperature (0–2) |
| `systemPrompt` | Override the system prompt for this request |
| `template` | Override the chat template (`llama2`, `tinyllama`, `llama4`, etc.) |

### Docker

The root `Dockerfile` builds:

- the React frontend
- a Linux `llama_infer` binary
- a runtime image that serves the web UI and API bridge

```bash
docker build -t cpi-chat-ui .
docker run --rm -p 3001:3001 --gpus all \
  -e LLAMA_MODEL_DIRS=/models \
  -e LLAMA_MODEL_PATH=/models/your-model.ll2c \
  -e LLAMA_TOKENIZER_PATH=/models/tokenizer.json \
  -e LLAMA_CHAT_TEMPLATE=tinyllama \
  -v /path/to/models:/models:ro \
  cpi-chat-ui
```

Open `http://localhost:3001` after the container starts.

## Style / Lint

Config files included:

- `.editorconfig`
- `.clang-format`
- `.clang-tidy`

CMake targets (if tools are installed):

```bash
cmake --build build --target format
cmake --build build --target format-check
cmake --build build --target tidy
```

Enable clang-tidy during normal builds:

```bash
cmake -S . -B build -DLLAMA_ENGINE_ENABLE_CLANG_TIDY=ON
```

## Quick System Check

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_stats.ps1
```

## GPU Cache Controls

By default the engine decides how many transformer layers to keep on GPU at startup based on free VRAM and
`--vram-safety-margin-mb`.

Auto mode is quality-first and speed-first:

- If the full transformer can fit on GPU in fp16 at runtime, it prefers that path and avoids packed/paged fallbacks.
- If full fp16 residency does not fit, it falls back to packed MLP caching and finally to streamed layers only when
  those memory-saving techniques are actually needed.

Startup prints the selected cache policy, estimated fp16 vs packed cache footprints, and the effective VRAM budget
used for the decision.

Available overrides:

- `--gpu-cache-all`: try to cache every layer on GPU; if not all layers fit, the engine falls back to the maximum that does.
- `--gpu-cache-layers N`: request a fixed number of layers to cache on GPU.
- `--gpu-cache-limit-mb N`: cap the layer-cache budget in MB.
- `--vram-safety-margin-mb N`: reserve this much VRAM before deciding how many layers to cache.

Examples:

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 128 --int8-streaming

./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 128 --int8-streaming --gpu-cache-all

./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 128 --int8-streaming --gpu-cache-layers 16
```

## Prepare Weights

1. Extract full model tensors from HF safetensors:

```bash
py tools/convert_hf_to_bins.py \
  --hf-dir /path/to/Llama-2-7b-hf \
  --out-dir /path/to/llama2_bins
```

2. Pack into memory-mappable model file:

```bash
py tools/pack_ll2c.py \
  --input-dir /path/to/llama2_bins \
  --output /path/to/llama2.ll2c
```

Optional: emit packed per-layer int8 tensors for faster/lower-RAM streamed decode:

```bash
py tools/pack_ll2c.py \
  --input-dir /path/to/llama2_bins \
  --output /path/to/llama2-streaming.ll2c \
  --emit-streaming-int8
```

To preserve response quality, the packer emits row-wise calibrated packed int8 copies
for the largest MLP matrices (`w1`/`w2`/`w3`) and leaves attention weights in fp16.

Optional: emit packed per-layer int4 tensors (smaller model artifact footprint):

```bash
py tools/pack_ll2c.py \
  --input-dir /path/to/llama2_bins \
  --output /path/to/llama2-streaming-int4.ll2c \
  --emit-streaming-int4
```

Optional: streaming-only model (omits fp16 layer tensors, smallest footprint, requires streaming quant enabled):

```bash
py tools/pack_ll2c.py \
  --input-dir /path/to/llama2_bins \
  --output /path/to/llama2-streaming-only.ll2c \
  --emit-streaming-int8 \
  --omit-fp16-layer-tensors
```

3. Validate packed model integrity (recommended):

```bash
py tools/validate_ll2c.py /path/to/llama2.ll2c
```

TinyLlama / GQA models are supported with the same flow:

```bash
py tools/convert_hf_to_bins.py --hf-dir /path/to/TinyLlama-1.1B-Chat-v1.0 --out-dir /path/to/tinyllama_bins
py tools/pack_ll2c.py --input-dir /path/to/tinyllama_bins --output /path/to/tinyllama-1.1b-chat.ll2c
```

## Run

Token-id mode:

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --tokens "1,29871,13" --max-new 32 --temp 0.8 --max-context 2048
```

Greedy fast path (best raw speed, no full logits copy each step):

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Hello from CUDA" \
  --tokenizer /path/to/tokenizer.model \
  --max-new 64 --temp 0.0 --repeat-penalty 1.0
```

Anti-loop and stop control:

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Write exactly five numbered CUDA crash debugging steps." \
  --tokenizer /path/to/tokenizer.model \
  --max-new 80 --temp 0.7 --top-k 30 --top-p 0.85 \
  --repeat-penalty 1.15 --no-repeat-ngram 4 --stop-text "6."
```

Decode loop guard is enabled by default and stops obvious repetitive tails early.
Disable it with `--no-loop-guard` if you want raw unbounded generation.
For small models that produce one decent sentence and then drift, add `--sentence-stop`.

Streamed low-bit decode (optional, coherence-preserving and faster on partially uncached 7B runs):

```bash
./build/Release/llama_infer /path/to/model-streaming-rowwise.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 128 --int8-streaming
```

Quality note:

- The default fp16 decode path is the coherence-safe path and is the one used for
  the newest fusion / cuBLASLt / exact attention / prompt-prefill optimizations.
- The row-wise calibrated streaming quant path is intended for streamed MLP weights.
  Use `--weight-quant int8` (or legacy `--int8-streaming`) for highest regular-quant quality.
  Use `--weight-quant int4` (or `--int4-streaming`) for smaller low-bit artifacts.
  Chunked prompt-prefill now supports that path too when KV stays device-resident.

Packed streaming-only model decode:

```bash
./build/Release/llama_infer /path/to/model-streaming-only.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 128 --weight-quant int8
```

Paged KV cache mode (reduces persistent KV VRAM use, slower but quality-preserving):

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 512 --paged-kv-cache
```

Tokenizer JSON mode (native HF BPE path, no Python required):

```bash
./build/Release/llama_infer /path/to/tinyllama.ll2c \
  --prompt "Who are you" \
  --chat-template tinyllama \
  --tokenizer /path/to/tokenizer.json \
  --max-new 64
```

Llama 3 (rope-theta override):

```bash
./build/Release/llama_infer /path/to/llama3-model.ll2c \
  --prompt "Hello" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.json \
  --rope-theta 500000 \
  --max-new 64
```

TinyLlama chat reliability note:

- Prefer `tokenizer.json` (not `tokenizer.model`) for TinyLlama checkpoints.
- `--chat-template tinyllama` uses a simpler instruction-style prompt that is more stable.
- `--chat-template tinyllama-chatml` keeps the role-marker format for experimentation.
- Use `--dump-tokenizer-meta --dump-prompt-tokens` to verify prompt tokenization when output looks malformed.
- Use `--inspect-next-topk 8 --trace-steps 8` to see whether drift starts at the first token or later in the decode.
- The CLI prints `[perf] generated_tokens=... elapsed_ms=... tok_per_s=...` after generation.
- Add `--benchmark` to split prompt-prefill and decode timing:

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 128 --benchmark
```

- Add `--benchmark-reps N --benchmark-warmup M` to run a quiet benchmark sweep and print averages:

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Answer briefly: what is CUDA?" \
  --chat-template llama2 \
  --tokenizer /path/to/tokenizer.model \
  --max-new 32 --temp 0.0 --max-context 128 \
  --weight-quant int4 --benchmark --benchmark-warmup 1 --benchmark-reps 3
```

- Add `--benchmark-phases` to print a resident decode phase breakdown for `RMSNorm`, `QKV`, KV store, attention, `WO`, MLP, and `LM head`.
- Benchmark output now includes `tq3_cached_active=0|1` so TurboQuant runs can verify that the true cached path is active.
- Add `--no-split-attention` to force the single-CTA attention path when comparing the split-attention backend at longer contexts.
- Fully resident decode autotunes multiple custom launch shapes for `QKV`, `WO`, `LM head`, and row-wise int8 `MLP` kernels at startup.
  This mode disables the greedy CUDA graph so the per-phase timings stay meaningful.

- Llama2-7B TurboQuant preset:

```bash
python tools/tq3_llama2_benchmark.py
```

This writes:

- `artifacts/tq3_llama2_benchmark_latest.json`
- `artifacts/tq3_llama2_benchmark_latest.csv`

with `decode_tok_per_s`, decode phase timings, fallback reason, and `tq3_cached_active`.

- Use `cuda_bandwidth_bench` to measure device-memory bandwidth and estimate a practical tokens/s roofline:

```bash
./build/Release/cuda_bandwidth_bench --mb 1024 --warmup 3 --iters 100 --repeats 5 --bytes-per-token-gb 8.03
```

This prints:

- the GPU's theoretical memory bandwidth from CUDA device attributes
- sustained read, write, copy-kernel, and `cudaMemcpy` D2D bandwidth
- an optional `read_best_tok_per_s` roofline estimate when `--bytes-per-token-gb` is provided

Optional parity check (CPU reference vs GPU logits):

```bash
./build/Release/llama_infer /path/to/model.ll2c \
  --prompt "Hello" \
  --tokenizer /path/to/tokenizer.model \
  --parity-check --max-new 1 --temp 0.0
```

Tokenizer fallback is also supported via `spm_encode/spm_decode` executables
(used automatically when C++ linking is unavailable). After `vcpkg install` they are expected at:

`third_party/vcpkg/installed/x64-windows/tools/sentencepiece/`

## OOM Guardrails

- Startup prints per-device free/total VRAM.
- Decoder uses streamed layer weights to avoid loading all FP16 weights into VRAM at once.
- KV cache is allocated on device and bounded by `--max-context`.
- `--paged-kv-cache` moves the full KV backing store to pinned host memory and uses a device staging buffer, which is slower but can avoid decode-time VRAM exhaustion at longer contexts.
- Packed `.ll2c` files generated with `--emit-streaming-int8` or `--emit-streaming-int4` store optional row-wise calibrated low-bit copies of the large MLP layer tensors plus per-row scales; these are used only when streaming quant is enabled (`--weight-quant int8|int4`, `--int8-streaming`, or `--int4-streaming`), so the default fp16 decode path remains unchanged.
- Host guardrails are enabled by default for inference:
  - `--max-cpu-percent` (default `85`)
  - `--max-memory-percent` (default `85`)
  - `--resource-sample-ms` (default `250`)
  - `--resource-sustain-ms` (default `5000`)
  - `--resource-throttle-ms` (default `50`)
- Behavior is throttle-first, then abort when over-limit pressure is sustained.
- TurboQuant cached mode (`--enable-tq-cached`) now uses strict preflight and fails fast if required TQ tensors are missing instead of silently falling back.

## Quality Eval Harness

Run a fixed prompt suite and write a JSON report with per-prompt scores:

```bash
python tools/eval_prompts.py \
  --model /path/to/model.ll2c \
  --tokenizer /path/to/tokenizer.model \
  --out eval_report.json
```

For TinyLlama `tokenizer.json` mode:

```bash
python tools/eval_prompts.py \
  --model /path/to/tinyllama.ll2c \
  --tokenizer /path/to/tokenizer.json \
  --chat-template tinyllama \
  --out eval_report_tinyllama.json
```

## Notes

- This implementation is functionally complete for decoder execution, but the current attention kernel is correctness-first and not yet FlashAttention-level optimized.
- Main optimization next steps: tiled warp-specialized decode attention, true batched prompt prefill, better layer prefetch, async overlap of H2D copies with compute, and multi-GPU tensor parallel decode.
- Base models (e.g. `Llama-2-7b-hf`) are not instruction tuned, so assistant-style prompts may look less polished than chat-tuned checkpoints.
- Decode defaults use `repetition_penalty=1.0` (faster/cleaner baseline). Increase with `--repeat-penalty` only when needed for repetition control.
