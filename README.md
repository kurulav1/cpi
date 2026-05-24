# CPI - Cross-Platform Inference

CPI is a local LLM inference engine with a CLI, REST API, and web UI. It supports CPU inference everywhere and CUDA acceleration when a CUDA toolchain is available.

## Highlights

- CPU and CUDA inference paths
- Native `tokenizer.json` and SentencePiece tokenizer support
- React web UI plus Node API bridge in `web/`
- Model auto-discovery and per-model web defaults
- Tools for converting Hugging Face weights into `.ll2c`

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

Convert Hugging Face weights into `.ll2c`:

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

### CPT `cpt_gpt` exports

CPI can run CPT's Hugging Face-style `cpt_gpt` export folders through a small Python backend. This is intended for tiny/debug CPT models while the native `llama_infer` engine grows direct CPT architecture support.

Export from CPT:

```powershell
python ..\cpt\tools\export_cpt_to_hf.py `
  --checkpoint C:\path\to\model.ckpt `
  --out-dir .\artifacts\cpt_tiny_hf `
  --model-name cpt-tiny `
  --overwrite
```

Run it in CPI:

```powershell
$env:LLAMA_MODEL_PATH=".\artifacts\cpt_tiny_hf"
.\start_local.bat
```

The export folder must contain `config.json` with `model_type: cpt_gpt` and `model.safetensors`. Byte-level exports use CPT's `byte_tokenizer.json`; tokenizer-backed exports use `tokenizer.json` and require the Python `tokenizers` package in CPI's environment.

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
