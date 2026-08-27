# CPI Benchmark Sweep; Llama-3.1-8B-Instruct

**Date:** 20260524T201004Z  
**Host:** redacted  
**GPU:** NVIDIA GeForce RTX 5090  
**CPU:** AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD  
**Prompt:** _Explain what a neural network is in one sentence._  
**max-new:** 64  reps=3  warmup=1

## Throughput & Memory; Llama-3.1-8B-Instruct

| Path | Context | Quant | Decode tok/s | Prefill ms | Decode ms | Peak RAM MB | Peak VRAM MB |
|------|--------:|------:|-------------:|-----------:|----------:|------------:|-------------:|
| CUDA | 128 | int4 | 72.70 | 384.5 | 522.7 | 7589 | 7982 |
| CUDA | 512 | int4 | 9.07 | 3045.3 | 4188.5 | 7590 | 7989 |
| CUDA | 2,048 | int4 | 76.57 | 363.7 | 496.2 | 7589 | 8083 |
| CUDA | 4,096 | int4 | 77.19 | 365.4 | 492.3 | 7589 | 8271 |

## Quantization Speedup; Llama-3.1-8B-Instruct

Speedup = quant decode tok/s ÷ fp16 decode tok/s for same path + context.

| Path | Context | fp16 tok/s | int8 tok/s | int8 ×speedup | int4 tok/s | int4 ×speedup |
|------|--------:|-----------:|-----------:|--------------:|-----------:|--------------:|
| CUDA | 128 |; |; |; | 72.70 |; |
| CUDA | 512 |; |; |; | 9.07 |; |
| CUDA | 2,048 |; |; |; | 76.57 |; |
| CUDA | 4,096 |; |; |; | 77.19 |; |
