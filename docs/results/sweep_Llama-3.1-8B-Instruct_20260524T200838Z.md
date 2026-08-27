# CPI Benchmark Sweep; Llama-3.1-8B-Instruct

**Date:** 20260524T200838Z  
**Host:** redacted  
**GPU:** NVIDIA GeForce RTX 5090  
**CPU:** AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD  
**Prompt:** _Explain what a neural network is in one sentence._  
**max-new:** 64  reps=3  warmup=1

## Throughput & Memory; Llama-3.1-8B-Instruct

| Path | Context | Quant | Decode tok/s | Prefill ms | Decode ms | Peak RAM MB | Peak VRAM MB |
|------|--------:|------:|-------------:|-----------:|----------:|------------:|-------------:|
| CUDA | 128 | fp16 | 58.27 | 36.7 | 703.7 | 15912 | 16741 |
| CUDA | 512 | fp16 | 58.05 | 37.5 | 706.3 | 15912 | 16783 |
| CUDA | 2,048 | fp16 | 58.66 | 36.6 | 698.9 | 15913 | 16985 |
| CUDA | 4,096 | fp16 | 58.02 | 37.4 | 706.6 | 15913 | 16796 |

## Quantization Speedup; Llama-3.1-8B-Instruct

Speedup = quant decode tok/s ÷ fp16 decode tok/s for same path + context.

| Path | Context | fp16 tok/s | int8 tok/s | int8 ×speedup | int4 tok/s | int4 ×speedup |
|------|--------:|-----------:|-----------:|--------------:|-----------:|--------------:|
| CUDA | 128 | 58.27 |; |; |; |; |
| CUDA | 512 | 58.05 |; |; |; |; |
| CUDA | 2,048 | 58.66 |; |; |; |; |
| CUDA | 4,096 | 58.02 |; |; |; |; |
