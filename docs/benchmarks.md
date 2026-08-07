# CPI Benchmark Methodology

This document describes how CPI's performance and quality are measured,
what hardware was used, and how to reproduce any reported result.

For background on the system being benchmarked, see [research.md](research.md).

---

## Quick Reproduction

```powershell
# 1. Build (CUDA or CPU-only)
cmake --preset cpu-release
cmake --build --preset cpu-release

# 2. Throughput sweep (fp16, context 128–4096)
python tools/bench_sweep.py `
    --model artifacts/mymodel.ll2c `
    --tokenizer artifacts/mymodel/hf/tokenizer.json `
    --context-lengths 128 512 2048 4096 `
    --quant-modes fp16 int8 int4

# 3. WikiText-2 perplexity
python tools/perplexity.py `
    --model-dir artifacts/mymodel/hf `
    --all-modes

# 4. Aggregate into a report
python tools/bench_report.py --patch-benchmarks
```

All outputs land in `docs/results/`. The report command updates the
*Latest Results* section at the bottom of this file.

---

## Cross-Engine Comparison Rules

Every CPI-vs-llama.cpp ratio in this repo follows these rules; a number that
doesn't is not comparable and should not be quoted.

1. **Interleave the runs.** The GPU throttles over a session (boost clocks
   drift ±6% within minutes on the reference box). Run A, B, A, B in one
   sitting -- never compare numbers taken in different sessions.
2. **Mind the timer scope.** `llama-bench` excludes sampling from its tok/s;
   CPI's `--benchmark` includes it. Reported ratios leave this in (it slightly
   flatters llama.cpp on decode); if you need it out, measure the sampler
   separately rather than adjusting by guess.
3. **Warm first, and defeat the prefix cache.** First-run prefill includes
   one-time costs (3.3x observed on cold DeepSeek). For CPI prefill timing,
   `--no-prefix-reuse` is mandatory: a repeated prompt otherwise skips prefill
   entirely, and `prefill_ms=0.00` means *not measured*, not fast.
4. **Same weights, stated build.** llama.cpp gets a GGUF converted from the
   identical checkpoint (`tools/ll2c_to_hf_safetensors.py` + llama.cpp's
   `convert_hf_to_gguf.py`), and the llama.cpp build hash is recorded next to
   the numbers.

---

## Metrics

### Throughput

| Metric | Unit | Definition |
|--------|------|------------|
| `decode_tok_per_s` | tok/s | Autoregressive decode throughput: new tokens generated per second, excluding prefill. This is the primary inference speed metric. |
| `total_tok_per_s` | tok/s | Total tokens (prompt + generated) per second, wall-clock. Useful for short-prompt scenarios. |
| `prefill_ms` | ms | Time to process the prompt tokens (forward pass over the full prompt). |
| `decode_ms` | ms | Total time for all decode steps (excludes prefill). |

Phase breakdowns (reported when `--benchmark-phases` is passed to `cpi`):

| Phase metric | Definition |
|---|---|
| `qkv_ms` | Query/Key/Value projection time |
| `attention_ms` | Attention + softmax time (CPU/CUDA fused kernel) |
| `mlp_ms` | MLP / feed-forward time (dominant phase for large models) |
| `moe_router_ms` | MoE gating network time |
| `moe_expert_ms` | MoE expert execution time |
| `lm_head_ms` | LM head projection + sampling time |

### Quality

#### Perplexity (WikiText-2)

Standard sliding-window perplexity on the WikiText-2 v1 test split
(Meister & Cotterell 2021). Lower is better.

| Metric | Definition |
|---|---|
| PPL | exp(mean cross-entropy loss). The primary quality metric for LM evaluation. |
| BPB | Bits per byte: NLL (nats) / ln(2) / 4.0. Normalises for vocabulary size; comparable across tokenizers. |
| NLL | Mean negative log-likelihood (nats) over scored tokens. |

**Sliding window parameters:**
- Stride: 512 tokens (tokens scored in each window beyond the context from the previous window)
- Max length: 1024 tokens (model context window per evaluation step)

Perplexity is measured using the HuggingFace `transformers` library against
the safetensors model directory. This makes fp16 / int8 / int4 comparison
fair: the same model weights are loaded with different `BitsAndBytesConfig`
settings.

#### Logit Parity (MoE models)

For MoE models the CI also checks top-k logit drift between fp16 and quantized
variants:

| Metric | Definition |
|---|---|
| `overlap@k` | Fraction of the fp16 top-k token IDs that appear in the quantized top-k. Higher is better. |
| `mean_abs` | Mean absolute logit difference across the top-k token IDs present in both sets. Lower is better. |
| `top1_match` | Whether the argmax token is the same between fp16 and the quantized variant. |

### Memory

| Metric | Unit | How measured |
|---|---|---|
| `peak_rss_mb` | MB | Peak RSS (Resident Set Size) of the `cpi` process tree, sampled every 150 ms by `bench_sweep.py` via `psutil`. Represents host RAM. |
| `peak_vram_mb` | MB | Peak VRAM delta above the pre-run baseline, sampled every 150 ms via `nvidia-smi --query-gpu=memory.used`. Represents GPU memory. |

---

## Benchmark Protocol

### Throughput (bench_sweep.py)

1. **Warm-up:** Each configuration runs `--benchmark-warmup 1` warm-up passes
   (not counted) before `--benchmark-reps 3` timed passes. The reported
   throughput is the average of the timed passes, as computed by `cpi`.
2. **Temperature:** 0 (greedy decoding) with `--top-k 1`. This eliminates
   sampling variance from timing measurements.
3. **Context filling:** The prompt is fixed. Context length is set via
   `--max-context`; the engine pre-allocates KV cache for this length.
4. **Isolation:** One run at a time. No concurrent processes on the GPU.

### Perplexity (perplexity.py)

1. The full WikiText-2 test set is tokenised with the model's own tokenizer.
2. A sliding window of length `max_length=1024` is advanced in steps of
   `stride=512`. Only the `stride` new tokens in each step are scored
   (cross-entropy against the ground truth); the first `max_length − stride`
   tokens are context.
3. The mean NLL is exponentiated to yield PPL.

### Logit Parity (moe_cuda_bench.py)

1. The fp16 model is run with `--inspect-next-topk 256` to collect logits for
   the first predicted token.
2. The quantized model is run with the same prompt and the same flag.
3. Top-k overlap and mean absolute difference are computed over the shared
   token IDs.

---

## Reference Hardware

Results in `docs/results/` are labelled with hardware metadata:

| Field | Description |
|---|---|
| `gpu` | nvidia-smi GPU name, e.g. `NVIDIA GeForce RTX 4090` |
| `cpu` | Processor model from `/proc/cpuinfo` or `wmic` |
| `ram_total_mb` | Total host RAM |
| `platform` | `platform.platform()` string |

Reproduce on different hardware by running the sweep and perplexity scripts;
the JSON output includes hardware metadata automatically.

---

## Adding a New Model

1. Convert to `.ll2c`:
   ```bash
   python tools/convert_hf_to_bins.py --hf-dir /path/to/hf --out-dir /path/to/bins
   python tools/pack_ll2c.py --input-dir /path/to/bins --output artifacts/mymodel.ll2c
   python tools/validate_ll2c.py artifacts/mymodel.ll2c
   ```

2. Optionally quantize:
   ```bash
   python tools/quantize_ll2c_streaming.py \
       --input artifacts/mymodel.ll2c \
       --output artifacts/mymodel-int8.ll2c --mode int8
   python tools/quantize_ll2c_streaming.py \
       --input artifacts/mymodel.ll2c \
       --output artifacts/mymodel-int4.ll2c --mode int4
   ```

3. Run the sweep:
   ```bash
   python tools/bench_sweep.py \
       --model artifacts/mymodel.ll2c \
       --tokenizer /path/to/tokenizer.json \
       --quant-modes fp16 int8 int4
   ```

4. Run perplexity (if you have the safetensors directory):
   ```bash
   python tools/perplexity.py --model-dir /path/to/hf --all-modes
   ```

5. Update this document:
   ```bash
   python tools/bench_report.py --patch-benchmarks
   ```

---

## CI Integration

The GitHub Actions workflow (`.github/workflows/cuda-moe-regression.yml`)
runs MoE parity and performance gate checks on every push. Gate thresholds
are defined in `tools/ci/moe_gate_thresholds.json`. Results are uploaded as
workflow artifacts and rendered in the job summary.

### Per-kernel perf regression gate

`tools/ci/kernel_perf_gate.py` guards the low-level CUDA kernels independently
of any model. It runs the microbenchmarks (`int4_gemv_bench`,
`attention_decode_bench`), parses achieved GB/s for every shape, and compares
against a committed baseline (`tools/ci/kernel_perf_baseline.json`).

```bash
python tools/ci/kernel_perf_gate.py --update   # (re)capture the baseline
python tools/ci/kernel_perf_gate.py            # gate; nonzero exit on regression
```

- Each shape is measured **best-of-N** (`--repeat`, default 3); the max over
  runs is the least clock/thermal/contention-perturbed estimate.
- **Run with the GPU idle** (stop the web server) -- a co-resident model adds
  contention/clock noise that inflates false positives.
- Per-bench tolerance: `int4_gemv` 12% (rock-stable across independent
  windows -- a real, tight gate, and the load-bearing decode signal),
  `attention_decode` 50% (a **coarse** gate). The attention microbench has a
  ~40-45% run-to-run *noise floor* on a consumer WDDM GPU: best-of-N absorbs
  drift within a capture window but not the GPU-temperature difference between a
  baseline and a gate run minutes later. So the attention gate catches only
  gross (2-10x) regressions -- a fallback kernel, wrong head_dim path, disabled
  coarsening; for a tight attention signal, run the microbench standalone on an
  idle, thermally-steady GPU and read %-of-roofline directly. Override the
  tolerance for all benches with `--tolerance`.

The refactors that share the sampler and decode loop across engines touch only
host-side orchestration -- the CUDA kernels and the LlamaEngine forward are
byte-identical -- so these kernel numbers are unchanged by construction; the gate
is what keeps them that way going forward.

---

## Latest Results

<!-- bench-report:start -->

# CPI Research Benchmark Report

## Perplexity (WikiText-2)

### Perplexity: Llama-3.1-8B-Instruct

Date: 2026-05-24T20:23:36.107158Z &nbsp;|&nbsp; GPU: NVIDIA GeForce RTX 5090, 32607 MiB  
Corpus: wikitext-2-v1/test &nbsp;|&nbsp; Stride: 256 &nbsp;|&nbsp; Window: 512

| Mode | PPL ↓ | BPB ↓ | NLL ↓ | Tokens | Eval time |
|------|------:|------:|------:|-------:|----------:|
| fp16 | 6.7634 | 0.6894 | 1.9115 | 512 | 43.7s |

## Throughput, Latency & Memory

### Sweep: Llama-3.1-8B-Instruct

Date: 20260524T200838Z &nbsp;|&nbsp; GPU: NVIDIA GeForce RTX 5090 &nbsp;|&nbsp; CPU: AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD

#### Throughput & Latency

| Path | Context | Quant | Decode tok/s | Prefill ms | Decode ms | Peak RAM MB | Peak VRAM MB |
|------|--------:|------:|-------------:|-----------:|----------:|------------:|-------------:|
| CUDA | 128 | fp16 | 58.27 | 36.7 | 703.7 | 15912 | 16741 |
| CUDA | 512 | fp16 | 58.05 | 37.5 | 706.3 | 15912 | 16783 |
| CUDA | 2,048 | fp16 | 58.66 | 36.6 | 698.9 | 15913 | 16985 |
| CUDA | 4,096 | fp16 | 58.02 | 37.4 | 706.6 | 15913 | 16796 |

#### Quantization Speedup (vs fp16 baseline)

| Path | Context | fp16 tok/s | int8 tok/s | int8 speedup | int4 tok/s | int4 speedup |
|------|--------:|-----------:|-----------:|-------------:|-----------:|-------------:|
| CUDA | 128 | 58.27 | -- | -- | -- | -- |
| CUDA | 512 | 58.05 | -- | -- | -- | -- |
| CUDA | 2,048 | 58.66 | -- | -- | -- | -- |
| CUDA | 4,096 | 58.02 | -- | -- | -- | -- |

### Sweep: Llama-3.1-8B-Instruct

Date: 20260524T201004Z &nbsp;|&nbsp; GPU: NVIDIA GeForce RTX 5090 &nbsp;|&nbsp; CPU: AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD

#### Throughput & Latency

| Path | Context | Quant | Decode tok/s | Prefill ms | Decode ms | Peak RAM MB | Peak VRAM MB |
|------|--------:|------:|-------------:|-----------:|----------:|------------:|-------------:|
| CUDA | 128 | int4 | 72.70 | 384.5 | 522.7 | 7589 | 7982 |
| CUDA | 512 | int4 | 9.07 | 3045.3 | 4188.5 | 7590 | 7989 |
| CUDA | 2,048 | int4 | 76.57 | 363.7 | 496.2 | 7589 | 8083 |
| CUDA | 4,096 | int4 | 77.19 | 365.4 | 492.3 | 7589 | 8271 |


<!-- bench-report:end -->
