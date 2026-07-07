// Decode-attention bandwidth bench (harness-first: locate the roofline before
// optimizing). Decode attention has seq_q=1 attending over a long KV cache, so
// it is memory-bound reading K/V (~0.5 flop/byte) — tensor cores can't help a
// single-token query. The lever is bytes read, so the metric that matters is
// achieved KV-read bandwidth vs the ~1.79 TB/s peak, NOT FLOPs.
//
// Measures launch_attention_step_batched_paged (the serving path) across batch
// and context, reporting effective KV bandwidth = (K+V bytes that MUST be read
// once) / time. Low % of peak => headroom (poor coalescing, redundant per-head
// re-reads, or low occupancy); near peak => the kernel is at its floor and a
// rewrite won't help (the same conclusion int4_gemv_bench reached for GEMV).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "runtime/kernels.cuh"

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t _e = (x);                                                             \
    if (_e != cudaSuccess) {                                                          \
      std::printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(_e)); \
      return 2;                                                                       \
    }                                                                                 \
  } while (0)

namespace {
constexpr double kPeakGBs = 1792.0;  // RTX 5090 GDDR7 ~1.79 TB/s

struct Shape {
  int num_heads;
  int num_kv_heads;
  int head_dim;
  const char* name;
};
}  // namespace

int main() {
  const Shape shapes[] = {
      {32, 8, 128, "Llama3.1-8B"},
      {16, 4, 256, "Qwen3.5-9B"},
  };
  const int batches[] = {1, 8, 32};
  const int contexts[] = {512, 2048, 4096, 8192, 16384, 32768};
  const int block_size = 32;

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

  std::printf("decode-attention bench (peak %.0f GB/s) — effective KV-read bandwidth\n", kPeakGBs);
  std::printf("%-12s %-5s %-6s %-9s  %-10s  %s\n", "model", "B", "ctx", "ms/step", "GB/s", "%peak");
  std::printf("--------------------------------------------------------------------\n");

  for (const Shape& s : shapes) {
    const int kv_hidden = s.num_kv_heads * s.head_dim;
    for (int batch : batches) {
      for (int L : contexts) {
        const int blocks_per_seq = (L + block_size - 1) / block_size;
        const int max_blocks = blocks_per_seq;
        const int scratch_chunks = blocks_per_seq;
        const int pool_blocks = batch * blocks_per_seq;

        // One decode step: query is a single token per (seq, head); K/V live in a
        // shared block pool addressed per-sequence by a contiguous block table.
        std::vector<half> q(static_cast<std::size_t>(batch) * s.num_heads * s.head_dim);
        for (auto& x : q) x = __float2half(uni(rng));
        std::vector<half> kv_pool(static_cast<std::size_t>(pool_blocks) * block_size * kv_hidden);
        for (auto& x : kv_pool) x = __float2half(uni(rng) * 0.1f);
        std::vector<int> block_tables(static_cast<std::size_t>(batch) * max_blocks);
        std::vector<int> seq_lens(batch, L);
        for (int b = 0; b < batch; ++b)
          for (int c = 0; c < blocks_per_seq; ++c)
            block_tables[static_cast<std::size_t>(b) * max_blocks + c] = b * blocks_per_seq + c;

        half *dq, *dk, *dv, *dout;
        int *dbt, *dsl;
        float *sm, *sl, *so;
        const std::size_t out_elems = static_cast<std::size_t>(batch) * s.num_heads * s.head_dim;
        CK(cudaMalloc(&dq, q.size() * sizeof(half)));
        CK(cudaMalloc(&dk, kv_pool.size() * sizeof(half)));
        CK(cudaMalloc(&dv, kv_pool.size() * sizeof(half)));
        CK(cudaMalloc(&dout, out_elems * sizeof(half)));
        CK(cudaMalloc(&dbt, block_tables.size() * sizeof(int)));
        CK(cudaMalloc(&dsl, seq_lens.size() * sizeof(int)));
        CK(cudaMalloc(
            &sm, static_cast<std::size_t>(batch) * s.num_heads * scratch_chunks * sizeof(float)));
        CK(cudaMalloc(
            &sl, static_cast<std::size_t>(batch) * s.num_heads * scratch_chunks * sizeof(float)));
        CK(cudaMalloc(&so, static_cast<std::size_t>(batch) * s.num_heads * scratch_chunks *
                               s.head_dim * sizeof(float)));
        CK(cudaMemcpy(dq, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, kv_pool.data(), kv_pool.size() * sizeof(half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, kv_pool.data(), kv_pool.size() * sizeof(half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbt, block_tables.data(), block_tables.size() * sizeof(int),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsl, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice));

        auto launch = [&] {
          kernels::launch_attention_step_batched_paged(
              dq, dk, dv, dbt, dsl, max_blocks, L, dout, batch, s.num_heads, s.num_kv_heads,
              s.head_dim, block_size, 0, sm, sl, so, scratch_chunks);
        };

        const int warmup = 10, iters = 50;
        for (int i = 0; i < warmup; ++i) launch();
        CK(cudaDeviceSynchronize());
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; ++i) launch();
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        const double per_call_ms = ms / iters;

        // Minimum KV bytes that MUST be read: each cached K and V element once.
        const double kv_bytes = static_cast<double>(batch) * L * kv_hidden * 2.0 * sizeof(half);
        const double gbs = kv_bytes / (per_call_ms * 1e-3) / 1e9;

        std::printf("%-12s %-5d %-6d %-9.4f  %-10.1f  %.0f%%\n", s.name, batch, L, per_call_ms, gbs,
                    100.0 * gbs / kPeakGBs);

        cudaFree(dq);
        cudaFree(dk);
        cudaFree(dv);
        cudaFree(dout);
        cudaFree(dbt);
        cudaFree(dsl);
        cudaFree(sm);
        cudaFree(sl);
        cudaFree(so);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
      }
    }
  }
  return 0;
}
