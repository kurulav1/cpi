// Parity test for batched paged decode attention (P2 primitive).
//
// Runs one decode step for a BATCH of sequences (varied lengths, per-sequence
// NON-CONTIGUOUS block tables, shared KV pool) via
// launch_attention_step_batched_paged, and checks each sequence's output matches
// an independent single-sequence launch_attention_step_paged call (itself parity-
// verified against the contiguous kernel). Proves batching is a pure throughput
// transform: no cross-sequence interference, correct per-seq lengths/tables.
#include "runtime/kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t _e = (x);                                                        \
    if (_e != cudaSuccess) {                                                     \
      std::printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(_e)); \
      return 2;                                                                  \
    }                                                                            \
  } while (0)

int main() {
  const int batch = 3, num_heads = 28, num_kv_heads = 4, head_dim = 128, block_size = 32;
  const int kv_hidden = num_kv_heads * head_dim;
  const std::vector<int> seq_lens = {40, 100, 200};
  const int max_seq = *std::max_element(seq_lens.begin(), seq_lens.end());
  const int max_blocks = (max_seq + block_size - 1) / block_size;   // 7
  const int scratch_chunks = max_blocks;
  const int pool_blocks = 64;

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  auto rnd = [&](std::size_t n) { std::vector<half> h(n); for (auto& x : h) x = __float2half(uni(rng)); return h; };

  // Shared KV pool (flat cache of pool_blocks*block_size tokens).
  std::vector<half> kp(static_cast<std::size_t>(pool_blocks) * block_size * kv_hidden, __float2half(0.0f));
  std::vector<half> vp = kp;
  // Per-sequence q + a NON-CONTIGUOUS block table (distinct physical blocks per seq).
  std::vector<half> q(static_cast<std::size_t>(batch) * num_heads * head_dim);
  { auto qq = rnd(q.size()); q = qq; }
  std::vector<int> block_tables(static_cast<std::size_t>(batch) * max_blocks, 0);
  int next_phys = pool_blocks - 1;  // hand out physical blocks top-down -> non-contiguous vs logical
  for (int b = 0; b < batch; ++b) {
    const int nb = (seq_lens[b] + block_size - 1) / block_size;
    std::vector<half> kf = rnd(static_cast<std::size_t>(seq_lens[b]) * kv_hidden);
    std::vector<half> vf = rnd(static_cast<std::size_t>(seq_lens[b]) * kv_hidden);
    for (int c = 0; c < nb; ++c) {
      const int phys = next_phys--;                 // descending, interleaved across seqs
      block_tables[static_cast<std::size_t>(b) * max_blocks + c] = phys;
      const int clen = std::min(block_size, seq_lens[b] - c * block_size);
      for (int local = 0; local < clen; ++local) {
        const int t = c * block_size + local;
        for (int d = 0; d < kv_hidden; ++d) {
          kp[(static_cast<std::size_t>(phys) * block_size + local) * kv_hidden + d] = kf[static_cast<std::size_t>(t) * kv_hidden + d];
          vp[(static_cast<std::size_t>(phys) * block_size + local) * kv_hidden + d] = vf[static_cast<std::size_t>(t) * kv_hidden + d];
        }
      }
    }
  }

  half *dq, *dkp, *dvp, *dout_b; int *dbt, *dsl;
  float *sm, *sl, *so;
  const std::size_t out_elems = static_cast<std::size_t>(batch) * num_heads * head_dim;
  CK(cudaMalloc(&dq, q.size() * sizeof(half)));
  CK(cudaMalloc(&dkp, kp.size() * sizeof(half)));
  CK(cudaMalloc(&dvp, vp.size() * sizeof(half)));
  CK(cudaMalloc(&dout_b, out_elems * sizeof(half)));
  CK(cudaMalloc(&dbt, block_tables.size() * sizeof(int)));
  CK(cudaMalloc(&dsl, seq_lens.size() * sizeof(int)));
  CK(cudaMalloc(&sm, static_cast<std::size_t>(batch) * num_heads * scratch_chunks * sizeof(float)));
  CK(cudaMalloc(&sl, static_cast<std::size_t>(batch) * num_heads * scratch_chunks * sizeof(float)));
  CK(cudaMalloc(&so, static_cast<std::size_t>(batch) * num_heads * scratch_chunks * head_dim * sizeof(float)));
  CK(cudaMemcpy(dq, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dkp, kp.data(), kp.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dvp, vp.data(), vp.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbt, block_tables.data(), block_tables.size() * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dsl, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice));

  // Batched: all sequences in one launch.
  kernels::launch_attention_step_batched_paged(dq, dkp, dvp, dbt, dsl, max_blocks, max_seq, dout_b,
                                               batch, num_heads, num_kv_heads, head_dim, block_size,
                                               0, sm, sl, so, scratch_chunks);
  CK(cudaDeviceSynchronize());
  std::vector<half> out_batched(out_elems);
  CK(cudaMemcpy(out_batched.data(), dout_b, out_elems * sizeof(half), cudaMemcpyDeviceToHost));

  // Reference: one independent single-sequence paged call per sequence.
  double max_abs = 0.0;
  half* dout_ref;
  CK(cudaMalloc(&dout_ref, static_cast<std::size_t>(num_heads) * head_dim * sizeof(half)));
  for (int b = 0; b < batch; ++b) {
    const half* qb = dq + (static_cast<std::size_t>(b) * num_heads) * head_dim;
    const int* btb = dbt + static_cast<std::size_t>(b) * max_blocks;
    kernels::launch_attention_step_paged(qb, dkp, dvp, btb, dout_ref, seq_lens[b], num_heads,
                                         num_kv_heads, head_dim, block_size, 0, sm, sl, so, scratch_chunks);
    CK(cudaDeviceSynchronize());
    std::vector<half> ref(static_cast<std::size_t>(num_heads) * head_dim);
    CK(cudaMemcpy(ref.data(), dout_ref, ref.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double d = std::fabs(static_cast<double>(__half2float(ref[i])) -
                                 static_cast<double>(__half2float(out_batched[static_cast<std::size_t>(b) * num_heads * head_dim + i])));
      max_abs = std::max(max_abs, d);
    }
  }

  std::printf("paged_batched_attention_parity: batch=%d seq_lens={40,100,200} heads=%d kv=%d hd=%d\n",
              batch, num_heads, num_kv_heads, head_dim);
  std::printf("  per-seq non-contiguous block tables, shared pool; max_abs_diff(batched vs single)=%.3e\n", max_abs);
  const bool pass = max_abs < 1e-3;
  std::printf("%s\n", pass ? "PASS (batched == per-sequence single, no cross-seq interference)" : "FAIL");
  return pass ? 0 : 1;
}
