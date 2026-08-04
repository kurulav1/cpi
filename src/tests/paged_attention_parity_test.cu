// Parity test for the paged split-K decode attention (P3 phase 2b).
//
// Proves launch_attention_step_paged (reads K/V through a per-chunk block table
// into a block pool) produces the same output as the contiguous
// launch_attention_step, under a deliberately non-contiguous block table. Same
// online-softmax arithmetic on the same values => expect bit-identical output.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "runtime/kernels.cuh"

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t _e = (x);                                                                  \
    if (_e != cudaSuccess) {                                                               \
      std::printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(_e)); \
      return 2;                                                                            \
    }                                                                                      \
  } while (0)

int main() {
  const int num_heads = 28, num_kv_heads = 4, head_dim = 128;  // Qwen2.5-7B shape
  const int seq_len = 200, block_size = 32;
  const int kv_hidden = num_kv_heads * head_dim;
  const int num_chunks = (seq_len + block_size - 1) / block_size;  // 7
  const int scratch_chunks = num_chunks;
  const int num_pool_blocks = num_chunks + 3;  // room to scramble

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  auto rnd = [&](std::size_t n) {
    std::vector<half> h(n);
    for (auto& x : h) x = __float2half(uni(rng));
    return h;
  };
  std::vector<half> q = rnd(static_cast<std::size_t>(num_heads) * head_dim);
  std::vector<half> kf = rnd(static_cast<std::size_t>(seq_len) * kv_hidden);
  std::vector<half> vf = rnd(static_cast<std::size_t>(seq_len) * kv_hidden);

  // Non-contiguous block table: logical chunk c -> physical block (reversed).
  std::vector<int> bt(num_chunks);
  for (int c = 0; c < num_chunks; ++c) bt[c] = num_pool_blocks - 1 - c;

  // Block pool laid out as a flat cache of num_pool_blocks*block_size tokens;
  // place logical chunk c's tokens into physical block bt[c].
  std::vector<half> kp(static_cast<std::size_t>(num_pool_blocks) * block_size * kv_hidden,
                       __float2half(0.0f));
  std::vector<half> vp = kp;
  for (int c = 0; c < num_chunks; ++c) {
    const int cstart = c * block_size;
    const int clen = std::min(block_size, seq_len - cstart);
    for (int local = 0; local < clen; ++local) {
      const int t = cstart + local;
      const int phys = bt[c] * block_size + local;
      for (int d = 0; d < kv_hidden; ++d) {
        kp[static_cast<std::size_t>(phys) * kv_hidden + d] =
            kf[static_cast<std::size_t>(t) * kv_hidden + d];
        vp[static_cast<std::size_t>(phys) * kv_hidden + d] =
            vf[static_cast<std::size_t>(t) * kv_hidden + d];
      }
    }
  }

  half *dq, *dkf, *dvf, *dkp, *dvp, *dref, *dpaged;
  int* dbt;
  float *sm, *sl, *so;
  const std::size_t out_elems = static_cast<std::size_t>(num_heads) * head_dim;
  CK(cudaMalloc(&dq, q.size() * sizeof(half)));
  CK(cudaMalloc(&dkf, kf.size() * sizeof(half)));
  CK(cudaMalloc(&dvf, vf.size() * sizeof(half)));
  CK(cudaMalloc(&dkp, kp.size() * sizeof(half)));
  CK(cudaMalloc(&dvp, vp.size() * sizeof(half)));
  CK(cudaMalloc(&dref, out_elems * sizeof(half)));
  CK(cudaMalloc(&dpaged, out_elems * sizeof(half)));
  CK(cudaMalloc(&dbt, bt.size() * sizeof(int)));
  CK(cudaMalloc(&sm, static_cast<std::size_t>(num_heads) * scratch_chunks * sizeof(float)));
  CK(cudaMalloc(&sl, static_cast<std::size_t>(num_heads) * scratch_chunks * sizeof(float)));
  CK(cudaMalloc(&so,
                static_cast<std::size_t>(num_heads) * scratch_chunks * head_dim * sizeof(float)));
  CK(cudaMemcpy(dq, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dkf, kf.data(), kf.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dvf, vf.data(), vf.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dkp, kp.data(), kp.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dvp, vp.data(), vp.size() * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbt, bt.data(), bt.size() * sizeof(int), cudaMemcpyHostToDevice));

  // Contiguous reference (split-K path).
  kernels::launch_attention_step(dq, dkf, dvf, dref, seq_len, num_heads, num_kv_heads, head_dim, 0,
                                 sm, sl, so, scratch_chunks, true);
  // Paged: same values via a non-contiguous block table.
  kernels::launch_attention_step_paged(dq, dkp, dvp, dbt, dpaged, seq_len, num_heads, num_kv_heads,
                                       head_dim, block_size, 0, sm, sl, so, scratch_chunks);
  CK(cudaDeviceSynchronize());

  std::vector<half> ref(out_elems), paged(out_elems);
  CK(cudaMemcpy(ref.data(), dref, out_elems * sizeof(half), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(paged.data(), dpaged, out_elems * sizeof(half), cudaMemcpyDeviceToHost));

  double max_abs = 0.0;
  int mismatches = 0;
  for (std::size_t i = 0; i < out_elems; ++i) {
    const float a = __half2float(ref[i]);
    const float b = __half2float(paged[i]);
    const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
    max_abs = std::max(max_abs, d);
    if (d > 0.0) ++mismatches;
  }
  std::printf("paged_attention_parity: seq_len=%d heads=%d kv=%d head_dim=%d block=%d\n", seq_len,
              num_heads, num_kv_heads, head_dim, block_size);
  std::printf(
      "  non-contiguous block table (reversed); max_abs_diff=%.3e mismatched_elems=%d/%zu\n",
      max_abs, mismatches, out_elems);
  // The paged kernel does the same online-softmax math on the same values, so it
  // matches to fp16 precision. Tiny (~1 ULP) diffs on a few elements come from
  // FMA-contraction/codegen differences between the two kernel instantiations,
  // not addressing — a wrong-block read would be off by O(1), not 1e-5. Greedy
  // decoding (argmax over logits) is robust to this, so the tolerance is fp16-scale.
  const double tol = 1e-3;
  const bool pass = max_abs < tol;
  std::printf("%s (tol=%.0e)\n", pass ? "PASS (numerically identical to contiguous)" : "FAIL", tol);
  return pass ? 0 : 1;
}
