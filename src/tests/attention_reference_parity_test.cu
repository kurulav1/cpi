// Reference-oracle parity for decode attention (Phase 0 safety net).
//
// Unlike paged_attention_parity_test (which checks paged == contiguous, a
// self-consistency test that passes even when both paths are wrong the same
// way), this test compares launch_attention_step against an independent naive
// float reference computed on the host. It sweeps head_dim ∈ {64,128,256} and a
// range of GQA ratios so a kernel that assumes head_dim <= blockDim (the class
// of bug that corrupted Qwen3.5's head_dim=256 attention) is caught: at 256 the
// buggy kernel scored cosine ≈ 0.65 against this oracle, far below the gate.
//
// Exercises all four host decode-attention launches against the reference: the
// tiled and split-K paths, plus their device-position variants (the CUDA-graph
// decode path, where seq_len is read from device
// memory). All must match the same reference.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

namespace {

// Host reference: naive per-head softmax attention over `seq_len` cached tokens,
// reading the same fp16 values the kernel reads (via __half2float) so the only
// difference is float summation order. K/V layout is [seq_len, num_kv_heads,
// head_dim] row-major, matching cache_index(t, kv, d) = (t*num_kv_heads+kv)*head_dim+d.
void reference_attention(const std::vector<half>& q, const std::vector<half>& k,
                         const std::vector<half>& v, std::vector<float>& out, int seq_len,
                         int num_heads, int num_kv_heads, int head_dim) {
  const int group = num_heads / num_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  out.assign(static_cast<std::size_t>(num_heads) * head_dim, 0.0f);
  std::vector<float> scores(seq_len);
  for (int h = 0; h < num_heads; ++h) {
    const int kv = h / group;
    float m = -1e30f;
    for (int t = 0; t < seq_len; ++t) {
      float s = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        const float qv = __half2float(q[static_cast<std::size_t>(h) * head_dim + d]);
        const float kvv =
            __half2float(k[(static_cast<std::size_t>(t) * num_kv_heads + kv) * head_dim + d]);
        s += qv * kvv;
      }
      s *= scale;
      scores[t] = s;
      m = std::max(m, s);
    }
    float l = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      scores[t] = std::exp(scores[t] - m);
      l += scores[t];
    }
    const float inv_l = 1.0f / l;
    for (int t = 0; t < seq_len; ++t) {
      const float w = scores[t] * inv_l;
      for (int d = 0; d < head_dim; ++d) {
        out[static_cast<std::size_t>(h) * head_dim + d] +=
            w * __half2float(v[(static_cast<std::size_t>(t) * num_kv_heads + kv) * head_dim + d]);
      }
    }
  }
}

struct Config {
  int num_heads;
  int num_kv_heads;
  int head_dim;
  const char* label;
};

}  // namespace

int main() {
  const Config configs[] = {
      {8, 8, 64, "MHA hd64"},         {8, 8, 128, "MHA hd128"},      {8, 8, 256, "MHA hd256"},
      {32, 8, 128, "Llama3.1 hd128"}, {28, 4, 128, "Qwen2.5 hd128"}, {16, 4, 256, "Qwen3.5 hd256"},
      {16, 4, 64, "GQA hd64"},        {16, 4, 256, "GQA4x hd256"},
      {8, 1, 256, "MQA hd256"},       {8, 1, 128, "MQA hd128"},
      {8, 1, 512, "MQA hd512"},       {16, 8, 512, "GQA hd512"},
  };
  // Long sequences exercise the coarsened split (blocks_per_chunk > 1) and the
  // reduce combining multiple coarse chunks, on both host and graph (dpos) paths.
  const int seq_lens[] = {1, 7, 33, 200, 600, 2050};
  // cosine gate: a correct kernel matches the oracle to > 0.9999; the head_dim=256
  // scalar-accumulator bug scored ~0.65. 0.999 cleanly separates the two.
  const double kCosGate = 0.999;

  std::mt19937 rng(20260705);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  auto rnd = [&](std::size_t n) {
    std::vector<half> h(n);
    for (auto& x : h) x = __float2half(uni(rng));
    return h;
  };

  int failures = 0, cases = 0;
  std::printf("%-16s %-6s %-11s  %-9s  %-9s  %s\n", "config", "seq", "path", "cosine", "max_abs",
              "result");
  std::printf("--------------------------------------------------------------------------\n");

  for (const Config& c : configs) {
    const int kv_hidden = c.num_kv_heads * c.head_dim;
    const std::size_t out_elems = static_cast<std::size_t>(c.num_heads) * c.head_dim;
    for (int seq_len : seq_lens) {
      std::vector<half> q = rnd(out_elems);
      std::vector<half> k = rnd(static_cast<std::size_t>(seq_len) * kv_hidden);
      std::vector<half> v = rnd(static_cast<std::size_t>(seq_len) * kv_hidden);
      std::vector<float> ref;
      reference_attention(q, k, v, ref, seq_len, c.num_heads, c.num_kv_heads, c.head_dim);

      half *dq, *dk, *dv, *dout;
      CK(cudaMalloc(&dq, q.size() * sizeof(half)));
      CK(cudaMalloc(&dk, k.size() * sizeof(half)));
      CK(cudaMalloc(&dv, v.size() * sizeof(half)));
      CK(cudaMalloc(&dout, out_elems * sizeof(half)));
      CK(cudaMemcpy(dq, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dk, k.data(), k.size() * sizeof(half), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dv, v.data(), v.size() * sizeof(half), cudaMemcpyHostToDevice));

      // Split-K scratch (used only when the dispatch selects that path; harmless
      // otherwise). Size for the worst-case chunking of this seq_len.
      const int chunk = 32;
      const int scratch_chunks = (seq_len + chunk - 1) / chunk + 2;
      float *sm, *sl, *so;
      CK(cudaMalloc(&sm, static_cast<std::size_t>(c.num_heads) * scratch_chunks * sizeof(float)));
      CK(cudaMalloc(&sl, static_cast<std::size_t>(c.num_heads) * scratch_chunks * sizeof(float)));
      CK(cudaMalloc(&so, static_cast<std::size_t>(c.num_heads) * scratch_chunks * c.head_dim *
                             sizeof(float)));

      // Device position for the CUDA-graph capture variant: seq_len = pos[0]+1.
      int* dpos;
      const int pos_val = seq_len - 1;
      CK(cudaMalloc(&dpos, sizeof(int)));
      CK(cudaMemcpy(dpos, &pos_val, sizeof(int), cudaMemcpyHostToDevice));

      // Paths: host tiled / host split-K / device-pos tiled / device-pos split-K.
      // The device-pos launches (the CUDA-graph decode path) forward to the same
      // cores; covering them here guards that path
      // directly rather than only by construction.
      const char* labels[4] = {"tiled", "split-K", "dpos-tiled", "dpos-splitK"};
      for (int path = 0; path < 4; ++path) {
        const bool split = (path == 1 || path == 3);
        const bool device_pos = (path >= 2);
        if (!device_pos && !split) {
          kernels::launch_attention_step(dq, dk, dv, dout, seq_len, c.num_heads, c.num_kv_heads,
                                         c.head_dim, 0);
        } else if (!device_pos && split) {
          kernels::launch_attention_step(dq, dk, dv, dout, seq_len, c.num_heads, c.num_kv_heads,
                                         c.head_dim, 0, sm, sl, so, scratch_chunks, true);
        } else if (device_pos && !split) {
          kernels::launch_attention_step_device_pos(dq, dk, dv, dout, dpos, c.num_heads,
                                                    c.num_kv_heads, c.head_dim, 0);
        } else {
          kernels::launch_attention_step_device_pos(dq, dk, dv, dout, dpos, c.num_heads,
                                                    c.num_kv_heads, c.head_dim, 0, sm, sl, so,
                                                    scratch_chunks, true);
        }
        CK(cudaDeviceSynchronize());
        std::vector<half> got(out_elems);
        CK(cudaMemcpy(got.data(), dout, out_elems * sizeof(half), cudaMemcpyDeviceToHost));

        double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0;
        for (std::size_t i = 0; i < out_elems; ++i) {
          const double a = ref[i];
          const double b = __half2float(got[i]);
          dot += a * b;
          na += a * a;
          nb += b * b;
          max_abs = std::max(max_abs, std::fabs(a - b));
        }
        const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        const bool pass = cos > kCosGate;
        if (!pass) ++failures;
        ++cases;
        std::printf("%-16s %-6d %-11s  %-9.5f  %-9.2e  %s\n", c.label, seq_len, labels[path], cos,
                    max_abs, pass ? "PASS" : "FAIL <<<");
      }

      cudaFree(dq);
      cudaFree(dk);
      cudaFree(dv);
      cudaFree(dout);
      cudaFree(sm);
      cudaFree(sl);
      cudaFree(so);
      cudaFree(dpos);
    }
  }

  std::printf("--------------------------------------------------------------------------\n");
  std::printf("%d/%d cases passed (cosine gate %.4f)\n", cases - failures, cases, kCosGate);
  return failures == 0 ? 0 : 1;
}
