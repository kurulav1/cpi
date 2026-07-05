// Reference-oracle parity for INT4 KV-cache decode attention (Phase 0 safety
// net). Compares launch_attention_step_int4 against an independent naive float
// reference computed on the DEQUANTIZED int4 K/V (so quantization error is
// shared and the test isolates kernel correctness), swept over head_dim ∈
// {128,256}. head_dim=256 exercises the serial fallback kernel, which is the
// path carrying the head_dim > blockDim scalar-accumulator class of bug.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

struct Config {
  int num_heads;
  int num_kv_heads;
  int head_dim;
  const char* label;
};

}  // namespace

int main() {
  const Config configs[] = {
      {28, 4, 128, "Qwen2.5 hd128"},
      {8, 8, 128, "MHA hd128"},
      {16, 4, 256, "hd256 (fallback)"},
      {8, 8, 256, "MHA hd256"},
  };
  const int seq_lens[] = {1, 33, 200};
  const double kCosGate = 0.999;

  std::mt19937 rng(4242);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

  int failures = 0, cases = 0;
  std::printf("%-18s %-6s %-8s  %-9s  %-9s  %s\n", "config", "seq", "path", "cosine", "max_abs",
              "result");
  std::printf("----------------------------------------------------------------------------\n");

  for (const Config& c : configs) {
    const int packed = c.head_dim / 2;
    const std::size_t out_elems = static_cast<std::size_t>(c.num_heads) * c.head_dim;
    for (int seq_len : seq_lens) {
      // Random fp16 Q; per-(token,kv_head) random K/V quantized to int4 exactly as
      // store_kv_int4_kernel does (absmax/7 scale, round-to-nearest, clamp [-8,7]).
      std::vector<half> q(out_elems);
      for (auto& x : q) x = __float2half(uni(rng));

      std::vector<int8_t> k_i4(static_cast<std::size_t>(seq_len) * c.num_kv_heads * packed);
      std::vector<int8_t> v_i4(k_i4.size());
      std::vector<half> k_scales(static_cast<std::size_t>(seq_len) * c.num_kv_heads);
      std::vector<half> v_scales(k_scales.size());
      // Dequantized K/V for the reference: [seq_len][num_kv_heads][head_dim].
      std::vector<float> kdq(static_cast<std::size_t>(seq_len) * c.num_kv_heads * c.head_dim);
      std::vector<float> vdq(kdq.size());

      auto quantize = [&](std::vector<int8_t>& cache, std::vector<half>& scales,
                          std::vector<float>& dq) {
        for (int t = 0; t < seq_len; ++t) {
          for (int kv = 0; kv < c.num_kv_heads; ++kv) {
            float vals[512];
            float amax = 0.0f;
            for (int d = 0; d < c.head_dim; ++d) {
              vals[d] = uni(rng);
              amax = std::max(amax, std::fabs(vals[d]));
            }
            const float s = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
            scales[static_cast<std::size_t>(t) * c.num_kv_heads + kv] = __float2half(s);
            const std::size_t cbase = (static_cast<std::size_t>(t) * c.num_kv_heads + kv) * packed;
            const std::size_t dbase =
                (static_cast<std::size_t>(t) * c.num_kv_heads + kv) * c.head_dim;
            for (int i = 0; i < packed; ++i) {
              const int q0 = std::max(-8, std::min(7, (int)std::lround(vals[2 * i] / s)));
              const int q1 = std::max(-8, std::min(7, (int)std::lround(vals[2 * i + 1] / s)));
              cache[cbase + i] = static_cast<int8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
              dq[dbase + 2 * i] = q0 * s;
              dq[dbase + 2 * i + 1] = q1 * s;
            }
          }
        }
      };
      quantize(k_i4, k_scales, kdq);
      quantize(v_i4, v_scales, vdq);

      // Reference attention over the dequantized int4 K/V.
      const int group = c.num_heads / c.num_kv_heads;
      const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
      std::vector<float> ref(out_elems, 0.0f);
      std::vector<float> scores(seq_len);
      for (int h = 0; h < c.num_heads; ++h) {
        const int kv = h / group;
        float m = -1e30f;
        for (int t = 0; t < seq_len; ++t) {
          const std::size_t kb = (static_cast<std::size_t>(t) * c.num_kv_heads + kv) * c.head_dim;
          float s = 0.0f;
          for (int d = 0; d < c.head_dim; ++d)
            s += __half2float(q[static_cast<std::size_t>(h) * c.head_dim + d]) * kdq[kb + d];
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
          const std::size_t vb = (static_cast<std::size_t>(t) * c.num_kv_heads + kv) * c.head_dim;
          for (int d = 0; d < c.head_dim; ++d)
            ref[static_cast<std::size_t>(h) * c.head_dim + d] += w * vdq[vb + d];
        }
      }

      half *dq_, *dout;
      int8_t *dk, *dv;
      half *dks, *dvs;
      CK(cudaMalloc(&dq_, q.size() * sizeof(half)));
      CK(cudaMalloc(&dout, out_elems * sizeof(half)));
      CK(cudaMalloc(&dk, k_i4.size()));
      CK(cudaMalloc(&dv, v_i4.size()));
      CK(cudaMalloc(&dks, k_scales.size() * sizeof(half)));
      CK(cudaMalloc(&dvs, v_scales.size() * sizeof(half)));
      CK(cudaMemcpy(dq_, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dk, k_i4.data(), k_i4.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dv, v_i4.data(), v_i4.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dks, k_scales.data(), k_scales.size() * sizeof(half), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dvs, v_scales.data(), v_scales.size() * sizeof(half), cudaMemcpyHostToDevice));

      const int chunkn = (seq_len + 31) / 32 + 2;
      float *sm, *sl, *so;
      CK(cudaMalloc(&sm, static_cast<std::size_t>(c.num_heads) * chunkn * sizeof(float)));
      CK(cudaMalloc(&sl, static_cast<std::size_t>(c.num_heads) * chunkn * sizeof(float)));
      CK(cudaMalloc(&so,
                    static_cast<std::size_t>(c.num_heads) * chunkn * c.head_dim * sizeof(float)));

      for (int path = 0; path < 2; ++path) {
        const bool split = (path == 1);
        if (split) {
          kernels::launch_attention_step_int4(dq_, dk, dv, dks, dvs, dout, seq_len, c.num_heads,
                                              c.num_kv_heads, c.head_dim, 0, sm, sl, so, chunkn,
                                              true);
        } else {
          kernels::launch_attention_step_int4(dq_, dk, dv, dks, dvs, dout, seq_len, c.num_heads,
                                              c.num_kv_heads, c.head_dim, 0, nullptr, nullptr,
                                              nullptr, 0, false);
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
        std::printf("%-18s %-6d %-8s  %-9.5f  %-9.2e  %s\n", c.label, seq_len,
                    split ? "split-K" : "fallback", cos, max_abs, pass ? "PASS" : "FAIL <<<");
      }

      cudaFree(dq_);
      cudaFree(dout);
      cudaFree(dk);
      cudaFree(dv);
      cudaFree(dks);
      cudaFree(dvs);
      cudaFree(sm);
      cudaFree(sl);
      cudaFree(so);
    }
  }

  std::printf("----------------------------------------------------------------------------\n");
  std::printf("%d/%d cases passed (cosine gate %.4f)\n", cases - failures, cases, kCosGate);
  return failures == 0 ? 0 : 1;
}
