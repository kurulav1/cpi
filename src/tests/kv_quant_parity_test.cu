// Parity test for the generalized quantized KV cache (launch_store_kv_quant /
// launch_attention_step_quant) across the supported (k_bits, v_bits, rotate_k)
// combinations. Two checks per config:
//   1. Store: run the device store kernel token by token, download the cache,
//      and verify each dequantized element is within half a quant step of the
//      host value (host applies the same Walsh-Hadamard rotation for rot
//      configs), which validates scales, packing, and the in-kernel rotation.
//   2. Attention: naive double-precision reference over the downloaded,
//      dequantized cache (with a host-rotated Q for rot configs, matching the
//      kernel's in-place Q rotation) vs the GPU output, fallback and split-K.
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

struct QuantConfig {
  int k_bits;
  int v_bits;
  bool rot;
  const char* label;
};

void fwht_host(std::vector<float>& x) {
  const int n = static_cast<int>(x.size());
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += 2 * len) {
      for (int j = i; j < i + len; ++j) {
        const float a = x[j];
        const float b = x[j + len];
        x[j] = a + b;
        x[j + len] = a - b;
      }
    }
  }
  const float rn = 1.0f / std::sqrt(static_cast<float>(n));
  for (auto& v : x) v *= rn;
}

float dequant_elem(const std::vector<int8_t>& cache, std::size_t row_base, int d, int bits,
                   float scale) {
  if (bits == 4) {
    const int8_t b = cache[row_base + (d >> 1)];
    const int q = (d & 1) ? (static_cast<int>(b) >> 4) : ((static_cast<int>(b) << 28) >> 28);
    return static_cast<float>(q) * scale;
  }
  return static_cast<float>(cache[row_base + d]) * scale;
}

}  // namespace

int main() {
  const int num_heads = 32;
  const int num_kv_heads = 8;
  const int head_dim = 128;
  const int max_context = 256;
  const QuantConfig configs[] = {
      {4, 4, false, "K4V4"},
      {4, 4, true, "K4V4+rot"},
      {8, 4, false, "K8V4"},
      {8, 8, false, "K8V8"},
  };
  const int seq_lens[] = {1, 33, 200};
  const double kCosGate = 0.999;

  std::mt19937 rng(777);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

  int failures = 0, cases = 0;
  std::printf("%-10s %-6s %-8s  %-9s  %-9s  %s\n", "config", "seq", "path", "cosine", "max_abs",
              "result");
  std::printf("----------------------------------------------------------------------------\n");

  for (const QuantConfig& c : configs) {
    const int k_row = head_dim * c.k_bits / 8;
    const int v_row = head_dim * c.v_bits / 8;
    const std::size_t out_elems = static_cast<std::size_t>(num_heads) * head_dim;
    const std::size_t kv_elems = static_cast<std::size_t>(num_kv_heads) * head_dim;

    for (int seq_len : seq_lens) {
      // Raw fp16 K/V per token, stored via the device kernel position by position.
      std::vector<std::vector<float>> k_raw(seq_len), v_raw(seq_len);
      std::vector<half> q(out_elems);
      for (auto& x : q) x = __float2half(uni(rng));

      half *dk_in, *dv_in;
      int8_t *dkc, *dvc;
      half *dks, *dvs;
      CK(cudaMalloc(&dk_in, kv_elems * sizeof(half)));
      CK(cudaMalloc(&dv_in, kv_elems * sizeof(half)));
      CK(cudaMalloc(&dkc, static_cast<std::size_t>(max_context) * num_kv_heads * k_row));
      CK(cudaMalloc(&dvc, static_cast<std::size_t>(max_context) * num_kv_heads * v_row));
      CK(cudaMalloc(&dks, static_cast<std::size_t>(max_context) * num_kv_heads * sizeof(half)));
      CK(cudaMalloc(&dvs, static_cast<std::size_t>(max_context) * num_kv_heads * sizeof(half)));

      for (int t = 0; t < seq_len; ++t) {
        k_raw[t].resize(kv_elems);
        v_raw[t].resize(kv_elems);
        std::vector<half> kh(kv_elems), vh(kv_elems);
        for (std::size_t i = 0; i < kv_elems; ++i) {
          k_raw[t][i] = uni(rng);
          v_raw[t][i] = uni(rng);
          kh[i] = __float2half(k_raw[t][i]);
          vh[i] = __float2half(v_raw[t][i]);
        }
        CK(cudaMemcpy(dk_in, kh.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv_in, vh.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice));
        kernels::launch_store_kv_quant(dk_in, dv_in, dkc, dvc, dks, dvs, t, num_kv_heads, head_dim,
                                       max_context, c.k_bits, c.v_bits, c.rot, 0);
      }
      CK(cudaDeviceSynchronize());

      std::vector<int8_t> kc(static_cast<std::size_t>(seq_len) * num_kv_heads * k_row);
      std::vector<int8_t> vc(static_cast<std::size_t>(seq_len) * num_kv_heads * v_row);
      std::vector<half> ksc(static_cast<std::size_t>(seq_len) * num_kv_heads);
      std::vector<half> vsc(ksc.size());
      CK(cudaMemcpy(kc.data(), dkc, kc.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(vc.data(), dvc, vc.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(ksc.data(), dks, ksc.size() * sizeof(half), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(vsc.data(), dvs, vsc.size() * sizeof(half), cudaMemcpyDeviceToHost));

      // Check 1: store parity. Each dequantized element must sit within ~half a
      // quant step of the host-side (optionally rotated) fp16 value. The small
      // slack absorbs fp16 input rounding and float-vs-host FWHT differences.
      bool store_ok = true;
      double store_worst = 0.0;
      for (int t = 0; t < seq_len && store_ok; ++t) {
        for (int kv = 0; kv < num_kv_heads; ++kv) {
          std::vector<float> kexp(k_raw[t].begin() + kv * head_dim,
                                  k_raw[t].begin() + (kv + 1) * head_dim);
          if (c.rot) fwht_host(kexp);
          const float ks = __half2float(ksc[static_cast<std::size_t>(t) * num_kv_heads + kv]);
          const float vs = __half2float(vsc[static_cast<std::size_t>(t) * num_kv_heads + kv]);
          const std::size_t kb = (static_cast<std::size_t>(t) * num_kv_heads + kv) *
                                 static_cast<std::size_t>(k_row);
          const std::size_t vb = (static_cast<std::size_t>(t) * num_kv_heads + kv) *
                                 static_cast<std::size_t>(v_row);
          for (int d = 0; d < head_dim; ++d) {
            const float kdq = dequant_elem(kc, kb, d, c.k_bits, ks);
            const float vdq = dequant_elem(vc, vb, d, c.v_bits, vs);
            const float kerr = std::fabs(kdq - kexp[d]) / std::max(ks, 1e-6f);
            const float verr =
                std::fabs(vdq - v_raw[t][kv * head_dim + d]) / std::max(vs, 1e-6f);
            store_worst = std::max(store_worst, static_cast<double>(std::max(kerr, verr)));
            if (kerr > 0.75f || verr > 0.75f) {
              store_ok = false;
              break;
            }
          }
          if (!store_ok) break;
        }
      }
      ++cases;
      if (!store_ok) ++failures;
      std::printf("%-10s %-6d %-8s  %-9.4f  %-9.2e  %s\n", c.label, seq_len, "store", 0.0,
                  store_worst, store_ok ? "PASS" : "FAIL <<<");

      // Reference attention over the downloaded, dequantized cache. For rot
      // configs the reference Q is host-rotated, matching the kernel.
      const int group = num_heads / num_kv_heads;
      const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
      std::vector<float> ref(out_elems, 0.0f);
      std::vector<double> scores(seq_len);
      for (int h = 0; h < num_heads; ++h) {
        const int kv = h / group;
        std::vector<float> qh(head_dim);
        for (int d = 0; d < head_dim; ++d) qh[d] = __half2float(q[h * head_dim + d]);
        if (c.rot) {
          fwht_host(qh);
          for (int d = 0; d < head_dim; ++d) qh[d] = __half2float(__float2half(qh[d]));
        }
        double m = -1e30;
        for (int t = 0; t < seq_len; ++t) {
          const float ks = __half2float(ksc[static_cast<std::size_t>(t) * num_kv_heads + kv]);
          const std::size_t kb = (static_cast<std::size_t>(t) * num_kv_heads + kv) *
                                 static_cast<std::size_t>(k_row);
          double s = 0.0;
          for (int d = 0; d < head_dim; ++d) s += qh[d] * dequant_elem(kc, kb, d, c.k_bits, ks);
          s *= scale;
          scores[t] = s;
          m = std::max(m, s);
        }
        double l = 0.0;
        for (int t = 0; t < seq_len; ++t) {
          scores[t] = std::exp(scores[t] - m);
          l += scores[t];
        }
        const double inv_l = 1.0 / l;
        for (int t = 0; t < seq_len; ++t) {
          const double w = scores[t] * inv_l;
          const float vs = __half2float(vsc[static_cast<std::size_t>(t) * num_kv_heads + kv]);
          const std::size_t vb = (static_cast<std::size_t>(t) * num_kv_heads + kv) *
                                 static_cast<std::size_t>(v_row);
          for (int d = 0; d < head_dim; ++d) {
            ref[static_cast<std::size_t>(h) * head_dim + d] +=
                static_cast<float>(w * dequant_elem(vc, vb, d, c.v_bits, vs));
          }
        }
      }

      half *dq_, *dout;
      CK(cudaMalloc(&dq_, q.size() * sizeof(half)));
      CK(cudaMalloc(&dout, out_elems * sizeof(half)));
      CK(cudaMemcpy(dq_, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
      const int chunkn = (seq_len + 31) / 32 + 2;
      float *sm, *sl, *so;
      CK(cudaMalloc(&sm, static_cast<std::size_t>(num_heads) * chunkn * sizeof(float)));
      CK(cudaMalloc(&sl, static_cast<std::size_t>(num_heads) * chunkn * sizeof(float)));
      CK(cudaMalloc(&so,
                    static_cast<std::size_t>(num_heads) * chunkn * head_dim * sizeof(float)));

      for (int path = 0; path < 2; ++path) {
        const bool split = (path == 1);
        kernels::launch_attention_step_quant(
            dq_, dkc, dvc, dks, dvs, dout, seq_len, num_heads, num_kv_heads, head_dim, c.k_bits,
            c.v_bits, c.rot, 0, split ? sm : nullptr, split ? sl : nullptr, split ? so : nullptr,
            split ? chunkn : 0, split);
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
        std::printf("%-10s %-6d %-8s  %-9.5f  %-9.2e  %s\n", c.label, seq_len,
                    split ? "split-K" : "fallback", cos, max_abs, pass ? "PASS" : "FAIL <<<");
      }

      cudaFree(dq_);
      cudaFree(dout);
      cudaFree(dk_in);
      cudaFree(dv_in);
      cudaFree(dkc);
      cudaFree(dvc);
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
