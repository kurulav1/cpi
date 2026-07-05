// Reference-oracle parity for batched paged decode attention (Phase 0 safety
// net, serving path).
//
// paged_batched_attention_parity_test checks batched == per-sequence-single — a
// self-consistency test that passes even when both paths share a bug. This test
// compares launch_attention_step_batched_paged against an INDEPENDENT naive
// float reference (per sequence, reading K/V from the shared pool through the
// block table), swept over head_dim ∈ {128,256} and GQA shapes. It catches the
// head_dim > blockDim scalar-accumulator class of bug in the batched split-K
// reduce kernel, which — unlike the non-batched split-K path (gated to
// head_dim==128) — runs at any head_dim and so silently corrupts head_dim=256
// (Qwen3.5) under batched serving.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// Per-sequence naive float reference, reading K/V from the shared pool via the
// block table exactly as the kernel does (so the oracle is independent of the
// kernel arithmetic). Writes head-major output for sequence b into out_b.
void reference_seq(const std::vector<half>& kp, const std::vector<half>& vp,
                   const std::vector<half>& q, const std::vector<int>& block_tables, int b,
                   int max_blocks, int seq_len, int num_heads, int num_kv_heads, int head_dim,
                   int block_size, std::vector<float>& out_b) {
  const int group = num_heads / num_kv_heads;
  const int kv_hidden = num_kv_heads * head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  out_b.assign(static_cast<std::size_t>(num_heads) * head_dim, 0.0f);
  std::vector<float> scores(seq_len);
  auto kv_base = [&](int t, int kv) -> std::size_t {
    const int c = t / block_size;
    const int local = t % block_size;
    const int phys = block_tables[static_cast<std::size_t>(b) * max_blocks + c];
    return (static_cast<std::size_t>(phys) * block_size + local) * kv_hidden +
           static_cast<std::size_t>(kv) * head_dim;
  };
  for (int h = 0; h < num_heads; ++h) {
    const int kv = h / group;
    const std::size_t qbase = (static_cast<std::size_t>(b) * num_heads + h) * head_dim;
    float m = -1e30f;
    for (int t = 0; t < seq_len; ++t) {
      const std::size_t kb = kv_base(t, kv);
      float s = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        s += __half2float(q[qbase + d]) * __half2float(kp[kb + d]);
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
      const std::size_t vb = kv_base(t, kv);
      for (int d = 0; d < head_dim; ++d) {
        out_b[static_cast<std::size_t>(h) * head_dim + d] += w * __half2float(vp[vb + d]);
      }
    }
  }
}

}  // namespace

int main() {
  const Config configs[] = {
      {28, 4, 128, "Qwen2.5 hd128"},
      {16, 4, 256, "Qwen3.5 hd256"},
      {8, 8, 256, "MHA hd256"},
  };
  const std::vector<int> seq_lens = {40, 100, 200};
  const int batch = static_cast<int>(seq_lens.size());
  const int block_size = 32;
  const double kCosGate = 0.999;

  std::mt19937 rng(913);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  auto rnd = [&](std::size_t n) {
    std::vector<half> h(n);
    for (auto& x : h) x = __float2half(uni(rng));
    return h;
  };

  const int max_seq = *std::max_element(seq_lens.begin(), seq_lens.end());
  const int max_blocks = (max_seq + block_size - 1) / block_size;
  const int scratch_chunks = max_blocks;

  int failures = 0, cases = 0;
  std::printf("%-16s %-24s  %-9s  %-9s  %s\n", "config", "per-seq cosine (len)", "min_cos",
              "max_abs", "result");
  std::printf("--------------------------------------------------------------------------------\n");

  for (const Config& c : configs) {
    const int kv_hidden = c.num_kv_heads * c.head_dim;
    const int pool_blocks = batch * max_blocks + 8;  // room to scramble

    // Shared KV pool + per-seq q + non-contiguous block tables (blocks handed out
    // top-down so logical != physical order).
    std::vector<half> kp(static_cast<std::size_t>(pool_blocks) * block_size * kv_hidden,
                         __float2half(0.0f));
    std::vector<half> vp = kp;
    std::vector<half> q = rnd(static_cast<std::size_t>(batch) * c.num_heads * c.head_dim);
    std::vector<int> block_tables(static_cast<std::size_t>(batch) * max_blocks, 0);
    int next_phys = pool_blocks - 1;
    for (int b = 0; b < batch; ++b) {
      const int nb = (seq_lens[b] + block_size - 1) / block_size;
      std::vector<half> kf = rnd(static_cast<std::size_t>(seq_lens[b]) * kv_hidden);
      std::vector<half> vf = rnd(static_cast<std::size_t>(seq_lens[b]) * kv_hidden);
      for (int cb = 0; cb < nb; ++cb) {
        const int phys = next_phys--;
        block_tables[static_cast<std::size_t>(b) * max_blocks + cb] = phys;
        const int clen = std::min(block_size, seq_lens[b] - cb * block_size);
        for (int local = 0; local < clen; ++local) {
          const int t = cb * block_size + local;
          for (int d = 0; d < kv_hidden; ++d) {
            kp[(static_cast<std::size_t>(phys) * block_size + local) * kv_hidden + d] =
                kf[static_cast<std::size_t>(t) * kv_hidden + d];
            vp[(static_cast<std::size_t>(phys) * block_size + local) * kv_hidden + d] =
                vf[static_cast<std::size_t>(t) * kv_hidden + d];
          }
        }
      }
    }

    half *dq, *dkp, *dvp, *dout;
    int *dbt, *dsl;
    float *sm, *sl, *so;
    const std::size_t out_elems = static_cast<std::size_t>(batch) * c.num_heads * c.head_dim;
    CK(cudaMalloc(&dq, q.size() * sizeof(half)));
    CK(cudaMalloc(&dkp, kp.size() * sizeof(half)));
    CK(cudaMalloc(&dvp, vp.size() * sizeof(half)));
    CK(cudaMalloc(&dout, out_elems * sizeof(half)));
    CK(cudaMalloc(&dbt, block_tables.size() * sizeof(int)));
    CK(cudaMalloc(&dsl, seq_lens.size() * sizeof(int)));
    CK(cudaMalloc(&sm,
                  static_cast<std::size_t>(batch) * c.num_heads * scratch_chunks * sizeof(float)));
    CK(cudaMalloc(&sl,
                  static_cast<std::size_t>(batch) * c.num_heads * scratch_chunks * sizeof(float)));
    CK(cudaMalloc(&so, static_cast<std::size_t>(batch) * c.num_heads * scratch_chunks * c.head_dim *
                           sizeof(float)));
    CK(cudaMemcpy(dq, q.data(), q.size() * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkp, kp.data(), kp.size() * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvp, vp.data(), vp.size() * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbt, block_tables.data(), block_tables.size() * sizeof(int),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsl, seq_lens.data(), seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice));

    kernels::launch_attention_step_batched_paged(dq, dkp, dvp, dbt, dsl, max_blocks, max_seq, dout,
                                                 batch, c.num_heads, c.num_kv_heads, c.head_dim,
                                                 block_size, 0, sm, sl, so, scratch_chunks);
    CK(cudaDeviceSynchronize());
    std::vector<half> got(out_elems);
    CK(cudaMemcpy(got.data(), dout, out_elems * sizeof(half), cudaMemcpyDeviceToHost));

    double min_cos = 1.0, max_abs = 0.0;
    char cbuf[128];
    int off = 0;
    for (int b = 0; b < batch; ++b) {
      std::vector<float> ref;
      reference_seq(kp, vp, q, block_tables, b, max_blocks, seq_lens[b], c.num_heads,
                    c.num_kv_heads, c.head_dim, block_size, ref);
      double dot = 0.0, na = 0.0, nb = 0.0;
      const std::size_t base = static_cast<std::size_t>(b) * c.num_heads * c.head_dim;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i];
        const double bb = __half2float(got[base + i]);
        dot += a * bb;
        na += a * a;
        nb += bb * bb;
        max_abs = std::max(max_abs, std::fabs(a - bb));
      }
      const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
      min_cos = std::min(min_cos, cos);
      off += std::snprintf(cbuf + off, sizeof(cbuf) - off, "%.4f(%d) ", cos, seq_lens[b]);
    }
    const bool pass = min_cos > kCosGate;
    if (!pass) ++failures;
    ++cases;
    std::printf("%-16s %-24s  %-9.5f  %-9.2e  %s\n", c.label, cbuf, min_cos, max_abs,
                pass ? "PASS" : "FAIL <<<");

    cudaFree(dq);
    cudaFree(dkp);
    cudaFree(dvp);
    cudaFree(dout);
    cudaFree(dbt);
    cudaFree(dsl);
    cudaFree(sm);
    cudaFree(sl);
    cudaFree(so);
  }

  std::printf("--------------------------------------------------------------------------------\n");
  std::printf("%d/%d configs passed (cosine gate %.4f)\n", cases - failures, cases, kCosGate);
  return failures == 0 ? 0 : 1;
}
