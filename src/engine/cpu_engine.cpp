// cpu_engine.cpp Ã¢â‚¬â€ CPU inference engine for LLaMA-family models.
//
// Optimization techniques applied:
//
//      Ã¢â‚¬â€ AVX2 256-bit vector arithmetic (8 FP32 values per SIMD lane).
//        Weights are stored as FP16 in the .ll2c file; the F16C instruction
//        _mm256_cvtph_ps converts 8 half-precision values to 8 single-
//        precision values in one instruction, keeping conversion overhead
//        close to zero.
//
//      Ã¢â‚¬â€ 4-output register blocking: each OpenMP thread processes four
//        consecutive output rows of the weight matrix per inner-loop trip,
//        loading the same 8-element slice of the input vector once and
//        multiplying it against four rows.  This quadruples arithmetic
//        intensity relative to one-row-at-a-time traversal.
//
//      Ã¢â‚¬â€ _mm_prefetch hints issued 20 iterations ahead inside the FP16
//        weight stream.  At 2 bytes per FP16 element, 20Ãƒâ€”8 = 160 elements
//        = 320 bytes Ã¢â€°Ë† 5 cache lines are prefetched per row per iteration,
//        hiding DRAM-to-L1 latency for the dominant memory-bound GEMV.
//
//     Ã¢â‚¬â€ #pragma omp parallel for with dynamic scheduling partitions the
//        output-row dimension across all available hardware threads, giving
//        near-linear scaling on multi-core CPUs.

#include "engine/cpu_engine.hpp"
#include "common.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// SIMD headers Ã¢â‚¬â€ included unconditionally; individual code paths are
// guarded by feature macros set by the compiler when AVX2 / F16C are enabled.
#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <immintrin.h>
#endif

// OpenMP
#if defined(_OPENMP)
#  include <omp.h>
#endif

namespace engine {
namespace {

// ============================================================
// Platform-portable prefetch
// ============================================================
inline void prefetch_r(const void* p) {
#if defined(_MSC_VER)
  _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p, 0, 1);
#endif
}

// ============================================================
// FP16 Ã¢â€ â€™ FP32 conversion
// ============================================================

// Software fallback: convert a single IEEE 754 half-precision value
// (stored as uint16_t) to a single-precision float.
inline float fp16_to_fp32(uint16_t h) {
  const uint32_t sign     = static_cast<uint32_t>(h >> 15) << 31;
  const uint32_t exp_h    = (h >> 10) & 0x1F;
  const uint32_t mant_h   = h & 0x3FF;
  uint32_t bits;
  if (exp_h == 0) {
    if (mant_h == 0) {
      bits = sign;
    } else {
      // Denormal: shift mantissa until leading 1 appears.
      uint32_t m = mant_h, e = 0;
      while (!(m & 0x400)) { m <<= 1; ++e; }
      bits = sign | ((127 - 15 - e + 1) << 23) | ((m & 0x3FF) << 13);
    }
  } else if (exp_h == 31) {
    bits = sign | 0x7F800000u | (mant_h << 13);  // Inf / NaN
  } else {
    bits = sign | ((exp_h + 112) << 23) | (mant_h << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// ============================================================
// SIMD GEMV helpers
// ============================================================

// Horizontal sum of an __m256 register (8 x float Ã¢â€ â€™ float).
#if defined(__AVX__)
inline float hsum256(__m256 v) {
  const __m128 hi  = _mm256_extractf128_ps(v, 1);
  const __m128 lo  = _mm256_castps256_ps128(v);
  const __m128 sum = _mm_add_ps(lo, hi);
  const __m128 h1  = _mm_hadd_ps(sum, sum);
  const __m128 h2  = _mm_hadd_ps(h1, h1);
  return _mm_cvtss_f32(h2);
}
#endif  // __AVX__

// MSVC with /arch:AVX2 supports _mm256_cvtph_ps but does not define __F16C__.
// Treat it as equivalent to having F16C so we get the fast path.
#if defined(__AVX2__) && !defined(__F16C__) && defined(_MSC_VER)
#  define CPU_ENGINE_HAVE_F16C 1
#elif defined(__F16C__)
#  define CPU_ENGINE_HAVE_F16C 1
#endif

// gemv_fp16_impl Ã¢â‚¬â€ y[0..M) = W[MÃƒâ€”N] * x[N]
//
// W  : row-major FP16 weight matrix, [M rows Ãƒâ€” N cols]
// x  : FP32 input vector, length N
// y  : FP32 output vector, length M (written, not accumulated)
//
// Inner loop processes 8 elements per AVX2 iteration, 4 output rows per
// OpenMP thread iteration (register blocking), with prefetch 160 FP16
// elements (320 bytes Ã¢â€°Ë† 5 cache lines) ahead on the first row pointer.
//
// Requires M to be a multiple of 4 and N to be a multiple of 8.
// The caller (gemv_fp16) pads if necessary.
static void gemv_fp16_impl(const uint16_t* __restrict__ W,
                           const float*    __restrict__ x,
                           float*          __restrict__ y,
                           int M, int N) {
#if defined(__AVX2__) && defined(CPU_ENGINE_HAVE_F16C)
  // --- AVX2 + F16C fast path ---
  // Outer loop parallelised over output rows in blocks of 4.
  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < M; i += 4) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const uint16_t* r0 = W + static_cast<std::ptrdiff_t>(i + 0) * N;
    const uint16_t* r1 = W + static_cast<std::ptrdiff_t>(i + 1) * N;
    const uint16_t* r2 = W + static_cast<std::ptrdiff_t>(i + 2) * N;
    const uint16_t* r3 = W + static_cast<std::ptrdiff_t>(i + 3) * N;

    for (int j = 0; j < N; j += 8) {
      // V6: prefetch ~20 iterations (160 fp16 elements = 320 bytes) ahead.
      // Only issue on the first two rows to avoid overfilling the prefetch
      // buffer; the remaining rows benefit from hardware prefetcher.
      prefetch_r(r0 + j + 160);
      prefetch_r(r1 + j + 160);

      // Load input slice (FP32, 8 elements = 32 bytes).
      const __m256 xv = _mm256_loadu_ps(x + j);

      // Load 8Ãƒâ€”FP16 per row, convert to FP32, fused-multiply-add.
      // _mm256_cvtph_ps takes a 128-bit register of 8 fp16 and returns
      // a 256-bit register of 8 fp32 Ã¢â‚¬â€ one instruction, no precision loss.
      acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(
                                 reinterpret_cast<const __m128i*>(r0 + j))),
                             xv, acc0);
      acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(
                                 reinterpret_cast<const __m128i*>(r1 + j))),
                             xv, acc1);
      acc2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(
                                 reinterpret_cast<const __m128i*>(r2 + j))),
                             xv, acc2);
      acc3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(
                                 reinterpret_cast<const __m128i*>(r3 + j))),
                             xv, acc3);
    }

    y[i + 0] = hsum256(acc0);
    y[i + 1] = hsum256(acc1);
    y[i + 2] = hsum256(acc2);
    y[i + 3] = hsum256(acc3);
  }

#elif defined(__AVX__)
  // --- AVX path (FP32 weights only, no F16C) ---
  // Convert entire weight rows to FP32 first (done inside parallel region).
  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < M; i += 4) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const uint16_t* r0 = W + static_cast<std::ptrdiff_t>(i + 0) * N;
    const uint16_t* r1 = W + static_cast<std::ptrdiff_t>(i + 1) * N;
    const uint16_t* r2 = W + static_cast<std::ptrdiff_t>(i + 2) * N;
    const uint16_t* r3 = W + static_cast<std::ptrdiff_t>(i + 3) * N;

    for (int j = 0; j < N; j += 8) {
      prefetch_r(r0 + j + 160);

      const __m256 xv = _mm256_loadu_ps(x + j);

      // Software FP16Ã¢â€ â€™FP32 conversion for 8 elements.
      float tmp0[8], tmp1[8], tmp2[8], tmp3[8];
      for (int k = 0; k < 8; ++k) {
        tmp0[k] = fp16_to_fp32(r0[j + k]);
        tmp1[k] = fp16_to_fp32(r1[j + k]);
        tmp2[k] = fp16_to_fp32(r2[j + k]);
        tmp3[k] = fp16_to_fp32(r3[j + k]);
      }
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(tmp0), xv));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(tmp1), xv));
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(tmp2), xv));
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(tmp3), xv));
    }

    y[i + 0] = hsum256(acc0);
    y[i + 1] = hsum256(acc1);
    y[i + 2] = hsum256(acc2);
    y[i + 3] = hsum256(acc3);
  }

#else
  // --- Scalar fallback ---
  // Still uses 4-output blocking and ILP-friendly accumulator structure so
  // the auto-vectoriser has a good chance of emitting SIMD code.
  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < M; i += 4) {
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const uint16_t* r0 = W + static_cast<std::ptrdiff_t>(i + 0) * N;
    const uint16_t* r1 = W + static_cast<std::ptrdiff_t>(i + 1) * N;
    const uint16_t* r2 = W + static_cast<std::ptrdiff_t>(i + 2) * N;
    const uint16_t* r3 = W + static_cast<std::ptrdiff_t>(i + 3) * N;
    for (int j = 0; j < N; ++j) {
      const float xj = x[j];
      acc0 += fp16_to_fp32(r0[j]) * xj;
      acc1 += fp16_to_fp32(r1[j]) * xj;
      acc2 += fp16_to_fp32(r2[j]) * xj;
      acc3 += fp16_to_fp32(r3[j]) * xj;
    }
    y[i + 0] = acc0;
    y[i + 1] = acc1;
    y[i + 2] = acc2;
    y[i + 3] = acc3;
  }
#endif
}

// gemv_fp32_impl Ã¢â‚¬â€ same blocking/prefetch structure as gemv_fp16_impl but for
// weights already stored as FP32 (e.g. dequantised INT8 MLP weights).
static void gemv_fp32_impl(const float* __restrict__ W,
                           const float* __restrict__ x,
                           float*       __restrict__ y,
                           int M, int N) {
#if defined(__AVX__)
  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < M; i += 4) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const float* r0 = W + static_cast<std::ptrdiff_t>(i + 0) * N;
    const float* r1 = W + static_cast<std::ptrdiff_t>(i + 1) * N;
    const float* r2 = W + static_cast<std::ptrdiff_t>(i + 2) * N;
    const float* r3 = W + static_cast<std::ptrdiff_t>(i + 3) * N;

    for (int j = 0; j < N; j += 8) {
      prefetch_r(r0 + j + 160);
      prefetch_r(r1 + j + 160);
      const __m256 xv = _mm256_loadu_ps(x + j);
#if defined(__FMA__)
      acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + j), xv, acc0);
      acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j), xv, acc1);
      acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j), xv, acc2);
      acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + j), xv, acc3);
#else
      acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(r0 + j), xv));
      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(r1 + j), xv));
      acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(r2 + j), xv));
      acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(r3 + j), xv));
#endif
    }
    y[i + 0] = hsum256(acc0);
    y[i + 1] = hsum256(acc1);
    y[i + 2] = hsum256(acc2);
    y[i + 3] = hsum256(acc3);
  }
#else
  #pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < M; i += 4) {
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const float* r0 = W + static_cast<std::ptrdiff_t>(i + 0) * N;
    const float* r1 = W + static_cast<std::ptrdiff_t>(i + 1) * N;
    const float* r2 = W + static_cast<std::ptrdiff_t>(i + 2) * N;
    const float* r3 = W + static_cast<std::ptrdiff_t>(i + 3) * N;
    for (int j = 0; j < N; ++j) {
      acc0 += r0[j] * x[j];
      acc1 += r1[j] * x[j];
      acc2 += r2[j] * x[j];
      acc3 += r3[j] * x[j];
    }
    y[i + 0] = acc0;
    y[i + 1] = acc1;
    y[i + 2] = acc2;
    y[i + 3] = acc3;
  }
#endif
}

}  // namespace

// ============================================================
// CpuLlamaEngine Ã¢â‚¬â€ public API
// ============================================================

// FP32-weight GEMV (used for dequantised INT8 MLP weights).
// Pads M and N to multiples of 4/8 as needed, mirroring gemv_fp16.
void CpuLlamaEngine::gemv_fp32(const float* W, const float* x, float* y,
                                int M, int N) {
  const int M4 = (M + 3) & ~3;
  const int N8 = (N + 7) & ~7;

  if (M4 == M && N8 == N) {
    gemv_fp32_impl(W, x, y, M, N);
    return;
  }
  thread_local std::vector<float> tmp_y;
  thread_local std::vector<float> tmp_x;
  tmp_y.assign(static_cast<std::size_t>(M4), 0.f);
  tmp_x.assign(static_cast<std::size_t>(N8), 0.f);
  std::copy(x, x + N, tmp_x.begin());

  if (M4 != M) {
    thread_local std::vector<float> tmp_w;
    tmp_w.assign(static_cast<std::size_t>(M4) * static_cast<std::size_t>(N8), 0.f);
    for (int i = 0; i < M; ++i) {
      std::copy(W + static_cast<std::ptrdiff_t>(i) * N,
                W + static_cast<std::ptrdiff_t>(i) * N + N,
                tmp_w.begin() + static_cast<std::ptrdiff_t>(i) * N8);
    }
    gemv_fp32_impl(tmp_w.data(), tmp_x.data(), tmp_y.data(), M4, N8);
  } else {
    gemv_fp32_impl(W, tmp_x.data(), tmp_y.data(), M, N8);
  }
  std::copy(tmp_y.begin(), tmp_y.begin() + M, y);
}

void CpuLlamaEngine::gemv_fp16(const uint16_t* W, const float* x, float* y,
                                int M, int N) {
  // Pad M and N to multiples of 4 and 8 respectively, so the blocked inner
  // loop never overreads.  We use a temporary output buffer when M is not
  // already aligned, copying only the valid prefix back to y.
  const int M4 = (M + 3) & ~3;
  const int N8 = (N + 7) & ~7;

  if (M4 == M && N8 == N) {
    // Common case: dimensions are already aligned.
    gemv_fp16_impl(W, x, y, M, N);
    return;
  }

  // Need temporary storage for the padded output rows.
  // Extra rows will be computed but discarded.
  thread_local std::vector<float> tmp_y;
  thread_local std::vector<float> tmp_x;
  tmp_y.assign(static_cast<std::size_t>(M4), 0.f);
  tmp_x.assign(static_cast<std::size_t>(N8), 0.f);
  std::copy(x, x + N, tmp_x.begin());

  // If M is not a multiple of 4 we cannot use W directly (the next rows
  // would be out of bounds).  Use a padded weight copy for the tail.
  if (M4 != M) {
    thread_local std::vector<uint16_t> tmp_w;
    tmp_w.assign(static_cast<std::size_t>(M4) * static_cast<std::size_t>(N8),
                 0u);
    for (int i = 0; i < M; ++i) {
      std::copy(W + static_cast<std::ptrdiff_t>(i) * N,
                W + static_cast<std::ptrdiff_t>(i) * N + N,
                tmp_w.begin() + static_cast<std::ptrdiff_t>(i) * N8);
    }
    gemv_fp16_impl(tmp_w.data(), tmp_x.data(), tmp_y.data(), M4, N8);
  } else {
    gemv_fp16_impl(W, tmp_x.data(), tmp_y.data(), M, N8);
  }
  std::copy(tmp_y.begin(), tmp_y.begin() + M, y);
}

void CpuLlamaEngine::normalize(const float* x,
                               const uint16_t* w,
                               const uint16_t* b,
                               float* out,
                               int n) {
  const float eps = cfg_.norm_eps > 0.0f ? cfg_.norm_eps : 1e-5f;
  if (cfg_.use_layernorm) {
    float sum = 0.0f;
    float sq = 0.0f;
    for (int i = 0; i < n; ++i) {
      sum += x[i];
      sq += x[i] * x[i];
    }
    const float mean = sum / static_cast<float>(n);
    const float var = std::max(0.0f, sq / static_cast<float>(n) - mean * mean);
    const float inv = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) {
      const float ww = fp16_to_fp32(w[i]);
      const float bb = b ? fp16_to_fp32(b[i]) : 0.0f;
      out[i] = (x[i] - mean) * inv * ww + bb;
    }
    return;
  }

  float ss = 0.f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float rms = 1.f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    const float bb = b ? fp16_to_fp32(b[i]) : 0.0f;
    out[i] = rms * x[i] * fp16_to_fp32(w[i]) + bb;
  }
}

// Rotary position embeddings (in-place) for q and k.
// Uses precomputed cos/sin tables built in initialize().
// For GQA, q has n_heads heads and k has n_kv_heads heads.
void CpuLlamaEngine::rope(float* q, float* k, int pos, int n_heads,
                           int n_kv_heads, int head_dim) {
  const float* cos_row = rope_cos_.data() + static_cast<std::ptrdiff_t>(pos) * (head_dim / 2);
  const float* sin_row = rope_sin_.data() + static_cast<std::ptrdiff_t>(pos) * (head_dim / 2);

  auto rotate_head = [&](float* h) {
    for (int d = 0; d < head_dim / 2; ++d) {
      const float x0 = h[d];
      const float x1 = h[d + head_dim / 2];
      h[d]              = x0 * cos_row[d] - x1 * sin_row[d];
      h[d + head_dim / 2] = x0 * sin_row[d] + x1 * cos_row[d];
    }
  };

  for (int h = 0; h < n_heads;    ++h) rotate_head(q + h * head_dim);
  for (int h = 0; h < n_kv_heads; ++h) rotate_head(k + h * head_dim);
}

// Multi-head attention (GQA-aware).
// Reads current Q/K/V from q_/k_/v_; writes result to att_.
// Stores the new K and V into k_cache_ / v_cache_ at position pos.
void CpuLlamaEngine::attention(int pos, int layer) {
  const int H       = cfg_.num_heads;
  const int H_kv    = cfg_.num_kv_heads;
  const int kv_mul  = H / H_kv;   // Q heads per KV head (GQA ratio)
  const int max_ctx = options_.max_context;
  const float scale = 1.f / std::sqrt(static_cast<float>(head_dim_));

  // Write current K and V into the cache at position pos.
  const std::ptrdiff_t kv_cache_stride =
      static_cast<std::ptrdiff_t>(max_ctx) * kv_dim_;
  float* kc = k_cache_.data() + static_cast<std::ptrdiff_t>(layer) * kv_cache_stride
              + static_cast<std::ptrdiff_t>(pos) * kv_dim_;
  float* vc = v_cache_.data() + static_cast<std::ptrdiff_t>(layer) * kv_cache_stride
              + static_cast<std::ptrdiff_t>(pos) * kv_dim_;
  std::copy(k_.begin(), k_.end(), kc);
  std::copy(v_.begin(), v_.end(), vc);

  const int full_seq_len = pos + 1;
  const int window = cfg_.sliding_window > 0 ? cfg_.sliding_window : 0;
  const int attn_seq_len = (window > 0) ? std::min(window, full_seq_len) : full_seq_len;
  const int attn_start = full_seq_len - attn_seq_len;

  // Compute attention for each query head in parallel.
  #pragma omp parallel for schedule(dynamic, 1)
  for (int h = 0; h < H; ++h) {
    const int h_kv = h / kv_mul;
    const float* q_h = q_.data() + static_cast<std::ptrdiff_t>(h) * head_dim_;

    // 1. Compute raw attention scores QK^T / sqrt(d).
    float* sc = scores_.data() + static_cast<std::ptrdiff_t>(h) * max_ctx;

    for (int t = 0; t < attn_seq_len; ++t) {
      const int kv_t = attn_start + t;
      const float* k_t = k_cache_.data()
                         + static_cast<std::ptrdiff_t>(layer) * kv_cache_stride
                         + static_cast<std::ptrdiff_t>(kv_t) * kv_dim_
                         + static_cast<std::ptrdiff_t>(h_kv) * head_dim_;
      float dot = 0.f;
      for (int d = 0; d < head_dim_; ++d) {
        dot += q_h[d] * k_t[d];
      }
      sc[t] = dot * scale;
    }

    // 2. Softmax over [0..attn_seq_len).
    float max_s = sc[0];
    for (int t = 1; t < attn_seq_len; ++t) {
      if (sc[t] > max_s) max_s = sc[t];
    }
    float sum_exp = 0.f;
    for (int t = 0; t < attn_seq_len; ++t) {
      sc[t] = std::exp(sc[t] - max_s);
      sum_exp += sc[t];
    }
    const float inv_sum = 1.f / sum_exp;
    for (int t = 0; t < attn_seq_len; ++t) sc[t] *= inv_sum;

    // 3. Weighted sum of V.
    float* out_h = att_.data() + static_cast<std::ptrdiff_t>(h) * head_dim_;
    std::fill(out_h, out_h + head_dim_, 0.f);
    for (int t = 0; t < attn_seq_len; ++t) {
      const int kv_t = attn_start + t;
      const float* v_t = v_cache_.data()
                         + static_cast<std::ptrdiff_t>(layer) * kv_cache_stride
                         + static_cast<std::ptrdiff_t>(kv_t) * kv_dim_
                         + static_cast<std::ptrdiff_t>(h_kv) * head_dim_;
      const float alpha = sc[t];
      for (int d = 0; d < head_dim_; ++d) {
        out_h[d] += alpha * v_t[d];
      }
    }
  }
}

// SwiGLU MLP block:
//   ff1 = w1 * x_norm  (gate,   [inter])
//   ff3 = w3 * x_norm  (up,     [inter])
//   ff1[i] = ff1[i] * sigmoid(ff1[i]) * ff3[i]  (SiLU gate)
//   ff2 = w2 * ff1     (down,   [hidden])
//   x  += ff2          (residual)
void CpuLlamaEngine::mlp(int layer) {
  const auto& lw = layers_[static_cast<std::size_t>(layer)];
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;

  // Dispatch to FP16 or FP32 GEMV depending on how MLP weights were loaded.
  auto mlp_gemv = [&](const uint16_t* w_fp16, const float* w_fp32,
                      float* out, int M, int N) {
    if (w_fp32) gemv_fp32(w_fp32, x_norm_.data(), out, M, N);
    else        gemv_fp16(w_fp16, x_norm_.data(), out, M, N);
  };

  mlp_gemv(lw.w1_fp16, lw.w1_fp32, ff1_.data(), I, H);
  mlp_gemv(lw.w3_fp16, lw.w3_fp32, ff3_.data(), I, H);

  // SiLU activation fused with gating: ff1[i] = SiLU(ff1[i]) * ff3[i]
  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  for (int i = 0; i < I; ++i) {
    const float g = ff1_[i];
    ff1_[i] = (g / (1.f + std::exp(-g))) * ff3_[i];
  }

  if (lw.w2_fp32) gemv_fp32(lw.w2_fp32, ff1_.data(), ff2_.data(), H, I);
  else            gemv_fp16(lw.w2_fp16, ff1_.data(), ff2_.data(), H, I);

  for (int i = 0; i < H; ++i) {
    x_[i] += ff2_[i];
  }
}

void CpuLlamaEngine::mlp_moe(int layer) {
  const auto& lw = layers_[static_cast<std::size_t>(layer)];
  const int H = cfg_.hidden_size;
  const int I = cfg_.effective_expert_intermediate_size() > 0
      ? cfg_.effective_expert_intermediate_size()
      : cfg_.intermediate_size;
  const int top_k = std::max(1, std::min(cfg_.num_experts_per_tok > 0 ? cfg_.num_experts_per_tok : 2,
                                         cfg_.num_local_experts));

  auto gemv_any = [&](const uint16_t* w_fp16, const float* w_fp32, float* out, int M, int N, const float* in) {
    if (w_fp32) {
      gemv_fp32(w_fp32, in, out, M, N);
    } else {
      gemv_fp16(w_fp16, in, out, M, N);
    }
  };

  gemv_any(lw.router_fp16, lw.router_fp32, moe_router_logits_.data(), cfg_.num_local_experts, H, x_norm_.data());

  float max_logit = moe_router_logits_[0];
  for (int i = 1; i < cfg_.num_local_experts; ++i) {
    max_logit = std::max(max_logit, moe_router_logits_[static_cast<std::size_t>(i)]);
  }
  float sum = 0.0f;
  for (int i = 0; i < cfg_.num_local_experts; ++i) {
    const float v = std::exp(moe_router_logits_[static_cast<std::size_t>(i)] - max_logit);
    moe_router_logits_[static_cast<std::size_t>(i)] = v;
    sum += v;
  }
  const float inv_sum = 1.0f / std::max(sum, 1e-8f);
  for (int i = 0; i < cfg_.num_local_experts; ++i) {
    moe_router_logits_[static_cast<std::size_t>(i)] *= inv_sum;
  }

  std::vector<int> picked(static_cast<std::size_t>(top_k), 0);
  std::vector<float> picked_p(static_cast<std::size_t>(top_k), 0.0f);
  std::vector<char> used(static_cast<std::size_t>(cfg_.num_local_experts), 0);
  for (int k = 0; k < top_k; ++k) {
    int best = -1;
    float best_p = -1.0f;
    for (int e = 0; e < cfg_.num_local_experts; ++e) {
      if (used[static_cast<std::size_t>(e)]) continue;
      const float p = moe_router_logits_[static_cast<std::size_t>(e)];
      if (p > best_p) {
        best = e;
        best_p = p;
      }
    }
    picked[static_cast<std::size_t>(k)] = std::max(0, best);
    picked_p[static_cast<std::size_t>(k)] = std::max(0.0f, best_p);
    if (best >= 0) used[static_cast<std::size_t>(best)] = 1;
  }
  float picked_sum = 0.0f;
  for (float p : picked_p) picked_sum += p;
  const float inv_picked = 1.0f / std::max(picked_sum, 1e-8f);
  for (float& p : picked_p) p *= inv_picked;

  std::fill(moe_accum_.begin(), moe_accum_.end(), 0.0f);
  for (int k = 0; k < top_k; ++k) {
    const int e = picked[static_cast<std::size_t>(k)];
    const float weight = picked_p[static_cast<std::size_t>(k)];
    if (weight <= 0.0f) continue;

    gemv_any(lw.expert_w1_fp16[static_cast<std::size_t>(e)],
             lw.expert_w1_fp32[static_cast<std::size_t>(e)],
             ff1_.data(), I, H, x_norm_.data());
    gemv_any(lw.expert_w3_fp16[static_cast<std::size_t>(e)],
             lw.expert_w3_fp32[static_cast<std::size_t>(e)],
             ff3_.data(), I, H, x_norm_.data());
    for (int i = 0; i < I; ++i) {
      const float g = ff1_[i];
      ff1_[i] = (g / (1.0f + std::exp(-g))) * ff3_[i];
    }
    gemv_any(lw.expert_w2_fp16[static_cast<std::size_t>(e)],
             lw.expert_w2_fp32[static_cast<std::size_t>(e)],
             ff2_.data(), H, I, ff1_.data());
    for (int i = 0; i < H; ++i) {
      moe_accum_[static_cast<std::size_t>(i)] += weight * ff2_[i];
    }
  }

  for (int i = 0; i < H; ++i) {
    x_[i] += moe_accum_[static_cast<std::size_t>(i)];
  }
}

// Full single-token forward pass.
// After this call x_ contains the updated residual and logits_ contains
// the unnormalised next-token logit vector.
void CpuLlamaEngine::forward_token(int token, int pos) {
  const int H    = cfg_.hidden_size;
  const int V    = cfg_.vocab_size;
  const int NL   = cfg_.num_layers;
  const int NH   = cfg_.num_heads;
  const int NKV  = cfg_.num_kv_heads;

  // 1. Token embedding lookup: x = embed[token]
  const uint16_t* emb_row =
      tok_embeddings_ + static_cast<std::ptrdiff_t>(token) * H;
  for (int i = 0; i < H; ++i) {
    x_[i] = fp16_to_fp32(emb_row[i]);
  }

  // 2. Transformer layers
  for (int l = 0; l < NL; ++l) {
    const auto& lw = layers_[static_cast<std::size_t>(l)];

    // --- Attention sub-block ---
    normalize(x_.data(), lw.norm_att, lw.norm_att_bias, x_norm_.data(), H);

    gemv_fp16(lw.wq, x_norm_.data(), q_.data(), q_dim_, H);
    gemv_fp16(lw.wk, x_norm_.data(), k_.data(), kv_dim_, H);
    gemv_fp16(lw.wv, x_norm_.data(), v_.data(), kv_dim_, H);

    rope(q_.data(), k_.data(), pos, NH, NKV, head_dim_);
    attention(pos, l);

    gemv_fp16(lw.wo, att_.data(), ff2_.data(), H, q_dim_);
    if (lw.bo) {
      for (int i = 0; i < H; ++i) {
        ff2_[i] += fp16_to_fp32(lw.bo[i]);
      }
    }

    for (int i = 0; i < H; ++i) {
      x_[i] += ff2_[i];
    }

    // --- FFN sub-block ---
    normalize(x_.data(), lw.norm_ffn, lw.norm_ffn_bias, x_norm_.data(), H);
    if (cfg_.is_moe()) {
      mlp_moe(l);
    } else {
      mlp(l);
    }
  }

  // 3. Final RMSNorm + LM head
  normalize(x_.data(), norm_out_, norm_out_bias_, x_norm_.data(), H);
  gemv_fp16(lm_head_, x_norm_.data(), logits_.data(), V, H);
  if (lm_head_bias_) {
    for (int i = 0; i < V; ++i) {
      logits_[static_cast<std::size_t>(i)] += fp16_to_fp32(lm_head_bias_[static_cast<std::size_t>(i)]);
    }
  }
}

// Temperature + top-k sampling with repetition penalty.
int CpuLlamaEngine::sample_token(float temperature, int top_k,
                                  const std::vector<int>& history,
                                  float rep_penalty) {
  const int V = cfg_.vocab_size;
  float* logits = logits_.data();

  // Repetition penalty: downscale logits of recently generated tokens.
  if (rep_penalty != 1.f && !history.empty()) {
    for (int id : history) {
      if (id >= 0 && id < V) {
        logits[id] = (logits[id] > 0.f) ? logits[id] / rep_penalty
                                         : logits[id] * rep_penalty;
      }
    }
  }

  // Greedy argmax when temperature is effectively zero.
  if (temperature < 1e-6f) {
    return static_cast<int>(
        std::max_element(logits, logits + V) - logits);
  }

  // Apply temperature.
  for (int i = 0; i < V; ++i) logits[i] /= temperature;

  // Top-k filtering: zero out all but the top_k largest logits.
  if (top_k > 0 && top_k < V) {
    thread_local std::vector<float> scratch;
    scratch.assign(logits, logits + V);
    std::nth_element(scratch.begin(), scratch.begin() + top_k,
                     scratch.end(), std::greater<float>{});
    const float cutoff = scratch[static_cast<std::size_t>(top_k - 1)];
    for (int i = 0; i < V; ++i) {
      if (logits[i] < cutoff) logits[i] = -1e38f;
    }
  }

  // Softmax.
  const float mx = *std::max_element(logits, logits + V);
  double sum = 0.0;
  for (int i = 0; i < V; ++i) {
    logits[i] = std::exp(logits[i] - mx);
    sum += logits[i];
  }
  for (int i = 0; i < V; ++i) logits[i] /= static_cast<float>(sum);

  // Multinomial sample.
  thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  float r = dist(rng);
  float cdf = 0.f;
  for (int i = 0; i < V; ++i) {
    cdf += logits[i];
    if (r <= cdf) return i;
  }
  return V - 1;
}

// ============================================================
// initialize
// ============================================================
void CpuLlamaEngine::initialize(const EngineOptions& options) {
  options_ = options;
  weights_.open(options.model_path);
  cfg_ = weights_.config();

  const int H    = cfg_.hidden_size;
  const int I    = cfg_.intermediate_size;
  const int expert_inter = cfg_.effective_expert_intermediate_size() > 0
      ? cfg_.effective_expert_intermediate_size()
      : I;
  const int ffn_inter = std::max(I, expert_inter);
  const int V    = cfg_.vocab_size;
  const int NL   = cfg_.num_layers;
  const int NH   = cfg_.num_heads;
  const int NKV  = cfg_.num_kv_heads;
  const std::string wq_name = "layers.0.attention.wq";
  const std::string wk_name = "layers.0.attention.wk";
  if (!weights_.has_tensor(wq_name) || !weights_.has_tensor(wk_name)) {
    LLAMA_ENGINE_THROW("missing attention projection tensors for CPU init");
  }
  const std::size_t row_bytes = static_cast<std::size_t>(H) * sizeof(uint16_t);
  const std::size_t wq_bytes = weights_.tensor_bytes(wq_name);
  const std::size_t wk_bytes = weights_.tensor_bytes(wk_name);
  if (row_bytes == 0 || (wq_bytes % row_bytes) != 0 || (wk_bytes % row_bytes) != 0) {
    LLAMA_ENGINE_THROW("invalid attention projection tensor shape for CPU init");
  }
  q_dim_ = static_cast<int>(wq_bytes / row_bytes);
  if (q_dim_ <= 0 || (q_dim_ % NH) != 0) {
    LLAMA_ENGINE_THROW("invalid q_proj shape for CPU init");
  }
  head_dim_ = q_dim_ / NH;
  kv_dim_   = NKV * head_dim_;
  if (static_cast<int>(wk_bytes / row_bytes) != kv_dim_) {
    LLAMA_ENGINE_THROW("invalid k_proj shape for CPU init");
  }

  if (options_.verbose) {
    std::cout << "[cpu] Initializing CPU engine\n"
              << "[cpu] hidden=" << H
              << " inter=" << I
              << " layers=" << NL
              << " heads=" << NH
              << " kv_heads=" << NKV
              << " vocab=" << V
              << " q_dim=" << q_dim_
              << " head_dim=" << head_dim_
              << " kv_dim=" << kv_dim_;
    if (cfg_.use_layernorm) {
      std::cout << " norm=layernorm";
    }
    if (cfg_.is_moe()) {
      std::cout << " moe_experts=" << cfg_.num_local_experts
                << " moe_topk=" << (cfg_.num_experts_per_tok > 0 ? cfg_.num_experts_per_tok : 2)
                << " expert_inter=" << expert_inter;
    }
    std::cout << "\n";
#if defined(__AVX2__) && defined(CPU_ENGINE_HAVE_F16C)
    std::cout << "[cpu] SIMD: AVX2 + F16C (8-wide FP32, on-the-fly FP16 conversion)\n";
#elif defined(__AVX__)
    std::cout << "[cpu] SIMD: AVX (8-wide FP32, software FP16 conversion)\n";
#else
    std::cout << "[cpu] SIMD: scalar fallback\n";
#endif
#if defined(_OPENMP)
    std::cout << "[cpu] OpenMP threads: " << omp_get_max_threads() << "\n";
#endif
  }

  // --- Bind static weight pointers ---
  auto ptr16 = [&](const std::string& name) -> const uint16_t* {
    return reinterpret_cast<const uint16_t*>(weights_.tensor_data(name));
  };

  tok_embeddings_ = ptr16("tok_embeddings.weight");
  norm_out_       = ptr16("norm.weight");
  norm_out_bias_  = weights_.has_tensor("norm.bias") ? ptr16("norm.bias") : nullptr;
  lm_head_ = weights_.has_tensor("output.weight")
                 ? ptr16("output.weight")
                 : tok_embeddings_;  // tied weights
  lm_head_bias_ = weights_.has_tensor("output.bias") ? ptr16("output.bias") : nullptr;

  // Helper: dequantise INT8 row-major weight [M Ãƒâ€” N] with per-row or global
  // FP32 scales into a pre-allocated FP32 buffer.
  // If scale_count == 1 the single value is a global scale for the whole
  // tensor; otherwise scale_count == M and each row has its own scale.
  auto dequant_int8 = [](const int8_t* src, const float* scales,
                          std::size_t scale_count,
                          int M, int N, float* dst) {
    const bool rowwise = (scale_count > 1);
    for (int i = 0; i < M; ++i) {
      const float s = rowwise ? scales[i] : scales[0];
      const int8_t* row = src + static_cast<std::ptrdiff_t>(i) * N;
      float* drow = dst + static_cast<std::ptrdiff_t>(i) * N;
      for (int j = 0; j < N; ++j) {
        drow[j] = static_cast<float>(row[j]) * s;
      }
    }
  };
  auto dequant_int4 = [](const int8_t* src, const float* scales,
                         std::size_t scale_count,
                         int M, int N, float* dst) {
    const bool rowwise = (scale_count > 1);
    const int packed_cols = (N + 1) / 2;
    for (int i = 0; i < M; ++i) {
      const float s = rowwise ? scales[i] : scales[0];
      const int8_t* row = src + static_cast<std::ptrdiff_t>(i) * packed_cols;
      float* drow = dst + static_cast<std::ptrdiff_t>(i) * N;
      for (int j = 0; j < N; ++j) {
        const uint8_t b = static_cast<uint8_t>(row[j / 2]);
        const uint8_t nib = (j & 1) == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
        const int8_t q = (nib >= 8) ? static_cast<int8_t>(nib) - 16 : static_cast<int8_t>(nib);
        drow[j] = static_cast<float>(q) * s;
      }
    }
  };

  // Returns the FP16 mmap pointer if the tensor is FP16, or fills fp32_buf
  // and returns null if stored as packed INT8/INT4.
  auto load_weight = [&](const std::string& name, int out_dim, int in_dim,
                         std::vector<float>& fp32_buf) -> const uint16_t* {
    if (weights_.has_tensor(name)) {
      return ptr16(name);  // FP16 path, zero-copy
    }
    const std::string i8name = name + ".int8";
    const std::string i4name = name + ".int4";
    const std::string scname = name + ".scale";
    if (!weights_.has_tensor(scname) ||
        (!weights_.has_tensor(i8name) && !weights_.has_tensor(i4name))) {
      LLAMA_ENGINE_THROW("tensor not found: " + name +
                         " (and no INT8/INT4 alternative)");
    }
    const auto* scales = reinterpret_cast<const float*>(weights_.tensor_data(scname));
    const std::size_t scale_count = weights_.tensor_bytes(scname) / sizeof(float);

    fp32_buf.resize(static_cast<std::size_t>(out_dim) *
                    static_cast<std::size_t>(in_dim));
    if (weights_.has_tensor(i8name)) {
      const auto* src = reinterpret_cast<const int8_t*>(weights_.tensor_data(i8name));
      dequant_int8(src, scales, scale_count, out_dim, in_dim, fp32_buf.data());
    } else {
      const auto* src = reinterpret_cast<const int8_t*>(weights_.tensor_data(i4name));
      dequant_int4(src, scales, scale_count, out_dim, in_dim, fp32_buf.data());
    }
    return nullptr;
  };

  // --- Bind per-layer weight pointers ---
  layers_.resize(static_cast<std::size_t>(NL));
  // Each dense layer needs up to 3 dequant buffers (w1, w2, w3).
  dequant_mlp_.resize(static_cast<std::size_t>(NL) * 3);
  dequant_moe_.clear();

  for (int l = 0; l < NL; ++l) {
    const std::string p = "layers." + std::to_string(l);
    auto& lw = layers_[static_cast<std::size_t>(l)];
    lw.norm_att = ptr16(p + ".attention_norm.weight");
    lw.norm_ffn = ptr16(p + ".ffn_norm.weight");
    lw.norm_att_bias = weights_.has_tensor(p + ".attention_norm.bias")
        ? ptr16(p + ".attention_norm.bias")
        : nullptr;
    lw.norm_ffn_bias = weights_.has_tensor(p + ".ffn_norm.bias")
        ? ptr16(p + ".ffn_norm.bias")
        : nullptr;
    lw.wq       = ptr16(p + ".attention.wq");
    lw.wk       = ptr16(p + ".attention.wk");
    lw.wv       = ptr16(p + ".attention.wv");
    lw.wo       = ptr16(p + ".attention.wo");
    lw.bo       = weights_.has_tensor(p + ".attention.bo")
        ? ptr16(p + ".attention.bo")
        : nullptr;

    auto& buf_w1 = dequant_mlp_[static_cast<std::size_t>(l) * 3 + 0];
    auto& buf_w2 = dequant_mlp_[static_cast<std::size_t>(l) * 3 + 1];
    auto& buf_w3 = dequant_mlp_[static_cast<std::size_t>(l) * 3 + 2];

    if (!cfg_.is_moe()) {
      lw.w1_fp16 = load_weight(p + ".feed_forward.w1", I, H, buf_w1);
      lw.w2_fp16 = load_weight(p + ".feed_forward.w2", H, I, buf_w2);
      lw.w3_fp16 = load_weight(p + ".feed_forward.w3", I, H, buf_w3);

      if (!buf_w1.empty()) lw.w1_fp32 = buf_w1.data();
      if (!buf_w2.empty()) lw.w2_fp32 = buf_w2.data();
      if (!buf_w3.empty()) lw.w3_fp32 = buf_w3.data();
    } else {
      std::vector<float> router_buf;
      lw.router_fp16 = load_weight(p + ".feed_forward.router", cfg_.num_local_experts, H, router_buf);
      if (!router_buf.empty()) {
        dequant_moe_.push_back(std::move(router_buf));
        lw.router_fp32 = dequant_moe_.back().data();
      }

      const int experts = std::max(0, cfg_.num_local_experts);
      lw.expert_w1_fp16.resize(static_cast<std::size_t>(experts), nullptr);
      lw.expert_w2_fp16.resize(static_cast<std::size_t>(experts), nullptr);
      lw.expert_w3_fp16.resize(static_cast<std::size_t>(experts), nullptr);
      lw.expert_w1_fp32.resize(static_cast<std::size_t>(experts), nullptr);
      lw.expert_w2_fp32.resize(static_cast<std::size_t>(experts), nullptr);
      lw.expert_w3_fp32.resize(static_cast<std::size_t>(experts), nullptr);

      for (int e = 0; e < experts; ++e) {
        const std::string eb = p + ".feed_forward.experts." + std::to_string(e);
        std::vector<float> ew1;
        std::vector<float> ew2;
        std::vector<float> ew3;
        lw.expert_w1_fp16[static_cast<std::size_t>(e)] = load_weight(eb + ".w1", expert_inter, H, ew1);
        lw.expert_w2_fp16[static_cast<std::size_t>(e)] = load_weight(eb + ".w2", H, expert_inter, ew2);
        lw.expert_w3_fp16[static_cast<std::size_t>(e)] = load_weight(eb + ".w3", expert_inter, H, ew3);
        if (!ew1.empty()) {
          dequant_moe_.push_back(std::move(ew1));
          lw.expert_w1_fp32[static_cast<std::size_t>(e)] = dequant_moe_.back().data();
        }
        if (!ew2.empty()) {
          dequant_moe_.push_back(std::move(ew2));
          lw.expert_w2_fp32[static_cast<std::size_t>(e)] = dequant_moe_.back().data();
        }
        if (!ew3.empty()) {
          dequant_moe_.push_back(std::move(ew3));
          lw.expert_w3_fp32[static_cast<std::size_t>(e)] = dequant_moe_.back().data();
        }
      }
    }
  }

  // --- Allocate scratch buffers ---
  const int max_ctx = options_.max_context;
  x_.assign(static_cast<std::size_t>(H),      0.f);
  x_norm_.assign(static_cast<std::size_t>(H), 0.f);
  q_.assign(static_cast<std::size_t>(q_dim_), 0.f);
  k_.assign(static_cast<std::size_t>(NKV * head_dim_), 0.f);
  v_.assign(static_cast<std::size_t>(NKV * head_dim_), 0.f);
  att_.assign(static_cast<std::size_t>(q_dim_), 0.f);
  ff1_.assign(static_cast<std::size_t>(ffn_inter), 0.f);
  ff2_.assign(static_cast<std::size_t>(H), 0.f);
  ff3_.assign(static_cast<std::size_t>(ffn_inter), 0.f);
  logits_.assign(static_cast<std::size_t>(V), 0.f);
  moe_router_logits_.assign(static_cast<std::size_t>(std::max(1, cfg_.num_local_experts)), 0.f);
  moe_accum_.assign(static_cast<std::size_t>(H), 0.f);
  // scores_: per head Ãƒâ€” max_context (used inside attention())
  scores_.assign(static_cast<std::size_t>(NH) *
                 static_cast<std::size_t>(max_ctx), 0.f);

  // --- KV cache ---
  const std::size_t kv_cache_elems =
      static_cast<std::size_t>(NL) *
      static_cast<std::size_t>(max_ctx) *
      static_cast<std::size_t>(kv_dim_);
  k_cache_.assign(kv_cache_elems, 0.f);
  v_cache_.assign(kv_cache_elems, 0.f);

  if (options_.verbose) {
    const double kv_mb = static_cast<double>(kv_cache_elems * 2 * sizeof(float)) / (1024.0 * 1024.0);
    std::cout << "[cpu] KV cache: " << std::fixed
              << static_cast<int>(kv_mb) << " MB (FP32, "
              << max_ctx << " tokens)\n";
  }

  // --- Precompute RoPE tables ---
  // cos[pos][d] = cos(pos / 10000^(2d/head_dim))
  // sin[pos][d] = sin(pos / 10000^(2d/head_dim))
  const int half_hd = head_dim_ / 2;
  rope_cos_.resize(static_cast<std::size_t>(max_ctx) *
                   static_cast<std::size_t>(half_hd));
  rope_sin_.resize(static_cast<std::size_t>(max_ctx) *
                   static_cast<std::size_t>(half_hd));
  for (int p = 0; p < max_ctx; ++p) {
    for (int d = 0; d < half_hd; ++d) {
      const float freq =
          1.f / std::pow(10000.f, static_cast<float>(2 * d) /
                                      static_cast<float>(head_dim_));
      rope_cos_[static_cast<std::size_t>(p) * half_hd + d] =
          std::cos(static_cast<float>(p) * freq);
      rope_sin_[static_cast<std::size_t>(p) * half_hd + d] =
          std::sin(static_cast<float>(p) * freq);
    }
  }

  if (options_.verbose) {
    std::cout << "[cpu] Ready.\n";
  }
}

// ============================================================
// generate
// ============================================================
std::vector<int> CpuLlamaEngine::generate(
    const std::vector<int>& prompt_tokens, int max_new_tokens,
    float temperature) {
  return generate_stream(prompt_tokens, max_new_tokens, temperature,
                         [](int) { return true; });
}

// ============================================================
// generate_stream
// ============================================================
std::vector<int> CpuLlamaEngine::generate_stream(
    const std::vector<int>& prompt_tokens, int max_new_tokens,
    float temperature, const std::function<bool(int)>& on_token) {
  const int eos_id = 2;  // Llama2 EOS token
  const int max_ctx = options_.max_context;
  const int top_k   = options_.top_k;
  const float rep_p = options_.repetition_penalty;

  last_benchmark_stats_ = BenchmarkStats{};
  last_benchmark_stats_.prompt_tokens =
      static_cast<int>(prompt_tokens.size());

  std::vector<int> output = prompt_tokens;
  std::vector<int> history;

  // --- Prefill ---
  // Process every prompt token to fill the KV cache.  The last forward_token
  // call leaves logits_ primed to predict the first new token Ã¢â‚¬â€ no extra
  // forward pass is needed before the first sample.
  const auto prefill_start = std::chrono::steady_clock::now();
  int pos = 0;
  if (prompt_tokens.empty()) {
    // Seed with BOS so logits_ is initialised before the first sample.
    forward_token(1 /*BOS*/, 0);
    pos = 1;
  } else {
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()); ++i) {
      if (i >= max_ctx) break;
      forward_token(prompt_tokens[static_cast<std::size_t>(i)], i);
      pos = i + 1;
    }
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  last_benchmark_stats_.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();

  // --- Decode ---
  // logits_ is already set by the last prefill forward_token call.
  // Correct autoregressive loop: sample from current logits_, output the
  // token, then run forward_token on it to get logits for the next step.
  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < max_new_tokens; ++step) {
    const int next = sample_token(temperature, top_k, history, rep_p);

    history.push_back(next);
    output.push_back(next);
    ++last_benchmark_stats_.generated_tokens;

    if (next == eos_id) break;
    if (!on_token(next)) break;
    if (pos >= max_ctx) break;

    // Forward on the sampled token Ã¢â€ â€™ updates logits_ for the next step.
    forward_token(next, pos++);
  }
  const auto decode_end = std::chrono::steady_clock::now();
  last_benchmark_stats_.decode_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start)
          .count();

  return output;
}

// ============================================================
// inspect_next_logits
// ============================================================
std::vector<std::pair<int, float>> CpuLlamaEngine::inspect_next_logits(
    const std::vector<int>& prompt_tokens, int top_k) {
  for (int i = 0; i < static_cast<int>(prompt_tokens.size()); ++i) {
    if (i >= options_.max_context) break;
    forward_token(prompt_tokens[static_cast<std::size_t>(i)], i);
  }

  const int V = cfg_.vocab_size;
  const int k = std::min(top_k, V);

  // Collect (id, logit) pairs and partial-sort for the top-k.
  std::vector<std::pair<int, float>> pairs;
  pairs.reserve(static_cast<std::size_t>(V));
  for (int i = 0; i < V; ++i) {
    pairs.emplace_back(i, logits_[static_cast<std::size_t>(i)]);
  }
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) {
                      return a.second > b.second;
                    });
  pairs.resize(static_cast<std::size_t>(k));
  return pairs;
}

}  // namespace engine
