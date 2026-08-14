// AVX-512 batched GEMM for the CPU engine's prefill (the wide sibling of
// gemm_fp16_impl in cpu_engine.cpp).
//
// This is the ONLY translation unit compiled with AVX-512 flags, and it must
// stay that way: the compiler may emit AVX-512 instructions for ANY code in
// this file, so nothing here can run before the caller's runtime CPUID check
// (cpu_engine.cpp: cpu_supports_avx512). Keep detection logic and anything
// else that executes unconditionally OUT of this file.
//
// Why a separate width tier at all: Zen 4 double-pumps 512-bit ops, so raw
// FMA throughput matches AVX2; the wins are the 32-register file (a 4-row x
// 4-token tile with 16 zmm accumulators + 4 weight + 4 input vectors = 24
// live registers, which AVX2's 16 ymm cannot hold without spilling) and
// half the instruction count per element (16-wide FMA and 16-wide fp16
// conversion), which relieves the frontend/port pressure that held the AVX2
// kernel to ~30% of its FMA peak.
//
// Layout contract matches gemm_fp16: Y[B][M] = X[B][N] * W^T, W row-major
// fp16, X/Y row-major fp32; each thread's 4-row weight block (4 x N fp16 =
// 32 KB at N = 4096) stays L1/L2-resident across the token tiles, so weights
// stream from DRAM once per call, not once per token.

#include "engine/cpu_gemm_avx512.hpp"

#if defined(__AVX512F__)

#include <immintrin.h>

#include <cstddef>

namespace engine {
namespace detail {

bool gemm_fp16_avx512_compiled() {
  return true;
}

void gemm_fp16_avx512(const std::uint16_t* W, const float* X, float* Y, int M, int N, int B) {
#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < M; i += 4) {
    const std::uint16_t* r0 = W + static_cast<std::ptrdiff_t>(i + 0) * N;
    const std::uint16_t* r1 = W + static_cast<std::ptrdiff_t>(i + 1) * N;
    const std::uint16_t* r2 = W + static_cast<std::ptrdiff_t>(i + 2) * N;
    const std::uint16_t* r3 = W + static_cast<std::ptrdiff_t>(i + 3) * N;
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const float* x0 = X + static_cast<std::ptrdiff_t>(b + 0) * N;
      const float* x1 = X + static_cast<std::ptrdiff_t>(b + 1) * N;
      const float* x2 = X + static_cast<std::ptrdiff_t>(b + 2) * N;
      const float* x3 = X + static_cast<std::ptrdiff_t>(b + 3) * N;
      __m512 a00 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps();
      __m512 a02 = _mm512_setzero_ps(), a03 = _mm512_setzero_ps();
      __m512 a10 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps();
      __m512 a12 = _mm512_setzero_ps(), a13 = _mm512_setzero_ps();
      __m512 a20 = _mm512_setzero_ps(), a21 = _mm512_setzero_ps();
      __m512 a22 = _mm512_setzero_ps(), a23 = _mm512_setzero_ps();
      __m512 a30 = _mm512_setzero_ps(), a31 = _mm512_setzero_ps();
      __m512 a32 = _mm512_setzero_ps(), a33 = _mm512_setzero_ps();
      for (int j = 0; j < N; j += 16) {
        const __m512 w0 = _mm512_cvtph_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + j)));
        const __m512 w1 = _mm512_cvtph_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + j)));
        const __m512 w2 = _mm512_cvtph_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + j)));
        const __m512 w3 = _mm512_cvtph_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r3 + j)));
        const __m512 v0 = _mm512_loadu_ps(x0 + j);
        const __m512 v1 = _mm512_loadu_ps(x1 + j);
        const __m512 v2 = _mm512_loadu_ps(x2 + j);
        const __m512 v3 = _mm512_loadu_ps(x3 + j);
        a00 = _mm512_fmadd_ps(w0, v0, a00);
        a01 = _mm512_fmadd_ps(w0, v1, a01);
        a02 = _mm512_fmadd_ps(w0, v2, a02);
        a03 = _mm512_fmadd_ps(w0, v3, a03);
        a10 = _mm512_fmadd_ps(w1, v0, a10);
        a11 = _mm512_fmadd_ps(w1, v1, a11);
        a12 = _mm512_fmadd_ps(w1, v2, a12);
        a13 = _mm512_fmadd_ps(w1, v3, a13);
        a20 = _mm512_fmadd_ps(w2, v0, a20);
        a21 = _mm512_fmadd_ps(w2, v1, a21);
        a22 = _mm512_fmadd_ps(w2, v2, a22);
        a23 = _mm512_fmadd_ps(w2, v3, a23);
        a30 = _mm512_fmadd_ps(w3, v0, a30);
        a31 = _mm512_fmadd_ps(w3, v1, a31);
        a32 = _mm512_fmadd_ps(w3, v2, a32);
        a33 = _mm512_fmadd_ps(w3, v3, a33);
      }
      float* y0 = Y + static_cast<std::ptrdiff_t>(b + 0) * M + i;
      float* y1 = Y + static_cast<std::ptrdiff_t>(b + 1) * M + i;
      float* y2 = Y + static_cast<std::ptrdiff_t>(b + 2) * M + i;
      float* y3 = Y + static_cast<std::ptrdiff_t>(b + 3) * M + i;
      y0[0] = _mm512_reduce_add_ps(a00);
      y1[0] = _mm512_reduce_add_ps(a01);
      y2[0] = _mm512_reduce_add_ps(a02);
      y3[0] = _mm512_reduce_add_ps(a03);
      y0[1] = _mm512_reduce_add_ps(a10);
      y1[1] = _mm512_reduce_add_ps(a11);
      y2[1] = _mm512_reduce_add_ps(a12);
      y3[1] = _mm512_reduce_add_ps(a13);
      y0[2] = _mm512_reduce_add_ps(a20);
      y1[2] = _mm512_reduce_add_ps(a21);
      y2[2] = _mm512_reduce_add_ps(a22);
      y3[2] = _mm512_reduce_add_ps(a23);
      y0[3] = _mm512_reduce_add_ps(a30);
      y1[3] = _mm512_reduce_add_ps(a31);
      y2[3] = _mm512_reduce_add_ps(a32);
      y3[3] = _mm512_reduce_add_ps(a33);
    }
    for (; b < B; ++b) {
      const float* xb = X + static_cast<std::ptrdiff_t>(b) * N;
      __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
      __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
      for (int j = 0; j < N; j += 16) {
        const __m512 xv = _mm512_loadu_ps(xb + j);
        a0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(
                                 reinterpret_cast<const __m256i*>(r0 + j))),
                             xv, a0);
        a1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(
                                 reinterpret_cast<const __m256i*>(r1 + j))),
                             xv, a1);
        a2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(
                                 reinterpret_cast<const __m256i*>(r2 + j))),
                             xv, a2);
        a3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(
                                 reinterpret_cast<const __m256i*>(r3 + j))),
                             xv, a3);
      }
      float* yb = Y + static_cast<std::ptrdiff_t>(b) * M + i;
      yb[0] = _mm512_reduce_add_ps(a0);
      yb[1] = _mm512_reduce_add_ps(a1);
      yb[2] = _mm512_reduce_add_ps(a2);
      yb[3] = _mm512_reduce_add_ps(a3);
    }
  }
}

}  // namespace detail
}  // namespace engine

#else  // !__AVX512F__: this build has no AVX-512 kernel; the dispatch never calls it.

namespace engine {
namespace detail {

bool gemm_fp16_avx512_compiled() {
  return false;
}

void gemm_fp16_avx512(const std::uint16_t*, const float*, float*, int, int, int) {}

}  // namespace detail
}  // namespace engine

#endif
