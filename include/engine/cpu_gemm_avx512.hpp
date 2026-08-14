#pragma once

#include <cstdint>

namespace engine {
namespace detail {

// AVX-512 batched GEMM for the CPU prefill path (cpu_gemm_avx512.cpp, the
// only TU built with AVX-512 flags). Callers MUST check both
// gemm_fp16_avx512_compiled() (this build has the kernel) and a runtime CPUID
// probe before calling; requires M % 4 == 0 and N % 16 == 0.
bool gemm_fp16_avx512_compiled();
void gemm_fp16_avx512(const std::uint16_t* W, const float* X, float* Y, int M, int N, int B);

// There is deliberately no AVX-512 GEMV here for decode. One was written and
// measured on Zen 5 and it was not faster than the AVX2 kernel in cpu_engine.cpp
// (17.6 vs 17.8 tok/s on Llama-3.2-1B): decode reads each weight exactly once,
// so it is bound by DRAM bandwidth and not by how wide the arithmetic is.
// Prefill is different, which is why the GEMM above earns its width: it reuses
// a weight block across the token dimension.

}  // namespace detail
}  // namespace engine
