#pragma once

#include <cstdint>

namespace engine {
namespace detail {

// AVX-512 batched GEMM for the CPU prefill path (cpu_gemm_avx512.cpp -- the
// only TU built with AVX-512 flags). Callers MUST check both
// gemm_fp16_avx512_compiled() (this build has the kernel) and a runtime CPUID
// probe before calling; requires M % 4 == 0 and N % 16 == 0.
bool gemm_fp16_avx512_compiled();
void gemm_fp16_avx512(const std::uint16_t* W, const float* X, float* Y, int M, int N, int B);

}  // namespace detail
}  // namespace engine
