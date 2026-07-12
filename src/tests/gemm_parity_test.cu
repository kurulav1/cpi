// Correctness gate for the sequence-mode fp16 GEMM.
//
// 1. vs a CPU fp32 reference, over shapes that exercise the tile edges (dims that
//    are not multiples of 32, tokens=1, K smaller than a tile).
// 2. vs the existing GEMV row-by-row: the GEMM must agree with it to within fp16
//    tolerance. It will NOT agree bit-for-bit -- the two sum K in different orders --
//    which is exactly why the executor keeps single-token work on the GEMV path.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "runtime/kernels.cuh"

#define CK(call)                                                                       \
  do {                                                                                 \
    cudaError_t e = (call);                                                            \
    if (e != cudaSuccess) {                                                            \
      std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);      \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

namespace {

struct Shape {
  int tokens, out_features, in_features;
};

}  // namespace

int main() {
  const Shape shapes[] = {
      {1, 64, 64},      // single token: the decode shape
      {2, 32, 32},      // exactly one tile
      {17, 48, 80},     // every dim ragged
      {64, 768, 768},   // Gemma 4 E2B vision hidden
      {196, 3072, 768}, // vision MLP up-projection, a real patch count
      {33, 1152, 4304}, // 26B vision down-projection shape, ragged tokens
  };

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  int failures = 0;

  for (const Shape& s : shapes) {
    const std::size_t wn = static_cast<std::size_t>(s.out_features) * s.in_features;
    const std::size_t xn = static_cast<std::size_t>(s.tokens) * s.in_features;
    const std::size_t yn = static_cast<std::size_t>(s.tokens) * s.out_features;

    std::vector<__half> hw(wn), hx(xn);
    for (auto& v : hw) v = __float2half(dist(rng));
    for (auto& v : hx) v = __float2half(dist(rng));

    __half *dw = nullptr, *dx = nullptr, *dy = nullptr, *dy_gemv = nullptr;
    CK(cudaMalloc(&dw, wn * sizeof(__half)));
    CK(cudaMalloc(&dx, xn * sizeof(__half)));
    CK(cudaMalloc(&dy, yn * sizeof(__half)));
    CK(cudaMalloc(&dy_gemv, yn * sizeof(__half)));
    CK(cudaMemcpy(dw, hw.data(), wn * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), xn * sizeof(__half), cudaMemcpyHostToDevice));

    kernels::launch_rowmajor_half_gemm_f16(dw, dx, dy, s.out_features, s.in_features, s.tokens, 0);
    // The GEMV one token at a time -- the path decode actually takes.
    for (int t = 0; t < s.tokens; ++t) {
      kernels::launch_rowmajor_half_gemv_f16(dw, dx + static_cast<std::size_t>(t) * s.in_features,
                                             dy_gemv + static_cast<std::size_t>(t) * s.out_features,
                                             s.out_features, s.in_features, 0, 0, 0, 0);
    }
    CK(cudaDeviceSynchronize());

    std::vector<__half> hy(yn), hy_gemv(yn);
    CK(cudaMemcpy(hy.data(), dy, yn * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy_gemv.data(), dy_gemv, yn * sizeof(__half), cudaMemcpyDeviceToHost));

    double worst_ref = 0.0, worst_gemv = 0.0;
    for (int t = 0; t < s.tokens; ++t) {
      for (int m = 0; m < s.out_features; ++m) {
        double ref = 0.0;
        for (int k = 0; k < s.in_features; ++k) {
          ref += static_cast<double>(__half2float(hw[static_cast<std::size_t>(m) * s.in_features + k])) *
                 static_cast<double>(__half2float(hx[static_cast<std::size_t>(t) * s.in_features + k]));
        }
        const std::size_t i = static_cast<std::size_t>(t) * s.out_features + m;
        const double got = __half2float(hy[i]);
        const double gemv = __half2float(hy_gemv[i]);
        const double denom = std::max(1.0, std::fabs(ref));
        worst_ref = std::max(worst_ref, std::fabs(got - ref) / denom);
        worst_gemv = std::max(worst_gemv, std::fabs(got - gemv) / denom);
      }
    }

    const bool ok = worst_ref < 2e-2 && worst_gemv < 2e-2;
    if (!ok) ++failures;
    std::printf("  [%s] tokens=%-4d out=%-5d in=%-5d | vs CPU ref: %.5f | vs GEMV: %.5f\n",
                ok ? "ok" : "FAIL", s.tokens, s.out_features, s.in_features, worst_ref, worst_gemv);

    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dy_gemv);
  }

  if (failures == 0) {
    std::printf("\nPARITY OK - sequence GEMM matches the CPU reference and the GEMV\n");
    return 0;
  }
  std::printf("\nPARITY FAILED - %d shape(s) disagree\n", failures);
  return 1;
}
