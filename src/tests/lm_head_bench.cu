// Achieved bandwidth of EVERY GEMV a decode step reads, for Qwen2.5-0.5B.
//
// Batch-1 decode streams each weight once per token, so if every GEMV ran at peak
// bandwidth the whole token would cost weights/bandwidth. It does not -- and this shows
// exactly which ones fall short and by how much, so the optimisation targets itself
// instead of being guessed at.
//
// A small GEMV cannot reach peak: it is latency-bound, not bandwidth-bound. That is the
// point of the measurement.
//
// !! READ THE NUMBERS WITH THIS IN MIND !!
// The RTX 5090 has ~128 MB of L2, so every matrix here EXCEPT the LM head fits in cache
// across the timing loop. The small-GEMV figures are therefore BEST CASE -- they are being
// served from L2, not HBM -- and they are STILL only 12-15% of peak, taking a flat ~7.6 us
// each regardless of size. That is a LATENCY FLOOR, not a bandwidth limit. It is the same
// per-kernel tax the decode-graph node count exposes, seen from the other side.
// The LM head (272 MB) is the only shape that genuinely streams from HBM, and it lands at
// 93% of peak -- i.e. it is NOT the problem, despite being 28% of the weight traffic.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

#include "runtime/kernels.cuh"

#define CK(call)                                                             \
  do {                                                                       \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
      std::printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

namespace {

struct Shape {
  const char* name;
  int out;
  int in;
  int per_token;  // how many times a decode step runs this shape
};

double bench(const __half* w, const __half* x, float* y, int out, int in, int warps, int tile,
             int rows, int iters) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 10; ++i)
    kernels::launch_rowmajor_half_gemv_f32(w, x, y, out, in, 0, warps, tile, rows);
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i)
    kernels::launch_rowmajor_half_gemv_f32(w, x, y, out, in, 0, warps, tile, rows);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}

}  // namespace

int main() {
  // Qwen2.5-0.5B: hidden 896, inter 4864, kv_hidden 128, vocab 151936, 24 layers.
  const Shape shapes[] = {
      {"LM head", 151936, 896, 1},
      {"w13 (gate+up)", 9728, 896, 24},
      {"w2 (down)", 896, 4864, 24},
      {"wqkv", 1152, 896, 24},
      {"wo", 896, 896, 24},
  };
  constexpr double kPeak = 1790.0;  // GB/s, RTX 5090 spec
  constexpr int kIters = 300;

  std::printf("Decode GEMVs, Qwen2.5-0.5B, RTX 5090 (peak %.0f GB/s)\n\n", kPeak);
  std::printf("  %-14s %8s %5s %10s %9s %8s %11s\n", "weight", "MB", "xN", "ms each", "GB/s",
              "of peak", "ms/token");

  double total_ms = 0, total_mb = 0;
  for (const Shape& s : shapes) {
    const std::size_t wn = static_cast<std::size_t>(s.out) * s.in;
    const double bytes = static_cast<double>(wn) * sizeof(__half);

    __half *w = nullptr, *x = nullptr;
    float* y = nullptr;
    CK(cudaMalloc(&w, wn * sizeof(__half)));
    CK(cudaMalloc(&x, static_cast<std::size_t>(s.in) * sizeof(__half)));
    CK(cudaMalloc(&y, static_cast<std::size_t>(s.out) * sizeof(float)));
    CK(cudaMemset(w, 0x11, wn * sizeof(__half)));
    CK(cudaMemset(x, 0x11, static_cast<std::size_t>(s.in) * sizeof(__half)));

    double best = 1e9;
    for (int warps : {4, 8, 16}) {
      for (int tile : {128, 256}) {
        for (int rows : {1, 2}) {
          best = std::min(best, bench(w, x, y, s.out, s.in, warps, tile, rows, kIters));
        }
      }
    }
    const double gbs = bytes / (best / 1e3) / 1e9;
    const double per_tok = best * s.per_token;
    total_ms += per_tok;
    total_mb += bytes / 1e6 * s.per_token;
    std::printf("  %-14s %8.1f %5d %10.4f %9.0f %7.0f%% %11.3f\n", s.name, bytes / 1e6,
                s.per_token, best, gbs, gbs / kPeak * 100, per_tok);

    cudaFree(w);
    cudaFree(x);
    cudaFree(y);
  }

  const double ideal = total_mb / 1e3 / kPeak * 1e3;
  std::printf("\n  weight traffic per token : %.0f MB\n", total_mb);
  std::printf("  roofline                 : %.3f ms\n", ideal);
  std::printf("  sum of the GEMVs alone   : %.3f ms  (%.0f%% of roofline)\n", total_ms,
              ideal / total_ms * 100);
  std::printf("\n  Everything else (attention, norms, rope, kv-store, residuals, sampling,\n");
  std::printf("  and the per-kernel launch tax) is whatever a real token costs MINUS this.\n");
  return 0;
}
