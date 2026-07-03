// Bench + parity harness for the int4 weight-only decode GEMV
// (kernels::launch_weight_only_int4_matvec_dp4a). Validates the kernel against a
// CPU reference (max abs error) and reports achieved memory bandwidth for the
// real 32B / 8B MLP+projection shapes. This is the optimization loop and
// regression gate for the int4 GEMV rewrite.
//
// Build: a CMake target links this against llama_engine. Run with no args.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "runtime/kernels.cuh"

namespace {

#define CK(call)                                                                         \
  do {                                                                                   \
    cudaError_t _e = (call);                                                             \
    if (_e != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, \
                   __LINE__);                                                            \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

constexpr double kPeakGBs = 1792.0;  // RTX 5090 GDDR7 ~1.79 TB/s

int signed_int4(unsigned nibble) {
  const int v = static_cast<int>(nibble & 0x0Fu);
  return v < 8 ? v : v - 16;
}

struct Shape {
  const char* name;
  int out_features;
  int in_features;
};

// Returns max abs error vs CPU reference and the achieved GB/s.
void run_shape(const Shape& s, std::mt19937& rng) {
  const int out = s.out_features;
  const int in = s.in_features;
  const int packed_cols = (in + 1) / 2;

  // Host data: packed int4 weights, per-row fp32 scales, int8 activation, act scale.
  std::vector<std::int8_t> h_w(static_cast<std::size_t>(out) * packed_cols);
  std::vector<float> h_wscale(out);
  std::vector<std::int8_t> h_x(in);
  std::vector<int> h_x_nibblepairs;  // unused
  float h_xscale = 0.0f;

  std::uniform_int_distribution<int> nib(0, 15);
  std::uniform_int_distribution<int> act(-127, 127);
  std::uniform_real_distribution<float> sc(0.001f, 0.05f);

  for (auto& b : h_w) b = static_cast<std::int8_t>((nib(rng) & 0x0F) | ((nib(rng) & 0x0F) << 4));
  for (auto& v : h_wscale) v = sc(rng);
  for (auto& v : h_x) v = static_cast<std::int8_t>(act(rng));
  h_xscale = sc(rng);

  // CPU reference: y[row] = (sum_col signed_int4(w[row,col]) * x[col]) * wscale[row] * xscale
  std::vector<float> ref(out);
  for (int r = 0; r < out; ++r) {
    long long acc = 0;
    const std::int8_t* row = h_w.data() + static_cast<std::size_t>(r) * packed_cols;
    for (int c = 0; c < in; ++c) {
      const unsigned byte = static_cast<unsigned char>(row[c >> 1]);
      const unsigned nibble = (c & 1) ? (byte >> 4) : (byte & 0x0F);
      acc += static_cast<long long>(signed_int4(nibble)) * static_cast<long long>(h_x[c]);
    }
    ref[r] = static_cast<float>(acc) * h_wscale[r] * h_xscale;
  }

  // Device buffers.
  std::int8_t* d_w = nullptr;
  float* d_wscale = nullptr;
  std::int8_t* d_x = nullptr;
  float* d_xscale = nullptr;
  half* d_y = nullptr;
  CK(cudaMalloc(&d_w, h_w.size()));
  CK(cudaMalloc(&d_wscale, out * sizeof(float)));
  CK(cudaMalloc(&d_x, in));
  CK(cudaMalloc(&d_xscale, sizeof(float)));
  CK(cudaMalloc(&d_y, out * sizeof(half)));
  CK(cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_wscale, h_wscale.data(), out * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_x, h_x.data(), in, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_xscale, &h_xscale, sizeof(float), cudaMemcpyHostToDevice));

  // Use the engine's tuned default config (warps=8, tile=256, wpr=2).
  auto launch = [&] {
    kernels::launch_weight_only_int4_matvec_dp4a(d_w, d_wscale, d_x, d_xscale, d_y, out, in,
                                                 /*stream=*/0, /*warps=*/8, /*tile=*/256,
                                                 /*wpr=*/2);
  };

  // Parity.
  launch();
  CK(cudaDeviceSynchronize());
  std::vector<half> h_y(out);
  CK(cudaMemcpy(h_y.data(), d_y, out * sizeof(half), cudaMemcpyDeviceToHost));
  double max_abs = 0.0, max_rel = 0.0;
  for (int r = 0; r < out; ++r) {
    const double got = __half2float(h_y[r]);
    const double want = ref[r];
    const double ae = std::abs(got - want);
    max_abs = std::max(max_abs, ae);
    const double denom = std::max(1e-6, std::abs(want));
    max_rel = std::max(max_rel, ae / denom);
  }

  // Bench.
  const int warmup = 20, iters = 200;
  for (int i = 0; i < warmup; ++i) launch();
  CK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i) launch();
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms = 0.0f;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  const double per_call_ms = ms / iters;
  const double weight_bytes = static_cast<double>(out) * packed_cols;  // dominant traffic
  const double gbs = weight_bytes / (per_call_ms * 1e-3) / 1e9;

  std::printf(
      "%-22s out=%-6d in=%-6d  %.4f ms  %.1f GB/s (%.0f%% peak)  max_abs=%.4g max_rel=%.4g %s\n",
      s.name, out, in, per_call_ms, gbs, 100.0 * gbs / kPeakGBs, max_abs, max_rel,
      (max_rel < 1e-3 ? "PARITY_OK" : "PARITY_FAIL"));

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaFree(d_w);
  cudaFree(d_wscale);
  cudaFree(d_x);
  cudaFree(d_xscale);
  cudaFree(d_y);
}

}  // namespace

int main() {
  std::mt19937 rng(1234);
  const Shape shapes[] = {
      {"32B w1/w3", 27648, 5120}, {"32B w2", 5120, 27648}, {"32B wqkv", 5120 + 2 * 1024, 5120},
      {"8B w1/w3", 14336, 4096},  {"8B w2", 4096, 14336},
  };
  std::printf("int4 GEMV bench (peak %.0f GB/s)\n", kPeakGBs);
  for (const auto& s : shapes) {
    run_shape(s, rng);
  }
  return 0;
}
