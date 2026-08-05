// Achieved bandwidth of every GEMV a decode step reads, plus the non-GEMV decode kernels.
//
// Batch-1 decode streams each weight once per token, so if every GEMV ran at peak bandwidth a
// token would cost weights/bandwidth. It does not, and this shows which shapes fall short and
// by how much, so the optimisation targets itself instead of being guessed at.
//
// Read the numbers with two caveats:
//
//   * Time kernels in a CUDA graph, not back-to-back on a stream. A stream launch carries
//     several microseconds of driver submission overhead, which is what you end up timing
//     it makes unrelated kernels of different sizes all look like they cost the same. Decode
//     runs inside a graph anyway. bench_graph() below does this; bench() is kept for contrast.
//
//   * A loop over one matrix measures L2, not HBM. Everything but the LM head fits in cache
//     here, so those figures are best-case. Real decode streams each weight from HBM once.
//
// Neither caveat is academic: both will happily report a starved kernel as a fast one.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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

// Peak DRAM bandwidth for the "% of roofline" column, auto-detected from the active GPU
// (memory clock x bus width). Override with CPI_PEAK_GBS=<GB/s> if the query is off.
double peak_gbs() {
  static const double v = [] {
    if (const char* e = std::getenv("CPI_PEAK_GBS")) {
      const double x = std::atof(e);
      if (x > 0.0) return x;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    int mem_clock_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    const double g = 2.0 * (mem_clock_khz * 1.0e3) * (bus_bits / 8.0) / 1.0e9;
    return g > 1.0 ? g : 1792.0;
  }();
  return v;
}

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

double bench_splitk(const __half* w, const __half* x, __half* y, int out, int in, int iters) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 10; ++i) kernels::launch_gemv_splitk_f16(w, x, y, out, in, 0);
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) kernels::launch_gemv_splitk_f16(w, x, y, out, in, 0);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / iters;
}

// Times the kernel inside a CUDA graph; which is what decode actually runs, and which
// excludes the per-launch driver overhead that stream timing folds into the result.
double bench_graph(const __half* w, const __half* x, float* y, int out, int in, int warps,
                   int tile, int rows, int reps) {
  constexpr int kChain = 50;
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaGraph_t g;
  cudaGraphExec_t ge;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < kChain; ++i)
    kernels::launch_rowmajor_half_gemv_f32(w, x, y, out, in, s, warps, tile, rows);
  cudaStreamEndCapture(s, &g);
  cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s);
  cudaStreamSynchronize(s);
  cudaEventRecord(a, s);
  for (int i = 0; i < reps; ++i) cudaGraphLaunch(ge, s);
  cudaEventRecord(b, s);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
  return ms / (reps * kChain);
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
  const double kPeak = peak_gbs();  // GB/s, queried from the active device
  constexpr int kIters = 300;

  std::printf("Decode GEMVs, Qwen2.5-0.5B (peak %.0f GB/s)\n\n", kPeak);
  std::printf("  %-14s %8s %7s %10s %8s %10s %8s %9s\n", "weight", "MB", "rows", "1warp/row",
              "of peak", "split-K", "of peak", "speedup");

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
    __half* yh = nullptr;
    CK(cudaMalloc(&yh, static_cast<std::size_t>(s.out) * sizeof(__half)));
    const double sk = bench_splitk(w, x, yh, s.out, s.in, kIters);

    // The one that counts: same kernel, timed inside a graph.
    double g = 1e9;
    for (int warps : {4, 8, 16})
      for (int tile : {128, 256})
        for (int rows : {1, 2}) g = std::min(g, bench_graph(w, x, y, s.out, s.in, warps, tile, rows, 20));

    const double gbs = bytes / (g / 1e3) / 1e9;
    const double per_tok = g * s.per_token;
    total_ms += per_tok;
    total_mb += bytes / 1e6 * s.per_token;
    std::printf("  %-14s %8.1f %7d %10.1f %10.1f %9.1f %8.0f%% %8.3f\n", s.name, bytes / 1e6, s.out,
                best * 1e3, sk * 1e3, g * 1e3, gbs / kPeak * 100, per_tok);

    cudaFree(w);
    cudaFree(x);
    cudaFree(y);
    cudaFree(yh);
  }

  // ---------------------------------------------------------------------------------
  // Non-GEMV decode kernels, at the same shapes, timed the same way.
  {
    constexpr int kHidden = 896, kHeads = 14, kKv = 2, kHeadDim = 64, kSeq = 512, kLayers = 24;
    constexpr int kChain = 50, kReps = 20;
    __half *x = nullptr, *wt = nullptr, *y = nullptr, *q = nullptr, *kc = nullptr, *vc = nullptr,
           *ao = nullptr;
    CK(cudaMalloc(&x, kHidden * sizeof(__half)));
    CK(cudaMalloc(&wt, kHidden * sizeof(__half)));
    CK(cudaMalloc(&y, kHidden * sizeof(__half)));
    CK(cudaMalloc(&q, kHeads * kHeadDim * sizeof(__half)));
    CK(cudaMalloc(&kc, static_cast<std::size_t>(kSeq) * kKv * kHeadDim * sizeof(__half)));
    CK(cudaMalloc(&vc, static_cast<std::size_t>(kSeq) * kKv * kHeadDim * sizeof(__half)));
    CK(cudaMalloc(&ao, kHeads * kHeadDim * sizeof(__half)));
    float *ct = nullptr, *st = nullptr;
    CK(cudaMalloc(&ct, kSeq * kHeadDim * sizeof(float)));
    CK(cudaMalloc(&st, kSeq * kHeadDim * sizeof(float)));

    cudaStream_t s;
    cudaStreamCreate(&s);
    auto time_graph = [&](auto&& emit) {
      cudaGraph_t g;
      cudaGraphExec_t ge;
      cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
      for (int i = 0; i < kChain; ++i) emit();
      cudaStreamEndCapture(s, &g);
      cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s);
      cudaStreamSynchronize(s);
      cudaEventRecord(a, s);
      for (int i = 0; i < kReps; ++i) cudaGraphLaunch(ge, s);
      cudaEventRecord(b, s);
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      cudaEventDestroy(a);
      cudaEventDestroy(b);
      cudaGraphExecDestroy(ge);
      cudaGraphDestroy(g);
      return static_cast<double>(ms) / (kReps * kChain) * 1e3;  // us
    };

    const double t_norm = time_graph(
        [&] { kernels::launch_rmsnorm(x, wt, y, 1, kHidden, 1e-6f, s); });
    const double t_rope = time_graph([&] {
      kernels::launch_rope_inplace_table(q, q, kHeads, kKv, kHeadDim, 100, ct, st, s);
    });
    // Attention needs the same split-K scratch the engine gives it; without it the kernel
    // silently drops to the fallback path and the timing is meaningless.
    constexpr int kChunks = 64;
    float *sm = nullptr, *sl = nullptr, *so = nullptr;
    CK(cudaMalloc(&sm, kHeads * kChunks * sizeof(float)));
    CK(cudaMalloc(&sl, kHeads * kChunks * sizeof(float)));
    CK(cudaMalloc(&so, static_cast<std::size_t>(kHeads) * kChunks * kHeadDim * sizeof(float)));
    const double t_attn = time_graph([&] {
      kernels::launch_attention_step(q, kc, vc, ao, kSeq, kHeads, kKv, kHeadDim, s, sm, sl, so,
                                     kChunks, true);
    });

    std::printf("\n  Non-GEMV decode kernels, in-graph, Qwen2.5-0.5B shapes (seq=%d):\n\n", kSeq);
    std::printf("  %-22s %10s %6s %12s\n", "kernel", "us each", "xN", "us/token");
    std::printf("  %-22s %10.2f %6d %12.1f\n", "rmsnorm", t_norm, 2 * kLayers, t_norm * 2 * kLayers);
    std::printf("  %-22s %10.2f %6d %12.1f\n", "rope", t_rope, kLayers, t_rope * kLayers);
    std::printf("  %-22s %10.2f %6d %12.1f\n", "attention_step", t_attn, kLayers, t_attn * kLayers);
    std::printf("  %-22s %10s %6s %12.1f\n", "TOTAL (these only)", "", "",
                (t_norm * 2 + t_rope + t_attn) * kLayers);

    cudaStreamDestroy(s);
    cudaFree(x); cudaFree(wt); cudaFree(y); cudaFree(q); cudaFree(kc); cudaFree(vc);
    cudaFree(ao); cudaFree(ct); cudaFree(st); cudaFree(sm); cudaFree(sl); cudaFree(so);
  }

  // ---------------------------------------------------------------------------------
  // Cost of a host sync between graph launches: decode launches the graph and then blocks on
  // it once per token, so this measures whether that round trip is significant.
  {
    constexpr int kNodes = 300;  // ~ the real decode graph (294)
    constexpr int kLaunches = 200;
    __half *w = nullptr, *x = nullptr;
    float* y = nullptr;
    CK(cudaMalloc(&w, 1152ull * 896 * sizeof(__half)));
    CK(cudaMalloc(&x, 896 * sizeof(__half)));
    CK(cudaMalloc(&y, 1152 * sizeof(float)));
    CK(cudaMemset(w, 0x11, 1152ull * 896 * sizeof(__half)));

    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < kNodes; ++i)
      kernels::launch_rowmajor_half_gemv_f32(w, x, y, 1152, 896, s, 4, 128, 1);
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    // (1) sync after every launch; what decode does today.
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kLaunches; ++i) {
      cudaGraphLaunch(ge, s);
      cudaStreamSynchronize(s);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double per_sync =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / kLaunches;

    // (2) launch back-to-back, sync once at the end.
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kLaunches; ++i) cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);
    t1 = std::chrono::high_resolution_clock::now();
    const double per_pipelined =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / kLaunches;

    std::printf("\n  Cost of a host sync between graph launches (%d-node graph):\n\n", kNodes);
    std::printf("    sync after every launch : %7.3f ms   <- what decode does per token\n",
                per_sync);
    std::printf("    sync once at the end    : %7.3f ms\n", per_pipelined);
    std::printf("    HOST-SYNC TAX           : %7.3f ms per launch\n", per_sync - per_pipelined);

    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
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
