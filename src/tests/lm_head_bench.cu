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

// Time the kernel INSIDE A CUDA GRAPH -- which is what decode actually runs.
//
// Timing back-to-back launches on a stream (bench() above) does NOT measure the kernel: on
// Windows/WDDM each stream launch carries several microseconds of driver overhead, and that
// overhead is what you end up timing. It is why wqkv and wo both came out at a flat ~7.6 us
// despite being different sizes, and why giving them 8x the parallelism (split-K) changed
// nothing -- there was nothing to speed up, they were WAITING on the driver, not computing.
// A graph pays that cost once at instantiation, not per node. Capture N launches, replay,
// and the per-kernel number is a real duration.
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
  constexpr double kPeak = 1790.0;  // GB/s, RTX 5090 spec
  constexpr int kIters = 300;

  std::printf("Decode GEMVs, Qwen2.5-0.5B, RTX 5090 (peak %.0f GB/s)\n\n", kPeak);
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
  // The GEMVs turned out to be ~88% of roofline, i.e. innocent. So the rest of the token
  // is the ~197 NON-GEMV kernels. Time those in-graph too, at the 0.5B's real shapes,
  // before optimising anything -- the last two hypotheses died on measurements like this.
  // ---------------------------------------------------------------------------------
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
    // Give attention the SAME split-K scratch the engine's graph gives it. Without the
    // scratch it silently drops to the fallback path and reports ~255 us -- which would make
    // a token 6 ms when a real one is 1.75 ms. Whenever a micro-bench claims a single op
    // costs more than the whole operation containing it, the BENCH is wrong, not the engine.
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

  const double ideal = total_mb / 1e3 / kPeak * 1e3;
  std::printf("\n  weight traffic per token : %.0f MB\n", total_mb);
  std::printf("  roofline                 : %.3f ms\n", ideal);
  std::printf("  sum of the GEMVs alone   : %.3f ms  (%.0f%% of roofline)\n", total_ms,
              ideal / total_ms * 100);
  std::printf("\n  Everything else (attention, norms, rope, kv-store, residuals, sampling,\n");
  std::printf("  and the per-kernel launch tax) is whatever a real token costs MINUS this.\n");
  return 0;
}
