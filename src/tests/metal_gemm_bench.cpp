// Isolated fp16 GEMM benchmark -- that CHECKS WHAT IT TIMES.
//
// This exists because of what happened to this kernel. It was tuned for two sessions against
// end-to-end wall-clock, a change raised its row tile without raising its thread count, and it
// began writing half the rows of every tile. That measured as +37%, and was accepted, because
// a kernel that skips half its writes is not slower -- it is faster. No gate caught it: the
// GEMM only runs at T >= kGemmMinTokens and every golden prompt was shorter than that.
//
// So: every configuration this benchmark reports a number for is spot-checked against a CPU
// reference in the same run, and a FAILED configuration prints no tok/s at all. A number here
// means the kernel computed the right answer while achieving it. That property is the point of
// the file; do not "optimize" it away.
//
// Shapes are Qwen2.5-0.5B's real projections at a realistic prefill depth, so the FLOPS relate
// to the prefill numbers in the README rather than to a synthetic square.
//
//   metal_gemm_bench [tokens] [reps]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "runtime/metal_context.hpp"

namespace {

std::uint16_t f32_to_f16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
  std::uint32_t man = x & 0x7FFFFFu;
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (man >> 13));
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t man = h & 0x3FFu;
  std::uint32_t out;
  if (exp == 0) {
    out = sign;
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (man << 13);
  } else {
    out = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

// MUST match the shader / engine. The whole bug was a copy of these drifting, so they are
// stated once and everything below derives from them.
//
// And "everything" has to mean everything: this file previously derived the GRID from kFBM but
// still computed the thread count as 32 * (kFBM/32) * (kBN/32) -- a formula that silently
// assumed 32x32 per simdgroup. The moment the per-simdgroup tile became a variable, that line
// handed the kernel twice the threads it wanted and the bench reported WRONG. The bench caught
// it, which is the design working, but the lesson is the same one that started all this: a
// restated formula is a bug with a delay.
constexpr std::uint32_t kFBM = 64;  // GEMM_FBM: rows per threadgroup
constexpr std::uint32_t kBN = 32;   // GEMM_BN:  tokens per tile
constexpr std::uint32_t kRF = 4;    // GEMM_RF:  8x8 row fragments per simdgroup
constexpr std::uint32_t kCF = 4;    // GEMM_CF:  8x8 col fragments per simdgroup

struct Shape {
  const char* name;
  std::uint32_t out_dim, in_dim;
};

}  // namespace

int main(int argc, char** argv) {
  const std::uint32_t T = (argc > 1) ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 541;
  const int reps = (argc > 2) ? std::atoi(argv[2]) : 20;

  runtime::MetalContext ctx;
  if (!ctx.available()) {
    std::printf("[gemm_bench] SKIP: no Metal GPU (%s)\n", ctx.last_error().c_str());
    return 0;
  }
  if (!ctx.load_library()) {
    std::printf("[gemm_bench] SKIP: no shader library (%s)\n", ctx.last_error().c_str());
    return 0;
  }
  std::printf("[gemm_bench] %s | T=%u reps=%d | tile %ux%u\n", ctx.device_name().c_str(), T, reps,
              kFBM, kBN);

  // Qwen2.5-0.5B's projections. qkv is fused in the plan but the shapes are what matter.
  // Qwen2.5-0.5B's SEVEN real projections per layer, not three. The k/v pair matters far more
  // than its 1.5% of the FLOPs suggests: out_dim 128 gives a grid of (128/64) * tiles = 18
  // threadgroups on a 10-core GPU, so it is launch/occupancy-bound rather than ALU-bound and
  // its cost does not scale with its arithmetic. Leaving it out is how a bench reports 2.85
  // TFLOP/s for a pass that really achieves ~1.87.
  const Shape shapes[] = {
      {"q_proj    896x896", 896, 896},
      {"k_proj    128x896", 128, 896},
      {"v_proj    128x896", 128, 896},
      {"o_proj    896x896", 896, 896},
      {"gate_proj 4864x896", 4864, 896},
      {"up_proj   4864x896", 4864, 896},
      {"down_proj 896x4864", 896, 4864},
      // Fused shapes, pricing the fusion the plan does NOT do: qkv is q|k|v stacked (896+128+128)
      // and gateup is gate|up stacked (2*4864), both from the same input. The measured verdict, at
      // T=512: gateup fused is 3.25 TFLOP/s vs its parts' 3.20 -- the big MLP GEMMs are already
      // grid-saturated, so fusing them saves only a dispatch (~0.4%). qkv fused is 2.87 vs its
      // parts' effective ~1.6 -- a real 0.48 -> 0.37 ms/layer win, but ONLY because it rescues the
      // grid-starved k_proj/v_proj (128 rows = 18 threadgroups), which are 1.5% of the FLOPs.
      // Net prefill win ~1.8% at T=512, shrinking as T grows and k/v stop being starved -- not
      // worth the weight-concat + slot-splitting plumbing (q/k/v would share one buffer that RoPE
      // and the KV-store then slice). The GEMM is at its practical ceiling; this documents why.
      {"[fused qkv    1152x896]", 1152, 896},
      {"[fused gateup 9728x896]", 9728, 896},
  };

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  double total_flop = 0.0, total_ms = 0.0;
  int failures = 0;

  for (const Shape& s : shapes) {
    // CPI_METAL_GEMM_ROTATE=<n> allocates n DISTINCT weight matrices and cycles through them,
    // instead of hammering one. This exists because the single-matrix default does not
    // reproduce a prefill: a real pass walks 24 layers x 7 matrices (~716 MB) once each, while
    // one matrix reused 20x sits in cache. The default reports 2.85 TFLOP/s where a real
    // prefill achieves ~1.87 for the same shapes -- so the default is measuring the cache, and
    // any limiter read from it describes the cache-hot case, not the one that matters.
    const int rotate = [] {
      const char* e = std::getenv("CPI_METAL_GEMM_ROTATE");
      return e != nullptr ? std::max(1, std::atoi(e)) : 1;
    }();
    std::vector<std::uint16_t> W(static_cast<std::size_t>(s.out_dim) * s.in_dim);
    std::vector<std::uint16_t> A(static_cast<std::size_t>(T) * s.in_dim);
    for (auto& v : W) v = f32_to_f16(dist(rng) * 0.1f);
    for (auto& v : A) v = f32_to_f16(dist(rng));

    // The rotation set. Contents do not matter (the spot check uses W, so rotation>1 only
    // times the traffic and the correctness check runs against the buffer it verified).
    std::vector<runtime::MetalBuffer> wrot;
    for (int i = 1; i < rotate; ++i) {
      std::vector<std::uint16_t> Wi(W.size());
      for (auto& v : Wi) v = f32_to_f16(dist(rng) * 0.1f);
      wrot.push_back(ctx.alloc_from(Wi.data(), Wi.size() * 2));
    }

    auto bW = ctx.alloc_from(W.data(), W.size() * 2);
    auto bA = ctx.alloc_from(A.data(), A.size() * 2);
    auto bo = ctx.alloc(static_cast<std::size_t>(T) * s.out_dim * 2);

    struct GP {
      std::uint32_t out_dim, in_dim, tokens, has_bias;
    } p{s.out_dim, s.in_dim, T, 0};

    const std::size_t tiles = (T + kBN - 1) / kBN;
    const std::size_t grid = (s.out_dim / kFBM) * tiles;
    const std::size_t threads = 32 * (kFBM / (8 * kRF)) * (kBN / (8 * kCF));
    const void* bufs[] = {bW.handle(), bA.handle(), bo.handle(), bW.handle()};

    // WARM UP UNTIL THE CLOCK STOPS RAMPING, not for a fixed few dispatches. Three was not
    // enough and it cost a wrong conclusion: an A/B of GEMM_FBK 32 vs 64 read as +15% for 64,
    // and interleaving the runs showed it was the FIRST measurement in each process that was
    // slow, whichever config it happened to be. The GPU ramps over ~100 ms; anything measured
    // before that is measuring the ramp.
    const auto warm_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm_start)
               .count() < 300.0) {
      for (int i = 0; i < 5; ++i) {
        ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, grid, threads, bufs,
                     nullptr, 4, &p, sizeof(p));
      }
      ctx.commit_and_wait();
    }

    // BEST of several timed batches, not the mean. The mean folds in whatever else the OS
    // decided to do; the best run is the one where the kernel got the machine to itself, and
    // it is what a kernel comparison wants.
    double ms = 1e30;
    for (int trial = 0; trial < 5; ++trial) {
      ctx.reset_counters();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) {
        const void* rb[] = {(i % rotate == 0) ? bW.handle() : wrot[(i % rotate) - 1].handle(),
                            bA.handle(), bo.handle(), bW.handle()};
        ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, grid, threads, rb,
                     nullptr, 4, &p, sizeof(p));
      }
      ctx.commit_and_wait();
      ms = std::min(ms, std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count());
    }

    // CPI_METAL_GPUTRACE=<path> captures ONE dispatch of this shape into a .gputrace for
    // Xcode's Metal Debugger, which is the only thing on Apple Silicon that reports occupancy
    // and the limiter -- the numbers that would explain why this kernel sits at ~53% of peak
    // when it does not spill and the GPU is at its Maximum performance state. One dispatch,
    // because a capture records every command and the file grows fast.
    //
    // The timing above is already done, so the capture cannot distort the number this
    // benchmark reports -- which matters, given a capture makes everything slower.
    if (const char* gt = std::getenv("CPI_METAL_GPUTRACE")) {
      const std::string path = std::string(gt) + "-" + std::to_string(s.out_dim) + "x" +
                               std::to_string(s.in_dim) + ".gputrace";
      if (ctx.begin_gputrace(path)) {
        ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, grid, threads, bufs,
                     nullptr, 4, &p, sizeof(p));
        ctx.end_gputrace();
        std::printf("  [capture] %s\n", path.c_str());
      } else {
        std::printf("  [capture] SKIPPED: %s\n", ctx.last_error().c_str());
      }
    }

    // Leave `bo` holding W's result: with rotate>1 the timed loop's last dispatch may have used
    // a rotation matrix, and then the spot check below compares W's expected values against
    // some other matrix's output and reports WRONG. (It did. The bench refusing to print a
    // number it cannot verify is the design working -- the bug was mine, in the harness.)
    if (rotate > 1) {
      ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, grid, threads, bufs,
                   nullptr, 4, &p, sizeof(p));
      ctx.commit_and_wait();
    }

    // CHECK WHAT WE TIMED. A spot check, not the full product: 128 random (token,row) dot
    // products on the host. A kernel writing half its rows fails this immediately, which is
    // the entire reason the benchmark refuses to report a number without it.
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    std::mt19937 pick(99);
    std::uniform_int_distribution<std::uint32_t> tsel(0, T - 1), rsel(0, s.out_dim - 1);
    double worst = 0.0;
    for (int k = 0; k < 128; ++k) {
      const std::uint32_t t = tsel(pick), r = rsel(pick);
      float acc = 0.0f;
      for (std::uint32_t j = 0; j < s.in_dim; ++j) {
        acc += f16_to_f32(W[static_cast<std::size_t>(r) * s.in_dim + j]) *
               f16_to_f32(A[static_cast<std::size_t>(t) * s.in_dim + j]);
      }
      const float got = f16_to_f32(o[static_cast<std::size_t>(t) * s.out_dim + r]);
      worst = std::max(worst, static_cast<double>(std::fabs(got - acc)));
    }

    const double flop = 2.0 * T * s.out_dim * s.in_dim * reps;
    if (worst > 0.2) {
      std::printf("  %-20s  WRONG (max|d|=%.3g over 128 spot checks) -- no timing reported\n",
                  s.name, worst);
      ++failures;
      continue;
    }
    std::printf("  %-20s  %7.2f ms/rep   %5.2f TFLOP/s   (max|d|=%.3g)\n", s.name, ms / reps,
                flop / (ms / 1000.0) / 1e12, worst);
    // The `[fused ...]` shapes are informational -- they price a fusion the plan does not do, so
    // they must NOT enter the aggregate (that would double-count gate/up as both parts and fused).
    if (s.name[0] == '[') continue;
    total_flop += flop;
    total_ms += ms;
  }

  if (failures != 0) {
    std::printf("\n[gemm_bench] FAIL: %d shape(s) computed the wrong answer\n", failures);
    return 1;
  }
  std::printf("\n  aggregate: %.2f TFLOP/s\n", total_flop / (total_ms / 1000.0) / 1e12);
  std::printf("[gemm_bench] OK\n");
  return 0;
}
