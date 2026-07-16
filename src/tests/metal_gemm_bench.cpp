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
constexpr std::uint32_t kFBM = 64;  // GEMM_FBM: rows per threadgroup
constexpr std::uint32_t kBN = 64;   // GEMM_BN:  tokens per tile

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
  const Shape shapes[] = {
      {"q/o_proj  896x896", 896, 896},
      {"gate/up   4864x896", 4864, 896},
      {"down      896x4864", 896, 4864},
  };

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  double total_flop = 0.0, total_ms = 0.0;
  int failures = 0;

  for (const Shape& s : shapes) {
    std::vector<std::uint16_t> W(static_cast<std::size_t>(s.out_dim) * s.in_dim);
    std::vector<std::uint16_t> A(static_cast<std::size_t>(T) * s.in_dim);
    for (auto& v : W) v = f32_to_f16(dist(rng) * 0.1f);
    for (auto& v : A) v = f32_to_f16(dist(rng));

    auto bW = ctx.alloc_from(W.data(), W.size() * 2);
    auto bA = ctx.alloc_from(A.data(), A.size() * 2);
    auto bo = ctx.alloc(static_cast<std::size_t>(T) * s.out_dim * 2);

    struct GP {
      std::uint32_t out_dim, in_dim, tokens, has_bias;
    } p{s.out_dim, s.in_dim, T, 0};

    const std::size_t tiles = (T + kBN - 1) / kBN;
    const std::size_t grid = (s.out_dim / kFBM) * tiles;
    const std::size_t threads = 32 * (kFBM / 32) * (kBN / 32);
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
        ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, grid, threads, bufs,
                     nullptr, 4, &p, sizeof(p));
      }
      ctx.commit_and_wait();
      ms = std::min(ms, std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count());
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
