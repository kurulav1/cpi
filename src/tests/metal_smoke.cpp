// Runs the Metal shaders against a CPU reference and reports the error.
//
// This is the first thing to run on real Apple Silicon. In CI it still builds and
// still runs, but MTLCreateSystemDefaultDevice() returns nil inside GitHub's macOS
// VM, so it reports SKIP and exits 0 -- a green CI here means "it compiles and
// links", never "the kernels are correct". Only a real GPU can say that.
//
// Tolerances are loose on purpose: the shaders accumulate in fp32 but read/write
// fp16, and the accumulation ORDER differs from the CPU reference, so exact
// equality is not the bar. What we are checking is that each kernel computes the
// right function -- a transposed index or a bad stride blows past these bounds by
// orders of magnitude.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "runtime/metal_context.hpp"

namespace {

// Minimal fp16 <-> fp32, so this test depends on nothing but the context.
std::uint16_t f32_to_f16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
  std::uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h >> 15) << 31;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;
  std::uint32_t bits;
  if (exp == 0) {
    bits = sign;  // treat denormals as zero; irrelevant at these magnitudes
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

struct Result {
  double max_abs = 0.0;
  double mean_abs = 0.0;
};

Result compare(const std::vector<float>& got, const std::vector<float>& want) {
  Result r;
  double sum = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
    r.max_abs = std::max(r.max_abs, d);
    sum += d;
  }
  r.mean_abs = want.empty() ? 0.0 : sum / static_cast<double>(want.size());
  return r;
}

int failures = 0;

void check(const std::string& name, const Result& r, double tol) {
  const bool ok = r.max_abs <= tol && std::isfinite(r.max_abs);
  std::printf("  %-14s max_abs=%.5f  mean_abs=%.6f  tol=%.3f  %s\n", name.c_str(), r.max_abs,
              r.mean_abs, tol, ok ? "PASS" : "FAIL");
  if (!ok) ++failures;
}

// Parameter blocks -- must match the layouts in cpi_kernels.metal exactly.
struct NormParams {
  std::uint32_t rows, cols;
  float eps;
  std::uint32_t weight_offset, has_weight;
};
struct GemvParams {
  std::uint32_t out_dim, in_dim, tokens, has_bias;
};
struct ElemParams {
  std::uint32_t n;
  float scale;
};

}  // namespace

int main() {
  runtime::MetalContext ctx;

  if (!ctx.available()) {
    std::printf("[metal_smoke] SKIP: %s\n", ctx.last_error().c_str());
    std::printf("[metal_smoke] (expected inside a VM -- GitHub's macOS runners have no GPU)\n");
    return 0;
  }
  std::printf("[metal_smoke] device: %s\n", ctx.device_name().c_str());

  if (!ctx.load_library()) {
    std::printf("[metal_smoke] FAIL: %s\n", ctx.last_error().c_str());
    std::printf("[metal_smoke] set CPI_METALLIB to the compiled cpi_kernels.metallib\n");
    return 1;
  }

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // ---- RMSNorm -----------------------------------------------------------
  {
    const std::uint32_t rows = 4, cols = 512;
    std::vector<std::uint16_t> x(rows * cols), w(cols);
    std::vector<float> xf(rows * cols), wf(cols);
    for (auto& v : xf) v = dist(rng);
    for (auto& v : wf) v = dist(rng);
    for (std::size_t i = 0; i < xf.size(); ++i) x[i] = f32_to_f16(xf[i]);
    for (std::size_t i = 0; i < wf.size(); ++i) w[i] = f32_to_f16(wf[i]);

    auto bx = ctx.alloc_from(x.data(), x.size() * 2);
    auto bw = ctx.alloc_from(w.data(), w.size() * 2);
    auto bo = ctx.alloc(x.size() * 2);

    NormParams p{rows, cols, 1e-6f, 0, 1};
    const void* bufs[] = {bx.handle(), bw.handle(), bo.handle()};
    ctx.dispatch("cpi_rmsnorm", runtime::MetalContext::Grid::Groups, rows, 256, bufs, nullptr, 3,
                 &p, sizeof(p));
    ctx.commit_and_wait();

    std::vector<float> want(rows * cols), got(rows * cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
      double ss = 0.0;
      for (std::uint32_t c = 0; c < cols; ++c) {
        const float v = f16_to_f32(x[r * cols + c]);
        ss += static_cast<double>(v) * v;
      }
      const double inv = 1.0 / std::sqrt(ss / cols + 1e-6);
      for (std::uint32_t c = 0; c < cols; ++c) {
        want[r * cols + c] =
            static_cast<float>(f16_to_f32(x[r * cols + c]) * inv * f16_to_f32(w[c]));
      }
    }
    const auto* out = static_cast<const std::uint16_t*>(bo.contents());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(out[i]);
    check("rmsnorm", compare(got, want), 0.01);
  }

  // ---- GEMV, with and without a bias -------------------------------------
  // Qwen2's Q/K/V projections carry a bias and Llama's do not, so both paths of
  // the same kernel need exercising. A bias that is silently dropped still yields
  // fluent-looking output, which is exactly why it gets its own check.
  for (int with_bias = 0; with_bias <= 1; ++with_bias) {
    const std::uint32_t out_dim = 128, in_dim = 256;
    std::vector<std::uint16_t> W(out_dim * in_dim), xin(in_dim), bias(out_dim);
    for (auto& v : W) v = f32_to_f16(dist(rng) * 0.1f);
    for (auto& v : xin) v = f32_to_f16(dist(rng));
    for (auto& v : bias) v = f32_to_f16(dist(rng) * 2.0f);

    auto bW = ctx.alloc_from(W.data(), W.size() * 2);
    auto bx = ctx.alloc_from(xin.data(), xin.size() * 2);
    auto bo = ctx.alloc(out_dim * 2);
    auto bb = ctx.alloc_from(bias.data(), bias.size() * 2);

    GemvParams p{out_dim, in_dim, 1, static_cast<std::uint32_t>(with_bias)};
    // The bias buffer is always bound; has_bias decides whether it is read.
    const void* bufs[] = {bW.handle(), bx.handle(), bo.handle(), bb.handle()};
    // 256 threads = 8 simdgroups = 8 rows per threadgroup.
    const std::size_t rows_per_tg = 8;
    const std::size_t groups = (out_dim + rows_per_tg - 1) / rows_per_tg;
    ctx.dispatch("cpi_gemv_f16", runtime::MetalContext::Grid::Groups, groups, 256, bufs, nullptr, 4,
                 &p, sizeof(p));
    ctx.commit_and_wait();

    std::vector<float> want(out_dim), got(out_dim);
    for (std::uint32_t r = 0; r < out_dim; ++r) {
      float acc = 0.0f;
      for (std::uint32_t c = 0; c < in_dim; ++c) {
        acc += f16_to_f32(W[r * in_dim + c]) * f16_to_f32(xin[c]);
      }
      if (with_bias) acc += f16_to_f32(bias[r]);
      want[r] = acc;
    }
    const auto* out = static_cast<const std::uint16_t*>(bo.contents());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(out[i]);
    check(with_bias ? "gemv_f16+bias" : "gemv_f16", compare(got, want), 0.05);
  }

  // ---- SiLU-mul (SwiGLU) -------------------------------------------------
  {
    const std::uint32_t n = 1024;
    std::vector<std::uint16_t> a(n), b(n);
    for (auto& v : a) v = f32_to_f16(dist(rng) * 3.0f);
    for (auto& v : b) v = f32_to_f16(dist(rng));

    auto ba = ctx.alloc_from(a.data(), n * 2);
    auto bb = ctx.alloc_from(b.data(), n * 2);
    auto bo = ctx.alloc(n * 2);

    ElemParams p{n, 0.0f};
    const void* bufs[] = {ba.handle(), bb.handle(), bo.handle()};
    ctx.dispatch("cpi_silu_mul", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr, 3, &p,
                 sizeof(p));
    ctx.commit_and_wait();

    std::vector<float> want(n), got(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      const float av = f16_to_f32(a[i]);
      want[i] = (av / (1.0f + std::exp(-av))) * f16_to_f32(b[i]);
    }
    const auto* out = static_cast<const std::uint16_t*>(bo.contents());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(out[i]);
    check("silu_mul", compare(got, want), 0.01);
  }

  // ---- Argmax ------------------------------------------------------------
  {
    const std::uint32_t n = 151936;  // a real vocab size
    std::vector<float> logits(n);
    for (auto& v : logits) v = dist(rng);
    const std::uint32_t planted = 90210;
    logits[planted] = 99.0f;

    const std::uint32_t parts = 256;
    auto bl = ctx.alloc_from(logits.data(), n * sizeof(float));
    auto bpv = ctx.alloc(parts * sizeof(float));
    auto bpi = ctx.alloc(parts * sizeof(std::int32_t));
    auto bo = ctx.alloc(sizeof(std::int32_t));

    ElemParams p1{n, 0.0f};
    const void* bufs1[] = {bl.handle(), bpv.handle(), bpi.handle()};
    ctx.dispatch("cpi_argmax_partial", runtime::MetalContext::Grid::Groups, parts, 256, bufs1,
                 nullptr, 3, &p1, sizeof(p1));

    ElemParams p2{parts, 0.0f};
    const void* bufs2[] = {bpv.handle(), bpi.handle(), bo.handle()};
    ctx.dispatch("cpi_argmax_reduce", runtime::MetalContext::Grid::Groups, 1, 256, bufs2, nullptr,
                 3, &p2, sizeof(p2));
    ctx.commit_and_wait();

    const std::int32_t got = *static_cast<const std::int32_t*>(bo.contents());
    const bool ok = got == static_cast<std::int32_t>(planted);
    std::printf("  %-14s got=%d want=%u  %s\n", "argmax", got, planted, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
  }

  std::printf("[metal_smoke] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
