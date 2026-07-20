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
// Must match the layouts in 70_linear_attn.metal exactly.
struct LinSplitParams {
  std::uint32_t heads, head_dim;
};
struct LinGateParams {
  std::uint32_t n;
};
struct LinMulVecParams {
  std::uint32_t n;
  float scale;
};
struct LinConvParams {
  std::uint32_t channels, kernel_size;
};
struct LinRepeatParams {
  std::uint32_t num_key_heads, head_repeat, key_head_dim, value_head_dim;
};
struct LinAttnParams {
  std::uint32_t key_head_dim, value_head_dim;
  float rms_eps;
};
// Must match the layouts in 80_vision.metal exactly.
struct VisPatchParams {
  std::uint32_t hidden, patch_dim, pos_table_size, tokens;
};
struct VisRope2DParams {
  std::uint32_t num_heads, head_dim, pairs_per_half, tokens;
};
struct VisPoolParams {
  std::uint32_t tokens, hidden, k, cells_x, out_tokens;
  float gain;
};
struct VisStdParams {
  std::uint32_t n, hidden;
};
// Mirrors BiAttnParams in 00_common.metal.
struct BiAttnParams {
  std::uint32_t tokens, heads, head_dim;
  float scale;
};
// Mirrors LayerNormParams in 00_common.metal.
struct LayerNormParams {
  std::uint32_t rows, cols;
  float eps;
  std::uint32_t has_bias;
};

float sigmoidf(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
float siluf(float x) {
  return x * sigmoidf(x);
}
float softplusf(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return std::exp(x);
  return std::log1p(std::exp(x));
}

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

  // ---- GEMM (fp16), the multi-token prefill path -------------------------
  // MUST match GEMM_FBM / GEMM_BN in cpi_kernels.metal (and kGemmFBM / kGemmBN in the engine).
  constexpr std::uint32_t kSmokeGemmFBM = 64;   // rows per threadgroup
  constexpr std::uint32_t kSmokeGemmBN = 32;    // tokens per tile
  // This kernel had NO check of its own: it was benchmarked heavily and gated only
  // indirectly, through end-to-end goldens. A prefill chunk and a token-at-a-time run
  // disagreed on Metal, and the split fell exactly on the GEMV/GEMM boundary, so the
  // question "is the GEMM itself right?" needs an answer that does not route through
  // another Metal path. Sweeping the token count matters: the kernel pads its tile out to
  // GEMM_BN, so partial tiles (T not a multiple of 64) are the interesting case.
  for (int tokens : {1, 8, 16, 24, 64, 100}) {
    const std::uint32_t out_dim = 256, in_dim = 128;  // out_dim % 128, in_dim % 32: the guard
    const std::uint32_t T = static_cast<std::uint32_t>(tokens);
    std::vector<std::uint16_t> W(out_dim * in_dim), A(T * in_dim), bias(out_dim);
    for (auto& v : W) v = f32_to_f16(dist(rng) * 0.1f);
    for (auto& v : A) v = f32_to_f16(dist(rng));
    for (auto& v : bias) v = f32_to_f16(dist(rng) * 2.0f);

    auto bW = ctx.alloc_from(W.data(), W.size() * 2);
    auto bA = ctx.alloc_from(A.data(), A.size() * 2);
    auto bo = ctx.alloc(static_cast<std::size_t>(T) * out_dim * 2);
    auto bb = ctx.alloc_from(bias.data(), bias.size() * 2);

    GemvParams p{out_dim, in_dim, T, 1};
    const void* bufs[] = {bW.handle(), bA.handle(), bo.handle(), bb.handle()};
    // Derive the dispatch from the tile constants with the same rule the engine uses, rather
    // than restating 256/128 as literals. Restating them is exactly how this kernel broke: the
    // shader's row tile moved and a hardcoded thread count stayed behind. If FBM changes and
    // this test still passes, the test is worth something.
    const std::size_t tiles = (T + kSmokeGemmBN - 1) / kSmokeGemmBN;
    const std::size_t groups = (out_dim / kSmokeGemmFBM) * tiles;
    const std::size_t threads = 32 * (kSmokeGemmFBM / 32) * (kSmokeGemmBN / 32);
    ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, groups, threads, bufs,
                 nullptr, 4, &p, sizeof(p));
    ctx.commit_and_wait();

    std::vector<float> want(static_cast<std::size_t>(T) * out_dim), got(want.size());
    for (std::uint32_t t = 0; t < T; ++t) {
      for (std::uint32_t r = 0; r < out_dim; ++r) {
        float acc = 0.0f;
        for (std::uint32_t c = 0; c < in_dim; ++c) {
          acc += f16_to_f32(W[r * in_dim + c]) * f16_to_f32(A[t * in_dim + c]);
        }
        want[t * out_dim + r] = acc + f16_to_f32(bias[r]);
      }
    }
    const auto* out = static_cast<const std::uint16_t*>(bo.contents());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(out[i]);
    check("gemm_f16 T=" + std::to_string(tokens), compare(got, want), 0.05);
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

  // ---- Quantized GEMV (int4 and int8) ------------------------------------
  // Quantize a known matrix on the host exactly as PlanMetalEngine does, run the
  // Metal kernel, and compare against a CPU dot product of the DEQUANTIZED weights.
  // That separates two questions the end-to-end test conflates: is the kernel right,
  // and is the quantization format right. Comparing against the dequantized weights
  // (not the originals) means quantization ERROR cannot mask a kernel BUG.
  for (int bits : {8, 4}) {
    const std::uint32_t out_dim = 64, in_dim = 256, group = 64;
    const std::uint32_t groups = in_dim / group;
    const float max_q = (bits == 4) ? 7.0f : 127.0f;

    std::vector<float> W(out_dim * in_dim);
    std::vector<std::uint16_t> xin(in_dim);
    for (auto& v : W) v = dist(rng);
    for (auto& v : xin) v = f32_to_f16(dist(rng));

    const std::size_t packed_row = (bits == 4) ? (in_dim + 1) / 2 : in_dim;
    std::vector<std::uint8_t> packed(out_dim * packed_row, 0);
    std::vector<float> scales(out_dim * groups);
    std::vector<float> deq(out_dim * in_dim);  // what the kernel SHOULD compute with

    for (std::uint32_t r = 0; r < out_dim; ++r) {
      for (std::uint32_t g = 0; g < groups; ++g) {
        float amax = 0.0f;
        for (std::uint32_t j = g * group; j < (g + 1) * group; ++j) {
          amax = std::max(amax, std::fabs(W[r * in_dim + j]));
        }
        float sc = std::max(amax / max_q, 1e-8f);
        scales[r * groups + g] = sc;
        for (std::uint32_t j = g * group; j < (g + 1) * group; ++j) {
          int q = static_cast<int>(std::lround(W[r * in_dim + j] / sc));
          q = (bits == 4) ? std::max(-8, std::min(7, q)) : std::max(-127, std::min(127, q));
          deq[r * in_dim + j] = static_cast<float>(q) * sc;
          if (bits == 4) {
            const std::uint8_t nib = static_cast<std::uint8_t>(q < 0 ? q + 16 : q);
            std::uint8_t& b = packed[r * packed_row + j / 2];
            b = (j & 1) == 0 ? static_cast<std::uint8_t>((b & 0xF0u) | nib)
                             : static_cast<std::uint8_t>((b & 0x0Fu) | (nib << 4));
          } else {
            packed[r * packed_row + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
          }
        }
      }
    }

    auto bq = ctx.alloc_from(packed.data(), packed.size());
    auto bx = ctx.alloc_from(xin.data(), xin.size() * 2);
    auto bo = ctx.alloc(out_dim * 2);
    auto bs = ctx.alloc_from(scales.data(), scales.size() * sizeof(float));

    struct QP {
      std::uint32_t out_dim, in_dim, tokens, bits, group, groups, has_bias;
    } p{out_dim, in_dim, 1, static_cast<std::uint32_t>(bits), group, groups, 0};

    const void* bufs[] = {bq.handle(), bx.handle(), bo.handle(), bs.handle(), bq.handle()};
    ctx.dispatch("cpi_gemv_quant", runtime::MetalContext::Grid::Groups, (out_dim + 7) / 8, 256,
                 bufs, nullptr, 5, &p, sizeof(p));
    ctx.commit_and_wait();

    std::vector<float> want(out_dim), got(out_dim);
    for (std::uint32_t r = 0; r < out_dim; ++r) {
      float acc = 0.0f;
      for (std::uint32_t j = 0; j < in_dim; ++j) acc += deq[r * in_dim + j] * f16_to_f32(xin[j]);
      want[r] = acc;
    }
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
    check(bits == 4 ? "gemv_int4" : "gemv_int8", compare(got, want), 0.05);
  }

  // ---- Quantized GEMM (int4/int8), the multi-token prefill path ----------
  // Same story as the fp16 GEMM above: this kernel had no check of its own, and it only runs
  // at T >= kGemmMinTokens, which no golden prompt reaches. It carries the 8B's entire
  // quantized prefill.
  // MUST match GEMM_BM / GEMM_QBN in cpi_kernels.metal (kGemmFBM's quant siblings).
  constexpr std::uint32_t kSmokeGemmBM = 64;    // quant rows per threadgroup
  constexpr std::uint32_t kSmokeGemmQBN = 128;  // quant tokens per tile
  for (int bits : {8, 4}) {
    for (int tokens : {1, 16, 24, 100, 128, 200}) {
      // The engine's guard: cols % 64, in_dim % GEMM_QBK(32), and the quant group >= QBK
      // (a K-block carries one scale per row, so it must sit inside one group).
      const std::uint32_t out_dim = 128, in_dim = 256, group = 64;
      const std::uint32_t groups_n = in_dim / group;
      const std::uint32_t T = static_cast<std::uint32_t>(tokens);
      const float max_q = (bits == 4) ? 7.0f : 127.0f;

      std::vector<float> W(out_dim * in_dim);
      std::vector<std::uint16_t> A(static_cast<std::size_t>(T) * in_dim);
      for (auto& v : W) v = dist(rng);
      for (auto& v : A) v = f32_to_f16(dist(rng));

      const std::size_t packed_row = (bits == 4) ? (in_dim + 1) / 2 : in_dim;
      std::vector<std::uint8_t> packed(out_dim * packed_row, 0);
      std::vector<float> scales(out_dim * groups_n);
      std::vector<float> deq(out_dim * in_dim);
      for (std::uint32_t r = 0; r < out_dim; ++r) {
        for (std::uint32_t g = 0; g < groups_n; ++g) {
          float amax = 0.0f;
          for (std::uint32_t j = g * group; j < (g + 1) * group; ++j) {
            amax = std::max(amax, std::fabs(W[r * in_dim + j]));
          }
          const float sc = std::max(amax / max_q, 1e-8f);
          scales[r * groups_n + g] = sc;
          for (std::uint32_t j = g * group; j < (g + 1) * group; ++j) {
            int q = static_cast<int>(std::lround(W[r * in_dim + j] / sc));
            q = (bits == 4) ? std::max(-8, std::min(7, q)) : std::max(-127, std::min(127, q));
            deq[r * in_dim + j] = static_cast<float>(q) * sc;
            if (bits == 4) {
              const std::uint8_t nib = static_cast<std::uint8_t>(q < 0 ? q + 16 : q);
              std::uint8_t& b = packed[r * packed_row + j / 2];
              b = (j & 1) == 0 ? static_cast<std::uint8_t>((b & 0xF0u) | nib)
                               : static_cast<std::uint8_t>((b & 0x0Fu) | (nib << 4));
            } else {
              packed[r * packed_row + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
            }
          }
        }
      }

      auto bq = ctx.alloc_from(packed.data(), packed.size());
      auto bx = ctx.alloc_from(A.data(), A.size() * 2);
      auto bo = ctx.alloc(static_cast<std::size_t>(T) * out_dim * 2);
      auto bs = ctx.alloc_from(scales.data(), scales.size() * sizeof(float));

      struct QP {
        std::uint32_t out_dim, in_dim, tokens, bits, group, groups, has_bias;
      } p{out_dim, in_dim, T, static_cast<std::uint32_t>(bits), group, groups_n, 0};

      // Derived from the tile constants, the same rule the engine uses -- never restated.
      const std::size_t tiles = (T + kSmokeGemmQBN - 1) / kSmokeGemmQBN;
      const std::size_t grid = (out_dim / kSmokeGemmBM) * tiles;
      const std::size_t threads = 32 * (kSmokeGemmBM / 32) * (kSmokeGemmQBN / 32);
      const void* bufs[] = {bq.handle(), bx.handle(), bo.handle(), bs.handle(), bq.handle()};
      ctx.dispatch("cpi_gemm_quant", runtime::MetalContext::Grid::Groups, grid, threads, bufs,
                   nullptr, 5, &p, sizeof(p));
      ctx.commit_and_wait();

      // Against the DEQUANTIZED weights, so quantization error cannot mask a kernel bug.
      std::vector<float> want(static_cast<std::size_t>(T) * out_dim), got(want.size());
      for (std::uint32_t t = 0; t < T; ++t) {
        for (std::uint32_t r = 0; r < out_dim; ++r) {
          float acc = 0.0f;
          for (std::uint32_t j = 0; j < in_dim; ++j) {
            acc += deq[r * in_dim + j] * f16_to_f32(A[t * in_dim + j]);
          }
          want[t * out_dim + r] = acc;
        }
      }
      const auto* o = static_cast<const std::uint16_t*>(bo.contents());
      for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
      check("gemm_int" + std::to_string(bits) + " T=" + std::to_string(tokens), compare(got, want),
            0.05);
    }
  }

  // ---- Attention (decode, prefill scalar, prefill matrix-unit) -----------
  // The last kernel family with no reference check, and the largest. THREE kernels hide
  // behind one op, chosen by (tokens, head_dim), so the sweep below is built to land on each:
  //   T < Q_BLOCK(8)          -> cpi_attention_decode      (one threadgroup per query)
  //   T >= 8, head_dim <= 128 -> cpi_attention_prefill_mm  (simdgroup matrix units)
  //   T >= 8, head_dim > 128  -> cpi_attention_prefill     (scalar; Gemma's 256)
  // A T that is not a multiple of Q_BLOCK matters (partial query block), as does GQA
  // (heads != kv_heads, so several queries share a KV head) and the sliding window.
  //
  // The reference is a plain three-loop attention in fp32: scores, softmax, weighted V. The
  // kernels compute it with an ONLINE softmax over key blocks, which is a different order of
  // operations, so agreement here is meaningful rather than tautological.
  {
    struct AP {  // MUST match AttnParams in cpi_kernels.metal
      std::uint32_t heads, kv_heads, head_dim, position, max_context, window;
      float scale;
      std::uint32_t use_position_buffer, tokens;
      std::uint32_t paged, block_size;  // contiguous here: paged = 0
    };
    struct Case {
      const char* name;
      std::uint32_t heads, kv_heads, head_dim, tokens, window, base;
    };
    // `base` is the position of the first query, and for decode it is the whole point.
    // A decode at position 0 attends to exactly ONE key: the softmax weight is 1 and the
    // output is just V[0], so it exercises no softmax, no accumulation, no online rescale --
    // it passes at max_abs == 0.00000 while proving nothing. Decode is tested at position 40,
    // against a populated cache, which is the shape it actually runs in (one query, many keys,
    // several KEY_BLOCK iterations with a running max).
    const Case cases[] = {
        {"decode hd64 @pos40", 4, 2, 64, 1, 0, 40},      // 41 keys: 2 key blocks
        {"decode hd128 @pos40", 4, 4, 128, 1, 0, 40},    // no GQA
        {"decode hd256 @pos40", 4, 2, 256, 1, 0, 40},    // Gemma's head_dim
        {"decode hd64 @pos40 win16", 4, 2, 64, 1, 16, 40},  // windowed decode
        {"decode hd128 T=4 @pos20", 4, 4, 128, 4, 0, 20},   // still decode: T < 8
        {"prefill_mm hd64 T=8", 4, 2, 64, 8, 0, 0},         // exactly one query block
        {"prefill_mm hd128 T=16", 8, 2, 128, 16, 0, 0},
        {"prefill_mm hd128 T=33", 4, 2, 128, 33, 0, 0},     // partial query block (4x8 + 1)
        {"prefill_mm hd64 window16", 4, 2, 64, 33, 16, 0},  // sliding window
        {"prefill scalar hd256 T=16", 4, 2, 256, 16, 0, 0},  // Gemma's head_dim
        {"prefill scalar hd256 T=33", 4, 1, 256, 33, 0, 0},  // MQA + partial block
    };
    constexpr std::uint32_t kQBlockSmoke = 8;  // MUST match Q_BLOCK in cpi_kernels.metal

    for (const Case& c : cases) {
      const std::uint32_t hd = c.head_dim, T = c.tokens;
      const std::uint32_t q_dim = c.heads * hd, kv_dim = c.kv_heads * hd;
      const std::uint32_t max_ctx = 64;
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

      std::vector<std::uint16_t> q(static_cast<std::size_t>(T) * q_dim);
      std::vector<std::uint16_t> kc(static_cast<std::size_t>(max_ctx) * kv_dim);
      std::vector<std::uint16_t> vc(static_cast<std::size_t>(max_ctx) * kv_dim);
      for (auto& v : q) v = f32_to_f16(dist(rng));
      for (auto& v : kc) v = f32_to_f16(dist(rng));
      for (auto& v : vc) v = f32_to_f16(dist(rng));

      auto bq = ctx.alloc_from(q.data(), q.size() * 2);
      auto bk = ctx.alloc_from(kc.data(), kc.size() * 2);
      auto bv = ctx.alloc_from(vc.data(), vc.size() * 2);
      auto bo = ctx.alloc(static_cast<std::size_t>(T) * q_dim * 2);
      const std::int32_t base = static_cast<std::int32_t>(c.base);
      auto bp = ctx.alloc_from(&base, sizeof(base));

      AP p{c.heads, c.kv_heads, hd, 0, max_ctx, c.window, scale, 1, T, 0, 0};
      const void* bufs[] = {bq.handle(), bk.handle(), bv.handle(), bo.handle(), bp.handle()};
      // The engine's own choice, reproduced -- testing a kernel production does not pick
      // would prove nothing about production.
      if (T >= kQBlockSmoke && hd <= 128) {
        // The matrix kernel takes a block table at buffer(5) for paged prefill, so its params
        // block sits at 6. p.paged is 0 here, so the table is never read -- but the binding must
        // exist or dispatch() would write the params over it. (This is what the gate caught when
        // the kernel gained that buffer and this caller did not: check_metal_bindings.py checks
        // the SHADER's ordering, not that a caller passes the right buffer count.)
        const std::size_t blocks = (T + kQBlockSmoke - 1) / kQBlockSmoke;
        const void* mmbufs[] = {bq.handle(), bk.handle(), bv.handle(),
                                bo.handle(), bp.handle(), bk.handle()};
        // This kernel is specialized on its shape, so the constants have to be supplied here too,
        // in the order it declares them. Omitting them fails pipeline creation outright -- which
        // is how the gate caught this caller when the kernel gained them.
        const std::uint32_t spec[3] = {hd, kv_dim, q_dim};
        ctx.set_next_specialization(spec, 3);
        ctx.dispatch("cpi_attention_prefill_mm", runtime::MetalContext::Grid::Groups,
                     c.heads * blocks, 256, mmbufs, nullptr, 6, &p, sizeof(p));
      } else if (T >= kQBlockSmoke) {
        const std::size_t blocks = (T + kQBlockSmoke - 1) / kQBlockSmoke;
        ctx.dispatch("cpi_attention_prefill", runtime::MetalContext::Grid::Groups,
                     c.heads * blocks, 256, bufs, nullptr, 5, &p, sizeof(p));
      } else {
        ctx.dispatch("cpi_attention_decode", runtime::MetalContext::Grid::Groups, c.heads * T, 256,
                     bufs, nullptr, 5, &p, sizeof(p));
      }
      ctx.commit_and_wait();

      std::vector<float> want(static_cast<std::size_t>(T) * q_dim), got(want.size());
      const std::uint32_t group = c.heads / c.kv_heads;
      for (std::uint32_t t = 0; t < T; ++t) {
        const std::uint32_t pos = base + t;
        std::uint32_t start = 0;
        if (c.window != 0 && pos + 1 > c.window) start = pos + 1 - c.window;
        for (std::uint32_t h = 0; h < c.heads; ++h) {
          const std::uint32_t kvh = h / group;
          std::vector<float> sc;
          float mx = -1e30f;
          for (std::uint32_t j = start; j <= pos; ++j) {
            float d = 0.0f;
            for (std::uint32_t i = 0; i < hd; ++i) {
              d += f16_to_f32(q[t * q_dim + h * hd + i]) * f16_to_f32(kc[j * kv_dim + kvh * hd + i]);
            }
            d *= scale;
            sc.push_back(d);
            mx = std::max(mx, d);
          }
          float sum = 0.0f;
          for (float& v : sc) {
            v = std::exp(v - mx);
            sum += v;
          }
          for (std::uint32_t i = 0; i < hd; ++i) {
            float acc = 0.0f;
            for (std::size_t jj = 0; jj < sc.size(); ++jj) {
              acc += sc[jj] * f16_to_f32(vc[(start + jj) * kv_dim + kvh * hd + i]);
            }
            want[t * q_dim + h * hd + i] = acc / sum;
          }
        }
      }
      const auto* o = static_cast<const std::uint16_t*>(bo.contents());
      for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
      check(std::string("attn ") + c.name, compare(got, want), 0.02);
    }
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

  // ---- linear attention (delta-net) family --------------------------------
  //
  // These six have no end-to-end golden on this backend yet -- the Qwen3.5 checkpoint that
  // exercises them is not converted here -- so a CPU reference IS the gate. Two of them carry
  // state across steps (the conv window and the recurrent matrix), and the reference reproduces
  // that state update too, which is the part a shape-only check would miss.
  {
    const std::uint32_t heads = 6, head_dim = 40;
    const std::uint32_t n = heads * head_dim;
    std::vector<std::uint16_t> src(heads * head_dim * 2);
    for (auto& v : src) v = f32_to_f16(dist(rng));
    auto bs = ctx.alloc_from(src.data(), src.size() * 2);
    auto b1 = ctx.alloc(n * 2), b2 = ctx.alloc(n * 2);
    LinSplitParams p{heads, head_dim};
    const void* bufs[] = {bs.handle(), b1.handle(), b2.handle()};
    ctx.dispatch("cpi_split_head_halves", runtime::MetalContext::Grid::Threads, n, 256, bufs,
                 nullptr, 3, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> w1(n), w2(n), g1(n), g2(n);
    for (std::uint32_t h = 0; h < heads; ++h)
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        w1[h * head_dim + d] = f16_to_f32(src[h * 2 * head_dim + d]);
        w2[h * head_dim + d] = f16_to_f32(src[h * 2 * head_dim + head_dim + d]);
      }
    const auto* o1 = static_cast<const std::uint16_t*>(b1.contents());
    const auto* o2 = static_cast<const std::uint16_t*>(b2.contents());
    for (std::uint32_t i = 0; i < n; ++i) { g1[i] = f16_to_f32(o1[i]); g2[i] = f16_to_f32(o2[i]); }
    check("split_halves", compare(g1, w1), 0.001);
    check("split_halves.2", compare(g2, w2), 0.001);
  }
  {
    const std::uint32_t n = 517;  // deliberately not a multiple of the threadgroup
    std::vector<std::uint16_t> val(n), gate(n);
    for (auto& v : val) v = f32_to_f16(dist(rng));
    for (auto& v : gate) v = f32_to_f16(dist(rng) * 4.0f);
    auto bv = ctx.alloc_from(val.data(), n * 2), bg = ctx.alloc_from(gate.data(), n * 2);
    LinGateParams p{n};
    const void* bufs[] = {bv.handle(), bg.handle()};
    ctx.dispatch("cpi_sigmoid_gate_inplace", runtime::MetalContext::Grid::Threads, n, 256, bufs,
                 nullptr, 2, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> want(n), got(n);
    for (std::uint32_t i = 0; i < n; ++i)
      want[i] = f16_to_f32(val[i]) * sigmoidf(f16_to_f32(gate[i]));
    const auto* o = static_cast<const std::uint16_t*>(bv.contents());
    for (std::uint32_t i = 0; i < n; ++i) got[i] = f16_to_f32(o[i]);
    check("sigmoid_gate", compare(got, want), 0.002);
  }
  {
    const std::uint32_t n = 517;
    const float scale = 0.75f;
    std::vector<std::uint16_t> in(n), vec(n);
    for (auto& v : in) v = f32_to_f16(dist(rng));
    for (auto& v : vec) v = f32_to_f16(dist(rng));
    auto bi = ctx.alloc_from(in.data(), n * 2), bvv = ctx.alloc_from(vec.data(), n * 2);
    auto bo = ctx.alloc(n * 2);
    LinMulVecParams p{n, scale};
    const void* bufs[] = {bi.handle(), bvv.handle(), bo.handle()};
    ctx.dispatch("cpi_mul_vec", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr, 3, &p,
                 sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> want(n), got(n);
    for (std::uint32_t i = 0; i < n; ++i)
      want[i] = f16_to_f32(in[i]) * f16_to_f32(vec[i]) * scale;
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    for (std::uint32_t i = 0; i < n; ++i) got[i] = f16_to_f32(o[i]);
    check("mul_vec", compare(got, want), 0.002);
  }
  {
    // Two consecutive steps, so the shifted conv window is checked and not just the first token.
    const std::uint32_t ch = 96, ks = 4, sl = ks - 1;
    std::vector<std::uint16_t> wt(ch * ks), x0(ch), x1(ch);
    std::vector<float> st(ch * sl);
    for (auto& v : wt) v = f32_to_f16(dist(rng) * 0.5f);
    for (auto& v : x0) v = f32_to_f16(dist(rng));
    for (auto& v : x1) v = f32_to_f16(dist(rng));
    for (auto& v : st) v = dist(rng);
    auto bw = ctx.alloc_from(wt.data(), wt.size() * 2);
    auto bst = ctx.alloc_from(st.data(), st.size() * 4);
    auto bx = ctx.alloc_from(x0.data(), ch * 2);
    LinConvParams p{ch, ks};
    const void* bufs[] = {bw.handle(), bst.handle(), bx.handle()};
    auto ref_step = [&](const std::vector<std::uint16_t>& xin, std::vector<float>& state) {
      std::vector<float> o(ch);
      for (std::uint32_t c = 0; c < ch; ++c) {
        const float input = f16_to_f32(xin[c]);
        float acc = f16_to_f32(wt[c * ks + ks - 1]) * input;
        for (std::uint32_t j = 0; j < sl; ++j) acc += f16_to_f32(wt[c * ks + j]) * state[c * sl + j];
        for (std::uint32_t j = 0; j + 1 < sl; ++j) state[c * sl + j] = state[c * sl + j + 1];
        state[c * sl + sl - 1] = input;
        o[c] = siluf(acc);
      }
      return o;
    };
    std::vector<float> ref_state = st;
    const std::vector<float> want0 = ref_step(x0, ref_state);
    ctx.dispatch("cpi_linear_conv1d_silu", runtime::MetalContext::Grid::Threads, ch, 256, bufs,
                 nullptr, 3, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> got0(ch);
    {
      const auto* o = static_cast<const std::uint16_t*>(bx.contents());
      for (std::uint32_t i = 0; i < ch; ++i) got0[i] = f16_to_f32(o[i]);
    }
    check("conv1d_silu", compare(got0, want0), 0.01);

    const std::vector<float> want1 = ref_step(x1, ref_state);
    std::memcpy(bx.contents(), x1.data(), ch * 2);
    ctx.dispatch("cpi_linear_conv1d_silu", runtime::MetalContext::Grid::Threads, ch, 256, bufs,
                 nullptr, 3, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> got1(ch);
    {
      const auto* o = static_cast<const std::uint16_t*>(bx.contents());
      for (std::uint32_t i = 0; i < ch; ++i) got1[i] = f16_to_f32(o[i]);
    }
    check("conv1d_silu.t2", compare(got1, want1), 0.01);
  }
  {
    const std::uint32_t kh = 4, rep_n = 2, kdim = 32, vdim = 48;
    const std::uint32_t vh = kh * rep_n;
    std::vector<std::uint16_t> mix(kh * kdim * 2 + vh * vdim);
    for (auto& v : mix) v = f32_to_f16(dist(rng));
    auto bm = ctx.alloc_from(mix.data(), mix.size() * 2);
    auto bq = ctx.alloc(vh * kdim * 2), bk = ctx.alloc(vh * kdim * 2), bv = ctx.alloc(vh * vdim * 2);
    LinRepeatParams p{kh, rep_n, kdim, vdim};
    const void* bufs[] = {bm.handle(), bq.handle(), bk.handle(), bv.handle()};
    const std::size_t width = std::max(kdim, vdim);
    ctx.dispatch("cpi_repeat_linear_heads", runtime::MetalContext::Grid::Threads, width * kh, 64,
                 bufs, nullptr, 4, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> wq(vh * kdim), wk(vh * kdim), wv(vh * vdim);
    for (std::uint32_t h = 0; h < kh; ++h)
      for (std::uint32_t r = 0; r < rep_n; ++r) {
        const std::uint32_t vhh = h * rep_n + r;
        for (std::uint32_t d = 0; d < kdim; ++d) {
          wq[vhh * kdim + d] = f16_to_f32(mix[h * kdim + d]);
          wk[vhh * kdim + d] = f16_to_f32(mix[kh * kdim + h * kdim + d]);
        }
        for (std::uint32_t d = 0; d < vdim; ++d)
          wv[vhh * vdim + d] =
              f16_to_f32(mix[kh * 2 * kdim + (h * rep_n + r) * vdim + d]);
      }
    auto pull = [&](const runtime::MetalBuffer& b, std::size_t n) {
      std::vector<float> o(n);
      const auto* h = static_cast<const std::uint16_t*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) o[i] = f16_to_f32(h[i]);
      return o;
    };
    check("repeat_heads.q", compare(pull(bq, vh * kdim), wq), 0.001);
    check("repeat_heads.k", compare(pull(bk, vh * kdim), wk), 0.001);
    check("repeat_heads.v", compare(pull(bv, vh * vdim), wv), 0.001);
  }
  {
    // The recurrent step, run TWICE so the carried state is part of what is checked.
    const std::uint32_t H = 3, kdim = 32, vdim = 32, ss = kdim * vdim;
    const float eps = 1e-6f;
    std::vector<std::uint16_t> q(H * kdim), k(H * kdim), v(H * vdim), z(H * vdim), av(H), bv2(H),
        dtb(H);
    std::vector<std::uint16_t> nw(vdim), alog(H);  // fp16: the container holds them that way
    std::vector<float> st(H * ss);
    auto fill = [&](std::vector<std::uint16_t>& x, float s) {
      for (auto& e : x) e = f32_to_f16(dist(rng) * s);
    };
    fill(q, 1.0f); fill(k, 1.0f); fill(v, 1.0f); fill(z, 1.0f);
    fill(av, 1.0f); fill(bv2, 1.0f); fill(dtb, 0.5f);
    for (auto& e : nw) e = f32_to_f16(dist(rng));
    for (auto& e : alog) e = f32_to_f16(dist(rng) * 0.5f);
    for (auto& e : st) e = dist(rng) * 0.1f;

    auto bq2 = ctx.alloc_from(q.data(), q.size()*2), bk2 = ctx.alloc_from(k.data(), k.size()*2);
    auto bv3 = ctx.alloc_from(v.data(), v.size()*2), bz = ctx.alloc_from(z.data(), z.size()*2);
    auto ba = ctx.alloc_from(av.data(), av.size()*2), bb = ctx.alloc_from(bv2.data(), bv2.size()*2);
    auto bnw = ctx.alloc_from(nw.data(), nw.size()*2), bal = ctx.alloc_from(alog.data(), alog.size()*2);
    auto bdt = ctx.alloc_from(dtb.data(), dtb.size()*2), bst2 = ctx.alloc_from(st.data(), st.size()*4);
    auto bo2 = ctx.alloc(H * vdim * 2);
    LinAttnParams p{kdim, vdim, eps};
    const void* bufs[] = {bq2.handle(), bk2.handle(), bv3.handle(), bz.handle(), ba.handle(),
                          bb.handle(), bnw.handle(), bal.handle(), bdt.handle(), bst2.handle(),
                          bo2.handle()};
    auto ref = [&](std::vector<float>& state) {
      std::vector<float> out(H * vdim);
      for (std::uint32_t h = 0; h < H; ++h) {
        float qss = 0, kss = 0;
        for (std::uint32_t i = 0; i < kdim; ++i) {
          qss += f16_to_f32(q[h*kdim+i]) * f16_to_f32(q[h*kdim+i]);
          kss += f16_to_f32(k[h*kdim+i]) * f16_to_f32(k[h*kdim+i]);
        }
        const float qn = 1.0f/std::sqrt(qss+1e-6f)/std::sqrt((float)kdim);
        const float kn = 1.0f/std::sqrt(kss+1e-6f);
        const float beta = sigmoidf(f16_to_f32(bv2[h]));
        const float decay = std::exp(-std::exp(f16_to_f32(alog[h])) *
                                     softplusf(f16_to_f32(av[h]) + f16_to_f32(dtb[h])));
        std::vector<float> qs(kdim), ks(kdim);
        for (std::uint32_t i = 0; i < kdim; ++i) {
          qs[i] = f16_to_f32(q[h*kdim+i]) * qn;
          ks[i] = f16_to_f32(k[h*kdim+i]) * kn;
        }
        float* S = state.data() + h*ss;
        for (std::uint32_t i = 0; i < ss; ++i) S[i] *= decay;
        std::vector<float> delta(vdim);
        for (std::uint32_t d = 0; d < vdim; ++d) {
          float m = 0;
          for (std::uint32_t c = 0; c < kdim; ++c) m += S[c*vdim+d] * ks[c];
          delta[d] = (f16_to_f32(v[h*vdim+d]) - m) * beta;
        }
        for (std::uint32_t c = 0; c < kdim; ++c)
          for (std::uint32_t d = 0; d < vdim; ++d) S[c*vdim+d] += ks[c] * delta[d];
        std::vector<float> o(vdim);
        float oss = 0;
        for (std::uint32_t d = 0; d < vdim; ++d) {
          float sum = 0;
          for (std::uint32_t c = 0; c < kdim; ++c) sum += S[c*vdim+d] * qs[c];
          o[d] = sum; oss += sum*sum;
        }
        const float inv = 1.0f/std::sqrt(oss/(float)vdim + eps);
        for (std::uint32_t d = 0; d < vdim; ++d)
          out[h*vdim+d] = o[d]*inv*f16_to_f32(nw[d])*siluf(f16_to_f32(z[h*vdim+d]));
      }
      return out;
    };
    std::vector<float> rstate = st;
    for (int step = 0; step < 2; ++step) {
      const std::vector<float> want = ref(rstate);
      ctx.dispatch("cpi_linear_attention_step", runtime::MetalContext::Grid::Groups, H, 256, bufs,
                   nullptr, 11, &p, sizeof(p));
      ctx.commit_and_wait();
      std::vector<float> got(H * vdim);
      const auto* o = static_cast<const std::uint16_t*>(bo2.contents());
      for (std::uint32_t i = 0; i < H*vdim; ++i) got[i] = f16_to_f32(o[i]);
      check(step == 0 ? "lin_attn_step" : "lin_attn_step.t2", compare(got, want), 0.02);
    }
  }

  // ---- vision tower ------------------------------------------------------
  //
  // No Gemma 4 checkpoint is converted here either, so the CPU reference is the gate. Every one
  // of these includes PADDING PATCHES (a negative position), because that is the case that fails
  // silently: padding mixed into a real cell is a plausible-looking number, not a crash.
  {
    const std::uint32_t tok = 7, hid = 24, pdim = 12, pts = 8;
    std::vector<std::uint16_t> proj(hid * pdim), ptab(2 * pts * hid);
    std::vector<float> pix(tok * pdim);
    std::vector<std::int32_t> px(tok), py(tok);
    for (auto& v : proj) v = f32_to_f16(dist(rng) * 0.3f);
    for (auto& v : ptab) v = f32_to_f16(dist(rng) * 0.3f);
    for (auto& v : pix) v = 0.5f + dist(rng) * 0.5f;
    for (std::uint32_t t = 0; t < tok; ++t) { px[t] = t % 4; py[t] = t / 4; }
    px[5] = -1; py[6] = -1;  // two padding patches, one per coordinate

    auto bp = ctx.alloc_from(proj.data(), proj.size()*2);
    auto bpix = ctx.alloc_from(pix.data(), pix.size()*4);
    auto bt = ctx.alloc_from(ptab.data(), ptab.size()*2);
    auto bx = ctx.alloc_from(px.data(), px.size()*4), by = ctx.alloc_from(py.data(), py.size()*4);
    auto bo = ctx.alloc(tok * hid * 2);
    VisPatchParams p{hid, pdim, pts, tok};
    const void* bufs[] = {bp.handle(), bpix.handle(), bt.handle(), bx.handle(), by.handle(),
                          bo.handle()};
    ctx.dispatch("cpi_patch_embed", runtime::MetalContext::Grid::Threads, tok * hid, 256, bufs,
                 nullptr, 6, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> want(tok * hid), got(tok * hid);
    for (std::uint32_t t = 0; t < tok; ++t)
      for (std::uint32_t h = 0; h < hid; ++h) {
        if (px[t] < 0 || py[t] < 0) { want[t*hid+h] = 0.0f; continue; }
        float acc = 0.0f;
        for (std::uint32_t k = 0; k < pdim; ++k)
          acc += f16_to_f32(proj[h*pdim+k]) * (2.0f * (pix[t*pdim+k] - 0.5f));
        acc += f16_to_f32(ptab[(std::size_t)px[t]*hid + h]);
        acc += f16_to_f32(ptab[(std::size_t)pts*hid + (std::size_t)py[t]*hid + h]);
        want[t*hid+h] = acc;
      }
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    for (std::uint32_t i = 0; i < tok*hid; ++i) got[i] = f16_to_f32(o[i]);
    check("patch_embed", compare(got, want), 0.02);
  }
  {
    const std::uint32_t tok = 5, heads = 3, hd = 16, hp = hd / 4;
    std::vector<std::uint16_t> x(tok * heads * hd);
    std::vector<float> ct(64 * hp), st(64 * hp);
    std::vector<std::int32_t> px(tok), py(tok);
    for (auto& v : x) v = f32_to_f16(dist(rng));
    for (std::size_t i = 0; i < ct.size(); ++i) { ct[i] = std::cos(0.1f*i); st[i] = std::sin(0.1f*i); }
    for (std::uint32_t t = 0; t < tok; ++t) { px[t] = t; py[t] = (t*2) % 5; }
    const std::vector<std::uint16_t> x0 = x;

    auto bx = ctx.alloc_from(x.data(), x.size()*2);
    auto bpx = ctx.alloc_from(px.data(), px.size()*4), bpy = ctx.alloc_from(py.data(), py.size()*4);
    auto bc = ctx.alloc_from(ct.data(), ct.size()*4), bs = ctx.alloc_from(st.data(), st.size()*4);
    VisRope2DParams p{heads, hd, hp, tok};
    const void* bufs[] = {bx.handle(), bpx.handle(), bpy.handle(), bc.handle(), bs.handle()};
    ctx.dispatch("cpi_rope_2d_inplace", runtime::MetalContext::Grid::Threads,
                 tok * heads * 2 * hp, 256, bufs, nullptr, 5, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> want(x.size()), got(x.size());
    for (std::size_t i = 0; i < x0.size(); ++i) want[i] = f16_to_f32(x0[i]);
    for (std::uint32_t t = 0; t < tok; ++t)
      for (std::uint32_t hh = 0; hh < heads; ++hh)
        for (std::uint32_t a = 0; a < 2; ++a)
          for (std::uint32_t j = 0; j < hp; ++j) {
            const std::size_t row = ((std::size_t)t*heads + hh)*hd;
            const std::uint32_t base = a * (hd/2);
            const std::int32_t pos = a == 0 ? px[t] : py[t];
            const float c = ct[(std::size_t)pos*hp + j], sn = st[(std::size_t)pos*hp + j];
            const float v0 = f16_to_f32(x0[row + base + j]);
            const float v1 = f16_to_f32(x0[row + base + j + hp]);
            want[row + base + j] = v0*c - v1*sn;
            want[row + base + j + hp] = v1*c + v0*sn;
          }
    const auto* o = static_cast<const std::uint16_t*>(bx.contents());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
    check("rope_2d", compare(got, want), 0.005);
  }
  {
    const std::uint32_t tok = 12, hid = 16, k = 2, cx = 2, outt = 4;
    const float gain = 1.5f;
    std::vector<std::uint16_t> in(tok * hid);
    std::vector<std::int32_t> px(tok), py(tok);
    for (auto& v : in) v = f32_to_f16(dist(rng));
    for (std::uint32_t t = 0; t < tok; ++t) { px[t] = t % 4; py[t] = (t / 4) % 4; }
    px[3] = -1; py[9] = -1;  // padding must be skipped but must NOT shrink the divisor

    auto bi = ctx.alloc_from(in.data(), in.size()*2);
    auto bpx = ctx.alloc_from(px.data(), px.size()*4), bpy = ctx.alloc_from(py.data(), py.size()*4);
    auto bo = ctx.alloc(outt * hid * 2);
    VisPoolParams p{tok, hid, k, cx, outt, gain};
    const void* bufs[] = {bi.handle(), bpx.handle(), bpy.handle(), bo.handle()};
    ctx.dispatch("cpi_avg_pool_patches", runtime::MetalContext::Grid::Threads, outt * hid, 256,
                 bufs, nullptr, 4, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> want(outt*hid), got(outt*hid);
    for (std::uint32_t cell = 0; cell < outt; ++cell)
      for (std::uint32_t h = 0; h < hid; ++h) {
        float acc = 0.0f;
        for (std::uint32_t t = 0; t < tok; ++t) {
          if (px[t] < 0 || py[t] < 0) continue;
          const std::uint32_t c = ((std::uint32_t)px[t]/k) + cx*((std::uint32_t)py[t]/k);
          if (c == cell) acc += f16_to_f32(in[(std::size_t)t*hid + h]);
        }
        want[cell*hid+h] = acc / float(k*k) * gain;
      }
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    for (std::uint32_t i = 0; i < outt*hid; ++i) got[i] = f16_to_f32(o[i]);
    check("avg_pool", compare(got, want), 0.01);
  }
  {
    const std::uint32_t tok = 9, hid = 32, n = tok * hid;
    std::vector<std::uint16_t> x(n), bias(hid), scl(hid);
    for (auto& v : x) v = f32_to_f16(dist(rng));
    for (auto& v : bias) v = f32_to_f16(dist(rng) * 0.5f);
    for (auto& v : scl) v = f32_to_f16(0.5f + dist(rng) * 0.25f);
    const std::vector<std::uint16_t> x0 = x;
    auto bx = ctx.alloc_from(x.data(), n*2);
    auto bb = ctx.alloc_from(bias.data(), hid*2), bsc = ctx.alloc_from(scl.data(), hid*2);
    VisStdParams p{n, hid};
    const void* bufs[] = {bx.handle(), bb.handle(), bsc.handle()};
    ctx.dispatch("cpi_standardize", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr, 3,
                 &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> want(n), got(n);
    for (std::uint32_t i = 0; i < n; ++i)
      want[i] = (f16_to_f32(x0[i]) - f16_to_f32(bias[i % hid])) * f16_to_f32(scl[i % hid]);
    const auto* o = static_cast<const std::uint16_t*>(bx.contents());
    for (std::uint32_t i = 0; i < n; ++i) got[i] = f16_to_f32(o[i]);
    check("standardize", compare(got, want), 0.005);
  }

  {
    // LayerNorm, gated against an fp64 reference rather than a rearrangement of the kernel's
    // own algebra -- a reference sharing the kernel's formulation confirms only that it was
    // transcribed twice.
    //
    // TWO row scales, deliberately. Row 1 carries values around 1600, which is what the
    // Qwen3.5 vision tower's last block actually produces. Squared that is ~2.6e6, past
    // fp16's 65504, so this row is what makes the case discriminating: patching the kernel's
    // variance accumulator to half moves it from max_abs 0.0005 to 1.21. A case built only
    // from unit-scale noise passes either way, which is how a tower that runs and is wrong
    // gets shipped.
    //
    // It does NOT discriminate the one-pass E[x^2] - E[x]^2 form: this data is zero-mean, so
    // nothing cancels, and that variant was measured to pass here unchanged. Catching it
    // would need a large-mean/small-variance row -- not added, because the tower's real
    // activations are near zero-mean (max |mean|/stddev 0.5) and a test should gate risks
    // this model actually has.
    const std::uint32_t rows = 4, cols = 768, n = rows * cols;
    std::vector<std::uint16_t> x(n), w(cols), bs(cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
      const float scale = (r == 1) ? 1600.0f : 1.0f;
      for (std::uint32_t c = 0; c < cols; ++c) x[r * cols + c] = f32_to_f16(dist(rng) * scale);
    }
    for (auto& v : w) v = f32_to_f16(0.5f + dist(rng) * 0.25f);
    for (auto& v : bs) v = f32_to_f16(dist(rng) * 0.5f);
    auto bx = ctx.alloc_from(x.data(), n * 2);
    auto bw = ctx.alloc_from(w.data(), cols * 2);
    auto bbias = ctx.alloc_from(bs.data(), cols * 2);
    auto bo = ctx.alloc(n * 2);
    LayerNormParams p{rows, cols, 1e-6f, 1u};
    const void* bufs[] = {bx.handle(), bw.handle(), bbias.handle(), bo.handle()};
    ctx.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups, rows, 256, bufs, nullptr,
                 4, &p, sizeof(p));
    ctx.commit_and_wait();
    std::vector<float> want(n), got(n);
    for (std::uint32_t r = 0; r < rows; ++r) {
      double mean = 0.0;
      for (std::uint32_t c = 0; c < cols; ++c) mean += f16_to_f32(x[r * cols + c]);
      mean /= double(cols);
      double var = 0.0;
      for (std::uint32_t c = 0; c < cols; ++c) {
        const double d = static_cast<double>(f16_to_f32(x[r * cols + c])) - mean;
        var += d * d;
      }
      var /= double(cols);
      const double inv = 1.0 / std::sqrt(var + 1e-6);
      for (std::uint32_t c = 0; c < cols; ++c) {
        want[r * cols + c] =
            static_cast<float>((static_cast<double>(f16_to_f32(x[r * cols + c])) - mean) * inv *
                                   f16_to_f32(w[c]) + f16_to_f32(bs[c]));
      }
    }
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    for (std::uint32_t i = 0; i < n; ++i) got[i] = f16_to_f32(o[i]);
    check("layernorm", compare(got, want), 0.01);
  }

  {
    // Bidirectional attention, against a plain fp64 softmax over ALL keys.
    //
    // tokens deliberately exceeds BIATTN_CHUNK so the online-softmax path runs more than one
    // chunk. With a single chunk the running-max rescale is never exercised and a kernel that
    // drops it entirely still passes -- which was the state of this kernel before the control
    // below caught it.
    const std::uint32_t tokens = 100, heads = 3, hd = 64, dim = heads * hd;
    const std::uint32_t n = tokens * dim;
    std::vector<std::uint16_t> q(n), k(n), v(n);
    for (auto& x : q) x = f32_to_f16(dist(rng));
    for (auto& x : k) x = f32_to_f16(dist(rng));
    for (auto& x : v) x = f32_to_f16(dist(rng));
    auto bq = ctx.alloc_from(q.data(), n * 2), bk = ctx.alloc_from(k.data(), n * 2);
    auto bv = ctx.alloc_from(v.data(), n * 2), bo = ctx.alloc(n * 2);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    BiAttnParams p{tokens, heads, hd, scale};
    const void* bufs[] = {bq.handle(), bk.handle(), bv.handle(), bo.handle()};
    ctx.dispatch("cpi_attention_bidirectional", runtime::MetalContext::Grid::Groups,
                 tokens * heads, 64, bufs, nullptr, 4, &p, sizeof(p));
    ctx.commit_and_wait();

    std::vector<float> want(n), got(n);
    for (std::uint32_t t = 0; t < tokens; ++t) {
      for (std::uint32_t h = 0; h < heads; ++h) {
        const std::uint32_t off = h * hd;
        std::vector<double> sc(tokens);
        double mx = -1e300;
        for (std::uint32_t j = 0; j < tokens; ++j) {  // every key, both directions
          double d = 0.0;
          for (std::uint32_t e = 0; e < hd; ++e)
            d += static_cast<double>(f16_to_f32(q[t * dim + off + e])) *
                 f16_to_f32(k[j * dim + off + e]);
          sc[j] = d * scale;
          mx = std::max(mx, sc[j]);
        }
        double sum = 0.0;
        for (std::uint32_t j = 0; j < tokens; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
        for (std::uint32_t e = 0; e < hd; ++e) {
          double a = 0.0;
          for (std::uint32_t j = 0; j < tokens; ++j)
            a += sc[j] * f16_to_f32(v[j * dim + off + e]);
          want[t * dim + off + e] = static_cast<float>(a / sum);
        }
      }
    }
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    for (std::uint32_t i = 0; i < n; ++i) got[i] = f16_to_f32(o[i]);
    check("bidir_attn", compare(got, want), 0.01);
  }

  std::printf("[metal_smoke] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
