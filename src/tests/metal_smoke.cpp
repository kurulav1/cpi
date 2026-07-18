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

  // ---- GEMM (fp16), the multi-token prefill path -------------------------
  // MUST match GEMM_FBM / GEMM_BN in cpi_kernels.metal (and kGemmFBM / kGemmBN in the engine).
  constexpr std::uint32_t kSmokeGemmFBM = 64;   // rows per threadgroup
  constexpr std::uint32_t kSmokeGemmBN = 64;    // tokens per tile
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

  std::printf("[metal_smoke] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
