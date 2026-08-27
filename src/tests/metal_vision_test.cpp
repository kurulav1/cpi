// Gates the Qwen3.5 vision tower's front end against the HuggingFace oracle, stage by stage.
//
//   ./metal_vision_test <oracle_dir>      (oracle_dir = tools/qwen35_vision_oracle.py --out)
//
// Compares the patch embedding, the interpolated position table, and their sum against
// stage_01/02/03 of the reference dump. Splitting them matters: the patch embed is a plain
// GEMM and the position table is fiddly index arithmetic, so a combined check would leave two
// suspects for one failure.
//
// The position embedding is computed on the HOST, deliberately. It is a bilinear resample of a
// 48x48 learned table plus a reordering, costing O(patches * hidden) once per image, against
// twelve transformer blocks costing O(patches^2 * dim) each. Putting it in a kernel would add a
// bespoke gather to the surface area that can be silently wrong, and buy nothing measurable.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/plan_metal_engine.hpp"
#include "model/config_json.hpp"
#include "model/json_mini.hpp"
#include "model/safetensors_loader.hpp"
#include "model/weight_loader.hpp"
#include "runtime/fp16.hpp"
#include "runtime/metal_context.hpp"

namespace {

int failures = 0;

std::vector<float> read_f32(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<float> v(static_cast<std::size_t>(bytes) / sizeof(float));
  if (std::fread(v.data(), 1, static_cast<std::size_t>(bytes), f) !=
      static_cast<std::size_t>(bytes)) {
    std::fprintf(stderr, "short read on %s\n", path.c_str());
    std::exit(2);
  }
  std::fclose(f);
  return v;
}

std::string read_text(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::string s;
  char buf[4096];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

// Was a local truncating copy. That mattered here specifically: this harness is the reference
// the engine is compared against, and the container it is compared to was packed by pack_ll2c.py
// with numpy's IEEE rounding. Truncating made END_TO_END look better than it was; 0.00257
// against 0.00383 with a correct converter; and that flattering 1.49x was half the apparent
// engine-vs-harness gap. See include/runtime/fp16.hpp.
inline std::uint16_t f32_to_f16(float f) {
  return cpi::f32_to_f16(f);
}
inline float f16_to_f32(std::uint16_t h) {
  return cpi::f16_to_f32(h);
}

// Reports max and mean absolute error. Both, because a mean alone hides a single wrong element
// among thousands of right ones; which is what an off-by-one in a permutation looks like.
void check(const char* name, const std::vector<float>& got, const std::vector<float>& want,
           float tol) {
  if (got.size() != want.size()) {
    std::printf("  %-22s SIZE MISMATCH got=%zu want=%zu  FAIL\n", name, got.size(), want.size());
    ++failures;
    return;
  }
  // NaN must be caught explicitly. std::max(x, NaN) returns x; every comparison with NaN is
  // false; so an all-NaN result accumulates a max of 0 and passes every tolerance in this
  // file. That is not hypothetical: the engine path returned NaN and this reported pass.
  std::size_t nans = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i])) ++nans;
  }
  if (nans != 0) {
    std::printf("  %-22s %zu/%zu non-finite values  FAIL\n", name, nans, got.size());
    ++failures;
    return;
  }
  float mx = 0.0f;
  double sum = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float d = std::fabs(got[i] - want[i]);
    mx = std::max(mx, d);
    sum += d;
  }
  const double mean = sum / static_cast<double>(got.size());
  const bool ok = mx <= tol;
  std::printf("  %-22s max_abs=%.6f  mean_abs=%.6f  tol=%.3f  %s\n", name, mx, mean, tol,
              ok ? "PASS" : "FAIL");
  if (!ok) ++failures;
}

struct GemvParams {
  std::uint32_t out_dim, in_dim, tokens, has_bias;
};
struct ElemParams {
  std::uint32_t n;
  float scale;
};

// Must match plan_metal_engine.cpp / the shader. cpi_gemm_f16 maps tgid -> (row block, token
// tile), so the grid has to be derived the same way here or the launch silently covers the
// wrong rows.
constexpr int kGemmBN = 32;
constexpr int kGemmFBM = 64;
constexpr int kGemmRF = 4;
constexpr int kGemmCF = 4;
constexpr int kGemmTG = 32 * (kGemmFBM / (8 * kGemmRF)) * (kGemmBN / (8 * kGemmCF));

// Mirrors Qwen3_5VisionModel.fast_pos_embed_interpolate.
//
// Two things here are easy to get wrong and neither announces itself:
//
//   1. The sample positions are linspace(0, side-1, n); endpoint inclusive. Using n even
//      steps (i * side / n) instead of n - 1 (i * (side-1) / (n-1)) shifts every sample and
//      still produces a smooth, plausible table.
//
//   2. The result is reordered into merge-unit-major order before being returned, so patch
//      (i, j) does not land at row i*w + j. The blocks therefore see patches grouped by the
//      2x2 unit the merger will later fold, not in raster order.
std::vector<float> interpolate_pos_embed(const std::vector<float>& table, int side, int h, int w,
                                         int hidden, int merge) {
  auto axis = [](int n, int s) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      v[static_cast<std::size_t>(i)] =
          (n > 1) ? static_cast<float>(i) * static_cast<float>(s - 1) / static_cast<float>(n - 1)
                  : 0.0f;
    }
    return v;
  };
  const std::vector<float> hi = axis(h, side);
  const std::vector<float> wi = axis(w, side);

  std::vector<float> raster(static_cast<std::size_t>(h) * w * hidden, 0.0f);
  for (int i = 0; i < h; ++i) {
    const int fh = static_cast<int>(hi[static_cast<std::size_t>(i)]);
    const int ch = std::min(fh + 1, side - 1);
    const float dh = hi[static_cast<std::size_t>(i)] - static_cast<float>(fh);
    for (int j = 0; j < w; ++j) {
      const int fw = static_cast<int>(wi[static_cast<std::size_t>(j)]);
      const int cw = std::min(fw + 1, side - 1);
      const float dw = wi[static_cast<std::size_t>(j)] - static_cast<float>(fw);
      const int idx[4] = {fh * side + fw, fh * side + cw, ch * side + fw, ch * side + cw};
      const float wt[4] = {(1.0f - dh) * (1.0f - dw), (1.0f - dh) * dw, dh * (1.0f - dw), dh * dw};
      float* dst = raster.data() + (static_cast<std::size_t>(i) * w + j) * hidden;
      for (int c = 0; c < 4; ++c) {
        const float* src = table.data() + static_cast<std::size_t>(idx[c]) * hidden;
        for (int d = 0; d < hidden; ++d) dst[d] += wt[c] * src[d];
      }
    }
  }

  // Merge-unit-major reorder: (h/m, m, w/m, m, C) -> (h/m, w/m, m, m, C).
  std::vector<float> out(raster.size());
  std::size_t o = 0;
  for (int bh = 0; bh < h / merge; ++bh) {
    for (int bw = 0; bw < w / merge; ++bw) {
      for (int mi = 0; mi < merge; ++mi) {
        for (int mj = 0; mj < merge; ++mj) {
          const std::size_t src =
              (static_cast<std::size_t>(bh * merge + mi) * w + (bw * merge + mj)) * hidden;
          std::memcpy(out.data() + o, raster.data() + src, sizeof(float) * hidden);
          o += static_cast<std::size_t>(hidden);
        }
      }
    }
  }
  return out;
}

struct VisionBlockWeights {
  std::vector<float> norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b;
  std::vector<float> norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b;
};
struct LayerNormParams {
  std::uint32_t rows, cols;
  float eps;
  std::uint32_t has_bias;
};
struct BiAttnParams {
  std::uint32_t tokens, heads, head_dim;
  float scale;
  std::uint32_t row_stride;
};
struct VisRopeParams {
  std::uint32_t tokens, heads, head_dim, row_stride;
};

std::vector<std::uint16_t> to_f16(const std::vector<float>& v) {
  std::vector<std::uint16_t> h(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) h[i] = f32_to_f16(v[i]);
  return h;
}

// One Qwen3_5VisionBlock:
//   x += proj(attn(rope(qkv(LN1(x)))));  x += fc2(gelu(fc1(LN2(x))))
//
// Pre-norm on both halves, and the residual is the unnormalised x; writing the norm back into
// x instead would still train-shaped-run and be wrong.
struct BlockDebug {
  // qkv_preroped is captured before the rotation, because that is what the reference's hook on
  // attn.qkv sees. Comparing the rotated buffer against it reports a failure for q and k that
  // is not a failure; which it did, and cost a bisect step.
  // attn_out is captured after the output projection, because the reference's hook is on the
  // whole attention MODULE and proj is its last step. Capturing the pre-proj value and
  // comparing it to that hook reports a failure that is not one; the second time this exact
  // mistake was made in this file, after the pre/post-rope one above.
  std::vector<float> norm1, qkv_preroped, norm2, attn_out, inter;
};

std::vector<float> run_vision_block(runtime::MetalContext& ctx, const std::vector<float>& x_in,
                                    const VisionBlockWeights& w, const std::vector<float>& cosb,
                                    const std::vector<float>& sinb, int tokens, int hidden,
                                    int heads, int head_dim, int inter, BlockDebug* dbg = nullptr) {
  const std::size_t n = static_cast<std::size_t>(tokens) * hidden;
  auto gemm = [&](runtime::MetalBuffer& bw, runtime::MetalBuffer& bin, runtime::MetalBuffer& bout,
                  runtime::MetalBuffer& bbias, int out_dim, int in_dim) {
    GemvParams p{static_cast<std::uint32_t>(out_dim), static_cast<std::uint32_t>(in_dim),
                 static_cast<std::uint32_t>(tokens), 1u};
    const void* bufs[] = {bw.handle(), bin.handle(), bout.handle(), bbias.handle()};
    const std::size_t tiles = static_cast<std::size_t>((tokens + kGemmBN - 1) / kGemmBN);
    const std::size_t groups = (static_cast<std::size_t>(out_dim) / kGemmFBM) * tiles;
    ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, groups, kGemmTG, bufs,
                 nullptr, 4, &p, sizeof(p));
  };

  auto grab = [&](runtime::MetalBuffer& b, std::size_t count) {
    const auto* q = static_cast<const std::uint16_t*>(b.contents());
    std::vector<float> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = f16_to_f32(q[i]);
    return v;
  };
  auto hx = to_f16(x_in);
  auto bx = ctx.alloc_from(hx.data(), hx.size() * 2);  // residual stream
  auto bnorm = ctx.alloc(n * 2);
  auto bqkv = ctx.alloc(n * 3 * 2);
  auto battn = ctx.alloc(n * 2);
  auto bproj = ctx.alloc(n * 2);
  auto binter = ctx.alloc(static_cast<std::size_t>(tokens) * inter * 2);

  auto h_n1w = to_f16(w.norm1_w), h_n1b = to_f16(w.norm1_b);
  auto h_qkvw = to_f16(w.qkv_w), h_qkvb = to_f16(w.qkv_b);
  auto h_pw = to_f16(w.proj_w), h_pb = to_f16(w.proj_b);
  auto h_n2w = to_f16(w.norm2_w), h_n2b = to_f16(w.norm2_b);
  auto h_f1w = to_f16(w.fc1_w), h_f1b = to_f16(w.fc1_b);
  auto h_f2w = to_f16(w.fc2_w), h_f2b = to_f16(w.fc2_b);
  auto bn1w = ctx.alloc_from(h_n1w.data(), h_n1w.size() * 2);
  auto bn1b = ctx.alloc_from(h_n1b.data(), h_n1b.size() * 2);
  auto bqw = ctx.alloc_from(h_qkvw.data(), h_qkvw.size() * 2);
  auto bqb = ctx.alloc_from(h_qkvb.data(), h_qkvb.size() * 2);
  auto bpw = ctx.alloc_from(h_pw.data(), h_pw.size() * 2);
  auto bpb = ctx.alloc_from(h_pb.data(), h_pb.size() * 2);
  auto bn2w = ctx.alloc_from(h_n2w.data(), h_n2w.size() * 2);
  auto bn2b = ctx.alloc_from(h_n2b.data(), h_n2b.size() * 2);
  auto bf1w = ctx.alloc_from(h_f1w.data(), h_f1w.size() * 2);
  auto bf1b = ctx.alloc_from(h_f1b.data(), h_f1b.size() * 2);
  auto bf2w = ctx.alloc_from(h_f2w.data(), h_f2w.size() * 2);
  auto bf2b = ctx.alloc_from(h_f2b.data(), h_f2b.size() * 2);
  auto bcos = ctx.alloc_from(cosb.data(), cosb.size() * sizeof(float));
  auto bsin = ctx.alloc_from(sinb.data(), sinb.size() * sizeof(float));

  // ---- attention half ----
  {
    LayerNormParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(hidden), 1e-6f,
                      1u};
    const void* bufs[] = {bx.handle(), bn1w.handle(), bn1b.handle(), bnorm.handle()};
    ctx.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups,
                 static_cast<std::size_t>(tokens), 256, bufs, nullptr, 4, &p, sizeof(p));
  }
  gemm(bqw, bnorm, bqkv, bqb, hidden * 3, hidden);
  if (dbg != nullptr) {
    ctx.commit_and_wait();  // debug path only: splits the pass so these can be read back
    dbg->norm1 = grab(bnorm, n);
    dbg->qkv_preroped = grab(bqkv, n * 3);
  }

  // qkv is [3][heads][head_dim] per token, so q, k and v are contiguous blocks of `hidden`.
  // Rotate q and k in place at their own offsets; row_stride is 3*hidden because they live
  // inside the fused block.
  {
    VisRopeParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(heads),
                    static_cast<std::uint32_t>(head_dim), static_cast<std::uint32_t>(hidden * 3)};
    const std::size_t work = static_cast<std::size_t>(tokens) * heads * (head_dim / 2);
    for (int part = 0; part < 2; ++part) {  // 0 = q, 1 = k; v is not rotated
      const void* bufs[] = {bqkv.handle(), bcos.handle(), bsin.handle()};
      const std::size_t offs[] = {static_cast<std::size_t>(part) * hidden * 2, 0, 0};
      ctx.dispatch("cpi_rope_vision", runtime::MetalContext::Grid::Threads, work, 256, bufs, offs,
                   3, &p, sizeof(p));
    }
  }
  {
    BiAttnParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(heads),
                   static_cast<std::uint32_t>(head_dim),
                   1.0f / std::sqrt(static_cast<float>(head_dim)),
                   static_cast<std::uint32_t>(hidden * 3)};
    // q, k, v at offsets 0, hidden, 2*hidden; but the kernel strides by heads*head_dim, which
    // is `hidden`, so each needs its own base offset into the fused buffer.
    const void* bufs[] = {bqkv.handle(), bqkv.handle(), bqkv.handle(), battn.handle()};
    const std::size_t offs[] = {0, static_cast<std::size_t>(hidden) * 2,
                                static_cast<std::size_t>(hidden) * 4, 0};
    ctx.dispatch("cpi_attention_bidirectional", runtime::MetalContext::Grid::Groups,
                 static_cast<std::size_t>(tokens) * heads, 64, bufs, offs, 4, &p, sizeof(p));
  }
  gemm(bpw, battn, bproj, bpb, hidden, hidden);
  if (dbg != nullptr) {
    ctx.commit_and_wait();
    dbg->attn_out = grab(bproj, n);
  }
  {
    ElemParams p{static_cast<std::uint32_t>(n), 1.0f};
    const void* bufs[] = {bproj.handle(), bx.handle()};
    ctx.dispatch("cpi_add_inplace", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr, 2,
                 &p, sizeof(p));
  }

  // ---- MLP half ----
  {
    LayerNormParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(hidden), 1e-6f,
                      1u};
    const void* bufs[] = {bx.handle(), bn2w.handle(), bn2b.handle(), bnorm.handle()};
    ctx.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups,
                 static_cast<std::size_t>(tokens), 256, bufs, nullptr, 4, &p, sizeof(p));
  }
  gemm(bf1w, bnorm, binter, bf1b, inter, hidden);
  {
    ElemParams p{static_cast<std::uint32_t>(tokens) * static_cast<std::uint32_t>(inter), 1.0f};
    const void* bufs[] = {binter.handle()};
    ctx.dispatch("cpi_gelu", runtime::MetalContext::Grid::Threads,
                 static_cast<std::size_t>(tokens) * inter, 256, bufs, nullptr, 1, &p, sizeof(p));
  }
  gemm(bf2w, binter, bproj, bf2b, hidden, inter);
  {
    ElemParams p{static_cast<std::uint32_t>(n), 1.0f};
    const void* bufs[] = {bproj.handle(), bx.handle()};
    ctx.dispatch("cpi_add_inplace", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr, 2,
                 &p, sizeof(p));
  }

  ctx.commit_and_wait();
  if (dbg != nullptr) {
    dbg->norm2 = grab(bnorm, n);  // norm2 overwrote norm1 in the same buffer
    dbg->inter = grab(binter, static_cast<std::size_t>(tokens) * inter);
  }
  return grab(bx, n);
}

// The patch merger: LayerNorm per patch, fold each 2x2 unit into one row, fc1 -> exact GELU ->
// fc2. The fold needs no data movement; patches are already in merge-unit-major order, so
// four consecutive rows are one unit and [tokens][hidden] reinterprets as [tokens/4][4*hidden].
std::vector<float> run_merger(runtime::MetalContext& ctx, const std::vector<float>& in,
                              const std::string& dir, int tokens, int hidden, int merge,
                              int inter_m, int out_hidden, int soft) {
  const std::string mp = dir + "/weights/merger_";
  auto hx = to_f16(in);
  auto bx = ctx.alloc_from(hx.data(), hx.size() * 2);
  auto bn = ctx.alloc(hx.size() * 2);
  auto bi = ctx.alloc(static_cast<std::size_t>(soft) * inter_m * 2);
  auto bo = ctx.alloc(static_cast<std::size_t>(soft) * out_hidden * 2);
  auto up = [&](const std::vector<float>& v) {
    auto h = to_f16(v);
    return ctx.alloc_from(h.data(), h.size() * 2);
  };
  auto bnw = up(read_f32(mp + "norm_weight.f32")), bnb = up(read_f32(mp + "norm_bias.f32"));
  auto b1w = up(read_f32(mp + "linear_fc1_weight.f32"));
  auto b1b = up(read_f32(mp + "linear_fc1_bias.f32"));
  auto b2w = up(read_f32(mp + "linear_fc2_weight.f32"));
  auto b2b = up(read_f32(mp + "linear_fc2_bias.f32"));
  {
    LayerNormParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(hidden), 1e-6f,
                      1u};
    const void* bufs[] = {bx.handle(), bnw.handle(), bnb.handle(), bn.handle()};
    ctx.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups,
                 static_cast<std::size_t>(tokens), 256, bufs, nullptr, 4, &p, sizeof(p));
  }
  auto mgemm = [&](runtime::MetalBuffer& w2, runtime::MetalBuffer& in2, runtime::MetalBuffer& out2,
                   runtime::MetalBuffer& b2, int od, int idim) {
    GemvParams p{static_cast<std::uint32_t>(od), static_cast<std::uint32_t>(idim),
                 static_cast<std::uint32_t>(soft), 1u};
    const void* bufs[] = {w2.handle(), in2.handle(), out2.handle(), b2.handle()};
    const std::size_t tiles = static_cast<std::size_t>((soft + kGemmBN - 1) / kGemmBN);
    const std::size_t groups = (static_cast<std::size_t>(od) / kGemmFBM) * tiles;
    ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, groups, kGemmTG, bufs,
                 nullptr, 4, &p, sizeof(p));
  };
  mgemm(b1w, bn, bi, b1b, inter_m, inter_m);
  {
    ElemParams p{static_cast<std::uint32_t>(soft) * static_cast<std::uint32_t>(inter_m), 1.0f};
    const void* bufs[] = {bi.handle()};
    // erf, not tanh; the merger's nn.GELU() defaults to approximate='none'.
    ctx.dispatch("cpi_gelu_erf", runtime::MetalContext::Grid::Threads,
                 static_cast<std::size_t>(soft) * inter_m, 256, bufs, nullptr, 1, &p, sizeof(p));
  }
  mgemm(b2w, bi, bo, b2b, out_hidden, inter_m);
  ctx.commit_and_wait();
  const auto* o = static_cast<const std::uint16_t*>(bo.contents());
  std::vector<float> out(static_cast<std::size_t>(soft) * out_hidden);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = f16_to_f32(o[i]);
  return out;
}

// Per-token cos/sin for the vision RoPE, half a head_dim wide (cos[i] == cos[i+half], so only
// the first half is stored).
//
// The frequency vector is [row_freqs | col_freqs], each head_dim/4 long; not one geometric
// series over head_dim/2. And the per-token (row, col) must be enumerated in the same
// merge-unit-major order the position table uses, or the rotation is applied to the wrong
// patches while still looking like a well-formed rotation.
void build_vision_rope(int h, int w, int head_dim, int merge, float theta, std::vector<float>& cosb,
                       std::vector<float>& sinb) {
  const int half = head_dim / 2;
  const int quarter = head_dim / 4;
  const int tokens = h * w;
  cosb.assign(static_cast<std::size_t>(tokens) * half, 0.0f);
  sinb.assign(static_cast<std::size_t>(tokens) * half, 0.0f);
  int idx = 0;
  for (int bh = 0; bh < h / merge; ++bh) {
    for (int bw = 0; bw < w / merge; ++bw) {
      for (int mi = 0; mi < merge; ++mi) {
        for (int mj = 0; mj < merge; ++mj, ++idx) {
          const int row = bh * merge + mi;
          const int col = bw * merge + mj;
          for (int j = 0; j < quarter; ++j) {
            const float inv =
                1.0f / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(half));
            const float ar = static_cast<float>(row) * inv;
            const float ac = static_cast<float>(col) * inv;
            cosb[static_cast<std::size_t>(idx) * half + j] = std::cos(ar);
            sinb[static_cast<std::size_t>(idx) * half + j] = std::sin(ar);
            cosb[static_cast<std::size_t>(idx) * half + quarter + j] = std::cos(ac);
            sinb[static_cast<std::size_t>(idx) * half + quarter + j] = std::sin(ac);
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: metal_vision_test <oracle_dir>\n");
    return 2;
  }
  // Unbuffered: this test can crash the process (it drives a GPU and mmaps a multi-GB
  // container), and a buffered stdout loses every line printed before the fault; which reads
  // as "it produced no output" rather than "it died at stage N".
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  const std::string dir = argv[1];
  const std::string manifest = read_text(dir + "/manifest.json");
  const std::string geo = engine::mini::json_extract_object(manifest, "geometry");
  const int hidden = engine::mini::json_get_int(geo, "hidden_size");
  const int grid_h = engine::mini::json_get_int(geo, "grid_h");
  const int grid_w = engine::mini::json_get_int(geo, "grid_w");
  const int side = engine::mini::json_get_int(geo, "num_grid_per_side");
  const int merge = engine::mini::json_get_int(geo, "spatial_merge_size");
  const int in_ch = engine::mini::json_get_int(geo, "in_channels");
  const int psize = engine::mini::json_get_int(geo, "patch_size");
  const int tsize = engine::mini::json_get_int(geo, "temporal_patch_size");
  const int tokens = grid_h * grid_w;
  const int patch_dim = in_ch * tsize * psize * psize;

  std::printf("[metal_vision] grid=%dx%d tokens=%d hidden=%d patch_dim=%d\n", grid_h, grid_w,
              tokens, hidden, patch_dim);

  runtime::MetalContext ctx;
  if (!ctx.available()) {
    std::printf("[metal_vision] SKIP: %s\n", ctx.last_error().c_str());
    return 0;
  }
  // Without this every dispatch below is a silent no-op: the buffers keep whatever they were
  // uploaded with, and the comparison fails with numbers that look like a wrong kernel rather
  // than a missing library; hence the explicit failure here.
  if (!ctx.load_library()) {
    std::printf("[metal_vision] FAIL: %s\n", ctx.last_error().c_str());
    return 1;
  }

  const std::vector<float> pixels = read_f32(dir + "/stage_00_input_pixels.f32");
  const std::vector<float> want_pe = read_f32(dir + "/stage_01_patch_embed.f32");
  const std::vector<float> want_pos = read_f32(dir + "/stage_02_pos_embed.f32");
  const std::vector<float> want_sum = read_f32(dir + "/stage_03_pos_embed_added.f32");
  const std::vector<float> w_proj = read_f32(dir + "/weights/patch_embed_proj_weight.f32");
  const std::vector<float> b_proj = read_f32(dir + "/weights/patch_embed_proj_bias.f32");
  const std::vector<float> pos_tab = read_f32(dir + "/weights/pos_embed_weight.f32");

  std::vector<float> our_patch_embed;  // filled by stage 1, consumed by the end-to-end run
  std::vector<float> our_sum;

  // ---- stage 1: patch embed ----
  // Conv3d with stride == kernel_size over an input reshaped so each sample is one patch, so
  // it reduces to Linear(patch_dim -> hidden). The [768,3,2,16,16] weight is already contiguous
  // in exactly that order, so it needs no rearrangement; only reinterpretation.
  {
    std::vector<std::uint16_t> hw(w_proj.size()), hx(pixels.size()), hb(b_proj.size());
    for (std::size_t i = 0; i < w_proj.size(); ++i) hw[i] = f32_to_f16(w_proj[i]);
    for (std::size_t i = 0; i < pixels.size(); ++i) hx[i] = f32_to_f16(pixels[i]);
    for (std::size_t i = 0; i < b_proj.size(); ++i) hb[i] = f32_to_f16(b_proj[i]);
    auto bw = ctx.alloc_from(hw.data(), hw.size() * 2);
    auto bx = ctx.alloc_from(hx.data(), hx.size() * 2);
    auto bb = ctx.alloc_from(hb.data(), hb.size() * 2);
    auto bo = ctx.alloc(static_cast<std::size_t>(tokens) * hidden * 2);
    GemvParams p{static_cast<std::uint32_t>(hidden), static_cast<std::uint32_t>(patch_dim),
                 static_cast<std::uint32_t>(tokens), 1u};
    const void* bufs[] = {bw.handle(), bx.handle(), bo.handle(), bb.handle()};
    const std::size_t tiles = static_cast<std::size_t>((tokens + kGemmBN - 1) / kGemmBN);
    const std::size_t groups = (static_cast<std::size_t>(hidden) / kGemmFBM) * tiles;
    ctx.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, groups, kGemmTG, bufs,
                 nullptr, 4, &p, sizeof(p));
    ctx.commit_and_wait();
    const auto* o = static_cast<const std::uint16_t*>(bo.contents());
    std::vector<float> got(static_cast<std::size_t>(tokens) * hidden);
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
    // Looser than the smoke kernels: this is a 1536-deep fp16 dot product against an fp32
    // reference, so the tolerance has to cover honest accumulation error.
    std::printf("    got[0..3] = %.4f %.4f %.4f %.4f\n", got[0], got[1], got[2], got[3]);
    std::printf("    want[0..3]= %.4f %.4f %.4f %.4f\n", want_pe[0], want_pe[1], want_pe[2],
                want_pe[3]);
    our_patch_embed = got;
    check("patch_embed", got, want_pe, 0.05f);
  }

  // ---- stage 2: interpolated position table (host) ----
  const std::vector<float> pos =
      interpolate_pos_embed(pos_tab, side, grid_h, grid_w, hidden, merge);
  check("pos_embed", pos, want_pos, 1e-4f);

  // ---- stage 3: their sum ----
  {
    std::vector<std::uint16_t> ha(want_pe.size()), hb(pos.size());
    for (std::size_t i = 0; i < want_pe.size(); ++i) ha[i] = f32_to_f16(want_pe[i]);
    for (std::size_t i = 0; i < pos.size(); ++i) hb[i] = f32_to_f16(pos[i]);
    auto ba = ctx.alloc_from(ha.data(), ha.size() * 2);
    auto bb = ctx.alloc_from(hb.data(), hb.size() * 2);
    ElemParams p{static_cast<std::uint32_t>(ha.size()), 1.0f};
    // cpi_add_inplace is (in, out) and writes buffer 1, so the accumulator must be second.
    const void* bufs[] = {bb.handle(), ba.handle()};
    ctx.dispatch("cpi_add_inplace", runtime::MetalContext::Grid::Threads, ha.size(), 256, bufs,
                 nullptr, 2, &p, sizeof(p));
    ctx.commit_and_wait();
    const auto* o = static_cast<const std::uint16_t*>(ba.contents());
    std::vector<float> got(ha.size());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
    // Fed the oracle's patch embed, not ours, so this isolates the add and the position table
    // from any error already measured in stage 1.
    check("pos_embed_added", got, want_sum, 0.02f);
    // Our own sum, from our own patch embed; the stage check above deliberately used the
    // oracle's, to isolate the add. This is what the end-to-end run starts from.
    our_sum.resize(our_patch_embed.size());
    for (std::size_t i = 0; i < our_sum.size(); ++i) our_sum[i] = our_patch_embed[i] + pos[i];
  }

  // ---- stage 4: one full transformer block ----
  //
  // Fed the oracle's stage 3, so this measures the block alone rather than accumulating the
  // front end's error. If it matches, the same code runs 12 times for the rest of the tower.
  {
    const int heads = engine::mini::json_get_int(geo, "num_heads");
    const int head_dim = engine::mini::json_get_int(geo, "head_dim");
    const int inter = engine::mini::json_get_int(geo, "intermediate_size");
    const std::vector<float> want_b0 = read_f32(dir + "/stage_04_block_00.f32");
    const std::string wp = dir + "/weights/blocks_0_";

    std::vector<float> h_x = want_sum;  // block input (oracle's, to isolate block 0)
    VisionBlockWeights bw{
        read_f32(wp + "norm1_weight.f32"),          read_f32(wp + "norm1_bias.f32"),
        read_f32(wp + "attn_qkv_weight.f32"),       read_f32(wp + "attn_qkv_bias.f32"),
        read_f32(wp + "attn_proj_weight.f32"),      read_f32(wp + "attn_proj_bias.f32"),
        read_f32(wp + "norm2_weight.f32"),          read_f32(wp + "norm2_bias.f32"),
        read_f32(wp + "mlp_linear_fc1_weight.f32"), read_f32(wp + "mlp_linear_fc1_bias.f32"),
        read_f32(wp + "mlp_linear_fc2_weight.f32"), read_f32(wp + "mlp_linear_fc2_bias.f32")};

    std::vector<float> cosb, sinb;
    build_vision_rope(grid_h, grid_w, head_dim, merge, 10000.0f, cosb, sinb);

    BlockDebug dbg;
    const std::vector<float> got =
        run_vision_block(ctx, h_x, bw, cosb, sinb, tokens, hidden, heads, head_dim, inter, &dbg);
    check("b0_norm1", dbg.norm1, read_f32(dir + "/sub_b0_norm1.f32"), 0.05f);
    check("b0_qkv", dbg.qkv_preroped, read_f32(dir + "/sub_b0_qkv.f32"), 0.05f);
    check("b0_attn", dbg.attn_out, read_f32(dir + "/sub_b0_attn.f32"), 0.05f);
    check("b0_norm2", dbg.norm2, read_f32(dir + "/sub_b0_norm2.f32"), 0.05f);
    check("block_00", got, want_b0, 0.05f);

    // ---- the remaining 11 blocks ----
    //
    // Chained: block N is fed the port's own output for N-1, not the oracle's. That is the
    // point; per-block checks against the reference would hide error that only shows up once
    // it compounds, and the tower has to survive its own output twelve times over.
    const int depth = engine::mini::json_get_int(geo, "depth");
    std::vector<float> cur = got;
    for (int L = 1; L < depth; ++L) {
      const std::string p2 = dir + "/weights/blocks_" + std::to_string(L) + "_";
      VisionBlockWeights bl{
          read_f32(p2 + "norm1_weight.f32"),          read_f32(p2 + "norm1_bias.f32"),
          read_f32(p2 + "attn_qkv_weight.f32"),       read_f32(p2 + "attn_qkv_bias.f32"),
          read_f32(p2 + "attn_proj_weight.f32"),      read_f32(p2 + "attn_proj_bias.f32"),
          read_f32(p2 + "norm2_weight.f32"),          read_f32(p2 + "norm2_bias.f32"),
          read_f32(p2 + "mlp_linear_fc1_weight.f32"), read_f32(p2 + "mlp_linear_fc1_bias.f32"),
          read_f32(p2 + "mlp_linear_fc2_weight.f32"), read_f32(p2 + "mlp_linear_fc2_bias.f32")};
      cur = run_vision_block(ctx, cur, bl, cosb, sinb, tokens, hidden, heads, head_dim, inter);
      // Per-block relative error, reported not gated. Smooth growth is fp16 accumulation; a
      // jump at one block is a bug in that block. Absolute error alone cannot tell them apart
      // once the activations span three orders of magnitude across the stack.
      const std::vector<float> ref =
          read_f32(dir + "/stage_" + (L + 4 < 10 ? std::string("0") : std::string("")) +
                   std::to_string(L + 4) + "_block_" +
                   (L < 10 ? std::string("0") : std::string("")) + std::to_string(L) + ".f32");
      float mx = 0.0f, ref_mx = 0.0f;
      for (std::size_t i = 0; i < cur.size(); ++i) {
        mx = std::max(mx, std::fabs(cur[i] - ref[i]));
        ref_mx = std::max(ref_mx, std::fabs(ref[i]));
      }
      std::printf("      block %02d: max_abs=%9.4f  |ref|max=%9.4f  rel=%.5f\n", L, mx, ref_mx,
                  mx / (ref_mx + 1e-9f));
    }
    // Gated on relative error, not absolute. The stack's activations span 8 -> 1657 across
    // blocks, so one absolute tolerance is either meaningless early or unenforceable late.
    //
    // The threshold is derived, not chosen to pass. Rounding the reference to fp16 and back
    // representation error with no arithmetic at all; costs max_abs 0.48 on this tensor. The
    // port lands at 10.06, which is 21x that floor, and the mean ratio is 24x. Those agreeing
    // is the signal: a real bug puts a few elements far out, giving a high max ratio against a
    // low mean one. The per-block curve above says the same thing; relative error is flat at
    // ~0.0005 for blocks 1-10 and only moves when block 11's activations jump 34x.
    //
    // 0.02 sits ~3x above the measured 0.006 and far below anything a bug produced here: the
    // fused-qkv stride bug measured 0.45 relative, the causal-mask control 0.99.
    const std::vector<float> ref11 = read_f32(dir + "/stage_15_block_11.f32");
    float mx = 0.0f, ref_mx = 0.0f;
    for (std::size_t i = 0; i < cur.size(); ++i) {
      mx = std::max(mx, std::fabs(cur[i] - ref11[i]));
      ref_mx = std::max(ref_mx, std::fabs(ref11[i]));
    }
    std::size_t nan_c = 0;
    for (float v : cur) {
      if (!std::isfinite(v)) ++nan_c;
    }
    const float rel = (nan_c != 0) ? 1e9f : mx / (ref_mx + 1e-9f);
    const bool ok = rel <= 0.02f;
    std::printf("  %-22s rel=%.5f  max_abs=%.4f  |ref|max=%.1f  tol_rel=0.020  %s\n",
                "block_11_chained", rel, mx, ref_mx, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
  }

  // ---- the merger: 2x2 patches -> one soft token ----
  //
  // Fed the oracle's block_11 so this measures the merger alone rather than inheriting the
  // stack's accumulated error.
  //
  // The 2x2 fold needs no data movement. The norm is applied per patch over hidden, and the
  // patches are already in merge-unit-major order, so four consecutive rows are one unit
  // regrouping [tokens][768] as [tokens/4][3072] is a reinterpretation of the same buffer.
  {
    const int inter_m = hidden * merge * merge;  // 3072
    const int out_hidden = engine::mini::json_get_int(geo, "out_hidden_size");
    const int soft = tokens / (merge * merge);
    const std::vector<float> want = read_f32(dir + "/stage_16_merger.f32");
    const std::vector<float> got = run_merger(ctx, read_f32(dir + "/stage_15_block_11.f32"), dir,
                                              tokens, hidden, merge, inter_m, out_hidden, soft);
    std::printf("      %d patches -> %d soft tokens of %d\n", tokens, soft, out_hidden);
    check("merger", got, want, 0.05f);
  }

  // ---- end to end: pixels -> soft tokens, entirely on our own outputs ----
  //
  // Every check above feeds a stage the oracle's input, which isolates that stage but says
  // nothing about whether the errors compound. This runs the whole tower on its own output and
  // is the only number that describes what the port actually produces.
  {
    const int heads = engine::mini::json_get_int(geo, "num_heads");
    const int head_dim = engine::mini::json_get_int(geo, "head_dim");
    const int inter = engine::mini::json_get_int(geo, "intermediate_size");
    const int depth = engine::mini::json_get_int(geo, "depth");
    const int inter_m = hidden * merge * merge;
    const int out_hidden = engine::mini::json_get_int(geo, "out_hidden_size");
    const int soft = tokens / (merge * merge);

    std::vector<float> cosb, sinb;
    build_vision_rope(grid_h, grid_w, head_dim, merge, 10000.0f, cosb, sinb);
    std::vector<float> cur = our_sum;
    for (int L = 0; L < depth; ++L) {
      const std::string bp = dir + "/weights/blocks_" + std::to_string(L) + "_";
      VisionBlockWeights bl{
          read_f32(bp + "norm1_weight.f32"),          read_f32(bp + "norm1_bias.f32"),
          read_f32(bp + "attn_qkv_weight.f32"),       read_f32(bp + "attn_qkv_bias.f32"),
          read_f32(bp + "attn_proj_weight.f32"),      read_f32(bp + "attn_proj_bias.f32"),
          read_f32(bp + "norm2_weight.f32"),          read_f32(bp + "norm2_bias.f32"),
          read_f32(bp + "mlp_linear_fc1_weight.f32"), read_f32(bp + "mlp_linear_fc1_bias.f32"),
          read_f32(bp + "mlp_linear_fc2_weight.f32"), read_f32(bp + "mlp_linear_fc2_bias.f32")};
      cur = run_vision_block(ctx, cur, bl, cosb, sinb, tokens, hidden, heads, head_dim, inter);
    }
    const std::vector<float> got =
        run_merger(ctx, cur, dir, tokens, hidden, merge, inter_m, out_hidden, soft);
    const std::vector<float> want = read_f32(dir + "/stage_16_merger.f32");
    float mx = 0.0f, ref_mx = 0.0f;
    for (std::size_t i = 0; i < got.size(); ++i) {
      mx = std::max(mx, std::fabs(got[i] - want[i]));
      ref_mx = std::max(ref_mx, std::fabs(want[i]));
    }
    std::size_t nan_e = 0;
    for (float v : got) {
      if (!std::isfinite(v)) ++nan_e;
    }
    const float rel = (nan_e != 0) ? 1e9f : mx / (ref_mx + 1e-9f);
    const bool ok = rel <= 0.02f;
    std::printf("  %-22s rel=%.5f  max_abs=%.4f  |ref|max=%.2f  tol_rel=0.020  %s\n", "END_TO_END",
                rel, mx, ref_mx, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
  }

  // ---- the same tower, weights from a .ll2c container ----
  //
  // Everything above reads the oracle's own fp32 weight dumps, so it gates the arithmetic while
  // assuming the plumbing. This re-runs it against the container the converter produces, which
  // is what the engine will actually load. A name that does not survive conversion, a tensor
  // stored transposed, a geometry field dropped by the packer's whitelist; none of those can
  // be seen upstream of here.
  if (argc >= 3) {
    // Loader-agnostic. This block byte-checks tensors straight out of the container, so it has
    // to know the format; and the engine now accepts both .ll2c and .cpi/safetensors. Opening
    // WeightLoader unconditionally is what made a repacked .cpi throw "expected LL2CUDA manifest"
    // here while the engine loaded it fine.
    const std::string cpath = argv[2];
    const bool use_st = (cpath.size() > 4 && cpath.compare(cpath.size() - 4, 4, ".cpi") == 0) ||
                        std::filesystem::is_directory(cpath);
    model::WeightLoader wl;
    model::SafetensorsLoader st;
    model::LlamaConfig c;
    if (use_st) {
      st.open(cpath);
      c = model::config_from_json(st.metadata_json());
    } else {
      wl.open(cpath);
      c = wl.config();
    }
    const auto ct_has = [&](const std::string& n) {
      return use_st ? st.has_tensor(n) : wl.has_tensor(n);
    };
    const auto ct_data = [&](const std::string& n) {
      return use_st ? st.tensor_data(n) : wl.tensor_data(n);
    };
    const auto ct_bytes = [&](const std::string& n) {
      return use_st ? st.tensor_bytes(n) : wl.tensor_bytes(n);
    };
    std::printf("      container: v-geometry depth=%d hidden=%d heads=%d merge=%d out=%d\n",
                c.vision_depth, c.vision_hidden_size, c.vision_num_heads,
                c.vision_spatial_merge_size, c.vision_out_hidden_size);
    if (!c.has_vision_tower()) {
      std::printf("  %-22s container carries no vision geometry  FAIL\n", "container_tower");
      ++failures;
    } else if (c.vision_depth != engine::mini::json_get_int(geo, "depth") ||
               c.vision_hidden_size != hidden ||
               c.vision_out_hidden_size != engine::mini::json_get_int(geo, "out_hidden_size")) {
      std::printf("  %-22s container geometry disagrees with the oracle  FAIL\n",
                  "container_tower");
      ++failures;
    } else {
      // The container stores fp16; the oracle dumps fp32. Compare the weights directly rather
      // than only the outputs; a transposed or truncated tensor can still produce plausible
      // activations, and this says which tensor rather than which stage.
      int checked = 0, bad = 0;
      const std::vector<std::pair<std::string, std::string>> pairs = {
          {"vision.patch_embed.weight", "patch_embed_proj_weight"},
          {"vision.patch_embed.bias", "patch_embed_proj_bias"},
          {"vision.pos_embed.weight", "pos_embed_weight"},
          {"vision.merger.fc1.weight", "merger_linear_fc1_weight"},
          {"vision.merger.fc2.weight", "merger_linear_fc2_weight"},
          {"vision.merger.norm.weight", "merger_norm_weight"},
          {"vision.blocks.0.attn.qkv.weight", "blocks_0_attn_qkv_weight"},
          {"vision.blocks.11.mlp.fc2.weight", "blocks_11_mlp_linear_fc2_weight"},
      };
      for (const auto& pr : pairs) {
        if (!ct_has(pr.first)) {
          std::printf("      MISSING from container: %s\n", pr.first.c_str());
          ++bad;
          continue;
        }
        const std::vector<float> ref = read_f32(dir + "/weights/" + pr.second + ".f32");
        const auto* got = reinterpret_cast<const std::uint16_t*>(ct_data(pr.first));
        const std::size_t n2 = ct_bytes(pr.first) / 2;
        if (n2 != ref.size()) {
          std::printf("      SIZE %s: container %zu vs oracle %zu\n", pr.first.c_str(), n2,
                      ref.size());
          ++bad;
          continue;
        }
        float mx = 0.0f;
        for (std::size_t i = 0; i < n2; ++i) {
          mx = std::max(mx, std::fabs(f16_to_f32(got[i]) - ref[i]));
        }
        // bf16 -> fp16 is the only conversion between them, so anything past a few thousandths
        // means a different tensor, not a rounding difference.
        if (mx > 0.01f) {
          std::printf("      DIFFERS %s: max_abs=%.5f\n", pr.first.c_str(), mx);
          ++bad;
        }
        ++checked;
      }
      std::printf("  %-22s %d tensors checked, %d wrong  %s\n", "container_weights", checked, bad,
                  bad == 0 ? "PASS" : "FAIL");
      if (bad != 0) ++failures;

      // And the engine's own path, end to end: open the container, hand it the same pixels,
      // compare its soft tokens to the oracle's. Everything above this exercises the test's
      // reimplementation of the tower; this exercises the code that will actually ship, through
      // the real weight loader. Without it the two can drift and every check above stays green.
      {
        engine::PlanMetalEngine eng;
        eng.open(argv[2], 512);
        const std::vector<float> soft = eng.encode_image(pixels, grid_h, grid_w);
        const std::vector<float> want16 = read_f32(dir + "/stage_16_merger.f32");
        float mx = 0.0f, ref_mx = 0.0f;
        for (std::size_t i = 0; i < soft.size() && i < want16.size(); ++i) {
          mx = std::max(mx, std::fabs(soft[i] - want16[i]));
          ref_mx = std::max(ref_mx, std::fabs(want16[i]));
        }
        std::size_t nan_s = 0;
        for (float v : soft) {
          if (!std::isfinite(v)) ++nan_s;
        }
        if (nan_s != 0) {
          std::printf("      engine produced %zu/%zu NON-FINITE values\n", nan_s, soft.size());
        }
        const bool sized = soft.size() == want16.size() && nan_s == 0;
        const float rel = (nan_s != 0) ? 1e9f : mx / (ref_mx + 1e-9f);
        const bool ok2 = sized && rel <= 0.02f;
        std::printf(
            "      engine soft: %zu floats (want %zu); first= %.4f %.4f %.4f | "
            "oracle= %.4f %.4f %.4f\n",
            soft.size(), want16.size(), soft[0], soft[1], soft[2], want16[0], want16[1], want16[2]);
        std::printf("  %-22s rel=%.8f  max_abs=%.6f  %s\n", "ENGINE_encode_image", rel, mx,
                    ok2 ? "PASS" : "FAIL");
        if (!ok2) ++failures;
      }
    }
  } else {
    std::printf("      (no container given: pass one as argv[2] to gate the conversion)\n");
  }

  // ---- the splice: soft tokens standing in for placeholder tokens ----
  //
  // There is no oracle past this point, the reference dump stops at soft tokens, so this
  // gates the two things that can be checked without one: that the spliced rows are the ones
  // that reach the residual stream, and that a prompt containing them generates without
  // diverging into NaN. It does not check that the model's answer is right; that needs a full
  // multimodal comparison against HuggingFace, which is a heavier harness than this.
  if (argc >= 3) {
    engine::PlanMetalEngine eng;
    eng.open(argv[2], 512);
    const std::vector<float> soft = eng.encode_image(pixels, grid_h, grid_w);
    const int out_hidden = engine::mini::json_get_int(geo, "out_hidden_size");
    const int n_soft = static_cast<int>(soft.size()) / out_hidden;

    // A prompt of arbitrary text tokens with n_soft placeholders in the middle. The ids do not
    // matter for the placeholders, their embeddings are overwritten, but they must be in
    // range, and using a real id rather than 0 is deliberate: 0 often maps to a padding
    // embedding that is all zeros, which would make a failed splice look like a working one.
    std::vector<int> toks = {760, 6511};
    std::vector<std::vector<float>> embeds(toks.size());
    for (int i = 0; i < n_soft; ++i) {
      toks.push_back(9338);
      embeds.emplace_back(soft.begin() + static_cast<std::size_t>(i) * out_hidden,
                          soft.begin() + static_cast<std::size_t>(i + 1) * out_hidden);
    }
    for (int i = 0; i < 2; ++i) {
      toks.push_back(314);
      embeds.emplace_back();
    }
    eng.prefill_multimodal(toks, embeds);
    std::printf("      spliced %d soft tokens into a %zu-token prompt\n", n_soft, toks.size());

    // Generation must survive the spliced prompt. NaN or a hang here is the failure mode that
    // matters; a wrong-but-finite answer cannot be judged without a multimodal oracle.
    bool finite = true;
    int next = toks.back();
    for (int i = 0; i < 4; ++i) {
      const std::vector<float>& lg = eng.forward_token(next, static_cast<int>(toks.size()) + i);
      int best = 0;
      for (std::size_t j = 0; j < lg.size(); ++j) {
        if (!std::isfinite(lg[j])) {
          finite = false;
          break;
        }
        if (lg[j] > lg[static_cast<std::size_t>(best)]) best = static_cast<int>(j);
      }
      if (!finite) break;
      next = best;
    }
    std::printf("  %-22s %s\n", "SPLICE_generation",
                finite ? "logits finite over 4 steps  PASS" : "NON-FINITE logits  FAIL");
    if (!finite) ++failures;

    // control: the same prompt with no splice must produce different logits. "Finite" alone
    // says nothing; a splice that silently did nothing also produces finite logits, and that
    // is precisely the failure this whole file keeps running into. Comparing against the
    // unspliced run is the only thing that shows the soft tokens reached the residual stream.
    {
      engine::PlanMetalEngine eng2;
      eng2.open(argv[2], 512);
      const std::vector<std::vector<float>> none(toks.size());
      eng2.prefill_multimodal(toks, none);
      const std::vector<float>& plain =
          eng2.forward_token(toks.back(), static_cast<int>(toks.size()));
      engine::PlanMetalEngine eng3;
      eng3.open(argv[2], 512);
      eng3.prefill_multimodal(toks, embeds);
      const std::vector<float>& with =
          eng3.forward_token(toks.back(), static_cast<int>(toks.size()));
      double diff = 0.0;
      const std::size_t m = std::min(plain.size(), with.size());
      for (std::size_t i = 0; i < m; ++i) {
        diff = std::max(diff, static_cast<double>(std::fabs(plain[i] - with[i])));
      }
      const bool changed = diff > 1e-3;
      std::printf("  %-22s max logit delta vs unspliced = %.4f  %s\n", "SPLICE_is_load_bearing",
                  diff, changed ? "PASS" : "FAIL (splice did nothing)");
      if (!changed) ++failures;
    }
  }

  // ---- M-RoPE positions, against the transformers dump ----
  //
  // These numbers came out of the reference for <vision_start> + 16 image + <vision_end> + 4
  // text with a 4x4 merged grid. Hardcoded rather than recomputed here: a check that derives
  // the expectation the same way the code does only proves the code is self-consistent.
  {
    const int IMG = 248056;
    std::vector<int> toks = {248053};
    for (int i = 0; i < 16; ++i) toks.push_back(IMG);
    toks.push_back(248054);
    for (int i = 0; i < 4; ++i) toks.push_back(3838 + i);
    const std::vector<std::int32_t> want_t = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 1, 1, 1, 5, 6, 7, 8, 9};
    const std::vector<std::int32_t> want_h = {0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3,
                                              3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9};
    const std::vector<std::int32_t> want_w = {0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2,
                                              3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const std::vector<std::int32_t> got = engine::build_mrope_positions(toks, IMG, 4, 4);
    const int n = static_cast<int>(toks.size());
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
      if (got[i] != want_t[i]) ++wrong;
      if (got[n + i] != want_h[i]) ++wrong;
      if (got[2 * n + i] != want_w[i]) ++wrong;
    }
    std::printf("  %-22s %d/%d position components wrong  %s\n", "mrope_positions", wrong, 3 * n,
                wrong == 0 ? "PASS" : "FAIL");
    if (wrong != 0) {
      std::printf("      got t: ");
      for (int i = 0; i < n; ++i) std::printf("%d ", got[i]);
      std::printf("\n      got h: ");
      for (int i = 0; i < n; ++i) std::printf("%d ", got[n + i]);
      std::printf("\n      got w: ");
      for (int i = 0; i < n; ++i) std::printf("%d ", got[2 * n + i]);
      std::printf("\n");
      ++failures;
    }
  }

  // ---- the end: Metal's token stream vs the reference's ----
  //
  // Everything else in this file compares activations. This compares answers, which is the only
  // check that covers the tower, the splice, M-RoPE and the text stack at once; and the only
  // one that can tell a wrong-but-finite result from a right one.
  //
  // Reference stream from tools/qwen35_multimodal_oracle.py on the same synthetic image and
  // prompt. Hardcoded: regenerating it here would just be running the oracle twice.
  if (argc >= 3) {
    const int IMG = 248056;
    engine::PlanMetalEngine eng;
    eng.open(argv[2], 512);
    const std::vector<float> soft = eng.encode_image(pixels, grid_h, grid_w);
    const int out_hidden = engine::mini::json_get_int(geo, "out_hidden_size");
    const int n_soft = static_cast<int>(soft.size()) / out_hidden;
    const int mh = grid_h / merge, mw = grid_w / merge;

    const std::vector<float> want_soft = read_f32(dir + "/stage_16_merger.f32");
    // CPI_MM_ORACLE_SOFT=1 splices the oracle's soft tokens instead of our tower's. That splits
    // the multimodal drift in two: whatever remains with the oracle's inputs belongs to the
    // splice, M-RoPE and the text stack, and whatever the swap removes was the tower's own fp16
    // error propagating. Without this the two are indistinguishable, and 16 of the 22 positions
    // carry tower output; enough that "the logits drift" says nothing about where.
    const std::vector<float>& soft_src =
        (std::getenv("CPI_MM_ORACLE_SOFT") != nullptr) ? want_soft : soft;
    if (std::getenv("CPI_MM_ORACLE_SOFT") != nullptr) {
      std::printf("      [CPI_MM_ORACLE_SOFT: splicing the oracle's soft tokens, not ours]\n");
    }

    std::vector<int> toks = {248053};
    std::vector<std::vector<float>> embeds(1);
    for (int i = 0; i < n_soft; ++i) {
      toks.push_back(IMG);
      embeds.emplace_back(soft_src.begin() + static_cast<std::size_t>(i) * out_hidden,
                          soft_src.begin() + static_cast<std::size_t>(i + 1) * out_hidden);
    }
    toks.push_back(248054);
    embeds.emplace_back();
    for (int t : {3838, 374, 419, 30}) {
      toks.push_back(t);
      embeds.emplace_back();
    }

    const std::vector<std::int32_t> mpos = engine::build_mrope_positions(toks, IMG, mh, mw);

    // Prefill everything except the last token, then push that one through forward_token.
    //
    // prefill_multimodal consumes every token it is handed; prefill_chunks(n) covers all n
    // and there is no accessor for the logits it leaves behind. So prefilling all of `toks` and
    // then calling forward_token(toks.back()) feeds the final token twice, at n-1 and again at
    // n, which corrupts the KV cache and shifts every position after it.
    //
    // This loop did exactly that, and it is what MULTIMODAL_stream was really measuring. The
    // same mistake in qwen35_text_test produced the last prompt token eight times in a row,
    // which is what made it visible: here it merely looked like a plausible near-tie at step 5.
    const int n_tok = static_cast<int>(toks.size());
    const std::vector<int> head(toks.begin(), toks.end() - 1);
    const std::vector<std::vector<float>> head_embeds(embeds.begin(), embeds.end() - 1);
    // mpos is [3][n_tok]; the prefill needs [3][n_tok-1] with the same axis-major layout.
    std::vector<std::int32_t> head_mpos(static_cast<std::size_t>(3) * (n_tok - 1));
    for (int axis = 0; axis < 3; ++axis) {
      for (int t = 0; t < n_tok - 1; ++t) {
        head_mpos[static_cast<std::size_t>(axis) * (n_tok - 1) + t] =
            mpos[static_cast<std::size_t>(axis) * n_tok + t];
      }
    }
    eng.prefill_multimodal(head, head_embeds, head_mpos);

    // Two counters, because M-RoPE makes them diverge. The rotary position continues the merged
    // counter: mrope_next_position reports where the next token goes, so the last prompt token
    // is one before it, and generated tokens step up by one from there. The cache position is the
    // real sequence index; head.size() tokens were prefilled, so the last prompt token occupies
    // slot head.size()-1 and the first generated token lands at head.size(). Passing the rotary
    // position for both is the bug this file used to carry: K/V written into an image slot and
    // attention truncated to the rotary position.
    const int rope_pos = engine::mrope_next_position(toks, IMG, mh, mw) - 1;
    const int cache_pos = static_cast<int>(head.size());  // last prompt token's slot + 1
    std::vector<int> got;
    int next = toks.back();
    for (int i = 0; i < 8; ++i) {
      const std::vector<float>& lg = eng.forward_token(next, rope_pos + i, cache_pos + i);

      // ---- step 0 against the oracle's dense logits ----
      //
      // MULTIMODAL_stream compares eight argmax decisions. That is 8 bits of evidence about a
      // 248320-wide vector, and it cannot tell "our logits are slightly off" from "our logits
      // are badly off but the ordering survived". qwen35_multimodal_oracle.py has been dumping
      // first_step_logits.f32 since it was written and nothing has ever compared against it
      // the same written-but-unrun pattern as the fp16 fixture.
      //
      // Step 0 specifically, because it is the last position where the two runs are guaranteed
      // to be looking at identical input: from step 1 on, any divergence feeds itself.
      if (i == 0) {
        const std::string mmdir = dir + "/../q35_mm/first_step_logits.f32";
        std::FILE* probe = std::fopen(mmdir.c_str(), "rb");
        if (probe == nullptr) {
          std::printf("      %-20s no q35_mm dump next to the oracle dir; skipped\n",
                      "FIRST_STEP_LOGITS");
        } else {
          std::fclose(probe);
          const std::vector<float> ref = read_f32(mmdir);
          if (ref.size() != lg.size()) {
            std::printf("  %-22s size %zu vs oracle %zu  FAIL\n", "FIRST_STEP_LOGITS", lg.size(),
                        ref.size());
            ++failures;
          } else {
            // Compare the SHAPE of the distribution, not raw magnitudes: a constant offset
            // across every logit is invisible to softmax and to argmax, so it is not an error.
            // Centre both on their own mean before differencing.
            double mg = 0.0, mr = 0.0;
            for (std::size_t j = 0; j < lg.size(); ++j) {
              mg += lg[j];
              mr += ref[j];
            }
            mg /= static_cast<double>(lg.size());
            mr /= static_cast<double>(ref.size());
            float mx = 0.0f;
            double sum = 0.0;
            int arg_g = 0, arg_r = 0;
            for (std::size_t j = 0; j < lg.size(); ++j) {
              const float d =
                  std::fabs((lg[j] - static_cast<float>(mg)) - (ref[j] - static_cast<float>(mr)));
              mx = std::max(mx, d);
              sum += d;
              if (lg[j] > lg[static_cast<std::size_t>(arg_g)]) arg_g = static_cast<int>(j);
              if (ref[j] > ref[static_cast<std::size_t>(arg_r)]) arg_r = static_cast<int>(j);
            }
            // Also price the pair that actually decides step 5, at step 0: if OUR gap between
            // them is already drifting here, the divergence is not born at step 5.
            const float g35 = lg[3160] - lg[3878];
            const float r35 = ref[3160] - ref[3878];
            std::printf(
                "      argmax ours=%d oracle=%d | logit(3160)-logit(3878): ours=%+.4f "
                "oracle=%+.4f  drift=%+.4f\n",
                arg_g, arg_r, g35, r35, g35 - r35);
            // Tolerance 0.30 on the mean. The behavioural gate is MULTIMODAL_stream below, which
            // is token-identical to the fp32 reference over 8 steps; this is the drift tripwire
            // on top of it. It is set above the text-only control's 0.12 on purpose: this prompt
            // carries fp16 TOWER output at 16 of its 22 positions and a 3-D M-RoPE rotary, where
            // the text control has neither, so 0.22 here is the same order of error the clean 1-D
            // path shows at 0.12; not a defect.
            //
            // The history is worth keeping: this read 0.484 while the KV cache was being indexed
            // by the rotary position instead of the sequence index (11 of 21 prompt tokens
            // dropped from attention). Splitting the two positions took it to 0.218 and the
            // stream to 8/8. Do not widen this further to absorb a real regression; if it
            // climbs back toward 0.4, the cache/rotary split has broken again.
            const double mean_abs = sum / static_cast<double>(lg.size());
            std::printf("  %-22s max_abs=%.4f  mean_abs=%.4f  tol_mean=0.30  %s\n",
                        "FIRST_STEP_LOGITS", mx, mean_abs, mean_abs <= 0.30 ? "PASS" : "FAIL");
            if (mean_abs > 0.30) ++failures;
          }
        }
      }
      int best = 0;
      for (std::size_t j = 1; j < lg.size(); ++j) {
        if (lg[j] > lg[static_cast<std::size_t>(best)]) best = static_cast<int>(j);
      }
      // Top-2 margin at every step. A divergence at a step where the top two logits are
      // within fp16 noise is a tipped near-tie; one where the winner leads comfortably is a
      // real disagreement. Printing it is the difference between knowing and guessing.
      int second = (best == 0) ? 1 : 0;
      for (std::size_t j = 0; j < lg.size(); ++j) {
        if (static_cast<int>(j) == best) continue;
        if (lg[j] > lg[static_cast<std::size_t>(second)]) second = static_cast<int>(j);
      }
      std::printf("      step %d: top1=%d (%.4f)  top2=%d (%.4f)  margin=%.4f\n", i, best,
                  lg[static_cast<std::size_t>(best)], second, lg[static_cast<std::size_t>(second)],
                  lg[static_cast<std::size_t>(best)] - lg[static_cast<std::size_t>(second)]);
      got.push_back(best);
      next = best;
    }
    const std::vector<int> want = {198, 760, 2099, 4774, 264, 3160, 5072, 314};
    int match = 0;
    for (std::size_t i = 0; i < want.size() && i < got.size(); ++i) {
      if (got[i] == want[i])
        ++match;
      else
        break;
    }
    std::printf("      metal : ");
    for (int v : got) std::printf("%d ", v);
    std::printf("\n      oracle: ");
    for (int v : want) std::printf("%d ", v);
    std::printf("\n  %-22s %d/8 leading tokens match  %s\n", "MULTIMODAL_stream", match,
                match == 8 ? "PASS" : "FAIL");
    if (match != 8) ++failures;
  }

  std::printf("[metal_vision] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
