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
#include <string>
#include <vector>

#include "model/json_mini.hpp"
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
  if (std::fread(v.data(), 1, static_cast<std::size_t>(bytes), f) != static_cast<std::size_t>(bytes)) {
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

std::uint16_t f32_to_f16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
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

// Reports max and mean absolute error. Both, because a mean alone hides a single wrong element
// among thousands of right ones -- which is what an off-by-one in a permutation looks like.
void check(const char* name, const std::vector<float>& got, const std::vector<float>& want,
           float tol) {
  if (got.size() != want.size()) {
    std::printf("  %-22s SIZE MISMATCH got=%zu want=%zu  FAIL\n", name, got.size(), want.size());
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
//   1. The sample positions are linspace(0, side-1, n) -- endpoint INCLUSIVE. Using n even
//      steps (i * side / n) instead of n - 1 (i * (side-1) / (n-1)) shifts every sample and
//      still produces a smooth, plausible table.
//
//   2. The result is reordered into merge-unit-major order before being returned, so patch
//      (i, j) does NOT land at row i*w + j. The blocks therefore see patches grouped by the
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: metal_vision_test <oracle_dir>\n");
    return 2;
  }
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
  // than a missing library. Cost me a debugging round; hence the explicit failure here.
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

  // ---- stage 1: patch embed ----
  // Conv3d with stride == kernel_size over an input reshaped so each sample IS one patch, so
  // it reduces to Linear(patch_dim -> hidden). The [768,3,2,16,16] weight is already contiguous
  // in exactly that order, so it needs no rearrangement -- only reinterpretation.
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
    std::printf("    want[0..3]= %.4f %.4f %.4f %.4f\n", want_pe[0], want_pe[1],
                want_pe[2], want_pe[3]);
    check("patch_embed", got, want_pe, 0.05f);
  }

  // ---- stage 2: interpolated position table (host) ----
  const std::vector<float> pos = interpolate_pos_embed(pos_tab, side, grid_h, grid_w, hidden, merge);
  check("pos_embed", pos, want_pos, 1e-4f);

  // ---- stage 3: their sum ----
  {
    std::vector<std::uint16_t> ha(want_pe.size()), hb(pos.size());
    for (std::size_t i = 0; i < want_pe.size(); ++i) ha[i] = f32_to_f16(want_pe[i]);
    for (std::size_t i = 0; i < pos.size(); ++i) hb[i] = f32_to_f16(pos[i]);
    auto ba = ctx.alloc_from(ha.data(), ha.size() * 2);
    auto bb = ctx.alloc_from(hb.data(), hb.size() * 2);
    ElemParams p{static_cast<std::uint32_t>(ha.size()), 1.0f};
    // cpi_add_inplace is (in, out) and writes buffer 1, so the accumulator must be SECOND.
    const void* bufs[] = {bb.handle(), ba.handle()};
    ctx.dispatch("cpi_add_inplace", runtime::MetalContext::Grid::Threads, ha.size(), 256, bufs,
                 nullptr, 2, &p, sizeof(p));
    ctx.commit_and_wait();
    const auto* o = static_cast<const std::uint16_t*>(ba.contents());
    std::vector<float> got(ha.size());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
    // Fed the ORACLE's patch embed, not ours, so this isolates the add and the position table
    // from any error already measured in stage 1.
    check("pos_embed_added", got, want_sum, 0.02f);
  }

  std::printf("[metal_vision] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
