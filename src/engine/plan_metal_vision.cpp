// The Qwen3.5 vision tower, inside the engine.
//
// Separate translation unit from plan_metal_engine.cpp, which is already ~2200 lines. Nothing
// here is shared with the text path except the kernels and the weight source: the tower runs
// once per image, over a patch grid rather than a token sequence, at its own hidden size. Its
// buffers are therefore allocated per call rather than drawn from the engine's slot pool, which
// is sized for text geometry -- the tower's cost is twelve blocks of GEMMs, so a handful of
// allocations does not register.
//
// The arithmetic here is the same sequence metal_vision_test gates against the HuggingFace
// oracle. That test now also runs THIS path (see encode_image there), so the two cannot drift.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>

#include "engine/plan_metal_engine.hpp"
#include "runtime/fp16.hpp"

namespace engine {

namespace {

// Was a local copy that truncated and flushed subnormals; see include/runtime/fp16.hpp for what
// that cost and metal_fp16_test for the gate.
inline std::uint16_t f32_to_f16_bits(float f) { return cpi::f32_to_f16(f); }
inline float f16_bits_to_f32(std::uint16_t h) { return cpi::f16_to_f32(h); }

struct GemvParams {
  std::uint32_t out_dim, in_dim, tokens, has_bias;
};
struct ElemParams {
  std::uint32_t n;
  float scale;
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

// Must match plan_metal_engine.cpp and the shader.
constexpr int kGemmBN = 32;
constexpr int kGemmFBM = 64;
constexpr int kGemmRF = 4;
constexpr int kGemmCF = 4;
constexpr int kGemmTG = 32 * (kGemmFBM / (8 * kGemmRF)) * (kGemmBN / (8 * kGemmCF));

// Bilinear resample of the learned position table onto the patch grid, then a reorder into
// merge-unit-major order. Both details are load-bearing and neither announces itself if wrong:
// the samples are linspace(0, side-1, n) with the endpoint INCLUDED, and patch (i, j) does NOT
// land at row i*w + j -- the blocks see patches grouped by the 2x2 unit the merger later folds.
//
// Host-side on purpose: O(patches * hidden) once per image against twelve blocks at
// O(patches^2 * dim), so a bespoke gather kernel would add surface area that can be silently
// wrong and buy nothing measurable.
std::vector<float> interpolate_pos_embed(const float* table, int side, int h, int w, int hidden,
                                         int merge) {
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
        const float* src = table + static_cast<std::size_t>(idx[c]) * hidden;
        for (int d = 0; d < hidden; ++d) dst[d] += wt[c] * src[d];
      }
    }
  }
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

// Per-token cos/sin for the vision rotation, half a head_dim wide (cos[i] == cos[i+half]).
// The frequency vector is [row_freqs | col_freqs], each head_dim/4 long -- not one geometric
// series over head_dim/2 -- and the tokens are enumerated in the same merge-unit-major order
// the position table uses.
void build_rope_tables(int h, int w, int head_dim, int merge, float theta,
                       std::vector<float>& cosb, std::vector<float>& sinb) {
  const int half = head_dim / 2;
  const int quarter = head_dim / 4;
  cosb.assign(static_cast<std::size_t>(h) * w * half, 0.0f);
  sinb.assign(static_cast<std::size_t>(h) * w * half, 0.0f);
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
            cosb[static_cast<std::size_t>(idx) * half + j] = std::cos(static_cast<float>(row) * inv);
            sinb[static_cast<std::size_t>(idx) * half + j] = std::sin(static_cast<float>(row) * inv);
            cosb[static_cast<std::size_t>(idx) * half + quarter + j] =
                std::cos(static_cast<float>(col) * inv);
            sinb[static_cast<std::size_t>(idx) * half + quarter + j] =
                std::sin(static_cast<float>(col) * inv);
          }
        }
      }
    }
  }
}

}  // namespace

// Builds the 3-D (t, h, w) position ids M-RoPE needs, as one [3][tokens] array.
//
// The rule, read off the reference's own output rather than from its source:
//
//   - a text token takes t = h = w = next, and advances next by 1;
//   - an image span takes base = next and assigns t = base, h = base + row, w = base + col over
//     its MERGED grid, then advances next by max(merged_h, merged_w).
//
// That last step is the one that matters and the one a reimplementation gets wrong. An image of
// 16 tokens advances the counter by 4, not 16, so every text token AFTER an image sits four
// positions past the image's start -- 5..9 for the reference prompt, where a 1-D counter puts
// them at 17..21. Getting the image span right and the tail wrong still corrupts the answer.
//
// Verified against the transformers dump for <vision_start> + 16 image + <vision_end> + 4 text;
// see metal_vision_test's mrope_positions case.
std::vector<std::int32_t> build_mrope_positions(const std::vector<int>& tokens, int image_token_id,
                                                int merged_h, int merged_w) {
  const int n = static_cast<int>(tokens.size());
  std::vector<std::int32_t> pos(static_cast<std::size_t>(3) * n, 0);
  auto set = [&](int idx, int t, int h, int w) {
    pos[static_cast<std::size_t>(idx)] = t;
    pos[static_cast<std::size_t>(n + idx)] = h;
    pos[static_cast<std::size_t>(2 * n + idx)] = w;
  };
  int next = 0;
  int i = 0;
  while (i < n) {
    if (tokens[static_cast<std::size_t>(i)] != image_token_id) {
      set(i, next, next, next);
      ++next;
      ++i;
      continue;
    }
    // A run of image tokens. Its length must be exactly merged_h * merged_w -- one placeholder
    // per soft token -- or the grid this walks does not describe the run.
    int run = 0;
    while (i + run < n && tokens[static_cast<std::size_t>(i + run)] == image_token_id) ++run;
    const int expect = merged_h * merged_w;
    if (run != expect) {
      throw std::runtime_error("image span is " + std::to_string(run) + " tokens but the " +
                               std::to_string(merged_h) + "x" + std::to_string(merged_w) +
                               " merged grid needs " + std::to_string(expect));
    }
    const int base = next;
    for (int k = 0; k < run; ++k) {
      set(i + k, base, base + k / merged_w, base + k % merged_w);
    }
    next = base + std::max(merged_h, merged_w);
    i += run;
  }
  return pos;
}

// Where the M-RoPE counter ended -- the position the FIRST generated token takes. Not the token
// count: an image span advances the counter by its merged extent, so after 16 image tokens the
// two differ by 12 and using the token count rotates every generated token wrongly.
int mrope_next_position(const std::vector<int>& tokens, int image_token_id, int merged_h,
                        int merged_w) {
  const std::vector<std::int32_t> pos =
      build_mrope_positions(tokens, image_token_id, merged_h, merged_w);
  const int n = static_cast<int>(tokens.size());
  if (n == 0) return 0;
  // The last token's t axis, plus one. For a trailing text token all three axes agree; for a
  // trailing IMAGE token the counter has already advanced past the span, so take the max.
  int last = 0;
  for (int axis = 0; axis < 3; ++axis) {
    last = std::max(last, static_cast<int>(pos[static_cast<std::size_t>(axis) * n + (n - 1)]));
  }
  return last + 1;
}

std::vector<float> PlanMetalEngine::encode_image(const std::vector<float>& patches, int grid_h,
                                                 int grid_w) {
  const model::LlamaConfig& c = cfg_;
  if (!c.has_vision_tower()) {
    throw std::runtime_error(
        "this model has no vision tower (container carries vision_depth = 0). A text-only "
        "checkpoint, or one converted before container v7 carried the tower's geometry.");
  }
  const int merge = c.vision_spatial_merge_size;
  if (grid_h % merge != 0 || grid_w % merge != 0) {
    throw std::runtime_error("patch grid must be a whole number of " + std::to_string(merge) +
                             "x" + std::to_string(merge) + " merge units in both axes");
  }
  const int hidden = c.vision_hidden_size;
  const int heads = c.vision_num_heads;
  const int head_dim = hidden / heads;
  const int inter = c.vision_intermediate_size;
  const int tokens = grid_h * grid_w;
  const int patch_dim = c.vision_in_channels * c.vision_temporal_patch_size * c.vision_patch_size *
                        c.vision_patch_size;
  const int side = static_cast<int>(std::lround(std::sqrt(
      static_cast<double>(c.vision_num_position_embeddings))));
  const int inter_m = hidden * merge * merge;
  const int out_hidden = c.vision_out_hidden_size;
  const int soft = tokens / (merge * merge);

  if (static_cast<int>(patches.size()) != tokens * patch_dim) {
    throw std::runtime_error("patch buffer is " + std::to_string(patches.size()) +
                             " floats, expected " + std::to_string(tokens * patch_dim) + " (" +
                             std::to_string(tokens) + " patches x " + std::to_string(patch_dim) +
                             ")");
  }

  const std::size_t n = static_cast<std::size_t>(tokens) * hidden;
  auto to_f16 = [](const std::vector<float>& v) {
    std::vector<std::uint16_t> h(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) h[i] = f32_to_f16_bits(v[i]);
    return h;
  };
  opplan::WeightSource& ws = weight_source();
  // Resolve EVERY vision tensor before the first dispatch. WeightSource::fp16 uploads on first
  // use -- it allocates a Metal buffer and memcpys into it -- and doing that while a compute
  // encoder is open is not something the encoder's already-queued work is ordered against.
  // Touching them all up front also keeps the twelve blocks free of upload stalls.
  std::vector<std::string> names = {"patch_embed.weight", "patch_embed.bias", "pos_embed.weight",
                                    "merger.norm.weight", "merger.norm.bias", "merger.fc1.weight",
                                    "merger.fc1.bias",    "merger.fc2.weight", "merger.fc2.bias"};
  for (int L = 0; L < c.vision_depth; ++L) {
    const std::string B = "blocks." + std::to_string(L) + ".";
    for (const char* t : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                          "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                          "attn.proj.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight",
                          "mlp.fc2.bias"}) {
      names.push_back(B + t);
    }
  }
  std::unordered_map<std::string, const void*> wcache;
  for (const std::string& nm : names) wcache[nm] = ws.fp16("vision." + nm);

  auto W = [&](const std::string& name) -> const void* {
    const auto it = wcache.find(name);
    if (it == wcache.end()) {
      throw std::runtime_error("vision weight not prefetched: " + name);
    }
    return it->second;
  };

  auto gemm = [&](const void* w, runtime::MetalBuffer& in, runtime::MetalBuffer& out,
                  const void* bias, int od, int idim, int rows) {
    GemvParams p{static_cast<std::uint32_t>(od), static_cast<std::uint32_t>(idim),
                 static_cast<std::uint32_t>(rows), 1u};
    const void* bufs[] = {w, in.handle(), out.handle(), bias};
    const std::size_t tiles = static_cast<std::size_t>((rows + kGemmBN - 1) / kGemmBN);
    const std::size_t groups = (static_cast<std::size_t>(od) / kGemmFBM) * tiles;
    ctx_.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, groups, kGemmTG, bufs,
                  nullptr, 4, &p, sizeof(p));
  };

  // ---- patch embed: a Conv3d whose stride equals its kernel, i.e. a Linear ----
  auto hp = to_f16(patches);
  auto bpix = ctx_.alloc_from(hp.data(), hp.size() * 2);
  auto bx = ctx_.alloc(n * 2);
  gemm(W("patch_embed.weight"), bpix, bx, W("patch_embed.bias"), hidden, patch_dim, tokens);
  ctx_.commit_and_wait();

  // ---- + interpolated position table ----
  {
    // From the LOADER, not the WeightSource. WeightSource::fp16 hands back an opaque Metal
    // buffer handle -- its own comment says these are never dereferenced as pointers -- and
    // reading one as fp16 data is how this first went wrong: a SIGBUS, then, once the layout
    // shifted, 16384 NaNs that the test happily reported as PASS.
    //
    // The position table is the one vision tensor the CPU has to read, because the resample is
    // done host-side; everything else stays on the GPU as a handle.
    const auto* tab = reinterpret_cast<const std::uint16_t*>(
        weights_.tensor_data("vision.pos_embed.weight"));
    std::vector<float> tab32(static_cast<std::size_t>(c.vision_num_position_embeddings) * hidden);
    for (std::size_t i = 0; i < tab32.size(); ++i) tab32[i] = f16_bits_to_f32(tab[i]);
    const std::vector<float> pos =
        interpolate_pos_embed(tab32.data(), side, grid_h, grid_w, hidden, merge);
    // Add the position table in fp32 and round ONCE, rather than rounding the table to fp16
    // first and adding in the kernel.
    //
    // The interpolated table is not the stored one: bilinear blending of four bf16 rows with
    // fractional weights produces values with full f32 mantissa content, so rounding it before
    // the add throws away bits the add would otherwise keep. Measured on the M4, this halves the
    // tower's error -- ENGINE_encode_image rel 0.01010 -> 0.00508 -- and it is what the
    // reference (and metal_vision_test's END_TO_END path) does.
    //
    // Host-side rather than a kernel: one pass over tokens*hidden, against twelve blocks of
    // GEMMs. It does not register, and the buffer is already host-addressable.
    //
    // The single rounding here is IEEE round-to-nearest (include/runtime/fp16.hpp). It used to
    // be a local truncating converter, which put a second, one-directional error on top of the
    // one this fix removes.
    auto* dst = static_cast<std::uint16_t*>(bx.contents());
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = f32_to_f16_bits(f16_bits_to_f32(dst[i]) + pos[i]);
    }
  }

  std::vector<float> cosb, sinb;
  build_rope_tables(grid_h, grid_w, head_dim, merge, 10000.0f, cosb, sinb);
  auto bcos = ctx_.alloc_from(cosb.data(), cosb.size() * sizeof(float));
  auto bsin = ctx_.alloc_from(sinb.data(), sinb.size() * sizeof(float));

  auto bnorm = ctx_.alloc(n * 2);
  auto bqkv = ctx_.alloc(n * 3 * 2);
  auto battn = ctx_.alloc(n * 2);
  auto btmp = ctx_.alloc(n * 2);
  auto binter = ctx_.alloc(static_cast<std::size_t>(tokens) * inter * 2);

  // ---- 12 blocks ----
  for (int L = 0; L < c.vision_depth; ++L) {
    const std::string B = "blocks." + std::to_string(L) + ".";
    {
      LayerNormParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(hidden),
                        1e-6f, 1u};
      const void* bufs[] = {bx.handle(), W(B + "norm1.weight"), W(B + "norm1.bias"),
                            bnorm.handle()};
      ctx_.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups,
                    static_cast<std::size_t>(tokens), 256, bufs, nullptr, 4, &p, sizeof(p));
    }
    gemm(W(B + "attn.qkv.weight"), bnorm, bqkv, W(B + "attn.qkv.bias"), hidden * 3, hidden, tokens);
    // q and k are rotated in place inside the fused qkv block; v is not rotated. row_stride is
    // 3*hidden because they live there rather than in buffers of their own.
    {
      VisRopeParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(heads),
                      static_cast<std::uint32_t>(head_dim), static_cast<std::uint32_t>(hidden * 3)};
      const std::size_t work = static_cast<std::size_t>(tokens) * heads * (head_dim / 2);
      for (int part = 0; part < 2; ++part) {
        const void* bufs[] = {bqkv.handle(), bcos.handle(), bsin.handle()};
        const std::size_t offs[] = {static_cast<std::size_t>(part) * hidden * 2, 0, 0};
        ctx_.dispatch("cpi_rope_vision", runtime::MetalContext::Grid::Threads, work, 256, bufs,
                      offs, 3, &p, sizeof(p));
      }
    }
    {
      BiAttnParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(heads),
                     static_cast<std::uint32_t>(head_dim),
                     1.0f / std::sqrt(static_cast<float>(head_dim)),
                     static_cast<std::uint32_t>(hidden * 3)};
      const void* bufs[] = {bqkv.handle(), bqkv.handle(), bqkv.handle(), battn.handle()};
      const std::size_t offs[] = {0, static_cast<std::size_t>(hidden) * 2,
                                  static_cast<std::size_t>(hidden) * 4, 0};
      ctx_.dispatch("cpi_attention_bidirectional", runtime::MetalContext::Grid::Groups,
                    static_cast<std::size_t>(tokens) * heads, 64, bufs, offs, 4, &p, sizeof(p));
    }
    gemm(W(B + "attn.proj.weight"), battn, btmp, W(B + "attn.proj.bias"), hidden, hidden, tokens);
    {
      ElemParams p{static_cast<std::uint32_t>(n), 1.0f};
      const void* bufs[] = {btmp.handle(), bx.handle()};
      ctx_.dispatch("cpi_add_inplace", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr,
                    2, &p, sizeof(p));
    }
    {
      LayerNormParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(hidden),
                        1e-6f, 1u};
      const void* bufs[] = {bx.handle(), W(B + "norm2.weight"), W(B + "norm2.bias"),
                            bnorm.handle()};
      ctx_.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups,
                    static_cast<std::size_t>(tokens), 256, bufs, nullptr, 4, &p, sizeof(p));
    }
    gemm(W(B + "mlp.fc1.weight"), bnorm, binter, W(B + "mlp.fc1.bias"), inter, hidden, tokens);
    {
      // tanh GELU here: the BLOCKS specify hidden_act = "gelu_pytorch_tanh". The merger below
      // uses the exact erf form instead -- they are different functions and the tower uses both.
      ElemParams p{static_cast<std::uint32_t>(tokens) * static_cast<std::uint32_t>(inter), 1.0f};
      const void* bufs[] = {binter.handle()};
      ctx_.dispatch("cpi_gelu", runtime::MetalContext::Grid::Threads,
                    static_cast<std::size_t>(tokens) * inter, 256, bufs, nullptr, 1, &p, sizeof(p));
    }
    gemm(W(B + "mlp.fc2.weight"), binter, btmp, W(B + "mlp.fc2.bias"), hidden, inter, tokens);
    {
      ElemParams p{static_cast<std::uint32_t>(n), 1.0f};
      const void* bufs[] = {btmp.handle(), bx.handle()};
      ctx_.dispatch("cpi_add_inplace", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr,
                    2, &p, sizeof(p));
    }
    // Diagnostic (CPI_VISION_DUMP_BLOCKS=<dir>): write this block's residual stream, so the
    // engine's per-block error curve can be laid next to the one metal_vision_test prints for
    // the harness. Comparing the two curves is what localises a divergence to a block instead of
    // leaving "the output is wrong" across twelve of them; it is how the fp32 position add above
    // was tracked down, and it showed the remainder is diffuse rather than any single block.
    //
    // Splitting the command buffer here is numerically neutral -- MetalContext barriers before
    // every dispatch by default (metal_context.mm:430) -- so this observes without perturbing.
    // Verified: with the dump on, ENGINE_encode_image is bit-identical at rel=0.01009724.
    if (const char* dd = std::getenv("CPI_VISION_DUMP_BLOCKS")) {
      ctx_.commit_and_wait();
      const auto* q = static_cast<const std::uint16_t*>(bx.contents());
      std::vector<float> v(n);
      for (std::size_t i = 0; i < n; ++i) v[i] = f16_bits_to_f32(q[i]);
      char path[512];
      std::snprintf(path, sizeof path, "%s/engine_block_%02d.f32", dd, L);
      if (std::FILE* f = std::fopen(path, "wb")) {
        std::fwrite(v.data(), sizeof(float), v.size(), f);
        std::fclose(f);
      }
    }
  }

  // ---- merger: LayerNorm per patch, then each 2x2 unit becomes one row ----
  //
  // The fold moves no data. Patches are already in merge-unit-major order, so four consecutive
  // rows ARE one unit and [tokens][hidden] reinterprets as [tokens/4][4*hidden] in place.
  auto bmi = ctx_.alloc(static_cast<std::size_t>(soft) * inter_m * 2);
  auto bout = ctx_.alloc(static_cast<std::size_t>(soft) * out_hidden * 2);
  {
    LayerNormParams p{static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(hidden), 1e-6f,
                      1u};
    const void* bufs[] = {bx.handle(), W("merger.norm.weight"), W("merger.norm.bias"),
                          bnorm.handle()};
    ctx_.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups,
                  static_cast<std::size_t>(tokens), 256, bufs, nullptr, 4, &p, sizeof(p));
  }
  gemm(W("merger.fc1.weight"), bnorm, bmi, W("merger.fc1.bias"), inter_m, inter_m, soft);
  {
    // erf, not tanh -- the merger's nn.GELU() defaults to approximate='none'.
    ElemParams p{static_cast<std::uint32_t>(soft) * static_cast<std::uint32_t>(inter_m), 1.0f};
    const void* bufs[] = {bmi.handle()};
    ctx_.dispatch("cpi_gelu_erf", runtime::MetalContext::Grid::Threads,
                  static_cast<std::size_t>(soft) * inter_m, 256, bufs, nullptr, 1, &p, sizeof(p));
  }
  gemm(W("merger.fc2.weight"), bmi, bout, W("merger.fc2.bias"), out_hidden, inter_m, soft);
  ctx_.commit_and_wait();

  const auto* o = static_cast<const std::uint16_t*>(bout.contents());
  std::vector<float> result(static_cast<std::size_t>(soft) * out_hidden);
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = f16_bits_to_f32(o[i]);
  return result;
}

}  // namespace engine
