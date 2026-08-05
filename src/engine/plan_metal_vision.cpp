// The Qwen3.5 vision tower, inside the engine.
//
// Separate translation unit from plan_metal_engine.cpp, which is already ~2200 lines. Nothing
// here is shared with the text path except the kernels and the weight source: the tower runs
// once per image, over a patch grid rather than a token sequence, at its own hidden size. Its
// buffers are therefore allocated per call rather than drawn from the engine's slot pool, which
// is sized for text geometry; the tower's cost is twelve blocks of GEMMs, so a handful of
// allocations does not register.
//
// The arithmetic here is the same sequence metal_vision_test gates against the HuggingFace
// oracle. That test now also runs this path (see encode_image there), so the two cannot drift.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>

#include "engine/plan_metal_engine.hpp"
#include "engine/plan_model_config.hpp"
#include "model/json_mini.hpp"
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
// the samples are linspace(0, side-1, n) with the endpoint included, and patch (i, j) does not
// land at row i*w + j; the blocks see patches grouped by the 2x2 unit the merger later folds.
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
// The frequency vector is [row_freqs | col_freqs], each head_dim/4 long; not one geometric
// series over head_dim/2; and the tokens are enumerated in the same merge-unit-major order
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
//     its merged grid, then advances next by max(merged_h, merged_w).
//
// That last step is the one that matters and the one a reimplementation gets wrong. An image of
// 16 tokens advances the counter by 4, not 16, so every text token after an image sits four
// positions past the image's start; 5..9 for the reference prompt, where a 1-D counter puts
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
    // A run of image tokens. Its length must be exactly merged_h * merged_w; one placeholder
    // per soft token; or the grid this walks does not describe the run.
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

// Where the M-RoPE counter ended; the position the first generated token takes. Not the token
// count: an image span advances the counter by its merged extent, so after 16 image tokens the
// two differ by 12 and using the token count rotates every generated token wrongly.
int mrope_next_position(const std::vector<int>& tokens, int image_token_id, int merged_h,
                        int merged_w) {
  const std::vector<std::int32_t> pos =
      build_mrope_positions(tokens, image_token_id, merged_h, merged_w);
  const int n = static_cast<int>(tokens.size());
  if (n == 0) return 0;
  // The last token's t axis, plus one. For a trailing text token all three axes agree; for a
  // trailing image token the counter has already advanced past the span, so take the max.
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
  // Resolve every vision tensor before the first dispatch. WeightSource::fp16 uploads on first
  // use, it allocates a Metal buffer and memcpys into it, and doing that while a compute
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
    // From the loader, not the WeightSource. WeightSource::fp16 hands back an opaque Metal
    // buffer handle, its own comment says these are never dereferenced as pointers, and
    // reading one as fp16 data is how this first went wrong: a SIGBUS, then, once the layout
    // shifted, 16384 NaNs that the test happily reported as pass.
    //
    // The position table is the one vision tensor the CPU has to read, because the resample is
    // done host-side; everything else stays on the GPU as a handle.
    const auto* tab = reinterpret_cast<const std::uint16_t*>(
        raw_data("vision.pos_embed.weight"));
    std::vector<float> tab32(static_cast<std::size_t>(c.vision_num_position_embeddings) * hidden);
    for (std::size_t i = 0; i < tab32.size(); ++i) tab32[i] = f16_bits_to_f32(tab[i]);
    const std::vector<float> pos =
        interpolate_pos_embed(tab32.data(), side, grid_h, grid_w, hidden, merge);
    // Add the position table in fp32 and round once, rather than rounding the table to fp16
    // first and adding in the kernel.
    //
    // The interpolated table is not the stored one: bilinear blending of four bf16 rows with
    // fractional weights produces values with full f32 mantissa content, so rounding it before
    // the add throws away bits the add would otherwise keep. Measured, this halves the
    // tower's error, ENGINE_encode_image rel 0.01010 -> 0.00508, and it is what the
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
      // tanh GELU here: the blocks specify hidden_act = "gelu_pytorch_tanh". The merger below
      // uses the exact erf form instead; they are different functions and the tower uses both.
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
    // Splitting the command buffer here is numerically neutral; MetalContext barriers before
    // every dispatch by default (metal_context.mm:430); so this observes without perturbing.
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
  // rows are one unit and [tokens][hidden] reinterprets as [tokens/4][4*hidden] in place.
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
    // erf, not tanh; the merger's nn.GELU() defaults to approximate='none'.
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

// ---------------------------------------------------------------------------
// Gemma 4's vision tower. Unlike Qwen3.5's hand-rolled sequence above, this one is a plan
// (build_gemma4_vision_plan, the same description PlanCudaEngine constructs) walked by the
// engine's own execute_ops with the slot resolver pointed at vision-sized buffers.

void PlanMetalEngine::init_gemma_vision(const std::string& model_dir) {
  namespace fs = std::filesystem;
  const fs::path cj = fs::path(model_dir) / "config.json";
  std::error_code ec;
  if (!fs::exists(cj, ec)) return;
  const opplan::Gemma4VisionGeometry v =
      engine::parse_gemma4_vision_config(engine::mini::read_text_file(cj));
  if (!v.present) return;  // a text-only export; the text path is unaffected
  if (!raw_has("model.vision_tower.patch_embedder.input_proj.weight")) return;
  if (v.kv_heads != v.heads) {
    // cpi_attention_bidirectional takes one row stride for q/k/v; see the executor's check.
    throw std::runtime_error("Gemma 4 vision tower with kv_heads != heads is not supported");
  }

  // cos/sin over the spatial half-dim, exactly PlanCudaEngine::build_vision_rope_tables:
  // inv_freq = theta^(-2j/spatial_dim) with spatial_dim = head_dim/2, head_dim/4 entries per
  // position, both axes sharing the table.
  {
    const int pairs = v.head_dim / 4;
    const int spatial_dim = v.head_dim / 2;
    const int max_pos = 4096;  // patch-grid coordinates; far above any real image
    std::vector<float> cs(static_cast<std::size_t>(max_pos) * pairs), ss(cs.size());
    for (int p = 0; p < max_pos; ++p) {
      for (int j = 0; j < pairs; ++j) {
        const float inv = std::pow(v.rope_theta,
                                   -static_cast<float>(2 * j) / static_cast<float>(spatial_dim));
        const float ang = static_cast<float>(p) * inv;
        cs[static_cast<std::size_t>(p) * pairs + j] = std::cos(ang);
        ss[static_cast<std::size_t>(p) * pairs + j] = std::sin(ang);
      }
    }
    gvis_rope_cos_ = ctx_.alloc_from(cs.data(), cs.size() * sizeof(float));
    gvis_rope_sin_ = ctx_.alloc_from(ss.data(), ss.size() * sizeof(float));
  }

  // Sized for the soft-token budget times the pool window; 280 * 3 * 3 = 2520 patches on E2B
  //; rounded up. The image preprocessor never emits more (it downscales to the budget).
  gvis_max_patches_ = 2560;
  const int P = gvis_max_patches_;
  const int H = v.hidden;
  const int qdim = v.heads * v.head_dim;
  const int kvdim = v.kv_heads * v.head_dim;
  const int patch_dim = 3 * v.patch_size * v.patch_size;

  vslots_.clear();
  vslots_.resize(static_cast<std::size_t>(opplan::Slot::Count));
  auto al = [&](opplan::Slot s, std::size_t per_token) {
    vslots_[static_cast<std::size_t>(s)] =
        ctx_.alloc(static_cast<std::size_t>(P) * per_token * sizeof(std::uint16_t));
  };
  using opplan::Slot;
  al(Slot::X, static_cast<std::size_t>(H));
  al(Slot::XNorm, static_cast<std::size_t>(H));
  // Tmp doubles as the projector's output, which is text-hidden wide.
  al(Slot::Tmp, static_cast<std::size_t>(std::max(H, cfg_.hidden_size)));
  al(Slot::Q, static_cast<std::size_t>(qdim));
  al(Slot::K, static_cast<std::size_t>(kvdim));
  al(Slot::V, static_cast<std::size_t>(kvdim));
  al(Slot::Att, static_cast<std::size_t>(qdim));
  al(Slot::Gate, static_cast<std::size_t>(v.intermediate));
  al(Slot::Up, static_cast<std::size_t>(v.intermediate));
  al(Slot::Inter, static_cast<std::size_t>(v.intermediate));

  gvis_pix_buf_ = ctx_.alloc(static_cast<std::size_t>(P) * patch_dim * sizeof(float));
  gvis_posx_buf_ = ctx_.alloc(static_cast<std::size_t>(P) * sizeof(std::int32_t));
  gvis_posy_buf_ = ctx_.alloc(static_cast<std::size_t>(P) * sizeof(std::int32_t));
  // clip_in staging: the widest gemv input is the MLP down-projection's (intermediate).
  gvis_clamp_buf_ = ctx_.alloc(static_cast<std::size_t>(P) *
                               static_cast<std::size_t>(std::max(H, v.intermediate)) *
                               sizeof(std::uint16_t));
  {
    std::vector<std::uint16_t> ones(static_cast<std::size_t>(H), cpi::f32_to_f16(1.0f));
    gvis_ones_buf_ = ctx_.alloc_from(ones.data(), ones.size() * sizeof(std::uint16_t));
  }

  // Last, so a throw above leaves gvis_.present false and the text path standing.
  gvis_plan_ = opplan::build_gemma4_vision_plan(v, *wsrc_);
  gvis_ = v;
}

std::vector<float> PlanMetalEngine::encode_image(const std::vector<float>& pixels,
                                                 const std::vector<int>& pos_x,
                                                 const std::vector<int>& pos_y, int out_tokens) {
  if (!gvis_.present) throw std::runtime_error("this model has no vision tower");
  const int P = static_cast<int>(pos_x.size());
  if (P <= 0 || static_cast<int>(pos_y.size()) != P) {
    throw std::runtime_error("pos_x / pos_y must be one entry per patch");
  }
  if (P > gvis_max_patches_) throw std::runtime_error("too many patches for the vision buffers");
  const int H = gvis_.hidden;
  const int patch_dim = 3 * gvis_.patch_size * gvis_.patch_size;
  if (static_cast<int>(pixels.size()) != P * patch_dim) {
    throw std::runtime_error("pixel buffer is " + std::to_string(pixels.size()) +
                             " floats, expected " + std::to_string(P * patch_dim));
  }

  std::memcpy(gvis_pix_buf_.contents(), pixels.data(), pixels.size() * sizeof(float));
  auto* px = static_cast<std::int32_t*>(gvis_posx_buf_.contents());
  auto* py = static_cast<std::int32_t*>(gvis_posy_buf_.contents());
  for (int i = 0; i < P; ++i) {
    px[i] = pos_x[static_cast<std::size_t>(i)];
    py[i] = pos_y[static_cast<std::size_t>(i)];
  }

  vision_pass_ = true;
  try {
    // The tower is bidirectional and cacheless: one pass over all P patches.
    execute_ops(gvis_plan_.prologue, -1, 0, P);
    for (const opplan::LayerPlan& lp : gvis_plan_.layers) {
      execute_ops(lp.ops, lp.layer_index, 0, P);
    }
    execute_ops(gvis_plan_.epilogue, -1, 0, P);

    // Pool the patches down to out_tokens soft tokens, then project into text space; direct
    // dispatches, not plan ops, because pooling changes the token count (see the builder note).
    const int k = gvis_.pooling_kernel;
    int max_x = 0;
    for (int i = 0; i < P; ++i) max_x = std::max(max_x, pos_x[static_cast<std::size_t>(i)]);
    const int cells_x = (max_x / k) + 1;
    {
      struct VisPoolParams {
        std::uint32_t tokens, hidden, k, cells_x, out_tokens;
        float gain;
      } p{static_cast<std::uint32_t>(P),       static_cast<std::uint32_t>(H),
          static_cast<std::uint32_t>(k),       static_cast<std::uint32_t>(cells_x),
          static_cast<std::uint32_t>(out_tokens),
          std::sqrt(static_cast<float>(H))};
      const void* bufs[] = {slot(opplan::Slot::X), gvis_posx_buf_.handle(),
                            gvis_posy_buf_.handle(), slot(opplan::Slot::XNorm)};
      ctx_.dispatch("cpi_avg_pool_patches", runtime::MetalContext::Grid::Threads,
                    static_cast<std::size_t>(out_tokens) * static_cast<std::size_t>(H), 256, bufs,
                    nullptr, 4, &p, sizeof(p));
    }
    // Projector: weightless RMS norm (ones weight, mirroring CUDA's launch_rmsnorm with
    // d_ones_) -> linear into the text hidden size.
    {
      struct NormParams {
        std::uint32_t rows, cols;
        float eps;
        std::uint32_t weight_offset, has_weight;
      } p{static_cast<std::uint32_t>(out_tokens), static_cast<std::uint32_t>(H), gvis_.rms_eps, 0u,
          1u};
      const void* bufs[] = {slot(opplan::Slot::XNorm), gvis_ones_buf_.handle(),
                            slot(opplan::Slot::XNorm)};
      ctx_.dispatch("cpi_rmsnorm", runtime::MetalContext::Grid::Groups,
                    static_cast<std::size_t>(out_tokens), 256, bufs, nullptr, 3, &p, sizeof(p));
    }
    {
      const void* w = wsrc_->fp16("model.embed_vision.embedding_projection.weight");
      GemvParams p{static_cast<std::uint32_t>(cfg_.hidden_size), static_cast<std::uint32_t>(H),
                   static_cast<std::uint32_t>(out_tokens), 0u};
      const void* bufs[] = {w, slot(opplan::Slot::XNorm), slot(opplan::Slot::Tmp), w};
      // text hidden is a multiple of kGemmFBM on every Gemma 4 (E2B: 1536) and the vision hidden
      // a multiple of 32, so the blocked GEMM's geometry holds; its store masks the token tail.
      const std::size_t tiles =
          static_cast<std::size_t>((out_tokens + kGemmBN - 1) / kGemmBN);
      const std::size_t groups =
          (static_cast<std::size_t>(cfg_.hidden_size) / kGemmFBM) * tiles;
      ctx_.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, groups, kGemmTG, bufs,
                    nullptr, 4, &p, sizeof(p));
    }
    ctx_.commit_and_wait();
  } catch (...) {
    vision_pass_ = false;
    throw;
  }
  vision_pass_ = false;

  const auto* o = static_cast<const std::uint16_t*>(
      vslots_[static_cast<std::size_t>(opplan::Slot::Tmp)].contents());
  std::vector<float> out(static_cast<std::size_t>(out_tokens) *
                         static_cast<std::size_t>(cfg_.hidden_size));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = f16_bits_to_f32(o[i]);
  return out;
}

std::vector<int> PlanMetalEngine::generate_multimodal(const std::vector<int>& tokens,
                                                      const std::vector<std::vector<float>>& embeds,
                                                      const std::vector<int>& limits, int max_new,
                                                      float temp,
                                                      const std::function<bool(int)>& on_token) {
  const int n = static_cast<int>(tokens.size());
  if (n == 0) return {};
  if (!limits.empty() && static_cast<int>(limits.size()) != n) {
    throw std::runtime_error("limits must have one entry per token");
  }
  // A fresh sequence: prefix reuse would pair a previous run's KV with this image's splice.
  reset_kv_cache();

  embeds_ = &embeds;
  limits_active_ = !limits.empty();
  int pos = 0;
  try {
    for (const int chunk : prefill_chunks(n)) {
      if (limits_active_) {
        const std::size_t need = static_cast<std::size_t>(chunk) * sizeof(std::int32_t);
        if (seq_limits_buf_.size() < need) seq_limits_buf_ = ctx_.alloc(need);
        auto* dst = static_cast<std::int32_t*>(seq_limits_buf_.contents());
        for (int t = 0; t < chunk; ++t) {
          const int lim = limits[static_cast<std::size_t>(pos + t)];
          // A span token's keys must all be written when its chunk's attention runs; a span
          // straddling a chunk boundary would attend over unwritten cache. Prompts are far
          // below one chunk today; splitting chunks at span edges is the fix if that changes.
          if (lim > pos + chunk) {
            throw std::runtime_error("bidirectional image span straddles a prefill chunk "
                                     "boundary (limit " +
                                     std::to_string(lim) + " > chunk end " +
                                     std::to_string(pos + chunk) + ")");
          }
          dst[t] = lim;
        }
        // CPI_LIMITS_TEST=1: clamp every token to key 0 only. If the output does not change
        // under this, the kernels are not reading the limits at all.
        if (std::getenv("CPI_LIMITS_TEST") != nullptr) {
          for (int t = 0; t < chunk; ++t) dst[t] = 1;
        }
      }
      const std::vector<int> ids(tokens.begin() + pos, tokens.begin() + pos + chunk);
      encode_prefill(ids, pos);
      ctx_.commit_and_wait();
      pos += chunk;
    }
  } catch (...) {
    embeds_ = nullptr;
    limits_active_ = false;
    throw;
  }
  embeds_ = nullptr;
  limits_active_ = false;

  // The prefill's epilogue left the last token's logits in logits_buf_ (float). Sample from
  // there, then continue on the ordinary decode path.
  std::vector<int> out;
  std::uint32_t rng = 0x9E3779B9u;  // deterministic; --image has no seed flag
  auto sample_from = [&](const float* lg) -> int {
    const int V = cfg_.vocab_size;
    if (temp <= 0.0f) {
      int best = 0;
      for (int i = 1; i < V; ++i) {
        if (lg[i] > lg[best]) best = i;
      }
      return best;
    }
    float m = lg[0];
    for (int i = 1; i < V; ++i) m = std::max(m, lg[i]);
    double sum = 0.0;
    std::vector<double> p(static_cast<std::size_t>(V));
    for (int i = 0; i < V; ++i) {
      p[static_cast<std::size_t>(i)] = std::exp((lg[i] - m) / temp);
      sum += p[static_cast<std::size_t>(i)];
    }
    rng = rng * 1664525u + 1013904223u;
    double r = (rng / 4294967296.0) * sum;
    for (int i = 0; i < V; ++i) {
      r -= p[static_cast<std::size_t>(i)];
      if (r <= 0.0) return i;
    }
    return V - 1;
  };

  int next = sample_from(static_cast<const float*>(logits_buf_.contents()));
  for (int i = 0; i < max_new; ++i) {
    if (next == eos_token_id_) break;
    out.push_back(next);
    if (on_token && !on_token(next)) break;
    if (i + 1 >= max_new) break;
    const std::vector<float>& lg = forward_token(next, n + i);
    next = sample_from(lg.data());
  }
  return out;
}

}  // namespace engine
