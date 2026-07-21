// Expanding an image placeholder into a real image span.
//
// The chat text carries a single `<|image|>` placeholder token -- exactly how HF's
// processor does it -- and this expands that one token into
//   <boi> <image> x N <eoi>
// where the N image tokens' EMBEDDINGS are the vision tower's soft tokens. Doing it this
// way means the ordinary chat-template rendering (CLI or web) needs no image-awareness
// at all: it just leaves a placeholder where the picture goes.
//
// A TEMPLATE over the engine, not a PlanCudaEngine function: PlanMetalEngine now carries the
// same four-method surface (has_vision / can_sequence_prefill / vision_config / encode_image),
// and this file is the one place the splice layout lives -- a per-backend copy is how the two
// would drift. Header-only so a CUDA-free build never sees a CUDA type.

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/image_preprocess.hpp"
#include "model/png.hpp"

namespace app {
namespace image_prompt {

// Gemma 4's markers. (Ids, not strings: the tokenizer knows them as added tokens.)
constexpr int kBoiToken = 255999;
constexpr int kImageToken = 258880;
constexpr int kEoiToken = 258882;
constexpr int kSoftTokenBudget = 280;  // vision_soft_tokens_per_image

struct ImagePrompt {
  std::vector<int> tokens;
  std::vector<std::vector<float>> embeds;  // per token; empty = look the token up as usual
  std::vector<int> limits;                 // per-token key limit (bidirectional image span)
  int image_tokens = 0;
  int grid_w = 0, grid_h = 0;
};

// Encodes an image and splices it in at the first `<|image|>` placeholder in
// `base_tokens`. Throws if the model has no vision tower or there is no placeholder.
//
// `causal_span` forces the image to be causal instead of bidirectional (bisection aid).
template <typename Engine>
ImagePrompt expand(Engine& eng, const std::vector<int>& base_tokens, const std::string& png_path,
                   bool causal_span = false) {
  if (!eng.has_vision()) throw std::runtime_error("this model has no vision tower");
  if (!eng.can_sequence_prefill())
    throw std::runtime_error("this plan cannot sequence-prefill, which images require");

  const auto it = std::find(base_tokens.begin(), base_tokens.end(), kImageToken);
  if (it == base_tokens.end())
    throw std::runtime_error("the prompt has no <|image|> placeholder to expand");
  const std::size_t at = static_cast<std::size_t>(it - base_tokens.begin());

  const auto& v = eng.vision_config();
  const model::image::Image img = model::image::load_png(png_path);
  // Not padded: we encode one image, and HF's padding exists only for batching (it then
  // masks the padding out of attention). Unmasked padding patches would join the
  // encoder's bidirectional attention as zero-valued keys and dilute every softmax.
  const model::image::PatchGrid grid = model::image::to_patches(
      img, v.patch_size, v.pooling_kernel, kSoftTokenBudget, /*pad_to_budget=*/false);

  const std::vector<float> soft =
      eng.encode_image(grid.pixels, grid.pos_x, grid.pos_y, grid.soft_tokens);
  // CPI_VISION_SOFT_DUMP=<file>: raw f32 soft tokens. Both backends' towers flow through this
  // one engine-agnostic point, so diffing two dumps of the same image gates the tower itself.
  if (const char* dump = std::getenv("CPI_VISION_SOFT_DUMP")) {
    if (std::FILE* f = std::fopen(dump, "wb")) {
      std::fwrite(soft.data(), sizeof(float), soft.size(), f);
      std::fclose(f);
    }
  }
  const int H = static_cast<int>(soft.size()) / grid.soft_tokens;
  // One image token per LIVE pooled cell -- NOT per padded cell. Using the padded budget
  // slides the span out of alignment with the real soft tokens.
  const int n_image = grid.live_soft_tokens;

  ImagePrompt out;
  out.image_tokens = n_image;
  out.grid_w = grid.grid_w;
  out.grid_h = grid.grid_h;

  auto push = [&](int tok, const float* emb) {
    out.tokens.push_back(tok);
    if (emb) {
      out.embeds.emplace_back(emb, emb + H);
    } else {
      out.embeds.emplace_back();
    }
  };

  for (std::size_t i = 0; i < at; ++i) push(base_tokens[i], nullptr);
  const int span_start = static_cast<int>(out.tokens.size());
  push(kBoiToken, nullptr);
  for (int i = 0; i < n_image; ++i) push(kImageToken, &soft[static_cast<std::size_t>(i) * H]);
  push(kEoiToken, nullptr);
  const int span_end = static_cast<int>(out.tokens.size());
  for (std::size_t i = at + 1; i < base_tokens.size(); ++i) push(base_tokens[i], nullptr);

  // The image span is bidirectional (every token in it sees the whole span); the text
  // around it stays causal.
  out.limits.resize(out.tokens.size());
  for (int i = 0; i < static_cast<int>(out.tokens.size()); ++i) {
    const bool in_span = !causal_span && i >= span_start && i < span_end;
    out.limits[static_cast<std::size_t>(i)] = in_span ? span_end : i + 1;
  }
  if (causal_span) out.limits.clear();
  return out;
}

}  // namespace image_prompt
}  // namespace app
