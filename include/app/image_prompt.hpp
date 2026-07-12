// Expanding an image placeholder into a real image span.
//
// The chat text carries a single `<|image|>` placeholder token -- exactly how HF's
// processor does it -- and this expands that one token into
//   <boi> <image> x N <eoi>
// where the N image tokens' EMBEDDINGS are the vision tower's soft tokens. Doing it this
// way means the ordinary chat-template rendering (CLI or web) needs no image-awareness
// at all: it just leaves a placeholder where the picture goes.

#pragma once

#include <string>
#include <vector>

#include "engine/plan_cuda_engine.hpp"
#include "model/tokenizer.hpp"

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
ImagePrompt expand(engine::PlanCudaEngine& eng, const std::vector<int>& base_tokens,
                   const std::string& png_path, bool causal_span = false);

}  // namespace image_prompt
}  // namespace app
