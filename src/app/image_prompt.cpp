#include "app/image_prompt.hpp"

#include <algorithm>
#include <stdexcept>

#include "model/image_preprocess.hpp"
#include "model/png.hpp"

namespace app {
namespace image_prompt {

ImagePrompt expand(engine::PlanCudaEngine& eng, const std::vector<int>& base_tokens,
                   const std::string& png_path, bool causal_span) {
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
