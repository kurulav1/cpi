// Image -> patch grid, the way Gemma 4's processor does it.
//
// Hand-rolled, including the bicubic resample. The resize is the one step where a
// small deviation would quietly cost accuracy rather than fail loudly, so it matches
// PIL's filter (the one HF actually uses) and is gated against it.

#pragma once

#include <vector>

#include "model/png.hpp"

namespace model {
namespace image {

struct PatchGrid {
  std::vector<float> pixels;  // [num_patches, 3 * patch^2], channel INNERMOST, in [0,1]
  std::vector<int> pos_x;     // patch grid coords; -1 marks a padding patch
  std::vector<int> pos_y;
  int num_patches = 0;   // including padding
  int soft_tokens = 0;   // num_patches / pooling^2 -- what the text stream must reserve
  int grid_w = 0;
  int grid_h = 0;
};

// Resize preserving aspect ratio to the largest size that fits the patch budget and
// whose sides divide (patch_size * pooling_kernel), then cut into patches.
//
//   max_soft_tokens - the model's vision_soft_tokens_per_image (280 for Gemma 4)
//
// If `pad_to_budget`, the grid is padded up to max_soft_tokens worth of patches with
// (-1,-1) padding patches, so the token stream always reserves the same number of
// image tokens.
PatchGrid to_patches(const Image& img, int patch_size, int pooling_kernel, int max_soft_tokens,
                     bool pad_to_budget = true);

// PIL-compatible bicubic resample (a = -0.5, with antialiasing when downscaling).
// Exposed so it can be gated directly against PIL.
Image resize_bicubic(const Image& src, int out_w, int out_h);

}  // namespace image
}  // namespace model
