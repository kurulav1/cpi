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
  // What the ENCODER emits: padded_patches / pooling^2 (HF's output_length).
  int soft_tokens = 0;
  // What the TEXT STREAM reserves: the LIVE grid only, (grid_w/k) * (grid_h/k). The
  // trailing padded cells are not image tokens -- inserting them shifts the whole span
  // out of alignment with the real soft tokens, and the model then reports seeing no
  // image at all.
  int live_soft_tokens = 0;
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

// A Qwen2-VL / Qwen3.5 patch tensor: what encode_image consumes, and nothing Gemma-shaped.
struct Qwen2VLPatches {
  std::vector<float> pixels;  // [grid_h * grid_w, 3 * temporal_patch * patch^2], normalised
  int grid_h = 0;             // patches down (resized_h / patch)
  int grid_w = 0;             // patches across
};

// Preprocess an RGB image the way Qwen2VLImageProcessor does, hand-rolled and gated against it
// (tools/qwen35_preproc_oracle.py, tests/metal_preproc_test).
//
// Four steps, each of which a reimplementation gets wrong quietly:
//   1. SMART RESIZE -- the target size is not fixed. Both sides are rounded to a multiple of
//      patch*merge with the pixel area held inside [min_pixels, max_pixels]; a 140x200 image
//      becomes 224x320, not something derived from a single "shortest edge".
//   2. bicubic resample to that size (PIL filter, a = -0.5).
//   3. normalise: (pixel/255 - mean) / std, per channel.
//   4. patchify into MERGE-UNIT-MAJOR order -- the same order the tower's position table and
//      RoPE use -- with the single frame repeated temporal_patch times.
Qwen2VLPatches qwen2vl_preprocess(const Image& img, int patch_size, int temporal_patch_size,
                                  int merge_size, const float mean[3], const float std[3],
                                  long min_pixels, long max_pixels);

}  // namespace image
}  // namespace model
