// Image -> patch grid (Gemma 4's processor, hand-rolled).

#include "model/image_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace model {
namespace image {
namespace {

// PIL's BICUBIC kernel (a = -0.5). Note PIL uses a = -0.5, not the -0.75 that OpenCV
// uses; they give visibly different results, and HF resamples with PIL.
float bicubic_kernel(float x) {
  constexpr float a = -0.5f;
  x = std::fabs(x);
  if (x < 1.0f) {
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
  }
  if (x < 2.0f) {
    return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
  }
  return 0.0f;
}

// One resampling pass along a single axis, reproducing PIL's 8-bit path exactly.
//
// Two details make it exact rather than merely close, and both are invisible until you
// diff against PIL:
//   - the filter coefficients are quantised to fixed point (22 fractional bits), not
//     used as doubles;
//   - the horizontal pass writes an 8-BIT intermediate, so the vertical pass resamples
//     already-rounded values.
// Downscaling stretches the filter support by 1/scale; that is what "antialias" means.
// Without it a 2x downscale point-samples and aliases badly (tens of levels off).
constexpr int kPrecisionBits = 22;

std::uint8_t clip8(std::int64_t v) {
  const std::int64_t x = v >> kPrecisionBits;
  if (x < 0) return 0;
  if (x > 255) return 255;
  return static_cast<std::uint8_t>(x);
}

void resample_axis(const std::vector<std::uint8_t>& src, std::vector<std::uint8_t>& dst,
                   int src_len, int dst_len, int other_len, int channels, bool horizontal) {
  const double scale = static_cast<double>(src_len) / static_cast<double>(dst_len);
  const double filter_scale = scale >= 1.0 ? scale : 1.0;
  const double support = 2.0 * filter_scale;  // bicubic support is 2

  for (int i = 0; i < dst_len; ++i) {
    const double center = (static_cast<double>(i) + 0.5) * scale;
    int lo = static_cast<int>(center - support + 0.5);
    int hi = static_cast<int>(center + support + 0.5);
    lo = std::max(lo, 0);
    hi = std::min(hi, src_len);
    if (hi <= lo) {
      lo = std::min(std::max(static_cast<int>(center), 0), src_len - 1);
      hi = lo + 1;
    }

    // double weights, normalised, then quantised to fixed point exactly as PIL does
    std::vector<double> w(static_cast<std::size_t>(hi - lo));
    double wsum = 0.0;
    for (int k = lo; k < hi; ++k) {
      const double t = (static_cast<double>(k) - center + 0.5) / filter_scale;
      const double weight = bicubic_kernel(t);
      w[static_cast<std::size_t>(k - lo)] = weight;
      wsum += weight;
    }
    std::vector<std::int64_t> kk(w.size());
    for (std::size_t j = 0; j < w.size(); ++j) {
      const double norm = wsum != 0.0 ? w[j] / wsum : 0.0;
      const double scaled = norm * (1 << kPrecisionBits);
      kk[j] = scaled < 0 ? -static_cast<std::int64_t>(-scaled + 0.5)
                         : static_cast<std::int64_t>(scaled + 0.5);
    }

    for (int j = 0; j < other_len; ++j) {
      for (int c = 0; c < channels; ++c) {
        std::int64_t acc = static_cast<std::int64_t>(1) << (kPrecisionBits - 1);  // rounding
        for (int k = lo; k < hi; ++k) {
          const std::size_t idx =
              horizontal
                  ? (static_cast<std::size_t>(j) * src_len + k) * channels + c
                  : (static_cast<std::size_t>(k) * other_len + j) * channels + c;
          acc += static_cast<std::int64_t>(src[idx]) * kk[static_cast<std::size_t>(k - lo)];
        }
        const std::size_t out =
            horizontal ? (static_cast<std::size_t>(j) * dst_len + i) * channels + c
                       : (static_cast<std::size_t>(i) * other_len + j) * channels + c;
        dst[out] = clip8(acc);
      }
    }
  }
}

}  // namespace

Image resize_bicubic(const Image& src, int out_w, int out_h) {
  if (out_w <= 0 || out_h <= 0) throw std::runtime_error("image: bad resize target");
  if (out_w == src.width && out_h == src.height) return src;

  constexpr int kC = 3;
  // Horizontal pass, then vertical; with an 8-bit intermediate, as PIL does.
  std::vector<std::uint8_t> mid(static_cast<std::size_t>(out_w) * src.height * kC);
  resample_axis(src.rgb, mid, src.width, out_w, src.height, kC, /*horizontal=*/true);

  Image dst;
  dst.width = out_w;
  dst.height = out_h;
  dst.rgb.assign(static_cast<std::size_t>(out_w) * out_h * kC, 0);
  resample_axis(mid, dst.rgb, src.height, out_h, out_w, kC, /*horizontal=*/false);
  return dst;
}

PatchGrid to_patches(const Image& img, int patch_size, int pooling_kernel, int max_soft_tokens,
                     bool pad_to_budget) {
  if (patch_size <= 0 || pooling_kernel <= 0 || max_soft_tokens <= 0)
    throw std::runtime_error("image: bad patch parameters");

  // Largest aspect-preserving size that (a) fits the patch budget and (b) has both
  // sides divisible by patch*pooling; so the patch grid pools evenly.
  const int max_patches = max_soft_tokens * pooling_kernel * pooling_kernel;
  const double total_px = static_cast<double>(img.width) * img.height;
  const double target_px = static_cast<double>(max_patches) * patch_size * patch_size;
  const double factor = std::sqrt(target_px / total_px);
  const int side_mult = patch_size * pooling_kernel;
  int th = static_cast<int>(std::floor(factor * img.height / side_mult)) * side_mult;
  int tw = static_cast<int>(std::floor(factor * img.width / side_mult)) * side_mult;
  // A very lopsided or very small image can round a side to zero; one cell is the floor.
  th = std::max(th, side_mult);
  tw = std::max(tw, side_mult);

  const Image r = resize_bicubic(img, tw, th);

  PatchGrid g;
  g.grid_w = tw / patch_size;
  g.grid_h = th / patch_size;
  const int live = g.grid_w * g.grid_h;
  const int patch_dim = 3 * patch_size * patch_size;
  const int total = pad_to_budget ? max_patches : live;

  g.pixels.assign(static_cast<std::size_t>(total) * patch_dim, 0.0f);
  g.pos_x.assign(static_cast<std::size_t>(total), -1);  // -1 = padding
  g.pos_y.assign(static_cast<std::size_t>(total), -1);
  g.num_patches = total;
  g.soft_tokens = total / (pooling_kernel * pooling_kernel);
  g.live_soft_tokens = (g.grid_w / pooling_kernel) * (g.grid_h / pooling_kernel);

  for (int py = 0; py < g.grid_h; ++py) {
    for (int px = 0; px < g.grid_w; ++px) {
      const int p = py * g.grid_w + px;  // patches are row-major
      g.pos_x[static_cast<std::size_t>(p)] = px;
      g.pos_y[static_cast<std::size_t>(p)] = py;
      float* out = &g.pixels[static_cast<std::size_t>(p) * patch_dim];
      // Within a patch the order is [y][x][channel]; CHANNEL INNERMOST.
      for (int y = 0; y < patch_size; ++y) {
        for (int x = 0; x < patch_size; ++x) {
          const int sy = py * patch_size + y;
          const int sx = px * patch_size + x;
          const std::uint8_t* src = &r.rgb[(static_cast<std::size_t>(sy) * tw + sx) * 3];
          float* dstp = out + (static_cast<std::size_t>(y) * patch_size + x) * 3;
          dstp[0] = static_cast<float>(src[0]) / 255.0f;  // rescale to [0,1]; no mean/std
          dstp[1] = static_cast<float>(src[1]) / 255.0f;
          dstp[2] = static_cast<float>(src[2]) / 255.0f;
        }
      }
    }
  }
  return g;
}

namespace {

// Qwen2-VL smart_resize: round both sides to a multiple of `factor`, keeping the pixel area in
// [min_pixels, max_pixels] and the aspect ratio. Verified against the processor: 140x200,
// factor 32 -> 224x320.
void smart_resize(int h, int w, int factor, long min_pixels, long max_pixels, int& out_h,
                  int& out_w) {
  auto round_mult = [factor](double v) {
    return static_cast<int>(std::lround(v / factor)) * factor;
  };
  int hb = std::max(factor, round_mult(h));
  int wb = std::max(factor, round_mult(w));
  const double area = static_cast<double>(h) * w;
  if (static_cast<long>(hb) * wb > max_pixels) {
    const double beta = std::sqrt(area / static_cast<double>(max_pixels));
    hb = static_cast<int>(std::floor(h / beta / factor)) * factor;
    wb = static_cast<int>(std::floor(w / beta / factor)) * factor;
  } else if (static_cast<long>(hb) * wb < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / area);
    hb = static_cast<int>(std::ceil(h * beta / factor)) * factor;
    wb = static_cast<int>(std::ceil(w * beta / factor)) * factor;
  }
  out_h = std::max(factor, hb);
  out_w = std::max(factor, wb);
}

}  // namespace

Qwen2VLPatches qwen2vl_preprocess(const Image& img, int patch_size, int temporal_patch_size,
                                  int merge_size, const float mean[3], const float std[3],
                                  long min_pixels, long max_pixels) {
  if (patch_size <= 0 || temporal_patch_size <= 0 || merge_size <= 0) {
    throw std::runtime_error("qwen2vl: bad patch parameters");
  }
  const int factor = patch_size * merge_size;
  int rh = 0, rw = 0;
  smart_resize(img.height, img.width, factor, min_pixels, max_pixels, rh, rw);

  const Image r = resize_bicubic(img, rw, rh);

  Qwen2VLPatches out;
  out.grid_h = rh / patch_size;
  out.grid_w = rw / patch_size;
  const int gh = out.grid_h, gw = out.grid_w;
  const int C = 3;
  const int patch_dim = C * temporal_patch_size * patch_size * patch_size;
  out.pixels.assign(static_cast<std::size_t>(gh) * gw * patch_dim, 0.0f);

  // Rows in MERGE-UNIT-MAJOR order: (block_h, block_w, merge_h, merge_w), matching the tower.
  // Row content: (channel, temporal, patch_h, patch_w). The single frame is repeated across the
  // temporal axis, so every temporal slice is identical; but it must still be written, or the
  // patch_dim stride is wrong and the tower reads channels out of a neighbouring patch.
  const int bh = gh / merge_size, bw = gw / merge_size;
  std::size_t row = 0;
  for (int ph_b = 0; ph_b < bh; ++ph_b) {
    for (int pw_b = 0; pw_b < bw; ++pw_b) {
      for (int mi = 0; mi < merge_size; ++mi) {
        for (int mj = 0; mj < merge_size; ++mj) {
          const int patch_row = ph_b * merge_size + mi;  // which patch down
          const int patch_col = pw_b * merge_size + mj;  // which patch across
          float* dst = out.pixels.data() + row * patch_dim;
          for (int c = 0; c < C; ++c) {
            for (int tp = 0; tp < temporal_patch_size; ++tp) {
              for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                  const int sy = patch_row * patch_size + y;
                  const int sx = patch_col * patch_size + x;
                  const std::uint8_t v =
                      r.rgb[(static_cast<std::size_t>(sy) * rw + sx) * 3 + c];
                  const float norm = (static_cast<float>(v) / 255.0f - mean[c]) / std[c];
                  const std::size_t idx =
                      ((static_cast<std::size_t>(c) * temporal_patch_size + tp) * patch_size + y) *
                          patch_size +
                      x;
                  dst[idx] = norm;
                }
              }
            }
          }
          ++row;
        }
      }
    }
  }
  return out;
}

}  // namespace image
}  // namespace model
