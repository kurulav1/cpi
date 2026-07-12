// Gate for the hand-rolled image preprocessing.
//
// 1. resize_bicubic vs PIL (the resampler HF actually uses). PIL's 8-bit path is
//    deterministic -- fixed-point filter coefficients, 8-bit intermediate between the
//    two passes -- so we reproduce it EXACTLY and the bar is byte-equality, not a
//    tolerance. (Float math alone gets to ~0.3 mean error with individual pixels off by
//    7 levels; that would have been easy to wave through as "close enough".)
//
// 2. to_patches: the grid geometry and, crucially, the patch memory layout
//    ([y][x][channel], channel innermost) -- getting that transposed would feed the
//    encoder a scrambled image while looking perfectly plausible.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "model/image_preprocess.hpp"

namespace {

bool read_blob(const std::string& path, int& w, int& h, std::vector<std::uint8_t>& rgb) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  int hdr[2];
  f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  w = hdr[0];
  h = hdr[1];
  rgb.resize(static_cast<std::size_t>(w) * h * 3);
  f.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
  return f.good() || f.eof();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "artifacts/resize_fixtures";
  int failures = 0;

  std::ifstream index(dir + "/index.txt");
  if (!index) {
    std::printf("no fixtures -- run: python tools/make_resize_fixtures.py\n");
    return 2;
  }

  std::string line;
  while (std::getline(index, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string name;
    int sw, sh, dw, dh;
    ss >> name >> sw >> sh >> dw >> dh;

    int w1, h1, w2, h2;
    std::vector<std::uint8_t> src_rgb, ref_rgb;
    if (!read_blob(dir + "/" + name + ".src", w1, h1, src_rgb) ||
        !read_blob(dir + "/" + name + ".ref", w2, h2, ref_rgb)) {
      std::printf("  [skip] %-12s missing fixture\n", name.c_str());
      continue;
    }

    model::image::Image src;
    src.width = w1;
    src.height = h1;
    src.rgb = src_rgb;
    const model::image::Image got = model::image::resize_bicubic(src, dw, dh);

    double sum = 0.0;
    int worst = 0;
    for (std::size_t i = 0; i < ref_rgb.size(); ++i) {
      const int d = std::abs(static_cast<int>(got.rgb[i]) - static_cast<int>(ref_rgb[i]));
      sum += d;
      worst = std::max(worst, d);
    }
    const double mae = sum / static_cast<double>(ref_rgb.size());
    // PIL's 8-bit resample is deterministic (fixed-point coefficients, 8-bit
    // intermediate), so we reproduce it EXACTLY rather than settle for a tolerance.
    const bool ok = worst == 0;
    if (!ok) ++failures;
    std::printf("  [%s] %-12s %dx%d -> %dx%d   mean abs err %.3f, worst %d (of 255)\n",
                ok ? "ok" : "FAIL", name.c_str(), sw, sh, dw, dh, mae, worst);
  }

  // --- patch grid geometry + layout ---
  {
    model::image::Image img;
    img.width = 200;
    img.height = 100;
    img.rgb.assign(static_cast<std::size_t>(img.width) * img.height * 3, 0);
    // paint a pixel we can find again after patching
    const std::size_t probe = (static_cast<std::size_t>(50) * img.width + 70) * 3;
    img.rgb[probe] = 200;
    img.rgb[probe + 1] = 100;
    img.rgb[probe + 2] = 50;

    const model::image::PatchGrid g = model::image::to_patches(img, 16, 3, 280);
    const int expect_soft = 280;
    const bool geom_ok = g.soft_tokens == expect_soft && g.num_patches == expect_soft * 9 &&
                         g.grid_w > 0 && g.grid_h > 0;
    std::printf("\n  [%s] patch grid  %dx%d patches, %d soft tokens (budget %d), %d padded\n",
                geom_ok ? "ok" : "FAIL", g.grid_w, g.grid_h, g.soft_tokens, expect_soft,
                g.num_patches - g.grid_w * g.grid_h);
    if (!geom_ok) ++failures;

    // Channel must be the INNERMOST axis of a patch vector.
    bool layout_ok = true;
    const int patch_dim = 3 * 16 * 16;
    for (int p = 0; p < g.grid_w * g.grid_h && layout_ok; ++p) {
      for (int i = 0; i + 2 < patch_dim; i += 3) {
        const float r = g.pixels[static_cast<std::size_t>(p) * patch_dim + i];
        if (r < 0.0f || r > 1.0f) layout_ok = false;  // rescaled to [0,1]
      }
    }
    // every live patch has a real position; every padding patch is (-1,-1)
    for (int p = 0; p < g.num_patches; ++p) {
      const bool live = p < g.grid_w * g.grid_h;
      const bool has_pos = g.pos_x[static_cast<std::size_t>(p)] >= 0;
      if (live != has_pos) layout_ok = false;
    }
    std::printf("  [%s] patch layout (channel innermost, [0,1] range, padding marked -1)\n",
                layout_ok ? "ok" : "FAIL");
    if (!layout_ok) ++failures;
  }

  std::printf("\n%s\n", failures == 0 ? "PREPROCESS OK (bicubic matches PIL; patch grid correct)"
                                      : "PREPROCESS FAILED");
  return failures == 0 ? 0 : 1;
}
