// Pixel-exact gate for the hand-rolled PNG decoder.
//
// tools/make_png_fixtures.py writes PNGs (via Python's zlib, i.e. a real DEFLATE
// encoder we did not write) plus the expected RGB bytes. Our decoder must reproduce
// them EXACTLY -- image decoding is lossless, so anything short of byte-equality is a
// bug, not a tolerance.
//
// The fixtures deliberately cover the cases that break naive inflaters: dynamic
// Huffman blocks, stored (uncompressed) blocks, overlapping LZ77 back-references, all
// five scanline filters, and every supported colour type.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "model/png.hpp"

namespace {

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "artifacts/png_fixtures";

  // name, description
  const std::pair<const char*, const char*> cases[] = {
      {"rgb_gradient", "RGB, dynamic Huffman, gradient (exercises all filters)"},
      {"rgba_noise", "RGBA, incompressible noise (forces stored blocks)"},
      {"gray_steps", "greyscale, long runs (overlapping back-references)"},
      {"palette_flag", "palette (PLTE) indexed colour"},
      {"gray_alpha", "greyscale + alpha"},
      {"tiny_1x1", "1x1 edge case"},
  };

  int failures = 0, ran = 0;
  for (const auto& c : cases) {
    const std::string png = dir + "/" + c.first + ".png";
    const std::string raw = dir + "/" + c.first + ".rgb";
    const auto png_bytes = read_file(png);
    const auto expect = read_file(raw);
    if (png_bytes.empty() || expect.empty()) {
      std::printf("  [skip] %-14s (fixture missing -- run tools/make_png_fixtures.py)\n", c.first);
      continue;
    }
    ++ran;

    try {
      const model::image::Image img = model::image::decode_png(png_bytes);
      if (img.rgb.size() != expect.size()) {
        std::printf("  [FAIL] %-14s size %zu, expected %zu\n", c.first, img.rgb.size(),
                    expect.size());
        ++failures;
        continue;
      }
      std::size_t bad = 0;
      for (std::size_t i = 0; i < expect.size(); ++i) {
        if (img.rgb[i] != expect[i]) ++bad;
      }
      if (bad != 0) {
        std::printf("  [FAIL] %-14s %zu/%zu bytes differ  (%s)\n", c.first, bad, expect.size(),
                    c.second);
        ++failures;
      } else {
        std::printf("  [ok]   %-14s %dx%d  (%s)\n", c.first, img.width, img.height, c.second);
      }
    } catch (const std::exception& e) {
      std::printf("  [FAIL] %-14s threw: %s\n", c.first, e.what());
      ++failures;
    }
  }

  if (ran == 0) {
    std::printf("\nno fixtures found -- run: python tools/make_png_fixtures.py\n");
    return 2;
  }
  std::printf("\n%s\n", failures == 0
                            ? "PNG OK (decoder is byte-exact against the reference encoder)"
                            : "PNG FAILED");
  return failures == 0 ? 0 : 1;
}
