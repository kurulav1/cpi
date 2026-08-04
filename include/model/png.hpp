// Hand-rolled PNG decoder.
//
// No libpng, no zlib; the DEFLATE decompressor (RFC 1951), the PNG filters
// (RFC 2083) and the chunk parser are all here. That is the point: CPI does not take
// third-party dependencies, and an image decoder is not an exception.
//
// Supported: 8-bit non-interlaced PNG, colour types 0 (grey), 2 (RGB), 3 (palette),
// 4 (grey+alpha), 6 (RGBA). Everything decodes to 8-bit RGB. 16-bit and Adam7
// interlacing throw rather than silently mangling the image.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace model {
namespace image {

struct Image {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;  // [height * width * 3], row-major
};

// Throws std::runtime_error on a malformed or unsupported file.
Image decode_png(const std::vector<std::uint8_t>& bytes);
Image load_png(const std::string& path);

// Raw DEFLATE (RFC 1951) and zlib (RFC 1950) streams. Exposed because they are
// generally useful, and independently testable.
std::vector<std::uint8_t> inflate(const std::uint8_t* data, std::size_t size);
std::vector<std::uint8_t> zlib_inflate(const std::uint8_t* data, std::size_t size);

}  // namespace image
}  // namespace model
