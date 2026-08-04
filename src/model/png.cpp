// Hand-rolled PNG decoder: DEFLATE (RFC 1951) + PNG filters (RFC 2083).
//
// The DEFLATE side is a straight canonical-Huffman inflater. The only subtlety worth
// calling out is that DEFLATE packs bits LSB-first within a byte, but Huffman CODES are
// read MSB-first; a bit reader that gets that backwards decodes the fixed-Huffman
// blocks fine and then falls apart on dynamic ones, which is a miserable bug to chase.
// Here, read_bits() is LSB-first and decode_symbol() walks the code bit by bit,
// accumulating in code-order.

#include "model/png.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace model {
namespace image {
namespace {

class BitReader {
 public:
  BitReader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  // DEFLATE stores bits LSB-first within each byte.
  std::uint32_t read_bits(int count) {
    std::uint32_t value = 0;
    for (int i = 0; i < count; ++i) {
      value |= static_cast<std::uint32_t>(read_bit()) << i;
    }
    return value;
  }

  int read_bit() {
    if (bit_pos_ == 0 && byte_pos_ >= size_) {
      throw std::runtime_error("png: DEFLATE stream truncated");
    }
    const int bit = (data_[byte_pos_] >> bit_pos_) & 1;
    if (++bit_pos_ == 8) {
      bit_pos_ = 0;
      ++byte_pos_;
    }
    return bit;
  }

  void align_to_byte() {
    if (bit_pos_ != 0) {
      bit_pos_ = 0;
      ++byte_pos_;
    }
  }

  const std::uint8_t* raw(std::size_t n) {
    if (byte_pos_ + n > size_) throw std::runtime_error("png: DEFLATE stored block truncated");
    const std::uint8_t* p = data_ + byte_pos_;
    byte_pos_ += n;
    return p;
  }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t byte_pos_ = 0;
  int bit_pos_ = 0;
};

// Canonical Huffman, built from code lengths (RFC 1951 §3.2.2). Decoding walks the code
// one bit at a time, comparing against the first code of each length; simple, and fast
// enough that image decode is nowhere near the bottleneck.
class Huffman {
 public:
  explicit Huffman(const std::vector<int>& lengths) {
    int max_len = 0;
    for (int l : lengths) max_len = std::max(max_len, l);
    if (max_len == 0) return;  // empty table (legal: e.g. no distance codes)

    counts_.assign(static_cast<std::size_t>(max_len) + 1, 0);
    for (int l : lengths) {
      if (l > 0) ++counts_[static_cast<std::size_t>(l)];
    }

    // symbols sorted by (length, symbol)
    std::vector<int> offsets(static_cast<std::size_t>(max_len) + 2, 0);
    for (int l = 1; l <= max_len; ++l) {
      offsets[static_cast<std::size_t>(l) + 1] = offsets[static_cast<std::size_t>(l)] +
                                                 counts_[static_cast<std::size_t>(l)];
    }
    symbols_.assign(static_cast<std::size_t>(offsets[static_cast<std::size_t>(max_len) + 1]), 0);
    for (std::size_t s = 0; s < lengths.size(); ++s) {
      if (lengths[s] > 0) {
        symbols_[static_cast<std::size_t>(offsets[static_cast<std::size_t>(lengths[s])]++)] =
            static_cast<int>(s);
      }
    }
    max_len_ = max_len;
  }

  int decode(BitReader& br) const {
    if (max_len_ == 0) throw std::runtime_error("png: decode from an empty Huffman table");
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= max_len_; ++len) {
      code |= br.read_bit();  // Huffman codes are MSB-first, so accumulate upward
      const int count = counts_[static_cast<std::size_t>(len)];
      if (code - first < count) {
        return symbols_[static_cast<std::size_t>(index + (code - first))];
      }
      index += count;
      first = (first + count) << 1;
      code <<= 1;
    }
    throw std::runtime_error("png: invalid Huffman code");
  }

 private:
  std::vector<int> counts_;
  std::vector<int> symbols_;
  int max_len_ = 0;
};

// RFC 1951 §3.2.5
constexpr int kLengthBase[29] = {3,  4,  5,  6,  7,  8,  9,  10,  11,  13,  15,  17,  19,  23, 27,
                                 31, 35, 43, 51, 59, 67, 83, 99,  115, 131, 163, 195, 227, 258};
constexpr int kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                  2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr int kDistBase[30] = {1,    2,    3,    4,    5,    7,     9,     13,    17,   25,
                               33,   49,   65,   97,   129,  193,   257,   385,   513,  769,
                               1025, 1537, 2049, 3073, 4097, 6145,  8193,  12289, 16385, 24577};
constexpr int kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

void inflate_block(BitReader& br, const Huffman& lit, const Huffman& dist,
                   std::vector<std::uint8_t>& out) {
  for (;;) {
    const int sym = lit.decode(br);
    if (sym < 256) {
      out.push_back(static_cast<std::uint8_t>(sym));
    } else if (sym == 256) {
      return;  // end of block
    } else {
      const int li = sym - 257;
      if (li >= 29) throw std::runtime_error("png: bad length symbol");
      const int length =
          kLengthBase[li] + static_cast<int>(br.read_bits(kLengthExtra[li]));
      const int di = dist.decode(br);
      if (di >= 30) throw std::runtime_error("png: bad distance symbol");
      const int distance = kDistBase[di] + static_cast<int>(br.read_bits(kDistExtra[di]));
      if (static_cast<std::size_t>(distance) > out.size())
        throw std::runtime_error("png: back-reference before the start of the stream");
      // Byte-by-byte on purpose: overlapping copies (distance < length) are legal and
      // must repeat what was just written. memcpy would be wrong here.
      const std::size_t start = out.size() - static_cast<std::size_t>(distance);
      for (int i = 0; i < length; ++i) out.push_back(out[start + static_cast<std::size_t>(i)]);
    }
  }
}

Huffman fixed_literal_table() {
  std::vector<int> lengths(288);
  for (int i = 0; i < 144; ++i) lengths[static_cast<std::size_t>(i)] = 8;
  for (int i = 144; i < 256; ++i) lengths[static_cast<std::size_t>(i)] = 9;
  for (int i = 256; i < 280; ++i) lengths[static_cast<std::size_t>(i)] = 7;
  for (int i = 280; i < 288; ++i) lengths[static_cast<std::size_t>(i)] = 8;
  return Huffman(lengths);
}

// Dynamic block: the code lengths are themselves Huffman-coded (RFC 1951 §3.2.7).
void read_dynamic_tables(BitReader& br, Huffman& lit, Huffman& dist) {
  const int hlit = static_cast<int>(br.read_bits(5)) + 257;
  const int hdist = static_cast<int>(br.read_bits(5)) + 1;
  const int hclen = static_cast<int>(br.read_bits(4)) + 4;

  static constexpr int kOrder[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                     11, 4,  12, 3, 13, 2, 14, 1, 15};
  std::vector<int> cl_lengths(19, 0);
  for (int i = 0; i < hclen; ++i) {
    cl_lengths[static_cast<std::size_t>(kOrder[i])] = static_cast<int>(br.read_bits(3));
  }
  const Huffman cl(cl_lengths);

  std::vector<int> lengths;
  lengths.reserve(static_cast<std::size_t>(hlit + hdist));
  while (static_cast<int>(lengths.size()) < hlit + hdist) {
    const int sym = cl.decode(br);
    if (sym < 16) {
      lengths.push_back(sym);
    } else if (sym == 16) {  // repeat the previous length 3-6 times
      if (lengths.empty()) throw std::runtime_error("png: repeat with no previous length");
      const int n = 3 + static_cast<int>(br.read_bits(2));
      const int prev = lengths.back();
      for (int i = 0; i < n; ++i) lengths.push_back(prev);
    } else if (sym == 17) {  // 3-10 zeros
      const int n = 3 + static_cast<int>(br.read_bits(3));
      for (int i = 0; i < n; ++i) lengths.push_back(0);
    } else {  // 18: 11-138 zeros
      const int n = 11 + static_cast<int>(br.read_bits(7));
      for (int i = 0; i < n; ++i) lengths.push_back(0);
    }
  }
  if (static_cast<int>(lengths.size()) != hlit + hdist)
    throw std::runtime_error("png: code-length sequence overran its table");

  lit = Huffman(std::vector<int>(lengths.begin(), lengths.begin() + hlit));
  dist = Huffman(std::vector<int>(lengths.begin() + hlit, lengths.end()));
}

int paeth(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

std::uint32_t be32(const std::uint8_t* p) {
  return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
         (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

}  // namespace

std::vector<std::uint8_t> inflate(const std::uint8_t* data, std::size_t size) {
  BitReader br(data, size);
  std::vector<std::uint8_t> out;
  for (;;) {
    const int final_block = br.read_bit();
    const int type = static_cast<int>(br.read_bits(2));
    if (type == 0) {  // stored
      br.align_to_byte();
      const std::uint8_t* hdr = br.raw(4);
      const std::size_t len = static_cast<std::size_t>(hdr[0]) | (static_cast<std::size_t>(hdr[1]) << 8);
      const std::uint8_t* payload = br.raw(len);
      out.insert(out.end(), payload, payload + len);
    } else if (type == 1) {
      static const Huffman lit = fixed_literal_table();
      static const Huffman dist(std::vector<int>(30, 5));
      inflate_block(br, lit, dist, out);
    } else if (type == 2) {
      Huffman lit((std::vector<int>())), dist((std::vector<int>()));
      read_dynamic_tables(br, lit, dist);
      inflate_block(br, lit, dist, out);
    } else {
      throw std::runtime_error("png: reserved DEFLATE block type");
    }
    if (final_block) break;
  }
  return out;
}

std::vector<std::uint8_t> zlib_inflate(const std::uint8_t* data, std::size_t size) {
  if (size < 2) throw std::runtime_error("png: zlib stream too short");
  // byte 0 = CMF (method in the low nibble), byte 1 = FLG (FDICT is bit 5 of FLG).
  const int cm = data[0] & 0x0f;
  if (cm != 8) throw std::runtime_error("png: unsupported zlib compression method");
  if ((data[1] & 0x20) != 0) throw std::runtime_error("png: zlib preset dictionary unsupported");
  return inflate(data + 2, size - 2);  // trailing adler32 is simply not checked
}

Image decode_png(const std::vector<std::uint8_t>& bytes) {
  static constexpr std::uint8_t kMagic[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  if (bytes.size() < 8 || std::memcmp(bytes.data(), kMagic, 8) != 0)
    throw std::runtime_error("png: not a PNG file");

  int width = 0, height = 0, bit_depth = 0, color_type = 0, interlace = 0;
  std::vector<std::uint8_t> idat;
  std::vector<std::uint8_t> palette;
  bool seen_ihdr = false;

  std::size_t pos = 8;
  while (pos + 8 <= bytes.size()) {
    const std::uint32_t len = be32(&bytes[pos]);
    const char* type = reinterpret_cast<const char*>(&bytes[pos + 4]);
    const std::size_t data_pos = pos + 8;
    if (data_pos + len + 4 > bytes.size()) throw std::runtime_error("png: chunk overruns the file");

    if (std::memcmp(type, "IHDR", 4) == 0) {
      if (len < 13) throw std::runtime_error("png: short IHDR");
      width = static_cast<int>(be32(&bytes[data_pos]));
      height = static_cast<int>(be32(&bytes[data_pos + 4]));
      bit_depth = bytes[data_pos + 8];
      color_type = bytes[data_pos + 9];
      interlace = bytes[data_pos + 12];
      seen_ihdr = true;
    } else if (std::memcmp(type, "PLTE", 4) == 0) {
      palette.assign(&bytes[data_pos], &bytes[data_pos] + len);
    } else if (std::memcmp(type, "IDAT", 4) == 0) {
      idat.insert(idat.end(), &bytes[data_pos], &bytes[data_pos] + len);
    } else if (std::memcmp(type, "IEND", 4) == 0) {
      break;
    }
    pos = data_pos + len + 4;  // + CRC
  }

  if (!seen_ihdr) throw std::runtime_error("png: missing IHDR");
  if (width <= 0 || height <= 0) throw std::runtime_error("png: bad dimensions");
  if (bit_depth != 8)
    throw std::runtime_error("png: only 8-bit images are supported (got bit depth " +
                             std::to_string(bit_depth) + ")");
  if (interlace != 0) throw std::runtime_error("png: interlaced (Adam7) images are unsupported");

  int channels = 0;
  switch (color_type) {
    case 0: channels = 1; break;  // grey
    case 2: channels = 3; break;  // RGB
    case 3: channels = 1; break;  // palette index
    case 4: channels = 2; break;  // grey + alpha
    case 6: channels = 4; break;  // RGBA
    default: throw std::runtime_error("png: unsupported colour type");
  }
  if (color_type == 3 && palette.size() < 3)
    throw std::runtime_error("png: palette image with no PLTE");

  const std::vector<std::uint8_t> raw = zlib_inflate(idat.data(), idat.size());
  const std::size_t stride = static_cast<std::size_t>(width) * channels;
  if (raw.size() < (stride + 1) * static_cast<std::size_t>(height))
    throw std::runtime_error("png: decompressed data is shorter than the image");

  // Undo the per-scanline filters (RFC 2083 §6). Each row is prefixed with its filter
  // type and predicts from the pixel to the left (a), above (b) and above-left (c).
  std::vector<std::uint8_t> lines(static_cast<std::size_t>(height) * stride);
  for (int y = 0; y < height; ++y) {
    const std::size_t src = static_cast<std::size_t>(y) * (stride + 1);
    const int filter = raw[src];
    const std::uint8_t* in = &raw[src + 1];
    std::uint8_t* cur = &lines[static_cast<std::size_t>(y) * stride];
    const std::uint8_t* up = y > 0 ? &lines[static_cast<std::size_t>(y - 1) * stride] : nullptr;

    for (std::size_t i = 0; i < stride; ++i) {
      const int a = i >= static_cast<std::size_t>(channels)
                        ? cur[i - static_cast<std::size_t>(channels)]
                        : 0;
      const int b = up ? up[i] : 0;
      const int c = (up && i >= static_cast<std::size_t>(channels))
                        ? up[i - static_cast<std::size_t>(channels)]
                        : 0;
      int value = in[i];
      switch (filter) {
        case 0: break;                                   // None
        case 1: value += a; break;                       // Sub
        case 2: value += b; break;                       // Up
        case 3: value += (a + b) / 2; break;             // Average
        case 4: value += paeth(a, b, c); break;          // Paeth
        default: throw std::runtime_error("png: unknown scanline filter");
      }
      cur[i] = static_cast<std::uint8_t>(value & 0xff);
    }
  }

  // Everything lands as 8-bit RGB; alpha is dropped (the vision tower takes RGB).
  Image img;
  img.width = width;
  img.height = height;
  img.rgb.resize(static_cast<std::size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::uint8_t* px = &lines[static_cast<std::size_t>(y) * stride +
                                      static_cast<std::size_t>(x) * channels];
      std::uint8_t* out = &img.rgb[(static_cast<std::size_t>(y) * width + x) * 3];
      switch (color_type) {
        case 0:
        case 4:
          out[0] = out[1] = out[2] = px[0];
          break;
        case 2:
        case 6:
          out[0] = px[0];
          out[1] = px[1];
          out[2] = px[2];
          break;
        case 3: {
          const std::size_t idx = static_cast<std::size_t>(px[0]) * 3;
          if (idx + 2 >= palette.size()) throw std::runtime_error("png: palette index out of range");
          out[0] = palette[idx];
          out[1] = palette[idx + 1];
          out[2] = palette[idx + 2];
          break;
        }
        default:
          break;
      }
    }
  }
  return img;
}

Image load_png(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("png: cannot open " + path);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
  return decode_png(bytes);
}

}  // namespace image
}  // namespace model
