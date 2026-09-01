#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// SHA-256, hand-rolled like everything else here.
//
// The token-stream hash is FNV-1a, which is fine for "did these two runs agree"
// inside one machine. This is for the other job: the input side of a
// reproducibility claim, where the digest is pasted into an issue or a paper and
// someone else is expected to produce the same one from the same files. That
// audience knows what a SHA-256 is and cannot check an ad-hoc 64-bit hash against
// anything, so the recognisable standard is worth the hundred lines.
//
// FIPS 180-4. Verified against the standard vectors in sha256_test.
namespace cpi::util {

class Sha256 {
public:
  Sha256() = default;

  void update(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_ += len;
    while (len > 0) {
      // Not std::min: <windows.h> defines min as a macro, and the call expands into
      // a syntax error wherever this header is included after it.
      const std::size_t room = sizeof(buf_) - buf_len_;
      const std::size_t take = (len < room) ? len : room;
      std::memcpy(buf_ + buf_len_, p, take);
      buf_len_ += take;
      p += take;
      len -= take;
      if (buf_len_ == sizeof(buf_)) {
        compress(buf_);
        buf_len_ = 0;
      }
    }
  }

  void update(const std::string& s) {
    update(s.data(), s.size());
  }

  // Returns the digest as lowercase hex. Finalising twice would be a bug, so the
  // object is single-use by convention.
  [[nodiscard]] std::string hex() {
    const std::uint64_t bits = total_ * 8;
    // Padding: a 0x80 byte, zeros, then the length as a big-endian 64-bit count.
    std::uint8_t pad[128] = {0x80};
    const std::size_t pad_len = (buf_len_ < 56) ? (56 - buf_len_) : (120 - buf_len_);
    update(pad, pad_len);
    std::uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) {
      len_be[i] = static_cast<std::uint8_t>((bits >> (56 - 8 * i)) & 0xFF);
    }
    // total_ must not count the length block itself, so write it directly.
    std::memcpy(buf_ + buf_len_, len_be, 8);
    compress(buf_);

    std::string out;
    out.reserve(64);
    char tmp[3];
    for (std::uint32_t w : h_) {
      for (int i = 3; i >= 0; --i) {
        std::snprintf(tmp, sizeof(tmp), "%02x", static_cast<unsigned>((w >> (8 * i)) & 0xFF));
        out += tmp;
      }
    }
    return out;
  }

private:
  static std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  void compress(const std::uint8_t block[64]) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};

    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) |
             (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    std::uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = hh + S1 + ch + k[i] + w[i];
      const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += hh;
  }

  std::array<std::uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::uint8_t buf_[64]{};
  std::size_t buf_len_ = 0;
  std::uint64_t total_ = 0;
};

// Convenience for the common cases.
[[nodiscard]] inline std::string sha256_hex(const std::string& s) {
  Sha256 h;
  h.update(s);
  return h.hex();
}

}  // namespace cpi::util
