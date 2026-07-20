#pragma once

// IEEE 754 binary16 <-> binary32, in one place.
//
// This exists because there were FIVE hand-rolled copies of these two functions (plan_metal_
// engine.cpp, plan_metal_vision.cpp, metal_vision_test.cpp, metal_smoke.cpp, metal_gemm_bench.cpp)
// and every one of them was wrong in the same two ways:
//
//   * f32 -> f16 TRUNCATED the mantissa (`man >> 13`) instead of rounding to nearest even, so
//     every conversion of a value that is not exactly representable was biased toward zero.
//   * both directions FLUSHED fp16 subnormals to zero (`if (exp <= 0) return sign`), discarding
//     the entire range below 2^-14 that fp16 in fact represents down to 2^-24.
//
// Metal's own `half()` does neither, so host and device disagreed on 0.39% of a bf16 checkpoint's
// weights and on ~14% of arbitrary f32 data (measured: input pixels took a mean signed error of
// -1.5e-4, against +5.8e-7 for correct rounding).
//
// A fixture for exactly this was sitting unused at ~/models/f16_vals.txt + f16_expect.txt; the
// old converter failed it on 16296 of 23009 values. metal_fp16_test now gates this header, and
// will read that fixture if handed it.
//
// Deliberately ONE portable implementation rather than an #ifdef split onto __fp16: a second code
// path is a second thing to verify, and these run once per weight upload where the cost is noise
// next to the memcpy they feed.

#include <cstdint>
#include <cstring>

namespace cpi {

// Round-to-nearest-even, with subnormals, and overflow to infinity. Matches numpy's
// astype(float16), IEEE 754, and Metal's half().
inline std::uint16_t f32_to_f16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::uint32_t mag = x & 0x7FFFFFFFu;  // |f|, as bits

  // NaN stays NaN (with a non-zero payload so it cannot decay to infinity); Inf stays Inf.
  if (mag >= 0x7F800000u) {
    return static_cast<std::uint16_t>(sign | 0x7C00u | (mag > 0x7F800000u ? 0x0200u : 0x0000u));
  }
  // 0x477FF000 is 65520.0f, the midpoint between fp16's largest finite value and 2^16. Anything
  // at or above it rounds to infinity -- NOT to 65504, which is what a truncating converter
  // returns and is the one case where truncation looks "safer" while being non-conforming.
  if (mag >= 0x477FF000u) return static_cast<std::uint16_t>(sign | 0x7C00u);

  // Normal fp16 range: |f| >= 2^-14 (0x38800000).
  if (mag >= 0x38800000u) {
    const std::uint32_t man = mag & 0x007FFFFFu;
    std::uint32_t h = (((mag >> 23) - 127u + 15u) << 10) | (man >> 13);
    const std::uint32_t rem = man & 0x1FFFu;  // the 13 bits being dropped
    // Halfway ties go to even. A carry out of the mantissa lands in the exponent field, which is
    // exactly right: 0x03FF + 1 == 0x0400 is the next binade with mantissa 0.
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) ++h;
    return static_cast<std::uint16_t>(sign | h);
  }

  // Subnormal fp16: the value is (integer mantissa) * 2^-24, so the result is round(|f| * 2^24).
  const std::int32_t e = static_cast<std::int32_t>(mag >> 23) - 127;
  const std::uint32_t man = (mag & 0x007FFFFFu) | 0x00800000u;  // restore the implicit 1
  const int shift = -e - 1;                                     // 14..24 over the subnormal range
  if (shift > 24) return static_cast<std::uint16_t>(sign);      // < 2^-25 (or the tie at it) -> 0
  std::uint32_t h = man >> shift;
  const std::uint32_t rem = man & ((1u << shift) - 1u);
  const std::uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (h & 1u) != 0u)) ++h;
  // A carry here promotes the largest subnormal to the smallest normal, which is correct.
  return static_cast<std::uint16_t>(sign | h);
}

inline float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t man = h & 0x03FFu;

  if (exp == 0u) {
    if (man == 0u) {  // signed zero
      float z;
      std::memcpy(&z, &sign, sizeof(z));
      return z;
    }
    // Subnormal: man * 2^-24, exactly. 2^-24 is a power of two and man <= 1023, so the product
    // is exact in f32 and needs no bit surgery.
    const float v = static_cast<float>(man) * 5.9604644775390625e-08f;
    return (sign != 0u) ? -v : v;
  }
  std::uint32_t out;
  if (exp == 31u) {
    out = sign | 0x7F800000u | (man << 13);  // Inf, or NaN with its payload preserved
  } else {
    out = sign | ((exp - 15u + 127u) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

}  // namespace cpi
