// Unit test for kernels::launch_dequant_mxfp4 (the K3 MXFP4 weight format).
// Validates: (1) the E2M1 value set matches the OCP spec, (2) exactly-representable values
// round-trip exact, (3) the DEVICE kernel matches a host reference decode, (4) round-trip error is
// sane. MXFP4 is not yet wired into the matvec path; this gates the decode math in isolation first.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "runtime/kernels.cuh"

namespace {

float e2m1(std::uint8_t nib) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = mag[nib & 0x7];
  return (nib & 0x8) ? -v : v;
}
std::uint8_t nearest_e2m1(float x) {
  std::uint8_t best = 0;
  float be = 1e30f;
  for (int n = 0; n < 16; ++n) {
    float e = std::fabs(x - e2m1((std::uint8_t)n));
    if (e < be) {
      be = e;
      best = (std::uint8_t)n;
    }
  }
  return best;
}
// Block-32 MXFP4 quantizer: E8M0 scale exp = floor(log2(amax)) - 2, then nearest E2M1 per element.
void quantize_mxfp4(const float* W, int rows, int cols, std::vector<std::uint8_t>& packed,
                    std::vector<std::uint8_t>& scales) {
  const int nblk = (cols + 31) / 32;
  packed.assign((size_t)rows * ((cols + 1) / 2), 0);
  scales.assign((size_t)rows * nblk, 127);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < nblk; ++b) {
      float amax = 0;
      for (int c = b * 32; c < std::min(cols, (b + 1) * 32); ++c)
        amax = std::max(amax, std::fabs(W[(size_t)r * cols + c]));
      int se = (amax > 0) ? (int)std::floor(std::log2(amax)) - 2 : -127;
      se = std::max(-127, std::min(127, se));
      scales[(size_t)r * nblk + b] = (std::uint8_t)(se + 127);
      const float inv = exp2f(-se);
      for (int c = b * 32; c < std::min(cols, (b + 1) * 32); ++c) {
        std::uint8_t nib = nearest_e2m1(W[(size_t)r * cols + c] * inv);
        size_t pi = (size_t)r * ((cols + 1) / 2) + (c >> 1);
        packed[pi] =
            (c & 1) ? (packed[pi] & 0x0f) | (nib << 4) : (packed[pi] & 0xf0) | (nib & 0x0f);
      }
    }
  }
}
float host_decode(const std::vector<std::uint8_t>& packed, const std::vector<std::uint8_t>& scales,
                  int r, int c, int cols) {
  const int nblk = (cols + 31) / 32, pc = (cols + 1) / 2;
  std::uint8_t byte = packed[(size_t)r * pc + (c >> 1)];
  std::uint8_t nib = (c & 1) ? (byte >> 4) : (byte & 0x0f);
  return e2m1(nib) * exp2f((int)scales[(size_t)r * nblk + c / 32] - 127);
}

}  // namespace

int main() {
  int fail = 0;

  // (1) OCP value set.
  const float spec[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  for (int n = 0; n < 16; ++n)
    if (e2m1((std::uint8_t)n) != spec[n]) {
      std::printf("FAIL: E2M1[%d]=%g expected %g\n", n, e2m1((std::uint8_t)n), spec[n]);
      fail = 1;
    }

  // (2) exactly-representable values round-trip exact.
  {
    std::vector<float> ex = {6, 4, 3, 2, 1.5f, 1, 0.5f, 0, -6, -3};
    std::vector<std::uint8_t> p, s;
    quantize_mxfp4(ex.data(), 1, (int)ex.size(), p, s);
    for (int c = 0; c < (int)ex.size(); ++c)
      if (host_decode(p, s, 0, c, (int)ex.size()) != ex[c]) {
        std::printf("FAIL: representable %g did not round-trip\n", ex[c]);
        fail = 1;
      }
  }

  // (3)+(4) device kernel == host reference, on a random outlier-laden matrix.
  const int rows = 64, cols = 256;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> W((size_t)rows * cols);
  for (auto& v : W) v = nd(rng);
  for (int r = 0; r < rows; ++r) W[(size_t)r * cols + 100] *= 12.0f;
  std::vector<std::uint8_t> packed, scales;
  quantize_mxfp4(W.data(), rows, cols, packed, scales);

  std::uint8_t *dp, *ds;
  half* dout;
  cudaMalloc(&dp, packed.size());
  cudaMalloc(&ds, scales.size());
  cudaMalloc(&dout, (size_t)rows * cols * sizeof(half));
  cudaMemcpy(dp, packed.data(), packed.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(ds, scales.data(), scales.size(), cudaMemcpyHostToDevice);
  kernels::launch_dequant_mxfp4(dp, ds, dout, rows, cols, nullptr);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("FAIL: kernel launch: %s\n", cudaGetErrorString(err));
    return 1;
  }
  std::vector<half> devh((size_t)rows * cols);
  cudaMemcpy(devh.data(), dout, devh.size() * sizeof(half), cudaMemcpyDeviceToHost);

  int mism = 0;
  double num = 0, den = 0;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      float dev = __half2float(devh[(size_t)r * cols + c]);
      float ref = host_decode(packed, scales, r, c, cols);
      if (__float2half(ref) != devh[(size_t)r * cols + c]) mism++;
      num += (dev - W[(size_t)r * cols + c]) * (dev - W[(size_t)r * cols + c]);
      den += W[(size_t)r * cols + c] * (double)W[(size_t)r * cols + c];
    }
  if (mism != 0) {
    std::printf("FAIL: device != host reference (%d mismatches)\n", mism);
    fail = 1;
  }
  const double rel = 100.0 * std::sqrt(num / den);
  if (rel > 25.0) {  // sanity ceiling; typical ~14% on this outlier matrix
    std::printf("FAIL: round-trip error %.2f%% too high\n", rel);
    fail = 1;
  }

  std::printf("%s: MXFP4 dequant (device==host, LUT+representable exact, round-trip %.2f%%)\n",
              fail ? "FAIL" : "PASS", rel);
  return fail;
}
