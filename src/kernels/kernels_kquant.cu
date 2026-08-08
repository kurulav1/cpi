// Device-side ggml k-quant dequantization.
//
// First brick of keeping GGUF's quantized weights packed on the GPU. Today the
// loader unpacks k-quants on the CPU into fp16, which costs the model's fp16
// size in host RAM and in VRAM, i.e. it discards the reason the file was
// quantized in the first place. Uploading the packed blocks and unpacking them
// on the device removes the host cost immediately, and the per-block arithmetic
// here is the same arithmetic a native quantized matvec will need inline.
//
// The layouts are ggml's and are a wire format; kquant_dequant_test gates every
// type against the host implementation in gguf_loader.cpp, which is itself
// gated against an fp16 oracle. A silent mistake here reads as slightly wrong
// weights, never as a crash, so the test is the point.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include "model/gguf_kquant.hpp"
#include "runtime/kernels.cuh"

namespace kernels {
namespace {

using model::kquant::kSuperBlock;

__device__ __forceinline__ float half_bits_to_float(std::uint16_t h) {
  __half v;
  memcpy(&v, &h, sizeof(v));
  return __half2float(v);
}

// Q4_K/Q5_K pack 8 sub-block (scale, min) pairs as 6-bit fields across 12 bytes.
__device__ __forceinline__ void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d,
                                                 std::uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<std::uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// One super-block per thread block; threads split the 256 outputs. Each type
// reads its own header, so the block pointer is derived from blockIdx rather
// than walked, which is what makes the whole thing parallel.
__global__ void dequant_q4_k_kernel(const std::uint8_t* __restrict__ base, __half* __restrict__ out,
                                    std::size_t blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= blocks) return;
  const std::uint8_t* p = base + b * model::kquant::kQ4KBlockBytes;
  std::uint16_t dh, mh;
  memcpy(&dh, p, 2);
  memcpy(&mh, p + 2, 2);
  const float d = half_bits_to_float(dh);
  const float dmin = half_bits_to_float(mh);
  const std::uint8_t* scales = p + 4;
  const std::uint8_t* qs = p + 16;
  __half* y = out + b * kSuperBlock;

  for (int i = threadIdx.x; i < static_cast<int>(kSuperBlock); i += blockDim.x) {
    const int j = i / 64;            // which 64-wide half-group
    const int rem = i % 64;          // position inside it
    const int is = 2 * j + (rem / 32);  // sub-block index for the scale pair
    const int l = rem % 32;
    std::uint8_t sc, m;
    get_scale_min_k4(is, scales, &sc, &m);
    const std::uint8_t byte = qs[j * 32 + l];
    const int q = (rem < 32) ? (byte & 0x0F) : (byte >> 4);
    y[i] = __float2half(d * sc * static_cast<float>(q) - dmin * m);
  }
}

__global__ void dequant_q5_k_kernel(const std::uint8_t* __restrict__ base, __half* __restrict__ out,
                                    std::size_t blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= blocks) return;
  const std::uint8_t* p = base + b * model::kquant::kQ5KBlockBytes;
  std::uint16_t dh, mh;
  memcpy(&dh, p, 2);
  memcpy(&mh, p + 2, 2);
  const float d = half_bits_to_float(dh);
  const float dmin = half_bits_to_float(mh);
  const std::uint8_t* scales = p + 4;
  const std::uint8_t* qh = p + 16;
  const std::uint8_t* ql = p + 48;
  __half* y = out + b * kSuperBlock;

  for (int i = threadIdx.x; i < static_cast<int>(kSuperBlock); i += blockDim.x) {
    const int j = i / 64;
    const int rem = i % 64;
    const int is = 2 * j + (rem / 32);
    const int l = rem % 32;
    std::uint8_t sc, m;
    get_scale_min_k4(is, scales, &sc, &m);
    const std::uint8_t byte = ql[j * 32 + l];
    // The high bit for this position lives in qh, two bits per 64-wide group.
    const std::uint8_t hbit = static_cast<std::uint8_t>(1u << (2 * j + (rem / 32)));
    const int base_q = (rem < 32) ? (byte & 0x0F) : (byte >> 4);
    const int q = base_q + ((qh[l] & hbit) ? 16 : 0);
    y[i] = __float2half(d * sc * static_cast<float>(q) - dmin * m);
  }
}

__global__ void dequant_q6_k_kernel(const std::uint8_t* __restrict__ base, __half* __restrict__ out,
                                    std::size_t blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= blocks) return;
  const std::uint8_t* p = base + b * model::kquant::kQ6KBlockBytes;
  const std::uint8_t* ql = p;
  const std::uint8_t* qh = p + 128;
  const auto* sc = reinterpret_cast<const std::int8_t*>(p + 192);
  std::uint16_t dh;
  memcpy(&dh, p + 208, 2);
  const float d = half_bits_to_float(dh);
  __half* y = out + b * kSuperBlock;

  for (int i = threadIdx.x; i < static_cast<int>(kSuperBlock); i += blockDim.x) {
    const int n = i / 128;      // which 128-wide half
    const int rem = i % 128;    // position within it
    const int quarter = rem / 32;
    const int l = rem % 32;
    const std::uint8_t* qln = ql + n * 64;
    const std::uint8_t* qhn = qh + n * 32;
    const std::int8_t* scn = sc + n * 8;
    // ggml's order within a 128-group: quarters 0,1 take the low nibbles of
    // ql[l] and ql[l+32]; quarters 2,3 take their high nibbles. So the nibble
    // half comes from the high bit of `quarter` and the ql offset from its low
    // bit -- swapping the two is undetectable until index 32.
    const int lo_hi = quarter >> 1;      // 0: low nibble, 1: high nibble
    const int half_off = (quarter & 1) * 32;
    const int q = static_cast<int>(lo_hi ? (qln[l + half_off] >> 4) : (qln[l + half_off] & 0x0F)) |
                  (((qhn[l] >> (2 * quarter)) & 3) << 4);
    y[i] = __float2half(d * static_cast<float>(scn[quarter * 2 + (l / 16)]) *
                        static_cast<float>(q - 32));
  }
}

}  // namespace

void launch_dequant_kquant(const std::uint8_t* blocks_in, KQuantType type, std::size_t blocks,
                           __half* out, cudaStream_t stream) {
  constexpr int kThreads = 128;
  const dim3 grid(static_cast<unsigned>(blocks));
  switch (type) {
    case KQuantType::Q4_K:
      dequant_q4_k_kernel<<<grid, kThreads, 0, stream>>>(blocks_in, out, blocks);
      break;
    case KQuantType::Q5_K:
      dequant_q5_k_kernel<<<grid, kThreads, 0, stream>>>(blocks_in, out, blocks);
      break;
    case KQuantType::Q6_K:
      dequant_q6_k_kernel<<<grid, kThreads, 0, stream>>>(blocks_in, out, blocks);
      break;
  }
}

}  // namespace kernels
