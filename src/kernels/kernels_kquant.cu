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

// Per-thread unpack of one k-quant super-block element. Shared by the dequant
// kernels above and the matvec below, so the two cannot drift: the matvec is
// only trustworthy because this arithmetic is the arithmetic kquant_dequant_test
// gates bit-for-bit.
__device__ __forceinline__ float kquant_value(const std::uint8_t* p, KQuantType type, int i) {
  if (type == KQuantType::Q4_K) {
    std::uint16_t dh, mh;
    memcpy(&dh, p, 2);
    memcpy(&mh, p + 2, 2);
    const float d = half_bits_to_float(dh);
    const float dmin = half_bits_to_float(mh);
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* qs = p + 16;
    const int j = i / 64;
    const int rem = i % 64;
    const int l = rem % 32;
    std::uint8_t sc, m;
    get_scale_min_k4(2 * j + (rem / 32), scales, &sc, &m);
    const std::uint8_t byte = qs[j * 32 + l];
    const int q = (rem < 32) ? (byte & 0x0F) : (byte >> 4);
    return d * sc * static_cast<float>(q) - dmin * m;
  }
  if (type == KQuantType::Q5_K) {
    std::uint16_t dh, mh;
    memcpy(&dh, p, 2);
    memcpy(&mh, p + 2, 2);
    const float d = half_bits_to_float(dh);
    const float dmin = half_bits_to_float(mh);
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* qh = p + 16;
    const std::uint8_t* ql = p + 48;
    const int j = i / 64;
    const int rem = i % 64;
    const int l = rem % 32;
    std::uint8_t sc, m;
    get_scale_min_k4(2 * j + (rem / 32), scales, &sc, &m);
    const std::uint8_t byte = ql[j * 32 + l];
    const std::uint8_t hbit = static_cast<std::uint8_t>(1u << (2 * j + (rem / 32)));
    const int q = ((rem < 32) ? (byte & 0x0F) : (byte >> 4)) + ((qh[l] & hbit) ? 16 : 0);
    return d * sc * static_cast<float>(q) - dmin * m;
  }
  // Q6_K
  const std::uint8_t* ql = p;
  const std::uint8_t* qh = p + 128;
  const auto* sc = reinterpret_cast<const std::int8_t*>(p + 192);
  std::uint16_t dh;
  memcpy(&dh, p + 208, 2);
  const float d = half_bits_to_float(dh);
  const int n = i / 128;
  const int rem = i % 128;
  const int quarter = rem / 32;
  const int l = rem % 32;
  const std::uint8_t* qln = ql + n * 64;
  const std::uint8_t* qhn = qh + n * 32;
  const std::int8_t* scn = sc + n * 8;
  const int half_off = (quarter & 1) * 32;
  const int q =
      static_cast<int>((quarter >> 1) ? (qln[l + half_off] >> 4) : (qln[l + half_off] & 0x0F)) |
      (((qhn[l] >> (2 * quarter)) & 3) << 4);
  return d * static_cast<float>(scn[quarter * 2 + (l / 16)]) * static_cast<float>(q - 32);
}

// y = W x with W still packed: one row per thread block, the row's super-blocks
// unpacked on the fly and multiplied into x. This is the point of the whole
// exercise -- the weight never exists as fp16 anywhere, so a Q4_K_M model is
// resident at its file size instead of ~3x it.
// Per-type packed matvec. The generic kernel above re-derives everything for
// every weight: it branches on the runtime type, reloads d/dmin, and re-decodes
// the packed 6-bit scale nibbles once per element. Since the block is exactly
// one super-block wide, a thread handles the SAME offset inside every
// super-block down the row, so all of that index math is loop-invariant and
// hoists out. What remains in the loop is the three loads that actually differ
// per super-block: the fp16 scale pair, the thread's own scale byte(s), and its
// quantized byte.
//
// A 256-thread block reads exactly the 144/176/210 bytes of one super-block per
// iteration and nothing else, so no bytes are fetched twice within a block.
template <KQuantType TYPE>
__global__ void kquant_matvec_fast_kernel(const std::uint8_t* __restrict__ w,
                                          const __half* __restrict__ x, __half* __restrict__ y,
                                          int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int i = static_cast<int>(threadIdx.x);
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);

  std::size_t block_bytes;
  if (TYPE == KQuantType::Q4_K) {
    block_bytes = model::kquant::kQ4KBlockBytes;
  } else if (TYPE == KQuantType::Q5_K) {
    block_bytes = model::kquant::kQ5KBlockBytes;
  } else {
    block_bytes = model::kquant::kQ6KBlockBytes;
  }
  const std::uint8_t* p =
      w + static_cast<std::size_t>(row) * static_cast<std::size_t>(blocks_per_row) * block_bytes;

  // Loop-invariant offsets for this thread's slot in a super-block.
  int off_a = 0;  // quantized byte
  int off_b = 0;  // high-bit byte (Q5_K/Q6_K)
  int scale_idx = 0;
  bool hi_nibble = false;
  int shift = 0;
  if (TYPE == KQuantType::Q4_K || TYPE == KQuantType::Q5_K) {
    const int j = i / 64;
    const int rem = i % 64;
    const int l = rem % 32;
    scale_idx = 2 * j + (rem / 32);
    hi_nibble = rem >= 32;
    if (TYPE == KQuantType::Q4_K) {
      off_a = 16 + j * 32 + l;
    } else {
      off_a = 48 + j * 32 + l;
      off_b = 16 + l;
    }
  } else {
    const int n = i / 128;
    const int rem = i % 128;
    const int quarter = rem / 32;
    const int l = rem % 32;
    off_a = n * 64 + l + (quarter & 1) * 32;
    off_b = 128 + n * 32 + l;
    scale_idx = 192 + n * 8 + quarter * 2 + (l / 16);
    hi_nibble = (quarter >> 1) != 0;
    shift = 2 * quarter;
  }

  float acc = 0.0f;
  for (int b = 0; b < blocks_per_row; ++b) {
    float v;
    if (TYPE == KQuantType::Q6_K) {
      std::uint16_t dh;
      memcpy(&dh, p + 208, 2);
      const int q = static_cast<int>(hi_nibble ? (__ldg(p + off_a) >> 4) : (__ldg(p + off_a) & 0x0F)) |
                    (((__ldg(p + off_b) >> shift) & 3) << 4);
      const auto sc = static_cast<const std::int8_t>(__ldg(p + scale_idx));
      v = half_bits_to_float(dh) * static_cast<float>(sc) * static_cast<float>(q - 32);
    } else {
      std::uint16_t dh, mh;
      memcpy(&dh, p, 2);
      memcpy(&mh, p + 2, 2);
      std::uint8_t sc, m;
      get_scale_min_k4(scale_idx, p + 4, &sc, &m);
      const std::uint8_t byte = __ldg(p + off_a);
      int q = hi_nibble ? (byte >> 4) : (byte & 0x0F);
      if (TYPE == KQuantType::Q5_K) {
        q += (__ldg(p + off_b) & static_cast<std::uint8_t>(1u << scale_idx)) ? 16 : 0;
      }
      v = half_bits_to_float(dh) * static_cast<float>(sc) * static_cast<float>(q) -
          half_bits_to_float(mh) * static_cast<float>(m);
    }
    acc += v * __half2float(__ldg(x + b * static_cast<int>(kSuperBlock) + i));
    p += block_bytes;
  }

  __shared__ float warp_sums[8];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) total += warp_sums[k];
    y[row] = __float2half(total);
  }
}

__global__ void kquant_matvec_kernel(const std::uint8_t* __restrict__ w, KQuantType type,
                                     std::size_t block_bytes, const __half* __restrict__ x,
                                     __half* __restrict__ y, int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);
  const std::uint8_t* row_base =
      w + static_cast<std::size_t>(row) * blocks_per_row * block_bytes;

  float acc = 0.0f;
  for (int b = 0; b < blocks_per_row; ++b) {
    const std::uint8_t* p = row_base + static_cast<std::size_t>(b) * block_bytes;
    const int col0 = b * static_cast<int>(kSuperBlock);
    for (int i = threadIdx.x; i < static_cast<int>(kSuperBlock); i += blockDim.x) {
      acc += kquant_value(p, type, i) * __half2float(x[col0 + i]);
    }
  }

  // Block reduction: warp shuffles then one value per warp through shared memory.
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  }
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    const int warps = (blockDim.x + warpSize - 1) / warpSize;
    for (int i = 0; i < warps; ++i) total += warp_sums[i];
    y[row] = __float2half(total);
  }
}

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

void launch_kquant_matvec(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                          int cols, cudaStream_t stream) {
  std::size_t block_bytes = model::kquant::kQ6KBlockBytes;
  if (type == KQuantType::Q4_K) block_bytes = model::kquant::kQ4KBlockBytes;
  if (type == KQuantType::Q5_K) block_bytes = model::kquant::kQ5KBlockBytes;
  constexpr int kThreads = 256;  // one super-block wide, so a thread's slot is fixed
  switch (type) {
    case KQuantType::Q4_K:
      kquant_matvec_fast_kernel<KQuantType::Q4_K><<<rows, kThreads, 0, stream>>>(w, x, y, rows, cols);
      return;
    case KQuantType::Q5_K:
      kquant_matvec_fast_kernel<KQuantType::Q5_K><<<rows, kThreads, 0, stream>>>(w, x, y, rows, cols);
      return;
    case KQuantType::Q6_K:
      kquant_matvec_fast_kernel<KQuantType::Q6_K><<<rows, kThreads, 0, stream>>>(w, x, y, rows, cols);
      return;
  }
  kquant_matvec_kernel<<<rows, kThreads, 0, stream>>>(w, type, block_bytes, x, y, rows, cols);
}

}  // namespace kernels
