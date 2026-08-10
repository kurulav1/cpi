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

__device__ __forceinline__ std::uint32_t byte_at(std::uint32_t w, int i) {
  return (w >> (8 * i)) & 0xFFu;
}

__device__ __forceinline__ void scale_min_from_hdr(int idx, std::uint32_t y, std::uint32_t z,
                                                   std::uint32_t w32, float* sc, float* m) {
  if (idx < 4) {
    *sc = static_cast<float>(byte_at(y, idx) & 63u);
    *m = static_cast<float>(byte_at(z, idx) & 63u);
  } else {
    const std::uint32_t a = byte_at(w32, idx - 4);  // q[idx + 4]
    const std::uint32_t b = byte_at(y, idx - 4);    // q[idx - 4]
    const std::uint32_t c = byte_at(z, idx - 4);    // q[idx]
    *sc = static_cast<float>((a & 0x0Fu) | ((b >> 6) << 4));
    *m = static_cast<float>((a >> 4) | ((c >> 6) << 4));
  }
}

// One warp per super-block, each lane taking a 4-byte slice of the quantized
// payload -- 8 values, written as two 8-byte stores.
//
// This replaced a per-element version that re-decoded the 6-bit scale fields and
// reloaded d/dmin for every weight, the same cost the matvec used to pay. It
// matters beyond load time: prefill expands whole weights through this path, and
// that showed up as a fixed ~80 ms per prefill against a ~14 ms bandwidth
// budget.
//
// Arithmetic per element is unchanged, so the result stays bit-identical to the
// host reference that kquant_dequant_test gates against.
// This lane's eight weights from one super-block, as values, plus where they sit
// inside the block. Shared by the dequant kernel and the batched matmul so the
// two cannot drift -- the matmul is trustworthy because this is the arithmetic
// kquant_dequant_test gates bit-for-bit against the host.
template <KQuantType TYPE>
__device__ __forceinline__ void kq_block_values(const std::uint8_t* __restrict__ p, int lane,
                                                float* lo, float* hi, int* out_off_lo,
                                                int* out_off_hi) {
  int off_lo;
  int off_hi;

  if (TYPE == KQuantType::Q6_K) {
    const int n = lane >> 4;
    const int rest = (lane << 2) & 63;
    const int h = rest >> 5;
    const int l0 = rest & 31;
    const std::uint8_t* qlp = p + (lane << 2);
    const std::uint32_t qw =
        static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qlp)) |
        (static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qlp + 2)) << 16);
    const std::uint8_t* qhp = p + 128 + (n << 5) + l0;
    const std::uint32_t hw =
        static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qhp)) |
        (static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qhp + 2)) << 16);
    const int sbase = 192 + (n << 3) + (l0 >> 4);
    const float scl =
        static_cast<float>(*reinterpret_cast<const std::int8_t*>(p + sbase + (h << 1)));
    const float sch =
        static_cast<float>(*reinterpret_cast<const std::int8_t*>(p + sbase + ((2 + h) << 1)));
    std::uint16_t dbits;
    memcpy(&dbits, p + 208, 2);
    const float d = half_bits_to_float(dbits);
    const int sl = 2 * h;
    const int sh = 4 + 2 * h;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const std::uint32_t byte = (qw >> (8 * k)) & 0xFFu;
      const std::uint32_t hb = (hw >> (8 * k)) & 0xFFu;
      const int qlo = static_cast<int>((byte & 0x0Fu) | (((hb >> sl) & 3u) << 4)) - 32;
      const int qhi = static_cast<int>((byte >> 4) | (((hb >> sh) & 3u) << 4)) - 32;
      lo[k] = d * scl * static_cast<float>(qlo);
      hi[k] = d * sch * static_cast<float>(qhi);
    }
    off_lo = (n << 7) + (h << 5) + l0;
    off_hi = off_lo + 64;
  } else {
    const uint4 hdr = *reinterpret_cast<const uint4*>(p);
    const float d = half_bits_to_float(static_cast<std::uint16_t>(hdr.x & 0xFFFFu));
    const float dmin = half_bits_to_float(static_cast<std::uint16_t>(hdr.x >> 16));
    const int j = lane >> 3;
    const int t = lane & 7;
    const int qoff = (TYPE == KQuantType::Q4_K) ? 16 : 48;
    const std::uint32_t qw = *reinterpret_cast<const std::uint32_t*>(p + qoff + (lane << 2));
    std::uint32_t hw = 0;
    if (TYPE == KQuantType::Q5_K) {
      hw = *reinterpret_cast<const std::uint32_t*>(p + 16 + (t << 2));
    }
    float sc0, m0, sc1, m1;
    scale_min_from_hdr(2 * j, hdr.y, hdr.z, hdr.w, &sc0, &m0);
    scale_min_from_hdr(2 * j + 1, hdr.y, hdr.z, hdr.w, &sc1, &m1);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const std::uint32_t byte = (qw >> (8 * k)) & 0xFFu;
      std::uint32_t qlo = byte & 0x0Fu;
      std::uint32_t qhi = byte >> 4;
      if (TYPE == KQuantType::Q5_K) {
        const std::uint32_t hb = (hw >> (8 * k)) & 0xFFu;
        qlo |= ((hb >> (2 * j)) & 1u) << 4;
        qhi |= ((hb >> (2 * j + 1)) & 1u) << 4;
      }
      lo[k] = d * sc0 * static_cast<float>(qlo) - dmin * m0;
      hi[k] = d * sc1 * static_cast<float>(qhi) - dmin * m1;
    }
    off_lo = (j << 6) + (t << 2);
    off_hi = off_lo + 32;
  }
  *out_off_lo = off_lo;
  *out_off_hi = off_hi;
}

template <KQuantType TYPE>
__global__ void dequant_kquant_vec_kernel(const std::uint8_t* __restrict__ base,
                                          __half* __restrict__ out, std::size_t blocks) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const std::size_t b = static_cast<std::size_t>(blockIdx.x) * 8 + warp;
  if (b >= blocks) return;

  std::size_t block_bytes;
  if (TYPE == KQuantType::Q4_K) {
    block_bytes = model::kquant::kQ4KBlockBytes;
  } else if (TYPE == KQuantType::Q5_K) {
    block_bytes = model::kquant::kQ5KBlockBytes;
  } else {
    block_bytes = model::kquant::kQ6KBlockBytes;
  }
  __half* y = out + b * kSuperBlock;

  float lo[4];
  float hi[4];
  int off_lo;
  int off_hi;
  kq_block_values<TYPE>(base + b * block_bytes, lane, lo, hi, &off_lo, &off_hi);

  // Four halves per store; the offsets are always a multiple of four.
  const __half2 lo0 = __floats2half2_rn(lo[0], lo[1]);
  const __half2 lo1 = __floats2half2_rn(lo[2], lo[3]);
  const __half2 hi0 = __floats2half2_rn(hi[0], hi[1]);
  const __half2 hi1 = __floats2half2_rn(hi[2], hi[3]);
  uint2 vlo;
  uint2 vhi;
  memcpy(&vlo.x, &lo0, 4);
  memcpy(&vlo.y, &lo1, 4);
  memcpy(&vhi.x, &hi0, 4);
  memcpy(&vhi.y, &hi1, 4);
  *reinterpret_cast<uint2*>(y + off_lo) = vlo;
  *reinterpret_cast<uint2*>(y + off_hi) = vhi;
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
// Decodes one (scale, min) pair straight out of the header registers. The 12
// scale bytes arrive as part of a single 16-byte load, so indexing them through
// a local array would spill to local memory for a runtime index; shifting the
// registers keeps it in registers.
__device__ __forceinline__ float2 load_x4(const __half* p, float2* second) {
  // Four contiguous halves as one 8-byte load. Offsets into x are always a
  // multiple of four halves here, so the 8-byte alignment holds.
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
  *second = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
  return a;
}

// One super-block's contribution to one row, given this lane's slice of x
// already in registers. Splitting it out is what lets a block walk several rows
// against a single load of x.
template <KQuantType TYPE>
__device__ __forceinline__ float kq_block_dot(const std::uint8_t* __restrict__ p, int lane,
                                              const float* xa, const float* xh) {
  if (TYPE == KQuantType::Q6_K) {
    const int n = lane >> 4;
    const int rest = (lane << 2) & 63;
    const int h = rest >> 5;
    const int l0 = rest & 31;

    const std::uint8_t* qlp = p + (lane << 2);
    const std::uint32_t qw =
        static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qlp)) |
        (static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qlp + 2)) << 16);
    const std::uint8_t* qhp = p + 128 + (n << 5) + l0;
    const std::uint32_t hw =
        static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qhp)) |
        (static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(qhp + 2)) << 16);

    const int sbase = 192 + (n << 3) + (l0 >> 4);
    const float scl = static_cast<float>(*reinterpret_cast<const std::int8_t*>(p + sbase + (h << 1)));
    const float sch =
        static_cast<float>(*reinterpret_cast<const std::int8_t*>(p + sbase + ((2 + h) << 1)));
    std::uint16_t dbits;
    memcpy(&dbits, p + 208, 2);
    const float d = half_bits_to_float(dbits);

    const int sl = 2 * h;
    const int sh = 4 + 2 * h;
    float sq_lo = 0.0f;
    float sq_hi = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const std::uint32_t byte = (qw >> (8 * k)) & 0xFFu;
      const std::uint32_t hb = (hw >> (8 * k)) & 0xFFu;
      const int qlo = static_cast<int>((byte & 0x0Fu) | (((hb >> sl) & 3u) << 4)) - 32;
      const int qhi = static_cast<int>((byte >> 4) | (((hb >> sh) & 3u) << 4)) - 32;
      sq_lo += static_cast<float>(qlo) * xa[k];
      sq_hi += static_cast<float>(qhi) * xh[k];
    }
    return d * (scl * sq_lo + sch * sq_hi);
  }

  const uint4 hdr = *reinterpret_cast<const uint4*>(p);
  const float d = half_bits_to_float(static_cast<std::uint16_t>(hdr.x & 0xFFFFu));
  const float dmin = half_bits_to_float(static_cast<std::uint16_t>(hdr.x >> 16));
  const int j = lane >> 3;
  const int t = lane & 7;

  const int qoff = (TYPE == KQuantType::Q4_K) ? 16 : 48;
  const std::uint32_t qw = *reinterpret_cast<const std::uint32_t*>(p + qoff + (lane << 2));
  std::uint32_t hw = 0;
  if (TYPE == KQuantType::Q5_K) {
    hw = *reinterpret_cast<const std::uint32_t*>(p + 16 + (t << 2));
  }

  float sc0, m0, sc1, m1;
  scale_min_from_hdr(2 * j, hdr.y, hdr.z, hdr.w, &sc0, &m0);
  scale_min_from_hdr(2 * j + 1, hdr.y, hdr.z, hdr.w, &sc1, &m1);

  float sq0 = 0.0f;
  float sx0 = 0.0f;
  float sq1 = 0.0f;
  float sx1 = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const std::uint32_t byte = (qw >> (8 * k)) & 0xFFu;
    std::uint32_t qlo = byte & 0x0Fu;
    std::uint32_t qhi = byte >> 4;
    if (TYPE == KQuantType::Q5_K) {
      const std::uint32_t hb = (hw >> (8 * k)) & 0xFFu;
      qlo |= ((hb >> (2 * j)) & 1u) << 4;
      qhi |= ((hb >> (2 * j + 1)) & 1u) << 4;
    }
    sq0 += static_cast<float>(qlo) * xa[k];
    sx0 += xa[k];
    sq1 += static_cast<float>(qhi) * xh[k];
    sx1 += xh[k];
  }
  // Scales are uniform across a 32-element group, so applying them to this
  // lane's partial sums is the same arithmetic as applying them per weight.
  return d * (sc0 * sq0 + sc1 * sq1) - dmin * (m0 * sx0 + m1 * sx1);
}

// Vector-load packed matvec. The scalar kernel was bound by load-instruction
// issue rather than bandwidth: it fetched one byte of qs, one half of x and the
// scale bytes separately for EVERY weight, roughly four loads per element.
//
// A warp owns one super-block and each lane takes a 4-byte slice of its
// quantized payload, which is 8 weights (4 low nibbles, 4 high nibbles). Those
// two runs of 4 always fall in exactly two 32-element scale groups, so scales
// apply once per run rather than once per weight, and the whole 16-byte header
// (d, dmin and all 12 scale bytes) arrives in one load.
//
// ROWS rows share each load of x, which is otherwise re-read once per row and
// accounts for half the remaining load instructions.
//
// Q6_K blocks are 210 bytes, so consecutive blocks are only 2-byte aligned and
// it uses uint16 pairs where the others use uint32.
__device__ __forceinline__ void store_out(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }

// WPR warps cooperate on one row, so a block covers 8/WPR rows.
//
// One warp per row suits tall matrices, but the attention projections are short:
// at 1024 rows that launches 128 blocks onto 170 SMs, leaving most of the GPU
// idle with nothing to hide memory latency behind. Splitting a row across warps
// multiplies the grid by WPR and costs one shared-memory reduction.
template <KQuantType TYPE, int WPR, typename OutT>
__global__ void kquant_matvec_vec_kernel(const std::uint8_t* __restrict__ w,
                                         const __half* __restrict__ x, OutT* __restrict__ y,
                                         int rows, int cols) {
  constexpr int kWarps = 8;  // 256 threads
  constexpr int kRowsPerBlock = kWarps / WPR;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);

  std::size_t block_bytes;
  if (TYPE == KQuantType::Q4_K) {
    block_bytes = model::kquant::kQ4KBlockBytes;
  } else if (TYPE == KQuantType::Q5_K) {
    block_bytes = model::kquant::kQ5KBlockBytes;
  } else {
    block_bytes = model::kquant::kQ6KBlockBytes;
  }
  const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * block_bytes;

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int sub = warp % WPR;  // which slice of this row's super-blocks
  const int row = blockIdx.x * kRowsPerBlock + warp / WPR;

  // This lane's two runs of four weights sit at fixed offsets inside every
  // super-block, so the x offsets are loop-invariant.
  int xo;
  int xstride;
  if (TYPE == KQuantType::Q6_K) {
    const int rest = (lane << 2) & 63;
    xo = ((lane >> 4) << 7) + ((rest >> 5) << 5) + (rest & 31);
    xstride = 64;
  } else {
    xo = ((lane >> 3) << 6) + ((lane & 7) << 2);
    xstride = 32;
  }

  float acc = 0.0f;
  if (row < rows) {
    const std::uint8_t* p = w + static_cast<std::size_t>(row) * row_bytes +
                            static_cast<std::size_t>(sub) * block_bytes;
    // Two super-blocks per iteration so both sets of loads are in flight before
    // either is consumed. Profiling put 59% of warp stalls on memory scoreboard
    // dependencies at 90% occupancy and only 48% DRAM utilisation, which is a
    // shortage of loads in flight rather than of warps.
#pragma unroll 2
    for (int b = sub; b < blocks_per_row; b += WPR) {
      const __half* xb = x + b * static_cast<int>(kSuperBlock);
      float2 a1;
      float2 h1;
      const float2 a0 = load_x4(xb + xo, &a1);
      const float2 h0 = load_x4(xb + xo + xstride, &h1);
      const float xa[4] = {a0.x, a0.y, a1.x, a1.y};
      const float xh[4] = {h0.x, h0.y, h1.x, h1.y};
      acc += kq_block_dot<TYPE>(p, lane, xa, xh);
      p += block_bytes * WPR;
    }
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);

  if (WPR == 1) {
    if (lane == 0 && row < rows) store_out(y + row, acc);
    return;
  }
  __shared__ float parts[kWarps];
  if (lane == 0) parts[warp] = acc;
  __syncthreads();
  if (lane == 0 && sub == 0 && row < rows) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < WPR; ++k) total += parts[warp + k];
    store_out(y + row, total);
  }
}

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
  constexpr int kThreads = 256;  // 8 warps, one super-block each
  const unsigned grid = static_cast<unsigned>((blocks + 7) / 8);
  switch (type) {
    case KQuantType::Q4_K:
      dequant_kquant_vec_kernel<KQuantType::Q4_K><<<grid, kThreads, 0, stream>>>(blocks_in, out, blocks);
      break;
    case KQuantType::Q5_K:
      dequant_kquant_vec_kernel<KQuantType::Q5_K><<<grid, kThreads, 0, stream>>>(blocks_in, out, blocks);
      break;
    case KQuantType::Q6_K:
      dequant_kquant_vec_kernel<KQuantType::Q6_K><<<grid, kThreads, 0, stream>>>(blocks_in, out, blocks);
      break;
  }
}

// Batched packed matmul: y[b, n] = sum_k x[b, k] * W[n, k], with W left in its
// k-quant blocks.
//
// The alternative is what prefill and batched decode did before: expand the
// whole weight to fp16 and hand it to cuBLAS. That reads the packed weight once
// but writes and then re-reads its fp16 form, roughly eight times the packed
// bytes, which is why batch-64 decode ran at 137 tok/s against 323 for fp16.
//
// Here one block owns an output row and its eight warps split the row's
// super-blocks. Each lane unpacks its eight weights once and reuses them across
// a tile of BMAX batch elements, so the weight is read ceil(batch/BMAX) times
// rather than once per element. That beats the expand-and-GEMM traffic while
// ceil(batch/BMAX) stays under about eight; past that cuBLAS on the expanded
// weight wins, and tensor cores widen the gap, so the launcher hands large
// batches back to the caller.
template <KQuantType TYPE, int BMAX>
__global__ void kquant_matmul_kernel(const std::uint8_t* __restrict__ w,
                                     const __half* __restrict__ x, __half* __restrict__ y, int rows,
                                     int cols, int batch, int ldy) {
  constexpr int kWarps = 8;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);

  std::size_t block_bytes;
  if (TYPE == KQuantType::Q4_K) {
    block_bytes = model::kquant::kQ4KBlockBytes;
  } else if (TYPE == KQuantType::Q5_K) {
    block_bytes = model::kquant::kQ5KBlockBytes;
  } else {
    block_bytes = model::kquant::kQ6KBlockBytes;
  }
  const std::uint8_t* rowp =
      w + static_cast<std::size_t>(row) * static_cast<std::size_t>(blocks_per_row) * block_bytes;

  __shared__ float parts[kWarps][BMAX];

  for (int b0 = 0; b0 < batch; b0 += BMAX) {
    const int bn = min(BMAX, batch - b0);
    float acc[BMAX];
#pragma unroll
    for (int t = 0; t < BMAX; ++t) acc[t] = 0.0f;

    const std::uint8_t* p = rowp + static_cast<std::size_t>(warp) * block_bytes;
    for (int sb = warp; sb < blocks_per_row; sb += kWarps) {
      float wlo[4];
      float whi[4];
      int off_lo;
      int off_hi;
      kq_block_values<TYPE>(p, lane, wlo, whi, &off_lo, &off_hi);
      const int xbase = sb * static_cast<int>(kSuperBlock);
      // The bound is BMAX, a compile-time constant, with the tail masked rather
      // than shortening the loop. A runtime bound leaves the compiler unable to
      // prove the index range, so acc[] lands in local memory and every
      // accumulate becomes a load-modify-store -- which measured as roughly 2%
      // of fp32 peak.
#pragma unroll
      for (int t = 0; t < BMAX; ++t) {
        if (t >= bn) continue;
        const __half* xb = x + static_cast<std::size_t>(b0 + t) * static_cast<std::size_t>(cols) +
                           xbase;
        float2 a1;
        float2 h1;
        const float2 a0 = load_x4(xb + off_lo, &a1);
        const float2 h0 = load_x4(xb + off_hi, &h1);
        acc[t] += wlo[0] * a0.x + wlo[1] * a0.y + wlo[2] * a1.x + wlo[3] * a1.y + whi[0] * h0.x +
                  whi[1] * h0.y + whi[2] * h1.x + whi[3] * h1.y;
      }
      p += block_bytes * kWarps;
    }

#pragma unroll
    for (int t = 0; t < BMAX; ++t) {
      if (t >= bn) continue;
      float v = acc[t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
      if (lane == 0) parts[warp][t] = v;
    }
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < bn) {
      const int t = static_cast<int>(threadIdx.x);
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) total += parts[k][t];
      // ldy, not rows: a split QKV writes into a row range of the fused output.
      y[static_cast<std::size_t>(b0 + t) * static_cast<std::size_t>(ldy) + row] =
          __float2half(total);
    }
    __syncthreads();
  }
}

// Activations in the form the packed matmul wants: int8 with one scale per
// 32-element group, plus that group's sum. This is llama.cpp's q8_1 idea.
//
// The sum is needed because a k-quant weight is `d*sc*q - dmin*m`: the dot
// splits into an integer part (sum q*xq, which dp4a does four at a time) and a
// correction proportional to sum(x) over the group. Precomputing the sum here
// keeps it out of the inner loop.
//
// One warp per group: 32 lanes, one element each.
__global__ void quantize_q8_1_groups_kernel(const __half* __restrict__ x, std::int8_t* __restrict__ q,
                                            float* __restrict__ scale, float* __restrict__ gsum,
                                            int cols, int groups_per_row) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int g = blockIdx.x * (blockDim.x >> 5) + warp;
  const int total = gridDim.x * (blockDim.x >> 5);
  (void)total;
  const int row = g / groups_per_row;
  const int gi = g % groups_per_row;
  const std::size_t base = static_cast<std::size_t>(row) * cols + gi * 32;

  const float v = __half2float(x[base + lane]);
  float amax = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
  }
  const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
  const int qi = __float2int_rn(v / s);
  q[base + lane] = static_cast<std::int8_t>(max(-127, min(127, qi)));

  float sum = static_cast<float>(qi);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
  if (lane == 0) {
    scale[g] = s;
    // Stored already multiplied by the scale: the inner loop wants sum(x), not
    // sum(xq).
    gsum[g] = sum * s;
  }
}

// dp4a form of the batched packed matmul, Q4_K/Q5_K only (their scales are per
// 32 elements, which is what the activation groups are cut to).
//
// Per lane per super-block the previous kernel did eight float multiply-adds per
// batch element. Here the four low nibbles are already four int8 lanes of one
// word (`qw & 0x0F0F0F0F`), so one dp4a does that whole run, and a second
// against 0x01010101 gets the group sum the `-dmin*m` term needs.
template <KQuantType TYPE, int BMAX>
__global__ void kquant_matmul_dp4a_kernel(const std::uint8_t* __restrict__ w,
                                          const std::int8_t* __restrict__ xq,
                                          const float* __restrict__ xs,
                                          const float* __restrict__ xsum, __half* __restrict__ y,
                                          int rows, int cols, int batch, int ldy) {
  constexpr int kWarps = 8;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);
  const int groups_per_row = cols >> 5;

  const std::size_t block_bytes =
      TYPE == KQuantType::Q4_K ? model::kquant::kQ4KBlockBytes : model::kquant::kQ5KBlockBytes;
  const std::uint8_t* rowp =
      w + static_cast<std::size_t>(row) * static_cast<std::size_t>(blocks_per_row) * block_bytes;

  const int j = lane >> 3;
  const int t4 = lane & 7;
  const int off_lo = (j << 6) + (t4 << 2);
  const int off_hi = off_lo + 32;

  __shared__ float parts[kWarps][BMAX];

  for (int b0 = 0; b0 < batch; b0 += BMAX) {
    const int bn = min(BMAX, batch - b0);
    float acc[BMAX];
#pragma unroll
    for (int t = 0; t < BMAX; ++t) acc[t] = 0.0f;

    const std::uint8_t* p = rowp + static_cast<std::size_t>(warp) * block_bytes;
    for (int sb = warp; sb < blocks_per_row; sb += kWarps) {
      const uint4 hdr = *reinterpret_cast<const uint4*>(p);
      const float d = half_bits_to_float(static_cast<std::uint16_t>(hdr.x & 0xFFFFu));
      const float dmin = half_bits_to_float(static_cast<std::uint16_t>(hdr.x >> 16));
      const int qoff = (TYPE == KQuantType::Q4_K) ? 16 : 48;
      const std::uint32_t qw = *reinterpret_cast<const std::uint32_t*>(p + qoff + (lane << 2));
      float sc0, m0, sc1, m1;
      scale_min_from_hdr(2 * j, hdr.y, hdr.z, hdr.w, &sc0, &m0);
      scale_min_from_hdr(2 * j + 1, hdr.y, hdr.z, hdr.w, &sc1, &m1);

      int qlo = static_cast<int>(qw & 0x0F0F0F0Fu);
      int qhi = static_cast<int>((qw >> 4) & 0x0F0F0F0Fu);
      if (TYPE == KQuantType::Q5_K) {
        const std::uint32_t hw = *reinterpret_cast<const std::uint32_t*>(p + 16 + (t4 << 2));
        const std::uint32_t bit_lo = (hw >> (2 * j)) & 0x01010101u;
        const std::uint32_t bit_hi = (hw >> (2 * j + 1)) & 0x01010101u;
        qlo |= static_cast<int>(bit_lo << 4);
        qhi |= static_cast<int>(bit_hi << 4);
      }

      const int xbase = sb * static_cast<int>(kSuperBlock);
      const int gbase = sb << 3;  // eight 32-groups per super-block
#pragma unroll
      for (int t = 0; t < BMAX; ++t) {
        if (t >= bn) continue;
        const std::size_t xrow = static_cast<std::size_t>(b0 + t) * static_cast<std::size_t>(cols);
        const std::size_t grow =
            static_cast<std::size_t>(b0 + t) * static_cast<std::size_t>(groups_per_row);
        const int xl = *reinterpret_cast<const int*>(xq + xrow + xbase + off_lo);
        const int xh = *reinterpret_cast<const int*>(xq + xrow + xbase + off_hi);
        const int dot_lo = __dp4a(qlo, xl, 0);
        const int dot_hi = __dp4a(qhi, xh, 0);
        const int sum_lo = __dp4a(xl, 0x01010101, 0);
        const int sum_hi = __dp4a(xh, 0x01010101, 0);
        const float slo = xs[grow + gbase + 2 * j];
        const float shi = xs[grow + gbase + 2 * j + 1];
        acc[t] += slo * (d * sc0 * static_cast<float>(dot_lo) -
                         dmin * m0 * static_cast<float>(sum_lo)) +
                  shi * (d * sc1 * static_cast<float>(dot_hi) -
                         dmin * m1 * static_cast<float>(sum_hi));
      }
      p += block_bytes * kWarps;
    }

#pragma unroll
    for (int t = 0; t < BMAX; ++t) {
      if (t >= bn) continue;
      float v = acc[t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
      if (lane == 0) parts[warp][t] = v;
    }
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < bn) {
      const int t = static_cast<int>(threadIdx.x);
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) total += parts[k][t];
      y[static_cast<std::size_t>(b0 + t) * static_cast<std::size_t>(ldy) + row] =
          __float2half(total);
    }
    __syncthreads();
  }
  (void)xsum;
}


// Q4_K matmul on int8 tensor cores -- the MMQ shape, adapted from
// moe_int4_grouped_mma_kernel.
//
// Everything above this in the file multiplies packed weights with scalar FMAs,
// which is why cuBLAS on an expanded weight kept winning above a batch of about
// eight: it has tensor cores and they do not. This one keeps the weight packed
// AND uses them, so it should beat both.
//
// Two things make Q4_K fit the MoE kernel's tiling almost unchanged. A 32-column
// group of one row is a contiguous 32-byte run of qs -- low nibbles when the
// group index is even, high when odd -- because a super-block's 256 weights are
// laid out as four 32-byte runs each supplying a low and a high group. And the
// scale index inside the block is just (group % 8).
//
// The one real difference from the MoE version is the minimum term. int4 there
// is symmetric, so the int32 dot times two scales is the whole answer. A k-quant
// weight is d*sc*q - dmin*m, so the dot needs a rank-1 correction:
//   sum_i w_i x_i = d*sc * (sum_i q_i xq_i) * as - dmin*m * (sum_i x_i)
// The activation group sum is precomputed by quantize_q8_1_groups_kernel, so the
// correction costs one multiply-add per accumulator per K-group and never
// touches the tensor cores.
namespace {
// M tile is fixed by the batch range this path serves. NT is the number of
// 8-wide n-tiles each warp-half owns, so BN = 2 * NT * 8: NT=4 gives a 64-wide
// output tile, NT=2 gives 32. NT is the register knob -- each n-tile carries 4
// accumulators, and at NT=4 the kernel needed 128 registers per thread, which
// capped it at 2 blocks/SM and 33% theoretical occupancy.
constexpr int kMmqBM = 64;
}  // namespace

// A whole super-block of K per staging round (llama.cpp's MMQ_ITER_K=256).
//
// Staging 32 columns at a time cost two __syncthreads per 32 columns: with
// Kg=128 that is 256 barriers per block to cover one row, against 8 mma
// instructions each. Widening to 256 amortises it 8x.
//
// An earlier attempt at this was slower, because widening the tile is only half
// the change -- the staging has to stay fully parallel and vectorized. Here every
// thread moves 16 bytes at a time with uint4 loads and stores:
//   As: 64 rows x 256 bytes = 1024 uint4 units, 4 per thread.
//   Bs: each row's 256 weights come from 128 bytes of qs, so 32 rows is 256
//       uint4 units, 1 per thread, expanding to two 16-byte runs (the low and
//       high nibbles of a 16-byte chunk land 32 columns apart).
template <int NT>
__global__ __launch_bounds__(256) void kquant_mmq_q4k_kernel(
    const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
    const float* __restrict__ as, const float* __restrict__ gsum, __half* __restrict__ y, int rows,
    int cols, int batch, int ldy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  (void)w; (void)xq; (void)as; (void)gsum; (void)y; (void)rows; (void)cols; (void)batch; (void)ldy;
#else
  constexpr int kMmqBN = 2 * NT * 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3, mt = warp & 3, nhalf = warp >> 2;
  const int blockNrow = blockIdx.x * kMmqBN;
  const int Kg = cols >> 5;
  const int blocks_per_row = cols >> 8;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks_per_row) * model::kquant::kQ4KBlockBytes;

  __shared__ __align__(16) std::int8_t As[kMmqBM][256];
  __shared__ __align__(16) std::int8_t Bs[kMmqBN][256];
  __shared__ float Bsc[kMmqBN][8];
  __shared__ float Bdm[kMmqBN][8];

  float facc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) facc[i][j] = 0.0f;

  for (int sb = 0; sb < blocks_per_row; ++sb) {
    const int k0 = sb << 8;

    // Activations: 16 uint4 per row, 4 units per thread.
#pragma unroll
    for (int i = 0; i < (kMmqBM * 16) / 256; ++i) {
      const int u = tid + i * 256;
      const int r = u >> 4, c = (u & 15) << 4;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < batch) v = *reinterpret_cast<const uint4*>(xq + static_cast<std::size_t>(r) * cols + k0 + c);
      *reinterpret_cast<uint4*>(&As[r][c]) = v;
    }

    // Weights: one uint4 of qs per thread expands to two 16-byte runs of Bs.
    {
      const int r = tid >> 3, chunk = (tid & 7) << 4;  // 8 chunks of 16 bytes = 128
      if (r < kMmqBN) {
        const int wrow = blockNrow + r;
        uint4 q = make_uint4(0, 0, 0, 0);
        if (wrow < rows) {
          const std::uint8_t* p = w + static_cast<std::size_t>(wrow) * row_bytes +
                                  static_cast<std::size_t>(sb) * model::kquant::kQ4KBlockBytes;
          q = *reinterpret_cast<const uint4*>(p + 16 + chunk);
        }
        const int j = chunk >> 5, l0 = chunk & 31;
        const int lo_col = (j << 6) + l0, hi_col = lo_col + 32;
        uint4 lo, hi;
        const std::uint32_t* qw = &q.x;
        std::uint32_t* lw = &lo.x;
        std::uint32_t* hw = &hi.x;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          lw[k] = qw[k] & 0x0F0F0F0Fu;
          hw[k] = (qw[k] >> 4) & 0x0F0F0F0Fu;
        }
        *reinterpret_cast<uint4*>(&Bs[r][lo_col]) = lo;
        *reinterpret_cast<uint4*>(&Bs[r][hi_col]) = hi;
      }
    }

    // Scales: one thread per output row decodes all eight groups from one header.
    if (tid < kMmqBN) {
      const int wrow = blockNrow + tid;
      float d = 0.0f, dmin = 0.0f;
      uint4 hdr = make_uint4(0, 0, 0, 0);
      if (wrow < rows) {
        hdr = *reinterpret_cast<const uint4*>(w + static_cast<std::size_t>(wrow) * row_bytes +
                                              static_cast<std::size_t>(sb) *
                                                  model::kquant::kQ4KBlockBytes);
        d = half_bits_to_float(static_cast<std::uint16_t>(hdr.x & 0xFFFFu));
        dmin = half_bits_to_float(static_cast<std::uint16_t>(hdr.x >> 16));
      }
#pragma unroll
      for (int gi = 0; gi < 8; ++gi) {
        float sc = 0.0f, m = 0.0f;
        scale_min_from_hdr(gi, hdr.y, hdr.z, hdr.w, &sc, &m);
        Bsc[tid][gi] = d * sc;
        Bdm[tid][gi] = dmin * m;
      }
    }
    __syncthreads();

    const int arow = 16 * mt;
    const int m_a = arow + g, m_b = arow + g + 8;
    const bool ok_a = m_a < batch, ok_b = m_b < batch;
#pragma unroll
    for (int gi = 0; gi < 8; ++gi) {
      const int kg = (sb << 3) + gi, co = gi << 5;
      const int a0 = *reinterpret_cast<int*>(&As[m_a][co + t * 4]);
      const int a1 = *reinterpret_cast<int*>(&As[m_b][co + t * 4]);
      const int a2 = *reinterpret_cast<int*>(&As[m_a][co + 16 + t * 4]);
      const int a3 = *reinterpret_cast<int*>(&As[m_b][co + 16 + t * 4]);
      const float as_a = ok_a ? as[static_cast<std::size_t>(m_a) * Kg + kg] : 0.0f;
      const float as_b = ok_b ? as[static_cast<std::size_t>(m_b) * Kg + kg] : 0.0f;
      const float gs_a = ok_a ? gsum[static_cast<std::size_t>(m_a) * Kg + kg] : 0.0f;
      const float gs_b = ok_b ? gsum[static_cast<std::size_t>(m_b) * Kg + kg] : 0.0f;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int nbase = nhalf * (NT * 8) + nt * 8;
        const int b0 = *reinterpret_cast<int*>(&Bs[nbase + g][co + t * 4]);
        const int b1 = *reinterpret_cast<int*>(&Bs[nbase + g][co + 16 + t * 4]);
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        const int na = nbase + 2 * t, nb = nbase + 2 * t + 1;
        facc[nt][0] += static_cast<float>(c0) * as_a * Bsc[na][gi] - Bdm[na][gi] * gs_a;
        facc[nt][1] += static_cast<float>(c1) * as_a * Bsc[nb][gi] - Bdm[nb][gi] * gs_a;
        facc[nt][2] += static_cast<float>(c2) * as_b * Bsc[na][gi] - Bdm[na][gi] * gs_b;
        facc[nt][3] += static_cast<float>(c3) * as_b * Bsc[nb][gi] - Bdm[nb][gi] * gs_b;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int nbase = nhalf * (NT * 8) + nt * 8;
    const int rr[4] = {g, g, g + 8, g + 8};
    const int cc[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
#pragma unroll
    for (int en = 0; en < 4; ++en) {
      const int lm = 16 * mt + rr[en], gn = blockNrow + nbase + cc[en];
      if (lm < batch && gn < rows) {
        y[static_cast<std::size_t>(lm) * static_cast<std::size_t>(ldy) + gn] =
            __float2half(facc[nt][en]);
      }
    }
  }
#endif
}

// Q4_K on int8 tensor cores. Same activation preparation as the dp4a path; the
// difference is the inner product runs on mma instead of scalar FMAs. sm_80+
// only -- the caller checks compute capability.
bool launch_kquant_mmq(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                       int cols, int batch, int ldy, std::int8_t* xq, float* xs, float* xsum,
                       cudaStream_t stream) {
  if (type != KQuantType::Q4_K) return false;
  if (batch < 1 || batch > kMmqBM || cols % 256 != 0) return false;
  const int groups_per_row = cols >> 5;
  const int total_groups = batch * groups_per_row;
  quantize_q8_1_groups_kernel<<<(total_groups + 7) / 8, 256, 0, stream>>>(x, xq, xs, xsum, cols,
                                                                         groups_per_row);
  // NT trades output-tile width against registers and grid size.
  static const int nt = []() {
    const char* e = std::getenv("CPI_KQUANT_MMQ_NT");
    return e != nullptr ? std::atoi(e) : 2;
  }();
  if (nt >= 4) {
    kquant_mmq_q4k_kernel<4><<<(rows + 63) / 64, 256, 0, stream>>>(w, xq, xs, xsum, y, rows, cols,
                                                                   batch, ldy);
  } else {
    kquant_mmq_q4k_kernel<2><<<(rows + 31) / 32, 256, 0, stream>>>(w, xq, xs, xsum, y, rows, cols,
                                                                   batch, ldy);
  }
  return true;
}

// Quantize activations once per GEMM, then run the dp4a matmul.
bool launch_kquant_matmul_dp4a(const std::uint8_t* w, KQuantType type, const half* x, half* y,
                               int rows, int cols, int batch, int ldy, std::int8_t* xq, float* xs,
                               float* xsum, cudaStream_t stream) {
  if (type == KQuantType::Q6_K) return false;  // per-16 scales, different grouping
  if (batch < 2 || cols % 32 != 0) return false;
  const int groups_per_row = cols >> 5;
  const int total_groups = batch * groups_per_row;
  constexpr int kQThreads = 256;  // 8 warps, one group each
  const int qgrid = (total_groups + 7) / 8;
  quantize_q8_1_groups_kernel<<<qgrid, kQThreads, 0, stream>>>(x, xq, xs, xsum, cols,
                                                               groups_per_row);
  constexpr int kThreads = 256;
  if (batch <= 4) {
    if (type == KQuantType::Q4_K) {
      kquant_matmul_dp4a_kernel<KQuantType::Q4_K, 4>
          <<<rows, kThreads, 0, stream>>>(w, xq, xs, xsum, y, rows, cols, batch, ldy);
    } else {
      kquant_matmul_dp4a_kernel<KQuantType::Q5_K, 4>
          <<<rows, kThreads, 0, stream>>>(w, xq, xs, xsum, y, rows, cols, batch, ldy);
    }
  } else {
    if (type == KQuantType::Q4_K) {
      kquant_matmul_dp4a_kernel<KQuantType::Q4_K, 16>
          <<<rows, kThreads, 0, stream>>>(w, xq, xs, xsum, y, rows, cols, batch, ldy);
    } else {
      kquant_matmul_dp4a_kernel<KQuantType::Q5_K, 16>
          <<<rows, kThreads, 0, stream>>>(w, xq, xs, xsum, y, rows, cols, batch, ldy);
    }
  }
  return true;
}

bool launch_kquant_matmul(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                          int cols, int batch, int ldy, cudaStream_t stream) {
  // This kernel is the mat-vec shape widened by a batch tile: warps split one
  // row and reduce partial products. llama.cpp uses that shape (MMVQ) only for
  // small batches and switches to MMQ above -- activations quantized to q8_1,
  // weight tiles staged in shared memory, int8 tensor cores. CPI has no MMQ
  // equivalent, so past the crossover the expanded-weight cuBLAS GEMM is the
  // better answer: it at least gets fp16 tensor cores.
  static const int max_batch = []() {
    const char* e = std::getenv("CPI_KQUANT_MM_MAX");
    return e != nullptr ? std::atoi(e) : 8;
  }();
  if (batch < 2 || batch > max_batch) return false;
  constexpr int kThreads = 256;
#define CPI_KQ_MM(BM)                                                             \
  switch (type) {                                                                 \
    case KQuantType::Q4_K:                                                        \
      kquant_matmul_kernel<KQuantType::Q4_K, BM>                                  \
          <<<rows, kThreads, 0, stream>>>(w, x, y, rows, cols, batch, ldy);            \
      return true;                                                                \
    case KQuantType::Q5_K:                                                        \
      kquant_matmul_kernel<KQuantType::Q5_K, BM>                                  \
          <<<rows, kThreads, 0, stream>>>(w, x, y, rows, cols, batch, ldy);            \
      return true;                                                                \
    case KQuantType::Q6_K:                                                        \
      kquant_matmul_kernel<KQuantType::Q6_K, BM>                                  \
          <<<rows, kThreads, 0, stream>>>(w, x, y, rows, cols, batch, ldy);            \
      return true;                                                                \
  }
  if (batch <= 4) {
    CPI_KQ_MM(4)
  } else {
    CPI_KQ_MM(16)
  }
#undef CPI_KQ_MM
  return false;
}

template <typename OutT>
static void launch_kquant_matvec_impl(const std::uint8_t* w, KQuantType type, const half* x, OutT* y,
                                      int rows, int cols, cudaStream_t stream) {
  constexpr int kThreads = 256;  // 8 warps
  // Warps per row. Short matrices need several or the grid cannot fill the GPU;
  // tall ones do better with one warp per row and no shared reduction.
  static const int forced = []() {
    const char* e = std::getenv("CPI_KQUANT_WPR");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  // Flat 8, measured end to end: 222.0 tok/s against 220.5 / 218.7 / 210.9 for
  // 8 / 4 / 1 in an earlier sweep. Two shape-aware rules were tried and both
  // lost (205.4 and 216.8) even though isolated per-shape benchmarks favoured
  // them -- a matvec measured alone owns the whole GPU and a warm L2, which is
  // nothing like running back to back inside the decode graph. Only end-to-end
  // numbers decide this.
  int wpr = forced;
  if (wpr <= 0) wpr = 8;
  const int rows_per_block = 8 / wpr;
  const int grid = (rows + rows_per_block - 1) / rows_per_block;

#define CPI_KQ_DISPATCH(WPRV)                                     switch (type) {                                                   case KQuantType::Q4_K:                                            kquant_matvec_vec_kernel<KQuantType::Q4_K, WPRV, OutT>              <<<grid, kThreads, 0, stream>>>(w, x, y, rows, cols);       return;                                                       case KQuantType::Q5_K:                                            kquant_matvec_vec_kernel<KQuantType::Q5_K, WPRV, OutT>              <<<grid, kThreads, 0, stream>>>(w, x, y, rows, cols);       return;                                                       case KQuantType::Q6_K:                                            kquant_matvec_vec_kernel<KQuantType::Q6_K, WPRV, OutT>              <<<grid, kThreads, 0, stream>>>(w, x, y, rows, cols);       return;                                                     }

  if (wpr >= 8) {
    CPI_KQ_DISPATCH(8)
  } else if (wpr >= 4) {
    CPI_KQ_DISPATCH(4)
  } else if (wpr >= 2) {
    CPI_KQ_DISPATCH(2)
  } else {
    CPI_KQ_DISPATCH(1)
  }
#undef CPI_KQ_DISPATCH
}

void launch_kquant_matvec(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                          int cols, cudaStream_t stream) {
  launch_kquant_matvec_impl(w, type, x, y, rows, cols, stream);
}

void launch_kquant_matvec_f32(const std::uint8_t* w, KQuantType type, const half* x, float* y,
                              int rows, int cols, cudaStream_t stream) {
  launch_kquant_matvec_impl(w, type, x, y, rows, cols, stream);
}

}  // namespace kernels
