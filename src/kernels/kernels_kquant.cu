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
#include <cstdio>
#include <cstring>

#include <cuda_pipeline.h>

#include "model/gguf_kquant.hpp"
#include "runtime/kernels.cuh"

namespace kernels {

// Per-box tuning knobs for the k-quant kernels.
//
// These were `static const` reads of one env var each, which meant a value could
// only be chosen before the process started -- fine for a manual sweep, useless
// for a tuner that has to re-time an incumbent against a candidate inside one
// run. Every wrong call in this kernel's history came from comparing numbers
// taken in different processes, where thermal drift is larger than the effect.
// Defaults are the hand-measured RTX 5090 values; env still overrides, and
// set_kquant_tuning lets the autotuner search.
namespace {
KQuantTuning g_kq_tune = []() {
  KQuantTuning t;
  const auto env_int = [](const char* name, int fallback) {
    const char* e = std::getenv(name);
    return e != nullptr ? std::atoi(e) : fallback;
  };
  t.mm_max = env_int("CPI_KQUANT_MM_MAX", t.mm_max);
  t.mmq_nt = env_int("CPI_KQUANT_MMQ_NT", t.mmq_nt);
  t.mmq_async_nt = env_int("CPI_KQUANT_MMQ_ANT", t.mmq_async_nt);
  t.mmq_async = env_int("CPI_KQUANT_MMQ_ASYNC", t.mmq_async);
  t.matvec_wpr = env_int("CPI_KQUANT_WPR", t.matvec_wpr);
  t.matvec_dp4a = env_int("CPI_KQUANT_MATVEC_DP4A", t.matvec_dp4a);
  t.matvec_dp4a16 = env_int("CPI_KQUANT_MATVEC_DP4A16", t.matvec_dp4a16);
  t.matvec_dp4a_q6k = env_int("CPI_KQUANT_MATVEC_DP4A_Q6K", t.matvec_dp4a_q6k);
  t.matvec_share_x = env_int("CPI_KQUANT_MATVEC_SHARE_X", t.matvec_share_x);
  t.matvec_gsum = env_int("CPI_KQUANT_MATVEC_GSUM", t.matvec_gsum);
  t.mmq_q6k = env_int("CPI_KQUANT_MMQ_Q6K", t.mmq_q6k);
  t.mmq_splitk = env_int("CPI_KQUANT_MMQ_SPLITK", t.mmq_splitk);
  t.mmq_split_target = env_int("CPI_KQUANT_MMQ_SPLIT_TARGET", t.mmq_split_target);
  t.mmq_streamk = env_int("CPI_KQUANT_MMQ_STREAMK", t.mmq_streamk);
  t.mmq_q6k_nt = env_int("CPI_KQUANT_MMQ_Q6K_NT", t.mmq_q6k_nt);
  t.matvec_vecx = env_int("CPI_KQUANT_MATVEC_VECX", t.matvec_vecx);
  return t;
}();
}  // namespace

void set_kquant_tuning(const KQuantTuning& t) { g_kq_tune = t; }
KQuantTuning get_kquant_tuning() { return g_kq_tune; }

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
    block_bytes = model::kquant::kQ6KResidentBytes;
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
__device__ __forceinline__ float load_out(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_out(const float* p) { return *p; }
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }

// WPR warps cooperate on one row, so a block covers 8/WPR rows.
//
// One warp per row suits tall matrices, but the attention projections are short:
// at 1024 rows that launches 128 blocks onto 170 SMs, leaving most of the GPU
// idle with nothing to hide memory latency behind. Splitting a row across warps
// multiplies the grid by WPR and costs one shared-memory reduction.
template <KQuantType TYPE, int WPR, typename OutT, bool ACC>
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
    block_bytes = model::kquant::kQ6KResidentBytes;
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
    if (lane == 0 && row < rows) {
      // ACC folds the residual add into this store: the fp16 path has always
      // done it in the projection epilogue, and the packed path was paying a
      // whole extra kernel per layer for it.
      store_out(y + row, ACC ? acc + load_out(y + row) : acc);
    }
    return;
  }
  __shared__ float parts[kWarps];
  if (lane == 0) parts[warp] = acc;
  __syncthreads();
  if (lane == 0 && sub == 0 && row < rows) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < WPR; ++k) total += parts[warp + k];
    store_out(y + row, ACC ? total + load_out(y + row) : total);
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
    block_bytes = model::kquant::kQ6KResidentBytes;
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
    block_bytes = model::kquant::kQ6KResidentBytes;
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

// 16-byte-granular XOR swizzle for the MMQ shared tiles. Tile rows are 256 B =
// exactly 64 banks x 4 B, so unswizzled, the 8 rows a fragment load touches
// all start on the SAME bank and every ld.shared serializes ~8-way -- the cost
// no prior falsification touched (stores are fewer, mma count was ruled out,
// and it is invisible to DRAM counters). XOR-ing bits 4..6 of the column with
// the low row bits rotates each row's chunks across banks at zero memory cost
// (llama.cpp buys the same property by padding its sram stride to %8==4).
// Bits 0..3 are untouched, so 4- and 16-byte alignment survive.
__device__ __forceinline__ int mmq_swz(int r, int c) { return c ^ ((r & 7) << 4); }

// Same idea for the Q6_K scale tile, whose rows are 64 B = 16 words: the four
// Bsc loads in the mma epilogue come from rows nbase + 2t (+1), i.e. a 32-word
// stride across the thread quad -- same bank four ways. XOR-ing the float
// index with row bits 1..2 spreads the quad across four bank groups; rows na
// (even) and na+1 share an offset, so a thread's own pair stays adjacent.
__device__ __forceinline__ int mmq_swz_sc(int r, int j) { return j ^ (((r >> 1) & 3) << 2); }

// ldmatrix fragment loads: one instruction replaces four (x4) or two (x2)
// address computations + ld.shared, with the PTX-defined per-thread fragment
// distribution (thread = row lane>>2, bytes (lane&3)*4) exactly matching what
// the manual loads produced for mma.m16n8k32.
__device__ __forceinline__ void ldsm_x4(int& r0, int& r1, int& r2, int& r3, const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(int& r0, int& r1, const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}

// Tile-set layout for the Q6_K stream-K kernel, host-visible so the launcher
// can size the dynamic-shared launch. The 64-batch x 128-row shape is 56 KB --
// past the 48 KB static limit -- which is exactly why Q6_K used to decline
// batches above 32 and fall back to dequant+cuBLAS.
template <int NT, int BM>
struct Q6kSmemLayout {
  std::int8_t As[BM][256];
  std::int8_t Bs[(128 / BM) * NT * 8][256];
  float Bsc[(128 / BM) * NT * 8][16];
};

// Static shared when the layout fits 48 KB, dynamic (opted in via
// cudaFuncSetAttribute at launch) when it does not. A __shared__ variable
// inside the member function still has kernel lifetime.
template <bool DYN, typename T>
struct MmqSmem;
template <typename T>
struct MmqSmem<false, T> {
  __device__ __forceinline__ T* get() {
    __shared__ __align__(16) T s;
    return &s;
  }
};
template <typename T>
struct MmqSmem<true, T> {
  __device__ __forceinline__ T* get() {
    extern __shared__ __align__(16) char mmq_dyn_smem[];
    return reinterpret_cast<T*>(mmq_dyn_smem);
  }
};
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

// Double-buffered Q4_K MMQ: cp.async stages the next K-tile while the current
// one is being multiplied.
//
// The previous version alternated [stage | compute] with a barrier between, so
// every warp waited out a full tile load before touching the tensor cores. That
// is why it moved 6.7x fewer bytes than expand-and-cuBLAS and still only matched
// it: ~190 GB/s effective against the ~590 GB/s the same engine reaches on fp16.
//
// cp.async can only copy bytes, which forces the useful change: the nibble
// unpack moves out of staging and into the fragment build. Shared now holds the
// RAW super-block -- 128 bytes of qs per row instead of 256 unpacked -- and the
// b-fragment does `word & 0x0F0F0F0F` on the way to the mma. That halves the
// weight tile as a side effect, which is what makes double buffering fit in
// static shared memory.
//
// M tile is 32 here, not 64: two tiles in flight double the activation storage,
// and batches above 32 keep the single-buffered kernel.
// SPLITK divides the super-block loop across gridDim.y. The loop is the whole
// cost: a block walks every super-block serially, each stage a
// __pipeline_wait_prior and a __syncthreads, and the microbenchmark shows the
// kernel does not care how many bytes it reads -- w13 has 3.5x wq's weight and
// takes the same time, both at nsb=16, i.e. ~1.8 us per iteration at B=8. More
// blocks does not help either (w13's 448 are no faster per call than wk's 16),
// because the chain is inside a block. So shorten the chain and multiply the
// blocks. Partials go to a float scratch via atomicAdd and a finalize pass folds
// them into y.
//
// Note this is NOT the split-K that failed for the matvec: there a 4096-wide row
// was 16 super-blocks and one block already consumed all 16 across its warps, so
// chunking only idled warps. Here one block walks all 16 in sequence.
//
// M2 serves batch 33..64 in ONE pass over the staged weights: two 32-row batch
// halves share every Bq fragment (the mma loop keeps the weight fragment in
// registers and issues it against both halves), so the weight is read once
// where the pre-M2 alternative was re-dequantizing the entire model to fp16
// every step -- the audit's 11.1 ms/step of dequant at B=64. M2 forces the
// single-buffered activation panel (both halves staged = 16 KB) and NT=2.
//
// FALSIFIED HERE (2026-08-12, #6 for this kernel family): folding the 6-bit
// scale headers into a pre-decoded float2 (d*sc, dmin*m) shared tile at stage
// time, llama.cpp-style, replacing the per-(group, n-tile) in-loop decode.
// Bit-identical, gated, and SLOWER: 1508.6 -> 1426.9 tok/s at B=32 and 621.6
// -> 539.7 at B=8, interleaved same-binary-pair medians. The in-loop decode
// pipelines behind the mma for free; a synchronous fold serializes at the
// tile boundary. Same verdict as the earlier decode-into-shared attempt --
// scale handling is NOT this kernel's bottleneck, stop trying to hoist it.
template <int NT, bool SPLITK, bool M2 = false>
__global__ __launch_bounds__(256) void kquant_mmq_q4k_async_kernel(
    const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
    const float* __restrict__ as, const float* __restrict__ gsum, __half* __restrict__ y,
    float* __restrict__ partial, int rows, int cols, int batch, int ldy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  (void)w; (void)xq; (void)as; (void)gsum; (void)y; (void)rows; (void)cols; (void)batch; (void)ldy;
#else
  constexpr int kBM = 32;
  constexpr int kMH = M2 ? 2 : 1;    // batch halves per pass
  constexpr int kBMT = 32 * kMH;     // activation rows staged
  // Warps split across the M and N tiles, and the split has to follow kBM: with
  // 16 rows per m-tile, kBM=32 leaves only two m-tiles, so the other warps must
  // go to N. Keeping the 64-row mapping here left warps 4..7 with every row out
  // of range -- half the block computing nothing, which is exactly the factor
  // between this kernel's 33% theoretical and 17% achieved occupancy.
  constexpr int kMT = kBM / 16;   // m-tiles per block
  constexpr int kNH = 8 / kMT;    // warps left for the n dimension
  constexpr int kBN = kNH * NT * 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3, mt = warp % kMT, nhalf = warp / kMT;
  const int blockNrow = blockIdx.x * kBN;
  const int Kg = cols >> 5;
  const int nsb = cols >> 8;
  const std::size_t row_bytes =
      static_cast<std::size_t>(nsb) * model::kquant::kQ4KBlockBytes;

  // At NT>=4 (kBN=128, the audit's restaging fix) the double-buffered tile set
  // no longer fits 48 KB static shared, so the ACTIVATION panel drops to a
  // single buffer staged with plain loads: it is the small, L2-hot side (8 KB
  // a tile against Bq's 16), and the whole point of the wide tile is that this
  // panel is re-read less often per weight byte. Weights and headers keep the
  // cp.async double buffer either way.
  constexpr int kAsBufs = (NT >= 4 || M2) ? 1 : 2;
  __shared__ __align__(16) std::int8_t As[kAsBufs][kBMT][256];
  __shared__ __align__(16) std::uint8_t Bq[2][kBN][128];  // raw qs, unpacked at use
  __shared__ __align__(16) std::uint8_t Hd[2][kBN][16];   // raw header: d, dmin, 12 scale bytes

  // Synchronous activation stage for the single-buffer shape. Safe without an
  // extra barrier: the previous iteration's trailing __syncthreads means every
  // warp is done computing from As[0] before any warp overwrites it.
  const auto stage_a_sync = [&](int sb) {
#pragma unroll
    for (int i = 0; i < (kBMT * 16) / 256; ++i) {
      const int u = tid + i * 256;
      const int r = u >> 4, c = (u & 15) << 4;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < batch) {
        v = *reinterpret_cast<const uint4*>(xq + static_cast<std::size_t>(r) * cols + (sb << 8) +
                                            c);
      }
      *reinterpret_cast<uint4*>(&As[0][r][mmq_swz(r, c)]) = v;
    }
  };

  // Issue one K-tile's copies into buffer `buf`.
  const auto stage = [&](int sb, int buf) {
    // Activations: 32 rows x 256 B = 512 uint4, two per thread.
    if (kAsBufs == 2) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int u = tid + i * 256;
        const int r = u >> 4, c = (u & 15) << 4;
        if (r < batch) {
          __pipeline_memcpy_async(&As[buf % kAsBufs][r][mmq_swz(r, c)],
                                  xq + static_cast<std::size_t>(r) * cols + (sb << 8) + c, 16);
        }
      }
    }
    // Weights: kBN rows x 128 B of qs, 8 uint4 per row. Swizzled like As --
    // Bq rows are 128 B (exactly one bank wrap), so unswizzled fragment loads
    // serialize the same way.
#pragma unroll
    for (int i = 0; i < (kBN + 31) / 32; ++i) {
      const int u = tid + i * 256;
      const int r = u >> 3, c = (u & 7) << 4;
      const int wrow = blockNrow + r;
      if (r < kBN && wrow < rows) {
        const std::uint8_t* p = w + static_cast<std::size_t>(wrow) * row_bytes +
                                static_cast<std::size_t>(sb) * model::kquant::kQ4KBlockBytes;
        __pipeline_memcpy_async(&Bq[buf][r][mmq_swz(r, c)], p + 16 + c, 16);
      }
    }
    // Headers: kBN rows x 16 B.
    if (tid < kBN) {
      const int wrow = blockNrow + tid;
      if (wrow < rows) {
        __pipeline_memcpy_async(&Hd[buf][tid][0],
                                w + static_cast<std::size_t>(wrow) * row_bytes +
                                    static_cast<std::size_t>(sb) * model::kquant::kQ4KBlockBytes,
                                16);
      }
    }
    __pipeline_commit();
  };

  float facc[kMH][NT][4];
#pragma unroll
  for (int m = 0; m < kMH; ++m)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) facc[m][i][j] = 0.0f;

  int sb_lo = 0, sb_hi = nsb;
  if (SPLITK) {
    const int chunk = (nsb + static_cast<int>(gridDim.y) - 1) / static_cast<int>(gridDim.y);
    sb_lo = static_cast<int>(blockIdx.y) * chunk;
    sb_hi = min(sb_lo + chunk, nsb);
    if (sb_lo >= sb_hi) return;
  }

  stage(sb_lo, 0);
  for (int sb = sb_lo; sb < sb_hi; ++sb) {
    const int buf = (sb - sb_lo) & 1;
    if (sb + 1 < sb_hi) stage(sb + 1, buf ^ 1);
    if (kAsBufs == 1) stage_a_sync(sb);
    // Wait only for this tile; the next one keeps flying.
    __pipeline_wait_prior(sb + 1 < sb_hi ? 1 : 0);
    __syncthreads();
    const int abuf = kAsBufs == 1 ? 0 : buf;

    const int arow = 16 * mt;

#pragma unroll
    for (int gi = 0; gi < 8; ++gi) {
      const int kg = (sb << 3) + gi, co = gi << 5;
      // A fragments per batch half: k = co..co+31, split 0..15 / 16..31 across
      // a0a1 / a2a3. Loaded up front so the nt loop can replay each staged
      // weight fragment against both halves without touching shared again.
      int a0[kMH], a1[kMH], a2[kMH], a3[kMH];
      float as_a[kMH], as_b[kMH], gs_a[kMH], gs_b[kMH];
#pragma unroll
      for (int mh = 0; mh < kMH; ++mh) {
        const int m_a = 32 * mh + arow + g, m_b = 32 * mh + arow + g + 8;
        const bool ok_a = m_a < batch, ok_b = m_b < batch;
        // Unguarded ldmatrix: rows past the batch hold zeros (sync staging) or
        // stale bytes (cp.async staging), and either way the products below
        // are zeroed by as/gs = 0 -- an int8 mma cannot produce NaN.
        {
          const int ar = 32 * mh + arow + (lane & 15);
          const int ac = co + ((lane >> 4) << 4);
          ldsm_x4(a0[mh], a1[mh], a2[mh], a3[mh], &As[abuf][ar][mmq_swz(ar, ac)]);
        }
        as_a[mh] = ok_a ? as[static_cast<std::size_t>(m_a) * Kg + kg] : 0.0f;
        as_b[mh] = ok_b ? as[static_cast<std::size_t>(m_b) * Kg + kg] : 0.0f;
        gs_a[mh] = ok_a ? gsum[static_cast<std::size_t>(m_a) * Kg + kg] : 0.0f;
        gs_b[mh] = ok_b ? gsum[static_cast<std::size_t>(m_b) * Kg + kg] : 0.0f;
      }

      // Column co..co+31 lives in qs[j*32 + l], j = co/64, low nibble when
      // (co%64) < 32. Four consecutive columns are four consecutive bytes.
      const int j = co >> 6;
      const bool hi_nib = ((co >> 5) & 1) != 0;

#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int nbase = nhalf * (NT * 8) + nt * 8;
        int wi0, wi1;
        {
          const int br = nbase + (lane & 7);
          const int bc = (j << 5) + ((lane & 8) << 1);
          ldsm_x2(wi0, wi1, &Bq[buf][br][mmq_swz(br, bc)]);
        }
        const std::uint32_t w0 = static_cast<std::uint32_t>(wi0);
        const std::uint32_t w1 = static_cast<std::uint32_t>(wi1);
        const int b0 = static_cast<int>(hi_nib ? ((w0 >> 4) & 0x0F0F0F0Fu) : (w0 & 0x0F0F0F0Fu));
        const int b1 = static_cast<int>(hi_nib ? ((w1 >> 4) & 0x0F0F0F0Fu) : (w1 & 0x0F0F0F0Fu));
        // Scales for the two output rows this thread accumulates -- shared by
        // both batch halves, so decoded once per n-tile.
        const int na = nbase + 2 * t, nb = nbase + 2 * t + 1;
        const uint4 ha = *reinterpret_cast<const uint4*>(&Hd[buf][na][0]);
        const uint4 hb = *reinterpret_cast<const uint4*>(&Hd[buf][nb][0]);
        float sca, ma, scb, mb;
        scale_min_from_hdr(gi, ha.y, ha.z, ha.w, &sca, &ma);
        scale_min_from_hdr(gi, hb.y, hb.z, hb.w, &scb, &mb);
        const float da = half_bits_to_float(static_cast<std::uint16_t>(ha.x & 0xFFFFu));
        const float dmina = half_bits_to_float(static_cast<std::uint16_t>(ha.x >> 16));
        const float db = half_bits_to_float(static_cast<std::uint16_t>(hb.x & 0xFFFFu));
        const float dminb = half_bits_to_float(static_cast<std::uint16_t>(hb.x >> 16));
#pragma unroll
        for (int mh = 0; mh < kMH; ++mh) {
          int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
              "{%8,%9}, {%0,%1,%2,%3};\n"
              : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
              : "r"(a0[mh]), "r"(a1[mh]), "r"(a2[mh]), "r"(a3[mh]), "r"(b0), "r"(b1));
          facc[mh][nt][0] += static_cast<float>(c0) * as_a[mh] * (da * sca) - (dmina * ma) * gs_a[mh];
          facc[mh][nt][1] += static_cast<float>(c1) * as_a[mh] * (db * scb) - (dminb * mb) * gs_a[mh];
          facc[mh][nt][2] += static_cast<float>(c2) * as_b[mh] * (da * sca) - (dmina * ma) * gs_b[mh];
          facc[mh][nt][3] += static_cast<float>(c3) * as_b[mh] * (db * scb) - (dminb * mb) * gs_b[mh];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int mh = 0; mh < kMH; ++mh) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int nbase = nhalf * (NT * 8) + nt * 8;
      const int rr[4] = {g, g, g + 8, g + 8};
      const int cc[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
#pragma unroll
      for (int en = 0; en < 4; ++en) {
        const int lm = 32 * mh + 16 * mt + rr[en], gn = blockNrow + nbase + cc[en];
        if (lm < batch && gn < rows) {
          if (SPLITK) {
            atomicAdd(&partial[static_cast<std::size_t>(lm) * rows + gn], facc[mh][nt][en]);
          } else {
            y[static_cast<std::size_t>(lm) * static_cast<std::size_t>(ldy) + gn] =
                __float2half(facc[mh][nt][en]);
          }
        }
      }
    }
  }
#endif
}

// Folds split-K partials into y and leaves the scratch zeroed for the next call.
__global__ void kquant_mmq_finalize_kernel(float* __restrict__ partial, __half* __restrict__ y,
                                           int rows, int batch, int ldy) {
  const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (n >= rows) return;
  for (int m = 0; m < batch; ++m) {
    const std::size_t i = static_cast<std::size_t>(m) * rows + n;
    y[static_cast<std::size_t>(m) * static_cast<std::size_t>(ldy) + n] = __float2half(partial[i]);
    partial[i] = 0.0f;
  }
}

// Q4_K on int8 tensor cores. Same activation preparation as the dp4a path; the
// difference is the inner product runs on mma instead of scalar FMAs. sm_80+
// only -- the caller checks compute capability.
// FALSIFIED TWICE MORE, and the second one closes off the compute theories.
//
// (a) llama.cpp does NOT use cp.async for Q6_K either. Its load_tiles_q6_K uses
// plain loads and unpacks 6-bit into int8 in shared at staging time -- the same
// shape as the kernel below, down to the __vsubss4(x, 0x20202020) centering. So
// the 210-byte alignment wall is not what separates the two, and the aligned
// repack is not the lever it looked like.
//
// (b) The mma decomposition is not the bottleneck either. llama.cpp issues its
// two mma over DISJOINT k-ranges with full b fragments and scales each
// accumulator afterwards, where this kernel issues two over the same range with
// half of b zeroed -- nominally a 2x waste. Rewriting it to two m16n8k16 over
// disjoint ranges is numerically exact and bought NOTHING: w2 34.0 -> 34.5 us,
// wv 28.3 -> 35.9, i.e. worse. Halving the tensor-core work changed nothing, so
// this kernel is not mma-bound. Reverted.
//
// What that leaves is the DRAM access pattern, and the staging below is the
// suspect. Each thread takes a contiguous 16-byte chunk at offset 16*t and reads
// it as eight uint16 -- two-byte loads are forced, because a 210-byte block
// stride leaves p only 2-byte aligned. At any one instruction the warp is then
// reading 16*t + 2*k: two bytes per thread at a sixteen-byte stride, so a 32-byte
// sector serves two lanes and roughly seven eighths of every fetch is discarded.
//
// That is invisible from L2 and ruinous from DRAM, which is exactly the signature
// measured: 0.72 us/MB here against the Q4_K MMQ's 0.85 when the weight is
// cache-resident, but ~138 GB/s against its ~495 in the engine. llama.cpp's
// equivalent runs ~33.9 us a call where this one runs ~102.7.
//
// FALSIFIED as well. Staging the raw block into shared first with consecutive
// lanes on consecutive halfwords -- perfectly coalesced global reads, unpack
// moved to a second pass off shared -- is numerically identical and SLOWER:
// defaults 1507 -> 1497 tok/s and Q6_K-MMQ 1462 -> 1427 at B=32.
//
// The reason is that the strided pattern was never costing DRAM bandwidth. At
// step k the warp touches 16*t + 2*k, and the sectors it pulls are the same ones
// steps k+1..k+7 need, so L1 absorbs the reuse and total DRAM traffic is
// unchanged. Only the instruction count differs, and the extra shared buffer plus
// its __syncthreads cost more than that. Strided-within-a-warp is a bandwidth
// problem only when the lines are not revisited.
//
// STREAM-K IS NOW TRIED (the last structural difference llama.cpp's profile
// showed): the decomposition below plus its fixup kernel, exactly the
// mul_mat_q / mul_mat_q_stream_k_fixup shape. Engine kernel-time per step at
// B=32 (nsys, graph-trace=node, per shape):
//   w2 (4096x14336): dequant+cuBLAS 135.1 us -> stream-K NT=4 118.7 (incl 8.3
//                    fixup); NT=2 was ~142, so the tile-shape verdict below
//                    flipped under this launch.
//   wv (1024x4096):  dequant+cuBLAS ~13 us -> stream-K NT=4 28.9, of which
//                    15.5 is fixup -- 16 tiles x 16 super-blocks split ~16 ways
//                    means the reconciliation costs more than the matmul.
//   LM head (128256): 1245.6 -> 956.2 us from NT=4 alone (data-parallel branch,
//                    no fixup); this one runs force_q6k in EVERY batched
//                    config, so it is a win independent of mmq_q6k.
// Net: layer weights at parity with dequant-and-cuBLAS (w2's win pays for wv's
// fixup), ~1% of a batched step won on the head. The remaining per-call gap to
// llama.cpp's 33.9 us is no longer decomposition: with the launch structure
// matched, what differs is the kernel body -- the staging unpack writes Bs one
// BYTE at a time (32 shared byte-stores per thread per super-block) where
// llama.cpp packs int words, and that is the one cost falsification has not
// touched (the coalescing experiment restaged the GLOBAL reads, not the shared
// stores).
//
// Q6_K CANNOT USE THE ASYNC STAGING
// STYLE THAT MAKES THE Q4_K MMQ FAST. cp.async requires natural 4/8/16-byte
// alignment on its source, and a Q6_K block is 210 bytes -- not a multiple of 4,
// let alone 16 -- so p shifts alignment every block. Q4_K's 144-byte blocks are a
// multiple of 16, which is exactly why its async kernel works. Writing the async
// version anyway produced worst_rel 1.0 and out-of-bounds writes that corrupted
// unrelated Q4_K results; the same alignment wall forced uint16 loads in the
// staging below.
//
// That matters because async staging is worth 22.8% on Q4_K (forcing it to the
// scalar-staging kernel costs 1259 -> 972 tok/s), and it is the largest known
// term in Q6_K MMQ being ~2.5x Q4_K per byte. So closing that gap needs Q6_K
// REPACKED to an aligned stride (224 or 256 bytes a block), which is a converter
// change, not a kernel one -- and note the reason is alignment, not the per-16
// scale grouping that an earlier note blamed.
//
// Q6_K on the tensor cores. STILL OFF BY DEFAULT for layer weights, but the
// story moved: with stream-K + NT=4 it is break-even against dequant-and-cuBLAS
// at B=32 (interleaved medians 1613.9 vs 1613.2 tok/s) and -4.2% at B=8 (was
// -9.5%). The force_q6k LM head keeps using this path in every batched config
// and got 19% faster per call from NT=4 alone. CPI_KQUANT_MMQ_Q6K=1 turns the
// layer path on.
//
// Q6_K is a quarter of a Q4_K_M by bytes -- ffn_down
// for most layers plus wv -- and it was the reason a batched step still expanded
// weights to fp16: with no MMQ path it fell to dequant-and-cuBLAS, which nsys put
// at 15.3% of batched time even with MMQ forced on everywhere else.
//
// Two things make this simpler than the Q4_K kernel rather than harder:
//
//   - Q6_K values are q-32 with q in 0..63, so they land in -32..31 and fit int8
//     directly. There is no -dmin*m term, so no group sums and no correction --
//     the mma result is the dot product already.
//   - The scale for element e is exactly scales[e>>4], once the layout is worked
//     through: the per-16 index the reference unpack builds from n, h and l0
//     equals e>>4 for the low half and e>>4 for the high half too, since off_hi
//     is off_lo + 64.
//
// The one real obstacle is that a scale per 16 does not line up with
// mma.m16n8k32, which accumulates 32 k at once. So each 32-k step issues two
// mma, one per 16-k half, with the other half of the b fragment zeroed. That
// doubles the mma count at unchanged memory traffic -- an acceptable trade for a
// kernel measured at ~426 GB/s, i.e. bound by the weight read rather than the
// tensor cores.
// One (row-tile, k-range) piece of the Q6_K MMQ: stages the activation and
// weight tiles for super-blocks [kb0_start, kb0_stop) of row tile `tile_it`
// and runs them through the mma. Factored out of the kernel so the stream-K
// launch below can walk several pieces per block. MODE picks the epilogue:
//   0  write the accumulators to y. Used when the piece reaches the tile's
//      k-END -- under stream-K exactly one piece per tile does, and the
//      earlier pieces arrive through the fixup pass.
//   1  atomicAdd into the batch x rows float scratch (the pre-stream-K
//      split-K, kept for CPI_KQUANT_MMQ_STREAMK=0).
//   2  store raw float partials to this block's slot of the stream-K fixup
//      buffer: aux[blockIdx.x][m][n_local].
//
// BM is the M (batch) tile: 16 rows per m-tile warp group, so BM=64 splits the
// 8 warps 4x2 over MxN and BM=32 splits them 2x4. The BM=32 shape exists
// because the audit's dominant matmul cost was ACTIVATION RESTAGING: every
// block re-stages the whole batch panel, so tile-N width is the divisor on
// that traffic, and shrinking the M tile to what a batch<=32 call actually
// uses buys the shared memory to double N: BM=32/NT=4 is a 32x128 tile in the
// same 48 KB that BM=64/NT=4 spends on 64x64.
template <int NT, int BM, int MODE>
__device__ __forceinline__ void kquant_mmq_q6k_process_tile(
    std::int8_t (&As)[BM][256], std::int8_t (&Bs)[(128 / BM) * NT * 8][256],
    float (&Bsc)[(128 / BM) * NT * 8][16], const std::uint8_t* __restrict__ w,
    const std::int8_t* __restrict__ xq, const float* __restrict__ as, __half* __restrict__ y,
    float* __restrict__ aux, int rows, int cols, int batch, int ldy, int tile_it, int kb0_start,
    int kb0_stop) {
  constexpr int kMT = BM / 16;      // m-tiles, one warp group each
  constexpr int kNH = 8 / kMT;      // warp groups across N
  constexpr int kMmqBN = kNH * NT * 8;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3, mt = warp % kMT, nhalf = warp / kMT;
  const int blockNrow = tile_it * kMmqBN;
  const int Kg = cols >> 5;
  const int blocks_per_row = cols >> 8;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks_per_row) * model::kquant::kQ6KResidentBytes;

  float facc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) facc[i][j] = 0.0f;

  for (int sb = kb0_start; sb < kb0_stop; ++sb) {
    const int k0 = sb << 8;

#pragma unroll
    for (int i = 0; i < (BM * 16) / 256; ++i) {
      const int u = tid + i * 256;
      const int r = u >> 4, c = (u & 15) << 4;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < batch) {
        v = *reinterpret_cast<const uint4*>(xq + static_cast<std::size_t>(r) * cols + k0 + c);
      }
      *reinterpret_cast<uint4*>(&As[r][mmq_swz(r, c)]) = v;
    }

    // Weights: eight threads a row, each unpacking a 16-byte run of ql into
    // sixteen low and sixteen high values. A 16-byte run never crosses the
    // 32-byte or 64-byte boundaries that move l0, h and n, so those are constant
    // across the run and the matching qh bytes are contiguous too.
    //
    // The loop, not `if (r < kMmqBN)`: 256 threads cover 32 rows a pass, so
    // NT=4's 64-row tile needs two. The old guard silently staged HALF the
    // tile at NT=4 -- rows 32..63 multiplied stale shared memory -- and the
    // rows<=32 unit-test slice masked it via the gn<rows store guard, which is
    // how ebbe5f2 recorded NT=4 as "numerically correct". --batched-check is
    // what caught it.
#pragma unroll
    for (int u = tid; u < kMmqBN * 8; u += 256) {
      {
        const int r = u >> 3, c = (u & 7) << 4;
        const int wrow = blockNrow + r;
        const std::uint8_t* p =
            (wrow < rows) ? w + static_cast<std::size_t>(wrow) * row_bytes +
                                static_cast<std::size_t>(sb) * model::kquant::kQ6KResidentBytes
                          : nullptr;
        const int n = c >> 6, h = (c >> 5) & 1, l0 = c & 31;
        const int off_lo = (n << 7) + (h << 5) + l0;
        const int sl = 2 * h, sh = 4 + 2 * h;
        // Read the run as uint16 pairs -- a 210-byte block leaves p only 2-byte
        // aligned so wider loads are out (llama.cpp pays the same get_int_b2
        // tax) -- but UNPACK IN WORDS, NOT BYTES. The old loop computed each
        // value separately and issued a byte-wide shared store per weight: 32
        // st.b8 and ~96 per-byte ALU ops per 16-byte run. The audit's head A/B
        // put this kernel at 379 GB/s against llama.cpp's 1317 on identical
        // bytes with staging as the only untested difference; their
        // load_tiles_q6_K assembles four recentered values per int in
        // registers (__vsubss4) and stores words. Same here: the whole-word
        // shifts keep each byte lane's 2 qh bits and 4 ql bits in place (the
        // cross-byte spill of >> lands above the per-lane mask), so this is
        // arithmetic-identical to the byte loop, at 8 st.b32 and ~16 word ops.
        const std::uint16_t* qlp = reinterpret_cast<const std::uint16_t*>(p + c);
        const std::uint16_t* qhp =
            reinterpret_cast<const std::uint16_t*>(p + 128 + (n << 5) + l0);
#pragma unroll
        for (int g2 = 0; g2 < 4; ++g2) {
          std::uint32_t ql32 = 0u;
          std::uint32_t qh32 = 0u;
          if (p != nullptr) {
            ql32 = static_cast<std::uint32_t>(qlp[2 * g2]) |
                   (static_cast<std::uint32_t>(qlp[2 * g2 + 1]) << 16);
            qh32 = static_cast<std::uint32_t>(qhp[2 * g2]) |
                   (static_cast<std::uint32_t>(qhp[2 * g2 + 1]) << 16);
          }
          const std::uint32_t lo_u =
              (ql32 & 0x0F0F0F0Fu) | (((qh32 >> sl) & 0x03030303u) << 4);
          const std::uint32_t hi_u =
              ((ql32 >> 4) & 0x0F0F0F0Fu) | (((qh32 >> sh) & 0x03030303u) << 4);
          const int lo = __vsubss4(static_cast<int>(lo_u), 0x20202020);
          const int hi = __vsubss4(static_cast<int>(hi_u), 0x20202020);
          *reinterpret_cast<int*>(&Bs[r][mmq_swz(r, off_lo + 4 * g2)]) = lo;
          *reinterpret_cast<int*>(&Bs[r][mmq_swz(r, off_lo + 64 + 4 * g2)]) = hi;
        }
        // Two of the sixteen scale groups per pass.
        float d = 0.0f;
        if (p != nullptr) {
          std::uint16_t dbits;
          memcpy(&dbits, p + 208, 2);
          d = half_bits_to_float(dbits);
        }
        const int j0 = (u & 7) << 1;
        for (int j = j0; j < j0 + 2; ++j) {
          Bsc[r][mmq_swz_sc(r, j)] =
              (p != nullptr)
                  ? d * static_cast<float>(*reinterpret_cast<const std::int8_t*>(p + 192 + j))
                  : 0.0f;
        }
      }
    }
    __syncthreads();

    const int arow = 16 * mt;
    const int m_a = arow + g, m_b = arow + g + 8;
    const bool ok_a = m_a < batch, ok_b = m_b < batch;
#pragma unroll
    for (int gi = 0; gi < 8; ++gi) {
      const int kg = (sb << 3) + gi, co = gi << 5;
      // One ldmatrix.x4 replaces the four a-fragment loads: lanes 0-15 address
      // rows arow..arow+15 at k-chunk co, lanes 16-31 the same rows at co+16;
      // the returned per-thread fragments are exactly the old a0..a3.
      int a0, a1, a2, a3;
      {
        const int ar = arow + (lane & 15);
        const int ac = co + ((lane >> 4) << 4);
        ldsm_x4(a0, a1, a2, a3, &As[ar][mmq_swz(ar, ac)]);
      }
      const float as_a = ok_a ? as[static_cast<std::size_t>(m_a) * Kg + kg] : 0.0f;
      const float as_b = ok_b ? as[static_cast<std::size_t>(m_b) * Kg + kg] : 0.0f;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int nbase = nhalf * (NT * 8) + nt * 8;
        int b0, b1;
        {
          const int br = nbase + (lane & 7);
          const int bc = co + ((lane & 8) << 1);
          ldsm_x2(b0, b1, &Bs[br][mmq_swz(br, bc)]);
        }
        // Halves of the 32-k step carry different weight scales, so they cannot
        // share an accumulator. Zeroing the other half of b is what splits them.
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(0));
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(0), "r"(b1));
        const int na = nbase + 2 * t, nb = nbase + 2 * t + 1;
        // na is even and nb = na + 1, so both share one swizzle offset.
        const int jl = mmq_swz_sc(na, gi << 1), jh = jl + 1;
        facc[nt][0] += as_a * (static_cast<float>(c0) * Bsc[na][jl] +
                               static_cast<float>(d0) * Bsc[na][jh]);
        facc[nt][1] += as_a * (static_cast<float>(c1) * Bsc[nb][jl] +
                               static_cast<float>(d1) * Bsc[nb][jh]);
        facc[nt][2] += as_b * (static_cast<float>(c2) * Bsc[na][jl] +
                               static_cast<float>(d2) * Bsc[na][jh]);
        facc[nt][3] += as_b * (static_cast<float>(c3) * Bsc[nb][jl] +
                               static_cast<float>(d3) * Bsc[nb][jh]);
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
        if (MODE == 1) {
          atomicAdd(&aux[static_cast<std::size_t>(lm) * rows + gn], facc[nt][en]);
        } else if (MODE == 2) {
          aux[static_cast<std::size_t>(blockIdx.x) * (kMmqBN * BM) +
              static_cast<std::size_t>(lm) * kMmqBN + (nbase + cc[en])] = facc[nt][en];
        } else {
          y[static_cast<std::size_t>(lm) * static_cast<std::size_t>(ldy) + gn] =
              __float2half(facc[nt][en]);
        }
      }
    }
  }
}

// Q6_K MMQ, one row tile per blockIdx.x. SPLITK chunks the super-block loop
// across blockIdx.y with atomicAdd partials -- the pre-stream-K split, kept
// behind CPI_KQUANT_MMQ_STREAMK=0.
template <int NT, bool SPLITK>
__global__ __launch_bounds__(256) void kquant_mmq_q6k_kernel(
    const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
    const float* __restrict__ as, __half* __restrict__ y, float* __restrict__ partial, int rows,
    int cols, int batch, int ldy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  (void)w; (void)xq; (void)as; (void)y; (void)partial; (void)rows; (void)cols; (void)batch;
  (void)ldy;
#else
  constexpr int kMmqBN = 2 * NT * 8;
  __shared__ __align__(16) std::int8_t As[kMmqBM][256];
  __shared__ __align__(16) std::int8_t Bs[kMmqBN][256];
  __shared__ float Bsc[kMmqBN][16];  // d*scale per 16 weights

  const int blocks_per_row = cols >> 8;
  int sb_lo = 0, sb_hi = blocks_per_row;
  if (SPLITK) {
    // Same lever that took 26% off the Q4_K kernel: the super-block loop is
    // serial per block and there are too few blocks to hide it.
    const int chunk = (blocks_per_row + static_cast<int>(gridDim.y) - 1) / static_cast<int>(gridDim.y);
    sb_lo = static_cast<int>(blockIdx.y) * chunk;
    sb_hi = min(sb_lo + chunk, blocks_per_row);
    if (sb_lo >= sb_hi) return;
  }
  kquant_mmq_q6k_process_tile<NT, kMmqBM, SPLITK ? 1 : 0>(As, Bs, Bsc, w, xq, as, y, partial, rows,
                                                          cols, batch, ldy,
                                                          static_cast<int>(blockIdx.x), sb_lo,
                                                          sb_hi);
#endif
}

// Stream-K for the Q6_K MMQ -- the decomposition llama.cpp's mul_mat_q uses,
// which its profile shows beating both CPI paths on this exact shape. The work
// space is the flat sequence of (row-tile, super-block) units with the tile
// index varying slowest; each block of a FIXED grid takes the contiguous slice
// [kbc, kbc_stop). The grid no longer has to divide the tile count, so the
// last-wave imbalance and the serial per-block super-block chain both
// disappear: every block finishes within one super-block of its peers. A piece
// that reaches its tile's k-end writes y directly (there is exactly one such
// piece per tile); a block's trailing partial piece stores raw float partials
// to its tmp slot, and kquant_mmq_q6k_streamk_fixup_kernel folds those into y.
// Unlike the atomicAdd split, the result is deterministic: the fixup adds
// partials in a fixed order.
template <int NT, int BM>
__global__ __launch_bounds__(256) void kquant_mmq_q6k_streamk_kernel(
    const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
    const float* __restrict__ as, __half* __restrict__ y, float* __restrict__ tmp, int rows,
    int cols, int batch, int ldy) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  (void)w; (void)xq; (void)as; (void)y; (void)tmp; (void)rows; (void)cols; (void)batch; (void)ldy;
#else
  constexpr int kMmqBN = (128 / BM) * NT * 8;
  using Smem = Q6kSmemLayout<NT, BM>;
  constexpr bool kDyn = sizeof(Smem) > 48 * 1024;
  Smem* sm = MmqSmem<kDyn, Smem>().get();
  auto& As = sm->As;
  auto& Bs = sm->Bs;
  auto& Bsc = sm->Bsc;  // d*scale per 16 weights

  const int nsb = cols >> 8;
  const int nty = (rows + kMmqBN - 1) / kMmqBN;
  const std::int64_t total = static_cast<std::int64_t>(nty) * nsb;
  std::int64_t kbc = static_cast<std::int64_t>(blockIdx.x) * total / gridDim.x;
  const std::int64_t kbc_stop = static_cast<std::int64_t>(blockIdx.x + 1) * total / gridDim.x;

  // One super-block is one work unit (llama.cpp's MMQ_ITER_K == 256 == the
  // Q6_K super-block), so no rounding of kbc to iteration boundaries is needed.
  int kb0_start = static_cast<int>(kbc % nsb);
  std::int64_t want = kb0_start + (kbc_stop - kbc);
  int kb0_stop = want < nsb ? static_cast<int>(want) : nsb;
  while (kbc < kbc_stop && kb0_stop == nsb) {
    kquant_mmq_q6k_process_tile<NT, BM, 0>(As, Bs, Bsc, w, xq, as, y, tmp, rows, cols, batch, ldy,
                                           static_cast<int>(kbc / nsb), kb0_start, kb0_stop);
    kbc += nsb - kb0_start;  // advance to the next tile boundary
    kb0_start = 0;
    want = kbc_stop - kbc;
    kb0_stop = want < nsb ? static_cast<int>(want) : nsb;
  }
  if (kbc >= kbc_stop) return;
  // Trailing piece that ends mid-tile: park it in the fixup buffer. Writing y
  // here would race with the block that owns this tile's k-end.
  kquant_mmq_q6k_process_tile<NT, BM, 2>(As, Bs, Bsc, w, xq, as, y, tmp, rows, cols, batch, ldy,
                                         static_cast<int>(kbc / nsb), kb0_start, kb0_stop);
#endif
}

// Reconciles stream-K partial tiles. Fixup block b mirrors MMQ block b: if
// that block wrote a tile's k-end to y after starting mid-tile, the missing
// k-prefix sits in the tmp slots of the blocks just before it, so walk
// backwards summing them and fold the total into y. Exactly one fixup block
// touches a given tile, so plain loads and stores suffice; blocks with nothing
// to reconcile exit in a few instructions. Must be launched with the SAME grid
// size as the stream-K kernel -- the kbc arithmetic has to reproduce its
// partitioning exactly.
template <int NT, int BM>
__global__ void kquant_mmq_q6k_streamk_fixup_kernel(__half* __restrict__ y,
                                                    const float* __restrict__ tmp, int rows,
                                                    int cols, int batch, int ldy) {
  constexpr int kMmqBN = (128 / BM) * NT * 8;
  constexpr int kElems = (kMmqBN * BM) / 256;  // tile outputs per thread
  const int nsb = cols >> 8;
  const int nty = (rows + kMmqBN - 1) / kMmqBN;
  const std::int64_t total = static_cast<std::int64_t>(nty) * nsb;
  const int bidx0 = static_cast<int>(blockIdx.x);
  const std::int64_t kbc0 = static_cast<std::int64_t>(bidx0) * total / gridDim.x;
  const std::int64_t kbc0_stop = static_cast<std::int64_t>(bidx0 + 1) * total / gridDim.x;

  if (kbc0 == kbc0_stop) return;  // block had no work
  if (kbc0 % nsb == 0) return;    // block started at a tile boundary: its write was complete
  const std::int64_t it = kbc0 / nsb;
  // Block never reached its first tile's k-end, so it wrote tmp, not y; the
  // block that DID write that tile's end runs the reconciliation instead.
  if (it == kbc0_stop / nsb && kbc0_stop % nsb != 0) return;

  float sum[kElems];
#pragma unroll
  for (int e = 0; e < kElems; ++e) sum[e] = 0.0f;

  int bidx = bidx0 - 1;
  std::int64_t kbc_stop_w = kbc0;
  while (bidx >= 0) {
    const std::int64_t kbc = static_cast<std::int64_t>(bidx) * total / gridDim.x;
    if (kbc == kbc_stop_w) {  // empty block, keep walking
      --bidx;
      continue;
    }
    // Non-empty predecessor: its range ends at kbc_stop_w, mid-tile `it`, so
    // its trailing piece is a partial of this tile.
    const float* src = tmp + static_cast<std::size_t>(bidx) * (kMmqBN * BM);
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      sum[e] += src[static_cast<int>(threadIdx.x) + e * 256];
    }
    // If it covered the tile's beginning (started at the boundary or in an
    // earlier tile), the prefix is complete.
    if (kbc % nsb == 0 || kbc / nsb < it) break;
    kbc_stop_w = kbc;
    --bidx;
  }

#pragma unroll
  for (int e = 0; e < kElems; ++e) {
    const int idx = static_cast<int>(threadIdx.x) + e * 256;
    const int m = idx / kMmqBN;
    if (m >= batch) continue;
    const int gn = static_cast<int>(it) * kMmqBN + (idx % kMmqBN);
    if (gn >= rows) continue;
    __half* p = &y[static_cast<std::size_t>(m) * static_cast<std::size_t>(ldy) + gn];
    *p = __float2half(__half2float(*p) + sum[e]);
  }
}

float* g_mmq_partial = nullptr;
std::size_t g_mmq_partial_elems = 0;

bool ensure_mmq_partial(std::size_t elems) {
  if (elems <= g_mmq_partial_elems) return g_mmq_partial != nullptr;
  capture_guard("kquant MMQ split-K partials");
  if (g_mmq_partial) cudaFree(g_mmq_partial);
  g_mmq_partial = nullptr;
  g_mmq_partial_elems = 0;
  if (cudaMalloc(&g_mmq_partial, elems * sizeof(float)) != cudaSuccess) return false;
  cudaMemset(g_mmq_partial, 0, elems * sizeof(float));
  g_mmq_partial_elems = elems;
  return true;
}

// Stream-K fixup buffer: one tile of raw float partials per block. Separate
// from g_mmq_partial -- that scratch must stay zeroed between calls (the
// atomicAdd path accumulates into it), while this one is plain-stored every
// launch and never needs zeroing. Sharing them would break the finalize
// kernel's leaves-it-zeroed invariant.
float* g_mmq_fixup = nullptr;
std::size_t g_mmq_fixup_elems = 0;

bool ensure_mmq_fixup(std::size_t elems) {
  if (elems <= g_mmq_fixup_elems) return g_mmq_fixup != nullptr;
  capture_guard("kquant MMQ stream-K fixup partials");
  if (g_mmq_fixup) cudaFree(g_mmq_fixup);
  g_mmq_fixup = nullptr;
  g_mmq_fixup_elems = 0;
  if (cudaMalloc(&g_mmq_fixup, elems * sizeof(float)) != cudaSuccess) return false;
  g_mmq_fixup_elems = elems;
  return true;
}

// Grid for the stream-K Q6_K MMQ: one wave -- SM count times this kernel's
// measured occupancy -- so all blocks run concurrently and the kbc ranges
// partition the work exactly once. Cached after the first call; the queries
// are device-attribute reads, not stream work, so they are capture-safe.
int q6k_streamk_grid(int nt_sel, int bm) {
  static int nsm = 0;
  static int occ[3][2] = {{0, 0}, {0, 0}, {0, 0}};  // [NT 1,2,4][BM 32,64]
  if (nsm == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        nsm <= 0) {
      nsm = 128;
    }
  }
  // The dynamic-shared shapes (64x128 at 56 KB, 32x256 at 88 KB) are
  // occupancy 1 by construction: one wave is exactly the SM count.
  if (bm > 32 || nt_sel >= 8) return nsm;
  const int i = nt_sel >= 4 ? 2 : (nt_sel <= 1 ? 0 : 1);
  const int j = bm <= 32 ? 0 : 1;
  if (occ[i][j] == 0) {
    int o = 0;
    cudaError_t err = cudaErrorUnknown;
    const auto query = [&](auto kern) {
      err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, kern, 256, 0);
    };
    if (j == 0) {
      if (i == 2) query(kquant_mmq_q6k_streamk_kernel<4, 32>);
      else if (i == 0) query(kquant_mmq_q6k_streamk_kernel<1, 32>);
      else query(kquant_mmq_q6k_streamk_kernel<2, 32>);
    } else {
      if (i == 2) query(kquant_mmq_q6k_streamk_kernel<4, 64>);
      else if (i == 0) query(kquant_mmq_q6k_streamk_kernel<1, 64>);
      else query(kquant_mmq_q6k_streamk_kernel<2, 64>);
    }
    occ[i][j] = (err == cudaSuccess && o > 0) ? o : 1;
  }
  return nsm * occ[i][j];
}

// Stream-K launch for the Q6_K MMQ. Returns false only when the fixup scratch
// cannot be allocated AND a fallback is impossible; in practice it degrades to
// exact one-tile-per-block tiling, which needs no scratch.
void launch_kquant_mmq_q6k_streamk(const std::uint8_t* w, const std::int8_t* xq, const float* xs,
                                   half* y, int rows, int cols, int batch, int ldy,
                                   cudaStream_t stream) {
  // Batches that fit half the M tile take the 32x128 shape in static shared;
  // larger batches take 64x128 in 56 KB of dynamic shared at occupancy 1 --
  // the N width is what divides activation restaging, so it stays 128.
  const int bm = batch <= 32 ? 32 : 64;
  // NT=8 at BM=32 is the 32x256 tile in 88 KB of dynamic shared (occupancy 1)
  // -- the next step on the width/occupancy frontier, opt-in via
  // CPI_KQUANT_MMQ_Q6K_NT=8.
  const int nt_sel =
      bm == 64 ? 8
               : (g_kq_tune.mmq_q6k_nt >= 8
                      ? 8
                      : (g_kq_tune.mmq_q6k_nt >= 4 ? 4 : (g_kq_tune.mmq_q6k_nt <= 1 ? 1 : 2)));
  const int bn = (128 / bm) * nt_sel * 8;
  const int nty = (rows + bn - 1) / bn;
  const int nsb = cols >> 8;
  const std::int64_t total = static_cast<std::int64_t>(nty) * nsb;
  int nblocks =
      g_kq_tune.mmq_streamk > 1 ? g_kq_tune.mmq_streamk : q6k_streamk_grid(nt_sel, bm);
  // llama.cpp's shortcut: when whole tiles already fill the waves >=90%
  // (the LM head's tiles, for instance), plain one-tile-per-block is the
  // same work with no fixup pass.
  const int waves = (nty + nblocks - 1) / nblocks;
  if (100 * nty >= 90 * nblocks * waves) {
    nblocks = nty;
  } else if (static_cast<std::int64_t>(nblocks) * 2 > total) {
    // Tiny shapes (wv: 8 tiles x 16 super-blocks): one unit per block split
    // every tile ~16 ways and the fixup cost more than the matmul (11.3 vs
    // 9.9 us in the 2026-08-12 audit). Keep at least two units per block, so
    // the walk halves and the tmp traffic halves, at worst half a wave idle.
    nblocks = static_cast<int>(total / 2 > nty ? total / 2 : nty);
  }
  bool fixup = (nty % nblocks) != 0;
  if (fixup && !ensure_mmq_fixup(static_cast<std::size_t>(nblocks) * bn * bm)) {
    nblocks = nty;  // no scratch: block boundaries land on tile boundaries
    fixup = false;
  }
#define CPI_MMQ_Q6K_SK(N, M)                                                                     \
  do {                                                                                           \
    constexpr unsigned kSmemB = sizeof(Q6kSmemLayout<N, M>) > 48u * 1024u                        \
                                    ? static_cast<unsigned>(sizeof(Q6kSmemLayout<N, M>))         \
                                    : 0u;                                                        \
    if (kSmemB > 0) {                                                                            \
      static const bool attr_once = []() {                                                       \
        return cudaFuncSetAttribute(kquant_mmq_q6k_streamk_kernel<N, M>,                         \
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,                 \
                                    static_cast<int>(sizeof(Q6kSmemLayout<N, M>))) ==            \
               cudaSuccess;                                                                      \
      }();                                                                                       \
      (void)attr_once;                                                                           \
    }                                                                                            \
    kquant_mmq_q6k_streamk_kernel<N, M>                                                          \
        <<<static_cast<unsigned>(nblocks), 256, kSmemB, stream>>>(w, xq, xs, y, g_mmq_fixup,     \
                                                                  rows, cols, batch, ldy);       \
    if (fixup) {                                                                                 \
      kquant_mmq_q6k_streamk_fixup_kernel<N, M>                                                  \
          <<<static_cast<unsigned>(nblocks), 256, 0, stream>>>(y, g_mmq_fixup, rows, cols,       \
                                                               batch, ldy);                      \
    }                                                                                            \
  } while (0)
  if (bm == 32) {
    if (nt_sel == 8) {
      CPI_MMQ_Q6K_SK(8, 32);
    } else if (nt_sel == 4) {
      CPI_MMQ_Q6K_SK(4, 32);
    } else if (nt_sel == 1) {
      CPI_MMQ_Q6K_SK(1, 32);
    } else {
      CPI_MMQ_Q6K_SK(2, 32);
    }
  } else {
    // One shape above batch 32: 64x128 in dynamic shared, occupancy 1.
    CPI_MMQ_Q6K_SK(8, 64);
  }
#undef CPI_MMQ_Q6K_SK
}

// The decline conditions of launch_kquant_mmq, checkable WITHOUT launching.
// packed_qkv_matmul is all-or-nothing across q/k/v: probing by launching meant
// a layer whose wv declines had already run -- and then discarded -- the fused
// wq+wk MMQ, every layer, every step (~430 us/step at B=32 in the audit).
bool kquant_mmq_accepts(KQuantType type, int batch, int cols, bool force_q6k) {
  if (type != KQuantType::Q4_K && type != KQuantType::Q6_K) return false;
  if (type == KQuantType::Q6_K && g_kq_tune.mmq_q6k == 0 && !force_q6k) return false;
  // Above batch 32 Q6_K runs the 64x128 dynamic-shared shape. (The 64x64
  // static shape it briefly had there measured -1.8% against dequant+cuBLAS;
  // the width, not the M tile, was what mattered.)
  if (batch < 1 || batch > kMmqBM || cols % 256 != 0) return false;
  return true;
}

bool launch_kquant_mmq(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                       int cols, int batch, int ldy, std::int8_t* xq, float* xs, float* xsum,
                       cudaStream_t stream, bool reuse_x, bool force_q6k) {
  // force_q6k is for callers whose alternative is dequant-and-cuBLAS rather than
  // a good kernel -- the LM head. For layer weights Q6_K MMQ is gated by mmq_q6k.
  if (!kquant_mmq_accepts(type, batch, cols, force_q6k)) return false;
  const int groups_per_row = cols >> 5;
  const int total_groups = batch * groups_per_row;
  // The activation is quantized per call, and at batch the q, k and v
  // projections all read the same one -- so it was being done three times a
  // layer. The microbenchmark shows why that matters: Q4_K MMQ takes about the
  // same 41 us for 2.4 MB as for 9.4, i.e. a large per-call fixed cost, and this
  // is a big part of it. Same redundancy the single-stream path had; the caller
  // is the only thing that knows x is unchanged, so it has to say so.
  if (!reuse_x) {
    quantize_q8_1_groups_kernel<<<(total_groups + 7) / 8, 256, 0, stream>>>(x, xq, xs, xsum, cols,
                                                                           groups_per_row);
  }
  if (type == KQuantType::Q6_K) {
    // Q6_K needs no group sums: its values are (q-32), which is already signed,
    // so the mma result is the dot product with no correction term.
    if (g_kq_tune.mmq_streamk != 0) {
      launch_kquant_mmq_q6k_streamk(w, xq, xs, y, rows, cols, batch, ldy, stream);
      return true;
    }
    // NT sets rows per block (2*NT*8), so it sets the grid. The Q6_K weight this
    // path serves is wv at 1024 rows: at NT=2 that is 32 blocks on a 170-SM part,
    // which is why the kernel measured 139 us a call against the Q4_K MMQ's 78
    // for roughly a seventh of the bytes. NT=1 quadruples the grid.
    const int bb6 = (rows + 31) / 32;
    int sp6 = 1;
    if (g_kq_tune.mmq_splitk > 0 && bb6 < g_kq_tune.mmq_split_target) {
      sp6 = (g_kq_tune.mmq_split_target + bb6 - 1) / bb6;
      sp6 = min(sp6, max(1, (cols >> 8) / 2));
    }
    const bool us6 = sp6 > 1 && ensure_mmq_partial(static_cast<std::size_t>(batch) * rows);
    const dim3 g6(static_cast<unsigned>(bb6), us6 ? static_cast<unsigned>(sp6) : 1u);
    // NT sets the N tile (2*NT*8 columns a block) and so the grid. Under THIS
    // launch NT=2 measured best (ebbe5f2); the stream-K launch above flipped
    // that to NT=4, which is why the two paths read separate knobs.
#define CPI_MMQ_Q6K(N)                                                                         do {                                                                                           const dim3 gg(static_cast<unsigned>((rows + (2 * (N) * 8) - 1) / (2 * (N) * 8)),                            us6 ? static_cast<unsigned>(sp6) : 1u);                                        if (us6) {                                                                                     kquant_mmq_q6k_kernel<N, true>                                                                    <<<gg, 256, 0, stream>>>(w, xq, xs, y, g_mmq_partial, rows, cols, batch, ldy);            kquant_mmq_finalize_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(g_mmq_partial, y,                                                                             rows, batch, ldy);        } else {                                                                                        kquant_mmq_q6k_kernel<N, false>                                                                    <<<gg, 256, 0, stream>>>(w, xq, xs, y, nullptr, rows, cols, batch, ldy);                }                                                                                           } while (0)
    if (g_kq_tune.mmq_q6k_nt >= 4) {
      CPI_MMQ_Q6K(4);
    } else if (g_kq_tune.mmq_q6k_nt <= 1) {
      CPI_MMQ_Q6K(1);
    } else {
      CPI_MMQ_Q6K(2);
    }
#undef CPI_MMQ_Q6K
    return true;
  }
  // NT trades output-tile width against registers and grid size.
  const int nt = g_kq_tune.mmq_nt;
  // The double-buffered kernel carries two K-tiles of activations, so its M tile
  // is 32; larger batches use the single-buffered one.
  const bool async_on = g_kq_tune.mmq_async != 0;
  if (async_on && batch <= 64) {
    // NT=2 (64-wide output tile) measured better than NT=1 at every batch this
    // path serves: 475.2/690.7/938.9 against 429.0/664.6/823.8 at B=8/16/32 in
    // one run. It halves the grid, and for a 4096-row matrix that is 64 blocks
    // against 170 SMs -- so the extra work per block outweighs filling the GPU,
    // which is the opposite of what the block count suggests. Measure, do not
    // reason from grid size here.
    // Batch 33..64 takes the dual-half M2 shape: NT=2 (the wide-N tile's shared
    // budget goes to the 64-row activation panel instead), weights staged once
    // and replayed against both batch halves.
    const bool m2 = batch > 32;
    const int ant =
        m2 ? 2 : (g_kq_tune.mmq_async_nt > 0 ? g_kq_tune.mmq_async_nt : (batch > 16 ? 4 : 2));
    const int base_blocks = ant >= 4 ? (rows + 127) / 128
                                     : (ant >= 2 ? (rows + 63) / 64 : (rows + 31) / 32);
    const int nsb = cols >> 8;
    // Enough blocks to cover the SMs, but never so few super-blocks per block
    // that the pipeline has nothing to overlap.
    int split = 1;
    if (g_kq_tune.mmq_splitk > 0 && base_blocks < g_kq_tune.mmq_split_target) {
      split = (g_kq_tune.mmq_split_target + base_blocks - 1) / base_blocks;
      split = min(split, max(1, nsb / 2));
    }
    const bool use_split =
        split > 1 && ensure_mmq_partial(static_cast<std::size_t>(batch) * rows);
    const dim3 grid(static_cast<unsigned>(base_blocks), use_split ? static_cast<unsigned>(split) : 1u);
#define CPI_MMQ_ASYNC(N, S, M)                                                                 kquant_mmq_q4k_async_kernel<N, S, M>                                                             <<<grid, 256, 0, stream>>>(w, xq, xs, xsum, y, g_mmq_partial, rows, cols, batch, ldy)
    if (m2) {
      if (use_split) { CPI_MMQ_ASYNC(2, true, true); } else { CPI_MMQ_ASYNC(2, false, true); }
    } else if (ant >= 4) {
      if (use_split) { CPI_MMQ_ASYNC(4, true, false); } else { CPI_MMQ_ASYNC(4, false, false); }
    } else if (ant >= 2) {
      if (use_split) { CPI_MMQ_ASYNC(2, true, false); } else { CPI_MMQ_ASYNC(2, false, false); }
    } else {
      if (use_split) { CPI_MMQ_ASYNC(1, true, false); } else { CPI_MMQ_ASYNC(1, false, false); }
    }
#undef CPI_MMQ_ASYNC
    if (use_split) {
      kquant_mmq_finalize_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(g_mmq_partial, y, rows,
                                                                        batch, ldy);
    }
    return true;
  }
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
  const int max_batch = g_kq_tune.mm_max;
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

// Integer-dot matvec: activations to int8 once, then dp4a instead of per-weight
// float unpacking.
//
// Profiling the fp16-activation matvec on w13 put compute at 62.8% against DRAM
// at 56.8% -- it is arithmetic-limited, and the arithmetic is converting each
// 4-bit weight to float and multiplying. Four low nibbles of a 4-byte slice are
// already four int8 lanes (`qw & 0x0F0F0F0F`), so one dp4a replaces four
// convert-and-multiply pairs, and a second against 0x01010101 gives the group
// sum the `- dmin*m` term needs.
//
// A matvec is where this trade is cheapest: the activation is quantized once and
// reused down every row, so the cost is one pass over `cols` against rows*cols
// of saved arithmetic. The same idea lost in the batched matmul, where the
// quantization happened per weight rather than per activation.
//
// It does change numerics -- int8 activations are what llama.cpp's MMVQ uses,
// and it is why decode stops being bit-identical to the fp16 path. Opt-in.
template <KQuantType TYPE, int WPR, bool ACC>
__global__ void kquant_matvec_dp4a_kernel(const std::uint8_t* __restrict__ w,
                                          const std::int8_t* __restrict__ xq,
                                          const float* __restrict__ xs, __half* __restrict__ y,
                                          int rows, int cols) {
  constexpr int kWarps = 8;
  constexpr int kRowsPerBlock = kWarps / WPR;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warpid = static_cast<int>(threadIdx.x) >> 5;
  const int sub = warpid % WPR;
  const int row = blockIdx.x * kRowsPerBlock + warpid / WPR;
  if (row >= rows) return;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);
  const std::size_t block_bytes =
      TYPE == KQuantType::Q4_K ? model::kquant::kQ4KBlockBytes : model::kquant::kQ5KBlockBytes;
  const std::uint8_t* p = w + static_cast<std::size_t>(row) * blocks_per_row * block_bytes +
                          static_cast<std::size_t>(sub) * block_bytes;

  const int j = lane >> 3;
  const int t = lane & 7;
  const int off_lo = (j << 6) + (t << 2);
  const int off_hi = off_lo + 32;

  // The activation scales are NOT staged in shared. Tried it: one per 32 inputs
  // is 512 B for a 4096-wide matvec, so every block already finds them in L1,
  // and with eight warps per row a block covers a single output row -- the
  // staging pass and its __syncthreads are then paid per row to save loads that
  // were already free. Cost 3 points against llama.cpp.
  float acc = 0.0f;
#pragma unroll 2
  for (int b = sub; b < blocks_per_row; b += WPR) {
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
      const std::uint32_t hw = *reinterpret_cast<const std::uint32_t*>(p + 16 + (t << 2));
      qlo |= static_cast<int>(((hw >> (2 * j)) & 0x01010101u) << 4);
      qhi |= static_cast<int>(((hw >> (2 * j + 1)) & 0x01010101u) << 4);
    }

    const int xbase = b * static_cast<int>(kSuperBlock);
    const int gbase = b << 3;  // eight 32-groups per super-block
    const int xl = *reinterpret_cast<const int*>(xq + xbase + off_lo);
    const int xh = *reinterpret_cast<const int*>(xq + xbase + off_hi);
    const int dot_lo = __dp4a(qlo, xl, 0);
    const int dot_hi = __dp4a(qhi, xh, 0);
    const int sum_lo = __dp4a(xl, 0x01010101, 0);
    const int sum_hi = __dp4a(xh, 0x01010101, 0);
    // Per-lane partial sums: the group scale is uniform across the run, so
    // applying it here is the same arithmetic as applying it per weight.
    acc += xs[gbase + 2 * j] * (d * sc0 * static_cast<float>(dot_lo) -
                                dmin * m0 * static_cast<float>(sum_lo)) +
           xs[gbase + 2 * j + 1] * (d * sc1 * static_cast<float>(dot_hi) -
                                    dmin * m1 * static_cast<float>(sum_hi));
    p += block_bytes * WPR;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (WPR == 1) {
    if (lane == 0) y[row] = __float2half(ACC ? acc + __half2float(y[row]) : acc);
    return;
  }
  __shared__ float parts[32];
  if (lane == 0) parts[warpid] = acc;
  __syncthreads();
  if (lane == 0 && sub == 0) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < WPR; ++k) total += parts[warpid + k];
    y[row] = __float2half(ACC ? total + __half2float(y[row]) : total);
  }
}

// Same integer dot as above, but each lane takes an 8-byte slice of qs instead
// of 4, so one scale decode covers sixteen weights instead of eight.
//
// The dot was never the cost. Per super-block a lane ran two
// scale_min_from_hdr decodes (~12 bit ops) and ~13 float ops to apply them,
// against four dp4a -- the scale handling is roughly six times the dot. Both
// halves of an 8-byte slice land in the same pair of 32-groups, so their dots
// can be summed as integers first and scaled once, which halves the scale work
// and the qs load instructions for identical bytes moved.
//
// Sixteen lanes now cover one super-block, so a warp covers two per iteration.
template <KQuantType TYPE, int WPR, bool ACC>
__global__ void kquant_matvec_dp4a16_kernel(const std::uint8_t* __restrict__ w,
                                            const std::int8_t* __restrict__ xq,
                                            const float* __restrict__ xs, __half* __restrict__ y,
                                            int rows, int cols) {
  constexpr int kWarps = 8;
  constexpr int kRowsPerBlock = kWarps / WPR;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warpid = static_cast<int>(threadIdx.x) >> 5;
  const int sub = warpid % WPR;
  const int row = blockIdx.x * kRowsPerBlock + warpid / WPR;
  if (row >= rows) return;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);
  const std::size_t block_bytes =
      TYPE == KQuantType::Q4_K ? model::kquant::kQ4KBlockBytes : model::kquant::kQ5KBlockBytes;

  const int sb_off = lane >> 4;   // which of the warp's two super-blocks
  const int r = lane & 15;        // which 8-byte slice of the 128-byte qs
  const int j = r >> 2;           // 32-byte run, i.e. scale-group pair (2j, 2j+1)
  const int t = r & 3;            // slice within the run
  const int off_lo = (j << 6) + (t << 3);
  const int off_hi = off_lo + 32;
  const int qoff = (TYPE == KQuantType::Q4_K) ? 16 : 48;

  const std::uint8_t* base = w + static_cast<std::size_t>(row) * blocks_per_row * block_bytes;

  float acc = 0.0f;
  for (int b = sub * 2; b < blocks_per_row; b += WPR * 2) {
    const int sb = b + sb_off;
    if (sb >= blocks_per_row) break;
    const std::uint8_t* p = base + static_cast<std::size_t>(sb) * block_bytes;

    const uint4 hdr = *reinterpret_cast<const uint4*>(p);
    const float d = half_bits_to_float(static_cast<std::uint16_t>(hdr.x & 0xFFFFu));
    const float dmin = half_bits_to_float(static_cast<std::uint16_t>(hdr.x >> 16));
    float sc0, m0, sc1, m1;
    scale_min_from_hdr(2 * j, hdr.y, hdr.z, hdr.w, &sc0, &m0);
    scale_min_from_hdr(2 * j + 1, hdr.y, hdr.z, hdr.w, &sc1, &m1);

    const uint2 qw = *reinterpret_cast<const uint2*>(p + qoff + (r << 3));
    uint2 hw;
    if (TYPE == KQuantType::Q5_K) hw = *reinterpret_cast<const uint2*>(p + 16 + (t << 3));

    const int xbase = sb * static_cast<int>(kSuperBlock);
    int dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const std::uint32_t word = k == 0 ? qw.x : qw.y;
      int qlo = static_cast<int>(word & 0x0F0F0F0Fu);
      int qhi = static_cast<int>((word >> 4) & 0x0F0F0F0Fu);
      if (TYPE == KQuantType::Q5_K) {
        const std::uint32_t h = k == 0 ? hw.x : hw.y;
        qlo |= static_cast<int>(((h >> (2 * j)) & 0x01010101u) << 4);
        qhi |= static_cast<int>(((h >> (2 * j + 1)) & 0x01010101u) << 4);
      }
      const int xl = *reinterpret_cast<const int*>(xq + xbase + off_lo + (k << 2));
      const int xh = *reinterpret_cast<const int*>(xq + xbase + off_hi + (k << 2));
      dot_lo = __dp4a(qlo, xl, dot_lo);
      dot_hi = __dp4a(qhi, xh, dot_hi);
      sum_lo = __dp4a(xl, 0x01010101, sum_lo);
      sum_hi = __dp4a(xh, 0x01010101, sum_hi);
    }

    // Both slices sit in the same two groups, so the scales apply once to the
    // summed integer dots rather than once per four weights.
    const int gbase = sb << 3;
    acc += xs[gbase + 2 * j] * (d * sc0 * static_cast<float>(dot_lo) -
                                dmin * m0 * static_cast<float>(sum_lo)) +
           xs[gbase + 2 * j + 1] * (d * sc1 * static_cast<float>(dot_hi) -
                                    dmin * m1 * static_cast<float>(sum_hi));
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  if (WPR == 1) {
    if (lane == 0) y[row] = __float2half(ACC ? acc + __half2float(y[row]) : acc);
    return;
  }
  __shared__ float parts[32];
  if (lane == 0) parts[warpid] = acc;
  __syncthreads();
  if (lane == 0 && sub == 0) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < WPR; ++k) total += parts[warpid + k];
    y[row] = __float2half(ACC ? total + __half2float(y[row]) : total);
  }
}

// The same trade taken one step further: a 16-byte slice of qs per lane, so one
// scale decode covers thirty-two weights. Widening 4 -> 8 bytes was worth 5.3%,
// which said the scale work really was the limiter; this halves what is left.
//
// Eight lanes now cover a super-block and a warp covers four, so this wants four
// warps per row rather than eight -- four warps x four super-blocks is exactly a
// 4096-wide row. At WPR=8 half the block would find nothing to do. Two rows per
// block keeps the same warps in flight.
// Split-K does not apply here, though the shape of the problem invites it. A
// per-shape microbenchmark showed the narrow projections at a fraction of the
// MLP matrices' throughput -- wk at 102 GB/s against w1's 1684 -- which reads as
// a starved GPU wanting more blocks. Two things were wrong with that. The bench
// launches outside a graph, and on Windows the per-launch cost dominates: every
// shape came out at 18-26 us regardless of size, so it was measuring launch
// overhead, not bandwidth. And a 4096-wide row is only 16 super-blocks, which
// one block already consumes entirely (4 warps x 4 each), so there is no column
// depth left to split -- chunking it just idles three warps in four. Measured
// end to end it cost 8.8%: 220.5 against 241.9 tok/s. Use nsys, not that bench,
// to decide anything about this kernel.
template <KQuantType TYPE, int WPR, bool ACC, bool GSUM, bool VECX>
__global__ void kquant_matvec_dp4a32_kernel(const std::uint8_t* __restrict__ w,
                                            const std::int8_t* __restrict__ xq,
                                            const float* __restrict__ xs,
                                            const float* __restrict__ gsum, __half* __restrict__ y,
                                            int rows, int cols) {
  constexpr int kWarps = 8;
  constexpr int kRowsPerBlock = kWarps / WPR;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warpid = static_cast<int>(threadIdx.x) >> 5;
  const int sub = warpid % WPR;
  const int row = blockIdx.x * kRowsPerBlock + warpid / WPR;
  if (row >= rows) return;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);
  const std::size_t block_bytes =
      TYPE == KQuantType::Q4_K ? model::kquant::kQ4KBlockBytes : model::kquant::kQ5KBlockBytes;

  const int sb_off = lane >> 3;  // which of the warp's four super-blocks
  const int r = lane & 7;        // which 16-byte slice of the 128-byte qs
  const int j = r >> 1;          // 32-byte run, i.e. scale-group pair (2j, 2j+1)
  const int t = r & 1;           // half of the run
  const int off_lo = (j << 6) + (t << 4);
  const int off_hi = off_lo + 32;
  const int qoff = (TYPE == KQuantType::Q4_K) ? 16 : 48;

  const std::uint8_t* base = w + static_cast<std::size_t>(row) * blocks_per_row * block_bytes;

  float acc = 0.0f;
  for (int b = sub * 4; b < blocks_per_row; b += WPR * 4) {
    const int sb = b + sb_off;
    if (sb >= blocks_per_row) break;
    const std::uint8_t* p = base + static_cast<std::size_t>(sb) * block_bytes;

    const uint4 hdr = *reinterpret_cast<const uint4*>(p);
    const float d = half_bits_to_float(static_cast<std::uint16_t>(hdr.x & 0xFFFFu));
    const float dmin = half_bits_to_float(static_cast<std::uint16_t>(hdr.x >> 16));
    float sc0, m0, sc1, m1;
    scale_min_from_hdr(2 * j, hdr.y, hdr.z, hdr.w, &sc0, &m0);
    scale_min_from_hdr(2 * j + 1, hdr.y, hdr.z, hdr.w, &sc1, &m1);

    const uint4 qw = *reinterpret_cast<const uint4*>(p + qoff + (r << 4));
    uint4 hw;
    if (TYPE == KQuantType::Q5_K) hw = *reinterpret_cast<const uint4*>(p + 16 + (t << 4));

    const int xbase = sb * static_cast<int>(kSuperBlock);
    // The four xl a lane wants are at off_lo + 0,4,8,12 -- sixteen contiguous
    // bytes, and off_lo is a multiple of sixteen. Reading them as one int4
    // instead of four ints is the same traffic in a quarter of the load
    // instructions, which is what is left to win here now that the dot is not
    // the limiter.
    int4 xl4, xh4;
    if (VECX) {
      xl4 = *reinterpret_cast<const int4*>(xq + xbase + off_lo);
      xh4 = *reinterpret_cast<const int4*>(xq + xbase + off_hi);
    }
    int dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const std::uint32_t word = k == 0 ? qw.x : (k == 1 ? qw.y : (k == 2 ? qw.z : qw.w));
      int qlo = static_cast<int>(word & 0x0F0F0F0Fu);
      int qhi = static_cast<int>((word >> 4) & 0x0F0F0F0Fu);
      if (TYPE == KQuantType::Q5_K) {
        const std::uint32_t h = k == 0 ? hw.x : (k == 1 ? hw.y : (k == 2 ? hw.z : hw.w));
        qlo |= static_cast<int>(((h >> (2 * j)) & 0x01010101u) << 4);
        qhi |= static_cast<int>(((h >> (2 * j + 1)) & 0x01010101u) << 4);
      }
      const int xl = VECX ? (k == 0 ? xl4.x : (k == 1 ? xl4.y : (k == 2 ? xl4.z : xl4.w)))
                          : *reinterpret_cast<const int*>(xq + xbase + off_lo + (k << 2));
      const int xh = VECX ? (k == 0 ? xh4.x : (k == 1 ? xh4.y : (k == 2 ? xh4.z : xh4.w)))
                          : *reinterpret_cast<const int*>(xq + xbase + off_hi + (k << 2));
      dot_lo = __dp4a(qlo, xl, dot_lo);
      dot_hi = __dp4a(qhi, xh, dot_hi);
      if (!GSUM) {
        sum_lo = __dp4a(xl, 0x01010101, sum_lo);
        sum_hi = __dp4a(xh, 0x01010101, sum_hi);
      }
    }

    // The -dmin*m term wants sum(x) over the whole group, not over this lane's
    // half of it, and the quantizer already wrote exactly that per group. Half
    // the dp4a here used to recompute it against 0x01010101 -- once per row, for
    // a quantity that does not depend on the row. The two lanes sharing a run
    // split the group between them, so letting t == 0 carry the entire
    // correction sums to the same value.
    const int gbase = sb << 3;
    if (GSUM) {
      acc += xs[gbase + 2 * j] * d * sc0 * static_cast<float>(dot_lo) +
             xs[gbase + 2 * j + 1] * d * sc1 * static_cast<float>(dot_hi);
      if (t == 0) {
        acc -= dmin * (m0 * gsum[gbase + 2 * j] + m1 * gsum[gbase + 2 * j + 1]);
      }
    } else {
      acc += xs[gbase + 2 * j] * (d * sc0 * static_cast<float>(dot_lo) -
                                  dmin * m0 * static_cast<float>(sum_lo)) +
             xs[gbase + 2 * j + 1] * (d * sc1 * static_cast<float>(dot_hi) -
                                      dmin * m1 * static_cast<float>(sum_hi));
    }
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  __shared__ float parts[32];
  if (lane == 0) parts[warpid] = acc;
  __syncthreads();
  if (lane == 0 && sub == 0) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < WPR; ++k) total += parts[warpid + k];
    y[row] = __float2half(ACC ? total + __half2float(y[row]) : total);
  }
}

// Q6_K on the integer path. Q6_K is a quarter of a Q4_K_M model by bytes --
// ffn_down for over half the layers, plus the output head -- and it was taking
// the fp16-activation matvec while everything else moved to dp4a.
//
// The reason it was excluded was that Q6_K carries a scale per 16 weights, not
// per 32, so it does not line up with a 32-wide mma. A matvec does not need it
// to. Each lane takes an 8-byte slice of ql, and because l0 is then a multiple
// of eight, l0>>4 is constant across the run: one scale pair covers the whole
// slice, exactly as for Q4_K. The 32-group the activation was quantized in is
// likewise constant, since off_lo advances inside a 32-aligned run.
//
// Blocks are 210 bytes, so p is only 2-byte aligned and the slice has to be read
// as uint16 pairs rather than a uint2 -- the same reason kq_block_values does.
//
// Values are (q - 32) with q in 0..63, so the correction term is 32*sum(x)
// against Q4_K's dmin*m*sum(x); the shape of the arithmetic is identical.
template <int WPR, bool ACC>
__global__ void kquant_matvec_dp4a_q6k_kernel(const std::uint8_t* __restrict__ w,
                                              const std::int8_t* __restrict__ xq,
                                              const float* __restrict__ xs,
                                              __half* __restrict__ y, int rows, int cols) {
  constexpr int kWarps = 8;
  constexpr int kRowsPerBlock = kWarps / WPR;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warpid = static_cast<int>(threadIdx.x) >> 5;
  const int sub = warpid % WPR;
  const int row = blockIdx.x * kRowsPerBlock + warpid / WPR;
  if (row >= rows) return;
  const int blocks_per_row = cols / static_cast<int>(kSuperBlock);
  constexpr std::size_t block_bytes = model::kquant::kQ6KResidentBytes;

  const int sb_off = lane >> 4;  // which of the warp's two super-blocks
  const int b0 = (lane & 15) << 3;  // 8-byte slice of the 128-byte ql
  const int n = b0 >> 6;
  const int h = (b0 >> 5) & 1;
  const int l0 = b0 & 31;
  const int off_lo = (n << 7) + (h << 5) + l0;
  const int off_hi = off_lo + 64;
  const int sl = 2 * h;
  const int sh = 4 + 2 * h;

  const std::uint8_t* base = w + static_cast<std::size_t>(row) * blocks_per_row * block_bytes;

  float acc = 0.0f;
  for (int b = sub * 2; b < blocks_per_row; b += WPR * 2) {
    const int sb = b + sb_off;
    if (sb >= blocks_per_row) break;
    const std::uint8_t* p = base + static_cast<std::size_t>(sb) * block_bytes;

    std::uint16_t dbits;
    memcpy(&dbits, p + 208, 2);
    const float d = half_bits_to_float(dbits);
    const int sbase = 192 + (n << 3) + (l0 >> 4);
    const float scl = static_cast<float>(*reinterpret_cast<const std::int8_t*>(p + sbase + sl));
    const float sch = static_cast<float>(*reinterpret_cast<const std::int8_t*>(p + sbase + sh));

    const std::uint16_t* qlp = reinterpret_cast<const std::uint16_t*>(p + b0);
    const std::uint16_t* qhp =
        reinterpret_cast<const std::uint16_t*>(p + 128 + (n << 5) + l0);

    const int xbase = sb * static_cast<int>(kSuperBlock);
    // Eight contiguous activation bytes per half, off_lo being a multiple of
    // eight: one int2 rather than two ints, as in the Q4_K kernel.
    const int2 xl2 = *reinterpret_cast<const int2*>(xq + xbase + off_lo);
    const int2 xh2 = *reinterpret_cast<const int2*>(xq + xbase + off_hi);
    int dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const std::uint32_t word = static_cast<std::uint32_t>(qlp[2 * k]) |
                                 (static_cast<std::uint32_t>(qlp[2 * k + 1]) << 16);
      const std::uint32_t hword = static_cast<std::uint32_t>(qhp[2 * k]) |
                                  (static_cast<std::uint32_t>(qhp[2 * k + 1]) << 16);
      const int qlo = static_cast<int>((word & 0x0F0F0F0Fu) |
                                       (((hword >> sl) & 0x03030303u) << 4));
      const int qhi = static_cast<int>(((word >> 4) & 0x0F0F0F0Fu) |
                                       (((hword >> sh) & 0x03030303u) << 4));
      const int xl = k == 0 ? xl2.x : xl2.y;
      const int xh = k == 0 ? xh2.x : xh2.y;
      dot_lo = __dp4a(qlo, xl, dot_lo);
      dot_hi = __dp4a(qhi, xh, dot_hi);
      sum_lo = __dp4a(xl, 0x01010101, sum_lo);
      sum_hi = __dp4a(xh, 0x01010101, sum_hi);
    }

    // The group-sum shortcut the Q4_K kernel uses does NOT work here, and the
    // gate says so loudly (worst_rel 2.8). It needs one weight scale across the
    // lanes sharing an activation group, and Q6_K carries a scale per 16 weights
    // where the group is 32: the four lanes covering a group split across two
    // different scales, so no single lane can carry the whole -32*sum(x) term.
    // Per-16 sums would fix it, but the quantizer only emits per-32.
    const int glo = (sb << 3) + (n << 2) + h;
    acc += xs[glo] * d * scl * static_cast<float>(dot_lo - 32 * sum_lo) +
           xs[glo + 2] * d * sch * static_cast<float>(dot_hi - 32 * sum_hi);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
  __shared__ float parts[32];
  if (lane == 0) parts[warpid] = acc;
  __syncthreads();
  if (lane == 0 && sub == 0) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < WPR; ++k) total += parts[warpid + k];
    y[row] = __float2half(ACC ? total + __half2float(y[row]) : total);
  }
}




namespace {
// Scratch for the int8 activation form, grown to the widest matvec seen. Kept
// module-local so the dp4a path can be switched on without touching every call
// site; it is a few tens of KB.
std::int8_t* g_dp4a_xq = nullptr;
// Width the quantized activation in the scratch currently holds, so a caller
// that knows x is unchanged can skip re-quantizing it. -1 means nothing valid.
int g_dp4a_ready_cols = -1;
float* g_dp4a_xs = nullptr;
// The quantizer writes a scale AND a group sum. The matvec only needs the
// scale -- it derives its own partial sums with dp4a -- but the sum still needs
// somewhere to land: aliasing it onto the scale buffer silently overwrote every
// scale and produced fluent garbage.
float* g_dp4a_sum = nullptr;
int g_dp4a_cols = 0;

// Must be reserved BEFORE any capture: cudaMalloc inside a captured stream is
// rejected, and this scratch is sized from a kernel that the decode graph
// captures. Reserving lazily crashed capture with "operation failed due to a
// previous error during capture" -- the same trap the GEMM scratch hit.
bool g_capture_active = false;

bool ensure_dp4a_scratch(int cols) {
  if (cols <= g_dp4a_cols) return g_dp4a_xq != nullptr;
  capture_guard("kquant dp4a activation scratch");
  if (g_dp4a_xq) cudaFree(g_dp4a_xq);
  if (g_dp4a_xs) cudaFree(g_dp4a_xs);
  if (g_dp4a_sum) cudaFree(g_dp4a_sum);
  g_dp4a_xq = nullptr;
  g_dp4a_xs = nullptr;
  g_dp4a_sum = nullptr;
  g_dp4a_cols = 0;
  const std::size_t meta = static_cast<std::size_t>(cols / 32) * sizeof(float);
  g_dp4a_ready_cols = -1;
  if (cudaMalloc(&g_dp4a_xq, static_cast<std::size_t>(cols)) != cudaSuccess) return false;
  if (cudaMalloc(&g_dp4a_xs, meta) != cudaSuccess || cudaMalloc(&g_dp4a_sum, meta) != cudaSuccess) {
    cudaFree(g_dp4a_xq);
    g_dp4a_xq = nullptr;
    return false;
  }
  g_dp4a_cols = cols;
  return true;
}

template <bool ACC>
bool try_dp4a_matvec(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                     int cols, cudaStream_t stream, bool reuse) {
  if (g_kq_tune.matvec_dp4a == 0) return false;
  if (cols % 256 != 0 || !ensure_dp4a_scratch(cols)) return false;
  const int groups = cols >> 5;
  // A layer's q, k and v projections read one activation, and this model cannot
  // fuse them into a packed triple because wv is Q6_K where wq and wk are Q4_K.
  // Quantizing it once per group of sharers rather than once per projection is
  // two fewer kernels per layer, 64 fewer per token. The caller is the only
  // thing that knows x is unchanged, so it has to say so -- keying on the
  // pointer would silently reuse across layers that share a residual buffer.
  const bool have_x = g_dp4a_ready_cols == cols && g_kq_tune.matvec_share_x;
  if (!(reuse && have_x)) {
    quantize_q8_1_groups_kernel<<<(groups + 7) / 8, 256, 0, stream>>>(x, g_dp4a_xq, g_dp4a_xs,
                                                                      g_dp4a_sum, cols, groups);
    g_dp4a_ready_cols = cols;
  }
  constexpr int kThreads = 256;
  const int wpr = 8;
  const int grid = (rows + (8 / wpr) - 1) / (8 / wpr);
  if (type == KQuantType::Q6_K) {
    if (g_kq_tune.matvec_dp4a_q6k == 0) return false;
    kquant_matvec_dp4a_q6k_kernel<8, ACC>
        <<<grid, kThreads, 0, stream>>>(w, g_dp4a_xq, g_dp4a_xs, y, rows, cols);
  } else if (g_kq_tune.matvec_dp4a16 == 2) {
    // Four warps per row: four warps x four super-blocks covers a 4096-wide row
    // exactly, so widening the slice does not idle half the block.
    const dim3 grid32(static_cast<unsigned>((rows + 1) / 2));
#define CPI_KQ_MV32(T, G, V)                                                                kquant_matvec_dp4a32_kernel<T, 4, ACC, G, V>                                                  <<<grid32, kThreads, 0, stream>>>(w, g_dp4a_xq, g_dp4a_xs, g_dp4a_sum, y, rows, cols)
#define CPI_KQ_MV32_G(T, G)                                                                 if (g_kq_tune.matvec_vecx) {                                                                CPI_KQ_MV32(T, G, true);                                                                } else {                                                                                    CPI_KQ_MV32(T, G, false);                                                               }
    if (type == KQuantType::Q4_K) {
      if (g_kq_tune.matvec_gsum) {
        CPI_KQ_MV32_G(KQuantType::Q4_K, true)
      } else {
        CPI_KQ_MV32_G(KQuantType::Q4_K, false)
      }
    } else {
      if (g_kq_tune.matvec_gsum) {
        CPI_KQ_MV32_G(KQuantType::Q5_K, true)
      } else {
        CPI_KQ_MV32_G(KQuantType::Q5_K, false)
      }
    }
#undef CPI_KQ_MV32_G
#undef CPI_KQ_MV32
  } else if (g_kq_tune.matvec_dp4a16) {
    if (type == KQuantType::Q4_K) {
      kquant_matvec_dp4a16_kernel<KQuantType::Q4_K, 8, ACC>
          <<<grid, kThreads, 0, stream>>>(w, g_dp4a_xq, g_dp4a_xs, y, rows, cols);
    } else {
      kquant_matvec_dp4a16_kernel<KQuantType::Q5_K, 8, ACC>
          <<<grid, kThreads, 0, stream>>>(w, g_dp4a_xq, g_dp4a_xs, y, rows, cols);
    }
  } else if (type == KQuantType::Q4_K) {
    kquant_matvec_dp4a_kernel<KQuantType::Q4_K, 8, ACC>
        <<<grid, kThreads, 0, stream>>>(w, g_dp4a_xq, g_dp4a_xs, y, rows, cols);
  } else {
    kquant_matvec_dp4a_kernel<KQuantType::Q5_K, 8, ACC>
        <<<grid, kThreads, 0, stream>>>(w, g_dp4a_xq, g_dp4a_xs, y, rows, cols);
  }
  return true;
}
}  // namespace

template <typename OutT, bool ACC>
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

#define CPI_KQ_DISPATCH(WPRV)                                     switch (type) {                                                   case KQuantType::Q4_K:                                            kquant_matvec_vec_kernel<KQuantType::Q4_K, WPRV, OutT, ACC>              <<<grid, kThreads, 0, stream>>>(w, x, y, rows, cols);       return;                                                       case KQuantType::Q5_K:                                            kquant_matvec_vec_kernel<KQuantType::Q5_K, WPRV, OutT, ACC>              <<<grid, kThreads, 0, stream>>>(w, x, y, rows, cols);       return;                                                       case KQuantType::Q6_K:                                            kquant_matvec_vec_kernel<KQuantType::Q6_K, WPRV, OutT, ACC>              <<<grid, kThreads, 0, stream>>>(w, x, y, rows, cols);       return;                                                     }

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

void set_capture_active(bool active) { g_capture_active = active; }
bool capture_active() { return g_capture_active; }

bool capture_guard(const char* what) {
  if (!g_capture_active) return false;
  static const bool strict = [] {
    const char* e = std::getenv("CPI_CAPTURE_STRICT");
    return e && *e == '1';
  }();
  std::fprintf(stderr, "[capture] ALLOCATION DURING GRAPH CAPTURE: %s\n", what);
  std::fprintf(stderr, "[capture]   capture fails later at cudaStreamEndCapture naming nothing;\n");
  std::fprintf(stderr, "[capture]   reserve this before capture (see the reserve_* helpers).\n");
  if (strict) std::abort();
  return true;
}

void reserve_kquant_dp4a_scratch(int cols) { ensure_dp4a_scratch(cols); }

bool acquire_kquant_q8_1_scratch(int cols, std::int8_t** xq, float** xs, float** gsum) {
  if (g_kq_tune.matvec_dp4a == 0 || cols % 256 != 0 || !ensure_dp4a_scratch(cols)) return false;
  *xq = g_dp4a_xq;
  *xs = g_dp4a_xs;
  *gsum = g_dp4a_sum;
  // Whoever fills it is standing in for quantize_q8_1_groups_kernel, so the
  // matvec that follows can be told to reuse it.
  g_dp4a_ready_cols = cols;
  return true;
}

void launch_kquant_matvec(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                          int cols, cudaStream_t stream, bool reuse_x) {
  if (try_dp4a_matvec<false>(w, type, x, y, rows, cols, stream, reuse_x)) return;
  launch_kquant_matvec_impl<half, false>(w, type, x, y, rows, cols, stream);
}

void launch_kquant_matvec_residual(const std::uint8_t* w, KQuantType type, const half* x, half* y,
                                   int rows, int cols, cudaStream_t stream, bool reuse_x) {
  if (try_dp4a_matvec<true>(w, type, x, y, rows, cols, stream, reuse_x)) return;
  launch_kquant_matvec_impl<half, true>(w, type, x, y, rows, cols, stream);
}

// The LM head stays on fp16 activations deliberately. It is 7.5% of a decode
// step on its own -- 202 launches of ~296 us for 431 MB of Q6_K -- so it looked
// like the last easy dp4a win. Wiring it through the integer path (templating
// the kernels on output type, since this one writes fp32 logits) works and is
// gate-clean, but nsys put the kernel at 291 us against 296: 1.6% of the kernel,
// 0.12% of the step, invisible end to end. It was already at 1480 GB/s of a
// ~1792 GB/s part, i.e. bandwidth-bound, which is the same reason Q6_K gained
// least from dp4a everywhere else. Not worth changing logit numerics for.
void launch_kquant_matvec_f32(const std::uint8_t* w, KQuantType type, const half* x, float* y,
                              int rows, int cols, cudaStream_t stream) {
  launch_kquant_matvec_impl<float, false>(w, type, x, y, rows, cols, stream);
}

}  // namespace kernels
