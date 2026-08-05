#pragma once

// Device-side helpers shared by the quantized-KV kernel files (contiguous and
// paged). Header-only so each translation unit instantiates what it needs.
//
// Quantization recipe (must stay in lockstep across store and read paths and
// the host references in the parity tests): per (token, kv_head) symmetric
// absmax scale, fp16; 4-bit = two signed nibbles per byte (low nibble first,
// range [-8, 7], scale = absmax/7), 8-bit = signed byte (range [-127, 127],
// scale = absmax/127). Optional QuaRot R3: an unrandomized head_dim
// Walsh-Hadamard applied to K before quantization, matched by the same
// transform on Q at read time.

#include <cuda_fp16.h>

namespace kernels {
namespace kvq {

// In-place Walsh-Hadamard butterfly over buf[0..n). Requires blockDim.x >= n,
// n a power of two, and buf fully populated and synced before the call.
// Unnormalized; the caller applies the 1/sqrt(n) factor.
__device__ __forceinline__ void fwht_shared(float* buf, int tid, int n) {
  for (int len = 1; len < n; len <<= 1) {
    float a = 0.0f, b = 0.0f;
    if (tid < n) {
      a = buf[tid];
      b = buf[tid ^ len];
    }
    __syncthreads();
    if (tid < n) {
      buf[tid] = (tid & len) ? (b - a) : (a + b);
    }
    __syncthreads();
  }
}

// Dequantize element d of a K/V row (int4 packed nibbles or int8 bytes).
template <int Bits>
__device__ __forceinline__ float kv_load(const int8_t* row, int d, float scale);

template <>
__device__ __forceinline__ float kv_load<4>(const int8_t* row, int d, float scale) {
  const int8_t b = row[d >> 1];
  return static_cast<float>((d & 1) ? (static_cast<int>(b) >> 4)
                                    : ((static_cast<int>(b) << 28) >> 28)) *
         scale;
}

template <>
__device__ __forceinline__ float kv_load<8>(const int8_t* row, int d, float scale) {
  return static_cast<float>(row[d]) * scale;
}

// Quantize one K/V head pair staged in shared memory and write cache rows plus
// scales. Call with blockDim.x == head_dim threads; smem layout is
// [head_dim k_buf][head_dim v_buf][num_warps k_warp][num_warps v_warp][2].
// k_buf/v_buf must hold the fp32 values (K already rotated if applicable) and
// be synced. k_row_out/v_row_out point at this (token, head)'s cache row;
// k_scale_out/v_scale_out at its scale slots.
template <int KBits, int VBits>
__device__ __forceinline__ void quant_store_head(float* smem, int tid, int head_dim,
                                                 int8_t* k_row_out, int8_t* v_row_out,
                                                 half* k_scale_out, half* v_scale_out) {
  const int warp_id = tid / 32;
  const int lane = tid % 32;
  const int num_warps = blockDim.x / 32;
  float* k_buf = smem;
  float* v_buf = k_buf + head_dim;
  float* k_warp = v_buf + head_dim;
  float* v_warp = k_warp + num_warps;
  float* scales = v_warp + num_warps;

  float kabs = fabsf(k_buf[tid]);
  float vabs = fabsf(v_buf[tid]);
  for (int off = 16; off > 0; off >>= 1) {
    kabs = fmaxf(kabs, __shfl_down_sync(0xffffffffu, kabs, off));
    vabs = fmaxf(vabs, __shfl_down_sync(0xffffffffu, vabs, off));
  }
  if (lane == 0) {
    k_warp[warp_id] = kabs;
    v_warp[warp_id] = vabs;
  }
  __syncthreads();

  constexpr float kMaxQ = (KBits == 4) ? 7.0f : 127.0f;
  constexpr float vMaxQ = (VBits == 4) ? 7.0f : 127.0f;
  if (tid == 0) {
    float km = 0.0f, vm = 0.0f;
    for (int i = 0; i < num_warps; ++i) {
      km = fmaxf(km, k_warp[i]);
      vm = fmaxf(vm, v_warp[i]);
    }
    const float ks = (km > 0.0f) ? (km / kMaxQ) : 1.0f;
    const float vs = (vm > 0.0f) ? (vm / vMaxQ) : 1.0f;
    scales[0] = ks;
    scales[1] = vs;
    *k_scale_out = __float2half(ks);
    *v_scale_out = __float2half(vs);
  }
  __syncthreads();

  const float ks = scales[0];
  const float vs = scales[1];
  if (KBits == 4) {
    const int packed = head_dim / 2;
    if (tid < packed) {
      const int q0 = max(-8, min(7, __float2int_rn(k_buf[2 * tid] / ks)));
      const int q1 = max(-8, min(7, __float2int_rn(k_buf[2 * tid + 1] / ks)));
      k_row_out[tid] = static_cast<int8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  } else {
    const int qi = max(-127, min(127, __float2int_rn(k_buf[tid] / ks)));
    k_row_out[tid] = static_cast<int8_t>(qi);
  }
  if (VBits == 4) {
    const int packed = head_dim / 2;
    if (tid < packed) {
      const int q0 = max(-8, min(7, __float2int_rn(v_buf[2 * tid] / vs)));
      const int q1 = max(-8, min(7, __float2int_rn(v_buf[2 * tid + 1] / vs)));
      v_row_out[tid] = static_cast<int8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  } else {
    const int qi = max(-127, min(127, __float2int_rn(v_buf[tid] / vs)));
    v_row_out[tid] = static_cast<int8_t>(qi);
  }
}

}  // namespace kvq
}  // namespace kernels
