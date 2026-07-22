// kernels_ops_matvec.cu
//
// CUDA kernels and host launch wrappers for pointwise ops, quant/dequant,
// weight-only matvec, projection GEMV, and argmax helpers.

#include <cstdlib>
#include <cuda_fp16.h>
#include <math_constants.h>  // CUDART_INF_F (device top-k reductions)
#include <sm_61_intrinsics.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "runtime/kernels.cuh"

namespace kernels {
namespace {

inline int choose_reduction_threads(int cols) {
  return (cols <= 1024) ? 128 : 256;
}

__device__ __forceinline__ half bf16_bits_to_half(std::uint16_t bits) {
  const unsigned int u = static_cast<unsigned int>(bits) << 16;
  const float f = __uint_as_float(u);
  return __float2half(f);
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

__device__ __forceinline__ void warp_argmax(float& value, int& index) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffu, value, offset);
    const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (other_value > value) {
      value = other_value;
      index = other_index;
    }
  }
}

// Pointwise math, quantization, and dequantization kernels.
__global__ void add_inplace_kernel(half* x, const half* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  x[i] = __hadd(x[i], y[i]);
}

__global__ void add_inplace_half2_kernel(half2* x, const half2* y, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) {
    return;
  }
  x[i] = __hadd2(x[i], y[i]);
}

__global__ void silu_mul_kernel(const half* gate, const half* up, half* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float g = __half2float(gate[i]);
  const float u = __half2float(up[i]);
  const float s = g / (1.0f + expf(-g));
  out[i] = __float2half(s * u);
}

__global__ void silu_mul_half2_kernel(const half2* gate, const half2* up, half2* out, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) {
    return;
  }

  const half2 g2 = gate[i];
  const half2 u2 = up[i];
  const float g0 = __half2float(__low2half(g2));
  const float g1 = __half2float(__high2half(g2));
  const float u0 = __half2float(__low2half(u2));
  const float u1 = __half2float(__high2half(u2));
  const float s0 = g0 / (1.0f + expf(-g0));
  const float s1 = g1 / (1.0f + expf(-g1));
  out[i] = __halves2half2(__float2half(s0 * u0), __float2half(s1 * u1));
}

// GeGLU: gelu(gate) * up, using the tanh GELU approximation (Gemma's
// gelu_pytorch_tanh). Same fused gate*up shape as silu_mul, different nonlinearity.
__device__ __forceinline__ float gelu_tanh_f32(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// Gated GLU read straight off the fused gate+up buffer.
//
// The prefill MLP emits w13 as one [tokens, 2*inter] matrix. Reading gate at [t][i] and up at
// [t][inter+i] avoids splitting it into two buffers first -- those copies only un-interleave
// data this kernel reads elementwise anyway, and prefill is bound by host-side API calls, so a
// copy that buys nothing still costs.
//
// Elementwise math is unchanged, so this is bit-identical to split-then-glu.
__global__ void gated_glu_interleaved_kernel(const half* __restrict__ ff13, half* __restrict__ out,
                                             int inter, int tokens, int gelu) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = tokens * inter;
  if (idx >= total) {
    return;
  }
  const int t = idx / inter;
  const int i = idx - t * inter;
  const std::size_t base = static_cast<std::size_t>(t) * 2 * inter;
  const float g = __half2float(ff13[base + i]);
  const float u = __half2float(ff13[base + inter + i]);
  const float act = gelu ? gelu_tanh_f32(g) : (g / (1.0f + expf(-g)));
  out[idx] = __float2half(act * u);
}

__global__ void gelu_mul_kernel(const half* gate, const half* up, half* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = __float2half(gelu_tanh_f32(__half2float(gate[i])) * __half2float(up[i]));
}

__global__ void gelu_mul_half2_kernel(const half2* gate, const half2* up, half2* out, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) {
    return;
  }
  const half2 g2 = gate[i];
  const half2 u2 = up[i];
  const float o0 = gelu_tanh_f32(__half2float(__low2half(g2))) * __half2float(__low2half(u2));
  const float o1 = gelu_tanh_f32(__half2float(__high2half(g2))) * __half2float(__high2half(u2));
  out[i] = __halves2half2(__float2half(o0), __float2half(o1));
}

__global__ void apply_sigmoid_gate_inplace_kernel(half* values, const half* gate, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float v = __half2float(values[i]);
  const float g = __half2float(gate[i]);
  values[i] = __float2half(v * (1.0f / (1.0f + expf(-g))));
}

__global__ void split_interleaved_head_halves_kernel(const half* src, half* first, half* second,
                                                     int head_dim) {
  const int head = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= head_dim) {
    return;
  }
  const std::size_t src_base =
      static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim * 2);
  const std::size_t dst_base = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim);
  first[dst_base + static_cast<std::size_t>(d)] = src[src_base + static_cast<std::size_t>(d)];
  second[dst_base + static_cast<std::size_t>(d)] =
      src[src_base + static_cast<std::size_t>(head_dim + d)];
}

__device__ __forceinline__ float sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_f32(float x) {
  return x * sigmoid_f32(x);
}

__device__ __forceinline__ float softplus_f32(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

__global__ void qwen35_linear_conv1d_silu_kernel(const half* conv_weight, float* conv_state,
                                                 half* qkv_mix, int channels, int kernel_size) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }

  const int state_len = kernel_size - 1;
  const std::size_t weight_base =
      static_cast<std::size_t>(channel) * static_cast<std::size_t>(kernel_size);
  const float input = __half2float(qkv_mix[channel]);
  float out =
      __half2float(conv_weight[weight_base + static_cast<std::size_t>(kernel_size - 1)]) * input;
  if (state_len > 0) {
    float* state_row =
        conv_state + static_cast<std::size_t>(channel) * static_cast<std::size_t>(state_len);
    for (int j = 0; j < state_len; ++j) {
      out += __half2float(conv_weight[weight_base + static_cast<std::size_t>(j)]) * state_row[j];
    }
    for (int j = 0; j + 1 < state_len; ++j) {
      state_row[j] = state_row[j + 1];
    }
    state_row[state_len - 1] = input;
  }
  qkv_mix[channel] = __float2half(silu_f32(out));
}

__global__ void qwen35_repeat_linear_heads_kernel(const half* qkv_mix, half* q_out, half* k_out,
                                                  half* v_out, int num_key_heads, int head_repeat,
                                                  int key_head_dim, int value_head_dim) {
  const int key_head = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;

  const std::size_t q_base =
      static_cast<std::size_t>(key_head) * static_cast<std::size_t>(key_head_dim);
  const std::size_t k_base =
      static_cast<std::size_t>(num_key_heads) * static_cast<std::size_t>(key_head_dim) + q_base;
  const std::size_t v_src_base =
      static_cast<std::size_t>(num_key_heads * 2 * key_head_dim) +
      static_cast<std::size_t>(key_head * head_repeat) * static_cast<std::size_t>(value_head_dim);

  for (int rep = 0; rep < head_repeat; ++rep) {
    const int value_head = key_head * head_repeat + rep;
    const std::size_t dst_q_base =
        static_cast<std::size_t>(value_head) * static_cast<std::size_t>(key_head_dim);
    const std::size_t dst_k_base =
        static_cast<std::size_t>(value_head) * static_cast<std::size_t>(key_head_dim);
    const std::size_t dst_v_base =
        static_cast<std::size_t>(value_head) * static_cast<std::size_t>(value_head_dim);
    if (d < key_head_dim) {
      q_out[dst_q_base + static_cast<std::size_t>(d)] =
          qkv_mix[q_base + static_cast<std::size_t>(d)];
      k_out[dst_k_base + static_cast<std::size_t>(d)] =
          qkv_mix[k_base + static_cast<std::size_t>(d)];
    }
    if (d < value_head_dim) {
      v_out[dst_v_base + static_cast<std::size_t>(d)] =
          qkv_mix[v_src_base +
                  static_cast<std::size_t>(rep) * static_cast<std::size_t>(value_head_dim) +
                  static_cast<std::size_t>(d)];
    }
  }
}

__global__ void qwen35_linear_attention_step_kernel(const half* q, const half* k, const half* v,
                                                    const half* z, const half* a, const half* b,
                                                    const float* norm_weight, const float* a_log,
                                                    const half* dt_bias, float* recurrent_state,
                                                    half* out, int key_head_dim, int value_head_dim,
                                                    float rms_eps) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int state_stride = key_head_dim * value_head_dim;
  const std::size_t q_base =
      static_cast<std::size_t>(head) * static_cast<std::size_t>(key_head_dim);
  const std::size_t v_base =
      static_cast<std::size_t>(head) * static_cast<std::size_t>(value_head_dim);
  float* state =
      recurrent_state + static_cast<std::size_t>(head) * static_cast<std::size_t>(state_stride);

  extern __shared__ float shared[];
  float* q_shared = shared;
  float* k_shared = q_shared + key_head_dim;
  float* scratch = k_shared + key_head_dim;
  __shared__ float reduce_buf[256];
  __shared__ float q_norm_factor;
  __shared__ float k_norm_factor;
  __shared__ float beta_shared;
  __shared__ float decay_shared;
  __shared__ float out_inv_shared;

  float q_ss = 0.0f;
  float k_ss = 0.0f;
  for (int idx = tid; idx < key_head_dim; idx += blockDim.x) {
    const float qv = __half2float(q[q_base + static_cast<std::size_t>(idx)]);
    const float kv = __half2float(k[q_base + static_cast<std::size_t>(idx)]);
    q_ss += qv * qv;
    k_ss += kv * kv;
  }
  reduce_buf[tid] = q_ss;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_buf[tid] += reduce_buf[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    q_norm_factor = rsqrtf(reduce_buf[0] + 1.0e-6f) / sqrtf(static_cast<float>(key_head_dim));
  }
  __syncthreads();

  reduce_buf[tid] = k_ss;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_buf[tid] += reduce_buf[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    k_norm_factor = rsqrtf(reduce_buf[0] + 1.0e-6f);
    beta_shared = sigmoid_f32(__half2float(b[head]));
    decay_shared = expf(-expf(a_log[head]) *
                        softplus_f32(__half2float(a[head]) + __half2float(dt_bias[head])));
  }
  __syncthreads();

  for (int idx = tid; idx < key_head_dim; idx += blockDim.x) {
    q_shared[idx] = __half2float(q[q_base + static_cast<std::size_t>(idx)]) * q_norm_factor;
    k_shared[idx] = __half2float(k[q_base + static_cast<std::size_t>(idx)]) * k_norm_factor;
  }
  __syncthreads();

  for (int idx = tid; idx < state_stride; idx += blockDim.x) {
    state[idx] *= decay_shared;
  }
  __syncthreads();

  for (int dv = tid; dv < value_head_dim; dv += blockDim.x) {
    float kv_mem = 0.0f;
    for (int kd = 0; kd < key_head_dim; ++kd) {
      kv_mem += state[static_cast<std::size_t>(kd) * static_cast<std::size_t>(value_head_dim) +
                      static_cast<std::size_t>(dv)] *
                k_shared[kd];
    }
    scratch[dv] = (__half2float(v[v_base + static_cast<std::size_t>(dv)]) - kv_mem) * beta_shared;
  }
  __syncthreads();

  for (int idx = tid; idx < state_stride; idx += blockDim.x) {
    const int kd = idx / value_head_dim;
    const int dv = idx % value_head_dim;
    state[idx] += k_shared[kd] * scratch[dv];
  }
  __syncthreads();

  float out_ss = 0.0f;
  for (int dv = tid; dv < value_head_dim; dv += blockDim.x) {
    float sum = 0.0f;
    for (int kd = 0; kd < key_head_dim; ++kd) {
      sum += state[static_cast<std::size_t>(kd) * static_cast<std::size_t>(value_head_dim) +
                   static_cast<std::size_t>(dv)] *
             q_shared[kd];
    }
    scratch[dv] = sum;
    out_ss += sum * sum;
  }
  reduce_buf[tid] = out_ss;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_buf[tid] += reduce_buf[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_inv_shared = rsqrtf(reduce_buf[0] / static_cast<float>(value_head_dim) + rms_eps);
  }
  __syncthreads();

  for (int dv = tid; dv < value_head_dim; dv += blockDim.x) {
    const float gated = scratch[dv] * out_inv_shared * norm_weight[dv] *
                        silu_f32(__half2float(z[v_base + static_cast<std::size_t>(dv)]));
    out[v_base + static_cast<std::size_t>(dv)] = __float2half(gated);
  }
}

__global__ void scale_copy_kernel(half* dst, const half* src, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = __float2half(__half2float(src[i]) * scale);
}

__global__ void scale_add_inplace_kernel(half* dst, const half* src, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float acc = __half2float(dst[i]) + __half2float(src[i]) * scale;
  dst[i] = __float2half(acc);
}

__global__ void moe_router_topk_softmax_kernel(const half* logits, int experts, int top_k,
                                               int* topk_idx, float* topk_prob,
                                               const half* per_expert_scale) {
  extern __shared__ float probs[];
  if (threadIdx.x != 0) {
    return;
  }

  if (experts <= 0 || top_k <= 0) {
    return;
  }

  float max_logit = __half2float(logits[0]);
  for (int e = 1; e < experts; ++e) {
    max_logit = fmaxf(max_logit, __half2float(logits[e]));
  }

  float sum = 0.0f;
  for (int e = 0; e < experts; ++e) {
    const float p = expf(__half2float(logits[e]) - max_logit);
    probs[e] = p;
    sum += p;
  }
  const float inv_sum = 1.0f / fmaxf(sum, 1.0e-8f);
  for (int e = 0; e < experts; ++e) {
    probs[e] *= inv_sum;
  }

  constexpr int kMaxTopK = 8;
  int picked[kMaxTopK];
  float picked_prob[kMaxTopK];
  const int capped_topk = top_k > kMaxTopK ? kMaxTopK : top_k;
  for (int k = 0; k < capped_topk; ++k) {
    picked[k] = -1;
    picked_prob[k] = 0.0f;
  }

  for (int k = 0; k < capped_topk; ++k) {
    int best_e = -1;
    float best_p = -1.0f;
    for (int e = 0; e < experts; ++e) {
      bool used = false;
      for (int prev = 0; prev < k; ++prev) {
        if (picked[prev] == e) {
          used = true;
          break;
        }
      }
      if (used) {
        continue;
      }
      const float p = probs[e];
      if (p > best_p) {
        best_p = p;
        best_e = e;
      }
    }
    picked[k] = best_e;
    picked_prob[k] = best_p > 0.0f ? best_p : 0.0f;
  }

  float picked_sum = 0.0f;
  for (int k = 0; k < capped_topk; ++k) {
    picked_sum += picked_prob[k];
  }
  const float inv_pick_sum = 1.0f / fmaxf(picked_sum, 1.0e-8f);

  for (int k = 0; k < top_k; ++k) {
    if (k < capped_topk) {
      const int e = picked[k] >= 0 ? picked[k] : 0;
      topk_idx[k] = e;
      // A learned per-expert gain on the routing weight, applied after the top-k
      // renormalisation (Gemma MoE). Null for routers that do not have one, which
      // leaves the weight exactly as before.
      const float gain = per_expert_scale ? __half2float(per_expert_scale[e]) : 1.0f;
      topk_prob[k] = picked_prob[k] * inv_pick_sum * gain;
    } else {
      topk_idx[k] = 0;
      topk_prob[k] = 0.0f;
    }
  }
}

__global__ void convert_bf16_to_fp16_kernel(const std::uint16_t* src, half* dst, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx] = bf16_bits_to_half(src[idx]);
}

__global__ void dequant_int8_to_fp16_kernel(const int8_t* src, half* dst, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = __float2half(static_cast<float>(src[i]) * scale);
}

__global__ void dequant_rowwise_int8_to_fp16_kernel(const int8_t* src, const float* scales,
                                                    half* dst, int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = rows * cols;
  if (idx >= n) {
    return;
  }
  const int row = idx / cols;
  dst[idx] = __float2half(static_cast<float>(src[idx]) * scales[row]);
}

__global__ void quantize_rowwise_fp16_to_int8_kernel(const half* src, int8_t* dst, float* scales,
                                                     int cols, int max_q) {
  extern __shared__ float smax[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  const int base = row * cols;

  float local_max = 0.0f;
  if ((cols & 1) == 0) {
    const half2* src2 = reinterpret_cast<const half2*>(src + base);
    const int cols2 = cols / 2;
    for (int col2 = tid; col2 < cols2; col2 += blockDim.x) {
      const half2 v2 = src2[col2];
      local_max = fmaxf(local_max, fabsf(__half2float(__low2half(v2))));
      local_max = fmaxf(local_max, fabsf(__half2float(__high2half(v2))));
    }
  } else {
    for (int col = tid; col < cols; col += blockDim.x) {
      local_max = fmaxf(local_max, fabsf(__half2float(src[base + col])));
    }
  }

  local_max = warp_max(local_max);
  if (lane == 0) {
    smax[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max = (lane < warp_count) ? smax[lane] : 0.0f;
    block_max = warp_max(block_max);
    if (lane == 0) {
      const float max_q_f = static_cast<float>(max_q);
      float scale = block_max / max_q_f;
      if (scale < 1.0e-8f) {
        scale = 1.0e-8f;
      }
      scales[row] = scale;
      smax[0] = 1.0f / scale;
    }
  }
  __syncthreads();

  const float inv_scale = smax[0];
  if ((cols & 1) == 0) {
    const half2* src2 = reinterpret_cast<const half2*>(src + base);
    char2* dst2 = reinterpret_cast<char2*>(dst + base);
    const int cols2 = cols / 2;
    for (int col2 = tid; col2 < cols2; col2 += blockDim.x) {
      const half2 v2 = src2[col2];
      int q0 = __float2int_rn(__half2float(__low2half(v2)) * inv_scale);
      int q1 = __float2int_rn(__half2float(__high2half(v2)) * inv_scale);
      q0 = max(-max_q, min(max_q, q0));
      q1 = max(-max_q, min(max_q, q1));
      dst2[col2] = make_char2(static_cast<signed char>(q0), static_cast<signed char>(q1));
    }
  } else {
    for (int col = tid; col < cols; col += blockDim.x) {
      const float v = __half2float(src[base + col]) * inv_scale;
      int q = __float2int_rn(v);
      q = max(-max_q, min(max_q, q));
      dst[base + col] = static_cast<int8_t>(q);
    }
  }
}

// One block per (row, group). The group's max is reduced in shared memory, its
// scale written to scales[row, g], and the group's elements quantised with it.
// Deliberately separate from quantize_rowwise_fp16_to_int8_kernel rather than a
// generalisation of it: that kernel runs per-token inside LlamaEngine's captured
// decode graphs, and this one only ever runs at load time on weights.
__global__ void quantize_groupwise_fp16_to_int8_kernel(const half* src, int8_t* dst, float* scales,
                                                       int cols, int group, int n_groups,
                                                       int max_q) {
  extern __shared__ float smax[];
  const int row = blockIdx.x;
  const int g = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;

  const int col0 = g * group;
  const int col1 = min(col0 + group, cols);
  const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

  float local_max = 0.0f;
  for (int col = col0 + tid; col < col1; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__half2float(src[base + col])));
  }

  local_max = warp_max(local_max);
  if (lane == 0) {
    smax[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max = (lane < warp_count) ? smax[lane] : 0.0f;
    block_max = warp_max(block_max);
    if (lane == 0) {
      float scale = block_max / static_cast<float>(max_q);
      if (scale < 1.0e-8f) {
        scale = 1.0e-8f;
      }
      scales[static_cast<std::size_t>(row) * n_groups + g] = scale;
      smax[0] = 1.0f / scale;
    }
  }
  __syncthreads();

  const float inv_scale = smax[0];
  for (int col = col0 + tid; col < col1; col += blockDim.x) {
    int q = __float2int_rn(__half2float(src[base + col]) * inv_scale);
    q = max(-max_q, min(max_q, q));
    dst[base + col] = static_cast<int8_t>(q);
  }
}

__global__ void pack_rowwise_int8_to_int4_kernel(const int8_t* src, int8_t* dst, int rows,
                                                 int cols) {
  const int row = blockIdx.x;
  const int col2 = blockIdx.y * blockDim.x + threadIdx.x;
  const int packed_cols = (cols + 1) / 2;
  if (row >= rows || col2 >= packed_cols) {
    return;
  }

  const int col = col2 * 2;
  const std::size_t src_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  auto clamp_i4 = [](int v) -> std::uint8_t {
    const int q = max(-8, min(7, v));
    return static_cast<std::uint8_t>(q < 0 ? q + 16 : q);
  };

  const std::uint8_t lo = clamp_i4(static_cast<int>(src[src_base + static_cast<std::size_t>(col)]));
  std::uint8_t hi = 0;
  if (col + 1 < cols) {
    hi = clamp_i4(static_cast<int>(src[src_base + static_cast<std::size_t>(col + 1)]));
  }

  const std::size_t dst_base =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  dst[dst_base + static_cast<std::size_t>(col2)] = static_cast<int8_t>(lo | (hi << 4));
}

// Weight-only int8 GEMV kernels.

// Weight-only int8/int4 matvec kernels moved to kernels_weight_only_matvec.cu.

template <typename OutT>
__device__ __forceinline__ void store_projection_value(OutT* y, int row, float value);

template <>
__device__ __forceinline__ void store_projection_value<half>(half* y, int row, float value) {
  y[row] = __float2half(value);
}

template <>
__device__ __forceinline__ void store_projection_value<float>(float* y, int row, float value) {
  y[row] = value;
}

// Shared-memory half2 GEMV used by resident projection layers and the LM head.
// `residual` (optional): fuse the residual add into the epilogue --
//     residual[row] = __hadd(residual[row], (half)dot)
// instead of storing the dot and running a separate add_inplace kernel over it.
// BYTE-IDENTICAL: the dot is rounded to fp16 first (exactly what the unfused store does)
// and combined with __hadd (exactly what add_inplace does). At batch 1 a kernel costs a
// fixed ~1.7 us however little it does, so deleting the add is worth more than the add.
template <int WarpsPerBlock, int TilePairs, int RowsPerWarp, typename OutT>
__global__ void rowmajor_half_gemv_kernel(const half* w, const half* x, OutT* y, int out_features,
                                          int in_features, half* residual = nullptr) {
  static_assert(RowsPerWarp >= 1, "RowsPerWarp must be >= 1");
  __shared__ half2 x_tile[TilePairs];
  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const int row_base = (blockIdx.x * WarpsPerBlock + warp_id) * RowsPerWarp;
  if (row_base >= out_features) {
    return;
  }

  float local[RowsPerWarp] = {0.0f};
  const half* row_ptrs[RowsPerWarp];
#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    const int row = row_base + r;
    row_ptrs[r] = (row < out_features)
                      ? (w + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features))
                      : nullptr;
  }
  if ((in_features & 1) == 0) {
    const int pairs = in_features / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* row_ptrs2[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
      row_ptrs2[r] = row_ptrs[r] ? reinterpret_cast<const half2*>(row_ptrs[r]) : nullptr;
    }
    for (int tile_base = 0; tile_base < pairs; tile_base += TilePairs) {
      const int tile_count = min(TilePairs, pairs - tile_base);
      // Use min(blockDim.x, TilePairs) as stride so all threads participate
      // even when blockDim.x > TilePairs (e.g. warps=8, tile=128).
      for (int idx = threadIdx.x; idx < tile_count; idx += blockDim.x) {
        x_tile[idx] = x2[tile_base + idx];
      }
      __syncthreads();

      for (int idx = lane; idx < tile_count; idx += warpSize) {
        const float2 xv = __half22float2(x_tile[idx]);
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
          if (row_ptrs2[r]) {
            const float2 wv = __half22float2(row_ptrs2[r][tile_base + idx]);
            local[r] += wv.x * xv.x + wv.y * xv.y;
          }
        }
      }
      __syncthreads();
    }
  } else {
    for (int col = lane; col < in_features; col += warpSize) {
      const float xv = __half2float(x[col]);
#pragma unroll
      for (int r = 0; r < RowsPerWarp; ++r) {
        if (row_ptrs[r]) {
          local[r] += __half2float(row_ptrs[r][col]) * xv;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    local[r] = warp_sum(local[r]);
    if (lane == 0) {
      const int row = row_base + r;
      if (row < out_features) {
        if (residual != nullptr) {
          residual[row] = __hadd(residual[row], __float2half(local[r]));
        } else {
          store_projection_value<OutT>(y, row, local[r]);
        }
      }
    }
  }
}

// Single-block argmax for the final logits vector.
// Phase 1: each block takes a contiguous slice of the logits and reduces it to one
// (value, index) pair. Phase 2: one block reduces the per-block pairs. Together these do the
// same job as argmax_float_kernel with the whole GPU instead of a single SM.
//
// Tie-break: `>` (strictly greater) everywhere, exactly as the single-block kernel, so a
// duplicated maximum resolves the same way it always did within a slice. Across slices the
// decomposition differs, so an EXACT tie between two equal maxima in different blocks could
// resolve differently -- the greedy 6-model gate is what proves this does not bite.
__global__ void argmax_partial_kernel(const float* __restrict__ logits, int n,
                                      float* __restrict__ part_val, int* __restrict__ part_idx) {
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  __shared__ float warp_max[32];
  __shared__ int warp_idx[32];

  const int chunk = (n + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * chunk;
  const int end = min(begin + chunk, n);

  float local_max = -3.402823466e+38F;
  int local_idx = begin;
  for (int i = begin + tid; i < end; i += blockDim.x) {
    const float v = logits[i];
    if (v > local_max) {
      local_max = v;
      local_idx = i;
    }
  }

  warp_argmax(local_max, local_idx);
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_idx[warp] = local_idx;
  }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38F;
    int i = (lane < warp_count) ? warp_idx[lane] : begin;
    warp_argmax(v, i);
    if (lane == 0) {
      part_val[blockIdx.x] = v;
      part_idx[blockIdx.x] = i;
    }
  }
}

__global__ void argmax_reduce_kernel(const float* __restrict__ part_val,
                                     const int* __restrict__ part_idx, int parts,
                                     int* __restrict__ out_index) {
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  __shared__ float warp_max[32];
  __shared__ int warp_idx[32];

  float local_max = -3.402823466e+38F;
  int local_idx = 0;
  for (int i = tid; i < parts; i += blockDim.x) {
    const float v = part_val[i];
    if (v > local_max) {
      local_max = v;
      local_idx = part_idx[i];
    }
  }

  warp_argmax(local_max, local_idx);
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_idx[warp] = local_idx;
  }
  __syncthreads();

  if (warp == 0) {
    float v = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38F;
    int i = (lane < warp_count) ? warp_idx[lane] : 0;
    warp_argmax(v, i);
    if (lane == 0) {
      *out_index = i;
    }
  }
}

__global__ void argmax_float_kernel(const float* logits, int n, int* out_index) {
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  __shared__ float warp_max[32];
  __shared__ int warp_idx[32];

  float local_max = -3.402823466e+38F;
  int local_idx = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    const float v = logits[i];
    if (v > local_max) {
      local_max = v;
      local_idx = i;
    }
  }

  warp_argmax(local_max, local_idx);
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_idx[warp] = local_idx;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38F;
    int block_idx = (lane < warp_count) ? warp_idx[lane] : 0;
    warp_argmax(block_max, block_idx);
    if (lane == 0) {
      *out_index = block_idx;
    }
  }
}
}  // namespace

// Host launch wrappers for pointwise/quant/matvec/projection kernels.
__global__ void add_bias_broadcast_kernel(half* out, const half* bias, int rows, int cols) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (col < cols) {
    out[row * cols + col] = __hadd(out[row * cols + col], bias[col]);
  }
}

__global__ void add_bias_inplace_float_from_half_kernel(float* out, const half* bias, int n) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (i < n) {
    out[i] += __half2float(bias[i]);
  }
}

void launch_add_bias_broadcast(half* out, const half* bias, int rows, int cols,
                               cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  constexpr int threads = 256;
  const dim3 grid(static_cast<unsigned>((cols + threads - 1) / threads),
                  static_cast<unsigned>(rows));
  add_bias_broadcast_kernel<<<grid, threads, 0, stream>>>(out, bias, rows, cols);
}

void launch_add_bias_inplace_float_from_half(float* out, const half* bias, int n,
                                             cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  add_bias_inplace_float_from_half_kernel<<<blocks, threads, 0, stream>>>(out, bias, n);
}

void launch_add_inplace(half* x, const half* y, int n, cudaStream_t stream) {
  constexpr int threads = 256;
  const bool aligned =
      ((reinterpret_cast<std::uintptr_t>(x) | reinterpret_cast<std::uintptr_t>(y)) &
       (alignof(half2) - 1)) == 0;
  if ((n & 1) == 0 && aligned) {
    const int n2 = n / 2;
    const int blocks = (n2 + threads - 1) / threads;
    add_inplace_half2_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<half2*>(x),
                                                             reinterpret_cast<const half2*>(y), n2);
    return;
  }

  const int blocks = (n + threads - 1) / threads;
  add_inplace_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}

void launch_gated_glu_interleaved(const half* ff13, half* out, int inter, int tokens, bool gelu,
                                  cudaStream_t stream) {
  const int total = tokens * inter;
  if (total <= 0) {
    return;
  }
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  gated_glu_interleaved_kernel<<<blocks, kThreads, 0, stream>>>(ff13, out, inter, tokens,
                                                                gelu ? 1 : 0);
}

void launch_silu_mul(const half* gate, const half* up, half* out, int n, cudaStream_t stream) {
  constexpr int threads = 256;
  const bool aligned =
      ((reinterpret_cast<std::uintptr_t>(gate) | reinterpret_cast<std::uintptr_t>(up) |
        reinterpret_cast<std::uintptr_t>(out)) &
       (alignof(half2) - 1)) == 0;
  if ((n & 1) == 0 && aligned) {
    const int n2 = n / 2;
    const int blocks = (n2 + threads - 1) / threads;
    silu_mul_half2_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half2*>(gate),
                                                          reinterpret_cast<const half2*>(up),
                                                          reinterpret_cast<half2*>(out), n2);
    return;
  }

  const int blocks = (n + threads - 1) / threads;
  silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

void launch_gelu_mul(const half* gate, const half* up, half* out, int n, cudaStream_t stream) {
  constexpr int threads = 256;
  const bool aligned =
      ((reinterpret_cast<std::uintptr_t>(gate) | reinterpret_cast<std::uintptr_t>(up) |
        reinterpret_cast<std::uintptr_t>(out)) &
       (alignof(half2) - 1)) == 0;
  if ((n & 1) == 0 && aligned) {
    const int n2 = n / 2;
    const int blocks = (n2 + threads - 1) / threads;
    gelu_mul_half2_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half2*>(gate),
                                                          reinterpret_cast<const half2*>(up),
                                                          reinterpret_cast<half2*>(out), n2);
    return;
  }
  const int blocks = (n + threads - 1) / threads;
  gelu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

void launch_scale_copy(half* dst, const half* src, int n, float scale, cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scale_copy_kernel<<<blocks, threads, 0, stream>>>(dst, src, n, scale);
}

void launch_scale_add_inplace(half* dst, const half* src, int n, float scale, cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scale_add_inplace_kernel<<<blocks, threads, 0, stream>>>(dst, src, n, scale);
}

void launch_moe_router_topk_softmax(const half* logits, int experts, int top_k, int* topk_idx,
                                    float* topk_prob, cudaStream_t stream,
                                    const half* per_expert_scale) {
  if (experts <= 0 || top_k <= 0) {
    return;
  }
  const std::size_t smem = static_cast<std::size_t>(experts) * sizeof(float);
  moe_router_topk_softmax_kernel<<<1, 32, smem, stream>>>(logits, experts, top_k, topk_idx,
                                                          topk_prob, per_expert_scale);
}

void launch_dequant_int8_to_fp16(const int8_t* src, half* dst, int n, float scale,
                                 cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  dequant_int8_to_fp16_kernel<<<blocks, threads, 0, stream>>>(src, dst, n, scale);
}

void launch_dequant_rowwise_int8_to_fp16(const int8_t* src, const float* scales, half* dst,
                                         int rows, int cols, cudaStream_t stream) {
  constexpr int threads = 256;
  const int n = rows * cols;
  const int blocks = (n + threads - 1) / threads;
  dequant_rowwise_int8_to_fp16_kernel<<<blocks, threads, 0, stream>>>(src, scales, dst, rows, cols);
}

void launch_quantize_rowwise_fp16_to_int8(const half* src, int8_t* dst, float* scales, int rows,
                                          int cols, cudaStream_t stream, int max_q) {
  if (max_q <= 0) {
    max_q = 127;
  }
  const int threads = choose_reduction_threads(cols);
  constexpr int kWarpSize = 32;
  const int warp_count = (threads + kWarpSize - 1) / kWarpSize;
  quantize_rowwise_fp16_to_int8_kernel<<<
      rows, threads, static_cast<std::size_t>(warp_count) * sizeof(float), stream>>>(
      src, dst, scales, cols, max_q);
}

void launch_quantize_groupwise_fp16_to_int8(const half* src, int8_t* dst, float* scales, int rows,
                                            int cols, int group, cudaStream_t stream, int max_q) {
  if (max_q <= 0) {
    max_q = 127;
  }
  if (group <= 0 || group > cols) {
    group = cols;  // degenerates to one scale per row
  }
  const int n_groups = quant_group_count(cols, group);
  const int threads = choose_reduction_threads(group);
  constexpr int kWarpSize = 32;
  const int warp_count = (threads + kWarpSize - 1) / kWarpSize;
  const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(n_groups));
  quantize_groupwise_fp16_to_int8_kernel<<<
      grid, threads, static_cast<std::size_t>(warp_count) * sizeof(float), stream>>>(
      src, dst, scales, cols, group, n_groups, max_q);
}

void launch_pack_rowwise_int8_to_int4(const int8_t* src, int8_t* dst, int rows, int cols,
                                      cudaStream_t stream) {
  constexpr int threads = 256;
  const int packed_cols = (cols + 1) / 2;
  const dim3 grid(static_cast<unsigned int>(rows),
                  static_cast<unsigned int>((packed_cols + threads - 1) / threads));
  pack_rowwise_int8_to_int4_kernel<<<grid, threads, 0, stream>>>(src, dst, rows, cols);
}

// GEMV shaped for streaming weights from HBM: one warp per output row, 128-bit loads, no
// shared tile.
//
// The tiled kernel above reads the weight row as half2 -- 32 bits per thread. That is fine
// out of L2, but decode streams each weight from HBM exactly once, and 32-bit requests leave
// far too few bytes in flight to cover the latency. Reading int4 (8 halves) quarters the
// request count and keeps several loads in flight per thread. The shared staging of x is
// dropped: x is a few KB, every block reads it, and it is served from cache anyway.
//
// Not bit-identical to the tiled kernel: each lane accumulates a different subset of the row,
// so the fp32 summation order differs. Values agree to fp32 rounding.
template <int Warps, typename OutT>
__global__ void gemv_wide_kernel(const half* __restrict__ w, const half* __restrict__ x,
                                 OutT* __restrict__ y, int out_features, int in_features,
                                 half* __restrict__ residual) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  if (row >= out_features) {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int vecs = in_features / 8;  // 8 halves per int4
  const int4* wrow = reinterpret_cast<const int4*>(
      w + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features));
  const int4* xv = reinterpret_cast<const int4*>(x);

  // Prefetch kUnroll chunks before consuming any of them.
  //
  // Loading one int4 and immediately using it keeps only ~1 load in flight per thread: `vecs`
  // is a runtime value, so nvcc cannot unroll the loop, and each iteration's math depends on
  // that iteration's load. Issuing several loads up front turns latency into throughput.
  //
  // The accumulation order is unchanged (lane, lane+32, lane+64, ...), so this is bit-identical
  // to the un-prefetched kernel -- purely a scheduling change.
  constexpr int kUnroll = 4;
  float acc = 0.0f;
  int4 wbuf[kUnroll];
  int4 xbuf[kUnroll];
  for (int base = lane; base < vecs; base += warpSize * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = base + u * warpSize;
      if (i < vecs) {
        wbuf[u] = wrow[i];
        xbuf[u] = xv[i];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = base + u * warpSize;
      if (i < vecs) {
        const half2* wh = reinterpret_cast<const half2*>(&wbuf[u]);
        const half2* xh = reinterpret_cast<const half2*>(&xbuf[u]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 a = __half22float2(wh[j]);
          const float2 b = __half22float2(xh[j]);
          acc += a.x * b.x + a.y * b.y;
        }
      }
    }
  }
  // Tail for in_features not divisible by 8.
  for (int i = vecs * 8 + lane; i < in_features; i += warpSize) {
    acc += __half2float(w[static_cast<std::size_t>(row) * in_features + i]) * __half2float(x[i]);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if (residual != nullptr) {
      residual[row] = __hadd(residual[row], __float2half(acc));
    } else {
      store_projection_value<OutT>(y, row, acc);
    }
  }
}

// Segment-routed gemv_wide_kernel: up to three projections sharing one input vector run as
// ONE launch (q|k|v, gate|up). A warp resolves its global row to (matrix, output, local row)
// and then runs gemv_wide_kernel's exact inner loop -- same lane stride, same unroll, same
// warp_sum, same fp16 rounding -- so a cat'd launch is bit-identical to the split launches;
// only the grid packing differs. Warp-per-row with no shared memory, so partial trailing
// blocks need no cross-warp care.
template <int Warps>
__global__ void gemv_wide_cat_kernel(const half* __restrict__ w0, half* __restrict__ y0, int n0,
                                     const half* __restrict__ w1, half* __restrict__ y1, int n1,
                                     const half* __restrict__ w2, half* __restrict__ y2, int n2,
                                     const half* __restrict__ x, int in_features) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  const half* w = nullptr;
  half* y = nullptr;
  int r = row;
  if (row < n0) {
    w = w0;
    y = y0;
  } else if (row < n0 + n1) {
    r = row - n0;
    w = w1;
    y = y1;
  } else if (row < n0 + n1 + n2) {
    r = row - n0 - n1;
    w = w2;
    y = y2;
  } else {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int vecs = in_features / 8;
  const int4* wrow =
      reinterpret_cast<const int4*>(w + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features));
  const int4* xv = reinterpret_cast<const int4*>(x);

  constexpr int kUnroll = 4;
  float acc = 0.0f;
  int4 wbuf[kUnroll];
  int4 xbuf[kUnroll];
  for (int base = lane; base < vecs; base += warpSize * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = base + u * warpSize;
      if (i < vecs) {
        wbuf[u] = wrow[i];
        xbuf[u] = xv[i];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = base + u * warpSize;
      if (i < vecs) {
        const half2* wh = reinterpret_cast<const half2*>(&wbuf[u]);
        const half2* xh = reinterpret_cast<const half2*>(&xbuf[u]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 a = __half22float2(wh[j]);
          const float2 b = __half22float2(xh[j]);
          acc += a.x * b.x + a.y * b.y;
        }
      }
    }
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    y[r] = __float2half(acc);
  }
}

// Opt-out, not opt-in: the wide kernel is the default because the profile says the tiled one
// is 4x off. LLAMA_INFER_TILED_GEMV=1 restores the old kernel for A/B and bisection.
static bool use_wide_gemv() {
  static const bool v = [] {
    const char* e = std::getenv("LLAMA_INFER_TILED_GEMV");
    return !(e && *e == '1');
  }();
  return v;
}

template <typename OutT>
static void launch_rowmajor_half_gemv(const half* w, const half* x, OutT* y, int out_features,
                                      int in_features, cudaStream_t stream, int warps_per_block,
                                      int tile_pairs, int rows_per_warp,
                                      half* residual = nullptr) {
  if (use_wide_gemv() && (in_features % 8) == 0) {
    // Do not route the short shapes here to the split-K kernel: it measured slower end to end.
    // The shared-memory reduction costs more than the extra parallelism buys at these sizes,
    // and the tuner still picks cuBLAS for wqkv/wo regardless. Their remaining gap to roofline
    // is a kernel-duration floor, not an occupancy problem.
    constexpr int kWarps = 4;
    const int blocks = (out_features + kWarps - 1) / kWarps;
    gemv_wide_kernel<kWarps, OutT>
        <<<blocks, kWarps * 32, 0, stream>>>(w, x, y, out_features, in_features, residual);
    return;
  }
  const int warps = (warps_per_block > 0) ? warps_per_block : ((out_features >= 8192) ? 8 : 4);
  const int tile = (tile_pairs > 0) ? tile_pairs : 128;
  const int rows = (rows_per_warp > 0) ? rows_per_warp : 1;
  auto launch = [&](auto warps_tag, auto tile_tag, auto rows_tag) {
    constexpr int kWarps = decltype(warps_tag)::value;
    constexpr int kTile = decltype(tile_tag)::value;
    constexpr int kRows = decltype(rows_tag)::value;
    constexpr int kThreads = kWarps * 32;
    const int blocks = (out_features + kWarps * kRows - 1) / (kWarps * kRows);
    rowmajor_half_gemv_kernel<kWarps, kTile, kRows>
        <<<blocks, kThreads, 0, stream>>>(w, x, y, out_features, in_features, residual);
  };
  auto launch_rows = [&](auto warps_tag, auto tile_tag) {
    if (rows >= 2) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 2>{});
    } else {
      launch(warps_tag, tile_tag, std::integral_constant<int, 1>{});
    }
  };

  if (warps >= 16) {
    if (tile >= 512) {
      launch_rows(std::integral_constant<int, 16>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_rows(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{});
    } else {
      launch_rows(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (warps >= 8) {
    if (tile >= 512) {
      launch_rows(std::integral_constant<int, 8>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_rows(std::integral_constant<int, 8>{}, std::integral_constant<int, 256>{});
    } else {
      launch_rows(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (tile >= 512) {
    launch_rows(std::integral_constant<int, 4>{}, std::integral_constant<int, 512>{});
  } else if (tile >= 256) {
    launch_rows(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
  } else {
    launch_rows(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
  }
}

void launch_apply_sigmoid_gate_inplace(half* values, const half* gate, int n, cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  apply_sigmoid_gate_inplace_kernel<<<blocks, threads, 0, stream>>>(values, gate, n);
}

void launch_split_interleaved_head_halves(const half* src, half* first, half* second, int heads,
                                          int head_dim, cudaStream_t stream) {
  constexpr int threads = 256;
  const dim3 grid(static_cast<unsigned int>(heads),
                  static_cast<unsigned int>((head_dim + threads - 1) / threads));
  split_interleaved_head_halves_kernel<<<grid, threads, 0, stream>>>(src, first, second, head_dim);
}

void launch_linear_conv1d_silu(const half* conv_weight, float* conv_state, half* qkv_mix,
                                      int channels, int kernel_size, cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (channels + threads - 1) / threads;
  qwen35_linear_conv1d_silu_kernel<<<blocks, threads, 0, stream>>>(conv_weight, conv_state, qkv_mix,
                                                                   channels, kernel_size);
}

void launch_repeat_linear_heads(const half* qkv_mix, half* q_out, half* k_out, half* v_out,
                                       int num_key_heads, int num_value_heads, int key_head_dim,
                                       int value_head_dim, cudaStream_t stream) {
  constexpr int threads = 256;
  const int head_repeat = num_value_heads / num_key_heads;
  const int width = key_head_dim > value_head_dim ? key_head_dim : value_head_dim;
  const dim3 grid(static_cast<unsigned int>(num_key_heads),
                  static_cast<unsigned int>((width + threads - 1) / threads));
  qwen35_repeat_linear_heads_kernel<<<grid, threads, 0, stream>>>(
      qkv_mix, q_out, k_out, v_out, num_key_heads, head_repeat, key_head_dim, value_head_dim);
}

void launch_linear_attention_step(const half* q, const half* k, const half* v, const half* z,
                                         const half* a, const half* b, const float* norm_weight,
                                         const float* a_log, const half* dt_bias,
                                         float* recurrent_state, half* out, int num_heads,
                                         int key_head_dim, int value_head_dim, float rms_eps,
                                         cudaStream_t stream) {
  int threads = key_head_dim > value_head_dim ? key_head_dim : value_head_dim;
  if (threads < 32) {
    threads = 32;
  } else if (threads > 256) {
    threads = 256;
  }
  const std::size_t shared_bytes =
      static_cast<std::size_t>(key_head_dim * 2 + value_head_dim) * sizeof(float);
  qwen35_linear_attention_step_kernel<<<num_heads, threads, shared_bytes, stream>>>(
      q, k, v, z, a, b, norm_weight, a_log, dt_bias, recurrent_state, out, key_head_dim,
      value_head_dim, rms_eps);
}

void launch_rowmajor_half_gemv_f16(const half* w, const half* x, half* y, int out_features,
                                   int in_features, cudaStream_t stream, int warps_per_block,
                                   int tile_pairs, int rows_per_warp, half* residual) {
  launch_rowmajor_half_gemv(w, x, y, out_features, in_features, stream, warps_per_block, tile_pairs,
                            rows_per_warp, residual);
}

void launch_rowmajor_half_gemv_cat(const half* w0, half* y0, int n0, const half* w1, half* y1,
                                   int n1, const half* w2, half* y2, int n2, const half* x,
                                   int in_features, cudaStream_t stream) {
  constexpr int kWarps = 4;
  const int total = n0 + n1 + n2;
  const int blocks = (total + kWarps - 1) / kWarps;
  gemv_wide_cat_kernel<kWarps>
      <<<blocks, kWarps * 32, 0, stream>>>(w0, y0, n0, w1, y1, n1, w2, y2, n2, x, in_features);
}

void launch_rowmajor_half_gemv_f32(const half* w, const half* x, float* y, int out_features,
                                   int in_features, cudaStream_t stream, int warps_per_block,
                                   int tile_pairs, int rows_per_warp) {
  launch_rowmajor_half_gemv(w, x, y, out_features, in_features, stream, warps_per_block, tile_pairs,
                            rows_per_warp);
}

int argmax_partition_count(int n) {
  constexpr int kItemsPerBlock = 256 * 8;
  const int parts = (n + kItemsPerBlock - 1) / kItemsPerBlock;
  return (parts < 1) ? 1 : ((parts > kArgmaxMaxParts) ? kArgmaxMaxParts : parts);
}

void launch_argmax_float(const float* logits, int n, int* out_index, cudaStream_t stream,
                         float* part_val, int* part_idx, int parts) {
  constexpr int threads = 256;
  // Two-phase when the caller supplies scratch. The single-block kernel below launches
  // <<<1, 256>>> over the WHOLE vocab -- one SM out of 170, each thread walking ~594 logits
  // in a serial dependent max-chain. It measured 149.6 us per token on Qwen2.5's 151936-wide
  // head: 12% of all GPU time, to read 608 KB that peak bandwidth would deliver in 0.34 us.
  // It cost as much as the LM head GEMV, which reads 272 MB.
  if (part_val != nullptr && part_idx != nullptr && parts > 1) {
    argmax_partial_kernel<<<parts, threads, 0, stream>>>(logits, n, part_val, part_idx);
    argmax_reduce_kernel<<<1, threads, 0, stream>>>(part_val, part_idx, parts, out_index);
    return;
  }
  argmax_float_kernel<<<1, threads, 0, stream>>>(logits, n, out_index);
}

// â”€â”€ device top-k candidate selection (sampled decode) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
namespace {

constexpr int kTopkChunk = 2048;    // elements per partition block (8 KB shared)
constexpr int kTopkThreads = 256;

// Block-wide argmax over a shared buffer; returns {value,index} on thread 0 and
// masks the winner so the next iteration finds the next-largest.
template <int THREADS>
__device__ __forceinline__ void block_argmax_mask(float* s, int len, float* red_v, int* red_i,
                                                  float* out_v, int* out_i) {
  float bv = -CUDART_INF_F;
  int bi = -1;
  for (int i = threadIdx.x; i < len; i += THREADS) {
    const float v = s[i];
    if (v > bv) { bv = v; bi = i; }
  }
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, bv, off);
    const int oi = __shfl_down_sync(0xffffffffu, bi, off);
    if (ov > bv) { bv = ov; bi = oi; }
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mv = -CUDART_INF_F;
    int mi = -1;
    for (int w = 0; w < THREADS / 32; ++w) {
      if (red_v[w] > mv) { mv = red_v[w]; mi = red_i[w]; }
    }
    *out_v = mv;
    *out_i = mi;
    if (mi >= 0) s[mi] = -CUDART_INF_F;  // mask for the next iteration
  }
  __syncthreads();
}

// Phase 1: each block takes the top-k of its own chunk (kept in shared memory).
__global__ void topk_partial_kernel(const float* __restrict__ logits, int n, int k,
                                    float* __restrict__ part_val, int* __restrict__ part_idx) {
  __shared__ float s[kTopkChunk];
  __shared__ float red_v[kTopkThreads / 32];
  __shared__ int red_i[kTopkThreads / 32];
  __shared__ float win_v;
  __shared__ int win_i;

  const int base = blockIdx.x * kTopkChunk;
  for (int i = threadIdx.x; i < kTopkChunk; i += kTopkThreads) {
    const int g = base + i;
    s[i] = (g < n) ? logits[g] : -CUDART_INF_F;
  }
  __syncthreads();

  for (int t = 0; t < k; ++t) {
    block_argmax_mask<kTopkThreads>(s, kTopkChunk, red_v, red_i, &win_v, &win_i);
    if (threadIdx.x == 0) {
      const int slot = blockIdx.x * k + t;
      part_val[slot] = win_v;
      part_idx[slot] = (win_i >= 0) ? (base + win_i) : -1;
    }
    __syncthreads();
  }
}

// Phase 2: one block merges the per-block partials into the final top-k.
__global__ void topk_merge_kernel(float* __restrict__ part_val, const int* __restrict__ part_idx,
                                  int m, int k, float* __restrict__ out_val,
                                  int* __restrict__ out_idx) {
  __shared__ float red_v[kTopkThreads / 32];
  __shared__ int red_i[kTopkThreads / 32];
  __shared__ float win_v;
  __shared__ int win_i;
  for (int t = 0; t < k; ++t) {
    // Same reduction, but masking the (global) partial buffer rather than shared.
    block_argmax_mask<kTopkThreads>(part_val, m, red_v, red_i, &win_v, &win_i);
    if (threadIdx.x == 0) {
      out_val[t] = win_v;
      out_idx[t] = (win_i >= 0) ? part_idx[win_i] : -1;
    }
    __syncthreads();
  }
}

// Gathers every finite logit >= *threshold. Reproduces the host sampler's
// candidate set exactly, ties at the k-th value included.
__global__ void gather_ge_threshold_kernel(const float* __restrict__ logits, int n,
                                           const float* __restrict__ threshold,
                                           int* __restrict__ out_idx, float* __restrict__ out_val,
                                           int* __restrict__ out_count, int capacity) {
  const float thr = *threshold;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float v = logits[i];
    if (!isfinite(v) || v < thr) continue;
    const int slot = atomicAdd(out_count, 1);
    if (slot < capacity) {
      out_idx[slot] = i;
      out_val[slot] = v;
    }
  }
}

}  // namespace

int topk_partition_count(int n) { return (n + kTopkChunk - 1) / kTopkChunk; }

void launch_topk_float(const float* logits, int n, int k, float* part_val, int* part_idx,
                       float* out_val, int* out_idx, cudaStream_t stream) {
  const int blocks = topk_partition_count(n);
  topk_partial_kernel<<<blocks, kTopkThreads, 0, stream>>>(logits, n, k, part_val, part_idx);
  topk_merge_kernel<<<1, kTopkThreads, 0, stream>>>(part_val, part_idx, blocks * k, k, out_val,
                                                    out_idx);
}

void launch_gather_ge_threshold(const float* logits, int n, const float* threshold, int* out_idx,
                                float* out_val, int* out_count, int capacity,
                                cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = min(1024, (n + threads - 1) / threads);
  gather_ge_threshold_kernel<<<blocks, threads, 0, stream>>>(logits, n, threshold, out_idx, out_val,
                                                             out_count, capacity);
}

void launch_convert_bf16_to_fp16(const std::uint16_t* src, half* dst, int n, cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  convert_bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
}

}  // namespace kernels
