// Linear-attention (delta-net) kernels -- the Qwen3.5 family.
//
// These are the Metal side of the six ops the CUDA plan engine already runs: SplitHeadHalves,
// SigmoidGate, LinearConv1d, RepeatLinearHeads, MulVec and LinearAttentionStep. The arithmetic
// mirrors kernels_ops_matvec.cu exactly, including where it accumulates in fp32 and where it
// rounds to half, so the two backends can be compared token-for-token rather than approximately.
//
// Two of them carry persistent state across decode steps, which nothing else in this backend
// does: the conv1d keeps the last kernel_size-1 inputs per channel, and the attention step keeps
// a [key_head_dim x value_head_dim] matrix per head. Both are fp32 and both are read-modify-write
// within a single step, so neither can be replayed and neither may run twice on the same token.

#include <metal_stdlib>
using namespace metal;

struct LinSplitParams {
  uint heads;
  uint head_dim;
};

struct LinGateParams {
  uint n;
};

struct LinMulVecParams {
  uint n;
  float scale;
};

struct LinConvParams {
  uint channels;
  uint kernel_size;
};

struct LinRepeatParams {
  uint num_key_heads;
  uint head_repeat;
  uint key_head_dim;
  uint value_head_dim;
};

struct LinAttnParams {
  uint key_head_dim;
  uint value_head_dim;
  float rms_eps;
};

inline float lin_sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}
inline float lin_silu(float x) {
  return x * lin_sigmoid(x);
}
// Matches the CUDA softplus, saturating branches included: without them exp() overflows to inf
// for large positive x and the decay below becomes NaN rather than 0.
inline float lin_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return exp(x);
  return log(1.0f + exp(x));
}

// src is [head][2 * head_dim] with the two halves laid out back to back; split them into two
// [head][head_dim] tensors.
kernel void cpi_split_head_halves(device const half* src [[buffer(0)]],
                                  device half* first [[buffer(1)]],
                                  device half* second [[buffer(2)]],
                                  constant LinSplitParams& p [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
  const uint n = p.heads * p.head_dim;
  if (gid >= n) return;
  const uint head = gid / p.head_dim;
  const uint d = gid % p.head_dim;
  const uint src_base = head * 2u * p.head_dim;
  first[gid] = src[src_base + d];
  second[gid] = src[src_base + p.head_dim + d];
}

kernel void cpi_sigmoid_gate_inplace(device half* values [[buffer(0)]],
                                     device const half* gate [[buffer(1)]],
                                     constant LinGateParams& p [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  values[gid] = half(float(values[gid]) * lin_sigmoid(float(gate[gid])));
}

kernel void cpi_mul_vec(device const half* in [[buffer(0)]], device const half* vec [[buffer(1)]],
                        device half* out [[buffer(2)]], constant LinMulVecParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  out[gid] = half(float(in[gid]) * float(vec[gid]) * p.scale);
}

// Depthwise causal conv1d over the decode history, then SiLU, in place.
//
// conv_state is [channels][kernel_size-1] fp32 and holds this channel's previous inputs oldest
// first. One thread per channel: the whole window is that thread's own, so the shift is a private
// read-modify-write and needs no barrier -- but it also means this kernel must run exactly once
// per token, since replaying it would advance the history twice.
kernel void cpi_linear_conv1d_silu(device const half* conv_weight [[buffer(0)]],
                                   device float* conv_state [[buffer(1)]],
                                   device half* qkv_mix [[buffer(2)]],
                                   constant LinConvParams& p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.channels) return;
  const uint state_len = p.kernel_size - 1u;
  const uint wbase = gid * p.kernel_size;
  const float input = float(qkv_mix[gid]);
  float out = float(conv_weight[wbase + p.kernel_size - 1u]) * input;
  if (state_len > 0u) {
    device float* row = conv_state + gid * state_len;
    for (uint j = 0u; j < state_len; ++j) out += float(conv_weight[wbase + j]) * row[j];
    for (uint j = 0u; j + 1u < state_len; ++j) row[j] = row[j + 1u];
    row[state_len - 1u] = input;
  }
  qkv_mix[gid] = half(lin_silu(out));
}

// qkv_mix packs [q per key head | k per key head | v per value head]; fan the shared q/k of each
// key head out to its head_repeat value heads so the attention step sees one head layout.
kernel void cpi_repeat_linear_heads(device const half* qkv_mix [[buffer(0)]],
                                    device half* q_out [[buffer(1)]],
                                    device half* k_out [[buffer(2)]],
                                    device half* v_out [[buffer(3)]],
                                    constant LinRepeatParams& p [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
  // Flat grid of num_key_heads * max(key_head_dim, value_head_dim): MetalContext dispatches 1D,
  // and a 2D grid here would buy nothing but an API to maintain.
  const uint width = max(p.key_head_dim, p.value_head_dim);
  const uint key_head = gid / width;
  const uint d = gid % width;
  if (key_head >= p.num_key_heads) return;

  const uint q_base = key_head * p.key_head_dim;
  const uint k_base = p.num_key_heads * p.key_head_dim + q_base;
  const uint v_src_base =
      p.num_key_heads * 2u * p.key_head_dim + key_head * p.head_repeat * p.value_head_dim;

  for (uint rep = 0u; rep < p.head_repeat; ++rep) {
    const uint value_head = key_head * p.head_repeat + rep;
    if (d < p.key_head_dim) {
      q_out[value_head * p.key_head_dim + d] = qkv_mix[q_base + d];
      k_out[value_head * p.key_head_dim + d] = qkv_mix[k_base + d];
    }
    if (d < p.value_head_dim) {
      v_out[value_head * p.value_head_dim + d] =
          qkv_mix[v_src_base + rep * p.value_head_dim + d];
    }
  }
}

// One decode step of gated delta-net linear attention. One threadgroup per head.
//
//   q, k        L2-normalised (q additionally scaled by 1/sqrt(key_head_dim))
//   state      *= decay, where decay = exp(-exp(a_log) * softplus(a + dt_bias))
//   delta       = (v - state^T k) * beta,   beta = sigmoid(b)
//   state      += k (outer) delta
//   out         = RMSNorm(state^T q) * norm_weight * silu(z)
//
// The state is [key_head_dim][value_head_dim] fp32 per head, read and written in place, so this
// kernel is likewise once-per-token. Reductions use simd_sum across the threadgroup's simdgroups
// rather than the CUDA tree reduction; the arithmetic is otherwise identical, including the fp32
// accumulation and the 1e-6 inside the q/k normalisers (which is NOT rms_eps -- that one applies
// only to the output norm).
#define LIN_ATTN_TG 256
#define LIN_ATTN_MAXDIM 256

kernel void cpi_linear_attention_step(device const half* q [[buffer(0)]],
                                      device const half* k [[buffer(1)]],
                                      device const half* v [[buffer(2)]],
                                      device const half* z [[buffer(3)]],
                                      device const half* a [[buffer(4)]],
                                      device const half* b [[buffer(5)]],
                                      // half, not float: the .ll2c container casts every weight
                                      // to fp16, so there is no f32 source to read these from.
                                      // CUDA reads them as f32 straight from safetensors, so the
                                      // two backends differ here by fp16 rounding on a_log -- it
                                      // feeds exp(-exp(a_log)*softplus(..)), so watch this first
                                      // if a layer diverges by a small constant factor.
                                      device const half* norm_weight [[buffer(6)]],
                                      device const half* a_log [[buffer(7)]],
                                      device const half* dt_bias [[buffer(8)]],
                                      device float* recurrent_state [[buffer(9)]],
                                      device half* out [[buffer(10)]],
                                      constant LinAttnParams& p [[buffer(11)]],
                                      uint head [[threadgroup_position_in_grid]],
                                      uint lid [[thread_position_in_threadgroup]],
                                      uint nthr [[threads_per_threadgroup]],
                                      uint simd_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
  const uint kd_n = p.key_head_dim;
  const uint vd_n = p.value_head_dim;
  const uint state_stride = kd_n * vd_n;
  const uint q_base = head * kd_n;
  const uint v_base = head * vd_n;
  device float* state = recurrent_state + head * state_stride;

  threadgroup float q_sh[LIN_ATTN_MAXDIM];
  threadgroup float k_sh[LIN_ATTN_MAXDIM];
  threadgroup float scratch[LIN_ATTN_MAXDIM];
  threadgroup float partial[LIN_ATTN_TG / 32];
  threadgroup float shared_scalar[5];  // q_norm, k_norm, beta, decay, out_inv

  // ---- q and k norms -------------------------------------------------------
  float q_ss = 0.0f, k_ss = 0.0f;
  for (uint i = lid; i < kd_n; i += nthr) {
    const float qv = float(q[q_base + i]);
    const float kv = float(k[q_base + i]);
    q_ss += qv * qv;
    k_ss += kv * kv;
  }
  // Two reductions, folded into one pass over the shared buffer each.
  for (uint which = 0u; which < 2u; ++which) {
    const float mine = (which == 0u) ? q_ss : k_ss;
    const float sg = simd_sum(mine);
    if (lane == 0u) partial[simd_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      float tot = 0.0f;
      for (uint s = 0u; s < nthr / 32u; ++s) tot += partial[s];
      if (which == 0u) {
        shared_scalar[0] = rsqrt(tot + 1.0e-6f) / sqrt(float(kd_n));
      } else {
        shared_scalar[1] = rsqrt(tot + 1.0e-6f);
        shared_scalar[2] = lin_sigmoid(float(b[head]));
        shared_scalar[3] =
            exp(-exp(float(a_log[head])) * lin_softplus(float(a[head]) + float(dt_bias[head])));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float q_norm = shared_scalar[0];
  const float k_norm = shared_scalar[1];
  const float beta = shared_scalar[2];
  const float decay = shared_scalar[3];

  for (uint i = lid; i < kd_n; i += nthr) {
    q_sh[i] = float(q[q_base + i]) * q_norm;
    k_sh[i] = float(k[q_base + i]) * k_norm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- decay the state, then the delta rule --------------------------------
  for (uint i = lid; i < state_stride; i += nthr) state[i] *= decay;
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  for (uint dv = lid; dv < vd_n; dv += nthr) {
    float kv_mem = 0.0f;
    for (uint kd = 0u; kd < kd_n; ++kd) kv_mem += state[kd * vd_n + dv] * k_sh[kd];
    scratch[dv] = (float(v[v_base + dv]) - kv_mem) * beta;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = lid; i < state_stride; i += nthr) {
    const uint kd = i / vd_n;
    const uint dv = i % vd_n;
    state[i] += k_sh[kd] * scratch[dv];
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  // ---- read out, RMS-normalise, gate ---------------------------------------
  float out_ss = 0.0f;
  for (uint dv = lid; dv < vd_n; dv += nthr) {
    float sum = 0.0f;
    for (uint kd = 0u; kd < kd_n; ++kd) sum += state[kd * vd_n + dv] * q_sh[kd];
    scratch[dv] = sum;
    out_ss += sum * sum;
  }
  {
    const float sg = simd_sum(out_ss);
    if (lane == 0u) partial[simd_id] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      float tot = 0.0f;
      for (uint s = 0u; s < nthr / 32u; ++s) tot += partial[s];
      shared_scalar[4] = rsqrt(tot / float(vd_n) + p.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float out_inv = shared_scalar[4];

  for (uint dv = lid; dv < vd_n; dv += nthr) {
    out[v_base + dv] = half(scratch[dv] * out_inv * float(norm_weight[dv]) *
                            lin_silu(float(z[v_base + dv])));
  }
}
