// Metal compute shaders for the op-plan decode path.
//
// These mirror the CUDA kernels op-for-op (see include/runtime/kernels.cuh); the
// op-plan IR in include/engine/op_plan.hpp is what both backends execute, so the
// semantics here must match the CUDA ones exactly -- same normalisation, same RoPE
// convention, same causal masking. Numerics are NOT bit-identical across backends
// (different accumulation order), so correctness is judged against the CPU
// reference engine, not against CUDA bit-for-bit.
//
// Conventions shared with the CUDA side:
//   - weights are row-major fp16, out_dim x in_dim
//   - the residual stream and all slots are fp16; reductions accumulate in fp32
//   - RoPE is the half-split (NeoX) convention: element i pairs with i + head_dim/2
//   - the KV cache is [max_context][kv_heads * head_dim], fp16

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------------
// Parameter blocks. One per op family, bound at buffer(N) as a `constant` ref.
// Kept flat and POD so the C++ side can memcpy them without a shared header.
// ---------------------------------------------------------------------------

struct NormParams {
  uint rows;      // number of independent groups (tokens x groups-per-token)
  uint cols;      // elements per group
  float eps;
  uint weight_offset;  // 1 => scale by (1 + w) [Gemma], 0 => scale by w
  uint has_weight;     // 0 => weightless (ones)
};

struct GemvParams {
  uint out_dim;
  uint in_dim;
  uint tokens;    // T: batched over tokens (sequence prefill); 1 for decode
  uint has_bias;  // 0 => the bias buffer is bound but never read (Llama); 1 => Qwen2's Q/K/V
};

struct RopeParams {
  uint heads;
  uint head_dim;
  uint position;   // used when positions_buf is not bound
  uint tokens;
  float theta;
  uint use_position_buffer;  // read the position from a device buffer instead
  uint row_stride;           // elements between consecutive tokens in the slot
};

struct ElemParams {
  uint n;       // total elements
  float scale;
};

struct KvParams {
  uint kv_heads;
  uint head_dim;
  uint position;
  uint max_context;
  uint use_position_buffer;
};

struct AttnParams {
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint position;      // index of the query token (0-based)
  uint max_context;
  uint window;        // 0 = full causal; else sliding window length
  float scale;        // usually 1/sqrt(head_dim)
  uint use_position_buffer;
};

struct EmbedParams {
  uint hidden;
  uint tokens;
};

// ---------------------------------------------------------------------------
// RMSNorm -- one threadgroup per row. x is cached in registers only when it
// fits; otherwise re-read. Reduction is fp32 in threadgroup memory.
// ---------------------------------------------------------------------------

kernel void cpi_rmsnorm(
    device const half*  x       [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device half*        out     [[buffer(2)]],
    constant NormParams& p      [[buffer(3)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  nthr [[threads_per_threadgroup]]) {
  if (gid >= p.rows) return;

  const uint base = gid * p.cols;

  // Sum of squares, fp32.
  float ss = 0.0f;
  for (uint i = lid; i < p.cols; i += nthr) {
    const float v = float(x[base + i]);
    ss += v * v;
  }
  ss = simd_sum(ss);

  // Combine across simdgroups within the threadgroup.
  threadgroup float partial[32];
  const uint simd_id  = lid / 32u;
  const uint simd_lane = lid % 32u;
  const uint n_simd = (nthr + 31u) / 32u;
  if (simd_lane == 0u) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0u) {
    float v = (simd_lane < n_simd) ? partial[simd_lane] : 0.0f;
    v = simd_sum(v);
    if (simd_lane == 0u) partial[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv = rsqrt(partial[0] / float(p.cols) + p.eps);

  for (uint i = lid; i < p.cols; i += nthr) {
    float w = 1.0f;
    if (p.has_weight != 0u) {
      w = float(weight[i]);
      if (p.weight_offset != 0u) w += 1.0f;  // Gemma stores (w - 1)
    }
    out[base + i] = half(float(x[base + i]) * inv * w);
  }
}

// ---------------------------------------------------------------------------
// GEMV: out[out_dim] = W[out_dim x in_dim] . in[in_dim], row-major, fp16.
// One simdgroup per output row; each lane strides the row, fp32 accumulate.
// ---------------------------------------------------------------------------

kernel void cpi_gemv_f16(
    device const half*  W     [[buffer(0)]],
    device const half*  in    [[buffer(1)]],
    device half*        out   [[buffer(2)]],
    device const half*  bias  [[buffer(3)]],
    constant GemvParams& p    [[buffer(4)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint nthr  [[threads_per_threadgroup]]) {
  // Grid: one threadgroup per (token, row-block). Each simdgroup does one row.
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id      = lid / 32u;
  const uint lane         = lid % 32u;

  const uint rows_per_tg = simds_per_tg;
  const uint token = gid / ((p.out_dim + rows_per_tg - 1u) / rows_per_tg);
  const uint blk   = gid % ((p.out_dim + rows_per_tg - 1u) / rows_per_tg);
  const uint row   = blk * rows_per_tg + simd_id;
  if (token >= p.tokens || row >= p.out_dim) return;

  device const half* wrow = W + (ulong)row * (ulong)p.in_dim;
  device const half* xrow = in + (ulong)token * (ulong)p.in_dim;

  float acc = 0.0f;
  for (uint i = lane; i < p.in_dim; i += 32u) {
    acc += float(wrow[i]) * float(xrow[i]);
  }
  acc = simd_sum(acc);
  if (lane == 0u) {
    if (p.has_bias != 0u) acc += float(bias[row]);
    out[(ulong)token * (ulong)p.out_dim + row] = half(acc);
  }
}

// LM head: same as the GEMV but the output is fp32 (logits feed the sampler).
kernel void cpi_lm_head(
    device const half*  W     [[buffer(0)]],
    device const half*  in    [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant GemvParams& p    [[buffer(3)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint nthr  [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id      = lid / 32u;
  const uint lane         = lid % 32u;
  const uint row = gid * simds_per_tg + simd_id;
  if (row >= p.out_dim) return;

  device const half* wrow = W + (ulong)row * (ulong)p.in_dim;

  float acc = 0.0f;
  for (uint i = lane; i < p.in_dim; i += 32u) {
    acc += float(wrow[i]) * float(in[i]);
  }
  acc = simd_sum(acc);
  if (lane == 0u) out[row] = acc;
}

// ---------------------------------------------------------------------------
// RoPE, half-split (NeoX) convention: element i is paired with i + head_dim/2.
// In-place over `heads` heads of `head_dim`.
// ---------------------------------------------------------------------------

kernel void cpi_rope(
    device half*        x         [[buffer(0)]],
    constant RopeParams& p        [[buffer(1)]],
    device const int*   positions [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint half_dim = p.head_dim / 2u;
  const uint per_token = p.heads * half_dim;
  const uint total = per_token * p.tokens;
  if (gid >= total) return;

  const uint token = gid / per_token;
  const uint rem   = gid % per_token;
  const uint head  = rem / half_dim;
  const uint i     = rem % half_dim;

  uint pos = p.position;
  if (p.use_position_buffer != 0u) pos = uint(positions[0]);
  pos += token;  // sequence prefill: token t sits at position base + t

  const float freq  = pow(p.theta, -float(2u * i) / float(p.head_dim));
  const float angle = float(pos) * freq;
  const float c = cos(angle);
  const float s = sin(angle);

  const uint stride = (p.row_stride != 0u) ? p.row_stride : (p.heads * p.head_dim);
  const uint base = token * stride + head * p.head_dim + i;

  const float a = float(x[base]);
  const float b = float(x[base + half_dim]);
  x[base]            = half(a * c - b * s);
  x[base + half_dim] = half(a * s + b * c);
}

// ---------------------------------------------------------------------------
// Elementwise ops.
// ---------------------------------------------------------------------------

kernel void cpi_scale_copy(
    device const half*  in  [[buffer(0)]],
    device half*        out [[buffer(1)]],
    constant ElemParams& p  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  out[gid] = half(float(in[gid]) * p.scale);
}

kernel void cpi_copy(
    device const half*  in  [[buffer(0)]],
    device half*        out [[buffer(1)]],
    constant ElemParams& p  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  out[gid] = in[gid];
}

kernel void cpi_add_inplace(
    device const half*  in  [[buffer(0)]],
    device half*        out [[buffer(1)]],
    constant ElemParams& p  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  out[gid] = half(float(out[gid]) + float(in[gid]));
}

// SwiGLU: out = silu(a) * b
kernel void cpi_silu_mul(
    device const half*  a   [[buffer(0)]],
    device const half*  b   [[buffer(1)]],
    device half*        out [[buffer(2)]],
    constant ElemParams& p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const float av = float(a[gid]);
  const float silu = av / (1.0f + exp(-av));
  out[gid] = half(silu * float(b[gid]));
}

// GeGLU: out = gelu(a) * b. Tanh approximation -- matches the CUDA kernel.
kernel void cpi_gelu_mul(
    device const half*  a   [[buffer(0)]],
    device const half*  b   [[buffer(1)]],
    device half*        out [[buffer(2)]],
    constant ElemParams& p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const float x = float(a[gid]);
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  const float inner = k * (x + 0.044715f * x * x * x);
  const float gelu = 0.5f * x * (1.0f + tanh(inner));
  out[gid] = half(gelu * float(b[gid]));
}

// ---------------------------------------------------------------------------
// Embedding lookup. The token id lives in a device buffer so the whole decode
// step stays on-GPU (the CUDA path does the same, for graph capture).
// ---------------------------------------------------------------------------

kernel void cpi_embedding_lookup(
    device const half*  table  [[buffer(0)]],
    device const int*   tokens [[buffer(1)]],
    device half*        out    [[buffer(2)]],
    constant EmbedParams& p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = p.hidden * p.tokens;
  if (gid >= total) return;
  const uint t = gid / p.hidden;
  const uint i = gid % p.hidden;
  const int tok = tokens[t];
  out[gid] = table[(ulong)tok * (ulong)p.hidden + i];
}

// ---------------------------------------------------------------------------
// KV store: append this token's K and V to the layer's cache at `position`.
// Cache layout [max_context][kv_heads * head_dim], matching the CUDA side.
// ---------------------------------------------------------------------------

kernel void cpi_kv_store(
    device const half*  k         [[buffer(0)]],
    device const half*  v         [[buffer(1)]],
    device half*        k_cache   [[buffer(2)]],
    device half*        v_cache   [[buffer(3)]],
    constant KvParams&  p         [[buffer(4)]],
    device const int*   positions [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint kv_dim = p.kv_heads * p.head_dim;
  if (gid >= kv_dim) return;

  uint pos = p.position;
  if (p.use_position_buffer != 0u) pos = uint(positions[0]);
  if (pos >= p.max_context) return;

  const ulong dst = (ulong)pos * (ulong)kv_dim + gid;
  k_cache[dst] = k[gid];
  v_cache[dst] = v[gid];
}

// ---------------------------------------------------------------------------
// Single-query attention over the cache. One threadgroup per query head.
// GQA: query head h reads kv head h / (heads / kv_heads).
//
// Online (streaming) softmax so the scores never need a second pass over the
// cache: track the running max and the running sum, rescaling the accumulator
// when a new max appears. Same algorithm as the CUDA decode kernel.
// ---------------------------------------------------------------------------

kernel void cpi_attention_decode(
    device const half*  q         [[buffer(0)]],
    device const half*  k_cache   [[buffer(1)]],
    device const half*  v_cache   [[buffer(2)]],
    device half*        out       [[buffer(3)]],
    constant AttnParams& p        [[buffer(4)]],
    device const int*   positions [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint head = gid;
  if (head >= p.heads) return;

  uint pos = p.position;
  if (p.use_position_buffer != 0u) pos = uint(positions[0]);

  const uint group   = p.heads / p.kv_heads;   // query heads per kv head
  const uint kv_head = head / group;
  const uint kv_dim  = p.kv_heads * p.head_dim;

  // Causal window: attend to [start, pos].
  uint start = 0u;
  if (p.window != 0u && pos + 1u > p.window) start = pos + 1u - p.window;

  device const half* qh = q + head * p.head_dim;

  threadgroup float tg_max;
  threadgroup float tg_sum;
  threadgroup float tg_acc[256];   // head_dim accumulator (head_dim <= 256)
  threadgroup float tg_red[32];

  for (uint i = lid; i < p.head_dim; i += nthr) tg_acc[i] = 0.0f;
  if (lid == 0u) { tg_max = -INFINITY; tg_sum = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Walk the cache one key at a time. Each thread handles a slice of head_dim
  // for the dot product; the threadgroup reduces to one score per key.
  for (uint t = start; t <= pos; ++t) {
    device const half* kt = k_cache + (ulong)t * (ulong)kv_dim + kv_head * p.head_dim;

    float dot = 0.0f;
    for (uint i = lid; i < p.head_dim; i += nthr) {
      dot += float(qh[i]) * float(kt[i]);
    }
    dot = simd_sum(dot);

    const uint simd_id   = lid / 32u;
    const uint simd_lane = lid % 32u;
    const uint n_simd    = (nthr + 31u) / 32u;
    if (simd_lane == 0u) tg_red[simd_id] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
      float v = (simd_lane < n_simd) ? tg_red[simd_lane] : 0.0f;
      v = simd_sum(v);
      if (simd_lane == 0u) tg_red[0] = v * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float score = tg_red[0];

    // Online softmax rescale.
    const float old_max = tg_max;
    const float new_max = max(old_max, score);
    const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);
    const float w = exp(score - new_max);

    device const half* vt = v_cache + (ulong)t * (ulong)kv_dim + kv_head * p.head_dim;
    for (uint i = lid; i < p.head_dim; i += nthr) {
      tg_acc[i] = tg_acc[i] * rescale + w * float(vt[i]);
    }
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + w;
      tg_max = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / tg_sum;
  for (uint i = lid; i < p.head_dim; i += nthr) {
    out[head * p.head_dim + i] = half(tg_acc[i] * inv);
  }
}

// ---------------------------------------------------------------------------
// Argmax over fp32 logits, two-phase (the vocab is far too big for one group).
// Phase 1 writes per-partition (value, index); phase 2 reduces them.
// ---------------------------------------------------------------------------

kernel void cpi_argmax_partial(
    device const float* logits    [[buffer(0)]],
    device float*       part_val  [[buffer(1)]],
    device int*         part_idx  [[buffer(2)]],
    constant ElemParams& p        [[buffer(3)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]],
    uint ngrp [[threadgroups_per_grid]]) {
  float best_v = -INFINITY;
  int   best_i = -1;

  for (uint i = gid * nthr + lid; i < p.n; i += nthr * ngrp) {
    const float v = logits[i];
    if (v > best_v) { best_v = v; best_i = int(i); }
  }

  threadgroup float tv[256];
  threadgroup int   ti[256];
  tv[lid] = best_v;
  ti[lid] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = nthr / 2u; s > 0u; s >>= 1u) {
    if (lid < s && tv[lid + s] > tv[lid]) {
      tv[lid] = tv[lid + s];
      ti[lid] = ti[lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0u) {
    part_val[gid] = tv[0];
    part_idx[gid] = ti[0];
  }
}

kernel void cpi_argmax_reduce(
    device const float* part_val [[buffer(0)]],
    device const int*   part_idx [[buffer(1)]],
    device int*         out      [[buffer(2)]],
    constant ElemParams& p       [[buffer(3)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  float best_v = -INFINITY;
  int   best_i = -1;
  for (uint i = lid; i < p.n; i += nthr) {
    if (part_val[i] > best_v) { best_v = part_val[i]; best_i = part_idx[i]; }
  }

  threadgroup float tv[256];
  threadgroup int   ti[256];
  tv[lid] = best_v;
  ti[lid] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = nthr / 2u; s > 0u; s >>= 1u) {
    if (lid < s && tv[lid + s] > tv[lid]) {
      tv[lid] = tv[lid + s];
      ti[lid] = ti[lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0u) out[0] = ti[0];
}
