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

// Tokens per threadgroup in the GEMV. The weight row is read once and reused across
// the tile, so this is the prefill's bandwidth-amplification factor. 8 fits in
// registers comfortably; decode uses a single token and is unaffected.
#define GEMV_TILE 8

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
  uint position;  // position of the FIRST token in the batch
  uint max_context;
  uint use_position_buffer;
  uint tokens;  // T: 1 for decode, N for a prefill chunk
};

struct AttnParams {
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint position;  // position of the FIRST query token (0-based)
  uint max_context;
  uint window;  // 0 = full causal; else sliding window length
  float scale;  // usually 1/sqrt(head_dim)
  uint use_position_buffer;
  uint tokens;  // T: 1 for decode, N for a prefill chunk
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
//
// The weights are read 128 BITS AT A TIME (uint4 = 8 halves), not one half at a
// time. A decode GEMV is pure bandwidth -- it touches every weight exactly once and
// never reuses one -- so the load width IS the kernel. Reading 16-bit scalars leaves
// most of each memory transaction unused and cannot keep enough requests in flight to
// cover DRAM latency. The same mistake, and the same fix, as the CUDA backend.
//
// Requires in_dim % 8 == 0 for the wide path (true of every shape here: 896, 1024,
// 2048, 3072, 4864). The scalar loop remains as the tail and the fallback.
// ---------------------------------------------------------------------------

// Dot-product of 8 halves packed in a uint4 against another, accumulating in fp32.
static inline float dot8_f32(uint4 a, uint4 b) {
  const half2 a0 = as_type<half2>(a.x), a1 = as_type<half2>(a.y);
  const half2 a2 = as_type<half2>(a.z), a3 = as_type<half2>(a.w);
  const half2 b0 = as_type<half2>(b.x), b1 = as_type<half2>(b.y);
  const half2 b2 = as_type<half2>(b.z), b3 = as_type<half2>(b.w);
  return float(a0.x) * float(b0.x) + float(a0.y) * float(b0.y) +
         float(a1.x) * float(b1.x) + float(a1.y) * float(b1.y) +
         float(a2.x) * float(b2.x) + float(a2.y) * float(b2.y) +
         float(a3.x) * float(b3.x) + float(a3.y) * float(b3.y);
}

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

  // TOKEN TILING. A prefill pushes T tokens through the same weights, so the weight
  // row is loaded ONCE and reused across a tile of tokens. Without this, a "batched"
  // GEMV is really T separate GEMVs: it re-streams the entire weight matrix per token
  // and saves nothing on a bandwidth-bound machine. With it, a tile of TILE tokens
  // reads the weights once, so prefill traffic drops by up to TILE-fold.
  //
  // Decode is TILE=1 and behaves exactly as before.
  const uint rows_per_tg = simds_per_tg;
  const uint row_blocks  = (p.out_dim + rows_per_tg - 1u) / rows_per_tg;

  const uint tile = gid / row_blocks;          // which group of tokens
  const uint blk  = gid % row_blocks;          // which group of rows
  const uint row  = blk * rows_per_tg + simd_id;
  const uint t0   = tile * GEMV_TILE;
  if (t0 >= p.tokens || row >= p.out_dim) return;

  const uint nt = min((uint)GEMV_TILE, p.tokens - t0);  // tokens actually in this tile

  device const half* wrow = W + (ulong)row * (ulong)p.in_dim;

  float acc[GEMV_TILE];
  for (uint t = 0u; t < GEMV_TILE; ++t) acc[t] = 0.0f;

  uint i = 0u;
  if ((p.in_dim & 7u) == 0u) {
    device const uint4* w4 = (device const uint4*)wrow;
    const uint n4 = p.in_dim >> 3;
    for (uint k = lane; k < n4; k += 32u) {
      const uint4 wv = w4[k];  // <-- read once, used by every token in the tile
      for (uint t = 0u; t < nt; ++t) {
        device const uint4* x4 = (device const uint4*)(in + (ulong)(t0 + t) * (ulong)p.in_dim);
        acc[t] += dot8_f32(wv, x4[k]);
      }
    }
    i = n4 << 3;
  }
  for (uint k = i + lane; k < p.in_dim; k += 32u) {  // tail / fallback
    const float wv = float(wrow[k]);
    for (uint t = 0u; t < nt; ++t) {
      acc[t] += wv * float(in[(ulong)(t0 + t) * (ulong)p.in_dim + k]);
    }
  }

  for (uint t = 0u; t < nt; ++t) {
    const float s = simd_sum(acc[t]);
    if (lane == 0u) {
      float v = s;
      if (p.has_bias != 0u) v += float(bias[row]);
      out[(ulong)(t0 + t) * (ulong)p.out_dim + row] = half(v);
    }
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

  // The LM head is the biggest single read of a decode step (vocab x hidden), so the
  // load width matters most here.
  float acc = 0.0f;
  uint i = 0u;

  if ((p.in_dim & 7u) == 0u) {
    device const uint4* w4 = (device const uint4*)wrow;
    device const uint4* x4 = (device const uint4*)in;
    const uint n4 = p.in_dim >> 3;
    for (uint k = lane; k < n4; k += 32u) {
      acc += dot8_f32(w4[k], x4[k]);
    }
    i = n4 << 3;
  }
  for (uint k = i + lane; k < p.in_dim; k += 32u) {
    acc += float(wrow[k]) * float(in[k]);
  }

  acc = simd_sum(acc);
  if (lane == 0u) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Weight-only int4 / int8 GEMV.
//
// Metal has NO __dp4a, so this does not try to mirror the CUDA kernel's packed
// integer dot product. It does not need to: an int4 decode is BANDWIDTH-bound, and
// reading 0.5 bytes per weight instead of 2 is itself the whole win. Dequantise on
// the fly and multiply in fp32.
//
// Format (identical to the CUDA path -- see kernels_weight_only_matvec.cu):
//   * nibble -> signed:  (n ^ 0x8) - 0x8   [so 0..15 maps to -8..7]
//   * a byte packs two weights: low nibble = even column, high nibble = odd column
//   * y[row] = sum_g scale[row][g] * sum_{j in g} (q[j] * x[j])
//     -- weight-only: activations stay fp32, and the scale is applied per group.
// ---------------------------------------------------------------------------

struct QuantParams {
  uint out_dim;
  uint in_dim;
  uint tokens;
  uint bits;      // 4 or 8
  uint group;     // 0 = one scale per row
  uint groups;    // scales per row
  uint has_bias;  // Qwen2's Q/K/V carry one; quantizing the weights does not remove it
};

kernel void cpi_gemv_quant(
    device const uchar*  qw     [[buffer(0)]],
    device const half*   in     [[buffer(1)]],
    device half*         out    [[buffer(2)]],
    device const float*  scales [[buffer(3)]],
    device const half*   bias   [[buffer(4)]],
    constant QuantParams& p     [[buffer(5)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint nthr  [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id      = lid / 32u;
  const uint lane         = lid % 32u;

  const uint row_blocks = (p.out_dim + simds_per_tg - 1u) / simds_per_tg;
  const uint tile = gid / row_blocks;
  const uint blk  = gid % row_blocks;
  const uint row  = blk * simds_per_tg + simd_id;
  const uint t0   = tile * GEMV_TILE;
  if (t0 >= p.tokens || row >= p.out_dim) return;
  const uint nt = min((uint)GEMV_TILE, p.tokens - t0);

  device const float* srow = scales + (ulong)row * (ulong)p.groups;
  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;

  float acc[GEMV_TILE];
  for (uint t = 0u; t < GEMV_TILE; ++t) acc[t] = 0.0f;

  if (p.bits == 4u) {
    // 128-BIT loads: a uint4 is 32 packed int4 weights. This used to load a bare `uint`
    // (8 weights, 32 bits) and it cost ~40% of decode throughput -- the same narrow-load
    // mistake as the fp16 GEMV, made again in a new kernel. On a bandwidth-bound machine
    // the LOAD WIDTH IS THE KERNEL, whatever the element size.
    //
    // A group size that is a multiple of 32 keeps all 32 weights inside one scale group.
    device const uint4* w128 =
        (device const uint4*)(qw + (ulong)row * (ulong)((p.in_dim + 1u) / 2u));
    const uint n32 = p.in_dim >> 5;  // 32 int4 weights per uint4
    for (uint k = lane; k < n32; k += 32u) {
      const uint4 packed = w128[k];
      const uint j0 = k << 5;
      const float sc = srow[j0 / gsz];

      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)p.in_dim + j0;
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const uint word = (w == 0u) ? packed.x : (w == 1u) ? packed.y
                          : (w == 2u) ? packed.z : packed.w;
          for (uint e = 0u; e < 8u; ++e) {
            const int nib = int((word >> (4u * e)) & 0xFu);
            const int q = (nib ^ 0x8) - 0x8;
            sub += float(q) * float(x[w * 8u + e]);
          }
        }
        acc[t] += sub * sc;
      }
    }
    // Tail: whatever does not fill a uint4.
    device const uint* w32 = (device const uint*)(qw + (ulong)row * (ulong)((p.in_dim + 1u) / 2u));
    for (uint k = (n32 << 2) + lane; k < (p.in_dim >> 3); k += 32u) {
      const uint packed = w32[k];
      const uint j0 = k << 3;
      const float sc = srow[j0 / gsz];
      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)p.in_dim + j0;
        float sub = 0.0f;
        for (uint e = 0u; e < 8u; ++e) {
          const int nib = int((packed >> (4u * e)) & 0xFu);
          sub += float((nib ^ 0x8) - 0x8) * float(x[e]);
        }
        acc[t] += sub * sc;
      }
    }
  } else {
    // int8, 128 bits at a time: a uint4 is SIXTEEN weights. Loading them one byte at a
    // time made int8 slower than fp16 -- which is absurd, since it reads half the bytes
    // -- for exactly the reason the fp16 GEMV needed 128-bit loads. Narrow loads waste
    // the transaction and cannot cover DRAM latency, whatever the element size.
    device const uint4* w16 = (device const uint4*)(qw + (ulong)row * (ulong)p.in_dim);
    const uint n16 = p.in_dim >> 4;  // 16 int8s per uint4
    for (uint k = lane; k < n16; k += 32u) {
      const uint4 packed = w16[k];
      const uint j0 = k << 4;
      const float sc = srow[j0 / gsz];  // gsz is a multiple of 16 here

      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)p.in_dim + j0;
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const uint word = (w == 0u) ? packed.x : (w == 1u) ? packed.y
                          : (w == 2u) ? packed.z : packed.w;
          for (uint e = 0u; e < 4u; ++e) {
            const int q = int(char((word >> (8u * e)) & 0xFFu));
            sub += float(q) * float(x[w * 4u + e]);
          }
        }
        acc[t] += sub * sc;
      }
    }
    // Tail: whatever does not fill a uint4.
    for (uint k = (n16 << 4) + lane; k < p.in_dim; k += 32u) {
      const int q = int(((device const char*)(qw + (ulong)row * (ulong)p.in_dim))[k]);
      const float sc = srow[k / gsz];
      for (uint t = 0u; t < nt; ++t) {
        acc[t] += float(q) * sc * float(in[(ulong)(t0 + t) * (ulong)p.in_dim + k]);
      }
    }
  }

  for (uint t = 0u; t < nt; ++t) {
    float s = simd_sum(acc[t]);
    if (lane == 0u) {
      if (p.has_bias != 0u) s += float(bias[row]);
      out[(ulong)(t0 + t) * (ulong)p.out_dim + row] = half(s);
    }
  }
}

// Same, but fp32 output: the LM head feeds the sampler.
kernel void cpi_lm_head_quant(
    device const uchar*  qw     [[buffer(0)]],
    device const half*   in     [[buffer(1)]],
    device float*        out    [[buffer(2)]],
    device const float*  scales [[buffer(3)]],
    constant QuantParams& p     [[buffer(4)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint nthr  [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id      = lid / 32u;
  const uint lane         = lid % 32u;
  const uint row = gid * simds_per_tg + simd_id;
  if (row >= p.out_dim) return;

  device const float* srow = scales + (ulong)row * (ulong)p.groups;
  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;
  float acc = 0.0f;

  if (p.bits == 4u) {
    device const uint* w32 = (device const uint*)(qw + (ulong)row * (ulong)((p.in_dim + 1u) / 2u));
    const uint n8 = p.in_dim >> 3;
    for (uint k = lane; k < n8; k += 32u) {
      const uint packed = w32[k];
      const uint j0 = k << 3;
      float sub = 0.0f;
      for (uint e = 0u; e < 8u; ++e) {
        const int nib = int((packed >> (4u * e)) & 0xFu);
        const int q = (nib ^ 0x8) - 0x8;
        sub += float(q) * float(in[j0 + e]);
      }
      acc += sub * srow[j0 / gsz];
    }
  } else {
    device const char* w8 = (device const char*)(qw + (ulong)row * (ulong)p.in_dim);
    for (uint k = lane; k < p.in_dim; k += 32u) {
      acc += float(w8[k]) * srow[k / gsz] * float(in[k]);
    }
  }

  acc = simd_sum(acc);
  if (lane == 0u) out[row] = acc;
}

// ---------------------------------------------------------------------------
// RoPE, half-split (NeoX) convention: element i is paired with i + head_dim/2.
// In-place over `heads` heads of `head_dim`.
// ---------------------------------------------------------------------------

// NOTE the buffer order: the params block is ALWAYS the last binding, in every
// kernel here. MetalContext::dispatch() binds it at index n_buffers, so a kernel
// that puts it anywhere else would have its params written over another buffer --
// which compiles cleanly and only misbehaves on real hardware.
kernel void cpi_rope(
    device half*        x         [[buffer(0)]],
    device const int*   positions [[buffer(1)]],
    constant RopeParams& p        [[buffer(2)]],
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

// GeGLU: out = gelu(a) * b. Tanh approximation, same as the CUDA kernel.
//
// ⚠ DO NOT write this as 0.5 * x * (1 + tanh(inner)).
//
// Metal's tanh overflows. It evaluates as (exp(2z) - 1) / (exp(2z) + 1), and exp(2z)
// exceeds fp32 for z beyond ~44, giving inf/inf = NaN -- where CUDA's tanhf saturates
// to 1 and is fine. Gemma reaches z ~ 62 on ordinary activations, so this produced NaN
// on real inputs, for some tokens and not others (it depends on the gate value). Llama
// never hit it because SwiGLU uses a sigmoid, which is naturally safe.
//
// 0.5 * (1 + tanh(z)) is exactly sigmoid(2z), and sigmoid is safe at BOTH ends:
// exp(-large) -> 0 gives 1, and exp(+large) -> inf gives 1/(1+inf) = 0. Same maths, no
// overflow.
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
  const float gelu = x / (1.0f + exp(-2.0f * inner));  // == 0.5*x*(1 + tanh(inner))
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
    device const int*   positions [[buffer(4)]],
    constant KvParams&  p         [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint total = kv_dim * p.tokens;
  if (gid >= total) return;

  const uint t = gid / kv_dim;   // which token in the batch
  const uint i = gid % kv_dim;   // which element of its K/V

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);
  const uint pos = base + t;
  if (pos >= p.max_context) return;

  const ulong dst = (ulong)pos * (ulong)kv_dim + i;
  k_cache[dst] = k[(ulong)t * (ulong)kv_dim + i];
  v_cache[dst] = v[(ulong)t * (ulong)kv_dim + i];
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
    device const int*   positions [[buffer(4)]],
    constant AttnParams& p        [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  // One threadgroup per (query token, head). Decode is just T = 1, so prefill and
  // decode share this kernel instead of duplicating the softmax and the GQA maths.
  const uint total = p.heads * p.tokens;
  if (gid >= total) return;
  const uint t    = gid / p.heads;  // which query token in the batch
  const uint head = gid % p.heads;

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  // This query's absolute position. CAUSALITY falls straight out of it: token t
  // attends to keys [start, base+t] and no further, so no explicit mask is needed.
  const uint pos = base + t;

  const uint group   = p.heads / p.kv_heads;   // query heads per kv head
  const uint kv_head = head / group;
  const uint kv_dim  = p.kv_heads * p.head_dim;
  const uint q_dim   = p.heads * p.head_dim;

  // Causal window: attend to [start, pos].
  uint start = 0u;
  if (p.window != 0u && pos + 1u > p.window) start = pos + 1u - p.window;

  device const half* qh = q + (ulong)t * (ulong)q_dim + head * p.head_dim;

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
    // Every thread read tg_max above to get old_max. Thread 0 is about to overwrite
    // it. Without this barrier a fast thread 0 could publish the NEW max before a
    // slower thread has read the old one, which silently corrupts that thread's
    // rescale factor -- wrong attention, non-deterministically, only on hardware.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + w;
      tg_max = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / tg_sum;
  for (uint i = lid; i < p.head_dim; i += nthr) {
    out[(ulong)t * (ulong)q_dim + head * p.head_dim + i] = half(tg_acc[i] * inv);
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
