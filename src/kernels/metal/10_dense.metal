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
// Fused per-head RMSNorm + RoPE. Gemma normalizes each attention head (q_norm/k_norm) and then
// immediately rotates it, so the plan runs two ops that each read and write the same head
// vector. This does both in one pass: the head is loaded once into threadgroup memory,
// normalized there, rotated, and written back once. One threadgroup per (token, head).
//
// Numerics match [cpi_rmsnorm; cpi_rope] exactly -- the normalized values are rounded to half
// before rotating, the same as the two-op sequence writing and re-reading the slot.
// ---------------------------------------------------------------------------
kernel void cpi_rmsnorm_rope(
    device half*        x         [[buffer(0)]],
    device const half*  weight    [[buffer(1)]],
    device const int*   positions [[buffer(2)]],
    constant NormRopeParams& p    [[buffer(3)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  nthr [[threads_per_threadgroup]]) {
  const uint token = gid / p.heads;
  const uint head  = gid % p.heads;
  if (token >= p.tokens) return;

  const uint stride = (p.row_stride != 0u) ? p.row_stride : (p.heads * p.head_dim);
  const uint base = token * stride + head * p.head_dim;

  threadgroup float hv[ATTN_MAX_HEAD_DIM];
  threadgroup float partial[32];

  float ss = 0.0f;
  for (uint i = lid; i < p.head_dim; i += nthr) {
    const float v = float(x[base + i]);
    hv[i] = v;
    ss += v * v;
  }
  ss = simd_sum(ss);
  const uint simd_id = lid / 32u;
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

  const float inv = rsqrt(partial[0] / float(p.head_dim) + p.eps);
  for (uint i = lid; i < p.head_dim; i += nthr) {
    float w = 1.0f;
    if (p.has_weight != 0u) {
      w = float(weight[i]);
      if (p.weight_offset != 0u) w += 1.0f;  // Gemma stores (w - 1)
    }
    // Round to half here: the unfused pair stores the normed head before RoPE reloads it.
    hv[i] = float(half(hv[i] * inv * w));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint rot = (p.rotary_dim != 0u) ? p.rotary_dim : p.head_dim;
  const uint half_rot = rot / 2u;
  uint pos = (p.use_position_buffer != 0u) ? uint(positions[0]) : p.position;
  pos += token;  // sequence chunk: row t is token base+t

  for (uint i = lid; i < p.head_dim; i += nthr) {
    float v;
    if (i < half_rot) {
      const float freq = pow(p.theta, -float(2u * i) / float(rot));
      const float ang = float(pos) * freq;
      v = hv[i] * cos(ang) - hv[i + half_rot] * sin(ang);
    } else if (i < rot) {
      const uint j = i - half_rot;
      const float freq = pow(p.theta, -float(2u * j) / float(rot));
      const float ang = float(pos) * freq;
      v = hv[j] * sin(ang) + hv[i] * cos(ang);
    } else {
      v = hv[i];  // lanes past the rotary width pass through
    }
    x[base + i] = half(v);
  }
}

// ---------------------------------------------------------------------------
// Fused residual-add + RMSNorm. The plan does `X += delta` (AddInplace) then
// `XNorm = rmsnorm(X)` back to back, each a full read+write of the residual. This does both
// in one pass: add delta into X (written back), then normalise the summed value. Saves one
// round-trip of the whole activation through device memory per fusion site.
// ---------------------------------------------------------------------------
kernel void cpi_add_rmsnorm(
    device half*        x       [[buffer(0)]],  // residual, updated in place (X += delta)
    device const half*  delta   [[buffer(1)]],  // block output to add
    device const half*  weight  [[buffer(2)]],
    device half*        out     [[buffer(3)]],  // normalised result (XNorm)
    constant NormParams& p      [[buffer(4)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  nthr [[threads_per_threadgroup]]) {
  if (gid >= p.rows) return;
  const uint base = gid * p.cols;

  // Add delta into the residual, write it back, and accumulate the sum of squares in one sweep.
  float ss = 0.0f;
  for (uint i = lid; i < p.cols; i += nthr) {
    const float v = float(x[base + i]) + float(delta[base + i]);
    x[base + i] = half(v);  // updated residual, read again below by this same thread
    ss += v * v;
  }
  ss = simd_sum(ss);

  threadgroup float partial[32];
  const uint simd_id = lid / 32u;
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
      if (p.weight_offset != 0u) w += 1.0f;
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
  // Four-rows-per-simdgroup DECODE path: one simdgroup owns four consecutive rows and the
  // activation is loaded once for all of them. An fp16 square GEMV is half activation
  // re-reads by traffic on the one-row shape (row = in_dim halfs, activation = in_dim halfs),
  // so this cuts total traffic up to 1.6x. The executor passes 64-thread threadgroups under
  // the SAME predicate (tokens == 1, in_dim % 8 == 0) -- keep them in lockstep.
  if (p.tokens == 1u && (p.in_dim & 7u) == 0u) {
    const uint r0 = (gid * simds_per_tg + simd_id) * 4u;
    if (r0 >= p.out_dim) return;
    const uint nrows = min(4u, p.out_dim - r0);
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const uint4* x4 = (device const uint4*)in;
    const uint n4 = p.in_dim >> 3;
    for (uint k = lane; k < n4; k += 32u) {
      const uint4 xv = x4[k];  // read once, dotted against four weight rows
      for (uint r = 0u; r < nrows; ++r) {
        device const uint4* w4 = (device const uint4*)(W + (ulong)(r0 + r) * (ulong)p.in_dim);
        acc4[r] += dot8_f32(w4[k], xv);
      }
    }
    for (uint r = 0u; r < nrows; ++r) {
      const float s = simd_sum(acc4[r]);
      if (lane == 0u) {
        float v = s;
        if (p.has_bias != 0u) v += float(bias[r0 + r]);
        out[r0 + r] = half(v);
      }
    }
    return;
  }

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

  // The LM head is the biggest single read of a decode step (vocab x hidden), so the load
  // width matters most here -- and so does the four-rows-per-simdgroup shape: the head is
  // always a single token, and one-row-per-simdgroup re-reads the whole activation once per
  // vocab row. Executor passes 64-thread threadgroups under the same predicate.
  if ((p.in_dim & 7u) == 0u) {
    const uint r0 = (gid * simds_per_tg + simd_id) * 4u;
    if (r0 >= p.out_dim) return;
    const uint nrows = min(4u, p.out_dim - r0);
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const uint4* x4 = (device const uint4*)in;
    const uint n4 = p.in_dim >> 3;
    for (uint k = lane; k < n4; k += 32u) {
      const uint4 xv = x4[k];
      for (uint r = 0u; r < nrows; ++r) {
        device const uint4* w4 = (device const uint4*)(W + (ulong)(r0 + r) * (ulong)p.in_dim);
        acc4[r] += dot8_f32(w4[k], xv);
      }
    }
    for (uint r = 0u; r < nrows; ++r) {
      const float s = simd_sum(acc4[r]);
      if (lane == 0u) out[r0 + r] = s;
    }
    return;
  }

  const uint row = gid * simds_per_tg + simd_id;
  if (row >= p.out_dim) return;

  device const half* wrow = W + (ulong)row * (ulong)p.in_dim;
  float acc = 0.0f;
  for (uint k = lane; k < p.in_dim; k += 32u) {
    acc += float(wrow[k]) * float(in[k]);
  }
  acc = simd_sum(acc);
  if (lane == 0u) out[row] = acc;
}


// ---------------------------------------------------------------------------
// True LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias.
//
// The fp32 accumulators are LOAD-BEARING and measured to be so. The Qwen3.5 vision tower's
// last block holds outliers around 1657 (its rows are otherwise ~unit scale), and 1657^2 is
// 2.7e6 -- past fp16's 65504. Swapping the variance accumulator to half makes metal_smoke's
// layernorm case go from max_abs 0.0005 to 1.21, so that test genuinely pins this.
//
// The two passes are DEFENSIVE, not required by this model. The one-pass identity
// var = E[x^2] - E[x]^2 cancels catastrophically only when |mean| >> stddev, and these
// activations are near zero-mean (max |mean|/stddev across the tower is 0.5), so both forms
// agree here -- verified by patching the kernel to the one-pass form and seeing the test
// pass unchanged. Kept anyway: the second pass re-reads a row that is already in cache, and
// nothing guarantees the next checkpoint's activations stay centred.
// ---------------------------------------------------------------------------
kernel void cpi_layernorm(
    device const half*  x       [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const half*  bias    [[buffer(2)]],
    device half*        out     [[buffer(3)]],
    constant LayerNormParams& p [[buffer(4)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  nthr [[threads_per_threadgroup]]) {
  if (gid >= p.rows) return;

  const uint base = gid * p.cols;
  threadgroup float partial[32];
  const uint simd_id   = lid / 32u;
  const uint simd_lane = lid % 32u;
  const uint n_simd    = (nthr + 31u) / 32u;

  // ---- pass 1: mean ----
  float sum = 0.0f;
  for (uint i = lid; i < p.cols; i += nthr) sum += float(x[base + i]);
  sum = simd_sum(sum);
  if (simd_lane == 0u) partial[simd_id] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0u) {
    float v = (simd_lane < n_simd) ? partial[simd_lane] : 0.0f;
    v = simd_sum(v);
    if (simd_lane == 0u) partial[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = partial[0] / float(p.cols);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- pass 2: variance about that mean ----
  float ss = 0.0f;
  for (uint i = lid; i < p.cols; i += nthr) {
    const float d = float(x[base + i]) - mean;
    ss += d * d;
  }
  ss = simd_sum(ss);
  if (simd_lane == 0u) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reduces into a slot it also reads from, which is safe only because simd_sum completes
  // for the whole simdgroup before any lane writes. `mean` is already in registers by this
  // point, so reusing partial[] costs nothing and keeps one shared array.
  if (simd_id == 0u) {
    float v = (simd_lane < n_simd) ? partial[simd_lane] : 0.0f;
    v = simd_sum(v);
    if (simd_lane == 0u) partial[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt(partial[0] / float(p.cols) + p.eps);

  for (uint i = lid; i < p.cols; i += nthr) {
    const float n = (float(x[base + i]) - mean) * inv * float(weight[i]);
    out[base + i] = half(p.has_bias != 0u ? n + float(bias[i]) : n);
  }
}
