kernel void cpi_rope(
    device half*        x         [[buffer(0)]],
    device const int*   positions [[buffer(1)]],
    constant RopeParams& p        [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  // Only the leading rotary_dim lanes of each head rotate; the rest pass through untouched.
  // rot_dim == 0 means "the whole head", which is every model without a partial factor.
  const uint rot_dim = (p.rotary_dim != 0u) ? p.rotary_dim : p.head_dim;
  const uint half_dim = rot_dim / 2u;
  const uint per_token = p.heads * half_dim;
  const uint total = per_token * p.tokens;
  if (gid >= total) return;

  const uint token = gid / per_token;
  const uint rem   = gid % per_token;
  const uint head  = rem / half_dim;
  const uint i     = rem % half_dim;

  uint pos;
  if (p.mrope_t != 0u || p.mrope_h != 0u || p.mrope_w != 0u) {
    // M-RoPE: pick this lane's axis, then read that axis's position for this token. The buffer
    // is [3][tokens] -- t, then h, then w.
    //
    // The layout is interleaved, not chunked: [T,H,W,T,H,W,...], not [TTT...HHH...WWW]. Qwen3.5
    // sets mrope_interleaved=true and transformers' apply_interleaved_mrope builds it as "start
    // from T everywhere, then overwrite stride-3 slots" --
    //
    //     freqs_t = freqs[0]
    //     for dim, offset in ((1,1), (2,2)):
    //         freqs_t[..., offset : mrope_section[dim]*3 : 3] = freqs[dim, ..., ...]
    //
    // so H takes lanes 1,4,7,... below section[1]*3 and W takes 2,5,8,... below section[2]*3,
    // and every lane neither claims stays T. For [11,11,10] that is 11 T, 11 H, 10 W -- the same
    // counts the chunked reading gives, which is why the section values look interchangeable and
    // are not.
    //
    // This was chunked, and nothing caught it. Both M-RoPE gates are blind to the lane mapping:
    // mrope_reduces_to_1d feeds t == h == w, where every mapping produces identical output
    // bit-for-bit, and mrope_splits_axes only asserts the result differs from 1-D rope -- which a
    // wrong mapping does just as well as a right one. Same shape as the position-0 rotary trap.
    //
    // A model with mrope_interleaved=false would need the chunked rule back, behind a flag.
    uint axis = 0u;  // T unless another axis claims this lane
    const uint slot = i % 3u;
    if (slot == 1u && i < p.mrope_h * 3u) {
      axis = 1u;  // H
    } else if (slot == 2u && i < p.mrope_w * 3u) {
      axis = 2u;  // W
    }
    pos = uint(positions[axis * p.tokens + token]);
  } else if (p.per_row_positions != 0u) {
    pos = uint(positions[token]);  // batched decode: row t is its own sequence
  } else {
    pos = (p.use_position_buffer != 0u) ? uint(positions[0]) : p.position;
    pos += token;  // sequence prefill: token t sits at position base + t
  }

  // The frequency denominator is the rotary width, not the head width. Using head_dim here
  // stretches every angle by head_dim/rotary_dim and still looks perfectly correct at
  // position 0, where the angle is zero regardless.
  const float freq  = pow(p.theta, -float(2u * i) / float(rot_dim));
  const float angle = float(pos) * freq;
  const float c = cos(angle);
  const float s = sin(angle);

  const uint stride = (p.row_stride != 0u) ? p.row_stride : (p.heads * p.head_dim);
  // Indexed by the head stride, not the rotary stride: lane i of head h still lives at
  // h * head_dim + i, and only the pairing distance shrinks to half of rotary_dim.
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
// Do not write this as 0.5 * x * (1 + tanh(inner)).
//
// Metal's tanh overflows. It evaluates as (exp(2z) - 1) / (exp(2z) + 1), and exp(2z)
// exceeds fp32 for z beyond ~44, giving inf/inf = NaN -- where CUDA's tanhf saturates
// to 1 and is fine. Gemma reaches z ~ 62 on ordinary activations, so this produced NaN
// on real inputs, for some tokens and not others (it depends on the gate value). Llama
// never hit it because SwiGLU uses a sigmoid, which is naturally safe.
//
// 0.5 * (1 + tanh(z)) is exactly sigmoid(2z), and sigmoid is safe at both ends:
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
  // p.scale (0 means 1) divides the product before it is stored, and the plan multiplies the same
  // factor back after the following down-projection. That projection is linear, so the round trip
  // is exact -- and with a power of two the scaling is a pure exponent shift, so it costs no
  // mantissa bits either. Nothing is approximated here.
  //
  // It exists because Gemma 4's norm gains are large (7-47 in E2B): activations reach ~165 into
  // the MLP and the GeGLU product passes fp16's 65504, going inf then NaN. Saturating instead kept
  // it finite but lossy, and the loss compounded into repetition over long generations. The clamp
  // stays as a backstop, a no-op for values that fit, so no existing model's arithmetic changes.
  // in2 may be a window of a wider row than out's. Flat indexing is only correct when the two
  // strides match, which is why an aux_offset plan used to be forced to one token at a time --
  // this makes it correct for a whole prefill chunk instead.
  uint bidx = gid;
  if (p.row_len != 0u) {
    const uint t = gid / p.row_len;
    const uint j = gid % p.row_len;
    bidx = t * p.in2_stride + p.in2_offset + j;
  }
  const float prod = gelu * float(b[bidx]);
  const float scaled = (p.scale != 0.0f) ? (prod * p.scale) : prod;
  out[gid] = half(clamp(scaled, -65504.0f, 65504.0f));
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

// Embedding lookup from a quantized table: same gather, dequantizing as it reads. The row is
// gathered rather than multiplied, so dequant is one scalar multiply per element and the kernel
// stays bandwidth-bound on a quarter the bytes. Gemma 4's embed_tokens_per_layer is
// [vocab 262144][35 layers x 256] = 4.70 GB fp16, most of that model.
//
// Layout matches WeightSource::quant: int4 packed two per byte low-then-high, decoded as
// (n ^ 8) - 8; int8 plain; fp16 scales, one per `group` elements, or one per row when group == 0.
kernel void cpi_embedding_lookup_quant(
    device const uchar* table  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const int*   tokens [[buffer(2)]],
    device half*        out    [[buffer(3)]],
    constant EmbedQuantParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = p.hidden * p.tokens;
  if (gid >= total) return;
  const uint t = gid / p.hidden;
  const uint i = gid % p.hidden;
  const int tok = tokens[t];

  const uint gsz = (p.group > 0u) ? p.group : p.hidden;
  const uint groups = (p.hidden + gsz - 1u) / gsz;
  const float sc = float(scales[(ulong)tok * (ulong)groups + i / gsz]);

  float w;
  if (p.bits == 4u) {
    // Two values per byte: element i lives in byte i/2, low nibble for even i.
    const ulong row = (ulong)tok * (ulong)((p.hidden + 1u) / 2u);
    const uchar byte = table[row + (i >> 1u)];
    const uchar nib = (i & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
    w = float(int(nib ^ 0x08u) - 8);
  } else {
    w = float(as_type<char>(table[(ulong)tok * (ulong)p.hidden + i]));
  }
  out[gid] = half(w * sc);
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
// Prefill attention: one threadgroup per (query block, head).
//
// The decode kernel below takes one threadgroup per (query token, head), and each of those
// walks the whole KV cache itself. Across a prompt that is O(T^2) of device traffic, and on
// the 8B it measured ~80 GB at ~73 GB/s -- genuinely bandwidth-bound, and 23% of prefill.
//
// A query block fixes it: the block of keys a threadgroup pulls in now serves Q_BLOCK
// queries instead of one, so the same answer costs Q_BLOCK times less traffic. The keys and
// values are re-read once per query within the block, but that is an L1 hit -- a key block
// is a few KB and the reuse is immediate.
//
// Each query keeps its own online-softmax state (running max, running sum), because each
// attends to a different prefix: causality masks key > that query's position.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Plain (ungated) GELU, tanh approximation -- gelu_pytorch_tanh.
//
// The vision MLP is fc1 -> gelu -> fc2, not a gated GeGLU, so cpi_gelu_mul does not fit.
// Same sigmoid formulation as that kernel and for the same reason: writing this as
// 0.5*x*(1 + tanh(inner)) gives NaN on Metal for |inner| beyond ~44, where CUDA's tanhf
// saturates. See the note above cpi_gelu_mul.
// ---------------------------------------------------------------------------
kernel void cpi_gelu(
    device half*        x   [[buffer(0)]],
    constant ElemParams& p  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const float v = float(x[gid]);
  const float k = 0.7978845608028654f;  // sqrt(2/pi)
  const float inner = k * (v + 0.044715f * v * v * v);
  x[gid] = half(v / (1.0f + exp(-2.0f * inner)));
}

// ---------------------------------------------------------------------------
// RoPE for a vision tower, from precomputed per-token cos/sin tables.
//
// The rotation itself is the same rotate_half pairing cpi_rope uses -- (i, i+head_dim/2) --
// but the angles are not derivable from a scalar position: each patch has a (row, column),
// and the frequency vector is [row_freqs | col_freqs] rather than one geometric series. So the
// table is built on the host (tokens * head_dim/2 floats, once per image) and read here.
//
// Not cpi_rope_2d_inplace from the Gemma 4 tower, which looks applicable and is not: it pairs
// lanes within each axis's half (j, j+pairs_per_half), where this pairs across the midpoint.
// Same idea, different lanes, silently wrong output.
// ---------------------------------------------------------------------------
kernel void cpi_rope_vision(
    device half*        x    [[buffer(0)]],
    device const float* cosb [[buffer(1)]],
    device const float* sinb [[buffer(2)]],
    constant VisRopeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint half_dim = p.head_dim / 2u;
  const uint total = p.tokens * p.heads * half_dim;
  if (gid >= total) return;

  const uint i     = gid % half_dim;
  const uint head  = (gid / half_dim) % p.heads;
  const uint token = gid / (half_dim * p.heads);

  const float c = cosb[token * half_dim + i];
  const float s = sinb[token * half_dim + i];

  // row_stride lets this address q and k inside a fused qkv block without copying them out.
  const ulong base = (ulong)token * (ulong)p.row_stride + (ulong)head * (ulong)p.head_dim + i;
  const float a = float(x[base]);
  const float b = float(x[base + half_dim]);
  x[base]            = half(a * c - b * s);
  x[base + half_dim] = half(a * s + b * c);
}

// ---------------------------------------------------------------------------
// Exact (erf) GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
//
// NOT interchangeable with cpi_gelu above. The vision tower uses both: its blocks specify
// hidden_act = "gelu_pytorch_tanh", but the patch merger constructs a bare nn.GELU(), whose
// default is approximate='none'. The two differ by up to ~4e-4 -- small, systematic, and
// exactly the kind of thing that gets absorbed into a loosened tolerance instead of fixed.
// ---------------------------------------------------------------------------
// metal_stdlib has no erf (nor erfc), so this is Abramowitz & Stegun 7.1.26 -- max absolute
// error 1.5e-7, which is four orders below fp16's resolution and two below the 4e-4 gap
// between exact and tanh GELU that this exists to close.
inline float cpi_erf(float x) {
  const float sign = (x < 0.0f) ? -1.0f : 1.0f;
  const float a = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float poly = t * (0.254829592f +
                     t * (-0.284496736f +
                     t * (1.421413741f +
                     t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - poly * exp(-a * a));
}

kernel void cpi_gelu_erf(
    device half*        x   [[buffer(0)]],
    constant ElemParams& p  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const float v = float(x[gid]);
  x[gid] = half(0.5f * v * (1.0f + cpi_erf(v * 0.70710678118654752f)));  // 1/sqrt(2)
}
