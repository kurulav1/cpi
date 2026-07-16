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

// Keys scored per pass in the attention kernel. The online softmax folds in one block at
// a time, so this is the barrier-amortisation factor.
#define KEY_BLOCK 32

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
  // Batched decode: every row is a DIFFERENT sequence, so row t takes positions[t] rather
  // than base+t. The default (0) keeps the prefill meaning, where the rows are consecutive
  // tokens of one sequence.
  uint per_row_positions;
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
  // Paged prefill (cpi_attention_prefill_mm only; the others ignore these). 0 = the KV is one
  // contiguous run and a key's row IS its position.
  uint paged;
  uint block_size;  // tokens per KV block; only meaningful when paged != 0
};

struct QuantParams {
  uint out_dim;
  uint in_dim;
  uint tokens;
  uint bits;      // 4 or 8
  uint group;     // 0 = one scale per row
  uint groups;    // scales per row
  uint has_bias;  // Qwen2's Q/K/V carry one; quantizing the weights does not remove it
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

#define GEMM_BM 64  // rows per threadgroup
// Tokens per threadgroup. Like the K-depth, the two GEMMs want DIFFERENT widths, measured on
// the 0.5B fp16 prefill and the int4:
//
//   fp16  (GEMM_BN 64): narrow is better. This kernel STAGES the activation tile in
//     threadgroup memory, so a wider tile costs more threadgroup memory and drops occupancy.
//     128 measured 2017 -> 1663 tok/s.
//   quant (GEMM_QBN 128): wide is better. The quant kernel reads activation fragments straight
//     from device (no As tile), so widening only adds simdgroups and weight reuse -- the staged
//     weight block now serves 128 tokens -- at the SAME threadgroup memory. 64 -> 128 measured
//     1527 -> 1649 tok/s.
#define GEMM_BN 64
#define GEMM_QBN 128
// Simdgroups tile the 64-row output as a grid, each owning a 32x32 sub-tile (4x4 fragments).
// Fragments of 8x8 each simdgroup owns: GEMM_RF rows x GEMM_CF cols. This sets arithmetic
// intensity: per 8-deep step a simdgroup issues (RF + CF) simdgroup_loads to do (RF * CF) matrix
// ops, so 4x4 is 8 loads per 16 ops (2.0) and 8x4 would be 12 per 32 (2.67). Everything else --
// thread count, simdgroup grid, row offsets -- DERIVES from these two, so the tile can be swept
// without a hardcoded number drifting out of step, which is how this kernel once came to write
// half its rows.
//
// 4x4 IS THE CEILING, AND THE REASON IS THE REGISTER FILE. Measured with metal_gemm_bench
// (correct at both settings, interleaved, warm):
//
//   RF=4, FBM=64  (32x32/simdgroup, 16 accumulator fragments)   2.90 TFLOP/s
//   RF=8, FBM=128 (64x32/simdgroup, 32 accumulator fragments)   0.38 TFLOP/s   <- 7.6x SLOWER
//
// 32 accumulator fragments is 64 float registers per lane before wf/af, and it spills. Raising
// arithmetic intensity by holding more of the output per thread is therefore not available on
// this hardware: the trade buys fewer loads and pays for them in spill traffic, at a ruinous
// rate.
//
// Worth stating plainly because this REINSTATES a finding that was wrongly overturned. Two
// sessions concluded this kernel was against a register-file wall; that was then "corrected" to
// a simdgroup-starvation diagnosis, on the strength of a 2764 -> 3780 tok/s result which turned
// out to be the kernel skipping half its writes. The starvation theory is dead (more simdgroups
// measured slower once the geometry was consistent). The register-file wall is real.
#define GEMM_RF 4
#define GEMM_CF 4
#define GEMM_SG_ROWS (8u * GEMM_RF)  // rows per simdgroup
#define GEMM_SG_TOKS (8u * GEMM_CF)  // tokens per simdgroup
#define GEMM_SG_COLS (GEMM_BN / GEMM_SG_TOKS)
#define GEMM_QSG_COLS (GEMM_QBN / 32)
// K per stage: how much arithmetic each barrier buys, since the fill and the matrix ops are
// separated by barriers.
//
//   fp16  (GEMM_FBK 32): the depth does not matter. Measured with metal_gemm_bench at 16 / 32
//     / 64: 2.85 / 2.89 / 2.86 TFLOP/s -- a 1.5% spread, i.e. nothing. This block used to
//     claim "fp16 (GEMM_BK 64): deeper is better -- 8 matrix steps per fill instead of 4",
//     describing a GEMM_BK that no kernel had read since the fp16 path moved to GEMM_FBK.
//     The claim was documenting a constant that did nothing. It is deleted rather than
//     corrected in place, because a dead define is how it survived.
//   quant (GEMM_QBK 32): deeper is WORSE. The quant tile also holds a scale row, so doubling
//     K doubles its threadgroup memory, and the 8B's GEMM is compute-bound -- it needs the
//     occupancy to hide latency more than it needs fewer barriers. Measured: 161 -> 148 tok/s.
//     (Unlike the fp16 claim above, GEMM_QBK is live and this has not been re-measured with
//     the current harness.)
//
// GEMM_QBK must also not exceed the quantization group (64 by default), or a K-block would
// straddle two scales. The dispatch guards that.
// GEMM_FBM is the fp16 row tile, and it MUST stay in step with the host's kGemmFBM, which
// derives both the thread count (one simdgroup per 32x32 sub-tile of FBM x BN) and the grid
// (out_dim / FBM row blocks) from it. That coupling is the whole reason this comment exists:
//
// This tile was once raised 64 -> 128 "over 8 simdgroups (256 threads)", reported as
// 2764 -> 3780 tok/s. The kernel changed; the host's thread count did not. So it ran 128
// threads = 4 simdgroups against a 128-row tile and wrote only half the rows of each one --
// and the "speedup" was simply the work it skipped. Nothing failed: the GEMM only runs at
// T >= kGemmMinTokens, and no golden prompt is that long. Re-measured with the geometry
// actually consistent, 128 rows over 8 simdgroups is ~3% SLOWER than 64 over 4, so the tile
// is back to 64 and the premise (more simdgroups hide fragment-load latency) is unsupported.
//
// Both configurations are now checked against a CPU reference in metal_smoke, which derives
// its dispatch from these same constants rather than restating them.
#define GEMM_QBK 32
#define GEMM_FBM 64
#define GEMM_FBK 32

// ---------------------------------------------------------------------------
// Blocked fp16 GEMM (prefill). A K-block of both operands is staged in threadgroup memory,
// so a weight loaded from device is reused by every token in the tile. Two earlier attempts
// streamed the operands from device on every K-step and LOST to the plain GEMV: the matrix
// units were never the bottleneck, the traffic feeding them was.
//
// REGISTER TILING is the second half of that same idea. With one 8x64 strip per simdgroup
// the inner loop issued nine fragment loads to do eight matrix ops -- starved on
// threadgroup traffic instead of device traffic. Each simdgroup now owns a 32x32 output
// tile (4x4 fragments), so the same eight loads feed sixteen matrix ops. The threadgroup is
// 128 threads rather than 256: fewer threads each holding more accumulators is the point.
// ---------------------------------------------------------------------------
kernel void cpi_gemm_f16(device const half* w [[buffer(0)]], device const half* in [[buffer(1)]],
                         device half* out [[buffer(2)]], device const half* bias [[buffer(3)]],
                         constant GemvParams& p [[buffer(4)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint nthr [[threads_per_threadgroup]]) {
  const uint sgid = lid / 32u;
  const uint lane = lid % 32u;
  const uint sg_row = sgid / GEMM_SG_COLS;
  const uint sg_col = sgid % GEMM_SG_COLS;

  const uint row_blocks = p.out_dim / GEMM_FBM;
  const uint row0 = (tgid % row_blocks) * GEMM_FBM;
  const uint tok0 = (tgid / row_blocks) * GEMM_BN;
  if (tok0 >= p.tokens) return;

  threadgroup half Ws[GEMM_FBM * GEMM_FBK];
  threadgroup half As[GEMM_BN * GEMM_FBK];

  simdgroup_float8x8 acc[GEMM_RF][GEMM_CF];
  for (uint i = 0u; i < GEMM_RF; ++i)
    for (uint j = 0u; j < GEMM_CF; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const uint chunks = GEMM_FBK / 8u;  // 8 halves == one 128-bit load

  for (uint k0 = 0u; k0 < p.in_dim; k0 += GEMM_FBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = lid; c < GEMM_FBM * chunks; c += nthr) {
      const uint r = c / chunks, sub = c % chunks;
      *(threadgroup uint4*)(Ws + r * GEMM_FBK + sub * 8u) =
          *(device const uint4*)(w + (ulong)(row0 + r) * (ulong)p.in_dim + k0 + sub * 8u);
    }
    for (uint c = lid; c < GEMM_BN * chunks; c += nthr) {
      const uint t = c / chunks, sub = c % chunks;
      threadgroup uint4* dst = (threadgroup uint4*)(As + t * GEMM_FBK + sub * 8u);
      if (tok0 + t < p.tokens) {
        *dst = *(device const uint4*)(in + (ulong)(tok0 + t) * (ulong)p.in_dim + k0 + sub * 8u);
      } else {
        *dst = uint4(0u);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0u; kk < GEMM_FBK; kk += 8u) {
      simdgroup_half8x8 wf[GEMM_RF], af[GEMM_CF];
      // W is [rows, cols], so a weight fragment loads TRANSPOSED: [8k x 8rows].
      for (uint i = 0u; i < GEMM_RF; ++i)
        simdgroup_load(wf[i], Ws + (sg_row * GEMM_SG_ROWS + i * 8u) * GEMM_FBK + kk, GEMM_FBK,
                       ulong2(0, 0), true);
      for (uint j = 0u; j < GEMM_CF; ++j)
        simdgroup_load(af[j], As + (sg_col * GEMM_SG_TOKS + j * 8u) * GEMM_FBK + kk, GEMM_FBK);
      for (uint i = 0u; i < GEMM_RF; ++i)
        for (uint j = 0u; j < GEMM_CF; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[j], wf[i], acc[i][j]);
    }
  }

  // The K-loop is done with the weight tile; reuse it to stage the accumulators out.
  threadgroup float* mine = (threadgroup float*)Ws + sgid * 64u;
  for (uint i = 0u; i < GEMM_RF; ++i) {
    for (uint j = 0u; j < GEMM_CF; ++j) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_store(acc[i][j], mine, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64u; e += 32u) {
        const uint t = e / 8u;  // token within the fragment
        const uint r = e % 8u;  // row within the fragment
        const uint tok = tok0 + sg_col * GEMM_SG_TOKS + j * 8u + t;
        if (tok >= p.tokens) continue;
        const uint row = row0 + sg_row * GEMM_SG_ROWS + i * 8u + r;
        float v = mine[t * 8u + r];
        if (p.has_bias != 0u) v += float(bias[row]);
        out[(ulong)tok * (ulong)p.out_dim + row] = half(v);
      }
    }
  }
}

// Fills one K-block of the weight tile, dequantizing on the way in.
//
// The tile is stored K-MAJOR -- Ws[k][row], not Ws[row][k]. The weights are [out_dim, in_dim]
// row-major, so a matrix fragment of them is [8k x 8rows]: the TRANSPOSE of how they sit in
// memory. Taking that transpose in simdgroup_load costs real time on every K-step. The
// dequant loop writes scalars anyway, so it can just as easily scatter them into K-major
// order once, and the fragment load becomes an ordinary one.
static inline void load_qblock(threadgroup half* Ws, threadgroup float* sc_row,
                               device const uchar* qw, device const float* scales, uint row0,
                               uint k0, uint groups, uint gsz, uint bits, uint packed_row, uint lid,
                               uint nthr) {
  // Hoisted out of the dequant inner loop: one scale load per row, not one per weight.
  for (uint r = lid; r < GEMM_BM; r += nthr) {
    sc_row[r] = scales[(ulong)(row0 + r) * (ulong)groups + k0 / gsz];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One 8-weight chunk per thread per pass: a single wide load, no byte read twice. Reading
  // the packed weights a byte at a time is the defect that has bitten every kernel here.
  const uint chunks = GEMM_QBK / 8u;
  for (uint c = lid; c < GEMM_BM * chunks; c += nthr) {
    const uint r = c / chunks, sub = c % chunks;
    const uint col0 = k0 + sub * 8u;
    const float sc = sc_row[r];
    if (bits == 4u) {
      const uint packed =  // 4 bytes == 8 packed int4 weights
          *(device const uint*)(qw + (ulong)(row0 + r) * (ulong)packed_row + (col0 >> 1));
      for (uint e = 0u; e < 8u; ++e) {
        const int nib = int((packed >> (4u * e)) & 0xFu);
        Ws[(sub * 8u + e) * GEMM_BM + r] = half(float((nib ^ 0x8) - 0x8) * sc);
      }
    } else {
      const uint2 packed = *(device const uint2*)(qw + (ulong)(row0 + r) * (ulong)packed_row + col0);
      for (uint e = 0u; e < 8u; ++e) {
        const uint word = (e < 4u) ? packed.x : packed.y;
        const int q = int(as_type<char>(uchar((word >> (8u * (e & 3u))) & 0xFFu)));
        Ws[(sub * 8u + e) * GEMM_BM + r] = half(float(q) * sc);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Quantized blocked GEMM. The 8B's prefill is quantized, so an fp16-only GEMM did nothing
// for it -- this is where the prefill gap against llama.cpp actually lived.
//
// Quantization only changes how the weight tile is FILLED. Dequantize a K-block into
// threadgroup memory once -- every weight unpacked exactly once -- and from there it is an
// ordinary fp16 matmul, identical to cpi_gemm_f16: the same 4x4 register tile, the same
// reuse across every token. The quantized GEMV re-unpacked every weight for every token.
// ---------------------------------------------------------------------------
kernel void cpi_gemm_quant(device const uchar* qw [[buffer(0)]],
                           device const half* in [[buffer(1)]], device half* out [[buffer(2)]],
                           device const float* scales [[buffer(3)]],
                           device const half* bias [[buffer(4)]],
                           constant QuantParams& p [[buffer(5)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint nthr [[threads_per_threadgroup]]) {
  const uint sgid = lid / 32u;
  const uint lane = lid % 32u;
  const uint sg_row = sgid / GEMM_QSG_COLS;
  const uint sg_col = sgid % GEMM_QSG_COLS;

  const uint row_blocks = p.out_dim / GEMM_BM;
  const uint row0 = (tgid % row_blocks) * GEMM_BM;
  const uint tok0 = (tgid / row_blocks) * GEMM_QBN;
  if (tok0 >= p.tokens) return;

  // The ACTIVATIONS are not staged. They could be -- and were -- but this GEMM turned out to
  // be occupancy-limited rather than barrier-limited (deepening the K-block halves the
  // barriers and made it 8% SLOWER), so threadgroup memory is the scarce resource. Fragments
  // load straight from device instead: the token block's activations are a few hundred KB and
  // every row-block threadgroup reads the same ones, so they sit in L2. Only the weights,
  // which must be dequantized exactly once, are worth the staging.
  threadgroup half Ws[GEMM_BM * GEMM_QBK];
  threadgroup float sc_row[GEMM_BM];

  simdgroup_float8x8 acc[4][4];
  for (uint i = 0u; i < 4u; ++i)
    for (uint j = 0u; j < 4u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;
  const uint packed_row = (p.bits == 4u) ? ((p.in_dim + 1u) / 2u) : p.in_dim;

  // A tail tile reads past p.tokens. That stays inside the slot buffer, which is sized for a
  // whole prefill chunk, and a garbage token only ever corrupts its OWN output row -- which
  // the store below drops. So there is nothing to mask here.
  device const half* atile = in + (ulong)(tok0 + sg_col * 32u) * (ulong)p.in_dim;

  for (uint k0 = 0u; k0 < p.in_dim; k0 += GEMM_QBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_qblock(Ws, sc_row, qw, scales, row0, k0, p.groups, gsz, p.bits, packed_row, lid, nthr);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0u; kk < GEMM_QBK; kk += 8u) {
      simdgroup_half8x8 wf[4], af[4];
      // Ws is K-major, so [8k x 8rows] loads straight -- no transpose.
      for (uint i = 0u; i < 4u; ++i)
        simdgroup_load(wf[i], Ws + kk * GEMM_BM + sg_row * 32u + i * 8u, GEMM_BM);
      for (uint j = 0u; j < 4u; ++j)
        simdgroup_load(af[j], atile + (ulong)(j * 8u) * (ulong)p.in_dim + k0 + kk, p.in_dim);
      for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[j], wf[i], acc[i][j]);
    }
  }

  threadgroup float* mine = (threadgroup float*)Ws + sgid * 64u;
  for (uint i = 0u; i < 4u; ++i) {
    for (uint j = 0u; j < 4u; ++j) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_store(acc[i][j], mine, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64u; e += 32u) {
        const uint t = e / 8u;
        const uint r = e % 8u;
        const uint tok = tok0 + sg_col * 32u + j * 8u + t;
        if (tok >= p.tokens) continue;
        const uint row = row0 + sg_row * 32u + i * 8u + r;
        float v = mine[t * 8u + r];
        if (p.has_bias != 0u) v += float(bias[row]);
        out[(ulong)tok * (ulong)p.out_dim + row] = half(v);
      }
    }
  }
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

// ---------------------------------------------------------------------------
// FUSED quantized GEMV over 2 or 3 concatenated weight matrices that share the same input.
// Decode fires ~7 separate GEMV dispatches per layer; three of them (Q, K, V) read the same
// normalized input, and two more (gate, up) do too. Each tiny GEMV is launch/latency-bound,
// not bandwidth-bound, so collapsing the group into ONE dispatch removes that fixed per-token
// cost -- which is the dominant share of a bandwidth-light int4 decode. No weight
// pre-concatenation: each output row across the [n0 + n1 + n2] space is ROUTED to its matrix.
// ---------------------------------------------------------------------------
struct QCatParams {
  uint n0, n1, n2;  // rows (out_dim) of each sub-matrix; n2 == 0 for the 2-way (gate/up) case
  uint in_dim;
  uint tokens;
  uint bits;
  uint group;
  uint groups;
  uint has_bias;  // bias is ONE concatenated tensor over the [n0+n1+n2] rows (Qwen's bqkv)
};

// One output row's quantized dot product against the token tile, written to `out`.
static inline void qcat_row(device const uchar* qw, device const float* scales,
                            device const half* in, device half* out, device const half* bias,
                            uint row, uint global_row, uint out_dim_m, uint has_bias, uint in_dim,
                            uint bits, uint gsz, uint groups, uint t0, uint tokens, uint lane) {
  const uint packed_row = (bits == 4u) ? ((in_dim + 1u) / 2u) : in_dim;
  device const float* srow = scales + (ulong)row * (ulong)groups;
  const uint nt = min((uint)GEMV_TILE, tokens - t0);
  float acc[GEMV_TILE];
  for (uint t = 0u; t < GEMV_TILE; ++t) acc[t] = 0.0f;

  if (bits == 4u) {
    device const uint4* w128 = (device const uint4*)(qw + (ulong)row * (ulong)packed_row);
    const uint n32 = in_dim >> 5;  // 32 int4 weights per uint4 (128-bit load)
    for (uint k = lane; k < n32; k += 32u) {
      const uint4 packed = w128[k];
      const uint j0 = k << 5;
      const float sc = srow[j0 / gsz];
      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)in_dim + j0;
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const uint word =
              (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
          for (uint e = 0u; e < 8u; ++e) {
            const int nib = int((word >> (4u * e)) & 0xFu);
            sub += float((nib ^ 0x8) - 0x8) * float(x[w * 8u + e]);
          }
        }
        acc[t] += sub * sc;
      }
    }
    device const uint* w32 = (device const uint*)(qw + (ulong)row * (ulong)packed_row);
    for (uint k = (n32 << 2) + lane; k < (in_dim >> 3); k += 32u) {
      const uint packed = w32[k];
      const uint j0 = k << 3;
      const float sc = srow[j0 / gsz];
      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)in_dim + j0;
        float sub = 0.0f;
        for (uint e = 0u; e < 8u; ++e) {
          const int nib = int((packed >> (4u * e)) & 0xFu);
          sub += float((nib ^ 0x8) - 0x8) * float(x[e]);
        }
        acc[t] += sub * sc;
      }
    }
  } else {
    device const uint4* w16 = (device const uint4*)(qw + (ulong)row * (ulong)in_dim);
    const uint n16 = in_dim >> 4;  // 16 int8s per uint4
    for (uint k = lane; k < n16; k += 32u) {
      const uint4 packed = w16[k];
      const uint j0 = k << 4;
      const float sc = srow[j0 / gsz];
      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)in_dim + j0;
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const uint word =
              (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
          for (uint e = 0u; e < 4u; ++e) {
            const int q = int(char((word >> (8u * e)) & 0xFFu));
            sub += float(q) * float(x[w * 4u + e]);
          }
        }
        acc[t] += sub * sc;
      }
    }
    for (uint k = (n16 << 4) + lane; k < in_dim; k += 32u) {
      const int q = int(((device const char*)(qw + (ulong)row * (ulong)in_dim))[k]);
      const float sc = srow[k / gsz];
      for (uint t = 0u; t < nt; ++t)
        acc[t] += float(q) * sc * float(in[(ulong)(t0 + t) * (ulong)in_dim + k]);
    }
  }

  for (uint t = 0u; t < nt; ++t) {
    float s = simd_sum(acc[t]);
    if (lane == 0u) {
      if (has_bias != 0u) s += float(bias[global_row]);
      out[(ulong)(t0 + t) * (ulong)out_dim_m + row] = half(s);
    }
  }
}

kernel void cpi_gemv_quant_cat(
    device const half*   in   [[buffer(0)]],
    device const uchar*  qw0  [[buffer(1)]], device const float* sc0 [[buffer(2)]],
    device half*         out0 [[buffer(3)]],
    device const uchar*  qw1  [[buffer(4)]], device const float* sc1 [[buffer(5)]],
    device half*         out1 [[buffer(6)]],
    device const uchar*  qw2  [[buffer(7)]], device const float* sc2 [[buffer(8)]],
    device half*         out2 [[buffer(9)]],
    device const half*   bias [[buffer(10)]],
    constant QCatParams& p    [[buffer(11)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;

  const uint total = p.n0 + p.n1 + p.n2;
  const uint row_blocks = (total + simds_per_tg - 1u) / simds_per_tg;
  const uint tile = gid / row_blocks;
  const uint t0 = tile * GEMV_TILE;
  const uint grow = (gid % row_blocks) * simds_per_tg + simd_id;  // global row
  if (t0 >= p.tokens || grow >= total) return;

  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;
  if (grow < p.n0) {
    qcat_row(qw0, sc0, in, out0, bias, grow, grow, p.n0, p.has_bias, p.in_dim, p.bits, gsz,
             p.groups, t0, p.tokens, lane);
  } else if (grow < p.n0 + p.n1) {
    const uint r = grow - p.n0;
    qcat_row(qw1, sc1, in, out1, bias, r, grow, p.n1, p.has_bias, p.in_dim, p.bits, gsz, p.groups,
             t0, p.tokens, lane);
  } else {
    const uint r = grow - p.n0 - p.n1;
    qcat_row(qw2, sc2, in, out2, bias, r, grow, p.n2, p.has_bias, p.in_dim, p.bits, gsz, p.groups,
             t0, p.tokens, lane);
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

  uint pos;
  if (p.per_row_positions != 0u) {
    pos = uint(positions[token]);  // batched decode: row t is its own sequence
  } else {
    pos = (p.use_position_buffer != 0u) ? uint(positions[0]) : p.position;
    pos += token;  // sequence prefill: token t sits at position base + t
  }

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
// Prefill attention: one threadgroup per (QUERY BLOCK, head).
//
// The decode kernel below takes one threadgroup per (query token, head), and each of those
// walks the whole KV cache itself. Across a prompt that is O(T^2) of DEVICE traffic, and on
// the 8B it measured ~80 GB at ~73 GB/s -- genuinely bandwidth-bound, and 23% of prefill.
//
// A query block fixes it: the block of keys a threadgroup pulls in now serves Q_BLOCK
// queries instead of one, so the same answer costs Q_BLOCK times less traffic. The keys and
// values are re-read once per query WITHIN the block, but that is an L1 hit -- a key block
// is a few KB and the reuse is immediate.
//
// Each query keeps its OWN online-softmax state (running max, running sum), because each
// attends to a different prefix: causality masks key > that query's position.
// ---------------------------------------------------------------------------
#define Q_BLOCK 8

// ---------------------------------------------------------------------------
// Prefill attention on the MATRIX UNITS (head_dim <= 128). The scalar kernel below scores
// QK^T and does P.V with per-thread dot products -- the matrix units sit idle through the
// whole of attention, which the RCA put at ~21% of prefill. This computes both products with
// simdgroup_matrix (fp32 accumulate, the fast path), keeping the same per-query online softmax.
//
// One threadgroup per (query block of 8, head); 8 simdgroups. Q_BLOCK==8 is exactly one 8-row
// matrix fragment. For a KEY_BLOCK of 32 keys:
//   scoring  S[8q x 8k]  = sum over head_dim/8 of  Q[8q x 8d] @ (K[8k x 8d])^T   (K loaded transposed)
//   P.V      O[8q x 8d] += sum over the key groups of  P[8q x 8k] @ V[8k x 8d]
// The online rescale between key blocks is applied to the fp32 accumulator in threadgroup
// memory (a per-row scalar, which a matrix fragment cannot scale), then the P.V matmul
// accumulates into it. Gemma's head_dim 256 does not fit the fragment plan and keeps the
// scalar kernel.
// ---------------------------------------------------------------------------
kernel void cpi_attention_prefill_mm(device const half* q [[buffer(0)]],
                                     device const half* k_cache [[buffer(1)]],
                                     device const half* v_cache [[buffer(2)]],
                                     device half* out [[buffer(3)]],
                                     device const int* positions [[buffer(4)]],
                                     // ONE sequence's logical->physical block map, used only when
                                     // p.paged != 0. Bound to a dummy otherwise (the params block
                                     // must stay last, so the slot cannot simply be absent).
                                     device const int* block_tables [[buffer(5)]],
                                     constant AttnParams& p [[buffer(6)]],
                                     uint gid [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]],
                                     uint nthr [[threads_per_threadgroup]]) {
  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * Q_BLOCK;
  if (t0 >= p.tokens) return;
  const uint nq = min((uint)Q_BLOCK, p.tokens - t0);

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint q_dim = p.heads * p.head_dim;
  const uint hd = p.head_dim;  // <= 128

  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;
  const uint n_simd = nthr / 32u;

  threadgroup half q_sh[Q_BLOCK * 128];         // [query][d]
  threadgroup float acc[Q_BLOCK * 128];         // [query][d] fp32 output accumulator
  threadgroup float sc_sh[Q_BLOCK * KEY_BLOCK]; // scores [query][key]
  threadgroup half pw_sh[Q_BLOCK * KEY_BLOCK];  // softmax weights [query][key], half for the matmul
  threadgroup float m_sh[Q_BLOCK];              // running max per query
  threadgroup float l_sh[Q_BLOCK];              // running sum per query
  threadgroup float r_sh[Q_BLOCK];              // this block's rescale per query

  // Zero the FULL Q_BLOCK rows, not just nq: the matrix ops load 8-row fragments, so the
  // padding rows must be defined (0) rather than garbage that could turn into NaN.
  for (uint c = lid; c < Q_BLOCK * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    q_sh[qi * hd + i] = (qi < nq) ? q[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i] : half(0);
    acc[qi * hd + i] = 0.0f;
  }
  for (uint qi = lid; qi < Q_BLOCK; qi += nthr) {
    m_sh[qi] = -INFINITY;
    l_sh[qi] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint last_pos = base + t0 + nq - 1u;
  const uint first_pos = base + t0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;

  const uint dfs = hd / 8u;  // head_dim fragments

  for (uint kb = start; kb <= last_pos; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, last_pos - kb + 1u);
    const uint n_kg = (nk + 7u) / 8u;  // 8-key groups this block

    // Where this key block physically lives. ONE lookup per block, not per key: the caller
    // guarantees block_size % KEY_BLOCK == 0 and no sliding window (so start == 0 and every kb
    // is KEY_BLOCK-aligned), which together mean the whole run [kb, kb+KEY_BLOCK) sits inside a
    // single physical block and stays contiguous -- so the simdgroup_loads below keep working
    // unchanged, just from a different base. Break either guarantee and a key block could
    // straddle two blocks, which this addressing cannot express; the dispatch enforces both.
    const uint kphys = (p.paged != 0u)
                           ? uint(block_tables[kb / p.block_size]) * p.block_size + (kb % p.block_size)
                           : kb;

    // --- Scoring: one simdgroup per key group computes S[8q x 8k] over the head dim. ---
    if (simd_id < n_kg) {
      const uint kg = simd_id;
      simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (uint df = 0u; df < dfs; ++df) {
        simdgroup_half8x8 Qf, Kf;
        simdgroup_load(Qf, q_sh + df * 8u, hd);  // [8q x 8d]
        // K is [key][d]; load the [8k x 8d] tile TRANSPOSED to get [8d x 8k].
        simdgroup_load(Kf, k_cache + (ulong)(kphys + kg * 8u) * (ulong)kv_dim + kv_head * hd + df * 8u,
                       kv_dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(S, Qf, Kf, S);
      }
      // Store straight into sc_sh[query][key] at this key group's column offset.
      simdgroup_store(S, sc_sh + kg * 8u, KEY_BLOCK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scale + causal/window mask, in place.
    for (uint c = lid; c < nq * nk; c += nthr) {
      const uint qi = c / nk, j = c % nk;
      const uint key = kb + j;
      const uint pos_i = base + t0 + qi;
      uint start_i = 0u;
      if (p.window != 0u && pos_i + 1u > p.window) start_i = pos_i + 1u - p.window;
      const bool keep = (key <= pos_i && key >= start_i);
      sc_sh[qi * KEY_BLOCK + j] = keep ? sc_sh[qi * KEY_BLOCK + j] * p.scale : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, one simdgroup per query; lane j owns key j (KEY_BLOCK == 32 == simd width).
    const uint qi = simd_id;
    if (qi < nq) {
      const float sc = (lane < nk) ? sc_sh[qi * KEY_BLOCK + lane] : -INFINITY;
      const float bmax = simd_max(sc);
      const float old_max = m_sh[qi];
      const float new_max = max(old_max, bmax);
      const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);
      const float w =
          (lane < nk && sc != -INFINITY && new_max != -INFINITY) ? exp(sc - new_max) : 0.0f;
      pw_sh[qi * KEY_BLOCK + lane] = half(w);  // all 32 lanes: 0 past nk, so P.V masks itself
      const float bsum = simd_sum(w);
      if (lane == 0u) {
        l_sh[qi] = l_sh[qi] * rescale + bsum;
        m_sh[qi] = new_max;
        r_sh[qi] = rescale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rescale the running fp32 accumulator by this block's per-query factor.
    for (uint c = lid; c < nq * hd; c += nthr) acc[c] = acc[c] * r_sh[c / hd];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- P.V: O[8q x 8d] += P[8q x 8k] @ V[8k x 8d], accumulated into the fp32 acc. ---
    for (uint dg = simd_id; dg < dfs; dg += n_simd) {
      simdgroup_float8x8 O;
      simdgroup_load(O, acc + dg * 8u, hd);  // current (rescaled) accumulator
      for (uint kg = 0u; kg < n_kg; ++kg) {
        simdgroup_half8x8 Pf, Vf;
        simdgroup_load(Pf, pw_sh + kg * 8u, KEY_BLOCK);  // [8q x 8k]
        simdgroup_load(Vf, v_cache + (ulong)(kphys + kg * 8u) * (ulong)kv_dim + kv_head * hd + dg * 8u,
                       kv_dim);  // [8k x 8d]
        simdgroup_multiply_accumulate(O, Pf, Vf, O);
      }
      simdgroup_store(O, acc + dg * 8u, hd);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint c = lid; c < nq * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    const float inv = (l_sh[qi] > 0.0f) ? 1.0f / l_sh[qi] : 0.0f;
    out[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i] = half(acc[qi * hd + i] * inv);
  }
}

kernel void cpi_attention_prefill(device const half* q [[buffer(0)]],
                                  device const half* k_cache [[buffer(1)]],
                                  device const half* v_cache [[buffer(2)]],
                                  device half* out [[buffer(3)]],
                                  device const int* positions [[buffer(4)]],
                                  constant AttnParams& p [[buffer(5)]],
                                  uint gid [[threadgroup_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  uint nthr [[threads_per_threadgroup]]) {
  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * Q_BLOCK;
  if (t0 >= p.tokens) return;
  const uint nq = min((uint)Q_BLOCK, p.tokens - t0);

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint q_dim = p.heads * p.head_dim;
  const uint hd = p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;
  const uint n_simd = nthr / 32u;

  threadgroup float q_sh[Q_BLOCK * 256];
  threadgroup float acc[Q_BLOCK * 256];
  threadgroup float sc_sh[Q_BLOCK * KEY_BLOCK];
  threadgroup float w_sh[Q_BLOCK * KEY_BLOCK];
  threadgroup float m_sh[Q_BLOCK];  // running max, per query
  threadgroup float l_sh[Q_BLOCK];  // running sum, per query
  threadgroup float r_sh[Q_BLOCK];  // this block's rescale, per query

  for (uint c = lid; c < nq * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    q_sh[qi * hd + i] = float(q[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i]);
    acc[qi * hd + i] = 0.0f;
  }
  for (uint qi = lid; qi < nq; qi += nthr) {
    m_sh[qi] = -INFINITY;
    l_sh[qi] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The block's queries span positions [base+t0, base+t0+nq-1]; the last one reaches
  // furthest, and the first one's window (if any) starts earliest.
  const uint last_pos = base + t0 + nq - 1u;
  const uint first_pos = base + t0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;

  for (uint kb = start; kb <= last_pos; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, last_pos - kb + 1u);

    // Score every (query, key) pair: ONE THREAD per pair, a full dot product with no
    // cross-lane reduction. The previous version gave each pair a whole simdgroup and summed
    // across 32 lanes -- five shuffle steps of reduction overhead for ~two useful multiplies
    // per lane. nq*nk <= Q_BLOCK*KEY_BLOCK == nthr, so this is a single pass at full occupancy.
    for (uint pidx = lid; pidx < nq * nk; pidx += nthr) {
      const uint qi = pidx / nk, j = pidx % nk;
      const uint key = kb + j;
      const uint pos_i = base + t0 + qi;

      uint start_i = 0u;
      if (p.window != 0u && pos_i + 1u > p.window) start_i = pos_i + 1u - p.window;

      float d = -INFINITY;
      if (key <= pos_i && key >= start_i) {  // causal mask, plus the sliding window
        device const half* kt = k_cache + (ulong)key * (ulong)kv_dim + kv_head * hd;
        float acc_d = 0.0f;
        for (uint i = 0u; i < hd; ++i) acc_d += q_sh[qi * hd + i] * float(kt[i]);
        d = acc_d * p.scale;
      }
      sc_sh[qi * KEY_BLOCK + j] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fold the block into each query's own online softmax. ONE SIMDGROUP per query, so all
    // 256 threads work -- the old thread-per-query version left 8 threads busy and 248 idle at
    // the barrier below. KEY_BLOCK == 32 == simd width, so lane j owns key j and the max/sum are
    // lane-parallel reductions.
    const uint qi = simd_id;
    if (qi < nq) {
      // A fully masked block leaves new_max at -INFINITY; guard so exp(-inf - -inf) is not NaN.
      const float sc = (lane < nk) ? sc_sh[qi * KEY_BLOCK + lane] : -INFINITY;
      const float bmax = simd_max(sc);

      const float old_max = m_sh[qi];
      const float new_max = max(old_max, bmax);
      const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

      const float w =
          (lane < nk && sc != -INFINITY && new_max != -INFINITY) ? exp(sc - new_max) : 0.0f;
      if (lane < nk) w_sh[qi * KEY_BLOCK + lane] = w;
      const float bsum = simd_sum(w);

      if (lane == 0u) {
        l_sh[qi] = l_sh[qi] * rescale + bsum;
        m_sh[qi] = new_max;
        r_sh[qi] = rescale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = lid; c < nq * hd; c += nthr) {
      const uint qi = c / hd, i = c % hd;
      float a = acc[qi * hd + i] * r_sh[qi];
      for (uint j = 0u; j < nk; ++j) {
        device const half* vt = v_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * hd;
        a += w_sh[qi * KEY_BLOCK + j] * float(vt[i]);
      }
      acc[qi * hd + i] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint c = lid; c < nq * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    const float inv = (l_sh[qi] > 0.0f) ? 1.0f / l_sh[qi] : 0.0f;
    out[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i] = half(acc[qi * hd + i] * inv);
  }
}

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
  // One threadgroup per (query token, head). Decode is just T = 1, so prefill and decode
  // share this kernel instead of duplicating the softmax and the GQA maths.
  //
  // KEYS ARE PROCESSED IN BLOCKS. The first version walked the cache ONE KEY AT A TIME,
  // with two threadgroup barriers per key -- about 1100 barriers per (token, head) on a
  // 551-token prompt, which made prefill attention O(T^2) in barriers as well as in work.
  // Scoring a block of 32 keys and folding it into the online softmax ONCE cuts the
  // barrier count ~20x and lets the score dot-products run across all simdgroups at once.
  const uint total = p.heads * p.tokens;
  if (gid >= total) return;
  const uint t    = gid / p.heads;
  const uint head = gid % p.heads;

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);
  const uint pos = base + t;  // causality: token t attends to keys [start, base+t]

  const uint group   = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim  = p.kv_heads * p.head_dim;
  const uint q_dim   = p.heads * p.head_dim;

  uint start = 0u;
  if (p.window != 0u && pos + 1u > p.window) start = pos + 1u - p.window;

  device const half* qh = q + (ulong)t * (ulong)q_dim + head * p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane    = lid % 32u;
  const uint n_simd  = nthr / 32u;

  threadgroup float q_sh[256];    // the query, read once instead of per key
  threadgroup float sc_sh[KEY_BLOCK];
  threadgroup float w_sh[KEY_BLOCK];
  threadgroup float tg_acc[256];
  threadgroup float tg_max;
  threadgroup float tg_sum;

  for (uint i = lid; i < p.head_dim; i += nthr) {
    q_sh[i] = float(qh[i]);
    tg_acc[i] = 0.0f;
  }
  if (lid == 0u) {
    tg_max = -INFINITY;
    tg_sum = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kb = start; kb <= pos; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, pos - kb + 1u);

    // Score the whole block: each simdgroup takes a key and reduces its dot product.
    for (uint j = simd_id; j < nk; j += n_simd) {
      device const half* kt = k_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * p.head_dim;
      float d = 0.0f;
      for (uint i = lane; i < p.head_dim; i += 32u) d += q_sh[i] * float(kt[i]);
      d = simd_sum(d);
      if (lane == 0u) sc_sh[j] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, folded in ONCE for the block rather than once per key.
    float bmax = -INFINITY;
    for (uint j = 0u; j < nk; ++j) bmax = max(bmax, sc_sh[j]);

    const float old_max = tg_max;
    const float new_max = max(old_max, bmax);
    const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

    for (uint j = lid; j < nk; j += nthr) w_sh[j] = exp(sc_sh[j] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bsum = 0.0f;
    for (uint j = 0u; j < nk; ++j) bsum += w_sh[j];

    for (uint i = lid; i < p.head_dim; i += nthr) {
      float a = tg_acc[i] * rescale;
      for (uint j = 0u; j < nk; ++j) {
        device const half* vt = v_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * p.head_dim;
        a += w_sh[j] * float(vt[i]);
      }
      tg_acc[i] = a;
    }

    // Every thread read tg_max above; thread 0 is about to overwrite it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + bsum;
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
// Batched paged decode -- the kernels continuous batching needs.
//
// The kernels above serve ONE sequence, whose KV is contiguous: the key at logical
// position p lives at p * kv_dim. Continuous batching cannot use that layout, because the
// sequences in a batch start, grow and finish independently -- a contiguous per-sequence
// cache would have to reserve max_context for every slot. Instead the cache is one pool per
// layer, carved into fixed-size blocks, and each sequence owns a block table mapping its
// logical positions onto physical blocks it does not have to own contiguously:
//
//   phys = block_table[b][pos / block_size] * block_size + (pos % block_size)
//
// That is deliberately the same arithmetic, pool layout and block-table encoding the CUDA
// paged kernels use, so a block table built by the shared host-side allocator
// (engine::SequenceBlockTable, which has no backend in it at all) means the same thing on
// both backends. Blocks are refcounted, so two sequences sharing a system prompt share its
// blocks outright -- which is why the table is a gather rather than a base offset.
// ---------------------------------------------------------------------------

struct KvPagedParams {
  uint kv_hidden;   // kv_heads * head_dim
  uint max_blocks;  // row stride of block_tables
  uint block_size;  // tokens per block
  uint batch;
};

// Scatter each sequence's one new K/V row into its own block. One threadgroup per sequence.
kernel void cpi_kv_store_batched_paged(
    device const half*  k_src        [[buffer(0)]],  // [batch][kv_hidden]
    device const half*  v_src        [[buffer(1)]],
    device half*        k_pool       [[buffer(2)]],
    device half*        v_pool       [[buffer(3)]],
    device const int*   block_tables [[buffer(4)]],  // [batch][max_blocks]
    device const int*   positions    [[buffer(5)]],  // [batch] -- each row's own position
    constant KvPagedParams& p        [[buffer(6)]],
    uint b    [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  if (b >= p.batch) return;

  const uint pos = uint(positions[b]);
  device const int* bt = block_tables + (ulong)b * (ulong)p.max_blocks;
  const uint phys = uint(bt[pos / p.block_size]) * p.block_size + (pos % p.block_size);

  device half* kd = k_pool + (ulong)phys * (ulong)p.kv_hidden;
  device half* vd = v_pool + (ulong)phys * (ulong)p.kv_hidden;
  device const half* ks = k_src + (ulong)b * (ulong)p.kv_hidden;
  device const half* vs = v_src + (ulong)b * (ulong)p.kv_hidden;
  for (uint d = lid; d < p.kv_hidden; d += nthr) {
    kd[d] = ks[d];
    vd[d] = vs[d];
  }
}

struct AttnPagedParams {
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint max_blocks;
  uint block_size;
  uint window;  // 0 = full causal; else sliding window length
  float scale;
  uint batch;
};

// One threadgroup per (sequence, head). Each sequence attends over its OWN length, gathered
// through its OWN block table -- the ragged batch the scheduler hands us. Structurally this
// is cpi_attention_decode with the contiguous key address replaced by a paged gather; the
// online softmax is unchanged, so the two agree key-for-key when a block table happens to be
// identity (which is what the parity gate checks).
kernel void cpi_attention_decode_batched_paged(
    device const half*  q            [[buffer(0)]],  // [batch][heads*head_dim]
    device const half*  k_pool       [[buffer(1)]],
    device const half*  v_pool       [[buffer(2)]],
    device half*        out          [[buffer(3)]],  // [batch][heads*head_dim]
    device const int*   block_tables [[buffer(4)]],  // [batch][max_blocks]
    device const int*   seq_lens     [[buffer(5)]],  // [batch] -- length INCLUDING the new token
    constant AttnPagedParams& p      [[buffer(6)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint total = p.batch * p.heads;
  if (gid >= total) return;
  const uint b    = gid / p.heads;
  const uint head = gid % p.heads;

  const uint seq_len = uint(seq_lens[b]);
  if (seq_len == 0u) return;
  const uint pos = seq_len - 1u;  // the query is this sequence's newest token

  const uint group   = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim  = p.kv_heads * p.head_dim;
  const uint q_dim   = p.heads * p.head_dim;

  uint start = 0u;
  if (p.window != 0u && seq_len > p.window) start = seq_len - p.window;

  device const int* bt = block_tables + (ulong)b * (ulong)p.max_blocks;
  device const half* qh = q + (ulong)b * (ulong)q_dim + head * p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane    = lid % 32u;
  const uint n_simd  = nthr / 32u;

  threadgroup float q_sh[256];
  threadgroup float sc_sh[KEY_BLOCK];
  threadgroup float w_sh[KEY_BLOCK];
  threadgroup float tg_acc[256];
  threadgroup float tg_max;
  threadgroup float tg_sum;

  for (uint i = lid; i < p.head_dim; i += nthr) {
    q_sh[i] = float(qh[i]);
    tg_acc[i] = 0.0f;
  }
  if (lid == 0u) {
    tg_max = -INFINITY;
    tg_sum = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kb = start; kb <= pos; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, pos - kb + 1u);

    for (uint j = simd_id; j < nk; j += n_simd) {
      const uint kpos = kb + j;
      const uint phys = uint(bt[kpos / p.block_size]) * p.block_size + (kpos % p.block_size);
      device const half* kt = k_pool + (ulong)phys * (ulong)kv_dim + kv_head * p.head_dim;
      float d = 0.0f;
      for (uint i = lane; i < p.head_dim; i += 32u) d += q_sh[i] * float(kt[i]);
      d = simd_sum(d);
      if (lane == 0u) sc_sh[j] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bmax = -INFINITY;
    for (uint j = 0u; j < nk; ++j) bmax = max(bmax, sc_sh[j]);

    const float old_max = tg_max;
    const float new_max = max(old_max, bmax);
    const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

    for (uint j = lid; j < nk; j += nthr) w_sh[j] = exp(sc_sh[j] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bsum = 0.0f;
    for (uint j = 0u; j < nk; ++j) bsum += w_sh[j];

    for (uint i = lid; i < p.head_dim; i += nthr) {
      float a = tg_acc[i] * rescale;
      for (uint j = 0u; j < nk; ++j) {
        const uint kpos = kb + j;
        const uint phys = uint(bt[kpos / p.block_size]) * p.block_size + (kpos % p.block_size);
        device const half* vt = v_pool + (ulong)phys * (ulong)kv_dim + kv_head * p.head_dim;
        a += w_sh[j] * float(vt[i]);
      }
      tg_acc[i] = a;
    }

    // Every thread read tg_max above; thread 0 is about to overwrite it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + bsum;
      tg_max = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / tg_sum;
  for (uint i = lid; i < p.head_dim; i += nthr) {
    out[(ulong)b * (ulong)q_dim + head * p.head_dim + i] = half(tg_acc[i] * inv);
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
