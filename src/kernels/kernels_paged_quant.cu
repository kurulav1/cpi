// kernels_paged_quant.cu
//
// Quantized-KV kernels for the block-paged pool: prefill/decode KV scatter and
// batched GQA-shared split-K decode attention. Same quantization recipe as the
// contiguous path (kv_quant_detail.cuh): per (token, kv_head) fp16 absmax
// scale, 4- or 8-bit elements, optional R3 Hadamard on K matched by a Q
// rotation at read time.
//
// Pool layout mirrors the fp16 paged pool: a flat run of
// (num_blocks * block_size) physical token slots per layer, row stride
// head_dim * bits / 8 bytes per (slot, kv_head), with a parallel
// [slots, kv_heads] fp16 scale array. Block indirection is unchanged
// (tokens-per-block invariant), so a block plus its scale slice is
// self-describing and fork-shareable exactly like an fp16 block.
//
// v1 scope: the GQA-shared attention requires head_dim 128 and a real GQA
// group with group_size * 32 >= head_dim (the engine gates eligibility);
// chunk coarsening (blocks_per_chunk > 1) is a follow-up.
//
// fp16 sink/window override (the quality tier): each sequence owns a stable
// quality slot (assigned by the scheduler for its whole lifetime) in the
// per-slot side buffers [slots, sink_n|win_n, kv_heads, head_dim]. The store
// kernels dual-write the (rotated) fp16 K and raw fp16 V of sink positions
// (< sink_n) and of every position into ring slot position % win_n; the
// decode attention reads those fp16 copies for sink and recent-window tokens
// and the quantized pool otherwise, exactly like the contiguous-cache
// kernels. The prefill store only ring-writes each launch's tail win_n
// positions: two positions in one grid sharing position % win_n would race,
// and only the tail can still be inside the window when the launch's last
// position is current (chunks launch in ascending order on one stream, so
// later chunks overwrite older residues correctly). The paged prefill
// ATTENTION still reads the quantized pool only (the quant copy of every
// position is always written, so this is correct; it just processes the
// prompt at the no-window tier).

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "runtime/kernels.cuh"
#include "runtime/kv_quant_detail.cuh"

namespace kernels {
namespace {

__device__ __forceinline__ float pq_warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

constexpr float kPqNegInf = -3.402823466e+38F;

// Mirror of kv_fp16_source in kernels_attention_decode_int4.cu: sink tokens
// (absolute < sink_n) and recent tokens (absolute >= seq_end - win_n) read the
// fp16 side buffers written by the store kernels instead of the quantized
// pool. `sink`/`ring` are already slot-resolved.
__device__ __forceinline__ const half* pq_fp16_source(const half* sink, const half* ring, int t_abs,
                                                      int seq_end_abs, int sink_n, int win_n,
                                                      int num_kv_heads, int kv_head, int head_dim) {
  if (t_abs < sink_n && sink != nullptr) {
    return sink + (t_abs * num_kv_heads + kv_head) * head_dim;
  }
  if (win_n > 0 && ring != nullptr && t_abs >= seq_end_abs - win_n) {
    return ring + ((t_abs % win_n) * num_kv_heads + kv_head) * head_dim;
  }
  return nullptr;
}

// Dual-write one head's fp16 K/V (K already rotated when RotK) into the
// slot-resolved sink/ring side buffers. `ring_write` lets the prefill store
// restrict ring writes to its race-free tail.
__device__ __forceinline__ void pq_sink_ring_store(const float* k_buf, const float* v_buf,
                                                   half* k_sink, half* v_sink, half* k_ring,
                                                   half* v_ring, int sink_n, int win_n, int pos,
                                                   bool ring_write, int num_kv_heads, int kv_head,
                                                   int head_dim, int tid) {
  if (ring_write && win_n > 0 && k_ring != nullptr) {
    const int slot = pos % win_n;
    const int ri = (slot * num_kv_heads + kv_head) * head_dim + tid;
    k_ring[ri] = __float2half(k_buf[tid]);
    v_ring[ri] = __float2half(v_buf[tid]);
  }
  if (pos < sink_n && k_sink != nullptr) {
    const int si = (pos * num_kv_heads + kv_head) * head_dim + tid;
    k_sink[si] = __float2half(k_buf[tid]);
    v_sink[si] = __float2half(v_buf[tid]);
  }
}

// Scatter + quantize `rows` contiguous KV rows into the paged pool at
// positions base_pos..base_pos+rows-1. Grid: (rows, num_kv_heads);
// block: head_dim threads. src_stride is in halves (kv_hidden for the plain
// layout, q_hidden + 2*kv_hidden when reading K/V in place from fused QKV).
template <int KBits, int VBits, bool RotK>
__global__ void store_kv_paged_quant_kernel(const half* k_src, const half* v_src, int src_stride,
                                            int8_t* k_pool, int8_t* v_pool, half* k_scales,
                                            half* v_scales, const int* __restrict__ block_table,
                                            int base_pos, int rows, int num_kv_heads, int head_dim,
                                            int block_size, half* k_sink, half* v_sink,
                                            half* k_ring, half* v_ring, int sink_n, int win_n) {
  extern __shared__ float smem[];
  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int pos = base_pos + row;
  const int phys = block_table[pos / block_size] * block_size + (pos % block_size);
  const int k_row_b = head_dim * KBits / 8;
  const int v_row_b = head_dim * VBits / 8;

  float* k_buf = smem;
  float* v_buf = k_buf + head_dim;
  const half* ks = k_src + static_cast<std::size_t>(row) * src_stride + kv_head * head_dim;
  const half* vs = v_src + static_cast<std::size_t>(row) * src_stride + kv_head * head_dim;
  k_buf[tid] = __half2float(ks[tid]);
  v_buf[tid] = __half2float(vs[tid]);
  __syncthreads();
  if (RotK) {
    kvq::fwht_shared(k_buf, tid, head_dim);
    k_buf[tid] *= rsqrtf(static_cast<float>(head_dim));
    __syncthreads();
  }
  // Ring writes only for this launch's tail win_n positions: residues are then
  // unique within the grid (no WAW race), and any position still inside the
  // window when the launch's last position is current lies in that tail.
  pq_sink_ring_store(k_buf, v_buf, k_sink, v_sink, k_ring, v_ring, sink_n, win_n, pos,
                     pos >= base_pos + rows - win_n, num_kv_heads, kv_head, head_dim, tid);
  const std::size_t sh = static_cast<std::size_t>(phys) * num_kv_heads + kv_head;
  kvq::quant_store_head<KBits, VBits>(smem, tid, head_dim, k_pool + sh * k_row_b,
                                      v_pool + sh * v_row_b, k_scales + sh, v_scales + sh);
}

// Batched decode scatter: one token per sequence, each through its own block
// table at its own position. Grid: (batch, num_kv_heads); block: head_dim.
template <int KBits, int VBits, bool RotK>
__global__ void store_kv_batched_paged_quant_kernel(
    const half* k_src, const half* v_src, int8_t* k_pool, int8_t* v_pool, half* k_scales,
    half* v_scales, const int* __restrict__ block_tables, const int* __restrict__ positions,
    int max_blocks, int batch, int num_kv_heads, int head_dim, int block_size,
    const int* __restrict__ slot_ids, half* k_sink, half* v_sink, half* k_ring, half* v_ring,
    int sink_n, int win_n) {
  extern __shared__ float smem[];
  const int b = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (b >= batch) return;

  const int pos = positions[b];
  const int* bt = block_tables + static_cast<std::size_t>(b) * max_blocks;
  const int phys = bt[pos / block_size] * block_size + (pos % block_size);
  const int kv_hidden = num_kv_heads * head_dim;
  const int k_row_b = head_dim * KBits / 8;
  const int v_row_b = head_dim * VBits / 8;

  float* k_buf = smem;
  float* v_buf = k_buf + head_dim;
  const half* ks = k_src + static_cast<std::size_t>(b) * kv_hidden + kv_head * head_dim;
  const half* vs = v_src + static_cast<std::size_t>(b) * kv_hidden + kv_head * head_dim;
  k_buf[tid] = __half2float(ks[tid]);
  v_buf[tid] = __half2float(vs[tid]);
  __syncthreads();
  if (RotK) {
    kvq::fwht_shared(k_buf, tid, head_dim);
    k_buf[tid] *= rsqrtf(static_cast<float>(head_dim));
    __syncthreads();
  }
  if (slot_ids != nullptr && (k_sink != nullptr || k_ring != nullptr)) {
    const int slot = slot_ids[b];
    const std::size_t soff = static_cast<std::size_t>(slot) * sink_n * kv_hidden;
    const std::size_t roff = static_cast<std::size_t>(slot) * win_n * kv_hidden;
    pq_sink_ring_store(k_buf, v_buf, k_sink ? k_sink + soff : nullptr,
                       v_sink ? v_sink + soff : nullptr, k_ring ? k_ring + roff : nullptr,
                       v_ring ? v_ring + roff : nullptr, sink_n, win_n, pos, true, num_kv_heads,
                       kv_head, head_dim, tid);
  }
  const std::size_t sh = static_cast<std::size_t>(phys) * num_kv_heads + kv_head;
  kvq::quant_store_head<KBits, VBits>(smem, tid, head_dim, k_pool + sh * k_row_b,
                                      v_pool + sh * v_row_b, k_scales + sh, v_scales + sh);
}

// Batched + paged GQA-shared split-K pass 1 over the quantized pool. Structure
// follows gqa_split_chunk_stats_batched_core with blocks_per_chunk == 1: grid
// (num_kv_heads, paged_block, batch), group_size warps share each dequantized
// KV tile. K and V are dequantized into the shared tile in disjoint phases.
template <int HeadDim, int KBits, int VBits, bool RotK>
__global__ void attention_step_gqa_batched_paged_quant_kernel(
    const half* q, const int8_t* k_pool, const int8_t* v_pool, const half* k_scales,
    const half* v_scales, const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int max_blocks, float* chunk_m, float* chunk_l, float* chunk_o, int num_heads, int num_kv_heads,
    int group_size, int block_size, int scratch_chunks, const int* __restrict__ slot_ids,
    const half* k_sink, const half* v_sink, const half* k_ring, const half* v_ring, int sink_n,
    int win_n) {
  const int b = blockIdx.z;
  const int seq_len = seq_lens[b];
  const int kv_head = blockIdx.x;
  const int pb = blockIdx.y;  // paged block index (== chunk)
  const int tok0 = pb * block_size;
  if (tok0 >= seq_len) return;

  extern __shared__ unsigned char smem_bytes[];
  half* q_sh = reinterpret_cast<half*>(smem_bytes);                        // group_size*HeadDim
  half* kv_tile = q_sh + group_size * HeadDim;                             // block_size*HeadDim
  float* w_sh = reinterpret_cast<float*>(kv_tile + block_size * HeadDim);  // group_size*block_size
  float* sc_sh = w_sh + group_size * block_size;                           // block_size (scales)
  float* rot_buf = sc_sh + block_size;                                     // HeadDim (RotK only)

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  const float scale = rsqrtf(static_cast<float>(HeadDim));
  const int q_head = kv_head * group_size + warp_id;
  const int k_row_b = HeadDim * KBits / 8;
  const int v_row_b = HeadDim * VBits / 8;

  const int* block_table = block_tables + static_cast<std::size_t>(b) * max_blocks;
  const half* q_seq = q + (static_cast<std::size_t>(b) * num_heads + q_head) * HeadDim;
  constexpr int kOutPerLane = HeadDim / 32;

  for (int d = lane; d < HeadDim; d += 32) q_sh[warp_id * HeadDim + d] = q_seq[d];
  __syncthreads();
  if (RotK) {
    // Rotate each of the group's query heads to match the rotated K basis.
    for (int g = 0; g < group_size; ++g) {
      if (tid < HeadDim) rot_buf[tid] = __half2float(q_sh[g * HeadDim + tid]);
      __syncthreads();
      kvq::fwht_shared(rot_buf, tid, HeadDim);
      if (tid < HeadDim) {
        q_sh[g * HeadDim + tid] = __float2half(rot_buf[tid] * rsqrtf(static_cast<float>(HeadDim)));
      }
      __syncthreads();
    }
  }

  const int tile_tokens = min(block_size, seq_len - tok0);
  const int phys_row0 = block_table[pb] * block_size;

  // Slot-resolved fp16 sink/ring bases for this sequence's quality slot; null
  // when the quality tier is off (then every token dequantizes from the pool).
  const int kv_hidden = num_kv_heads * HeadDim;
  const half* seq_k_sink = nullptr;
  const half* seq_v_sink = nullptr;
  const half* seq_k_ring = nullptr;
  const half* seq_v_ring = nullptr;
  if (slot_ids != nullptr && (k_sink != nullptr || k_ring != nullptr)) {
    const int slot = slot_ids[b];
    const std::size_t soff = static_cast<std::size_t>(slot) * sink_n * kv_hidden;
    const std::size_t roff = static_cast<std::size_t>(slot) * win_n * kv_hidden;
    if (k_sink != nullptr) {
      seq_k_sink = k_sink + soff;
      seq_v_sink = v_sink + soff;
    }
    if (k_ring != nullptr) {
      seq_k_ring = k_ring + roff;
      seq_v_ring = v_ring + roff;
    }
  }
  // Whether any token in this tile takes the fp16 override (sink or window),
  // and whether the whole tile does (then the quant stage is skipped entirely
  // and the tile stages straight from fp16, which is cheaper than dequant;
  // at shallow depths every tile is inside the window, so this keeps the
  // quality tier from costing anything there). A tile straddling the sink /
  // window boundary takes the stage-then-overlay path.
  const bool tile_has_fp16 = (seq_k_sink != nullptr && tok0 < sink_n) ||
                             (seq_k_ring != nullptr && tok0 + tile_tokens > seq_len - win_n);
  const int fp16_from = (seq_k_ring != nullptr) ? max(0, seq_len - win_n) : 0x7fffffff;
  const bool tile_full_fp16 =
      (seq_k_sink != nullptr && tok0 + tile_tokens <= sink_n) || (tok0 >= fp16_from);

  // Phase 1: stage the K tile into shared memory. Fully fp16-covered tiles
  // copy straight from the quality-slot side buffers; otherwise dequantize
  // from the pool (scales hoisted to shared once per tile; rows load as
  // 16-byte segments) and overlay any fp16 rows afterwards.
  if (!tile_full_fp16 && tid < tile_tokens) {
    sc_sh[tid] =
        __half2float(k_scales[static_cast<std::size_t>(phys_row0 + tid) * num_kv_heads + kv_head]);
  }
  __syncthreads();
  if (tile_full_fp16) {
    constexpr int hsegs = HeadDim / 8;  // 8 halves per 16-byte segment
    for (int idx = tid; idx < tile_tokens * hsegs; idx += blockDim.x) {
      const int t = idx / hsegs;
      const int s = idx - t * hsegs;
      const half* kf = pq_fp16_source(seq_k_sink, seq_k_ring, tok0 + t, seq_len, sink_n, win_n,
                                      num_kv_heads, kv_head, HeadDim);
      reinterpret_cast<uint4*>(kv_tile + t * HeadDim)[s] = reinterpret_cast<const uint4*>(kf)[s];
    }
  } else {
    const int segs = k_row_b / 16;
    const int elems = (KBits == 4) ? 32 : 16;
    for (int idx = tid; idx < tile_tokens * segs; idx += blockDim.x) {
      const int t = idx / segs;
      const int s = idx - t * segs;
      const std::size_t sh = static_cast<std::size_t>(phys_row0 + t) * num_kv_heads + kv_head;
      const uint4 w = reinterpret_cast<const uint4*>(k_pool + sh * k_row_b)[s];
      const int8_t* b = reinterpret_cast<const int8_t*>(&w);
      const float sc = sc_sh[t];
      half* dst = kv_tile + t * HeadDim + s * elems;
      if (KBits == 4) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int bb = b[j];
          dst[2 * j] = __float2half(static_cast<float>((bb << 28) >> 28) * sc);
          dst[2 * j + 1] = __float2half(static_cast<float>(bb >> 4) * sc);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          dst[j] = __float2half(static_cast<float>(b[j]) * sc);
        }
      }
    }
    // fp16 override: overwrite sink / recent-window rows of the tile with the
    // exact (rotated) fp16 K from the quality-slot side buffers, 16B segments.
    // The barrier is required: the quant loop's segments (32/16 elements) and
    // the overlay's (8 halves) partition rows differently across threads, so
    // without it a late quant write can clobber another thread's overlay.
    if (tile_has_fp16) {
      __syncthreads();
      constexpr int hsegs = HeadDim / 8;  // 8 halves per 16-byte segment
      for (int idx = tid; idx < tile_tokens * hsegs; idx += blockDim.x) {
        const int t = idx / hsegs;
        const int s = idx - t * hsegs;
        const half* kf = pq_fp16_source(seq_k_sink, seq_k_ring, tok0 + t, seq_len, sink_n, win_n,
                                        num_kv_heads, kv_head, HeadDim);
        if (kf != nullptr) {
          reinterpret_cast<uint4*>(kv_tile + t * HeadDim)[s] =
              reinterpret_cast<const uint4*>(kf)[s];
        }
      }
    }
  }
  __syncthreads();

  float score = kPqNegInf;
  if (lane < tile_tokens && warp_id < group_size) {
    const half2* qh = reinterpret_cast<const half2*>(q_sh + warp_id * HeadDim);
    const half2* kt = reinterpret_cast<const half2*>(kv_tile + lane * HeadDim);
    float dot = 0.0f;
#pragma unroll
    for (int p = 0; p < HeadDim / 2; ++p) {
      const float2 a = __half22float2(qh[p]);
      const float2 bb = __half22float2(kt[p]);
      dot += a.x * bb.x + a.y * bb.y;
    }
    score = dot * scale;
  }
  float tile_m = score;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tile_m = fmaxf(tile_m, __shfl_xor_sync(0xffffffffu, tile_m, o));
  const float weight = (lane < tile_tokens && warp_id < group_size) ? expf(score - tile_m) : 0.0f;
  float tile_l = weight;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tile_l += __shfl_xor_sync(0xffffffffu, tile_l, o);
  if (warp_id < group_size) w_sh[warp_id * block_size + lane] = weight;
  __syncthreads();  // all warps done with K; safe to overwrite the tile with V

  // Phase 2: stage the V tile the same way, then weighted sum per head.
  if (!tile_full_fp16 && tid < tile_tokens) {
    sc_sh[tid] =
        __half2float(v_scales[static_cast<std::size_t>(phys_row0 + tid) * num_kv_heads + kv_head]);
  }
  __syncthreads();
  if (tile_full_fp16) {
    constexpr int hsegs = HeadDim / 8;
    for (int idx = tid; idx < tile_tokens * hsegs; idx += blockDim.x) {
      const int t = idx / hsegs;
      const int s = idx - t * hsegs;
      const half* vf = pq_fp16_source(seq_v_sink, seq_v_ring, tok0 + t, seq_len, sink_n, win_n,
                                      num_kv_heads, kv_head, HeadDim);
      reinterpret_cast<uint4*>(kv_tile + t * HeadDim)[s] = reinterpret_cast<const uint4*>(vf)[s];
    }
  } else {
    const int segs = v_row_b / 16;
    const int elems = (VBits == 4) ? 32 : 16;
    for (int idx = tid; idx < tile_tokens * segs; idx += blockDim.x) {
      const int t = idx / segs;
      const int s = idx - t * segs;
      const std::size_t sh = static_cast<std::size_t>(phys_row0 + t) * num_kv_heads + kv_head;
      const uint4 w = reinterpret_cast<const uint4*>(v_pool + sh * v_row_b)[s];
      const int8_t* b = reinterpret_cast<const int8_t*>(&w);
      const float sc = sc_sh[t];
      half* dst = kv_tile + t * HeadDim + s * elems;
      if (VBits == 4) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int bb = b[j];
          dst[2 * j] = __float2half(static_cast<float>((bb << 28) >> 28) * sc);
          dst[2 * j + 1] = __float2half(static_cast<float>(bb >> 4) * sc);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          dst[j] = __float2half(static_cast<float>(b[j]) * sc);
        }
      }
    }
    // fp16 override for V, same rows as the K phase (same barrier rationale).
    if (tile_has_fp16) {
      __syncthreads();
      constexpr int hsegs = HeadDim / 8;
      for (int idx = tid; idx < tile_tokens * hsegs; idx += blockDim.x) {
        const int t = idx / hsegs;
        const int s = idx - t * hsegs;
        const half* vf = pq_fp16_source(seq_v_sink, seq_v_ring, tok0 + t, seq_len, sink_n, win_n,
                                        num_kv_heads, kv_head, HeadDim);
        if (vf != nullptr) {
          reinterpret_cast<uint4*>(kv_tile + t * HeadDim)[s] =
              reinterpret_cast<const uint4*>(vf)[s];
        }
      }
    }
  }
  __syncthreads();

  if (warp_id < group_size) {
    const int chunk_index = (b * num_heads + q_head) * scratch_chunks + pb;
#pragma unroll
    for (int i = 0; i < kOutPerLane; ++i) {
      const int d = lane + i * 32;
      float o = 0.0f;
      for (int t = 0; t < tile_tokens; ++t) {
        o += w_sh[warp_id * block_size + t] * __half2float(kv_tile[t * HeadDim + d]);
      }
      chunk_o[static_cast<std::size_t>(chunk_index) * HeadDim + d] = o;
    }
    if (lane == 0) {
      chunk_m[chunk_index] = tile_m;
      chunk_l[chunk_index] = tile_l;
    }
  }
}

// Paged prefill attention over the quantized pool: same causal online-softmax
// structure as attention_prefill_kernel_tiled_paged, with dequantizing K/V
// reads through the block table and (for the R3 format) an in-place Q
// rotation to match the rotated K basis. Grid: (num_heads, num_tokens);
// block: 4 warps.
//
// fp16 quality reads: sink positions (< sink_n, rotated-K fp16 side buffer,
// slot-resolved by the caller) and this chunk's own positions
// [start_position, start_position + num_tokens) (the unrotated fp16 K/V the
// store just quantized, still in k_src/v_src) read exact fp16 instead of the
// quantized pool. The recent-window ring is NOT readable here (its residues
// are only tail-written per launch), but the chunk source covers a superset
// of the intra-chunk window. Chunk-source K is unrotated, so under RotK the
// raw Q is kept alongside the rotated copy; the rotation is orthogonal, so
// the two score paths agree in scale (dot(Rq, Rk) == dot(q, k)).
template <int WarpsPerBlock, int KBits, int VBits, bool RotK>
__global__ void attention_prefill_paged_quant_kernel(
    const half* q, const int8_t* k_pool, const int8_t* v_pool, const half* k_scales,
    const half* v_scales, const int* __restrict__ block_table, half* out, int num_tokens,
    int start_position, int num_heads, int num_kv_heads, int head_dim, int block_size,
    const half* k_src, const half* v_src, int src_stride, const half* k_sink, const half* v_sink,
    int sink_n) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  half* q_raw = q_shared + head_dim;  // unrotated copy (RotK only; else aliases q_shared)
  float* score_shared = reinterpret_cast<float*>(q_shared + (RotK ? 2 : 1) * head_dim);
  float* alpha_shared = score_shared + WarpsPerBlock;
  float* beta_shared = alpha_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;
  float* rot_buf = stats_shared + 2;  // [head_dim] (RotK only)

  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  if (token >= num_tokens) return;

  const int hidden = num_heads * head_dim;
  const int q_base = token * hidden + head * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int k_row = head_dim * KBits / 8;
  const int v_row = head_dim * VBits / 8;
  const int limit = start_position + token + 1;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[q_base + d];
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;
    stats_shared[1] = 0.0f;
  }
  __syncthreads();
  if (RotK) {
    for (int d = tid; d < head_dim; d += blockDim.x) {
      q_raw[d] = q_shared[d];
      rot_buf[d] = __half2float(q_shared[d]);
    }
    __syncthreads();
    kvq::fwht_shared(rot_buf, tid, head_dim);
    const float rn = rsqrtf(static_cast<float>(head_dim));
    for (int d = tid; d < head_dim; d += blockDim.x) {
      q_shared[d] = __float2half(rot_buf[d] * rn);
    }
    __syncthreads();
  } else {
    q_raw = q_shared;  // no rotation: one copy serves both score paths
  }

  constexpr int kOutPerThread = (256 + WarpsPerBlock * 32 - 1) / (WarpsPerBlock * 32);
  float acc[kOutPerThread];
#pragma unroll
  for (int j = 0; j < kOutPerThread; ++j) acc[j] = 0.0f;
  for (int tile_base = 0; tile_base < limit; tile_base += WarpsPerBlock) {
    const int t = tile_base + warp_id;
    float score = -1.0e30f;
    if (warp_id < WarpsPerBlock && t < limit) {
      // fp16 quality reads: sink (rotated K, dot with rotated Q) and this
      // chunk's own source rows (unrotated K, dot with raw Q).
      const half* kf = nullptr;
      const half* qf = nullptr;
      if (t < sink_n && k_sink != nullptr) {
        kf = k_sink + (t * num_kv_heads + kv_head) * head_dim;
        qf = q_shared;
      } else if (k_src != nullptr && t >= start_position) {
        kf = k_src + static_cast<std::size_t>(t - start_position) * src_stride + kv_head * head_dim;
        qf = q_raw;
      }
      float partial = 0.0f;
      if (kf != nullptr) {
        for (int i = lane; i < head_dim; i += warpSize) {
          partial += __half2float(qf[i]) * __half2float(kf[i]);
        }
      } else {
        const int phys = block_table[t / block_size] * block_size + (t % block_size);
        const std::size_t sh = static_cast<std::size_t>(phys) * num_kv_heads + kv_head;
        const float kscale = __half2float(k_scales[sh]);
        const int8_t* krow = k_pool + sh * k_row;
        if (KBits == 4) {
          for (int i = lane; i < head_dim / 2; i += warpSize) {
            const int8_t b = krow[i];
            const float k0 = static_cast<float>(((int)b << 28) >> 28) * kscale;
            const float k1 = static_cast<float>((int)b >> 4) * kscale;
            partial += __half2float(q_shared[2 * i]) * k0;
            partial += __half2float(q_shared[2 * i + 1]) * k1;
          }
        } else {
          for (int i = lane; i < head_dim; i += warpSize) {
            partial += __half2float(q_shared[i]) * static_cast<float>(krow[i]) * kscale;
          }
        }
      }
      score = pq_warp_sum(partial) * scale;
    }
    if (lane == 0 && warp_id < WarpsPerBlock) {
      score_shared[warp_id] = score;
    }
    __syncthreads();

    if (tid == 0) {
      float running_m = stats_shared[0];
      float running_l = stats_shared[1];
      const int tile_tokens = min(WarpsPerBlock, limit - tile_base);
      for (int i = 0; i < tile_tokens; ++i) {
        const float token_score = score_shared[i];
        const float new_m = fmaxf(running_m, token_score);
        const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
        const float beta = expf(token_score - new_m);
        running_l = running_l * alpha + beta;
        running_m = new_m;
        alpha_shared[i] = alpha;
        beta_shared[i] = beta;
      }
      stats_shared[0] = running_m;
      stats_shared[1] = running_l;
    }
    __syncthreads();

    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
      float acc_local = acc[j];
      const int tile_tokens = min(WarpsPerBlock, limit - tile_base);
      for (int i = 0; i < tile_tokens; ++i) {
        const int tt = tile_base + i;
        float vv;
        if (tt < sink_n && v_sink != nullptr) {
          vv = __half2float(v_sink[(tt * num_kv_heads + kv_head) * head_dim + d]);
        } else if (v_src != nullptr && tt >= start_position) {
          vv = __half2float(v_src[static_cast<std::size_t>(tt - start_position) * src_stride +
                                  kv_head * head_dim + d]);
        } else {
          const int phys = block_table[tt / block_size] * block_size + (tt % block_size);
          const std::size_t sh = static_cast<std::size_t>(phys) * num_kv_heads + kv_head;
          vv = kvq::kv_load<VBits>(v_pool + sh * v_row, d, __half2float(v_scales[sh]));
        }
        acc_local = acc_local * alpha_shared[i] + beta_shared[i] * vv;
      }
      acc[j] = acc_local;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(stats_shared[1], 1e-8f);
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
    out[q_base + d] = __float2half(acc[j] * inv_l);
  }
}

// Pass-2 reduce across a sequence's chunks (local clone of the fp16 batched
// reduce; float scratch only, format-independent). Warp-parallel like its fp16
// twin in kernels_attention_decode.cu: the serial per-chunk walk with two block
// barriers was a latency chain (~96 us at 8k context), and the online-softmax
// merge is associative, so each warp merges a strided subset barrier-free and
// warp 0 combines the warp partials.
constexpr int kPqReduceMaxWarps = 8;

__global__ void pq_chunk_reduce_batched_kernel(const float* chunk_m, const float* chunk_l,
                                               const float* chunk_o, half* out,
                                               const int* __restrict__ seq_lens, int num_heads,
                                               int head_dim, int chunk_size, int scratch_chunks) {
  const int b = blockIdx.y;
  const int head = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const int chunk_count = (seq_lens[b] + chunk_size - 1) / chunk_size;
  const std::size_t hb = (static_cast<std::size_t>(b) * num_heads + head) * scratch_chunks;

  constexpr int kLaneAcc = kTiledMaxHeadDim / 32;
  float acc[kLaneAcc];
#pragma unroll
  for (int i = 0; i < kLaneAcc; ++i) acc[i] = 0.0f;
  float m = kPqNegInf, l = 0.0f;
  for (int c = warp; c < chunk_count; c += nwarps) {
    const float cm = chunk_m[hb + c];
    const float cl = chunk_l[hb + c];
    const float new_m = fmaxf(m, cm);
    const float alpha = (l == 0.0f) ? 0.0f : expf(m - new_m);
    const float beta = (cl == 0.0f) ? 0.0f : expf(cm - new_m);
    const std::size_t base = (hb + c) * static_cast<std::size_t>(head_dim);
    int j = 0;
    for (int d = lane; d < head_dim; d += 32, ++j)
      acc[j] = acc[j] * alpha + chunk_o[base + d] * beta;
    l = l * alpha + cl * beta;
    m = new_m;
  }

  __shared__ float warp_m[kPqReduceMaxWarps], warp_l[kPqReduceMaxWarps];
  __shared__ float warp_o[kPqReduceMaxWarps][kTiledMaxHeadDim];
  if (lane == 0) {
    warp_m[warp] = m;
    warp_l[warp] = l;
  }
  int js = 0;
  for (int d = lane; d < head_dim; d += 32, ++js) warp_o[warp][d] = acc[js];
  __syncthreads();

  if (warp == 0) {
    float fm = kPqNegInf, fl = 0.0f;
    float facc[kLaneAcc];
#pragma unroll
    for (int i = 0; i < kLaneAcc; ++i) facc[i] = 0.0f;
    // An empty warp's partial is (m=-inf, l=0); the guards make it a no-op.
    for (int w = 0; w < nwarps; ++w) {
      const float cm = warp_m[w], cl = warp_l[w];
      const float new_m = fmaxf(fm, cm);
      const float alpha = (fl == 0.0f) ? 0.0f : expf(fm - new_m);
      const float beta = (cl == 0.0f) ? 0.0f : expf(cm - new_m);
      int j = 0;
      for (int d = lane; d < head_dim; d += 32, ++j)
        facc[j] = facc[j] * alpha + warp_o[w][d] * beta;
      fl = fl * alpha + cl * beta;
      fm = new_m;
    }
    const float inv_l = 1.0f / fmaxf(fl, 1e-8f);
    half* out_seq = out + (static_cast<std::size_t>(b) * num_heads + head) * head_dim;
    int j = 0;
    for (int d = lane; d < head_dim; d += 32, ++j) out_seq[d] = __float2half(facc[j] * inv_l);
  }
}

}  // namespace

void launch_store_kv_paged_quant(const half* k_src, const half* v_src, int src_stride,
                                 int8_t* k_pool, int8_t* v_pool, half* k_scales, half* v_scales,
                                 const int* block_table, int base_pos, int rows, int num_kv_heads,
                                 int head_dim, int block_size, int k_bits, int v_bits,
                                 bool rotate_k, cudaStream_t stream, half* k_sink, half* v_sink,
                                 half* k_ring, half* v_ring, int sink_n, int win_n) {
  const int num_warps = head_dim / 32;
  const std::size_t smem =
      static_cast<std::size_t>(2 * head_dim + 2 * num_warps + 2) * sizeof(float);
  const dim3 grid(rows, num_kv_heads);
  const dim3 block(head_dim);
  rotate_k = rotate_k && head_dim == 128;
#define CPI_PQ_STORE(KB, VB, RK)                                                                 \
  store_kv_paged_quant_kernel<KB, VB, RK><<<grid, block, smem, stream>>>(                        \
      k_src, v_src, src_stride, k_pool, v_pool, k_scales, v_scales, block_table, base_pos, rows, \
      num_kv_heads, head_dim, block_size, k_sink, v_sink, k_ring, v_ring, sink_n, win_n)
  if (k_bits == 4 && v_bits == 4 && rotate_k) {
    CPI_PQ_STORE(4, 4, true);
  } else if (k_bits == 4 && v_bits == 4) {
    CPI_PQ_STORE(4, 4, false);
  } else if (k_bits == 8 && v_bits == 4) {
    CPI_PQ_STORE(8, 4, false);
  } else {
    CPI_PQ_STORE(8, 8, false);
  }
#undef CPI_PQ_STORE
}

void launch_store_kv_batched_paged_quant(const half* k_src, const half* v_src, int8_t* k_pool,
                                         int8_t* v_pool, half* k_scales, half* v_scales,
                                         const int* block_tables, const int* positions,
                                         int max_blocks, int batch, int num_kv_heads, int head_dim,
                                         int block_size, int k_bits, int v_bits, bool rotate_k,
                                         cudaStream_t stream, const int* slot_ids, half* k_sink,
                                         half* v_sink, half* k_ring, half* v_ring, int sink_n,
                                         int win_n) {
  const int num_warps = head_dim / 32;
  const std::size_t smem =
      static_cast<std::size_t>(2 * head_dim + 2 * num_warps + 2) * sizeof(float);
  const dim3 grid(batch, num_kv_heads);
  const dim3 block(head_dim);
  rotate_k = rotate_k && head_dim == 128;
#define CPI_PQ_BSTORE(KB, VB, RK)                                                                  \
  store_kv_batched_paged_quant_kernel<KB, VB, RK><<<grid, block, smem, stream>>>(                  \
      k_src, v_src, k_pool, v_pool, k_scales, v_scales, block_tables, positions, max_blocks,       \
      batch, num_kv_heads, head_dim, block_size, slot_ids, k_sink, v_sink, k_ring, v_ring, sink_n, \
      win_n)
  if (k_bits == 4 && v_bits == 4 && rotate_k) {
    CPI_PQ_BSTORE(4, 4, true);
  } else if (k_bits == 4 && v_bits == 4) {
    CPI_PQ_BSTORE(4, 4, false);
  } else if (k_bits == 8 && v_bits == 4) {
    CPI_PQ_BSTORE(8, 4, false);
  } else {
    CPI_PQ_BSTORE(8, 8, false);
  }
#undef CPI_PQ_BSTORE
}

void launch_attention_prefill_paged_quant(const half* q, const int8_t* k_pool, const int8_t* v_pool,
                                          const half* k_scales, const half* v_scales,
                                          const int* block_table, half* out, int num_tokens,
                                          int start_position, int num_heads, int num_kv_heads,
                                          int head_dim, int block_size, int k_bits, int v_bits,
                                          bool rotate_k, cudaStream_t stream, const half* k_src,
                                          const half* v_src, int src_stride, const half* k_sink,
                                          const half* v_sink, int sink_n) {
  constexpr int warps = 4;
  constexpr int threads = warps * 32;
  const bool rot = rotate_k && head_dim == threads;
  std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                     static_cast<std::size_t>(3 * warps + 2) * sizeof(float);
  if (rot) {
    smem += static_cast<std::size_t>(head_dim) * sizeof(float);  // rot_buf
    smem += static_cast<std::size_t>(head_dim) * sizeof(half);   // unrotated q copy
  }
  const dim3 grid(num_heads, num_tokens);
#define CPI_PQ_PREFILL(KB, VB, RK)                                                             \
  attention_prefill_paged_quant_kernel<warps, KB, VB, RK><<<grid, threads, smem, stream>>>(    \
      q, k_pool, v_pool, k_scales, v_scales, block_table, out, num_tokens, start_position,     \
      num_heads, num_kv_heads, head_dim, block_size, k_src, v_src, src_stride, k_sink, v_sink, \
      sink_n)
  if (k_bits == 4 && v_bits == 4 && rot) {
    CPI_PQ_PREFILL(4, 4, true);
  } else if (k_bits == 4 && v_bits == 4) {
    CPI_PQ_PREFILL(4, 4, false);
  } else if (k_bits == 8 && v_bits == 4) {
    CPI_PQ_PREFILL(8, 4, false);
  } else {
    CPI_PQ_PREFILL(8, 8, false);
  }
#undef CPI_PQ_PREFILL
}

bool paged_quant_attention_supported(int num_heads, int num_kv_heads, int head_dim,
                                     int block_size) {
  const int kv_hs = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = num_heads / kv_hs;
  return head_dim == 128 && group_size > 1 && (num_heads % kv_hs) == 0 && block_size <= 32 &&
         group_size <= 32 && group_size * 32 >= head_dim;
}

void launch_attention_step_batched_paged_quant(
    const half* q, const int8_t* k_pool, const int8_t* v_pool, const half* k_scales,
    const half* v_scales, const int* block_tables, const int* seq_lens, int max_blocks,
    int max_seq_len, half* out, int batch, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int k_bits, int v_bits, bool rotate_k, cudaStream_t stream, float* scratch_m,
    float* scratch_l, float* scratch_o, int scratch_chunks, const int* slot_ids, const half* k_sink,
    const half* v_sink, const half* k_ring, const half* v_ring, int sink_n, int win_n) {
  const int group_size = num_heads / num_kv_heads;
  const int total_blocks = min(scratch_chunks, (max_seq_len + block_size - 1) / block_size);
  const int threads_g = group_size * 32;
  const bool rot = rotate_k && head_dim == 128 && threads_g >= head_dim;
  std::size_t smem =
      (static_cast<std::size_t>(group_size) * head_dim +
       static_cast<std::size_t>(block_size) * head_dim) *
          sizeof(half) +
      (static_cast<std::size_t>(group_size) * block_size + block_size) * sizeof(float);
  if (rot) smem += static_cast<std::size_t>(head_dim) * sizeof(float);
  const dim3 grid(num_kv_heads, total_blocks, batch);

#define CPI_PQ_ATTN(KB, VB, RK)                                                                  \
  attention_step_gqa_batched_paged_quant_kernel<128, KB, VB, RK>                                 \
      <<<grid, threads_g, smem, stream>>>(                                                       \
          q, k_pool, v_pool, k_scales, v_scales, block_tables, seq_lens, max_blocks, scratch_m,  \
          scratch_l, scratch_o, num_heads, num_kv_heads, group_size, block_size, scratch_chunks, \
          slot_ids, k_sink, v_sink, k_ring, v_ring, sink_n, win_n)
  if (k_bits == 4 && v_bits == 4 && rot) {
    CPI_PQ_ATTN(4, 4, true);
  } else if (k_bits == 4 && v_bits == 4) {
    CPI_PQ_ATTN(4, 4, false);
  } else if (k_bits == 8 && v_bits == 4) {
    CPI_PQ_ATTN(8, 4, false);
  } else {
    CPI_PQ_ATTN(8, 8, false);
  }
#undef CPI_PQ_ATTN

  const dim3 rgrid(num_heads, batch);
  pq_chunk_reduce_batched_kernel<<<rgrid, 128, 0, stream>>>(scratch_m, scratch_l, scratch_o, out,
                                                            seq_lens, num_heads, head_dim,
                                                            block_size, scratch_chunks);
}

}  // namespace kernels
