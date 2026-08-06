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
// chunk coarsening (blocks_per_chunk > 1) and the fp16 sink/window override
// for multi-sequence batches are follow-ups.

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

// Scatter + quantize `rows` contiguous KV rows into the paged pool at
// positions base_pos..base_pos+rows-1. Grid: (rows, num_kv_heads);
// block: head_dim threads. src_stride is in halves (kv_hidden for the plain
// layout, q_hidden + 2*kv_hidden when reading K/V in place from fused QKV).
template <int KBits, int VBits, bool RotK>
__global__ void store_kv_paged_quant_kernel(const half* k_src, const half* v_src, int src_stride,
                                            int8_t* k_pool, int8_t* v_pool, half* k_scales,
                                            half* v_scales, const int* __restrict__ block_table,
                                            int base_pos, int rows, int num_kv_heads, int head_dim,
                                            int block_size) {
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
    int max_blocks, int batch, int num_kv_heads, int head_dim, int block_size) {
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
    int max_blocks, float* chunk_m, float* chunk_l, float* chunk_o, int num_heads,
    int num_kv_heads, int group_size, int block_size, int scratch_chunks) {
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
  float* rot_buf = w_sh + group_size * block_size;                         // HeadDim (RotK only)

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

  // Phase 1: dequantize the K tile into shared memory.
  for (int i = tid; i < tile_tokens * HeadDim; i += blockDim.x) {
    const int t = i / HeadDim;
    const int d = i - t * HeadDim;
    const std::size_t sh = static_cast<std::size_t>(phys_row0 + t) * num_kv_heads + kv_head;
    kv_tile[t * HeadDim + d] =
        __float2half(kvq::kv_load<KBits>(k_pool + sh * k_row_b, d, __half2float(k_scales[sh])));
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
  const float weight =
      (lane < tile_tokens && warp_id < group_size) ? expf(score - tile_m) : 0.0f;
  float tile_l = weight;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tile_l += __shfl_xor_sync(0xffffffffu, tile_l, o);
  if (warp_id < group_size) w_sh[warp_id * block_size + lane] = weight;
  __syncthreads();  // all warps done with K; safe to overwrite the tile with V

  // Phase 2: dequantize the V tile, then weighted sum per query head.
  for (int i = tid; i < tile_tokens * HeadDim; i += blockDim.x) {
    const int t = i / HeadDim;
    const int d = i - t * HeadDim;
    const std::size_t sh = static_cast<std::size_t>(phys_row0 + t) * num_kv_heads + kv_head;
    kv_tile[t * HeadDim + d] =
        __float2half(kvq::kv_load<VBits>(v_pool + sh * v_row_b, d, __half2float(v_scales[sh])));
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
template <int WarpsPerBlock, int KBits, int VBits, bool RotK>
__global__ void attention_prefill_paged_quant_kernel(
    const half* q, const int8_t* k_pool, const int8_t* v_pool, const half* k_scales,
    const half* v_scales, const int* __restrict__ block_table, half* out, int num_tokens,
    int start_position, int num_heads, int num_kv_heads, int head_dim, int block_size) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
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
    for (int d = tid; d < head_dim; d += blockDim.x) rot_buf[d] = __half2float(q_shared[d]);
    __syncthreads();
    kvq::fwht_shared(rot_buf, tid, head_dim);
    const float rn = rsqrtf(static_cast<float>(head_dim));
    for (int d = tid; d < head_dim; d += blockDim.x) {
      q_shared[d] = __float2half(rot_buf[d] * rn);
    }
    __syncthreads();
  }

  constexpr int kOutPerThread = (256 + WarpsPerBlock * 32 - 1) / (WarpsPerBlock * 32);
  float acc[kOutPerThread];
#pragma unroll
  for (int j = 0; j < kOutPerThread; ++j) acc[j] = 0.0f;
  for (int tile_base = 0; tile_base < limit; tile_base += WarpsPerBlock) {
    const int t = tile_base + warp_id;
    float score = -1.0e30f;
    if (warp_id < WarpsPerBlock && t < limit) {
      const int phys = block_table[t / block_size] * block_size + (t % block_size);
      const std::size_t sh = static_cast<std::size_t>(phys) * num_kv_heads + kv_head;
      const float kscale = __half2float(k_scales[sh]);
      const int8_t* krow = k_pool + sh * k_row;
      float partial = 0.0f;
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
        const int phys = block_table[tt / block_size] * block_size + (tt % block_size);
        const std::size_t sh = static_cast<std::size_t>(phys) * num_kv_heads + kv_head;
        const float vv =
            kvq::kv_load<VBits>(v_pool + sh * v_row, d, __half2float(v_scales[sh]));
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
// reduce; float scratch only, format-independent).
__global__ void pq_chunk_reduce_batched_kernel(const float* chunk_m, const float* chunk_l,
                                               const float* chunk_o, half* out,
                                               const int* __restrict__ seq_lens, int num_heads,
                                               int head_dim, int chunk_size, int scratch_chunks) {
  __shared__ float scale_shared[3];
  const int b = blockIdx.y;
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int chunk_count = (seq_lens[b] + chunk_size - 1) / chunk_size;
  float acc[kAccPerThread];
#pragma unroll
  for (int i = 0; i < kAccPerThread; ++i) acc[i] = 0.0f;
  float running_m = kPqNegInf, running_l = 0.0f;
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    if (tid == 0) {
      const int idx = (b * num_heads + head) * scratch_chunks + chunk;
      const float cm = chunk_m[idx], cl = chunk_l[idx];
      const float new_m = fmaxf(running_m, cm);
      const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float beta = (cl == 0.0f) ? 0.0f : expf(cm - new_m);
      running_l = running_l * alpha + cl * beta;
      running_m = new_m;
      scale_shared[0] = alpha;
      scale_shared[1] = beta;
      scale_shared[2] = running_l;
    }
    __syncthreads();
    const float alpha = scale_shared[0], beta = scale_shared[1];
    const std::size_t base =
        (static_cast<std::size_t>(b * num_heads + head) * scratch_chunks + chunk) *
        static_cast<std::size_t>(head_dim);
    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
      acc[j] = acc[j] * alpha + chunk_o[base + d] * beta;
    }
    __syncthreads();
  }
  const float inv_l = 1.0f / fmaxf(scale_shared[2], 1e-8f);
  half* out_seq = out + (static_cast<std::size_t>(b) * num_heads + head) * head_dim;
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
    out_seq[d] = __float2half(acc[j] * inv_l);
  }
}

}  // namespace

void launch_store_kv_paged_quant(const half* k_src, const half* v_src, int src_stride,
                                 int8_t* k_pool, int8_t* v_pool, half* k_scales, half* v_scales,
                                 const int* block_table, int base_pos, int rows, int num_kv_heads,
                                 int head_dim, int block_size, int k_bits, int v_bits,
                                 bool rotate_k, cudaStream_t stream) {
  const int num_warps = head_dim / 32;
  const std::size_t smem =
      static_cast<std::size_t>(2 * head_dim + 2 * num_warps + 2) * sizeof(float);
  const dim3 grid(rows, num_kv_heads);
  const dim3 block(head_dim);
  rotate_k = rotate_k && head_dim == 128;
#define CPI_PQ_STORE(KB, VB, RK)                                                              \
  store_kv_paged_quant_kernel<KB, VB, RK><<<grid, block, smem, stream>>>(                     \
      k_src, v_src, src_stride, k_pool, v_pool, k_scales, v_scales, block_table, base_pos,    \
      rows, num_kv_heads, head_dim, block_size)
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
                                         int max_blocks, int batch, int num_kv_heads,
                                         int head_dim, int block_size, int k_bits, int v_bits,
                                         bool rotate_k, cudaStream_t stream) {
  const int num_warps = head_dim / 32;
  const std::size_t smem =
      static_cast<std::size_t>(2 * head_dim + 2 * num_warps + 2) * sizeof(float);
  const dim3 grid(batch, num_kv_heads);
  const dim3 block(head_dim);
  rotate_k = rotate_k && head_dim == 128;
#define CPI_PQ_BSTORE(KB, VB, RK)                                                             \
  store_kv_batched_paged_quant_kernel<KB, VB, RK><<<grid, block, smem, stream>>>(             \
      k_src, v_src, k_pool, v_pool, k_scales, v_scales, block_tables, positions, max_blocks,  \
      batch, num_kv_heads, head_dim, block_size)
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

void launch_attention_prefill_paged_quant(const half* q, const int8_t* k_pool,
                                          const int8_t* v_pool, const half* k_scales,
                                          const half* v_scales, const int* block_table,
                                          half* out, int num_tokens, int start_position,
                                          int num_heads, int num_kv_heads, int head_dim,
                                          int block_size, int k_bits, int v_bits, bool rotate_k,
                                          cudaStream_t stream) {
  constexpr int warps = 4;
  constexpr int threads = warps * 32;
  const bool rot = rotate_k && head_dim == threads;
  std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                     static_cast<std::size_t>(3 * warps + 2) * sizeof(float);
  if (rot) smem += static_cast<std::size_t>(head_dim) * sizeof(float);
  const dim3 grid(num_heads, num_tokens);
#define CPI_PQ_PREFILL(KB, VB, RK)                                                            \
  attention_prefill_paged_quant_kernel<warps, KB, VB, RK><<<grid, threads, smem, stream>>>(   \
      q, k_pool, v_pool, k_scales, v_scales, block_table, out, num_tokens, start_position,    \
      num_heads, num_kv_heads, head_dim, block_size)
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
    float* scratch_l, float* scratch_o, int scratch_chunks) {
  const int group_size = num_heads / num_kv_heads;
  const int total_blocks = min(scratch_chunks, (max_seq_len + block_size - 1) / block_size);
  const int threads_g = group_size * 32;
  const bool rot = rotate_k && head_dim == 128 && threads_g >= head_dim;
  std::size_t smem = (static_cast<std::size_t>(group_size) * head_dim +
                      static_cast<std::size_t>(block_size) * head_dim) *
                         sizeof(half) +
                     static_cast<std::size_t>(group_size) * block_size * sizeof(float);
  if (rot) smem += static_cast<std::size_t>(head_dim) * sizeof(float);
  const dim3 grid(num_kv_heads, total_blocks, batch);

#define CPI_PQ_ATTN(KB, VB, RK)                                                                \
  attention_step_gqa_batched_paged_quant_kernel<128, KB, VB, RK>                               \
      <<<grid, threads_g, smem, stream>>>(q, k_pool, v_pool, k_scales, v_scales, block_tables, \
                                          seq_lens, max_blocks, scratch_m, scratch_l,          \
                                          scratch_o, num_heads, num_kv_heads, group_size,      \
                                          block_size, scratch_chunks)
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
