// kernels.cuh
//
// Public launch-function declarations for all CUDA kernels used by the
// inference runtime.  Each function in this header is a thin host-side wrapper
// that selects the appropriate kernel variant (tiled, vectorised, fallback,
// device-position, etc.) and issues a single asynchronous kernel launch on the
// supplied CUDA stream.
//
// Naming conventions:
//   launch_*            - host-side wrapper; always async on `stream`
//   *_device_pos        - variant where the sequence position is read from a
//                         device-side int pointer (required for CUDA Graph
//                         compatibility where the position cannot be a host
//                         constant at graph-capture time)
//   *_batched           - processes a batch of vectors in a single launch
//   *_dp4a              - uses SM 6.1+ dp4a int8 dot-product instruction
//   *_dual_*            - fuses two independent GEMV outputs in one kernel
//
// All fp16 tensors use the CUDA half / half2 types from <cuda_fp16.h>.
// All kernels accumulate in fp32 internally unless noted otherwise.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace kernels {

// launch_rmsnorm
//
// Applies Root Mean Square Layer Normalisation to every row of `x`:
//   y[row, d] = x[row, d] * rsqrt(mean(x[row]^2) + eps) * weight[d]
//
// Parameters:
//   x       - input activations, row-major fp16 [rows, cols]
//   weight  - per-dimension scale, fp16 [cols]
//   y       - output, row-major fp16 [rows, cols]; may alias x
//   rows    - number of rows (one CUDA block per row)
//   cols    - row width; if even, uses half2 vectorised loads
//   eps     - small constant added to the mean-square for numerical stability
//   stream  - CUDA stream for async launch
//
// Algorithm: two-pass block reduction (warp sums -> warp-0 cross-warp sum)
// with fp32 accumulation, then a single pass applying the computed scale.
void launch_rmsnorm(const half* x,
                    const half* weight,
                    half* y,
                    int rows,
                    int cols,
                    float eps,
                    cudaStream_t stream);

// launch_layernorm
//
// Applies true LayerNorm over each row:
//   y[row, d] = (x[row, d] - mean(row)) * rsqrt(var(row) + eps) * weight[d] + bias[d]
//
// bias may be null (treated as zeros).
void launch_layernorm(const half* x,
                      const half* weight,
                      const half* bias,
                      half* y,
                      int rows,
                      int cols,
                      float eps,
                      cudaStream_t stream);

// launch_embedding_lookup
//
// Gathers embedding rows for a sequence of token IDs:
//   out[i, :] = embedding[token_ids[i], :]
//
// Parameters:
//   embedding  - weight table, row-major fp16 [vocab_size, hidden]
//   token_ids  - integer token indices on device [num_tokens]
//   out        - output activations, fp16 [num_tokens, hidden]
//   num_tokens - number of tokens to look up (one CUDA block per token)
//   hidden     - embedding dimension; if divisible by 8 uses int4 vectorised
//                copies (128-bit loads), else by 2 uses half2, else scalar
//   stream     - CUDA stream for async launch
void launch_embedding_lookup(const half* embedding,
                             const int* token_ids,
                             half* out,
                             int num_tokens,
                             int hidden,
                             cudaStream_t stream);

// launch_rope_inplace
//
// Applies Rotary Position Embedding (RoPE) in-place to query and key vectors
// for a single token using on-the-fly trigonometric computation:
//   theta_i = position * rope_theta^(-2i / head_dim)
//   [q_{2i}, q_{2i+1}] = rotate([q_{2i}, q_{2i+1}], theta_i)
//
// Parameters:
//   q            - query buffer, fp16 [num_heads_q, head_dim]; modified in place
//   k            - key buffer, fp16 [num_heads_k, head_dim]; modified in place
//   num_heads_q  - number of query heads
//   num_heads_k  - number of key heads (may differ for GQA/MQA)
//   head_dim     - dimension per head; must be even
//   position     - absolute token position used to compute rotation angles
//   rope_theta   - RoPE base frequency (10000 for Llama 2, 500000 for Llama 3+)
//   stream       - CUDA stream for async launch
//
// Grid: max(num_heads_q, num_heads_k) blocks, head_dim/2 threads per block.
void launch_rope_inplace(half* q,
                         half* k,
                         int num_heads_q,
                         int num_heads_k,
                         int head_dim,
                         int position,
                         float rope_theta,
                         cudaStream_t stream);

// launch_rope_inplace_table
//
// In-place RoPE variant that reads precomputed cos/sin values from device
// tables instead of computing trigonometric functions on the fly.  Avoids
// repeated powf/cosf/sinf overhead for long contexts.
//
// Parameters:
//   q/k          - see launch_rope_inplace
//   num_heads_q/k, head_dim, stream - see launch_rope_inplace
//   position     - row index into cos_table and sin_table
//   cos_table    - fp32 table, row-major [max_position, head_dim/2]
//   sin_table    - fp32 table, row-major [max_position, head_dim/2]
void launch_rope_inplace_table(half* q,
                               half* k,
                               int num_heads_q,
                               int num_heads_k,
                               int head_dim,
                               int position,
                               const float* cos_table,
                               const float* sin_table,
                               cudaStream_t stream);

// launch_rope_inplace_device_pos
//
// In-place RoPE using a precomputed cos/sin table where the position is read
// from a device pointer rather than a host integer.  Required for CUDA Graph
// capture: the position value is fixed inside the graph and updated each step
// by an increment kernel, so the host never needs to re-capture.
//
// Parameters:
//   position  - device pointer to a single int holding the current position
//   All other parameters: see launch_rope_inplace_table
void launch_rope_inplace_device_pos(half* q,
                                    half* k,
                                    int num_heads_q,
                                    int num_heads_k,
                                    int head_dim,
                                    const int* position,
                                    const float* cos_table,
                                    const float* sin_table,
                                    cudaStream_t stream);

// launch_rope_inplace_batched
//
// In-place RoPE for a full prompt chunk where multiple tokens are processed
// in a single launch.  Each token gets the rotation for its absolute position
// (start_position + token_index).
//
// Parameters:
//   q            - fp16 [num_tokens, num_heads_q * head_dim]; modified in place
//   k            - fp16 [num_tokens, num_heads_k * head_dim]; modified in place
//   num_tokens   - number of tokens in this chunk
//   num_heads_q/k, head_dim, cos_table, sin_table - see above
//   start_position - position of the first token in the chunk
//   stream       - CUDA stream for async launch
//
// Grid: (max(num_heads_q, num_heads_k), num_tokens), head_dim/2 threads.
void launch_rope_inplace_batched(half* q,
                                 half* k,
                                 int num_tokens,
                                 int num_heads_q,
                                 int num_heads_k,
                                 int head_dim,
                                 int start_position,
                                 const float* cos_table,
                                 const float* sin_table,
                                 cudaStream_t stream);

// launch_attention_step
//
// Computes single-token causal self-attention using the full K/V cache up to
// the given sequence length.  Supports Grouped Query Attention (GQA) and
// Multi-Query Attention (MQA) via the num_kv_heads parameter.
//
// Parameters:
//   q            - current token query, fp16 [num_heads, head_dim]
//   k_cache      - key cache, fp16 [max_context, num_kv_heads, head_dim]
//   v_cache      - value cache, fp16 [max_context, num_kv_heads, head_dim]
//   out          - output, fp16 [num_heads, head_dim]
//   seq_len      - number of valid KV positions (causal limit)
//   num_heads    - number of query attention heads
//   num_kv_heads - number of KV heads (<= num_heads; 1 = MQA)
//   head_dim     - dimension per head
//   stream       - CUDA stream
//
// Optional split-K scratch buffers (all must be provided together):
//   scratch_m      - per-chunk running softmax max, fp32
//                    [num_heads * scratch_chunks]
//   scratch_l      - per-chunk running softmax denominator, fp32
//                    [num_heads * scratch_chunks]
//   scratch_o      - per-chunk partial output, fp32
//                    [num_heads * scratch_chunks * head_dim]
//   scratch_chunks - maximum number of KV chunks (determines grid.y)
//   allow_split    - if false, skip the split-K path even when buffers present
//
// Kernel selection:
//   - split-K path: head_dim==128, seq_len>=64, scratch buffers provided ->
//     chunk_stats + chunk_reduce two-pass kernels
//   - tiled path: head_dim even and <=256 -> flash-attention tile merge
//   - fallback: scalar per-token online softmax
void launch_attention_step(const half* q,
                           const half* k_cache,
                           const half* v_cache,
                           half* out,
                           int seq_len,
                           int num_heads,
                           int num_kv_heads,
                           int head_dim,
                           cudaStream_t stream,
                           float* scratch_m = nullptr,
                           float* scratch_l = nullptr,
                           float* scratch_o = nullptr,
                           int scratch_chunks = 0,
                           bool allow_split = true);

// launch_attention_step_device_pos
//
// Device-position variant of launch_attention_step.  seq_len is derived on
// device as position[0] + 1, enabling CUDA Graph-friendly decode loops where
// the host does not need to re-capture the graph each step.
//
// Parameters:
//   position     - device pointer to the current (0-based) decode position;
//                  seq_len = position[0] + 1 is computed inside the kernel
//   All other parameters: see launch_attention_step
//
// Note: when split-K scratch buffers are provided the grid is launched with
// scratch_chunks columns so all chunks run unconditionally; individual blocks
// whose chunk_start >= seq_len exit early.
void launch_attention_step_device_pos(const half* q,
                                      const half* k_cache,
                                      const half* v_cache,
                                      half* out,
                                      const int* position,
                                      int num_heads,
                                      int num_kv_heads,
                                      int head_dim,
                                      cudaStream_t stream,
                                      float* scratch_m = nullptr,
                                      float* scratch_l = nullptr,
                                      float* scratch_o = nullptr,
                                      int scratch_chunks = 0,
                                      bool allow_split = true);

// launch_store_kv_device_pos
//
// Writes the current token's K and V vectors into the layer-level KV cache
// at the row selected by the device-side decode position:
//   k_cache[position[0], :] = k[:]
//   v_cache[position[0], :] = v[:]
//
// Parameters:
//   k/v          - source vectors, fp16 [kv_hidden]
//   k_cache/v_cache - destination caches, fp16 [max_context, kv_hidden]
//   position     - device pointer to the current position index
//   kv_hidden    - flattened KV hidden size (num_kv_heads * head_dim)
//   max_context  - cache capacity; out-of-bounds positions are silently skipped
//   stream       - CUDA stream
//
// Uses 128-bit vectorised (int4) stores when all four pointers are 16-byte
// aligned and kv_hidden is divisible by 8.
void launch_store_kv_device_pos(const half* k,
                                const half* v,
                                half* k_cache,
                                half* v_cache,
                                const int* position,
                                int kv_hidden,
                                int max_context,
                                cudaStream_t stream);

// launch_copy_int / launch_increment_int
//
// Tiny scalar device-side operations used to maintain the decode position
// counter inside a CUDA Graph without host involvement.
//
// launch_copy_int:      dst[0] = src[0]   (single-thread kernel, 1x1 grid)
// launch_increment_int: value[0] += 1     (single-thread kernel, 1x1 grid)
void launch_copy_int(const int* src, int* dst, cudaStream_t stream);
void launch_increment_int(int* value, cudaStream_t stream);

// launch_attention_prefill
//
// Computes full causal self-attention for an entire prompt chunk of
// num_tokens tokens.  Each token attends to all prior cached tokens plus
// its own in-chunk prefix (causal mask enforced via the loop limit
// start_position + token + 1).
//
// Parameters:
//   q            - query matrix, fp16 [num_tokens, num_heads * head_dim]
//   k_cache      - key cache, fp16 [max_context, num_kv_heads * head_dim]
//   v_cache      - value cache, fp16 [max_context, num_kv_heads * head_dim]
//   out          - output, same layout as q
//   num_tokens   - number of tokens in this prefill chunk
//   start_position - cache position of the first token in the chunk
//   num_heads/num_kv_heads/head_dim - architecture parameters
//   stream       - CUDA stream
//
// Grid: (num_heads, num_tokens); one block per (head, token) pair.
// Kernel selection: tiled (flash-attention style) when head_dim even and
// <=256, otherwise scalar per-token online softmax fallback.
void launch_attention_prefill(const half* q,
                              const half* k_cache,
                              const half* v_cache,
                              half* out,
                              int num_tokens,
                              int start_position,
                              int num_heads,
                              int num_kv_heads,
                              int head_dim,
                              cudaStream_t stream);

// launch_add_inplace
//
// Element-wise in-place addition: x[i] += y[i] for all i in [0, n).
//
// Parameters:
//   x      - input/output fp16 vector [n]; modified in place
//   y      - addend fp16 vector [n]
//   n      - number of elements
//   stream - CUDA stream
//
// Uses half2 vectorised adds when n is even and both pointers are
// 2-byte aligned; falls back to scalar half adds otherwise.
void launch_add_inplace(half* x, const half* y, int n, cudaStream_t stream);

// launch_add_bias_broadcast
//
// Adds a bias vector to every row of a 2-D fp16 matrix in place:
//   out[row, col] += bias[col]  for all rows in [0, rows), cols in [0, cols)
//
// Used to apply QKV projection biases during chunked prefill where the output
// has shape [num_tokens, dim] and the bias has shape [dim].
//
// Parameters:
//   out    - input/output fp16 matrix [rows, cols]; modified in place
//   bias   - fp16 bias vector [cols]
//   rows   - number of rows (token count for prefill)
//   cols   - number of columns (projection output dimension)
//   stream - CUDA stream
void launch_add_bias_broadcast(half* out, const half* bias, int rows, int cols, cudaStream_t stream);

// launch_add_bias_inplace_float_from_half
//
// Adds an fp16 bias vector to an fp32 vector in place:
//   out[i] += float(bias[i])
void launch_add_bias_inplace_float_from_half(float* out,
                                             const half* bias,
                                             int n,
                                             cudaStream_t stream);

// launch_silu_mul
//
// Applies the SwiGLU activation pointwise:
//   out[i] = silu(gate[i]) * up[i]   where silu(x) = x * sigmoid(x)
//
// Parameters:
//   gate   - gate activations, fp16 [n]
//   up     - up-projection activations, fp16 [n]
//   out    - output, fp16 [n]; may not alias gate or up
//   n      - number of elements
//   stream - CUDA stream
//
// Uses half2 vectorised paths when n is even and all pointers are aligned.
void launch_silu_mul(const half* gate,
                     const half* up,
                     half* out,
                     int n,
                     cudaStream_t stream);

// launch_scale_copy
//
// Scales an fp16 vector into an fp16 destination:
//   dst[i] = fp16(src[i] * scale)
//
// Parameters:
//   dst    - output fp16 vector [n]
//   src    - input fp16 vector [n]
//   n      - number of elements
//   scale  - fp32 scalar multiplier
//   stream - CUDA stream
void launch_scale_copy(half* dst,
                       const half* src,
                       int n,
                       float scale,
                       cudaStream_t stream);

// launch_scale_add_inplace
//
// In-place scaled accumulation:
//   dst[i] += fp16(src[i] * scale)
//
// Parameters:
//   dst    - input/output fp16 vector [n]
//   src    - input fp16 vector [n]
//   n      - number of elements
//   scale  - fp32 scalar multiplier
//   stream - CUDA stream
void launch_scale_add_inplace(half* dst,
                              const half* src,
                              int n,
                              float scale,
                              cudaStream_t stream);

// launch_moe_router_topk_softmax
//
// Computes softmax probabilities from router logits and selects top-k experts.
// Output probabilities are renormalized over selected experts.
//
// Parameters:
//   logits      - router logits, fp16 [experts]
//   experts     - number of experts
//   top_k       - experts selected per token (typically 2)
//   topk_idx    - selected expert indices [top_k]
//   topk_prob   - selected normalized gate probabilities [top_k]
//   stream      - CUDA stream
void launch_moe_router_topk_softmax(const half* logits,
                                    int experts,
                                    int top_k,
                                    int* topk_idx,
                                    float* topk_prob,
                                    cudaStream_t stream);

// launch_dequant_int8_to_fp16
//
// Dequantises an int8 tensor to fp16 using a single global scale:
//   dst[i] = fp16(src[i] * scale)
//
// Parameters:
//   src    - quantised int8 tensor [n]
//   dst    - output fp16 tensor [n]
//   n      - number of elements
//   scale  - scalar dequantisation factor (host float)
//   stream - CUDA stream
void launch_dequant_int8_to_fp16(const std::int8_t* src,
                                 half* dst,
                                 int n,
                                 float scale,
                                 cudaStream_t stream);

// launch_dequant_rowwise_int8_to_fp16
//
// Dequantises a row-major int8 matrix to fp16 with one scale per row:
//   dst[row, col] = fp16(src[row, col] * scales[row])
//
// Parameters:
//   src    - quantised int8 matrix [rows, cols]
//   scales - per-row fp32 dequantisation scales [rows]
//   dst    - output fp16 matrix [rows, cols]
//   rows   - number of rows
//   cols   - number of columns per row
//   stream - CUDA stream
void launch_dequant_rowwise_int8_to_fp16(const std::int8_t* src,
                                         const float* scales,
                                         half* dst,
                                         int rows,
                                         int cols,
                                         cudaStream_t stream);

// launch_quantize_rowwise_fp16_to_int8
//
// Quantises fp16 activations to int8 with one scale per row. The quant range
// is controlled by max_q:
//   scales[row] = max(abs(src[row, :])) / max_q
//   dst[row, col] = clamp(round(src[row, col] / scales[row]), -max_q, max_q)
// Typical values are max_q=127 (INT8) and max_q=7 (INT4 pre-pack path).
//
// Parameters:
//   src    - input fp16 activations [rows, cols]
//   dst    - output int8 activations [rows, cols]
//   scales - output per-row dequantisation scales [rows]; written by kernel
//   rows   - number of rows (one CUDA block per row)
//   cols   - columns per row; if even uses half2 vectorised max reduction
//   stream - CUDA stream
//   max_q  - positive symmetric quant bound (default 127)
void launch_quantize_rowwise_fp16_to_int8(const half* src,
                                          std::int8_t* dst,
                                          float* scales,
                                          int rows,
                                          int cols,
                                          cudaStream_t stream,
                                          int max_q = 127);

// launch_pack_rowwise_int8_to_int4
//
// Packs row-major signed int8 values into signed int4 (two values per byte).
// Input is expected to be within the int4 range [-8, 7]; values are clamped.
void launch_pack_rowwise_int8_to_int4(const std::int8_t* src,
                                      std::int8_t* dst,
                                      int rows,
                                      int cols,
                                      cudaStream_t stream);

// launch_weight_only_int8_matvec
//
// Weight-only int8 matrix-vector multiply with per-row scales and fp16 input:
//   y[row] = fp16(dot(w[row, :], fp32(x[:])) * scales[row])
//
// Parameters:
//   w            - weight matrix, row-major int8 [out_features, in_features]
//   scales       - per-row fp32 dequantisation scales [out_features]
//   x            - input activation vector, fp16 [in_features]
//   y            - output vector, fp16 [out_features]
//   out_features - number of output rows (one block per row)
//   in_features  - inner dimension; reduction done with shared-memory tree
//   stream       - CUDA stream
void launch_weight_only_int8_matvec(const std::int8_t* w,
                                    const float* scales,
                                    const half* x,
                                    half* y,
                                    int out_features,
                                    int in_features,
                                    cudaStream_t stream);

// launch_weight_only_int8_matvec_batched
//
// Batched weight-only int8 GEMV: same weight matrix shared by all batch rows.
//   y[b, row] = fp16(dot(w[row, :], fp32(x[b, :])) * scales[row])
//
// Parameters:
//   w            - weight matrix, row-major int8 [out_features, in_features]
//   scales       - per-row fp32 scales [out_features]
//   x            - input batch, fp16 [batch_size, in_features]
//   y            - output batch, fp16 [batch_size, out_features]
//   batch_size   - number of independent input vectors
//   out_features - number of output rows
//   in_features  - inner dimension
//   stream       - CUDA stream
//
// Grid: (out_features, batch_size).
void launch_weight_only_int8_matvec_batched(const std::int8_t* w,
                                            const float* scales,
                                            const half* x,
                                            half* y,
                                            int batch_size,
                                            int out_features,
                                            int in_features,
                                            cudaStream_t stream);

// launch_weight_only_int8_matvec_batched_dp4a
//
// Batched int8 x int8 GEMV using SM 6.1+ dp4a packed 4-element dot products.
// Both the weight matrix and input activations are quantised to int8.  The
// final result is rescaled by the product of per-row weight scale and per-batch
// activation scale:
//   y[b, row] = fp16(idot(w[row,:], x[b,:]) * w_scales[row] * x_scales[b])
//
// Parameters:
//   w/x            - int8 weight [out_features, in_features] and
//                    activation [batch_size, in_features] matrices
//   w_scales       - per-row fp32 weight dequantisation scales [out_features]
//   x_scales       - per-batch fp32 activation scales [batch_size]
//   y              - output fp16 [batch_size, out_features]
//   batch_size/out_features/in_features - tensor dimensions
//   stream         - CUDA stream
//
// Uses int4 (128-bit) loads when in_features is divisible by 16, then int
// (32-bit / 4 elements) loads for the remainder, then scalar for any tail.
void launch_weight_only_int8_matvec_batched_dp4a(const std::int8_t* w,
                                                 const float* w_scales,
                                                 const std::int8_t* x,
                                                 const float* x_scales,
                                                 half* y,
                                                 int batch_size,
                                                 int out_features,
                                                 int in_features,
                                                 cudaStream_t stream);

// launch_weight_only_int8_matvec_dp4a
//
// Single-row (batch=1) int8 x int8 GEMV using dp4a.  The activation scale is
// a single device float (*x_scale) rather than a per-batch array.
//
// Template-dispatch tuning parameters (0 = use runtime defaults):
//   warps_per_block - total warps per CUDA block (4, 8, or 16)
//   tile_packed4    - number of packed-4 input elements staged in shared
//                     memory per tile (128, 256, or 512)
//   warps_per_row   - warps that cooperate on a single output row (1, 2, or 4)
//                     enabling intra-row split-K; must divide warps_per_block
//
// Parameters:
//   w/w_scales   - int8 weight matrix [out_features, in_features] and scales
//   x/x_scale    - int8 input [in_features] and single device scale pointer
//   y            - fp16 output [out_features]
//   stream       - CUDA stream
void launch_weight_only_int8_matvec_dp4a(const std::int8_t* w,
                                         const float* w_scales,
                                         const std::int8_t* x,
                                         const float* x_scale,
                                         half* y,
                                         int out_features,
                                         int in_features,
                                         cudaStream_t stream,
                                         int warps_per_block = 0,
                                         int tile_packed4 = 0,
                                         int warps_per_row = 1);

// launch_weight_only_int8_matvec_dual_dp4a
//
// Fused dual-output single-row int8 x int8 GEMV.  Computes two independent
// GEMV operations (w_a * x and w_b * x) sharing the same input activation
// vector and scale in a single kernel launch.  The shared x tile is staged in
// shared memory once and reused for both weight matrices, halving global load
// traffic for x compared to two separate launches.
//
// Parameters:
//   w_a/w_scales_a - first weight matrix int8 [out_features, in_features]
//                    and per-row fp32 scales [out_features]
//   w_b/w_scales_b - second weight matrix and scales (same shapes as w_a)
//   x/x_scale      - shared input vector int8 [in_features] and device scale
//   y_a/y_b        - separate fp16 output vectors [out_features]
//   out_features/in_features - weight matrix dimensions
//   stream         - CUDA stream
//   warps_per_block/tile_packed4/warps_per_row - see launch_weight_only_int8_matvec_dp4a
void launch_weight_only_int8_matvec_dual_dp4a(const std::int8_t* w_a,
                                              const float* w_scales_a,
                                              const std::int8_t* w_b,
                                              const float* w_scales_b,
                                              const std::int8_t* x,
                                              const float* x_scale,
                                              half* y_a,
                                              half* y_b,
                                              int out_features,
                                              int in_features,
                                              cudaStream_t stream,
                                              int warps_per_block = 0,
                                              int tile_packed4 = 0,
                                              int warps_per_row = 1);

// launch_weight_only_int4_matvec
//
// Weight-only int4 matrix-vector multiply with per-row scales and fp16 input.
// Weights are packed row-major with two signed int4 values per byte
// (low nibble first), matching .int4 tensor layout.
void launch_weight_only_int4_matvec(const std::int8_t* w_packed,
                                    const float* scales,
                                    const half* x,
                                    half* y,
                                    int out_features,
                                    int in_features,
                                    cudaStream_t stream);

// launch_weight_only_int4_matvec_batched
//
// Batched variant of launch_weight_only_int4_matvec using fp16 activations.
void launch_weight_only_int4_matvec_batched(const std::int8_t* w_packed,
                                            const float* scales,
                                            const half* x,
                                            half* y,
                                            int batch_size,
                                            int out_features,
                                            int in_features,
                                            cudaStream_t stream);

// launch_weight_only_int4_matvec_batched_dp4a
//
// Batched int4(weight) x int8(activation) GEMV using dp4a. The input
// activations are int8 with one scale per batch row.
void launch_weight_only_int4_matvec_batched_dp4a(const std::int8_t* w_packed,
                                                 const float* w_scales,
                                                 const std::int8_t* x,
                                                 const float* x_scales,
                                                 half* y,
                                                 int batch_size,
                                                 int out_features,
                                                 int in_features,
                                                 cudaStream_t stream);

// launch_weight_only_int4_matvec_dp4a
//
// Single-row (batch=1) int4(weight) x int8(activation) GEMV using dp4a.
void launch_weight_only_int4_matvec_dp4a(const std::int8_t* w_packed,
                                         const float* w_scales,
                                         const std::int8_t* x,
                                         const float* x_scale,
                                         half* y,
                                         int out_features,
                                         int in_features,
                                         cudaStream_t stream,
                                         int warps_per_block = 0,
                                         int tile_packed4 = 0,
                                         int warps_per_row = 1);

// launch_weight_only_int4_matvec_dual_dp4a
//
// Dual-output dp4a GEMV for two packed-int4 weight matrices sharing the same
// int8 input activation vector and scale.
void launch_weight_only_int4_matvec_dual_dp4a(const std::int8_t* w_a_packed,
                                              const float* w_scales_a,
                                              const std::int8_t* w_b_packed,
                                              const float* w_scales_b,
                                              const std::int8_t* x,
                                              const float* x_scale,
                                              half* y_a,
                                              half* y_b,
                                              int out_features,
                                              int in_features,
                                              cudaStream_t stream,
                                              int warps_per_block = 0,
                                              int tile_packed4 = 0,
                                              int warps_per_row = 1);

// launch_rowmajor_half_gemv_f16
//
// Batch-1 row-major fp16 GEMV with fp16 output, optimised for resident
// decode projection layers (Q, K, V, O, gate, up, down):
//   y[row] = fp16(dot(w[row, :], fp32(x[:])))
//
// The kernel stages x tiles in shared memory and uses warp-level half2
// vectorised dot products with fp32 accumulation.  Multiple rows can be
// assigned to a single warp (rows_per_warp) to improve arithmetic intensity
// when out_features is large.
//
// Parameters:
//   w            - row-major fp16 weight matrix [out_features, in_features]
//   x            - fp16 input vector [in_features]
//   y            - fp16 output vector [out_features]
//   out_features - number of output elements
//   in_features  - inner dimension
//   stream       - CUDA stream
//
// Tuning parameters (0 = use runtime defaults):
//   warps_per_block - warps per block (4, 8, or 16; auto-selected by
//                     out_features >= 8192 heuristic when 0)
//   tile_pairs      - half2 elements staged per shared-memory tile (128 or 256)
//   rows_per_warp   - output rows assigned to each warp (1 or 2)
void launch_rowmajor_half_gemv_f16(const half* w,
                                   const half* x,
                                   half* y,
                                   int out_features,
                                   int in_features,
                                   cudaStream_t stream,
                                   int warps_per_block = 0,
                                   int tile_pairs = 0,
                                   int rows_per_warp = 1);

// launch_rowmajor_half_gemv_f32
//
// Batch-1 row-major fp16 GEMV with fp32 output, used for the LM head where
// the logit vector must remain in full precision before argmax / sampling:
//   y[row] = float(dot(w[row, :], fp32(x[:])))
//
// All parameters are identical to launch_rowmajor_half_gemv_f16 except:
//   y - fp32 output vector [out_features]
//
// The same templated kernel (rowmajor_half_gemv_kernel) is instantiated with
// OutT=float; the only difference is the final store path.
void launch_rowmajor_half_gemv_f32(const half* w,
                                   const half* x,
                                   float* y,
                                   int out_features,
                                   int in_features,
                                   cudaStream_t stream,
                                   int warps_per_block = 0,
                                   int tile_pairs = 0,
                                   int rows_per_warp = 1);

// launch_argmax_float
//
// Finds the index of the maximum element in a float vector:
//   *out_index = argmax(logits[0..n-1])
//
// Parameters:
//   logits     - fp32 input vector [n] on device
//   n          - number of elements
//   out_index  - device pointer; receives the 0-based index of the maximum
//   stream     - CUDA stream
//
// Uses a two-level warp-argmax reduction (per-warp then across warps in
// warp 0) within a single block of 256 threads.
void launch_argmax_float(const float* logits,
                         int n,
                         int* out_index,
                         cudaStream_t stream);

// launch_convert_bf16_to_fp16
//
// Converts raw BF16 bit patterns to fp16:
//   dst[i] = fp16(bfloat16_to_float(src[i]))
//
// Parameters:
//   src    - input BF16 values stored as uint16 bit patterns [n]
//   dst    - output fp16 tensor [n]
//   n      - number of elements
//   stream - CUDA stream
void launch_convert_bf16_to_fp16(const std::uint16_t* src,
                                 half* dst,
                                 int n,
                                 cudaStream_t stream);

// launch_store_kv_int4
//
// Quantizes the current-token K and V vectors to symmetric per-head INT4 and
// writes them into the layer KV cache at sequence position `position`.
// Two signed 4-bit values are packed per byte, low nibble first:
//   packed[i] = (q[2i] & 0xF) | ((q[2i+1] & 0xF) << 4)
//   scale      = max(|group|) / 7.0  (range [-8, 7], per head)
//
// Parameters:
//   k / v          - fp16 K and V vectors [num_kv_heads * head_dim]
//   k_cache_i4     - INT4 K cache (layer slice) [max_context, num_kv_heads, head_dim/2]
//   v_cache_i4     - INT4 V cache (layer slice)
//   k_scales       - fp16 K scale table (layer slice) [max_context, num_kv_heads]
//   v_scales       - fp16 V scale table (layer slice)
//   position       - current sequence position (host integer)
//   num_kv_heads   - number of KV heads
//   head_dim       - dimension per head; must be a multiple of 32
//   max_context    - cache capacity
//   stream         - CUDA stream
void launch_store_kv_int4(const half*  k,
                           const half*  v,
                           int8_t*      k_cache_i4,
                           int8_t*      v_cache_i4,
                           half*        k_scales,
                           half*        v_scales,
                           int          position,
                           int          num_kv_heads,
                           int          head_dim,
                           int          max_context,
                           cudaStream_t stream);

// launch_attention_step_int4
//
// Decode-step attention reading INT4-compressed KV cache.  Reads 4x less
// VRAM bandwidth than the fp16 equivalent by unpacking nibbles and
// dequantizing on the fly using the per-head scales written by
// launch_store_kv_int4.  Functionally equivalent to launch_attention_step.
//
// When scratch_m/l/o are non-null and allow_split is true the kernel uses the
// same split-K two-pass approach as launch_attention_step (chunk_stats then
// chunk_reduce), enabling full SM utilization at long context.
//
// Parameters:
//   q              - query fp16 [num_heads * head_dim]
//   k_cache_i4     - INT4 K cache (layer slice) [max_context, num_kv_heads, head_dim/2]
//   v_cache_i4     - INT4 V cache (layer slice)
//   k_scales       - fp16 K scales (layer slice) [max_context, num_kv_heads]
//   v_scales       - fp16 V scales
//   out            - output fp16 [num_heads * head_dim]
//   seq_len        - number of valid KV positions (causal limit)
//   num_heads / num_kv_heads / head_dim - architecture parameters
//   stream         - CUDA stream
//   scratch_m/l/o  - split-K scratch buffers (same layout as launch_attention_step)
//   scratch_chunks - capacity of scratch buffers in chunks
//   allow_split    - enable split-K path (requires head_dim==128, seq_len>=64, scratch)
void launch_attention_step_int4(const half*   q,
                                 const int8_t* k_cache_i4,
                                 const int8_t* v_cache_i4,
                                 const half*   k_scales,
                                 const half*   v_scales,
                                 half*         out,
                                 int           seq_len,
                                 int           num_heads,
                                 int           num_kv_heads,
                                 int           head_dim,
                                 cudaStream_t  stream,
                                 float*        scratch_m    = nullptr,
                                 float*        scratch_l    = nullptr,
                                 float*        scratch_o    = nullptr,
                                 int           scratch_chunks = 0,
                                 bool          allow_split  = false);

// ── TurboQuant 3-bit (TQ3) kernels ───────────────────────────────────────────

// launch_hadamard_rotate_fp16
//
// Applies an in-place block-diagonal randomised Walsh-Hadamard Transform to an
// fp16 vector.  The vector is split into (n / block_size) sub-blocks; each sub-block
// b gets the transform D_b * H_{block_size} / sqrt(block_size) where D_b is the
// corresponding slice of the ±1 diagonal `signs`.
//
// For power-of-2 hidden sizes, set block_size = n (single block, same as before).
// For non-power-of-2 hidden sizes, set block_size = n & -n (largest pow-2 factor).
//
// Parameters:
//   x          - fp16 vector [n]; modified in place
//   signs      - int8 ±1 diagonal values [n]; must match those used at conversion
//   n          - total vector length; must be a multiple of block_size
//   block_size - WHT sub-block size; must be a power of 2 and <= 4096
//   stream     - CUDA stream
//
// Launches (n / block_size) CUDA blocks with 512 threads each.
void launch_hadamard_rotate_fp16(half* x, const int8_t* signs, int n, int block_size, cudaStream_t stream);

// launch_tq3_gemv_f16
//
// Weight-only 3-bit TQ3 matrix-vector product:
//   y[row] = fp16( dot(dequant(w_packed[row]), x[:]) * scales[row] )
//
// Weights are stored as 10 packed 3-bit indices per uint32 word.  Each index
// selects one of 8 reconstruction values from the shared codebook.  x must
// already be rotated by the Hadamard transform (see launch_hadamard_rotate_fp16).
//
// Parameters:
//   w_packed     - packed weight matrix [out_features, words_per_row] uint32,
//                  where words_per_row = ceil(in_features / 10)
//   codebook     - 8-entry FP16 reconstruction table [8]; shared across all rows
//   scales       - per-row FP16 dequantisation scale [out_features]
//   x            - rotated fp16 input vector [in_features]
//   y            - fp16 output vector [out_features]
//   out_features - number of output rows
//   in_features  - inner dimension; currently only power-of-2 ≤ 4096 is
//                  supported (non-power-of-2 falls back to fp16 in the engine)
//   stream       - CUDA stream
//
// Grid: ceil(out_features/8) blocks, 256 threads (8 warps, one per row).
// Shared memory: in_features*2 + 32 bytes.
void launch_tq3_gemv_f16(const uint32_t* w_packed,
                          const half*     codebook,
                          const half*     scales,
                          const half*     x,
                          half*           y,
                          int             out_features,
                          int             in_features,
                          cudaStream_t    stream);

// Builds packed sign bits for projected coordinates of x:
//   bit[j] = sign(signs[j] * x[indices[j]]) >= 0
// Output is packed little-endian in uint32 words.
void launch_tq_qjl_pack_sign_bits(const half*     x,
                                  const int32_t*  indices,
                                  const int8_t*   signs,
                                  uint32_t*       out_bits,
                                  int             qjl_dim,
                                  cudaStream_t    stream);

// Adds residual 1-bit correction (Qprod stage-B style) to y:
//   y[row] += scales[row] * corr(sign(row_bits), sign(x_bits))
// where corr is normalized sign-agreement in [-1, 1].
void launch_tq_qjl_residual_add_f16(const uint32_t* row_bits,
                                    const half*     scales,
                                    const uint32_t* x_bits,
                                    half*           y,
                                    int             out_features,
                                    int             qjl_dim,
                                    cudaStream_t    stream);

}  // namespace kernels
