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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace kernels {

// Decode-attention accumulator sizing (shared by the fp16 and int4 kernel TUs).
// The tiled/split-K decode kernels give each thread a fixed-size per-head_dim
// accumulator array of kAccPerThread floats; a block runs blockDim (<=
// WarpsPerBlock*32) threads, so it covers head_dim up to blockDim*kAccPerThread.
// Each templated kernel static_asserts that bound against kTiledMaxHeadDim, so
// raising the head_dim cap without growing the accumulator is a build error, not
// silent corruption (the head_dim=256 bug this guards against scored cosine
// ~0.65 vs an independent reference).
// 512 accommodates Gemma 4's full-attention layers (global_head_dim=512); the
// warps=4 tiled path runs 128 threads × kAccPerThread=8 = 1024 >= 512, so the
// per-thread accumulator still covers it (static_assert enforces this).
constexpr int kTiledMaxHeadDim = 512;
constexpr int kAccPerThread = 8;

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
void launch_rmsnorm(const half* x, const half* weight, half* y, int rows, int cols, float eps,
                    cudaStream_t stream);

// Batched RMSNorm over T rows with separate in/out row strides (DeepSeek kv_a_layernorm over the
// [kv_lora] prefix of MlaCkv). One launch for all T tokens. Requires cols % 8 == 0.
void launch_rmsnorm_seq_strided(const half* x, const half* weight, half* y, int cols, int in_stride,
                                int out_stride, int T, float eps, cudaStream_t stream);

// launch_rmsnorm_offset
//
// Qwen3.5-style RMSNorm where the stored weight is an additive offset from 1:
//   y[row, d] = x[row, d] * rsqrt(mean(x[row]^2) + eps) * (1 + weight[d])
//
// Parameters match launch_rmsnorm.
void launch_rmsnorm_offset(const half* x, const half* weight, half* y, int rows, int cols,
                           float eps, cudaStream_t stream);

// launch_layernorm
//
// Applies true LayerNorm over each row:
//   y[row, d] = (x[row, d] - mean(row)) * rsqrt(var(row) + eps) * weight[d] + bias[d]
//
// bias may be null (treated as zeros).
void launch_layernorm(const half* x, const half* weight, const half* bias, half* y, int rows,
                      int cols, float eps, cudaStream_t stream);

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
void launch_embedding_lookup(const half* embedding, const int* token_ids, half* out, int num_tokens,
                             int hidden, cudaStream_t stream);

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
void launch_rope_inplace(half* q, half* k, int num_heads_q, int num_heads_k, int head_dim,
                         int position, float rope_theta, cudaStream_t stream);

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
void launch_rope_inplace_table(half* q, half* k, int num_heads_q, int num_heads_k, int head_dim,
                               int position, const float* cos_table, const float* sin_table,
                               cudaStream_t stream);

// launch_rope_inplace_partial_table
//
// Partial-RoPE variant that rotates only the first rotary_dim channels of each
// head and leaves the remaining channels unchanged.
void launch_rope_inplace_partial_table(half* q, half* k, int num_heads_q, int num_heads_k,
                                       int head_dim, int rotary_dim, int position,
                                       const float* cos_table, const float* sin_table,
                                       cudaStream_t stream);

// Persistent-decode interpreter (kernels_persistent_decode.cu): the whole single-token
// forward as one cooperative launch over a device array of resolved op descriptors.
// The engine compiles its (already-data) plan into these; kinds it cannot express reject
// the plan at compile time and the decode falls back to the graph path.
struct PersistOp {
  enum Kind {
    kEmbed = 0,
    kScale,
    kCopy,
    kAdd,
    kGeluMul,
    kRmsNorm,
    kRope,
    kGemv,
    kLmHead,
    kKvStore,
    kAttnStats,
    kAttnReduce,
  };
  int kind = 0;
  int rows = 0, cols = 0, in_dim = 0;
  int heads = 0, kv_heads = 0, head_dim = 0, window = 0;
  int aux_offset = 0, chunk_size = 0, chunks = 0;
  float scale = 1.0f, eps = 1e-6f;
  const half* weight = nullptr;
  const half* in = nullptr;
  const half* in2 = nullptr;
  half* out = nullptr;
  float* fout = nullptr;  // LmHead logits
  const float* cosT = nullptr;
  const float* sinT = nullptr;
  float* sm = nullptr;  // attention scratch
  float* sl = nullptr;
  float* so = nullptr;
  half* kcache = nullptr;
  half* vcache = nullptr;
  // Set by the compiler when the next op reads only same-grid-stride elements of this op's
  // output (a pure elementwise chain): the grid.sync after this op is skipped, because block
  // b's writes are exactly what block b reads next.
  int no_sync = 0;
};

// Grid size the cooperative launch can co-schedule (SMs x occupancy for the kernel).
int persistent_decode_max_blocks();

// One cooperative launch executing `n_ops` descriptors. Returns false if the launch is
// rejected (co-residency), letting the caller fall back to the graph path.
bool launch_persistent_decode(const PersistOp* ops, int n_ops, const int* tok, const int* pos,
                              int blocks, cudaStream_t stream);

// launch_attention_split_any / _device_pos
//
// Split-K decode attention for any even head_dim <= kTiledMaxHeadDim, sliding-window aware.
// Grid is (num_heads, scratch_chunks) with per-chunk seq/window guards, so it is graph-safe
// and fills the GPU where the single-block-per-head tiled path cannot (Gemma 4: 8 heads on a
// large GPU). The reduce merges chunk softmax stats exactly (log-sum-exp).
void launch_attention_split_any(const half* q, const half* k_cache, const half* v_cache, half* out,
                                int seq_len, int window, int num_heads, int num_kv_heads,
                                int head_dim, float* scratch_m, float* scratch_l, float* scratch_o,
                                int chunk_size, int scratch_chunks, cudaStream_t stream);
void launch_attention_split_any_device_pos(const half* q, const half* k_cache, const half* v_cache,
                                           half* out, const int* position, int window,
                                           int num_heads, int num_kv_heads, int head_dim,
                                           float* scratch_m, float* scratch_l, float* scratch_o,
                                           int chunk_size, int scratch_chunks, cudaStream_t stream);
// Multi-query (T <= 16) device-position form for the graphed speculative verify: query
// token t attends causally over cache[0 .. *position + t], q/out use the sequence-slot
// stride, scratch gains a leading token axis (sized tokens * num_heads * scratch_chunks).
void launch_attention_split_any_mt_device_pos(const half* q, const half* k_cache,
                                              const half* v_cache, half* out, const int* position,
                                              int tokens, int window, int num_heads,
                                              int num_kv_heads, int head_dim, float* scratch_m,
                                              float* scratch_l, float* scratch_o, int chunk_size,
                                              int scratch_chunks, cudaStream_t stream);
// Sequence form of the device-position KV append: `rows` K/V rows land at device base
// position + row. Graph-capturable replacement for the host-offset seq memcpys.
void launch_store_kv_seq_device_pos(const half* k, const half* v, half* k_cache, half* v_cache,
                                    const int* position, int kv_hidden, int rows, int max_context,
                                    cudaStream_t stream);

// Tree-verify attention (EAGLE tree rounds): `rows` draft-tree rows attend the committed
// cache [0, *cache_len) plus in-batch scratch rows admitted by their ancestor bitmask
// (bit j of anc_mask[row]; a row's own bit is set). Sibling rows share positions, so
// their K/V live in [rows, kv_hidden] scratch, not the cache; causality is the mask.
// Fixed shape + device cache_len = graph-capturable. rows <= 32 (bitmask width).
void launch_attention_tree_masked(const half* q, const half* k_cache, const half* v_cache,
                                  const half* k_scratch, const half* v_scratch, half* out,
                                  const int* cache_len, const unsigned int* anc_mask, int rows,
                                  int num_heads, int num_kv_heads, int head_dim, int q_row_stride,
                                  cudaStream_t stream);

// Split-K form of the tree-verify attention (the one-block-per-(row,head) form above is
// latency-bound at depth): cache chunks go through the guarded fixed-grid stats core at
// constant per-row length *cache_len, the masked in-batch columns become one virtual
// chunk at slot ceil(cache_len/chunk_size), and the reduce merges both. scratch_chunks
// must include the +1 spare slot for the virtual chunk.
void launch_attention_tree_split(const half* q, const half* k_cache, const half* v_cache,
                                 const half* k_scratch, const half* v_scratch, half* out,
                                 const int* cache_len, const unsigned int* anc_mask, int rows,
                                 int num_heads, int num_kv_heads, int head_dim, float* scratch_m,
                                 float* scratch_l, float* scratch_o, int chunk_size,
                                 int scratch_chunks, cudaStream_t stream);

// launch_rmsnorm_add_scale
//
// Decode fusion: x = (x + rmsnorm_w(tmp)) * alpha over one row. Replaces the
// [RmsNorm -> AddInplace -> optional ScaleCopy] sandwich-norm tail (three ~12 us kernel
// entries) with one.
void launch_rmsnorm_add_scale(half* x, const half* tmp, const half* w, int cols, float eps,
                              float alpha, cudaStream_t stream);

// launch_rmsnorm_rope
//
// Decode fusion: per-head rmsnorm then table RoPE, in place. pos_ptr (device) wins over
// position (host) when non-null, so the kernel is graph-capturable.
void launch_rmsnorm_rope(half* s, const half* w, int heads, int head_dim, int position,
                         const int* pos_ptr, const float* cos_table, const float* sin_table,
                         float eps, cudaStream_t stream);

// launch_rope_inplace_partial_table_device_pos
//
// The partial-RoPE variant with the position read from a device pointer; the piece that
// makes plans carrying partial RoPE (Gemma 4's full layers, Qwen3.5's attention layers)
// graph-capturable. Same table layout as launch_rope_inplace_partial_table.
void launch_rope_inplace_partial_table_device_pos(half* q, half* k, int num_heads_q,
                                                  int num_heads_k, int head_dim, int rotary_dim,
                                                  const int* position, const float* cos_table,
                                                  const float* sin_table, cudaStream_t stream);

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
void launch_rope_inplace_device_pos(half* q, half* k, int num_heads_q, int num_heads_k,
                                    int head_dim, const int* position, const float* cos_table,
                                    const float* sin_table, cudaStream_t stream);

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
// launch_rope_inplace_batched_strided
//
// Batched RoPE where Q and K rows carry an explicit stride, so they can be rotated in place
// inside the fused QKV buffer instead of being copied out to contiguous buffers first. Prefill
// is host-bound and those copies were 3 of the 7 cudaMemcpy2DAsync per layer.
// launch_rope_inplace_batched is this with the natural strides; bit-identical.
void launch_rope_inplace_batched_strided(half* q, half* k, int num_tokens, int num_heads_q,
                                         int num_heads_k, int head_dim, int start_position,
                                         const float* cos_table, const float* sin_table,
                                         int q_row_stride, int k_row_stride, cudaStream_t stream);

// Device-position twin (graph-capturable fixed-shape batched forwards).
void launch_rope_inplace_batched_strided_device_pos(half* q, half* k, int num_tokens,
                                                    int num_heads_q, int num_heads_k, int head_dim,
                                                    const int* start_position,
                                                    const float* cos_table, const float* sin_table,
                                                    int q_row_stride, int k_row_stride,
                                                    cudaStream_t stream);

// Tree twin: row i is roped at *start_position + row_off[i] (its tree depth; device
// buffer, constant per static tree shape). Sibling rows share a depth, which the
// sequential device-pos twin cannot express.
void launch_rope_inplace_batched_offsets_device_pos(half* q, half* k, int num_tokens,
                                                    int num_heads_q, int num_heads_k, int head_dim,
                                                    const int* start_position, const int* row_off,
                                                    const float* cos_table, const float* sin_table,
                                                    int q_row_stride, int k_row_stride,
                                                    cudaStream_t stream);

// logits[*index] = -inf (device-read index): chains argmax -> mask -> argmax for
// device-side top-k extraction with no host sync; graph-capturable.
void launch_mask_logit(float* logits, const int* index, cudaStream_t stream);

// EAGLE tree scatter: copy accepted verify rows' K/V (indices rows_idx[0..n)) from the
// per-layer [scr_rows, kv_hidden] scratch into every layer's cache at base..base+n-1.
void launch_eagle_tree_scatter(const half* k_scr, const half* v_scr, half* k_cache, half* v_cache,
                               const int* rows_idx, int n, int base, int kv_hidden, int scr_rows,
                               int num_layers, std::size_t cache_layer_stride,
                               cudaStream_t stream);

void launch_rope_inplace_batched(half* q, half* k, int num_tokens, int num_heads_q, int num_heads_k,
                                 int head_dim, int start_position, const float* cos_table,
                                 const float* sin_table, cudaStream_t stream);

// Per-position RoPE (P2 batched decode): row `i` rotated at positions[i].
void launch_rope_inplace_perpos(half* q, half* k, int num_tokens, int num_heads_q, int num_heads_k,
                                int head_dim, const int* positions, const float* cos_table,
                                const float* sin_table, cudaStream_t stream);

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
void launch_attention_step(const half* q, const half* k_cache, const half* v_cache, half* out,
                           int seq_len, int num_heads, int num_kv_heads, int head_dim,
                           cudaStream_t stream, float* scratch_m = nullptr,
                           float* scratch_l = nullptr, float* scratch_o = nullptr,
                           int scratch_chunks = 0, bool allow_split = true);

// Paged split-K decode attention (P3). K/V live in a block pool laid out like a
// flat cache of (num_blocks*block_size) tokens; block_table[c] gives the physical
// block for logical chunk c (block_size == the split chunk). Same math/output as
// launch_attention_step's split-K path; enables non-contiguous KV.
void launch_attention_step_paged(const half* q, const half* k_pool, const half* v_pool,
                                 const int* block_table, half* out, int seq_len, int num_heads,
                                 int num_kv_heads, int head_dim, int block_size,
                                 cudaStream_t stream, float* scratch_m, float* scratch_l,
                                 float* scratch_o, int scratch_chunks);

// Batched paged decode attention (P2 primitive): one decode step for `batch`
// sequences in one launch. block_tables/seq_lens are per-sequence; q/out/scratch
// are batched ([batch][num_heads][...]); all share the one KV block pool.
void launch_attention_step_batched_paged(const half* q, const half* k_pool, const half* v_pool,
                                         const int* block_tables, const int* seq_lens,
                                         int max_blocks, int max_seq_len, half* out, int batch,
                                         int num_heads, int num_kv_heads, int head_dim,
                                         int block_size, cudaStream_t stream, float* scratch_m,
                                         float* scratch_l, float* scratch_o, int scratch_chunks);

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
// `window` > 0 restricts attention to the last `window` keys (sliding-window
// layers); 0 = full causal. Windowed requests use the tiled/fallback path (the
// split-K / GQA-fused device-pos kernels are full-attention only).
void launch_attention_step_device_pos(const half* q, const half* k_cache, const half* v_cache,
                                      half* out, const int* position, int num_heads,
                                      int num_kv_heads, int head_dim, cudaStream_t stream,
                                      float* scratch_m = nullptr, float* scratch_l = nullptr,
                                      float* scratch_o = nullptr, int scratch_chunks = 0,
                                      bool allow_split = true, int window = 0);

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
void launch_store_kv_device_pos(const half* k, const half* v, half* k_cache, half* v_cache,
                                const int* position, int kv_hidden, int max_context,
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
//   causal       - true (default): each token attends only to its own prefix.
//                  false: every token attends to the whole sequence; the
//                  bidirectional self-attention a vision encoder needs.
//   limits       - optional device array [num_tokens]: the exclusive key limit for each
//                  token, overriding `causal`. This is what makes an image span
//                  bidirectional (every token in the span sees the whole span) while the
//                  surrounding text stays causal.
//   window       - sliding-window width; 0 = unlimited. Keys before (limit - window) are
//                  skipped. The decode path always did this; prefill did not (Llama has
//                  no windows), so a windowed model with a prompt longer than the window
//                  was silently wrong here.
// launch_build_attention_ptrs
//
// Builds the cublasGemmBatchedEx pointer arrays for the tensor-core prefill attention on the
// device. Doing this from the host into a reused pinned buffer is a race; the function runs
// once per layer, so the next layer overwrites the staging buffer while the previous async copy
// is still in flight. Building them on the compute stream makes the ordering structural.
void launch_build_attention_ptrs(const half* k_layer, const half* v_layer, const half* q,
                                 half* scores, half* out, void** ptrs, int num_heads, int group,
                                 int head_dim, int kchunk, int keys, int q_stride, int out_stride,
                                 int c0, cudaStream_t stream);

// launch_gated_glu_interleaved
//
// silu(gate)*up (or gelu) read straight off the fused w13 output [tokens, 2*inter]: gate at
// [t][i], up at [t][inter+i]. Prefill used to split that into two buffers with a pair of
// cudaMemcpy2DAsync first; copies that only un-interleaved data the next kernel reads
// elementwise anyway. Prefill is host-bound, so those copies cost real time.
// Bit-identical to split-then-glu.
void launch_gated_glu_interleaved(const half* ff13, half* out, int inter, int tokens, bool gelu,
                                  cudaStream_t stream);

// launch_softmax_causal_rows
//
// In-place causal row-softmax over a [heads][chunk][keys] score matrix. Row (h, i) belongs to
// query token (q_start + i) and may attend to keys j <= q_start + i; masked entries are set to
// zero (they feed a GEMM, not another softmax, so zero is what drops them).
//
// This is the middle of the tensor-core prefill attention: cuBLAS batched GEMM produces the
// scores, this normalises them, and a second GEMM consumes them as P.
void launch_softmax_causal_rows(half* scores, int heads, int chunk_stride, int rows, int keys,
                                int q_start, cudaStream_t stream, int window = 0);

void launch_attention_prefill(const half* q, const half* k_cache, const half* v_cache, half* out,
                              int num_tokens, int start_position, int num_heads, int num_kv_heads,
                              int head_dim, cudaStream_t stream, bool causal = true,
                              const int* limits = nullptr, int window = 0);

// Paged prefill attention + paged KV scatter (P3 phase 2d). K/V live in a block
// pool; block_table maps logical chunk -> physical block (block_size tokens each).
void launch_attention_prefill_paged(const half* q, const half* k_pool, const half* v_pool,
                                    const int* block_table, half* out, int num_tokens,
                                    int start_position, int num_heads, int num_kv_heads,
                                    int head_dim, int block_size, cudaStream_t stream);
void launch_store_kv_paged(half* k_pool, half* v_pool, const half* k_src, const half* v_src,
                           const int* block_table, int base_pos, int rows, int kv_hidden,
                           int block_size, cudaStream_t stream);
// Batched decode KV scatter (P2): one token per sequence to its own block table
// at positions[b].
void launch_store_kv_batched_paged(half* k_pool, half* v_pool, const half* k_src, const half* v_src,
                                   const int* block_tables, const int* positions, int max_blocks,
                                   int batch, int kv_hidden, int block_size, cudaStream_t stream);

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
void launch_add_bias_broadcast(half* out, const half* bias, int rows, int cols,
                               cudaStream_t stream);

// launch_add_bias_inplace_float_from_half
//
// Adds an fp16 bias vector to an fp32 vector in place:
//   out[i] += float(bias[i])
void launch_add_bias_inplace_float_from_half(float* out, const half* bias, int n,
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
void launch_silu_mul(const half* gate, const half* up, half* out, int n, cudaStream_t stream);
// GeGLU: gelu(gate) * up (tanh GELU approximation, Gemma). Same shape as silu_mul.
void launch_gelu_mul(const half* gate, const half* up, half* out, int n, cudaStream_t stream);

// launch_apply_sigmoid_gate_inplace
//
// Element-wise gated multiply used by Qwen3.5 full attention:
//   values[i] *= sigmoid(gate[i])
void launch_apply_sigmoid_gate_inplace(half* values, const half* gate, int n, cudaStream_t stream);

// launch_split_interleaved_head_halves
//
// Splits a tensor laid out as repeated per-head pairs:
//   src[head] = [first_half, second_half]
// into two separate outputs.
void launch_split_interleaved_head_halves(const half* src, half* first, half* second, int heads,
                                          int head_dim, cudaStream_t stream);

// Qwen3.5 linear-attention helpers.
void launch_linear_conv1d_silu(const half* conv_weight, float* conv_state, half* qkv_mix,
                               int channels, int kernel_size, cudaStream_t stream);

void launch_repeat_linear_heads(const half* qkv_mix, half* q_out, half* k_out, half* v_out,
                                int num_key_heads, int num_value_heads, int key_head_dim,
                                int value_head_dim, cudaStream_t stream);

void launch_linear_attention_step(const half* q, const half* k, const half* v, const half* z,
                                  const half* a, const half* b, const float* norm_weight,
                                  const float* a_log, const half* dt_bias, float* recurrent_state,
                                  half* out, int num_heads, int key_head_dim, int value_head_dim,
                                  float rms_eps, cudaStream_t stream);

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
void launch_scale_copy(half* dst, const half* src, int n, float scale, cudaStream_t stream);

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
void launch_scale_add_inplace(half* dst, const half* src, int n, float scale, cudaStream_t stream);

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
//   per_expert_scale - optional learned per-expert gain, fp16 [experts], applied to
//                      the weight after the top-k renormalisation (Gemma MoE has
//                      one; pass nullptr for routers that do not, which leaves the
//                      result exactly as it was).
void launch_moe_router_topk_softmax(const half* logits, int experts, int top_k, int* topk_idx,
                                    float* topk_prob, cudaStream_t stream,
                                    const half* per_expert_scale = nullptr, bool renorm = true);

// Sequence prefill: route all T tokens (logits [T, experts]) in one launch -> topk_idx/prob [T, top_k].
void launch_moe_router_topk_softmax_seq(const half* logits, int experts, int top_k, int* topk_idx,
                                        float* topk_prob, int T, cudaStream_t stream,
                                        const half* per_expert_scale = nullptr, bool renorm = true);

// launch_moe_router_sigmoid_topk
//
// Kimi-K3-style router: sigmoid gate per expert (independent, not softmax), optional grouped
// node-limited selection, top-k, then normalise the selected gates to sum 1. Supports top_k up to
// 32 (K3 uses 16, above the softmax router's hard cap of 8). Groundwork for K3; not yet wired to
// a model (its 896/top-16 MoE needs this router + expert streaming + MXFP4); gated in isolation by
// moe_router_sigmoid_test.
//   n_group / topk_group - grouped routing: experts split into n_group contiguous groups; a group's
//     score is the sum of its top-2 gates; the top `topk_group` groups are kept and top-k is taken
//     within them (DeepSeek-V3 style). n_group<=1 or a degenerate grouping -> flat top-k.
//   topk_weight - selected experts' normalised sigmoid gates [top_k].
void launch_moe_router_sigmoid_topk(const half* logits, int experts, int top_k, int n_group,
                                    int topk_group, int* topk_idx, float* topk_weight,
                                    cudaStream_t stream);

// launch_moe_gate_up_geglu / launch_moe_down_accum
//
// The expert feed-forward, with the selected experts read from device memory
// (topk_idx, written by the router); no host round-trip, so the decode graph
// stays capturable.
//
// Experts are one contiguous matrix each, so selecting expert e is a row offset:
//   gate_up: [num_experts * 2*inter, hidden]   rows e*2*inter .. +2*inter
//   down:    [num_experts * hidden,  inter ]   rows e*hidden  .. +hidden
// which means the ordinary weight quantiser applies to them unchanged.
//
//   w/scales/qbits/group - weight encoding (qbits 0 = fp16, 8 = int8, 4 = int4;
//                          group 0 = per-row scales, >0 = group-wise)
//   inter_out            - [top_k, inter] gelu(gate)*up per selected expert
//   topk_weight          - routing weight per selected expert [top_k]
//   y                    - [hidden], the routing-weighted sum over the top_k experts
void launch_moe_gate_up_geglu(const void* w, const float* scales, int qbits, int group,
                              const half* x, const int* topk_idx, half* inter_out, int inter,
                              int hidden, int top_k, cudaStream_t stream, bool use_gelu = true);

// dp4a int4 variant: `xq`/`x_scale` = the perm8-int8 quantised activation (quantise the fp16 input once
// with launch_quantize_fp16_to_int8_perm8_g32). group must be a power of two, multiple of 32, dividing
// hidden. Decode (batch-1) only.
void launch_moe_gate_up_geglu_dp4a(const void* wg, const float* sg, const std::int8_t* xq,
                                   const float* x_scale, const int* topk_idx, half* inter_out,
                                   int inter, int hidden, int top_k, int group, cudaStream_t stream,
                                   bool use_gelu = true);

// dp4a int4 down-accum: xq_all/xs_all = the top_k per-expert inter vectors, mt-quantised to perm8-int8
// (launch_quantize_fp16_to_int8_perm8_g32_mt with rows=top_k, cols=inter). Decode (batch-1) only.
void launch_moe_down_accum_dp4a(const void* wd, const float* sd, const std::int8_t* xq_all,
                                const float* xs_all, const int* topk_idx, const float* topk_weight,
                                half* y, int hidden, int inter, int top_k, int group,
                                cudaStream_t stream);

void launch_moe_down_accum(const void* w, const float* scales, int qbits, int group,
                           const half* inter_in, const int* topk_idx, const float* topk_weight,
                           half* y, int hidden, int inter, int top_k, cudaStream_t stream);

// int4-direct grouped MoE GEMM on int8 tensor cores: reads int4 expert weights directly (no dequant),
// int8 activations (natural per-32-group scales). off[] is device-resident (grid.z = expert). The
// fused gate_up output splits at column `split` into out_lo (gate) / out_hi (up); down passes split=N
// and out_hi=nullptr. Skips the dequant-all fixed cost that dominates short-prompt MoE prefill.
void launch_moe_int4_grouped_mma(const std::int8_t* xq, const float* as, const std::int8_t* wpacked,
                                 const float* ws, const int* off, half* out_lo, half* out_hi, int E,
                                 int N, int K, int max_ne, int group, int split, int lo_width,
                                 int hi_width, cudaStream_t stream);

// launch_mul_vec
//
// out[i] = in[i] * vec[i] * scale. Elementwise gain by a learned vector plus a
// constant (the MoE router's pre-projection scaling).
void launch_mul_vec(const half* in, const half* vec, half* out, int n, float scale,
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
void launch_dequant_int8_to_fp16(const std::int8_t* src, half* dst, int n, float scale,
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
void launch_dequant_rowwise_int8_to_fp16(const std::int8_t* src, const float* scales, half* dst,
                                         int rows, int cols, cudaStream_t stream);

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
void launch_quantize_rowwise_fp16_to_int8(const half* src, std::int8_t* dst, float* scales,
                                          int rows, int cols, cudaStream_t stream, int max_q = 127);

// launch_quantize_groupwise_fp16_to_int8
//
// Weight quantisation with one scale per group of `group` contiguous columns
// instead of one scale per row:
//   g              = col >> log2(group)
//   scales[row, g] = max(abs(src[row, g*group : (g+1)*group])) / max_q
//   dst[row, col]  = clamp(round(src[row, col] / scales[row, g]), -max_q, max_q)
//
// Why: a single row scale must span the row's largest magnitude, so one outlier
// weight coarsens every other weight in that row. At int8 (255 levels) that is
// survivable; at int4 (16 levels) across thousands of columns it is not, and the
// error compounds across layers. Narrowing each scale to ~128 columns keeps the
// outlier local. Cost is 32 bits per group, i.e. 4 + 32/group bits per weight
// (4.25 bits at group=128).
//
// `group` must be a positive power of two and <= cols. Scales are [rows,
// n_groups] row-major with n_groups = ceil(cols / group); use
// quant_group_count() to size the buffer. This is the load-time weight path;
// activation quantisation stays per-row (launch_quantize_rowwise_fp16_to_int8).
void launch_quantize_groupwise_fp16_to_int8(const half* src, std::int8_t* dst, float* scales,
                                            int rows, int cols, int group, cudaStream_t stream,
                                            int max_q = 127);

// Number of groups per row, i.e. the second dimension of the scales buffer.
// group <= 0 means "one scale per row" (n_groups == 1).
inline int quant_group_count(int cols, int group) {
  if (group <= 0 || group >= cols) {
    return 1;
  }
  return (cols + group - 1) / group;
}

// launch_pack_rowwise_int8_to_int4
//
// Packs row-major signed int8 values into signed int4 (two values per byte).
// Input is expected to be within the int4 range [-8, 7]; values are clamped.
// Layout-only: it does not touch scales, so it serves both the per-row and the
// group-wise quantisers.
void launch_pack_rowwise_int8_to_int4(const std::int8_t* src, std::int8_t* dst, int rows, int cols,
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
// launch_weight_only_int8_gemv_f32
//
// int8 weight-only GEMV with fp16 activations and float output: the LM head.
//   y[row] = scales[row] * sum_k w[row, k] * x[k]
// Only the weights are quantized; x stays fp16 and the dot accumulates in fp32. The dp4a
// kernels quantize the activation too, which is the wrong trade in the one layer that decides
// the output token (x is 8 KB; the weight is 1 GB). An 8B LM head is 1.05 GB fp16; int8 halves
// it, and that is 22% of everything an int4 8B reads per token.
void launch_weight_only_int8_gemv_f32(const std::int8_t* w, const float* scales, const half* x,
                                      float* y, int out_features, int in_features,
                                      cudaStream_t stream);

void launch_weight_only_int8_matvec(const std::int8_t* w, const float* scales, const half* x,
                                    half* y, int out_features, int in_features,
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
void launch_weight_only_int8_matvec_batched(const std::int8_t* w, const float* scales,
                                            const half* x, half* y, int batch_size,
                                            int out_features, int in_features, cudaStream_t stream);

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
void launch_weight_only_int8_matvec_batched_dp4a(const std::int8_t* w, const float* w_scales,
                                                 const std::int8_t* x, const float* x_scales,
                                                 half* y, int batch_size, int out_features,
                                                 int in_features, cudaStream_t stream);

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
void launch_weight_only_int8_matvec_dp4a(const std::int8_t* w, const float* w_scales,
                                         const std::int8_t* x, const float* x_scale, half* y,
                                         int out_features, int in_features, cudaStream_t stream,
                                         int warps_per_block = 0, int tile_packed4 = 0,
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
void launch_weight_only_int8_matvec_dual_dp4a(const std::int8_t* w_a, const float* w_scales_a,
                                              const std::int8_t* w_b, const float* w_scales_b,
                                              const std::int8_t* x, const float* x_scale, half* y_a,
                                              half* y_b, int out_features, int in_features,
                                              cudaStream_t stream, int warps_per_block = 0,
                                              int tile_packed4 = 0, int warps_per_row = 1);

// launch_weight_only_int4_matvec
//
// Weight-only int4 matrix-vector multiply with per-row scales and fp16 input.
// Weights are packed row-major with two signed int4 values per byte
// (low nibble first), matching .int4 tensor layout.
void launch_weight_only_int4_matvec(const std::int8_t* w_packed, const float* scales, const half* x,
                                    half* y, int out_features, int in_features,
                                    cudaStream_t stream);

// launch_weight_only_int{4,8}_matvec_grouped
//
// As above, but the weight scales are per group of `group` contiguous input
// features rather than per row (see launch_quantize_groupwise_fp16_to_int8).
// The scale can no longer be hoisted out of the dot product, so it is applied
// per element: y[row] = sum_col q[row,col] * scales[row, col>>shift] * x[col].
//
// `group` must be a positive power of two; `scales` is [out_features,
// quant_group_count(in_features, group)] row-major.
void launch_weight_only_int4_matvec_grouped(const std::int8_t* w_packed, const float* scales,
                                            const half* x, half* y, int out_features,
                                            int in_features, int group, cudaStream_t stream);

void launch_weight_only_int8_matvec_grouped(const std::int8_t* w, const float* scales,
                                            const half* x, half* y, int out_features,
                                            int in_features, int group, cudaStream_t stream);

// launch_weight_only_int4_matvec_batched
//
// Batched variant of launch_weight_only_int4_matvec using fp16 activations.
void launch_weight_only_int4_matvec_batched(const std::int8_t* w_packed, const float* scales,
                                            const half* x, half* y, int batch_size,
                                            int out_features, int in_features, cudaStream_t stream);

// launch_weight_only_int4_matvec_batched_dp4a
//
// Batched int4(weight) x int8(activation) GEMV using dp4a. The input
// activations are int8 with one scale per batch row.
void launch_weight_only_int4_matvec_batched_dp4a(const std::int8_t* w_packed, const float* w_scales,
                                                 const std::int8_t* x, const float* x_scales,
                                                 half* y, int batch_size, int out_features,
                                                 int in_features, cudaStream_t stream);

// launch_weight_only_int4_matvec_dp4a
//
// Single-row (batch=1) int4(weight) x int8(activation) GEMV using dp4a.
void launch_weight_only_int4_matvec_dp4a(const std::int8_t* w_packed, const float* w_scales,
                                         const std::int8_t* x, const float* x_scale, half* y,
                                         int out_features, int in_features, cudaStream_t stream,
                                         int warps_per_block = 0, int tile_packed4 = 0,
                                         int warps_per_row = 1);

// launch_weight_only_int4_matvec_dual_dp4a
//
// Dual-output dp4a GEMV for two packed-int4 weight matrices sharing the same
// int8 input activation vector and scale.
void launch_weight_only_int4_matvec_dual_dp4a(
    const std::int8_t* w_a_packed, const float* w_scales_a, const std::int8_t* w_b_packed,
    const float* w_scales_b, const std::int8_t* x, const float* x_scale, half* y_a, half* y_b,
    int out_features, int in_features, cudaStream_t stream, int warps_per_block = 0,
    int tile_packed4 = 0, int warps_per_row = 1);

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
// ── vision-encoder ops ──
//
// launch_rope_2d_inplace
//
// 2-D RoPE over a sequence of patches. A head's channels split into two halves: the
// first rotates by the patch's x coordinate, the second by its y. Within each half the
// rotation is rotate_half (channel j pairs with j + half/2); the same convention the
// 1-D table kernel uses, so cos/sin tables are built identically (here over the spatial
// half-dim: head_dim/4 entries per position).
//
//   x        - q or k, fp16 [tokens, num_heads, head_dim]; modified in place
//   pos_x/y  - per-patch integer coordinates on device [tokens]
void launch_rope_2d_inplace(half* x, const int* pos_x, const int* pos_y, int num_heads,
                            int head_dim, int tokens, const float* cos_table,
                            const float* sin_table, cudaStream_t stream);

// launch_patch_embed
//
// out[t] = proj . (2*(pixels[t] - 0.5)) + pos_table[0][x_t] + pos_table[1][y_t]
//
// Gemma applies no mean/std image normalisation; it rescales [0,1] to [-1,1] here.
// The position table is [2, pos_table_size, hidden]: x indexes plane 0, y plane 1, and
// the two are summed. Patches with a negative coordinate are padding and emit zeros.
//
//   pixels   - fp32 [tokens, patch_dim], patch_dim = 3 * patch_size^2
void launch_patch_embed(const half* proj, const float* pixels, const half* pos_table,
                        const int* pos_x, const int* pos_y, half* out, int tokens, int hidden,
                        int patch_dim, int pos_table_size, cudaStream_t stream);

// launch_avg_pool_patches
//
// 2-D average pool over k x k patch cells, then a `gain` (sqrt(hidden)) multiply.
// Padding patches are zero and contribute nothing, but the divisor stays k*k (as HF
// does); dividing by the live count instead would change the numbers.
void launch_avg_pool_patches(const half* in, const int* pos_x, const int* pos_y, half* out,
                             int tokens, int hidden, int k, int cells_x, int out_tokens, float gain,
                             cudaStream_t stream);

// launch_standardize
//
// x = (x - bias) * scale, broadcast over tokens. Null on checkpoints with
// standardize=false (Gemma 4 E2B); present on the 26B.
void launch_standardize(half* x, const half* bias, const half* scale, int tokens, int hidden,
                        cudaStream_t stream);

// launch_gelu_mul_strided
//
// out[t][i] = gelu(a[t][i]) * b[t*b_stride + i]. The per-layer-input gate multiplies by a
// slice of a wider per-token tensor, so b strides differently from a/out.
void launch_gelu_mul_strided(const half* a, const half* b, half* out, int n, int tokens,
                             int b_stride, cudaStream_t stream);

// launch_gemv_splitk_f16
//
// GEMV with one block per output row and 8 warps splitting the input dimension.
// The default GEMV gives one warp per row, so its parallelism is the row count; ample for
// a 151936-row LM head, and only ~10% of the GPU for a 896-row o_proj, which then runs 8x
// off peak because memory latency is never hidden. Use this for short-and-wide GEMVs.
// Reduction order differs from the one-warp kernel, so results are equal to fp32 rounding
// but not bit-identical; opt in per call site.
void launch_gemv_splitk_f16(const half* w, const half* x, half* y, int out_features,
                            int in_features, cudaStream_t stream, half* residual = nullptr);

// launch_swiglu_gemv_f16
//
// Fused SwiGLU projection: out[i] = silu(dot(w_gate[i], x)) * dot(w_up[i], x).
// Replaces gate-GEMV + up-GEMV + silu_mul (3 kernels -> 1). At batch 1 each kernel costs
// a fixed ~2.7us of scheduling regardless of its size, and that tax, not bandwidth,
// is what separates a small model from the roofline.
// g and u are rounded to fp16 before the multiply, exactly as the unfused path does when
// it stores them between kernels, so the result is byte-identical.
void launch_swiglu_gemv_f16(const half* w_gate, const half* w_up, const half* x, half* out,
                            int inter, int in_features, cudaStream_t stream);

// launch_rope_seq_table
//
// RoPE over a whole prompt chunk: token t sits at position start_position + t. Same
// rotate_half table convention as the single-token kernel, so a chunk prefilled here and
// the same tokens decoded one at a time agree exactly.
//   rotary_dim - 0 = rotate the full head; >0 = partial RoPE over the first rotary_dim
void launch_rope_seq_table(half* x, int num_heads, int head_dim, int start_position, int tokens,
                           const float* cos_table, const float* sin_table, int rotary_dim,
                           cudaStream_t stream);
// Device-position twin (base position read from device memory) so a captured graph can
// replay sequence RoPE at any position. Identical rotation math.
void launch_rope_seq_table_device_pos(half* x, int num_heads, int head_dim, const int* position_ptr,
                                      int tokens, const float* cos_table, const float* sin_table,
                                      int rotary_dim, cudaStream_t stream);

// launch_rowmajor_half_gemm_f16
//
// Sequence-mode GEMM: y[t, m] = sum_k w[m, k] * x[t, k], for `tokens` rows of x.
// Same weight layout as the GEMV below, so one bound weight serves both.
//
// A tiled GEMM sums K in a different order than the GEMV, so the two do not agree
// bit-for-bit; single-token work must keep using the GEMV or decode output shifts.
// in_min/in_max/out_min/out_max: clipped projections (Gemma 4 E2B's vision tower)
// clamp the activation on the way in and the result on the way out. +-inf = no clamp.
void launch_rowmajor_half_gemm_f16(const half* w, const half* x, half* y, int out_features,
                                   int in_features, int tokens, cudaStream_t stream,
                                   float in_min = -INFINITY, float in_max = INFINITY,
                                   float out_min = -INFINITY, float out_max = INFINITY);

//   residual (optional): fuses the residual add into the epilogue:
//     residual[row] = __hadd(residual[row], (half)dot)   and `y` is left untouched.
//   Saves a whole add_inplace launch. byte-identical: the dot is rounded to fp16 first
//   (what the unfused store does) and combined with __hadd (what add_inplace does).
void launch_rowmajor_half_gemv_f16(const half* w, const half* x, half* y, int out_features,
                                   int in_features, cudaStream_t stream, int warps_per_block = 0,
                                   int tile_pairs = 0, int rows_per_warp = 1,
                                   half* residual = nullptr);

// launch_quantize_fp16_to_int8_perm8
//
// Batch-1 rowwise int8 activation quantization (same max/scale/rounding as
// launch_quantize_rowwise_fp16_to_int8) with even/odd bytes deinterleaved per 8-column
// window; the layout launch_weight_only_int4_matvec_grouped_dp4a consumes. cols % 8 == 0.
void launch_quantize_fp16_to_int8_perm8(const half* src, std::int8_t* dst, float* scales, int cols,
                                        cudaStream_t stream);

// Group-32 variant (q8_1 style): scales[cols/32], no global max, multi-block. The dp4a
// grouped kernels consume these per-chunk. cols % 32 == 0.
void launch_quantize_fp16_to_int8_perm8_g32(const half* src, std::int8_t* dst, float* scales,
                                            int cols, cudaStream_t stream);
// Multi-row form: quantizes `rows` consecutive vectors in one launch (dst stride = cols,
// scale stride = cols/32 per row).
void launch_quantize_fp16_to_int8_perm8_g32_mt(const half* src, std::int8_t* dst, float* scales,
                                               int cols, int rows, cudaStream_t stream);

// launch_weight_only_int4_matvec_grouped_dp4a
//
// Batch-1 grouped int4(weight) x int8(activation) GEMV via dp4a. `xq` must hold the
// perm8-quantized activation and `x_scale` its scale. Requires in_features % 32 == 0 and
// group % 32 == 0 (silently returns otherwise; caller gates). `warps` (2/4/8) is the
// rows-per-block shape knob probed by the per-box autotuner; all variants are
// bit-identical (row math is warp-local).
void launch_weight_only_int4_matvec_grouped_dp4a(const std::int8_t* w_packed, const float* scales,
                                                 const std::int8_t* xq, const float* x_scale,
                                                 half* y, int out_features, int in_features,
                                                 int group, cudaStream_t stream, int warps = 4);

// launch_rmsnorm_quant_perm8
//
// rows=1 rmsnorm fused with perm8 int8 activation quantization (XNorm sites feeding dp4a
// projections). Bit-identical to [launch_rmsnorm; launch_quantize_fp16_to_int8_perm8]:
// the quantizer reads the fp16-rounded normed values. cols % 8 == 0 and cols <= 2048.
void launch_rmsnorm_quant_perm8(const half* x, const half* w, half* y, std::int8_t* xq,
                                float* xscale, int cols, float eps, cudaStream_t stream);

// launch_weight_only_int4_matvec_grouped_dp4a_cat
//
// Up to three grouped int4 dp4a GEMVs sharing one perm8-quantized activation, run as one
// launch (q|k|v, gate|up). Same per-row math as the non-cat variant; pass n2 = 0 for two
// segments. Same gating: in_features % 32 == 0, group % 32 == 0.
void launch_weight_only_int4_matvec_grouped_dp4a_cat(const std::int8_t* w0, const float* s0,
                                                     half* y0, int n0, const std::int8_t* w1,
                                                     const float* s1, half* y1, int n1,
                                                     const std::int8_t* w2, const float* s2,
                                                     half* y2, int n2, const std::int8_t* xq,
                                                     const float* x_scale, int in_features,
                                                     int group, cudaStream_t stream, int warps = 4);

// launch_weight_only_int4_matvec_grouped_dp4a_glu
//
// Fused GeGLU: out[r] = gelu_tanh(gate_r) * up_r with both grouped-int4 dp4a dots computed
// in one warp against the shared perm8 activation. Numerics match [gate gemv; up gemv;
// gelu_mul]: dots round to fp16 before the gelu. Same gating as the other dp4a launchers.
void launch_weight_only_int4_matvec_grouped_dp4a_glu(const std::int8_t* wg, const float* sg,
                                                     const std::int8_t* wu, const float* su,
                                                     const std::int8_t* xq, const float* x_scale,
                                                     half* y, int out_features, int in_features,
                                                     int group, cudaStream_t stream, int warps = 4);

// launch_weight_only_int4_matvec_grouped_dp4a_f32
//
// Float-output grouped dp4a GEMV for the quantized LM head (logits stay fp32).
// Consumes the perm8-g32 activation like the other dp4a launchers.
void launch_weight_only_int4_matvec_grouped_dp4a_f32(const std::int8_t* w_packed,
                                                     const float* scales, const std::int8_t* xq,
                                                     const float* x_scale, float* y,
                                                     int out_features, int in_features, int group,
                                                     cudaStream_t stream);

// launch_weight_only_int8_matvec_glu
//
// Fused GeGLU for rowwise-int8 weights with fp16 activations: out[r] = gelu(gate_r)*up_r,
// one row per warp, gelu on the fp16-rounded dots. in_features % 16 == 0.
void launch_weight_only_int8_matvec_glu(const std::int8_t* wg, const float* sg,
                                        const std::int8_t* wu, const float* su, const half* x,
                                        half* y, int out_features, int in_features,
                                        cudaStream_t stream);

// launch_half_gemv_glu
//
// Fused fp16 GeGLU: out[r] = gelu_tanh(gate_r) * up_r, paired-warp shape, gelu on the
// fp16-rounded dots. in_features % 8 == 0.
void launch_half_gemv_glu(const half* wg, const half* wu, const half* x, half* y, int out_features,
                          int in_features, cudaStream_t stream);

// launch_dequant_kquant
//
// Device-side ggml k-quant dequantization: `blocks` 256-weight super-blocks in, fp16
// out. The packed blocks are uploaded as-is, so the host never materializes the fp16
// copy. Gated against the host reference by kquant_dequant_test -- a mistake in this
// arithmetic reads as slightly-wrong weights, not as a failure.
enum class KQuantType { Q4_K, Q5_K, Q6_K };
void launch_dequant_kquant(const std::uint8_t* blocks_in, KQuantType type, std::size_t blocks,
                           half* out, cudaStream_t stream);

// launch_kquant_matvec
//
// y[rows] = W[rows, cols] * x[cols] with W left PACKED: the super-blocks are unpacked
// inside the kernel, so the weight never exists as fp16 and a quantized model stays
// resident at its file size. cols must be a multiple of 256. Gated by
// kquant_matvec_test against a host dequant + fp32 dot.
// Batched form: y[b, n] = sum_k x[b, k] * W[n, k], W left packed. x is
// [batch, cols] and y is [batch, rows], both row-major -- the layout the fp16
// GEMM path already produces. Returns false when the batch is outside the range
// this is worth doing for, which is the caller's cue to expand and use cuBLAS.
bool launch_kquant_matmul(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                          int cols, int batch, int ldy, cudaStream_t stream);

// Same, writing fp32. The LM head produces logits rather than activations.
void launch_kquant_matvec_f32(const std::uint8_t* w, KQuantType type, const half* x, float* y,
                              int rows, int cols, cudaStream_t stream);

void launch_kquant_matvec(const std::uint8_t* w, KQuantType type, const half* x, half* y, int rows,
                          int cols, cudaStream_t stream);

// launch_dequant_weight_rowwise_to_fp16
//
// Dequant in LlamaEngine's cached low-bit layout (row-wise scales; int4 packed as
// sequential column pairs, low nibble first). Prefill uses it to reach the tensor-core
// GEMM. Not interchangeable with launch_dequant_int4_grouped below: that reads the
// op-plan engine's packing.
// `group` <= 0 selects the one-scale-per-row layout; > 0 means each scale covers that
// many consecutive input columns (grouped int4), with quant_group_count(cols, group)
// scales per row. int8 is always row-wise.
void launch_dequant_weight_rowwise_to_fp16(const std::int8_t* w, const float* scales, half* out,
                                           int rows, int cols, bool int4, int group,
                                           cudaStream_t stream);

// launch_dequant_int4_grouped / launch_dequant_int8_rowwise
//
// Whole-matrix dequant to fp16 (prefill scratch: sequence mode runs the real GEMM over a
// dequantized copy; decode keeps the quant kernels).
void launch_dequant_int4_grouped(const std::int8_t* w_packed, const float* scales, half* out,
                                 int rows, int cols, int group, cudaStream_t stream);
void launch_dequant_int8_rowwise(const std::int8_t* w, const float* scales, half* out, int rows,
                                 int cols, cudaStream_t stream);

// launch_dequant_mxfp4
//
// Whole-matrix dequant of OCP MXFP4 to fp16. Each element is an E2M1 4-bit float (1 sign, 2 exp,
// 1 mantissa -> magnitudes {0,.5,1,1.5,2,3,4,6}); each block of 32 columns shares an E8M0 scale
// (8-bit power-of-two exponent, value = 2^(code - 127)). `packed` is [rows, cols/2] nibbles (two
// per byte, low nibble first); `scales` is [rows, cols/32] E8M0 bytes. cols % 32 == 0.
// value(row,col) = e2m1[nibble] * 2^(scales[row, col/32] - 127). This is the K3 weight format;
// the decode is validated in isolation (mxfp4_dequant_test) ahead of wiring it into the matvecs.
void launch_dequant_mxfp4(const std::uint8_t* packed, const std::uint8_t* scales, half* out,
                          int rows, int cols, cudaStream_t stream);

// launch_i32_scale_to_fp16
//
// Epilogue for the int8-direct sequence GEMM: fp16 y[t0+lt, r] = i32[lt, r] * sw[r] * sx[t].
void launch_i32_scale_to_fp16(const int* acc, const float* sw, const float* sx, half* y, int out,
                              int chunk, int t0, cudaStream_t stream);

// launch_weight_only_int4_matvec_grouped_dp4a_mt
//
// Multi-token (T <= 8) grouped dp4a matvec: the weights stream once for all T tokens'
// perm8/group-32 activations (x scales [token][group]). Serves speculative verify and
// short sequence remainders. Same gating as the other grouped dp4a launchers.
void launch_weight_only_int4_matvec_grouped_dp4a_mt(const std::int8_t* w_packed,
                                                    const float* scales, const std::int8_t* xq,
                                                    const float* x_scales, half* y,
                                                    int out_features, int in_features, int group,
                                                    int tokens, cudaStream_t stream);
// f32-output form for the LM head: logits for all T positions in one weight pass
// (y is [token][out_features] floats, ready for the argmax kernels).
void launch_weight_only_int4_matvec_grouped_dp4a_mt_f32(const std::int8_t* w_packed,
                                                        const float* scales, const std::int8_t* xq,
                                                        const float* x_scales, float* y,
                                                        int out_features, int in_features,
                                                        int group, int tokens, cudaStream_t stream);

// launch_rowmajor_half_gemv_cat
//
// Up to three batch-1 fp16 GEMVs sharing one input vector, run as one launch (q|k|v,
// gate|up). Per-row arithmetic mirrors the wide gemv exactly, so results are bit-identical
// to three separate launches. Requires in_features % 8 == 0. Pass n2 = 0 (w2/y2 null) for
// a two-segment cat.
void launch_rowmajor_half_gemv_cat(const half* w0, half* y0, int n0, const half* w1, half* y1,
                                   int n1, const half* w2, half* y2, int n2, const half* x,
                                   int in_features, cudaStream_t stream);

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
void launch_rowmajor_half_gemv_f32(const half* w, const half* x, float* y, int out_features,
                                   int in_features, cudaStream_t stream, int warps_per_block = 0,
                                   int tile_pairs = 0, int rows_per_warp = 1);

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
// Two-phase when part_val/part_idx scratch is supplied (sized argmax_partition_count(n)),
// otherwise a single block scans the whole vector. Supply the scratch: the single-block form
// launches <<<1,256>>> over the entire vocab, one SM of many, and measured 149.6 us on
// Qwen2.5's 151936-wide head, 12% of all GPU time, to read 608 KB.
constexpr int kArgmaxMaxParts = 1024;
int argmax_partition_count(int n);
void launch_argmax_float(const float* logits, int n, int* out_index, cudaStream_t stream,
                         float* part_val = nullptr, int* part_idx = nullptr, int parts = 0);

// launch_topk_float / launch_gather_ge_threshold
//
// Device-side top-k candidate selection for sampled decode. Together these let a
// large-vocab model sample without copying the whole logit vector to the host:
// the host only ever sees the ~k candidates.
//
// launch_topk_float: writes the k largest values (descending) and their indices.
//   part_val/part_idx - scratch, must hold topk_partition_count(n) * k entries
//   out_val/out_idx   - device [k]; out_val[k-1] is the k-th largest ("kth")
// Two phases: each block takes the top-k of its chunk in shared memory, then one
// block merges the partials. Cost is a few passes over the logits, not a sort.
//
// launch_gather_ge_threshold: appends every finite logit >= *threshold to
// out_idx/out_val and writes the count to out_count (capped at `capacity`).
// Taking the threshold from out_val[k-1] reproduces the host sampler exactly,
// including ties at the k-th value (which can yield more than k candidates).
// out_count must be zeroed before launch. If the count exceeds capacity the
// caller should fall back to the host path.
int topk_partition_count(int n);
void launch_topk_float(const float* logits, int n, int k, float* part_val, int* part_idx,
                       float* out_val, int* out_idx, cudaStream_t stream);
void launch_gather_ge_threshold(const float* logits, int n, const float* threshold, int* out_idx,
                                float* out_val, int* out_count, int capacity, cudaStream_t stream);

// Repetition penalty for the batched device top-k path, applied to [batch][vocab] logits before
// the top-k so the candidate set matches the host sampler's slow path. Both touch only rows whose
// penalties[b] > 1. Call sanitize first (non-finite -> -inf, clamp [+-80]), then the penalty over
// each row's unique seen token ids (seen_ids[i] in row seen_rows[i]); seen ids are unique per row
// so the penalty writes never collide.
void launch_sanitize_penalty_rows(float* logits, int vocab, const float* penalties, int batch,
                                  cudaStream_t stream);
void launch_repetition_penalty(float* logits, int vocab, const int* seen_ids, const int* seen_rows,
                               const float* penalties, int total, cudaStream_t stream);

// Batched greedy argmax over [batch][vocab] logits: one winner id per row into out_ids[batch].
// Sanitizes inline (non-finite -> -inf, clamp [+-80]) so the winner matches the host greedy path.
// blocked[batch] is a per-row id to exclude (EOS suppression below min_new_tokens); pass nullptr
// or -1 entries for none. For penalty rows, run sanitize + penalty first (as the top-k path does).
void launch_batched_argmax(const float* logits, int vocab, const int* blocked, int* out_ids,
                           int batch, cudaStream_t stream);

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
void launch_convert_bf16_to_fp16(const std::uint16_t* src, half* dst, int n, cudaStream_t stream);

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
void launch_store_kv_int4(const half* k, const half* v, int8_t* k_cache_i4, int8_t* v_cache_i4,
                          half* k_scales, half* v_scales, int position, int num_kv_heads,
                          int head_dim, int max_context, cudaStream_t stream);

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
void launch_attention_step_int4(const half* q, const int8_t* k_cache_i4, const int8_t* v_cache_i4,
                                const half* k_scales, const half* v_scales, half* out, int seq_len,
                                int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream,
                                float* scratch_m = nullptr, float* scratch_l = nullptr,
                                float* scratch_o = nullptr, int scratch_chunks = 0,
                                bool allow_split = false);

// launch_store_kv_quant / launch_attention_step_quant
//
// Generalized quantized-KV store and decode attention. K and V are stored per
// (token, kv_head) with one fp16 absmax scale each, at k_bits / v_bits per
// element (4 = packed signed nibbles as in the int4 path, 8 = signed bytes).
// Supported (k_bits, v_bits) combinations: (4,4), (8,4), (8,8).
//
// rotate_k applies a head_dim Walsh-Hadamard transform to K before quantizing
// (QuaRot R3); the attention launcher then applies the same transform to Q at
// read time so scores are computed in the rotated basis. Requires
// head_dim == 128 and is only meaningful with k_bits == 4; other combinations
// ignore the flag. The rotation is unrandomized (no sign vector), matching the
// validated quality-harness recipe.
//
// Cache row strides: K = head_dim * k_bits / 8 bytes, V = head_dim * v_bits / 8
// bytes per (token, kv_head). Scale tables are [max_context, num_kv_heads] fp16.
//
// Sink / recent-window fp16 override (quality: keeps attention sinks and the
// newest tokens exact): when k_sink/v_sink are non-null the store kernel also
// writes the (rotated) fp16 K and raw fp16 V of absolute positions < sink_n
// into them ([sink_n, num_kv_heads, head_dim]); when k_ring/v_ring are
// non-null every position is also written to ring slot position % win_n
// ([win_n, num_kv_heads, head_dim]). The attention kernels read these fp16
// buffers for sink tokens and for tokens younger than win_n, and the
// quantized cache otherwise. attn_start is the absolute position of the first
// cached token the (possibly offset) cache pointers refer to. Pass nulls and
// zeros to disable.
void launch_store_kv_quant(const half* k, const half* v, int8_t* k_cache, int8_t* v_cache,
                           half* k_scales, half* v_scales, half* k_sink, half* v_sink,
                           half* k_ring, half* v_ring, int sink_n, int win_n, int position,
                           int num_kv_heads, int head_dim, int max_context, int k_bits,
                           int v_bits, bool rotate_k, cudaStream_t stream);

void launch_attention_step_quant(const half* q, const int8_t* k_cache, const int8_t* v_cache,
                                 const half* k_scales, const half* v_scales, const half* k_sink,
                                 const half* v_sink, const half* k_ring, const half* v_ring,
                                 int sink_n, int win_n, int attn_start, half* out, int seq_len,
                                 int num_heads, int num_kv_heads, int head_dim, int k_bits,
                                 int v_bits, bool rotate_k, cudaStream_t stream,
                                 float* scratch_m = nullptr, float* scratch_l = nullptr,
                                 float* scratch_o = nullptr, int scratch_chunks = 0,
                                 bool allow_split = false);

// Quantized paged-pool KV (kernels_paged_quant.cu). Pool layout mirrors the
// fp16 paged pool: flat (num_blocks * block_size) physical token slots per
// layer, row stride head_dim * bits / 8 bytes per (slot, kv_head), parallel
// [slots, kv_heads] fp16 scale array. Same quant recipe as the contiguous
// quantized cache; rotate_k stores K Hadamard-rotated and the attention
// launcher rotates Q to match.

// fp16 sink/window quality tier for the paged pool: per-sequence stable
// quality slots in side buffers [slots, sink_n | win_n, kv_heads, head_dim]
// (one set per layer; the caller passes layer-resolved bases). The store
// kernels dual-write (rotated) fp16 K and raw fp16 V for sink positions
// (< sink_n) and into ring slot position % win_n; the batched decode
// attention reads those instead of the quantized pool for sink and
// recent-window tokens. slot_ids is a device array [batch]; the
// single-sequence prefill store takes slot-resolved pointers directly and
// only ring-writes each launch's tail win_n positions (residue-race rule).
// Pass nulls / zeros to disable.

// Scatter + quantize `rows` contiguous KV rows at positions
// base_pos..base_pos+rows-1 through `block_table`. src_stride is in halves
// (pass kv_hidden, or the fused-QKV row stride to read K/V in place).
void launch_store_kv_paged_quant(const half* k_src, const half* v_src, int src_stride,
                                 int8_t* k_pool, int8_t* v_pool, half* k_scales, half* v_scales,
                                 const int* block_table, int base_pos, int rows, int num_kv_heads,
                                 int head_dim, int block_size, int k_bits, int v_bits,
                                 bool rotate_k, cudaStream_t stream, half* k_sink = nullptr,
                                 half* v_sink = nullptr, half* k_ring = nullptr,
                                 half* v_ring = nullptr, int sink_n = 0, int win_n = 0);

// Batched decode scatter: one token per sequence via per-sequence block tables
// and positions (device arrays, [batch] and [batch * max_blocks]).
void launch_store_kv_batched_paged_quant(const half* k_src, const half* v_src, int8_t* k_pool,
                                         int8_t* v_pool, half* k_scales, half* v_scales,
                                         const int* block_tables, const int* positions,
                                         int max_blocks, int batch, int num_kv_heads,
                                         int head_dim, int block_size, int k_bits, int v_bits,
                                         bool rotate_k, cudaStream_t stream,
                                         const int* slot_ids = nullptr, half* k_sink = nullptr,
                                         half* v_sink = nullptr, half* k_ring = nullptr,
                                         half* v_ring = nullptr, int sink_n = 0, int win_n = 0);

// Chunked prefill attention over the quantized paged pool (causal, gather via
// the block table); the quant sibling of launch_attention_prefill_paged.
// k_src/v_src (the chunk's own fp16 K/V, row stride src_stride halves) and
// the slot-resolved fp16 sink buffers enable exact reads for this chunk's
// positions and the sink; pass nulls for pure quantized reads.
void launch_attention_prefill_paged_quant(const half* q, const int8_t* k_pool,
                                          const int8_t* v_pool, const half* k_scales,
                                          const half* v_scales, const int* block_table,
                                          half* out, int num_tokens, int start_position,
                                          int num_heads, int num_kv_heads, int head_dim,
                                          int block_size, int k_bits, int v_bits, bool rotate_k,
                                          cudaStream_t stream, const half* k_src = nullptr,
                                          const half* v_src = nullptr, int src_stride = 0,
                                          const half* k_sink = nullptr,
                                          const half* v_sink = nullptr, int sink_n = 0);

// Eligibility for the quantized batched paged attention (GQA-shared tier is
// the only implementation): head_dim 128, real GQA group, block_size <= 32,
// group_size * 32 >= head_dim (Q rotation needs one thread per element).
bool paged_quant_attention_supported(int num_heads, int num_kv_heads, int head_dim,
                                     int block_size);

// Batched split-K decode attention over the quantized paged pool (GQA-shared,
// one grid block per (kv_head, paged block, sequence)). Scratch layout and
// semantics match launch_attention_step_batched_paged.
void launch_attention_step_batched_paged_quant(
    const half* q, const int8_t* k_pool, const int8_t* v_pool, const half* k_scales,
    const half* v_scales, const int* block_tables, const int* seq_lens, int max_blocks,
    int max_seq_len, half* out, int batch, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int k_bits, int v_bits, bool rotate_k, cudaStream_t stream, float* scratch_m,
    float* scratch_l, float* scratch_o, int scratch_chunks, const int* slot_ids = nullptr,
    const half* k_sink = nullptr, const half* v_sink = nullptr, const half* k_ring = nullptr,
    const half* v_ring = nullptr, int sink_n = 0, int win_n = 0);

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
void launch_hadamard_rotate_fp16(half* x, const int8_t* signs, int n, int block_size,
                                 cudaStream_t stream);

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
void launch_tq3_gemv_f16(const uint32_t* w_packed, const half* codebook, const half* scales,
                         const half* x, half* y, int out_features, int in_features,
                         cudaStream_t stream);

// Builds packed sign bits for projected coordinates of x:
//   bit[j] = sign(signs[j] * x[indices[j]]) >= 0
// Output is packed little-endian in uint32 words.
void launch_tq_qjl_pack_sign_bits(const half* x, const int32_t* indices, const int8_t* signs,
                                  uint32_t* out_bits, int qjl_dim, cudaStream_t stream);

// Adds residual 1-bit correction (Qprod stage-B style) to y:
//   y[row] += scales[row] * corr(sign(row_bits), sign(x_bits))
// where corr is normalized sign-agreement in [-1, 1].
void launch_tq_qjl_residual_add_f16(const uint32_t* row_bits, const half* scales,
                                    const uint32_t* x_bits, half* y, int out_features, int qjl_dim,
                                    cudaStream_t stream);

}  // namespace kernels
