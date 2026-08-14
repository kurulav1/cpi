#pragma once

// LlamaEngine: CUDA-accelerated inference engine for LLaMA-family models.
//
// This header exposes the primary public interface together with the supporting
// configuration and statistics types needed to drive a full inference session.
//
// Typical usage:
//   1. Fill in an EngineOptions struct with the model path and runtime knobs.
//   2. Construct a LlamaEngine and call initialize(options).
//   3. Encode a prompt with model::Tokenizer, then call generate() or
//      generate_stream() to produce output token IDs.
//   4. Decode the returned IDs back to text with model::Tokenizer.
//
// The engine owns all CUDA resources (streams, events, device buffers, cuBLAS
// handles) and releases them in its destructor.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/batch_scheduler.hpp"
#include "engine/engine_types.hpp"
#include "engine/generation_constraints.hpp"
#include "engine/paged_kv.hpp"
#include "model/weight_loader.hpp"

namespace engine {

// Cache key that uniquely identifies a cublasLt matmul plan by its shape and
// output data type.  Used to look up pre-built plans in lt_plan_cache_.
struct LtMatmulPlanKey {
  int out_features = 0;                     // Number of output rows (M dimension).
  int in_features = 0;                      // Number of input columns (K dimension).
  int batch_size = 1;                       // Batch count for batched matmul.
  cudaDataType_t output_type = CUDA_R_16F;  // cuBLAS data type of the output matrix.

  // Returns true when all fields of this key are equal to other's fields.
  bool matches(const LtMatmulPlanKey& other) const {
    return out_features == other.out_features && in_features == other.in_features &&
           batch_size == other.batch_size && output_type == other.output_type;
  }
};

// A fully constructed cublasLt matmul plan together with its identifying key.
// Plans are created once and stored in lt_plan_cache_ to avoid repeated
// heuristic queries to the cuBLAS library.
struct LtMatmulPlan {
  LtMatmulPlanKey key{};                        // Shape/type key used to look up this plan.
  cublasLtMatmulDesc_t op_desc = nullptr;       // cuBLASLt operation descriptor.
  cublasLtMatrixLayout_t a_desc = nullptr;      // Layout descriptor for the A (weight) matrix.
  cublasLtMatrixLayout_t b_desc = nullptr;      // Layout descriptor for the B (activation) matrix.
  cublasLtMatrixLayout_t c_desc = nullptr;      // Layout descriptor for the C (output) matrix.
  cublasLtMatmulHeuristicResult_t heuristic{};  // Algorithm selected by the cuBLASLt heuristic.
  bool ready = false;                           // True once the plan has been fully initialised.
};

// Main inference engine for LLaMA-family models.
//
// LlamaEngine manages the full lifecycle of a CUDA-accelerated forward pass:
// weight loading, KV-cache allocation, prefill, and autoregressive decode.
// It is not copyable or movable; construct exactly one instance per GPU.
class LlamaEngine {
  // Speculative decoding drives a draft + target engine pair through their
  // internal decode/prefill/verify methods.
  // SpeculativeDecoder is a template now (one algorithm over both engines), so the friend
  // declaration has to be too. `friend class SpeculativeDecoder;` also implicitly declared it as
  // a non-template here, which then collided with the real definition; MSVC: "class template
  // has already been declared as a non-class template". Metal never saw this because it does not
  // include llama_engine.hpp.
  template <typename EngineT>
  friend class SpeculativeDecoder;

public:
  ~LlamaEngine();

  // Loads weights, allocates CUDA buffers, and warms up cuBLAS plans.
  // Must be called before any other method.  Throws on CUDA or I/O errors.
  void initialize(const EngineOptions& options);

  // Runs a full prefill + decode session and returns all generated token IDs.
  // Blocks until max_new_tokens have been produced or an EOS token is sampled.
  std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                            float temperature);

  // Streaming variant of generate().  Calls on_token(token_id) after each
  // newly generated token.  Generation stops when on_token returns false,
  // when max_new_tokens is reached, or when EOS is sampled.
  // Returns the full sequence of generated token IDs.
  std::vector<int> generate_stream(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                   float temperature, const std::function<bool(int)>& on_token,
                                   const GenerationConstraints* constraints = nullptr);

  // Runs a forward pass over prompt_tokens and returns the top_k (token_id,
  // logit_value) pairs from the next-token distribution.  Useful for
  // debugging and probability inspection without committing to a sample.
  std::vector<std::pair<int, float>> inspect_next_logits(const std::vector<int>& prompt_tokens,
                                                         int top_k);

  // Runs a numerical parity check of the GPU decode forward against an
  // independent CPU-reference forward for prompt_tokens, printing max_abs_diff and
  // the top-token match. Returns true when the GPU and CPU argmax agree and the
  // max logit diff is within tolerance, i.e. a pass/fail gate for verifying that
  // a forward-path change (e.g. kernel fusion) preserved correctness.
  bool run_parity_check(const std::vector<int>& prompt_tokens);

  // Returns the BenchmarkStats collected during the most recent generate() or
  // generate_stream() call.
  const BenchmarkStats& last_benchmark_stats() const {
    return last_benchmark_stats_;
  }

private:
  // Weight pointers for a single transformer layer that are kept resident on
  // the GPU device.  All pointers are untyped (void*) because the concrete
  // element type (fp16 or fp32) depends on the runtime dtype setting.
  // A weight left in its container's k-quant blocks instead of expanded to fp16.
  // Populated only when the model came from a GGUF and the packed path is
  // enabled; `data` null means this matrix is fp16 as before.
  struct PackedWeight {
    void* data = nullptr;
    int kind = -1;  // kernels::KQuantType
    int rows = 0;
    int cols = 0;
    [[nodiscard]] bool active() const {
      return data != nullptr && kind >= 0;
    }
  };

  struct LayerDeviceWeights {
    void* wqkv = nullptr;           // Fused QKV projection weight matrix.
    void* wo = nullptr;             // Output (post-attention) projection weight matrix.
    void* bo = nullptr;             // Optional output-projection bias [hidden].
    void* w13 = nullptr;            // Fused gate (w1) and up-projection (w3) weight for SwiGLU.
    void* w2 = nullptr;             // Down-projection weight matrix for the MLP block.
    void* norm_att = nullptr;       // RMSNorm scale vector applied before the attention block.
    void* norm_ffn = nullptr;       // RMSNorm scale vector applied before the FFN/MLP block.
    void* norm_att_bias = nullptr;  // Optional pre-attention norm bias [hidden].
    void* norm_ffn_bias = nullptr;  // Optional pre-FFN norm bias [hidden].
    void* bqkv = nullptr;  // Fused QKV bias vector [q_dim + kv_dim + kv_dim]; null if unused.
    void* q_norm =
        nullptr;  // Per-head RMSNorm scale for Q [head_dim]; null unless has_qk_norm (Qwen3).
    void* k_norm =
        nullptr;  // Per-head RMSNorm scale for K [head_dim]; null unless has_qk_norm (Qwen3).
    // Packed k-quant forms, used instead of the fp16 buffer above when active.
    // Only the matrices the container can serve packed are populated: Q/K carry
    // a RoPE permutation undone during host unpacking, and CPI's QKV is fused
    // while GGUF stores it split, so those stay fp16 for now.
    PackedWeight wqkv_packed;
    // Fallback when Q, K and V disagree on quant type and cannot share one
    // buffer: three separate matvecs into the fused output's row ranges.
    PackedWeight wq_packed;
    PackedWeight wk_packed;
    PackedWeight wv_packed;
    PackedWeight w2_packed;
    PackedWeight wo_packed;
    // gate and up concatenated; legal only because a k-quant row is a whole
    // number of super-blocks, so fusing rows is a byte append.
    PackedWeight w13_packed;
  };

  // Weight pointers for a single transformer layer stored in host page-locked
  // (pinned) memory.  QKV is stored split (separate q, k, v matrices) so that
  // streaming transfers can overlap with compute on the other stream.
  struct LayerHostPinnedWeights {
    void* wq = nullptr;             // Query projection weight matrix (host-pinned copy).
    void* wk = nullptr;             // Key projection weight matrix (host-pinned copy).
    void* wv = nullptr;             // Value projection weight matrix (host-pinned copy).
    void* wo = nullptr;             // Output projection weight matrix (host-pinned copy).
    void* bo = nullptr;             // Optional output-projection bias (host-pinned copy).
    void* w1 = nullptr;             // Gate projection weight matrix, SwiGLU (host-pinned copy).
    void* w2 = nullptr;             // Down-projection weight matrix (host-pinned copy).
    void* w3 = nullptr;             // Up-projection weight matrix, SwiGLU (host-pinned copy).
    void* norm_att = nullptr;       // Attention RMSNorm scale (host-pinned copy).
    void* norm_ffn = nullptr;       // FFN RMSNorm scale (host-pinned copy).
    void* norm_att_bias = nullptr;  // Optional attention norm bias (host-pinned copy).
    void* norm_ffn_bias = nullptr;  // Optional FFN norm bias (host-pinned copy).
    void* bqkv = nullptr;           // Fused QKV bias (host-pinned copy); null if unused.
  };

  // INT8-quantised MLP weights stored in host memory.
  // s_w* arrays hold per-row quantisation scales used to dequantise the
  // corresponding INT8 weight matrix during the forward pass.
  struct LayerHostInt8Weights {
    std::int8_t* w1 = nullptr;  // INT8 gate projection weight matrix.
    std::int8_t* w2 = nullptr;  // INT8 down-projection weight matrix.
    std::int8_t* w3 = nullptr;  // INT8 up-projection weight matrix.
    float* s_w1 = nullptr;      // Per-row dequantisation scales for w1.
    float* s_w2 = nullptr;      // Per-row dequantisation scales for w2.
    float* s_w3 = nullptr;      // Per-row dequantisation scales for w3.
  };

  // INT8-quantised MLP weights resident on the GPU device.
  // Layout mirrors LayerHostInt8Weights; pointers address device memory.
  struct LayerDeviceInt8Weights {
    std::int8_t* w1 = nullptr;  // Device INT8 gate projection weight matrix.
    std::int8_t* w2 = nullptr;  // Device INT8 down-projection weight matrix.
    std::int8_t* w3 = nullptr;  // Device INT8 up-projection weight matrix.
    bool mlp_int4 = false;      // True when w1/w2/w3 are packed INT4 (two signed nibbles per byte).
    int mlp_group = 0;          // >0: s_w1/s_w2/s_w3 hold one scale per group of N input columns
                                // (group-wise int4). 0: one scale per row (legacy). Self-describing
                                // so streaming (per-row) and resident (grouped) buffers can't confuse
                                // a consumer.
    float* s_w1 = nullptr;      // Device dequantisation scales for w1 ([rows] or [rows,n_groups]).
    float* s_w2 = nullptr;      // Device per-row dequantisation scales for w2.
    float* s_w3 = nullptr;      // Device per-row dequantisation scales for w3.
    // Projection weights quantised at layer-cache init time (null when unused).
    std::int8_t* wqkv = nullptr;  // Device INT8 fused QKV weight matrix.
    bool proj_int4 = false;     // True when wqkv/wo are packed INT4 (two signed nibbles per byte).
    float* s_wqkv = nullptr;    // Device per-row dequantisation scales for wqkv.
    std::int8_t* wo = nullptr;  // Device INT8 output projection weight matrix.
    float* s_wo = nullptr;      // Device per-row dequantisation scales for wo.
  };

  // Copies the embedding table, final RMSNorm, and LM-head weights from the
  // weight file into device memory.  Called once during initialize().
  void load_static_weights();

  // Allocates all per-token and per-layer device scratch buffers sized
  // according to options_ and the model config.
  void allocate_runtime_buffers();

  // Copies the configured number of layer weights into the persistent GPU
  // layer cache (layer_cache_ / layer_cache_i8_).
  void init_layer_cache();
  bool tq_cached_preflight_layers(int layers, std::string* reason) const;

  // Initialises layer_host_pinned_ with page-locked host copies of all
  // layers that are not kept in the GPU cache (streaming layers).
  void init_uncached_pinned_host_weights();

  // Initialises layer_host_int8_ with INT8-quantised host copies of the
  // MLP weights for all streaming layers.
  void init_uncached_int8_host_weights();

  // Transfers the weights for layer into the destination device structs dst
  // and dst_i8, issuing asynchronous copies on stream.
  void copy_layer_weights_to_device(int layer, LayerDeviceWeights* dst,
                                    LayerDeviceInt8Weights* dst_i8, cudaStream_t stream);

  // Zeros out the KV cache device and host buffers to start a fresh sequence.
  void reset_kv_cache();

  // Benchmarks the custom projection kernels to choose optimal tile/warp
  // parameters for resident (cached) layers.
  void tune_resident_projection_backends();

  // Executes the fused INT8 w1/w3 (gate + up) MLP projection for a resident
  // layer.  inter and hidden are the intermediate and hidden dimensions.
  void resident_int8_mlp_w13(const LayerDeviceInt8Weights& lw_i8, int inter, int hidden);

  // Executes the INT8 w2 (down) MLP projection for a resident layer.
  void resident_int8_mlp_w2(const LayerDeviceInt8Weights& lw_i8, int hidden, int inter);

  // Group-wise int4 MLP matvec over `rows` tokens against a pre-quantized perm8-g32 activation
  // (xq + x_scales). rows==1 -> single-token grouped_dp4a; rows>1 -> grouped_dp4a_mt looped in
  // batches of 8 (its per-launch token cap), weights streamed once per batch. Same dp4a speed as
  // the per-row path, with the group-wise weight-scale quality.
  void mlp_int4_grouped_dp4a(const std::int8_t* w, const float* scales, const std::int8_t* xq,
                             const float* x_scales, __half* y, int rows, int out_features,
                             int in_features, int group);

  // Processes the prompt token-by-token (sequential prefill) without using
  // chunked attention.  Used as a fallback when the prompt fits in a single
  // chunk or when split-attention is disabled. start_pos skips a leading prefix
  // whose KV is already resident (prefix reuse).
  void prefill_prompt_sequential(const std::vector<int>& prompt_tokens, int start_pos = 0);

  // Processes the prompt using chunked attention to support long sequences
  // efficiently, then falls back to sequential for the tail. start_pos skips a
  // shared prefix whose KV is already resident (prefix reuse).
  void prefill_prompt(const std::vector<int>& prompt_tokens, int start_pos = 0);

  // Runs `rows` tokens (already uploaded to d_token_id_) through the full
  // transformer in one batched pass at absolute positions
  // [base_pos, base_pos+rows), writing their K/V into the cache. Shared body of
  // prefill_prompt (per chunk) and verify_tokens (speculative decoding).
  void run_batched_chunk(int rows, int base_pos);

  // Speculative-decoding verify: runs the K `tokens` through the full model in
  // one batched pass at absolute positions [start_pos, start_pos+K), writes
  // their K/V into the cache, and fills out_argmax[i] with the greedy (argmax)
  // next token at position start_pos+i. Requires the batched full-attention
  // path; K must be <= prefill_chunk_size_.
  void verify_tokens(const std::vector<int>& tokens, int start_pos, std::vector<int>& out_argmax);

  // Returns true if the greedy-decode CUDA graph can be used for the current
  // engine state (e.g. all layers cached, temperature == 0 implied by caller).
  bool can_use_greedy_decode_graph() const;

  // Destroys the previously captured greedy-decode CUDA graph and resets the
  // associated state flags.
  void destroy_greedy_decode_graph();

  // Captures a CUDA graph of the greedy-decode forward pass for the current
  // model configuration to amortise kernel launch overhead in subsequent steps.
  void init_greedy_decode_graph();

  // Replays the greedy-decode CUDA graph for (token, position) and returns
  // the argmax token ID.  Faster than forward_token() for single-token steps.
  int decode_next_token_graph(int token, int position);

  // Destroys the logits-decode CUDA graph and resets associated state flags.
  void destroy_logits_decode_graph();

  // Captures a CUDA graph of the transformer body + LM head (no argmax) for
  // use by the sampling path to eliminate kernel launch overhead.
  void init_logits_decode_graph();

  // Replays the logits-decode CUDA graph and copies the full logit vector to h_logits.
  void decode_next_token_logits_graph(int token, int position, std::vector<float>& h_logits);

  // Tensor-core prefill attention: Q.K^T and P.V as cuBLAS batched GEMMs, with a masked softmax
  // between, over chunks of query rows. attention_prefill_kernel_tiled runs on the plain FMA
  // pipe; routing these two matmuls through cuBLAS puts them on the tensor cores.
  //
  // Returns false when not eligible (unsupported geometry, or the score matrix would be too
  // large), in which case the caller must fall back to the kernel. Only used for plain causal
  // text prefill: the vision/bidirectional paths need a per-token `limits` or sliding `window`
  // mask, which only the kernel implements.
  bool prefill_attention_tensorcore(const void* q, const void* k_layer, const void* v_layer,
                                    void* out, int rows, int base_pos, int num_heads,
                                    int num_kv_heads, int head_dim, int q_stride);

  // Eligibility + scratch allocation for the above, decided once per chunk. The caller needs the
  // answer before the layer loop: if it skips the Q/K/V split copies and the tensor-core path
  // then declined mid-loop, there would be no contiguous Q left to fall back to.
  bool prefill_tc_prepare(int rows, int base_pos, int num_heads, int num_kv_heads, int head_dim);

  // Runs the logits decode graph and leaves the logits on the device. Returns false if the
  // graph is unavailable (caller must fall back). This is the half of
  // decode_next_token_logits_graph that does not pay for the D2H copy.
  bool run_logits_decode_graph(int token, int position);

  // Device-side top-k sampling. The host-logits path copies the whole vocab to the host
  // every token, 608 KB for Qwen2.5's 151936 logits, and then sorts it there. Measured
  // at 0.73 ms/token, which is 45% of a 0.5B decode step and dwarfs every kernel in it.
  // Greedy (temp<=0) already argmaxes on-device and avoids this; temperature>0 is the path
  // real chat actually takes, and it was paying full price.
  //
  // Selects the candidate set on the GPU so the host only ever sees ~k entries. Returns
  // false when not eligible or when a pathological tie count overflows the buffer, in which
  // case the caller must use the host path. Mirrors PlanCudaEngine, which already does this.
  bool decode_next_token_device_topk(int token, int position, float temperature,
                                     const std::vector<int>& history, int* out_token);
  void ensure_device_topk_buffers();

  // Runs a full forward pass for a single token at position, writes the
  // full logit vector to *out_logits, and the argmax index to *out_argmax.
  void forward_token_logits(int token, int position, std::vector<float>* out_logits,
                            int* out_argmax);

  // Runs a full forward pass for a single token at position.
  // If compute_logits is true, results are written to out_logits/out_argmax.
  void forward_token(int token, int position, bool compute_logits, std::vector<float>* out_logits,
                     int* out_argmax);
  void forward_decode_layers(int token, int position);

  // Samples the next token ID given the current token at position, applying
  // temperature, top-k/p filtering, repetition penalty, and the no-repeat
  // n-gram constraint using history as the previously generated context.
  int decode_next_token(int token, int position, float temperature,
                        const std::vector<int>& history);

  // Greedy decode returning the top-1 argmax and, via `second`, the top-2 token.
  // Uses the host-logits forward (advances KV like decode_next_token). Used by
  // the speculative decoder's tree-opportunity probe.
  int decode_next_token2(int token, int position, int* second);

  // Launches the custom half-precision (FP16) projection kernel for a
  // weight matrix w of shape [out_features x in_features] applied to
  // activation vector x, writing the result to y.
  // warps_per_block, tile_pairs, and rows_per_warp tune the kernel launch.
  void resident_projection_half(const void* w, const void* x, void* y, int out_features,
                                int in_features, int warps_per_block = 0, int tile_pairs = 0,
                                int rows_per_warp = 1);
  // Same projection with the residual add folded into the epilogue, so the separate
  // add_inplace launch disappears. At batch 1 a kernel costs a fixed ~1.7 us whatever it
  // does, and adding `hidden` elements is far cheaper than launching a kernel to do it.
  void resident_projection_half_residual(const void* w, const void* x, void* residual,
                                         int out_features, int in_features, int warps_per_block = 0,
                                         int tile_pairs = 0, int rows_per_warp = 1);

  // Float32 variant of resident_projection_half() for models using fp32 weights.
  void resident_projection_float(const void* w, const void* x, void* y, int out_features,
                                 int in_features, int warps_per_block = 0, int tile_pairs = 0,
                                 int rows_per_warp = 1);

  // Single-sequence LM-head projection: x_norm[hidden] -> logits[vocab] (fp32). Uses the resident
  // int8 head (d_lm_head_i8_) when lm_head_int8_ is set; near-lossless and lets the fp16 head be
  // freed under single-sequence deployment; else the fp16 head. Bias is added by the caller.
  void project_lm_head_logits(const __half* x_norm, float* logits);

  // Applies either RMSNorm or true LayerNorm based on model config.
  // Optional bias is applied in-kernel for LayerNorm or as a post-add for RMSNorm.
  void launch_norm(const void* x, const void* weight, const void* bias, void* y, int rows,
                   int cols);

  // Adds an optional fp16 bias vector to a fp16 output tensor (1-D or row-broadcasted).
  void maybe_add_half_bias(void* out, const void* bias, int rows, int cols);

  // Enforces host CPU/RAM guardrails with "throttle then abort" semantics.
  void enforce_host_resource_limits(const char* stage);

  // Throws if cached TurboQuant initialisation exceeds the configured timeout.
  void check_tq_cached_init_timeout(const std::chrono::steady_clock::time_point& start,
                                    int layer_index);

  EngineOptions options_{};  // Runtime configuration supplied by the caller.
  // Active grammar for the in-flight generate_stream call, or null. Set at entry
  // and cleared on exit; read by decode_next_token to mask logits.
  grammar::GrammarSampler* active_grammar_ = nullptr;
  // When true, decode_next_token masks the EOS logit so it cannot be sampled
  // (min_new_tokens). Set per-step by the generate_stream decode loop.
  bool suppress_eos_ = false;
  // Tokens whose KV is currently resident in the cache (KV[0, size) valid for
  // exactly these tokens). Lets generate_stream reuse a shared prompt prefix and
  // prefill only the divergent tail. Cleared by reset_kv_cache (KV wiped).
  std::vector<int> resident_prefix_;
  // P3 paged KV (Phase 2a): block allocator + current sequence's block table.
  // Null unless options_.paged_blocks. Phase 2a uses contiguous blocks (the KV
  // cache is rounded to a whole number of blocks), so addressing is unchanged and
  // output is byte-identical; the block table validates position->block mapping
  // and is the hook for the non-contiguous gather kernel (Phase 2b).
  std::unique_ptr<BlockAllocator> block_alloc_;
  std::unique_ptr<SequenceBlockTable> seq_blocks_;
  int* d_block_table_ = nullptr;  // device: logical chunk -> physical block, for paged attention
  std::vector<int> block_table_host_;  // host mirror, for host-side paged KV-write addressing
  // P2 batched decode: per-sequence device state (positions, seq_lens, block tables).
  int* d_batch_positions_ = nullptr;
  int* d_batch_seq_lens_ = nullptr;
  int* d_batch_block_tables_ = nullptr;
  int* d_batch_kv_slots_ = nullptr;  // [batch] stable quality-slot ids (quant sink/window tier)
  int batch_buffers_max_seqs_ = 0;
  int batch_buffers_max_blocks_ = 0;
  // Quality-slot capacity of the fp16 sink/ring side buffers (d_kv_sink_/d_kv_ring_,
  // layout [layers, slots, sink_n | win_n, kv_heads, head_dim]). The contiguous
  // single-sequence cache uses exactly slot 0; the batched paged tier grows this on
  // demand as the scheduler hands out higher slot ids.
  int kv_quality_slots_ = 1;
  void ensure_kv_quality_slots(int min_slots);
  // Quality slot the next paged quant prefill writes into (set by the batch adapter
  // around prefill_suffix; 0 for the single-sequence path).
  int prefill_kv_slot_ = 0;
  // Actual element capacity of d_batch_block_tables_ (batch*max_blocks ints). The
  // batch size can shrink then regrow (e.g. a preempted request resuming), so this
  // must track the real allocation, not the product of independent high-watermarks.
  std::size_t batch_buffers_block_table_cap_ = 0;
  void ensure_batch_state_buffers(int batch, int max_blocks);

public:
  // One decode step for a batch of sequences (P2). tokens[b] is sequence b's
  // current token at positions[b]; block_tables_flat is [batch][max_blocks]
  // (logical chunk -> physical block). Returns each sequence's next (greedy) token.
  // fp16 resident (--gpu-cache-all), full-attention path only.
  // kv_slots (all decode_step_batched* methods): per-row stable quality-slot ids for
  // the quant sink/window tier; empty means "all rows slot 0" (single-sequence and
  // duplicate-row check callers). Ignored when the tier is off.
  std::vector<int> decode_step_batched(const std::vector<int>& tokens,
                                       const std::vector<int>& positions,
                                       const std::vector<int>& block_tables_flat, int max_blocks,
                                       const std::vector<int>& kv_slots = std::vector<int>());

  // Same forward as decode_step_batched but returns per-row logits [batch][vocab]
  // so each sequence can be sampled with its own params/grammar (streaming batcher).
  void decode_step_batched_logits(const std::vector<int>& tokens, const std::vector<int>& positions,
                                  const std::vector<int>& block_tables_flat, int max_blocks,
                                  std::vector<std::vector<float>>& out_logits,
                                  const std::vector<int>& kv_slots = std::vector<int>());

  // Device top-k fast path for the streaming batcher: same forward as decode_step_batched_logits,
  // then a per-row device top-k + gather so only each row's ~k candidates cross to the host (not
  // the full vocab). out_cand[b] is the row's candidate set, index-sorted to match the host
  // sampler. Returns false (caller falls back to the logits path) when k is out of the device
  // top-k range or a row hits a pathological tie count.
  bool decode_step_batched_topk(const std::vector<int>& tokens, const std::vector<int>& positions,
                                const std::vector<int>& block_tables_flat, int max_blocks,
                                const BatchTopkParams& sp,
                                std::vector<std::vector<detail::SampleCandidate>>& out_cand,
                                const std::vector<int>& kv_slots = std::vector<int>());

  // Greedy device argmax over a batched decode step: returns each row's winner id in out_ids
  // (resized to batch) via a device reduction, sanitizing + applying repetition penalty on-device
  // so the winners match the host greedy path. Returns false only if the batch is empty.
  bool decode_step_batched_argmax(const std::vector<int>& tokens, const std::vector<int>& positions,
                                  const std::vector<int>& block_tables_flat, int max_blocks,
                                  const BatchArgmaxParams& ap, std::vector<int>& out_ids,
                                  const std::vector<int>& kv_slots = std::vector<int>());

  // Parity gate for decode_step_batched: prefill `prompt_tokens`, then for
  // `num_steps` decode steps compare the batched path (N=1 and N=2 duplicate
  // rows) against the single-token path token-for-token. Prints pass/FAIL.
  void run_batched_decode_check(const std::vector<int>& prompt_tokens, int num_steps);

  // One request in a concurrently-scheduled batch.
  struct BatchRequest {
    std::vector<int> prompt;
    int max_new_tokens = 0;
    int eos_id = -1;           // stop when this token is produced (-1 = run to max_new_tokens)
    float temperature = 0.0f;  // 0 = greedy
  };

  // Iteration-level scheduler: prefill every request into its own paged block
  // table (non-contiguous physical blocks from the shared pool), then decode all
  // running sequences together one step at a time via decode_step_batched,
  // freeing a sequence's blocks when it hits eos / max_new_tokens. Ragged
  // lengths handled per-sequence. Greedy. Returns each request's generated
  // tokens. Blocking; the streaming server wrapper comes on top of this.
  // Requires --paged-blocks + --gpu-cache-all; the concurrent sequences' total
  // length must fit the max_context KV budget.
  std::vector<std::vector<int>> run_batch(const std::vector<BatchRequest>& requests);

  // Parity gate for the scheduler: builds several distinct-length sequences from
  // `base_prompt`, generates them concurrently via run_batch and each alone via
  // the single-sequence path, and compares token-for-token. Prints pass/FAIL.
  void run_scheduler_check(const std::vector<int>& base_prompt, int max_new, int eos_id);

  // Throughput benchmark: sweeps batch sizes, comparing serial single-sequence
  // generation against concurrent run_batch (decode tokens/sec + speedup).
  void run_batch_bench(const std::vector<int>& prompt, int max_new);

  // Coordinate descent over the k-quant kernel knobs on this box, timed on real
  // batched decode. See the definition for why it re-times the incumbent.
  void tune_kquant_knobs(int batch, int steps);

  // ---- Streaming batch scheduler (continuous batching for the server) --------
  //
  // The scheduler itself now lives in engine::BatchScheduler, which has no backend in it:
  // admission, preemption, block growth, the shared-prefix LRU and per-request sampling are
  // host bookkeeping, and it reaches a GPU through two virtual calls. It sat inside this class
  // for no better reason than that, which is what made continuous batching CUDA-only. The
  // methods below are thin delegations, kept so callers do not have to change and so the
  // engine still owns the backend half (prefill + batched decode).
  //
  // The types are aliases rather than definitions: LlamaEngine::StreamParams and
  // engine::StreamParams must be the same type, or the app layer could not be handed either
  // backend's scheduler.
  using StreamParams = engine::StreamParams;
  using StreamEvent = engine::StreamEvent;

  // Admit a request into the running batch (prefills its prompt into its own
  // paged blocks). Requires --paged-blocks + --gpu-cache-all.
  void stream_admit(const std::string& id, const std::vector<int>& prompt_tokens,
                    const StreamParams& params);
  // Advance one decode step over all running requests, sampling each with its own
  // params/grammar; appends one StreamEvent per running request. Retires and
  // frees blocks for finished requests. Returns false if nothing is running.
  bool stream_step(std::vector<StreamEvent>& events);
  // Cancel a running request by id: removes it from the batch and frees its paged
  // blocks (RAII). Returns true if the id was running. Used to reclaim serving
  // capacity when a client disconnects mid-generation.
  bool stream_cancel(const std::string& id);
  // Number of requests currently running.
  int stream_active() const;

  // The scheduler driving those methods, built on first use. Exposed because the app-layer
  // batch worker now takes a BatchScheduler& directly; it never needed the engine. No
  // options argument: this engine already carries them in EngineOptions.
  BatchScheduler& batch_scheduler() {
    return ensure_scheduler();
  }

private:
  // Single-sequence greedy decode (reference for the scheduler gate).
  std::vector<int> greedy_generate_single(const std::vector<int>& prompt, int max_new, int eos_id);
  // Shared batched-decode forward (embed -> layers -> final norm); leaves per-row
  // final hidden states in d_x_norm_. Returns batch size.
  int decode_step_batched_forward(const std::vector<int>& tokens, const std::vector<int>& positions,
                                  const std::vector<int>& block_tables_flat, int max_blocks,
                                  const std::vector<int>& kv_slots);
  // Throws with a clear message if the current model/mode isn't supported by the
  // batched decode path (only plain fp16 full-attention resident is).
  void require_batched_supported() const;
  // Project all `batch` rows of d_x_norm_ through the LM head into d_batch_logits_
  // ([batch][vocab], float) in one GEMM. Lazily (re)allocates d_batch_logits_.
  void batched_lm_head(int batch, int hidden, int vocab);
  float* d_batch_logits_ = nullptr;  // [max_batch * vocab] float LM-head output
  int d_batch_logits_cap_ = 0;       // capacity in rows
  // Persistent pinned host mirror for the [batch][vocab] D2H copy. Pinned so the copy
  // runs at full PCIe bandwidth and can be async; reused so no ~batch*vocab alloc per
  // decode step. Grown on demand alongside d_batch_logits_.
  float* h_batch_logits_ = nullptr;
  int h_batch_logits_cap_ = 0;  // capacity in rows
  // Device scratch for the batched device top-k fast path's repetition penalty: per-row penalty,
  // and the flattened (id, row) pairs of seen tokens. Grown on demand.
  float* d_penalty_ = nullptr;
  int d_penalty_cap_ = 0;  // rows
  int* d_seen_ids_ = nullptr;
  int* d_seen_rows_ = nullptr;
  int d_seen_cap_ = 0;  // pairs
  // Device scratch for the greedy argmax fast path: per-row blocked id and per-row winner id.
  int* d_argmax_blocked_ = nullptr;
  int* d_argmax_out_ = nullptr;
  int d_argmax_cap_ = 0;  // rows
  // Persistent split-K attention scratch for batched decode (grown on demand).
  float* d_bs_scratch_m_ = nullptr;
  float* d_bs_scratch_l_ = nullptr;
  float* d_bs_scratch_o_ = nullptr;
  std::size_t d_bs_stat_cap_ = 0;  // capacity in stat elements (m/l); o is *head_dim
  // Optional attribution (CPI_BATCH_PROFILE): cumulative seconds in the batched
  // forward vs the LM-head/logits-copy tail.
  double prof_fwd_s_ = 0.0;
  double prof_head_s_ = 0.0;

  // A running request in the streaming batch scheduler.
  // The backend half of continuous batching: the two operations BatchScheduler needs from a
  // GPU. Defined in llama_engine_batched_decode.cpp, where the CUDA state it touches lives.
  // Held as the base pointer: BatchBackend is complete here and has a virtual destructor, so
  // ~LlamaEngine (compiled where BatchAdapter is only forward-declared) can destroy it.
  class BatchAdapter;
  std::unique_ptr<BatchBackend> batch_adapter_;
  std::unique_ptr<BatchScheduler> scheduler_;
  // Built lazily on first admit, because it needs block_alloc_, which --paged-blocks creates.
  BatchScheduler& ensure_scheduler();

public:
private:
  model::WeightLoader weights_;  // Memory-mapped weight file handle.
  int attn_q_hidden_ = 0;        // Query projection width (rows in attention.wq).
  int attn_head_dim_ = 0;        // Per-head attention width (attn_q_hidden_ / num_heads).
  int attn_kv_hidden_ = 0;       // Key/value projection width (num_kv_heads * attn_head_dim_).
  // Physical per-layer KV token capacity: the stride of one layer's K/V region
  // and thus the paged block pool size (num_blocks = kv_capacity_tokens_/bs).
  // Equals max_context normally; with --paged-blocks it is sized up to available
  // VRAM so continuous batching can hold many concurrent sequences (each still
  // bounded by max_context) rather than one max_context sequence total.
  int kv_capacity_tokens_ = 0;
  bool has_any_layer_output_bias_ = false;  // Any layer has attention.bo.
  bool has_any_layer_norm_bias_ = false;    // Any layer has norm bias tensors.

  cublasHandle_t cublas_ = nullptr;       // Legacy cuBLAS handle used for prefill GEMMs.
  cublasLtHandle_t cublas_lt_ = nullptr;  // cuBLASLt handle used for decode projections.
  void* lt_workspace_ = nullptr;          // Device workspace buffer required by cuBLASLt.
  std::size_t lt_workspace_bytes_ =
      4 * 1024 * 1024;                       // Size of the cuBLASLt workspace (default 4 MiB).
  std::vector<LtMatmulPlan> lt_plan_cache_;  // Cache of pre-built cuBLASLt matmul plans.

  // Static (sequence-independent) device weight buffers.
  void* d_tok_embeddings_ = nullptr;  // Token embedding table on device [vocab_size x hidden_size].
  void* d_norm_out_ = nullptr;        // Final RMSNorm scale vector on device.
  void* d_norm_out_bias_ = nullptr;   // Optional final norm bias [hidden].
  void* d_lm_head_ = nullptr;         // LM-head projection weight matrix on device.
  // The head in its container's k-quant blocks; when active d_lm_head_ is never
  // allocated, since the head is read whole on every token.
  PackedWeight lm_head_packed_;
  // int8 LM head (weight-only). An 8B's LM head is 1.05 GB in fp16; 22% of everything an
  // int4 8B reads per token. Built only when weight quantization is on. The fp16 copy is kept:
  // the batched-decode path drives the LM head through cuBLAS and still needs it.
  std::int8_t* d_lm_head_i8_ = nullptr;
  float* d_lm_head_i8_scales_ = nullptr;
  bool lm_head_int8_ = false;
  void* d_lm_head_bias_ = nullptr;  // Optional lm_head bias [vocab].

  // Per-step decode scratch buffers on device.
  int* d_token_id_ = nullptr;    // Single-element device buffer holding the current input token ID.
  void* d_x_ = nullptr;          // Residual stream buffer (hidden state) for the current token.
  void* d_x_norm_ = nullptr;     // RMSNorm-normalised version of d_x_.
  void* d_qkv_ = nullptr;        // Fused QKV projection output buffer.
  void* d_q_ = nullptr;          // Query slice of d_qkv_ (or separate for prefill).
  void* d_k_ = nullptr;          // Key slice.
  void* d_v_ = nullptr;          // Value slice.
  void* d_prefill_q_ = nullptr;  // Full-sequence Q buffer used during chunked prefill.
  void* d_prefill_k_ = nullptr;  // Full-sequence K buffer used during chunked prefill.
  void* d_prefill_v_ = nullptr;  // Full-sequence V buffer used during chunked prefill.
  void* d_att_ = nullptr;        // Attention output accumulation buffer.
  void* d_ff13_ = nullptr;       // Fused gate+up (w1/w3) MLP output buffer.
  void* d_ff1_ = nullptr;        // Gate activation buffer (SwiGLU first operand).
  void* d_ff2_ = nullptr;        // Post-down-projection MLP output buffer.
  void* d_prefill_ff1_ = nullptr;         // Prefill-sized gate activation buffer.
  void* d_prefill_ff2_ = nullptr;         // Prefill-sized down-projection output buffer.
  // Prefill scratch for the dequantize-then-GEMM path: one weight matrix at a
  // time in fp16, so a prompt chunk reaches the tensor-core GEMM instead of the
  // batched matvec kernels (decode-shaped, ~20x off at prefill row counts).
  void* d_prefill_wdq_ = nullptr;
  std::size_t d_prefill_wdq_bytes_ = 0;
  // Uploads a tensor's packed k-quant blocks and records them in `out`. No-op
  // when the container cannot serve this tensor packed, which leaves the fp16
  // path in charge.
  void stage_packed_weight(const std::string& name, PackedWeight* out, int mask_bit);

  // Same, for a matrix CPI keeps fused that the container stores as two
  // row-blocks. Declines unless both halves are packed the same way.
  void stage_packed_pair(const std::string& first, const std::string& second,
                         PackedWeight* out, int mask_bit);

  // Same for a matrix the container stores as three row-blocks (Q, K, V).
  void stage_packed_triple(const std::string& a, const std::string& b, const std::string& c,
                           PackedWeight* out, int mask_bit);

  bool upload_packed_rows(const model::GgufLoader::PackedTensor& pk, void* dst);

  // Load-time H2D through a small pinned ring. cudaMemcpyAsync from the pageable
  // GGUF mapping moves ~3 GB/s on Windows because the driver stages every chunk
  // through its own tiny pinned buffer; hand-staging through two 64 MB pinned
  // buffers (host memcpy into one while the other DMAs) runs the same bytes at
  // an order of magnitude faster on a warm cache. The source is fully consumed
  // by the time this returns, so callers may free it; the device copy may still
  // be in flight on transfer_stream_. Falls back to a synchronous copy when the
  // pinned allocation is unavailable. Startup-scoped: free_load_staging()
  // releases the ring once the model is resident.
  bool staged_h2d(void* dst, const void* src, std::size_t bytes);
  void free_load_staging();
  void* load_staging_buf_[2] = {nullptr, nullptr};
  void* load_staging_ev_[2] = {nullptr, nullptr};
  bool load_staging_failed_ = false;
  // CPI_STARTUP_PROFILE accounting for the ring: where the staged time goes.
  double load_staging_memcpy_ms_ = 0.0;
  double load_staging_wait_ms_ = 0.0;
  std::size_t load_staging_bytes_ = 0;

  // Single-token QKV straight off the packed blocks. False when this layer kept
  // its fp16 fused matrix, which is the caller's cue to run the old path.
  // x_pre_quantized: the rmsnorm that produced x_norm already wrote its q8_1
  // form into the shared activation scratch, so no projection needs to redo it.
  bool packed_qkv_matvec(const LayerDeviceWeights& lw, const void* x_norm, void* qkv,
                         cudaStream_t stream, bool x_pre_quantized = false);

  // Batched forms. False means the batch is outside what the packed kernel is
  // worth doing for, and the caller should expand the weight and use cuBLAS.
  // reuse_x says the caller has already had this exact activation quantized by
  // an immediately preceding matvec of the same width, as q/k/v do.
  bool packed_matmul(const PackedWeight& w, const void* x, void* y, int batch, int ldy, int row0,
                     cudaStream_t stream, bool reuse_x = false);

  // A captured batched decode step. The graph is only valid for the shapes that
  // fixed its grids, so it is keyed on them and re-captured when any changes --
  // batch size above all, which moves every grid in the step. Held as void* so
  // the header does not need the CUDA graph types.
  void* batch_graph_exec_ = nullptr;
  int batch_graph_batch_ = -1;
  int batch_graph_blocks_ = -1;
  int batch_graph_bucket_ = -1;
  void reset_batch_graph();

  // fp16 logits scratch for the tensor-core LM head, plus its gate. Enabled by
  // default: unlike the layer weights, the path it replaces is dequant-and-cuBLAS
  // rather than a good kernel. CPI_LM_HEAD_MMQ=0 disables.
  void* d_batch_logits_h_ = nullptr;
  std::size_t d_batch_logits_h_cap_ = 0;
  bool ensure_batch_logits_half(std::size_t elems);
  static bool lm_head_mmq_enabled();

  // Scratch for the dp4a path's int8 activations, grown on demand.
  bool ensure_q8_scratch(int batch, int cols);

  std::int8_t* d_q8_x_ = nullptr;
  float* d_q8_scale_ = nullptr;
  float* d_q8_sum_ = nullptr;
  std::size_t q8_x_bytes_ = 0;
  std::size_t q8_meta_bytes_ = 0;

  bool packed_qkv_matmul(const LayerDeviceWeights& lw, const void* x, void* y, int batch, int ldy,
                         cudaStream_t stream);

  const void* dequant_packed_for_gemm(const PackedWeight& w, cudaStream_t stream);

  bool ensure_prefill_wdq(std::size_t need, cudaStream_t stream);

  const void* dequant_packed_qkv_for_gemm(const LayerDeviceWeights& lw, cudaStream_t stream);

  const void* dequant_weight_for_gemm(const std::int8_t* w, const float* scales, bool int4,
                                      int rows, int cols, int group);
  std::int8_t* d_prefill_i8_ = nullptr;   // INT8 quantised activations for prefill INT8 path.
  float* d_prefill_i8_scales_ = nullptr;  // Per-row scales accompanying d_prefill_i8_.
  float* d_prefill_perm8_scales_ = nullptr;  // Per-group(32) scales for the perm8-g32 activation
                                             // quant feeding grouped int4 dp4a MLP. [rows, cols/32].
  void* d_ff3_ = nullptr;     // Up-projection (w3) output buffer (SwiGLU second operand).
  void* d_logits_ = nullptr;  // Raw logit vector output from the LM head.
  int* d_argmax_ = nullptr;   // Single-element device buffer for the argmax result.
  // Tensor-core prefill attention (see prefill_attention_tensorcore).
  void* d_attn_scores_ = nullptr;  // [heads][chunk][keys] score matrix, fp16.
  std::size_t attn_scores_bytes_ = 0;
  // Contiguous K/V staging for paged single-stream prefill: the tensor-core
  // attention needs plain [pos, kv_hidden] K/V, so the block pool is gathered
  // here once per layer per chunk (K at the base, V at the row-count offset).
  void* d_paged_gather_kv_ = nullptr;
  std::size_t paged_gather_bytes_ = 0;
  void** d_gemm_ptrs_ = nullptr;        // device pointer arrays for cublasGemmBatchedEx (6 x heads)
  std::size_t gemm_ptrs_capacity_ = 0;  // pointers, not bytes
  float* d_argmax_part_val_ = nullptr;  // Per-block partials for the two-phase greedy argmax.
  int* d_argmax_part_idx_ = nullptr;
  int argmax_parts_ = 0;
  // Device top-k sampling scratch (see decode_next_token_device_topk). Allocated lazily on
  // the first sampled token, so greedy-only runs never pay for them.
  float* d_topk_part_val_ = nullptr;
  int* d_topk_part_idx_ = nullptr;
  float* d_topk_val_ = nullptr;
  int* d_topk_idx_ = nullptr;
  int* d_cand_idx_ = nullptr;
  float* d_cand_val_ = nullptr;
  int* d_cand_count_ = nullptr;
  bool device_topk_ready_ = false;
  int* d_decode_position_ = nullptr;  // Device copy of the current decode position index.
  float* d_rope_cos_ = nullptr;       // Precomputed RoPE cosine table [max_seq_len x head_dim].
  float* d_rope_sin_ = nullptr;       // Precomputed RoPE sine table [max_seq_len x head_dim].
  float* d_attn_chunk_m_ = nullptr;   // Running max values for online softmax (chunked attention).
  float* d_attn_chunk_l_ = nullptr;   // Running sum-of-exp values for online softmax.
  float* d_attn_chunk_o_ = nullptr;   // Running output accumulator for chunked attention.
  int attn_chunk_capacity_ = 0;       // Maximum number of KV tokens per attention chunk.

  // MoE decode scratch buffers (allocated only for MoE models).
  void* d_moe_router_w_ = nullptr;           // FP16 router weights [experts, hidden].
  void* d_moe_router_logits_ = nullptr;      // Router logits scratch [experts].
  std::int8_t* d_moe_router_w_q_ = nullptr;  // Packed router weights (INT8/INT4).
  float* d_moe_router_scales_ = nullptr;     // Router per-row scales [experts].
  void* d_moe_w1_ = nullptr;                 // FP16 expert w1 weights [expert_inter, hidden].
  void* d_moe_w2_ = nullptr;                 // FP16 expert w2 weights [hidden, expert_inter].
  void* d_moe_w3_ = nullptr;                 // FP16 expert w3 weights [expert_inter, hidden].
  std::int8_t* d_moe_w1_q_ = nullptr;        // Packed expert w1 weights (INT8/INT4).
  std::int8_t* d_moe_w2_q_ = nullptr;        // Packed expert w2 weights (INT8/INT4).
  std::int8_t* d_moe_w3_q_ = nullptr;        // Packed expert w3 weights (INT8/INT4).
  float* d_moe_s_w1_ = nullptr;              // Expert w1 per-row scales [expert_inter].
  float* d_moe_s_w2_ = nullptr;              // Expert w2 per-row scales [hidden].
  float* d_moe_s_w3_ = nullptr;              // Expert w3 per-row scales [expert_inter].
  int* d_moe_topk_idx_ = nullptr;            // Selected expert indices [top_k].
  float* d_moe_topk_prob_ = nullptr;         // Selected expert probabilities [top_k].

  // KV cache device and host (pinned) buffers (fp16 path).
  void* d_k_cache_ = nullptr;  // Device KV-cache for keys [num_layers x max_context x kv_dim].
  void* d_v_cache_ = nullptr;  // Device KV-cache for values [num_layers x max_context x kv_dim].
  void* h_k_cache_ = nullptr;  // Host-pinned mirror of d_k_cache_ for paged eviction.
  void* h_v_cache_ = nullptr;  // Host-pinned mirror of d_v_cache_ for paged eviction.

  // ── EAGLE speculative decoding (CPI_EAGLE=k; draft head from CPI_EAGLE_DIR) ──
  // One llama-style draft layer over [embed(token) | target post-norm feature]
  // pairs, sharing the target's embeddings and lm_head. Chain drafting; the
  // existing verify_tokens batched verify accepts/rejects.
  bool eagle_enabled_ = false;
  int eagle_k_ = 4;
  void* d_eagle_fc_ = nullptr;        // [hidden, 2*hidden]
  void* d_eagle_wq_ = nullptr;        // [q_hidden, hidden]
  void* d_eagle_wk_ = nullptr;        // [kv_hidden, hidden]
  void* d_eagle_wv_ = nullptr;
  void* d_eagle_wo_ = nullptr;        // [hidden, q_hidden]
  void* d_eagle_w1_ = nullptr;        // [inter, hidden]
  void* d_eagle_w3_ = nullptr;
  void* d_eagle_w2_ = nullptr;        // [hidden, inter]
  void* d_eagle_pnorm_ = nullptr;     // [hidden]
  __half* d_eagle_kcache_ = nullptr;  // draft KV [max_ctx pairs, kv_hidden]
  __half* d_eagle_vcache_ = nullptr;
  float* d_eagle_cos_ = nullptr;      // plain-theta rope tables (draft head has no rope scaling)
  float* d_eagle_sin_ = nullptr;
  __half* d_eagle_cat_ = nullptr;     // [2*hidden] fc input
  __half* d_eagle_x_ = nullptr;       // [hidden] draft hidden / residual / output feature
  __half* d_eagle_tmp_ = nullptr;     // [hidden]
  __half* d_eagle_norm_ = nullptr;    // [hidden]
  __half* d_eagle_q_ = nullptr;       // [q_hidden]
  __half* d_eagle_kv_ = nullptr;      // [2*kv_hidden] (k then v)
  __half* d_eagle_att_ = nullptr;     // [q_hidden]
  __half* d_eagle_gate_ = nullptr;    // [inter]
  __half* d_eagle_up_ = nullptr;      // [inter]
  __half* d_eagle_feats_ = nullptr;   // [max verify rows, hidden] true features from verify
  float* d_eagle_logits_ = nullptr;   // [vocab]
  int* d_eagle_tok_ = nullptr;        // [1]
  int* d_eagle_dtoks_ = nullptr;      // [16] device-side chained draft tokens
  // Graphed verify: a fixed-K, device-position replica of the verify forward
  // captured once and replayed per round (the eager verify's ~700 small
  // launches cost ~2ms/round).
  int* d_eagle_pos_ = nullptr;        // [1] device base position for the verify
  int* d_eagle_verdict_ = nullptr;    // [17] per-row argmax outputs
  float* d_eagle_mt_m_ = nullptr;     // mt split-K scratch [K, heads, chunks]
  float* d_eagle_mt_l_ = nullptr;
  float* d_eagle_mt_o_ = nullptr;     // [K, heads, chunks, head_dim]
  int eagle_mt_chunks_ = 0;
  cudaGraph_t eagle_vgraph_ = nullptr;
  cudaGraphExec_t eagle_vgraph_exec_ = nullptr;
  int eagle_vgraph_k_ = 0;            // K the graph was captured for (0 = none)
  void eagle_verify_forward(int K);   // graph-safe verify body (device pos)
  bool eagle_verify_graphed(const std::vector<int>& batch, int start_pos,
                            std::vector<int>& out_argmax);
  // ── EAGLE tree drafting (CPI_EAGLE_TREE=1, on top of CPI_EAGLE) ──
  // Static draft tree instead of a single chain: siblings rescue the round when
  // the top-1 draft misses (chain tokens/round is capped ~2.2 by the 69/43%
  // acceptance decay). Sibling verify rows share positions, so their K/V go to
  // per-layer scratch and attention uses ancestor bitmasks; the accepted path's
  // rows are scattered into the real cache after the verdict walk.
  bool eagle_tree_ = false;
  __half* d_eagle_tree_h_ = nullptr;   // [1 + nodes, hidden] stashed draft hiddens (DFS)
  __half* d_eagle_tree_k_ = nullptr;   // [num_layers, rows, kv_hidden] verify K scratch
  __half* d_eagle_tree_v_ = nullptr;   // [num_layers, rows, kv_hidden] verify V scratch
  int* d_eagle_row_off_ = nullptr;     // [rows] per-row depth (device, constant)
  unsigned int* d_eagle_anc_mask_ = nullptr;  // [rows] ancestor bitmasks (device, constant)
  int* d_eagle_scatter_ = nullptr;     // [8] accepted-row indices for the KV scatter
  // Batched draft levels: one forward per tree depth instead of one per node.
  // Row capacity 4 (widest level), node-scratch capacity 16 forwarded nodes.
  __half* d_eagle_bcat_ = nullptr;     // [4, 2*hidden] level fc inputs
  __half* d_eagle_bx_ = nullptr;       // [4, hidden] level hidden / residual
  __half* d_eagle_btmp_ = nullptr;     // [4, hidden]
  __half* d_eagle_bnorm_ = nullptr;    // [4, hidden]
  __half* d_eagle_bq_ = nullptr;       // [4, q_hidden]
  __half* d_eagle_batt_ = nullptr;     // [4, q_hidden]
  __half* d_eagle_bgate_ = nullptr;    // [4, inter]
  __half* d_eagle_bup_ = nullptr;      // [4, inter]
  __half* d_eagle_scrk_ = nullptr;     // [16, kv_hidden] drafted-node K (forward order)
  __half* d_eagle_scrv_ = nullptr;     // [16, kv_hidden]
  int* d_eagle_lvl_tok_ = nullptr;     // [16] per-row dtok slot of the input token
  int* d_eagle_lvl_feat_ = nullptr;    // [16] per-row stash row of the input feature
  unsigned int* d_eagle_lvl_mask_ = nullptr;  // [16] per-row draft-scratch ancestor mask
  int* d_eagle_lvl_dep_ = nullptr;     // [16] per-row depth (rope offset)
  void eagle_tree_level(int B, int row0, int n_scr);
  void eagle_tree_verify_forward(int K);
  std::vector<int> eagle_tree_generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                       const std::function<bool(int)>& on_token);
  bool eagle_tried_ = false;          // lazy one-shot load attempt
  bool eagle_load();                  // returns false (and disables) on any mismatch
  void eagle_free();
  // Runs one (feature, token) pair through the draft layer at pair row
  // `pair_idx`. The input token comes from token_dev (device int, chained
  // drafts) when non-null, else from token_host. With want_token the shared
  // lm_head argmax is written to dtok_out (device) without any host sync.
  // The output feature stays in d_eagle_x_.
  void eagle_step(const __half* feature, const int* token_dev, int token_host, int pair_idx,
                  bool want_token, int* dtok_out);
  std::vector<int> eagle_generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                  const std::function<bool(int)>& on_token);
  void eagle_prefill_pairs(const int* tokens, int count, int chunk_start);

  // Quantized KV cache (active when options_.kv_cache_int4 or CPI_KV_QUANT is set).
  // Replaces the fp16 buffers above; d_k_cache_ / d_v_cache_ are not allocated.
  // K and V each store head_dim * bits/8 bytes per (token, kv_head) plus one
  // fp16 absmax scale. kv_quant_rot_ stores K Hadamard-rotated (QuaRot R3);
  // the attention kernels rotate Q to match.
  bool kv_int4_enabled_ = false;         // True when the quantized KV cache is active.
  int kv_quant_kbits_ = 4;               // Bits per K element (4 or 8).
  int kv_quant_vbits_ = 4;               // Bits per V element (4 or 8).
  bool kv_quant_rot_ = false;            // Hadamard-rotate K before quantizing.
  int kv_quant_sink_ = 16;               // Sink tokens kept fp16 (0 disables).
  int kv_quant_win_ = 128;               // Recent-window tokens kept fp16 (0 disables).
  __half* d_kv_sink_k_ = nullptr;        // fp16 sink K [layers, sink_n, kv_heads, head_dim].
  __half* d_kv_sink_v_ = nullptr;        // fp16 sink V, same layout.
  __half* d_kv_ring_k_ = nullptr;        // fp16 recent-K ring [layers, win_n, kv_heads, head_dim].
  __half* d_kv_ring_v_ = nullptr;        // fp16 recent-V ring, same layout.
  std::int8_t* d_k_cache_i4_ = nullptr;  // Quantized K cache [layers, ctx, kv_heads, row_bytes].
  std::int8_t* d_v_cache_i4_ = nullptr;  // Quantized V cache [layers, ctx, kv_heads, row_bytes].
  __half* d_k_scales_ = nullptr;         // Per-head K dequant scales [layers, ctx, kv_heads].
  __half* d_v_scales_ = nullptr;         // Per-head V dequant scales [layers, ctx, kv_heads].

  // Per-layer weight management.
  LayerDeviceWeights layer_weights_{};  // Device weights for the currently active layer.
  LayerDeviceWeights
      streaming_layer_weights_[2]{};           // Double-buffer slots for streaming layer weights.
  LayerDeviceInt8Weights layer_weights_i8_{};  // INT8 device weights for the active layer.
  LayerDeviceInt8Weights
      streaming_layer_weights_i8_[2]{};          // Double-buffer INT8 slots for streaming.
  std::vector<LayerDeviceWeights> layer_cache_;  // Persistent device cache for GPU-resident layers.
  std::vector<LayerDeviceInt8Weights> layer_cache_i8_;  // INT8 cache for GPU-resident layers.
  std::vector<LayerHostPinnedWeights>
      layer_host_pinned_;                              // Host-pinned weights for streaming layers.
  std::vector<LayerHostInt8Weights> layer_host_int8_;  // Host INT8 weights for streaming layers.
  int cached_layer_count_ = 0;  // Number of layers held permanently in GPU memory.
  bool cached_int8_mlp_enabled_ =
      false;  // True when the resident layer cache uses INT8 MLP weights.
  bool cached_int8_proj_enabled_ =
      false;                      // True when QKV/wo are also cached as INT8 (fp16 copies freed).
  int prefill_chunk_size_ = 256;  // Tokens per batched prefill pass; 256 saturates the GPU (16 was
                                  // ~5.3x slower). Override via CPI_PREFILL_CHUNK_SIZE.
  BenchmarkStats last_benchmark_stats_{};  // Statistics from the most recent generate/stream call.

  // CUDA streams and synchronisation events.
  cudaStream_t compute_stream_ = nullptr;  // Primary stream for all compute kernels.
  cudaStream_t transfer_stream_ =
      nullptr;                        // Secondary stream for async host-to-device weight transfers.
  cudaEvent_t streaming_ready_[2]{};  // Signalled when a streamed layer has finished transferring.
  cudaEvent_t
      streaming_consumed_[2]{};  // Signalled when the compute stream has finished using a layer.
  cudaEvent_t benchmark_transfer_start_ = nullptr;  // Marks the start of a timed transfer window.
  cudaEvent_t benchmark_transfer_end_ = nullptr;    // Marks the end of a timed transfer window.
  bool benchmark_transfer_active_ = false;          // True while a timed transfer is in flight.

  // CUDA graph state for the greedy-decode fast path.
  cudaGraph_t greedy_decode_graph_ = nullptr;           // Captured decode graph object.
  cudaGraphExec_t greedy_decode_graph_exec_ = nullptr;  // Executable instance of the decode graph.
  // How many packed k-quant matvecs actually launched. There are several layer
  // bodies (two captured into CUDA graphs, one not), so "the output did not
  // change" cannot on its own tell you the packed path ran (it did not, once).
  std::uint64_t packed_matvec_calls_ = 0;
  std::uint64_t packed_matmul_calls_ = 0;
  std::uint64_t packed_matmul_declined_ = 0;

  bool greedy_decode_graph_ready_ = false;  // True once the graph has been captured and compiled.
  bool greedy_decode_graph_state_valid_ =
      false;  // True when cached graph inputs match the current state.
  int greedy_decode_graph_expected_token_ = 0;  // Token ID the graph was last compiled for.
  int greedy_decode_graph_expected_position_ =
      0;  // Sequence position the graph was last compiled for.

  // CUDA graph state for the logits-decode fast path (sampling with temperature).
  // Same transformer body as the greedy graph but outputs d_logits_ without argmax.
  cudaGraph_t logits_decode_graph_ = nullptr;
  cudaGraphExec_t logits_decode_graph_exec_ = nullptr;
  bool logits_decode_graph_ready_ = false;

  // TQ3 (TurboQuant 3-bit) state.
  // Enabled when the weight file contains tq3_codebook + tq3_signs_hidden tensors.
  // Covers wq/wk/wv/wo/w1/w3 (in_features == hidden_size, power of 2).
  // w2 (in_features == intermediate_size, not power of 2) stays fp16.
  bool tq3_enabled_ = false;
  bool tq_prod_enabled_ = false;  // True when residual 1-bit correction is active (Qprod).
  int tq_objective_file_ = 0;     // 0=mse, 1=prod from model metadata.
  int tq_qjl_dim_ = 0;            // Residual projection dimension for Qprod.
  int tq_qjl_words_ = 0;          // ceil(tq_qjl_dim_/32).

  // Device-side global TQ3 parameters (loaded once at init).
  half* d_tq3_codebook_ = nullptr;  // [8] FP16 reconstruction values.
  int8_t* d_tq3_signs_ = nullptr;   // [hidden_size] +/-1 Hadamard signs.
  int tq3_block_size_ = 0;          // WHT sub-block size (0 until loaded; defaults to hidden_size).
  int32_t* d_tq_qjl_indices_ = nullptr;  // [tq_qjl_dim_] projected coordinate indices.
  int8_t* d_tq_qjl_signs_ = nullptr;     // [tq_qjl_dim_] projected coordinate signs in {-1,+1}.
  uint32_t* d_tq_qjl_x_bits_ =
      nullptr;  // [tq_qjl_words_] packed signs for current rotated activation.
  // Rotated activation scratch buffer (same size as d_x_norm_).
  // Holds z = Π·x_norm after launch_hadamard_rotate_fp16; used as input to
  // TQ3 GEMV projections in place of the plain x_norm.
  void* d_x_tq3_ = nullptr;

  // Per-layer TQ3 weight buffers (GPU-resident, allocated in init_layer_cache).
  // Parallel to layer_cache_; only populated when tq3_enabled_ is true.
  struct LayerDeviceTq3Weights {
    uint32_t* wqkv = nullptr;  // packed [hidden+2*kv_hidden, ceil(hidden/10)]
    uint32_t* wo = nullptr;    // packed [hidden, ceil(hidden/10)]
    uint32_t* w13 = nullptr;   // packed [2*inter, ceil(hidden/10)]
    half* s_wqkv = nullptr;    // per-row fp16 scales [hidden+2*kv_hidden]
    half* s_wo = nullptr;      // per-row fp16 scales [hidden]
    half* s_w13 = nullptr;     // per-row fp16 scales [2*inter]
    uint32_t* r_wqkv =
        nullptr;               // packed residual signatures [hidden+2*kv_hidden, ceil(qjl_dim/32)].
    uint32_t* r_wo = nullptr;  // packed residual signatures [hidden, ceil(qjl_dim/32)].
    uint32_t* r_w13 = nullptr;  // packed residual signatures [2*inter, ceil(qjl_dim/32)].
    half* rs_wqkv = nullptr;    // residual correction scales [hidden+2*kv_hidden].
    half* rs_wo = nullptr;      // residual correction scales [hidden].
    half* rs_w13 = nullptr;     // residual correction scales [2*inter].
  };
  std::vector<LayerDeviceTq3Weights> layer_cache_tq3_;

  // Tuned parameters for the custom resident projection kernels.
  bool resident_custom_qkv_ = false;      // True when the custom kernel is used for QKV projection.
  bool resident_custom_wo_ = false;       // True when the custom kernel is used for Wo projection.
  bool resident_custom_lm_head_ = false;  // True when the custom kernel is used for the LM head.
  int resident_qkv_warps_ = 8;            // Warps-per-block for the QKV custom kernel.
  int resident_qkv_tile_pairs_ = 128;     // Tile-pair count for the QKV custom kernel.
  int resident_qkv_rows_per_warp_ = 1;    // Output rows handled per warp in the QKV kernel.
  int resident_wo_warps_ = 4;             // Warps-per-block for the Wo custom kernel.
  int resident_wo_tile_pairs_ = 128;      // Tile-pair count for the Wo custom kernel.
  int resident_wo_rows_per_warp_ = 1;     // Output rows per warp in the Wo kernel.
  int resident_lm_head_warps_ = 8;        // Warps-per-block for the LM-head custom kernel.
  int resident_lm_head_tile_pairs_ = 128;    // Tile-pair count for the LM-head custom kernel.
  int resident_lm_head_rows_per_warp_ = 1;   // Output rows per warp in the LM-head kernel.
  int resident_mlp_w13_warps_ = 4;           // Warps-per-block for the fused w1/w3 INT8 MLP kernel.
  int resident_mlp_w13_tile_packed4_ = 128;  // Packed-4 tile count for the w1/w3 INT8 kernel.
  int resident_mlp_w13_warps_per_row_ = 1;   // Warps per output row in the w1/w3 INT8 kernel.
  int resident_mlp_w2_warps_ = 4;            // Warps-per-block for the w2 INT8 MLP kernel.
  int resident_mlp_w2_tile_packed4_ = 128;   // Packed-4 tile count for the w2 INT8 kernel.
  int resident_mlp_w2_warps_per_row_ = 1;    // Warps per output row in the w2 INT8 kernel.
  int mlp_quant_group_ = 0;  // CPI_MLP_INT4_GROUP: group-wise int4 MLP weight-scale granularity
                             // (0 = per-row legacy; 128 = llama.cpp Q4_0-style). Resident cache only.
  int resident_int8_qkv_warps_ = 8;          // Warps-per-block for the INT8 QKV dp4a kernel.
  int resident_int8_qkv_tile_packed4_ = 256;  // Packed-4 tile count for the INT8 QKV kernel.
  int resident_int8_qkv_warps_per_row_ = 2;   // Warps per output row in the INT8 QKV kernel.
  int resident_int8_wo_warps_ = 8;            // Warps-per-block for the INT8 wo dp4a kernel.
  int resident_int8_wo_tile_packed4_ = 256;   // Packed-4 tile count for the INT8 wo kernel.
  int resident_int8_wo_warps_per_row_ = 2;    // Warps per output row in the INT8 wo kernel.

  // Host resource guardrail state.
  std::chrono::steady_clock::time_point last_resource_sample_time_{};
  bool resource_sample_ready_ = false;
  double sampled_cpu_percent_ = -1.0;
  double sampled_memory_percent_ = -1.0;
  bool over_limit_active_ = false;
  std::chrono::steady_clock::time_point over_limit_since_{};
  std::chrono::steady_clock::time_point last_over_limit_log_time_{};
};

}  // namespace engine
