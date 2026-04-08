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

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <utility>
#include <string>
#include <vector>

#include "model/weight_loader.hpp"

namespace engine {

// Cache key that uniquely identifies a cublasLt matmul plan by its shape and
// output data type.  Used to look up pre-built plans in lt_plan_cache_.
struct LtMatmulPlanKey {
  int out_features = 0;                        // Number of output rows (M dimension).
  int in_features = 0;                         // Number of input columns (K dimension).
  int batch_size = 1;                          // Batch count for batched matmul.
  cudaDataType_t output_type = CUDA_R_16F;     // cuBLAS data type of the output matrix.

  // Returns true when all fields of this key are equal to other's fields.
  bool matches(const LtMatmulPlanKey& other) const {
    return out_features == other.out_features && in_features == other.in_features && batch_size == other.batch_size &&
           output_type == other.output_type;
  }
};

// A fully constructed cublasLt matmul plan together with its identifying key.
// Plans are created once and stored in lt_plan_cache_ to avoid repeated
// heuristic queries to the cuBLAS library.
struct LtMatmulPlan {
  LtMatmulPlanKey key{};                          // Shape/type key used to look up this plan.
  cublasLtMatmulDesc_t op_desc = nullptr;         // cuBLASLt operation descriptor.
  cublasLtMatrixLayout_t a_desc = nullptr;        // Layout descriptor for the A (weight) matrix.
  cublasLtMatrixLayout_t b_desc = nullptr;        // Layout descriptor for the B (activation) matrix.
  cublasLtMatrixLayout_t c_desc = nullptr;        // Layout descriptor for the C (output) matrix.
  cublasLtMatmulHeuristicResult_t heuristic{};    // Algorithm selected by the cuBLASLt heuristic.
  bool ready = false;                             // True once the plan has been fully initialised.
};

// Runtime configuration options passed to LlamaEngine::initialize().
// All fields have conservative defaults suitable for single-sequence inference
// on a GPU with at least 16 GB of VRAM.
struct EngineOptions {
  std::string model_path;                          // Path to the compiled .bin weight file.
  int max_batch = 1;                               // Maximum number of sequences processed simultaneously.
  int max_context = 2048;                          // Maximum total sequence length (prompt + generated tokens).
  int tensor_parallel = 1;                         // Number of tensor-parallel GPU shards (reserved, currently 1).
  std::size_t vram_safety_margin_mb = 1024;        // VRAM headroom kept free when auto-sizing buffers (MiB).
  int gpu_cache_layers = -1;                       // Number of layers to keep permanently on GPU (-1 = auto).
  std::size_t gpu_cache_limit_mb = 0;              // Hard cap on GPU layer-cache memory (0 = unlimited).
  bool gpu_cache_all = false;                      // When true, all layers are kept resident on GPU.
  int top_k = 40;                                  // Top-K sampling: number of candidates to keep.
  float top_p = 0.9f;                              // Top-P (nucleus) sampling threshold.
  float repetition_penalty = 1.0f;                 // Multiplicative penalty applied to recently generated tokens.
  int no_repeat_ngram_size = 0;                    // Block repeated n-grams of this length (0 = disabled).
  bool int8_streaming = false;                     // Enable low-bit streaming MLP path (legacy flag name kept for compatibility).
  int streaming_quant_bits = 8;                    // Bit-width for regular streaming quantization (supported: 4, 8).
  bool profile_decode_phases = false;              // Emit per-phase timing into BenchmarkStats after each decode step.
  bool disable_split_attention = false;            // Force single-pass attention even for long sequences.
  bool loop_guard = true;                          // Abort generation when a repeated token loop is detected.
  bool paged_kv_cache = false;                     // Use the paged KV-cache manager instead of the flat cache.
  bool kv_cache_int4 = false;                      // Store KV cache as packed INT4 (4x less VRAM bandwidth, ~4x attention speedup).
  bool enable_tq_cached = false;                   // Opt-in: allow cached/resident TurboQuant paths after preflight checks.
  std::string tq_mode = "auto";                    // TurboQuant objective override: auto|mse|prod.
  bool verbose = true;                             // Print engine startup/runtime diagnostics to stdout.
  float rope_theta = 0.0f;                          // RoPE base frequency override (0 = read from model file / family default).
  bool enable_host_resource_limits = true;         // Enforce host CPU/RAM guardrails during startup and generation.
  double max_cpu_percent = 85.0;                   // Host CPU usage hard ceiling (%).
  double max_memory_percent = 85.0;                // Host physical memory usage hard ceiling (%).
  int resource_sample_interval_ms = 250;           // Min interval between host resource samples.
  int resource_sustain_ms = 5000;                  // Abort after this much sustained over-limit time.
  int resource_throttle_sleep_ms = 50;             // Sleep duration used while over limit before abort.
  int tq_cached_init_timeout_ms = 180000;          // Timeout budget for cached TurboQuant layer init.
  int tq_first_token_timeout_ms = 120000;          // Timeout budget for first decode token in TurboQuant cached mode.
  int eos_token_id = 2;                            // EOS token used by engine-level decode stop check (-1 disables).
};

// Timing and token-count statistics collected during the most recent generate()
// or generate_stream() call.  All *_ms fields are wall-clock milliseconds.
struct BenchmarkStats {
  double prefill_ms = 0.0;           // Total time spent processing the prompt (prefill phase).
  double decode_ms = 0.0;            // Total time spent in the autoregressive decode loop.
  double transfer_ms = 0.0;          // Total time spent streaming layer weights from host to device.
  double decode_rmsnorm_ms = 0.0;    // Cumulative time for RMSNorm kernels during decode.
  double decode_qkv_ms = 0.0;        // Cumulative time for QKV projection during decode.
  double decode_kv_store_ms = 0.0;   // Cumulative time for writing new KV entries to the cache.
  double decode_attention_ms = 0.0;  // Cumulative time for the attention score + softmax kernels.
  double decode_wo_ms = 0.0;         // Cumulative time for the output projection (Wo) during decode.
  double decode_mlp_ms = 0.0;        // Cumulative time for the MLP block during decode.
  double decode_moe_router_ms = 0.0; // Cumulative time for MoE router logits + top-k selection.
  double decode_moe_expert_ms = 0.0; // Cumulative time for MoE expert matvec compute.
  double decode_moe_merge_ms = 0.0;  // Cumulative time for MoE weighted merge kernels.
  double decode_lm_head_ms = 0.0;    // Cumulative time for the final LM-head projection + sampling.
  int prompt_tokens = 0;             // Number of tokens in the input prompt.
  int generated_tokens = 0;          // Number of tokens produced during decode.
  int streamed_layer_copies = 0;     // Number of layer weight transfers triggered during this run.
  int tq3_cached_active = 0;         // 1 when full resident TurboQuant cached path is active, else 0.
  // MoE routing snapshot for the most recent decoded token.
  // Flattened layout: [layer0_e0, layer0_e1, ..., layerN_e(top_k-1)].
  int moe_topk_layers = 0;
  int moe_topk_k = 0;
  std::vector<int> moe_topk_indices{};
  std::vector<float> moe_topk_probs{};
  std::string moe_quant_mode = "none"; // "fp16", "int8", "int4", or "none" for non-MoE.
};

// Main inference engine for LLaMA-family models.
//
// LlamaEngine manages the full lifecycle of a CUDA-accelerated forward pass:
// weight loading, KV-cache allocation, prefill, and autoregressive decode.
// It is not copyable or movable; construct exactly one instance per GPU.
class LlamaEngine {
 public:
  ~LlamaEngine();

  // Loads weights, allocates CUDA buffers, and warms up cuBLAS plans.
  // Must be called before any other method.  Throws on CUDA or I/O errors.
  void initialize(const EngineOptions& options);

  // Runs a full prefill + decode session and returns all generated token IDs.
  // Blocks until max_new_tokens have been produced or an EOS token is sampled.
  std::vector<int> generate(const std::vector<int>& prompt_tokens,
                            int max_new_tokens,
                            float temperature);

  // Streaming variant of generate().  Calls on_token(token_id) after each
  // newly generated token.  Generation stops when on_token returns false,
  // when max_new_tokens is reached, or when EOS is sampled.
  // Returns the full sequence of generated token IDs.
  std::vector<int> generate_stream(const std::vector<int>& prompt_tokens,
                                   int max_new_tokens,
                                   float temperature,
                                   const std::function<bool(int)>& on_token);

  // Runs a forward pass over prompt_tokens and returns the top_k (token_id,
  // logit_value) pairs from the next-token distribution.  Useful for
  // debugging and probability inspection without committing to a sample.
  std::vector<std::pair<int, float>> inspect_next_logits(const std::vector<int>& prompt_tokens, int top_k);

  // Runs a numerical parity check on the model's outputs for prompt_tokens,
  // printing diagnostic information to stdout.  Used for regression testing.
  void run_parity_check(const std::vector<int>& prompt_tokens);

  // Returns the BenchmarkStats collected during the most recent generate() or
  // generate_stream() call.
  const BenchmarkStats& last_benchmark_stats() const { return last_benchmark_stats_; }

 private:
  // Weight pointers for a single transformer layer that are kept resident on
  // the GPU device.  All pointers are untyped (void*) because the concrete
  // element type (fp16 or fp32) depends on the runtime dtype setting.
  struct LayerDeviceWeights {
    void* wqkv = nullptr;      // Fused QKV projection weight matrix.
    void* wo = nullptr;        // Output (post-attention) projection weight matrix.
    void* bo = nullptr;        // Optional output-projection bias [hidden].
    void* w13 = nullptr;       // Fused gate (w1) and up-projection (w3) weight for SwiGLU.
    void* w2 = nullptr;        // Down-projection weight matrix for the MLP block.
    void* norm_att = nullptr;  // RMSNorm scale vector applied before the attention block.
    void* norm_ffn = nullptr;  // RMSNorm scale vector applied before the FFN/MLP block.
    void* norm_att_bias = nullptr; // Optional pre-attention norm bias [hidden].
    void* norm_ffn_bias = nullptr; // Optional pre-FFN norm bias [hidden].
    void* bqkv = nullptr;      // Fused QKV bias vector [q_dim + kv_dim + kv_dim]; null if unused.
  };

  // Weight pointers for a single transformer layer stored in host page-locked
  // (pinned) memory.  QKV is stored split (separate q, k, v matrices) so that
  // streaming transfers can overlap with compute on the other stream.
  struct LayerHostPinnedWeights {
    void* wq = nullptr;       // Query projection weight matrix (host-pinned copy).
    void* wk = nullptr;       // Key projection weight matrix (host-pinned copy).
    void* wv = nullptr;       // Value projection weight matrix (host-pinned copy).
    void* wo = nullptr;       // Output projection weight matrix (host-pinned copy).
    void* bo = nullptr;       // Optional output-projection bias (host-pinned copy).
    void* w1 = nullptr;       // Gate projection weight matrix, SwiGLU (host-pinned copy).
    void* w2 = nullptr;       // Down-projection weight matrix (host-pinned copy).
    void* w3 = nullptr;       // Up-projection weight matrix, SwiGLU (host-pinned copy).
    void* norm_att = nullptr; // Attention RMSNorm scale (host-pinned copy).
    void* norm_ffn = nullptr; // FFN RMSNorm scale (host-pinned copy).
    void* norm_att_bias = nullptr; // Optional attention norm bias (host-pinned copy).
    void* norm_ffn_bias = nullptr; // Optional FFN norm bias (host-pinned copy).
    void* bqkv = nullptr;     // Fused QKV bias (host-pinned copy); null if unused.
  };

  // INT8-quantised MLP weights stored in host memory.
  // s_w* arrays hold per-row quantisation scales used to dequantise the
  // corresponding INT8 weight matrix during the forward pass.
  struct LayerHostInt8Weights {
    std::int8_t* w1 = nullptr; // INT8 gate projection weight matrix.
    std::int8_t* w2 = nullptr; // INT8 down-projection weight matrix.
    std::int8_t* w3 = nullptr; // INT8 up-projection weight matrix.
    float* s_w1 = nullptr;     // Per-row dequantisation scales for w1.
    float* s_w2 = nullptr;     // Per-row dequantisation scales for w2.
    float* s_w3 = nullptr;     // Per-row dequantisation scales for w3.
  };

  // INT8-quantised MLP weights resident on the GPU device.
  // Layout mirrors LayerHostInt8Weights; pointers address device memory.
  struct LayerDeviceInt8Weights {
    std::int8_t* w1 = nullptr; // Device INT8 gate projection weight matrix.
    std::int8_t* w2 = nullptr; // Device INT8 down-projection weight matrix.
    std::int8_t* w3 = nullptr; // Device INT8 up-projection weight matrix.
    bool mlp_int4 = false;     // True when w1/w2/w3 are packed INT4 (two signed nibbles per byte).
    float* s_w1 = nullptr;     // Device per-row dequantisation scales for w1.
    float* s_w2 = nullptr;     // Device per-row dequantisation scales for w2.
    float* s_w3 = nullptr;     // Device per-row dequantisation scales for w3.
    // Projection weights quantised at layer-cache init time (null when unused).
    std::int8_t* wqkv = nullptr; // Device INT8 fused QKV weight matrix.
    bool proj_int4 = false;      // True when wqkv/wo are packed INT4 (two signed nibbles per byte).
    float* s_wqkv = nullptr;     // Device per-row dequantisation scales for wqkv.
    std::int8_t* wo = nullptr;   // Device INT8 output projection weight matrix.
    float* s_wo = nullptr;       // Device per-row dequantisation scales for wo.
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
  void copy_layer_weights_to_device(int layer,
                                    LayerDeviceWeights* dst,
                                    LayerDeviceInt8Weights* dst_i8,
                                    cudaStream_t stream);

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

  // Processes the prompt token-by-token (sequential prefill) without using
  // chunked attention.  Used as a fallback when the prompt fits in a single
  // chunk or when split-attention is disabled.
  void prefill_prompt_sequential(const std::vector<int>& prompt_tokens);

  // Processes the prompt using chunked attention to support long sequences
  // efficiently, then falls back to sequential for the tail.
  void prefill_prompt(const std::vector<int>& prompt_tokens);

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

  // Runs a full forward pass for a single token at position, writes the
  // full logit vector to *out_logits, and the argmax index to *out_argmax.
  void forward_token_logits(int token, int position, std::vector<float>* out_logits, int* out_argmax);

  // Runs a full forward pass for a single token at position.
  // If compute_logits is true, results are written to out_logits/out_argmax.
  void forward_token(int token,
                     int position,
                     bool compute_logits,
                     std::vector<float>* out_logits,
                     int* out_argmax);
  void forward_decode_layers(int token, int position);

  // Samples the next token ID given the current token at position, applying
  // temperature, top-k/p filtering, repetition penalty, and the no-repeat
  // n-gram constraint using history as the previously generated context.
  int decode_next_token(int token,
                        int position,
                        float temperature,
                        const std::vector<int>& history);

  // Launches the custom half-precision (FP16) projection kernel for a
  // weight matrix w of shape [out_features x in_features] applied to
  // activation vector x, writing the result to y.
  // warps_per_block, tile_pairs, and rows_per_warp tune the kernel launch.
  void resident_projection_half(const void* w,
                                const void* x,
                                void* y,
                                int out_features,
                                int in_features,
                                int warps_per_block = 0,
                                int tile_pairs = 0,
                                int rows_per_warp = 1);

  // Float32 variant of resident_projection_half() for models using fp32 weights.
  void resident_projection_float(const void* w,
                                 const void* x,
                                 void* y,
                                 int out_features,
                                 int in_features,
                                 int warps_per_block = 0,
                                 int tile_pairs = 0,
                                 int rows_per_warp = 1);

  // Applies either RMSNorm or true LayerNorm based on model config.
  // Optional bias is applied in-kernel for LayerNorm or as a post-add for RMSNorm.
  void launch_norm(const void* x,
                   const void* weight,
                   const void* bias,
                   void* y,
                   int rows,
                   int cols);

  // Adds an optional fp16 bias vector to a fp16 output tensor (1-D or row-broadcasted).
  void maybe_add_half_bias(void* out, const void* bias, int rows, int cols);

  // Enforces host CPU/RAM guardrails with "throttle then abort" semantics.
  void enforce_host_resource_limits(const char* stage);

  // Throws if cached TurboQuant initialisation exceeds the configured timeout.
  void check_tq_cached_init_timeout(const std::chrono::steady_clock::time_point& start,
                                    int layer_index);

  EngineOptions options_{};         // Runtime configuration supplied by the caller.
  model::WeightLoader weights_;     // Memory-mapped weight file handle.
  int attn_q_hidden_ = 0;           // Query projection width (rows in attention.wq).
  int attn_head_dim_ = 0;           // Per-head attention width (attn_q_hidden_ / num_heads).
  int attn_kv_hidden_ = 0;          // Key/value projection width (num_kv_heads * attn_head_dim_).
  bool has_any_layer_output_bias_ = false; // Any layer has attention.bo.
  bool has_any_layer_norm_bias_ = false;   // Any layer has norm bias tensors.

  cublasHandle_t cublas_ = nullptr;           // Legacy cuBLAS handle used for prefill GEMMs.
  cublasLtHandle_t cublas_lt_ = nullptr;      // cuBLASLt handle used for decode projections.
  void* lt_workspace_ = nullptr;              // Device workspace buffer required by cuBLASLt.
  std::size_t lt_workspace_bytes_ = 4 * 1024 * 1024; // Size of the cuBLASLt workspace (default 4 MiB).
  std::vector<LtMatmulPlan> lt_plan_cache_;   // Cache of pre-built cuBLASLt matmul plans.

  // Static (sequence-independent) device weight buffers.
  void* d_tok_embeddings_ = nullptr; // Token embedding table on device [vocab_size x hidden_size].
  void* d_norm_out_ = nullptr;       // Final RMSNorm scale vector on device.
  void* d_norm_out_bias_ = nullptr;  // Optional final norm bias [hidden].
  void* d_lm_head_ = nullptr;        // LM-head projection weight matrix on device.
  void* d_lm_head_bias_ = nullptr;   // Optional lm_head bias [vocab].

  // Per-step decode scratch buffers on device.
  int* d_token_id_ = nullptr;     // Single-element device buffer holding the current input token ID.
  void* d_x_ = nullptr;           // Residual stream buffer (hidden state) for the current token.
  void* d_x_norm_ = nullptr;      // RMSNorm-normalised version of d_x_.
  void* d_qkv_ = nullptr;         // Fused QKV projection output buffer.
  void* d_q_ = nullptr;           // Query slice of d_qkv_ (or separate for prefill).
  void* d_k_ = nullptr;           // Key slice.
  void* d_v_ = nullptr;           // Value slice.
  void* d_prefill_q_ = nullptr;   // Full-sequence Q buffer used during chunked prefill.
  void* d_prefill_k_ = nullptr;   // Full-sequence K buffer used during chunked prefill.
  void* d_prefill_v_ = nullptr;   // Full-sequence V buffer used during chunked prefill.
  void* d_att_ = nullptr;         // Attention output accumulation buffer.
  void* d_ff13_ = nullptr;        // Fused gate+up (w1/w3) MLP output buffer.
  void* d_ff1_ = nullptr;         // Gate activation buffer (SwiGLU first operand).
  void* d_ff2_ = nullptr;         // Post-down-projection MLP output buffer.
  void* d_prefill_ff1_ = nullptr; // Prefill-sized gate activation buffer.
  void* d_prefill_ff2_ = nullptr; // Prefill-sized down-projection output buffer.
  std::int8_t* d_prefill_i8_ = nullptr;     // INT8 quantised activations for prefill INT8 path.
  float* d_prefill_i8_scales_ = nullptr;    // Per-row scales accompanying d_prefill_i8_.
  void* d_ff3_ = nullptr;         // Up-projection (w3) output buffer (SwiGLU second operand).
  void* d_logits_ = nullptr;      // Raw logit vector output from the LM head.
  int* d_argmax_ = nullptr;       // Single-element device buffer for the argmax result.
  int* d_decode_position_ = nullptr; // Device copy of the current decode position index.
  float* d_rope_cos_ = nullptr;   // Precomputed RoPE cosine table [max_seq_len x head_dim].
  float* d_rope_sin_ = nullptr;   // Precomputed RoPE sine table [max_seq_len x head_dim].
  float* d_attn_chunk_m_ = nullptr;  // Running max values for online softmax (chunked attention).
  float* d_attn_chunk_l_ = nullptr;  // Running sum-of-exp values for online softmax.
  float* d_attn_chunk_o_ = nullptr;  // Running output accumulator for chunked attention.
  int attn_chunk_capacity_ = 0;      // Maximum number of KV tokens per attention chunk.

  // MoE decode scratch buffers (allocated only for MoE models).
  void* d_moe_router_w_ = nullptr;          // FP16 router weights [experts, hidden].
  void* d_moe_router_logits_ = nullptr;     // Router logits scratch [experts].
  std::int8_t* d_moe_router_w_q_ = nullptr; // Packed router weights (INT8/INT4).
  float* d_moe_router_scales_ = nullptr;    // Router per-row scales [experts].
  void* d_moe_w1_ = nullptr;                // FP16 expert w1 weights [expert_inter, hidden].
  void* d_moe_w2_ = nullptr;                // FP16 expert w2 weights [hidden, expert_inter].
  void* d_moe_w3_ = nullptr;                // FP16 expert w3 weights [expert_inter, hidden].
  std::int8_t* d_moe_w1_q_ = nullptr;       // Packed expert w1 weights (INT8/INT4).
  std::int8_t* d_moe_w2_q_ = nullptr;       // Packed expert w2 weights (INT8/INT4).
  std::int8_t* d_moe_w3_q_ = nullptr;       // Packed expert w3 weights (INT8/INT4).
  float* d_moe_s_w1_ = nullptr;             // Expert w1 per-row scales [expert_inter].
  float* d_moe_s_w2_ = nullptr;             // Expert w2 per-row scales [hidden].
  float* d_moe_s_w3_ = nullptr;             // Expert w3 per-row scales [expert_inter].
  int* d_moe_topk_idx_ = nullptr;           // Selected expert indices [top_k].
  float* d_moe_topk_prob_ = nullptr;        // Selected expert probabilities [top_k].

  // KV cache device and host (pinned) buffers (fp16 path).
  void* d_k_cache_ = nullptr; // Device KV-cache for keys [num_layers x max_context x kv_dim].
  void* d_v_cache_ = nullptr; // Device KV-cache for values [num_layers x max_context x kv_dim].
  void* h_k_cache_ = nullptr; // Host-pinned mirror of d_k_cache_ for paged eviction.
  void* h_v_cache_ = nullptr; // Host-pinned mirror of d_v_cache_ for paged eviction.

  // INT4-compressed KV cache (active when options_.kv_cache_int4 is set).
  // Replaces the fp16 buffers above; d_k_cache_ / d_v_cache_ are not allocated.
  bool     kv_int4_enabled_ = false; // True when INT4 KV cache is active.
  std::int8_t* d_k_cache_i4_ = nullptr; // Packed INT4 K cache [layers, ctx, kv_heads, head_dim/2].
  std::int8_t* d_v_cache_i4_ = nullptr; // Packed INT4 V cache [layers, ctx, kv_heads, head_dim/2].
  __half*      d_k_scales_   = nullptr; // Per-head K dequant scales [layers, ctx, kv_heads].
  __half*      d_v_scales_   = nullptr; // Per-head V dequant scales [layers, ctx, kv_heads].

  // Per-layer weight management.
  LayerDeviceWeights layer_weights_{};                         // Device weights for the currently active layer.
  LayerDeviceWeights streaming_layer_weights_[2]{};           // Double-buffer slots for streaming layer weights.
  LayerDeviceInt8Weights layer_weights_i8_{};                  // INT8 device weights for the active layer.
  LayerDeviceInt8Weights streaming_layer_weights_i8_[2]{};    // Double-buffer INT8 slots for streaming.
  std::vector<LayerDeviceWeights> layer_cache_;                // Persistent device cache for GPU-resident layers.
  std::vector<LayerDeviceInt8Weights> layer_cache_i8_;         // INT8 cache for GPU-resident layers.
  std::vector<LayerHostPinnedWeights> layer_host_pinned_;      // Host-pinned weights for streaming layers.
  std::vector<LayerHostInt8Weights> layer_host_int8_;          // Host INT8 weights for streaming layers.
  int cached_layer_count_ = 0;          // Number of layers held permanently in GPU memory.
  bool cached_int8_mlp_enabled_ = false;  // True when the resident layer cache uses INT8 MLP weights.
  bool cached_int8_proj_enabled_ = false; // True when QKV/wo are also cached as INT8 (fp16 copies freed).
  int prefill_chunk_size_ = 16;          // Number of tokens processed per attention chunk during prefill.
  BenchmarkStats last_benchmark_stats_{};// Statistics from the most recent generate/stream call.

  // CUDA streams and synchronisation events.
  cudaStream_t compute_stream_ = nullptr;   // Primary stream for all compute kernels.
  cudaStream_t transfer_stream_ = nullptr;  // Secondary stream for async host-to-device weight transfers.
  cudaEvent_t streaming_ready_[2]{};        // Signalled when a streamed layer has finished transferring.
  cudaEvent_t streaming_consumed_[2]{};     // Signalled when the compute stream has finished using a layer.
  cudaEvent_t benchmark_transfer_start_ = nullptr; // Marks the start of a timed transfer window.
  cudaEvent_t benchmark_transfer_end_ = nullptr;   // Marks the end of a timed transfer window.
  bool benchmark_transfer_active_ = false;          // True while a timed transfer is in flight.

  // CUDA graph state for the greedy-decode fast path.
  cudaGraph_t greedy_decode_graph_ = nullptr;          // Captured decode graph object.
  cudaGraphExec_t greedy_decode_graph_exec_ = nullptr; // Executable instance of the decode graph.
  bool greedy_decode_graph_ready_ = false;              // True once the graph has been captured and compiled.
  bool greedy_decode_graph_state_valid_ = false;        // True when cached graph inputs match the current state.
  int greedy_decode_graph_expected_token_ = 0;          // Token ID the graph was last compiled for.
  int greedy_decode_graph_expected_position_ = 0;       // Sequence position the graph was last compiled for.

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
  bool tq_prod_enabled_ = false; // True when residual 1-bit correction is active (Qprod).
  int tq_objective_file_ = 0;    // 0=mse, 1=prod from model metadata.
  int tq_qjl_dim_ = 0;           // Residual projection dimension for Qprod.
  int tq_qjl_words_ = 0;         // ceil(tq_qjl_dim_/32).

  // Device-side global TQ3 parameters (loaded once at init).
  half*    d_tq3_codebook_ = nullptr;   // [8] FP16 reconstruction values.
  int8_t*  d_tq3_signs_    = nullptr;   // [hidden_size] +/-1 Hadamard signs.
  int      tq3_block_size_ = 0;         // WHT sub-block size (0 until loaded; defaults to hidden_size).
  int32_t* d_tq_qjl_indices_ = nullptr; // [tq_qjl_dim_] projected coordinate indices.
  int8_t*  d_tq_qjl_signs_   = nullptr; // [tq_qjl_dim_] projected coordinate signs in {-1,+1}.
  uint32_t* d_tq_qjl_x_bits_ = nullptr; // [tq_qjl_words_] packed signs for current rotated activation.
  // Rotated activation scratch buffer (same size as d_x_norm_).
  // Holds z = Π·x_norm after launch_hadamard_rotate_fp16; used as input to
  // TQ3 GEMV projections in place of the plain x_norm.
  void* d_x_tq3_ = nullptr;

  // Per-layer TQ3 weight buffers (GPU-resident, allocated in init_layer_cache).
  // Parallel to layer_cache_; only populated when tq3_enabled_ is true.
  struct LayerDeviceTq3Weights {
    uint32_t* wqkv = nullptr;  // packed [hidden+2*kv_hidden, ceil(hidden/10)]
    uint32_t* wo   = nullptr;  // packed [hidden, ceil(hidden/10)]
    uint32_t* w13  = nullptr;  // packed [2*inter, ceil(hidden/10)]
    half* s_wqkv   = nullptr;  // per-row fp16 scales [hidden+2*kv_hidden]
    half* s_wo     = nullptr;  // per-row fp16 scales [hidden]
    half* s_w13    = nullptr;  // per-row fp16 scales [2*inter]
    uint32_t* r_wqkv = nullptr; // packed residual signatures [hidden+2*kv_hidden, ceil(qjl_dim/32)].
    uint32_t* r_wo   = nullptr; // packed residual signatures [hidden, ceil(qjl_dim/32)].
    uint32_t* r_w13  = nullptr; // packed residual signatures [2*inter, ceil(qjl_dim/32)].
    half* rs_wqkv    = nullptr; // residual correction scales [hidden+2*kv_hidden].
    half* rs_wo      = nullptr; // residual correction scales [hidden].
    half* rs_w13     = nullptr; // residual correction scales [2*inter].
  };
  std::vector<LayerDeviceTq3Weights> layer_cache_tq3_;

  // Tuned parameters for the custom resident projection kernels.
  bool resident_custom_qkv_ = false;        // True when the custom kernel is used for QKV projection.
  bool resident_custom_wo_ = false;         // True when the custom kernel is used for Wo projection.
  bool resident_custom_lm_head_ = false;    // True when the custom kernel is used for the LM head.
  int resident_qkv_warps_ = 8;              // Warps-per-block for the QKV custom kernel.
  int resident_qkv_tile_pairs_ = 128;       // Tile-pair count for the QKV custom kernel.
  int resident_qkv_rows_per_warp_ = 1;      // Output rows handled per warp in the QKV kernel.
  int resident_wo_warps_ = 4;               // Warps-per-block for the Wo custom kernel.
  int resident_wo_tile_pairs_ = 128;        // Tile-pair count for the Wo custom kernel.
  int resident_wo_rows_per_warp_ = 1;       // Output rows per warp in the Wo kernel.
  int resident_lm_head_warps_ = 8;          // Warps-per-block for the LM-head custom kernel.
  int resident_lm_head_tile_pairs_ = 128;   // Tile-pair count for the LM-head custom kernel.
  int resident_lm_head_rows_per_warp_ = 1;  // Output rows per warp in the LM-head kernel.
  int resident_mlp_w13_warps_ = 4;           // Warps-per-block for the fused w1/w3 INT8 MLP kernel.
  int resident_mlp_w13_tile_packed4_ = 128;  // Packed-4 tile count for the w1/w3 INT8 kernel.
  int resident_mlp_w13_warps_per_row_ = 1;   // Warps per output row in the w1/w3 INT8 kernel.
  int resident_mlp_w2_warps_ = 4;            // Warps-per-block for the w2 INT8 MLP kernel.
  int resident_mlp_w2_tile_packed4_ = 128;   // Packed-4 tile count for the w2 INT8 kernel.
  int resident_mlp_w2_warps_per_row_ = 1;    // Warps per output row in the w2 INT8 kernel.
  int resident_int8_qkv_warps_ = 8;          // Warps-per-block for the INT8 QKV dp4a kernel.
  int resident_int8_qkv_tile_packed4_ = 256; // Packed-4 tile count for the INT8 QKV kernel.
  int resident_int8_qkv_warps_per_row_ = 2;  // Warps per output row in the INT8 QKV kernel.
  int resident_int8_wo_warps_ = 8;           // Warps-per-block for the INT8 wo dp4a kernel.
  int resident_int8_wo_tile_packed4_ = 256;  // Packed-4 tile count for the INT8 wo kernel.
  int resident_int8_wo_warps_per_row_ = 2;   // Warps per output row in the INT8 wo kernel.

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
