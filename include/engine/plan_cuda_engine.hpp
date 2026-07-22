// Generic per-layer op-plan CUDA executor. The forward is resolved at load into
// a data op-plan (include/engine/op_plan.hpp) and the hot loop just executes it —
// no model-specific control flow. This is the "exotic model" tier: it runs
// architectures whose per-layer geometry breaks LlamaEngine's uniform-geometry
// fast path (per-layer head_dim / kv-heads, cross-layer KV sharing, PLE, …).
//
// Gemma 4 is currently its sole tenant (MatFormer text tower: Per-Layer
// Embeddings, dual head_dim 256/512, partial+dual RoPE, weightless V-norm,
// QK-norm, sandwich norms, KV-cache sharing, per-layer scalar, GeGLU, logit
// softcap). Adding another exotic model = a new descriptor + plan recipe, not a
// new engine. The load path still parses the .cpi/Gemma descriptor; that is the
// model-specific *configuration* (params), which is expected — the goal is no
// model-specific *code* in the forward, which is now met.
// See memory: cpi-gemma4-arch for Gemma's full spec + parity oracle.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/decode_driver.hpp"
#include "engine/engine_types.hpp"
#include "engine/op_plan.hpp"
#include "engine/plan_model_config.hpp"
#include "model/safetensors_loader.hpp"

namespace engine {

struct GenerationConstraints;  // fwd (generation_constraints.hpp)

// Derives from runtime::SequenceModel so generate/generate_stream reuse the
// shared decode driver (loop + sampler + stops) instead of a bespoke greedy loop.
class PlanCudaEngine : public runtime::SequenceModel {
 public:
  PlanCudaEngine() = default;
  ~PlanCudaEngine();
  PlanCudaEngine(const PlanCudaEngine&) = delete;
  PlanCudaEngine& operator=(const PlanCudaEngine&) = delete;

  // Load from a .cpi blob produced by tools/convert_gemma4.py (reads the sibling
  // .manifest for config + tensor directory).
  void open(const std::string& cpi_path, int max_context = 4096);

  // --- engine interface (matches the other CUDA engines, so main.cpp dispatch
  //     and execute_engine_modes can drive it) ---
  void initialize(const EngineOptions& options);
  std::vector<int> generate(const std::vector<int>& prompt, int max_new, float temperature);
  std::vector<int> generate_stream(const std::vector<int>& prompt, int max_new, float temperature,
                                   const std::function<bool(int)>& on_token,
                                   const GenerationConstraints* constraints = nullptr);
  std::vector<std::pair<int, float>> inspect_next_logits(const std::vector<int>& prompt, int top_k);
  const BenchmarkStats& last_benchmark_stats() const { return stats_; }

  // Run the token sequence through the text tower and return the softcapped
  // logits of the LAST position. If per_layer_rms != nullptr, fills it with each
  // layer's output RMS (parity vs the oracle). Resets KV state each call.
  std::vector<float> forward_logits(const std::vector<int>& tokens,
                                    std::vector<float>* per_layer_rms = nullptr);

  // Perf validation: capture the single-token decode step (device-position ops)
  // into a CUDA graph and A/B its replay against the non-graph path at a fixed
  // decode position. Verifies logits parity (graph vs non-graph) then reports
  // tok/s for both. NOTE: uses the full-attention device-pos kernels, so it is
  // only correct/representative for position < sliding_window (fine for a fixed
  // short position); long-context sliding correctness needs a windowed kernel.
  // `pos_override` > prompt length pads the prefill (repeating the last token) so
  // the A/B runs at that decode position — use it to exercise the sliding window
  // (pos >= sliding_window), where graph and non-graph must still match.
  void benchmark_graph_decode(const std::vector<int>& prompt, int iters, int pos_override = 0);

  int bos_id() const { return cfg_.bos_token_id; }

  // --- runtime::SequenceModel (the decode driver calls these) ---
  int vocab() const override { return cfg_.vocab; }
  int eos_id() const override { return cfg_.eos_token_id; }
  int max_context() const override { return max_ctx_; }
  void step(int token, int position, bool want_logits) override {
    forward_one(token, position, want_logits);
  }
  void synchronize() override { cudaStreamSynchronize(stream_); }
  std::vector<float>& logits() override { return last_logits_; }
  // Device-argmax greedy fast path: for greedy decode the final softcap is
  // monotonic (argmax(softcap(x)) == argmax(x)), so we argmax d_logits_ on device
  // and copy back one int — skipping the 262K-vocab D2H + host softcap that
  // otherwise dominate E2B decode. Non-greedy falls back to the host sampler.
  int sample(const runtime::DecodeParams& params, const std::vector<int>& history) override;

 private:
  // Which model recipe builds the plan, and the geometry it builds from. BOTH now live in
  // engine/plan_model_config.hpp, a header with no CUDA in it, because PlanMetalEngine needs the
  // same answers -- "what shape is this model" is not a backend question. These aliases keep every
  // existing `cfg_.<field>` and `Family::Gemma4` in this engine compiling unchanged; the field
  // names in the shared struct are deliberately identical for that reason.
  //
  // The EXECUTOR, sampler, quantizer, decode graph and KV/state machinery are shared; only the
  // config parse, weight load and plan recipe differ -- so a "new model" is a recipe, not an
  // engine, and a new BACKEND for an existing model should be a routing change, not a fork.
  using Family = engine::PlanFamily;
  using Config = engine::PlanModelConfig;

  struct TensorMeta {
    std::string dtype;
    std::vector<int> shape;
    std::size_t offset = 0, bytes = 0;
  };

  void parse_manifest(const std::string& manifest_path);
  void load_all(const std::string& cpi_path);
  // ── Qwen3.5 recipe (config parse + weight load + plan build). The executor,
  //    sampler, quantizer, graph and state machinery below are all SHARED. ──
  void parse_qwen35_config(const std::string& model_dir);
  void load_qwen35_weights();
  void build_qwen35_plan();
  // Gemma 4 from a HuggingFace safetensors directory (the MoE checkpoint ships that
  // way). Fills Config + st_shapes_ so the SHARED Gemma loader and plan builder run
  // unchanged -- only the tensor-name prefix and the shape source differ.
  void parse_gemma4_st_config(const std::string& model_dir);

  // ── vision tower ──
  // Its own config, slots and plan, but the SAME executor (in sequence mode) and the
  // same ops. A vision encoder is a capability, not a second engine.
  struct VisionConfig {
    bool present = false;
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0;
    int patch_size = 0, pos_table_size = 0, pooling_kernel = 1;
    float rms_eps = 1e-6f, rope_theta = 100.0f;
    bool standardize = false;
    bool clipped_linears = false;  // E2B: every projection clamps its input AND output
  };
  // Per-projection activation bounds, keyed by the projection's tensor prefix.
  struct ClipBounds {
    float in_min, in_max, out_min, out_max;
  };
  std::unordered_map<std::string, ClipBounds> vclip_;
  VisionConfig vcfg_;
  opplan::ModelPlan vplan_;
  std::array<__half*, static_cast<std::size_t>(opplan::Slot::Count)> vslot_ptr_{};
  float* d_vis_pixels_ = nullptr;   // [max_patches, patch_dim]
  int* d_vis_pos_x_ = nullptr;      // [max_patches]
  int* d_vis_pos_y_ = nullptr;      // [max_patches]
  float* d_vrope_cos_ = nullptr;    // [max_pos, head_dim/4]
  float* d_vrope_sin_ = nullptr;
  int vis_max_patches_ = 0;
  // Vision slot buffers (vision hidden != text hidden, so they cannot share the text
  // slots). Freed with the rest.
  std::vector<__half*> vis_buffers_;

  // Sequence-mode text prefill buffers. seq_prefill_ok_ is false for plans the sequence
  // path cannot yet run (partial RoPE / delta-net), which then keep the token-by-token
  // prefill -- a capability decision on the PLAN, not on a model name.
  bool seq_prefill_ok_ = false;
  int seq_max_tokens_ = 0;
  std::vector<__half*> seq_buffers_;
  std::array<__half*, static_cast<std::size_t>(opplan::Slot::Count)> sslot_ptr_{};
  int* d_seq_tokens_ = nullptr;
  int* d_seq_limits_ = nullptr;
  void allocate_sequence_buffers(int max_tokens);
  bool plan_can_sequence_prefill() const;

  void parse_vision_config(const std::string& model_dir);
  void load_vision_weights();
  void build_vision_plan();
  void allocate_vision_buffers();
  void build_vision_rope_tables();

public:
  // Encodes one image into soft tokens in the TEXT embedding space.
  //   pixels  - [num_patches, 3*patch^2], row-major, channel-last, values in [0,1]
  //   pos_x/y - patch grid coordinates; -1 marks a padding patch
  // Returns [out_tokens, text_hidden] on the host.
  // Prefill a whole prompt in ONE sequence pass (instead of token-by-token).
  //   embeds  - optional per-token embedding overrides (image soft tokens); an entry
  //             with an empty vector means "look the token up as usual"
  //   limits  - optional per-token key limits (bidirectional image spans)
  // Returns logits for the LAST token.
  void prefill_sequence(const std::vector<int>& tokens, int start_position,
                        const std::vector<std::vector<float>>& embeds,
                        const std::vector<int>& limits);
  bool can_sequence_prefill() const { return seq_prefill_ok_; }

  // Prompt containing image soft tokens: one sequence prefill (which splices the
  // embeddings and applies the bidirectional span limits), then the ordinary decode
  // loop with the ordinary sampler. Returns the generated tokens.
  std::vector<int> generate_multimodal(const std::vector<int>& tokens,
                                       const std::vector<std::vector<float>>& embeds,
                                       const std::vector<int>& limits, int max_new,
                                       float temperature,
                                       const std::function<bool(int)>& on_token);

  std::vector<float> encode_image(const std::vector<float>& pixels, const std::vector<int>& pos_x,
                                  const std::vector<int>& pos_y, int out_tokens);
  // Stage taps for the parity gate: 1 = after the patch embedder, 2 = after the
  // encoder stack (pre-pool). Both return [num_patches, vision_hidden].
  std::vector<float> encode_image_stage(const std::vector<float>& pixels,
                                        const std::vector<int>& pos_x,
                                        const std::vector<int>& pos_y, int stage);
  bool has_vision() const { return vcfg_.present; }
  const VisionConfig& vision_config() const { return vcfg_; }

private:
  // The MoE feed-forward block, appended to a layer's ops. Emitted only when the
  // config declares experts, so non-MoE Gemma plans are byte-for-byte unchanged.
  void append_moe_ffn_ops(std::vector<opplan::Op>& ops, const std::string& p, int L);
  // Host fp16 bytes for a tensor, from whichever container backs this model.
  // Safetensors carries no shape/dtype we can use, so dims come from the config
  // and bf16 is converted to fp16 (the .cpi path already stores fp16 + shapes).
  std::vector<__half> read_host_fp16(const std::string& name, int rows, int cols);
  float* upload_f32(const std::string& name, int n);       // float weights (delta-net norm / A_log)
  __half* upload(const std::string& name, int rows = 0, int cols = 0);
  // Loads a weight and quantizes it to int4 on the GPU, freeing the fp16 copy
  // immediately. Done PER TENSOR during load: uploading the whole model in fp16
  // first and quantizing afterwards would peak at the fp16 footprint — exactly the
  // OOM this exists to avoid (Gemma 12B is ~24 GB fp16). Peak here is one tensor.
  void upload_int4(const std::string& name, int rows = 0, int cols = 0);
  std::pair<int, int> shape_of(const std::string& name) const;
  const __half* dev_at(const std::string& name) const;
  float scalar_value(const std::string& name);             // read a [1] tensor to host
  void build_rope_tables();
  void allocate_buffers();
  // Resolve the per-layer forward into a data op-plan once at load (conditionals
  // shared/keqv/full/PLE decided here, weights + geometry bound); the hot loop
  // then just iterates it. See include/engine/op_plan.hpp.
  void build_plan();
  // Execute an op list against `position` (RoPE/KV/attention need it). `layer` is
  // the owning layer for cache-indexed ops (KvStore/Attention); pass -1 for the
  // prologue/epilogue, which contain none.
  // Execution context. Decode runs one token against a KV cache; a vision encoder
  // runs a whole SEQUENCE of patches with no cache and no causal mask. Both drive the
  // same op list through the same executor -- only the context differs.
  //
  // INVARIANT: at tokens == 1 every op takes the identical kernel + arguments it took
  // before sequence mode existed. Byte-identical decode is therefore structural, not
  // something we test for and hope: a GEMM and a GEMV do not sum K in the same order
  // (measured drift 1e-3), so single-token work must never wander onto the GEMM path.
  struct ExecCtx {
    __half* const* slots = nullptr;  // which slot table (text or vision)
    int tokens = 1;                  // 1 = decode; N = sequence
    bool causal = true;              // false = bidirectional (vision)
    // cached: a TEXT prefill -- K/V are appended to this layer's KV cache and attention
    // runs over the cache. A vision tower sets this false: it keeps no cache and attends
    // over the K/V slots directly.
    bool cached = false;
    // Optional per-token key limits (device). Text prefill with an image uses this to
    // make the image span bidirectional while the surrounding text stays causal.
    const int* limits = nullptr;
  };
  ExecCtx decode_ctx() const { return ExecCtx{slot_ptr_.data(), 1, true}; }

  void execute_ops(const opplan::Op* ops, std::size_t n, int layer, int position) {
    execute_ops(ops, n, layer, position, decode_ctx());
  }
  void execute_ops(const opplan::Op* ops, std::size_t n, int layer, int position,
                   const ExecCtx& ctx);
  void run_layer(int layer, int position);  // executes plan_.layers[layer].ops
  float* lin_conv_state(int layer);         // this layer's rolling causal-conv window
  float* lin_recurrent_state(int layer);    // this layer's delta-net recurrent state
  // Capture the single-token forward (device-position ops) into a CUDA graph once,
  // for reuse across every logits step. See forward_one.
  void capture_decode_graph();
  // Process one token at `position` (reusing the KV cache from prior positions).
  // When compute_logits, fills last_logits_ with softcapped logits. When
  // per_layer_rms != nullptr, appends each layer's output RMS (and dumps the
  // hidden to $G4_DUMP_DIR if set) — the oracle parity path.
  void forward_one(int token, int position, bool compute_logits,
                   std::vector<float>* per_layer_rms = nullptr);
  void publish_host_logits();  // d_logits_ -> last_logits_ (softcapped)

  std::vector<float> last_logits_;
  BenchmarkStats stats_;
  // Sampling knobs from EngineOptions (fed to the shared decode driver).
  int samp_top_k_ = 40;
  float samp_top_p_ = 0.9f;
  float samp_rep_penalty_ = 1.0f;
  int samp_no_repeat_ngram_ = 0;

  int head_dim_of(int layer) const {
    return cfg_.layer_full[layer] ? cfg_.head_dim_full : cfg_.head_dim_sliding;
  }
  int kv_heads_of(int layer) const {
    return cfg_.layer_full[layer] ? cfg_.num_kv_heads_full : cfg_.num_kv_heads_sliding;
  }
  bool k_eq_v(int layer) const { return cfg_.attention_k_eq_v && cfg_.layer_full[layer]; }
  bool has_ple() const { return cfg_.hidden_size_per_layer_input > 0; }  // E2B: yes, 12B: no
  float rope_theta_of(int layer) const {
    return cfg_.layer_full[layer] ? cfg_.rope_theta_full : cfg_.rope_theta_sliding;
  }

  Config cfg_;
  int max_ctx_ = 4096;
  std::string cpi_path_;
  std::size_t data_start_ = 0;
  std::unordered_map<std::string, TensorMeta> meta_;
  std::unordered_map<std::string, void*> dev_;   // name -> device fp16 ptr
  std::unordered_map<std::string, float*> devf_; // name -> device float ptr (delta-net norm/A_log)

  // Weight source. Gemma ships a .cpi (fp16 + shapes in the manifest); Qwen3.5
  // ships safetensors (bf16, shapes only in its config), so the engine reads
  // whichever the model actually has rather than forcing a conversion.
  model::SafetensorsLoader st_;
  bool from_safetensors_ = false;

  // Derived geometry (Qwen3.5).
  int rotary_dim_ = 0;      // partial RoPE width (head_dim * partial_rotary_factor)
  int full_q_dim_ = 0, full_kv_dim_ = 0;
  int linear_k_dim_ = 0, linear_v_dim_ = 0, linear_conv_dim_ = 0;

  // Delta-net / gated-attention working buffers (allocated only for that family).
  __half* d_q_pair_ = nullptr;   // fused q|gate before the split
  __half* d_q_gate_ = nullptr;
  __half* d_lin_mix_ = nullptr;  // fused linear qkv mixture
  __half* d_lin_z_ = nullptr;
  __half* d_lin_a_ = nullptr;
  __half* d_lin_b_ = nullptr;
  __half* d_lin_q_ = nullptr;
  __half* d_lin_k_ = nullptr;
  __half* d_lin_v_ = nullptr;
  __half* d_lin_att_ = nullptr;

  // int4 weight-only quantization (--weight-quant int4). Gemma 12B is ~24 GB in
  // fp16 and OOMs on a 32 GB card alongside the desktop; int4 on the per-layer
  // projections takes the transformer from ~22 GB to ~5.5 GB. Embeddings stay fp16
  // (the token lookup and the tied LM head both need them).
  struct QuantWeight {
    std::int8_t* packed = nullptr;  // int4: two values per byte, row-major
    float* scales = nullptr;        // dequant scales, [rows, groups] row-major
    int group = 0;                  // 0 = one scale per row; >0 = group-wise
  };
  std::unordered_map<std::string, QuantWeight> qdev_;  // name -> quantized weight
  int weight_quant_bits_ = 0;                          // 0 = fp16, 4 = int4, 8 = int8
  // Scale granularity for quantized weights (input features per scale, a power of
  // two; 0 = one scale per row). int4 defaults to 128: a per-row scale must span
  // the row's largest weight, so at 16 levels a single outlier coarsens every
  // other weight in the row, and the error compounds across layers (this is what
  // made int4 unusable on Gemma 12B). int8's 255 levels tolerate per-row.
  int weight_quant_group_ = 0;

  // Weight-name prefix. Empty for a .cpi container (its manifest already stores
  // stripped names); "model.language_model." for a HuggingFace safetensors dir.
  // Keeping it here means ONE Gemma plan builder serves both containers instead of
  // a second copy that differs only in string keys.
  std::string wprefix_;
  // Tensor shapes for the safetensors path, where the container carries no shape
  // index (unlike .cpi's manifest) -- derived from config.json at load.
  std::unordered_map<std::string, std::pair<int, int>> st_shapes_;
  std::vector<float> layer_scalar_host_;         // per-layer scalar values

  cudaStream_t stream_ = nullptr;

  // rope tables per layer type: [max_ctx, head_dim/2] fp32
  float* d_cos_sliding_ = nullptr;
  float* d_sin_sliding_ = nullptr;
  float* d_cos_full_ = nullptr;
  float* d_sin_full_ = nullptr;
  __half* d_ones_ = nullptr;  // weightless v-norm weight (all ones)

  // per-layer K/V caches: caches_k_[L] is [max_ctx, kv_dim(L)]
  std::vector<__half*> caches_k_;
  std::vector<__half*> caches_v_;

  // Per-layer state for linear-attention ("delta-net") layers: a rolling causal
  // conv window and the recurrent state. The analogue of the KV cache for that op
  // family — allocated only when the plan actually emits those ops.
  float* d_lin_conv_state_ = nullptr;
  float* d_lin_recurrent_state_ = nullptr;
  int lin_conv_state_stride_ = 0;
  int lin_recurrent_state_stride_ = 0;

  // scratch
  __half* d_x_ = nullptr;        // [hidden] — also holds the residual (adds are in-place)
  __half* d_x_norm_ = nullptr;   // [hidden]
  __half* d_tmp_ = nullptr;      // [hidden]
  __half* d_q_ = nullptr;        // [num_heads * max_head_dim]
  __half* d_k_ = nullptr;        // [max_kv_dim]
  __half* d_v_ = nullptr;        // [max_kv_dim]
  __half* d_att_ = nullptr;      // [num_heads * max_head_dim]
  __half* d_gate_ = nullptr;     // [max_inter]
  __half* d_up_ = nullptr;       // [max_inter]
  __half* d_inter_ = nullptr;    // [max_inter]
  __half* d_ple_raw_ = nullptr;  // [num_layers * ple_dim]
  __half* d_ple_ = nullptr;      // [num_layers * ple_dim] per-layer inputs
  __half* d_ple_gate_ = nullptr; // [ple_dim]

  // ── Mixture-of-Experts working set ──
  // The expert selection lives on the DEVICE: the router writes d_moe_idx_/d_moe_w_
  // and the expert kernels read them there. Nothing round-trips to the host between
  // the router and the experts (a sync per layer per token would cost more than the
  // experts themselves, and would make the decode graph uncapturable).
  __half* d_moe_router_in_ = nullptr;  // [hidden]
  __half* d_moe_logits_ = nullptr;     // [num_experts]
  __half* d_moe_inter_ = nullptr;      // [top_k * moe_intermediate]
  __half* d_moe_out_ = nullptr;        // [hidden]
  int* d_moe_idx_ = nullptr;           // [top_k] selected experts
  float* d_moe_w_ = nullptr;           // [top_k] routing weights
  float* d_logits_ = nullptr;    // [vocab]
  int* d_tok_ = nullptr;         // current token id (device); EmbeddingLookup reads it
  int* d_position_ = nullptr;    // current decode position (device); device-pos ops read it
  int* d_argmax_ = nullptr;      // device-argmax result (greedy fast path)

  // Device top-k sampling (temperature>0 — the real chat path, since greedy only
  // covers temp<=0). Selects the candidate set on the GPU so the host never sees
  // the full vocab-sized logit vector; only the ~k candidates come back.
  static constexpr int kMaxDeviceTopK = 256;
  static constexpr int kCandCapacity = kMaxDeviceTopK + 64;  // slack for ties at the k-th logit
  // Persistent decode: the plan compiled to device-side op descriptors, executed as ONE
  // cooperative launch per token (kernels_persistent_decode.cu). Compiled after build_plan;
  // a plan with kinds the interpreter lacks compiles to "unsupported" and decode keeps the
  // graph path. persist_enabled_ also drops to false if the cooperative launch is rejected.
  bool compile_persistent_plan();
  void* d_persist_ops_ = nullptr;  // kernels::PersistOp[], device
  int persist_n_ops_ = 0;
  int persist_blocks_ = 0;
  bool persist_enabled_ = false;

  // Rowwise-int8 packed copy of the tied embedding table, used by the LM head ONLY on
  // quant runs (EmbeddingLookup keeps reading the fp16 table). nullptr = fp16 head.
  std::int8_t* d_head_q_ = nullptr;
  float* d_head_qs_ = nullptr;

  // Decode activation-quant scratch for the dp4a matvec paths (one vector + one scale).
  // int8 runs hold the plain rowwise layout, int4 runs the perm8 layout -- a run is one
  // or the other, so the buffer is shared. nullptr = dp4a routing off.
  std::int8_t* d_act_i8_ = nullptr;
  float* d_act_qs_ = nullptr;

  // Split-K decode-attention scratch for the wide-head (any-head_dim) path. Sized
  // num_heads x split_any_chunks_ (x maxhd for the partial outputs) in allocate_buffers.
  static constexpr int kSplitAnyChunk = 32;
  // Depth-bucketed attention grids: forward_one sets attn_chunk_budget_ (a power-of-two
  // chunk count covering the current position, min 512 tokens) and the decode graph is
  // re-captured whenever the bucket grows. 0 = fall back to the full scratch range.
  int attn_chunk_budget_ = 0;
  int graph_bucket_chunks_ = 0;
  float* d_split_m_ = nullptr;
  float* d_split_l_ = nullptr;
  float* d_split_o_ = nullptr;
  int split_any_chunks_ = 0;

  float* d_topk_part_val_ = nullptr;  // per-partition scratch
  int* d_topk_part_idx_ = nullptr;
  float* d_topk_val_ = nullptr;       // [k] descending; [k-1] is the threshold
  int* d_topk_idx_ = nullptr;
  int* d_cand_idx_ = nullptr;         // gathered candidates (>= threshold)
  float* d_cand_val_ = nullptr;
  int* d_cand_count_ = nullptr;
  bool device_topk_enabled_ = true;   // env LLAMA_INFER_PLAN_NO_DEVICE_TOPK=1 disables (A/B)
  // When true (the shared decode-loop path), forward_one leaves the logits on the
  // device (d_logits_) instead of copying 262K floats to host + softcapping every
  // step; sample() then does the device argmax (greedy) or a lazy host copy.
  bool defer_host_logits_ = false;
  // When true, execute_ops uses the device-position kernel variants (RoPE / KV
  // store / attention read d_position_) so the op sequence is CUDA-graph capturable.
  bool device_pos_mode_ = false;

  // Decode CUDA graph: the single-token forward captured once (device-position
  // ops), replayed for every logits step. decode_graph_enabled_ off falls back to
  // the eager path (env LLAMA_INFER_PLAN_NO_GRAPH=1, for A/B / debugging).
  cudaGraph_t decode_graph_ = nullptr;
  cudaGraphExec_t decode_graph_exec_ = nullptr;
  bool decode_graph_ready_ = false;
  bool decode_graph_enabled_ = true;

  // The forward as data (built once in build_plan) + the Slot→device-buffer map
  // the executor dereferences. plan_.layers[L] is layer L's resolved op list.
  opplan::ModelPlan plan_;
  std::array<__half*, static_cast<std::size_t>(opplan::Slot::Count)> slot_ptr_{};
};

}  // namespace engine
