// Generic per-layer op-plan CUDA executor. The forward is resolved at load into
// a data op-plan (include/engine/op_plan.hpp) and the hot loop just executes it,
// with no model-specific control flow. This is the "exotic model" tier: it runs
// architectures whose per-layer geometry breaks LlamaEngine's uniform-geometry
// fast path (per-layer head_dim / kv-heads, cross-layer KV sharing, PLE, …).
//
// Gemma 4 is currently its sole tenant (MatFormer text tower: Per-Layer
// Embeddings, dual head_dim 256/512, partial+dual RoPE, weightless V-norm,
// QK-norm, sandwich norms, KV-cache sharing, per-layer scalar, GeGLU, logit
// softcap). Adding another exotic model = a new descriptor + plan recipe, not a
// new engine. The load path still parses the .cpi/Gemma descriptor; that is the
// model-specific *configuration* (params), which is expected; the goal is no
// model-specific *code* in the forward, which is now met.
// See memory: cpi-gemma4-arch for Gemma's full spec + parity oracle.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  // logits of the last position. If per_layer_rms != nullptr, fills it with each
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
  // the A/B runs at that decode position; use it to exercise the sliding window
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
  // and copy back one int, skipping the 262K-vocab D2H + host softcap that
  // otherwise dominate E2B decode. Non-greedy falls back to the host sampler.
  int sample(const runtime::DecodeParams& params, const std::vector<int>& history) override;
  // Batched prompt ingestion via the sequence-prefill path (the multimodal machinery,
  // minus images). Falls back to per-token stepping for plans that cannot sequence.
  void prefill(const std::vector<int>& prompt) override {
    const int T = static_cast<int>(prompt.size());
    if (T > 1 && seq_prefill_ok_) {
      // Chunked: the sequence buffers cap at seq_max_tokens_; KV appends keep the
      // chunks exact.
      constexpr int kChunk = 2048;
      int pos = 0;
      while (pos < T) {
        int n = std::min(kChunk, T - pos);
        // Never emit a single-token chunk: prefill_sequence with T == 1 degenerates to
        // decode-mode kernels that read the (stale) device position. Shrink so the tail
        // has at least two tokens, and step a genuinely single-token remainder normally.
        if (n > 1 && T - pos - n == 1) n -= 1;
        if (n == 1) {
          step(prompt[pos], pos, pos == T - 1);
        } else {
          prefill_sequence(std::vector<int>(prompt.begin() + pos, prompt.begin() + pos + n), pos,
                           {}, {});
        }
        pos += n;
      }
      return;
    }
    runtime::SequenceModel::prefill(prompt);
  }

 private:
  // Which model recipe builds the plan, and the geometry it builds from. both now live in
  // engine/plan_model_config.hpp, a header with no CUDA in it, because PlanMetalEngine needs the
  // same answers; "what shape is this model" is not a backend question. These aliases keep every
  // existing `cfg_.<field>` and `Family::Gemma4` in this engine compiling unchanged; the field
  // names in the shared struct are deliberately identical for that reason.
  //
  // The executor, sampler, quantizer, decode graph and KV/state machinery are shared; only the
  // config parse, weight load and plan recipe differ; so a "new model" is a recipe, not an
  // engine, and a new backend for an existing model should be a routing change, not a fork.
  using Family = engine::PlanFamily;
  using Config = engine::PlanModelConfig;

  struct TensorMeta {
    std::string dtype;
    std::vector<int> shape;
    std::size_t offset = 0, bytes = 0;
  };

  void parse_manifest(const std::string& manifest_path);
  bool seq_gemm_cublas(const __half* w, const __half* x, __half* y, int out, int in, int T);
  // Grouped-GEMM MoE seq prefill (DeepSeek). gate_up builds+caches the host routing, gathers per
  // expert and runs gate/up GEMM + SiLU; down runs the down GEMM and scatters into MoeOut. Return
  // false to fall back to the per-token loop (routing not built, or no cuBLAS).
  bool moe_grouped_gate_up(const opplan::Op& op, __half* xnorm, int T);
  bool moe_grouped_down(const opplan::Op& op, __half* moe_out, int T);
  // Short-prompt MoE uses the int4-direct grouped GEMM (no dequant-all); long prompts keep the
  // dequant+cuBLAS path. Decision keyed on P = total routings (= T*top_k). See moe_use_int4_direct.
  bool moe_use_int4_direct(int P) const;
  // One cublasGemmGroupedBatchedEx over all (fp16) expert GEMMs: group g computes C[g] = A[g]^T . B[g]
  // with A [m,k] row-major (OP_T), B [n_g,k] row-major, C [n_g,m] row-major. Collapses the ~64
  // per-expert calls into one, killing the tiny-GEMM launch-latency floor. false on failure.
  bool grouped_gemm_ex(int m, int k, const std::vector<int>& n_per_group,
                       const std::vector<const void*>& A, const std::vector<const void*>& B,
                       const std::vector<void*>& C);
  void verify_greedy(const int* tokens, int k, int pos, int* out_argmax);
  void load_all(const std::string& cpi_path);
  // ── Qwen3.5 recipe (config parse + weight load + plan build). The executor,
  //    sampler, quantizer, graph and state machinery below are all shared. ──
  void parse_qwen35_config(const std::string& model_dir);
  void load_qwen35_weights();
  void build_qwen35_plan();
  // Gemma 4 from a HuggingFace safetensors directory (the MoE checkpoint ships that
  // way). Fills Config + st_shapes_ so the shared Gemma loader and plan builder run
  // unchanged; only the tensor-name prefix and the shape source differ.
  void parse_gemma4_st_config(const std::string& model_dir);

  // ── DeepSeek-V2 recipe (MLA latent attention + fine-grained MoE). Same shared
  //    executor/sampler/quantizer/graph below; only parse + load + plan differ. ──
  void parse_deepseek_config(const std::string& model_dir);
  void load_deepseek_weights();
  void build_deepseek_plan();
  void append_deepseek_moe_ffn_ops(std::vector<opplan::Op>& ops, const std::string& p);
  // Fuse DeepSeek's per-expert un-fused gate/up/down into the single [E*2*MI,H]/[E*H,MI]
  // tensors the shared MoE ops expect, quantizing from a host buffer (no such tensor exists
  // in the checkpoint to read by name). Registers under a synthetic `name`.
  void fuse_upload_experts_int4(const std::string& name, const std::vector<__half>& host_rowmajor,
                                int rows, int cols);
  // Upload a host-built fp16 [rows,cols] weight under `name` (e.g. the zero-padded o_proj).
  __half* upload_host_fp16(const std::string& name, const std::vector<__half>& host, int rows,
                           int cols);

  // ── vision tower ──
  // Its own config, slots and plan, but the same executor (in sequence mode) and the
  // same ops. A vision encoder is a capability, not a second engine.
  struct VisionConfig {
    bool present = false;
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0;
    int patch_size = 0, pos_table_size = 0, pooling_kernel = 1;
    float rms_eps = 1e-6f, rope_theta = 100.0f;
    bool standardize = false;
    bool clipped_linears = false;  // E2B: every projection clamps its input and output
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
  // prefill; a capability decision on the PLAN, not on a model name.
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
  // Encodes one image into soft tokens in the text embedding space.
  //   pixels  - [num_patches, 3*patch^2], row-major, channel-last, values in [0,1]
  //   pos_x/y - patch grid coordinates; -1 marks a padding patch
  // Returns [out_tokens, text_hidden] on the host.
  // Prefill a whole prompt in one sequence pass (instead of token-by-token).
  //   embeds  - optional per-token embedding overrides (image soft tokens); an entry
  //             with an empty vector means "look the token up as usual"
  //   limits  - optional per-token key limits (bidirectional image spans)
  // Returns logits for the last token.
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
  // immediately. Done per tensor during load: uploading the whole model in fp16
  // first and quantizing afterwards would peak at the fp16 footprint, exactly the
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
  // runs a whole sequence of patches with no cache and no causal mask. Both drive the
  // same op list through the same executor; only the context differs.
  //
  // INVARIANT: at tokens == 1 every op takes the identical kernel + arguments it took
  // before sequence mode existed. Byte-identical decode is therefore structural, not
  // something we test for and hope: a GEMM and a GEMV do not sum K in the same order
  // (measured drift 1e-3), so single-token work must never wander onto the GEMM path.
  struct ExecCtx {
    __half* const* slots = nullptr;  // which slot table (text or vision)
    int tokens = 1;                  // 1 = decode; N = sequence
    bool causal = true;              // false = bidirectional (vision)
    // cached: a text prefill; K/V are appended to this layer's KV cache and attention
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
  // hidden to $G4_DUMP_DIR if set); the oracle parity path.
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
  // fp16 and can exceed available VRAM; int4 on the per-layer
  // projections takes the transformer from ~22 GB to ~5.5 GB. Embeddings stay fp16
  // (the token lookup and the tied LM head both need them).
  struct QuantWeight {
    std::int8_t* packed = nullptr;  // int4: two values per byte, row-major
    float* scales = nullptr;        // dequant scales, [rows, groups] row-major
    int group = 0;                  // 0 = one scale per row; >0 = group-wise
    // Rowwise-int8 PREFILL copy: sequence GEMMs run 8-bit x 8-bit -> int32 through
    // cuBLAS with per-token activation scales (exact fold: y = idot * s_w[r] * s_x[t]).
    // For int8 configs these alias `packed`/`scales` (no extra VRAM); int4 configs pay
    // one extra int8 copy. nullptr = use the dequant-to-fp16 fallback.
    std::int8_t* pf_i8 = nullptr;
    float* pf_scales = nullptr;
    bool host = false;  // true = packed/scales live on host (expert streaming); stage_moe H2D's them
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
  // Keeping it here means one Gemma plan builder serves both containers instead of
  // a second copy that differs only in string keys.
  std::string wprefix_;
  // Tensor shapes for the safetensors path, where the container carries no shape
  // index (unlike .cpi's manifest); derived from config.json at load.
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
  // family, allocated only when the plan actually emits those ops.
  float* d_lin_conv_state_ = nullptr;
  float* d_lin_recurrent_state_ = nullptr;
  int lin_conv_state_stride_ = 0;
  int lin_recurrent_state_stride_ = 0;

  // scratch
  __half* d_x_ = nullptr;        // [hidden], also holds the residual (adds are in-place)
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
  // The expert selection lives on the device: the router writes d_moe_idx_/d_moe_w_
  // and the expert kernels read them there. Nothing round-trips to the host between
  // the router and the experts (a sync per layer per token would cost more than the
  // experts themselves, and would make the decode graph uncapturable).
  __half* d_moe_router_in_ = nullptr;  // [hidden]
  __half* d_moe_logits_ = nullptr;     // [num_experts]
  __half* d_moe_inter_ = nullptr;      // [top_k * moe_intermediate]
  __half* d_moe_out_ = nullptr;        // [hidden]
  int* d_moe_idx_ = nullptr;           // [top_k] selected experts
  float* d_moe_w_ = nullptr;           // [top_k] routing weights
  // Sequence-prefill routing: one [top_k] slot per token ([max_tokens][top_k]). The MoE ops loop the
  // T==1 kernels per token in seq mode (DeepSeek batched prefill), reading token t's slot.
  int* d_moe_idx_seq_ = nullptr;
  float* d_moe_w_seq_ = nullptr;

  // Grouped-GEMM MoE prefill (CPI_DS_MOE_GROUPED): route on the host, bucket tokens per expert,
  // then one dequant+cuBLAS GEMM per expert instead of a matvec per token. Buffers sized for the
  // worst case P = max_tokens * top_k routings. gate_g doubles as the SiLU-gated inter (in place).
  __half* d_moe_gather_ = nullptr;    // [P, hidden]   gathered token rows, grouped by expert
  __half* d_moe_gate_g_ = nullptr;    // [P, moe_inter] gate GEMM out, then SiLU(gate)*up in place
  __half* d_moe_up_g_ = nullptr;      // [P, moe_inter] up GEMM out
  __half* d_moe_outg_ = nullptr;      // [P, hidden]   per-expert down GEMM out
  __half* d_moe_dqw_ = nullptr;       // [E*2*MI, H] all experts dequanted once (reused for down)
  std::int8_t* d_moe_actq_ = nullptr; // [P, H] int8-quantized grouped activations (int4-direct path)
  float* d_moe_acts_ = nullptr;       // [P, H/32] per-32-group activation scales (int4-direct path)
  int* d_moe_off_dev_ = nullptr;      // [E+1] device copy of per-expert offsets (int4-direct grid.z)
  // cublasGemmGroupedBatchedEx wants the A/B/C pointer arrays in device memory (scalar arrays stay
  // host). Sized 2*E groups (gate_up's worst case). Verified in isolation.
  const void** d_gg_A_ = nullptr;
  const void** d_gg_B_ = nullptr;
  void** d_gg_C_ = nullptr;
  int* d_moe_perm_ = nullptr;         // [P] source token id per grouped row
  float* d_moe_rweight_ = nullptr;    // [P] routing weight per grouped row
  float* d_moe_out_f32_ = nullptr;    // [max_tokens, hidden] fp32 scatter accumulator
  std::vector<int> h_moe_idx_seq_;    // host copy of the [T*top_k] selections for routing
  std::vector<float> h_moe_w_seq_;    // host copy of the [T*top_k] weights
  std::vector<int> h_moe_off_;        // [E+1] grouped-row offset per expert (cached gate_up->down)
  std::vector<int> h_moe_perm_;       // [P] source token id per grouped row (cached gate_up->down)
  // Reused per-call scratch for the MoE grouped path, so each prefill layer refills rather than
  // reallocates (assign/clear keep capacity after the first call).
  std::vector<int> h_moe_cursor_;     // [E] per-expert fill cursor while bucketing
  std::vector<float> h_moe_rweight_;  // [P] routing weight per grouped row
  std::vector<int> gg_ng_;            // [G] tokens per group for grouped_gemm_ex
  std::vector<const void*> gg_pa_, gg_pb_;
  std::vector<void*> gg_pc_;
  // grouped_gemm_ex scalar-array scratch. transa/transb held as int (cublasOperation_t is int-backed;
  // cast at the call site) to keep cublas out of this header.
  std::vector<int> gg_ta_, gg_tb_;
  std::vector<int> gg_ma_, gg_na_, gg_ka_, gg_lda_, gg_ldb_, gg_ldc_, gg_gs_;
  std::vector<float> gg_alpha_, gg_beta_;
  int moe_grouped_total_ = 0;         // P for the current chunk (cached gate_up->down)
  // DeepSeek-V2 MLA working set + rope table.
  __half* d_mla_ckv_ = nullptr;      // [kv_lora + qk_rope]  (kv_a_proj output)
  __half* d_mla_latent_ = nullptr;   // [kv_lora]            (kv_a_layernorm output)
  __half* d_mla_kvb_ = nullptr;      // [nh*(qk_nope+v_head)] (kv_b_proj output)

  // ── Absorbed-MLA decode (DeepSeek): latent cache + absorbed weights ──
  bool mla_absorb_ = false;             // absorbed decode enabled (CPI_DS_NO_ABSORB=1 disables)
  std::vector<__half*> caches_lat_;     // per layer: [max_ctx, kv_lora + qk_rope]
  std::vector<__half*> mla_w_ukt_;      // per layer: [nh][kv_lora][qk_nope] fp16
  std::vector<__half*> mla_w_uv_;       // per layer: [nh][v_head][kv_lora] fp16
  __half* d_mla_qabs_ = nullptr;        // [nh, kv_lora + qk_rope]
  __half* d_mla_attlat_ = nullptr;      // [nh, kv_lora]
  float* d_mla_abs_m_ = nullptr;        // split-K scratch [nh, chunks]
  float* d_mla_abs_l_ = nullptr;
  float* d_mla_abs_o_ = nullptr;        // [nh, chunks, kv_lora]
  int mla_abs_chunks_ = 0;
  float* d_inv_freq_ = nullptr;      // [qk_rope/2]          (YARN inverse frequencies)
  float mla_attn_scaling_ = 1.0f;    // YARN mscale (1.0 for V2-Lite)
  // Expert streaming (CPI_MOE_STREAM): stage only the top_k selected experts into K contiguous
  // slots and feed the kernels an identity index [0..K-1] (so slot k == the k-th selected expert).
  // Step 1 stages from the resident matrix (D2D) to verify the remap is token-identical; Step 2
  // will source experts from host (H2D) so a >VRAM MoE (K3-scale) can run. The router still writes
  // d_moe_idx_ on device; we D2H those K indices to issue the per-expert copies.
  bool moe_stream_ = false;
  bool moe_experts_on_host_ = false;         // step 2: experts migrated to host, streamed H2D
  std::int8_t* d_moe_stage_gu_ = nullptr;    // [K*2*MI, (H+1)/2] int4 gate_up staging
  float* d_moe_stage_gu_s_ = nullptr;        // [K*2*MI, n_groups(H)] scales
  std::int8_t* d_moe_stage_dn_ = nullptr;    // [K*H, (MI+1)/2] int4 down staging
  float* d_moe_stage_dn_s_ = nullptr;        // [K*H, n_groups(MI)] scales
  int* d_moe_identity_idx_ = nullptr;        // [K] = {0,1,...,K-1}
  std::vector<int> h_moe_idx_;               // host scratch for the D2H of d_moe_idx_
  // Stage the top_k selected experts (indices in d_moe_idx_) from a quantized [E*rpe, ...] source
  // into dst_w/dst_s slots [0..K-1]. rpe = rows per expert (2*MI for gate_up, H for down).
  void stage_moe_experts(const void* qw, const float* qs, int rows_per_expert, int in_features,
                         int group, std::int8_t* dst_w, float* dst_s);
  float* d_logits_ = nullptr;    // [vocab]
  int* d_tok_ = nullptr;         // current token id (device); EmbeddingLookup reads it
  int* d_position_ = nullptr;    // current decode position (device); device-pos ops read it
  int* d_argmax_ = nullptr;      // device-argmax result (greedy fast path)
  float* d_argmax_pv_ = nullptr;  // two-phase argmax partition scratch
  int* d_argmax_pi_ = nullptr;
  int argmax_parts_ = 1;
  float* d_mt_logits_ = nullptr;  // [16][vocab] verify-batch logits (lazy, spec decode only)

  // ── Graphed speculative verify: the T=16 verify forward captured once with
  //    device-position seq kernels, replayed per verify with only d_position_ and
  //    d_seq_tokens_ updated. All scratch is lazy (spec decode only). ──
  bool spec_seq_device_pos_ = false;  // execute_ops: seq rope/kv-append/attention take
                                      // their device-position twins (graph-capturable)
  bool spec_graph_broken_ = false;    // init or capture failed once; stay on the eager path
  cudaGraph_t spec_graph_ = nullptr;
  cudaGraphExec_t spec_graph_exec_ = nullptr;
  int spec_attn_chunks_ = 0;          // fixed split-K chunk grid, sized from max_ctx_
  float* d_spec_split_m_ = nullptr;   // [16][heads][chunks] (o adds head_dim)
  float* d_spec_split_l_ = nullptr;
  float* d_spec_split_o_ = nullptr;
  int* d_spec_argmax_ = nullptr;      // [16] per-position verify argmaxes
  bool spec_graph_init();             // alloc scratch (+ capture unless CPI_CUDA_SPEC_GRAPH=0)
  void spec_forward_t16();            // the captured body (also runs eagerly for A/B)

  // ── Per-box tuning knobs. Defaults are the RTX 5090 hand-tuned values; CPI_AUTOTUNE=1
  //    coordinate-descends over real decode-graph replay time and persists the winners per
  //    (GPU, model geometry, quant) so every other box pays the search once. All gemv-warps
  //    variants are bit-identical (row math is warp-local); chunk-size variants regroup the
  //    exact split-K merge, the same tolerance class as the existing depth-adaptive switch. ──
  int tune_gemv_warps_ = 4;        // int4 dp4a wide/cat/glu rows per block (2/4/8)
  int tune_attn_cs_shallow_ = 16;  // split-K chunk size below the deep crossover
  int tune_attn_cs_deep_ = 32;     // ...and at/after it
  int tune_attn_deep_tokens_ = 2048;
  void autotune();                            // search + report (call after open())
  double time_decode_at(int pos, int iters);  // decode-graph replay ms/token at a depth
  std::string tuning_path() const;            // per-(GPU, model, quant) cache file
  bool load_tuning();
  void save_tuning() const;

  // Device top-k sampling (temperature>0, the real chat path, since greedy only
  // covers temp<=0). Selects the candidate set on the GPU so the host never sees
  // the full vocab-sized logit vector; only the ~k candidates come back.
  static constexpr int kMaxDeviceTopK = 256;
  static constexpr int kCandCapacity = kMaxDeviceTopK + 64;  // slack for ties at the k-th logit
  // Persistent decode: the plan compiled to device-side op descriptors, executed as one
  // cooperative launch per token (kernels_persistent_decode.cu). Compiled after build_plan;
  // a plan with kinds the interpreter lacks compiles to "unsupported" and decode keeps the
  // graph path. persist_enabled_ also drops to false if the cooperative launch is rejected.
  bool compile_persistent_plan();
  void* d_persist_ops_ = nullptr;  // kernels::PersistOp[], device
  int persist_n_ops_ = 0;
  int persist_blocks_ = 0;
  bool persist_enabled_ = false;

  // Rowwise-int8 packed copy of the tied embedding table, used by the LM head only on
  // quant runs (EmbeddingLookup keeps reading the fp16 table). nullptr = fp16 head.
  std::int8_t* d_head_q_ = nullptr;
  float* d_head_qs_ = nullptr;
  int head_qbits_ = 0;   // 8 = rowwise int8 head, 4 = grouped int4 head (dp4a f32 path)
  int head_qgroup_ = 0;

  // Decode activation-quant scratch for the dp4a matvec paths (one vector + one scale).
  // int8 runs hold the plain rowwise layout, int4 runs the perm8 layout; a run is one
  // or the other, so the buffer is shared. nullptr = dp4a routing off.
  std::int8_t* d_act_i8_ = nullptr;
  float* d_act_qs_ = nullptr;
  // Prefill dequant scratch: sequence mode runs fp16 GEMMs over a dequantized copy of
  // each quantized matrix (sized for the largest projection). nullptr on fp16 runs.
  __half* d_seq_dequant_ = nullptr;
  // int8-direct prefill scratch: per-token quantized activations + int32 GEMM output.
  std::int8_t* d_seq_x8_ = nullptr;
  float* d_seq_xs_ = nullptr;
  int* d_seq_i32_ = nullptr;
  // cuBLAS, for the sequence-mode GEMMs only (precedent: llama4 engine + bert embedder);
  // the hand-rolled GEMM cannot reach tensor-core rates. Decode never touches this.
  // CPI_CUDA_NO_CUBLAS=1 restores the fallback.
  void* cublas_ = nullptr;  // cublasHandle_t, kept as void* to keep cublas out of the header
  void* d_cublas_ws_ = nullptr;  // persistent cuBLAS workspace (set once; avoids per-call alloc)
  // Tensor-core attention prefill (full-attention layers): scores scratch + device
  // pointer arrays, the LlamaEngine machinery adapted. Lazy-allocated on first use.
  __half* d_pf_scores_ = nullptr;
  void** d_pf_ptrs_ = nullptr;
  bool prefill_attention_tc(const __half* q, const __half* k, const __half* v, __half* out,
                            int rows, int base_pos, int heads, int kv_heads, int hd,
                            int window = 0);

  // Split-K decode-attention scratch for the wide-head (any-head_dim) path. Sized
  // num_heads x split_any_chunks_ (x maxhd for the partial outputs) in allocate_buffers.
  static constexpr int kSplitAnyChunk = 16;
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
  bool device_topk_enabled_ = true;   // env CPI_PLAN_NO_DEVICE_TOPK=1 disables (A/B)
  // When true (the shared decode-loop path), forward_one leaves the logits on the
  // device (d_logits_) instead of copying 262K floats to host + softcapping every
  // step; sample() then does the device argmax (greedy) or a lazy host copy.
  bool defer_host_logits_ = false;
  // When true, execute_ops uses the device-position kernel variants (RoPE / KV
  // store / attention read d_position_) so the op sequence is CUDA-graph capturable.
  bool device_pos_mode_ = false;

  // Decode CUDA graph: the single-token forward captured once (device-position
  // ops), replayed for every logits step. decode_graph_enabled_ off falls back to
  // the eager path (env CPI_PLAN_NO_GRAPH=1, for A/B / debugging).
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
