// Gemma 4 (E2B) CUDA inference engine — a dedicated fork (not a LlamaEngine
// absorb), token-at-a-time, correctness-first. Implements the MatFormer text
// tower: Per-Layer Embeddings, dual head_dim (256 sliding / 512 full), partial +
// dual RoPE, weightless V-norm, QK-norm, sandwich norms, KV-cache sharing,
// per-layer scalar, GeGLU (double-wide on shared layers), final logit softcap.
// See memory: cpi-gemma4-arch for the full spec + parity oracle.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/decode_driver.hpp"
#include "engine/engine_types.hpp"

namespace engine {

struct GenerationConstraints;  // fwd (generation_constraints.hpp)

// Derives from runtime::SequenceModel so generate/generate_stream reuse the
// shared decode driver (loop + sampler + stops) instead of a bespoke greedy loop.
class Gemma4CudaEngine : public runtime::SequenceModel {
 public:
  Gemma4CudaEngine() = default;
  ~Gemma4CudaEngine();
  Gemma4CudaEngine(const Gemma4CudaEngine&) = delete;
  Gemma4CudaEngine& operator=(const Gemma4CudaEngine&) = delete;

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

  int bos_id() const { return cfg_.bos_token_id; }

  // --- runtime::SequenceModel (the decode driver calls these) ---
  int vocab() const override { return cfg_.vocab; }
  int eos_id() const override { return cfg_.eos_token_id; }
  int max_context() const override { return max_ctx_; }
  void step(int token, int position, bool want_logits) override {
    forward_one(token, position, want_logits);
  }
  std::vector<float>& logits() override { return last_logits_; }

 private:
  struct Config {
    int num_layers = 0, hidden = 0, num_heads = 0, num_kv_heads = 1;
    int head_dim = 256, global_head_dim = 512, intermediate = 0, vocab = 0;
    // Actual per-layer-type head_dim from the weights (E2B: 256/512; 12B: 256/512).
    int head_dim_sliding = 256, head_dim_full = 512;
    // num_kv_heads can differ per layer type (12B: sliding GQA-8, full MQA-1).
    int num_kv_heads_sliding = 1, num_kv_heads_full = 1;
    bool attention_k_eq_v = false;  // 12B full layers: V shares k_proj (no v_proj)
    int hidden_size_per_layer_input = 256, num_kv_shared_layers = 0, first_shared_layer = 0;
    int sliding_window = 0, bos_token_id = 2, eos_token_id = 1;
    float rms_eps = 1e-6f, final_logit_softcapping = 0.0f;
    float rope_theta_full = 1e6f, rope_theta_sliding = 1e4f, partial_rotary_full = 0.25f;
    bool use_double_wide_mlp = false, tie_word_embeddings = true;
    bool enable_moe_block = false;  // 26B-A4B: dense-MLP + top-k experts (not yet run)
    int num_experts = 0, top_k_experts = 0, moe_intermediate_size = 0;
    std::vector<int> layer_full;    // 1 if full_attention, 0 if sliding
    std::vector<int> kv_source;     // which layer's K/V each layer uses
  };

  struct TensorMeta {
    std::string dtype;
    std::vector<int> shape;
    std::size_t offset = 0, bytes = 0;
  };

  void parse_manifest(const std::string& manifest_path);
  void load_all(const std::string& cpi_path);
  __half* upload(const std::string& name);                 // load a tensor to device (fp16)
  float scalar_value(const std::string& name);             // read a [1] tensor to host
  void build_rope_tables();
  void allocate_buffers();
  void run_layer(int layer, int position);
  void build_per_layer_inputs(int token);
  // Process one token at `position` (reusing the KV cache from prior positions).
  // When compute_logits, fills last_logits_ with softcapped logits. When
  // per_layer_rms != nullptr, appends each layer's output RMS (and dumps the
  // hidden to $G4_DUMP_DIR if set) — the oracle parity path.
  void forward_one(int token, int position, bool compute_logits,
                   std::vector<float>* per_layer_rms = nullptr);

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
  float* d_logits_ = nullptr;    // [vocab]
};

}  // namespace engine
