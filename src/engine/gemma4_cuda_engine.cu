// Gemma 4 (E2B) CUDA engine — token-at-a-time forward. See the header and
// memory:cpi-gemma4-arch for the architecture spec. Reuses the shared kernels
// (rmsnorm scale=w, rope table, tiled decode attention, gelu-mul, gemv).
#include "engine/gemma4_cuda_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "runtime/kernels.cuh"

namespace engine {

namespace {
#define G4_CHECK(x)                                                                        \
  do {                                                                                     \
    cudaError_t _e = (x);                                                                  \
    if (_e != cudaSuccess)                                                                 \
      throw std::runtime_error(std::string("cuda error ") + cudaGetErrorString(_e) + " @" \
                               + __FILE__ + ":" + std::to_string(__LINE__));               \
  } while (0)

// the .cpi stores F16, so host reads are raw uint16 halves.
std::vector<__half> read_fp16(std::ifstream& f, std::size_t data_start, std::size_t off,
                              std::size_t bytes) {
  std::vector<__half> buf(bytes / sizeof(__half));
  f.seekg(static_cast<std::streamoff>(data_start + off));
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes));
  if (!f) throw std::runtime_error("failed reading tensor bytes from .cpi");
  return buf;
}
}  // namespace

Gemma4CudaEngine::~Gemma4CudaEngine() {
  for (auto& kv : dev_) cudaFree(kv.second);
  // shared-layer caches are aliases; free only the owning (< first_shared) ones.
  for (int L = 0; L < cfg_.first_shared_layer && L < static_cast<int>(caches_k_.size()); ++L) {
    cudaFree(caches_k_[L]);
    cudaFree(caches_v_[L]);
  }
  cudaFree(d_cos_sliding_); cudaFree(d_sin_sliding_);
  cudaFree(d_cos_full_); cudaFree(d_sin_full_);
  cudaFree(d_ones_);
  cudaFree(d_x_); cudaFree(d_x_norm_); cudaFree(d_resid_); cudaFree(d_tmp_);
  cudaFree(d_q_); cudaFree(d_k_); cudaFree(d_v_); cudaFree(d_att_);
  cudaFree(d_gate_); cudaFree(d_up_); cudaFree(d_inter_);
  cudaFree(d_ple_raw_); cudaFree(d_ple_); cudaFree(d_ple_gate_);
  cudaFree(d_logits_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Gemma4CudaEngine::parse_manifest(const std::string& manifest_path) {
  std::ifstream f(manifest_path);
  if (!f) throw std::runtime_error("cannot open manifest: " + manifest_path);
  auto get_int_list = [](const std::string& js) {
    std::vector<int> out;
    std::string cur;
    for (char c : js) {
      if (c == '-' || (c >= '0' && c <= '9')) cur += c;
      else { if (!cur.empty()) { out.push_back(std::stoi(cur)); cur.clear(); } }
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
  };
  std::string line;
  std::vector<std::string> layer_types;  // filled from CFGJSON layer_types
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind == "DATA_START") {
      ss >> data_start_;
    } else if (kind == "CFG") {
      std::string k, v;
      ss >> k >> v;
      if (k == "num_layers") cfg_.num_layers = std::stoi(v);
      else if (k == "hidden") cfg_.hidden = std::stoi(v);
      else if (k == "num_heads") cfg_.num_heads = std::stoi(v);
      else if (k == "num_kv_heads") cfg_.num_kv_heads = std::stoi(v);
      else if (k == "head_dim") cfg_.head_dim = std::stoi(v);
      else if (k == "global_head_dim") cfg_.global_head_dim = std::stoi(v);
      else if (k == "intermediate") cfg_.intermediate = std::stoi(v);
      else if (k == "vocab") cfg_.vocab = std::stoi(v);
      else if (k == "rms_eps") cfg_.rms_eps = std::stof(v);
      else if (k == "sliding_window") cfg_.sliding_window = std::stoi(v);
      else if (k == "final_logit_softcapping") cfg_.final_logit_softcapping = std::stof(v);
      else if (k == "hidden_size_per_layer_input") cfg_.hidden_size_per_layer_input = std::stoi(v);
      else if (k == "num_kv_shared_layers") cfg_.num_kv_shared_layers = std::stoi(v);
      else if (k == "first_shared_layer") cfg_.first_shared_layer = std::stoi(v);
      else if (k == "rope_theta_full") cfg_.rope_theta_full = std::stof(v);
      else if (k == "rope_theta_sliding") cfg_.rope_theta_sliding = std::stof(v);
      else if (k == "partial_rotary_full") cfg_.partial_rotary_full = std::stof(v);
      else if (k == "bos_token_id") cfg_.bos_token_id = std::stoi(v);
      else if (k == "eos_token_id") cfg_.eos_token_id = std::stoi(v);
      else if (k == "use_double_wide_mlp") cfg_.use_double_wide_mlp = (v == "True" || v == "true" || v == "1");
      else if (k == "enable_moe_block") cfg_.enable_moe_block = (v == "True" || v == "true" || v == "1");
      else if (k == "num_experts") cfg_.num_experts = std::stoi(v);
      else if (k == "top_k_experts") cfg_.top_k_experts = std::stoi(v);
      else if (k == "moe_intermediate_size") cfg_.moe_intermediate_size = std::stoi(v);
      else if (k == "tie_word_embeddings") cfg_.tie_word_embeddings = (v == "True" || v == "true" || v == "1");
    } else if (kind == "CFGJSON") {
      std::string k;
      ss >> k;
      std::string rest;
      std::getline(ss, rest);
      if (k == "kv_source") cfg_.kv_source = get_int_list(rest);
      else if (k == "layer_types") {
        // parse quoted strings; mark 1 for full_attention
        cfg_.layer_full.clear();
        std::size_t p = 0;
        while ((p = rest.find('"', p)) != std::string::npos) {
          std::size_t q = rest.find('"', p + 1);
          std::string s = rest.substr(p + 1, q - p - 1);
          cfg_.layer_full.push_back(s == "full_attention" ? 1 : 0);
          p = q + 1;
        }
      }
    } else if (kind == "TENSOR") {
      TensorMeta m;
      std::string name, shp;
      std::size_t bytes;
      ss >> name >> m.dtype >> shp >> m.offset >> bytes;
      m.bytes = bytes;
      std::string cur;
      for (char c : shp) { if (c == ',') { m.shape.push_back(std::stoi(cur)); cur.clear(); } else cur += c; }
      if (!cur.empty()) m.shape.push_back(std::stoi(cur));
      meta_[name] = m;
    }
  }
  if (cfg_.kv_source.empty())
    for (int i = 0; i < cfg_.num_layers; ++i) cfg_.kv_source.push_back(i);
}

__half* Gemma4CudaEngine::upload(const std::string& name) {
  auto it = meta_.find(name);
  if (it == meta_.end()) throw std::runtime_error("missing tensor: " + name);
  std::ifstream f(cpi_path_, std::ios::binary);
  auto host = read_fp16(f, data_start_, it->second.offset, it->second.bytes);
  __half* d = nullptr;
  G4_CHECK(cudaMalloc(&d, host.size() * sizeof(__half)));
  G4_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice));
  dev_[name] = d;
  return d;
}

float Gemma4CudaEngine::scalar_value(const std::string& name) {
  auto it = meta_.find(name);
  if (it == meta_.end()) throw std::runtime_error("missing scalar: " + name);
  std::ifstream f(cpi_path_, std::ios::binary);
  auto host = read_fp16(f, data_start_, it->second.offset, it->second.bytes);
  return __half2float(host[0]);
}

void Gemma4CudaEngine::build_rope_tables() {
  const int hs = cfg_.head_dim / 2;       // sliding pairs (128)
  const int hf = cfg_.global_head_dim / 2; // full pairs (256)
  std::vector<float> cs(static_cast<std::size_t>(max_ctx_) * hs), ss(cs.size());
  std::vector<float> cf(static_cast<std::size_t>(max_ctx_) * hf), sf(cf.size());
  // sliding: full rotary, inv_freq[i] = theta^(-2i/head_dim)
  for (int p = 0; p < max_ctx_; ++p)
    for (int i = 0; i < hs; ++i) {
      float inv = std::pow(cfg_.rope_theta_sliding, -2.0f * i / cfg_.head_dim);
      cs[static_cast<std::size_t>(p) * hs + i] = std::cos(p * inv);
      ss[static_cast<std::size_t>(p) * hs + i] = std::sin(p * inv);
    }
  // full: partial rotary — first rope_angles pairs rotated, rest identity (inv=0)
  const int rope_angles = static_cast<int>(cfg_.partial_rotary_full * cfg_.global_head_dim) / 2;
  for (int p = 0; p < max_ctx_; ++p)
    for (int i = 0; i < hf; ++i) {
      float inv = (i < rope_angles) ? std::pow(cfg_.rope_theta_full, -2.0f * i / cfg_.global_head_dim) : 0.0f;
      cf[static_cast<std::size_t>(p) * hf + i] = std::cos(p * inv);
      sf[static_cast<std::size_t>(p) * hf + i] = std::sin(p * inv);
    }
  auto up = [&](const std::vector<float>& v, float** d) {
    G4_CHECK(cudaMalloc(d, v.size() * sizeof(float)));
    G4_CHECK(cudaMemcpy(*d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  };
  up(cs, &d_cos_sliding_); up(ss, &d_sin_sliding_);
  up(cf, &d_cos_full_); up(sf, &d_sin_full_);
}

void Gemma4CudaEngine::allocate_buffers() {
  const int H = cfg_.hidden;
  const int maxhd = cfg_.global_head_dim;
  const int maxq = cfg_.num_heads * maxhd;
  const int maxkv = cfg_.num_kv_heads * maxhd;
  const int maxinter = cfg_.intermediate * (cfg_.use_double_wide_mlp ? 2 : 1);
  const int ple_tot = cfg_.num_layers * cfg_.hidden_size_per_layer_input;
  auto al = [&](__half** p, std::size_t n) { G4_CHECK(cudaMalloc(p, n * sizeof(__half))); };
  al(&d_x_, H); al(&d_x_norm_, H); al(&d_resid_, H); al(&d_tmp_, H);
  al(&d_q_, maxq); al(&d_k_, maxkv); al(&d_v_, maxkv); al(&d_att_, maxq);
  al(&d_gate_, maxinter); al(&d_up_, maxinter); al(&d_inter_, maxinter);
  al(&d_ple_raw_, ple_tot); al(&d_ple_, ple_tot); al(&d_ple_gate_, cfg_.hidden_size_per_layer_input);
  G4_CHECK(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg_.vocab) * sizeof(float)));
  // ones for weightless v-norm
  std::vector<__half> ones(maxhd, __float2half(1.0f));
  G4_CHECK(cudaMalloc(&d_ones_, ones.size() * sizeof(__half)));
  G4_CHECK(cudaMemcpy(d_ones_, ones.data(), ones.size() * sizeof(__half), cudaMemcpyHostToDevice));

  // per-layer K/V caches: own for L < first_shared, alias for shared.
  // Own K/V cache for each non-shared layer; shared layers (>= first_shared)
  // alias the cache of their kv_source (they reuse that layer's K/V, per HF's
  // num_kv_shared_layers). Aliases are not freed in the destructor.
  caches_k_.assign(cfg_.num_layers, nullptr);
  caches_v_.assign(cfg_.num_layers, nullptr);
  for (int L = 0; L < cfg_.num_layers; ++L) {
    if (L < cfg_.first_shared_layer || cfg_.first_shared_layer == 0) {
      const int kvd = head_dim_of(L) * cfg_.num_kv_heads;
      G4_CHECK(cudaMalloc(&caches_k_[L], static_cast<std::size_t>(max_ctx_) * kvd * sizeof(__half)));
      G4_CHECK(cudaMalloc(&caches_v_[L], static_cast<std::size_t>(max_ctx_) * kvd * sizeof(__half)));
    }
  }
  for (int L = cfg_.first_shared_layer; L < cfg_.num_layers && cfg_.first_shared_layer > 0; ++L) {
    int src = cfg_.kv_source[L];
    caches_k_[L] = caches_k_[src];
    caches_v_[L] = caches_v_[src];
  }
}

void Gemma4CudaEngine::load_all(const std::string& cpi_path) {
  cpi_path_ = cpi_path;
  // large tables + per-layer weights to device
  upload("embed_tokens.weight");
  upload("norm.weight");
  if (has_ple()) {  // E2B has Per-Layer Embeddings; 12B does not
    upload("embed_tokens_per_layer.weight");
    upload("per_layer_model_projection.weight");
    upload("per_layer_projection_norm.weight");
  }
  layer_scalar_host_.assign(cfg_.num_layers, 1.0f);
  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = "layers." + std::to_string(L) + ".";
    for (const char* t : {"input_layernorm.weight", "post_attention_layernorm.weight",
                          "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                          "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                          "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                          "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"})
      upload(p + t);
    if (has_ple())
      for (const char* t : {"per_layer_input_gate.weight", "per_layer_projection.weight",
                            "post_per_layer_input_norm.weight"})
        upload(p + t);
    layer_scalar_host_[L] = scalar_value(p + "layer_scalar");
  }
}

void Gemma4CudaEngine::open(const std::string& cpi_path, int max_context) {
  max_ctx_ = max_context;
  std::string manifest = cpi_path.substr(0, cpi_path.find_last_of('.')) + ".manifest";
  parse_manifest(manifest);
  if (static_cast<int>(cfg_.layer_full.size()) != cfg_.num_layers)
    throw std::runtime_error("layer_types count mismatch");
  if (cfg_.enable_moe_block)
    throw std::runtime_error(
        "Gemma 4 MoE (e.g. 26B-A4B, " + std::to_string(cfg_.num_experts) +
        " experts) is not yet runnable: the fp16 experts exceed 32GB VRAM, so it "
        "needs int4 weights + expert streaming. The forward math is specced in "
        "memory:cpi-gemma4-arch (dense-MLP + top-k experts, summed).");
  G4_CHECK(cudaStreamCreate(&stream_));
  load_all(cpi_path);
  build_rope_tables();
  allocate_buffers();
  G4_CHECK(cudaStreamSynchronize(stream_));
}

void Gemma4CudaEngine::build_per_layer_inputs(int token) {
  const int ple = cfg_.hidden_size_per_layer_input;
  const int tot = cfg_.num_layers * ple;
  static int* d_tok = nullptr;
  if (!d_tok) G4_CHECK(cudaMalloc(&d_tok, sizeof(int)));
  G4_CHECK(cudaMemcpyAsync(d_tok, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
  // ple_raw = embed_tokens_per_layer[token] * sqrt(ple)
  kernels::launch_embedding_lookup(static_cast<const __half*>(dev_["embed_tokens_per_layer.weight"]),
                                   d_tok, d_ple_raw_, 1, tot, stream_);
  kernels::launch_scale_copy(d_ple_raw_, d_ple_raw_, tot, std::sqrt((float)ple), stream_);
  // ple_proj = rmsnorm( (W_proj · embeds) * hidden^-0.5 )  [d_x_ currently = embeds]
  kernels::launch_rowmajor_half_gemv_f16(
      static_cast<const __half*>(dev_["per_layer_model_projection.weight"]), d_x_, d_ple_, tot,
      cfg_.hidden, stream_);
  kernels::launch_scale_copy(d_ple_, d_ple_, tot, std::pow((float)cfg_.hidden, -0.5f), stream_);
  kernels::launch_rmsnorm(d_ple_, static_cast<const __half*>(dev_["per_layer_projection_norm.weight"]),
                          d_ple_, cfg_.num_layers, ple, cfg_.rms_eps, stream_);
  // per_layer_inputs = (proj + raw) * 2^-0.5
  kernels::launch_add_inplace(d_ple_, d_ple_raw_, tot, stream_);
  kernels::launch_scale_copy(d_ple_, d_ple_, tot, std::pow(2.0f, -0.5f), stream_);
}

void Gemma4CudaEngine::run_layer(int layer, int position) {
  const std::string p = "layers." + std::to_string(layer) + ".";
  const int H = cfg_.hidden;
  const int hd = head_dim_of(layer);
  const int nq = cfg_.num_heads, nkv = cfg_.num_kv_heads;
  const int qdim = nq * hd, kvdim = nkv * hd;
  const bool full = cfg_.layer_full[layer] != 0;
  const bool shared = layer >= cfg_.first_shared_layer && cfg_.first_shared_layer > 0;
  const float* cosT = full ? d_cos_full_ : d_cos_sliding_;
  const float* sinT = full ? d_sin_full_ : d_sin_sliding_;
  auto W = [&](const char* t) { return static_cast<const __half*>(dev_[p + t]); };

  // --- attention block ---
  kernels::launch_rmsnorm(d_x_, W("input_layernorm.weight"), d_x_norm_, 1, H, cfg_.rms_eps, stream_);
  kernels::launch_rowmajor_half_gemv_f16(W("self_attn.q_proj.weight"), d_x_norm_, d_q_, qdim, H, stream_);
  kernels::launch_rmsnorm(d_q_, W("self_attn.q_norm.weight"), d_q_, nq, hd, cfg_.rms_eps, stream_);
  kernels::launch_rope_inplace_table(d_q_, d_q_, nq, 0, hd, position, cosT, sinT, stream_);

  __half* kc = caches_k_[layer];
  __half* vc = caches_v_[layer];
  if (!shared) {
    kernels::launch_rowmajor_half_gemv_f16(W("self_attn.k_proj.weight"), d_x_norm_, d_k_, kvdim, H, stream_);
    kernels::launch_rowmajor_half_gemv_f16(W("self_attn.v_proj.weight"), d_x_norm_, d_v_, kvdim, H, stream_);
    kernels::launch_rmsnorm(d_k_, W("self_attn.k_norm.weight"), d_k_, nkv, hd, cfg_.rms_eps, stream_);
    kernels::launch_rope_inplace_table(d_k_, d_k_, nkv, 0, hd, position, cosT, sinT, stream_);
    kernels::launch_rmsnorm(d_v_, d_ones_, d_v_, nkv, hd, cfg_.rms_eps, stream_);  // weightless v-norm
    G4_CHECK(cudaMemcpyAsync(kc + static_cast<std::size_t>(position) * kvdim, d_k_,
                             kvdim * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
    G4_CHECK(cudaMemcpyAsync(vc + static_cast<std::size_t>(position) * kvdim, d_v_,
                             kvdim * sizeof(__half), cudaMemcpyDeviceToDevice, stream_));
  }
  // net attention scale = 1.0: pre-scale q by sqrt(hd) to cancel the kernel's 1/sqrt(hd).
  kernels::launch_scale_copy(d_q_, d_q_, qdim, std::sqrt((float)hd), stream_);
  // sliding window: attend to the last `sliding_window` keys only (full layers: all).
  int k_start = 0;
  int seq = position + 1;
  if (!full && cfg_.sliding_window > 0 && seq > cfg_.sliding_window)
    k_start = seq - cfg_.sliding_window;
  const int att_seq = seq - k_start;
  kernels::launch_attention_step(d_q_, kc + static_cast<std::size_t>(k_start) * kvdim,
                                 vc + static_cast<std::size_t>(k_start) * kvdim, d_att_, att_seq,
                                 nq, nkv, hd, stream_);
  kernels::launch_rowmajor_half_gemv_f16(W("self_attn.o_proj.weight"), d_att_, d_tmp_, H, qdim, stream_);
  kernels::launch_rmsnorm(d_tmp_, W("post_attention_layernorm.weight"), d_tmp_, 1, H, cfg_.rms_eps, stream_);
  kernels::launch_add_inplace(d_x_, d_tmp_, H, stream_);

  // --- MLP block (GeGLU; double-wide on shared layers) ---
  const int inter = cfg_.intermediate * ((cfg_.use_double_wide_mlp && shared) ? 2 : 1);
  kernels::launch_rmsnorm(d_x_, W("pre_feedforward_layernorm.weight"), d_x_norm_, 1, H, cfg_.rms_eps, stream_);
  kernels::launch_rowmajor_half_gemv_f16(W("mlp.gate_proj.weight"), d_x_norm_, d_gate_, inter, H, stream_);
  kernels::launch_rowmajor_half_gemv_f16(W("mlp.up_proj.weight"), d_x_norm_, d_up_, inter, H, stream_);
  kernels::launch_gelu_mul(d_gate_, d_up_, d_inter_, inter, stream_);
  kernels::launch_rowmajor_half_gemv_f16(W("mlp.down_proj.weight"), d_inter_, d_tmp_, H, inter, stream_);
  kernels::launch_rmsnorm(d_tmp_, W("post_feedforward_layernorm.weight"), d_tmp_, 1, H, cfg_.rms_eps, stream_);
  kernels::launch_add_inplace(d_x_, d_tmp_, H, stream_);

  // --- Per-Layer Input injection (only when the model has PLE) ---
  if (has_ple()) {
    const int ple = cfg_.hidden_size_per_layer_input;
    const __half* ple_L = d_ple_ + static_cast<std::size_t>(layer) * ple;
    kernels::launch_rowmajor_half_gemv_f16(W("per_layer_input_gate.weight"), d_x_, d_ple_gate_, ple, H, stream_);
    kernels::launch_gelu_mul(d_ple_gate_, ple_L, d_ple_gate_, ple, stream_);  // gelu(gate)*ple_L
    kernels::launch_rowmajor_half_gemv_f16(W("per_layer_projection.weight"), d_ple_gate_, d_tmp_, H, ple, stream_);
    kernels::launch_rmsnorm(d_tmp_, W("post_per_layer_input_norm.weight"), d_tmp_, 1, H, cfg_.rms_eps, stream_);
    kernels::launch_add_inplace(d_x_, d_tmp_, H, stream_);
  }

  // --- per-layer scalar ---
  kernels::launch_scale_copy(d_x_, d_x_, H, layer_scalar_host_[layer], stream_);
}

// Process one token at `position`, reusing the KV cache from earlier positions.
// Positions must be presented in increasing order (prefill then decode) so the
// cache and the KV-share sources are populated before dependent layers read them.
void Gemma4CudaEngine::forward_one(int token, int position, bool compute_logits) {
  const int H = cfg_.hidden;
  static int* d_tok = nullptr;
  if (!d_tok) G4_CHECK(cudaMalloc(&d_tok, sizeof(int)));
  G4_CHECK(cudaMemcpyAsync(d_tok, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
  kernels::launch_embedding_lookup(static_cast<const __half*>(dev_["embed_tokens.weight"]), d_tok,
                                   d_x_, 1, H, stream_);
  kernels::launch_scale_copy(d_x_, d_x_, H, std::sqrt((float)H), stream_);  // embed * sqrt(hidden)
  if (has_ple()) build_per_layer_inputs(token);  // uses d_x_ (= embeds) before layers mutate it
  for (int L = 0; L < cfg_.num_layers; ++L) run_layer(L, position);
  if (!compute_logits) return;
  kernels::launch_rmsnorm(d_x_, static_cast<const __half*>(dev_["norm.weight"]), d_x_norm_, 1, H,
                          cfg_.rms_eps, stream_);
  kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(dev_["embed_tokens.weight"]),
                                         d_x_norm_, d_logits_, cfg_.vocab, H, stream_);
  last_logits_.resize(cfg_.vocab);
  G4_CHECK(cudaStreamSynchronize(stream_));
  G4_CHECK(cudaMemcpy(last_logits_.data(), d_logits_, cfg_.vocab * sizeof(float),
                      cudaMemcpyDeviceToHost));
  const float cap = cfg_.final_logit_softcapping;
  if (cap > 0.0f)
    for (float& v : last_logits_) v = cap * std::tanh(v / cap);
}

int Gemma4CudaEngine::argmax_last() const {
  int best = 0;
  for (int i = 1; i < static_cast<int>(last_logits_.size()); ++i)
    if (last_logits_[i] > last_logits_[best]) best = i;
  return best;
}

std::vector<float> Gemma4CudaEngine::forward_logits(const std::vector<int>& tokens,
                                                    std::vector<float>* per_layer_rms) {
  // Parity/debug entry: fresh sequence from position 0. When per_layer_rms is
  // requested, capture the last token's per-layer hidden (oracle comparison).
  const int H = cfg_.hidden;
  static int* d_tok = nullptr;
  if (!d_tok) G4_CHECK(cudaMalloc(&d_tok, sizeof(int)));
  for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
    const int tok = tokens[pos];
    const bool last = pos == static_cast<int>(tokens.size()) - 1;
    G4_CHECK(cudaMemcpyAsync(d_tok, &tok, sizeof(int), cudaMemcpyHostToDevice, stream_));
    kernels::launch_embedding_lookup(static_cast<const __half*>(dev_["embed_tokens.weight"]), d_tok,
                                     d_x_, 1, H, stream_);
    kernels::launch_scale_copy(d_x_, d_x_, H, std::sqrt((float)H), stream_);
    if (has_ple()) build_per_layer_inputs(tok);
    for (int L = 0; L < cfg_.num_layers; ++L) {
      run_layer(L, pos);
      if (last && per_layer_rms) {
        std::vector<__half> h(H);
        G4_CHECK(cudaStreamSynchronize(stream_));
        G4_CHECK(cudaMemcpy(h.data(), d_x_, H * sizeof(__half), cudaMemcpyDeviceToHost));
        double ss = 0;
        for (auto v : h) { float fv = __half2float(v); ss += (double)fv * fv; }
        per_layer_rms->push_back((float)std::sqrt(ss / H));
        if (const char* dir = std::getenv("G4_DUMP_DIR")) {
          std::vector<float> hf(H);
          for (int i = 0; i < H; ++i) hf[i] = __half2float(h[i]);
          std::string fn = std::string(dir) + "/cpi_layer_" + std::to_string(L) + ".f32";
          std::FILE* fp = std::fopen(fn.c_str(), "wb");
          if (fp) { std::fwrite(hf.data(), sizeof(float), H, fp); std::fclose(fp); }
        }
      }
    }
    if (last) {
      kernels::launch_rmsnorm(d_x_, static_cast<const __half*>(dev_["norm.weight"]), d_x_norm_, 1, H,
                              cfg_.rms_eps, stream_);
      kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(dev_["embed_tokens.weight"]),
                                             d_x_norm_, d_logits_, cfg_.vocab, H, stream_);
      last_logits_.resize(cfg_.vocab);
      G4_CHECK(cudaStreamSynchronize(stream_));
      G4_CHECK(cudaMemcpy(last_logits_.data(), d_logits_, cfg_.vocab * sizeof(float),
                          cudaMemcpyDeviceToHost));
      const float cap = cfg_.final_logit_softcapping;
      if (cap > 0.0f)
        for (float& v : last_logits_) v = cap * std::tanh(v / cap);
    }
  }
  return last_logits_;
}

void Gemma4CudaEngine::initialize(const EngineOptions& options) {
  // Keep the eos from the manifest (Gemma <eos>=1); the CLI's --eos default (2)
  // would otherwise clobber it. Turn-level stops (<end_of_turn>) come via stop
  // texts. Cap the context to what the caller requested (default 2048).
  open(options.model_path, options.max_context > 0 ? options.max_context : 4096);
}

std::vector<int> Gemma4CudaEngine::generate_stream(const std::vector<int>& prompt, int max_new,
                                                   float /*temperature*/,
                                                   const std::function<bool(int)>& on_token,
                                                   const GenerationConstraints* /*constraints*/) {
  // Incremental: prefill the prompt once (O(n)), then decode one token at a time
  // appending to the KV cache (greedy). Position advances monotonically so the
  // KV-share sources stay populated.
  std::vector<int> out;
  const int P = static_cast<int>(prompt.size());
  if (P == 0) return out;
  stats_ = BenchmarkStats{};
  stats_.prompt_tokens = P;
  for (int i = 0; i < P; ++i) forward_one(prompt[i], i, i == P - 1);
  int pos = P;
  for (int step = 0; step < max_new && pos < max_ctx_; ++step) {
    const int next = argmax_last();
    out.push_back(next);
    stats_.generated_tokens++;
    if (on_token && !on_token(next)) break;
    if (next == cfg_.eos_token_id) break;
    forward_one(next, pos, true);
    ++pos;
  }
  return out;
}

std::vector<int> Gemma4CudaEngine::generate(const std::vector<int>& prompt, int max_new,
                                            float temperature) {
  return generate_stream(prompt, max_new, temperature, nullptr, nullptr);
}

std::vector<std::pair<int, float>> Gemma4CudaEngine::inspect_next_logits(
    const std::vector<int>& prompt, int top_k) {
  for (int i = 0; i < static_cast<int>(prompt.size()); ++i)
    forward_one(prompt[i], i, i == static_cast<int>(prompt.size()) - 1);
  std::vector<int> idx(last_logits_.size());
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) idx[i] = i;
  const int k = std::min(top_k, static_cast<int>(idx.size()));
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return last_logits_[a] > last_logits_[b]; });
  std::vector<std::pair<int, float>> out;
  for (int i = 0; i < k; ++i) out.emplace_back(idx[i], last_logits_[idx[i]]);
  return out;
}

}  // namespace engine
