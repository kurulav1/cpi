// Generic per-layer op-plan CUDA executor — token-at-a-time forward driven by a
// data op-plan (include/engine/op_plan.hpp), not model-specific control flow.
// Gemma 4 is currently its sole tenant; see memory:cpi-gemma4-arch for that
// model's spec. Reuses the shared kernels (rmsnorm scale=w, rope table, tiled
// decode attention, gelu-mul, gemv).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "engine/generation_constraints.hpp"
#include "engine/plan_cuda_engine.hpp"
#include "engine/sampling.hpp"
#include "model/json_mini.hpp"
#include "runtime/kernels.cuh"

namespace engine {

namespace {
#define G4_CHECK(x)                                                                         \
  do {                                                                                      \
    cudaError_t _e = (x);                                                                   \
    if (_e != cudaSuccess)                                                                  \
      throw std::runtime_error(std::string("cuda error ") + cudaGetErrorString(_e) + " @" + \
                               __FILE__ + ":" + std::to_string(__LINE__));                  \
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

PlanCudaEngine::~PlanCudaEngine() {
  for (auto& kv : dev_) cudaFree(kv.second);
  for (auto& kv : qdev_) {
    cudaFree(kv.second.packed);
    cudaFree(kv.second.scales);
  }
  for (auto& kv : devf_) cudaFree(kv.second);
  cudaFree(d_q_pair_);
  cudaFree(d_q_gate_);
  cudaFree(d_lin_mix_);
  cudaFree(d_lin_z_);
  cudaFree(d_lin_a_);
  cudaFree(d_lin_b_);
  cudaFree(d_lin_q_);
  cudaFree(d_lin_k_);
  cudaFree(d_lin_v_);
  cudaFree(d_lin_att_);
  cudaFree(d_lin_conv_state_);
  cudaFree(d_lin_recurrent_state_);
  // shared-layer caches are aliases; free only the owning (< first_shared) ones.
  for (int L = 0; L < cfg_.first_shared_layer && L < static_cast<int>(caches_k_.size()); ++L) {
    cudaFree(caches_k_[L]);
    cudaFree(caches_v_[L]);
  }
  cudaFree(d_cos_sliding_);
  cudaFree(d_sin_sliding_);
  cudaFree(d_cos_full_);
  cudaFree(d_sin_full_);
  cudaFree(d_ones_);
  cudaFree(d_x_);
  cudaFree(d_x_norm_);
  cudaFree(d_tmp_);
  cudaFree(d_q_);
  cudaFree(d_k_);
  cudaFree(d_v_);
  cudaFree(d_att_);
  cudaFree(d_gate_);
  cudaFree(d_up_);
  cudaFree(d_inter_);
  cudaFree(d_ple_raw_);
  cudaFree(d_ple_);
  cudaFree(d_ple_gate_);
  cudaFree(d_logits_);
  cudaFree(d_tok_);
  cudaFree(d_position_);
  cudaFree(d_argmax_);
  cudaFree(d_topk_part_val_);
  cudaFree(d_topk_part_idx_);
  cudaFree(d_topk_val_);
  cudaFree(d_topk_idx_);
  cudaFree(d_cand_idx_);
  cudaFree(d_cand_val_);
  cudaFree(d_cand_count_);
  if (decode_graph_exec_) cudaGraphExecDestroy(decode_graph_exec_);
  if (decode_graph_) cudaGraphDestroy(decode_graph_);
  if (stream_) cudaStreamDestroy(stream_);
}

void PlanCudaEngine::parse_manifest(const std::string& manifest_path) {
  std::ifstream f(manifest_path);
  if (!f) throw std::runtime_error("cannot open manifest: " + manifest_path);
  auto get_int_list = [](const std::string& js) {
    std::vector<int> out;
    std::string cur;
    for (char c : js) {
      if (c == '-' || (c >= '0' && c <= '9'))
        cur += c;
      else {
        if (!cur.empty()) {
          out.push_back(std::stoi(cur));
          cur.clear();
        }
      }
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
      if (k == "num_layers")
        cfg_.num_layers = std::stoi(v);
      else if (k == "hidden")
        cfg_.hidden = std::stoi(v);
      else if (k == "num_heads")
        cfg_.num_heads = std::stoi(v);
      else if (k == "num_kv_heads")
        cfg_.num_kv_heads = std::stoi(v);
      else if (k == "head_dim")
        cfg_.head_dim = std::stoi(v);
      else if (k == "global_head_dim")
        cfg_.global_head_dim = std::stoi(v);
      else if (k == "head_dim_sliding")
        cfg_.head_dim_sliding = std::stoi(v);
      else if (k == "head_dim_full")
        cfg_.head_dim_full = std::stoi(v);
      else if (k == "num_kv_heads_sliding")
        cfg_.num_kv_heads_sliding = std::stoi(v);
      else if (k == "num_kv_heads_full")
        cfg_.num_kv_heads_full = std::stoi(v);
      else if (k == "attention_k_eq_v")
        cfg_.attention_k_eq_v = (v == "True" || v == "true" || v == "1");
      else if (k == "intermediate")
        cfg_.intermediate = std::stoi(v);
      else if (k == "vocab")
        cfg_.vocab = std::stoi(v);
      else if (k == "rms_eps")
        cfg_.rms_eps = std::stof(v);
      else if (k == "sliding_window")
        cfg_.sliding_window = std::stoi(v);
      else if (k == "final_logit_softcapping")
        cfg_.final_logit_softcapping = std::stof(v);
      else if (k == "hidden_size_per_layer_input")
        cfg_.hidden_size_per_layer_input = std::stoi(v);
      else if (k == "num_kv_shared_layers")
        cfg_.num_kv_shared_layers = std::stoi(v);
      else if (k == "first_shared_layer")
        cfg_.first_shared_layer = std::stoi(v);
      else if (k == "rope_theta_full")
        cfg_.rope_theta_full = std::stof(v);
      else if (k == "rope_theta_sliding")
        cfg_.rope_theta_sliding = std::stof(v);
      else if (k == "partial_rotary_full")
        cfg_.partial_rotary_full = std::stof(v);
      else if (k == "bos_token_id")
        cfg_.bos_token_id = std::stoi(v);
      else if (k == "eos_token_id")
        cfg_.eos_token_id = std::stoi(v);
      else if (k == "use_double_wide_mlp")
        cfg_.use_double_wide_mlp = (v == "True" || v == "true" || v == "1");
      else if (k == "enable_moe_block")
        cfg_.enable_moe_block = (v == "True" || v == "true" || v == "1");
      else if (k == "num_experts")
        cfg_.num_experts = std::stoi(v);
      else if (k == "top_k_experts")
        cfg_.top_k_experts = std::stoi(v);
      else if (k == "moe_intermediate_size")
        cfg_.moe_intermediate_size = std::stoi(v);
      else if (k == "tie_word_embeddings")
        cfg_.tie_word_embeddings = (v == "True" || v == "true" || v == "1");
    } else if (kind == "CFGJSON") {
      std::string k;
      ss >> k;
      std::string rest;
      std::getline(ss, rest);
      if (k == "kv_source")
        cfg_.kv_source = get_int_list(rest);
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
      for (char c : shp) {
        if (c == ',') {
          m.shape.push_back(std::stoi(cur));
          cur.clear();
        } else
          cur += c;
      }
      if (!cur.empty()) m.shape.push_back(std::stoi(cur));
      meta_[name] = m;
    }
  }
  if (cfg_.kv_source.empty())
    for (int i = 0; i < cfg_.num_layers; ++i) cfg_.kv_source.push_back(i);
}

// Host fp16 for a tensor, from whichever container backs this model.
//  .cpi        : already fp16, shapes in the manifest (rows/cols ignored).
//  safetensors : bf16, and the header carries no shape we use — so the caller
//                passes the dims from the config, and we convert bf16 -> fp16.
std::vector<__half> PlanCudaEngine::read_host_fp16(const std::string& name, int rows, int cols) {
  if (!from_safetensors_) {
    auto it = meta_.find(name);
    if (it == meta_.end()) throw std::runtime_error("missing tensor: " + name);
    std::ifstream f(cpi_path_, std::ios::binary);
    return read_fp16(f, data_start_, it->second.offset, it->second.bytes);
  }
  if (!st_.has_tensor(name)) throw std::runtime_error("missing tensor: " + name);
  const auto* src = reinterpret_cast<const std::uint16_t*>(st_.tensor_ptr(name));
  const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  if (n * sizeof(std::uint16_t) > st_.tensor_bytes(name))
    throw std::runtime_error("tensor smaller than declared dims: " + name);
  std::vector<__half> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = __float2half(mini::bf16_to_float(src[i]));
  return out;
}

// Shapes for the safetensors path. A .cpi carries them in its manifest; a HF
// directory does not, so they are derived from config.json into st_shapes_ at load.
// Resolving here is what lets the SHARED Gemma loader call upload("...") with no
// dims and still work against either container.
std::pair<int, int> PlanCudaEngine::shape_of(const std::string& name) const {
  auto it = st_shapes_.find(name);
  if (it == st_shapes_.end()) throw std::runtime_error("no shape registered for: " + name);
  return it->second;
}

// Weight lookup that FAILS LOUDLY. dev_ is a map, so dev_[name] on a missing key
// silently inserts nullptr -- which the ops then read as "weightless norm" or a GEMV
// against a null matrix, producing zeros instead of an error. Every bound weight goes
// through here; genuinely weightless ops pass nullptr explicitly.
const __half* PlanCudaEngine::dev_at(const std::string& name) const {
  auto it = dev_.find(name);
  if (it == dev_.end() || it->second == nullptr)
    throw std::runtime_error("weight not loaded: " + name);
  return static_cast<const __half*>(it->second);
}

__half* PlanCudaEngine::upload(const std::string& name, int rows, int cols) {
  if (from_safetensors_ && (rows == 0 || cols == 0)) {
    const auto s = shape_of(name);
    rows = s.first;
    cols = s.second;
  }
  const auto host = read_host_fp16(name, rows, cols);
  __half* d = nullptr;
  G4_CHECK(cudaMalloc(&d, host.size() * sizeof(__half)));
  G4_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice));
  dev_[name] = d;
  return d;
}

// float32 weights (the delta-net RMS-norm weight and A_log ship as f32).
float* PlanCudaEngine::upload_f32(const std::string& name, int n) {
  if (!st_.has_tensor(name)) throw std::runtime_error("missing tensor: " + name);
  const auto* src = reinterpret_cast<const float*>(st_.tensor_ptr(name));
  float* d = nullptr;
  G4_CHECK(cudaMalloc(&d, static_cast<std::size_t>(n) * sizeof(float)));
  G4_CHECK(cudaMemcpy(d, src, static_cast<std::size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
  devf_[name] = d;
  return d;
}

// Upload a [rows, cols] weight and quantize it on-device to int8 or int4, freeing
// the fp16 straight away. Per-tensor so peak VRAM is one tensor, not the whole
// fp16 model (Gemma 12B is ~24 GB in fp16 and would OOM before we could shrink it).
void PlanCudaEngine::upload_int4(const std::string& name, int rows, int cols) {
  if (from_safetensors_ && (rows == 0 || cols == 0)) {
    const auto s = shape_of(name);
    rows = s.first;
    cols = s.second;
  }
  if (rows == 0 || cols == 0) {  // .cpi: shapes come from the manifest
    auto it = meta_.find(name);
    if (it == meta_.end()) throw std::runtime_error("missing tensor: " + name);
    const auto& shape = it->second.shape;
    if (shape.size() != 2) {  // only 2-D projections are quantized
      upload(name);
      return;
    }
    rows = shape[0];
    cols = shape[1];
  }
  const int bits = weight_quant_bits_;
  // A group must be a power of two that divides the row evenly. When the requested
  // group does not divide it, HALVE until it does rather than dropping to per-row:
  // per-row IS the failure mode group-wise scales exist to fix, so falling back to
  // it would silently reintroduce it on exactly the tensors that need it. Gemma MoE
  // hits this -- its down projections are 704 and 2112 wide, neither divisible by
  // 128, so they land on 64.
  int group = weight_quant_group_;
  if (group > 0 && (group & (group - 1)) != 0) {
    group = 0;  // not a power of two: caller error, take the safe path
  }
  while (group > 0 && (group > cols || (cols % group) != 0)) {
    group /= 2;
  }
  if (group == 1) {
    group = 0;  // a scale per weight is just fp16 with extra steps
  }

  auto host = read_host_fp16(name, rows, cols);
  __half* d_fp16 = nullptr;
  G4_CHECK(cudaMalloc(&d_fp16, host.size() * sizeof(__half)));
  G4_CHECK(cudaMemcpy(d_fp16, host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice));

  // fp16 -> int8 with a symmetric scale. max_q sets the level count: 127 for int8,
  // 7 for int4 (which is then packed two-per-byte). group picks the granularity.
  std::int8_t* d_i8 = nullptr;
  float* d_scales = nullptr;
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const int n_groups = kernels::quant_group_count(cols, group);
  G4_CHECK(cudaMalloc(&d_i8, n));
  G4_CHECK(cudaMalloc(&d_scales, static_cast<std::size_t>(rows) * n_groups * sizeof(float)));
  if (group > 0) {
    kernels::launch_quantize_groupwise_fp16_to_int8(d_fp16, d_i8, d_scales, rows, cols, group,
                                                    stream_, bits == 4 ? 7 : 127);
  } else {
    kernels::launch_quantize_rowwise_fp16_to_int8(d_fp16, d_i8, d_scales, rows, cols, stream_,
                                                  bits == 4 ? 7 : 127);
  }

  std::int8_t* d_w = d_i8;
  if (bits == 4) {
    std::int8_t* d_packed = nullptr;
    G4_CHECK(cudaMalloc(&d_packed, static_cast<std::size_t>(rows) * ((cols + 1) / 2)));
    kernels::launch_pack_rowwise_int8_to_int4(d_i8, d_packed, rows, cols, stream_);
    G4_CHECK(cudaStreamSynchronize(stream_));
    cudaFree(d_i8);
    d_w = d_packed;
  } else {
    G4_CHECK(cudaStreamSynchronize(stream_));
  }

  cudaFree(d_fp16);  // the whole point: the fp16 never coexists with the next tensor
  qdev_[name] = {d_w, d_scales, group};
}

float PlanCudaEngine::scalar_value(const std::string& name) {
  if (from_safetensors_) {
    // Optional: a checkpoint with no layer_scalar means a 1.0 (no-op) gain.
    if (!st_.has_tensor(name)) return 1.0f;
    const auto* src = reinterpret_cast<const std::uint16_t*>(st_.tensor_ptr(name));
    return mini::bf16_to_float(src[0]);
  }
  auto it = meta_.find(name);
  if (it == meta_.end()) throw std::runtime_error("missing scalar: " + name);
  std::ifstream f(cpi_path_, std::ios::binary);
  auto host = read_fp16(f, data_start_, it->second.offset, it->second.bytes);
  return __half2float(host[0]);
}

void PlanCudaEngine::build_rope_tables() {
  if (cfg_.family == Family::Qwen35) {
    // One partial-rotary table: [max_ctx, rotary_dim/2], inv_freq = theta^(-2i/rotary_dim).
    const int half = rotary_dim_ / 2;
    std::vector<float> cs(static_cast<std::size_t>(max_ctx_) * half), ss(cs.size());
    for (int pos = 0; pos < max_ctx_; ++pos) {
      for (int i = 0; i < half; ++i) {
        const float inv = std::pow(cfg_.rope_theta,
                                   -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim_));
        const float ang = static_cast<float>(pos) * inv;
        cs[static_cast<std::size_t>(pos) * half + i] = std::cos(ang);
        ss[static_cast<std::size_t>(pos) * half + i] = std::sin(ang);
      }
    }
    G4_CHECK(cudaMalloc(&d_cos_full_, cs.size() * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_sin_full_, ss.size() * sizeof(float)));
    G4_CHECK(cudaMemcpy(d_cos_full_, cs.data(), cs.size() * sizeof(float), cudaMemcpyHostToDevice));
    G4_CHECK(cudaMemcpy(d_sin_full_, ss.data(), ss.size() * sizeof(float), cudaMemcpyHostToDevice));
    return;
  }
  // Tables sized to the ACTUAL per-layer-type head_dim (E2B full=512, 12B full=256).
  const int hd_s = cfg_.head_dim_sliding, hd_f = cfg_.head_dim_full;
  const int hs = hd_s / 2;  // sliding pairs
  const int hf = hd_f / 2;  // full pairs
  std::vector<float> cs(static_cast<std::size_t>(max_ctx_) * hs), ss(cs.size());
  std::vector<float> cf(static_cast<std::size_t>(max_ctx_) * hf), sf(cf.size());
  // sliding: full rotary, inv_freq[i] = theta^(-2i/head_dim)
  for (int p = 0; p < max_ctx_; ++p)
    for (int i = 0; i < hs; ++i) {
      float inv = std::pow(cfg_.rope_theta_sliding, -2.0f * i / hd_s);
      cs[static_cast<std::size_t>(p) * hs + i] = std::cos(p * inv);
      ss[static_cast<std::size_t>(p) * hs + i] = std::sin(p * inv);
    }
  // full: partial rotary — first rope_angles pairs rotated, rest identity (inv=0)
  const int rope_angles = static_cast<int>(cfg_.partial_rotary_full * hd_f) / 2;
  for (int p = 0; p < max_ctx_; ++p)
    for (int i = 0; i < hf; ++i) {
      float inv = (i < rope_angles) ? std::pow(cfg_.rope_theta_full, -2.0f * i / hd_f) : 0.0f;
      cf[static_cast<std::size_t>(p) * hf + i] = std::cos(p * inv);
      sf[static_cast<std::size_t>(p) * hf + i] = std::sin(p * inv);
    }
  auto up = [&](const std::vector<float>& v, float** d) {
    G4_CHECK(cudaMalloc(d, v.size() * sizeof(float)));
    G4_CHECK(cudaMemcpy(*d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  };
  up(cs, &d_cos_sliding_);
  up(ss, &d_sin_sliding_);
  up(cf, &d_cos_full_);
  up(sf, &d_sin_full_);
}

void PlanCudaEngine::allocate_buffers() {
  if (cfg_.family == Family::Qwen35) {
    const int H = cfg_.hidden;
    const int nvh = cfg_.linear_num_value_heads;
    auto al = [&](__half** p, std::size_t n) { G4_CHECK(cudaMalloc(p, n * sizeof(__half))); };
    al(&d_x_, H);
    al(&d_x_norm_, H);
    al(&d_tmp_, H);
    al(&d_q_pair_, static_cast<std::size_t>(full_q_dim_) * 2);
    al(&d_q_, full_q_dim_);
    al(&d_q_gate_, full_q_dim_);
    al(&d_k_, full_kv_dim_);
    al(&d_v_, full_kv_dim_);
    al(&d_att_, std::max(full_q_dim_, H));
    al(&d_gate_, cfg_.intermediate);
    al(&d_up_, cfg_.intermediate);
    al(&d_inter_, cfg_.intermediate);
    al(&d_lin_mix_, linear_conv_dim_);
    al(&d_lin_z_, linear_v_dim_);
    al(&d_lin_a_, nvh);
    al(&d_lin_b_, nvh);
    al(&d_lin_q_, static_cast<std::size_t>(nvh) * cfg_.linear_key_head_dim);
    al(&d_lin_k_, static_cast<std::size_t>(nvh) * cfg_.linear_key_head_dim);
    al(&d_lin_v_, static_cast<std::size_t>(nvh) * cfg_.linear_value_head_dim);
    al(&d_lin_att_, linear_v_dim_);
    G4_CHECK(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg_.vocab) * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_tok_, sizeof(int)));
    G4_CHECK(cudaMalloc(&d_position_, sizeof(int)));
    G4_CHECK(cudaMalloc(&d_argmax_, sizeof(int)));
    const int topk_parts = kernels::topk_partition_count(cfg_.vocab);
    const std::size_t part_n = static_cast<std::size_t>(topk_parts) * kMaxDeviceTopK;
    G4_CHECK(cudaMalloc(&d_topk_part_val_, part_n * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_topk_part_idx_, part_n * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_topk_val_, kMaxDeviceTopK * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_topk_idx_, kMaxDeviceTopK * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_cand_idx_, kCandCapacity * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_cand_val_, kCandCapacity * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_cand_count_, sizeof(int)));

    // KV cache only for full-attention layers; delta-net layers carry state instead.
    caches_k_.assign(cfg_.num_layers, nullptr);
    caches_v_.assign(cfg_.num_layers, nullptr);
    for (int L = 0; L < cfg_.num_layers; ++L) {
      if (!cfg_.layer_full[L]) continue;
      const std::size_t bytes = static_cast<std::size_t>(max_ctx_) * full_kv_dim_ * sizeof(__half);
      G4_CHECK(cudaMalloc(&caches_k_[L], bytes));
      G4_CHECK(cudaMalloc(&caches_v_[L], bytes));
    }
    // Delta-net state (zeroed: the recurrence starts empty).
    const std::size_t conv_n =
        static_cast<std::size_t>(cfg_.num_layers) * std::max(1, lin_conv_state_stride_);
    const std::size_t rec_n =
        static_cast<std::size_t>(cfg_.num_layers) * std::max(1, lin_recurrent_state_stride_);
    G4_CHECK(cudaMalloc(&d_lin_conv_state_, conv_n * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_lin_recurrent_state_, rec_n * sizeof(float)));
    G4_CHECK(cudaMemset(d_lin_conv_state_, 0, conv_n * sizeof(float)));
    G4_CHECK(cudaMemset(d_lin_recurrent_state_, 0, rec_n * sizeof(float)));
    return;
  }

  const int H = cfg_.hidden;
  const int maxhd = std::max(cfg_.head_dim_full, cfg_.head_dim_sliding);
  const int maxq = cfg_.num_heads * maxhd;
  const int maxkv = std::max(cfg_.num_kv_heads_sliding * cfg_.head_dim_sliding,
                             cfg_.num_kv_heads_full * cfg_.head_dim_full);
  const int maxinter = cfg_.intermediate * (cfg_.use_double_wide_mlp ? 2 : 1);
  const int ple_tot = cfg_.num_layers * cfg_.hidden_size_per_layer_input;
  auto al = [&](__half** p, std::size_t n) { G4_CHECK(cudaMalloc(p, n * sizeof(__half))); };
  al(&d_x_, H);
  al(&d_x_norm_, H);
  al(&d_tmp_, H);
  al(&d_q_, maxq);
  al(&d_k_, maxkv);
  al(&d_v_, maxkv);
  al(&d_att_, maxq);
  al(&d_gate_, maxinter);
  al(&d_up_, maxinter);
  al(&d_inter_, maxinter);
  if (cfg_.enable_moe_block) {
    al(&d_moe_router_in_, H);
    al(&d_moe_logits_, cfg_.num_experts);
    al(&d_moe_inter_, static_cast<std::size_t>(cfg_.top_k_experts) * cfg_.moe_intermediate_size);
    al(&d_moe_out_, H);
    G4_CHECK(cudaMalloc(&d_moe_idx_, static_cast<std::size_t>(cfg_.top_k_experts) * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_moe_w_, static_cast<std::size_t>(cfg_.top_k_experts) * sizeof(float)));
  }
  al(&d_ple_raw_, ple_tot);
  al(&d_ple_, ple_tot);
  al(&d_ple_gate_, cfg_.hidden_size_per_layer_input);
  G4_CHECK(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg_.vocab) * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_tok_, sizeof(int)));
  G4_CHECK(cudaMalloc(&d_position_, sizeof(int)));
  G4_CHECK(cudaMalloc(&d_argmax_, sizeof(int)));
  // Device top-k sampling scratch (sized for the largest k we accept).
  const int topk_parts = kernels::topk_partition_count(cfg_.vocab);
  const std::size_t part_n = static_cast<std::size_t>(topk_parts) * kMaxDeviceTopK;
  G4_CHECK(cudaMalloc(&d_topk_part_val_, part_n * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_topk_part_idx_, part_n * sizeof(int)));
  G4_CHECK(cudaMalloc(&d_topk_val_, kMaxDeviceTopK * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_topk_idx_, kMaxDeviceTopK * sizeof(int)));
  G4_CHECK(cudaMalloc(&d_cand_idx_, kCandCapacity * sizeof(int)));
  G4_CHECK(cudaMalloc(&d_cand_val_, kCandCapacity * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_cand_count_, sizeof(int)));
  // ones for weightless v-norm
  // d_ones_ backs every WEIGHTLESS norm, so it must be as long as the widest vector any
  // of them normalises -- not just the widest head. The vision projector normalises over
  // the vision hidden size, which is far wider than a head: sizing this to maxhd read off
  // the end of the buffer (silently, with garbage "ones").
  std::vector<__half> ones(std::max(maxhd, cfg_.hidden), __float2half(1.0f));
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
      const int kvd = head_dim_of(L) * kv_heads_of(L);
      G4_CHECK(
          cudaMalloc(&caches_k_[L], static_cast<std::size_t>(max_ctx_) * kvd * sizeof(__half)));
      G4_CHECK(
          cudaMalloc(&caches_v_[L], static_cast<std::size_t>(max_ctx_) * kvd * sizeof(__half)));
    }
  }
  for (int L = cfg_.first_shared_layer; L < cfg_.num_layers && cfg_.first_shared_layer > 0; ++L) {
    int src = cfg_.kv_source[L];
    caches_k_[L] = caches_k_[src];
    caches_v_[L] = caches_v_[src];
  }
}

// ── Qwen3.5 recipe ───────────────────────────────────────────────────────────
// Config + weights + plan. Everything else (executor, sampler, quantizer, decode
// graph, KV/state) is the SAME shared machinery Gemma runs on.

// Gemma 4 from a HuggingFace safetensors directory. The .cpi path gets its config
// from the container manifest; this reads the same facts out of config.json and,
// crucially, registers every tensor's SHAPE (the safetensors header carries none we
// use). With Config + st_shapes_ + wprefix_ filled in, the shared Gemma loader and
// plan builder run unchanged -- the MoE checkpoint is not a second Gemma port.
void PlanCudaEngine::parse_gemma4_st_config(const std::string& model_dir) {
  const std::string raw = mini::read_text_file(std::filesystem::path(model_dir) / "config.json");

  // The PARSE lives in engine/plan_model_config.cpp, with no CUDA in it, so PlanMetalEngine can
  // reach the same geometry. What stays here is the SHAPE REGISTRY below, which is specific to how
  // this engine resolves tensors. Splitting it that way is what unblocks Gemma 4 on Metal.
  cfg_ = engine::parse_gemma4_text_config(raw);

  // --- shape registry (see shape_of) ---
  const int H = cfg_.hidden;
  const int E = cfg_.num_experts;
  const int MI = cfg_.moe_intermediate_size;
  auto put = [&](const std::string& n, int r, int c) { st_shapes_[wprefix_ + n] = {r, c}; };
  put("embed_tokens.weight", cfg_.vocab, H);
  put("norm.weight", 1, H);
  // Per-Layer Embeddings (E2B has them; 12B and the MoE do not).
  const int ple = cfg_.hidden_size_per_layer_input;
  if (ple > 0) {
    const int ple_vocab =
        cfg_.vocab_size_per_layer_input > 0 ? cfg_.vocab_size_per_layer_input : cfg_.vocab;
    put("embed_tokens_per_layer.weight", ple_vocab, cfg_.num_layers * ple);
    put("per_layer_model_projection.weight", cfg_.num_layers * ple, H);
    put("per_layer_projection_norm.weight", 1, ple);
  }
  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = "layers." + std::to_string(L) + ".";
    const bool full = cfg_.layer_full[L] != 0;
    const int hd = full ? cfg_.head_dim_full : cfg_.head_dim_sliding;
    const int nkv = full ? cfg_.num_kv_heads_full : cfg_.num_kv_heads_sliding;
    const int qdim = cfg_.num_heads * hd, kvdim = nkv * hd;
    put(p + "input_layernorm.weight", 1, H);
    put(p + "post_attention_layernorm.weight", 1, H);
    put(p + "pre_feedforward_layernorm.weight", 1, H);
    put(p + "post_feedforward_layernorm.weight", 1, H);
    put(p + "self_attn.q_proj.weight", qdim, H);
    put(p + "self_attn.k_proj.weight", kvdim, H);
    put(p + "self_attn.v_proj.weight", kvdim, H);
    put(p + "self_attn.o_proj.weight", H, qdim);
    put(p + "self_attn.q_norm.weight", 1, hd);
    put(p + "self_attn.k_norm.weight", 1, hd);
    put(p + "mlp.gate_proj.weight", cfg_.intermediate, H);
    put(p + "mlp.up_proj.weight", cfg_.intermediate, H);
    put(p + "mlp.down_proj.weight", H, cfg_.intermediate);
    // The KV-shared layers run a DOUBLE-WIDE MLP. This override must come AFTER the
    // puts above -- registering it first just gets overwritten, which silently reads
    // half of each shared layer's weight and produces garbage.
    if (cfg_.use_double_wide_mlp && cfg_.first_shared_layer > 0 && L >= cfg_.first_shared_layer) {
      put(p + "mlp.gate_proj.weight", cfg_.intermediate * 2, H);
      put(p + "mlp.up_proj.weight", cfg_.intermediate * 2, H);
      put(p + "mlp.down_proj.weight", H, cfg_.intermediate * 2);
    }
    if (ple > 0) {
      put(p + "per_layer_input_gate.weight", ple, H);
      put(p + "per_layer_projection.weight", H, ple);
      put(p + "post_per_layer_input_norm.weight", 1, H);
    }
    if (cfg_.enable_moe_block) {
      put(p + "pre_feedforward_layernorm_2.weight", 1, H);
      put(p + "post_feedforward_layernorm_1.weight", 1, H);
      put(p + "post_feedforward_layernorm_2.weight", 1, H);
      put(p + "router.proj.weight", E, H);
      put(p + "router.scale", 1, H);
      put(p + "router.per_expert_scale", 1, E);
      // The experts ship as 3-D [E, ...] but are contiguous, so they are indexed as
      // one tall 2-D matrix and quantised like any other weight; picking an expert is
      // just a row offset.
      put(p + "experts.gate_up_proj", E * 2 * MI, H);
      put(p + "experts.down_proj", E * H, MI);
    }
  }
}

void PlanCudaEngine::parse_qwen35_config(const std::string& model_dir) {
  const std::string raw = mini::read_text_file(std::filesystem::path(model_dir) / "config.json");
  const std::string tc = mini::json_extract_object(raw, "text_config");
  if (tc.empty()) throw std::runtime_error("Qwen3.5 config is missing text_config");

  cfg_.family = Family::Qwen35;
  cfg_.vocab = mini::json_get_int(tc, "vocab_size", 0);
  cfg_.hidden = mini::json_get_int(tc, "hidden_size", 0);
  cfg_.intermediate = mini::json_get_int(tc, "intermediate_size", 0);
  cfg_.num_layers = mini::json_get_int(tc, "num_hidden_layers", 0);
  cfg_.num_heads = mini::json_get_int(tc, "num_attention_heads", 0);
  cfg_.num_kv_heads = mini::json_get_int(tc, "num_key_value_heads", 0);
  cfg_.head_dim = mini::json_get_int(tc, "head_dim", 0);
  cfg_.linear_num_key_heads = mini::json_get_int(tc, "linear_num_key_heads", 0);
  cfg_.linear_num_value_heads = mini::json_get_int(tc, "linear_num_value_heads", 0);
  cfg_.linear_key_head_dim = mini::json_get_int(tc, "linear_key_head_dim", 0);
  cfg_.linear_value_head_dim = mini::json_get_int(tc, "linear_value_head_dim", 0);
  cfg_.linear_conv_kernel_dim = mini::json_get_int(tc, "linear_conv_kernel_dim", 0);
  cfg_.max_position_embeddings = mini::json_get_int(tc, "max_position_embeddings", 0);
  cfg_.eos_token_id =
      mini::json_get_int(tc, "eos_token_id", mini::json_get_int(raw, "eos_token_id", -1));
  cfg_.rms_eps = mini::json_get_float(tc, "rms_norm_eps", 1e-6f);
  const std::string rope = mini::json_extract_object(tc, "rope_parameters");
  cfg_.rope_theta = mini::json_get_float(rope, "rope_theta", 1e7f);
  cfg_.partial_rotary_factor = mini::json_get_float(rope, "partial_rotary_factor", 1.0f);

  // layer_full doubles as the layer-kind mask: 1 = full attention, 0 = delta-net.
  cfg_.layer_full.clear();
  for (const std::string& t : mini::json_get_string_array(tc, "layer_types"))
    cfg_.layer_full.push_back(t == "full_attention" ? 1 : 0);
  if (static_cast<int>(cfg_.layer_full.size()) != cfg_.num_layers)
    throw std::runtime_error("Qwen3.5 layer_types does not match num_hidden_layers");

  // Derived geometry.
  rotary_dim_ =
      static_cast<int>(std::lround(static_cast<float>(cfg_.head_dim) * cfg_.partial_rotary_factor));
  if (rotary_dim_ <= 0 || (rotary_dim_ % 2) != 0) rotary_dim_ = cfg_.head_dim - (cfg_.head_dim % 2);
  full_q_dim_ = cfg_.num_heads * cfg_.head_dim;
  full_kv_dim_ = cfg_.num_kv_heads * cfg_.head_dim;
  linear_k_dim_ = cfg_.linear_num_key_heads * cfg_.linear_key_head_dim;
  linear_v_dim_ = cfg_.linear_num_value_heads * cfg_.linear_value_head_dim;
  linear_conv_dim_ = linear_k_dim_ * 2 + linear_v_dim_;
  lin_conv_state_stride_ = linear_conv_dim_ * std::max(0, cfg_.linear_conv_kernel_dim - 1);
  lin_recurrent_state_stride_ =
      cfg_.linear_num_value_heads * cfg_.linear_key_head_dim * cfg_.linear_value_head_dim;
  cfg_.bos_token_id = cfg_.eos_token_id >= 0 ? cfg_.eos_token_id : 0;

  if (cfg_.vocab <= 0 || cfg_.hidden <= 0 || cfg_.num_layers <= 0 || cfg_.head_dim <= 0 ||
      cfg_.linear_num_value_heads <= 0 || cfg_.linear_conv_kernel_dim <= 0)
    throw std::runtime_error("Qwen3.5 config.json is missing required text_config fields");
}

void PlanCudaEngine::load_qwen35_weights() {
  const bool quant = weight_quant_bits_ == 4 || weight_quant_bits_ == 8;
  const int H = cfg_.hidden;
  auto proj = [&](const std::string& n, int rows, int cols) {
    if (quant)
      upload_int4(n, rows, cols);
    else
      upload(n, rows, cols);
  };

  upload("model.language_model.embed_tokens.weight", cfg_.vocab, H);
  upload("model.language_model.norm.weight", 1, H);
  if (st_.has_tensor("lm_head.weight")) upload("lm_head.weight", cfg_.vocab, H);

  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = "model.language_model.layers." + std::to_string(L) + ".";
    upload(p + "input_layernorm.weight", 1, H);
    upload(p + "post_attention_layernorm.weight", 1, H);
    proj(p + "mlp.gate_proj.weight", cfg_.intermediate, H);
    proj(p + "mlp.up_proj.weight", cfg_.intermediate, H);
    proj(p + "mlp.down_proj.weight", H, cfg_.intermediate);

    if (cfg_.layer_full[L]) {  // full attention (q_proj is a FUSED q|gate)
      proj(p + "self_attn.q_proj.weight", full_q_dim_ * 2, H);
      proj(p + "self_attn.k_proj.weight", full_kv_dim_, H);
      proj(p + "self_attn.v_proj.weight", full_kv_dim_, H);
      proj(p + "self_attn.o_proj.weight", H, full_q_dim_);
      upload(p + "self_attn.q_norm.weight", 1, cfg_.head_dim);
      upload(p + "self_attn.k_norm.weight", 1, cfg_.head_dim);
    } else {  // gated delta-net
      proj(p + "linear_attn.in_proj_qkv.weight", linear_conv_dim_, H);
      proj(p + "linear_attn.in_proj_z.weight", linear_v_dim_, H);
      proj(p + "linear_attn.in_proj_a.weight", cfg_.linear_num_value_heads, H);
      proj(p + "linear_attn.in_proj_b.weight", cfg_.linear_num_value_heads, H);
      proj(p + "linear_attn.out_proj.weight", H, linear_v_dim_);
      upload(p + "linear_attn.conv1d.weight", linear_conv_dim_, cfg_.linear_conv_kernel_dim);
      upload(p + "linear_attn.dt_bias", 1, cfg_.linear_num_value_heads);
      upload_f32(p + "linear_attn.norm.weight", cfg_.linear_value_head_dim);
      upload_f32(p + "linear_attn.A_log", cfg_.linear_num_value_heads);
    }
  }
}

void PlanCudaEngine::load_all(const std::string& cpi_path) {
  cpi_path_ = cpi_path;
  const std::string& W = wprefix_;  // "" for .cpi, "model.language_model." for HF
  // large tables + per-layer weights to device
  upload(W + "embed_tokens.weight");
  upload(W + "norm.weight");
  if (has_ple()) {  // E2B has Per-Layer Embeddings; 12B does not
    upload(W + "embed_tokens_per_layer.weight");
    upload(W + "per_layer_model_projection.weight");
    upload(W + "per_layer_projection_norm.weight");
  }
  layer_scalar_host_.assign(cfg_.num_layers, 1.0f);
  // The dense projections carry ~all the weight; norms/embeddings stay fp16.
  const bool quantize = weight_quant_bits_ == 4 || weight_quant_bits_ == 8;
  // LLAMA_INFER_PLAN_INT4_GROUP=attn|mlp restricts quantization to one group
  // (bisection aid when a model degrades under int4).
  const char* group_env = std::getenv("LLAMA_INFER_PLAN_INT4_GROUP");
  const std::string group = group_env ? group_env : "";
  const auto quantizable = [&group](const char* t) {
    const bool is_attn =
        std::strncmp(t, "self_attn.", 10) == 0 && std::strstr(t, "_proj.") != nullptr;
    const bool is_mlp = std::strncmp(t, "mlp.", 4) == 0 && std::strstr(t, "_proj.") != nullptr;
    if (group == "attn") return is_attn;
    if (group == "mlp") return is_mlp;
    return is_attn || is_mlp;
  };
  const auto load_weight = [&](const std::string& full, const char* t) {
    if (quantize && quantizable(t))
      upload_int4(full);  // quantizes per weight_quant_bits_
    else
      upload(full);
  };
  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = W + "layers." + std::to_string(L) + ".";
    for (const char* t :
         {"input_layernorm.weight", "post_attention_layernorm.weight",
          "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
          "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
          "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight",
          "mlp.up_proj.weight", "mlp.down_proj.weight"})
      load_weight(p + t, t);
    // k_eq_v full layers have no v_proj (V reuses the raw k_proj output).
    if (!k_eq_v(L)) load_weight(p + "self_attn.v_proj.weight", "self_attn.v_proj.weight");
    if (has_ple())
      for (const char* t : {"per_layer_input_gate.weight", "per_layer_projection.weight",
                            "post_per_layer_input_norm.weight"})
        upload(p + t);
    if (cfg_.enable_moe_block) {
      // Norms and the tiny router stay fp16; the EXPERTS are ~90% of the model, so
      // they go through the same quantiser as any other projection (they are plain
      // 2-D matrices here -- see parse_gemma4_st_config).
      for (const char* t :
           {"pre_feedforward_layernorm_2.weight", "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight", "router.proj.weight", "router.scale",
            "router.per_expert_scale"})
        upload(p + t);
      if (quantize) {
        upload_int4(p + "experts.gate_up_proj");
        upload_int4(p + "experts.down_proj");
      } else {
        upload(p + "experts.gate_up_proj");
        upload(p + "experts.down_proj");
      }
    }
    layer_scalar_host_[L] = scalar_value(p + "layer_scalar");
  }
}

void PlanCudaEngine::open(const std::string& cpi_path, int max_context) {
  max_ctx_ = max_context;
  if (std::getenv("LLAMA_INFER_PLAN_NO_GRAPH")) decode_graph_enabled_ = false;
  if (std::getenv("LLAMA_INFER_PLAN_NO_DEVICE_TOPK")) device_topk_enabled_ = false;

  // A directory ⇒ HuggingFace safetensors; a file ⇒ .cpi. Within a directory the
  // architecture comes from config.json's model_type -- the engine reads whichever
  // container and architecture the model actually ships, with no name matching on
  // the path.
  std::error_code ec;
  if (std::filesystem::is_directory(cpi_path, ec) && !ec) {
    from_safetensors_ = true;
    st_.open(cpi_path);
    const std::string raw = mini::read_text_file(std::filesystem::path(cpi_path) / "config.json");
    const std::string tc = mini::json_extract_object(raw, "text_config");
    const std::string mt = mini::json_get_string(tc.empty() ? raw : tc, "model_type", "");
    if (mt.rfind("gemma4", 0) == 0) {
      wprefix_ = "model.language_model.";
      parse_gemma4_st_config(cpi_path);
      if (max_ctx_ <= 0) max_ctx_ = 4096;
      // Gemma's dual/partial RoPE has no device-position kernel, same as Qwen's.
      decode_graph_enabled_ = false;
      G4_CHECK(cudaStreamCreate(&stream_));
      load_all(cpi_path);
      build_rope_tables();
      allocate_buffers();
      build_plan();
      // The vision tower, when the checkpoint ships one. Opt out with
      // LLAMA_INFER_NO_VISION=1 (it costs VRAM even on text-only prompts).
      // Sequence prefill is a capability of the PLAN: partial RoPE and delta-net have no
      // sequence kernels yet, so those plans keep the token-by-token prefill.
      seq_prefill_ok_ = plan_can_sequence_prefill();
      if (seq_prefill_ok_) allocate_sequence_buffers(std::min(max_ctx_, 4096));

      parse_vision_config(cpi_path);
      if (vcfg_.present && !std::getenv("LLAMA_INFER_NO_VISION")) {
        vis_max_patches_ = 4096;
        load_vision_weights();
        build_vision_rope_tables();
        allocate_vision_buffers();
        build_vision_plan();
      } else {
        vcfg_.present = false;
      }
      G4_CHECK(cudaStreamSynchronize(stream_));
      return;
    }
    parse_qwen35_config(cpi_path);
    if (max_ctx_ <= 0 ||
        (cfg_.max_position_embeddings > 0 && max_ctx_ > cfg_.max_position_embeddings))
      max_ctx_ = std::min(max_ctx_ > 0 ? max_ctx_ : 2048, cfg_.max_position_embeddings);
    // Partial RoPE has no device-position kernel, so this plan cannot be graphed.
    // A capability decision made on the PLAN, not on a model name.
    decode_graph_enabled_ = false;
    G4_CHECK(cudaStreamCreate(&stream_));
    load_qwen35_weights();
    build_rope_tables();
    allocate_buffers();
    build_plan();
    G4_CHECK(cudaStreamSynchronize(stream_));
    return;
  }

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
  build_plan();  // resolve the per-layer forward into a data op-plan (once)
  G4_CHECK(cudaStreamSynchronize(stream_));
}

// Resolve the per-layer forward into a data op-plan once at load. This is the
// exact op sequence the former imperative run_layer emitted, with the per-layer
// conditionals (shared / keqv / full / PLE) decided here and weights + geometry
// bound — so the hot loop (run_layer, below) is branch-free over identity.
// The MoE feed-forward, appended to a layer whose dense MLP output is already in
// Slot::Tmp (pre-norm). Transcribed from Gemma4TextDecoderLayer.forward:
//
//   h1 = post_ffn_norm_1(dense_mlp_out)
//   tkw, tki = router(X)                       <-- the PRE-MLP residual, not the MLP output
//   h2 = post_ffn_norm_2( experts( pre_ffn_norm_2(X), tki, tkw ) )
//   Tmp = h1 + h2                              <-- caller then applies post_ffn_norm + residual
//
// Note both branches read X, which no op here writes, so the ordering is safe.
void PlanCudaEngine::append_moe_ffn_ops(std::vector<opplan::Op>& ops, const std::string& p, int L) {
  using namespace opplan;
  const int H = cfg_.hidden;
  const int E = cfg_.num_experts;
  const int K = cfg_.top_k_experts;
  const int MI = cfg_.moe_intermediate_size;
  auto W = [&](const char* t) { return dev_at(p + t); };

  // h1 = post_feedforward_layernorm_1(dense MLP output), left in Tmp.
  {
    Op o;
    o.kind = OpKind::RmsNorm;
    o.in = Slot::Tmp;
    o.out = Slot::Tmp;
    o.weight = W("post_feedforward_layernorm_1.weight");
    o.rows = 1;
    o.cols = H;
    ops.push_back(o);
  }

  // --- router(X): weightless RMS norm -> * scale * H^-0.5 -> proj -> top-k ---
  {
    Op o;
    o.kind = OpKind::RmsNorm;
    o.in = Slot::X;
    o.out = Slot::MoeRouterIn;
    o.weight = nullptr;
    o.rows = 1;
    o.cols = H;  // with_scale=False in HF
    ops.push_back(o);
  }
  {
    Op o;
    o.kind = OpKind::MulVec;
    o.in = Slot::MoeRouterIn;
    o.out = Slot::MoeRouterIn;
    o.weight = W("router.scale");
    o.cols = H;
    o.scale = std::pow(static_cast<float>(H), -0.5f);
    ops.push_back(o);
  }
  {
    Op o;
    o.kind = OpKind::Gemv;
    o.in = Slot::MoeRouterIn;
    o.out = Slot::MoeLogits;
    o.weight = W("router.proj.weight");
    o.cols = E;
    o.in_dim = H;  // tiny: stays fp16
    ops.push_back(o);
  }
  {
    Op o;
    o.kind = OpKind::MoeRouterTopk;
    o.in = Slot::MoeLogits;
    o.weight = W("router.per_expert_scale");
    o.cols = E;
    o.heads = K;
    ops.push_back(o);
  }

  // --- experts(pre_feedforward_layernorm_2(X)) ---
  // XNorm is free here: the dense MLP already consumed it.
  {
    Op o;
    o.kind = OpKind::RmsNorm;
    o.in = Slot::X;
    o.out = Slot::XNorm;
    o.weight = W("pre_feedforward_layernorm_2.weight");
    o.rows = 1;
    o.cols = H;
    ops.push_back(o);
  }
  auto bind_expert = [&](Op& o, const char* t) {
    const auto q = qdev_.find(p + t);
    if (q != qdev_.end()) {
      o.qweight = q->second.packed;
      o.qscales = q->second.scales;
      o.qbits = weight_quant_bits_;
      o.qgroup = q->second.group;
    } else {
      o.weight = dev_at(p + t);
    }
  };
  {
    Op o;
    o.kind = OpKind::MoeGateUpGeglu;
    o.in = Slot::XNorm;
    o.out = Slot::MoeInter;
    o.cols = MI;
    o.in_dim = H;
    o.heads = K;
    bind_expert(o, "experts.gate_up_proj");
    ops.push_back(o);
  }
  {
    Op o;
    o.kind = OpKind::MoeDownAccum;
    o.in = Slot::MoeInter;
    o.out = Slot::MoeOut;
    o.cols = H;
    o.in_dim = MI;
    o.heads = K;
    bind_expert(o, "experts.down_proj");
    ops.push_back(o);
  }
  {
    Op o;
    o.kind = OpKind::RmsNorm;
    o.in = Slot::MoeOut;
    o.out = Slot::MoeOut;
    o.weight = W("post_feedforward_layernorm_2.weight");
    o.rows = 1;
    o.cols = H;
    ops.push_back(o);
  }

  // Tmp = h1 + h2
  {
    Op o;
    o.kind = OpKind::AddInplace;
    o.out = Slot::Tmp;
    o.in = Slot::MoeOut;
    o.cols = H;
    ops.push_back(o);
  }
  (void)L;
}

// ───────────────────────── vision tower ─────────────────────────
// Same executor, same ops, sequence mode. Only the config, slots and plan differ.

void PlanCudaEngine::parse_vision_config(const std::string& model_dir) {
  const std::string raw = mini::read_text_file(std::filesystem::path(model_dir) / "config.json");
  const std::string vc = mini::json_extract_object(raw, "vision_config");
  if (vc.empty()) return;

  vcfg_.hidden = mini::json_get_int(vc, "hidden_size", 0);
  vcfg_.layers = mini::json_get_int(vc, "num_hidden_layers", 0);
  vcfg_.heads = mini::json_get_int(vc, "num_attention_heads", 0);
  vcfg_.kv_heads = mini::json_get_int(vc, "num_key_value_heads", vcfg_.heads);
  vcfg_.head_dim = mini::json_get_int(vc, "head_dim", 0);
  vcfg_.intermediate = mini::json_get_int(vc, "intermediate_size", 0);
  vcfg_.patch_size = mini::json_get_int(vc, "patch_size", 0);
  vcfg_.pos_table_size = mini::json_get_int(vc, "position_embedding_size", 0);
  vcfg_.pooling_kernel = mini::json_get_int(vc, "pooling_kernel_size", 1);
  vcfg_.rms_eps = mini::json_get_float(vc, "rms_norm_eps", 1e-6f);
  vcfg_.standardize = mini::json_get_bool(vc, "standardize", false);
  vcfg_.clipped_linears = mini::json_get_bool(vc, "use_clipped_linears", false);
  const std::string rp = mini::json_extract_object(vc, "rope_parameters");
  vcfg_.rope_theta = mini::json_get_float(rp.empty() ? vc : rp, "rope_theta", 100.0f);

  vcfg_.present = vcfg_.hidden > 0 && vcfg_.layers > 0 && vcfg_.patch_size > 0;
}

// cos/sin over the SPATIAL half-dim: inv_freq = theta^(-2j/spatial_dim) with
// spatial_dim = head_dim/2, giving head_dim/4 entries per position. Both axes share
// the table -- x and y get identical frequency ranges (they are separate positions,
// not separate frequency bands).
void PlanCudaEngine::build_vision_rope_tables() {
  const int pairs = vcfg_.head_dim / 4;
  const int spatial_dim = vcfg_.head_dim / 2;
  const int max_pos = 4096;  // patch-grid coordinates; far above any real image
  std::vector<float> cs(static_cast<std::size_t>(max_pos) * pairs), ss(cs.size());
  for (int p = 0; p < max_pos; ++p) {
    for (int j = 0; j < pairs; ++j) {
      const float inv =
          std::pow(vcfg_.rope_theta, -static_cast<float>(2 * j) / static_cast<float>(spatial_dim));
      const float ang = static_cast<float>(p) * inv;
      cs[static_cast<std::size_t>(p) * pairs + j] = std::cos(ang);
      ss[static_cast<std::size_t>(p) * pairs + j] = std::sin(ang);
    }
  }
  G4_CHECK(cudaMalloc(&d_vrope_cos_, cs.size() * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_vrope_sin_, ss.size() * sizeof(float)));
  G4_CHECK(cudaMemcpy(d_vrope_cos_, cs.data(), cs.size() * sizeof(float), cudaMemcpyHostToDevice));
  G4_CHECK(cudaMemcpy(d_vrope_sin_, ss.data(), ss.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void PlanCudaEngine::load_vision_weights() {
  const int H = vcfg_.hidden;
  const int patch_dim = 3 * vcfg_.patch_size * vcfg_.patch_size;
  const std::string v = "model.vision_tower.";

  // Vision weights stay fp16: the tower is small (E2B ~100M params) and runs once per
  // image, so quantising it would trade real accuracy for irrelevant savings.
  st_shapes_[v + "patch_embedder.input_proj.weight"] = {H, patch_dim};
  st_shapes_[v + "patch_embedder.position_embedding_table"] = {2 * vcfg_.pos_table_size, H};
  upload(v + "patch_embedder.input_proj.weight");
  upload(v + "patch_embedder.position_embedding_table");
  if (vcfg_.standardize) {
    st_shapes_[v + "std_bias"] = {1, H};
    st_shapes_[v + "std_scale"] = {1, H};
    upload(v + "std_bias");
    upload(v + "std_scale");
  }

  const int qdim = vcfg_.heads * vcfg_.head_dim;
  const int kvdim = vcfg_.kv_heads * vcfg_.head_dim;
  for (int L = 0; L < vcfg_.layers; ++L) {
    const std::string p = v + "encoder.layers." + std::to_string(L) + ".";
    auto put = [&](const std::string& n, int r, int c) { st_shapes_[p + n] = {r, c}; };
    // NOTE the ".linear." infix: the projections are wrapped in Gemma4ClippableLinear.
    // Its clamp bounds are +-inf and are absent from the checkpoint, so it is a plain
    // Linear -- do not implement the clipping.
    put("input_layernorm.weight", 1, H);
    put("post_attention_layernorm.weight", 1, H);
    put("pre_feedforward_layernorm.weight", 1, H);
    put("post_feedforward_layernorm.weight", 1, H);
    put("self_attn.q_proj.linear.weight", qdim, H);
    put("self_attn.k_proj.linear.weight", kvdim, H);
    put("self_attn.v_proj.linear.weight", kvdim, H);
    put("self_attn.o_proj.linear.weight", H, qdim);
    put("self_attn.q_norm.weight", 1, vcfg_.head_dim);
    put("self_attn.k_norm.weight", 1, vcfg_.head_dim);
    put("mlp.gate_proj.linear.weight", vcfg_.intermediate, H);
    put("mlp.up_proj.linear.weight", vcfg_.intermediate, H);
    put("mlp.down_proj.linear.weight", H, vcfg_.intermediate);
    // Clipped projections: E2B ships input/output activation bounds per linear and HF
    // CLAMPS with them. The 26B sets use_clipped_linears=false and ships none. Skipping
    // these does not crash -- it silently changes the numbers, which is why the gate
    // caught it and inspection did not.
    if (vcfg_.clipped_linears) {
      for (const char* proj :
           {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
        const std::string base = p + proj + ".";
        vclip_[base] =
            ClipBounds{scalar_value(base + "input_min"), scalar_value(base + "input_max"),
                       scalar_value(base + "output_min"), scalar_value(base + "output_max")};
      }
    }
    const char* tensors[] = {
        "input_layernorm.weight",           "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        "self_attn.q_proj.linear.weight",   "self_attn.k_proj.linear.weight",
        "self_attn.v_proj.linear.weight",   "self_attn.o_proj.linear.weight",
        "self_attn.q_norm.weight",          "self_attn.k_norm.weight",
        "mlp.gate_proj.linear.weight",      "mlp.up_proj.linear.weight",
        "mlp.down_proj.linear.weight"};
    for (const char* t : tensors) upload(p + t);
  }

  // The projector into text-embedding space.
  st_shapes_["model.embed_vision.embedding_projection.weight"] = {cfg_.hidden, H};
  upload("model.embed_vision.embedding_projection.weight");
}

void PlanCudaEngine::allocate_vision_buffers() {
  const int H = vcfg_.hidden;
  const int patch_dim = 3 * vcfg_.patch_size * vcfg_.patch_size;
  const int P = vis_max_patches_;
  const int qdim = vcfg_.heads * vcfg_.head_dim;
  const int kvdim = vcfg_.kv_heads * vcfg_.head_dim;

  auto al = [&](opplan::Slot slot, std::size_t per_token) {
    __half* d = nullptr;
    G4_CHECK(cudaMalloc(&d, static_cast<std::size_t>(P) * per_token * sizeof(__half)));
    vslot_ptr_[static_cast<int>(slot)] = d;
    vis_buffers_.push_back(d);
  };
  using opplan::Slot;
  al(Slot::X, H);
  al(Slot::XNorm, H);
  al(Slot::Tmp, static_cast<std::size_t>(std::max(H, cfg_.hidden)));  // also the projector output
  al(Slot::Q, qdim);
  al(Slot::K, kvdim);
  al(Slot::V, kvdim);
  al(Slot::Att, qdim);
  al(Slot::Gate, vcfg_.intermediate);
  al(Slot::Up, vcfg_.intermediate);
  al(Slot::Inter, vcfg_.intermediate);

  // Defensive: a tower whose hidden exceeds the text hidden would still overrun d_ones_.
  if (vcfg_.hidden > std::max(cfg_.hidden, std::max(cfg_.head_dim_full, cfg_.head_dim_sliding))) {
    std::vector<__half> ones(vcfg_.hidden, __float2half(1.0f));
    cudaFree(d_ones_);
    G4_CHECK(cudaMalloc(&d_ones_, ones.size() * sizeof(__half)));
    G4_CHECK(
        cudaMemcpy(d_ones_, ones.data(), ones.size() * sizeof(__half), cudaMemcpyHostToDevice));
  }

  G4_CHECK(cudaMalloc(&d_vis_pixels_, static_cast<std::size_t>(P) * patch_dim * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_vis_pos_x_, static_cast<std::size_t>(P) * sizeof(int)));
  G4_CHECK(cudaMalloc(&d_vis_pos_y_, static_cast<std::size_t>(P) * sizeof(int)));
}

void PlanCudaEngine::build_vision_plan() {
  using namespace opplan;
  const int H = vcfg_.hidden;
  const int patch_dim = 3 * vcfg_.patch_size * vcfg_.patch_size;
  const std::string v = "model.vision_tower.";

  {  // prologue: pixels -> patch embeddings (+ the two-axis learned position table)
    Op o;
    o.kind = OpKind::PatchEmbed;
    o.out = Slot::X;
    o.cols = H;
    o.in_dim = patch_dim;
    o.rows = vcfg_.pos_table_size;
    o.weight = dev_at(v + "patch_embedder.input_proj.weight");
    o.aux_ptr = dev_at(v + "patch_embedder.position_embedding_table");
    vplan_.prologue.push_back(o);
  }

  vplan_.layers.assign(vcfg_.layers, LayerPlan{});
  for (int L = 0; L < vcfg_.layers; ++L) {
    const std::string p = v + "encoder.layers." + std::to_string(L) + ".";
    auto& ops = vplan_.layers[L].ops;
    vplan_.layers[L].layer_index = L;
    auto W = [&](const char* t) { return dev_at(p + t); };
    const int nq = vcfg_.heads, nkv = vcfg_.kv_heads, hd = vcfg_.head_dim;
    const int qdim = nq * hd, kvdim = nkv * hd;

    auto rms = [&](Slot in, Slot out, const __half* w, int rows, int cols) {
      Op o;
      o.kind = OpKind::RmsNorm;
      o.in = in;
      o.out = out;
      o.weight = w;
      o.rows = rows;
      o.cols = cols;
      ops.push_back(o);
    };
    auto gemv = [&](Slot in, Slot out, const char* t, int out_dim, int in_dim) {
      Op o;
      o.kind = OpKind::Gemv;
      o.in = in;
      o.out = out;
      o.cols = out_dim;
      o.in_dim = in_dim;
      o.weight = dev_at(p + t);
      // "<proj>.linear.weight" -> "<proj>." is the key the bounds were stored under.
      const std::string tn(t);
      const std::string base = p + tn.substr(0, tn.find("linear.weight"));
      const auto c = vclip_.find(base);
      if (c != vclip_.end()) {
        o.clip_in_min = c->second.in_min;
        o.clip_in_max = c->second.in_max;
        o.clip_out_min = c->second.out_min;
        o.clip_out_max = c->second.out_max;
      }
      ops.push_back(o);
    };
    auto rope2d = [&](Slot s, int heads) {
      Op o;
      o.kind = OpKind::Rope2D;
      o.in = s;
      o.out = s;
      o.heads = heads;
      o.head_dim = hd;
      ops.push_back(o);
    };
    auto add_x = [&](Slot src) {
      Op o;
      o.kind = OpKind::AddInplace;
      o.out = Slot::X;
      o.in = src;
      o.cols = H;
      ops.push_back(o);
    };

    // The same sandwich-norm shape as a Gemma TEXT layer. The differences are that
    // attention is bidirectional and the positions are 2-D.
    rms(Slot::X, Slot::XNorm, W("input_layernorm.weight"), 1, H);
    gemv(Slot::XNorm, Slot::Q, "self_attn.q_proj.linear.weight", qdim, H);
    gemv(Slot::XNorm, Slot::K, "self_attn.k_proj.linear.weight", kvdim, H);
    gemv(Slot::XNorm, Slot::V, "self_attn.v_proj.linear.weight", kvdim, H);
    rms(Slot::Q, Slot::Q, W("self_attn.q_norm.weight"), nq, hd);
    rms(Slot::K, Slot::K, W("self_attn.k_norm.weight"), nkv, hd);
    rms(Slot::V, Slot::V, nullptr, nkv, hd);  // weightless v-norm, as in the text tower
    rope2d(Slot::Q, nq);
    rope2d(Slot::K, nkv);
    {  // scaling = 1.0: pre-scale q by sqrt(hd) to cancel the kernel's 1/sqrt(hd)
      Op o;
      o.kind = OpKind::ScaleCopy;
      o.in = Slot::Q;
      o.out = Slot::Q;
      o.cols = qdim;
      o.scale = std::sqrt(static_cast<float>(hd));
      ops.push_back(o);
    }
    {
      Op o;
      o.kind = OpKind::Attention;
      o.in = Slot::Q;
      o.out = Slot::Att;
      o.heads = nq;
      o.kv_heads = nkv;
      o.head_dim = hd;
      o.full_attention = true;
      ops.push_back(o);
    }
    gemv(Slot::Att, Slot::Tmp, "self_attn.o_proj.linear.weight", H, qdim);
    rms(Slot::Tmp, Slot::Tmp, W("post_attention_layernorm.weight"), 1, H);
    add_x(Slot::Tmp);

    rms(Slot::X, Slot::XNorm, W("pre_feedforward_layernorm.weight"), 1, H);
    gemv(Slot::XNorm, Slot::Gate, "mlp.gate_proj.linear.weight", vcfg_.intermediate, H);
    gemv(Slot::XNorm, Slot::Up, "mlp.up_proj.linear.weight", vcfg_.intermediate, H);
    {
      Op o;
      o.kind = OpKind::GeluMul;
      o.in = Slot::Gate;
      o.in2 = Slot::Up;
      o.out = Slot::Inter;
      o.cols = vcfg_.intermediate;
      ops.push_back(o);
    }
    gemv(Slot::Inter, Slot::Tmp, "mlp.down_proj.linear.weight", H, vcfg_.intermediate);
    rms(Slot::Tmp, Slot::Tmp, W("post_feedforward_layernorm.weight"), 1, H);
    add_x(Slot::Tmp);
  }

  // Epilogue: standardize, when the checkpoint has it. Pooling and projection are NOT
  // ops -- pooling changes the token count, so the executor has to be re-entered with
  // the new count; see encode_image.
  if (vcfg_.standardize) {
    Op o;
    o.kind = OpKind::Standardize;
    o.in = Slot::X;
    o.out = Slot::X;
    o.cols = H;
    o.weight = dev_at(v + "std_bias");
    o.aux_ptr = dev_at(v + "std_scale");
    vplan_.epilogue.push_back(o);
  }
  (void)patch_dim;
}

std::vector<float> PlanCudaEngine::encode_image_stage(const std::vector<float>& pixels,
                                                      const std::vector<int>& pos_x,
                                                      const std::vector<int>& pos_y, int stage) {
  using namespace opplan;
  const int P = static_cast<int>(pos_x.size());
  const int H = vcfg_.hidden;
  G4_CHECK(cudaMemcpyAsync(d_vis_pixels_, pixels.data(), pixels.size() * sizeof(float),
                           cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_vis_pos_x_, pos_x.data(), pos_x.size() * sizeof(int),
                           cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_vis_pos_y_, pos_y.data(), pos_y.size() * sizeof(int),
                           cudaMemcpyHostToDevice, stream_));
  ExecCtx ctx{vslot_ptr_.data(), P, /*causal=*/false};
  execute_ops(vplan_.prologue.data(), vplan_.prologue.size(), 0, 0, ctx);
  // stage >= 0: run that many whole layers, return Slot::X.
  // stage <  0: run the first |stage| OPS of layer 0 and return Slot::Q -- lets the
  //             gate look inside a layer (e.g. Q straight after RoPE).
  std::size_t n;
  const __half* src;
  if (stage < 0) {
    const auto& ops = vplan_.layers[0].ops;
    const std::size_t take = std::min<std::size_t>(static_cast<std::size_t>(-stage), ops.size());
    execute_ops(ops.data(), take, 0, 0, ctx);
    if (take <= 1) {  // only the input norm has run; Q is not written yet
      n = static_cast<std::size_t>(P) * H;
      src = vslot_ptr_[static_cast<int>(Slot::XNorm)];
    } else {
      n = static_cast<std::size_t>(P) * vcfg_.heads * vcfg_.head_dim;
      src = vslot_ptr_[static_cast<int>(Slot::Q)];
    }
  } else {
    const int nl = std::min(stage, vcfg_.layers);
    for (int L = 0; L < nl; ++L)
      execute_ops(vplan_.layers[L].ops.data(), vplan_.layers[L].ops.size(), L, 0, ctx);
    n = static_cast<std::size_t>(P) * H;
    src = vslot_ptr_[static_cast<int>(Slot::X)];
  }
  G4_CHECK(cudaStreamSynchronize(stream_));

  std::vector<__half> host(n);
  G4_CHECK(cudaMemcpy(host.data(), src, n * sizeof(__half), cudaMemcpyDeviceToHost));
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = __half2float(host[i]);
  return out;
}

std::vector<float> PlanCudaEngine::encode_image(const std::vector<float>& pixels,
                                                const std::vector<int>& pos_x,
                                                const std::vector<int>& pos_y, int out_tokens) {
  using namespace opplan;
  if (!vcfg_.present) throw std::runtime_error("this model has no vision tower");
  const int P = static_cast<int>(pos_x.size());
  if (P > vis_max_patches_) throw std::runtime_error("too many patches for the vision buffers");
  const int H = vcfg_.hidden;

  G4_CHECK(cudaMemcpyAsync(d_vis_pixels_, pixels.data(), pixels.size() * sizeof(float),
                           cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_vis_pos_x_, pos_x.data(), pos_x.size() * sizeof(int),
                           cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_vis_pos_y_, pos_y.data(), pos_y.size() * sizeof(int),
                           cudaMemcpyHostToDevice, stream_));

  // The tower is bidirectional and cacheless: ONE sequence pass over all P patches.
  ExecCtx ctx{vslot_ptr_.data(), P, /*causal=*/false};
  execute_ops(vplan_.prologue.data(), vplan_.prologue.size(), 0, 0, ctx);
  for (int L = 0; L < vcfg_.layers; ++L)
    execute_ops(vplan_.layers[L].ops.data(), vplan_.layers[L].ops.size(), L, 0, ctx);
  execute_ops(vplan_.epilogue.data(), vplan_.epilogue.size(), 0, 0, ctx);

  // Pool the patches down to out_tokens soft tokens, then project into text space.
  const int k = vcfg_.pooling_kernel;
  int max_x = 0;
  for (int i = 0; i < P; ++i) max_x = std::max(max_x, pos_x[i]);
  const int cells_x = (max_x / k) + 1;
  __half* pooled = vslot_ptr_[static_cast<int>(Slot::XNorm)];
  kernels::launch_avg_pool_patches(vslot_ptr_[static_cast<int>(Slot::X)], d_vis_pos_x_,
                                   d_vis_pos_y_, pooled, P, H, k, cells_x, out_tokens,
                                   std::sqrt(static_cast<float>(H)), stream_);

  // projector: weightless RMS norm -> linear into the text hidden size
  kernels::launch_rmsnorm(pooled, d_ones_, pooled, out_tokens, H, vcfg_.rms_eps, stream_);
  __half* proj_out = vslot_ptr_[static_cast<int>(Slot::Tmp)];
  kernels::launch_rowmajor_half_gemm_f16(dev_at("model.embed_vision.embedding_projection.weight"),
                                         pooled, proj_out, cfg_.hidden, H, out_tokens, stream_);
  G4_CHECK(cudaStreamSynchronize(stream_));

  const std::size_t n = static_cast<std::size_t>(out_tokens) * cfg_.hidden;
  std::vector<__half> host(n);
  G4_CHECK(cudaMemcpy(host.data(), proj_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = __half2float(host[i]);
  return out;
}

void PlanCudaEngine::build_plan() {
  using namespace opplan;
  slot_ptr_[static_cast<int>(Slot::X)] = d_x_;
  slot_ptr_[static_cast<int>(Slot::XNorm)] = d_x_norm_;
  slot_ptr_[static_cast<int>(Slot::Q)] = d_q_;
  slot_ptr_[static_cast<int>(Slot::K)] = d_k_;
  slot_ptr_[static_cast<int>(Slot::V)] = d_v_;
  slot_ptr_[static_cast<int>(Slot::Att)] = d_att_;
  slot_ptr_[static_cast<int>(Slot::Tmp)] = d_tmp_;
  slot_ptr_[static_cast<int>(Slot::Gate)] = d_gate_;
  slot_ptr_[static_cast<int>(Slot::Up)] = d_up_;
  slot_ptr_[static_cast<int>(Slot::Inter)] = d_inter_;
  slot_ptr_[static_cast<int>(Slot::PleGate)] = d_ple_gate_;
  slot_ptr_[static_cast<int>(Slot::PleRaw)] = d_ple_raw_;
  slot_ptr_[static_cast<int>(Slot::PleAll)] = d_ple_;
  // Gated / linear-attention working set (null for models that never emit those
  // ops, so Gemma's plan is byte-for-byte unchanged).
  slot_ptr_[static_cast<int>(Slot::QPair)] = d_q_pair_;
  slot_ptr_[static_cast<int>(Slot::QGate)] = d_q_gate_;
  slot_ptr_[static_cast<int>(Slot::LinMix)] = d_lin_mix_;
  slot_ptr_[static_cast<int>(Slot::LinZ)] = d_lin_z_;
  slot_ptr_[static_cast<int>(Slot::LinA)] = d_lin_a_;
  slot_ptr_[static_cast<int>(Slot::LinB)] = d_lin_b_;
  slot_ptr_[static_cast<int>(Slot::LinQ)] = d_lin_q_;
  slot_ptr_[static_cast<int>(Slot::LinK)] = d_lin_k_;
  slot_ptr_[static_cast<int>(Slot::LinV)] = d_lin_v_;
  slot_ptr_[static_cast<int>(Slot::LinAtt)] = d_lin_att_;
  slot_ptr_[static_cast<int>(Slot::MoeRouterIn)] = d_moe_router_in_;
  slot_ptr_[static_cast<int>(Slot::MoeLogits)] = d_moe_logits_;
  slot_ptr_[static_cast<int>(Slot::MoeInter)] = d_moe_inter_;
  slot_ptr_[static_cast<int>(Slot::MoeOut)] = d_moe_out_;

  if (cfg_.family == Family::Qwen35) {
    build_qwen35_plan();
    return;
  }

  const int H = cfg_.hidden;

  // --- prologue: token -> embeddings (+ scale, and the PLE build when present) ---
  // Mirrors the former forward_one head + build_per_layer_inputs, as data.
  {
    std::vector<Op>& pro = plan_.prologue;
    auto emb = [&](const char* name, Slot out, int dim) {
      Op o;
      o.kind = OpKind::EmbeddingLookup;
      o.out = out;
      o.cols = dim;
      o.weight = dev_at(name);
      pro.push_back(o);
    };
    auto sc = [&](Slot s, int len, float scl) {
      Op o;
      o.kind = OpKind::ScaleCopy;
      o.in = s;
      o.out = s;
      o.cols = len;
      o.scale = scl;
      pro.push_back(o);
    };
    emb((wprefix_ + "embed_tokens.weight").c_str(), Slot::X, H);
    sc(Slot::X, H, std::sqrt((float)H));  // embed * sqrt(hidden)
    // Image embeddings are spliced in HERE -- before anything reads X. In particular the
    // per-layer-input projection below reads X, and HF projects it from the POST-scatter
    // embeddings.
    plan_.embed_ready = pro.size();
    if (has_ple()) {
      const int ple = cfg_.hidden_size_per_layer_input;
      const int tot = cfg_.num_layers * ple;
      // ple_raw = embed_tokens_per_layer[token] * sqrt(ple)
      emb((wprefix_ + "embed_tokens_per_layer.weight").c_str(), Slot::PleRaw, tot);
      sc(Slot::PleRaw, tot, std::sqrt((float)ple));
      // ple = rmsnorm( (W_proj · x) * hidden^-0.5 )   [x = the scaled embeds in Slot::X]
      Op g;
      g.kind = OpKind::Gemv;
      g.in = Slot::X;
      g.out = Slot::PleAll;
      g.weight = dev_at(wprefix_ + "per_layer_model_projection.weight");
      g.cols = tot;
      g.in_dim = cfg_.hidden;
      pro.push_back(g);
      sc(Slot::PleAll, tot, std::pow((float)cfg_.hidden, -0.5f));
      Op n;
      n.kind = OpKind::RmsNorm;
      n.in = Slot::PleAll;
      n.out = Slot::PleAll;
      n.weight = dev_at(wprefix_ + "per_layer_projection_norm.weight");
      n.rows = cfg_.num_layers;
      n.cols = ple;
      pro.push_back(n);
      // per_layer_inputs = (ple + ple_raw) * 2^-0.5
      Op a;
      a.kind = OpKind::AddInplace;
      a.out = Slot::PleAll;
      a.in = Slot::PleRaw;
      a.cols = tot;
      pro.push_back(a);
      sc(Slot::PleAll, tot, std::pow(2.0f, -0.5f));
    }
  }

  plan_.layers.assign(cfg_.num_layers, LayerPlan{});
  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = wprefix_ + "layers." + std::to_string(L) + ".";
    auto W = [&](const char* t) { return dev_at(p + t); };
    const int hd = head_dim_of(L);
    const int nq = cfg_.num_heads, nkv = kv_heads_of(L);
    const int qdim = nq * hd, kvdim = nkv * hd;
    const bool full = cfg_.layer_full[L] != 0;
    const bool shared = L >= cfg_.first_shared_layer && cfg_.first_shared_layer > 0;
    const bool keqv = k_eq_v(L);
    const RopeTable rt = full ? RopeTable::Full : RopeTable::Sliding;
    const int inter = cfg_.intermediate * ((cfg_.use_double_wide_mlp && shared) ? 2 : 1);

    LayerPlan& lp = plan_.layers[L];
    lp.layer_index = L;
    auto& ops = lp.ops;
    auto rms = [&](Slot in, Slot out, const __half* w, int rows, int cols) {
      Op o;
      o.kind = OpKind::RmsNorm;
      o.in = in;
      o.out = out;
      o.weight = w;
      o.rows = rows;
      o.cols = cols;
      ops.push_back(o);
    };
    // Resolved by NAME so a weight that was quantized at load binds its int4 form.
    // The choice is made here, once — the hot loop just runs whatever the op holds.
    auto gemv = [&](Slot in, Slot out, const char* t, int out_dim, int in_dim) {
      Op o;
      o.kind = OpKind::Gemv;
      o.in = in;
      o.out = out;
      o.cols = out_dim;
      o.in_dim = in_dim;
      const std::string full = p + t;
      const auto q = qdev_.find(full);
      if (q != qdev_.end()) {
        o.qweight = q->second.packed;
        o.qscales = q->second.scales;
        o.qbits = weight_quant_bits_;
        o.qgroup = q->second.group;
      } else {
        o.weight = dev_at(full);
      }
      ops.push_back(o);
    };
    auto rope = [&](Slot s, int heads) {
      Op o;
      o.kind = OpKind::Rope;
      o.in = s;
      o.out = s;
      o.heads = heads;
      o.head_dim = hd;
      o.rope_table = rt;
      ops.push_back(o);
    };
    auto scale = [&](Slot s, int len, float sc) {
      Op o;
      o.kind = OpKind::ScaleCopy;
      o.in = s;
      o.out = s;
      o.cols = len;
      o.scale = sc;
      ops.push_back(o);
    };
    auto add_x = [&](Slot src) {
      Op o;
      o.kind = OpKind::AddInplace;
      o.out = Slot::X;
      o.in = src;
      o.cols = H;
      ops.push_back(o);
    };

    // --- attention block ---
    rms(Slot::X, Slot::XNorm, W("input_layernorm.weight"), 1, H);
    gemv(Slot::XNorm, Slot::Q, "self_attn.q_proj.weight", qdim, H);
    rms(Slot::Q, Slot::Q, W("self_attn.q_norm.weight"), nq, hd);
    rope(Slot::Q, nq);
    if (!shared) {
      gemv(Slot::XNorm, Slot::K, "self_attn.k_proj.weight", kvdim, H);
      if (keqv) {  // V shares the raw k_proj output (before k_norm/rope)
        Op o;
        o.kind = OpKind::CopySlot;
        o.in = Slot::K;
        o.out = Slot::V;
        o.cols = kvdim;
        ops.push_back(o);
      } else {
        gemv(Slot::XNorm, Slot::V, "self_attn.v_proj.weight", kvdim, H);
      }
      rms(Slot::K, Slot::K, W("self_attn.k_norm.weight"), nkv, hd);
      rope(Slot::K, nkv);
      rms(Slot::V, Slot::V, nullptr, nkv, hd);  // weightless v-norm (ones)
      Op st;
      st.kind = OpKind::KvStore;
      st.cols = kvdim;
      ops.push_back(st);
    }
    // net attention scale = 1.0: pre-scale q by sqrt(hd) to cancel the kernel's 1/sqrt(hd).
    scale(Slot::Q, qdim, std::sqrt((float)hd));
    {
      Op o;
      o.kind = OpKind::Attention;
      o.in = Slot::Q;
      o.out = Slot::Att;
      o.heads = nq;
      o.kv_heads = nkv;
      o.head_dim = hd;
      o.full_attention = full;
      o.sliding_window = cfg_.sliding_window;
      ops.push_back(o);
    }
    gemv(Slot::Att, Slot::Tmp, "self_attn.o_proj.weight", H, qdim);
    rms(Slot::Tmp, Slot::Tmp, W("post_attention_layernorm.weight"), 1, H);
    add_x(Slot::Tmp);

    // --- MLP block (GeGLU; double-wide on shared layers) ---
    rms(Slot::X, Slot::XNorm, W("pre_feedforward_layernorm.weight"), 1, H);
    gemv(Slot::XNorm, Slot::Gate, "mlp.gate_proj.weight", inter, H);
    gemv(Slot::XNorm, Slot::Up, "mlp.up_proj.weight", inter, H);
    {
      Op o;
      o.kind = OpKind::GeluMul;
      o.in = Slot::Gate;
      o.in2 = Slot::Up;
      o.out = Slot::Inter;
      o.cols = inter;
      ops.push_back(o);
    }
    gemv(Slot::Inter, Slot::Tmp, "mlp.down_proj.weight", H, inter);
    // MoE models run a second, ROUTED feed-forward in parallel with the dense one and
    // sum them. Slot::Tmp holds the dense output (pre-norm) on entry, and the summed
    // result on exit, so the shared tail below is identical either way.
    // LLAMA_INFER_PLAN_NO_MOE=1 drops the routed branch, leaving the dense MLP only.
    // Bisection aid: it separates "the shared Gemma path is wrong" from "the MoE ops
    // are wrong" (the model is degraded but must stay finite and coherent-ish).
    if (cfg_.enable_moe_block && !std::getenv("LLAMA_INFER_PLAN_NO_MOE"))
      append_moe_ffn_ops(ops, p, L);
    rms(Slot::Tmp, Slot::Tmp, W("post_feedforward_layernorm.weight"), 1, H);
    add_x(Slot::Tmp);

    // --- Per-Layer Input injection (only when the model has PLE) ---
    if (has_ple()) {
      const int ple = cfg_.hidden_size_per_layer_input;
      gemv(Slot::X, Slot::PleGate, "per_layer_input_gate.weight", ple, H);
      Op g;
      g.kind = OpKind::GeluMul;
      g.in = Slot::PleGate;
      g.out = Slot::PleGate;
      g.cols = ple;
      g.in2 = Slot::PleAll;    // gelu(gate) * per_layer_input[L]
      g.aux_offset = L * ple;  // ...which is layer L's window of it
      ops.push_back(g);
      gemv(Slot::PleGate, Slot::Tmp, "per_layer_projection.weight", H, ple);
      rms(Slot::Tmp, Slot::Tmp, W("post_per_layer_input_norm.weight"), 1, H);
      add_x(Slot::Tmp);
    }

    // --- per-layer scalar ---
    scale(Slot::X, H, layer_scalar_host_[L]);
  }

  // --- epilogue: final norm -> LM head (float logits). Softcap stays host-side. ---
  {
    std::vector<Op>& epi = plan_.epilogue;
    Op n;
    n.kind = OpKind::RmsNorm;
    n.in = Slot::X;
    n.out = Slot::XNorm;
    n.weight = dev_at(wprefix_ + "norm.weight");
    n.rows = 1;
    n.cols = H;
    epi.push_back(n);
    Op h;
    h.kind = OpKind::LmHead;
    h.in = Slot::XNorm;
    h.weight = dev_at(wprefix_ + "embed_tokens.weight");
    h.cols = cfg_.vocab;
    h.in_dim = H;
    epi.push_back(h);
  }
}

// Execute an op list at `position`. Branch-free over model identity — the only
// runtime-varying inputs are `position` (RoPE / KV store / attention window) and
// the executor's device token buffer (EmbeddingLookup). `layer` selects the
// cache for KvStore/Attention; the prologue/epilogue pass -1 (no such ops).
void PlanCudaEngine::execute_ops(const opplan::Op* ops, std::size_t n, int layer, int position,
                                 const ExecCtx& ctx) {
  using namespace opplan;
  auto S = [&](Slot s) -> __half* { return const_cast<__half*>(ctx.slots[static_cast<int>(s)]); };
  // The plan holds weights as opaque void* so the IR stays backend-neutral (see
  // op_plan.hpp). This is the CUDA executor's cast back to its own half type.
  auto HW = [](const void* p) { return static_cast<const __half*>(p); };
  auto QW = [](const void* p) { return static_cast<const std::int8_t*>(p); };
  auto QS = [](const void* p) { return static_cast<const float*>(p); };
  // Sequence mode multiplies an op's extent by the token count. RmsNorm normalises
  // over `cols` in groups of `rows`, and slots are [tokens][rows*cols] contiguous, so
  // N tokens is just N times as many groups -- no new kernel.
  const int T = ctx.tokens;
  const bool seq = T > 1;
  auto rows_x = [&](int rows) { return rows * T; };
  auto len_x = [&](int len) { return len * T; };
  for (std::size_t idx = 0; idx < n; ++idx) {
    const Op& op = ops[idx];
    switch (op.kind) {
      case OpKind::EmbeddingLookup:
        kernels::launch_embedding_lookup(HW(op.weight), seq ? d_seq_tokens_ : d_tok_, S(op.out), T,
                                         op.cols, stream_);
        break;
      case OpKind::LmHead:
        // In sequence mode only the LAST token's logits are wanted -- running the head
        // over the whole prompt would cost a 262K-wide GEMM per token for nothing.
        kernels::launch_rowmajor_half_gemv_f32(
            HW(op.weight),
            S(op.in) + static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(op.in_dim),
            d_logits_, op.cols, op.in_dim, stream_);
        break;
      case OpKind::RmsNorm:
        // norm_offset ⇒ the weight is applied as (1 + w) (Qwen3.5-style).
        if (op.norm_offset) {
          kernels::launch_rmsnorm_offset(S(op.in), HW(op.weight), S(op.out), rows_x(op.rows),
                                         op.cols, cfg_.rms_eps, stream_);
        } else {
          kernels::launch_rmsnorm(S(op.in), HW(op.weight) ? HW(op.weight) : d_ones_, S(op.out),
                                  rows_x(op.rows), op.cols, cfg_.rms_eps, stream_);
        }
        break;
      case OpKind::Gemv:
        // Weight encoding AND scale granularity were chosen at load; the op just
        // carries them.
        if (op.qbits == 4 && op.qgroup > 0) {
          kernels::launch_weight_only_int4_matvec_grouped(QW(op.qweight), QS(op.qscales), S(op.in),
                                                          S(op.out), op.cols, op.in_dim, op.qgroup,
                                                          stream_);
        } else if (op.qbits == 8 && op.qgroup > 0) {
          kernels::launch_weight_only_int8_matvec_grouped(QW(op.qweight), QS(op.qscales), S(op.in),
                                                          S(op.out), op.cols, op.in_dim, op.qgroup,
                                                          stream_);
        } else if (op.qbits == 4) {
          kernels::launch_weight_only_int4_matvec(QW(op.qweight), QS(op.qscales), S(op.in),
                                                  S(op.out), op.cols, op.in_dim, stream_);
        } else if (op.qbits == 8) {
          kernels::launch_weight_only_int8_matvec(QW(op.qweight), QS(op.qscales), S(op.in),
                                                  S(op.out), op.cols, op.in_dim, stream_);
        } else if (seq) {
          kernels::launch_rowmajor_half_gemm_f16(HW(op.weight), S(op.in), S(op.out), op.cols,
                                                 op.in_dim, T, stream_, op.clip_in_min,
                                                 op.clip_in_max, op.clip_out_min, op.clip_out_max);
        } else {
          kernels::launch_rowmajor_half_gemv_f16(HW(op.weight), S(op.in), S(op.out), op.cols,
                                                 op.in_dim, stream_);
        }
        break;
      case OpKind::PatchEmbed:
        kernels::launch_patch_embed(HW(op.weight), d_vis_pixels_, HW(op.aux_ptr), d_vis_pos_x_,
                                    d_vis_pos_y_, S(op.out), T, op.cols, op.in_dim, op.rows,
                                    stream_);
        break;
      case OpKind::Rope2D:
        kernels::launch_rope_2d_inplace(S(op.in), d_vis_pos_x_, d_vis_pos_y_, op.heads, op.head_dim,
                                        T, d_vrope_cos_, d_vrope_sin_, stream_);
        break;
      case OpKind::AvgPoolPatches:
        kernels::launch_avg_pool_patches(S(op.in), d_vis_pos_x_, d_vis_pos_y_, S(op.out), T,
                                         op.cols, op.heads, op.kv_heads, op.rows, op.scale,
                                         stream_);
        break;
      case OpKind::Standardize:
        kernels::launch_standardize(S(op.in), HW(op.weight), HW(op.aux_ptr), T, op.cols, stream_);
        break;
      case OpKind::MulVec:
        kernels::launch_mul_vec(S(op.in), HW(op.weight), S(op.out), op.cols, op.scale, stream_);
        break;
      case OpKind::MoeRouterTopk:
        // Writes the selected experts to DEVICE buffers; the expert ops below read
        // them there, so a token never round-trips to the host mid-layer.
        kernels::launch_moe_router_topk_softmax(S(op.in), op.cols, op.heads, d_moe_idx_, d_moe_w_,
                                                stream_, HW(op.weight));
        break;
      case OpKind::MoeGateUpGeglu:
        kernels::launch_moe_gate_up_geglu(
            op.qbits ? static_cast<const void*>(op.qweight) : static_cast<const void*>(op.weight),
            QS(op.qscales), op.qbits, op.qgroup, S(op.in), d_moe_idx_, S(op.out), op.cols,
            op.in_dim, op.heads, stream_);
        break;
      case OpKind::MoeDownAccum:
        kernels::launch_moe_down_accum(
            op.qbits ? static_cast<const void*>(op.qweight) : static_cast<const void*>(op.weight),
            QS(op.qscales), op.qbits, op.qgroup, S(op.in), d_moe_idx_, d_moe_w_, S(op.out), op.cols,
            op.in_dim, op.heads, stream_);
        break;
      case OpKind::Rope: {
        const float* cosT = op.rope_table == RopeTable::Full ? d_cos_full_ : d_cos_sliding_;
        const float* sinT = op.rope_table == RopeTable::Full ? d_sin_full_ : d_sin_sliding_;
        if (op.rotary_dim > 0) {
          // Partial RoPE over Q (in) and K (in2) together. No device-position
          // variant exists, so a plan using it runs without the decode graph
          // (capability tiering: the plan decides, not the model name).
          kernels::launch_rope_inplace_partial_table(S(op.in), S(op.in2), op.heads, op.kv_heads,
                                                     op.head_dim, op.rotary_dim, position, cosT,
                                                     sinT, stream_);
        } else if (seq) {
          // Sequence prefill: token t sits at position (chunk start + t).
          kernels::launch_rope_seq_table(S(op.in), op.heads, op.head_dim, position, T, cosT, sinT,
                                         0, stream_);
        } else if (device_pos_mode_) {
          // Single-tensor RoPE via the device-position kernel (k-branch guarded off
          // by num_heads_k=0). Graph-capturable: position read from d_position_.
          kernels::launch_rope_inplace_device_pos(S(op.in), nullptr, op.heads, 0, op.head_dim,
                                                  d_position_, cosT, sinT, stream_);
        } else {
          kernels::launch_rope_inplace_table(S(op.in), S(op.out), op.heads, 0, op.head_dim,
                                             position, cosT, sinT, stream_);
        }
        break;
      }
      case OpKind::ScaleCopy:
        kernels::launch_scale_copy(S(op.in), S(op.out), len_x(op.cols), op.scale, stream_);
        break;
      case OpKind::CopySlot:
        G4_CHECK(cudaMemcpyAsync(S(op.out), S(op.in),
                                 static_cast<std::size_t>(len_x(op.cols)) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, stream_));
        break;
      case OpKind::KvStore: {
        __half* kc = caches_k_[layer];
        __half* vc = caches_v_[layer];
        const std::size_t kvdim = static_cast<std::size_t>(op.cols);
        if (device_pos_mode_) {
          kernels::launch_store_kv_device_pos(S(Slot::K), S(Slot::V), kc, vc, d_position_,
                                              static_cast<int>(kvdim), max_ctx_, stream_);
        } else {
          // Sequence prefill appends T rows at once: the K/V slots are [T][kvdim] and the
          // cache rows are contiguous, so it is the same copy, T times as long.
          const std::size_t rows = static_cast<std::size_t>(T);
          G4_CHECK(cudaMemcpyAsync(kc + static_cast<std::size_t>(position) * kvdim, S(Slot::K),
                                   rows * kvdim * sizeof(__half), cudaMemcpyDeviceToDevice,
                                   stream_));
          G4_CHECK(cudaMemcpyAsync(vc + static_cast<std::size_t>(position) * kvdim, S(Slot::V),
                                   rows * kvdim * sizeof(__half), cudaMemcpyDeviceToDevice,
                                   stream_));
        }
        break;
      }
      case OpKind::Attention: {
        if (seq && ctx.cached) {
          // TEXT prefill: K/V were appended to this layer's cache by KvStore, so attend
          // over the CACHE (it also holds any earlier context). Sliding layers pass their
          // window, and ctx.limits (when set) makes an image span bidirectional.
          const int window = op.full_attention ? 0 : op.sliding_window;
          kernels::launch_attention_prefill(S(op.in), caches_k_[layer], caches_v_[layer], S(op.out),
                                            T, position, op.heads, op.kv_heads, op.head_dim,
                                            stream_,
                                            /*causal=*/true, ctx.limits, window);
          break;
        }
        if (seq) {
          // Cacheless sequence (a vision tower): the K/V slots ARE the "cache", and
          // causal comes from the context (false for a bidirectional encoder).
          kernels::launch_attention_prefill(S(op.in), S(Slot::K), S(Slot::V), S(op.out), T, 0,
                                            op.heads, op.kv_heads, op.head_dim, stream_,
                                            ctx.causal);
          break;
        }
        __half* kc = caches_k_[layer];
        __half* vc = caches_v_[layer];
        const std::size_t kvdim = static_cast<std::size_t>(op.kv_heads) * op.head_dim;
        if (device_pos_mode_) {
          // Device-position kernel: seq derived from d_position_ on-device (graph-
          // safe). Sliding layers pass their window so k_start is computed on device.
          const int window = op.full_attention ? 0 : op.sliding_window;
          kernels::launch_attention_step_device_pos(S(op.in), kc, vc, S(op.out), d_position_,
                                                    op.heads, op.kv_heads, op.head_dim, stream_,
                                                    nullptr, nullptr, nullptr, 0, true, window);
        } else {
          const int seq = position + 1;
          int k_start = 0;
          if (!op.full_attention && op.sliding_window > 0 && seq > op.sliding_window)
            k_start = seq - op.sliding_window;
          const int att_seq = seq - k_start;
          kernels::launch_attention_step(S(op.in), kc + static_cast<std::size_t>(k_start) * kvdim,
                                         vc + static_cast<std::size_t>(k_start) * kvdim, S(op.out),
                                         att_seq, op.heads, op.kv_heads, op.head_dim, stream_);
        }
        break;
      }
      case OpKind::GeluMul: {
        const __half* b = HW(op.aux_ptr) ? HW(op.aux_ptr) : (S(op.in2) + op.aux_offset);
        if (seq && op.in2 == Slot::PleAll) {
          // b is a window of a WIDER per-token tensor, so it strides differently.
          const int stride = cfg_.num_layers * cfg_.hidden_size_per_layer_input;
          kernels::launch_gelu_mul_strided(S(op.in), b, S(op.out), op.cols, T, stride, stream_);
        } else {
          kernels::launch_gelu_mul(S(op.in), b, S(op.out), len_x(op.cols), stream_);
        }
        break;
      }
      case OpKind::AddInplace:
        kernels::launch_add_inplace(S(op.out), S(op.in), len_x(op.cols), stream_);
        break;

      // ── gated / linear-attention ("delta-net") extensions ──
      case OpKind::SiluMul: {
        const __half* b = HW(op.aux_ptr) ? HW(op.aux_ptr) : S(op.in2);
        kernels::launch_silu_mul(S(op.in), b, S(op.out), len_x(op.cols), stream_);
        break;
      }
      case OpKind::SplitHeadHalves:
        kernels::launch_split_interleaved_head_halves(S(op.in), S(op.out), S(op.out2), op.heads,
                                                      op.head_dim, stream_);
        break;
      case OpKind::SigmoidGate:
        kernels::launch_apply_sigmoid_gate_inplace(S(op.out), S(op.in2), op.cols, stream_);
        break;
      case OpKind::LinearConv1d:
        kernels::launch_linear_conv1d_silu(HW(op.weight), lin_conv_state(layer), S(op.in), op.cols,
                                           op.conv_kernel, stream_);
        break;
      case OpKind::RepeatLinearHeads:
        kernels::launch_repeat_linear_heads(S(op.in), S(Slot::LinQ), S(Slot::LinK), S(Slot::LinV),
                                            op.num_k_heads, op.num_v_heads, op.key_head_dim,
                                            op.value_head_dim, stream_);
        break;
      case OpKind::LinearAttentionStep:
        kernels::launch_linear_attention_step(
            S(Slot::LinQ), S(Slot::LinK), S(Slot::LinV), S(Slot::LinZ), S(Slot::LinA),
            S(Slot::LinB), op.auxf_a, op.auxf_b, HW(op.aux_ptr), lin_recurrent_state(layer),
            S(Slot::LinAtt), op.num_v_heads, op.key_head_dim, op.value_head_dim, op.eps, stream_);
        break;
    }
  }
}

float* PlanCudaEngine::lin_conv_state(int layer) {
  return d_lin_conv_state_ + static_cast<std::size_t>(layer) * lin_conv_state_stride_;
}

float* PlanCudaEngine::lin_recurrent_state(int layer) {
  return d_lin_recurrent_state_ + static_cast<std::size_t>(layer) * lin_recurrent_state_stride_;
}

// Qwen3.5 as a PLAN: hybrid full-attention + gated delta-net. Mirrors the fork's
// forward exactly; the ops, state and executor are all shared machinery.
void PlanCudaEngine::build_qwen35_plan() {
  using namespace opplan;
  const int H = cfg_.hidden;
  const float eps = cfg_.rms_eps;

  // Prologue: embed only (no scale, no PLE).
  {
    Op e;
    e.kind = OpKind::EmbeddingLookup;
    e.out = Slot::X;
    e.cols = H;
    e.weight = static_cast<const __half*>(dev_["model.language_model.embed_tokens.weight"]);
    plan_.prologue.push_back(e);
  }

  plan_.layers.assign(cfg_.num_layers, LayerPlan{});
  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = "model.language_model.layers." + std::to_string(L) + ".";
    LayerPlan& lp = plan_.layers[L];
    lp.layer_index = L;
    auto& ops = lp.ops;

    auto W = [&](const char* t) { return static_cast<const __half*>(dev_[p + t]); };
    auto rms = [&](Slot in, Slot out, const char* t, int rows, int cols) {
      Op o;
      o.kind = OpKind::RmsNorm;
      o.in = in;
      o.out = out;
      o.weight = W(t);
      o.rows = rows;
      o.cols = cols;
      o.norm_offset = true;  // Qwen3.5 uses (1 + w)
      ops.push_back(o);
    };
    auto gemv = [&](Slot in, Slot out, const char* t, int out_dim, int in_dim) {
      Op o;
      o.kind = OpKind::Gemv;
      o.in = in;
      o.out = out;
      o.cols = out_dim;
      o.in_dim = in_dim;
      const auto q = qdev_.find(p + t);
      if (q != qdev_.end()) {
        o.qweight = q->second.packed;
        o.qscales = q->second.scales;
        o.qbits = weight_quant_bits_;
        o.qgroup = q->second.group;
      } else {
        o.weight = W(t);
      }
      ops.push_back(o);
    };
    auto add_x = [&](Slot src) {
      Op o;
      o.kind = OpKind::AddInplace;
      o.out = Slot::X;
      o.in = src;
      o.cols = H;
      ops.push_back(o);
    };

    rms(Slot::X, Slot::XNorm, "input_layernorm.weight", 1, H);

    if (cfg_.layer_full[L]) {
      // ── full attention (q_proj is a fused q|gate) ──
      gemv(Slot::XNorm, Slot::QPair, "self_attn.q_proj.weight", full_q_dim_ * 2, H);
      gemv(Slot::XNorm, Slot::K, "self_attn.k_proj.weight", full_kv_dim_, H);
      gemv(Slot::XNorm, Slot::V, "self_attn.v_proj.weight", full_kv_dim_, H);
      {
        Op o;
        o.kind = OpKind::SplitHeadHalves;
        o.in = Slot::QPair;
        o.out = Slot::Q;
        o.out2 = Slot::QGate;
        o.heads = cfg_.num_heads;
        o.head_dim = cfg_.head_dim;
        ops.push_back(o);
      }
      rms(Slot::Q, Slot::Q, "self_attn.q_norm.weight", cfg_.num_heads, cfg_.head_dim);
      rms(Slot::K, Slot::K, "self_attn.k_norm.weight", cfg_.num_kv_heads, cfg_.head_dim);
      {
        Op o;
        o.kind = OpKind::Rope;
        o.in = Slot::Q;
        o.in2 = Slot::K;  // partial, Q+K together
        o.heads = cfg_.num_heads;
        o.kv_heads = cfg_.num_kv_heads;
        o.head_dim = cfg_.head_dim;
        o.rotary_dim = rotary_dim_;
        o.rope_table = RopeTable::Full;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::KvStore;
        o.cols = full_kv_dim_;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::Attention;
        o.in = Slot::Q;
        o.out = Slot::Att;
        o.heads = cfg_.num_heads;
        o.kv_heads = cfg_.num_kv_heads;
        o.head_dim = cfg_.head_dim;
        o.full_attention = true;
        o.sliding_window = 0;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::SigmoidGate;
        o.out = Slot::Att;
        o.in2 = Slot::QGate;
        o.cols = full_q_dim_;
        ops.push_back(o);
      }
      gemv(Slot::Att, Slot::Tmp, "self_attn.o_proj.weight", H, full_q_dim_);
    } else {
      // ── gated delta-net ──
      gemv(Slot::XNorm, Slot::LinMix, "linear_attn.in_proj_qkv.weight", linear_conv_dim_, H);
      gemv(Slot::XNorm, Slot::LinZ, "linear_attn.in_proj_z.weight", linear_v_dim_, H);
      gemv(Slot::XNorm, Slot::LinA, "linear_attn.in_proj_a.weight", cfg_.linear_num_value_heads, H);
      gemv(Slot::XNorm, Slot::LinB, "linear_attn.in_proj_b.weight", cfg_.linear_num_value_heads, H);
      {
        Op o;
        o.kind = OpKind::LinearConv1d;
        o.in = Slot::LinMix;
        o.weight = W("linear_attn.conv1d.weight");
        o.cols = linear_conv_dim_;
        o.conv_kernel = cfg_.linear_conv_kernel_dim;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::RepeatLinearHeads;
        o.in = Slot::LinMix;
        o.num_k_heads = cfg_.linear_num_key_heads;
        o.num_v_heads = cfg_.linear_num_value_heads;
        o.key_head_dim = cfg_.linear_key_head_dim;
        o.value_head_dim = cfg_.linear_value_head_dim;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::LinearAttentionStep;
        o.auxf_a = devf_[p + "linear_attn.norm.weight"];
        o.auxf_b = devf_[p + "linear_attn.A_log"];
        o.aux_ptr = W("linear_attn.dt_bias");
        o.num_v_heads = cfg_.linear_num_value_heads;
        o.key_head_dim = cfg_.linear_key_head_dim;
        o.value_head_dim = cfg_.linear_value_head_dim;
        o.eps = eps;
        ops.push_back(o);
      }
      gemv(Slot::LinAtt, Slot::Tmp, "linear_attn.out_proj.weight", H, linear_v_dim_);
    }
    add_x(Slot::Tmp);

    // ── MLP (SwiGLU — SiLU, not GeLU) ──
    rms(Slot::X, Slot::XNorm, "post_attention_layernorm.weight", 1, H);
    gemv(Slot::XNorm, Slot::Gate, "mlp.gate_proj.weight", cfg_.intermediate, H);
    gemv(Slot::XNorm, Slot::Up, "mlp.up_proj.weight", cfg_.intermediate, H);
    {
      Op o;
      o.kind = OpKind::SiluMul;
      o.in = Slot::Gate;
      o.in2 = Slot::Up;
      o.out = Slot::Inter;
      o.cols = cfg_.intermediate;
      ops.push_back(o);
    }
    gemv(Slot::Inter, Slot::Tmp, "mlp.down_proj.weight", H, cfg_.intermediate);
    add_x(Slot::Tmp);
  }

  // Epilogue: final (1+w) norm -> LM head.
  {
    Op n;
    n.kind = OpKind::RmsNorm;
    n.in = Slot::X;
    n.out = Slot::XNorm;
    n.weight = static_cast<const __half*>(dev_["model.language_model.norm.weight"]);
    n.rows = 1;
    n.cols = H;
    n.norm_offset = true;
    plan_.epilogue.push_back(n);
    Op h;
    h.kind = OpKind::LmHead;
    h.in = Slot::XNorm;
    const char* head = dev_.count("lm_head.weight") ? "lm_head.weight"
                                                    : "model.language_model.embed_tokens.weight";
    h.weight = static_cast<const __half*>(dev_[head]);
    h.cols = cfg_.vocab;
    h.in_dim = H;
    plan_.epilogue.push_back(h);
  }
}

// Hot loop for one tower layer: execute its resolved op-plan at `position`.
void PlanCudaEngine::run_layer(int layer, int position) {
  const std::vector<opplan::Op>& ops = plan_.layers[layer].ops;
  execute_ops(ops.data(), ops.size(), layer, position);
}

// Process one token at `position`, reusing the KV cache from earlier positions.
// Positions must be presented in increasing order (prefill then decode) so the
// cache and the KV-share sources are populated before dependent layers read them.
// Sequence-mode buffers: the same slots as decode, but [max_tokens] deep.
// Can this plan run a sequence prefill? Decided by the OPS it emits, not by a model
// name: partial RoPE (Qwen) and the delta-net ops have no sequence kernels yet.
bool PlanCudaEngine::plan_can_sequence_prefill() const {
  using namespace opplan;
  for (const auto& lp : plan_.layers) {
    for (const Op& o : lp.ops) {
      if (o.kind == OpKind::Rope && o.rotary_dim > 0) return false;
      if (o.kind == OpKind::LinearAttentionStep || o.kind == OpKind::LinearConv1d) return false;
    }
  }
  return !plan_.layers.empty();
}

void PlanCudaEngine::allocate_sequence_buffers(int max_tokens) {
  const int H = cfg_.hidden;
  const int maxhd = std::max(cfg_.head_dim_full, cfg_.head_dim_sliding);
  const int maxq = cfg_.num_heads * maxhd;
  const int maxkv = std::max(cfg_.num_kv_heads_sliding * cfg_.head_dim_sliding,
                             cfg_.num_kv_heads_full * cfg_.head_dim_full);
  const int maxinter = cfg_.intermediate * (cfg_.use_double_wide_mlp ? 2 : 1);
  seq_max_tokens_ = max_tokens;

  auto al = [&](opplan::Slot slot, std::size_t per_token) {
    __half* d = nullptr;
    G4_CHECK(cudaMalloc(&d, static_cast<std::size_t>(max_tokens) * per_token * sizeof(__half)));
    sslot_ptr_[static_cast<int>(slot)] = d;
    seq_buffers_.push_back(d);
  };
  using opplan::Slot;
  al(Slot::X, H);
  al(Slot::XNorm, H);
  al(Slot::Tmp, H);
  al(Slot::Q, maxq);
  al(Slot::K, maxkv);
  al(Slot::V, maxkv);
  al(Slot::Att, maxq);
  al(Slot::Gate, maxinter);
  al(Slot::Up, maxinter);
  al(Slot::Inter, maxinter);
  if (has_ple()) {
    const int ple = cfg_.hidden_size_per_layer_input;
    al(Slot::PleGate, ple);
    al(Slot::PleRaw, static_cast<std::size_t>(cfg_.num_layers) * ple);
    al(Slot::PleAll, static_cast<std::size_t>(cfg_.num_layers) * ple);
  }
  G4_CHECK(cudaMalloc(&d_seq_tokens_, static_cast<std::size_t>(max_tokens) * sizeof(int)));
  G4_CHECK(cudaMalloc(&d_seq_limits_, static_cast<std::size_t>(max_tokens) * sizeof(int)));
}

// Prefill a whole prompt in ONE pass instead of token by token.
//
// Two things this buys, beyond speed:
//   - image soft tokens can be spliced over the embeddings of the image placeholder
//     tokens (they are used AS-IS: unlike text embeddings they get no sqrt(hidden) scale);
//   - a per-token key limit makes an image span BIDIRECTIONAL while the surrounding text
//     stays causal, which is what Gemma's use_bidirectional_attention="vision" means.
void PlanCudaEngine::prefill_sequence(const std::vector<int>& tokens, int start_position,
                                      const std::vector<std::vector<float>>& embeds,
                                      const std::vector<int>& limits) {
  using namespace opplan;
  const int T = static_cast<int>(tokens.size());
  if (T == 0) return;
  if (!seq_prefill_ok_) throw std::runtime_error("this plan cannot sequence-prefill");
  if (T > seq_max_tokens_) throw std::runtime_error("prompt longer than the sequence buffers");
  const int H = cfg_.hidden;

  G4_CHECK(cudaMemcpyAsync(d_seq_tokens_, tokens.data(), static_cast<std::size_t>(T) * sizeof(int),
                           cudaMemcpyHostToDevice, stream_));
  const int* dlimits = nullptr;
  if (!limits.empty()) {
    if (static_cast<int>(limits.size()) != T)
      throw std::runtime_error("limits must have one entry per token");
    G4_CHECK(cudaMemcpyAsync(d_seq_limits_, limits.data(),
                             static_cast<std::size_t>(T) * sizeof(int), cudaMemcpyHostToDevice,
                             stream_));
    dlimits = d_seq_limits_;
  }

  ExecCtx ctx{sslot_ptr_.data(), T, /*causal=*/true, /*cached=*/true, dlimits};

  // Prologue up to the point where the token embeddings are final...
  const std::size_t ready = std::min(plan_.embed_ready, plan_.prologue.size());
  execute_ops(plan_.prologue.data(), ready, 0, start_position, ctx);

  // ...splice the image soft tokens over the placeholder tokens' embeddings (used AS-IS:
  // unlike text embeddings they get no sqrt(hidden) scale)...
  if (!embeds.empty()) {
    std::vector<__half> rows(static_cast<std::size_t>(T) * H);
    G4_CHECK(cudaMemcpy(rows.data(), sslot_ptr_[static_cast<int>(Slot::X)],
                        rows.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    bool any = false;
    for (std::size_t i = 0; i < embeds.size() && i < static_cast<std::size_t>(T); ++i) {
      if (embeds[i].empty()) continue;
      if (static_cast<int>(embeds[i].size()) != H)
        throw std::runtime_error("embedding override has the wrong width");
      for (int j = 0; j < H; ++j)
        rows[i * static_cast<std::size_t>(H) + j] = __float2half(embeds[i][j]);
      any = true;
    }
    if (any)
      G4_CHECK(cudaMemcpy(sslot_ptr_[static_cast<int>(Slot::X)], rows.data(),
                          rows.size() * sizeof(__half), cudaMemcpyHostToDevice));
  }

  // ...and only THEN the rest of the prologue, which reads X (the per-layer-input
  // projection does, and HF projects it from the post-splice embeddings).
  execute_ops(plan_.prologue.data() + ready, plan_.prologue.size() - ready, 0, start_position, ctx);

  for (int L = 0; L < cfg_.num_layers; ++L)
    execute_ops(plan_.layers[L].ops.data(), plan_.layers[L].ops.size(), L, start_position, ctx);
  execute_ops(plan_.epilogue.data(), plan_.epilogue.size(), 0, start_position, ctx);
  G4_CHECK(cudaStreamSynchronize(stream_));
  publish_host_logits();  // logits of the LAST prompt token, ready for the first sample
}

void PlanCudaEngine::forward_one(int token, int position, bool compute_logits,
                                 std::vector<float>* per_layer_rms) {
  const int H = cfg_.hidden;

  // Fast path: every logits step is the same single-token forward, so replay a
  // captured CUDA graph (device-position ops) instead of re-launching ~600 kernels.
  // Skipped for the per-layer-rms parity dump (it syncs mid-forward). The graph
  // writes the KV cache at d_position_, so it interleaves correctly with the eager
  // prefill (which fills earlier rows via host position).
  if (compute_logits && per_layer_rms == nullptr && decode_graph_enabled_) {
    if (!decode_graph_ready_) capture_decode_graph();
    G4_CHECK(cudaMemcpyAsync(d_tok_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
    G4_CHECK(cudaMemcpyAsync(d_position_, &position, sizeof(int), cudaMemcpyHostToDevice, stream_));
    G4_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
    // Shared-loop path leaves logits on-device for sample() (greedy argmax / lazy
    // host copy); other callers get host logits now.
    if (!defer_host_logits_) publish_host_logits();
    return;
  }

  // Publish the token; the prologue's EmbeddingLookup ops read d_tok_ on-device.
  G4_CHECK(cudaMemcpyAsync(d_tok_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
  execute_ops(plan_.prologue.data(), plan_.prologue.size(), -1, position);
  for (int L = 0; L < cfg_.num_layers; ++L) {
    run_layer(L, position);
    if (per_layer_rms) {  // oracle parity: capture this layer's output hidden
      std::vector<__half> h(H);
      G4_CHECK(cudaStreamSynchronize(stream_));
      G4_CHECK(cudaMemcpy(h.data(), d_x_, H * sizeof(__half), cudaMemcpyDeviceToHost));
      double ss = 0;
      for (auto v : h) {
        float fv = __half2float(v);
        ss += (double)fv * fv;
      }
      per_layer_rms->push_back((float)std::sqrt(ss / H));
      if (const char* dir = std::getenv("G4_DUMP_DIR")) {
        std::vector<float> hf(H);
        for (int i = 0; i < H; ++i) hf[i] = __half2float(h[i]);
        std::string fn = std::string(dir) + "/cpi_layer_" + std::to_string(L) + ".f32";
        std::FILE* fp = std::fopen(fn.c_str(), "wb");
        if (fp) {
          std::fwrite(hf.data(), sizeof(float), H, fp);
          std::fclose(fp);
        }
      }
    }
  }
  if (!compute_logits) return;
  execute_ops(plan_.epilogue.data(), plan_.epilogue.size(), -1, position);
  if (!defer_host_logits_) publish_host_logits();
}

// Copy d_logits_ to host (last_logits_) and apply the final logit softcap. Used by
// the non-greedy sample path and the host-logits callers (forward_logits/inspect).
void PlanCudaEngine::publish_host_logits() {
  last_logits_.resize(cfg_.vocab);
  G4_CHECK(cudaStreamSynchronize(stream_));
  G4_CHECK(cudaMemcpy(last_logits_.data(), d_logits_, cfg_.vocab * sizeof(float),
                      cudaMemcpyDeviceToHost));
  const float cap = cfg_.final_logit_softcapping;
  if (cap > 0.0f)
    for (float& v : last_logits_) v = cap * std::tanh(v / cap);
}

// Device-argmax greedy fast path (monotonic softcap ⇒ argmax(softcap)==argmax);
// non-greedy brings logits to host (softcapped) and reuses the shared sampler.
int PlanCudaEngine::sample(const runtime::DecodeParams& params, const std::vector<int>& history) {
  const bool greedy_fast_path = params.temperature <= 0.0f && params.repetition_penalty <= 1.0f &&
                                params.no_repeat_ngram_size <= 1 && !params.grammar_mask;
  if (greedy_fast_path) {
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_), cfg_.vocab, d_argmax_,
                                 stream_);
    G4_CHECK(cudaStreamSynchronize(stream_));
    int next = 0;
    G4_CHECK(cudaMemcpy(&next, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
    return next;
  }

  // Device top-k: select the candidate set on the GPU so the host never touches the
  // full vocab (temperature>0 is the real chat path — greedy only covers temp<=0).
  // Eligibility mirrors the shared host top-k sampler exactly.
  const bool device_topk = device_topk_enabled_ && params.temperature > 0.0f && params.top_k > 0 &&
                           params.top_k <= kMaxDeviceTopK && params.top_k < cfg_.vocab &&
                           params.repetition_penalty <= 1.0f && params.no_repeat_ngram_size <= 1 &&
                           !params.grammar_mask;
  if (device_topk) {
    const int k = params.top_k;
    G4_CHECK(cudaMemsetAsync(d_cand_count_, 0, sizeof(int), stream_));
    kernels::launch_topk_float(d_logits_, cfg_.vocab, k, d_topk_part_val_, d_topk_part_idx_,
                               d_topk_val_, d_topk_idx_, stream_);
    // Threshold = the k-th largest RAW logit. The final softcap is monotonic, so
    // {i : raw_i >= raw_kth} is exactly the host's softcapped candidate set.
    kernels::launch_gather_ge_threshold(d_logits_, cfg_.vocab, d_topk_val_ + (k - 1), d_cand_idx_,
                                        d_cand_val_, d_cand_count_, kCandCapacity, stream_);
    G4_CHECK(cudaStreamSynchronize(stream_));

    int count = 0;
    G4_CHECK(cudaMemcpy(&count, d_cand_count_, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > 0 && count <= kCandCapacity) {
      std::vector<int> idx(count);
      std::vector<float> val(count);
      G4_CHECK(cudaMemcpy(idx.data(), d_cand_idx_, count * sizeof(int), cudaMemcpyDeviceToHost));
      G4_CHECK(cudaMemcpy(val.data(), d_cand_val_, count * sizeof(float), cudaMemcpyDeviceToHost));

      const float cap = cfg_.final_logit_softcapping;
      std::vector<detail::SampleCandidate> cand(static_cast<std::size_t>(count));
      for (int i = 0; i < count; ++i) {
        const float v = val[i];
        cand[i] = {idx[i], cap > 0.0f ? cap * std::tanh(v / cap) : v};
      }
      // The gather appends atomically (arbitrary order); the host gathers in index
      // order. Sort to match so the shared sampler sees an identical candidate
      // vector — that is what makes a seeded run byte-identical across both paths.
      std::sort(cand.begin(), cand.end(),
                [](const detail::SampleCandidate& a, const detail::SampleCandidate& b) {
                  return a.id < b.id;
                });
      return detail::dispatch_sample_from_candidates(cand, params.temperature, params.top_p);
    }
    // Pathological tie count overflowed the buffer → fall through to the host path.
  }

  publish_host_logits();
  return runtime::SequenceModel::sample(params, history);
}

std::vector<float> PlanCudaEngine::forward_logits(const std::vector<int>& tokens,
                                                  std::vector<float>* per_layer_rms) {
  // Parity/debug entry: fresh sequence from position 0. Drives forward_one; the
  // last token requests the per-layer dump for oracle comparison.
  defer_host_logits_ = false;  // this entry returns host logits
  for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
    const bool last = pos == static_cast<int>(tokens.size()) - 1;
    forward_one(tokens[pos], pos, last, last ? per_layer_rms : nullptr);
  }
  return last_logits_;
}

void PlanCudaEngine::capture_decode_graph() {
  // Capture the single-token forward (device-position ops) once; replayed for
  // every logits step. Buffers referenced (d_tok_/d_position_/caches/weights/
  // d_logits_) are stable for the engine's lifetime, so one capture serves all
  // positions and all generate() calls. Sync first so no prefill work is pending.
  G4_CHECK(cudaStreamSynchronize(stream_));
  device_pos_mode_ = true;
  G4_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  execute_ops(plan_.prologue.data(), plan_.prologue.size(), -1, 0);
  for (int L = 0; L < cfg_.num_layers; ++L) run_layer(L, 0);
  execute_ops(plan_.epilogue.data(), plan_.epilogue.size(), -1, 0);
  G4_CHECK(cudaStreamEndCapture(stream_, &decode_graph_));
  G4_CHECK(cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0));
  device_pos_mode_ = false;
  decode_graph_ready_ = true;
}

void PlanCudaEngine::benchmark_graph_decode(const std::vector<int>& prompt, int iters,
                                            int pos_override) {
  using clock = std::chrono::steady_clock;
  if (prompt.empty()) {
    std::printf("[graph-bench] empty prompt\n");
    return;
  }
  const int token = prompt.back();
  // Effective prefill: the prompt, padded with its last token up to pos_override.
  std::vector<int> pre = prompt;
  while (static_cast<int>(pre.size()) < pos_override) pre.push_back(token);
  const int pos = static_cast<int>(pre.size());  // decode position for the A/B

  // Prefill (non-graph, host position) to fill the KV cache for 0..pos-1.
  device_pos_mode_ = false;
  for (int i = 0; i < pos; ++i) forward_one(pre[i], i, /*compute_logits=*/false);
  G4_CHECK(cudaStreamSynchronize(stream_));

  // The GPU forward (prologue -> layers -> epilogue), no host copy. Honours
  // device_pos_mode_ (host position `pos` is ignored by the device-pos kernels).
  auto gpu_forward = [&]() {
    execute_ops(plan_.prologue.data(), plan_.prologue.size(), -1, pos);
    for (int L = 0; L < cfg_.num_layers; ++L) run_layer(L, pos);
    execute_ops(plan_.epilogue.data(), plan_.epilogue.size(), -1, pos);
  };
  auto read_softcapped_logits = [&]() {
    std::vector<float> lg(cfg_.vocab);
    G4_CHECK(cudaMemcpy(lg.data(), d_logits_, cfg_.vocab * sizeof(float), cudaMemcpyDeviceToHost));
    const float cap = cfg_.final_logit_softcapping;
    if (cap > 0.0f)
      for (float& v : lg) v = cap * std::tanh(v / cap);
    return lg;
  };
  auto argmax = [&](const std::vector<float>& v) {
    int a = 0;
    for (int i = 1; i < (int)v.size(); ++i)
      if (v[i] > v[a]) a = i;
    return a;
  };

  // Reference: non-graph forward at (token, pos).
  device_pos_mode_ = false;
  G4_CHECK(cudaMemcpyAsync(d_tok_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
  gpu_forward();
  G4_CHECK(cudaStreamSynchronize(stream_));
  const std::vector<float> ref = read_softcapped_logits();

  // Capture the device-position forward into a CUDA graph.
  device_pos_mode_ = true;
  G4_CHECK(cudaMemcpyAsync(d_tok_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_position_, &pos, sizeof(int), cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaStreamSynchronize(stream_));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  G4_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  gpu_forward();
  G4_CHECK(cudaStreamEndCapture(stream_, &graph));
  G4_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Correctness: graph logits vs non-graph reference (valid since pos < window
  // makes full-attention == sliding here).
  G4_CHECK(cudaGraphLaunch(exec, stream_));
  G4_CHECK(cudaStreamSynchronize(stream_));
  const std::vector<float> glog = read_softcapped_logits();
  float maxdiff = 0.0f;
  for (int i = 0; i < cfg_.vocab; ++i) maxdiff = std::max(maxdiff, std::fabs(glog[i] - ref[i]));
  const int a_ref = argmax(ref), a_g = argmax(glog);
  std::printf("[graph-bench] pos=%d argmax non-graph=%d graph=%d %s | max|logit diff|=%.4f\n", pos,
              a_ref, a_g, a_ref == a_g ? "MATCH" : "MISMATCH", maxdiff);

  auto time_loop = [&](const std::function<void()>& fn) {
    G4_CHECK(cudaStreamSynchronize(stream_));
    const auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) fn();
    G4_CHECK(cudaStreamSynchronize(stream_));
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  };

  device_pos_mode_ = false;
  const double ng_ms = time_loop([&]() { gpu_forward(); });
  const double g_ms = time_loop([&]() { G4_CHECK(cudaGraphLaunch(exec, stream_)); });

  const double ng_per = ng_ms / iters, g_per = g_ms / iters;
  std::printf(
      "[graph-bench] iters=%d  non-graph %.3f ms/tok (%.1f tok/s)  graph %.3f ms/tok "
      "(%.1f tok/s)  speedup %.2fx\n",
      iters, ng_per, 1000.0 / ng_per, g_per, 1000.0 / g_per, ng_per / g_per);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  device_pos_mode_ = false;
}

void PlanCudaEngine::initialize(const EngineOptions& options) {
  // Keep the eos from the manifest (Gemma <eos>=1); the CLI's --eos default (2)
  // would otherwise clobber it. Turn-level stops (<end_of_turn>) come via stop
  // texts. Cap the context to what the caller requested (default 2048).
  samp_top_k_ = options.top_k;
  samp_top_p_ = options.top_p;
  samp_rep_penalty_ = options.repetition_penalty;
  samp_no_repeat_ngram_ = options.no_repeat_ngram_size;
  // Weight-only quantization (--weight-quant int8|int4). Must be set BEFORE open():
  // load_all quantizes each projection as it streams in, so the fp16 model never
  // lands on the GPU whole (Gemma 12B is ~24 GB fp16 and would OOM first).
  if (options.int8_streaming) {
    if (options.streaming_quant_bits == 4)
      weight_quant_bits_ = 4;
    else if (options.streaming_quant_bits == 8)
      weight_quant_bits_ = 8;
  }
  // Scale granularity. int4 defaults to group-wise (128 input features per scale):
  // per-row int4 gives one scale for a whole 3840-15360-wide row, so a single
  // outlier weight coarsens the entire row, and over 48 layers x 7 matmuls that
  // compounds into garbage (Gemma 12B produced "t/t/t/t/..."). Group-wise costs
  // 32 bits per group -- 4.25 bits/weight instead of 4. int8's 255 levels do not
  // need it, so it stays per-row and byte-identical to before.
  // Override with LLAMA_INFER_PLAN_QUANT_GROUP (0 = per-row, else a power of two).
  if (weight_quant_bits_ == 4) {
    weight_quant_group_ = 128;
  }
  if (const char* g = std::getenv("LLAMA_INFER_PLAN_QUANT_GROUP")) {
    weight_quant_group_ = std::atoi(g);
  }
  open(options.model_path, options.max_context > 0 ? options.max_context : 4096);
}

std::vector<int> PlanCudaEngine::generate_stream(const std::vector<int>& prompt, int max_new,
                                                 float temperature,
                                                 const std::function<bool(int)>& on_token,
                                                 const GenerationConstraints* constraints) {
  // Delegate the prefill+decode+sample+stop loop to the shared driver (which
  // reuses the canonical sampler); this engine only supplies step()/logits().
  // Incremental: positions advance monotonically so the KV-share sources stay
  // populated. (Grammar constraints aren't supported by this fork engine.)
  runtime::DecodeParams p;
  p.max_new_tokens = max_new;
  p.temperature = temperature;
  p.top_k = samp_top_k_;
  p.top_p = samp_top_p_;
  p.repetition_penalty = samp_rep_penalty_;
  p.no_repeat_ngram_size = samp_no_repeat_ngram_;
  // Honour a caller-supplied seed so temperature>0 generation is reproducible
  // (this engine previously ignored it, unlike the other engines).
  p.seed = constraints ? constraints->seed : -1;
  // Contract: generate_stream returns prompt+generated. main_modes strips
  // prompt_tokens.size(), and ONLY when the result is longer than the prompt — so
  // returning generated-only silently produced an EMPTY final "Decoded text".
  // It went unnoticed because the streamed on_token text still looked correct.
  p.include_prompt = !prompt.empty();
  // sample() pulls logits off the device lazily (greedy argmax / host copy), so
  // step() need not copy 262K floats to host every token.
  defer_host_logits_ = true;
  return runtime::run_decode(*this, prompt, p, on_token, &stats_);
}

std::vector<int> PlanCudaEngine::generate_multimodal(const std::vector<int>& tokens,
                                                     const std::vector<std::vector<float>>& embeds,
                                                     const std::vector<int>& limits, int max_new,
                                                     float temperature,
                                                     const std::function<bool(int)>& on_token) {
  defer_host_logits_ = false;  // the shared sampler reads host logits
  prefill_sequence(tokens, 0, embeds, limits);

  runtime::DecodeParams p;
  p.max_new_tokens = max_new;
  p.temperature = temperature;
  p.top_k = samp_top_k_;
  p.top_p = samp_top_p_;
  p.repetition_penalty = samp_rep_penalty_;
  p.seed = -1;

  std::vector<int> history = tokens;
  std::vector<int> out;
  int pos = static_cast<int>(tokens.size());
  for (int i = 0; i < max_new; ++i) {
    const int next = sample(p, history);  // logits() already holds the last token's
    if (next == cfg_.eos_token_id) break;
    out.push_back(next);
    history.push_back(next);
    if (on_token && !on_token(next)) break;
    step(next, pos++, /*want_logits=*/true);
  }
  return out;
}

std::vector<int> PlanCudaEngine::generate(const std::vector<int>& prompt, int max_new,
                                          float temperature) {
  return generate_stream(prompt, max_new, temperature, nullptr, nullptr);
}

std::vector<std::pair<int, float>> PlanCudaEngine::inspect_next_logits(
    const std::vector<int>& prompt, int top_k) {
  defer_host_logits_ = false;  // needs host logits for the top-k sort
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
