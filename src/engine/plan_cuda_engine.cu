// Generic per-layer op-plan CUDA executor: token-at-a-time forward driven by a
// data op-plan (include/engine/op_plan.hpp), not model-specific control flow.
// Gemma 4 is currently its sole tenant; see memory:cpi-gemma4-arch for that
// model's spec. Reuses the shared kernels (rmsnorm scale=w, rope table, tiled
// decode attention, gelu-mul, gemv).
#include <cublas_v2.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "engine/generation_constraints.hpp"
#include "engine/plan_cuda_engine.hpp"
#include "engine/deepseek_rope.hpp"
#include "engine/mla_absorb.hpp"
#include "engine/mla_assemble.hpp"
#include "engine/moe_grouped.hpp"
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
  for (auto* p : caches_lat_) cudaFree(p);
  for (auto* p : mla_w_ukt_) cudaFree(p);
  for (auto* p : mla_w_uv_) cudaFree(p);
  cudaFree(d_mla_qabs_);
  cudaFree(d_mla_attlat_);
  cudaFree(d_mla_abs_m_);
  cudaFree(d_mla_abs_l_);
  cudaFree(d_mla_abs_o_);
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
  cudaFree(d_argmax_pv_);
  cudaFree(d_argmax_pi_);
  cudaFree(d_mt_logits_);
  if (spec_graph_exec_) cudaGraphExecDestroy(spec_graph_exec_);
  if (spec_graph_) cudaGraphDestroy(spec_graph_);
  cudaFree(d_spec_split_m_);
  cudaFree(d_spec_split_l_);
  cudaFree(d_spec_split_o_);
  cudaFree(d_spec_argmax_);
  cudaFree(d_topk_part_val_);
  cudaFree(d_topk_part_idx_);
  cudaFree(d_topk_val_);
  cudaFree(d_topk_idx_);
  cudaFree(d_cand_idx_);
  cudaFree(d_cand_val_);
  cudaFree(d_cand_count_);
  cudaFree(d_persist_ops_);
  if (cublas_ != nullptr) cublasDestroy(static_cast<cublasHandle_t>(cublas_));
  cudaFree(d_cublas_ws_);
  cudaFree(d_pf_scores_);
  cudaFree(d_pf_ptrs_);
  cudaFree(d_seq_dequant_);
  cudaFree(d_moe_idx_seq_);
  cudaFree(d_moe_w_seq_);
  cudaFree(d_moe_gather_);
  cudaFree(d_moe_gate_g_);
  cudaFree(d_moe_up_g_);
  cudaFree(d_moe_outg_);
  cudaFree(d_moe_dqw_);
  cudaFree(d_gg_A_);
  cudaFree(d_gg_B_);
  cudaFree(d_gg_C_);
  cudaFree(d_moe_perm_);
  cudaFree(d_moe_rweight_);
  cudaFree(d_moe_out_f32_);
  cudaFree(d_seq_x8_);
  cudaFree(d_seq_xs_);
  cudaFree(d_seq_i32_);
  cudaFree(d_head_q_);
  cudaFree(d_head_qs_);
  cudaFree(d_split_m_);
  cudaFree(d_split_l_);
  cudaFree(d_split_o_);
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
//  safetensors : bf16, and the header carries no shape we use, so the caller
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
// Resolving here is what lets the shared Gemma loader call upload("...") with no
// dims and still work against either container.
std::pair<int, int> PlanCudaEngine::shape_of(const std::string& name) const {
  auto it = st_shapes_.find(name);
  if (it == st_shapes_.end()) throw std::runtime_error("no shape registered for: " + name);
  return it->second;
}

// Weight lookup that fails loudly. dev_ is a map, so dev_[name] on a missing key
// silently inserts nullptr; which the ops then read as "weightless norm" or a GEMV
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
  // group does not divide it, halve until it does rather than dropping to per-row:
  // per-row is the failure mode group-wise scales exist to fix, so falling back to
  // it would silently reintroduce it on exactly the tensors that need it. Gemma MoE
  // hits this; its down projections are 704 and 2112 wide, neither divisible by
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

  // Rowwise-int8 prefill copy (see QuantWeight): int8-rowwise configs alias their main
  // weights below; everything else pays one int8 copy so sequence GEMMs can run 8-bit.
  std::int8_t* d_pf = nullptr;
  float* d_pf_s = nullptr;
  static const bool int8_prefill = std::getenv("CPI_CUDA_INT8_PREFILL") != nullptr;
  if (int8_prefill && !(bits == 8 && group == 0)) {
    G4_CHECK(cudaMalloc(&d_pf, n));
    G4_CHECK(cudaMalloc(&d_pf_s, static_cast<std::size_t>(rows) * sizeof(float)));
    kernels::launch_quantize_rowwise_fp16_to_int8(d_fp16, d_pf, d_pf_s, rows, cols, stream_, 127);
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
  QuantWeight qw;
  qw.packed = d_w;
  qw.scales = d_scales;
  qw.group = group;
  qw.pf_i8 = (bits == 8 && group == 0) ? d_w : d_pf;  // alias when already rowwise int8
  qw.pf_scales = (bits == 8 && group == 0) ? d_scales : d_pf_s;
  if (moe_stream_ && bits == 4 && name.find("experts.") != std::string::npos) {
    // Expert streaming (step 2): keep the quantized experts on host and free the device copy (the
    // memory win). They stream K-at-a-time per layer (stage_moe_experts, now H2D). Device VRAM no
    // longer holds ~90% of the model; a >VRAM MoE becomes runnable. (Host store is pageable; for a
    // model too big for host RAM too, this would be mmap-backed; the K3 case.)
    const std::size_t wbytes = static_cast<std::size_t>(rows) * ((cols + 1) / 2);
    const std::size_t sbytes = static_cast<std::size_t>(rows) * n_groups * sizeof(float);
    void* hw = std::malloc(wbytes);
    void* hs = std::malloc(sbytes);
    if (!hw || !hs)
      throw std::runtime_error("host alloc failed for streamed experts (RAM too small): " + name);
    G4_CHECK(cudaMemcpy(hw, d_w, wbytes, cudaMemcpyDeviceToHost));
    G4_CHECK(cudaMemcpy(hs, d_scales, sbytes, cudaMemcpyDeviceToHost));
    cudaFree(d_w);
    cudaFree(d_scales);
    qw.packed = static_cast<std::int8_t*>(hw);
    qw.scales = static_cast<float*>(hs);
    qw.pf_i8 = nullptr;
    qw.pf_scales = nullptr;
    qw.host = true;
    moe_experts_on_host_ = true;
  }
  qdev_[name] = qw;
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
  // Tables sized to the actual per-layer-type head_dim (E2B full=512, 12B full=256).
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
  // full: partial rotary, first rope_angles pairs rotated, rest identity (inv=0)
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
    argmax_parts_ = kernels::argmax_partition_count(cfg_.vocab);
    G4_CHECK(cudaMalloc(&d_argmax_pv_, static_cast<std::size_t>(argmax_parts_) * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_argmax_pi_, static_cast<std::size_t>(argmax_parts_) * sizeof(int)));
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

  if (cfg_.family == Family::DeepSeekV2) {
    const int H = cfg_.hidden;
    const int nh = cfg_.num_heads;
    const int qkhd = cfg_.head_dim;                    // qk_nope+qk_rope (V padded to this)
    const int kvw = nh * qkhd;                         // K/V per-token width
    const int E = cfg_.num_experts, K = cfg_.top_k_experts, MI = cfg_.moe_intermediate_size;
    const int maxinter = std::max(cfg_.intermediate, MI * std::max(1, cfg_.n_shared_experts));
    auto al = [&](__half** p, std::size_t n) { G4_CHECK(cudaMalloc(p, n * sizeof(__half))); };
    al(&d_x_, H);
    al(&d_x_norm_, H);
    al(&d_tmp_, H);
    al(&d_q_, kvw);
    al(&d_k_, kvw);
    al(&d_v_, kvw);
    al(&d_att_, kvw);
    al(&d_gate_, maxinter);
    al(&d_up_, maxinter);
    al(&d_inter_, maxinter);
    al(&d_mla_ckv_, cfg_.kv_lora_rank + cfg_.qk_rope_head_dim);
    al(&d_mla_latent_, cfg_.kv_lora_rank);
    al(&d_mla_kvb_, static_cast<std::size_t>(nh) * (cfg_.qk_nope_head_dim + cfg_.v_head_dim));
    // MoE working set (experts always int4).
    al(&d_moe_router_in_, H);
    al(&d_moe_logits_, E);
    al(&d_moe_inter_, static_cast<std::size_t>(K) * MI);
    al(&d_moe_out_, H);
    G4_CHECK(cudaMalloc(&d_moe_idx_, static_cast<std::size_t>(K) * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_moe_w_, static_cast<std::size_t>(K) * sizeof(float)));
    // YARN inverse frequencies (filled in open()).
    G4_CHECK(cudaMalloc(&d_inv_freq_, static_cast<std::size_t>(cfg_.qk_rope_head_dim / 2) * sizeof(float)));
    // Sampling / logits scratch (same as the other families).
    G4_CHECK(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg_.vocab) * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_tok_, sizeof(int)));
    G4_CHECK(cudaMalloc(&d_position_, sizeof(int)));
    G4_CHECK(cudaMalloc(&d_argmax_, sizeof(int)));
    argmax_parts_ = kernels::argmax_partition_count(cfg_.vocab);
    G4_CHECK(cudaMalloc(&d_argmax_pv_, static_cast<std::size_t>(argmax_parts_) * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_argmax_pi_, static_cast<std::size_t>(argmax_parts_) * sizeof(int)));
    const int topk_parts = kernels::topk_partition_count(cfg_.vocab);
    const std::size_t part_n = static_cast<std::size_t>(topk_parts) * kMaxDeviceTopK;
    G4_CHECK(cudaMalloc(&d_topk_part_val_, part_n * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_topk_part_idx_, part_n * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_topk_val_, kMaxDeviceTopK * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_topk_idx_, kMaxDeviceTopK * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_cand_idx_, kCandCapacity * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_cand_val_, kCandCapacity * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_cand_count_, sizeof(int)));
    std::vector<__half> ones(std::max(qkhd, H), __float2half(1.0f));
    G4_CHECK(cudaMalloc(&d_ones_, ones.size() * sizeof(__half)));
    G4_CHECK(cudaMemcpy(d_ones_, ones.data(), ones.size() * sizeof(__half), cudaMemcpyHostToDevice));
    // Split-K decode-attention scratch (head_dim 192 > 128 -> the wide-head path).
    split_any_chunks_ = (max_ctx_ + kSplitAnyChunk - 1) / kSplitAnyChunk;
    const std::size_t split_cells = static_cast<std::size_t>(nh) * split_any_chunks_;
    G4_CHECK(cudaMalloc(&d_split_m_, split_cells * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_split_l_, split_cells * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_split_o_, split_cells * static_cast<std::size_t>(qkhd) * sizeof(float)));
    // Per-layer K/V caches (reconstructed K/V at width nh*qkhd).
    caches_k_.assign(cfg_.num_layers, nullptr);
    caches_v_.assign(cfg_.num_layers, nullptr);
    for (int L = 0; L < cfg_.num_layers; ++L) {
      const std::size_t bytes = static_cast<std::size_t>(max_ctx_) * kvw * sizeof(__half);
      G4_CHECK(cudaMalloc(&caches_k_[L], bytes));
      G4_CHECK(cudaMalloc(&caches_v_[L], bytes));
    }
    // Absorbed-MLA decode: latent caches + absorbed query/output scratch. The
    // materialized caches above stay (prefill uses them); decode attends over
    // the latent rows instead, cutting attention reads ~9x at depth.
    // mla_absorb_ was decided (and the absorbed weights uploaded) before this runs.
    if (mla_absorb_) {
      const int kva_w = cfg_.kv_lora_rank + cfg_.qk_rope_head_dim;
      caches_lat_.assign(cfg_.num_layers, nullptr);
      for (int L = 0; L < cfg_.num_layers; ++L) {
        const std::size_t bytes = static_cast<std::size_t>(max_ctx_) * kva_w * sizeof(__half);
        G4_CHECK(cudaMalloc(&caches_lat_[L], bytes));
      }
      al(&d_mla_qabs_, static_cast<std::size_t>(nh) * kva_w);
      al(&d_mla_attlat_, static_cast<std::size_t>(nh) * cfg_.kv_lora_rank);
      mla_abs_chunks_ = (max_ctx_ + 31) / 32;  // must match the launcher's kChunk
      const std::size_t cells = static_cast<std::size_t>(nh) * mla_abs_chunks_;
      G4_CHECK(cudaMalloc(&d_mla_abs_m_, cells * sizeof(float)));
      G4_CHECK(cudaMalloc(&d_mla_abs_l_, cells * sizeof(float)));
      G4_CHECK(cudaMalloc(&d_mla_abs_o_,
                          cells * static_cast<std::size_t>(cfg_.kv_lora_rank) * sizeof(float)));
    }
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
    moe_stream_ = std::getenv("CPI_MOE_STREAM") != nullptr;
    if (moe_stream_) {
      const std::size_t K = cfg_.top_k_experts;
      const int MI = cfg_.moe_intermediate_size;  // H is already in scope (cfg_.hidden)
      // The expert quant group is adaptive (see upload_int4): weight_quant_group_ halved until it
      // divides the row (gate_up over H=2816 keeps 128; down over MI=704 lands on 64). Size the
      // staging scales with the same effective group, or the per-expert scale copy overflows.
      auto eff_group = [&](int cols) {
        int g = weight_quant_group_;
        if (g > 0 && (g & (g - 1)) != 0) g = 0;
        while (g > 0 && (g > cols || cols % g != 0)) g /= 2;
        return g == 1 ? 0 : g;
      };
      const int ng_gu = kernels::quant_group_count(H, eff_group(H));
      const int ng_dn = kernels::quant_group_count(MI, eff_group(MI));
      const std::size_t gu_rows = K * 2 * static_cast<std::size_t>(MI);
      const std::size_t dn_rows = K * static_cast<std::size_t>(H);
      G4_CHECK(cudaMalloc(&d_moe_stage_gu_, gu_rows * ((H + 1) / 2)));
      G4_CHECK(cudaMalloc(&d_moe_stage_gu_s_, gu_rows * ng_gu * sizeof(float)));
      G4_CHECK(cudaMalloc(&d_moe_stage_dn_, dn_rows * ((MI + 1) / 2)));
      G4_CHECK(cudaMalloc(&d_moe_stage_dn_s_, dn_rows * ng_dn * sizeof(float)));
      G4_CHECK(cudaMalloc(&d_moe_identity_idx_, K * sizeof(int)));
      std::vector<int> ident(K);
      for (std::size_t k = 0; k < K; ++k) ident[k] = static_cast<int>(k);
      G4_CHECK(cudaMemcpy(d_moe_identity_idx_, ident.data(), K * sizeof(int),
                          cudaMemcpyHostToDevice));
      std::fprintf(stderr, "[moe] expert streaming ON (staging %zu experts/layer)\n", K);
    }
  }
  al(&d_ple_raw_, ple_tot);
  al(&d_ple_, ple_tot);
  al(&d_ple_gate_, cfg_.hidden_size_per_layer_input);
  G4_CHECK(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg_.vocab) * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_tok_, sizeof(int)));
  G4_CHECK(cudaMalloc(&d_position_, sizeof(int)));
  G4_CHECK(cudaMalloc(&d_argmax_, sizeof(int)));
  argmax_parts_ = kernels::argmax_partition_count(cfg_.vocab);
  G4_CHECK(cudaMalloc(&d_argmax_pv_, static_cast<std::size_t>(argmax_parts_) * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_argmax_pi_, static_cast<std::size_t>(argmax_parts_) * sizeof(int)));
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
  // d_ones_ backs every weightless norm, so it must be as long as the widest vector any
  // of them normalises; not just the widest head. The vision projector normalises over
  // the vision hidden size, which is far wider than a head: sizing this to maxhd read off
  // the end of the buffer (silently, with garbage "ones").
  std::vector<__half> ones(std::max(maxhd, cfg_.hidden), __float2half(1.0f));
  G4_CHECK(cudaMalloc(&d_ones_, ones.size() * sizeof(__half)));
  G4_CHECK(cudaMemcpy(d_ones_, ones.data(), ones.size() * sizeof(__half), cudaMemcpyHostToDevice));

  // Split-K decode-attention scratch for the wide-head path (head_dim > 128, window or not).
  // Without it every decode attention runs one block per head (8 blocks), and
  // the whole windowed KV walk serialises inside them. ~2 MB at Gemma E2B geometry.
  split_any_chunks_ = (max_ctx_ + kSplitAnyChunk - 1) / kSplitAnyChunk;
  const std::size_t split_cells =
      static_cast<std::size_t>(cfg_.num_heads) * static_cast<std::size_t>(split_any_chunks_);
  G4_CHECK(cudaMalloc(&d_split_m_, split_cells * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_split_l_, split_cells * sizeof(float)));
  G4_CHECK(cudaMalloc(&d_split_o_, split_cells * static_cast<std::size_t>(maxhd) * sizeof(float)));

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
// graph, KV/state) is the same shared machinery Gemma runs on.

// Gemma 4 from a HuggingFace safetensors directory. The .cpi path gets its config
// from the container manifest; this reads the same facts out of config.json and,
// crucially, registers every tensor's shape (the safetensors header carries none we
// use). With Config + st_shapes_ + wprefix_ filled in, the shared Gemma loader and
// plan builder run unchanged; the MoE checkpoint is not a second Gemma port.
void PlanCudaEngine::parse_gemma4_st_config(const std::string& model_dir) {
  const std::string raw = mini::read_text_file(std::filesystem::path(model_dir) / "config.json");

  // The parse lives in engine/plan_model_config.cpp, with no CUDA in it, so PlanMetalEngine can
  // reach the same geometry. What stays here is the shape registry below, which is specific to how
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
    // The KV-shared layers run a double-wide MLP. This override must come after the
    // puts above; registering it first just gets overwritten, which silently reads
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

void PlanCudaEngine::parse_deepseek_config(const std::string& model_dir) {
  // DeepSeek-V2-Lite config.json is flat (no text_config nesting).
  const std::string raw = mini::read_text_file(std::filesystem::path(model_dir) / "config.json");
  cfg_.family = Family::DeepSeekV2;
  cfg_.vocab = mini::json_get_int(raw, "vocab_size", 0);
  cfg_.hidden = mini::json_get_int(raw, "hidden_size", 0);
  cfg_.intermediate = mini::json_get_int(raw, "intermediate_size", 0);  // the dense-layer MLP width
  cfg_.num_layers = mini::json_get_int(raw, "num_hidden_layers", 0);
  cfg_.num_heads = mini::json_get_int(raw, "num_attention_heads", 0);
  cfg_.rms_eps = mini::json_get_float(raw, "rms_norm_eps", 1e-6f);
  cfg_.max_position_embeddings = mini::json_get_int(raw, "max_position_embeddings", 4096);
  cfg_.eos_token_id = mini::json_get_int(raw, "eos_token_id", -1);
  cfg_.bos_token_id = mini::json_get_int(raw, "bos_token_id", 0);
  cfg_.tie_word_embeddings = mini::json_get_bool(raw, "tie_word_embeddings", false);

  // MLA geometry. q_lora_rank is `null` for V2-Lite (json_get_int -> 0 = direct q_proj).
  cfg_.q_lora_rank = mini::json_get_int(raw, "q_lora_rank", 0);
  cfg_.kv_lora_rank = mini::json_get_int(raw, "kv_lora_rank", 0);
  cfg_.qk_nope_head_dim = mini::json_get_int(raw, "qk_nope_head_dim", 0);
  cfg_.qk_rope_head_dim = mini::json_get_int(raw, "qk_rope_head_dim", 0);
  cfg_.v_head_dim = mini::json_get_int(raw, "v_head_dim", 0);
  // The shared Attention op + KV cache use one head_dim; we pad V up to the qk head dim (see the
  // o_proj padding in the loader), so every per-layer head_dim is qk_nope+qk_rope.
  const int qk_head_dim = cfg_.qk_nope_head_dim + cfg_.qk_rope_head_dim;
  cfg_.head_dim = qk_head_dim;
  cfg_.head_dim_full = cfg_.head_dim_sliding = qk_head_dim;
  cfg_.num_kv_heads = cfg_.num_kv_heads_full = cfg_.num_kv_heads_sliding = cfg_.num_heads;

  // MoE.
  cfg_.num_experts = mini::json_get_int(raw, "n_routed_experts", 0);
  cfg_.top_k_experts = mini::json_get_int(raw, "num_experts_per_tok", 0);
  cfg_.moe_intermediate_size = mini::json_get_int(raw, "moe_intermediate_size", 0);
  cfg_.n_shared_experts = mini::json_get_int(raw, "n_shared_experts", 0);
  cfg_.first_k_dense_replace = mini::json_get_int(raw, "first_k_dense_replace", 0);
  cfg_.routed_scaling_factor = mini::json_get_float(raw, "routed_scaling_factor", 1.0f);
  cfg_.enable_moe_block = cfg_.num_experts > 0;

  // YARN rope (used at load to build inv_freq + the mscale attention scaling).
  cfg_.rope_theta = mini::json_get_float(raw, "rope_theta", 10000.0f);
  const std::string rope = mini::json_extract_object(raw, "rope_scaling");
  if (!rope.empty()) {
    cfg_.yarn_factor = mini::json_get_float(rope, "factor", 1.0f);
    cfg_.yarn_mscale = mini::json_get_float(rope, "mscale", 1.0f);
    cfg_.yarn_beta_fast = mini::json_get_float(rope, "beta_fast", 32.0f);
    cfg_.yarn_beta_slow = mini::json_get_float(rope, "beta_slow", 1.0f);
    cfg_.yarn_original_max_pos =
        mini::json_get_int(rope, "original_max_position_embeddings", 4096);
  }

  // All layers use MLA (full-attention kind); layers >= first_k_dense_replace are MoE.
  cfg_.layer_full.assign(cfg_.num_layers, 1);

  if (cfg_.vocab <= 0 || cfg_.hidden <= 0 || cfg_.num_layers <= 0 || cfg_.num_heads <= 0 ||
      cfg_.kv_lora_rank <= 0 || cfg_.qk_nope_head_dim <= 0 || cfg_.qk_rope_head_dim <= 0 ||
      cfg_.v_head_dim <= 0)
    throw std::runtime_error("DeepSeek-V2 config.json is missing required MLA/geometry fields");
}

// --- DeepSeek-V2 loader/plan (Phase 1c loader landed; MLA op + plan follow). ---

// Upload a host-built fp16 [rows,cols] weight under `name` (e.g. the zero-padded o_proj that has no
// counterpart tensor to read from the checkpoint).
__half* PlanCudaEngine::upload_host_fp16(const std::string& name, const std::vector<__half>& host,
                                         int rows, int cols) {
  (void)rows;
  (void)cols;
  __half* d = nullptr;
  G4_CHECK(cudaMalloc(&d, host.size() * sizeof(__half)));
  G4_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice));
  dev_[name] = d;
  return d;
}

// Quantize a host-built fp16 [rows,cols] weight to int4 (group-wise) and register it under `name`.
// Same quant path as upload_int4 but from a caller buffer; DeepSeek's fused expert matrices are
// synthesized in host memory (no such tensor exists to read by name). Experts are the memory hog, so
// they are always int4 regardless of the global weight-quant flag (mirrors the Gemma MoE requirement).
void PlanCudaEngine::fuse_upload_experts_int4(const std::string& name,
                                              const std::vector<__half>& host, int rows, int cols) {
  // CPI_DS_EXPERT_BITS=8 keeps experts int8 (diagnostic: distinguishes int4 quant drift from a bug).
  static const int ebits = std::getenv("CPI_DS_EXPERT_BITS")
                               ? std::atoi(std::getenv("CPI_DS_EXPERT_BITS"))
                               : 4;
  int group = weight_quant_group_;
  if (group > 0 && (group & (group - 1)) != 0) group = 0;
  while (group > 0 && (group > cols || (cols % group) != 0)) group /= 2;
  if (group == 1) group = 0;

  __half* d_fp16 = nullptr;
  G4_CHECK(cudaMalloc(&d_fp16, host.size() * sizeof(__half)));
  G4_CHECK(cudaMemcpy(d_fp16, host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice));
  std::int8_t* d_i8 = nullptr;
  float* d_scales = nullptr;
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  const int max_q = ebits == 8 ? 127 : 7;
  const int n_groups = kernels::quant_group_count(cols, group);
  G4_CHECK(cudaMalloc(&d_i8, n));
  G4_CHECK(cudaMalloc(&d_scales, static_cast<std::size_t>(rows) * n_groups * sizeof(float)));
  if (group > 0)
    kernels::launch_quantize_groupwise_fp16_to_int8(d_fp16, d_i8, d_scales, rows, cols, group,
                                                    stream_, max_q);
  else
    kernels::launch_quantize_rowwise_fp16_to_int8(d_fp16, d_i8, d_scales, rows, cols, stream_, max_q);
  std::int8_t* d_w = d_i8;
  if (ebits == 4) {
    std::int8_t* d_packed = nullptr;
    G4_CHECK(cudaMalloc(&d_packed, static_cast<std::size_t>(rows) * ((cols + 1) / 2)));
    kernels::launch_pack_rowwise_int8_to_int4(d_i8, d_packed, rows, cols, stream_);
    G4_CHECK(cudaStreamSynchronize(stream_));
    cudaFree(d_i8);
    d_w = d_packed;
  } else {
    G4_CHECK(cudaStreamSynchronize(stream_));  // int8: keep d_i8 unpacked
  }
  cudaFree(d_fp16);

  QuantWeight qw;
  qw.packed = d_w;
  qw.scales = d_scales;
  qw.group = group;
  qw.pf_i8 = nullptr;
  qw.pf_scales = nullptr;
  qdev_[name] = qw;
}

void PlanCudaEngine::load_deepseek_weights() {
  const bool quant = weight_quant_bits_ == 4 || weight_quant_bits_ == 8;
  const int H = cfg_.hidden;
  const int nh = cfg_.num_heads;
  const int qk = cfg_.qk_nope_head_dim + cfg_.qk_rope_head_dim;  // 192, the padded head_dim
  const int vhd = cfg_.v_head_dim;                               // 128
  const int kva = cfg_.kv_lora_rank + cfg_.qk_rope_head_dim;     // 576
  const int kvb = nh * (cfg_.qk_nope_head_dim + vhd);            // nh*256
  const int MI = cfg_.moe_intermediate_size;                     // 1408
  // MLA/dense/shared projections honor --weight-quant int4 (like llama.cpp Q4) to cut decode bandwidth
  //; the shared expert alone is ~2x the routed experts in fp16 bytes. o_proj stays fp16 (host-built
  // padded tensor). Experts are always int4.
  auto proj = [&](const std::string& nm, int r, int c) {
    if (quant)
      upload_int4(nm, r, c);
    else
      upload(nm, r, c);
  };

  upload("model.embed_tokens.weight", cfg_.vocab, H);
  upload("model.norm.weight", 1, H);
  if (st_.has_tensor("lm_head.weight")) upload("lm_head.weight", cfg_.vocab, H);
  if (mla_absorb_) {
    mla_w_ukt_.assign(cfg_.num_layers, nullptr);
    mla_w_uv_.assign(cfg_.num_layers, nullptr);
  }

  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = "model.layers." + std::to_string(L) + ".";
    upload(p + "input_layernorm.weight", 1, H);
    upload(p + "post_attention_layernorm.weight", 1, H);

    // ── MLA attention weights ──
    if (cfg_.q_lora_rank > 0) {
      proj(p + "self_attn.q_a_proj.weight", cfg_.q_lora_rank, H);
      upload(p + "self_attn.q_a_layernorm.weight", 1, cfg_.q_lora_rank);
      proj(p + "self_attn.q_b_proj.weight", nh * qk, cfg_.q_lora_rank);
    } else {
      proj(p + "self_attn.q_proj.weight", nh * qk, H);
    }
    proj(p + "self_attn.kv_a_proj_with_mqa.weight", kva, H);
    upload(p + "self_attn.kv_a_layernorm.weight", 1, cfg_.kv_lora_rank);
    proj(p + "self_attn.kv_b_proj.weight", kvb, cfg_.kv_lora_rank);
    if (mla_absorb_) {
      // Absorbed-layout fp16 copies of kv_b's two halves: w_ukt [nh][kv_lora][qk_nope]
      // (K-nope rows transposed, output-major) and w_uv [nh][v_head][kv_lora].
      const int lora = cfg_.kv_lora_rank;
      const int nope = cfg_.qk_nope_head_dim;
      const int kvh = nope + vhd;
      const auto kb = read_host_fp16(p + "self_attn.kv_b_proj.weight", kvb, lora);
      std::vector<__half> ukt(static_cast<std::size_t>(nh) * lora * nope);
      std::vector<__half> uv(static_cast<std::size_t>(nh) * vhd * lora);
      for (int i = 0; i < nh; ++i) {
        for (int d = 0; d < nope; ++d) {
          const std::size_t src = (static_cast<std::size_t>(i) * kvh + d) * lora;
          for (int j = 0; j < lora; ++j) {
            ukt[(static_cast<std::size_t>(i) * lora + j) * nope + d] = kb[src + j];
          }
        }
        for (int d = 0; d < vhd; ++d) {
          const std::size_t src = (static_cast<std::size_t>(i) * kvh + nope + d) * lora;
          for (int j = 0; j < lora; ++j) {
            uv[(static_cast<std::size_t>(i) * vhd + d) * lora + j] = kb[src + j];
          }
        }
      }
      G4_CHECK(cudaMalloc(&mla_w_ukt_[L], ukt.size() * sizeof(__half)));
      G4_CHECK(cudaMemcpy(mla_w_ukt_[L], ukt.data(), ukt.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
      G4_CHECK(cudaMalloc(&mla_w_uv_[L], uv.size() * sizeof(__half)));
      G4_CHECK(cudaMemcpy(mla_w_uv_[L], uv.data(), uv.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    }
    // o_proj is [H, nh*v_head]; pad to [H, nh*qk] with zero columns per head so the standard
    // Attention op can run at head_dim=qk (V padded to qk) and o_proj still consumes it directly.
    {
      const auto oh = read_host_fp16(p + "self_attn.o_proj.weight", H, nh * vhd);
      std::vector<__half> pad(static_cast<std::size_t>(H) * nh * qk, __float2half(0.0f));
      for (int r = 0; r < H; ++r)
        for (int h = 0; h < nh; ++h)
          for (int d = 0; d < vhd; ++d)
            pad[(static_cast<std::size_t>(r) * nh + h) * qk + d] =
                oh[(static_cast<std::size_t>(r) * nh + h) * vhd + d];
      // int4 under --weight-quant (biggest attention read); else fp16. The padded zero columns
      // quantize to 0, so o_proj still consumes the padded Att slot directly either way.
      if (quant)
        fuse_upload_experts_int4(p + "self_attn.o_proj.weight", pad, H, nh * qk);
      else
        upload_host_fp16(p + "self_attn.o_proj.weight", pad, H, nh * qk);
    }

    // ── MoE (layers >= first_k_dense_replace) or a dense MLP (the first layers) ──
    const bool moe = L >= cfg_.first_k_dense_replace && cfg_.num_experts > 0;
    if (moe) {
      upload(p + "mlp.gate.weight", cfg_.num_experts, H);  // router logits proj (fp16)
      // Fuse the per-expert gate/up/down into the [E*2*MI,H] / [E*H,MI] layout the MoE ops expect.
      std::vector<__half> gate_up(static_cast<std::size_t>(cfg_.num_experts) * 2 * MI * H);
      std::vector<__half> down(static_cast<std::size_t>(cfg_.num_experts) * H * MI);
      for (int e = 0; e < cfg_.num_experts; ++e) {
        const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        const auto g = read_host_fp16(ep + "gate_proj.weight", MI, H);
        const auto u = read_host_fp16(ep + "up_proj.weight", MI, H);
        const auto dn = read_host_fp16(ep + "down_proj.weight", H, MI);
        std::copy(g.begin(), g.end(), gate_up.begin() + static_cast<std::size_t>(e) * 2 * MI * H);
        std::copy(u.begin(), u.end(),
                  gate_up.begin() + (static_cast<std::size_t>(e) * 2 * MI + MI) * H);
        std::copy(dn.begin(), dn.end(), down.begin() + static_cast<std::size_t>(e) * H * MI);
      }
      fuse_upload_experts_int4(p + "mlp.experts.gate_up_proj", gate_up, cfg_.num_experts * 2 * MI, H);
      fuse_upload_experts_int4(p + "mlp.experts.down_proj", down, cfg_.num_experts * H, MI);
      const int SI = MI * cfg_.n_shared_experts;  // shared-expert SwiGLU intermediate
      proj(p + "mlp.shared_experts.gate_proj.weight", SI, H);
      proj(p + "mlp.shared_experts.up_proj.weight", SI, H);
      proj(p + "mlp.shared_experts.down_proj.weight", H, SI);
    } else {
      proj(p + "mlp.gate_proj.weight", cfg_.intermediate, H);
      proj(p + "mlp.up_proj.weight", cfg_.intermediate, H);
      proj(p + "mlp.down_proj.weight", H, cfg_.intermediate);
    }
  }
}

// DeepSeek MoE FFN (router softmax-top-k without renorm + fused int4 experts + a shared SwiGLU),
// all reading Slot::XNorm (= post_attention_layernorm output), accumulating into Slot::X.
void PlanCudaEngine::append_deepseek_moe_ffn_ops(std::vector<opplan::Op>& ops, const std::string& p) {
  using namespace opplan;
  const int H = cfg_.hidden, E = cfg_.num_experts, K = cfg_.top_k_experts;
  const int MI = cfg_.moe_intermediate_size, SI = MI * std::max(1, cfg_.n_shared_experts);
  auto Wd = [&](const char* t) { return static_cast<const __half*>(dev_[p + t]); };
  static const int ebits = std::getenv("CPI_DS_EXPERT_BITS")
                               ? std::atoi(std::getenv("CPI_DS_EXPERT_BITS"))
                               : 4;
  auto bind_expert = [&](Op& o, const char* t) {
    const auto q = qdev_.find(p + t);  // fused experts are int4 (or int8 under CPI_DS_EXPERT_BITS=8)
    o.qweight = q->second.packed;
    o.qscales = q->second.scales;
    o.qbits = ebits;
    o.qgroup = q->second.group;
  };
  auto gemv = [&](Slot in, Slot out, const char* t, int out_dim, int in_dim) {
    Op o;
    o.kind = OpKind::Gemv;
    o.in = in;
    o.out = out;
    o.cols = out_dim;
    o.in_dim = in_dim;
    const auto q = qdev_.find(p + t);  // shared-expert projections are int4 under --weight-quant
    if (q != qdev_.end()) {
      o.qweight = q->second.packed;
      o.qscales = q->second.scales;
      o.qbits = weight_quant_bits_;
      o.qgroup = q->second.group;
    } else {
      o.weight = Wd(t);
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

  // router: logits = gate . XNorm ; softmax -> top-k, no renorm (norm_topk_prob=false), scale 1.0.
  gemv(Slot::XNorm, Slot::MoeLogits, "mlp.gate.weight", E, H);
  {
    Op o;
    o.kind = OpKind::MoeRouterTopk;
    o.in = Slot::MoeLogits;
    o.weight = nullptr;
    o.cols = E;
    o.heads = K;
    o.moe_renorm = false;
    ops.push_back(o);
  }
  // routed experts (SiLU SwiGLU) on XNorm -> MoeOut ; X += MoeOut.
  {
    Op o;
    o.kind = OpKind::MoeGateUpGeglu;
    o.in = Slot::XNorm;
    o.out = Slot::MoeInter;
    o.cols = MI;
    o.in_dim = H;
    o.heads = K;
    o.mlp_gelu = false;
    bind_expert(o, "mlp.experts.gate_up_proj");
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
    bind_expert(o, "mlp.experts.down_proj");
    ops.push_back(o);
  }
  add_x(Slot::MoeOut);
  // shared expert (SwiGLU) on XNorm -> Tmp ; X += Tmp.
  gemv(Slot::XNorm, Slot::Gate, "mlp.shared_experts.gate_proj.weight", SI, H);
  gemv(Slot::XNorm, Slot::Up, "mlp.shared_experts.up_proj.weight", SI, H);
  {
    Op o;
    o.kind = OpKind::SiluMul;
    o.in = Slot::Gate;
    o.in2 = Slot::Up;
    o.out = Slot::Inter;
    o.cols = SI;
    ops.push_back(o);
  }
  gemv(Slot::Inter, Slot::Tmp, "mlp.shared_experts.down_proj.weight", H, SI);
  add_x(Slot::Tmp);
}

void PlanCudaEngine::build_deepseek_plan() {
  using namespace opplan;
  // Wire the decode slot pointers (build_plan does this for the Gemma path; DeepSeek needs its own).
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
  slot_ptr_[static_cast<int>(Slot::MoeRouterIn)] = d_moe_router_in_;
  slot_ptr_[static_cast<int>(Slot::MoeLogits)] = d_moe_logits_;
  slot_ptr_[static_cast<int>(Slot::MoeInter)] = d_moe_inter_;
  slot_ptr_[static_cast<int>(Slot::MoeOut)] = d_moe_out_;
  slot_ptr_[static_cast<int>(Slot::MlaCkv)] = d_mla_ckv_;
  slot_ptr_[static_cast<int>(Slot::MlaLatent)] = d_mla_latent_;
  slot_ptr_[static_cast<int>(Slot::MlaKvb)] = d_mla_kvb_;
  slot_ptr_[static_cast<int>(Slot::MlaQAbs)] = d_mla_qabs_;
  slot_ptr_[static_cast<int>(Slot::MlaAttLat)] = d_mla_attlat_;

  const int H = cfg_.hidden, nh = cfg_.num_heads;
  const int qkhd = cfg_.head_dim, kvw = nh * qkhd;
  const int kva = cfg_.kv_lora_rank + cfg_.qk_rope_head_dim;
  const int kvb = nh * (cfg_.qk_nope_head_dim + cfg_.v_head_dim);
  if (cfg_.q_lora_rank > 0)
    throw std::runtime_error("DeepSeek q_lora_rank>0 (q-LoRA) not wired yet; V2-Lite is direct q_proj");

  {
    Op e;
    e.kind = OpKind::EmbeddingLookup;
    e.out = Slot::X;
    e.cols = H;
    e.weight = static_cast<const __half*>(dev_["model.embed_tokens.weight"]);
    plan_.prologue.push_back(e);
  }

  plan_.layers.assign(cfg_.num_layers, LayerPlan{});
  for (int L = 0; L < cfg_.num_layers; ++L) {
    const std::string p = "model.layers." + std::to_string(L) + ".";
    LayerPlan& lp = plan_.layers[L];
    lp.layer_index = L;
    auto& ops = lp.ops;
    auto W = [&](const char* t) { return static_cast<const __half*>(dev_[p + t]); };
    auto rms = [&](Slot in, Slot out, const char* t, int rows, int cols) {
      Op o;
      o.kind = OpKind::RmsNorm;  // plain DeepSeek RMSNorm (no 1+w offset)
      o.in = in;
      o.out = out;
      o.weight = W(t);
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
    // ── MLA attention ──
    gemv(Slot::XNorm, Slot::Q, "self_attn.q_proj.weight", nh * qkhd, H);
    gemv(Slot::XNorm, Slot::MlaCkv, "self_attn.kv_a_proj_with_mqa.weight", kva, H);
    rms(Slot::MlaCkv, Slot::MlaLatent, "self_attn.kv_a_layernorm.weight", 1, cfg_.kv_lora_rank);
    // Latent-only decode: with absorbed attention, decode never reads the
    // materialized per-head cache (prefill always restarts from position 0, so
    // no later prefill reads decode-written rows either). kv_b, assemble, and
    // the materialized store then only need to run in seq mode; a fused
    // rope-q + store-latent kernel replaces them per decode token. Spec decode
    // is the one seq path that attends over decode-written materialized rows,
    // so it keeps the dual-write decode.
    const bool latent_only = mla_absorb_ && std::getenv("CPI_CUDA_SPEC") == nullptr &&
                             std::getenv("CPI_DS_NO_LATENT_ONLY") == nullptr;
    gemv(Slot::MlaLatent, Slot::MlaKvb, "self_attn.kv_b_proj.weight", kvb, cfg_.kv_lora_rank);
    ops.back().decode_skip = latent_only;
    {
      Op o;
      o.kind = OpKind::MlaAssembleRope;
      o.heads = nh;
      o.head_dim = qkhd;
      o.key_head_dim = cfg_.qk_nope_head_dim;
      o.rotary_dim = cfg_.qk_rope_head_dim;
      o.value_head_dim = cfg_.v_head_dim;
      o.in_dim = cfg_.kv_lora_rank;
      o.scale = mla_attn_scaling_;
      o.aux_ptr = d_inv_freq_;
      o.decode_skip = latent_only;
      ops.push_back(o);
    }
    {
      Op o;
      o.kind = OpKind::KvStore;
      o.cols = kvw;
      o.decode_skip = latent_only;
      ops.push_back(o);
    }
    if (mla_absorb_) {
      Op o;
      o.kind = OpKind::MlaLatStore;
      o.heads = nh;
      o.head_dim = qkhd;
      o.key_head_dim = cfg_.qk_nope_head_dim;
      o.rotary_dim = cfg_.qk_rope_head_dim;
      o.in_dim = cfg_.kv_lora_rank;
      if (latent_only) {
        // Arms the fused decode path: rope q_pe/k_pe here (assemble is skipped)
        // and write the latent row straight from the kv_a outputs.
        o.aux_ptr = d_inv_freq_;
        o.scale = mla_attn_scaling_;
      }
      ops.push_back(o);
    }
    {
      Op o;
      o.kind = OpKind::Attention;
      o.in = Slot::Q;
      o.out = Slot::Att;
      o.heads = nh;
      o.kv_heads = nh;
      o.head_dim = qkhd;
      o.full_attention = true;
      o.sliding_window = 0;
      // Absorbed decode replaces this op at T==1; prefill still runs it.
      o.decode_skip = mla_absorb_;
      ops.push_back(o);
    }
    if (mla_absorb_) {
      Op o;
      o.kind = OpKind::MlaQAbsorb;
      o.heads = nh;
      o.head_dim = qkhd;
      o.key_head_dim = cfg_.qk_nope_head_dim;
      o.rotary_dim = cfg_.qk_rope_head_dim;
      o.in_dim = cfg_.kv_lora_rank;
      o.weight = mla_w_ukt_[L];
      ops.push_back(o);
      Op a;
      a.kind = OpKind::MlaAbsorbedAttn;
      a.heads = nh;
      a.head_dim = qkhd;
      a.in_dim = cfg_.kv_lora_rank;
      a.rotary_dim = cfg_.qk_rope_head_dim;
      a.value_head_dim = cfg_.v_head_dim;
      // Fused reduce+decompress writes Slot::Att directly via w_uv.
      a.weight = mla_w_uv_[L];
      // The materialized head's softmax scale; mscale is already in the roped parts.
      a.scale = 1.0f / std::sqrt(static_cast<float>(qkhd));
      ops.push_back(a);
    }
    gemv(Slot::Att, Slot::Tmp, "self_attn.o_proj.weight", H, nh * qkhd);
    add_x(Slot::Tmp);

    // ── MLP: dense (first_k_dense_replace layers) or MoE ──
    rms(Slot::X, Slot::XNorm, "post_attention_layernorm.weight", 1, H);
    if (L >= cfg_.first_k_dense_replace && cfg_.num_experts > 0) {
      append_deepseek_moe_ffn_ops(ops, p);
    } else {
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
  }

  {
    Op n;
    n.kind = OpKind::RmsNorm;
    n.in = Slot::X;
    n.out = Slot::XNorm;
    n.weight = static_cast<const __half*>(dev_["model.norm.weight"]);
    n.rows = 1;
    n.cols = H;
    plan_.epilogue.push_back(n);
    Op h;
    h.kind = OpKind::LmHead;
    h.in = Slot::XNorm;
    const char* head =
        dev_.count("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight";
    h.weight = static_cast<const __half*>(dev_[head]);
    h.cols = cfg_.vocab;
    h.in_dim = H;
    plan_.epilogue.push_back(h);
  }
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

    if (cfg_.layer_full[L]) {  // full attention (q_proj is a fused q|gate)
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
    // The PLE projections are ordinary 2-D matmuls and ride the same quantiser (and the
    // dp4a decode path) as attn/mlp; they were the last fp16 gemvs in the quant token
    // (~180 us/token): model_projection is [layers*ple x H], each layer adds a gate and a
    // down-projection below. Norms stay fp16 as everywhere else.
    if (weight_quant_bits_ == 4 || weight_quant_bits_ == 8)
      upload_int4(W + "per_layer_model_projection.weight");
    else
      upload(W + "per_layer_model_projection.weight");
    upload(W + "per_layer_projection_norm.weight");
  }
  layer_scalar_host_.assign(cfg_.num_layers, 1.0f);
  // The dense projections carry ~all the weight; norms/embeddings stay fp16.
  const bool quantize = weight_quant_bits_ == 4 || weight_quant_bits_ == 8;
  // The tied LM head re-reads the whole fp16 embedding table every token (~0.5 ms on E2B's
  // 262K x 1536). Quant runs give the head a rowwise-int8 packed copy; int8 even under
  // int4, since per-row int8 needs no groups; while EmbeddingLookup keeps the fp16 table.
  if (quantize && d_head_q_ == nullptr) {
    const __half* emb = static_cast<const __half*>(dev_at(W + "embed_tokens.weight"));
    const std::size_t n =
        static_cast<std::size_t>(cfg_.vocab) * static_cast<std::size_t>(cfg_.hidden);
    if (weight_quant_bits_ == 4 && (cfg_.hidden % 32) == 0) {
      // int4 runs: grouped-int4 head (~5 bpw with scales) on the dp4a f32 path; half
      // the bytes of the int8 head, which was already at roofline. Packed via the same
      // int8-then-pack pipeline as every projection.
      const int group = weight_quant_group_ > 0 ? weight_quant_group_ : 128;
      const int n_groups = kernels::quant_group_count(cfg_.hidden, group);
      std::int8_t* tmp_i8 = nullptr;
      G4_CHECK(cudaMalloc(&tmp_i8, n));
      G4_CHECK(cudaMalloc(&d_head_q_, n / 2));
      G4_CHECK(
          cudaMalloc(&d_head_qs_, static_cast<std::size_t>(cfg_.vocab) * n_groups * sizeof(float)));
      kernels::launch_quantize_groupwise_fp16_to_int8(emb, tmp_i8, d_head_qs_, cfg_.vocab,
                                                      cfg_.hidden, group, stream_, 7);
      kernels::launch_pack_rowwise_int8_to_int4(tmp_i8, d_head_q_, cfg_.vocab, cfg_.hidden,
                                                stream_);
      G4_CHECK(cudaStreamSynchronize(stream_));
      cudaFree(tmp_i8);
      head_qbits_ = 4;
      head_qgroup_ = group;
    } else {
      G4_CHECK(cudaMalloc(&d_head_q_, n));
      G4_CHECK(cudaMalloc(&d_head_qs_, static_cast<std::size_t>(cfg_.vocab) * sizeof(float)));
      kernels::launch_quantize_rowwise_fp16_to_int8(emb, d_head_q_, d_head_qs_, cfg_.vocab,
                                                    cfg_.hidden, stream_);
      head_qbits_ = 8;
    }
  }
  // CPI_PLAN_INT4_GROUP=attn|mlp restricts quantization to one group
  // (bisection aid when a model degrades under int4).
  const char* group_env = std::getenv("CPI_PLAN_INT4_GROUP");
  const std::string group = group_env ? group_env : "";
  const auto quantizable = [&group](const char* t) {
    const bool is_attn =
        std::strncmp(t, "self_attn.", 10) == 0 && std::strstr(t, "_proj.") != nullptr;
    const bool is_mlp = std::strncmp(t, "mlp.", 4) == 0 && std::strstr(t, "_proj.") != nullptr;
    const bool is_ple = std::strncmp(t, "per_layer_", 10) == 0;  // gate + projection, not norms
    if (group == "attn") return is_attn;
    if (group == "mlp") return is_mlp;
    return is_attn || is_mlp || is_ple;
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
    {
      // Fold the sqrt(head_dim) attention pre-scale into the q_norm weight (scaling
      // commutes with rotation): the per-layer ScaleCopy op it replaces cost a
      // kernel-latency floor every token.
      const int hd_l = head_dim_of(L);
      __half* wq = const_cast<__half*>(dev_at(p + "self_attn.q_norm.weight"));
      kernels::launch_scale_copy(wq, wq, hd_l, std::sqrt(static_cast<float>(hd_l)), stream_);
    }
    if (has_ple()) {
      for (const char* t : {"per_layer_input_gate.weight", "per_layer_projection.weight"})
        load_weight(p + t, t);
      upload(p + "post_per_layer_input_norm.weight");
    }
    if (cfg_.enable_moe_block) {
      // Norms and the tiny router stay fp16; the experts are ~90% of the model, so
      // they go through the same quantiser as any other projection (they are plain
      // 2-D matrices here; see parse_gemma4_st_config).
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
  if (std::getenv("CPI_PLAN_NO_GRAPH")) decode_graph_enabled_ = false;
  // dp4a decode routing needs a quantized copy of the current activation vector.
  // 64 KB covers any in_dim in the model zoo; CPI_CUDA_NO_DP4A keeps the fp16-activation
  // quant kernels (the pre-parity behaviour) for A/B and bisection.
  if ((weight_quant_bits_ == 4 || weight_quant_bits_ == 8) && d_act_i8_ == nullptr &&
      std::getenv("CPI_CUDA_NO_DP4A") == nullptr) {
    G4_CHECK(cudaMalloc(&d_act_i8_, std::size_t(16) * 65536));  // up to 16 tokens' activations
    G4_CHECK(cudaMalloc(&d_act_qs_, std::size_t(16) * (65536 / 32) *
                                        sizeof(float)));  // g32 x scales, up to 16 tokens
    // Largest projection on the supported models is <= 16384 x 4096 halfs (128 MB is the
    // ceiling; E2B's largest is 12288 x 1536 = 38 MB). Sized generously once.
    G4_CHECK(cudaMalloc(&d_seq_dequant_, std::size_t(16384) * 4096 * sizeof(__half)));
    // int8-direct prefill scratch: activations [4096 x 16384] int8 + per-token scales +
    // one 512-token int32 GEMM tile [16384 x 512].
    G4_CHECK(cudaMalloc(&d_seq_x8_, std::size_t(4096) * 16384));
    G4_CHECK(cudaMalloc(&d_seq_xs_, 4096 * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_seq_i32_, std::size_t(16384) * 512 * sizeof(int)));
  }
  if (std::getenv("CPI_PLAN_NO_DEVICE_TOPK")) device_topk_enabled_ = false;

  // A directory ⇒ HuggingFace safetensors; a file ⇒ .cpi. Within a directory the
  // architecture comes from config.json's model_type; the engine reads whichever
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
      // Decode-graph capture stays on: every op in this plan has a device-position variant
      // (partial RoPE gained its twin; launch_rope_inplace_partial_table_device_pos; the
      // sliding/full attention and KV store already had theirs). Gate: graphs-on and
      // CPI_PLAN_NO_GRAPH=1 must produce token-identical streams.
      G4_CHECK(cudaStreamCreate(&stream_));
      load_all(cpi_path);
      build_rope_tables();
      allocate_buffers();
      build_plan();
      persist_enabled_ = std::getenv("CPI_CUDA_PERSISTENT") != nullptr && compile_persistent_plan();
      // The vision tower, when the checkpoint ships one. Opt out with
      // CPI_NO_VISION=1 (it costs VRAM even on text-only prompts).
      // Sequence prefill is a capability of the plan: partial RoPE and delta-net have no
      // sequence kernels yet, so those plans keep the token-by-token prefill.
      seq_prefill_ok_ = plan_can_sequence_prefill();
      if (seq_prefill_ok_) allocate_sequence_buffers(std::min(max_ctx_, 4096));

      parse_vision_config(cpi_path);
      if (vcfg_.present && !std::getenv("CPI_NO_VISION")) {
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
    if (mt.rfind("deepseek_v2", 0) == 0) {
      // DeepSeek-V2 (MLA latent attention + fine-grained MoE). Weights are HF-native under "model.".
      // WIP bring-up: config parse is live; loader/ops/plan land phase by phase (see
      // cpi-deepseek-v2lite-status memory).
      wprefix_ = "model.";
      parse_deepseek_config(cpi_path);
      if (max_ctx_ <= 0) max_ctx_ = std::min(4096, cfg_.max_position_embeddings);
      G4_CHECK(cudaStreamCreate(&stream_));
      // Absorbed-MLA decode (default; CPI_DS_NO_ABSORB=1 restores materialized):
      // token-identical to the materialized path and reads ~9x fewer attention
      // bytes at depth, giving a much flatter decode curve (336/325/293 tok/s at
      // depth 7/512/2048 vs materialized 351/319/251). Decided before the weight
      // load: the loader uploads the absorbed-layout kv_b copies, and
      // allocate_buffers sizes the latent caches. Accumulators cover kv_lora <= 512.
      mla_absorb_ = std::getenv("CPI_DS_NO_ABSORB") == nullptr && cfg_.kv_lora_rank <= 512;
      load_deepseek_weights();
      allocate_buffers();
      // int4 decode routes through dp4a with int8 activations, which needs this scratch. The shared
      // path allocates it only for .cpi/.ll2c (this dir branch returns before it), so without this the
      // DeepSeek int4 projections silently fell to the slow fp16-activation matvec.
      if (weight_quant_bits_ == 4 && d_act_i8_ == nullptr &&
          std::getenv("CPI_CUDA_NO_DP4A") == nullptr) {
        G4_CHECK(cudaMalloc(&d_act_i8_, std::size_t(16) * 65536));
        G4_CHECK(cudaMalloc(&d_act_qs_, std::size_t(16) * (65536 / 32) * sizeof(float)));
      }
      {
        const engine::YarnRope y = engine::deepseek_yarn_rope(
            cfg_.qk_rope_head_dim, cfg_.rope_theta, cfg_.yarn_factor, cfg_.yarn_beta_fast,
            cfg_.yarn_beta_slow, cfg_.yarn_original_max_pos, cfg_.yarn_mscale, cfg_.yarn_mscale);
        mla_attn_scaling_ = y.attention_scaling;
        G4_CHECK(cudaMemcpy(d_inv_freq_, y.inv_freq.data(), y.inv_freq.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      }
      build_deepseek_plan();
      // Batched sequence prefill: MlaAssembleRope has a T>1 kernel and the MoE ops loop the T==1
      // kernels per token (d_moe_idx_seq_), so the whole prompt runs in one pass; the projections,
      // MLA attention and o_proj batch as GEMMs; only the experts stay matvec. Decode uses CUDA-graph
      // capture (MlaAssembleRope's device-position variant keeps the graph position-correct on replay;
      // verified token-identical to eager). Persistent decode stays off unless CPI_CUDA_PERSISTENT.
      // CPI_DS_NO_SEQ_PREFILL=1 forces the old token-by-token prefill (a correctness escape hatch).
      seq_prefill_ok_ =
          std::getenv("CPI_DS_NO_SEQ_PREFILL") == nullptr && plan_can_sequence_prefill();
      if (seq_prefill_ok_) allocate_sequence_buffers(std::min(max_ctx_, 4096));
      G4_CHECK(cudaStreamSynchronize(stream_));
      return;
    }
    parse_qwen35_config(cpi_path);
    if (max_ctx_ <= 0 ||
        (cfg_.max_position_embeddings > 0 && max_ctx_ > cfg_.max_position_embeddings))
      max_ctx_ = std::min(max_ctx_ > 0 ? max_ctx_ : 2048, cfg_.max_position_embeddings);
    // Partial RoPE gained its device-position twin (launch_rope_inplace_partial_table_device_pos),
    // so this plan is graph-capturable now: the delta-net state ops read/write fixed device
    // buffers and every position-dependent op has a device-position variant. Gate: graphs-on and
    // CPI_PLAN_NO_GRAPH=1 must produce token-identical streams.
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
  if (cfg_.enable_moe_block && weight_quant_bits_ != 4 && weight_quant_bits_ != 8)
    throw std::runtime_error(
        "Gemma 4 MoE (" + std::to_string(cfg_.num_experts) +
        " experts) needs quantized experts to fit in VRAM: run with --weight-quant int4 "
        "(or int8). The fp16 experts alone (~90% of the model) exceed 32GB.");
  G4_CHECK(cudaStreamCreate(&stream_));
  load_all(cpi_path);
  build_rope_tables();
  allocate_buffers();
  build_plan();  // resolve the per-layer forward into a data op-plan (once)
  persist_enabled_ = std::getenv("CPI_CUDA_PERSISTENT") != nullptr && compile_persistent_plan();
  // Sequence prefill: this route must set the capability flag too, or every .cpi model
  // prefills token-by-token (orders of magnitude slower than the batched path).
  seq_prefill_ok_ = plan_can_sequence_prefill();
  if (seq_prefill_ok_) allocate_sequence_buffers(std::min(max_ctx_, 4096));
  G4_CHECK(cudaStreamSynchronize(stream_));
}

// Resolve the per-layer forward into a data op-plan once at load. This is the
// exact op sequence the former imperative run_layer emitted, with the per-layer
// conditionals (shared / keqv / full / PLE) decided here and weights + geometry
// bound, so the hot loop (run_layer, below) is branch-free over identity.
// The MoE feed-forward, appended to a layer whose dense MLP output is already in
// Slot::Tmp (pre-norm). Transcribed from Gemma4TextDecoderLayer.forward:
//
//   h1 = post_ffn_norm_1(dense_mlp_out)
//   tkw, tki = router(X)                       <-- the pre-MLP residual, not the MLP output
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
      o.pf_qweight = q->second.pf_i8;
      o.pf_qscales = q->second.pf_scales;
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

// cos/sin over the spatial half-dim: inv_freq = theta^(-2j/spatial_dim) with
// spatial_dim = head_dim/2, giving head_dim/4 entries per position. Both axes share
// the table; x and y get identical frequency ranges (they are separate positions,
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
    // Note the ".linear." infix: the projections are wrapped in Gemma4ClippableLinear.
    // Its clamp bounds are +-inf and are absent from the checkpoint, so it is a plain
    // Linear; do not implement the clipping.
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
    // clamps with them. The 26B sets use_clipped_linears=false and ships none. Skipping
    // these does not crash; it silently changes the numbers, which is why the gate
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

    // The same sandwich-norm shape as a Gemma text layer. The differences are that
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

  // Epilogue: standardize, when the checkpoint has it. Pooling and projection are not
  // ops; pooling changes the token count, so the executor has to be re-entered with
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
  // stage <  0: run the first |stage| ops of layer 0 and return Slot::Q; lets the
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

  // The tower is bidirectional and cacheless: one sequence pass over all P patches.
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
  slot_ptr_[static_cast<int>(Slot::MlaCkv)] = d_mla_ckv_;
  slot_ptr_[static_cast<int>(Slot::MlaLatent)] = d_mla_latent_;
  slot_ptr_[static_cast<int>(Slot::MlaKvb)] = d_mla_kvb_;

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
    // Image embeddings are spliced in here; before anything reads X. In particular the
    // per-layer-input projection below reads X, and HF projects it from the post-scatter
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
      {  // binds the int4 form when the load quantized it (same rule as the layer gemvs)
        const auto q = qdev_.find(wprefix_ + "per_layer_model_projection.weight");
        if (q != qdev_.end()) {
          g.qweight = q->second.packed;
          g.qscales = q->second.scales;
          g.qbits = weight_quant_bits_;
          g.qgroup = q->second.group;
          g.pf_qweight = q->second.pf_i8;
          g.pf_qscales = q->second.pf_scales;
        } else {
          g.weight = dev_at(wprefix_ + "per_layer_model_projection.weight");
        }
      }
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
    // Resolved by name so a weight that was quantized at load binds its int4 form.
    // The choice is made here, once; the hot loop just runs whatever the op holds.
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
        o.pf_qweight = q->second.pf_i8;
        o.pf_qscales = q->second.pf_scales;
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
    // q/k/v all read XNorm, so they are emitted adjacent: the executor's cat peephole
    // runs adjacent same-input projections as one launch. The norms/ropes that consume
    // them follow after; on keqv layers the raw-K copy still happens before k_norm.
    gemv(Slot::XNorm, Slot::Q, "self_attn.q_proj.weight", qdim, H);
    if (!shared) {
      gemv(Slot::XNorm, Slot::K, "self_attn.k_proj.weight", kvdim, H);
      if (!keqv) gemv(Slot::XNorm, Slot::V, "self_attn.v_proj.weight", kvdim, H);
    }
    rms(Slot::Q, Slot::Q, W("self_attn.q_norm.weight"), nq, hd);
    rope(Slot::Q, nq);
    if (!shared) {
      if (keqv) {  // V shares the raw k_proj output (before k_norm/rope)
        Op o;
        o.kind = OpKind::CopySlot;
        o.in = Slot::K;
        o.out = Slot::V;
        o.cols = kvdim;
        ops.push_back(o);
      }
      rms(Slot::K, Slot::K, W("self_attn.k_norm.weight"), nkv, hd);
      rope(Slot::K, nkv);
      rms(Slot::V, Slot::V, nullptr, nkv, hd);  // weightless v-norm (ones)
      Op st;
      st.kind = OpKind::KvStore;
      st.cols = kvdim;
      ops.push_back(st);
    }
    // net attention scale = 1.0: sqrt(hd) is folded into the q_norm weight at load (it
    // commutes with the rotation), cancelling the kernel's internal 1/sqrt(hd) without a
    // per-layer ScaleCopy op.
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
    // MoE models run a second, routed feed-forward in parallel with the dense one and
    // sum them. Slot::Tmp holds the dense output (pre-norm) on entry, and the summed
    // result on exit, so the shared tail below is identical either way.
    // CPI_PLAN_NO_MOE=1 drops the routed branch, leaving the dense MLP only.
    // Bisection aid: it separates "the shared Gemma path is wrong" from "the MoE ops
    // are wrong" (the model is degraded but must stay finite and coherent-ish).
    if (cfg_.enable_moe_block && !std::getenv("CPI_PLAN_NO_MOE")) append_moe_ffn_ops(ops, p, L);
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

    // --- per-layer scalar (a checkpoint without one means 1.0; emitting a no-op
    // ScaleCopy per layer costs a real ~1 us kernel-latency floor each) ---
    if (layer_scalar_host_[L] != 1.0f) scale(Slot::X, H, layer_scalar_host_[L]);
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
    if (d_head_q_ != nullptr) {
      h.qweight = d_head_q_;
      h.qscales = d_head_qs_;
      h.qbits = head_qbits_;
      h.qgroup = head_qgroup_;
    } else {
      h.weight = dev_at(wprefix_ + "embed_tokens.weight");
    }
    h.cols = cfg_.vocab;
    h.in_dim = H;
    epi.push_back(h);
  }
}

// Execute an op list at `position`. Branch-free over model identity; the only
// runtime-varying inputs are `position` (RoPE / KV store / attention window) and
// the executor's device token buffer (EmbeddingLookup). `layer` selects the
// cache for KvStore/Attention; the prologue/epilogue pass -1 (no such ops).
// Sequence-mode GEMM: y[T,out] = x[T,in] * W[out,in]^T via cublasGemmEx (fp16 in, fp32
// accumulate); the classic row-major-as-column-major call. Falls back to the hand-rolled
// GEMM when cuBLAS is disabled or the op carries activation clips (vision projections).
bool PlanCudaEngine::seq_gemm_cublas(const __half* w, const __half* x, __half* y, int out, int in,
                                     int T) {
  static const bool disabled = std::getenv("CPI_CUDA_NO_CUBLAS") != nullptr;
  if (disabled) return false;
  if (cublas_ == nullptr) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return false;
    cublas_ = h;
    // A persistent workspace so cuBLAS does not allocate/free one per call (which serializes
    // the host and idles the GPU across the many per-layer projection GEMMs).
    constexpr std::size_t kWs = 32u << 20;
    if (cudaMalloc(&d_cublas_ws_, kWs) == cudaSuccess) cublasSetWorkspace(h, d_cublas_ws_, kWs);
  }
  cublasHandle_t h = static_cast<cublasHandle_t>(cublas_);
  cublasSetStream(h, stream_);
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, out, T, in, &alpha, w,
                                         CUDA_R_16F, in, x, CUDA_R_16F, in, &beta, y, CUDA_R_16F,
                                         out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return st == CUBLAS_STATUS_SUCCESS;
}

// One cublasGemmGroupedBatchedEx over every expert GEMM (group_size 1 each): C[g] = A[g]^T . B[g],
// A [m,k] row-major (OP_T), B [n_g,k] row-major, C [n_g,m] row-major, the same layout as
// seq_gemm_cublas, batched. The array params live in host memory; the pointers inside them are
// device addresses. Collapses ~64 tiny per-expert calls into one, removing the per-GEMM latency
// floor that dominated grouped-MoE prefill's fixed cost.
bool PlanCudaEngine::grouped_gemm_ex(int m, int k, const std::vector<int>& n_per_group,
                                     const std::vector<const void*>& A,
                                     const std::vector<const void*>& B,
                                     const std::vector<void*>& C) {
  const int G = static_cast<int>(n_per_group.size());
  if (G == 0) return true;
  if (cublas_ == nullptr) {
    cublasHandle_t nh = nullptr;
    if (cublasCreate(&nh) != CUBLAS_STATUS_SUCCESS) return false;
    cublas_ = nh;
    constexpr std::size_t kWs = 32u << 20;
    if (cudaMalloc(&d_cublas_ws_, kWs) == cudaSuccess) cublasSetWorkspace(nh, d_cublas_ws_, kWs);
  }
  cublasHandle_t h = static_cast<cublasHandle_t>(cublas_);
  cublasSetStream(h, stream_);
  // Reused member scratch (assign keeps capacity after the first call) so the twice-per-layer prefill
  // GEMM does not churn the heap.
  gg_ta_.assign(G, static_cast<int>(CUBLAS_OP_T));
  gg_tb_.assign(G, static_cast<int>(CUBLAS_OP_N));
  gg_ma_.assign(G, m);
  gg_na_.assign(n_per_group.begin(), n_per_group.end());
  gg_ka_.assign(G, k);
  gg_lda_.assign(G, k);
  gg_ldb_.assign(G, k);
  gg_ldc_.assign(G, m);
  gg_gs_.assign(G, 1);
  gg_alpha_.assign(G, 1.0f);
  gg_beta_.assign(G, 0.0f);
  // The pointer arrays must live in device memory; the scalar arrays stay on host.
  G4_CHECK(cudaMemcpyAsync(d_gg_A_, A.data(), G * sizeof(void*), cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_gg_B_, B.data(), G * sizeof(void*), cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_gg_C_, C.data(), G * sizeof(void*), cudaMemcpyHostToDevice, stream_));
  const cublasStatus_t st = cublasGemmGroupedBatchedEx(
      h, reinterpret_cast<cublasOperation_t*>(gg_ta_.data()),
      reinterpret_cast<cublasOperation_t*>(gg_tb_.data()), gg_ma_.data(), gg_na_.data(), gg_ka_.data(),
      gg_alpha_.data(), d_gg_A_, CUDA_R_16F, gg_lda_.data(), d_gg_B_, CUDA_R_16F, gg_ldb_.data(),
      gg_beta_.data(), d_gg_C_, CUDA_R_16F, gg_ldc_.data(), G, gg_gs_.data(), CUBLAS_COMPUTE_32F);
  return st == CUBLAS_STATUS_SUCCESS;
}

// Grouped-GEMM MoE gate_up (DeepSeek seq prefill). Routes on the host (T*top_k is tiny), buckets
// tokens by expert, gathers each expert's rows, then dequant+cuBLAS gate & up GEMMs per expert and
// a single SiLU-gate over the whole grouped batch. The gated inter stays in d_moe_gate_g_ (grouped
// layout) for moe_grouped_down. Returns false to fall back to the per-token loop.
// Short-prompt MoE prefill: read int4 expert weights directly (no dequant-all to fp16). At few
// tokens/expert the dequant-all (~28ms) + the fp16 re-read dominate; at many tokens/expert the
// dequant amortizes and cuBLAS's fp16 tensor-core GEMM wins, so we cross over on P (= T*top_k).
// CPI_DS_MOE_INT4=1 forces on, =0 forces off (A/B); CPI_DS_MOE_INT4_MAX_P overrides the threshold.
bool PlanCudaEngine::moe_use_int4_direct(int P) const {
  // The int4-direct path runs on int8 tensor cores (mma.m16n8k32.s8, sm_80+);
  // pre-Ampere devices take the fp16-dequant grouped path unconditionally.
  static const bool mma_ok = [] {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
      return false;
    return major >= 8;
  }();
  if (!mma_ok) return false;
  static const int forced = [] {
    const char* f = std::getenv("CPI_DS_MOE_INT4");
    return f ? (std::atoi(f) != 0 ? 1 : -1) : 0;
  }();
  if (forced != 0) return forced == 1;
  static const int max_p = [] {
    const char* t = std::getenv("CPI_DS_MOE_INT4_MAX_P");
    return t ? std::atoi(t) : 6144;  // ~1024 tokens at top_k=6: measured crossover ~1150 tok
  }();
  return P <= max_p;
}

bool PlanCudaEngine::moe_grouped_gate_up(const opplan::Op& op, __half* xnorm, int T) {
  moe_grouped_total_ = 0;  // set to P only on full success; keeps down() from reading partial state
  if (d_moe_gather_ == nullptr || op.qbits != 4 || d_seq_dequant_ == nullptr) return false;
  const int E = cfg_.num_experts, K = op.heads, H = op.in_dim, MI = op.cols, group = op.qgroup;
  const int P = T * K;
  // Pull the [T*top_k] selections/weights to the host and bucket them by expert.
  h_moe_idx_seq_.resize(P);
  h_moe_w_seq_.resize(P);
  G4_CHECK(cudaMemcpyAsync(h_moe_idx_seq_.data(), d_moe_idx_seq_, P * sizeof(int),
                           cudaMemcpyDeviceToHost, stream_));
  G4_CHECK(cudaMemcpyAsync(h_moe_w_seq_.data(), d_moe_w_seq_, P * sizeof(float),
                           cudaMemcpyDeviceToHost, stream_));
  G4_CHECK(cudaStreamSynchronize(stream_));
  std::vector<int>& off = h_moe_off_;
  off.assign(E + 1, 0);
  for (int i = 0; i < P; ++i) off[h_moe_idx_seq_[i] + 1]++;
  for (int e = 0; e < E; ++e) off[e + 1] += off[e];
  h_moe_perm_.resize(P);
  h_moe_cursor_.assign(off.begin(), off.begin() + E);
  h_moe_rweight_.resize(P);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < K; ++k) {
      const int e = h_moe_idx_seq_[t * K + k];
      const int pos = h_moe_cursor_[e]++;
      h_moe_perm_[pos] = t;
      h_moe_rweight_[pos] = h_moe_w_seq_[t * K + k];
    }
  G4_CHECK(cudaMemcpyAsync(d_moe_perm_, h_moe_perm_.data(), P * sizeof(int), cudaMemcpyHostToDevice,
                           stream_));
  G4_CHECK(cudaMemcpyAsync(d_moe_rweight_, h_moe_rweight_.data(), P * sizeof(float),
                           cudaMemcpyHostToDevice, stream_));
  engine::launch_moe_gather_rows(xnorm, d_moe_perm_, d_moe_gather_, H, P, stream_);

  // int4-direct path (short prompts): int8-quantize the gathered activations (natural g32), upload the
  // per-expert offsets, and run one grouped int8-mma GEMM that reads the int4 weights directly into
  // gate_g/up_g. Skips the dequant-all + fp16 re-read entirely.
  if (moe_use_int4_direct(P)) {
    int max_ne = 0;
    for (int e = 0; e < E; ++e) max_ne = std::max(max_ne, off[e + 1] - off[e]);
    G4_CHECK(cudaMemcpyAsync(d_moe_off_dev_, off.data(), (E + 1) * sizeof(int), cudaMemcpyHostToDevice,
                             stream_));
    kernels::launch_quantize_groupwise_fp16_to_int8(d_moe_gather_, d_moe_actq_, d_moe_acts_, P, H, 32,
                                                    stream_, 127);
    kernels::launch_moe_int4_grouped_mma(
        d_moe_actq_, d_moe_acts_, static_cast<const std::int8_t*>(op.qweight),
        static_cast<const float*>(op.qscales), d_moe_off_dev_, d_moe_gate_g_, d_moe_up_g_, E, 2 * MI, H,
        max_ne, group, /*split=*/MI, /*lo_width=*/MI, /*hi_width=*/MI, stream_);
    kernels::launch_silu_mul(d_moe_gate_g_, d_moe_up_g_, d_moe_gate_g_, P * MI, stream_);
    moe_grouped_total_ = P;
    return true;
  }

  // Dequant all experts' fused [E*2MI, H] gate_up in one launch, then one grouped GEMM: two groups
  // per expert (gate rows [e*2MI, +MI), up rows [+MI, +2MI)) into the separate gate_g/up_g buffers.
  kernels::launch_dequant_int4_grouped(static_cast<const std::int8_t*>(op.qweight),
                                       static_cast<const float*>(op.qscales), d_moe_dqw_,
                                       E * 2 * MI, H, group, stream_);
  std::vector<int>& ng_list = gg_ng_;
  std::vector<const void*>& A = gg_pa_;
  std::vector<const void*>& B = gg_pb_;
  std::vector<void*>& C = gg_pc_;
  ng_list.clear();
  A.clear();
  B.clear();
  C.clear();
  for (int e = 0; e < E; ++e) {
    const int n_e = off[e + 1] - off[e];
    if (n_e <= 0) continue;
    const __half* We = d_moe_dqw_ + static_cast<std::size_t>(e) * 2 * MI * H;
    const __half* Be = d_moe_gather_ + static_cast<std::size_t>(off[e]) * H;
    A.push_back(We);  // gate rows
    B.push_back(Be);
    C.push_back(d_moe_gate_g_ + static_cast<std::size_t>(off[e]) * MI);
    ng_list.push_back(n_e);
    A.push_back(We + static_cast<std::size_t>(MI) * H);  // up rows
    B.push_back(Be);
    C.push_back(d_moe_up_g_ + static_cast<std::size_t>(off[e]) * MI);
    ng_list.push_back(n_e);
  }
  if (!grouped_gemm_ex(MI, H, ng_list, A, B, C)) return false;
  // SiLU-gate the whole grouped batch in place: gate_g = silu(gate_g) * up_g.
  kernels::launch_silu_mul(d_moe_gate_g_, d_moe_up_g_, d_moe_gate_g_, P * MI, stream_);
  moe_grouped_total_ = P;  // success: down() may now consume the cached routing + gated inter
  return true;
}

// Grouped-GEMM MoE down (DeepSeek seq prefill). Per-expert down GEMM over the gated inter left by
// moe_grouped_gate_up, then a weighted scatter back into moe_out (fp32 accumulator -> fp16).
bool PlanCudaEngine::moe_grouped_down(const opplan::Op& op, __half* moe_out, int T) {
  if (moe_grouped_total_ <= 0 || op.qbits != 4 || d_seq_dequant_ == nullptr) return false;
  const int E = cfg_.num_experts, H = op.cols, MI = op.in_dim, group = op.qgroup;
  const int P = moe_grouped_total_;
  const std::vector<int>& off = h_moe_off_;
  if (moe_use_int4_direct(P)) {
    // int4-direct: int8-quantize the gated inter (d_moe_gate_g_ [P, MI]), then one grouped int8-mma
    // GEMM reading the int4 down weights directly into d_moe_outg_. off_dev was uploaded by gate_up
    // (same P -> same path). split=H, no hi output (down is not fused).
    int max_ne = 0;
    for (int e = 0; e < E; ++e) max_ne = std::max(max_ne, off[e + 1] - off[e]);
    kernels::launch_quantize_groupwise_fp16_to_int8(d_moe_gate_g_, d_moe_actq_, d_moe_acts_, P, MI, 32,
                                                    stream_, 127);
    kernels::launch_moe_int4_grouped_mma(
        d_moe_actq_, d_moe_acts_, static_cast<const std::int8_t*>(op.qweight),
        static_cast<const float*>(op.qscales), d_moe_off_dev_, d_moe_outg_, nullptr, E, H, MI, max_ne,
        group, /*split=*/H, /*lo_width=*/H, /*hi_width=*/0, stream_);
  } else {
    // Dequant all experts' [E*H, MI] down in one launch, then one grouped GEMM (one group/expert).
    kernels::launch_dequant_int4_grouped(static_cast<const std::int8_t*>(op.qweight),
                                         static_cast<const float*>(op.qscales), d_moe_dqw_, E * H, MI,
                                         group, stream_);
    std::vector<int>& ng_list = gg_ng_;
    std::vector<const void*>& A = gg_pa_;
    std::vector<const void*>& B = gg_pb_;
    std::vector<void*>& C = gg_pc_;
    ng_list.clear();
    A.clear();
    B.clear();
    C.clear();
    for (int e = 0; e < E; ++e) {
      const int n_e = off[e + 1] - off[e];
      if (n_e <= 0) continue;
      A.push_back(d_moe_dqw_ + static_cast<std::size_t>(e) * H * MI);  // down expert e at row e*H
      B.push_back(d_moe_gate_g_ + static_cast<std::size_t>(off[e]) * MI);
      C.push_back(d_moe_outg_ + static_cast<std::size_t>(off[e]) * H);
      ng_list.push_back(n_e);
    }
    if (!grouped_gemm_ex(H, MI, ng_list, A, B, C)) return false;
  }
  // Scatter each grouped row's weighted output back to its source token (fp32 accumulate; a token
  // receives top_k contributions from different experts), then convert to fp16 MoeOut.
  G4_CHECK(cudaMemsetAsync(d_moe_out_f32_, 0, static_cast<std::size_t>(T) * H * sizeof(float),
                           stream_));
  engine::launch_moe_scatter_accum(d_moe_outg_, d_moe_perm_, d_moe_rweight_, d_moe_out_f32_, H, P,
                                   stream_);
  engine::launch_moe_f32_to_f16(d_moe_out_f32_, moe_out, static_cast<std::size_t>(T) * H, stream_);
  moe_grouped_total_ = 0;
  return true;
}

// Tensor-core attention prefill for full-attention layers: batched QK^T -> causal
// softmax -> PV via cuBLAS, the LlamaEngine machinery (device-built pointer arrays keep
// GQA groups straight). The scalar fallback measured 5 ms per layer at 1216 tokens
// 32% of the whole prefill; this does the same work in tensor-core GEMMs. Sliding
// layers keep the masked tiled kernel (the window mask lives only there).
bool PlanCudaEngine::prefill_attention_tc(const __half* q, const __half* k, const __half* v,
                                          __half* out, int rows, int base_pos, int heads,
                                          int kv_heads, int hd, int window) {
  static const bool disabled = std::getenv("CPI_CUDA_NO_TC_PREFILL") != nullptr;
  if (disabled || cublas_ == nullptr) return false;
  if (heads <= 0 || kv_heads <= 0 || (heads % kv_heads) != 0) return false;
  constexpr int kChunk = 1024;
  if (d_pf_scores_ == nullptr) {
    // heads x kChunk x max_ctx fp16 scores + 6*heads device pointers.
    if (cudaMalloc(&d_pf_scores_, static_cast<std::size_t>(heads) * kChunk * max_ctx_ *
                                      sizeof(__half)) != cudaSuccess)
      return false;
    if (cudaMalloc(&d_pf_ptrs_, sizeof(void*) * 6 * heads) != cudaSuccess) return false;
  }
  cublasHandle_t h = static_cast<cublasHandle_t>(cublas_);
  cublasSetStream(h, stream_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const float zero = 0.0f, one = 1.0f;
  const int keys = base_pos + rows;
  const int group = heads / kv_heads;
  const int kv_stride = kv_heads * hd;
  const int q_stride = heads * hd;
  const int out_stride = heads * hd;
  for (int c0 = 0; c0 < rows; c0 += kChunk) {
    const int chunk = std::min(kChunk, rows - c0);
    kernels::launch_build_attention_ptrs(k, v, q, d_pf_scores_, out, d_pf_ptrs_, heads, group, hd,
                                         kChunk, keys, q_stride, out_stride, c0, stream_);
    const void* const* A1 = const_cast<const void* const*>(d_pf_ptrs_);
    const void* const* B1 = const_cast<const void* const*>(d_pf_ptrs_ + heads);
    void* const* C1 = d_pf_ptrs_ + 2 * heads;
    const void* const* A2 = const_cast<const void* const*>(d_pf_ptrs_ + 3 * heads);
    const void* const* B2 = const_cast<const void* const*>(d_pf_ptrs_ + 4 * heads);
    void* const* C2 = d_pf_ptrs_ + 5 * heads;
    if (cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, keys, chunk, hd, &scale, A1, CUDA_R_16F,
                            kv_stride, B1, CUDA_R_16F, q_stride, &zero, C1, CUDA_R_16F, keys, heads,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
      return false;
    kernels::launch_softmax_causal_rows(d_pf_scores_, heads, kChunk, chunk, keys, base_pos + c0,
                                        stream_, window);
    if (cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, hd, chunk, keys, &one, A2, CUDA_R_16F,
                            kv_stride, B2, CUDA_R_16F, keys, &zero, C2, CUDA_R_16F, out_stride,
                            heads, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
      return false;
  }
  return true;
}

void PlanCudaEngine::stage_moe_experts(const void* qw, const float* qs, int rpe, int in_features,
                                       int group, std::int8_t* dst_w, float* dst_s) {
  const int K = cfg_.top_k_experts;
  const std::size_t row_bytes = (in_features + 1) / 2;  // int4 packed row
  const std::size_t ng = kernels::quant_group_count(in_features, group);
  if (static_cast<int>(h_moe_idx_.size()) < K) h_moe_idx_.resize(K);
  // The router wrote the selection on device; bring the K indices to the host to issue the copies.
  // This host sync per MoE layer is what makes streaming graph-incompatible (MoE layers run eager).
  G4_CHECK(cudaMemcpyAsync(h_moe_idx_.data(), d_moe_idx_, static_cast<std::size_t>(K) * sizeof(int),
                           cudaMemcpyDeviceToHost, stream_));
  G4_CHECK(cudaStreamSynchronize(stream_));
  const std::int8_t* w = static_cast<const std::int8_t*>(qw);
  // Step 2: when the experts live on host, this is H2D (the memory win); step-1 D2D otherwise.
  const cudaMemcpyKind kind =
      moe_experts_on_host_ ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
  for (int k = 0; k < K; ++k) {
    const std::size_t e = static_cast<std::size_t>(h_moe_idx_[k]);
    const std::size_t rp = static_cast<std::size_t>(rpe);
    G4_CHECK(cudaMemcpyAsync(dst_w + static_cast<std::size_t>(k) * rp * row_bytes,
                             w + e * rp * row_bytes, rp * row_bytes, kind, stream_));
    G4_CHECK(cudaMemcpyAsync(dst_s + static_cast<std::size_t>(k) * rp * ng, qs + e * rp * ng,
                             rp * ng * sizeof(float), kind, stream_));
  }
}

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
  // N tokens is just N times as many groups; no new kernel.
  const int T = ctx.tokens;
  const bool seq = T > 1;
  auto rows_x = [&](int rows) { return rows * T; };
  auto len_x = [&](int len) { return len * T; };
  // CPI_CUDA_PROFILE=1: per-op-kind GPU time via event pairs, printed at exit. Decode-only
  // ranking tool; it serialises nothing but each op pays two event records, so its absolute
  // numbers are honest and its use is with CPI_PLAN_NO_GRAPH=1 (a captured graph
  // bypasses this function entirely).
  static const bool prof = std::getenv("CPI_CUDA_PROFILE") != nullptr;
  struct ProfBin {
    double ms = 0.0;
    std::uint64_t count = 0;
  };
  static std::map<std::string, ProfBin> prof_bins;
  static struct ProfDump {
    ~ProfDump() {
      if (prof_bins.empty()) return;
      std::fprintf(stderr, "[cuda-prof] per-op-kind GPU ms (sum / count / us-per):\n");
      double total = 0.0;
      for (const auto& kv : prof_bins) total += kv.second.ms;
      for (const auto& kv : prof_bins) {
        std::fprintf(stderr, "  %-18s %9.2f ms  %8llu  %6.2f us  %5.1f%%\n", kv.first.c_str(),
                     kv.second.ms, static_cast<unsigned long long>(kv.second.count),
                     1000.0 * kv.second.ms / static_cast<double>(kv.second.count),
                     100.0 * kv.second.ms / total);
      }
      std::fprintf(stderr, "  %-18s %9.2f ms\n", "TOTAL", total);
    }
  } prof_dump;
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;
  // dp4a activation-quant cache: adjacent quant Gemvs reading the same vector (q/k/v,
  // gate/up) share one quantize launch. Slots are reused within a layer (XNorm is written
  // twice with the same pointer and length), so identity alone would hit stale; any
  // non-Gemv op invalidates, which keeps exactly the adjacent-run reuse and nothing else.
  const void* actq_src = nullptr;
  int actq_len = 0;
  bool actq_set_now = false;
  const void* mtq_src = nullptr;  // multi-token act-quant cache (seq T <= 8)
  int mtq_len = 0;
  for (std::size_t idx = 0; idx < n; ++idx) {
    const Op& op = ops[idx];
    if (prof) {
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      cudaEventRecord(ev0, stream_);
    }

    // decode fusion peepholes; opt-in (CPI_PLAN_FUSE=1), because the premise was
    // measured and did not survive. The eager profiler prices every tiny op at ~11-14 us,
    // which made these two patterns (~200 of Gemma 4's ~870 per-token kernels) look like a
    // fifth of the token; fused-vs-unfused interleaved A/B under graph replay measured
    // 100.2/103.2 vs 101.1/105.5 tok/s; neutral to slightly negative. The ~12 us was
    // launch cost that graph replay already amortises, not execution the fusion could
    // remove: a limiter read off the eager path cannot be quoted as the graph's. Kept as an
    // experiment lever, not a default: fusion also changes reduction order, so opt-in runs
    // may shift near-tie tokens.
    // adjacent-projection cat; default on (CPI_CUDA_NO_CAT=1 restores split launches).
    // Two or three fp16 decode projections reading the same input vector (q|k|v, gate|up)
    // run as one launch. Unlike the opt-in fusions below this survives its measurement:
    // the 256-row k/v grids are pure exposed latency behind q's launch, and the cat kernel
    // is per-row bit-identical to the split wide gemv, so the token stream cannot move.
    // fp16 GLU fusion, checked before the cat peephole so [gate][up][gelu] outranks
    // a plain gate|up cat.
    static const bool no_cat = std::getenv("CPI_CUDA_NO_CAT") != nullptr;
    if (!seq && !no_cat && op.kind == OpKind::Gemv && op.qbits == 0 && op.bias == nullptr &&
        (op.in_dim % 8) == 0 && idx + 2 < n && ops[idx + 1].kind == OpKind::Gemv &&
        ops[idx + 1].qbits == 0 && ops[idx + 1].bias == nullptr && ops[idx + 1].in == op.in &&
        ops[idx + 1].in_dim == op.in_dim && ops[idx + 1].cols == op.cols &&
        ops[idx + 2].kind == OpKind::GeluMul && ops[idx + 2].aux_ptr == nullptr &&
        ops[idx + 2].aux_offset == 0 && ops[idx + 2].in == op.out &&
        ops[idx + 2].in2 == ops[idx + 1].out && ops[idx + 2].cols == op.cols &&
        ops[idx + 2].out != op.in) {
      kernels::launch_half_gemv_glu(HW(op.weight), HW(ops[idx + 1].weight), S(op.in),
                                    S(ops[idx + 2].out), op.cols, op.in_dim, stream_);
      idx += 2;
      goto op_done;
    }
    if (!seq && !no_cat && op.kind == OpKind::Gemv && op.qbits == 0 && op.bias == nullptr &&
        (op.in_dim % 8) == 0 && op.out != op.in && idx + 1 < n) {
      const Op& b = ops[idx + 1];
      if (b.kind == OpKind::Gemv && b.qbits == 0 && b.bias == nullptr && b.in == op.in &&
          b.in_dim == op.in_dim && b.out != b.in && b.out != op.out) {
        const Op* c = (idx + 2 < n) ? &ops[idx + 2] : nullptr;
        const bool three = c != nullptr && c->kind == OpKind::Gemv && c->qbits == 0 &&
                           c->bias == nullptr && c->in == op.in && c->in_dim == op.in_dim &&
                           c->out != c->in && c->out != op.out && c->out != b.out;
        kernels::launch_rowmajor_half_gemv_cat(HW(op.weight), S(op.out), op.cols, HW(b.weight),
                                               S(b.out), b.cols, three ? HW(c->weight) : nullptr,
                                               three ? S(c->out) : nullptr, three ? c->cols : 0,
                                               S(op.in), op.in_dim, stream_);
        idx += three ? 2 : 1;
        goto op_done;
      }
    }

    static const bool fuse = std::getenv("CPI_PLAN_FUSE") != nullptr;
    if (!seq && fuse) {
      // [RmsNorm(1 x cols, in==out)] [AddInplace(dst += that)] [optional ScaleCopy(dst *= a)]
      if (op.kind == OpKind::RmsNorm && !op.norm_offset && op.rows == 1 && op.in == op.out &&
          op.weight != nullptr && idx + 1 < n && ops[idx + 1].kind == OpKind::AddInplace &&
          ops[idx + 1].in == op.out && ops[idx + 1].out != op.out) {
        const Op& add = ops[idx + 1];
        float alpha = 1.0f;
        std::size_t consumed = 1;
        if (idx + 2 < n && ops[idx + 2].kind == OpKind::ScaleCopy && ops[idx + 2].in == add.out &&
            ops[idx + 2].out == add.out) {
          alpha = ops[idx + 2].scale;
          consumed = 2;
        }
        kernels::launch_rmsnorm_add_scale(S(add.out), S(op.in), HW(op.weight), op.cols,
                                          cfg_.rms_eps, alpha, stream_);
        idx += consumed;
        goto op_done;
      }
      // [RmsNorm(rows=heads, in==out, weighted)] [Rope(same slot, whole-head table rope)]
      if (op.kind == OpKind::RmsNorm && !op.norm_offset && op.rows > 1 && op.in == op.out &&
          op.weight != nullptr && idx + 1 < n && ops[idx + 1].kind == OpKind::Rope &&
          ops[idx + 1].in == op.in && ops[idx + 1].rotary_dim == 0 &&
          ops[idx + 1].heads == op.rows && ops[idx + 1].head_dim == op.cols) {
        const Op& rp = ops[idx + 1];
        const float* cosT = rp.rope_table == RopeTable::Full ? d_cos_full_ : d_cos_sliding_;
        const float* sinT = rp.rope_table == RopeTable::Full ? d_sin_full_ : d_sin_sliding_;
        kernels::launch_rmsnorm_rope(S(op.in), HW(op.weight), op.rows, op.cols, position,
                                     device_pos_mode_ ? d_position_ : nullptr, cosT, sinT,
                                     cfg_.rms_eps, stream_);
        idx += 1;
        goto op_done;
      }
    }

    // Ops marked decode_skip run only in sequence mode (absorbed-MLA decode
    // replaces the materialized attention ops at T==1).
    if (!seq && op.decode_skip) goto op_done;

    // Op-class ablation levers, graph-safe (they change what gets captured). Output is
    // garbage by design; like CPI_CUDA_ABLATE_ATTENTION they price an op class under
    // graph replay, which no profiler can do. Never set outside an experiment.
    {
      static const bool ab_elem = std::getenv("CPI_CUDA_ABLATE_ELEMENTWISE") != nullptr;
      static const bool ab_norm = std::getenv("CPI_CUDA_ABLATE_NORMS") != nullptr;
      if (!seq && ((ab_elem && (op.kind == OpKind::AddInplace || op.kind == OpKind::GeluMul ||
                                op.kind == OpKind::ScaleCopy || op.kind == OpKind::CopySlot)) ||
                   (ab_norm && (op.kind == OpKind::RmsNorm || op.kind == OpKind::Rope ||
                                op.kind == OpKind::KvStore)))) {
        goto op_done;
      }
    }

    switch (op.kind) {
      case OpKind::EmbeddingLookup:
        kernels::launch_embedding_lookup(HW(op.weight), seq ? d_seq_tokens_ : d_tok_, S(op.out), T,
                                         op.cols, stream_);
        break;
      case OpKind::LmHead:
        // In sequence mode only the last token's logits are wanted; running the head
        // over the whole prompt would cost a 262K-wide GEMM per token for nothing.
        if (op.qbits == 4 && d_act_i8_ != nullptr) {
          const __half* head_in =
              S(op.in) + static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(op.in_dim);
          if (actq_src != head_in || actq_len != op.in_dim) {
            kernels::launch_quantize_fp16_to_int8_perm8_g32(head_in, d_act_i8_, d_act_qs_,
                                                            op.in_dim, stream_);
          }
          kernels::launch_weight_only_int4_matvec_grouped_dp4a_f32(
              QW(op.qweight), QS(op.qscales), d_act_i8_, d_act_qs_, d_logits_, op.cols, op.in_dim,
              op.qgroup, stream_);
          break;
        }
        if (op.qbits == 8) {
          kernels::launch_weight_only_int8_gemv_f32(
              QW(op.qweight), QS(op.qscales),
              S(op.in) + static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(op.in_dim),
              d_logits_, op.cols, op.in_dim, stream_);
          break;
        }
        kernels::launch_rowmajor_half_gemv_f32(
            HW(op.weight),
            S(op.in) + static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(op.in_dim),
            d_logits_, op.cols, op.in_dim, stream_);
        break;
      case OpKind::RmsNorm:
        // An XNorm-site norm whose output feeds a dp4a int4 projection fuses the perm8
        // activation quantize into the norm (bit-identical: the fused kernel quantizes the
        // fp16-rounded values); the Gemv branch then hits the actq cache and skips its
        // quantize launch entirely.
        if (!seq && d_act_i8_ != nullptr && !op.norm_offset && op.rows == 1 &&
            op.weight != nullptr && (op.cols % 32) == 0 && op.cols <= 2048 && idx + 1 < n &&
            ops[idx + 1].kind == OpKind::Gemv && ops[idx + 1].qbits == 4 &&
            ops[idx + 1].qgroup > 0 && (ops[idx + 1].qgroup % 32) == 0 &&
            ops[idx + 1].in == op.out && ops[idx + 1].in_dim == op.cols &&
            ops[idx + 1].bias == nullptr) {
          kernels::launch_rmsnorm_quant_perm8(S(op.in), HW(op.weight), S(op.out), d_act_i8_,
                                              d_act_qs_, op.cols, cfg_.rms_eps, stream_);
          actq_src = S(op.out);
          actq_len = op.cols;
          actq_set_now = true;
          break;
        }
        // DeepSeek kv_a_layernorm normalizes only the [kv_lora] latent prefix of the wider
        // MlaCkv=[latent | k_pe] row (per-token stride kva != cols), so the contiguous seq
        // norm would read the wrong slices for t>0. Loop the T==1 norm per token with the true
        // input stride; output MlaLatent is contiguous [T][kv_lora].
        if (seq && op.in == Slot::MlaCkv && op.out == Slot::MlaLatent) {
          // All T tokens in one launch: normalize the [kv_lora] latent prefix (in_stride=kva) into
          // contiguous MlaLatent (out_stride=cols). The per-token loop was ~16.5k host launches.
          const int kva = op.cols + cfg_.qk_rope_head_dim;
          kernels::launch_rmsnorm_seq_strided(S(op.in), HW(op.weight) ? HW(op.weight) : d_ones_,
                                              S(op.out), op.cols, kva, op.cols, T, cfg_.rms_eps,
                                              stream_);
          break;
        }
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
        // Weight encoding and scale granularity were chosen at load; the op just
        // carries them.
        // int4 decode Gemvs route through dp4a with int8 activations: quantize the input
        // vector once per adjacent run, then integer dots. int8 stays on the wide
        // fp16-activation kernel; it is bandwidth-bound already and its dp4a variant
        // measured slower. CPI_CUDA_NO_DP4A=1 keeps the fp16-activation kernels for
        // int4 too.
        if (!seq && d_act_i8_ != nullptr && op.bias == nullptr && op.qbits == 4 && op.qgroup > 0 &&
            (op.qgroup % 32) == 0 && (op.in_dim % 32) == 0) {
          if (actq_src != S(op.in) || actq_len != op.in_dim) {
            kernels::launch_quantize_fp16_to_int8_perm8_g32(S(op.in), d_act_i8_, d_act_qs_,
                                                            op.in_dim, stream_);
            actq_src = S(op.in);
            actq_len = op.in_dim;
          }
          // GLU fusion: [gate Gemv][up Gemv][GeluMul] collapses into one kernel writing
          // the activated product; no Gate/Up round-trip, no elementwise launch.
          if (idx + 2 < n && ops[idx + 1].kind == OpKind::Gemv && ops[idx + 1].qbits == 4 &&
              ops[idx + 1].qgroup == op.qgroup && ops[idx + 1].bias == nullptr &&
              ops[idx + 1].in == op.in && ops[idx + 1].in_dim == op.in_dim &&
              ops[idx + 2].kind == OpKind::GeluMul && ops[idx + 2].aux_ptr == nullptr &&
              ops[idx + 2].aux_offset == 0 && ops[idx + 2].in == op.out &&
              ops[idx + 2].in2 == ops[idx + 1].out && ops[idx + 2].cols == op.cols &&
              ops[idx + 1].cols == op.cols && ops[idx + 2].out != op.in) {
            kernels::launch_weight_only_int4_matvec_grouped_dp4a_glu(
                QW(op.qweight), QS(op.qscales), QW(ops[idx + 1].qweight), QS(ops[idx + 1].qscales),
                d_act_i8_, d_act_qs_, S(ops[idx + 2].out), op.cols, op.in_dim, op.qgroup, stream_,
                tune_gemv_warps_);
            idx += 2;
            break;
          }
          // Adjacent same-input int4 Gemvs (q/k/v, gate/up) run as one cat launch,
          // mirroring the fp16 cat peephole; ~2 fewer medium-gemv launches per site.
          {
            auto same = [&](const Op& o2) {
              return o2.kind == OpKind::Gemv && o2.qbits == 4 && o2.qgroup == op.qgroup &&
                     o2.bias == nullptr && o2.in == op.in && o2.in_dim == op.in_dim &&
                     o2.out != o2.in;
            };
            const Op* b = (idx + 1 < n && same(ops[idx + 1]) && ops[idx + 1].out != op.out)
                              ? &ops[idx + 1]
                              : nullptr;
            const Op* c = (b != nullptr && idx + 2 < n && same(ops[idx + 2]) &&
                           ops[idx + 2].out != op.out && ops[idx + 2].out != b->out)
                              ? &ops[idx + 2]
                              : nullptr;
            if (b != nullptr) {
              kernels::launch_weight_only_int4_matvec_grouped_dp4a_cat(
                  QW(op.qweight), QS(op.qscales), S(op.out), op.cols, QW(b->qweight),
                  QS(b->qscales), S(b->out), b->cols, c ? QW(c->qweight) : nullptr,
                  c ? QS(c->qscales) : nullptr, c ? S(c->out) : nullptr, c ? c->cols : 0, d_act_i8_,
                  d_act_qs_, op.in_dim, op.qgroup, stream_, tune_gemv_warps_);
              idx += c ? 2 : 1;
              break;
            }
          }
          kernels::launch_weight_only_int4_matvec_grouped_dp4a(
              QW(op.qweight), QS(op.qscales), d_act_i8_, d_act_qs_, S(op.out), op.cols, op.in_dim,
              op.qgroup, stream_, tune_gemv_warps_);
          if (S(op.out) == actq_src) actq_src = nullptr;
          break;
        }
        // Short quantized sequences (a speculative verify, a tail chunk) stream the
        // weights once for all tokens through the multi-token dp4a kernel instead of
        // paying a whole-model dequant round.
        if (seq && T <= 16 && op.qbits == 4 && op.qgroup > 0 && (op.qgroup % 32) == 0 &&
            (op.in_dim % 32) == 0 && op.bias == nullptr && d_act_i8_ != nullptr) {
          if (mtq_src != S(op.in) || mtq_len != op.in_dim) {
            kernels::launch_quantize_fp16_to_int8_perm8_g32_mt(S(op.in), d_act_i8_, d_act_qs_,
                                                               op.in_dim, T, stream_);
            mtq_src = S(op.in);
            mtq_len = op.in_dim;
          }
          kernels::launch_weight_only_int4_matvec_grouped_dp4a_mt(
              QW(op.qweight), QS(op.qscales), d_act_i8_, d_act_qs_, S(op.out), op.cols, op.in_dim,
              op.qgroup, T, stream_);
          break;
        }
        // Sequence mode with quantized weights: dequantize into scratch and run the real
        // fp16 GEMM. The quant matvec kernels are batch-1; calling them with T tokens
        // computed only token 0 and filled the KV cache with garbage; the reason the
        // .cpi route historically never enabled sequence prefill.
        // int8-direct sequence GEMM: 8-bit x 8-bit -> int32 via cuBLAS, per-token rowwise
        // activation scales folded in the epilogue. Reads a quarter of the dequant
        // round-trip's bytes. Falls through to dequant+fp16 GEMM when the op has no
        // prefill copy or cuBLAS declines the shape.
        // measured and lost as a default (162-172 ms vs the dequant path's 146-150 on
        // p1200): plain cublasGemmEx int8 does not hit IMMA on this stack; doing this
        // right needs cublasLt ordered layouts. Kept as an opt-in experiment.
        static const bool int8_prefill = std::getenv("CPI_CUDA_INT8_PREFILL") != nullptr;
        if (seq && int8_prefill && op.qbits != 0 && op.pf_qweight != nullptr &&
            d_seq_x8_ != nullptr && (op.in_dim % 4) == 0 && cublas_ != nullptr) {
          if (actq_src != S(op.in) || actq_len != -op.in_dim) {  // negative key: seq form
            kernels::launch_quantize_rowwise_fp16_to_int8(S(op.in), d_seq_x8_, d_seq_xs_, T,
                                                          op.in_dim, stream_);
            actq_src = S(op.in);
            actq_len = -op.in_dim;
          }
          cublasHandle_t ch = static_cast<cublasHandle_t>(cublas_);
          cublasSetStream(ch, stream_);
          const int ialpha = 1, ibeta = 0;
          bool ok = true;
          for (int t0 = 0; t0 < T && ok; t0 += 512) {
            const int chunk = std::min(512, T - t0);
            ok =
                cublasGemmEx(ch, CUBLAS_OP_T, CUBLAS_OP_N, op.cols, chunk, op.in_dim, &ialpha,
                             op.pf_qweight, CUDA_R_8I, op.in_dim,
                             d_seq_x8_ + static_cast<std::size_t>(t0) * op.in_dim, CUDA_R_8I,
                             op.in_dim, &ibeta, d_seq_i32_, CUDA_R_32I, op.cols, CUBLAS_COMPUTE_32I,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
            if (ok) {
              kernels::launch_i32_scale_to_fp16(d_seq_i32_,
                                                static_cast<const float*>(op.pf_qscales), d_seq_xs_,
                                                S(op.out), op.cols, chunk, t0, stream_);
            }
          }
          if (ok) break;
          actq_src = nullptr;  // fall through to the dequant path below
        }
        if (seq && op.qbits != 0 && d_seq_dequant_ != nullptr) {
          if (op.qbits == 4 && op.qgroup > 0) {
            kernels::launch_dequant_int4_grouped(QW(op.qweight), QS(op.qscales), d_seq_dequant_,
                                                 op.cols, op.in_dim, op.qgroup, stream_);
          } else if (op.qbits == 8 && op.qgroup == 0) {
            kernels::launch_dequant_int8_rowwise(QW(op.qweight), QS(op.qscales), d_seq_dequant_,
                                                 op.cols, op.in_dim, stream_);
          } else {
            throw std::runtime_error("sequence prefill: unsupported quant layout");
          }
          if (!seq_gemm_cublas(d_seq_dequant_, S(op.in), S(op.out), op.cols, op.in_dim, T)) {
            kernels::launch_rowmajor_half_gemm_f16(
                d_seq_dequant_, S(op.in), S(op.out), op.cols, op.in_dim, T, stream_, op.clip_in_min,
                op.clip_in_max, op.clip_out_min, op.clip_out_max);
          }
          break;
        }
        // int8 GLU fusion: [gate Gemv][up Gemv][GeluMul] -> one paired-warp launch.
        if (!seq && op.qbits == 8 && op.qgroup == 0 && op.bias == nullptr &&
            (op.in_dim % 16) == 0 && idx + 2 < n && ops[idx + 1].kind == OpKind::Gemv &&
            ops[idx + 1].qbits == 8 && ops[idx + 1].qgroup == 0 && ops[idx + 1].bias == nullptr &&
            ops[idx + 1].in == op.in && ops[idx + 1].in_dim == op.in_dim &&
            ops[idx + 1].cols == op.cols && ops[idx + 2].kind == OpKind::GeluMul &&
            ops[idx + 2].aux_ptr == nullptr && ops[idx + 2].aux_offset == 0 &&
            ops[idx + 2].in == op.out && ops[idx + 2].in2 == ops[idx + 1].out &&
            ops[idx + 2].cols == op.cols && ops[idx + 2].out != op.in) {
          kernels::launch_weight_only_int8_matvec_glu(
              QW(op.qweight), QS(op.qscales), QW(ops[idx + 1].qweight), QS(ops[idx + 1].qscales),
              S(op.in), S(ops[idx + 2].out), op.cols, op.in_dim, stream_);
          idx += 2;
          break;
        }
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
          const bool clipped = std::isfinite(op.clip_in_min) || std::isfinite(op.clip_in_max) ||
                               std::isfinite(op.clip_out_min) || std::isfinite(op.clip_out_max);
          if (clipped ||
              !seq_gemm_cublas(HW(op.weight), S(op.in), S(op.out), op.cols, op.in_dim, T)) {
            kernels::launch_rowmajor_half_gemm_f16(
                HW(op.weight), S(op.in), S(op.out), op.cols, op.in_dim, T, stream_, op.clip_in_min,
                op.clip_in_max, op.clip_out_min, op.clip_out_max);
          }
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
        // Writes the selected experts to device buffers; the expert ops below read
        // them there, so a token never round-trips to the host mid-layer.
        if (seq) {
          // All T tokens in one launch (was a per-token loop = the top prefill cost).
          kernels::launch_moe_router_topk_softmax_seq(S(op.in), op.cols, op.heads, d_moe_idx_seq_,
                                                      d_moe_w_seq_, T, stream_, HW(op.weight),
                                                      op.moe_renorm);
        } else {
          kernels::launch_moe_router_topk_softmax(S(op.in), op.cols, op.heads, d_moe_idx_, d_moe_w_,
                                                  stream_, HW(op.weight), op.moe_renorm);
        }
        break;
      case OpKind::MoeGateUpGeglu:
        if (seq && d_moe_gather_ != nullptr && moe_grouped_gate_up(op, S(op.in), T)) {
          break;  // grouped-GEMM path handled gate/up + SiLU (inter stays grouped for down)
        }
        if (seq) {
          // Sequence prefill: loop the (verified) T==1 scalar kernel per token, reading token t's
          // routing slot and writing its [top_k*MI] inter block. Experts stay matvec; the win is that
          // the projections/attention around the MoE batch as GEMMs.
          for (int t = 0; t < T; ++t)
            kernels::launch_moe_gate_up_geglu(
                op.qbits ? static_cast<const void*>(op.qweight) : static_cast<const void*>(op.weight),
                QS(op.qscales), op.qbits, op.qgroup,
                S(op.in) + static_cast<std::size_t>(t) * op.in_dim,
                d_moe_idx_seq_ + static_cast<std::size_t>(t) * op.heads,
                S(op.out) + static_cast<std::size_t>(t) * op.heads * op.cols, op.cols, op.in_dim,
                op.heads, stream_, op.mlp_gelu);
        } else if (moe_stream_ && op.qbits) {
          // gate_up is [E*2*MI, H]: rows-per-expert = 2*cols, in_features = in_dim (=H).
          stage_moe_experts(op.qweight, QS(op.qscales), 2 * op.cols, op.in_dim, op.qgroup,
                            d_moe_stage_gu_, d_moe_stage_gu_s_);
          kernels::launch_moe_gate_up_geglu(d_moe_stage_gu_, d_moe_stage_gu_s_, op.qbits, op.qgroup,
                                            S(op.in), d_moe_identity_idx_, S(op.out), op.cols,
                                            op.in_dim, op.heads, stream_, op.mlp_gelu);
        } else if (!seq && op.qbits == 4 && op.qgroup > 0 && (op.qgroup % 32) == 0 &&
                   (op.in_dim % 32) == 0 && d_act_i8_ != nullptr) {
          // dp4a: quantise the (shared) activation once, then int8xint4 dot per expert.
          kernels::launch_quantize_fp16_to_int8_perm8_g32(S(op.in), d_act_i8_, d_act_qs_, op.in_dim,
                                                          stream_);
          kernels::launch_moe_gate_up_geglu_dp4a(op.qweight, QS(op.qscales), d_act_i8_, d_act_qs_,
                                                 d_moe_idx_, S(op.out), op.cols, op.in_dim, op.heads,
                                                 op.qgroup, stream_, op.mlp_gelu);
        } else {
          kernels::launch_moe_gate_up_geglu(
              op.qbits ? static_cast<const void*>(op.qweight) : static_cast<const void*>(op.weight),
              QS(op.qscales), op.qbits, op.qgroup, S(op.in), d_moe_idx_, S(op.out), op.cols,
              op.in_dim, op.heads, stream_, op.mlp_gelu);
        }
        break;
      case OpKind::MoeDownAccum:
        if (seq && moe_grouped_down(op, S(op.out), T)) {
          break;  // grouped-GEMM path handled down + weighted scatter into MoeOut
        }
        if (seq) {
          // Sequence prefill: loop the T==1 scalar down/accum per token. Reads token t's [top_k*MI]
          // inter block + its routing slot, accumulates into token t's output row.
          for (int t = 0; t < T; ++t)
            kernels::launch_moe_down_accum(
                op.qbits ? static_cast<const void*>(op.qweight) : static_cast<const void*>(op.weight),
                QS(op.qscales), op.qbits, op.qgroup,
                S(op.in) + static_cast<std::size_t>(t) * op.heads * op.in_dim,
                d_moe_idx_seq_ + static_cast<std::size_t>(t) * op.heads,
                d_moe_w_seq_ + static_cast<std::size_t>(t) * op.heads,
                S(op.out) + static_cast<std::size_t>(t) * op.cols, op.cols, op.in_dim, op.heads,
                stream_);
        } else if (moe_stream_ && op.qbits) {
          // down is [E*H, MI]: rows-per-expert = cols (=H), in_features = in_dim (=MI). The routing
          // weights (d_moe_w_) are indexed by selection position k, so they stay as-is.
          stage_moe_experts(op.qweight, QS(op.qscales), op.cols, op.in_dim, op.qgroup,
                            d_moe_stage_dn_, d_moe_stage_dn_s_);
          kernels::launch_moe_down_accum(d_moe_stage_dn_, d_moe_stage_dn_s_, op.qbits, op.qgroup,
                                         S(op.in), d_moe_identity_idx_, d_moe_w_, S(op.out), op.cols,
                                         op.in_dim, op.heads, stream_);
        } else if (!seq && op.qbits == 4 && op.qgroup > 0 && (op.qgroup % 32) == 0 &&
                   (op.in_dim % 32) == 0 && d_act_i8_ != nullptr) {
          // dp4a: quantise the top_k per-expert inter vectors (op.heads rows x op.in_dim), then dot.
          kernels::launch_quantize_fp16_to_int8_perm8_g32_mt(S(op.in), d_act_i8_, d_act_qs_,
                                                             op.in_dim, op.heads, stream_);
          kernels::launch_moe_down_accum_dp4a(op.qweight, QS(op.qscales), d_act_i8_, d_act_qs_,
                                              d_moe_idx_, d_moe_w_, S(op.out), op.cols, op.in_dim,
                                              op.heads, op.qgroup, stream_);
        } else {
          kernels::launch_moe_down_accum(
              op.qbits ? static_cast<const void*>(op.qweight) : static_cast<const void*>(op.weight),
              QS(op.qscales), op.qbits, op.qgroup, S(op.in), d_moe_idx_, d_moe_w_, S(op.out), op.cols,
              op.in_dim, op.heads, stream_);
        }
        break;
      case OpKind::MlaLatStore: {
        const int* dpos =
            (device_pos_mode_ || spec_seq_device_pos_) ? d_position_ : nullptr;
        if (!seq && op.aux_ptr != nullptr) {
          // Latent-only decode: assemble was skipped, so rope q_pe/k_pe here
          // and write the latent row straight from the kv_a outputs.
          engine::launch_mla_rope_q_store_lat(
              S(Slot::Q), S(Slot::MlaCkv), S(Slot::MlaLatent), caches_lat_[layer],
              static_cast<const float*>(op.aux_ptr), op.heads, op.key_head_dim, op.rotary_dim,
              op.head_dim, op.in_dim, position, dpos, max_ctx_, op.scale, stream_);
          break;
        }
        // Append [latent | roped k_pe] rows to the latent cache (k_pe gathered
        // from the assembled K); seq prefill, spec seq, and dual-write decode.
        const int rows = seq ? T : 1;
        engine::launch_mla_lat_store(S(Slot::MlaLatent), S(Slot::K), caches_lat_[layer],
                                     op.in_dim, op.key_head_dim, op.rotary_dim, op.head_dim,
                                     op.heads, rows, position, dpos, max_ctx_, stream_);
        break;
      }
      case OpKind::MlaQAbsorb: {
        if (seq) break;  // decode-only
        engine::launch_mla_q_absorb(S(Slot::Q), static_cast<const __half*>(op.weight),
                                    S(Slot::MlaQAbs), op.heads, op.key_head_dim, op.rotary_dim,
                                    op.head_dim, op.in_dim, stream_);
        break;
      }
      case OpKind::MlaAbsorbedAttn: {
        if (seq) break;  // decode-only
        const int kdim = op.in_dim + op.rotary_dim;
        engine::launch_mla_absorbed_attention(
            S(Slot::MlaQAbs), caches_lat_[layer], static_cast<const __half*>(op.weight),
            S(Slot::MlaAttLat), S(Slot::Att), op.heads, kdim, op.in_dim, op.value_head_dim,
            op.head_dim, op.scale, position + 1, device_pos_mode_ ? d_position_ : nullptr,
            d_mla_abs_m_, d_mla_abs_l_, d_mla_abs_o_, mla_abs_chunks_, device_pos_mode_, stream_);
        break;
      }
      case OpKind::MlaVDecompress: {
        break;  // folded into MlaAbsorbedAttn's fused reduce+decompress
      }
      case OpKind::MlaAssembleRope: {
        // DeepSeek MLA: assemble per-head K/V from the kv_b output + shared k_pe, rope q_pe/k_pe.
        // Runs at T==1 (MoE forces token-by-token, so no sequence path). heads=nh,
        // key_head_dim=qk_nope, rotary_dim=qk_rope, value_head_dim=v_head, in_dim=kv_lora.
        if (seq) {
          engine::launch_mla_assemble_rope_seq(
              S(Slot::Q), S(Slot::MlaKvb), S(Slot::MlaCkv), S(Slot::K), S(Slot::V),
              static_cast<const float*>(op.aux_ptr), op.heads, op.key_head_dim, op.rotary_dim,
              op.value_head_dim, op.in_dim, T, position, op.scale, stream_);
        } else {
          engine::launch_mla_assemble_rope(
              S(Slot::Q), S(Slot::MlaKvb), S(Slot::MlaCkv), S(Slot::K), S(Slot::V),
              static_cast<const float*>(op.aux_ptr), op.heads, op.key_head_dim, op.rotary_dim,
              op.value_head_dim, op.in_dim, position, device_pos_mode_ ? d_position_ : nullptr,
              op.scale, stream_);
        }
        break;
      }
      case OpKind::Rope: {
        const float* cosT = op.rope_table == RopeTable::Full ? d_cos_full_ : d_cos_sliding_;
        const float* sinT = op.rope_table == RopeTable::Full ? d_sin_full_ : d_sin_sliding_;
        if (op.rotary_dim > 0) {
          // Partial RoPE over Q (in) and K (in2) together. The device-position twin is what
          // makes partial-RoPE plans graph-capturable; it was the one missing kernel that
          // kept Gemma 4 decode launching ~700 kernels per token from the host.
          if (device_pos_mode_) {
            kernels::launch_rope_inplace_partial_table_device_pos(
                S(op.in), S(op.in2), op.heads, op.kv_heads, op.head_dim, op.rotary_dim, d_position_,
                cosT, sinT, stream_);
          } else {
            kernels::launch_rope_inplace_partial_table(S(op.in), S(op.in2), op.heads, op.kv_heads,
                                                       op.head_dim, op.rotary_dim, position, cosT,
                                                       sinT, stream_);
          }
        } else if (seq) {
          // Sequence prefill: token t sits at position (chunk start + t). The spec-verify
          // graph reads the base position from the device instead.
          if (spec_seq_device_pos_) {
            kernels::launch_rope_seq_table_device_pos(S(op.in), op.heads, op.head_dim, d_position_,
                                                      T, cosT, sinT, 0, stream_);
          } else {
            kernels::launch_rope_seq_table(S(op.in), op.heads, op.head_dim, position, T, cosT, sinT,
                                           0, stream_);
          }
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
        } else if (spec_seq_device_pos_) {
          // Graph-capturable seq append: T rows land at device base position + row (the
          // host-offset memcpys below would bake one position into the graph).
          kernels::launch_store_kv_seq_device_pos(S(Slot::K), S(Slot::V), kc, vc, d_position_,
                                                  static_cast<int>(kvdim), T, max_ctx_, stream_);
        } else {
          // Sequence prefill appends T rows at once: the K/V slots are [T][kvdim] and the
          // cache rows are contiguous, so it is the same copy, T times as long.
          const std::size_t rows = static_cast<std::size_t>(T);
          static const bool kv_debug =
              std::getenv("CPI_KV_DEBUG") != nullptr;  // cached, not per-append
          if (kv_debug)
            std::fprintf(stderr, "[kvdbg] layer=%d T=%d pos=%d kvdim=%zu kc=%p vc=%p K=%p V=%p\n",
                         layer, T, position, kvdim, (void*)kc, (void*)vc, (void*)S(Slot::K),
                         (void*)S(Slot::V));
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
          // text prefill: K/V were appended to this layer's cache by KvStore, so attend
          // over the cache (it also holds any earlier context). Full-attention layers with
          // no image limits take the tensor-core cuBLAS path; sliding layers pass their
          // window to the masked tiled kernel, and ctx.limits (when set) makes an image
          // span bidirectional.
          const int window = op.full_attention ? 0 : op.sliding_window;
          if (spec_seq_device_pos_) {
            // Spec-verify graph: multi-query split-K with per-token causal length derived
            // from the device position. Fixed max-context chunk grid; per-chunk guards
            // skip dead chunks, so one capture serves every depth.
            kernels::launch_attention_split_any_mt_device_pos(
                S(op.in), caches_k_[layer], caches_v_[layer], S(op.out), d_position_, T, window,
                op.heads, op.kv_heads, op.head_dim, d_spec_split_m_, d_spec_split_l_,
                d_spec_split_o_, 32, spec_attn_chunks_, stream_);
            break;
          }
          if (ctx.limits == nullptr &&
              prefill_attention_tc(S(op.in), caches_k_[layer], caches_v_[layer], S(op.out), T,
                                   position, op.heads, op.kv_heads, op.head_dim, window)) {
            break;
          }
          kernels::launch_attention_prefill(S(op.in), caches_k_[layer], caches_v_[layer], S(op.out),
                                            T, position, op.heads, op.kv_heads, op.head_dim,
                                            stream_,
                                            /*causal=*/true, ctx.limits, window);
          break;
        }
        if (seq) {
          // Cacheless sequence (a vision tower): the K/V slots are the "cache", and
          // causal comes from the context (false for a bidirectional encoder).
          kernels::launch_attention_prefill(S(op.in), S(Slot::K), S(Slot::V), S(op.out), T, 0,
                                            op.heads, op.kv_heads, op.head_dim, stream_,
                                            ctx.causal);
          break;
        }
        __half* kc = caches_k_[layer];
        __half* vc = caches_v_[layer];
        const std::size_t kvdim = static_cast<std::size_t>(op.kv_heads) * op.head_dim;
        {
          // CPI_CUDA_ABLATE_ATTENTION=1 skips every decode attention dispatch. The output is
          // garbage; that is the point: it prices attention under the graph by removing it,
          // which no profiler distortion can confuse. Never set outside an experiment.
          static const bool ablate_attn = std::getenv("CPI_CUDA_ABLATE_ATTENTION") != nullptr;
          if (ablate_attn && !seq) break;
          const int window = op.full_attention ? 0 : op.sliding_window;
          // Wide heads (Gemma's 256/512) take the any-head_dim split-K path when its scratch
          // exists: the tiled single-block-per-head kernel leaves a many-SM GPU running only 8
          // blocks, and the whole windowed KV walk serialises inside them. The split's grid is
          // (heads x chunks) with per-chunk seq/window guards, graph-safe, and eager and
          // graph share the same kernels so the two stay bit-identical.
          if (d_split_o_ != nullptr && op.head_dim > 128) {
            // Grid sizing: the depth bucket (set by forward_one; falls back to max)
            // and a sliding layer never needs more chunks than its window can span.
            int chunks = attn_chunk_budget_ > 0 ? attn_chunk_budget_ : split_any_chunks_;
            // Depth-adaptive chunk size: fine chunks maximize parallelism shallow, but
            // the fp32 (m,l,o) scratch written+re-read per chunk scales with chunk
            // count and dominates at depth. Deep buckets coarsen the chunks; the merge
            // math per key is unchanged.
            int cs = kSplitAnyChunk;
            {
              const int live_tokens = chunks * kSplitAnyChunk;
              // Shallow/deep chunk sizes and the crossover are per-box tuning knobs.
              cs = live_tokens >= tune_attn_deep_tokens_ ? tune_attn_cs_deep_
                                                         : tune_attn_cs_shallow_;
              if (cs != kSplitAnyChunk) chunks = (live_tokens + cs - 1) / cs;
            }
            if (window > 0) {
              const int wc = window / cs + 2;  // window can straddle chunk edges
              if (wc < chunks) chunks = wc;
            }
            if (device_pos_mode_) {
              kernels::launch_attention_split_any_device_pos(
                  S(op.in), kc, vc, S(op.out), d_position_, window, op.heads, op.kv_heads,
                  op.head_dim, d_split_m_, d_split_l_, d_split_o_, cs, chunks, stream_);
            } else {
              kernels::launch_attention_split_any(S(op.in), kc, vc, S(op.out), position + 1, window,
                                                  op.heads, op.kv_heads, op.head_dim, d_split_m_,
                                                  d_split_l_, d_split_o_, cs, chunks, stream_);
            }
            break;
          }
          if (device_pos_mode_) {
            // Device-position kernel: seq derived from d_position_ on-device (graph-
            // safe). Sliding layers pass their window so k_start is computed on device.
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
                                           vc + static_cast<std::size_t>(k_start) * kvdim,
                                           S(op.out), att_seq, op.heads, op.kv_heads, op.head_dim,
                                           stream_);
          }
        }
        break;
      }
      case OpKind::GeluMul: {
        const __half* b = HW(op.aux_ptr) ? HW(op.aux_ptr) : (S(op.in2) + op.aux_offset);
        if (seq && op.in2 == Slot::PleAll) {
          // b is a window of a wider per-token tensor, so it strides differently.
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
  op_done:
    if (!actq_set_now && op.kind != OpKind::Gemv) {
      actq_src = nullptr;
      mtq_src = nullptr;
    }
    actq_set_now = false;
    if (prof) {
      cudaEventRecord(ev1, stream_);
      cudaEventSynchronize(ev1);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev0, ev1);
      static const char* kProfNames[] = {
          "RmsNorm",   "Gemv",       "Rope",        "ScaleCopy",  "CopySlot",    "KvStore",
          "Attention", "GeluMul",    "AddInplace",  "AddRmsNorm", "Embedding",   "LmHead",
          "SiluMul",   "SplitHeads", "SigmoidGate", "LinConv1d",  "RepeatHeads", "LinAttnStep",
          "MulVec",    "MoeRouter",  "MoeGateUp",   "MoeDown",    "PatchEmbed",  "Rope2D",
          "AvgPool",   "Standardize"};
      const int ki = static_cast<int>(op.kind);
      std::string key = (ki >= 0 && ki < 26) ? kProfNames[ki] : "Unknown";
      // Split the two op kinds that hide very different shapes under one name.
      if (op.kind == OpKind::Gemv) {
        key += "_" + std::to_string(op.cols) + "x" + std::to_string(op.in_dim);
      } else if (op.kind == OpKind::RmsNorm) {
        key += "_" + std::to_string(op.rows) + "x" + std::to_string(op.cols);
      }
      prof_bins[key].ms += ms;
      prof_bins[key].count += 1;
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }
  }
}

float* PlanCudaEngine::lin_conv_state(int layer) {
  return d_lin_conv_state_ + static_cast<std::size_t>(layer) * lin_conv_state_stride_;
}

float* PlanCudaEngine::lin_recurrent_state(int layer) {
  return d_lin_recurrent_state_ + static_cast<std::size_t>(layer) * lin_recurrent_state_stride_;
}

// Qwen3.5 as a plan: hybrid full-attention + gated delta-net. Mirrors the fork's
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
        o.pf_qweight = q->second.pf_i8;
        o.pf_qscales = q->second.pf_scales;
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

    // ── MLP (SwiGLU: SiLU, not GeLU) ──
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
  // Debug: CPI_DS_DUMP=<file> dumps Slot::X (post-layer hidden) at position CPI_DS_DUMP_POS to compare
  // against the HF golden trace, layer by layer (localises the first divergent op).
  static const char* dsdump = std::getenv("CPI_DS_DUMP");
  if (dsdump) {
    static const int dpos = std::getenv("CPI_DS_DUMP_POS") ? std::atoi(std::getenv("CPI_DS_DUMP_POS")) : 0;
    if (position == dpos) {
      G4_CHECK(cudaStreamSynchronize(stream_));
      std::vector<__half> h(cfg_.hidden);
      G4_CHECK(cudaMemcpy(h.data(), slot_ptr_[static_cast<int>(opplan::Slot::X)],
                          cfg_.hidden * sizeof(__half), cudaMemcpyDeviceToHost));
      std::vector<float> f(cfg_.hidden);
      for (int i = 0; i < cfg_.hidden; ++i) f[i] = __half2float(h[i]);
      std::FILE* fp = std::fopen(dsdump, layer == 0 ? "wb" : "ab");
      std::fwrite(f.data(), sizeof(float), cfg_.hidden, fp);
      std::fclose(fp);
    }
  }
}

// Process one token at `position`, reusing the KV cache from earlier positions.
// Positions must be presented in increasing order (prefill then decode) so the
// cache and the KV-share sources are populated before dependent layers read them.
// Sequence-mode buffers: the same slots as decode, but [max_tokens] deep.
// Can this plan run a sequence prefill? Decided by the ops it emits, not by a model
// name: partial RoPE (Qwen) and the delta-net ops have no sequence kernels yet.
bool PlanCudaEngine::plan_can_sequence_prefill() const {
  using namespace opplan;
  for (const auto& lp : plan_.layers) {
    for (const Op& o : lp.ops) {
      if (o.kind == OpKind::Rope && o.rotary_dim > 0) return false;
      if (o.kind == OpKind::LinearAttentionStep || o.kind == OpKind::LinearConv1d) return false;
      // MoE routing/expert kernels are single-token. DeepSeek's seq path loops the T==1 kernels
      // per token (d_moe_idx_seq_ holds one [top_k] slot per token), so its MoE plans can
      // sequence-prefill; other MoE families (Gemma) have no seq loop wired, so they keep the
      // token-by-token prefill.
      if ((o.kind == OpKind::MoeRouterTopk || o.kind == OpKind::MoeGateUpGeglu ||
           o.kind == OpKind::MoeDownAccum) &&
          cfg_.family != Family::DeepSeekV2)
        return false;
    }
  }
  return !plan_.layers.empty();
}

void PlanCudaEngine::allocate_sequence_buffers(int max_tokens) {
  if (cfg_.family == Family::DeepSeekV2) {
    const int H = cfg_.hidden, nh = cfg_.num_heads, qkhd = cfg_.head_dim;
    const int kvw = nh * qkhd;
    const int E = cfg_.num_experts, K = cfg_.top_k_experts, MI = cfg_.moe_intermediate_size;
    const int maxinter = std::max(cfg_.intermediate, MI * std::max(1, cfg_.n_shared_experts));
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
    al(Slot::Q, kvw);
    al(Slot::K, kvw);
    al(Slot::V, kvw);
    al(Slot::Att, kvw);
    al(Slot::Gate, maxinter);
    al(Slot::Up, maxinter);
    al(Slot::Inter, maxinter);
    al(Slot::MlaCkv, cfg_.kv_lora_rank + cfg_.qk_rope_head_dim);
    al(Slot::MlaLatent, cfg_.kv_lora_rank);
    al(Slot::MlaKvb, static_cast<std::size_t>(nh) * (cfg_.qk_nope_head_dim + cfg_.v_head_dim));
    al(Slot::MoeRouterIn, H);
    al(Slot::MoeLogits, E);
    al(Slot::MoeInter, static_cast<std::size_t>(K) * MI);
    al(Slot::MoeOut, H);
    G4_CHECK(cudaMalloc(&d_moe_idx_seq_, static_cast<std::size_t>(max_tokens) * K * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_moe_w_seq_, static_cast<std::size_t>(max_tokens) * K * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_seq_tokens_, static_cast<std::size_t>(max_tokens) * sizeof(int)));
    G4_CHECK(cudaMalloc(&d_seq_limits_, static_cast<std::size_t>(max_tokens) * sizeof(int)));
    // Grouped-GEMM MoE scratch (default on; CPI_DS_NO_MOE_GROUPED=1 opts out to the per-token loop).
    // ~370 MB at max_tokens=4096 (gather/outg [P,H] + gate_g/up_g [P,MI] + out_f32 [T,H]).
    if (std::getenv("CPI_DS_NO_MOE_GROUPED") == nullptr) {
      const std::size_t P = static_cast<std::size_t>(max_tokens) * K;
      G4_CHECK(cudaMalloc(&d_moe_gather_, P * H * sizeof(__half)));
      G4_CHECK(cudaMalloc(&d_moe_gate_g_, P * MI * sizeof(__half)));
      G4_CHECK(cudaMalloc(&d_moe_up_g_, P * MI * sizeof(__half)));
      G4_CHECK(cudaMalloc(&d_moe_outg_, P * H * sizeof(__half)));
      // All experts dequanted at once (one launch) so the grouped GEMM reads contiguous fp16.
      // gate_up is the larger of the two: [E*2*MI, H]; down [E*H, MI] reuses it.
      G4_CHECK(cudaMalloc(&d_moe_dqw_, static_cast<std::size_t>(E) * 2 * MI * H * sizeof(__half)));
      G4_CHECK(cudaMalloc(&d_gg_A_, static_cast<std::size_t>(2) * E * sizeof(void*)));
      G4_CHECK(cudaMalloc(&d_gg_B_, static_cast<std::size_t>(2) * E * sizeof(void*)));
      G4_CHECK(cudaMalloc(&d_gg_C_, static_cast<std::size_t>(2) * E * sizeof(void*)));
      G4_CHECK(cudaMalloc(&d_moe_perm_, P * sizeof(int)));
      G4_CHECK(cudaMalloc(&d_moe_rweight_, P * sizeof(float)));
      G4_CHECK(cudaMalloc(&d_moe_out_f32_, static_cast<std::size_t>(max_tokens) * H * sizeof(float)));
      // int4-direct path (short-prompt): int8 activations [P,H] + per-32-group scales [P,H/32].
      // H is the wider of {H (gate_up act), MI (down act)}, so one buffer serves both.
      G4_CHECK(cudaMalloc(&d_moe_actq_, P * H));
      G4_CHECK(cudaMalloc(&d_moe_acts_, P * (H / 32) * sizeof(float)));
      G4_CHECK(cudaMalloc(&d_moe_off_dev_, static_cast<std::size_t>(E + 1) * sizeof(int)));
      h_moe_off_.assign(E + 1, 0);
    }
    return;
  }

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

// Prefill a whole prompt in one pass instead of token by token.
//
// Two things this buys, beyond speed:
//   - image soft tokens can be spliced over the embeddings of the image placeholder
//     tokens (they are used as-is: unlike text embeddings they get no sqrt(hidden) scale);
//   - a per-token key limit makes an image span bidirectional while the surrounding text
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

  // ...splice the image soft tokens over the placeholder tokens' embeddings (used as-is:
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

  // ...and only then the rest of the prologue, which reads X (the per-layer-input
  // projection does, and HF projects it from the post-splice embeddings).
  execute_ops(plan_.prologue.data() + ready, plan_.prologue.size() - ready, 0, start_position, ctx);

  for (int L = 0; L < cfg_.num_layers; ++L)
    execute_ops(plan_.layers[L].ops.data(), plan_.layers[L].ops.size(), L, start_position, ctx);
  execute_ops(plan_.epilogue.data(), plan_.epilogue.size(), 0, start_position, ctx);
  G4_CHECK(cudaStreamSynchronize(stream_));
  publish_host_logits();  // logits of the last prompt token, ready for the first sample
}

void PlanCudaEngine::forward_one(int token, int position, bool compute_logits,
                                 std::vector<float>* per_layer_rms) {
  const int H = cfg_.hidden;

  // Fast path: every logits step is the same single-token forward, so replay a
  // captured CUDA graph (device-position ops) instead of re-launching ~600 kernels.
  // Skipped for the per-layer-rms parity dump (it syncs mid-forward). The graph
  // writes the KV cache at d_position_, so it interleaves correctly with the eager
  // prefill (which fills earlier rows via host position).
  // Persistent decode: the whole token as one cooperative launch over the compiled plan.
  // Falls back to the graph path if the launch is ever rejected (co-residency).
  if (compute_logits && per_layer_rms == nullptr && persist_enabled_) {
    G4_CHECK(cudaMemcpyAsync(d_tok_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
    G4_CHECK(cudaMemcpyAsync(d_position_, &position, sizeof(int), cudaMemcpyHostToDevice, stream_));
    if (kernels::launch_persistent_decode(static_cast<const kernels::PersistOp*>(d_persist_ops_),
                                          persist_n_ops_, d_tok_, d_position_, persist_blocks_,
                                          stream_)) {
      if (!defer_host_logits_) publish_host_logits();
      return;
    }
    persist_enabled_ = false;
  }

  if (compute_logits && per_layer_rms == nullptr && decode_graph_enabled_) {
    // The split-attention grid is frozen at capture, so a graph captured for max_ctx
    // launches (heads x max_chunks) blocks per layer forever; at depth 400 with a 4096
    // context that is ~90% dead blocks, and it measured as most of attention's 1.3 ms.
    // Capture instead for a power-of-two depth bucket (>= 512 tokens) and re-capture when
    // generation crosses into the next bucket: a handful of ~ms captures per generation
    // buys grids at most 2x the live depth. Math is unchanged; dead chunks never
    // contribute; so graph and eager stay bit-identical.
    if (split_any_chunks_ > 0) {
      int need_chunks = (position + 1 + kSplitAnyChunk - 1) / kSplitAnyChunk;
      int bucket = 512 / kSplitAnyChunk;
      while (bucket < need_chunks) bucket *= 2;
      if (bucket > split_any_chunks_) bucket = split_any_chunks_;
      if (decode_graph_ready_ && bucket != graph_bucket_chunks_) decode_graph_ready_ = false;
      attn_chunk_budget_ = bucket;
    }
    if (!decode_graph_ready_) {
      capture_decode_graph();
      graph_bucket_chunks_ = attn_chunk_budget_;
    }
    G4_CHECK(cudaMemcpyAsync(d_tok_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
    G4_CHECK(cudaMemcpyAsync(d_position_, &position, sizeof(int), cudaMemcpyHostToDevice, stream_));
    G4_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
    // Shared-loop path leaves logits on-device for sample() (greedy argmax / lazy
    // host copy); other callers get host logits now.
    if (!defer_host_logits_) publish_host_logits();
    return;
  }

  // Publish the token; the prologue's EmbeddingLookup ops read d_tok_ on-device.
  // Eager decode sizes attention grids exactly (no bucket needed without a capture).
  attn_chunk_budget_ = (position + 1 + kSplitAnyChunk - 1) / kSplitAnyChunk;
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
    // Two-phase argmax: without partition scratch the single-block fallback walks the
    // whole vocab on one SM every greedy token. The partitioned kernel exists for
    // exactly this; it just needs the scratch passed.
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_), cfg_.vocab, d_argmax_,
                                 stream_, d_argmax_pv_, d_argmax_pi_, argmax_parts_);
    G4_CHECK(cudaStreamSynchronize(stream_));
    int next = 0;
    G4_CHECK(cudaMemcpy(&next, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
    return next;
  }

  // Device top-k: select the candidate set on the GPU so the host never touches the
  // full vocab (temperature>0 is the real chat path; greedy only covers temp<=0).
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
    // Threshold = the k-th largest raw logit. The final softcap is monotonic, so
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
      // vector, which is what makes a seeded run byte-identical across both paths.
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

// Compile the (already-data) decode plan into the persistent interpreter's device
// descriptors. Returns false, leaving the graph path in charge, when the plan uses any
// kind the interpreter does not implement (MoE, delta-net, quantized projections, biases,
// rotary_dim ropes). Every pointer is resolved here, once, exactly as build_plan resolved
// them for the eager executor.
bool PlanCudaEngine::compile_persistent_plan() {
  using namespace opplan;
  std::vector<kernels::PersistOp> pl;
  auto S = [&](Slot s) -> __half* { return const_cast<__half*>(slot_ptr_[static_cast<int>(s)]); };
  auto HW = [](const void* w) { return static_cast<const __half*>(w); };

  auto add_ops = [&](const std::vector<Op>& ops, int layer) -> bool {
    for (const Op& op : ops) {
      kernels::PersistOp d;
      switch (op.kind) {
        case OpKind::EmbeddingLookup:
          d.kind = kernels::PersistOp::kEmbed;
          d.weight = HW(op.weight);
          d.out = S(op.out);
          d.cols = op.cols;
          break;
        case OpKind::ScaleCopy:
          d.kind = kernels::PersistOp::kScale;
          d.in = S(op.in);
          d.out = S(op.out);
          d.cols = op.cols;
          d.scale = op.scale;
          break;
        case OpKind::CopySlot:
          d.kind = kernels::PersistOp::kCopy;
          d.in = S(op.in);
          d.out = S(op.out);
          d.cols = op.cols;
          break;
        case OpKind::AddInplace:
          d.kind = kernels::PersistOp::kAdd;
          d.in = S(op.in);
          d.out = S(op.out);
          d.cols = op.cols;
          break;
        case OpKind::GeluMul:
          d.kind = kernels::PersistOp::kGeluMul;
          d.in = S(op.in);
          d.in2 = op.aux_ptr != nullptr ? HW(op.aux_ptr) : S(op.in2);
          d.aux_offset = op.aux_ptr != nullptr ? 0 : op.aux_offset;
          d.out = S(op.out);
          d.cols = op.cols;
          break;
        case OpKind::RmsNorm:
          if (op.norm_offset) return false;
          d.kind = kernels::PersistOp::kRmsNorm;
          d.in = S(op.in);
          d.out = S(op.out);
          d.weight = HW(op.weight);
          d.rows = op.rows;
          d.cols = op.cols;
          d.eps = cfg_.rms_eps;
          break;
        case OpKind::Rope:
          if (op.rotary_dim > 0) return false;
          d.kind = kernels::PersistOp::kRope;
          d.out = S(op.in);  // in place
          d.heads = op.heads;
          d.head_dim = op.head_dim;
          d.cosT = op.rope_table == RopeTable::Full ? d_cos_full_ : d_cos_sliding_;
          d.sinT = op.rope_table == RopeTable::Full ? d_sin_full_ : d_sin_sliding_;
          break;
        case OpKind::Gemv:
          if (op.qbits != 0 || op.bias != nullptr || (op.in_dim % 8) != 0) return false;
          d.kind = kernels::PersistOp::kGemv;
          d.weight = HW(op.weight);
          d.in = S(op.in);
          d.out = S(op.out);
          d.rows = op.cols;
          d.in_dim = op.in_dim;
          break;
        case OpKind::LmHead:
          if (op.qbits != 0 || (op.in_dim % 8) != 0) return false;
          d.kind = kernels::PersistOp::kLmHead;
          d.weight = HW(op.weight);
          d.in = S(op.in);
          d.fout = d_logits_;
          d.rows = op.cols;
          d.in_dim = op.in_dim;
          break;
        case OpKind::KvStore:
          d.kind = kernels::PersistOp::kKvStore;
          d.in = S(Slot::K);
          d.in2 = S(Slot::V);
          d.cols = op.cols;
          d.kcache = caches_k_[layer];
          d.vcache = caches_v_[layer];
          break;
        case OpKind::Attention: {
          if (d_split_o_ == nullptr || op.head_dim > 512 || (op.head_dim % 2) != 0) return false;
          const int window = op.full_attention ? 0 : op.sliding_window;
          kernels::PersistOp st;
          st.kind = kernels::PersistOp::kAttnStats;
          st.in = S(op.in);
          st.heads = op.heads;
          st.kv_heads = op.kv_heads;
          st.head_dim = op.head_dim;
          st.window = window;
          st.chunk_size = kSplitAnyChunk;
          st.chunks = split_any_chunks_;
          st.sm = d_split_m_;
          st.sl = d_split_l_;
          st.so = d_split_o_;
          st.kcache = caches_k_[layer];
          st.vcache = caches_v_[layer];
          pl.push_back(st);
          d.kind = kernels::PersistOp::kAttnReduce;
          d.out = S(op.out);
          d.heads = op.heads;
          d.head_dim = op.head_dim;
          d.window = window;
          d.chunk_size = kSplitAnyChunk;
          d.chunks = split_any_chunks_;
          d.sm = d_split_m_;
          d.sl = d_split_l_;
          d.so = d_split_o_;
          break;
        }
        default:
          return false;
      }
      pl.push_back(d);
    }
    return true;
  };

  if (!add_ops(plan_.prologue, -1)) return false;
  for (int L = 0; L < cfg_.num_layers; ++L) {
    if (!add_ops(plan_.layers[L].ops, L)) return false;
  }
  if (!add_ops(plan_.epilogue, -1)) return false;

  // Elide the grid.sync between adjacent pure-elementwise ops of equal length: block b's
  // writes are exactly the elements block b reads next (same grid stride), so no cross-block
  // data moves. A windowed GeluMul reading the previous op's output is the one same-length
  // case where indices shift; keep its sync.
  auto elementwise = [](int k) {
    return k == kernels::PersistOp::kEmbed || k == kernels::PersistOp::kScale ||
           k == kernels::PersistOp::kCopy || k == kernels::PersistOp::kAdd ||
           k == kernels::PersistOp::kGeluMul;
  };
  for (std::size_t i = 0; i + 1 < pl.size(); ++i) {
    kernels::PersistOp& a = pl[i];
    const kernels::PersistOp& b = pl[i + 1];
    if (!elementwise(a.kind) || !elementwise(b.kind) || a.cols != b.cols) continue;
    if (b.kind == kernels::PersistOp::kGeluMul && b.aux_offset != 0 &&
        b.in2 == static_cast<const half*>(a.out)) {
      continue;
    }
    a.no_sync = 1;
  }

  persist_blocks_ = kernels::persistent_decode_max_blocks();
  if (persist_blocks_ <= 0) return false;
  persist_n_ops_ = static_cast<int>(pl.size());
  G4_CHECK(cudaMalloc(&d_persist_ops_, pl.size() * sizeof(kernels::PersistOp)));
  G4_CHECK(cudaMemcpy(d_persist_ops_, pl.data(), pl.size() * sizeof(kernels::PersistOp),
                      cudaMemcpyHostToDevice));
  return true;
}

void PlanCudaEngine::capture_decode_graph() {
  // Capture the single-token forward (device-position ops) once; replayed for
  // every logits step. Buffers referenced (d_tok_/d_position_/caches/weights/
  // d_logits_) are stable for the engine's lifetime, so one capture serves all
  // positions and all generate() calls. Sync first so no prefill work is pending.
  G4_CHECK(cudaStreamSynchronize(stream_));
  // Depth-bucket re-capture: drop the previous instantiation before capturing anew.
  if (decode_graph_exec_ != nullptr) {
    cudaGraphExecDestroy(decode_graph_exec_);
    decode_graph_exec_ = nullptr;
  }
  if (decode_graph_ != nullptr) {
    cudaGraphDestroy(decode_graph_);
    decode_graph_ = nullptr;
  }
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

// ── Per-box autotuning ──────────────────────────────────────────────────────────────
// The objective is the real workload: ms per decode-graph replay at a representative
// depth, not a kernel microbench (microbenches repeatedly ranked candidates wrong during
// the parity campaign; the graph replay is what generation actually runs).

double PlanCudaEngine::time_decode_at(int pos, int iters) {
  using clock = std::chrono::steady_clock;
  attn_chunk_budget_ = (pos + 1 + kSplitAnyChunk - 1) / kSplitAnyChunk;
  decode_graph_ready_ = false;  // knobs changed between calls; force a fresh capture
  const int tok = cfg_.bos_token_id;
  G4_CHECK(cudaMemcpyAsync(d_tok_, &tok, sizeof(int), cudaMemcpyHostToDevice, stream_));
  G4_CHECK(cudaMemcpyAsync(d_position_, &pos, sizeof(int), cudaMemcpyHostToDevice, stream_));
  capture_decode_graph();
  for (int i = 0; i < 5; ++i) G4_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
  G4_CHECK(cudaStreamSynchronize(stream_));
  const auto t0 = clock::now();
  for (int i = 0; i < iters; ++i) G4_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
  G4_CHECK(cudaStreamSynchronize(stream_));
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;
}

std::string PlanCudaEngine::tuning_path() const {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::string gpu = prop.name;
  for (char& ch : gpu) {
    if (!std::isalnum(static_cast<unsigned char>(ch))) ch = '_';
  }
  const char* base = std::getenv("LOCALAPPDATA");
  if (base == nullptr) base = std::getenv("HOME");
  if (base == nullptr) base = ".";
  return std::string(base) + "/cpi/tuning/" + gpu + "-L" + std::to_string(cfg_.num_layers) + "h" +
         std::to_string(cfg_.hidden) + "q" + std::to_string(weight_quant_bits_) + "g" +
         std::to_string(weight_quant_group_) + ".json";
}

// The file is written by save_tuning below, so a full JSON parser is overkill: pull each
// "key": <int> pair by name.
static int tuning_field(const std::string& s, const char* key, int fallback) {
  const std::string pat = std::string("\"") + key + "\":";
  const std::size_t at = s.find(pat);
  if (at == std::string::npos) return fallback;
  return std::atoi(s.c_str() + at + pat.size());
}

bool PlanCudaEngine::load_tuning() {
  std::ifstream in(tuning_path());
  if (!in.good()) return false;
  std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const int gw = tuning_field(s, "gemv_warps", tune_gemv_warps_);
  if (gw == 2 || gw == 4 || gw == 8) tune_gemv_warps_ = gw;
  const int csw = tuning_field(s, "attn_cs_shallow", tune_attn_cs_shallow_);
  if (csw == 16 || csw == 32) tune_attn_cs_shallow_ = csw;
  const int csd = tuning_field(s, "attn_cs_deep", tune_attn_cs_deep_);
  if (csd == 16 || csd == 32 || csd == 64) tune_attn_cs_deep_ = csd;
  const int th = tuning_field(s, "attn_deep_tokens", tune_attn_deep_tokens_);
  if (th >= 256 && th <= 65536) tune_attn_deep_tokens_ = th;
  std::fprintf(stderr, "[tune] loaded %s: gemv_warps=%d attn_cs=%d/%d@%d\n", tuning_path().c_str(),
               tune_gemv_warps_, tune_attn_cs_shallow_, tune_attn_cs_deep_, tune_attn_deep_tokens_);
  return true;
}

void PlanCudaEngine::save_tuning() const {
  const std::string path = tuning_path();
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  std::ofstream out(path, std::ios::trunc);
  if (!out.good()) {
    std::fprintf(stderr, "[tune] could not write %s\n", path.c_str());
    return;
  }
  out << "{\"gemv_warps\":" << tune_gemv_warps_ << ",\"attn_cs_shallow\":" << tune_attn_cs_shallow_
      << ",\"attn_cs_deep\":" << tune_attn_cs_deep_
      << ",\"attn_deep_tokens\":" << tune_attn_deep_tokens_ << "}\n";
  std::fprintf(stderr, "[tune] saved %s\n", path.c_str());
}

void PlanCudaEngine::autotune() {
  if (!decode_graph_enabled_ || d_tok_ == nullptr) return;
  const int iters = 40;
  const int shallow_pos = std::min(400, max_ctx_ - 2);
  const int deep_pos = std::min(3072, max_ctx_ - 2);
  struct Knob {
    const char* name;
    int* value;
    std::vector<int> candidates;
    int depth;
  };
  // Coordinate descent, one knob at a time, defaults kept unless a candidate wins by >1%
  // (jitter hysteresis: thermal drift is 3-5%, so each comparison re-times
  // the incumbent in the same pass instead of trusting an earlier number).
  std::vector<Knob> knobs = {
      {"gemv_warps", &tune_gemv_warps_, {2, 4, 8}, shallow_pos},
      {"attn_cs_shallow", &tune_attn_cs_shallow_, {16, 32}, shallow_pos},
      {"attn_cs_deep", &tune_attn_cs_deep_, {32, 64}, deep_pos},
  };
  std::fprintf(stderr, "[tune] autotuning (depths %d/%d, %d replays per candidate)\n", shallow_pos,
               deep_pos, iters);
  for (auto& k : knobs) {
    // Deep knobs are pointless when the context cannot reach the crossover.
    if (k.depth + 1 < tune_attn_deep_tokens_ && k.value == &tune_attn_cs_deep_) {
      continue;
    }
    // Throwaway run first: the very first capture+replay at a depth lands on cold clocks
    // and unramped power state, and a cold incumbent loses to any warm candidate. (This
    // fired on the first bring-up: warps=4 "lost" 28% purely to clock ramp.)
    time_decode_at(k.depth, 10);
    int best = *k.value;
    double best_ms = time_decode_at(k.depth, iters);
    const double incumbent_ms = best_ms;
    for (int cand : k.candidates) {
      if (cand == best) continue;
      *k.value = cand;
      const double ms = time_decode_at(k.depth, iters);
      if (ms < best_ms * 0.98) {  // 2% hysteresis toward the incumbent; drift is real
        best = cand;
        best_ms = ms;
      }
    }
    *k.value = best;
    std::fprintf(stderr, "[tune]   %s=%d (%.3f ms/tok; incumbent was %.3f)\n", k.name, best,
                 best_ms, incumbent_ms);
  }
  decode_graph_ready_ = false;  // generation re-captures at its own depth bucket
  attn_chunk_budget_ = 0;
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
  // Weight-only quantization (--weight-quant int8|int4). Must be set before open():
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
  // 32 bits per group; 4.25 bits/weight instead of 4. int8's 255 levels do not
  // need it, so it stays per-row and byte-identical to before.
  // Override with CPI_PLAN_QUANT_GROUP (0 = per-row, else a power of two).
  if (weight_quant_bits_ == 4) {
    weight_quant_group_ = 128;
  }
  if (const char* g = std::getenv("CPI_PLAN_QUANT_GROUP")) {
    weight_quant_group_ = std::atoi(g);
  }
  open(options.model_path, options.max_context > 0 ? options.max_context : 4096);
  // Per-box tuning: CPI_AUTOTUNE=1 searches on this GPU and persists the winners; every
  // other run silently applies the saved file for this (GPU, model, quant) if one exists.
  if (std::getenv("CPI_AUTOTUNE") != nullptr) {
    autotune();
    save_tuning();
  } else {
    load_tuning();
  }
}

// The verify-batch forward at fixed T=16 with device-position seq kernels: reads tokens
// from d_seq_tokens_ and the base position from d_position_, leaves the 16 per-position
// argmaxes in d_spec_argmax_. This is exactly what the spec graph captures; it also runs
// eagerly (CPI_CUDA_SPEC_GRAPH=0) so graph and eager replay the same kernel stream and
// must agree bitwise. Host `position` args are dead in this mode.
void PlanCudaEngine::spec_forward_t16() {
  using namespace opplan;
  constexpr int T = 16;
  ExecCtx ctx{sslot_ptr_.data(), T, /*causal=*/true, /*cached=*/true, nullptr};
  spec_seq_device_pos_ = true;
  execute_ops(plan_.prologue.data(), plan_.prologue.size(), 0, 0, ctx);
  for (int L = 0; L < cfg_.num_layers; ++L)
    execute_ops(plan_.layers[L].ops.data(), plan_.layers[L].ops.size(), L, 0, ctx);
  const Op& head = plan_.epilogue.back();
  for (std::size_t i = 0; i + 1 < plan_.epilogue.size(); ++i) {
    execute_ops(&plan_.epilogue[i], 1, 0, 0, ctx);
  }
  const __half* xn = sslot_ptr_[static_cast<int>(head.in)];
  kernels::launch_quantize_fp16_to_int8_perm8_g32_mt(xn, d_act_i8_, d_act_qs_, head.in_dim, T,
                                                     stream_);
  kernels::launch_weight_only_int4_matvec_grouped_dp4a_mt_f32(
      static_cast<const std::int8_t*>(head.qweight), static_cast<const float*>(head.qscales),
      d_act_i8_, d_act_qs_, d_mt_logits_, head.cols, head.in_dim, head.qgroup, T, stream_);
  for (int t = 0; t < T; ++t) {
    kernels::launch_argmax_float(d_mt_logits_ + static_cast<std::size_t>(t) * head.cols, head.cols,
                                 d_spec_argmax_ + t, stream_, d_argmax_pv_, d_argmax_pi_,
                                 argmax_parts_);
  }
  spec_seq_device_pos_ = false;
}

// One-time scratch + capture for the graphed verify. Returns true when the spec kernel
// path is usable (graphed or eager); a failure marks the path broken and the caller
// stays on the cuBLAS eager verify forever.
bool PlanCudaEngine::spec_graph_init() {
  if (spec_graph_broken_) return false;
  if (d_spec_argmax_ != nullptr) return true;  // already initialized
  try {
    const opplan::Op& head = plan_.epilogue.back();
    if (d_mt_logits_ == nullptr) {
      G4_CHECK(
          cudaMalloc(&d_mt_logits_, 16ull * static_cast<std::size_t>(head.cols) * sizeof(float)));
    }
    G4_CHECK(cudaMalloc(&d_spec_argmax_, 16 * sizeof(int)));
    int max_heads = 0, max_hd = 0;
    for (const auto& lp : plan_.layers) {
      for (const auto& o : lp.ops) {
        if (o.kind == opplan::OpKind::Attention) {
          max_heads = std::max(max_heads, o.heads);
          max_hd = std::max(max_hd, o.head_dim);
        }
      }
    }
    if (max_heads <= 0 || max_hd <= 0) throw std::runtime_error("no attention ops");
    spec_attn_chunks_ = (max_ctx_ + 31) / 32;
    const std::size_t cells =
        16ull * static_cast<std::size_t>(max_heads) * static_cast<std::size_t>(spec_attn_chunks_);
    G4_CHECK(cudaMalloc(&d_spec_split_m_, cells * sizeof(float)));
    G4_CHECK(cudaMalloc(&d_spec_split_l_, cells * sizeof(float)));
    G4_CHECK(
        cudaMalloc(&d_spec_split_o_, cells * static_cast<std::size_t>(max_hd) * sizeof(float)));
    static const bool graph_off = [] {
      const char* e = std::getenv("CPI_CUDA_SPEC_GRAPH");
      return e && *e == '0';
    }();
    if (!graph_off) {
      G4_CHECK(cudaStreamSynchronize(stream_));
      G4_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      spec_forward_t16();
      G4_CHECK(cudaStreamEndCapture(stream_, &spec_graph_));
      G4_CHECK(cudaGraphInstantiate(&spec_graph_exec_, spec_graph_, nullptr, nullptr, 0));
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[spec] graph init failed (%s); staying on the eager verify\n", e.what());
    spec_seq_device_pos_ = false;
    if (spec_graph_exec_) cudaGraphExecDestroy(spec_graph_exec_), spec_graph_exec_ = nullptr;
    if (spec_graph_) cudaGraphDestroy(spec_graph_), spec_graph_ = nullptr;
    spec_graph_broken_ = true;
    return false;
  }
}

// Greedy speculative verify: forward `k` tokens (the pending token plus k-1 drafts) at
// positions pos.., then compute each position's greedy argmax. Returns the argmaxes; the
// caller accepts the longest draft prefix they confirm. KV rows for rejected positions go
// stale in place; attention never reads past the current position, and later steps
// overwrite them.
//
// Fast path: the fixed-T=16 forward captured by spec_graph_init replays as one graph
// launch (drafts padded by repeating the last token; padded rows write KV that later
// steps overwrite before ever reading). Falls back to the eager seq forward when the
// batch would cross max_context or the plan is not an int4 one.
void PlanCudaEngine::verify_greedy(const int* tokens, int k, int pos, int* out_argmax) {
  using namespace opplan;
  {
    const Op& h = plan_.epilogue.back();
    if (h.qbits == 4 && d_act_i8_ != nullptr && pos + 16 <= max_ctx_ && spec_graph_init()) {
      int padded[16];
      for (int i = 0; i < 16; ++i) padded[i] = tokens[i < k ? i : k - 1];
      G4_CHECK(cudaMemcpyAsync(d_seq_tokens_, padded, 16 * sizeof(int), cudaMemcpyHostToDevice,
                               stream_));
      G4_CHECK(cudaMemcpyAsync(d_position_, &pos, sizeof(int), cudaMemcpyHostToDevice, stream_));
      if (spec_graph_exec_) {
        G4_CHECK(cudaGraphLaunch(spec_graph_exec_, stream_));
      } else {
        spec_forward_t16();  // CPI_CUDA_SPEC_GRAPH=0: same kernel stream, launched eagerly
      }
      int tmp[16];
      G4_CHECK(
          cudaMemcpyAsync(tmp, d_spec_argmax_, 16 * sizeof(int), cudaMemcpyDeviceToHost, stream_));
      G4_CHECK(cudaStreamSynchronize(stream_));
      for (int i = 0; i < k; ++i) out_argmax[i] = tmp[i];
      return;
    }
  }
  G4_CHECK(cudaMemcpyAsync(d_seq_tokens_, tokens, static_cast<std::size_t>(k) * sizeof(int),
                           cudaMemcpyHostToDevice, stream_));
  ExecCtx ctx{sslot_ptr_.data(), k, /*causal=*/true, /*cached=*/true, nullptr};
  execute_ops(plan_.prologue.data(), plan_.prologue.size(), 0, pos, ctx);
  for (int L = 0; L < cfg_.num_layers; ++L)
    execute_ops(plan_.layers[L].ops.data(), plan_.layers[L].ops.size(), L, pos, ctx);
  // Epilogue norm over all k rows; then the head + argmax per position (the seq LmHead
  // case computes only the last row).
  const Op& head = plan_.epilogue.back();
  for (std::size_t i = 0; i + 1 < plan_.epilogue.size(); ++i) {
    execute_ops(&plan_.epilogue[i], 1, 0, pos, ctx);
  }
  const __half* xn = sslot_ptr_[static_cast<int>(head.in)];
  if (head.qbits == 4 && d_act_i8_ != nullptr) {
    // Batched head: one weight pass produces all k positions' logits (the per-position
    // gemv re-read the 262K-row head k times and dominated the verify's GPU time).
    if (d_mt_logits_ == nullptr) {
      G4_CHECK(
          cudaMalloc(&d_mt_logits_, 16ull * static_cast<std::size_t>(head.cols) * sizeof(float)));
    }
    kernels::launch_quantize_fp16_to_int8_perm8_g32_mt(xn, d_act_i8_, d_act_qs_, head.in_dim, k,
                                                       stream_);
    kernels::launch_weight_only_int4_matvec_grouped_dp4a_mt_f32(
        static_cast<const std::int8_t*>(head.qweight), static_cast<const float*>(head.qscales),
        d_act_i8_, d_act_qs_, d_mt_logits_, head.cols, head.in_dim, head.qgroup, k, stream_);
    for (int t = 0; t < k; ++t) {
      kernels::launch_argmax_float(d_mt_logits_ + static_cast<std::size_t>(t) * head.cols,
                                   head.cols, d_argmax_, stream_, d_argmax_pv_, d_argmax_pi_,
                                   argmax_parts_);
      G4_CHECK(
          cudaMemcpyAsync(out_argmax + t, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    }
    G4_CHECK(cudaStreamSynchronize(stream_));
    return;
  }
  for (int t = 0; t < k; ++t) {
    const __half* row = xn + static_cast<std::size_t>(t) * head.in_dim;
    if (head.qbits == 8) {
      kernels::launch_weight_only_int8_gemv_f32(static_cast<const std::int8_t*>(head.qweight),
                                                static_cast<const float*>(head.qscales), row,
                                                d_logits_, head.cols, head.in_dim, stream_);
    } else {
      kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(head.weight), row,
                                             d_logits_, head.cols, head.in_dim, stream_);
    }
    kernels::launch_argmax_float(d_logits_, head.cols, d_argmax_, stream_, d_argmax_pv_,
                                 d_argmax_pi_, argmax_parts_);
    G4_CHECK(
        cudaMemcpyAsync(out_argmax + t, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
  }
  G4_CHECK(cudaStreamSynchronize(stream_));
}

// Prompt-lookup draft: find the most recent earlier occurrence of the last `ng` tokens
// and propose the tokens that followed it.
static int lookup_draft(const std::vector<int>& hist, int ng, int k, int* out) {
  const int n = static_cast<int>(hist.size());
  if (n < ng + 1) return 0;
  for (int start = n - ng - 1; start >= 0; --start) {
    bool match = true;
    for (int j = 0; j < ng; ++j) {
      if (hist[start + j] != hist[n - ng + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      int c = 0;
      for (int j = start + ng; j < n && c < k; ++j) out[c++] = hist[j];
      return c;
    }
  }
  return 0;
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
  // prompt_tokens.size(), and only when the result is longer than the prompt, so
  // returning generated-only silently produced an empty final "Decoded text".
  // It went unnoticed because the streamed on_token text still looked correct.
  p.include_prompt = !prompt.empty();
  // sample() pulls logits off the device lazily (greedy argmax / host copy), so
  // step() need not copy 262K floats to host every token.
  defer_host_logits_ = true;

  // Speculative greedy decode (CPI_CUDA_SPEC=<k>: up to k prompt-lookup drafts per
  // batched verify, batch = drafts+1 <= 16). Greedy-only and sampler-neutral (no
  // penalties, no grammar).
  //
  // Known-non-lossless (2026-08): the intended invariant, output identical to plain greedy, does
  // not hold. The batched verify uses the seq/mt kernels while the nd==0 fallback uses the decode
  // kernels; the two round differently on near-ties, so the emitted stream diverges from plain greedy
  // and depends on the draft policy (3-gram vs 6-gram give different output). It can cascade into
  // repetition loops. Making it lossless requires reconciling the verify and decode kernel numerics
  // (prefill-parity-class work); see memory:cpi-cuda-spec-decode. Until then this is experimental.
  const char* spec_env = std::getenv("CPI_CUDA_SPEC");
  const int spec_k = spec_env ? std::min(std::atoi(spec_env), 15) : 0;
  if (spec_k >= 1) {
    static bool warned = false;
    if (!warned) {
      std::fprintf(stderr,
                   "[warn] CPI_CUDA_SPEC is EXPERIMENTAL and NON-LOSSLESS: the batched verify and "
                   "plain decode use different kernels, so the output differs from greedy decoding "
                   "(and depends on the draft policy). Do not use it for reference/production "
                   "output. See memory:cpi-cuda-spec-decode.\n");
      warned = true;
    }
  }
  if (spec_k >= 1 && temperature <= 0.0f && seq_prefill_ok_ && !prompt.empty() &&
      p.repetition_penalty == 1.0f && p.no_repeat_ngram_size == 0) {
    using clock = std::chrono::steady_clock;
    const auto msf = [](clock::time_point a, clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const int P = static_cast<int>(prompt.size());
    if (P >= max_context())
      throw std::runtime_error("prompt (" + std::to_string(P) +
                               " tokens) does not fit the context (" +
                               std::to_string(max_context()) + "); raise --max-context");
    reset_state();
    stats_ = BenchmarkStats{};
    stats_.prompt_tokens = P;
    const auto t0 = clock::now();
    prefill(prompt);
    synchronize();
    stats_.prefill_ms = msf(t0, clock::now());

    std::vector<int> history = prompt;
    std::vector<int> out;
    const int max_ctx = max_context();
    int verify_calls = 0, drafted = 0, accepted_total = 0;
    const auto d0 = clock::now();
    int pos = P;
    bool stop = false;
    // First token from the prefill logits, via the same device-argmax sample().
    int cur = sample(p, history);
    out.push_back(cur);
    history.push_back(cur);
    stats_.generated_tokens++;
    if (on_token && !on_token(cur)) stop = true;
    if (!stop && is_stop(cur)) stop = true;
    // Invariant at loop top: `cur` is emitted but not yet forwarded; `pos` is its slot.
    while (!stop && stats_.generated_tokens < p.max_new_tokens && pos < max_ctx) {
      int drafts[16];
      const int room = max_ctx - pos - 1;  // verify writes KV rows pos..pos+nd
      const int nd = room >= 1 ? lookup_draft(history, 3, std::min(spec_k, room), drafts) : 0;
      if (nd <= 0) {
        step(cur, pos, /*want_logits=*/true);
        ++pos;
        cur = sample(p, history);
      } else {
        int batch[17], verdict[17];
        batch[0] = cur;
        for (int i = 0; i < nd; ++i) batch[1 + i] = drafts[i];
        verify_greedy(batch, nd + 1, pos, verdict);
        ++verify_calls;
        drafted += nd;
        int acc = 0;
        while (acc < nd && verdict[acc] == drafts[acc]) ++acc;
        accepted_total += acc;
        // KV rows past pos+acc are stale in place; the next forward overwrites them.
        pos += acc + 1;
        for (int i = 0; i < acc && !stop && stats_.generated_tokens < p.max_new_tokens; ++i) {
          out.push_back(drafts[i]);
          history.push_back(drafts[i]);
          stats_.generated_tokens++;
          if (on_token && !on_token(drafts[i])) stop = true;
          if (!stop && is_stop(drafts[i])) stop = true;
        }
        if (stop || stats_.generated_tokens >= p.max_new_tokens) break;
        cur = verdict[acc];  // bonus token: computed by the verify, not yet forwarded
      }
      out.push_back(cur);
      history.push_back(cur);
      stats_.generated_tokens++;
      if (on_token && !on_token(cur)) break;
      if (is_stop(cur)) break;
    }
    synchronize();
    stats_.decode_ms = msf(d0, clock::now());
    if (std::getenv("CPI_CUDA_SPEC_STATS")) {
      std::fprintf(stderr, "[spec] verifies=%d drafted=%d accepted=%d (%.1f%%) tokens=%d\n",
                   verify_calls, drafted, accepted_total,
                   drafted > 0 ? 100.0 * accepted_total / drafted : 0.0,
                   static_cast<int>(stats_.generated_tokens));
    }
    if (p.include_prompt) {
      std::vector<int> full = prompt;
      full.insert(full.end(), out.begin(), out.end());
      return full;
    }
    return out;
  }

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
