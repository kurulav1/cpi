#include "engine/llama4_cuda_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"
namespace engine {
namespace {
constexpr float kPi = 3.14159265358979323846f;
float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
  float out = 0.0f;
  std::memcpy(&out, &word, sizeof(out));
  return out;
}
std::uint16_t float_to_half_bits(float value) {
  const __half h = __float2half(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}
float half_bits_to_float(std::uint16_t bits) {
  __half h{};
  std::memcpy(&h, &bits, sizeof(bits));
  return __half2float(h);
}
}  // namespace
void Llama4CudaEngine::destroy() {
  auto free_device = [](void*& ptr) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  };
  auto free_int_device = [](int*& ptr) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  };

  for (auto& layer : layer_device_) {
    free_device(layer.norm_att);
    free_device(layer.norm_ffn);
    free_device(layer.wq);
    free_device(layer.wk);
    free_device(layer.wv);
    free_device(layer.wo);
    free_device(layer.q_norm);
    free_device(layer.k_norm);
  }
  layer_device_.clear();
  layer_host_moe_.clear();
  for (auto& sh : layer_device_shared_) {
    free_device(sh.sh_gate);
    free_device(sh.sh_up);
    free_device(sh.sh_down);
  }
  layer_device_shared_.clear();

  free_device(d_norm_out_);
  free_device(d_lm_head_);
  free_device(d_q_norm_unit_);
  free_device(d_k_norm_unit_);
  free_device(d_rope_cos_);
  free_device(d_rope_sin_);
  free_device(d_x_);
  free_device(d_x_norm_);
  free_device(d_q_);
  free_device(d_k_);
  free_device(d_v_);
  free_device(d_att_);
  free_device(d_tmp_hidden_);
  free_device(d_ff13_);
  free_device(d_ff_inter_);
  free_device(d_shared_gate_out_);
  free_device(d_shared_up_out_);
  free_device(d_logits_);
  free_device(d_k_cache_);
  free_device(d_v_cache_);
  free_device(d_expert_gate_up_w_);
  free_device(d_expert_down_w_);
  free_device(d_streamed_rowmajor_w_);

  if (d_bf16_stage_) {
    cudaFree(d_bf16_stage_);
    d_bf16_stage_ = nullptr;
  }
  bf16_stage_elems_ = 0;

  free_int_device(d_argmax_);

  if (cublas_) {
    cublasDestroy(cublas_);
    cublas_ = nullptr;
  }
  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
    compute_stream_ = nullptr;
  }

  h_tok_embeddings_bf16_ = nullptr;
  h_token_embedding_fp16_.clear();
  h_x_norm_fp16_.clear();
  h_x_norm_f32_.clear();
  h_logits_.clear();
}

void Llama4CudaEngine::allocate_runtime_buffers() {
  auto malloc_device = [](void** ptr, std::size_t bytes) {
    CUDA_CHECK(cudaMalloc(ptr, bytes));
  };

  const std::size_t half_bytes_hidden =
      static_cast<std::size_t>(hidden_) * sizeof(__half);
  const std::size_t half_bytes_kv =
      static_cast<std::size_t>(kv_dim_) * sizeof(__half);
  const std::size_t half_bytes_inter_full =
      static_cast<std::size_t>(inter_full_) * sizeof(__half);
  const std::size_t half_bytes_inter_shared =
      static_cast<std::size_t>(inter_shared_) * sizeof(__half);
  const std::size_t half_bytes_inter_expert =
      static_cast<std::size_t>(inter_expert_) * sizeof(__half);
  const std::size_t cache_bytes =
      static_cast<std::size_t>(n_layers_) *
      static_cast<std::size_t>(max_ctx_) *
      static_cast<std::size_t>(kv_dim_) *
      sizeof(__half);
  const std::size_t rope_bytes =
      static_cast<std::size_t>(max_ctx_) *
      static_cast<std::size_t>(head_dim_ / 2) *
      sizeof(float);
  const std::size_t expert_gate_up_bytes =
      static_cast<std::size_t>(hidden_) *
      static_cast<std::size_t>(inter_full_) *
      sizeof(__half);
  const std::size_t expert_down_bytes =
      static_cast<std::size_t>(inter_expert_) *
      static_cast<std::size_t>(hidden_) *
      sizeof(__half);
  const std::size_t streamed_rowmajor_bytes =
      std::max(static_cast<std::size_t>(inter_shared_) *
                   static_cast<std::size_t>(hidden_),
               static_cast<std::size_t>(hidden_) *
                   static_cast<std::size_t>(inter_shared_)) *
      sizeof(__half);

  malloc_device(&d_rope_cos_, rope_bytes);
  malloc_device(&d_rope_sin_, rope_bytes);
  malloc_device(&d_x_, half_bytes_hidden);
  malloc_device(&d_x_norm_, half_bytes_hidden);
  malloc_device(&d_q_, half_bytes_hidden);
  malloc_device(&d_k_, half_bytes_kv);
  malloc_device(&d_v_, half_bytes_kv);
  malloc_device(&d_att_, half_bytes_hidden);
  malloc_device(&d_tmp_hidden_, half_bytes_hidden);
  malloc_device(&d_ff13_, half_bytes_inter_full);
  malloc_device(&d_ff_inter_,
                std::max(half_bytes_inter_expert, half_bytes_inter_shared));
  malloc_device(&d_shared_gate_out_, half_bytes_inter_shared);
  malloc_device(&d_shared_up_out_, half_bytes_inter_shared);
  malloc_device(&d_logits_,
                static_cast<std::size_t>(vocab_size_) * sizeof(float));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_argmax_), sizeof(int)));
  malloc_device(&d_k_cache_, cache_bytes);
  malloc_device(&d_v_cache_, cache_bytes);
  malloc_device(&d_expert_gate_up_w_, expert_gate_up_bytes);
  malloc_device(&d_expert_down_w_, expert_down_bytes);
  malloc_device(&d_streamed_rowmajor_w_, streamed_rowmajor_bytes);

  // Size the staging buffer to the largest single weight matrix we'll stream:
  // gate_up per expert is [hidden, 2*inter] = hidden_ * inter_full_ elements.
  // A single-chunk DMA eliminates serialized round-trips from small chunking.
  bf16_stage_elems_ = static_cast<std::size_t>(hidden_) *
                      static_cast<std::size_t>(inter_full_);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bf16_stage_),
                        bf16_stage_elems_ * sizeof(std::uint16_t)));

  h_token_embedding_fp16_.resize(static_cast<std::size_t>(hidden_));
  h_x_norm_fp16_.resize(static_cast<std::size_t>(hidden_));
  h_x_norm_f32_.resize(static_cast<std::size_t>(hidden_));
  h_logits_.resize(static_cast<std::size_t>(vocab_size_));
}

void Llama4CudaEngine::copy_bf16_tensor_to_fp16_device(const std::uint16_t* src,
                                                       void* dst,
                                                       std::size_t elems) {
  if (src == nullptr || dst == nullptr || elems == 0) {
    return;
  }
  if (d_bf16_stage_ == nullptr || bf16_stage_elems_ == 0) {
    LLAMA_ENGINE_THROW("BF16 staging buffer is not initialized");
  }

  std::size_t offset = 0;
  auto* dst_half = reinterpret_cast<__half*>(dst);
  while (offset < elems) {
    const std::size_t chunk = std::min(bf16_stage_elems_, elems - offset);
    CUDA_CHECK(cudaMemcpyAsync(d_bf16_stage_,
                               src + offset,
                               chunk * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice,
                               compute_stream_));
    kernels::launch_convert_bf16_to_fp16(d_bf16_stage_,
                                         dst_half + offset,
                                         static_cast<int>(chunk),
                                         compute_stream_);
    offset += chunk;
  }
}


void Llama4CudaEngine::load_resident_weights() {
  auto load_tensor = [&](const std::string& name,
                         std::size_t expected_elems = 0) -> const std::uint16_t* {
    const std::size_t bytes = weights_.tensor_bytes(name);
    if (expected_elems > 0 &&
        bytes != expected_elems * sizeof(std::uint16_t)) {
      LLAMA_ENGINE_THROW("unexpected tensor size for " + name);
    }
    return reinterpret_cast<const std::uint16_t*>(weights_.tensor_ptr(name));
  };
  auto alloc_and_copy = [&](void** dst,
                            const std::uint16_t* src,
                            std::size_t elems) {
    CUDA_CHECK(cudaMalloc(dst, elems * sizeof(__half)));
    copy_bf16_tensor_to_fp16_device(src, *dst, elems);
  };

  h_tok_embeddings_bf16_ = load_tensor(
      "language_model.model.embed_tokens.weight",
      static_cast<std::size_t>(vocab_size_) * static_cast<std::size_t>(hidden_));

  alloc_and_copy(&d_norm_out_,
                 load_tensor("language_model.model.norm.weight",
                             static_cast<std::size_t>(hidden_)),
                 static_cast<std::size_t>(hidden_));

  const char* lm_head_name = weights_.has_tensor("language_model.lm_head.weight")
                                 ? "language_model.lm_head.weight"
                                 : "language_model.model.embed_tokens.weight";
  alloc_and_copy(&d_lm_head_,
                 load_tensor(lm_head_name,
                             static_cast<std::size_t>(vocab_size_) *
                                 static_cast<std::size_t>(hidden_)),
                 static_cast<std::size_t>(vocab_size_) *
                     static_cast<std::size_t>(hidden_));

  std::vector<std::uint16_t> unit_norm(static_cast<std::size_t>(head_dim_),
                                       float_to_half_bits(1.0f));
  CUDA_CHECK(cudaMalloc(&d_q_norm_unit_,
                        static_cast<std::size_t>(head_dim_) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_k_norm_unit_,
                        static_cast<std::size_t>(head_dim_) * sizeof(__half)));
  CUDA_CHECK(cudaMemcpyAsync(d_q_norm_unit_,
                             unit_norm.data(),
                             static_cast<std::size_t>(head_dim_) *
                                 sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice,
                             compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_k_norm_unit_,
                             unit_norm.data(),
                             static_cast<std::size_t>(head_dim_) *
                                 sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice,
                             compute_stream_));

  // Determine how many layers' shared expert weights fit in remaining VRAM.
  // We must reserve VRAM for all attention weights first (they must stay resident).
  // Attention per layer: wq(hiddenÃ—hidden) + wk+wv(2Ã—kv_dimÃ—hidden) + wo(hiddenÃ—hidden)
  //                    + q_norm + k_norm + norm_att + norm_ffn (tiny)
  std::size_t free_vram = 0, total_vram = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
  const std::size_t attn_fp16_per_layer =
      (2ULL * static_cast<std::size_t>(hidden_) * static_cast<std::size_t>(hidden_) +
       2ULL * static_cast<std::size_t>(kv_dim_) * static_cast<std::size_t>(hidden_)) *
      sizeof(__half);
  const std::size_t attn_total = static_cast<std::size_t>(n_layers_) * attn_fp16_per_layer;
  const std::size_t shared_fp16_per_layer =
      3ULL * static_cast<std::size_t>(inter_shared_) *
      static_cast<std::size_t>(hidden_) * sizeof(__half);
  // Leave 5% headroom for runtime buffers, KV cache, etc.
  const std::size_t vram_for_shared =
      (free_vram > attn_total + free_vram / 20)
          ? (free_vram - attn_total - free_vram / 20)
          : 0;
  n_shared_gpu_ = static_cast<int>(
      std::min(static_cast<std::size_t>(n_layers_),
               vram_for_shared / shared_fp16_per_layer));

  layer_device_.resize(static_cast<std::size_t>(n_layers_));
  layer_host_moe_.resize(static_cast<std::size_t>(n_layers_));
  layer_device_shared_.resize(static_cast<std::size_t>(n_layers_));
  for (int layer = 0; layer < n_layers_; ++layer) {
    const std::string prefix =
        "language_model.model.layers." + std::to_string(layer);
    auto& dev = layer_device_[static_cast<std::size_t>(layer)];
    auto& host = layer_host_moe_[static_cast<std::size_t>(layer)];

    alloc_and_copy(&dev.norm_att,
                   load_tensor(prefix + ".input_layernorm.weight",
                               static_cast<std::size_t>(hidden_)),
                   static_cast<std::size_t>(hidden_));
    alloc_and_copy(&dev.norm_ffn,
                   load_tensor(prefix + ".post_attention_layernorm.weight",
                               static_cast<std::size_t>(hidden_)),
                   static_cast<std::size_t>(hidden_));
    alloc_and_copy(&dev.wq,
                   load_tensor(prefix + ".self_attn.q_proj.weight",
                               static_cast<std::size_t>(hidden_) *
                                   static_cast<std::size_t>(hidden_)),
                   static_cast<std::size_t>(hidden_) *
                       static_cast<std::size_t>(hidden_));
    alloc_and_copy(&dev.wk,
                   load_tensor(prefix + ".self_attn.k_proj.weight",
                               static_cast<std::size_t>(kv_dim_) *
                                   static_cast<std::size_t>(hidden_)),
                   static_cast<std::size_t>(kv_dim_) *
                       static_cast<std::size_t>(hidden_));
    alloc_and_copy(&dev.wv,
                   load_tensor(prefix + ".self_attn.v_proj.weight",
                               static_cast<std::size_t>(kv_dim_) *
                                   static_cast<std::size_t>(hidden_)),
                   static_cast<std::size_t>(kv_dim_) *
                       static_cast<std::size_t>(hidden_));
    alloc_and_copy(&dev.wo,
                   load_tensor(prefix + ".self_attn.o_proj.weight",
                               static_cast<std::size_t>(hidden_) *
                                   static_cast<std::size_t>(hidden_)),
                   static_cast<std::size_t>(hidden_) *
                       static_cast<std::size_t>(hidden_));

    const std::string q_norm_name = prefix + ".self_attn.q_norm.weight";
    if (weights_.has_tensor(q_norm_name)) {
      alloc_and_copy(&dev.q_norm,
                     load_tensor(q_norm_name,
                                 static_cast<std::size_t>(head_dim_)),
                     static_cast<std::size_t>(head_dim_));
    }
    const std::string k_norm_name = prefix + ".self_attn.k_norm.weight";
    if (weights_.has_tensor(k_norm_name)) {
      alloc_and_copy(&dev.k_norm,
                     load_tensor(k_norm_name,
                                 static_cast<std::size_t>(head_dim_)),
                     static_cast<std::size_t>(head_dim_));
    }

    host.router = load_tensor(prefix + ".feed_forward.router.weight",
                              static_cast<std::size_t>(n_experts_) *
                                  static_cast<std::size_t>(hidden_));
    host.gate_up = load_tensor(prefix + ".feed_forward.experts.gate_up_proj",
                               static_cast<std::size_t>(n_experts_) *
                                   static_cast<std::size_t>(hidden_) *
                                   static_cast<std::size_t>(inter_full_));
    host.down_exp = load_tensor(prefix + ".feed_forward.experts.down_proj",
                                static_cast<std::size_t>(n_experts_) *
                                    static_cast<std::size_t>(inter_expert_) *
                                    static_cast<std::size_t>(hidden_));
    // Shared expert weights: load to GPU if VRAM permits, otherwise keep on host.
    auto& dev_sh = layer_device_shared_[static_cast<std::size_t>(layer)];
    const auto* sh_gate_src = reinterpret_cast<const std::uint16_t*>(
        weights_.tensor_ptr(prefix + ".feed_forward.shared_expert.gate_proj.weight"));
    const auto* sh_up_src = reinterpret_cast<const std::uint16_t*>(
        weights_.tensor_ptr(prefix + ".feed_forward.shared_expert.up_proj.weight"));
    const auto* sh_down_src = reinterpret_cast<const std::uint16_t*>(
        weights_.tensor_ptr(prefix + ".feed_forward.shared_expert.down_proj.weight"));
    if (layer < n_shared_gpu_) {
      alloc_and_copy(&dev_sh.sh_gate, sh_gate_src,
                     static_cast<std::size_t>(inter_shared_) *
                         static_cast<std::size_t>(hidden_));
      alloc_and_copy(&dev_sh.sh_up, sh_up_src,
                     static_cast<std::size_t>(inter_shared_) *
                         static_cast<std::size_t>(hidden_));
      alloc_and_copy(&dev_sh.sh_down, sh_down_src,
                     static_cast<std::size_t>(hidden_) *
                         static_cast<std::size_t>(inter_shared_));
      host.sh_gate = nullptr;
      host.sh_up = nullptr;
      host.sh_down = nullptr;
    } else {
      // Fall back to per-step streaming for layers that don't fit on GPU.
      dev_sh.sh_gate = nullptr;
      dev_sh.sh_up = nullptr;
      dev_sh.sh_down = nullptr;
      host.sh_gate = sh_gate_src;
      host.sh_up = sh_up_src;
      host.sh_down = sh_down_src;
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}


void Llama4CudaEngine::build_rope_tables() {
  const int half_dim = head_dim_ / 2;
  std::vector<float> rope_cos(
      static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(half_dim));
  std::vector<float> rope_sin(
      static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(half_dim));

  const float low_wavelen =
      static_cast<float>(rope_orig_max_pos_) / rope_low_freq_factor_;
  const float high_wavelen =
      static_cast<float>(rope_orig_max_pos_) / rope_high_freq_factor_;

  for (int d = 0; d < half_dim; ++d) {
    float theta_d =
        std::pow(rope_theta_,
                 -2.0f * static_cast<float>(d) /
                     static_cast<float>(head_dim_));
    const float wavelen = 2.0f * kPi / theta_d;
    if (wavelen > low_wavelen) {
      theta_d /= rope_scale_;
    } else if (wavelen >= high_wavelen) {
      const float smooth =
          (static_cast<float>(rope_orig_max_pos_) / wavelen -
           rope_low_freq_factor_) /
          (rope_high_freq_factor_ - rope_low_freq_factor_);
      const float scaled_theta = theta_d / rope_scale_;
      theta_d = (1.0f - smooth) * scaled_theta + smooth * theta_d;
    }

    for (int p = 0; p < max_ctx_; ++p) {
      const float angle = static_cast<float>(p) * theta_d;
      rope_cos[static_cast<std::size_t>(p) *
                   static_cast<std::size_t>(half_dim) +
               static_cast<std::size_t>(d)] = std::cos(angle);
      rope_sin[static_cast<std::size_t>(p) *
                   static_cast<std::size_t>(half_dim) +
               static_cast<std::size_t>(d)] = std::sin(angle);
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(d_rope_cos_,
                             rope_cos.data(),
                             rope_cos.size() * sizeof(float),
                             cudaMemcpyHostToDevice,
                             compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_rope_sin_,
                             rope_sin.data(),
                             rope_sin.size() * sizeof(float),
                             cudaMemcpyHostToDevice,
                             compute_stream_));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}


void Llama4CudaEngine::reset_kv_cache() {
  const std::size_t cache_bytes =
      static_cast<std::size_t>(n_layers_) *
      static_cast<std::size_t>(max_ctx_) *
      static_cast<std::size_t>(kv_dim_) *
      sizeof(__half);
  CUDA_CHECK(cudaMemsetAsync(d_k_cache_, 0, cache_bytes, compute_stream_));
  CUDA_CHECK(cudaMemsetAsync(d_v_cache_, 0, cache_bytes, compute_stream_));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

void Llama4CudaEngine::load_token_embedding_to_device(int token) {
  if (!h_tok_embeddings_bf16_) {
    LLAMA_ENGINE_THROW("token embeddings are not initialized");
  }
  token = std::clamp(token, 0, vocab_size_ - 1);
  const std::uint16_t* row =
      h_tok_embeddings_bf16_ +
      static_cast<std::size_t>(token) * static_cast<std::size_t>(hidden_);
  for (int i = 0; i < hidden_; ++i) {
    h_token_embedding_fp16_[static_cast<std::size_t>(i)] =
        float_to_half_bits(bf16_to_float(row[i]));
  }
  CUDA_CHECK(cudaMemcpyAsync(d_x_,
                             h_token_embedding_fp16_.data(),
                             static_cast<std::size_t>(hidden_) *
                                 sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice,
                             compute_stream_));
}


int Llama4CudaEngine::select_expert_cpu(int layer) {
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  CUDA_CHECK(cudaMemcpy(h_x_norm_fp16_.data(),
                        d_x_norm_,
                        static_cast<std::size_t>(hidden_) *
                            sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < hidden_; ++i) {
    h_x_norm_f32_[static_cast<std::size_t>(i)] =
        half_bits_to_float(h_x_norm_fp16_[static_cast<std::size_t>(i)]);
  }

  const std::uint16_t* router =
      layer_host_moe_[static_cast<std::size_t>(layer)].router;
  int best_idx = 0;
  float best_score = -std::numeric_limits<float>::infinity();
  for (int expert = 0; expert < n_experts_; ++expert) {
    const std::uint16_t* row =
        router + static_cast<std::size_t>(expert) *
                     static_cast<std::size_t>(hidden_);
    float sum = 0.0f;
    for (int i = 0; i < hidden_; ++i) {
      sum += bf16_to_float(row[i]) * h_x_norm_f32_[static_cast<std::size_t>(i)];
    }
    if (sum > best_score) {
      best_score = sum;
      best_idx = expert;
    }
  }
  return best_idx;
}


void Llama4CudaEngine::load_layer_moe_weights_to_device(int layer, int expert_idx) {
  if (expert_idx < 0 || expert_idx >= n_experts_) {
    LLAMA_ENGINE_THROW("expert index out of range");
  }

  const auto& host = layer_host_moe_[static_cast<std::size_t>(layer)];
  const std::uint16_t* gate_up =
      host.gate_up +
      static_cast<std::size_t>(expert_idx) *
          static_cast<std::size_t>(hidden_) *
          static_cast<std::size_t>(inter_full_);
  const std::uint16_t* down =
      host.down_exp +
      static_cast<std::size_t>(expert_idx) *
          static_cast<std::size_t>(inter_expert_) *
          static_cast<std::size_t>(hidden_);

  copy_bf16_tensor_to_fp16_device(
      gate_up,
      d_expert_gate_up_w_,
      static_cast<std::size_t>(hidden_) * static_cast<std::size_t>(inter_full_));
  copy_bf16_tensor_to_fp16_device(
      down,
      d_expert_down_w_,
      static_cast<std::size_t>(inter_expert_) * static_cast<std::size_t>(hidden_));
  ++last_benchmark_stats_.streamed_layer_copies;
}


void Llama4CudaEngine::rowmajor_projection_half(const void* w,
                                                const void* x,
                                                void* y,
                                                int out_features,
                                                int in_features) {
  kernels::launch_rowmajor_half_gemv_f16(static_cast<const __half*>(w),
                                         static_cast<const __half*>(x),
                                         static_cast<__half*>(y),
                                         out_features,
                                         in_features,
                                         compute_stream_);
}


void Llama4CudaEngine::rowmajor_projection_float(const void* w,
                                                 const void* x,
                                                 void* y,
                                                 int out_features,
                                                 int in_features) {
  kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(w),
                                         static_cast<const __half*>(x),
                                         static_cast<float*>(y),
                                         out_features,
                                         in_features,
                                         compute_stream_);
}


void Llama4CudaEngine::transposed_projection_half(const void* w_in_out,
                                                  const void* x,
                                                  void* y,
                                                  int in_features,
                                                  int out_features) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(cublas_,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            out_features,
                            1,
                            in_features,
                            &alpha,
                            w_in_out,
                            CUDA_R_16F,
                            out_features,
                            x,
                            CUDA_R_16F,
                            in_features,
                            &beta,
                            y,
                            CUDA_R_16F,
                            out_features,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


void Llama4CudaEngine::initialize(const EngineOptions& options) {
  destroy();
  options_ = options;
  if (options_.max_context > 0) {
    max_ctx_ = std::min(options_.max_context,
                        rope_orig_max_pos_ * static_cast<int>(rope_scale_));
  }
  kv_dim_ = n_kv_heads_ * head_dim_;

  try {
    weights_.open(options_.model_path);
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, compute_stream_));

    allocate_runtime_buffers();
    load_resident_weights();
    build_rope_tables();
    reset_kv_cache();

    if (options_.verbose) {
      // Estimate GPU VRAM used:
      //   attention weights + lm_head + shared expert weights (now GPU-resident)
      const long long shared_mb = static_cast<long long>(n_layers_) *
          3LL * inter_shared_ * hidden_ * 2 / (1024 * 1024);
      const long long attn_mb = static_cast<long long>(n_layers_) *
          (static_cast<long long>(hidden_) * hidden_ * 2LL +
           static_cast<long long>(kv_dim_) * hidden_ * 4LL) * 2 / (1024 * 1024);
      const long long lm_head_mb = static_cast<long long>(vocab_size_) * hidden_ * 2 / (1024 * 1024);
      std::cout << "[llama4_cuda] layers=" << n_layers_
                << " hidden=" << hidden_
                << " heads=" << n_heads_ << "/" << n_kv_heads_
                << " experts=" << n_experts_
                << " vocab=" << vocab_size_
                << " max_ctx=" << max_ctx_
                << " kv_dim=" << kv_dim_
                << " gpu_vram_est=" << (shared_mb + attn_mb + lm_head_mb) << "MB"
                << " (shared_exp=" << shared_mb << "MB)"
                << " shared_gpu_layers=" << n_shared_gpu_ << "/" << n_layers_
                << "\n";
    }
  } catch (...) {
    destroy();
    throw;
  }
}
}  // namespace engine