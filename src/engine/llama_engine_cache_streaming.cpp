#include "llama_engine_internal.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "common.hpp"
#include "runtime/cuda_utils.cuh"

namespace engine {
namespace {

std::size_t bytes_for_matrix(int rows, int cols) {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(__half);
}

}  // namespace

void LlamaEngine::copy_layer_weights_to_device(int layer,
                                               LayerDeviceWeights* dst,
                                               LayerDeviceInt8Weights* dst_i8,
                                               cudaStream_t stream) {
  enforce_host_resource_limits("copy_layer_weights_to_device.begin");
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);

  const LayerHostPinnedWeights* pinned = nullptr;
  if (layer >= 0 && layer < static_cast<int>(layer_host_pinned_.size())) {
    const auto& p = layer_host_pinned_[static_cast<std::size_t>(layer)];
    if (p.wq) {
      pinned = &p;
    }
  }

  const LayerHostInt8Weights* quant = nullptr;
  if (lowbit_streaming_enabled(options_) && layer >= 0 && layer < static_cast<int>(layer_host_int8_.size())) {
    const auto& q = layer_host_int8_[static_cast<std::size_t>(layer)];
    if (q.w1 || q.w2 || q.w3) {
      quant = &q;
    }
  }

  const auto load_fp16 = [&](const std::string& name, void* dst_fp16, std::size_t bytes, const void* src_override) {
    if (!weights_.has_tensor(name) && !src_override) {
      if (has_any_packed_lowbit_tensor(weights_, name)) {
        LLAMA_ENGINE_THROW("packed low-bit tensor requires --weight-quant int8|int4: " + name);
      }
      LLAMA_ENGINE_THROW("missing tensor: " + name);
    }
    const void* src = src_override ? src_override : weights_.tensor_data(name);
    CUDA_CHECK(cudaMemcpyAsync(dst_fp16, src, bytes, cudaMemcpyHostToDevice, stream));
  };
  const auto load_optional_fp16 = [&](const std::string& name,
                                      void* dst_fp16,
                                      std::size_t bytes,
                                      const void* src_override) {
    const void* src = src_override;
    if (!src && weights_.has_tensor(name)) {
      src = weights_.tensor_data(name);
    }
    if (src) {
      CUDA_CHECK(cudaMemcpyAsync(dst_fp16, src, bytes, cudaMemcpyHostToDevice, stream));
    } else {
      CUDA_CHECK(cudaMemsetAsync(dst_fp16, 0, bytes, stream));
    }
  };

  if (!benchmark_transfer_active_) {
    CUDA_CHECK(cudaEventRecord(benchmark_transfer_start_, stream));
    benchmark_transfer_active_ = true;
  }
  ++last_benchmark_stats_.streamed_layer_copies;

  const std::string p = "layers." + std::to_string(layer);
  load_fp16(p + ".attention_norm.weight", dst->norm_att, bytes_for_matrix(1, hidden), pinned ? pinned->norm_att : nullptr);
  load_fp16(p + ".ffn_norm.weight", dst->norm_ffn, bytes_for_matrix(1, hidden), pinned ? pinned->norm_ffn : nullptr);
  load_optional_fp16(p + ".attention_norm.bias", dst->norm_att_bias, bytes_for_matrix(1, hidden),
                     pinned ? pinned->norm_att_bias : nullptr);
  load_optional_fp16(p + ".ffn_norm.bias", dst->norm_ffn_bias, bytes_for_matrix(1, hidden),
                     pinned ? pinned->norm_ffn_bias : nullptr);
  load_fp16(p + ".attention.wo", dst->wo, bytes_for_matrix(hidden, q_hidden), pinned ? pinned->wo : nullptr);
  load_optional_fp16(p + ".attention.bo", dst->bo, bytes_for_matrix(1, hidden), pinned ? pinned->bo : nullptr);

  auto* wqkv_base = static_cast<__half*>(dst->wqkv);
  load_fp16(p + ".attention.wq", wqkv_base, bytes_for_matrix(q_hidden, hidden), pinned ? pinned->wq : nullptr);
  load_fp16(p + ".attention.wk",
            wqkv_base + static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden),
            bytes_for_matrix(kv_hidden, hidden),
            pinned ? pinned->wk : nullptr);
  load_fp16(p + ".attention.wv",
            wqkv_base + static_cast<std::size_t>(q_hidden + kv_hidden) * static_cast<std::size_t>(hidden),
            bytes_for_matrix(kv_hidden, hidden),
            pinned ? pinned->wv : nullptr);

  if (cfg.has_qkv_bias && dst->bqkv) {
    const std::string bname = p + ".attention.bqkv";
    const void* bsrc = (pinned && pinned->bqkv) ? pinned->bqkv : (weights_.has_tensor(bname) ? weights_.tensor_data(bname) : nullptr);
    if (bsrc) {
      CUDA_CHECK(cudaMemcpyAsync(dst->bqkv, bsrc, bytes_for_matrix(1, q_hidden + 2 * kv_hidden),
                                 cudaMemcpyHostToDevice, stream));
    }
  }

  if (quant) {
    if (!dst_i8 || !dst_i8->w1 || !dst_i8->w2 || !dst_i8->w3 || !dst_i8->s_w1 || !dst_i8->s_w2 || !dst_i8->s_w3) {
      LLAMA_ENGINE_THROW("missing weight-only int8 staging buffers");
    }
    dst_i8->mlp_int4 = false;
    dst_i8->proj_int4 = false;
    CUDA_CHECK(cudaMemcpyAsync(dst_i8->w1,
                               quant->w1,
                               static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst_i8->w2,
                               quant->w2,
                               static_cast<std::size_t>(hidden) * static_cast<std::size_t>(inter),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst_i8->w3,
                               quant->w3,
                               static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst_i8->s_w1,
                               quant->s_w1,
                               static_cast<std::size_t>(inter) * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst_i8->s_w2,
                               quant->s_w2,
                               static_cast<std::size_t>(hidden) * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst_i8->s_w3,
                               quant->s_w3,
                               static_cast<std::size_t>(inter) * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaEventRecord(benchmark_transfer_end_, stream));
    enforce_host_resource_limits("copy_layer_weights_to_device.end");
    return;
  }

  auto* w13_base = static_cast<__half*>(dst->w13);
  load_fp16(p + ".feed_forward.w1", w13_base, bytes_for_matrix(inter, hidden), pinned ? pinned->w1 : nullptr);
  load_fp16(p + ".feed_forward.w3",
            w13_base + static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden),
            bytes_for_matrix(inter, hidden),
            pinned ? pinned->w3 : nullptr);
  load_fp16(p + ".feed_forward.w2", dst->w2, bytes_for_matrix(hidden, inter), pinned ? pinned->w2 : nullptr);
  CUDA_CHECK(cudaEventRecord(benchmark_transfer_end_, stream));
  enforce_host_resource_limits("copy_layer_weights_to_device.end");
}

void LlamaEngine::init_uncached_pinned_host_weights() {
  const auto& cfg = weights_.config();

  if (cached_layer_count_ >= cfg.num_layers) {
    return;
  }

  layer_host_pinned_.resize(static_cast<std::size_t>(cfg.num_layers));
  std::size_t pinned_layers = 0;

  const auto pin_copy = [&](const std::string& name, void** out_ptr) -> bool {
    if (!weights_.has_tensor(name)) {
      if (has_any_packed_lowbit_tensor(weights_, name)) {
        LLAMA_ENGINE_THROW("packed low-bit tensor requires --weight-quant int8|int4: " + name);
      }
      LLAMA_ENGINE_THROW("missing tensor: " + name);
    }
    const std::size_t bytes = weights_.tensor_bytes(name);
    void* ptr = nullptr;
    const cudaError_t rc = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
    if (rc != cudaSuccess) {
      if (ptr) {
        cudaFreeHost(ptr);
      }
      cudaGetLastError();
      return false;
    }
    std::memcpy(ptr, weights_.tensor_data(name), bytes);
    *out_ptr = ptr;
    return true;
  };

  for (int layer = cached_layer_count_; layer < cfg.num_layers; ++layer) {
    auto& hp = layer_host_pinned_[static_cast<std::size_t>(layer)];
    const std::string p = "layers." + std::to_string(layer);
    bool ok = true;
    const auto pin_if_needed = [&](const std::string& name, void** out_ptr) {
      if (lowbit_streaming_enabled(options_) &&
          (has_any_packed_lowbit_tensor(weights_, name) || is_streaming_quantizable_tensor(name))) {
        return true;
      }
      return pin_copy(name, out_ptr);
    };

    ok = ok && pin_if_needed(p + ".attention.wq", &hp.wq);
    ok = ok && pin_if_needed(p + ".attention.wk", &hp.wk);
    ok = ok && pin_if_needed(p + ".attention.wv", &hp.wv);
    ok = ok && pin_if_needed(p + ".attention.wo", &hp.wo);
    if (weights_.has_tensor(p + ".attention.bo")) {
      ok = ok && pin_copy(p + ".attention.bo", &hp.bo);
    }
    ok = ok && pin_if_needed(p + ".feed_forward.w1", &hp.w1);
    ok = ok && pin_if_needed(p + ".feed_forward.w2", &hp.w2);
    ok = ok && pin_if_needed(p + ".feed_forward.w3", &hp.w3);
    ok = ok && pin_if_needed(p + ".attention_norm.weight", &hp.norm_att);
    ok = ok && pin_if_needed(p + ".ffn_norm.weight", &hp.norm_ffn);
    if (weights_.has_tensor(p + ".attention_norm.bias")) {
      ok = ok && pin_copy(p + ".attention_norm.bias", &hp.norm_att_bias);
    }
    if (weights_.has_tensor(p + ".ffn_norm.bias")) {
      ok = ok && pin_copy(p + ".ffn_norm.bias", &hp.norm_ffn_bias);
    }
    if (cfg.has_qkv_bias && weights_.has_tensor(p + ".attention.bqkv")) {
      pin_copy(p + ".attention.bqkv", &hp.bqkv);  // non-fatal if OOM; bias will be read from mmap
    }

    if (!ok) {
      if (hp.wq) cudaFreeHost(hp.wq);
      if (hp.wk) cudaFreeHost(hp.wk);
      if (hp.wv) cudaFreeHost(hp.wv);
      if (hp.wo) cudaFreeHost(hp.wo);
      if (hp.bo) cudaFreeHost(hp.bo);
      if (hp.w1) cudaFreeHost(hp.w1);
      if (hp.w2) cudaFreeHost(hp.w2);
      if (hp.w3) cudaFreeHost(hp.w3);
      if (hp.norm_att) cudaFreeHost(hp.norm_att);
      if (hp.norm_ffn) cudaFreeHost(hp.norm_ffn);
      if (hp.norm_att_bias) cudaFreeHost(hp.norm_att_bias);
      if (hp.norm_ffn_bias) cudaFreeHost(hp.norm_ffn_bias);
      if (hp.bqkv) cudaFreeHost(hp.bqkv);
      hp = {};
      break;
    }
    pinned_layers++;
  }

  if (options_.verbose && pinned_layers > 0) {
    std::cout << "[engine] pinned_uncached_layers: " << pinned_layers << "\n";
  }
}

void LlamaEngine::init_uncached_int8_host_weights() {
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int quant_bits = clamp_streaming_quant_bits(options_.streaming_quant_bits);

  if (cached_layer_count_ >= cfg.num_layers) {
    return;
  }

  layer_host_int8_.resize(static_cast<std::size_t>(cfg.num_layers));
  std::size_t qlayers = 0;

  const auto alloc_i8 = [&](std::size_t elems, std::int8_t** out) -> bool {
    void* ptr = nullptr;
    const cudaError_t rc = cudaHostAlloc(&ptr, elems, cudaHostAllocPortable);
    if (rc != cudaSuccess) {
      if (ptr) {
        cudaFreeHost(ptr);
      }
      cudaGetLastError();
      return false;
    }
    *out = static_cast<std::int8_t*>(ptr);
    return true;
  };

  const auto alloc_f32 = [&](std::size_t elems, float** out) -> bool {
    void* ptr = nullptr;
    const cudaError_t rc = cudaHostAlloc(&ptr, elems * sizeof(float), cudaHostAllocPortable);
    if (rc != cudaSuccess) {
      if (ptr) {
        cudaFreeHost(ptr);
      }
      cudaGetLastError();
      return false;
    }
    *out = static_cast<float*>(ptr);
    return true;
  };

  for (int layer = cached_layer_count_; layer < cfg.num_layers; ++layer) {
    auto& hq = layer_host_int8_[static_cast<std::size_t>(layer)];
    const std::string p = "layers." + std::to_string(layer);

    const auto maybe_alloc_and_load = [&](const std::string& name,
                                          int rows,
                                          int cols,
                                          std::int8_t** dst_w,
                                          float** dst_scales) -> bool {
      if (!is_streaming_quantizable_tensor(name) && !has_any_packed_lowbit_tensor(weights_, name)) {
        return true;
      }
      const std::size_t elems = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
      if (!alloc_i8(elems, dst_w) || !alloc_f32(static_cast<std::size_t>(rows), dst_scales)) {
        return false;
      }
      const bool prefer_i4 = (quant_bits == 4);
      if (prefer_i4 && has_packed_int4_tensor(weights_, name)) {
        unpack_rowwise_int4_to_int8(
            reinterpret_cast<const std::int8_t*>(weights_.tensor_data(int4_tensor_name(name))),
            rows,
            cols,
            *dst_w);
        load_rowwise_scales(weights_, name, rows, *dst_scales);
        return true;
      }
      if (!prefer_i4 && has_packed_int8_tensor(weights_, name)) {
        std::memcpy(*dst_w, weights_.tensor_data(int8_tensor_name(name)), elems);
        load_rowwise_scales(weights_, name, rows, *dst_scales);
        return true;
      }
      if (weights_.has_tensor(name)) {
        quantize_rowwise_to_int8(tensor_half(weights_, name), rows, cols, quant_bits, *dst_w, *dst_scales);
        return true;
      }
      if (has_packed_int8_tensor(weights_, name)) {
        std::memcpy(*dst_w, weights_.tensor_data(int8_tensor_name(name)), elems);
        load_rowwise_scales(weights_, name, rows, *dst_scales);
        return true;
      }
      if (has_packed_int4_tensor(weights_, name)) {
        unpack_rowwise_int4_to_int8(
            reinterpret_cast<const std::int8_t*>(weights_.tensor_data(int4_tensor_name(name))),
            rows,
            cols,
            *dst_w);
        load_rowwise_scales(weights_, name, rows, *dst_scales);
        return true;
      }
      return false;
    };

    bool ok = true;
    ok = ok && maybe_alloc_and_load(p + ".feed_forward.w1", inter, hidden, &hq.w1, &hq.s_w1);
    ok = ok && maybe_alloc_and_load(p + ".feed_forward.w2", hidden, inter, &hq.w2, &hq.s_w2);
    ok = ok && maybe_alloc_and_load(p + ".feed_forward.w3", inter, hidden, &hq.w3, &hq.s_w3);
    if (!ok) {
      if (hq.w1) cudaFreeHost(hq.w1);
      if (hq.w2) cudaFreeHost(hq.w2);
      if (hq.w3) cudaFreeHost(hq.w3);
      if (hq.s_w1) cudaFreeHost(hq.s_w1);
      if (hq.s_w2) cudaFreeHost(hq.s_w2);
      if (hq.s_w3) cudaFreeHost(hq.s_w3);
      hq = {};
      break;
    }
    if (hq.w1 || hq.w2 || hq.w3) {
      qlayers++;
    }
  }

  if (options_.verbose && qlayers > 0) {
    std::cout << "[engine] int" << quant_bits << "_uncached_layers: " << qlayers << "\n";
  }
}

}  // namespace engine
