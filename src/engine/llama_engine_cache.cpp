#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common.hpp"
#include "llama_engine_internal.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"
#include "runtime/system_info.hpp"
namespace engine {

namespace {
// Opt-in while the packed path is being wired matrix by matrix. Reading the
// generated text at each step is the only thing that catches a wrong unpack,
// so this stays off until every matrix has been through that.
// Bit per matrix: 1 = w2, 2 = wo, 4 = w13, 8 = lm head, 16 = fused qkv.
// One at a time is how the earlier
// unpacking bugs were caught -- a wrong unpack reads as fluent text, not a
// crash, so each matrix needs its own A/B against the fp16 path.
// Whether a tensor will be served packed, answerable before allocation so the
// fp16 buffer can be skipped rather than allocated and left unread.
int kquant_packed_mask();

// True when every part is packable AND they agree on quant type and width, the
// same condition the fused stagers apply. The allocation decision and the
// staging decision must not be able to disagree.
bool will_pack_fused(const model::WeightLoader& w, const std::vector<std::string>& names, int bit) {
  if ((kquant_packed_mask() & bit) == 0 || !w.is_gguf() || w.gguf() == nullptr) return false;
  int kind = -1;
  int cols = -1;
  for (const auto& n : names) {
    const auto pk = w.gguf()->packed_kquant(n);
    if (!pk.valid()) return false;
    if (kind < 0) {
      kind = pk.kind;
      cols = pk.cols;
    } else if (pk.kind != kind || pk.cols != cols) {
      return false;
    }
  }
  return true;
}

bool will_pack(const model::WeightLoader& w, const std::string& name, int bit) {
  if ((kquant_packed_mask() & bit) == 0 || !w.is_gguf() || w.gguf() == nullptr) return false;
  return w.gguf()->packed_kquant(name).valid();
}

int kquant_packed_mask() {
  static const int mask = []() {
    const char* env = std::getenv("CPI_KQUANT_PACKED");
    return env != nullptr ? std::atoi(env) : 0;
  }();
  return mask;
}
}  // namespace


namespace {

std::size_t bytes_for_matrix(int rows, int cols) {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(__half);
}

}  // namespace

bool LlamaEngine::tq_cached_preflight_layers(int layers, std::string* reason) const {
  if (!tq3_enabled_) {
    return true;
  }
  if (!options_.enable_tq_cached) {
    *reason = "tq_cached_opt_in_required";
    return false;
  }
  if (tq_prod_enabled_) {
    if (tq_qjl_dim_ <= 0 || tq_qjl_words_ <= 0 || !d_tq_qjl_indices_ || !d_tq_qjl_signs_) {
      *reason = "tq_prod_qjl_metadata_invalid";
      return false;
    }
  }

  const auto has_required_pair = [&](const std::string& base, const char* suffix0,
                                     const char* suffix1) {
    return weights_.has_tensor(base + suffix0) && weights_.has_tensor(base + suffix1);
  };

  for (int layer = 0; layer < layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer);
    const bool has_qkv = has_required_pair(p + ".attention.wq", ".tq3", ".tq3s") &&
                         has_required_pair(p + ".attention.wk", ".tq3", ".tq3s") &&
                         has_required_pair(p + ".attention.wv", ".tq3", ".tq3s");
    const bool has_wo = has_required_pair(p + ".attention.wo", ".tq3", ".tq3s");
    const bool has_w13 = has_required_pair(p + ".feed_forward.w1", ".tq3", ".tq3s") &&
                         has_required_pair(p + ".feed_forward.w3", ".tq3", ".tq3s");
    if (!has_qkv || !has_wo || !has_w13) {
      *reason = "tq_cached_missing_tq3_tensors_layer_" + std::to_string(layer);
      return false;
    }

    if (tq_prod_enabled_) {
      const bool has_r_qkv = has_required_pair(p + ".attention.wq", ".tq3r", ".tq3rs") &&
                             has_required_pair(p + ".attention.wk", ".tq3r", ".tq3rs") &&
                             has_required_pair(p + ".attention.wv", ".tq3r", ".tq3rs");
      const bool has_r_wo = has_required_pair(p + ".attention.wo", ".tq3r", ".tq3rs");
      const bool has_r_w13 = has_required_pair(p + ".feed_forward.w1", ".tq3r", ".tq3rs") &&
                             has_required_pair(p + ".feed_forward.w3", ".tq3r", ".tq3rs");
      if (!has_r_qkv || !has_r_wo || !has_r_w13) {
        *reason = "tq_cached_missing_qprod_tensors_layer_" + std::to_string(layer);
        return false;
      }
    }
  }
  return true;
}

// Uploads one packed tensor into `dst`, undoing the converter's Q/K row
// interleave on the way when the container reports one. The un-permute moves
// whole rows and a packed row is a contiguous run of super-blocks, so it is a
// gather of row-sized byte runs -- the same reordering the host unpacker does,
// applied to the blocks instead of to fp16.
bool LlamaEngine::upload_packed_rows(const model::GgufLoader::PackedTensor& pk, void* dst) {
  const std::size_t row_bytes = pk.bytes / static_cast<std::size_t>(pk.rows);
  if (pk.permute_heads <= 0 || pk.rows % pk.permute_heads != 0) {
    return cudaMemcpyAsync(dst, pk.data, pk.bytes, cudaMemcpyHostToDevice, transfer_stream_) ==
           cudaSuccess;
  }
  const int hd = pk.rows / pk.permute_heads;  // rows per head
  const int hh = hd / 2;
  std::vector<std::uint8_t> tmp(pk.bytes);
  const auto* src = reinterpret_cast<const std::uint8_t*>(pk.data);
  for (int h = 0; h < pk.permute_heads; ++h) {
    for (int r = 0; r < hd; ++r) {
      const int src_row = (r < hh) ? (r * 2) : ((r - hh) * 2 + 1);
      std::memcpy(tmp.data() + (static_cast<std::size_t>(h * hd + r) * row_bytes),
                  src + (static_cast<std::size_t>(h * hd + src_row) * row_bytes), row_bytes);
    }
  }
  // Synchronous: tmp is a local buffer and must not outlive the copy.
  return cudaMemcpy(dst, tmp.data(), pk.bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

// CPI fuses Q, K and V into one matrix; a GGUF stores them apart. Concatenating
// their packed rows is legal whenever the three share a quant type, which
// Q4_K_M does not always do -- it puts Q6_K in some layers' attn_v. When they
// disagree the fp16 path keeps the layer.
void LlamaEngine::stage_packed_triple(const std::string& a, const std::string& b,
                                      const std::string& c, PackedWeight* out, int mask_bit) {
  if ((kquant_packed_mask() & mask_bit) == 0) return;
  if (out == nullptr || !weights_.is_gguf() || weights_.gguf() == nullptr) return;
  const auto pa = weights_.gguf()->packed_kquant(a);
  const auto pb = weights_.gguf()->packed_kquant(b);
  const auto pc = weights_.gguf()->packed_kquant(c);
  if (!pa.valid() || !pb.valid() || !pc.valid()) return;
  if (pa.kind != pb.kind || pa.kind != pc.kind) return;
  if (pa.cols != pb.cols || pa.cols != pc.cols) return;
  void* dev = nullptr;
  if (cudaMalloc(&dev, pa.bytes + pb.bytes + pc.bytes) != cudaSuccess) return;
  auto* base = static_cast<std::uint8_t*>(dev);
  if (!upload_packed_rows(pa, base) || !upload_packed_rows(pb, base + pa.bytes) ||
      !upload_packed_rows(pc, base + pa.bytes + pb.bytes)) {
    cudaFree(dev);
    return;
  }
  out->data = dev;
  out->kind = pa.kind;
  out->rows = pa.rows + pb.rows + pc.rows;
  out->cols = pa.cols;
}

void LlamaEngine::stage_packed_pair(const std::string& first, const std::string& second,
                                    PackedWeight* out, int mask_bit) {
  if ((kquant_packed_mask() & mask_bit) == 0) return;
  if (out == nullptr || !weights_.is_gguf() || weights_.gguf() == nullptr) return;
  const auto a = weights_.gguf()->packed_kquant(first);
  const auto b = weights_.gguf()->packed_kquant(second);
  // Different quant types or widths cannot share one packed buffer; the mixed
  // k-quant mixes (Q4_K_M puts Q6_K in some layers) make this a real case.
  if (!a.valid() || !b.valid() || a.kind != b.kind || a.cols != b.cols) return;
  void* dev = nullptr;
  if (cudaMalloc(&dev, a.bytes + b.bytes) != cudaSuccess) return;
  auto* base = static_cast<std::uint8_t*>(dev);
  if (!upload_packed_rows(a, base) || !upload_packed_rows(b, base + a.bytes)) {
    cudaFree(dev);
    return;
  }
  out->data = dev;
  out->kind = a.kind;
  out->rows = a.rows + b.rows;
  out->cols = a.cols;
}

void LlamaEngine::stage_packed_weight(const std::string& name, PackedWeight* out,
                                      int mask_bit) {
  if ((kquant_packed_mask() & mask_bit) == 0) return;
  if (out == nullptr || !weights_.is_gguf() || weights_.gguf() == nullptr) return;
  const auto pk = weights_.gguf()->packed_kquant(name);
  if (!pk.valid() || pk.rows <= 0 || pk.cols <= 0) return;
  void* dev = nullptr;
  if (cudaMalloc(&dev, pk.bytes) != cudaSuccess) return;  // stay on fp16 rather than fail
  if (!upload_packed_rows(pk, dev)) {
    cudaFree(dev);
    return;
  }
  out->data = dev;
  out->kind = pk.kind;
  out->rows = pk.rows;
  out->cols = pk.cols;
}

void LlamaEngine::init_layer_cache() {
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int inter = cfg.intermediate_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  if (cfg.is_moe()) {
    cached_layer_count_ = 0;
    cached_int8_mlp_enabled_ = false;
    cached_int8_proj_enabled_ = false;
    if (options_.verbose) {
      std::cout << "[engine] MoE model detected: disabling resident dense layer cache (dynamic "
                   "expert routing path).\n";
    }
    return;
  }
  const int quant_bits = clamp_streaming_quant_bits(options_.streaming_quant_bits);
  // Attention projections cache at INT8 by default for quality. CPI_PROJ_INT4=1 caches them at
  // INT4 too (saves ~1.9 GB on the 32B); opt-in until the quality delta is validated per model.
  const bool enable_proj_int4 = quant_bits == 4 && std::getenv("CPI_PROJ_INT4") != nullptr;
  // CPI_MLP_INT4_GROUP=N (power of two >= 32, e.g. 128) stores MLP int4 weights with one scale per
  // group of N input columns instead of one per row; ~18% -> ~11% weight error (llama.cpp Q4_0
  // uses block-32). Opt-in until validated per model. Requires the perm8 grouped-dp4a path
  // (in_features % 32 == 0, group % 32 == 0) and a model that stores fp16 MLP weights on disk (we
  // quantize on load); otherwise we stay per-row so no consumer misreads the grouped scales.
  mlp_quant_group_ = 0;
  if (quant_bits == 4) {
    if (const char* g = std::getenv("CPI_MLP_INT4_GROUP")) {
      const int gv = std::atoi(g);
      if (gv >= 32 && (gv & (gv - 1)) == 0) {
        mlp_quant_group_ = gv;
      }
    }
  }
  const bool model_fp16_mlp = weights_.has_tensor("layers.0.feed_forward.w1") &&
                              !has_packed_int4_tensor(weights_, "layers.0.feed_forward.w1") &&
                              !has_packed_int8_tensor(weights_, "layers.0.feed_forward.w1");
  const int mlp_gq =
      (mlp_quant_group_ >= 32 && (mlp_quant_group_ % 32) == 0 && (hidden % 32) == 0 &&
       (inter % 32) == 0 && model_fp16_mlp)
          ? mlp_quant_group_
          : 0;
  cached_int8_mlp_enabled_ = false;
  const std::size_t fp16_attention_bytes =
      bytes_for_matrix(q_hidden, hidden) + bytes_for_matrix(hidden, q_hidden) +
      bytes_for_matrix(kv_hidden, hidden) * 2ULL + bytes_for_matrix(1, hidden) * 2ULL;
  // Resident footprint when QKV/wo are stored as regular low-bit (int8 or int4)
  // plus per-row float scales.
  const bool proj_i4_estimate = (quant_bits == 4) && enable_proj_int4;
  const std::size_t lowbit_qkv_bytes =
      proj_i4_estimate
          ? static_cast<std::size_t>(q_hidden + 2 * kv_hidden) *
                static_cast<std::size_t>((hidden + 1) / 2)
          : static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * static_cast<std::size_t>(hidden);
  const std::size_t lowbit_wo_bytes =
      proj_i4_estimate
          ? static_cast<std::size_t>(hidden) * static_cast<std::size_t>((q_hidden + 1) / 2)
          : static_cast<std::size_t>(hidden) * static_cast<std::size_t>(q_hidden);
  const std::size_t lowbit_attention_bytes =
      lowbit_qkv_bytes + static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(float) +
      lowbit_wo_bytes + static_cast<std::size_t>(hidden) * sizeof(float) +
      bytes_for_matrix(1, hidden) * 2ULL;
  const std::size_t fp16_mlp_bytes =
      bytes_for_matrix(inter, hidden) * 2ULL + bytes_for_matrix(hidden, inter);
  const std::size_t lowbit_w13_bytes =
      (quant_bits == 4)
          ? (static_cast<std::size_t>(inter) * static_cast<std::size_t>((hidden + 1) / 2) * 2ULL)
          : (static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden) * 2ULL);
  const std::size_t lowbit_w2_bytes =
      (quant_bits == 4)
          ? (static_cast<std::size_t>(hidden) * static_cast<std::size_t>((inter + 1) / 2))
          : (static_cast<std::size_t>(hidden) * static_cast<std::size_t>(inter));
  const std::size_t lowbit_mlp_bytes =
      lowbit_w13_bytes + lowbit_w2_bytes +
      static_cast<std::size_t>(inter + hidden + inter) * sizeof(float);
  // TQ3 layer footprint: packed 3-bit wqkv/wo/w13 + fp16 scales + fp16 w2 + 2 norms.
  // wqkv/wo/w13 are not loaded as fp16; int8 proj is also skipped.
  const std::size_t wpr_tq3 = static_cast<std::size_t>((hidden + 9) / 10);
  const std::size_t tq3_per_layer_bytes =
      static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * wpr_tq3 * sizeof(uint32_t) +
      static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half) +
      static_cast<std::size_t>(hidden) * wpr_tq3 * sizeof(uint32_t) +
      static_cast<std::size_t>(hidden) * sizeof(__half) +
      static_cast<std::size_t>(2 * inter) * wpr_tq3 * sizeof(uint32_t) +
      static_cast<std::size_t>(2 * inter) * sizeof(__half) + bytes_for_matrix(hidden, inter) +
      bytes_for_matrix(1, hidden) * 2ULL;
  const std::size_t qjl_words = static_cast<std::size_t>((tq_qjl_dim_ + 31) / 32);
  const std::size_t tqprod_residual_per_layer_bytes =
      (tq_prod_enabled_ && tq_qjl_dim_ > 0)
          ? (static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * qjl_words * sizeof(uint32_t) +
             static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half) +
             static_cast<std::size_t>(hidden) * qjl_words * sizeof(uint32_t) +
             static_cast<std::size_t>(hidden) * sizeof(__half) +
             static_cast<std::size_t>(2 * inter) * qjl_words * sizeof(uint32_t) +
             static_cast<std::size_t>(2 * inter) * sizeof(__half))
          : 0;
  const std::size_t tq_total_per_layer_bytes =
      tq3_per_layer_bytes + tqprod_residual_per_layer_bytes;
  const std::size_t fp16_per_layer_bytes = fp16_attention_bytes + fp16_mlp_bytes;
  // int8 proj is always used for GPU-cached layers; int8 MLP requires a pre-quantised model.
  // fp16_per_layer_bytes represents peak allocation cost (before fp16 proj is freed).
  const std::size_t int8_per_layer_bytes = lowbit_attention_bytes + lowbit_mlp_bytes;

  const auto layer_uses_cached_int8 = [&](int layer, bool enable_cached_int8) {
    return enable_cached_int8 && lowbit_streaming_enabled(options_) &&
           can_cache_layer_mlp_as_lowbit(weights_, layer, quant_bits);
  };
  const auto prefix_can_cache = [&](int layers, bool enable_cached_int8) {
    for (int layer = 0; layer < layers; ++layer) {
      if (layer_uses_cached_int8(layer, enable_cached_int8)) {
        continue;
      }
      if (!can_cache_layer_mlp_as_fp16(weights_, layer)) {
        return false;
      }
    }
    return true;
  };
  const auto prefix_cache_bytes = [&](int layers, bool enable_cached_int8) {
    std::size_t total = 0;
    for (int layer = 0; layer < layers; ++layer) {
      if (tq3_enabled_) {
        total += tq_total_per_layer_bytes;
      } else {
        const bool use_i8_mlp = layer_uses_cached_int8(layer, enable_cached_int8);
        // Proj is always int8; MLP is int8 only when the model has pre-quantised weights.
        total += lowbit_attention_bytes + (use_i8_mlp ? lowbit_mlp_bytes : fp16_mlp_bytes);
      }
    }
    return total;
  };
  // Peak extra memory needed during init: temporary fp16 proj copies freed after int8 quantization.
  // TQ3 skips fp16 proj staging entirely, so the peak overhead is zero.
  const std::size_t proj_init_peak_bytes =
      tq3_enabled_
          ? 0
          : bytes_for_matrix(q_hidden + 2 * kv_hidden, hidden) + bytes_for_matrix(hidden, q_hidden);
  const auto max_layers_for_budget = [&](std::size_t budget_bytes, bool enable_cached_int8) {
    // Reserve room for the temporary fp16 copies made during int8 quantization of projections.
    const std::size_t effective_budget =
        (budget_bytes > proj_init_peak_bytes) ? (budget_bytes - proj_init_peak_bytes) : 0;
    std::size_t used = 0;
    int layers = 0;
    for (; layers < cfg.num_layers; ++layers) {
      const bool use_cached_int8 = layer_uses_cached_int8(layers, enable_cached_int8);
      if (!tq3_enabled_ && !use_cached_int8 && !can_cache_layer_mlp_as_fp16(weights_, layers)) {
        break;
      }
      // TQ3: compact packed-weight footprint; fp16/int8: proj+MLP footprint.
      const std::size_t layer_bytes =
          tq3_enabled_
              ? tq_total_per_layer_bytes
              : lowbit_attention_bytes + (use_cached_int8 ? lowbit_mlp_bytes : fp16_mlp_bytes);
      if (used + layer_bytes > effective_budget) {
        break;
      }
      used += layer_bytes;
    }
    return layers;
  };
  std::size_t free_b = 0;
  std::size_t total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  const std::size_t safety = options_.vram_safety_margin_mb * 1024ULL * 1024ULL;
  const std::size_t cacheable_b = (free_b > safety) ? (free_b - safety) : 0;
  const std::size_t full_fp16_bytes =
      prefix_can_cache(cfg.num_layers, false) ? prefix_cache_bytes(cfg.num_layers, false) : 0;
  bool has_any_cached_int8_layer = false;
  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    if (can_cache_layer_mlp_as_lowbit(weights_, layer, quant_bits)) {
      has_any_cached_int8_layer = true;
      break;
    }
  }
  const std::size_t full_int8_bytes =
      (has_any_cached_int8_layer && prefix_can_cache(cfg.num_layers, true))
          ? prefix_cache_bytes(cfg.num_layers, true)
          : 0;
  const bool prefer_packed_cache = options_.prefer_lowbit_cache &&
                                   lowbit_streaming_enabled(options_) && has_any_cached_int8_layer;

  int requested_layers = 0;
  std::string cache_mode = "auto";
  std::string cache_policy = "partial_fp16";
  std::size_t effective_budget_b = cacheable_b;
  if (options_.gpu_cache_all) {
    cache_mode = "all";
    if (prefer_packed_cache) {
      if (full_int8_bytes > 0 && full_int8_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = true;
        cache_policy = "requested_full_packed_fit";
      } else if (full_fp16_bytes > 0 && full_fp16_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = false;
        cache_policy = "requested_full_fp16_fallback";
      } else {
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        if (int8_layers > 0) {
          cached_int8_mlp_enabled_ = true;
          requested_layers = int8_layers;
          cache_policy = "requested_partial_packed";
        } else {
          cached_int8_mlp_enabled_ = false;
          requested_layers = fp16_layers;
          cache_policy = "requested_partial_fp16_fallback";
        }
      }
    } else {
      if (full_fp16_bytes > 0 && full_fp16_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = false;
        cache_policy = "full_fp16_fit";
      } else if (full_int8_bytes > 0 && full_int8_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = true;
        cache_policy = "full_packed_fit";
      } else {
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        cached_int8_mlp_enabled_ = int8_layers > fp16_layers;
        requested_layers = cached_int8_mlp_enabled_ ? int8_layers : fp16_layers;
        cache_policy = cached_int8_mlp_enabled_ ? "packed_fallback_to_maximize_fit"
                                                : "fp16_fallback_to_maximize_fit";
      }
    }
  } else if (options_.gpu_cache_layers >= 0) {
    cache_mode = "layers";
    requested_layers = std::min(options_.gpu_cache_layers, cfg.num_layers);
    if (prefer_packed_cache) {
      if (requested_layers > 0 && prefix_can_cache(requested_layers, true) &&
          prefix_cache_bytes(requested_layers, true) <= effective_budget_b) {
        cached_int8_mlp_enabled_ = true;
        cache_policy = "requested_packed_fit";
      } else if (requested_layers > 0 && prefix_can_cache(requested_layers, false) &&
                 prefix_cache_bytes(requested_layers, false) <= effective_budget_b) {
        cached_int8_mlp_enabled_ = false;
        cache_policy = "requested_fp16_fallback";
      } else {
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        if (int8_layers > 0) {
          cached_int8_mlp_enabled_ = true;
          requested_layers = int8_layers;
          cache_policy = "requested_packed_partial";
        } else {
          cached_int8_mlp_enabled_ = false;
          requested_layers = fp16_layers;
          cache_policy = "requested_fp16_fallback";
        }
      }
    } else {
      if (requested_layers > 0 && prefix_can_cache(requested_layers, false) &&
          prefix_cache_bytes(requested_layers, false) <= effective_budget_b) {
        cached_int8_mlp_enabled_ = false;
        cache_policy = "requested_fp16_fit";
      } else if (requested_layers > 0 && prefix_can_cache(requested_layers, true) &&
                 prefix_cache_bytes(requested_layers, true) <= effective_budget_b) {
        cached_int8_mlp_enabled_ = true;
        cache_policy = "requested_packed_fit";
      } else {
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        cached_int8_mlp_enabled_ = int8_layers > fp16_layers;
        cache_policy =
            cached_int8_mlp_enabled_ ? "requested_packed_fallback" : "requested_fp16_fallback";
      }
    }
  } else if (options_.gpu_cache_limit_mb > 0) {
    cache_mode = "budget";
    effective_budget_b = std::min<std::size_t>(
        cacheable_b, static_cast<std::size_t>(options_.gpu_cache_limit_mb) * 1024ULL * 1024ULL);
    if (prefer_packed_cache) {
      if (full_int8_bytes > 0 && full_int8_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = true;
        cache_policy = "budget_full_packed_fit";
      } else if (full_fp16_bytes > 0 && full_fp16_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = false;
        cache_policy = "budget_full_fp16_fallback";
      } else {
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        if (int8_layers > 0) {
          cached_int8_mlp_enabled_ = true;
          requested_layers = int8_layers;
          cache_policy = "budget_partial_packed";
        } else {
          cached_int8_mlp_enabled_ = false;
          requested_layers = fp16_layers;
          cache_policy = "budget_partial_fp16_fallback";
        }
      }
    } else {
      if (full_fp16_bytes > 0 && full_fp16_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = false;
        cache_policy = "budget_full_fp16_fit";
      } else if (full_int8_bytes > 0 && full_int8_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = true;
        cache_policy = "budget_full_packed_fit";
      } else {
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        cached_int8_mlp_enabled_ = int8_layers > fp16_layers;
        requested_layers = cached_int8_mlp_enabled_ ? int8_layers : fp16_layers;
        cache_policy = cached_int8_mlp_enabled_ ? "budget_partial_packed" : "budget_partial_fp16";
      }
    }
  } else {
    if (prefer_packed_cache) {
      if (full_int8_bytes > 0 && full_int8_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = true;
        cache_policy = "requested_full_packed_fit";
      } else if (full_fp16_bytes > 0 && full_fp16_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = false;
        cache_policy = "requested_full_fp16_fallback";
      } else {
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        if (int8_layers > 0) {
          cached_int8_mlp_enabled_ = true;
          requested_layers = int8_layers;
          cache_policy = "requested_partial_packed";
        } else {
          cached_int8_mlp_enabled_ = false;
          requested_layers = fp16_layers;
          cache_policy = "requested_partial_fp16_fallback";
        }
      }
    } else {
      if (full_fp16_bytes > 0 && full_fp16_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = false;
        cache_policy = "full_fp16_fit";
      } else if (full_int8_bytes > 0 && full_int8_bytes <= effective_budget_b) {
        requested_layers = cfg.num_layers;
        cached_int8_mlp_enabled_ = true;
        cache_policy = "full_packed_fit";
      } else {
        const int fp16_layers = max_layers_for_budget(effective_budget_b, false);
        const int int8_layers = max_layers_for_budget(effective_budget_b, true);
        cached_int8_mlp_enabled_ = int8_layers > fp16_layers;
        requested_layers = cached_int8_mlp_enabled_ ? int8_layers : fp16_layers;
        cache_policy =
            cached_int8_mlp_enabled_ ? "partial_packed_to_reduce_streaming" : "partial_fp16";
      }
    }
  }

  if (tq3_enabled_ && options_.enable_tq_cached) {
    if (requested_layers != cfg.num_layers) {
      std::ostringstream oss;
      oss << "TurboQuant cached mode requires full residency: requested_layers=" << requested_layers
          << " total_layers=" << cfg.num_layers << " policy=" << cache_policy;
      CPI_THROW(oss.str());
    }
    std::string strict_reason;
    if (!tq_cached_preflight_layers(requested_layers, &strict_reason)) {
      CPI_THROW("TurboQuant cached preflight failed: " + strict_reason);
    }
    cache_policy = "tq_cached_strict";
  } else if (tq3_enabled_ && requested_layers > 0) {
    std::string fallback_reason;
    if (requested_layers < cfg.num_layers) {
      fallback_reason = "tq_cached_requires_full_residency";
    } else if (!tq_cached_preflight_layers(requested_layers, &fallback_reason)) {
      // fallback_reason populated by preflight helper
    }
    if (!fallback_reason.empty()) {
      if (options_.verbose) {
        std::cout << "[engine] TurboQuant cached preflight failed; falling back to uncached path: "
                  << fallback_reason << "\n";
      }
      requested_layers = 0;
      cached_int8_mlp_enabled_ = false;
      cache_policy = fallback_reason;
    }
  }

  const std::size_t estimated_cache_bytes =
      (requested_layers > 0 && prefix_can_cache(requested_layers, cached_int8_mlp_enabled_))
          ? prefix_cache_bytes(requested_layers, cached_int8_mlp_enabled_)
          : 0;

  if (options_.verbose) {
    std::cout << "[engine] gpu_cache_mode=" << cache_mode
              << " requested_layers=" << requested_layers
              << " budget=" << runtime::format_bytes(effective_budget_b) << " fp16_full_cache="
              << (full_fp16_bytes > 0 ? runtime::format_bytes(full_fp16_bytes) : "unavailable")
              << " packed_full_cache="
              << (full_int8_bytes > 0 ? runtime::format_bytes(full_int8_bytes) : "unavailable")
              << " estimated_layer_bytes_fp16=" << runtime::format_bytes(fp16_per_layer_bytes)
              << " estimated_layer_bytes_packed=" << runtime::format_bytes(int8_per_layer_bytes)
              << (tq3_enabled_ ? " estimated_layer_bytes_tq=" +
                                     runtime::format_bytes(tq_total_per_layer_bytes)
                               : "")
              << " estimated_cache_bytes=" << runtime::format_bytes(estimated_cache_bytes)
              << " mlp_cache="
              << (cached_int8_mlp_enabled_ ? ("int" + std::to_string(quant_bits)) : "fp16")
              << " policy=" << cache_policy << " safety_margin=" << options_.vram_safety_margin_mb
              << " MB\n";
  }

  if (requested_layers <= 0) {
    cached_layer_count_ = 0;
    return;
  }

  const auto cache_init_start = std::chrono::steady_clock::now();
  layer_cache_.resize(static_cast<std::size_t>(requested_layers));
  if (tq3_enabled_) {
    layer_cache_tq3_.resize(static_cast<std::size_t>(requested_layers));
  }
  layer_cache_i8_.resize(static_cast<std::size_t>(requested_layers));
  int built = 0;
  // CPI_STARTUP_PROFILE: split the per-layer cost into cudaMalloc churn vs the H2D/quant sync-wait.
  const bool cache_prof = std::getenv("CPI_STARTUP_PROFILE") != nullptr;
  double t_alloc_ms = 0, t_sync_ms = 0;
  using prof_clk = std::chrono::steady_clock;
  const auto prof_ms = [](prof_clk::time_point a) {
    return std::chrono::duration<double, std::milli>(prof_clk::now() - a).count();
  };

  for (int layer = 0; layer < requested_layers; ++layer) {
    enforce_host_resource_limits("cache_init.layer_begin");
    check_tq_cached_init_timeout(cache_init_start, layer);
    const auto t_alloc0 = prof_clk::now();
    auto& lw = layer_cache_[static_cast<std::size_t>(layer)];
    auto& lw_i8 = layer_cache_i8_[static_cast<std::size_t>(layer)];
    const bool use_cached_int8 =
        cached_int8_mlp_enabled_ && can_cache_layer_mlp_as_lowbit(weights_, layer, quant_bits);
    const bool use_cached_int4 = use_cached_int8 && (quant_bits == 4);
    lw_i8.mlp_int4 = use_cached_int4;
    lw_i8.mlp_group = use_cached_int4 ? mlp_gq : 0;
    // Group counts along each MLP matrix's input dimension (1 when per-row).
    const int mlp_ng_hidden = kernels::quant_group_count(hidden, lw_i8.mlp_group);  // w1, w3 [.,hidden]
    const int mlp_ng_inter = kernels::quant_group_count(inter, lw_i8.mlp_group);    // w2 [.,inter]
    // TQ3 path owns wqkv/wo/w13 via packed weights. Otherwise, only cache
    // projection weights in low-bit form when low-bit streaming was requested.
    const bool use_proj_int8 = !tq3_enabled_ && lowbit_streaming_enabled(options_);
    const bool use_proj_int4 = use_proj_int8 && (quant_bits == 4) && enable_proj_int4;
    lw_i8.proj_int4 = use_proj_int4;
    const std::size_t w1_bytes =
        use_cached_int4
            ? static_cast<std::size_t>(inter) * static_cast<std::size_t>((hidden + 1) / 2)
            : static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden);
    const std::size_t w2_bytes =
        use_cached_int4
            ? static_cast<std::size_t>(hidden) * static_cast<std::size_t>((inter + 1) / 2)
            : static_cast<std::size_t>(hidden) * static_cast<std::size_t>(inter);
    const std::size_t w3_bytes =
        use_cached_int4
            ? static_cast<std::size_t>(inter) * static_cast<std::size_t>((hidden + 1) / 2)
            : static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden);
    const std::string lp = "layers." + std::to_string(layer);
    // Must match stage_packed_triple exactly: fusing three row-blocks into one
    // buffer needs a shared quant type, and Q4_K_M puts Q6_K in some layers'
    // attn_v. Checking each tensor alone would skip the fp16 allocation for a
    // layer the stager then declines to pack, leaving nothing behind.
    // Either shape works now, so the fp16 fused matrix can go whenever all three
    // parts are packable at all.
    const bool pack_qkv = will_pack(weights_, lp + ".attention.wq", 16) &&
                          will_pack(weights_, lp + ".attention.wk", 16) &&
                          will_pack(weights_, lp + ".attention.wv", 16);
    const cudaError_t a0 =
        (tq3_enabled_ || pack_qkv)
            ? cudaSuccess
            : cudaMalloc(&lw.wqkv, bytes_for_matrix(q_hidden + 2 * kv_hidden, hidden));
    // A matrix that will live packed gets no fp16 buffer at all; that skipped
    // allocation is the whole point of packed residency. Every consumer either
    // multiplies the blocks directly or expands them into the shared prefill
    // scratch, so nothing reads the pointer that is left null here.
    const bool pack_wo = will_pack(weights_, lp + ".attention.wo", 2);
    const bool pack_w13 =
        will_pack_fused(weights_, {lp + ".feed_forward.w1", lp + ".feed_forward.w3"}, 4);
    const bool pack_w2 = will_pack(weights_, lp + ".feed_forward.w2", 1);
    const cudaError_t a1 = (tq3_enabled_ || pack_wo)
                               ? cudaSuccess
                               : cudaMalloc(&lw.wo, bytes_for_matrix(hidden, q_hidden));
    const cudaError_t a2 = (use_cached_int8 || tq3_enabled_ || pack_w13)
                               ? cudaSuccess
                               : cudaMalloc(&lw.w13, bytes_for_matrix(2 * inter, hidden));
    const cudaError_t a3 = (use_cached_int8 || pack_w2)
                               ? cudaSuccess
                               : cudaMalloc(&lw.w2, bytes_for_matrix(hidden, inter));
    const cudaError_t a4 = cudaMalloc(&lw.norm_att, bytes_for_matrix(1, hidden));
    const cudaError_t a5 = cudaMalloc(&lw.norm_ffn, bytes_for_matrix(1, hidden));
    const cudaError_t a16 = cudaMalloc(&lw.bo, bytes_for_matrix(1, hidden));
    const cudaError_t a17 = cudaMalloc(&lw.norm_att_bias, bytes_for_matrix(1, hidden));
    const cudaError_t a18 = cudaMalloc(&lw.norm_ffn_bias, bytes_for_matrix(1, hidden));
    // Qwen3 per-head QK-norm scales [head_dim]; only allocated when present.
    const int qk_norm_dim = attn_head_dim_ > 0 ? attn_head_dim_ : hidden / cfg.num_heads;
    const cudaError_t a19 =
        cfg.has_qk_norm ? cudaMalloc(&lw.q_norm, bytes_for_matrix(1, qk_norm_dim)) : cudaSuccess;
    const cudaError_t a20 =
        cfg.has_qk_norm ? cudaMalloc(&lw.k_norm, bytes_for_matrix(1, qk_norm_dim)) : cudaSuccess;
    const cudaError_t a6 = use_cached_int8 ? cudaMalloc(&lw_i8.w1, w1_bytes) : cudaSuccess;
    const cudaError_t a7 = use_cached_int8 ? cudaMalloc(&lw_i8.w2, w2_bytes) : cudaSuccess;
    const cudaError_t a8 = use_cached_int8 ? cudaMalloc(&lw_i8.w3, w3_bytes) : cudaSuccess;
    const cudaError_t a9 =
        use_cached_int8 ? cudaMalloc(&lw_i8.s_w1, static_cast<std::size_t>(inter) *
                                                      static_cast<std::size_t>(mlp_ng_hidden) *
                                                      sizeof(float))
                        : cudaSuccess;
    const cudaError_t a10 =
        use_cached_int8 ? cudaMalloc(&lw_i8.s_w2, static_cast<std::size_t>(hidden) *
                                                      static_cast<std::size_t>(mlp_ng_inter) *
                                                      sizeof(float))
                        : cudaSuccess;
    const cudaError_t a11 =
        use_cached_int8 ? cudaMalloc(&lw_i8.s_w3, static_cast<std::size_t>(inter) *
                                                      static_cast<std::size_t>(mlp_ng_hidden) *
                                                      sizeof(float))
                        : cudaSuccess;
    const cudaError_t a12 =
        use_proj_int8
            ? cudaMalloc(&lw_i8.wqkv, use_proj_int4
                                          ? static_cast<std::size_t>(q_hidden + 2 * kv_hidden) *
                                                static_cast<std::size_t>((hidden + 1) / 2)
                                          : static_cast<std::size_t>(q_hidden + 2 * kv_hidden) *
                                                static_cast<std::size_t>(hidden))
            : cudaSuccess;
    const cudaError_t a13 =
        use_proj_int8
            ? cudaMalloc(&lw_i8.s_wqkv,
                         static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(float))
            : cudaSuccess;
    const cudaError_t a14 =
        use_proj_int8
            ? cudaMalloc(&lw_i8.wo, use_proj_int4 ? static_cast<std::size_t>(hidden) *
                                                        static_cast<std::size_t>((q_hidden + 1) / 2)
                                                  : static_cast<std::size_t>(hidden) *
                                                        static_cast<std::size_t>(q_hidden))
            : cudaSuccess;
    const cudaError_t a15 =
        use_proj_int8 ? cudaMalloc(&lw_i8.s_wo, static_cast<std::size_t>(hidden) * sizeof(float))
                      : cudaSuccess;
    t_alloc_ms += prof_ms(t_alloc0);
    if (a0 != cudaSuccess || a1 != cudaSuccess || a2 != cudaSuccess || a3 != cudaSuccess ||
        a4 != cudaSuccess || a5 != cudaSuccess || a6 != cudaSuccess || a7 != cudaSuccess ||
        a8 != cudaSuccess || a9 != cudaSuccess || a10 != cudaSuccess || a11 != cudaSuccess ||
        a12 != cudaSuccess || a13 != cudaSuccess || a14 != cudaSuccess || a15 != cudaSuccess ||
        a16 != cudaSuccess || a17 != cudaSuccess || a18 != cudaSuccess || a19 != cudaSuccess ||
        a20 != cudaSuccess) {
      if (lw.wqkv) cudaFree(lw.wqkv);
      if (lw.wo) cudaFree(lw.wo);
      if (lw.bo) cudaFree(lw.bo);
      if (lw.w13) cudaFree(lw.w13);
      if (lw.wqkv_packed.data) cudaFree(lw.wqkv_packed.data);
      if (lw.wq_packed.data) cudaFree(lw.wq_packed.data);
      if (lw.wk_packed.data) cudaFree(lw.wk_packed.data);
      if (lw.wv_packed.data) cudaFree(lw.wv_packed.data);
      if (lw.w2_packed.data) cudaFree(lw.w2_packed.data);
      if (lw.wo_packed.data) cudaFree(lw.wo_packed.data);
      if (lw.w13_packed.data) cudaFree(lw.w13_packed.data);
      if (lw.w2) cudaFree(lw.w2);
      if (lw.norm_att) cudaFree(lw.norm_att);
      if (lw.norm_ffn) cudaFree(lw.norm_ffn);
      if (lw.norm_att_bias) cudaFree(lw.norm_att_bias);
      if (lw.norm_ffn_bias) cudaFree(lw.norm_ffn_bias);
      if (lw.q_norm) cudaFree(lw.q_norm);
      if (lw.k_norm) cudaFree(lw.k_norm);
      if (lw_i8.w1) cudaFree(lw_i8.w1);
      if (lw_i8.w2) cudaFree(lw_i8.w2);
      if (lw_i8.w3) cudaFree(lw_i8.w3);
      if (lw_i8.s_w1) cudaFree(lw_i8.s_w1);
      if (lw_i8.s_w2) cudaFree(lw_i8.s_w2);
      if (lw_i8.s_w3) cudaFree(lw_i8.s_w3);
      if (lw_i8.wqkv) cudaFree(lw_i8.wqkv);
      if (lw_i8.s_wqkv) cudaFree(lw_i8.s_wqkv);
      if (lw_i8.wo) cudaFree(lw_i8.wo);
      if (lw_i8.s_wo) cudaFree(lw_i8.s_wo);
      lw = {};
      lw_i8 = {};
      cudaGetLastError();
      break;
    }
    CUDA_CHECK(cudaMemsetAsync(lw.bo, 0, bytes_for_matrix(1, hidden), transfer_stream_));
    CUDA_CHECK(cudaMemsetAsync(lw.norm_att_bias, 0, bytes_for_matrix(1, hidden), transfer_stream_));
    CUDA_CHECK(cudaMemsetAsync(lw.norm_ffn_bias, 0, bytes_for_matrix(1, hidden), transfer_stream_));

    // Async H2D on transfer_stream_; quantization on compute_stream_ synchronized via events.
    // This overlaps wo H2D with wqkv quantization, and MLP H2D with wo quantization,
    // reducing per-layer sync count from 3 to 1.
    const auto load_async = [&](const std::string& name, void* dst, std::size_t bytes,
                                std::int8_t* tmp_i8, float* tmp_scales, int rows, int cols) {
      if (weights_.has_tensor(name)) {
        CUDA_CHECK(cudaMemcpyAsync(dst, weights_.tensor_data(name), bytes, cudaMemcpyHostToDevice,
                                   transfer_stream_));
        return;
      }
      if (!has_any_packed_lowbit_tensor(weights_, name)) {
        CPI_THROW("missing fp16/packed tensor for cached layer load: " + name);
      }
      if (!lowbit_streaming_enabled(options_)) {
        CPI_THROW("packed low-bit tensor requires --weight-quant int8|int4: " + name);
      }
      if (!tmp_i8 || !tmp_scales) {
        CPI_THROW("missing temporary low-bit buffers for cached layer load: " + name);
      }
      // Packed low-bit: async H2D on transfer_stream_, sync to compute_stream_ via
      // event before dequant to fp16.
      const bool prefer_i4 = (quant_bits == 4);
      if (prefer_i4 && has_packed_int4_tensor(weights_, name)) {
        std::vector<std::int8_t> unpacked(
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0);
        unpack_rowwise_int4_to_int8(
            reinterpret_cast<const std::int8_t*>(weights_.tensor_data(int4_tensor_name(name))),
            rows, cols, unpacked.data());
        CUDA_CHECK(cudaMemcpy(tmp_i8, unpacked.data(), unpacked.size(), cudaMemcpyHostToDevice));
      } else if (has_packed_int8_tensor(weights_, name)) {
        CUDA_CHECK(cudaMemcpyAsync(tmp_i8, weights_.tensor_data(int8_tensor_name(name)),
                                   static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
                                   cudaMemcpyHostToDevice, transfer_stream_));
      } else if (has_packed_int4_tensor(weights_, name)) {
        std::vector<std::int8_t> unpacked(
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0);
        unpack_rowwise_int4_to_int8(
            reinterpret_cast<const std::int8_t*>(weights_.tensor_data(int4_tensor_name(name))),
            rows, cols, unpacked.data());
        CUDA_CHECK(cudaMemcpy(tmp_i8, unpacked.data(), unpacked.size(), cudaMemcpyHostToDevice));
      } else {
        CPI_THROW("missing packed low-bit tensor data: " + name);
      }
      cudaEvent_t ev_i8 = nullptr;
      CUDA_CHECK(cudaEventCreateWithFlags(&ev_i8, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(ev_i8, transfer_stream_));
      CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, ev_i8, 0));
      CUDA_CHECK(cudaEventDestroy(ev_i8));
      const std::size_t scale_bytes = packed_quant_scale_bytes(weights_, name);
      if (scale_bytes == static_cast<std::size_t>(rows) * sizeof(float)) {
        CUDA_CHECK(cudaMemcpy(tmp_scales, weights_.tensor_data(quant_scale_name(name)), scale_bytes,
                              cudaMemcpyHostToDevice));
        kernels::launch_dequant_rowwise_int8_to_fp16(tmp_i8, tmp_scales, static_cast<half*>(dst),
                                                     rows, cols, compute_stream_);
      } else {
        kernels::launch_dequant_int8_to_fp16(tmp_i8, static_cast<half*>(dst), rows * cols,
                                             packed_quant_scale(weights_, name), compute_stream_);
      }
    };
    const auto load_optional_async = [&](const std::string& name, void* dst, std::size_t bytes) {
      if (!dst) return;
      if (!weights_.has_tensor(name)) {
        CUDA_CHECK(cudaMemsetAsync(dst, 0, bytes, transfer_stream_));
        return;
      }
      CUDA_CHECK(cudaMemcpyAsync(dst, weights_.tensor_data(name), bytes, cudaMemcpyHostToDevice,
                                 transfer_stream_));
    };
    const std::string p = "layers." + std::to_string(layer);
    auto* tmp_i8 = lowbit_streaming_enabled(options_) ? &streaming_layer_weights_i8_[0] : nullptr;
    auto* wqkv_base = static_cast<__half*>(lw.wqkv);
    auto* w13_base = static_cast<__half*>(lw.w13);
    const bool plain_fp16_cached_layer = !tq3_enabled_ && !use_proj_int8 && !use_cached_int8;

    const auto copy_fp16_direct = [&](const std::string& name, void* dst, std::size_t bytes) {
      if (!dst) {
        return;
      }
      if (!weights_.has_tensor(name)) {
        CPI_THROW("missing tensor: " + name);
      }
      // A container that can unpack on the device does so: the quantized bytes
      // cross the bus and expand in VRAM, instead of expanding to fp16 in host
      // RAM first. Anything else falls through to the plain copy.
      if (weights_.fill_device_fp16(name, dst, transfer_stream_)) {
        return;
      }
      CUDA_CHECK(cudaMemcpyAsync(dst, weights_.tensor_data(name), bytes, cudaMemcpyHostToDevice,
                                 transfer_stream_));
    };
    const auto copy_optional_fp16_direct = [&](const std::string& name, void* dst,
                                               std::size_t bytes) {
      if (!dst) {
        return;
      }
      if (weights_.has_tensor(name)) {
        CUDA_CHECK(cudaMemcpyAsync(dst, weights_.tensor_data(name), bytes, cudaMemcpyHostToDevice,
                                   transfer_stream_));
      } else {
        CUDA_CHECK(cudaMemsetAsync(dst, 0, bytes, transfer_stream_));
      }
    };

    if (plain_fp16_cached_layer) {
      copy_fp16_direct(p + ".attention_norm.weight", lw.norm_att, bytes_for_matrix(1, hidden));
      copy_optional_fp16_direct(p + ".attention_norm.bias", lw.norm_att_bias,
                                bytes_for_matrix(1, hidden));
      if (cfg.has_qk_norm) {
        const int qkd = attn_head_dim_ > 0 ? attn_head_dim_ : hidden / cfg.num_heads;
        copy_fp16_direct(p + ".attention.q_norm", lw.q_norm, bytes_for_matrix(1, qkd));
        copy_fp16_direct(p + ".attention.k_norm", lw.k_norm, bytes_for_matrix(1, qkd));
      }
      if (wqkv_base != nullptr) {
        copy_fp16_direct(p + ".attention.wq", wqkv_base, bytes_for_matrix(q_hidden, hidden));
        copy_fp16_direct(
            p + ".attention.wk",
            wqkv_base + static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden),
            bytes_for_matrix(kv_hidden, hidden));
        copy_fp16_direct(p + ".attention.wv",
                         wqkv_base + static_cast<std::size_t>(q_hidden + kv_hidden) *
                                         static_cast<std::size_t>(hidden),
                         bytes_for_matrix(kv_hidden, hidden));
      }
      if (lw.wo != nullptr) {
        copy_fp16_direct(p + ".attention.wo", lw.wo, bytes_for_matrix(hidden, q_hidden));
      }
      copy_optional_fp16_direct(p + ".attention.bo", lw.bo, bytes_for_matrix(1, hidden));
      if (cfg.has_qkv_bias && weights_.has_tensor(p + ".attention.bqkv")) {
        const std::size_t bias_bytes = bytes_for_matrix(1, q_hidden + 2 * kv_hidden);
        if (!lw.bqkv && cudaMalloc(&lw.bqkv, bias_bytes) == cudaSuccess) {
          CUDA_CHECK(cudaMemcpyAsync(lw.bqkv, weights_.tensor_data(p + ".attention.bqkv"),
                                     bias_bytes, cudaMemcpyHostToDevice, transfer_stream_));
        }
      }
      copy_fp16_direct(p + ".ffn_norm.weight", lw.norm_ffn, bytes_for_matrix(1, hidden));
      copy_optional_fp16_direct(p + ".ffn_norm.bias", lw.norm_ffn_bias,
                                bytes_for_matrix(1, hidden));
      if (w13_base != nullptr) {
        copy_fp16_direct(p + ".feed_forward.w1", w13_base, bytes_for_matrix(inter, hidden));
        copy_fp16_direct(
            p + ".feed_forward.w3",
            w13_base + static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden),
            bytes_for_matrix(inter, hidden));
      }
      if (lw.w2 != nullptr) {
        copy_fp16_direct(p + ".feed_forward.w2", lw.w2, bytes_for_matrix(hidden, inter));
      }
      // Packed k-quant residency (CPI_KQUANT_PACKED=1): keep the container's own
      // blocks on the device so decode can multiply them directly. Staged
      // alongside the fp16 copy for now -- correctness first; dropping the fp16
      // allocation is the step that actually saves the memory.
      stage_packed_triple(p + ".attention.wq", p + ".attention.wk", p + ".attention.wv",
                          &lw.wqkv_packed, 16);
      if (!lw.wqkv_packed.active()) {
        // Q, K and V disagree on quant type here (Q4_K_M puts Q6_K in some
        // layers' attn_v), so they cannot share one buffer. Fuse as much as
        // does agree: a 1024-row matvec is far off roofline (250 GB/s against
        // 900 for a 4096-row one), so folding K into Q is worth a buffer.
        stage_packed_pair(p + ".attention.wq", p + ".attention.wk", &lw.wq_packed, 16);
        if (!lw.wq_packed.active()) {
          stage_packed_weight(p + ".attention.wq", &lw.wq_packed, 16);
          stage_packed_weight(p + ".attention.wk", &lw.wk_packed, 16);
        }
        stage_packed_weight(p + ".attention.wv", &lw.wv_packed, 16);
      }
      stage_packed_weight(p + ".feed_forward.w2", &lw.w2_packed, 1);
      stage_packed_weight(p + ".attention.wo", &lw.wo_packed, 2);
      stage_packed_pair(p + ".feed_forward.w1", p + ".feed_forward.w3", &lw.w13_packed, 4);

      CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
      CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
      built++;
      enforce_host_resource_limits("cache_init.layer_end");
      check_tq_cached_init_timeout(cache_init_start, layer);
      continue;
    }

    // Load norm_att always; wqkv/wo skipped for TQ3 (covered by packed weights loaded later).
    load_async(p + ".attention_norm.weight", lw.norm_att, bytes_for_matrix(1, hidden), nullptr,
               nullptr, 1, hidden);
    load_optional_async(p + ".attention_norm.bias", lw.norm_att_bias, bytes_for_matrix(1, hidden));
    if (!tq3_enabled_) {
      load_async(p + ".attention.wq", wqkv_base, bytes_for_matrix(q_hidden, hidden), nullptr,
                 nullptr, q_hidden, hidden);
      load_async(p + ".attention.wk",
                 wqkv_base + static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden),
                 bytes_for_matrix(kv_hidden, hidden), nullptr, nullptr, kv_hidden, hidden);
      load_async(p + ".attention.wv",
                 wqkv_base + static_cast<std::size_t>(q_hidden + kv_hidden) *
                                 static_cast<std::size_t>(hidden),
                 bytes_for_matrix(kv_hidden, hidden), nullptr, nullptr, kv_hidden, hidden);
    }

    if (use_proj_int8) {
      // Record event after wqkv H2D: compute_stream_ must wait before quantization.
      cudaEvent_t ev_wqkv = nullptr;
      CUDA_CHECK(cudaEventCreateWithFlags(&ev_wqkv, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(ev_wqkv, transfer_stream_));

      // Load wo on transfer_stream_; overlaps with qkv quantization work.
      load_async(p + ".attention.wo", lw.wo, bytes_for_matrix(hidden, q_hidden), nullptr, nullptr,
                 hidden, q_hidden);

      // Record event after wo H2D.
      cudaEvent_t ev_wo = nullptr;
      CUDA_CHECK(cudaEventCreateWithFlags(&ev_wo, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(ev_wo, transfer_stream_));
      load_optional_async(p + ".attention.bo", lw.bo, bytes_for_matrix(1, hidden));

      CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, ev_wqkv, 0));
      if (use_proj_int4) {
        if (!tmp_i8 || !tmp_i8->w1 || !tmp_i8->w2 || !tmp_i8->s_w1 || !tmp_i8->s_w2) {
          CPI_THROW("missing temporary buffers for int4 projection packing");
        }
        kernels::launch_quantize_rowwise_fp16_to_int8(
            static_cast<const __half*>(lw.wqkv), tmp_i8->w1, tmp_i8->s_w1, q_hidden + 2 * kv_hidden,
            hidden, compute_stream_, 7);
        CUDA_CHECK(
            cudaMemcpyAsync(lw_i8.s_wqkv, tmp_i8->s_w1,
                            static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(float),
                            cudaMemcpyDeviceToDevice, compute_stream_));
        kernels::launch_pack_rowwise_int8_to_int4(tmp_i8->w1, lw_i8.wqkv, q_hidden + 2 * kv_hidden,
                                                  hidden, compute_stream_);
      } else {
        kernels::launch_quantize_rowwise_fp16_to_int8(
            static_cast<const __half*>(lw.wqkv), lw_i8.wqkv, lw_i8.s_wqkv, q_hidden + 2 * kv_hidden,
            hidden, compute_stream_);
      }

      CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, ev_wo, 0));
      if (use_proj_int4) {
        kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(lw.wo), tmp_i8->w2,
                                                      tmp_i8->s_w2, hidden, q_hidden,
                                                      compute_stream_, 7);
        CUDA_CHECK(cudaMemcpyAsync(lw_i8.s_wo, tmp_i8->s_w2,
                                   static_cast<std::size_t>(hidden) * sizeof(float),
                                   cudaMemcpyDeviceToDevice, compute_stream_));
        kernels::launch_pack_rowwise_int8_to_int4(tmp_i8->w2, lw_i8.wo, hidden, q_hidden,
                                                  compute_stream_);
      } else {
        kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(lw.wo), lw_i8.wo,
                                                      lw_i8.s_wo, hidden, q_hidden,
                                                      compute_stream_);
      }
      CUDA_CHECK(cudaEventDestroy(ev_wqkv));
      CUDA_CHECK(cudaEventDestroy(ev_wo));
    }
    if (!use_proj_int8) {
      load_optional_async(p + ".attention.bo", lw.bo, bytes_for_matrix(1, hidden));
    }

    if (cfg.has_qkv_bias && weights_.has_tensor(p + ".attention.bqkv")) {
      const std::size_t bias_bytes = bytes_for_matrix(1, q_hidden + 2 * kv_hidden);
      if (cudaMalloc(&lw.bqkv, bias_bytes) == cudaSuccess) {
        CUDA_CHECK(cudaMemcpyAsync(lw.bqkv, weights_.tensor_data(p + ".attention.bqkv"), bias_bytes,
                                   cudaMemcpyHostToDevice, transfer_stream_));
      }
    }
    load_async(p + ".ffn_norm.weight", lw.norm_ffn, bytes_for_matrix(1, hidden), nullptr, nullptr,
               1, hidden);
    load_optional_async(p + ".ffn_norm.bias", lw.norm_ffn_bias, bytes_for_matrix(1, hidden));
    if (use_cached_int8) {
      const auto copy_scales_to_device = [&](const std::string& name, float* dst_scales, int rows) {
        const std::size_t scale_bytes = packed_quant_scale_bytes(weights_, name);
        if (scale_bytes == static_cast<std::size_t>(rows) * sizeof(float)) {
          CUDA_CHECK(cudaMemcpyAsync(dst_scales, weights_.tensor_data(quant_scale_name(name)),
                                     scale_bytes, cudaMemcpyHostToDevice, transfer_stream_));
          return;
        }
        std::vector<float> host_scales(static_cast<std::size_t>(rows),
                                       packed_quant_scale(weights_, name));
        CUDA_CHECK(cudaMemcpy(dst_scales, host_scales.data(),
                              static_cast<std::size_t>(rows) * sizeof(float),
                              cudaMemcpyHostToDevice));
      };
      const auto load_cached_lowbit = [&](const std::string& name, int rows, int cols,
                                          std::int8_t* dst_w, float* dst_scales) {
        const std::size_t elems = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        const std::size_t packed_i4_bytes =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>((cols + 1) / 2);
        const bool target_i4 = lw_i8.mlp_int4;

        if (target_i4 && has_packed_int4_tensor(weights_, name)) {
          CUDA_CHECK(cudaMemcpyAsync(dst_w, weights_.tensor_data(int4_tensor_name(name)),
                                     packed_i4_bytes, cudaMemcpyHostToDevice, transfer_stream_));
          copy_scales_to_device(name, dst_scales, rows);
          return;
        }

        if (!target_i4 && has_packed_int8_tensor(weights_, name)) {
          CUDA_CHECK(cudaMemcpyAsync(dst_w, weights_.tensor_data(int8_tensor_name(name)), elems,
                                     cudaMemcpyHostToDevice, transfer_stream_));
          copy_scales_to_device(name, dst_scales, rows);
          return;
        }

        if (weights_.has_tensor(name)) {
          std::vector<std::int8_t> q(elems, 0);
          // Group-wise int4 (mlp_gq>0) narrows each scale to `mlp_gq` columns: rows*n_groups scales.
          // The int8 values (hence the packed nibbles) keep the same layout, so packing is unchanged.
          const int group = (target_i4 && mlp_gq > 0) ? mlp_gq : 0;
          const int ng = kernels::quant_group_count(cols, group);
          std::vector<float> s(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ng), 0.0f);
          if (group > 0) {
            quantize_groupwise_to_int8(tensor_half(weights_, name), rows, cols, group, quant_bits,
                                       q.data(), s.data());
          } else {
            quantize_rowwise_to_int8(tensor_half(weights_, name), rows, cols, quant_bits, q.data(),
                                     s.data());
          }
          if (target_i4) {
            std::vector<std::int8_t> packed(packed_i4_bytes, 0);
            pack_rowwise_int8_to_int4(q.data(), rows, cols, packed.data());
            CUDA_CHECK(cudaMemcpy(dst_w, packed.data(), packed.size(), cudaMemcpyHostToDevice));
          } else {
            CUDA_CHECK(cudaMemcpy(dst_w, q.data(), q.size(), cudaMemcpyHostToDevice));
          }
          CUDA_CHECK(
              cudaMemcpy(dst_scales, s.data(), s.size() * sizeof(float), cudaMemcpyHostToDevice));
          return;
        }

        if (has_packed_int8_tensor(weights_, name)) {
          if (target_i4) {
            std::vector<std::int8_t> q(elems, 0);
            std::memcpy(q.data(), weights_.tensor_data(int8_tensor_name(name)), elems);
            std::vector<std::int8_t> packed(packed_i4_bytes, 0);
            pack_rowwise_int8_to_int4(q.data(), rows, cols, packed.data());
            CUDA_CHECK(cudaMemcpy(dst_w, packed.data(), packed.size(), cudaMemcpyHostToDevice));
          } else {
            CUDA_CHECK(cudaMemcpyAsync(dst_w, weights_.tensor_data(int8_tensor_name(name)), elems,
                                       cudaMemcpyHostToDevice, transfer_stream_));
          }
          copy_scales_to_device(name, dst_scales, rows);
          return;
        }

        if (has_packed_int4_tensor(weights_, name)) {
          if (target_i4) {
            CUDA_CHECK(cudaMemcpyAsync(dst_w, weights_.tensor_data(int4_tensor_name(name)),
                                       packed_i4_bytes, cudaMemcpyHostToDevice, transfer_stream_));
          } else {
            std::vector<std::int8_t> unpacked(elems, 0);
            unpack_rowwise_int4_to_int8(
                reinterpret_cast<const std::int8_t*>(weights_.tensor_data(int4_tensor_name(name))),
                rows, cols, unpacked.data());
            CUDA_CHECK(cudaMemcpy(dst_w, unpacked.data(), unpacked.size(), cudaMemcpyHostToDevice));
          }
          copy_scales_to_device(name, dst_scales, rows);
          return;
        }

        CPI_THROW("missing low-bit/fp16 tensor for cached MLP: " + name);
      };
      load_cached_lowbit(p + ".feed_forward.w1", inter, hidden, lw_i8.w1, lw_i8.s_w1);
      load_cached_lowbit(p + ".feed_forward.w2", hidden, inter, lw_i8.w2, lw_i8.s_w2);
      load_cached_lowbit(p + ".feed_forward.w3", inter, hidden, lw_i8.w3, lw_i8.s_w3);
    } else if (tq3_enabled_) {
      // TQ3: only w2 stays fp16; w1/w3 are covered by packed TQ3 weights.
      load_async(p + ".feed_forward.w2", lw.w2, bytes_for_matrix(hidden, inter),
                 tmp_i8 ? tmp_i8->w2 : nullptr, tmp_i8 ? tmp_i8->s_w2 : nullptr, hidden, inter);
    } else {
      load_async(p + ".feed_forward.w1", w13_base, bytes_for_matrix(inter, hidden),
                 tmp_i8 ? tmp_i8->w1 : nullptr, tmp_i8 ? tmp_i8->s_w1 : nullptr, inter, hidden);
      load_async(p + ".feed_forward.w3",
                 w13_base + static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden),
                 bytes_for_matrix(inter, hidden), tmp_i8 ? tmp_i8->w3 : nullptr,
                 tmp_i8 ? tmp_i8->s_w3 : nullptr, inter, hidden);
      load_async(p + ".feed_forward.w2", lw.w2, bytes_for_matrix(hidden, inter),
                 tmp_i8 ? tmp_i8->w2 : nullptr, tmp_i8 ? tmp_i8->s_w2 : nullptr, hidden, inter);
    }

    // Single sync per layer (was 3): wait for all H2D transfers and GPU quantization.
    const auto t_sync0 = prof_clk::now();
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    t_sync_ms += prof_ms(t_sync0);

    // Free FP16 staging buffers now that quantization is confirmed complete.
    // (Skipped for TQ3: fp16 wqkv/wo were never allocated.)
    if (use_proj_int8) {
      cudaFree(lw.wqkv);
      lw.wqkv = nullptr;
      cudaFree(lw.wo);
      lw.wo = nullptr;
    }
    built++;
    enforce_host_resource_limits("cache_init.layer_end");
    check_tq_cached_init_timeout(cache_init_start, layer);
  }

  if (cache_prof) {
    std::cout << "[startup]   cache-copy.loop alloc_ms=" << static_cast<long>(t_alloc_ms)
              << " sync_wait_ms=" << static_cast<long>(t_sync_ms) << " layers=" << built << "\n";
  }
  cached_layer_count_ = built;

  // Size the prefill scratch now rather than on the first prompt. A packed
  // weight has no fp16 form for cuBLAS, so prefill expands the largest of them
  // into this buffer -- and growing it there costs a stream sync plus a
  // few-hundred-MB cudaMalloc on the very request whose latency matters most.
  // Steady-state prefill is unaffected; this is purely first-token latency.
  if (built > 0 && !layer_cache_.empty()) {
    std::size_t widest = 0;
    for (const auto& lw : layer_cache_) {
      for (const PackedWeight* w :
           {&lw.wqkv_packed, &lw.wq_packed, &lw.wk_packed, &lw.wv_packed, &lw.wo_packed,
            &lw.w13_packed, &lw.w2_packed}) {
        if (!w->active()) continue;
        widest = std::max(widest, static_cast<std::size_t>(w->rows) *
                                      static_cast<std::size_t>(w->cols) * sizeof(__half));
      }
    }
    if (widest > 0) ensure_prefill_wdq(widest, transfer_stream_);
  }

  cached_int8_proj_enabled_ = !tq3_enabled_ && lowbit_streaming_enabled(options_) && (built > 0);
  layer_cache_.resize(static_cast<std::size_t>(cached_layer_count_));
  layer_cache_i8_.resize(static_cast<std::size_t>(cached_layer_count_));

  // Load TQ3 packed weights into the layer cache for every successfully cached layer.
  if (tq3_enabled_ && cached_layer_count_ > 0) {
    layer_cache_tq3_.resize(static_cast<std::size_t>(cached_layer_count_));
    const int words_wqkv = (q_hidden + 9) / 10;
    const int words_wo = (hidden + 9) / 10;
    const int words_w13 = (hidden + 9) / 10;
    for (int layer = 0; layer < cached_layer_count_; ++layer) {
      enforce_host_resource_limits("cache_init.tq3_load_begin");
      check_tq_cached_init_timeout(cache_init_start, layer);
      auto& tq = layer_cache_tq3_[static_cast<std::size_t>(layer)];
      const std::string p = "layers." + std::to_string(layer);

      // wqkv (fused Q+K+V): out = hidden + 2*kv_hidden
      const std::string wq_tq3 = p + ".attention.wq.tq3";
      const std::string wk_tq3 = p + ".attention.wk.tq3";
      const std::string wv_tq3 = p + ".attention.wv.tq3";
      if (weights_.has_tensor(wq_tq3) && weights_.has_tensor(wk_tq3) &&
          weights_.has_tensor(wv_tq3)) {
        const std::size_t wq_bytes = weights_.tensor_bytes(wq_tq3);
        const std::size_t wk_bytes = weights_.tensor_bytes(wk_tq3);
        const std::size_t wv_bytes = weights_.tensor_bytes(wv_tq3);
        const std::size_t wqkv_bytes = wq_bytes + wk_bytes + wv_bytes;
        CUDA_CHECK(cudaMalloc(&tq.wqkv, wqkv_bytes));
        CUDA_CHECK(
            cudaMemcpy(tq.wqkv, weights_.tensor_data(wq_tq3), wq_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.wqkv)) + wq_bytes,
                              weights_.tensor_data(wk_tq3), wk_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.wqkv)) + wq_bytes + wk_bytes,
                       weights_.tensor_data(wv_tq3), wv_bytes, cudaMemcpyHostToDevice));

        // Scales: fuse Q+K+V scale vectors.
        const std::string sq = p + ".attention.wq.tq3s";
        const std::string sk = p + ".attention.wk.tq3s";
        const std::string sv = p + ".attention.wv.tq3s";
        const std::size_t sq_b = weights_.tensor_bytes(sq);
        const std::size_t sk_b = weights_.tensor_bytes(sk);
        const std::size_t sv_b = weights_.tensor_bytes(sv);
        CUDA_CHECK(cudaMalloc(&tq.s_wqkv, sq_b + sk_b + sv_b));
        CUDA_CHECK(cudaMemcpy(tq.s_wqkv, weights_.tensor_data(sq), sq_b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.s_wqkv)) + sq_b,
                              weights_.tensor_data(sk), sk_b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.s_wqkv)) + sq_b + sk_b,
                              weights_.tensor_data(sv), sv_b, cudaMemcpyHostToDevice));

        if (tq_prod_enabled_) {
          const std::string rq = p + ".attention.wq.tq3r";
          const std::string rk = p + ".attention.wk.tq3r";
          const std::string rv = p + ".attention.wv.tq3r";
          const std::string rsq = p + ".attention.wq.tq3rs";
          const std::string rsk = p + ".attention.wk.tq3rs";
          const std::string rsv = p + ".attention.wv.tq3rs";
          const std::size_t rq_b = weights_.tensor_bytes(rq);
          const std::size_t rk_b = weights_.tensor_bytes(rk);
          const std::size_t rv_b = weights_.tensor_bytes(rv);
          CUDA_CHECK(cudaMalloc(&tq.r_wqkv, rq_b + rk_b + rv_b));
          CUDA_CHECK(cudaMemcpy(tq.r_wqkv, weights_.tensor_data(rq), rq_b, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.r_wqkv)) + rq_b,
                                weights_.tensor_data(rk), rk_b, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.r_wqkv)) + rq_b + rk_b,
                                weights_.tensor_data(rv), rv_b, cudaMemcpyHostToDevice));

          const std::size_t rsq_b = weights_.tensor_bytes(rsq);
          const std::size_t rsk_b = weights_.tensor_bytes(rsk);
          const std::size_t rsv_b = weights_.tensor_bytes(rsv);
          CUDA_CHECK(cudaMalloc(&tq.rs_wqkv, rsq_b + rsk_b + rsv_b));
          CUDA_CHECK(
              cudaMemcpy(tq.rs_wqkv, weights_.tensor_data(rsq), rsq_b, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.rs_wqkv)) + rsq_b,
                                weights_.tensor_data(rsk), rsk_b, cudaMemcpyHostToDevice));
          CUDA_CHECK(
              cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.rs_wqkv)) + rsq_b + rsk_b,
                         weights_.tensor_data(rsv), rsv_b, cudaMemcpyHostToDevice));
        }
      }

      // wo
      const std::string wo_tq3 = p + ".attention.wo.tq3";
      if (weights_.has_tensor(wo_tq3)) {
        const std::size_t b = weights_.tensor_bytes(wo_tq3);
        CUDA_CHECK(cudaMalloc(&tq.wo, b));
        CUDA_CHECK(cudaMemcpy(tq.wo, weights_.tensor_data(wo_tq3), b, cudaMemcpyHostToDevice));
        const std::string s = p + ".attention.wo.tq3s";
        const std::size_t sb = weights_.tensor_bytes(s);
        CUDA_CHECK(cudaMalloc(&tq.s_wo, sb));
        CUDA_CHECK(cudaMemcpy(tq.s_wo, weights_.tensor_data(s), sb, cudaMemcpyHostToDevice));
        if (tq_prod_enabled_) {
          const std::string rb = p + ".attention.wo.tq3r";
          const std::string rs = p + ".attention.wo.tq3rs";
          const std::size_t rbb = weights_.tensor_bytes(rb);
          const std::size_t rsb = weights_.tensor_bytes(rs);
          CUDA_CHECK(cudaMalloc(&tq.r_wo, rbb));
          CUDA_CHECK(cudaMemcpy(tq.r_wo, weights_.tensor_data(rb), rbb, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMalloc(&tq.rs_wo, rsb));
          CUDA_CHECK(cudaMemcpy(tq.rs_wo, weights_.tensor_data(rs), rsb, cudaMemcpyHostToDevice));
        }
      }

      // w13 (fused w1 + w3)
      const std::string w1_tq3 = p + ".feed_forward.w1.tq3";
      const std::string w3_tq3 = p + ".feed_forward.w3.tq3";
      if (weights_.has_tensor(w1_tq3) && weights_.has_tensor(w3_tq3)) {
        const std::size_t w1b = weights_.tensor_bytes(w1_tq3);
        const std::size_t w3b = weights_.tensor_bytes(w3_tq3);
        CUDA_CHECK(cudaMalloc(&tq.w13, w1b + w3b));
        CUDA_CHECK(cudaMemcpy(tq.w13, weights_.tensor_data(w1_tq3), w1b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.w13)) + w1b,
                              weights_.tensor_data(w3_tq3), w3b, cudaMemcpyHostToDevice));
        const std::string s1 = p + ".feed_forward.w1.tq3s";
        const std::string s3 = p + ".feed_forward.w3.tq3s";
        const std::size_t s1b = weights_.tensor_bytes(s1);
        const std::size_t s3b = weights_.tensor_bytes(s3);
        CUDA_CHECK(cudaMalloc(&tq.s_w13, s1b + s3b));
        CUDA_CHECK(cudaMemcpy(tq.s_w13, weights_.tensor_data(s1), s1b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.s_w13)) + s1b,
                              weights_.tensor_data(s3), s3b, cudaMemcpyHostToDevice));
        if (tq_prod_enabled_) {
          const std::string r1 = p + ".feed_forward.w1.tq3r";
          const std::string r3 = p + ".feed_forward.w3.tq3r";
          const std::string rs1 = p + ".feed_forward.w1.tq3rs";
          const std::string rs3 = p + ".feed_forward.w3.tq3rs";
          const std::size_t r1b = weights_.tensor_bytes(r1);
          const std::size_t r3b = weights_.tensor_bytes(r3);
          CUDA_CHECK(cudaMalloc(&tq.r_w13, r1b + r3b));
          CUDA_CHECK(cudaMemcpy(tq.r_w13, weights_.tensor_data(r1), r1b, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.r_w13)) + r1b,
                                weights_.tensor_data(r3), r3b, cudaMemcpyHostToDevice));
          const std::size_t rs1b = weights_.tensor_bytes(rs1);
          const std::size_t rs3b = weights_.tensor_bytes(rs3);
          CUDA_CHECK(cudaMalloc(&tq.rs_w13, rs1b + rs3b));
          CUDA_CHECK(
              cudaMemcpy(tq.rs_w13, weights_.tensor_data(rs1), rs1b, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(static_cast<void*>(tq.rs_w13)) + rs1b,
                                weights_.tensor_data(rs3), rs3b, cudaMemcpyHostToDevice));
        }
      }
    }
    (void)words_wqkv;
    (void)words_wo;
    (void)words_w13;
    // Validate completeness: every cached layer must have all three TQ3 weight matrices.
    // A missing pointer means the .ll2c file lacks per-layer packed tensors; proceeding
    // would pass a null device pointer to a CUDA kernel and crash/hang the GPU.
    for (int layer = 0; layer < cached_layer_count_; ++layer) {
      const auto& tq = layer_cache_tq3_[static_cast<std::size_t>(layer)];
      if (!tq.wqkv || !tq.wo || !tq.w13) {
        CPI_THROW("TurboQuant model is missing packed weights for layer " + std::to_string(layer) +
                  " (wqkv=" + (tq.wqkv ? "ok" : "MISSING") + " wo=" + (tq.wo ? "ok" : "MISSING") +
                  " w13=" + (tq.w13 ? "ok" : "MISSING") +
                  "). Rebuild the model with turbo_quant_convert.py.");
      }
    }
  }
  if (options_.verbose && cached_layer_count_ > 0) {
    std::cout << "[engine] cached_layers_on_gpu: " << cached_layer_count_ << "/" << cfg.num_layers
              << "\n";
  }
  if (options_.verbose && cached_layer_count_ < requested_layers) {
    std::cout << "[engine] requested more GPU-cached layers than fit; using " << cached_layer_count_
              << " layer(s) instead.\n";
  }
}

}  // namespace engine
