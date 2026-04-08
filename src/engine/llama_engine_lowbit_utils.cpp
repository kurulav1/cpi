#include "llama_engine_internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <cuda_fp16.h>
namespace engine {
constexpr float kMinRowwiseQuantScale = 1e-8f;
constexpr std::array<const char*, 3> kMlpWeightSuffixes = {
    ".feed_forward.w1",
    ".feed_forward.w2",
    ".feed_forward.w3"};
std::string layer_prefix(int layer) {
  return "layers." + std::to_string(layer);
}
std::string layer_mlp_name(int layer, const char* suffix) {
  return layer_prefix(layer) + suffix;
}
const __half* tensor_half(const model::WeightLoader& weights, const std::string& name) {
  return reinterpret_cast<const __half*>(weights.tensor_data(name));
}

std::string int8_tensor_name(const std::string& base) { return base + ".int8"; }

std::string int4_tensor_name(const std::string& base) { return base + ".int4"; }

std::string quant_scale_name(const std::string& base) { return base + ".scale"; }

int clamp_streaming_quant_bits(int bits) {
  return bits == 4 ? 4 : 8;
}

int streaming_quant_maxq(int bits) {
  return clamp_streaming_quant_bits(bits) == 4 ? 7 : 127;
}

bool has_packed_int8_tensor(const model::WeightLoader& weights, const std::string& base) {
  return weights.has_tensor(int8_tensor_name(base)) && weights.has_tensor(quant_scale_name(base));
}

bool has_packed_int4_tensor(const model::WeightLoader& weights, const std::string& base) {
  return weights.has_tensor(int4_tensor_name(base)) && weights.has_tensor(quant_scale_name(base));
}

bool has_any_packed_lowbit_tensor(const model::WeightLoader& weights, const std::string& base) {
  return has_packed_int8_tensor(weights, base) || has_packed_int4_tensor(weights, base);
}

bool has_lowbit_source_tensor(const model::WeightLoader& weights, const std::string& name, int quant_bits) {
  if (weights.has_tensor(name)) {
    return true;
  }
  if (clamp_streaming_quant_bits(quant_bits) == 4 && has_packed_int4_tensor(weights, name)) {
    return true;
  }
  return has_packed_int8_tensor(weights, name);
}

bool can_cache_layer_mlp_as_lowbit(const model::WeightLoader& weights, int layer, int quant_bits) {
  for (const char* suffix : kMlpWeightSuffixes) {
    if (!has_lowbit_source_tensor(weights, layer_mlp_name(layer, suffix), quant_bits)) {
      return false;
    }
  }
  return true;
}

bool can_cache_layer_mlp_as_fp16(const model::WeightLoader& weights, int layer) {
  for (const char* suffix : kMlpWeightSuffixes) {
    if (!weights.has_tensor(layer_mlp_name(layer, suffix))) {
      return false;
    }
  }
  return true;
}

bool is_streaming_quantizable_tensor(const std::string& name) {
  for (const char* suffix : kMlpWeightSuffixes) {
    if (name.find(suffix) != std::string::npos) {
      return true;
    }
  }
  return false;
}

float packed_quant_scale(const model::WeightLoader& weights, const std::string& base) {
  const auto* ptr = weights.tensor_data(quant_scale_name(base));
  float scale = 1.0f;
  std::memcpy(&scale, ptr, sizeof(float));
  return scale;
}

std::size_t packed_quant_scale_bytes(const model::WeightLoader& weights, const std::string& base) {
  return weights.tensor_bytes(quant_scale_name(base));
}

void quantize_rowwise_to_int8(const __half* src,
                              int rows,
                              int cols,
                              int quant_bits,
                              std::int8_t* dst,
                              float* scales) {
  const int max_q = streaming_quant_maxq(quant_bits);
  const float max_q_f = static_cast<float>(max_q);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int row = 0; row < rows; ++row) {
    float max_abs = 0.0f;
    const std::size_t row_off = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    for (int col = 0; col < cols; ++col) {
      max_abs = std::max(max_abs, std::abs(__half2float(src[row_off + static_cast<std::size_t>(col)])));
    }
    float scale = max_abs / max_q_f;
    if (scale < kMinRowwiseQuantScale) {
      scale = kMinRowwiseQuantScale;
    }
    scales[row] = scale;
    for (int col = 0; col < cols; ++col) {
      const float q = __half2float(src[row_off + static_cast<std::size_t>(col)]) / scale;
      const float clamped = std::max(-max_q_f, std::min(max_q_f, q));
      dst[row_off + static_cast<std::size_t>(col)] = static_cast<std::int8_t>(std::lrint(clamped));
    }
  }
}

void unpack_rowwise_int4_to_int8(const std::int8_t* src,
                                 int rows,
                                 int cols,
                                 std::int8_t* dst) {
  const int packed_cols = (cols + 1) / 2;
  for (int row = 0; row < rows; ++row) {
    const std::size_t src_row = static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
    const std::size_t dst_row = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    for (int col = 0; col < cols; ++col) {
      const std::uint8_t byte = static_cast<std::uint8_t>(src[src_row + static_cast<std::size_t>(col / 2)]);
      const std::uint8_t nibble = (col & 1) == 0 ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
      const std::int8_t signed_q = (nibble >= 8) ? static_cast<std::int8_t>(nibble) - 16 : static_cast<std::int8_t>(nibble);
      dst[dst_row + static_cast<std::size_t>(col)] = signed_q;
    }
  }
}

void pack_rowwise_int8_to_int4(const std::int8_t* src,
                               int rows,
                               int cols,
                               std::int8_t* dst) {
  const int packed_cols = (cols + 1) / 2;
  for (int row = 0; row < rows; ++row) {
    const std::size_t src_row = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    const std::size_t dst_row = static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
    for (int col = 0; col < cols; col += 2) {
      const auto clamp_i4 = [](int v) -> std::uint8_t {
        const int q = std::max(-8, std::min(7, v));
        return static_cast<std::uint8_t>(q < 0 ? q + 16 : q);
      };
      const std::uint8_t lo = clamp_i4(static_cast<int>(src[src_row + static_cast<std::size_t>(col)]));
      std::uint8_t hi = 0;
      if (col + 1 < cols) {
        hi = clamp_i4(static_cast<int>(src[src_row + static_cast<std::size_t>(col + 1)]));
      }
      dst[dst_row + static_cast<std::size_t>(col / 2)] = static_cast<std::int8_t>(lo | (hi << 4));
    }
  }
}

void load_rowwise_scales(const model::WeightLoader& weights,
                         const std::string& base,
                         int rows,
                         float* dst_scales) {
  const std::size_t scale_bytes = packed_quant_scale_bytes(weights, base);
  if (scale_bytes == static_cast<std::size_t>(rows) * sizeof(float)) {
    std::memcpy(dst_scales, weights.tensor_data(quant_scale_name(base)), scale_bytes);
    return;
  }
  std::fill_n(dst_scales, rows, packed_quant_scale(weights, base));
}

bool lowbit_streaming_enabled(const engine::EngineOptions& options) {
  return options.int8_streaming;
}
}  // namespace engine