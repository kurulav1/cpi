#pragma once

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/llama_engine.hpp"

namespace engine {

int env_int_or_default(const char* name, int default_value);
std::size_t env_workspace_bytes_or_default(const char* name, std::size_t default_bytes);

const __half* tensor_half(const model::WeightLoader& weights, const std::string& name);
std::string int8_tensor_name(const std::string& base);
std::string int4_tensor_name(const std::string& base);
std::string quant_scale_name(const std::string& base);

int clamp_streaming_quant_bits(int bits);
bool has_packed_int8_tensor(const model::WeightLoader& weights, const std::string& base);
bool has_packed_int4_tensor(const model::WeightLoader& weights, const std::string& base);
bool has_any_packed_lowbit_tensor(const model::WeightLoader& weights, const std::string& base);
bool can_cache_layer_mlp_as_lowbit(const model::WeightLoader& weights, int layer, int quant_bits);
bool can_cache_layer_mlp_as_fp16(const model::WeightLoader& weights, int layer);
bool is_streaming_quantizable_tensor(const std::string& name);

float packed_quant_scale(const model::WeightLoader& weights, const std::string& base);
std::size_t packed_quant_scale_bytes(const model::WeightLoader& weights, const std::string& base);
void quantize_rowwise_to_int8(const __half* src, int rows, int cols, int quant_bits, std::int8_t* dst, float* scales);
void unpack_rowwise_int4_to_int8(const std::int8_t* src, int rows, int cols, std::int8_t* dst);
void pack_rowwise_int8_to_int4(const std::int8_t* src, int rows, int cols, std::int8_t* dst);
void load_rowwise_scales(const model::WeightLoader& weights, const std::string& base, int rows, float* dst_scales);
bool lowbit_streaming_enabled(const EngineOptions& options);

namespace detail {

void dispatch_linear_rowmajor_weight(cublasHandle_t handle,
                                     cublasLtHandle_t lt_handle,
                                     std::vector<LtMatmulPlan>* lt_cache,
                                     void* lt_workspace,
                                     std::size_t lt_workspace_bytes,
                                     cudaStream_t stream,
                                     const void* d_w_rowmajor,
                                     const void* d_x,
                                     void* d_y,
                                     int out_features,
                                     int in_features,
                                     int batch_size,
                                     cudaDataType_t output_type);

int dispatch_sample_from_logits(std::vector<float>& logits,
                                float temperature,
                                int top_k,
                                float top_p,
                                float repetition_penalty,
                                int no_repeat_ngram_size,
                                const std::vector<int>& history);

bool dispatch_has_degenerate_tail(const std::vector<int>& ids, std::size_t prompt_size);

}  // namespace detail

}  // namespace engine
