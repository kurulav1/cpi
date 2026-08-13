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
#include <random>
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

int env_int_or_default(const char* name, int default_value) {
  std::string raw;
#ifdef _WIN32
  char* dup = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&dup, &len, name) == 0 && dup != nullptr) {
    raw.assign(dup);
    std::free(dup);
  }
#else
  if (const char* ptr = std::getenv(name); ptr) {
    raw.assign(ptr);
  }
#endif
  if (raw.empty()) {
    return default_value;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw.c_str(), &end, 10);
  if (end == raw.c_str() || *end != '\0' ||
      parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

std::size_t env_workspace_bytes_or_default(const char* name, std::size_t default_bytes) {
  const int default_mb = static_cast<int>(default_bytes / (1024 * 1024));
  const int mb = env_int_or_default(name, default_mb);
  if (mb <= 0) {
    return default_bytes;
  }
  return static_cast<std::size_t>(mb) * static_cast<std::size_t>(1024 * 1024);
}

namespace {
bool linear_rowmajor_weight_lt(cublasLtHandle_t handle, std::vector<LtMatmulPlan>* cache,
                               void* workspace, std::size_t workspace_bytes, cudaStream_t stream,
                               const void* d_w_rowmajor, const void* d_x, void* d_y,
                               int out_features, int in_features, int batch_size,
                               cudaDataType_t output_type) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  // fp16 accumulation, for prefill GEMMs only (batch_size > 1).
  //
  // On GeForce parts, fp16 tensor ops that accumulate in fp32 run at roughly half the rate of
  // ones that accumulate in fp16, and prefill is dominated by these GEMMs. The precision cost is
  // small (logit differences well under a percent) and matches what ggml does for fp16 matmuls.
  //
  // Decode deliberately keeps fp32: at batch 1 the projections are memory-bound GEMVs, so fp16
  // accumulation buys nothing there, and decode is where a logit error selects the wrong token.
  //
  // CPI_GEMM_FP32_ACC=1 forces fp32 everywhere.
  static const bool force_fp32 = [] {
    const char* e = std::getenv("CPI_GEMM_FP32_ACC");
    return e && *e == '1';
  }();
  const bool fp16_acc = !force_fp32 && batch_size > 1 && output_type == CUDA_R_16F;
  const cublasComputeType_t compute_type = fp16_acc ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;

  LtMatmulPlan* plan = nullptr;
  const LtMatmulPlanKey key{out_features, in_features, batch_size, output_type};
  // batch_size is part of the key, and fp16_acc is a pure function of it, so a decode plan can
  // never be handed to prefill or vice versa.
  if (cache) {
    for (auto& cached : *cache) {
      if (cached.ready && cached.key.matches(key)) {
        plan = &cached;
        break;
      }
    }
  }

  if (!plan) {
    LtMatmulPlan created{};
    created.key = key;

    cublasLtMatmulPreference_t pref = nullptr;
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    const cudaDataType_t scale_type = fp16_acc ? CUDA_R_16F : CUDA_R_32F;

    do {
      if (cublasLtMatmulDescCreate(&created.op_desc, compute_type, scale_type) !=
          CUBLAS_STATUS_SUCCESS) {
        break;
      }
      if (cublasLtMatmulDescSetAttribute(created.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                         sizeof(transa)) != CUBLAS_STATUS_SUCCESS) {
        break;
      }
      if (cublasLtMatmulDescSetAttribute(created.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                         sizeof(transb)) != CUBLAS_STATUS_SUCCESS) {
        break;
      }
      if (cublasLtMatrixLayoutCreate(&created.a_desc, CUDA_R_16F, in_features, out_features,
                                     in_features) != CUBLAS_STATUS_SUCCESS ||
          cublasLtMatrixLayoutCreate(&created.b_desc, CUDA_R_16F, in_features, batch_size,
                                     in_features) != CUBLAS_STATUS_SUCCESS ||
          cublasLtMatrixLayoutCreate(&created.c_desc, output_type, out_features, batch_size,
                                     out_features) != CUBLAS_STATUS_SUCCESS) {
        break;
      }
      if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
        break;
      }
      if (cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes,
                                               sizeof(workspace_bytes)) != CUBLAS_STATUS_SUCCESS) {
        break;
      }

      int returned = 0;
      if (cublasLtMatmulAlgoGetHeuristic(handle, created.op_desc, created.a_desc, created.b_desc,
                                         created.c_desc, created.c_desc, pref, 1,
                                         &created.heuristic, &returned) != CUBLAS_STATUS_SUCCESS ||
          returned == 0) {
        break;
      }
      created.ready = true;
    } while (false);

    if (pref) {
      cublasLtMatmulPreferenceDestroy(pref);
    }
    if (!created.ready) {
      if (created.c_desc) {
        cublasLtMatrixLayoutDestroy(created.c_desc);
      }
      if (created.b_desc) {
        cublasLtMatrixLayoutDestroy(created.b_desc);
      }
      if (created.a_desc) {
        cublasLtMatrixLayoutDestroy(created.a_desc);
      }
      if (created.op_desc) {
        cublasLtMatmulDescDestroy(created.op_desc);
      }
      return false;
    }

    if (cache) {
      cache->push_back(created);
      plan = &cache->back();
    } else {
      return false;
    }
  }

  // alpha/beta must match the scale type, not the data type: with CUDA_R_16F they are read as
  // __half, and handing cuBLAS a float* there reinterprets the bits as garbage.
  const __half alpha_h = __float2half(1.0f);
  const __half beta_h = __float2half(0.0f);
  const void* alpha_p =
      fp16_acc ? static_cast<const void*>(&alpha_h) : static_cast<const void*>(&alpha);
  const void* beta_p =
      fp16_acc ? static_cast<const void*>(&beta_h) : static_cast<const void*>(&beta);

  return cublasLtMatmul(handle, plan->op_desc, alpha_p, d_w_rowmajor, plan->a_desc, d_x,
                        plan->b_desc, beta_p, d_y, plan->c_desc, d_y, plan->c_desc,
                        &plan->heuristic.algo, workspace, workspace_bytes,
                        stream) == CUBLAS_STATUS_SUCCESS;
}

void linear_rowmajor_weight(cublasHandle_t handle, cublasLtHandle_t lt_handle,
                            std::vector<LtMatmulPlan>* lt_cache, void* lt_workspace,
                            std::size_t lt_workspace_bytes, cudaStream_t stream,
                            const void* d_w_rowmajor, const void* d_x, void* d_y, int out_features,
                            int in_features, int batch_size, cudaDataType_t output_type) {
  const bool allow_lt = output_type != CUDA_R_32F;
  if (allow_lt && lt_handle &&
      linear_rowmajor_weight_lt(lt_handle, lt_cache, lt_workspace, lt_workspace_bytes, stream,
                                d_w_rowmajor, d_x, d_y, out_features, in_features, batch_size,
                                output_type)) {
    return;
  }

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  /*
   * HF weights are row-major [out_features, in_features].
   * Treat raw memory as column-major [in_features, out_features], then apply transpose:
   * y[out,1] = W^T[out,in] * x[in,1].
   */
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features,
                            &alpha, d_w_rowmajor, CUDA_R_16F, in_features, d_x, CUDA_R_16F,
                            in_features, &beta, d_y, output_type, out_features, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void require_tensor_bytes(const model::WeightLoader& weights, const std::string& name,
                          std::size_t expected_bytes) {
  if (!weights.has_tensor(name)) {
    CPI_THROW("missing tensor: " + name);
  }
  const std::size_t got = weights.tensor_bytes(name);
  if (got != expected_bytes) {
    CPI_THROW("tensor size mismatch for " + name + ": expected " + std::to_string(expected_bytes) +
              " bytes, got " + std::to_string(got) + " bytes");
  }
}

void require_fp16_or_packed_lowbit_bytes(const model::WeightLoader& weights,
                                         const std::string& name, std::size_t expected_fp16_bytes,
                                         std::size_t expected_int8_bytes,
                                         std::size_t expected_int4_bytes,
                                         std::size_t expected_scale_bytes = sizeof(float)) {
  if (weights.has_tensor(name)) {
    require_tensor_bytes(weights, name, expected_fp16_bytes);
    return;
  }
  const std::string q8name = int8_tensor_name(name);
  const std::string q4name = int4_tensor_name(name);
  const std::string sname = quant_scale_name(name);
  const bool has_q8 = weights.has_tensor(q8name) && weights.has_tensor(sname);
  const bool has_q4 = weights.has_tensor(q4name) && weights.has_tensor(sname);
  if (!has_q8 && !has_q4) {
    CPI_THROW("missing tensor: " + name + " (or packed int8/int4 alternative)");
  }
  if (has_q8) {
    require_tensor_bytes(weights, q8name, expected_int8_bytes);
  } else {
    require_tensor_bytes(weights, q4name, expected_int4_bytes);
  }
  const std::size_t got_scale = weights.tensor_bytes(sname);
  if (got_scale != sizeof(float) && got_scale != expected_scale_bytes) {
    CPI_THROW("tensor size mismatch for " + sname + ": expected " +
              std::to_string(expected_scale_bytes) + " or " + std::to_string(sizeof(float)) +
              " bytes, got " + std::to_string(got_scale) + " bytes");
  }
}

struct AttentionDims {
  int q_hidden = 0;
  int head_dim = 0;
  int kv_hidden = 0;
};

int infer_mat_rows(const model::WeightLoader& weights, const std::string& name, int cols) {
  if (cols <= 0) {
    CPI_THROW("invalid infer_mat_rows cols for tensor: " + name);
  }
  if (weights.has_tensor(name)) {
    const std::size_t bytes = weights.tensor_bytes(name);
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(__half);
    if (row_bytes == 0 || (bytes % row_bytes) != 0) {
      CPI_THROW("tensor size mismatch for " + name + ": not divisible by fp16 row bytes");
    }
    return static_cast<int>(bytes / row_bytes);
  }
  const std::string q8name = int8_tensor_name(name);
  if (weights.has_tensor(q8name)) {
    const std::size_t bytes = weights.tensor_bytes(q8name);
    const std::size_t row_bytes = static_cast<std::size_t>(cols);
    if (row_bytes == 0 || (bytes % row_bytes) != 0) {
      CPI_THROW("tensor size mismatch for " + q8name + ": not divisible by int8 row bytes");
    }
    return static_cast<int>(bytes / row_bytes);
  }
  const std::string q4name = int4_tensor_name(name);
  if (weights.has_tensor(q4name)) {
    const int packed_cols = (cols + 1) / 2;
    const std::size_t bytes = weights.tensor_bytes(q4name);
    const std::size_t row_bytes = static_cast<std::size_t>(packed_cols);
    if (row_bytes == 0 || (bytes % row_bytes) != 0) {
      CPI_THROW("tensor size mismatch for " + q4name + ": not divisible by int4 packed row bytes");
    }
    return static_cast<int>(bytes / row_bytes);
  }
  CPI_THROW("missing tensor: " + name + " (or packed int8/int4 alternative)");
}

AttentionDims infer_attention_dims(const model::WeightLoader& weights,
                                   const model::LlamaConfig& cfg) {
  if (cfg.num_heads <= 0 || cfg.num_kv_heads <= 0 || (cfg.num_heads % cfg.num_kv_heads) != 0) {
    CPI_THROW("invalid attention head config in header");
  }
  const int hidden = cfg.hidden_size;
  if (cfg.num_layers <= 0 || hidden <= 0) {
    CPI_THROW("invalid model config in header");
  }
  const std::string p = "layers.0";
  const int q_hidden = infer_mat_rows(weights, p + ".attention.wq", hidden);
  const int kv_hidden = infer_mat_rows(weights, p + ".attention.wk", hidden);
  const int vv_hidden = infer_mat_rows(weights, p + ".attention.wv", hidden);
  if (q_hidden <= 0 || (q_hidden % cfg.num_heads) != 0) {
    CPI_THROW("invalid attention.wq shape: q_hidden must be positive and divisible by num_heads");
  }
  if (kv_hidden <= 0 || kv_hidden != vv_hidden) {
    CPI_THROW("invalid attention.wk/wv shape: rows must be positive and equal");
  }
  const int head_dim = q_hidden / cfg.num_heads;
  if (head_dim <= 0) {
    CPI_THROW("invalid inferred attention head_dim");
  }
  if (kv_hidden != cfg.num_kv_heads * head_dim) {
    CPI_THROW("invalid attention dims: wk rows must equal num_kv_heads * (wq_rows / num_heads)");
  }
  return AttentionDims{q_hidden, head_dim, kv_hidden};
}

void validate_tensor_layout(const model::WeightLoader& weights) {
  const auto& cfg = weights.config();
  if (cfg.hidden_size <= 0 || cfg.intermediate_size <= 0 || cfg.vocab_size <= 0 ||
      cfg.num_layers <= 0 || cfg.num_heads <= 0 || cfg.num_kv_heads <= 0 ||
      cfg.num_heads % cfg.num_kv_heads != 0) {
    CPI_THROW("invalid model config in header");
  }

  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const bool is_moe = cfg.is_moe();
  const int moe_experts = std::max(0, cfg.num_local_experts);
  const int expert_inter = cfg.effective_expert_intermediate_size() > 0
                               ? cfg.effective_expert_intermediate_size()
                               : inter;
  const AttentionDims attn_dims = infer_attention_dims(weights, cfg);
  const int q_hidden = attn_dims.q_hidden;
  const int kv_hidden = attn_dims.kv_hidden;
  constexpr std::size_t hsz = sizeof(__half);

  require_tensor_bytes(
      weights, "tok_embeddings.weight",
      static_cast<std::size_t>(cfg.vocab_size) * static_cast<std::size_t>(hidden) * hsz);
  require_tensor_bytes(weights, "norm.weight", static_cast<std::size_t>(hidden) * hsz);

  if (weights.has_tensor("output.weight")) {
    require_tensor_bytes(
        weights, "output.weight",
        static_cast<std::size_t>(cfg.vocab_size) * static_cast<std::size_t>(hidden) * hsz);
  }
  if (weights.has_tensor("output.bias")) {
    require_tensor_bytes(weights, "output.bias", static_cast<std::size_t>(cfg.vocab_size) * hsz);
  }

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer);
    require_fp16_or_packed_lowbit_bytes(
        weights, p + ".attention_norm.weight", static_cast<std::size_t>(hidden) * hsz,
        static_cast<std::size_t>(hidden), static_cast<std::size_t>((hidden + 1) / 2));
    if (weights.has_tensor(p + ".attention_norm.bias")) {
      require_tensor_bytes(weights, p + ".attention_norm.bias",
                           static_cast<std::size_t>(hidden) * hsz);
    }
    require_fp16_or_packed_lowbit_bytes(
        weights, p + ".attention.wq",
        static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden) * hsz,
        static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden),
        static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>((hidden + 1) / 2));
    require_fp16_or_packed_lowbit_bytes(
        weights, p + ".attention.wk",
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden) * hsz,
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden),
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>((hidden + 1) / 2));
    require_fp16_or_packed_lowbit_bytes(
        weights, p + ".attention.wv",
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden) * hsz,
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>(hidden),
        static_cast<std::size_t>(kv_hidden) * static_cast<std::size_t>((hidden + 1) / 2));
    require_fp16_or_packed_lowbit_bytes(
        weights, p + ".attention.wo",
        static_cast<std::size_t>(hidden) * static_cast<std::size_t>(q_hidden) * hsz,
        static_cast<std::size_t>(hidden) * static_cast<std::size_t>(q_hidden),
        static_cast<std::size_t>(hidden) * static_cast<std::size_t>((q_hidden + 1) / 2));
    if (cfg.has_qkv_bias && weights.has_tensor(p + ".attention.bqkv")) {
      require_tensor_bytes(weights, p + ".attention.bqkv",
                           static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * hsz);
    }
    if (weights.has_tensor(p + ".attention.bo")) {
      require_tensor_bytes(weights, p + ".attention.bo", static_cast<std::size_t>(hidden) * hsz);
    }
    require_fp16_or_packed_lowbit_bytes(
        weights, p + ".ffn_norm.weight", static_cast<std::size_t>(hidden) * hsz,
        static_cast<std::size_t>(hidden), static_cast<std::size_t>((hidden + 1) / 2));
    if (weights.has_tensor(p + ".ffn_norm.bias")) {
      require_tensor_bytes(weights, p + ".ffn_norm.bias", static_cast<std::size_t>(hidden) * hsz);
    }
    if (!is_moe) {
      require_fp16_or_packed_lowbit_bytes(
          weights, p + ".feed_forward.w1",
          static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden) * hsz,
          static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden),
          static_cast<std::size_t>(inter) * static_cast<std::size_t>((hidden + 1) / 2),
          static_cast<std::size_t>(inter) * sizeof(float));
      require_fp16_or_packed_lowbit_bytes(
          weights, p + ".feed_forward.w2",
          static_cast<std::size_t>(hidden) * static_cast<std::size_t>(inter) * hsz,
          static_cast<std::size_t>(hidden) * static_cast<std::size_t>(inter),
          static_cast<std::size_t>(hidden) * static_cast<std::size_t>((inter + 1) / 2),
          static_cast<std::size_t>(hidden) * sizeof(float));
      require_fp16_or_packed_lowbit_bytes(
          weights, p + ".feed_forward.w3",
          static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden) * hsz,
          static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden),
          static_cast<std::size_t>(inter) * static_cast<std::size_t>((hidden + 1) / 2),
          static_cast<std::size_t>(inter) * sizeof(float));
    } else {
      if (moe_experts <= 0) {
        CPI_THROW("invalid MoE config: num_local_experts must be > 0");
      }
      require_fp16_or_packed_lowbit_bytes(
          weights, p + ".feed_forward.router",
          static_cast<std::size_t>(moe_experts) * static_cast<std::size_t>(hidden) * hsz,
          static_cast<std::size_t>(moe_experts) * static_cast<std::size_t>(hidden),
          static_cast<std::size_t>(moe_experts) * static_cast<std::size_t>((hidden + 1) / 2),
          static_cast<std::size_t>(moe_experts) * sizeof(float));
      for (int expert = 0; expert < moe_experts; ++expert) {
        const std::string ebase = p + ".feed_forward.experts." + std::to_string(expert);
        require_fp16_or_packed_lowbit_bytes(
            weights, ebase + ".w1",
            static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>(hidden) * hsz,
            static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>(hidden),
            static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>((hidden + 1) / 2),
            static_cast<std::size_t>(expert_inter) * sizeof(float));
        require_fp16_or_packed_lowbit_bytes(
            weights, ebase + ".w2",
            static_cast<std::size_t>(hidden) * static_cast<std::size_t>(expert_inter) * hsz,
            static_cast<std::size_t>(hidden) * static_cast<std::size_t>(expert_inter),
            static_cast<std::size_t>(hidden) * static_cast<std::size_t>((expert_inter + 1) / 2),
            static_cast<std::size_t>(hidden) * sizeof(float));
        require_fp16_or_packed_lowbit_bytes(
            weights, ebase + ".w3",
            static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>(hidden) * hsz,
            static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>(hidden),
            static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>((hidden + 1) / 2),
            static_cast<std::size_t>(expert_inter) * sizeof(float));
      }
    }
  }

  // Optional TQ3 metadata/tensors (TurboQuant 3-bit).
  const bool has_tq3_codebook = weights.has_tensor("tq3_codebook");
  const bool has_tq3_signs = weights.has_tensor("tq3_signs_hidden");
  if (has_tq3_codebook != has_tq3_signs) {
    CPI_THROW("incomplete TQ3 metadata: expected both tq3_codebook and tq3_signs_hidden");
  }
  if (has_tq3_codebook) {
    require_tensor_bytes(weights, "tq3_codebook", static_cast<std::size_t>(8) * sizeof(__half));
    require_tensor_bytes(weights, "tq3_signs_hidden",
                         static_cast<std::size_t>(hidden) * sizeof(std::int8_t));
  }

  int tq_objective = 0;
  if (weights.has_tensor("tq_objective")) {
    require_tensor_bytes(weights, "tq_objective", sizeof(std::int32_t));
    std::int32_t objective = 0;
    std::memcpy(&objective, weights.tensor_data("tq_objective"), sizeof(std::int32_t));
    tq_objective = static_cast<int>(objective);
    if (tq_objective != 0 && tq_objective != 1) {
      CPI_THROW("invalid tq_objective value: expected 0 (mse) or 1 (prod)");
    }
  }

  int qjl_dim = 0;
  if (tq_objective == 1) {
    if (!weights.has_tensor("tq_qjl_dim") || !weights.has_tensor("tq_qjl_seed") ||
        !weights.has_tensor("tq_qjl_indices_hidden") ||
        !weights.has_tensor("tq_qjl_signs_hidden")) {
      CPI_THROW(
          "incomplete Qprod metadata: expected "
          "tq_qjl_dim/tq_qjl_seed/tq_qjl_indices_hidden/tq_qjl_signs_hidden");
    }
    require_tensor_bytes(weights, "tq_qjl_dim", sizeof(std::int32_t));
    require_tensor_bytes(weights, "tq_qjl_seed", sizeof(std::int32_t));
    std::int32_t qjl_dim_i32 = 0;
    std::memcpy(&qjl_dim_i32, weights.tensor_data("tq_qjl_dim"), sizeof(std::int32_t));
    qjl_dim = static_cast<int>(qjl_dim_i32);
    if (qjl_dim <= 0 || qjl_dim > hidden) {
      CPI_THROW("invalid tq_qjl_dim: expected in [1, hidden_size]");
    }
    require_tensor_bytes(weights, "tq_qjl_indices_hidden",
                         static_cast<std::size_t>(qjl_dim) * sizeof(std::int32_t));
    require_tensor_bytes(weights, "tq_qjl_signs_hidden",
                         static_cast<std::size_t>(qjl_dim) * sizeof(std::int8_t));
  }

  const std::size_t tq3_words_per_row = static_cast<std::size_t>((hidden + 9) / 10);
  const std::size_t qjl_words_per_row = static_cast<std::size_t>((qjl_dim + 31) / 32);
  const auto validate_tq3_pair_if_present = [&](const std::string& base, int out_rows) {
    const std::string packed = base + ".tq3";
    const std::string scales = base + ".tq3s";
    const bool has_packed = weights.has_tensor(packed);
    const bool has_scales = weights.has_tensor(scales);
    if (!has_packed && !has_scales) {
      return;
    }
    if (!has_packed || !has_scales) {
      CPI_THROW("incomplete TQ3 tensor pair: expected both " + packed + " and " + scales);
    }
    const std::size_t expected_packed =
        static_cast<std::size_t>(out_rows) * tq3_words_per_row * sizeof(std::uint32_t);
    const std::size_t expected_scales = static_cast<std::size_t>(out_rows) * sizeof(__half);
    require_tensor_bytes(weights, packed, expected_packed);
    require_tensor_bytes(weights, scales, expected_scales);

    if (tq_objective == 1) {
      const std::string residual_bits = base + ".tq3r";
      const std::string residual_scales = base + ".tq3rs";
      if (!weights.has_tensor(residual_bits) || !weights.has_tensor(residual_scales)) {
        CPI_THROW("incomplete Qprod residual pair: expected both " + residual_bits + " and " +
                  residual_scales);
      }
      const std::size_t expected_rbits =
          static_cast<std::size_t>(out_rows) * qjl_words_per_row * sizeof(std::uint32_t);
      const std::size_t expected_rscales = static_cast<std::size_t>(out_rows) * sizeof(__half);
      require_tensor_bytes(weights, residual_bits, expected_rbits);
      require_tensor_bytes(weights, residual_scales, expected_rscales);
    }
  };

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer);
    validate_tq3_pair_if_present(p + ".attention.wq", q_hidden);
    validate_tq3_pair_if_present(p + ".attention.wk", kv_hidden);
    validate_tq3_pair_if_present(p + ".attention.wv", kv_hidden);
    validate_tq3_pair_if_present(p + ".attention.wo", hidden);
    if (!is_moe) {
      validate_tq3_pair_if_present(p + ".feed_forward.w1", inter);
      validate_tq3_pair_if_present(p + ".feed_forward.w3", inter);
    }
  }
}

}  // namespace

namespace detail {

void dispatch_linear_rowmajor_weight(cublasHandle_t handle, cublasLtHandle_t lt_handle,
                                     std::vector<LtMatmulPlan>* lt_cache, void* lt_workspace,
                                     std::size_t lt_workspace_bytes, cudaStream_t stream,
                                     const void* d_w_rowmajor, const void* d_x, void* d_y,
                                     int out_features, int in_features, int batch_size,
                                     cudaDataType_t output_type) {
  linear_rowmajor_weight(handle, lt_handle, lt_cache, lt_workspace, lt_workspace_bytes, stream,
                         d_w_rowmajor, d_x, d_y, out_features, in_features, batch_size,
                         output_type);
}

void launch_gated_glu(bool use_gelu, const __half* gate, const __half* up, __half* out, int n,
                      cudaStream_t stream) {
  if (use_gelu) {
    kernels::launch_gelu_mul(gate, up, out, n, stream);
  } else {
    kernels::launch_silu_mul(gate, up, out, n, stream);
  }
}

}  // namespace detail

void LlamaEngine::launch_norm(const void* x, const void* weight, const void* bias, void* y,
                              int rows, int cols) {
  const auto& cfg = weights_.config();
  const float eps = cfg.norm_eps > 0.0f ? cfg.norm_eps : 1e-5f;
  if (cfg.use_layernorm) {
    kernels::launch_layernorm(static_cast<const __half*>(x), static_cast<const __half*>(weight),
                              static_cast<const __half*>(bias), static_cast<__half*>(y), rows, cols,
                              eps, compute_stream_);
    return;
  }
  kernels::launch_rmsnorm(static_cast<const __half*>(x), static_cast<const __half*>(weight),
                          static_cast<__half*>(y), rows, cols, eps, compute_stream_);
  maybe_add_half_bias(y, bias, rows, cols);
}

void LlamaEngine::maybe_add_half_bias(void* out, const void* bias, int rows, int cols) {
  if (!out || !bias || rows <= 0 || cols <= 0) {
    return;
  }
  if (rows == 1) {
    kernels::launch_add_inplace(static_cast<__half*>(out), static_cast<const __half*>(bias), cols,
                                compute_stream_);
    return;
  }
  kernels::launch_add_bias_broadcast(static_cast<__half*>(out), static_cast<const __half*>(bias),
                                     rows, cols, compute_stream_);
}

LlamaEngine::~LlamaEngine() {
  free_load_staging();
  destroy_greedy_decode_graph();
  destroy_logits_decode_graph();
  for (auto& plan : lt_plan_cache_) {
    if (plan.c_desc) {
      cublasLtMatrixLayoutDestroy(plan.c_desc);
      plan.c_desc = nullptr;
    }
    if (plan.b_desc) {
      cublasLtMatrixLayoutDestroy(plan.b_desc);
      plan.b_desc = nullptr;
    }
    if (plan.a_desc) {
      cublasLtMatrixLayoutDestroy(plan.a_desc);
      plan.a_desc = nullptr;
    }
    if (plan.op_desc) {
      cublasLtMatmulDescDestroy(plan.op_desc);
      plan.op_desc = nullptr;
    }
    plan.ready = false;
  }
  lt_plan_cache_.clear();
  if (cublas_) {
    cublasDestroy(cublas_);
  }
  if (cublas_lt_) {
    cublasLtDestroy(cublas_lt_);
    cublas_lt_ = nullptr;
  }
  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
    compute_stream_ = nullptr;
  }
  if (transfer_stream_) {
    cudaStreamDestroy(transfer_stream_);
    transfer_stream_ = nullptr;
  }
  for (auto& ev : streaming_ready_) {
    if (ev) {
      cudaEventDestroy(ev);
      ev = nullptr;
    }
  }
  for (auto& ev : streaming_consumed_) {
    if (ev) {
      cudaEventDestroy(ev);
      ev = nullptr;
    }
  }
  if (benchmark_transfer_start_) {
    cudaEventDestroy(benchmark_transfer_start_);
    benchmark_transfer_start_ = nullptr;
  }
  if (benchmark_transfer_end_) {
    cudaEventDestroy(benchmark_transfer_end_);
    benchmark_transfer_end_ = nullptr;
  }

  auto free_ptr = [](void*& p) {
    if (p) {
      cudaFree(p);
      p = nullptr;
    }
  };

  free_ptr(d_tok_embeddings_);
  free_ptr(d_norm_out_);
  free_ptr(d_norm_out_bias_);
  free_ptr(d_attn_scores_);
  if (d_gemm_ptrs_) {
    cudaFree(d_gemm_ptrs_);
    d_gemm_ptrs_ = nullptr;
  }
  attn_scores_bytes_ = 0;
  free_ptr(d_lm_head_);
  if (d_lm_head_i8_) {
    cudaFree(d_lm_head_i8_);
    d_lm_head_i8_ = nullptr;
  }
  if (d_lm_head_i8_scales_) {
    cudaFree(d_lm_head_i8_scales_);
    d_lm_head_i8_scales_ = nullptr;
  }
  lm_head_int8_ = false;
  free_ptr(d_lm_head_bias_);
  free_ptr(lt_workspace_);
  if (d_token_id_) {
    cudaFree(d_token_id_);
    d_token_id_ = nullptr;
  }
  free_ptr(d_x_);
  free_ptr(d_x_norm_);
  free_ptr(d_qkv_);
  d_q_ = nullptr;
  d_k_ = nullptr;
  d_v_ = nullptr;
  free_ptr(d_prefill_q_);
  free_ptr(d_prefill_k_);
  free_ptr(d_prefill_v_);
  free_ptr(d_att_);
  free_ptr(d_ff13_);
  d_ff1_ = nullptr;
  d_ff2_ = nullptr;
  free_ptr(d_prefill_ff1_);
  free_ptr(d_prefill_ff2_);
  free_ptr(d_ff3_);
  if (d_prefill_i8_) {
    cudaFree(d_prefill_i8_);
    d_prefill_i8_ = nullptr;
  }
  if (d_prefill_wdq_) {
    cudaFree(d_prefill_wdq_);
    d_prefill_wdq_ = nullptr;
    d_prefill_wdq_bytes_ = 0;
  }
  if (d_q8_x_) {
    cudaFree(d_q8_x_);
    d_q8_x_ = nullptr;
    q8_x_bytes_ = 0;
  }
  if (d_q8_scale_) {
    cudaFree(d_q8_scale_);
    d_q8_scale_ = nullptr;
  }
  if (d_q8_sum_) {
    cudaFree(d_q8_sum_);
    d_q8_sum_ = nullptr;
    q8_meta_bytes_ = 0;
  }
  if (d_prefill_i8_scales_) {
    cudaFree(d_prefill_i8_scales_);
    d_prefill_i8_scales_ = nullptr;
  }
  if (d_prefill_perm8_scales_) {
    cudaFree(d_prefill_perm8_scales_);
    d_prefill_perm8_scales_ = nullptr;
  }
  free_ptr(d_logits_);
  if (d_argmax_) {
    cudaFree(d_argmax_);
    d_argmax_ = nullptr;
  }
  const auto free_typed = [](auto*& p) {
    if (p) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_typed(d_argmax_part_val_);
  free_typed(d_argmax_part_idx_);
  free_typed(d_topk_part_val_);
  free_typed(d_topk_part_idx_);
  free_typed(d_topk_val_);
  free_typed(d_topk_idx_);
  free_typed(d_cand_idx_);
  free_typed(d_cand_val_);
  free_typed(d_cand_count_);
  device_topk_ready_ = false;
  if (d_decode_position_) {
    cudaFree(d_decode_position_);
    d_decode_position_ = nullptr;
  }
  if (d_rope_cos_) {
    cudaFree(d_rope_cos_);
    d_rope_cos_ = nullptr;
  }
  if (d_rope_sin_) {
    cudaFree(d_rope_sin_);
    d_rope_sin_ = nullptr;
  }
  if (d_attn_chunk_m_) {
    cudaFree(d_attn_chunk_m_);
    d_attn_chunk_m_ = nullptr;
  }
  if (d_attn_chunk_l_) {
    cudaFree(d_attn_chunk_l_);
    d_attn_chunk_l_ = nullptr;
  }
  if (d_attn_chunk_o_) {
    cudaFree(d_attn_chunk_o_);
    d_attn_chunk_o_ = nullptr;
  }
  free_ptr(d_moe_router_w_);
  free_ptr(d_moe_router_logits_);
  if (d_moe_router_w_q_) {
    cudaFree(d_moe_router_w_q_);
    d_moe_router_w_q_ = nullptr;
  }
  if (d_moe_router_scales_) {
    cudaFree(d_moe_router_scales_);
    d_moe_router_scales_ = nullptr;
  }
  free_ptr(d_moe_w1_);
  free_ptr(d_moe_w2_);
  free_ptr(d_moe_w3_);
  if (d_moe_w1_q_) {
    cudaFree(d_moe_w1_q_);
    d_moe_w1_q_ = nullptr;
  }
  if (d_moe_w2_q_) {
    cudaFree(d_moe_w2_q_);
    d_moe_w2_q_ = nullptr;
  }
  if (d_moe_w3_q_) {
    cudaFree(d_moe_w3_q_);
    d_moe_w3_q_ = nullptr;
  }
  if (d_moe_s_w1_) {
    cudaFree(d_moe_s_w1_);
    d_moe_s_w1_ = nullptr;
  }
  if (d_moe_s_w2_) {
    cudaFree(d_moe_s_w2_);
    d_moe_s_w2_ = nullptr;
  }
  if (d_moe_s_w3_) {
    cudaFree(d_moe_s_w3_);
    d_moe_s_w3_ = nullptr;
  }
  if (d_moe_topk_idx_) {
    cudaFree(d_moe_topk_idx_);
    d_moe_topk_idx_ = nullptr;
  }
  if (d_moe_topk_prob_) {
    cudaFree(d_moe_topk_prob_);
    d_moe_topk_prob_ = nullptr;
  }
  free_ptr(d_k_cache_);
  free_ptr(d_v_cache_);
  if (d_block_table_) {
    cudaFree(d_block_table_);
    d_block_table_ = nullptr;
  }  // paged KV device block table
  if (d_batch_positions_) {
    cudaFree(d_batch_positions_);
    d_batch_positions_ = nullptr;
  }
  if (d_batch_seq_lens_) {
    cudaFree(d_batch_seq_lens_);
    d_batch_seq_lens_ = nullptr;
  }
  if (d_batch_block_tables_) {
    cudaFree(d_batch_block_tables_);
    d_batch_block_tables_ = nullptr;
  }
  if (d_batch_kv_slots_) {
    cudaFree(d_batch_kv_slots_);
    d_batch_kv_slots_ = nullptr;
  }
  if (d_batch_logits_) {
    cudaFree(d_batch_logits_);
    d_batch_logits_ = nullptr;
  }
  if (h_batch_logits_) {
    cudaFreeHost(h_batch_logits_);
    h_batch_logits_ = nullptr;
    h_batch_logits_cap_ = 0;
  }
  if (d_penalty_) {
    cudaFree(d_penalty_);
    d_penalty_ = nullptr;
    d_penalty_cap_ = 0;
  }
  if (d_seen_ids_) {
    cudaFree(d_seen_ids_);
    d_seen_ids_ = nullptr;
  }
  if (d_seen_rows_) {
    cudaFree(d_seen_rows_);
    d_seen_rows_ = nullptr;
  }
  d_seen_cap_ = 0;
  if (d_argmax_blocked_) {
    cudaFree(d_argmax_blocked_);
    d_argmax_blocked_ = nullptr;
  }
  if (d_argmax_out_) {
    cudaFree(d_argmax_out_);
    d_argmax_out_ = nullptr;
  }
  d_argmax_cap_ = 0;
  if (d_bs_scratch_m_) {
    cudaFree(d_bs_scratch_m_);
    d_bs_scratch_m_ = nullptr;
  }
  if (d_bs_scratch_l_) {
    cudaFree(d_bs_scratch_l_);
    d_bs_scratch_l_ = nullptr;
  }
  if (d_bs_scratch_o_) {
    cudaFree(d_bs_scratch_o_);
    d_bs_scratch_o_ = nullptr;
  }
  if (h_k_cache_) {
    cudaFreeHost(h_k_cache_);
    h_k_cache_ = nullptr;
  }
  if (h_v_cache_) {
    cudaFreeHost(h_v_cache_);
    h_v_cache_ = nullptr;
  }
  if (d_k_cache_i4_) {
    cudaFree(d_k_cache_i4_);
    d_k_cache_i4_ = nullptr;
  }
  if (d_v_cache_i4_) {
    cudaFree(d_v_cache_i4_);
    d_v_cache_i4_ = nullptr;
  }
  if (d_k_scales_) {
    cudaFree(d_k_scales_);
    d_k_scales_ = nullptr;
  }
  if (d_kv_sink_k_) {
    cudaFree(d_kv_sink_k_);
    d_kv_sink_k_ = nullptr;
  }
  if (d_kv_sink_v_) {
    cudaFree(d_kv_sink_v_);
    d_kv_sink_v_ = nullptr;
  }
  if (d_kv_ring_k_) {
    cudaFree(d_kv_ring_k_);
    d_kv_ring_k_ = nullptr;
  }
  if (d_kv_ring_v_) {
    cudaFree(d_kv_ring_v_);
    d_kv_ring_v_ = nullptr;
  }
  if (d_tq3_codebook_) {
    cudaFree(d_tq3_codebook_);
    d_tq3_codebook_ = nullptr;
  }
  if (d_tq3_signs_) {
    cudaFree(d_tq3_signs_);
    d_tq3_signs_ = nullptr;
  }
  if (d_tq_qjl_indices_) {
    cudaFree(d_tq_qjl_indices_);
    d_tq_qjl_indices_ = nullptr;
  }
  if (d_tq_qjl_signs_) {
    cudaFree(d_tq_qjl_signs_);
    d_tq_qjl_signs_ = nullptr;
  }
  if (d_tq_qjl_x_bits_) {
    cudaFree(d_tq_qjl_x_bits_);
    d_tq_qjl_x_bits_ = nullptr;
  }
  if (d_x_tq3_) {
    cudaFree(d_x_tq3_);
    d_x_tq3_ = nullptr;
  }
  for (auto& tq : layer_cache_tq3_) {
    if (tq.wqkv) {
      cudaFree(tq.wqkv);
    }
    if (tq.wo) {
      cudaFree(tq.wo);
    }
    if (tq.w13) {
      cudaFree(tq.w13);
    }
    if (tq.s_wqkv) {
      cudaFree(tq.s_wqkv);
    }
    if (tq.s_wo) {
      cudaFree(tq.s_wo);
    }
    if (tq.s_w13) {
      cudaFree(tq.s_w13);
    }
    if (tq.r_wqkv) {
      cudaFree(tq.r_wqkv);
    }
    if (tq.r_wo) {
      cudaFree(tq.r_wo);
    }
    if (tq.r_w13) {
      cudaFree(tq.r_w13);
    }
    if (tq.rs_wqkv) {
      cudaFree(tq.rs_wqkv);
    }
    if (tq.rs_wo) {
      cudaFree(tq.rs_wo);
    }
    if (tq.rs_w13) {
      cudaFree(tq.rs_w13);
    }
  }
  layer_cache_tq3_.clear();
  if (d_v_scales_) {
    cudaFree(d_v_scales_);
    d_v_scales_ = nullptr;
  }

  free_ptr(layer_weights_.wqkv);
  free_ptr(layer_weights_.wo);
  free_ptr(layer_weights_.bo);
  free_ptr(layer_weights_.w13);
  free_ptr(layer_weights_.w2);
  free_ptr(layer_weights_.norm_att);
  free_ptr(layer_weights_.norm_ffn);
  free_ptr(layer_weights_.norm_att_bias);
  free_ptr(layer_weights_.norm_ffn_bias);
  free_ptr(layer_weights_.bqkv);
  if (layer_weights_i8_.w1) cudaFree(layer_weights_i8_.w1);
  if (layer_weights_i8_.w2) cudaFree(layer_weights_i8_.w2);
  if (layer_weights_i8_.w3) cudaFree(layer_weights_i8_.w3);
  if (layer_weights_i8_.s_w1) cudaFree(layer_weights_i8_.s_w1);
  if (layer_weights_i8_.s_w2) cudaFree(layer_weights_i8_.s_w2);
  if (layer_weights_i8_.s_w3) cudaFree(layer_weights_i8_.s_w3);
  layer_weights_i8_ = {};

  for (auto& lw : streaming_layer_weights_) {
    free_ptr(lw.wqkv);
    free_ptr(lw.wo);
    free_ptr(lw.bo);
    free_ptr(lw.w13);
    free_ptr(lw.w2);
    free_ptr(lw.norm_att);
    free_ptr(lw.norm_ffn);
    free_ptr(lw.norm_att_bias);
    free_ptr(lw.norm_ffn_bias);
    free_ptr(lw.bqkv);
  }
  for (auto& iw : streaming_layer_weights_i8_) {
    if (iw.w1) cudaFree(iw.w1);
    if (iw.w2) cudaFree(iw.w2);
    if (iw.w3) cudaFree(iw.w3);
    if (iw.s_w1) cudaFree(iw.s_w1);
    if (iw.s_w2) cudaFree(iw.s_w2);
    if (iw.s_w3) cudaFree(iw.s_w3);
    iw = {};
  }

  for (auto& lw : layer_cache_) {
    free_ptr(lw.wqkv);
    free_ptr(lw.wo);
    free_ptr(lw.bo);
    free_ptr(lw.w13);
    free_ptr(lw.w2);
    free_ptr(lw.norm_att);
    free_ptr(lw.norm_ffn);
    free_ptr(lw.norm_att_bias);
    free_ptr(lw.norm_ffn_bias);
    free_ptr(lw.bqkv);
  }
  for (auto& iw : layer_cache_i8_) {
    if (iw.w1) cudaFree(iw.w1);
    if (iw.w2) cudaFree(iw.w2);
    if (iw.w3) cudaFree(iw.w3);
    if (iw.s_w1) cudaFree(iw.s_w1);
    if (iw.s_w2) cudaFree(iw.s_w2);
    if (iw.s_w3) cudaFree(iw.s_w3);
    iw = {};
  }

  for (auto& hp : layer_host_pinned_) {
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
  }

  for (auto& hq : layer_host_int8_) {
    if (hq.w1) cudaFreeHost(hq.w1);
    if (hq.w2) cudaFreeHost(hq.w2);
    if (hq.w3) cudaFreeHost(hq.w3);
    if (hq.s_w1) cudaFreeHost(hq.s_w1);
    if (hq.s_w2) cudaFreeHost(hq.s_w2);
    if (hq.s_w3) cudaFreeHost(hq.s_w3);
    hq = {};
  }
}

void LlamaEngine::initialize(const EngineOptions& options) {
  destroy_greedy_decode_graph();
  destroy_logits_decode_graph();
  options_ = options;
  options_.streaming_quant_bits = clamp_streaming_quant_bits(options_.streaming_quant_bits);
  attn_q_hidden_ = 0;
  attn_head_dim_ = 0;
  attn_kv_hidden_ = 0;
  resource_sample_ready_ = false;
  sampled_cpu_percent_ = -1.0;
  sampled_memory_percent_ = -1.0;
  over_limit_active_ = false;
  last_over_limit_log_time_ = std::chrono::steady_clock::time_point{};
  enforce_host_resource_limits("startup.begin");

  const auto startup_begin = std::chrono::steady_clock::now();
  static const bool startup_vram = std::getenv("CPI_STARTUP_PROFILE") != nullptr;
  const auto run_startup_phase = [&](const char* phase, auto&& fn) {
    const auto phase_begin = std::chrono::steady_clock::now();
    fn();
    enforce_host_resource_limits(phase);
    if (options_.verbose) {
      const auto phase_end = std::chrono::steady_clock::now();
      const auto phase_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_begin).count();
      std::cout << "[startup] phase=" << phase << " ms=" << phase_ms;
      if (startup_vram) {
        std::size_t vfree = 0, vtot = 0;
        if (cudaMemGetInfo(&vfree, &vtot) == cudaSuccess) {
          std::cout << " vram_used_gb=" << ((vtot - vfree) / (1024.0 * 1024.0 * 1024.0));
        }
      }
      std::cout << "\n";
    }
  };

  run_startup_phase("open", [&] { weights_.open(options.model_path); });
  run_startup_phase("validate", [&] { validate_tensor_layout(weights_); });

  if (options.max_batch != 1) {
    CPI_THROW("only max_batch=1 is currently supported");
  }

  const auto snap = runtime::collect_system_snapshot();
  if (options_.verbose) {
    std::cout << "[engine] OS: " << snap.os_name << "\n";
    std::cout << "[engine] CUDA devices: " << snap.device_count << "\n";
    for (int i = 0; i < snap.device_count; ++i) {
      std::cout << "  - GPU " << i << ": " << snap.gpu_names[i]
                << " free=" << runtime::format_bytes(snap.gpu_memory[i].free_bytes)
                << " total=" << runtime::format_bytes(snap.gpu_memory[i].total_bytes) << "\n";
    }
  }
  const auto& cfg = weights_.config();
  if (cfg.has_linear_attention()) {
    std::ostringstream oss;
    oss << "native CUDA runtime does not support linear-attention layers yet";
    if (cfg.model_family == model::ModelFamily::Qwen3_5) {
      oss << " (Qwen3.5 metadata was detected successfully, but kernels are not implemented yet)";
    }
    CPI_THROW(oss.str());
  }
  const AttentionDims attn_dims = infer_attention_dims(weights_, cfg);
  attn_q_hidden_ = attn_dims.q_hidden;
  attn_head_dim_ = attn_dims.head_dim;
  attn_kv_hidden_ = attn_dims.kv_hidden;
  has_any_layer_output_bias_ = false;
  has_any_layer_norm_bias_ = false;
  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer);
    if (weights_.has_tensor(p + ".attention.bo")) {
      has_any_layer_output_bias_ = true;
    }
    if (weights_.has_tensor(p + ".attention_norm.bias") ||
        weights_.has_tensor(p + ".ffn_norm.bias")) {
      has_any_layer_norm_bias_ = true;
    }
    if (has_any_layer_output_bias_ && has_any_layer_norm_bias_) {
      break;
    }
  }
  if (options_.verbose) {
    const char* family_str = "unknown";
    switch (cfg.model_family) {
      case model::ModelFamily::LLaMA2:
        family_str = "llama2";
        break;
      case model::ModelFamily::LLaMA3:
        family_str = "llama3";
        break;
      case model::ModelFamily::Mistral:
        family_str = "mistral";
        break;
      case model::ModelFamily::Mixtral:
        family_str = "mixtral";
        break;
      case model::ModelFamily::Phi3:
        family_str = "phi3";
        break;
      case model::ModelFamily::Qwen2:
        family_str = "qwen2";
        break;
      case model::ModelFamily::Qwen3_5:
        family_str = "qwen3_5";
        break;
      case model::ModelFamily::Qwen3:
        family_str = "qwen3";
        break;
      case model::ModelFamily::Gemma:
        family_str = "gemma";
        break;
      default:
        break;
    }
    std::cout << "[engine] model_family=" << family_str << " hidden=" << cfg.hidden_size
              << " attn_hidden=" << attn_q_hidden_ << " layers=" << cfg.num_layers
              << " heads=" << cfg.num_heads << " head_dim=" << attn_head_dim_
              << " kv_heads=" << cfg.num_kv_heads << " vocab=" << cfg.vocab_size;
    if (lowbit_streaming_enabled(options_)) {
      std::cout << " weight_quant=int" << options_.streaming_quant_bits;
    }
    if (cfg.has_qkv_bias) std::cout << " qkv_bias=yes";
    if (cfg.has_qk_norm) std::cout << " qk_norm=yes";
    if (cfg.mlp_gelu) std::cout << " mlp=gelu";
    if (cfg.scale_embeddings) std::cout << " embed_scale=yes";
    if (cfg.tie_word_embeddings) std::cout << " tied=yes";
    if (cfg.use_layernorm) std::cout << " norm=layernorm";
    if (has_any_layer_norm_bias_) std::cout << " norm_bias=yes";
    if (has_any_layer_output_bias_) std::cout << " o_proj_bias=yes";
    if (weights_.has_tensor("output.bias")) std::cout << " lm_head_bias=yes";
    if (cfg.sliding_window > 0) std::cout << " sliding_window=" << cfg.sliding_window;
    if (cfg.partial_rotary_factor != 1.0f)
      std::cout << " partial_rope=" << cfg.partial_rotary_factor;
    if (cfg.has_linear_attention()) {
      std::cout << " attention=linear-mixed";
    } else if (cfg.uses_non_full_attention()) {
      std::cout << " attention=nonfull";
    }
    if (cfg.is_moe()) {
      std::cout << " moe_experts=" << cfg.num_local_experts
                << " moe_topk=" << (cfg.effective_experts_per_tok());
      if (cfg.expert_intermediate_size > 0) {
        std::cout << " expert_inter=" << cfg.expert_intermediate_size;
      }
    }
    std::cout << "\n";
  }

  // A batched decode step ends in a stream sync to read the sampled tokens back,
  // and the next step cannot start until the host has them. On Windows that wake
  // is a WDDM scheduling event, and its latency lands squarely in the gap that
  // nsys shows between steps -- about 4.3 ms a step at B=32, most of which the
  // CUDA graph did not recover because it is not launch overhead. Spinning
  // instead of blocking trades a busy core for a faster wake.
  //
  // Must precede context creation, so it sits above cudaSetDevice. Off by
  // default: it burns a core, which is the wrong trade for a machine doing
  // anything else. CPI_CUDA_SPIN=1 enables.
  if (const char* spin = std::getenv("CPI_CUDA_SPIN"); spin != nullptr && *spin == '1') {
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
  }
  CUDA_CHECK(cudaSetDevice(0));
  // Prefill chunk = tokens per batched prefill pass. With attention on the tensor cores, prefill
  // is host-API-call bound, not GPU-bound, so doubling the chunk halves the passes and API traffic;
  // bigger is better. Chunking is exact (output unchanged for any size). Cost is VRAM: every prefill
  // activation buffer scales with the chunk, so size it from a memory budget, not a fixed constant
  // that would hurt small-VRAM GPUs.
  int prefill_chunk_target = env_int_or_default("CPI_PREFILL_CHUNK_SIZE", 0);
  if (prefill_chunk_target <= 0) {
    const int kv_dim = cfg.num_kv_heads * (cfg.hidden_size / std::max(1, cfg.num_heads));
    // x, x_norm, att, out + q/k/v + the three ff buffers, fp16.
    const std::size_t per_token = (static_cast<std::size_t>(cfg.hidden_size) * 4 +
                                   static_cast<std::size_t>(cfg.intermediate_size) * 3 +
                                   static_cast<std::size_t>(cfg.hidden_size + 2 * kv_dim)) *
                                  sizeof(__half);

    std::size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // Spend at most 768 MB, and never more than 5% of what is actually free.
    //
    // The ceiling used to be 2048 tokens on a 512 MB budget. Raising it to 4096 is worth more
    // than the row count suggests, because a chunk costs more than its rows: with packed
    // k-quant weights every chunk expands the whole model to fp16 once, about 11.6 ms on an 8B
    // Q4_K_M regardless of how many rows it then multiplies. Halving the chunk count on a
    // 4057-token prompt measured prefill 14793 -> 15848 tok/s (+7.1%) for +168 MiB of peak
    // VRAM. The free-memory clamp still keeps small cards where they were.
    const std::size_t budget = std::min<std::size_t>(std::size_t{768} << 20, free_bytes / 20);
    const std::size_t fits = (per_token > 0) ? (budget / per_token) : 2048;
    prefill_chunk_target =
        static_cast<int>(std::min<std::size_t>(4096, std::max<std::size_t>(256, fits)));
  }
  prefill_chunk_size_ = std::max(1, std::min(options_.max_context, prefill_chunk_target));
  if (options_.verbose) {
    std::cout << "[engine] prefill_chunk=" << prefill_chunk_size_ << "\n";
  }
  lt_workspace_bytes_ = env_workspace_bytes_or_default("CPI_LT_WORKSPACE_MB", lt_workspace_bytes_);
  CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));
  for (auto& ev : streaming_ready_) {
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  }
  for (auto& ev : streaming_consumed_) {
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, compute_stream_));
  }
  CUDA_CHECK(cudaEventCreate(&benchmark_transfer_start_));
  CUDA_CHECK(cudaEventCreate(&benchmark_transfer_end_));
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, compute_stream_));
  CUBLAS_CHECK(cublasLtCreate(&cublas_lt_));
  CUDA_CHECK(cudaMalloc(&lt_workspace_, lt_workspace_bytes_));

  run_startup_phase("static-load", [&] { load_static_weights(); });
  run_startup_phase("runtime-buffers", [&] { allocate_runtime_buffers(); });
  run_startup_phase("cache-copy", [&] {
    const bool prof = std::getenv("CPI_STARTUP_PROFILE") != nullptr;
    auto sub = [&](const char* n, auto&& fn) {
      const auto t0 = std::chrono::steady_clock::now();
      fn();
      if (prof) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        std::cout << "[startup]   cache-copy." << n << " ms=" << ms << "\n";
      }
    };
    sub("init_layer_cache", [&] { init_layer_cache(); });
    if (!cfg.is_moe()) {
      if (lowbit_streaming_enabled(options_)) {
        sub("uncached_pinned", [&] { init_uncached_pinned_host_weights(); });
        sub("uncached_int8", [&] { init_uncached_int8_host_weights(); });
      } else {
        sub("uncached_pinned", [&] { init_uncached_pinned_host_weights(); });
      }
    }
  });
  run_startup_phase("warmup", [&] { tune_resident_projection_backends(); });
  run_startup_phase("graph-init", [&] {
    init_greedy_decode_graph();
    init_logits_decode_graph();
  });

  // The pinned staging ring only serves load-time uploads; give the 128 MB back.
  free_load_staging();

  // Reclaim streaming staging when every layer is resident. The streaming decode double-buffer
  // (streaming_layer_weights_[2] / _i8_[2]) is only used for layers >= cached_layer_count_, so once
  // the model is fully cached it is never touched again; ~2.8 GB on the 32B (fp16 wqkv/wo/w13/w2
  // + int8 MLP per slot x2). Slot-0's int8 buffer was scratch during cache-copy only. Freeing it
  // returns that VRAM to steady-state headroom (nvidia-smi, and the KV pool) at zero quality cost.
  // The pointers are nulled, so the destructor's later free is a no-op.
  if (!weights_.config().is_moe() && cached_layer_count_ == weights_.config().num_layers) {
    const auto cf = [](void* p) {
      if (p) cudaFree(p);
    };
    for (auto& lw : streaming_layer_weights_) {
      cf(lw.wqkv);
      cf(lw.wo);
      cf(lw.bo);
      cf(lw.w13);
      cf(lw.w2);
      cf(lw.norm_att);
      cf(lw.norm_ffn);
      cf(lw.norm_att_bias);
      cf(lw.norm_ffn_bias);
      cf(lw.bqkv);
      cf(lw.q_norm);
      cf(lw.k_norm);
      lw = {};
    }
    for (auto& iw : streaming_layer_weights_i8_) {
      cf(iw.w1);
      cf(iw.w2);
      cf(iw.w3);
      cf(iw.s_w1);
      cf(iw.s_w2);
      cf(iw.s_w3);
      iw = {};
    }
    if (startup_vram) {
      std::size_t vfree = 0, vtot = 0;
      if (cudaMemGetInfo(&vfree, &vtot) == cudaSuccess) {
        std::cout << "[startup] freed streaming staging (fully cached); vram_used_gb="
                  << ((vtot - vfree) / (1024.0 * 1024.0 * 1024.0)) << "\n";
      }
    }
  }

  // Free the fp16 LM head once the int8 head serves inference. project_lm_head_logits() routes to
  // the resident int8 head (d_lm_head_i8_, near-lossless) when lm_head_int8_ is set, so the 1.45 GB
  // fp16 head is dead for single-sequence decode. Two consumers still need the fp16 head, so gate:
  // the batched lm-head (batched_lm_head) uses it and only runs under --paged-blocks; the autotuner
  // references it. CPI_FP16_LM_HEAD=1 keeps the head in fp16 (lm_head_int8_ stays false).
  if (lm_head_int8_ && d_lm_head_ != nullptr && !options_.paged_blocks &&
      std::getenv("CPI_AUTOTUNE") == nullptr) {
    cudaFree(d_lm_head_);
    d_lm_head_ = nullptr;
    if (startup_vram) {
      std::size_t vfree = 0, vtot = 0;
      if (cudaMemGetInfo(&vfree, &vtot) == cudaSuccess) {
        std::cout << "[startup] freed fp16 lm head (int8 head active); vram_used_gb="
                  << ((vtot - vfree) / (1024.0 * 1024.0 * 1024.0)) << "\n";
      }
    }
  }

  if (options_.verbose) {
    const auto startup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - startup_begin)
                                .count();
    std::cout << "[startup] total_ms=" << startup_ms << "\n";
  }
}

}  // namespace engine
