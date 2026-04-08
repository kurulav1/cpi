// Implements TensorParallelLinear: a row-parallel linear operator that
// distributes the weight matrix across multiple CUDA devices and executes
// independent fp16 GEMMs in parallel. Each device holds a contiguous vertical
// slice of the weight matrix (a subset of the output rows). After the GEMMs
// complete, each shard's result is copied back to the caller's output buffer on
// the primary device via cudaMemcpyAsync so the copies overlap on the stream.

#include "engine/tensor_parallel.hpp"

#include <algorithm>
#include <stdexcept>

#include <cuda_fp16.h>

#include "runtime/cuda_utils.cuh"

namespace engine {

// Destroys per-device cuBLAS handles and frees device weight/partial buffers.
// Each device must be set active before its resources are released.
TensorParallelLinear::~TensorParallelLinear() {
  for (auto& ctx : contexts_) {
    if (ctx.handle) {
      cudaSetDevice(ctx.device);
      cublasDestroy(ctx.handle);
      ctx.handle = nullptr;
    }
    if (ctx.d_weight) {
      cudaSetDevice(ctx.device);
      cudaFree(ctx.d_weight);
      ctx.d_weight = nullptr;
    }
    if (ctx.d_partial) {
      cudaSetDevice(ctx.device);
      cudaFree(ctx.d_partial);
      ctx.d_partial = nullptr;
    }
  }
}

// Distributes weight shards across devices and initialises cuBLAS handles.
//
// The out_features rows are divided greedily: rank r receives
//   rows_remaining / (world_size - r)
// rows, ensuring all output rows are covered even when out_features is not
// evenly divisible by world_size. Each shard is uploaded synchronously from
// the corresponding host pointer in shard_weights_fp16.
void TensorParallelLinear::initialize(int world_size,
                                      int in_features,
                                      int out_features,
                                      const std::vector<const void*>& shard_weights_fp16) {
  if (world_size <= 0) {
    throw std::invalid_argument("world_size must be > 0");
  }
  if (static_cast<int>(shard_weights_fp16.size()) < world_size) {
    throw std::invalid_argument("missing shard weight pointers");
  }

  in_features_ = in_features;
  out_features_ = out_features;

  contexts_.clear();
  contexts_.resize(world_size);

  int rows_remaining = out_features;
  for (int rank = 0; rank < world_size; ++rank) {
    auto& ctx = contexts_[rank];
    ctx.device = rank;
    // Divide remaining rows evenly across remaining devices so no row is missed.
    ctx.out_rows = rows_remaining / (world_size - rank);
    rows_remaining -= ctx.out_rows;

    CUDA_CHECK(cudaSetDevice(ctx.device));
    CUBLAS_CHECK(cublasCreate(&ctx.handle));

    const std::size_t shard_bytes =
        static_cast<std::size_t>(ctx.out_rows) * static_cast<std::size_t>(in_features_) * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&ctx.d_weight, shard_bytes));
    CUDA_CHECK(cudaMemcpy(ctx.d_weight, shard_weights_fp16[rank], shard_bytes, cudaMemcpyHostToDevice));
  }
}

// Runs a row-parallel fp16 GEMM on each device and concatenates the partial
// results into d_output_fp16 on the primary device.
//
// The operation is: d_output_fp16 = W_shard * d_input_fp16^T (column-major
// interpretation). d_partial is re-allocated per call because the batch size
// may change between invocations; this is acceptable for typical use where
// batch is constant during inference.
void TensorParallelLinear::forward(const void* d_input_fp16,
                                   int batch,
                                   void* d_output_fp16,
                                   cudaStream_t stream) {
  int row_offset = 0;

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  for (auto& ctx : contexts_) {
    CUDA_CHECK(cudaSetDevice(ctx.device));
    CUBLAS_CHECK(cublasSetStream(ctx.handle, stream));

    // Re-allocate the per-shard output buffer if necessary.
    if (ctx.d_partial) {
      cudaFree(ctx.d_partial);
      ctx.d_partial = nullptr;
    }

    const std::size_t out_bytes =
        static_cast<std::size_t>(batch) * static_cast<std::size_t>(ctx.out_rows) * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&ctx.d_partial, out_bytes));

    // Compute the shard GEMM: d_partial[out_rows, batch] = W[out_rows, in] * x[in, batch].
    // CUBLAS_OP_N on both operands because the weight shard is already stored
    // in the column-major layout expected by cuBLAS (row_major HF weights are
    // reinterpreted as column-major in the full linear path; here the shard
    // pointer arrives pre-transposed from the caller).
    CUBLAS_CHECK(cublasGemmEx(ctx.handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              ctx.out_rows,
                              batch,
                              in_features_,
                              &alpha,
                              ctx.d_weight,
                              CUDA_R_16F,
                              ctx.out_rows,
                              d_input_fp16,
                              CUDA_R_16F,
                              in_features_,
                              &beta,
                              ctx.d_partial,
                              CUDA_R_16F,
                              ctx.out_rows,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Copy this shard's result to the correct row range in the output buffer.
    // row_offset tracks the element (not byte) position because element size
    // is sizeof(__half) and we cast to char* for byte arithmetic.
    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<char*>(d_output_fp16) + static_cast<std::size_t>(row_offset) * sizeof(__half),
        ctx.d_partial,
        out_bytes,
        cudaMemcpyDeviceToDevice,
        stream));

    row_offset += ctx.out_rows * batch;
  }
}

}  // namespace engine
