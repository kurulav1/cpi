#pragma once

#include <cublas_v2.h>

#include <cstddef>
#include <vector>

// Tensor-parallel linear layer split across multiple CUDA devices.
//
// Implements row-parallel projection: the output rows of the weight matrix are
// evenly divided among the available GPUs. Each device computes its shard of
// the output independently using cuBLAS, and the results are concatenated in
// host-side order into a caller-supplied device buffer on the primary device.
//
// Typical use:
//   TensorParallelLinear op;
//   op.initialize(world_size, in_features, out_features, shard_ptrs);
//   op.forward(d_input, batch, d_output, stream);

namespace engine {

// Tensor-parallel linear operator (row-parallel projection).
// Splits output rows across available GPUs and concatenates results.
class TensorParallelLinear {
public:
  TensorParallelLinear() = default;
  ~TensorParallelLinear();

  // Initializes per-device cuBLAS handles and uploads weight shards to VRAM.
  //
  // world_size        - number of CUDA devices to distribute across (must be > 0).
  // in_features       - number of input columns (K dimension of the GEMM).
  // out_features      - total number of output rows before sharding (M dimension).
  // shard_weights_fp16 - host pointers to fp16 weight shards, one per device.
  //                      Each shard covers (out_features / world_size) rows.
  // devices           - optional rank->CUDA-device map. Empty => rank r uses device r (real
  //                      multi-GPU). Passing all-zeros runs every rank on device 0, which lets the
  //                      sharding math be verified on a single GPU (tensor_parallel_test); the
  //                      cross-device transport is the only piece that then needs real hardware.
  void initialize(int world_size, int in_features, int out_features,
                  const std::vector<const void*>& shard_weights_fp16,
                  const std::vector<int>& devices = {});

  // Runs the row-parallel fp16 GEMM across all devices and concatenates results.
  //
  // d_input_fp16  - device pointer (on device 0) to the [batch, in_features] input.
  // batch         - number of tokens in the batch (N dimension).
  // d_output_fp16 - device pointer (on device 0) to receive [batch, out_features] output.
  // stream        - CUDA stream to use on every device.
  //
  // The function issues a cublasGemmEx on each device and then copies each
  // shard's partial result back to d_output_fp16 at the appropriate row offset.
  void forward(const void* d_input_fp16, int batch, void* d_output_fp16, cudaStream_t stream);

private:
  // Per-device state: cuBLAS handle, weight shard, and temporary partial output.
  struct DeviceContext {
    int device = 0;
    cublasHandle_t handle = nullptr;
    void* d_weight = nullptr;
    int out_rows = 0;           // number of output rows assigned to this device
    void* d_partial = nullptr;  // temporary device buffer for the per-shard result
  };

  int in_features_ = 0;
  int out_features_ = 0;
  std::vector<DeviceContext> contexts_;
};

// Row-parallel linear: the complement of TensorParallelLinear. The weight is split by INPUT columns
// (the K dimension); each rank holds W[:, in_slice] and its slice of the input, computes a PARTIAL
// output over its inputs, and the partials are all-REDUCED (summed) into the full output. A
// column-parallel layer's sharded output feeds a row-parallel layer with no gather between them
// the standard Megatron tensor-parallel block. Verified single-GPU (row_parallel_test); the sum is
// where ncclAllReduce plugs in for real multi-GPU (the one cluster-gated step).
class RowParallelLinear {
public:
  RowParallelLinear() = default;
  ~RowParallelLinear();

  // shard_weights_fp16[r] = W[:, in_slice_r] as column-major [out_features, in_r] (ld =
  // out_features). The greedy split covers all input columns even when in_features is not divisible
  // by world_size. devices: optional rank->device map (see TensorParallelLinear); all-zeros =
  // single-GPU verify.
  void initialize(int world_size, int in_features, int out_features,
                  const std::vector<const void*>& shard_weights_fp16,
                  const std::vector<int>& devices = {});

  // shard_inputs_fp16[r] = device pointer to x_r, column-major [in_r, batch] (ld = in_r). Writes
  // the reduced [out_features, batch] result to d_output_fp16 on the primary device.
  void forward(const std::vector<const void*>& shard_inputs_fp16, int batch, void* d_output_fp16,
               cudaStream_t stream);

private:
  struct DeviceContext {
    int device = 0;
    cublasHandle_t handle = nullptr;
    void* d_weight = nullptr;
    int in_rows = 0;  // input columns assigned to this rank
    void* d_partial = nullptr;
  };
  int in_features_ = 0;
  int out_features_ = 0;
  std::vector<DeviceContext> contexts_;
};

}  // namespace engine
