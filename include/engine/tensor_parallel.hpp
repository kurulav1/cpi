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
  void initialize(int world_size,
                  int in_features,
                  int out_features,
                  const std::vector<const void*>& shard_weights_fp16);

  // Runs the row-parallel fp16 GEMM across all devices and concatenates results.
  //
  // d_input_fp16  - device pointer (on device 0) to the [batch, in_features] input.
  // batch         - number of tokens in the batch (N dimension).
  // d_output_fp16 - device pointer (on device 0) to receive [batch, out_features] output.
  // stream        - CUDA stream to use on every device.
  //
  // The function issues a cublasGemmEx on each device and then copies each
  // shard's partial result back to d_output_fp16 at the appropriate row offset.
  void forward(const void* d_input_fp16,
               int batch,
               void* d_output_fp16,
               cudaStream_t stream);

 private:
  // Per-device state: cuBLAS handle, weight shard, and temporary partial output.
  struct DeviceContext {
    int device = 0;
    cublasHandle_t handle = nullptr;
    void* d_weight = nullptr;
    int out_rows = 0;      // number of output rows assigned to this device
    void* d_partial = nullptr;  // temporary device buffer for the per-shard result
  };

  int in_features_ = 0;
  int out_features_ = 0;
  std::vector<DeviceContext> contexts_;
};

}  // namespace engine
