#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "runtime/kernels.cuh"

// Multi-GPU collective interface. Single-process model: one buffer per rank (single-node multi-GPU,
// or the single-GPU verification where every rank's buffer sits on device 0). world_size()==1 => the
// ops are no-ops, so the world=1 path is exactly today's single-GPU engine.
//
// LocalCollective implements the ops with local device copies/adds; verifiable on one box, and the
// correct single-node-multi-GPU behaviour via peer copies. A NcclCollective wrapping ncclCommInitAll
// (single-node, one process, multiple GPUs) is the only piece that needs real multi-GPU hardware; it
// slots in behind this interface, declared in an HPC build. This is the seam that keeps NCCL a single
// isolated dependency instead of threaded through the engine. Scope is single-NODE (see
// memory:cpi-multi-gpu-hpc-prep); multi-node (one process per rank) is out of scope.
//
// The tensor-parallel all-reduce (RowParallelLinear) and the expert-parallel combine
// (expert_parallel.hpp) are the call sites that route through all_reduce_sum_fp16 once wired.

namespace engine {

class Collective {
public:
  virtual ~Collective() = default;
  virtual int world_size() const = 0;

  // All-reduce (sum): recvbuf[r] receives the elementwise fp16 SUM over every rank's sendbuf. In-place
  // (sendbuf[r] == recvbuf[r]) is allowed. `count` is the element count per rank (all equal).
  virtual void all_reduce_sum_fp16(const std::vector<const void*>& sendbuf,
                                   const std::vector<void*>& recvbuf, int count,
                                   cudaStream_t stream) = 0;
};

// Single-process implementation: reduce into rank 0, broadcast to the rest. On one GPU all buffers are
// on device 0 (verification); on single-node multi-GPU these are peer copies. This exact class is what
// a NcclCollective replaces for cross-node.
class LocalCollective : public Collective {
public:
  explicit LocalCollective(int world_size) : world_size_(world_size) {}
  int world_size() const override { return world_size_; }

  void all_reduce_sum_fp16(const std::vector<const void*>& sendbuf,
                           const std::vector<void*>& recvbuf, int count,
                           cudaStream_t stream) override {
    if (world_size_ <= 0) return;
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(__half);
    // acc (rank 0) = sum of all sends. Read every send before any broadcast overwrites, so in-place
    // (send==recv) is safe.
    if (recvbuf[0] != sendbuf[0])
      cudaMemcpyAsync(recvbuf[0], sendbuf[0], bytes, cudaMemcpyDeviceToDevice, stream);
    for (int r = 1; r < world_size_; ++r)
      kernels::launch_add_inplace(static_cast<__half*>(recvbuf[0]),
                                  static_cast<const __half*>(sendbuf[r]), count, stream);
    for (int r = 1; r < world_size_; ++r)
      cudaMemcpyAsync(recvbuf[r], recvbuf[0], bytes, cudaMemcpyDeviceToDevice, stream);
  }

private:
  int world_size_;
};

}  // namespace engine
