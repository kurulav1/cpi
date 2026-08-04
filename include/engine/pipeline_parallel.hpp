#pragma once

// Pipeline parallelism (PP): the model's transformer layers are split into contiguous STAGES, one per
// rank. Rank r runs layers [lo, hi) on the activation it receives, then hands the resulting activation
// to rank r+1. The layer partition is pure host index math (below); the stage-to-stage handoff is a
// single activation transfer that becomes a point-to-point send on a real cluster and is a local
// device copy on one box; which is what makes the partition + handoff logic verifiable here
// (pipeline_parallel_test), exactly like the TP/EP bricks. Scope is single-NODE (see
// memory:cpi-multi-gpu-hpc-prep); the cross-rank transport is the only cluster-gated piece.
//
// The split is the same balanced greedy split TensorParallelLinear uses for output rows, so every
// parallelism dimension partitions the same way: each stage gets floor or ceil of layers/world,
// contiguous, no gap or overlap. It degrades cleanly when world_size > num_layers; the trailing
// ranks get an empty [x, x) range and act as identity pass-throughs.

namespace engine {

// [lo, hi) contiguous layers owned by `rank` under a balanced greedy split of `num_layers` across
// `world_size` ranks. Each rank gets floor or ceil(num_layers/world_size) layers; the ranges tile
// [0, num_layers) with no gap or overlap for any world_size (including world_size > num_layers, where
// the trailing ranks get an empty range). lo == hi means the stage runs no layers (identity handoff).
inline void pipeline_parallel_stage_range(int num_layers, int world_size, int rank, int* lo,
                                          int* hi) {
  int rem = num_layers;
  int off = 0;
  for (int r = 0; r <= rank; ++r) {
    const int count = rem / (world_size - r);  // floor/ceil balance, same as the TP row split
    if (r == rank) {
      *lo = off;
      *hi = off + count;
      return;
    }
    off += count;
    rem -= count;
  }
}

// Which rank runs `layer` (0 <= layer < num_layers), i.e. the inverse of the stage split. Returns
// world_size - 1 for an out-of-range layer (defensive; callers pass valid layer ids).
inline int pipeline_parallel_owner(int num_layers, int world_size, int layer) {
  for (int r = 0; r < world_size; ++r) {
    int lo = 0, hi = 0;
    pipeline_parallel_stage_range(num_layers, world_size, r, &lo, &hi);
    if (layer >= lo && layer < hi) return r;
  }
  return world_size - 1;
}

}  // namespace engine
