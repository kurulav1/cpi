#pragma once

// Expert parallelism (EP) dispatch for MoE: the experts are sharded across ranks (rank r owns a
// contiguous expert range), so a token's top-k selections may land on several ranks. Each rank runs
// the STANDARD MoE kernels on its own expert sub-matrix, with the selections REMAPPED to local expert
// indices and the off-rank selections' weights MASKED to zero; the per-rank outputs then SUM to the
// full MoE result (each global expert contributes exactly once, on its owner). That sum is the
// all_to_all-combine seam -- for real multi-GPU it becomes a collective; on one GPU it is a local add,
// which is what makes the dispatch/combine logic verifiable here (expert_parallel_test).
//
// Header-only host index logic (no GPU); pairs with the existing launch_moe_gate_up_geglu /
// launch_moe_down_accum kernels run on the sliced expert weights.

namespace engine {

// [lo, hi) global experts owned by `rank` under a balanced greedy split -- each rank gets floor or
// ceil(experts/world_size), the same partition policy as the TP row split and the PP stage split (see
// pipeline_parallel.hpp), so every parallelism dimension shards the same way. Contiguous, tiles
// [0, experts) with no gap or overlap for ANY world_size. Unlike an even-split-plus-last-rank-remainder
// scheme this stays load-balanced and degrades cleanly when world_size > experts (trailing ranks get
// an empty [x, x) range and own no experts) instead of piling every leftover onto the final rank.
inline void expert_parallel_range(int experts, int world_size, int rank, int* lo, int* hi) {
  int rem = experts;
  int off = 0;
  for (int r = 0; r <= rank; ++r) {
    const int count = rem / (world_size - r);
    if (r == rank) {
      *lo = off;
      *hi = off + count;
      return;
    }
    off += count;
    rem -= count;
  }
}

// For one token's top-k selections (global expert ids + weights), fill this rank's LOCAL expert
// indices and MASKED weights: on-rank selection -> (global - lo, weight); off-rank -> (0, 0). The
// zero weight makes the standard down-accum kernel drop the off-rank term, so running the kernels on
// rank r's [lo,hi) expert sub-matrix yields exactly r's partial of the full weighted sum.
inline void expert_parallel_dispatch(int experts, int world_size, int rank, const int* topk_idx,
                                     const float* topk_w, int top_k, int* local_idx,
                                     float* masked_w) {
  int lo = 0, hi = 0;
  expert_parallel_range(experts, world_size, rank, &lo, &hi);
  for (int k = 0; k < top_k; ++k) {
    const int e = topk_idx[k];
    if (e >= lo && e < hi) {
      local_idx[k] = e - lo;
      masked_w[k] = topk_w[k];
    } else {
      local_idx[k] = 0;
      masked_w[k] = 0.0f;
    }
  }
}

}  // namespace engine
