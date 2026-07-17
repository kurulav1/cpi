
kernel void cpi_argmax_partial(
    device const float* logits    [[buffer(0)]],
    device float*       part_val  [[buffer(1)]],
    device int*         part_idx  [[buffer(2)]],
    constant ElemParams& p        [[buffer(3)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]],
    uint ngrp [[threadgroups_per_grid]]) {
  float best_v = -INFINITY;
  int   best_i = -1;

  for (uint i = gid * nthr + lid; i < p.n; i += nthr * ngrp) {
    const float v = logits[i];
    if (v > best_v) { best_v = v; best_i = int(i); }
  }

  threadgroup float tv[256];
  threadgroup int   ti[256];
  tv[lid] = best_v;
  ti[lid] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = nthr / 2u; s > 0u; s >>= 1u) {
    if (lid < s && tv[lid + s] > tv[lid]) {
      tv[lid] = tv[lid + s];
      ti[lid] = ti[lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0u) {
    part_val[gid] = tv[0];
    part_idx[gid] = ti[0];
  }
}

kernel void cpi_argmax_reduce(
    device const float* part_val [[buffer(0)]],
    device const int*   part_idx [[buffer(1)]],
    device int*         out      [[buffer(2)]],
    constant ElemParams& p       [[buffer(3)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  float best_v = -INFINITY;
  int   best_i = -1;
  for (uint i = lid; i < p.n; i += nthr) {
    if (part_val[i] > best_v) { best_v = part_val[i]; best_i = part_idx[i]; }
  }

  threadgroup float tv[256];
  threadgroup int   ti[256];
  tv[lid] = best_v;
  ti[lid] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = nthr / 2u; s > 0u; s >>= 1u) {
    if (lid < s && tv[lid + s] > tv[lid]) {
      tv[lid] = tv[lid + s];
      ti[lid] = ti[lid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0u) out[0] = ti[0];
}
