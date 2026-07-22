
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

// ---------------------------------------------------------------------------
// Chained greedy decode. K token-steps are encoded into command buffers the host never waits
// on individually: each step's argmax feeds the NEXT step's input token entirely on the GPU,
// so the per-token host round-trip (commit, wait, read, write) disappears. The host reads the
// whole block's tokens back once, from `ring`.

struct ChainParams {
  uint idx;
  int value;
};

// dst[0] = value. The chained step's position: the host knows it at encode time but must not
// write pos_buf_ directly while earlier steps still execute.
kernel void cpi_set_i32(device int* dst [[buffer(0)]], constant ChainParams& p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid == 0u) dst[0] = p.value;
}

// Feed the argmax result forward as the next step's input token, and record it for the host.
kernel void cpi_chain_token(device const int* argmax_out [[buffer(0)]],
                            device int* tok [[buffer(1)]], device int* ring [[buffer(2)]],
                            constant ChainParams& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid == 0u) {
    tok[0] = argmax_out[0];
    ring[p.idx] = argmax_out[0];
  }
}
