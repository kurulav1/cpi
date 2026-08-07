struct MoeRouterParams {
  uint experts;
  uint top_k;
  uint has_expert_scale;
};

// Softmax over experts, then top-k, then renormalise over the picked ones.
//
// One thread does all of it, exactly as the CUDA kernel does. Not laziness: experts is 8 and
// top_k is 2, so this is a handful of FLOPs -- and the selection is a sequential
// argmax-without-replacement whose tie-breaking must match CUDA's, or the two backends pick
// different experts and diverge. `p > best_p` (strict) means the lowest index wins a tie;
// parallelising the reduction would quietly change that.
kernel void cpi_moe_router_topk(
    device const half*  logits       [[buffer(0)]],
    device const half*  expert_scale [[buffer(1)]],  // per-expert gain, or a dummy binding
    device int*         topk_idx     [[buffer(2)]],
    device float*       topk_prob    [[buffer(3)]],
    constant MoeRouterParams& p      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0u || p.experts == 0u || p.top_k == 0u) return;

  const uint E = min(p.experts, 64u);
  float probs[64];

  float max_logit = float(logits[0]);
  for (uint e = 1u; e < E; ++e) max_logit = max(max_logit, float(logits[e]));

  float sum = 0.0f;
  for (uint e = 0u; e < E; ++e) {
    const float pr = exp(float(logits[e]) - max_logit);
    probs[e] = pr;
    sum += pr;
  }
  const float inv_sum = 1.0f / max(sum, 1.0e-8f);
  for (uint e = 0u; e < E; ++e) probs[e] *= inv_sum;

  int picked[8];
  float picked_prob[8];
  const uint capped = min(p.top_k, 8u);
  for (uint k = 0u; k < capped; ++k) {
    picked[k] = -1;
    picked_prob[k] = 0.0f;
  }
  for (uint k = 0u; k < capped; ++k) {
    int best_e = -1;
    float best_p = -1.0f;
    for (uint e = 0u; e < E; ++e) {
      bool used = false;
      for (uint prev = 0u; prev < k; ++prev) {
        if (picked[prev] == int(e)) {
          used = true;
          break;
        }
      }
      if (used) continue;
      if (probs[e] > best_p) {  // STRICT: ties go to the lower index, as in CUDA
        best_p = probs[e];
        best_e = int(e);
      }
    }
    picked[k] = best_e;
    picked_prob[k] = best_p > 0.0f ? best_p : 0.0f;
  }

  float picked_sum = 0.0f;
  for (uint k = 0u; k < capped; ++k) picked_sum += picked_prob[k];
  const float inv_pick = 1.0f / max(picked_sum, 1.0e-8f);

  for (uint k = 0u; k < p.top_k; ++k) {
    if (k < capped) {
      const int e = picked[k] >= 0 ? picked[k] : 0;
      topk_idx[k] = e;
      // A learned per-expert gain applied AFTER the renormalisation (Gemma 4). Mixtral has
      // none, and then the weight is left exactly as it was.
      const float gain = (p.has_expert_scale != 0u) ? float(expert_scale[e]) : 1.0f;
      topk_prob[k] = picked_prob[k] * inv_pick * gain;
    } else {
      topk_idx[k] = 0;
      topk_prob[k] = 0.0f;
    }
  }
}

struct MoeExpertParams {
  uint inter;
  uint hidden;
  uint top_k;
  uint use_gelu;  // 0 = SiLU (Mixtral), 1 = tanh-GELU (Gemma 4)
};

// inter[k][r] = act(gate) * up, where gate and up are rows r and (inter + r) of the selected
// expert's fused gate_up matrix. One threadgroup per (r, k).
kernel void cpi_moe_gate_up(
    device const half*  w         [[buffer(0)]],  // [experts][2*inter][hidden]
    device const half*  x         [[buffer(1)]],  // [hidden]
    device const int*   topk_idx  [[buffer(2)]],
    device half*        inter_out [[buffer(3)]],  // [top_k][inter]
    constant MoeExpertParams& p   [[buffer(4)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint r = gid % p.inter;
  const uint k = gid / p.inter;
  if (r >= p.inter || k >= p.top_k) return;

  const ulong base = (ulong)topk_idx[k] * (ulong)(2u * p.inter);
  device const half* gate_row = w + (base + (ulong)r) * (ulong)p.hidden;
  device const half* up_row = w + (base + (ulong)(p.inter + r)) * (ulong)p.hidden;

  float g = 0.0f, u = 0.0f;
  for (uint c = lid; c < p.hidden; c += nthr) {
    const float xv = float(x[c]);
    g += float(gate_row[c]) * xv;
    u += float(up_row[c]) * xv;
  }
  g = simd_sum(g);
  u = simd_sum(u);

  threadgroup float gs[32], us[32];
  const uint simd_id = lid / 32u, lane = lid % 32u;
  if (lane == 0u) {
    gs[simd_id] = g;
    us[simd_id] = u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    const uint n_simd = nthr / 32u;
    float G = 0.0f, U = 0.0f;
    for (uint i = 0u; i < n_simd; ++i) {
      G += gs[i];
      U += us[i];
    }
    float act;
    if (p.use_gelu != 0u) {
      // Sigmoid form: Metal's tanh computes (exp(2z)-1)/(exp(2z)+1) and overflows to NaN past
      // z~44 where CUDA's tanhf saturates. 0.5*(1+tanh(z)) IS sigmoid(2z), safe at both ends.
      const float c = 0.7978845608028654f * (G + 0.044715f * G * G * G);
      act = G * (1.0f / (1.0f + exp(-2.0f * c)));
    } else {
      act = G / (1.0f + exp(-G));  // SiLU
    }
    inter_out[(ulong)k * (ulong)p.inter + r] = half(act * U);
  }
}

// y[r] = sum_k topk_weight[k] * dot(down[expert_k].row(r), inter[k]).
// One threadgroup per output row; it walks all top_k experts, so the weighted sum needs no
// atomics and no second pass.
kernel void cpi_moe_down_accum(
    device const half*  w        [[buffer(0)]],  // [experts][hidden][inter]
    device const half*  inter_in [[buffer(1)]],  // [top_k][inter]
    device const int*   topk_idx [[buffer(2)]],
    device const float* topk_w   [[buffer(3)]],
    device half*        y        [[buffer(4)]],  // [hidden]
    constant MoeExpertParams& p  [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint r = gid;
  if (r >= p.hidden) return;

  threadgroup float part[32];
  threadgroup float dot_k;
  const uint simd_id = lid / 32u, lane = lid % 32u;
  const uint n_simd = nthr / 32u;

  float acc = 0.0f;
  for (uint k = 0u; k < p.top_k; ++k) {
    device const half* row = w + ((ulong)topk_idx[k] * (ulong)p.hidden + (ulong)r) * (ulong)p.inter;
    device const half* xk = inter_in + (ulong)k * (ulong)p.inter;

    float d = 0.0f;
    for (uint c = lid; c < p.inter; c += nthr) d += float(row[c]) * float(xk[c]);
    d = simd_sum(d);
    if (lane == 0u) part[simd_id] = d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      float t = 0.0f;
      for (uint i = 0u; i < n_simd; ++i) t += part[i];
      dot_k = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc += topk_w[k] * dot_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // part/dot_k are reused next expert
  }
  if (lid == 0u) y[r] = half(acc);
}

// ---------------------------------------------------------------------------
// Argmax over fp32 logits, two-phase (the vocab is far too big for one group).
// Phase 1 writes per-partition (value, index); phase 2 reduces them.
// ---------------------------------------------------------------------------
