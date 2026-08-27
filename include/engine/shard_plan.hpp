#pragma once

// Shard-fit / memory planner: given a model's dimensions, a weight/KV quantisation, and a 3D
// parallelism split (tensor x pipeline x expert), compute the exact per-rank VRAM footprint and
// scan for the minimum world size that fits a target GPU. Pure host arithmetic, no GPU, no cluster,
// so it runs anywhere and is verifiable by reproducing the known single-GPU footprints at world=1.
//
// The per-tensor byte formulas mirror the engine's own resident-footprint accounting
// (llama_engine_cache.cpp): fp16 = rows*cols*2; int8 = rows*cols + one fp32 scale per output row;
// int4 = rows*ceil(cols/2) + one fp32 scale per output row (per-row scales, the engine default).
// This is a structural estimate of weights + KV; a coarse activation working-set is reported
// separately and labelled, since it is workload-dependent, not a fixed allocation.
//
// The `weight_quant` here is applied uniformly to every weight; the engine actually runs mixed
// precision (MLP at this quant, projections int8, embedding/head fp16 on CUDA). The int4-MLP
// dominates so the total still lands on the real footprint; validated against the known world=1
// numbers: Llama-8B fp16 = 8.03B*2 exactly, Qwen-32B int4 @32k ctx ~= 24.9 GB, Gemma-4 26B-A4B int4
// experts
// ~= 12 GB; but treat the per-component lines as an approximation of the true per-tensor policy.
//
// Sharding model (standard 3D mesh, world = tp * pp * ep):
//   * pp (pipeline): a rank owns ceil(num_layers / pp) contiguous layers. The embedding sits on the
//     first stage and the LM head on the last, so the planner charges each to the single stage that
//     holds it and reports the heaviest rank.
//   * tp (tensor): within a layer, the dense attention+MLP matrices and the KV heads are split tp
//   ways;
//     the vocab (embedding + head) is also split tp ways (Megatron vocab-parallel).
//   * ep (expert): the per-layer experts are split ep ways.
// This matches how the TP/EP/PP verification bricks shard (balanced floor/ceil); see
// tensor_parallel.hpp / expert_parallel.hpp / pipeline_parallel.hpp.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine {

enum class Quant { FP16, INT8, INT4 };

inline const char* quant_name(Quant q) {
  switch (q) {
    case Quant::FP16:
      return "fp16";
    case Quant::INT8:
      return "int8";
    case Quant::INT4:
      return "int4";
  }
  return "?";
}

inline double quant_bytes_per_elem(Quant q) {
  switch (q) {
    case Quant::FP16:
      return 2.0;
    case Quant::INT8:
      return 1.0;
    case Quant::INT4:
      return 0.5;
  }
  return 2.0;
}

struct ModelDims {
  std::string name;
  int hidden = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int intermediate = 0;  // dense MLP intermediate (0 for pure-MoE layers)
  int vocab = 0;
  int num_experts = 0;          // 0 => dense model
  int expert_intermediate = 0;  // per-expert MLP intermediate
  bool tie_embeddings = false;

  int q_hidden() const {
    return num_heads * head_dim;
  }
  int kv_hidden() const {
    return num_kv_heads * head_dim;
  }
  bool is_moe() const {
    return num_experts > 0 && expert_intermediate > 0;
  }
};

struct PlanConfig {
  Quant weight_quant = Quant::FP16;
  Quant kv_quant = Quant::FP16;
  int seq_len = 4096;
  int batch = 1;
  int tp = 1;  // tensor-parallel degree
  int pp = 1;  // pipeline-parallel degree
  int ep = 1;  // expert-parallel degree
  int world() const {
    return tp * pp * ep;
  }
};

struct Footprint {
  std::size_t embedding = 0;    // token embedding table (on the pp rank that holds it)
  std::size_t head = 0;         // LM head (0 if tied)
  std::size_t attn = 0;         // all attention weights on this rank
  std::size_t dense_mlp = 0;    // dense MLP weights on this rank
  std::size_t experts = 0;      // expert weights on this rank
  std::size_t kv_cache = 0;     // KV cache on this rank at seq_len x batch
  std::size_t activations = 0;  // coarse working-set estimate (labelled approximate)
  int layers_on_rank = 0;

  std::size_t weights() const {
    return embedding + head + attn + dense_mlp + experts;
  }
  std::size_t total() const {
    return weights() + kv_cache + activations;
  }
};

namespace detail {
// Bytes for one weight matrix with `rows` output rows and `cols` input columns at quant q, with the
// engine's per-output-row fp32 scale for the low-bit cases.
inline std::size_t matrix_bytes(long long rows, long long cols, Quant q) {
  const std::size_t r = static_cast<std::size_t>(rows);
  const std::size_t c = static_cast<std::size_t>(cols);
  switch (q) {
    case Quant::FP16:
      return r * c * 2;
    case Quant::INT8:
      return r * c + r * sizeof(float);
    case Quant::INT4:
      return r * ((c + 1) / 2) + r * sizeof(float);
  }
  return r * c * 2;
}
// Ceil-divide, for balanced shard sizing (the heaviest rank owns ceil(n/parts)).
inline int ceil_div(int n, int parts) {
  return parts > 0 ? (n + parts - 1) / parts : n;
}
}  // namespace detail

// Per-rank footprint for the heaviest rank under `pc`. Weights that TP splits are charged at their
// sharded size; the embedding/head are charged on their single pp stage (vocab split tp ways);
// experts are split ep ways; the KV cache is charged for this rank's layers and its tp share of KV
// heads.
inline Footprint estimate_footprint(const ModelDims& d, const PlanConfig& pc) {
  using detail::ceil_div;
  using detail::matrix_bytes;
  const Quant wq = pc.weight_quant;
  Footprint f;

  const int layers_on_rank = ceil_div(d.num_layers, pc.pp);
  f.layers_on_rank = layers_on_rank;

  // --- attention (dense matrices split tp ways; norms are tiny fp16 and replicated) ---
  const long long qh = d.q_hidden();
  const long long kvh = d.kv_hidden();
  const long long qh_tp = ceil_div(static_cast<int>(qh), pc.tp);
  const long long kvh_tp = ceil_div(static_cast<int>(kvh), pc.tp);
  std::size_t attn_per_layer = matrix_bytes(qh_tp, d.hidden, wq) +          // q proj
                               matrix_bytes(d.hidden, qh_tp, wq) +          // o proj
                               matrix_bytes(kvh_tp, d.hidden, wq) * 2 +     // k, v proj
                               matrix_bytes(1, d.hidden, Quant::FP16) * 2;  // 2 rmsnorms
  f.attn = attn_per_layer * static_cast<std::size_t>(layers_on_rank);

  // --- dense MLP (split tp ways along the intermediate dim) ---
  if (d.intermediate > 0) {
    const long long inter_tp = ceil_div(d.intermediate, pc.tp);
    const std::size_t mlp_per_layer = matrix_bytes(inter_tp, d.hidden, wq) * 2 +  // gate, up
                                      matrix_bytes(d.hidden, inter_tp, wq);       // down
    f.dense_mlp = mlp_per_layer * static_cast<std::size_t>(layers_on_rank);
  }

  // --- experts (split ep ways; each expert's matrices are not tp-split in this model) ---
  if (d.is_moe()) {
    const int experts_on_rank = ceil_div(d.num_experts, pc.ep);
    const std::size_t per_expert =
        matrix_bytes(2LL * d.expert_intermediate, d.hidden, wq) +  // gate_up
        matrix_bytes(d.hidden, d.expert_intermediate, wq);         // down
    f.experts = per_expert * static_cast<std::size_t>(experts_on_rank) *
                static_cast<std::size_t>(layers_on_rank);
  }

  // --- embedding + head (charged on their pp stage; vocab split tp ways) ---
  const long long vocab_tp = ceil_div(d.vocab, pc.tp);
  f.embedding = matrix_bytes(vocab_tp, d.hidden, wq);
  f.head = d.tie_embeddings ? 0 : matrix_bytes(vocab_tp, d.hidden, wq);

  // --- KV cache (this rank's layers x its tp-share of KV heads x seq x batch, K and V) ---
  const double kv_bpe = quant_bytes_per_elem(pc.kv_quant);
  const long long kv_elems = static_cast<long long>(layers_on_rank) * kvh_tp *
                             static_cast<long long>(pc.seq_len) * pc.batch * 2;  // K + V
  f.kv_cache = static_cast<std::size_t>(kv_elems * kv_bpe);

  // --- activations: coarse working-set (a handful of hidden-sized fp16 buffers over the batch x
  // seq token span). Labelled approximate; it is workload-dependent, not a fixed allocation. ---
  f.activations = static_cast<std::size_t>(4LL) * pc.batch * pc.seq_len * d.hidden * 2;

  return f;
}

struct FitOption {
  int tp = 1, pp = 1, ep = 1;
  std::size_t per_rank_bytes = 0;
  int world() const {
    return tp * pp * ep;
  }
};

// Scan tp/pp/ep over powers of two up to `max_degree` and return the feasible configs (heaviest
// rank
// <= budget_bytes), sorted by world size then per-rank bytes. ep is only varied for MoE models.
inline std::vector<FitOption> scan_fit(const ModelDims& d, Quant wq, Quant kvq, int seq, int batch,
                                       std::size_t budget_bytes, int max_degree = 64) {
  std::vector<FitOption> out;
  std::vector<int> degrees;
  for (int x = 1; x <= max_degree; x *= 2) degrees.push_back(x);
  for (int tp : degrees) {
    if (tp > d.num_heads && tp > 1) continue;  // can't split fewer heads than ranks
    for (int pp : degrees) {
      if (pp > d.num_layers && pp > 1) continue;
      for (int ep : degrees) {
        if (!d.is_moe() && ep > 1) continue;
        if (d.is_moe() && ep > d.num_experts && ep > 1) continue;
        PlanConfig pc;
        pc.weight_quant = wq;
        pc.kv_quant = kvq;
        pc.seq_len = seq;
        pc.batch = batch;
        pc.tp = tp;
        pc.pp = pp;
        pc.ep = ep;
        const std::size_t per_rank = estimate_footprint(d, pc).total();
        if (per_rank <= budget_bytes) out.push_back({tp, pp, ep, per_rank});
      }
    }
  }
  std::sort(out.begin(), out.end(), [](const FitOption& a, const FitOption& b) {
    if (a.world() != b.world()) return a.world() < b.world();
    return a.per_rank_bytes < b.per_rank_bytes;
  });
  return out;
}

}  // namespace engine
