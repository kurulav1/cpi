// Invariants for the shard-fit planner (engine/shard_plan.hpp); pure host arithmetic, no GPU.
// Checks that the sharding divisors actually divide the footprint the way TP/EP/PP claim, and that
// the fit scan only returns configs that fit and are ordered by world size. Complements the
// empirical validation (the planner reproduces the known single-GPU footprints of real models at
// world=1).
#include "engine/shard_plan.hpp"

#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* msg) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", msg);
  if (!ok) ++failures;
}

engine::ModelDims dense_model() {
  engine::ModelDims d;
  d.name = "dense-test";
  d.hidden = 4096;
  d.num_layers = 32;
  d.num_heads = 32;
  d.num_kv_heads = 8;
  d.head_dim = 128;
  d.intermediate = 14336;
  d.vocab = 128256;
  return d;
}

engine::ModelDims moe_model() {
  engine::ModelDims d = dense_model();
  d.name = "moe-test";
  d.intermediate = 0;
  d.num_experts = 64;
  d.expert_intermediate = 1408;
  return d;
}

engine::PlanConfig base_cfg() {
  engine::PlanConfig pc;
  pc.weight_quant = engine::Quant::FP16;
  pc.kv_quant = engine::Quant::FP16;
  pc.seq_len = 8192;
  pc.batch = 1;
  return pc;
}
}  // namespace

int main() {
  const engine::ModelDims dense = dense_model();

  // --- TP halves the dense MLP + attention weights (the tp-split matrices) ---
  {
    engine::PlanConfig w1 = base_cfg();
    engine::PlanConfig t2 = base_cfg();
    t2.tp = 2;
    const auto f1 = estimate_footprint(dense, w1);
    const auto f2 = estimate_footprint(dense, t2);
    const double mlp_ratio = double(f2.dense_mlp) / double(f1.dense_mlp);
    const double attn_ratio = double(f2.attn) / double(f1.attn);
    check(mlp_ratio > 0.49 && mlp_ratio < 0.51, "tp=2 halves the dense MLP weights");
    // Attention includes tiny replicated norms, so allow a hair above 0.5.
    check(attn_ratio > 0.49 && attn_ratio < 0.52, "tp=2 ~halves the attention weights");
  }

  // --- PP halves the number of layers on the heaviest rank ---
  {
    engine::PlanConfig p2 = base_cfg();
    p2.pp = 2;
    const auto f = estimate_footprint(dense, p2);
    check(f.layers_on_rank == 16, "pp=2 puts 16 of 32 layers on the heaviest rank");
    // Uneven: 32 layers over pp=5 -> ceil(32/5)=7 on the heaviest stage.
    engine::PlanConfig p5 = base_cfg();
    p5.pp = 5;
    check(estimate_footprint(dense, p5).layers_on_rank == 7,
          "pp=5 heaviest rank owns ceil(32/5)=7");
  }

  // --- EP halves the expert weights on a MoE model ---
  {
    const engine::ModelDims moe = moe_model();
    engine::PlanConfig e1 = base_cfg();
    engine::PlanConfig e2 = base_cfg();
    e2.ep = 2;
    const auto f1 = estimate_footprint(moe, e1);
    const auto f2 = estimate_footprint(moe, e2);
    check(f1.experts > 0 && f1.dense_mlp == 0, "MoE model has experts and no dense MLP");
    const double ratio = double(f2.experts) / double(f1.experts);
    check(ratio > 0.49 && ratio < 0.51, "ep=2 halves the expert weights");
  }

  // --- fit scan: every returned config fits the budget and results are world-sorted ---
  {
    const std::size_t budget = std::size_t(24) * 1024 * 1024 * 1024;  // 24 GB
    const auto opts =
        engine::scan_fit(dense, engine::Quant::FP16, engine::Quant::FP16, 8192, 1, budget);
    bool all_fit = true, sorted = true;
    for (std::size_t i = 0; i < opts.size(); ++i) {
      if (opts[i].per_rank_bytes > budget) all_fit = false;
      if (i && opts[i - 1].world() > opts[i].world()) sorted = false;
    }
    check(!opts.empty(), "fit scan finds at least one feasible split for the 8B on 24 GB");
    check(all_fit, "every scanned option fits the budget");
    check(sorted, "scan options are ordered by world size");
    // The 8B fp16 (~16 GB weights + KV) fits on one 24 GB GPU, so the smallest world must be 1.
    check(opts.front().world() == 1, "8B fp16 fits world=1 on 24 GB");
  }

  std::printf(failures ? "\n%d checks FAILED\n" : "\nall checks passed\n", failures);
  return failures ? 1 : 0;
}
