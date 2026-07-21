// Checks that build_llama_plan emits the op sequence a Llama-style decode needs.
//
// Runs everywhere -- no CUDA, no Metal, no GPU, no weights. The WeightSource is a
// stub that hands back a distinct fake handle per tensor name, so the test can also
// assert that each op got bound to the RIGHT tensor, not merely to something.
//
// This is what stops a backend bring-up from chasing a kernel bug that is really a
// plan bug.

#include "engine/op_plan_builder.hpp"

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace engine::opplan;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what.c_str());
    ++failures;
  }
}

// Hands out a unique non-null handle per name and records what was asked for.
class FakeWeights : public WeightSource {
public:
  const void* fp16(const std::string& name) const override {
    auto it = handles_.find(name);
    if (it == handles_.end()) {
      // Distinct, stable, non-null. Never dereferenced.
      const void* h = reinterpret_cast<const void*>(handles_.size() + 1);
      handles_[name] = h;
      return h;
    }
    return it->second;
  }
  bool has(const std::string& name) const override {
    return present_.count(name) != 0;
  }

  void add(const std::string& name) {
    present_.insert(name);
  }
  const void* handle_for(const std::string& name) const {
    return fp16(name);
  }

private:
  mutable std::map<std::string, const void*> handles_;
  std::set<std::string> present_;
};

const char* kind_name(OpKind k) {
  switch (k) {
    case OpKind::RmsNorm:
      return "RmsNorm";
    case OpKind::Gemv:
      return "Gemv";
    case OpKind::Rope:
      return "Rope";
    case OpKind::ScaleCopy:
      return "ScaleCopy";
    case OpKind::CopySlot:
      return "CopySlot";
    case OpKind::KvStore:
      return "KvStore";
    case OpKind::Attention:
      return "Attention";
    case OpKind::SiluMul:
      return "SiluMul";
    case OpKind::AddInplace:
      return "AddInplace";
    case OpKind::EmbeddingLookup:
      return "EmbeddingLookup";
    case OpKind::LmHead:
      return "LmHead";
    default:
      return "<other>";
  }
}

}  // namespace

int main() {
  // Qwen2.5-0.5B's actual geometry: GQA with 14 query heads over 2 kv heads.
  LlamaGeometry g;
  g.num_layers = 2;  // two is enough to prove the tower repeats
  g.hidden = 896;
  g.inter = 4864;
  g.heads = 14;
  g.kv_heads = 2;
  g.head_dim = 64;
  g.vocab = 151936;
  g.rms_eps = 1e-6f;
  g.rope_theta = 1000000.0f;

  FakeWeights w;
  w.add("output.weight");  // untied head

  const ModelPlan plan = build_llama_plan(g, w);

  // ---- prologue ----------------------------------------------------------
  expect(plan.prologue.size() == 1, "prologue is just the embedding lookup (no embed scale)");
  expect(plan.prologue[0].kind == OpKind::EmbeddingLookup, "prologue[0] is EmbeddingLookup");
  expect(plan.embed_ready == 1, "embed_ready points past the lookup");

  // ---- tower -------------------------------------------------------------
  expect(plan.layers.size() == 2, "two layers");

  const std::vector<OpKind> want = {
      OpKind::RmsNorm,     // attention norm
      OpKind::Gemv,        // wq
      OpKind::Gemv,        // wk
      OpKind::Gemv,        // wv
      OpKind::Rope,        // q
      OpKind::Rope,        // k
      OpKind::KvStore,     //
      OpKind::Attention,   //
      OpKind::Gemv,        // wo
      OpKind::AddInplace,  // residual
      OpKind::RmsNorm,     // ffn norm
      OpKind::Gemv,        // w1 gate
      OpKind::Gemv,        // w3 up
      OpKind::SiluMul,     // SwiGLU
      OpKind::Gemv,        // w2 down
      OpKind::AddInplace,  // residual
  };

  for (std::size_t L = 0; L < plan.layers.size(); ++L) {
    const auto& ops = plan.layers[L].ops;
    if (ops.size() != want.size()) {
      std::printf("  FAIL: layer %zu has %zu ops, want %zu\n", L, ops.size(), want.size());
      ++failures;
      continue;
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (ops[i].kind != want[i]) {
        std::printf("  FAIL: layer %zu op %zu is %s, want %s\n", L, i, kind_name(ops[i].kind),
                    kind_name(want[i]));
        ++failures;
      }
    }
    expect(plan.layers[L].layer_index == static_cast<int>(L), "layer_index is set");
  }

  // ---- geometry actually landed on the ops --------------------------------
  {
    const auto& ops = plan.layers[0].ops;
    expect(ops[1].cols == 14 * 64, "wq out_dim = heads * head_dim");
    expect(ops[2].cols == 2 * 64, "wk out_dim = kv_heads * head_dim (GQA, not heads)");
    expect(ops[3].cols == 2 * 64, "wv out_dim = kv_heads * head_dim");
    expect(ops[1].in_dim == 896, "wq in_dim = hidden");

    expect(ops[4].heads == 14, "RoPE over Q uses query head count");
    expect(ops[5].heads == 2, "RoPE over K uses KV head count");

    expect(ops[7].heads == 14 && ops[7].kv_heads == 2, "attention carries both head counts");
    expect(ops[7].full_attention, "Llama attention is full-causal");
    // 1/sqrt(64) = 0.125
    expect(ops[7].scale > 0.124f && ops[7].scale < 0.126f, "attention scale = 1/sqrt(head_dim)");

    expect(ops[8].cols == 896 && ops[8].in_dim == 14 * 64, "wo maps q_dim -> hidden");
    expect(ops[11].cols == 4864, "gate out_dim = inter");
    expect(ops[14].in_dim == 4864, "down in_dim = inter");
  }

  // ---- weights bound to the right tensors ---------------------------------
  {
    const auto& ops = plan.layers[1].ops;  // layer 1, so a wrong prefix would show
    expect(ops[1].weight == w.handle_for("layers.1.attention.wq"), "wq bound to layer 1's wq");
    expect(ops[2].weight == w.handle_for("layers.1.attention.wk"), "wk bound to layer 1's wk");
    expect(ops[8].weight == w.handle_for("layers.1.attention.wo"), "wo bound to layer 1's wo");
    expect(ops[11].weight == w.handle_for("layers.1.feed_forward.w1"), "gate bound to w1");
    expect(ops[13].weight == nullptr, "SiluMul takes no weight");

    // Llama/Mistral have no QKV bias. Binding one anyway would be silently wrong.
    expect(ops[1].bias == nullptr && ops[2].bias == nullptr && ops[3].bias == nullptr,
           "no QKV bias when has_qkv_bias is false");
  }

  // ---- Qwen2's QKV bias ---------------------------------------------------
  // Dropping it does not crash -- it yields fluent nonsense -- so it is asserted.
  {
    LlamaGeometry gq = g;
    gq.has_qkv_bias = true;
    FakeWeights wq;
    const ModelPlan pq = build_llama_plan(gq, wq);
    const auto& ops = pq.layers[1].ops;
    // The .ll2c fuses the three biases into one [bq | bk | bv] tensor, so all three
    // ops share a handle and differ only by offset. Getting the offsets wrong would
    // feed K the tail of Q's bias -- fluent nonsense, not a crash.
    const void* bqkv = wq.handle_for("layers.1.attention.bqkv");
    expect(ops[1].bias == bqkv && ops[2].bias == bqkv && ops[3].bias == bqkv,
           "Q/K/V biases all point at the fused bqkv tensor");
    expect(ops[1].bias_offset == 0, "Q bias starts at 0");
    expect(ops[2].bias_offset == 14 * 64, "K bias starts after Q (q_dim)");
    expect(ops[3].bias_offset == 14 * 64 + 2 * 64, "V bias starts after Q and K (q_dim + kv_dim)");
    expect(ops[8].bias == nullptr, "the o_proj has no bias");
    expect(ops[11].bias == nullptr, "the MLP has no bias");
  }

  // ---- Qwen3's per-head QK-norm -------------------------------------------
  // A per-head RMSNorm on Q and K, after projection and BEFORE RoPE. It is the
  // ordinary RmsNorm op with rows = heads, not a new kernel -- that is the point of
  // the seam. Note Qwen3's head_dim is NOT hidden/heads.
  {
    LlamaGeometry gq;
    gq.num_layers = 2;
    gq.hidden = 1024;
    gq.inter = 3072;
    gq.heads = 16;
    gq.kv_heads = 8;
    gq.head_dim = 128;  // q_dim = 2048 != hidden
    gq.vocab = 151936;
    gq.has_qk_norm = true;
    FakeWeights wq;
    const ModelPlan pq = build_llama_plan(gq, wq);
    const auto& ops = pq.layers[1].ops;

    expect(ops[4].kind == OpKind::RmsNorm, "Q-norm sits right after the Q/K/V projections");
    expect(ops[5].kind == OpKind::RmsNorm, "K-norm follows it");
    expect(ops[6].kind == OpKind::Rope, "and QK-norm comes BEFORE RoPE");

    expect(ops[4].rows == 16 && ops[4].cols == 128,
           "Q-norm is per-head: rows=heads, cols=head_dim");
    expect(ops[5].rows == 8 && ops[5].cols == 128, "K-norm uses the KV head count");
    expect(ops[4].weight == wq.handle_for("layers.1.attention.q_norm"), "Q-norm bound to q_norm");
    expect(ops[5].weight == wq.handle_for("layers.1.attention.k_norm"), "K-norm bound to k_norm");

    expect(ops[1].cols == 16 * 128, "wq out_dim = heads*head_dim = 2048, NOT hidden");
  }

  // ---- epilogue -----------------------------------------------------------
  expect(plan.epilogue.size() == 2, "epilogue is norm + lm_head");
  expect(plan.epilogue[0].kind == OpKind::RmsNorm, "epilogue[0] is the final norm");
  expect(plan.epilogue[1].kind == OpKind::LmHead, "epilogue[1] is the LM head");
  expect(plan.epilogue[1].cols == 151936, "lm head out_dim = vocab");
  expect(plan.epilogue[1].weight == w.handle_for("output.weight"),
         "untied head uses output.weight");

  // ---- a TIED head must fall back to the embedding table -------------------
  {
    FakeWeights tied;  // never add("output.weight")
    const ModelPlan p2 = build_llama_plan(g, tied);
    expect(p2.epilogue[1].weight == tied.handle_for("tok_embeddings.weight"),
           "tied head reuses tok_embeddings.weight");
  }

  // ---- embedding scale is opt-in ------------------------------------------
  {
    LlamaGeometry gs = g;
    gs.scale_embeddings = true;
    FakeWeights w2;
    const ModelPlan p3 = build_llama_plan(gs, w2);
    expect(p3.prologue.size() == 2, "scale_embeddings adds a ScaleCopy");
    expect(p3.prologue[1].kind == OpKind::ScaleCopy, "prologue[1] is ScaleCopy");
    expect(p3.embed_ready == 2, "embed_ready sits AFTER the scale (image splice point)");
    // sqrt(896) ~= 29.9333
    expect(p3.prologue[1].scale > 29.9f && p3.prologue[1].scale < 30.0f, "scale = sqrt(hidden)");
  }

  // ---- Gemma 4 -------------------------------------------------------------
  //
  // E2B's shape, scaled down so the op lists stay readable: alternating sliding/full layers with
  // a DIFFERENT head_dim each, the tail of them sharing KV, per-layer embeddings, and a
  // double-wide MLP on the shared layers. Each check below is a rule that fails silently -- the
  // model still runs and still produces fluent-looking text when any of them is wrong.
  {
    Gemma4Geometry gg;
    gg.num_layers = 6;
    gg.hidden = 64;
    gg.inter = 128;
    gg.vocab = 999;
    gg.heads = 4;
    // Deliberately NOT 1e-6f: that is Op::eps's own default, so a builder that never assigns eps
    // would still satisfy the "carries eps" check below. The control caught this test passing
    // while measuring nothing -- the value has to be one the default cannot accidentally match.
    gg.rms_eps = 7.5e-5f;
    gg.sliding_window = 512;
    gg.head_dim_sliding = 16;
    gg.head_dim_full = 32;
    gg.kv_heads_sliding = 2;
    gg.kv_heads_full = 1;
    gg.layer_full = {0, 0, 1, 0, 0, 1};
    gg.first_shared_layer = 4;  // layers 4,5 reuse someone else's K/V
    gg.use_double_wide_mlp = true;
    gg.ple = 8;
    gg.layer_scalar = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    gg.rope_theta_sliding = 10000.0f;
    gg.rope_theta_full = 1000000.0f;
    gg.partial_rotary_full = 0.25f;

    FakeWeights wg;
    const ModelPlan pg = build_gemma4_plan(gg, wg);

    expect(static_cast<int>(pg.layers.size()) == 6, "gemma4: one LayerPlan per layer");
    // The image splice point must sit after the embedding scale but BEFORE the PLE build, because
    // the per-layer inputs are projected FROM the embeddings. Splicing after the whole prologue
    // would compute them from the placeholder token instead of the image.
    expect(pg.embed_ready == 2, "gemma4: embed_ready precedes the PLE build");
    expect(pg.prologue.size() > 2, "gemma4: PLE build emitted into the prologue");

    // KV sharing: a shared layer must project NO K/V and store nothing -- emitting the projections
    // would not merely waste work, it would overwrite the cache it is supposed to be reading.
    auto has_kind = [](const std::vector<Op>& ops, OpKind k) {
      for (const Op& o : ops)
        if (o.kind == k) return true;
      return false;
    };
    expect(has_kind(pg.layers[0].ops, OpKind::KvStore), "gemma4: layer 0 stores KV");
    expect(!has_kind(pg.layers[4].ops, OpKind::KvStore), "gemma4: shared layer 4 stores no KV");
    expect(!has_kind(pg.layers[5].ops, OpKind::KvStore), "gemma4: shared layer 5 stores no KV");

    // Per-layer-type head_dim / kv-heads. One head_dim cannot describe this model.
    const Op* att0 = nullptr;
    const Op* att2 = nullptr;
    for (const Op& o : pg.layers[0].ops)
      if (o.kind == OpKind::Attention) att0 = &o;
    for (const Op& o : pg.layers[2].ops)
      if (o.kind == OpKind::Attention) att2 = &o;
    expect(att0 && att0->head_dim == 16 && att0->kv_heads == 2, "gemma4: sliding layer 16/2");
    expect(att2 && att2->head_dim == 32 && att2->kv_heads == 1, "gemma4: full layer 32/1");
    expect(att0 && !att0->full_attention, "gemma4: layer 0 is sliding");
    expect(att2 && att2->full_attention, "gemma4: layer 2 is full");

    // THE TRAP THAT PROMPTED THIS TEST: the CUDA executor reads a global rms_eps and would run
    // fine with op.eps unset; the Metal executor reads op.eps. A plan missing it normalises with
    // no epsilon on one backend only.
    int norms = 0, norms_with_eps = 0, offset_norms = 0;
    auto scan = [&](const std::vector<Op>& ops) {
      for (const Op& o : ops) {
        if (o.kind != OpKind::RmsNorm) continue;
        ++norms;
        if (o.eps == gg.rms_eps) ++norms_with_eps;
        if (o.norm_offset) ++offset_norms;
      }
    };
    scan(pg.prologue);
    for (const LayerPlan& lp : pg.layers) scan(lp.ops);
    scan(pg.epilogue);
    expect(norms > 0 && norms == norms_with_eps, "gemma4: EVERY RmsNorm carries eps");
    // Gemma 4 stores raw gains, not (w - 1). Applying the offset form doubles every norm, which
    // compounds and overflows fp16 around layer 2.
    expect(offset_norms == 0, "gemma4: no RmsNorm uses the (1+w) offset form");

    // RoPE carries its configuration as VALUES, not just as the rope_table enum. A backend that
    // computes angles in-shader (Metal) never reads rope_table, and Op::scale defaults to 1.0f --
    // a theta of 1.0 rotates every lane at the same frequency and is a wrong model, not an error.
    auto rope_of = [&](int L) {
      const Op* r = nullptr;
      for (const Op& o : pg.layers[L].ops)
        if (o.kind == OpKind::Rope && r == nullptr) r = &o;
      return r;
    };
    const Op* r0 = rope_of(0);  // sliding
    const Op* r2 = rope_of(2);  // full
    expect(r0 && r0->scale == 10000.0f, "gemma4: sliding layer rope theta 1e4");
    expect(r2 && r2->scale == 1000000.0f, "gemma4: full layer rope theta 1e6");
    expect(r0 && r0->rope_table == RopeTable::Sliding, "gemma4: sliding layer names its table");
    expect(r2 && r2->rope_table == RopeTable::Full, "gemma4: full layer names its table");
    // Partial rotary on the FULL layers only, and even: 32 * 0.25 = 8.
    expect(r0 && r0->rotary_dim == 0, "gemma4: sliding layer rotates all of head_dim");
    expect(r2 && r2->rotary_dim == 8 && r2->rotary_dim % 2 == 0,
           "gemma4: full layer rotary_dim is 0.25*head_dim and even");

    // PLE injection reads layer L's window of the packed per-layer table.
    for (int L = 0; L < gg.num_layers; ++L) {
      bool ok = false;
      for (const Op& o : pg.layers[L].ops)
        if (o.kind == OpKind::GeluMul && o.in2 == Slot::PleAll && o.aux_offset == L * gg.ple)
          ok = true;
      expect(ok, "gemma4: layer " + std::to_string(L) + " gates its own PLE window");
    }

    // Double-wide MLP on the shared layers only.
    auto mlp_width = [&](int L) {
      for (const Op& o : pg.layers[L].ops)
        if (o.kind == OpKind::GeluMul && o.out == Slot::Inter) return o.cols;
      return -1;
    };
    expect(mlp_width(0) == 128, "gemma4: normal layer MLP is inter");
    expect(mlp_width(4) == 256, "gemma4: shared layer MLP is double-wide");

    // Per-layer output gain, in order.
    expect(pg.layers[3].ops.back().kind == OpKind::ScaleCopy &&
               pg.layers[3].ops.back().scale == 4.0f,
           "gemma4: layer ends with its own scalar gain");

    // ---- the 12B / MoE shape: no PLE at all ----
    Gemma4Geometry g12 = gg;
    g12.ple = 0;
    g12.first_shared_layer = 0;  // no sharing either
    FakeWeights w12;
    const ModelPlan p12 = build_gemma4_plan(g12, w12);
    expect(p12.prologue.size() == 2, "gemma4: no PLE => prologue is just embed+scale");
    bool any_ple = false;
    for (const LayerPlan& lp : p12.layers)
      for (const Op& o : lp.ops)
        if (o.in2 == Slot::PleAll || o.out == Slot::PleGate) any_ple = true;
    expect(!any_ple, "gemma4: no PLE => no per-layer injection");
    for (int L = 0; L < g12.num_layers; ++L)
      expect(has_kind(p12.layers[L].ops, OpKind::KvStore),
             "gemma4: no sharing => every layer stores KV");
  }

  std::printf("op_plan_builder_test: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
