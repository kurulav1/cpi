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

  std::printf("op_plan_builder_test: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
