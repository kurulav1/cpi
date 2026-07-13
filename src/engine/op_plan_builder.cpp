// Backend-neutral plan builder for uniform-geometry Llama-style decoders.
//
// Emits exactly the eleven ops a dense fp16 decode needs, so any backend that
// implements those eleven can run the result. Compiles with no CUDA and no Metal.

#include "engine/op_plan_builder.hpp"

#include <cmath>
#include <stdexcept>

namespace engine {
namespace opplan {

namespace {

Op gemv(Slot in, Slot out, const void* w, int out_dim, int in_dim) {
  Op o;
  o.kind = OpKind::Gemv;
  o.in = in;
  o.out = out;
  o.weight = w;
  o.cols = out_dim;
  o.in_dim = in_dim;
  return o;
}

Op rmsnorm(Slot in, Slot out, const void* w, int cols, float eps) {
  Op o;
  o.kind = OpKind::RmsNorm;
  o.in = in;
  o.out = out;
  o.weight = w;
  o.rows = 1;
  o.cols = cols;
  o.eps = eps;
  return o;
}

Op add_inplace(Slot in) {
  Op o;
  o.kind = OpKind::AddInplace;
  o.in = in;
  o.out = Slot::X;
  return o;
}

}  // namespace

ModelPlan build_llama_plan(const LlamaGeometry& g, const WeightSource& w) {
  if (g.num_layers <= 0 || g.hidden <= 0 || g.heads <= 0 || g.kv_heads <= 0) {
    throw std::runtime_error("build_llama_plan: incomplete geometry");
  }
  if (g.heads % g.kv_heads != 0) {
    throw std::runtime_error("build_llama_plan: heads must be a multiple of kv_heads");
  }

  const int q_dim = g.heads * g.head_dim;
  const int kv_dim = g.kv_heads * g.head_dim;
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));

  ModelPlan plan;

  // ---- prologue: token id -> embeddings -----------------------------------
  {
    Op o;
    o.kind = OpKind::EmbeddingLookup;
    o.out = Slot::X;
    o.weight = w.fp16("tok_embeddings.weight");
    o.cols = g.hidden;
    plan.prologue.push_back(o);

    if (g.scale_embeddings) {
      Op s;
      s.kind = OpKind::ScaleCopy;
      s.in = Slot::X;
      s.out = Slot::X;
      s.cols = g.hidden;
      s.scale = std::sqrt(static_cast<float>(g.hidden));
      plan.prologue.push_back(s);
    }
    // Embeddings are final here; multimodal prefill splices at this index.
    plan.embed_ready = plan.prologue.size();
  }

  // ---- the tower ----------------------------------------------------------
  for (int L = 0; L < g.num_layers; ++L) {
    const std::string p = g.layer_prefix + std::to_string(L) + ".";
    LayerPlan lp;
    lp.layer_index = L;

    // Attention block.
    lp.ops.push_back(
        rmsnorm(Slot::X, Slot::XNorm, w.fp16(p + "attention_norm.weight"), g.hidden, g.rms_eps));
    lp.ops.push_back(gemv(Slot::XNorm, Slot::Q, w.fp16(p + "attention.wq"), q_dim, g.hidden));
    lp.ops.push_back(gemv(Slot::XNorm, Slot::K, w.fp16(p + "attention.wk"), kv_dim, g.hidden));
    lp.ops.push_back(gemv(Slot::XNorm, Slot::V, w.fp16(p + "attention.wv"), kv_dim, g.hidden));

    // RoPE rotates Q and K in place. Two ops -- they differ only in head count,
    // and keeping them separate is what lets a GQA model share one kernel.
    {
      Op q;
      q.kind = OpKind::Rope;
      q.in = Slot::Q;
      q.out = Slot::Q;
      q.heads = g.heads;
      q.head_dim = g.head_dim;
      q.scale = g.rope_theta;
      lp.ops.push_back(q);

      Op k;
      k.kind = OpKind::Rope;
      k.in = Slot::K;
      k.out = Slot::K;
      k.heads = g.kv_heads;
      k.head_dim = g.head_dim;
      k.scale = g.rope_theta;
      lp.ops.push_back(k);
    }

    {
      Op kv;
      kv.kind = OpKind::KvStore;
      kv.in = Slot::K;
      kv.in2 = Slot::V;
      kv.kv_heads = g.kv_heads;
      kv.head_dim = g.head_dim;
      lp.ops.push_back(kv);
    }

    {
      Op a;
      a.kind = OpKind::Attention;
      a.in = Slot::Q;
      a.out = Slot::Att;
      a.heads = g.heads;
      a.kv_heads = g.kv_heads;
      a.head_dim = g.head_dim;
      a.full_attention = true;
      a.sliding_window = 0;
      a.scale = attn_scale;
      lp.ops.push_back(a);
    }

    lp.ops.push_back(gemv(Slot::Att, Slot::Tmp, w.fp16(p + "attention.wo"), g.hidden, q_dim));
    lp.ops.push_back(add_inplace(Slot::Tmp));

    // MLP block (SwiGLU).
    lp.ops.push_back(
        rmsnorm(Slot::X, Slot::XNorm, w.fp16(p + "ffn_norm.weight"), g.hidden, g.rms_eps));
    lp.ops.push_back(
        gemv(Slot::XNorm, Slot::Gate, w.fp16(p + "feed_forward.w1"), g.inter, g.hidden));
    lp.ops.push_back(gemv(Slot::XNorm, Slot::Up, w.fp16(p + "feed_forward.w3"), g.inter, g.hidden));
    {
      Op s;
      s.kind = OpKind::SiluMul;
      s.in = Slot::Gate;
      s.in2 = Slot::Up;
      s.out = Slot::Inter;
      s.cols = g.inter;
      lp.ops.push_back(s);
    }
    lp.ops.push_back(
        gemv(Slot::Inter, Slot::Tmp, w.fp16(p + "feed_forward.w2"), g.hidden, g.inter));
    lp.ops.push_back(add_inplace(Slot::Tmp));

    plan.layers.push_back(std::move(lp));
  }

  // ---- epilogue: final norm -> logits -------------------------------------
  plan.epilogue.push_back(
      rmsnorm(Slot::X, Slot::XNorm, w.fp16("norm.weight"), g.hidden, g.rms_eps));
  {
    // A tied LM head reuses the embedding table -- same weights, transposed use.
    const char* head = w.has("output.weight") ? "output.weight" : "tok_embeddings.weight";
    Op o;
    o.kind = OpKind::LmHead;
    o.in = Slot::XNorm;
    o.weight = w.fp16(head);
    o.cols = g.vocab;
    o.in_dim = g.hidden;
    plan.epilogue.push_back(o);
  }

  return plan;
}

}  // namespace opplan
}  // namespace engine
