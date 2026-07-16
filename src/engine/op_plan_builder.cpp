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

// A projection. Asks the backend whether it wants this weight quantized; if so the op
// carries the packed handle and its scales instead of the fp16 one, and the executor
// dispatches a weight-only matvec. Norms and the embedding table are never quantized --
// they are lookups and elementwise scales, not matmuls, and quantizing them buys
// nothing while costing accuracy.
Op gemv(Slot in, Slot out, const WeightSource& w, const std::string& name, int out_dim,
        int in_dim) {
  Op o;
  o.kind = OpKind::Gemv;
  o.in = in;
  o.out = out;
  o.cols = out_dim;
  o.in_dim = in_dim;

  const QuantWeight q = w.quant(name, out_dim, in_dim);
  if (q.bits != 0) {
    o.qweight = q.packed;
    o.qscales = q.scales;
    o.qbits = q.bits;
    o.qgroup = q.group;
  } else {
    o.weight = w.fp16(name);
  }
  return o;
}

Op rmsnorm(Slot in, Slot out, const void* w, int cols, float eps, bool offset = false) {
  Op o;
  o.kind = OpKind::RmsNorm;
  o.in = in;
  o.out = out;
  o.weight = w;
  o.rows = 1;
  o.cols = cols;
  o.eps = eps;
  o.norm_offset = offset;  // Gemma: scale by (1 + w)
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
    lp.ops.push_back(rmsnorm(Slot::X, Slot::XNorm, w.fp16(p + "attention_norm.weight"), g.hidden,
                             g.rms_eps, g.norm_offset));
    {
      Op q = gemv(Slot::XNorm, Slot::Q, w, p + "attention.wq", q_dim, g.hidden);
      Op k = gemv(Slot::XNorm, Slot::K, w, p + "attention.wk", kv_dim, g.hidden);
      Op v = gemv(Slot::XNorm, Slot::V, w, p + "attention.wv", kv_dim, g.hidden);
      if (g.has_qkv_bias) {
        // One fused tensor, laid out [bq (q_dim) | bk (kv_dim) | bv (kv_dim)].
        const void* bqkv = w.fp16(p + "attention.bqkv");
        q.bias = bqkv;
        q.bias_offset = 0;
        k.bias = bqkv;
        k.bias_offset = q_dim;
        v.bias = bqkv;
        v.bias_offset = q_dim + kv_dim;
      }
      lp.ops.push_back(q);
      lp.ops.push_back(k);
      lp.ops.push_back(v);
    }

    // Qwen3: per-head RMSNorm on Q and K, after projection and BEFORE RoPE. One
    // [head_dim] weight shared across heads, so it is the ordinary RmsNorm op with
    // rows = heads -- a new capability, not a new kernel.
    if (g.has_qk_norm) {
      Op qn = rmsnorm(Slot::Q, Slot::Q, w.fp16(p + "attention.q_norm"), g.head_dim, g.rms_eps,
                      g.norm_offset);
      qn.rows = g.heads;
      lp.ops.push_back(qn);

      Op kn = rmsnorm(Slot::K, Slot::K, w.fp16(p + "attention.k_norm"), g.head_dim, g.rms_eps,
                      g.norm_offset);
      kn.rows = g.kv_heads;
      lp.ops.push_back(kn);
    }

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

    lp.ops.push_back(gemv(Slot::Att, Slot::Tmp, w, p + "attention.wo", g.hidden, q_dim));
    lp.ops.push_back(add_inplace(Slot::Tmp));

    // MLP block. Dense SwiGLU/GeGLU, or -- for an MoE model -- router + selected experts.
    // The norm and the residual add are shared: only what sits between them differs.
    lp.ops.push_back(rmsnorm(Slot::X, Slot::XNorm, w.fp16(p + "ffn_norm.weight"), g.hidden,
                             g.rms_eps, g.norm_offset));
    if (g.num_experts > 0) {
      const int einter = g.expert_inter > 0 ? g.expert_inter : g.inter;
      // Router logits: [experts]. Tiny (experts x hidden), so it stays fp16 even when the
      // projections are quantized -- quantizing a matrix this small buys nothing and the
      // routing decision is the one place an error picks a different expert outright.
      lp.ops.push_back(gemv(Slot::XNorm, Slot::MoeLogits, w, p + "feed_forward.router",
                            g.num_experts, g.hidden));
      {
        // softmax -> top-k -> renormalise over the picked ones. Mixtral has no per-expert
        // gain (that is Gemma 4), so weight stays null and the kernel skips it.
        Op o;
        o.kind = OpKind::MoeRouterTopk;
        o.in = Slot::MoeLogits;
        o.cols = g.num_experts;
        o.heads = g.experts_per_tok;
        lp.ops.push_back(o);
      }
      {
        // Per selected expert: act(gate) * up -> MoeInter[k]. Mixtral is SwiGLU, and the op
        // name says Geglu for historical reasons -- the kernel takes the activation from the
        // geometry, as the dense path does.
        Op o;
        o.kind = OpKind::MoeGateUpGeglu;
        o.in = Slot::XNorm;
        o.out = Slot::MoeInter;
        o.weight = w.fp16(p + "feed_forward.experts.gate_up");
        o.cols = einter;
        o.in_dim = g.hidden;
        o.heads = g.experts_per_tok;
        o.mlp_gelu = g.mlp_gelu;
        lp.ops.push_back(o);
      }
      {
        // sum_k routing_weight[k] * down[expert_k] . MoeInter[k] -> Tmp.
        Op o;
        o.kind = OpKind::MoeDownAccum;
        o.in = Slot::MoeInter;
        o.out = Slot::Tmp;
        o.weight = w.fp16(p + "feed_forward.experts.down");
        o.cols = g.hidden;
        o.in_dim = einter;
        o.heads = g.experts_per_tok;
        lp.ops.push_back(o);
      }
      lp.ops.push_back(add_inplace(Slot::Tmp));
      plan.layers.push_back(std::move(lp));
      continue;
    }
    lp.ops.push_back(gemv(Slot::XNorm, Slot::Gate, w, p + "feed_forward.w1", g.inter, g.hidden));
    lp.ops.push_back(gemv(Slot::XNorm, Slot::Up, w, p + "feed_forward.w3", g.inter, g.hidden));
    {
      // Gemma uses GeGLU where Llama uses SwiGLU. Both kernels already exist; the
      // model just picks one.
      Op s;
      s.kind = g.mlp_gelu ? OpKind::GeluMul : OpKind::SiluMul;
      s.in = Slot::Gate;
      s.in2 = Slot::Up;
      s.out = Slot::Inter;
      s.cols = g.inter;
      lp.ops.push_back(s);
    }
    lp.ops.push_back(gemv(Slot::Inter, Slot::Tmp, w, p + "feed_forward.w2", g.hidden, g.inter));
    lp.ops.push_back(add_inplace(Slot::Tmp));

    plan.layers.push_back(std::move(lp));
  }

  // ---- epilogue: final norm -> logits -------------------------------------
  plan.epilogue.push_back(
      rmsnorm(Slot::X, Slot::XNorm, w.fp16("norm.weight"), g.hidden, g.rms_eps, g.norm_offset));
  {
    // A tied LM head reuses the embedding table -- same weights, transposed use.
    const bool untied = w.has("output.weight");
    const std::string head = untied ? "output.weight" : "tok_embeddings.weight";
    Op o;
    o.kind = OpKind::LmHead;
    o.in = Slot::XNorm;
    o.cols = g.vocab;
    o.in_dim = g.hidden;

    // The LM head is the single biggest read of a decode step (vocab x hidden), so it
    // is the most valuable thing to quantize. But only when it is UNTIED: a tied head
    // shares storage with the embedding table, which is looked up as fp16, and
    // quantizing it would corrupt the lookup.
    const QuantWeight q = untied ? w.quant(head, g.vocab, g.hidden) : QuantWeight{};
    if (q.bits != 0) {
      o.qweight = q.packed;
      o.qscales = q.scales;
      o.qbits = q.bits;
      o.qgroup = q.group;
    } else {
      o.weight = w.fp16(head);
    }
    plan.epilogue.push_back(o);
  }

  return plan;
}

}  // namespace opplan
}  // namespace engine
