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

// ---------------------------------------------------------------------------
// Qwen3.5: mixed gated-delta-net / gated-full-attention decoder.
// ---------------------------------------------------------------------------
ModelPlan build_qwen35_plan(const Qwen35Geometry& g, const WeightSource& w) {
  if (static_cast<int>(g.layer_is_linear.size()) != g.num_layers) {
    throw std::runtime_error("build_qwen35_plan: layer_is_linear must have num_layers entries");
  }
  ModelPlan plan;
  const int H = g.hidden;
  const int q_dim = g.heads * g.head_dim;
  const int kv_dim = g.kv_heads * g.head_dim;
  const int lin_k_dim = g.lin_key_heads * g.lin_key_head_dim;
  const int lin_v_dim = g.lin_value_heads * g.lin_value_head_dim;
  const int lin_conv_dim = lin_k_dim * 2 + lin_v_dim;  // in_proj_qkv emits q|k|v back to back

  {
    Op e;
    e.kind = OpKind::EmbeddingLookup;
    e.out = Slot::X;
    e.cols = H;
    e.weight = w.fp16("tok_embeddings.weight");
    plan.prologue.push_back(e);
  }

  plan.layers.assign(g.num_layers, LayerPlan{});
  for (int L = 0; L < g.num_layers; ++L) {
    const std::string p = "layers." + std::to_string(L) + ".";
    LayerPlan& lp = plan.layers[L];
    lp.layer_index = L;
    auto& ops = lp.ops;

    // Qwen3.5 scales by (1 + w), and unlike Gemma the converter does NOT fold the +1 into the
    // stored weights -- that fold is gated on the Gemma family. So the offset is applied here,
    // matching the CUDA and CPU implementations. Getting it wrong scales every norm by ~2 and
    // compounds; it would surface as a divergence at the very first layer.
    auto rms = [&](Slot in, Slot out, const std::string& t, int rows, int cols) {
      Op o;
      o.kind = OpKind::RmsNorm;
      o.in = in;
      o.out = out;
      o.weight = w.fp16(p + t);
      o.rows = rows;
      o.cols = cols;
      o.norm_offset = true;
      ops.push_back(o);
    };
    auto gemv = [&](Slot in, Slot out, const std::string& t, int out_dim, int in_dim) {
      Op o;
      o.kind = OpKind::Gemv;
      o.in = in;
      o.out = out;
      o.cols = out_dim;
      o.in_dim = in_dim;
      const QuantWeight q = w.quant(p + t, out_dim, in_dim);
      if (q.packed != nullptr) {
        o.qweight = q.packed;
        o.qscales = q.scales;
        o.qbits = q.bits;
        o.qgroup = q.group;
      } else {
        o.weight = w.fp16(p + t);
      }
      ops.push_back(o);
    };
    auto add_x = [&](Slot src) {
      Op o;
      o.kind = OpKind::AddInplace;
      o.out = Slot::X;
      o.in = src;
      o.cols = H;
      ops.push_back(o);
    };

    if (!g.layer_is_linear[static_cast<std::size_t>(L)]) {
      // ---- gated full attention ----
      // wq is TWICE q_dim: it projects [q | gate] per head, and the gate multiplies the attention
      // output before o_proj. Treating it as an ordinary q projection reads a head_dim twice the
      // real one and converts cleanly into nonsense.
      rms(Slot::X, Slot::XNorm, "attention_norm.weight", 1, H);
      gemv(Slot::XNorm, Slot::QPair, "attention.wq", q_dim * 2, H);
      gemv(Slot::XNorm, Slot::K, "attention.wk", kv_dim, H);
      gemv(Slot::XNorm, Slot::V, "attention.wv", kv_dim, H);
      {
        Op o;
        o.kind = OpKind::SplitHeadHalves;
        o.in = Slot::QPair;
        o.out = Slot::Q;
        o.out2 = Slot::QGate;
        o.heads = g.heads;
        o.head_dim = g.head_dim;
        ops.push_back(o);
      }
      rms(Slot::Q, Slot::Q, "attention.q_norm", g.heads, g.head_dim);
      rms(Slot::K, Slot::K, "attention.k_norm", g.kv_heads, g.head_dim);
      {
        Op o;
        o.kind = OpKind::Rope;
        o.in = Slot::Q;
        o.in2 = Slot::K;
        o.heads = g.heads;
        o.kv_heads = g.kv_heads;
        o.head_dim = g.head_dim;
        o.rotary_dim = g.rotary_dim;  // partial: 0.25 of head_dim on this family
        o.rope_table = RopeTable::Full;
        o.scale = g.rope_theta;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::KvStore;
        o.in = Slot::K;
        o.in2 = Slot::V;
        o.kv_heads = g.kv_heads;
        o.head_dim = g.head_dim;
        o.cols = kv_dim;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::Attention;
        o.in = Slot::Q;
        o.out = Slot::Att;
        o.heads = g.heads;
        o.kv_heads = g.kv_heads;
        o.head_dim = g.head_dim;
        o.full_attention = true;
        o.sliding_window = 0;
        o.scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::SigmoidGate;
        o.out = Slot::Att;
        o.in2 = Slot::QGate;
        o.cols = q_dim;
        ops.push_back(o);
      }
      gemv(Slot::Att, Slot::Tmp, "attention.wo", H, q_dim);
    } else {
      // ---- gated delta-net ----
      // These layers ARE pre-normed, exactly like the attention ones -- the four projections read
      // the normalised state, not the raw residual. linear_attn.norm is a SECOND norm inside the
      // block (value_head_dim wide, gated by z) and does not stand in for this one. Feeding the
      // projections raw X leaves the whole block orthogonal to the reference at layer 1.
      rms(Slot::X, Slot::XNorm, "attention_norm.weight", 1, H);
      gemv(Slot::XNorm, Slot::LinMix, "linear_attn.in_proj_qkv", lin_conv_dim, H);
      gemv(Slot::XNorm, Slot::LinZ, "linear_attn.in_proj_z", lin_v_dim, H);
      gemv(Slot::XNorm, Slot::LinA, "linear_attn.in_proj_a", g.lin_value_heads, H);
      gemv(Slot::XNorm, Slot::LinB, "linear_attn.in_proj_b", g.lin_value_heads, H);
      {
        Op o;
        o.kind = OpKind::LinearConv1d;
        o.in = Slot::LinMix;
        o.weight = w.fp16(p + "linear_attn.conv1d");
        o.cols = lin_conv_dim;
        o.conv_kernel = g.lin_conv_kernel;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::RepeatLinearHeads;
        o.in = Slot::LinMix;
        o.num_k_heads = g.lin_key_heads;
        o.num_v_heads = g.lin_value_heads;
        o.key_head_dim = g.lin_key_head_dim;
        o.value_head_dim = g.lin_value_head_dim;
        ops.push_back(o);
      }
      {
        Op o;
        o.kind = OpKind::LinearAttentionStep;
        // auxf_a/auxf_b are declared const float* by the shared plan because CUDA reads them as
        // f32 from safetensors. On the .ll2c path every weight is fp16 (the converter casts all
        // of them), so these carry fp16 handles and the kernel reads them as half.
        o.auxf_a = static_cast<const float*>(w.fp16(p + "linear_attn.norm"));
        o.auxf_b = static_cast<const float*>(w.fp16(p + "linear_attn.a_log"));
        o.aux_ptr = w.fp16(p + "linear_attn.dt_bias");
        o.num_v_heads = g.lin_value_heads;
        o.key_head_dim = g.lin_key_head_dim;
        o.value_head_dim = g.lin_value_head_dim;
        o.eps = g.rms_eps;
        ops.push_back(o);
      }
      gemv(Slot::LinAtt, Slot::Tmp, "linear_attn.out_proj", H, lin_v_dim);
    }
    add_x(Slot::Tmp);

    // ---- MLP: SwiGLU, shared by both layer kinds ----
    rms(Slot::X, Slot::XNorm, "ffn_norm.weight", 1, H);
    gemv(Slot::XNorm, Slot::Gate, "feed_forward.w1", g.inter, H);
    gemv(Slot::XNorm, Slot::Up, "feed_forward.w3", g.inter, H);
    {
      Op o;
      o.kind = OpKind::SiluMul;
      o.in = Slot::Gate;
      o.in2 = Slot::Up;
      o.out = Slot::Inter;
      o.cols = g.inter;
      ops.push_back(o);
    }
    gemv(Slot::Inter, Slot::Tmp, "feed_forward.w2", H, g.inter);
    add_x(Slot::Tmp);
  }

  {
    Op o;
    o.kind = OpKind::RmsNorm;
    o.in = Slot::X;
    o.out = Slot::XNorm;
    o.weight = w.fp16("norm.weight");
    o.rows = 1;
    o.cols = H;
    o.norm_offset = true;
    plan.epilogue.push_back(o);
  }
  {
    Op o;
    o.kind = OpKind::LmHead;
    o.in = Slot::XNorm;
    o.cols = g.vocab;
    o.in_dim = H;
    o.weight = w.fp16(w.has("output.weight") ? "output.weight" : "tok_embeddings.weight");
    plan.epilogue.push_back(o);
  }
  return plan;
}

}  // namespace opplan
}  // namespace engine
