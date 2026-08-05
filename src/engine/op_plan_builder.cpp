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
// dispatches a weight-only matvec. Norms and the embedding table are never quantized
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

// How an embedding table is stored, for all three builders. A backend that can gather from a
// packed row takes the quantized table; everything else keeps the fp16 handle.
//
// A tied head costs nothing extra: WeightSource::quant caches by tensor name, so the LmHead's
// packed copy is the same buffer, and the fp16 one (materialised on demand) is never allocated.
Op make_embed(const WeightSource& w, const std::string& name, Slot out, int cols, int vocab) {
  Op o;
  o.kind = OpKind::EmbeddingLookup;
  o.out = out;
  o.cols = cols;
  if (w.quantize_embeddings() && vocab > 0) {
    const QuantWeight q = w.quant_embedding(name, vocab, cols);
    if (q.bits != 0) {
      o.qweight = q.packed;
      o.qscales = q.scales;
      o.qbits = q.bits;
      o.qgroup = q.group;
      return o;
    }
  }
  o.weight = w.fp16(name);
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
    plan.prologue.push_back(make_embed(w, "tok_embeddings.weight", Slot::X, g.hidden, g.vocab));

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

    // Qwen3: per-head RMSNorm on Q and K, after projection and before RoPE. One
    // [head_dim] weight shared across heads, so it is the ordinary RmsNorm op with
    // rows = heads; a new capability, not a new kernel.
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

    // RoPE rotates Q and K in place. Two ops; they differ only in head count,
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

    // MLP block. Dense SwiGLU/GeGLU, or, for an MoE model, router + selected experts.
    // The norm and the residual add are shared: only what sits between them differs.
    lp.ops.push_back(rmsnorm(Slot::X, Slot::XNorm, w.fp16(p + "ffn_norm.weight"), g.hidden,
                             g.rms_eps, g.norm_offset));
    if (g.num_experts > 0) {
      const int einter = g.expert_inter > 0 ? g.expert_inter : g.inter;
      // Router logits: [experts]. Tiny (experts x hidden), so it stays fp16 even when the
      // projections are quantized; quantizing a matrix this small buys nothing and the
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
        // name says Geglu for historical reasons; the kernel takes the activation from the
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
    // A tied LM head reuses the embedding table; same weights, transposed use.
    const bool untied = w.has("output.weight");
    const std::string head = untied ? "output.weight" : "tok_embeddings.weight";
    Op o;
    o.kind = OpKind::LmHead;
    o.in = Slot::XNorm;
    o.cols = g.vocab;
    o.in_dim = g.hidden;

    // The LM head is the single biggest read of a decode step (vocab x hidden), so it
    // is the most valuable thing to quantize, tied or not. A tied head shares weights
    // with the embedding table, not storage: WeightSource::quant packs a separate copy,
    // and the EmbeddingLookup op keeps its own fp16 handle, so the lookup is untouched.
    // (This used to skip tied heads, which left Qwen2.5 and Gemma paying a full fp16
    // vocab x hidden read per token in every quant mode; the single largest slice of
    // their decode traffic.)
    const QuantWeight q = w.quant(head, g.vocab, g.hidden);
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
    plan.prologue.push_back(make_embed(w, "tok_embeddings.weight", Slot::X, H, g.vocab));
  }
  // Embeddings are final here (no scale, no PLE); multimodal prefill splices at this index.
  // Every builder must set this: an unset (0) embed_ready puts the splice before the lookup,
  // and the lookup then overwrites every soft token. SPLICE_is_load_bearing gates it.
  plan.embed_ready = plan.prologue.size();

  plan.layers.assign(g.num_layers, LayerPlan{});
  for (int L = 0; L < g.num_layers; ++L) {
    const std::string p = "layers." + std::to_string(L) + ".";
    LayerPlan& lp = plan.layers[L];
    lp.layer_index = L;
    auto& ops = lp.ops;

    // Qwen3.5 scales by (1 + w), and unlike Gemma the converter does not fold the +1 into the
    // stored weights; that fold is gated on the Gemma family. So the offset is applied here,
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
      // wq is twice q_dim: it projects [q | gate] per head, and the gate multiplies the attention
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
      // These layers are pre-normed, exactly like the attention ones; the four projections read
      // the normalised state, not the raw residual. linear_attn.norm is a second norm inside the
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
    // Same tied-head quantization as the other builders: a separate packed copy for the
    // GEMV, the fp16 table untouched for the lookup.
    const std::string head = w.has("output.weight") ? "output.weight" : "tok_embeddings.weight";
    const QuantWeight q = w.quant(head, g.vocab, H);
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

ModelPlan build_gemma4_plan(const Gemma4Geometry& g, const WeightSource& w) {
  if (g.num_layers <= 0 || g.hidden <= 0 || g.heads <= 0)
    throw std::runtime_error("build_gemma4_plan: degenerate geometry");
  if (static_cast<int>(g.layer_full.size()) != g.num_layers)
    throw std::runtime_error("build_gemma4_plan: layer_full must be num_layers long");
  if (!g.layer_scalar.empty() && static_cast<int>(g.layer_scalar.size()) != g.num_layers)
    throw std::runtime_error("build_gemma4_plan: layer_scalar must be empty or num_layers long");

  ModelPlan plan;
  const int H = g.hidden;
  const std::string& WP = g.weight_prefix;

  // Every RmsNorm below carries `eps` explicitly. The CUDA executor reads a global cfg_.rms_eps
  // and would not notice it missing; the Metal executor reads op.eps, so a plan built without it
  // normalises by 1/sqrt(mean) with no epsilon at all. Same plan, two executors, and only one of
  // them would have told you.
  auto norm = [&](Slot in, Slot out, const void* weight, int rows, int cols) {
    Op o;
    o.kind = OpKind::RmsNorm;
    o.in = in;
    o.out = out;
    o.weight = weight;  // null => weightless (ones), used for Gemma's v-norm
    o.rows = rows;
    o.cols = cols;
    o.eps = g.rms_eps;
    return o;  // NOTE: norm_offset stays false; Gemma 4 stores raw gains, not (w - 1).
  };
  auto scale = [&](Slot s, int len, float sc) {
    Op o;
    o.kind = OpKind::ScaleCopy;
    o.in = s;
    o.out = s;
    o.cols = len;
    o.scale = sc;
    return o;
  };
  auto add_into = [&](Slot dst, Slot src, int cols) {
    Op o;
    o.kind = OpKind::AddInplace;
    o.out = dst;
    o.in = src;
    o.cols = cols;
    return o;
  };
  auto embed = [&](const std::string& name, Slot out, int dim) {
    return make_embed(w, name, out, dim, g.vocab);
  };

  // ── prologue: token -> embeddings (+ scale, and the PLE build when present) ──
  {
    std::vector<Op>& pro = plan.prologue;
    pro.push_back(embed(WP + "embed_tokens.weight", Slot::X, H));
    pro.push_back(scale(Slot::X, H, std::sqrt(static_cast<float>(H))));
    // Image embeddings splice in here, before anything reads X; and the per-layer-input
    // projection below does read X, which is why this marker is before it and not after the
    // prologue (HF projects the per-layer inputs from the post-scatter embeddings).
    plan.embed_ready = pro.size();
    if (g.has_ple()) {
      const int ple = g.ple;
      const int tot = g.num_layers * ple;
      // ple_raw = embed_tokens_per_layer[token] * sqrt(ple)
      pro.push_back(embed(WP + "embed_tokens_per_layer.weight", Slot::PleRaw, tot));
      pro.push_back(scale(Slot::PleRaw, tot, std::sqrt(static_cast<float>(ple))));
      // ple = rmsnorm( (W_proj . x) * hidden^-0.5 )   [x = the scaled embeddings in Slot::X]
      pro.push_back(
          gemv(Slot::X, Slot::PleAll, w, WP + "per_layer_model_projection.weight", tot, H));
      pro.push_back(scale(Slot::PleAll, tot, std::pow(static_cast<float>(H), -0.5f)));
      pro.push_back(norm(Slot::PleAll, Slot::PleAll,
                         w.fp16(WP + "per_layer_projection_norm.weight"), g.num_layers, ple));
      // per_layer_inputs = (ple + ple_raw) * 2^-0.5
      pro.push_back(add_into(Slot::PleAll, Slot::PleRaw, tot));
      pro.push_back(scale(Slot::PleAll, tot, std::pow(2.0f, -0.5f)));
    }
  }

  // ── the tower ──
  plan.layers.assign(g.num_layers, LayerPlan{});
  for (int L = 0; L < g.num_layers; ++L) {
    const std::string p = WP + "layers." + std::to_string(L) + ".";
    const int hd = g.head_dim_of(L);
    const int nq = g.heads, nkv = g.kv_heads_of(L);
    const int qdim = nq * hd, kvdim = nkv * hd;
    const bool full = g.layer_full[L] != 0;
    const bool shared = g.is_shared(L);
    const int inter = g.inter * ((g.use_double_wide_mlp && shared) ? 2 : 1);

    LayerPlan& lp = plan.layers[L];
    lp.layer_index = L;
    std::vector<Op>& ops = lp.ops;
    // Gemma 4's two RoPE configurations. rope_table names which for backends that precompute
    // tables (CUDA); scale and rotary_dim carry the same choice as values, for backends that
    // compute the angles in-shader (Metal, which does not look at rope_table at all). Both are set
    // because the plan is one description shared by both executors; setting only the enum leaves
    // Metal on Op::scale's default theta of 1.0, i.e. every lane at the same frequency.
    const float theta = full ? g.rope_theta_full : g.rope_theta_sliding;
    // Partial rotary applies to the full layers only, and must be even; an odd count leaves a
    // half-pair no rotation kernel can turn. 0 means "all of head_dim", which is what the sliding
    // layers want.
    int rot = 0;
    if (full && g.partial_rotary_full > 0.0f && g.partial_rotary_full < 1.0f) {
      rot = static_cast<int>(static_cast<float>(hd) * g.partial_rotary_full);
      rot -= rot % 2;
      if (rot <= 0) rot = hd - (hd % 2);
    }
    auto rope = [&](Slot s, int heads) {
      Op o;
      o.kind = OpKind::Rope;
      o.in = s;
      o.out = s;
      o.heads = heads;
      o.head_dim = hd;
      o.rope_table = full ? RopeTable::Full : RopeTable::Sliding;
      o.scale = theta;
      o.rotary_dim = rot;
      return o;
    };

    // --- attention ---
    ops.push_back(norm(Slot::X, Slot::XNorm, w.fp16(p + "input_layernorm.weight"), 1, H));
    ops.push_back(gemv(Slot::XNorm, Slot::Q, w, p + "self_attn.q_proj.weight", qdim, H));
    ops.push_back(norm(Slot::Q, Slot::Q, w.fp16(p + "self_attn.q_norm.weight"), nq, hd));
    ops.push_back(rope(Slot::Q, nq));
    // A KV-shared layer projects no K/V at all: it reads the cache another layer filled. Emitting
    // the projections anyway would not just waste work, it would overwrite that cache.
    if (!shared) {
      ops.push_back(gemv(Slot::XNorm, Slot::K, w, p + "self_attn.k_proj.weight", kvdim, H));
      if (g.k_eq_v(L)) {
        Op c;  // V shares the raw k_proj output; copied before k_norm and rope touch K.
        c.kind = OpKind::CopySlot;
        c.in = Slot::K;
        c.out = Slot::V;
        c.cols = kvdim;
        ops.push_back(c);
      } else {
        ops.push_back(gemv(Slot::XNorm, Slot::V, w, p + "self_attn.v_proj.weight", kvdim, H));
      }
      ops.push_back(norm(Slot::K, Slot::K, w.fp16(p + "self_attn.k_norm.weight"), nkv, hd));
      ops.push_back(rope(Slot::K, nkv));
      ops.push_back(norm(Slot::V, Slot::V, nullptr, nkv, hd));  // weightless v-norm (ones)
      // Every field matters here, and CUDA's own Gemma builder sets only `cols`; its executor
      // reads the K and V slots by hardcoded name, so the rest is implied. The Metal executor
      // takes them from the op (slot(op.in), slot(op.in2), op.kv_heads, op.head_dim), so an
      // under-specified KvStore writes nothing, the cache stays zero, and attention then
      // correctly returns zero over an empty cache. That is the same mistake as leaving
      // Op::scale unset on the Rope ops: a plan is one description read by two executors, so it
      // has to be complete, not merely sufficient for the one that infers the most.
      Op st;
      st.kind = OpKind::KvStore;
      st.in = Slot::K;
      st.in2 = Slot::V;
      st.kv_heads = nkv;
      st.head_dim = hd;
      st.cols = kvdim;
      ops.push_back(st);
    }
    {
      Op a;
      a.kind = OpKind::Attention;
      a.in = Slot::Q;
      a.out = Slot::Att;
      a.heads = nq;
      a.kv_heads = nkv;
      a.head_dim = hd;
      a.full_attention = full;
      a.sliding_window = g.sliding_window;
      // Gemma 4 specifies a net attention scale of 1.0, and the Metal executor applies
      // op.scale directly; so state 1.0 here rather than pre-scaling Q by sqrt(head_dim)
      // with a ScaleCopy and cancelling it against 1/sqrt(head_dim). The old pair cost a
      // dispatch per layer and an extra fp16 rounding of Q. (CUDA's own Gemma builder keeps
      // the pre-scale because its kernel applies 1/sqrt(head_dim) internally; if it ever
      // delegates here, its executor must honour op.scale first.)
      a.scale = 1.0f;
      ops.push_back(a);
    }
    ops.push_back(gemv(Slot::Att, Slot::Tmp, w, p + "self_attn.o_proj.weight", H, qdim));
    ops.push_back(norm(Slot::Tmp, Slot::Tmp, w.fp16(p + "post_attention_layernorm.weight"), 1, H));
    ops.push_back(add_into(Slot::X, Slot::Tmp, H));

    // --- MLP (GeGLU, double-wide on the shared layers) ---
    ops.push_back(norm(Slot::X, Slot::XNorm, w.fp16(p + "pre_feedforward_layernorm.weight"), 1, H));
    ops.push_back(gemv(Slot::XNorm, Slot::Gate, w, p + "mlp.gate_proj.weight", inter, H));
    ops.push_back(gemv(Slot::XNorm, Slot::Up, w, p + "mlp.up_proj.weight", inter, H));
    // NOTE: no fp16 headroom scaling here, deliberately. The GeGLU product did overflow fp16
    // while attention was returning zero; an unattenuated residual made the pre-FFN norm huge
    // but that was a consequence of the under-specified KvStore, not a property of the model.
    // With attention working, this product peaks around 45 against a 65504 ceiling. A scale op per
    // layer to fix a problem that no longer exists is just cost. The shader keeps a clamp as a
    // backstop, which is a no-op for anything in range.
    {
      Op m;
      m.kind = OpKind::GeluMul;
      m.in = Slot::Gate;
      m.in2 = Slot::Up;
      m.out = Slot::Inter;
      m.cols = inter;
      ops.push_back(m);
    }
    ops.push_back(gemv(Slot::Inter, Slot::Tmp, w, p + "mlp.down_proj.weight", H, inter));
    ops.push_back(
        norm(Slot::Tmp, Slot::Tmp, w.fp16(p + "post_feedforward_layernorm.weight"), 1, H));
    ops.push_back(add_into(Slot::X, Slot::Tmp, H));

    // --- Per-Layer Input injection ---
    if (g.has_ple()) {
      const int ple = g.ple;
      ops.push_back(gemv(Slot::X, Slot::PleGate, w, p + "per_layer_input_gate.weight", ple, H));
      Op gt;
      gt.kind = OpKind::GeluMul;
      gt.in = Slot::PleGate;
      gt.out = Slot::PleGate;
      gt.cols = ple;
      gt.in2 = Slot::PleAll;    // gelu(gate) * per_layer_input[L]...
      gt.aux_offset = L * ple;  // ...which is layer L's window of the packed table
      ops.push_back(gt);
      ops.push_back(gemv(Slot::PleGate, Slot::Tmp, w, p + "per_layer_projection.weight", H, ple));
      ops.push_back(
          norm(Slot::Tmp, Slot::Tmp, w.fp16(p + "post_per_layer_input_norm.weight"), 1, H));
      ops.push_back(add_into(Slot::X, Slot::Tmp, H));
    }

    // --- per-layer output gain (skipped when it is exactly 1.0: a multiply-by-one is a
    // dispatch per layer per token that changes nothing) ---
    const float ls = g.layer_scalar.empty() ? 1.0f : g.layer_scalar[L];
    if (ls != 1.0f) ops.push_back(scale(Slot::X, H, ls));
  }

  // ── epilogue: final norm -> LM head. The logit softcap stays host-side. ──
  {
    std::vector<Op>& epi = plan.epilogue;
    epi.push_back(norm(Slot::X, Slot::XNorm, w.fp16(WP + "norm.weight"), 1, H));
    Op h;
    h.kind = OpKind::LmHead;
    h.in = Slot::XNorm;
    h.cols = g.vocab;
    h.in_dim = H;
    // Tied head, quantized when the backend quantizes: the vocab x hidden read is the
    // single largest slice of a decode token (0.77 GB fp16 on E2B). WeightSource::quant
    // packs a separate copy; the prologue's EmbeddingLookup keeps its own fp16 handle.
    const QuantWeight q = w.quant(WP + "embed_tokens.weight", g.vocab, H);
    if (q.bits != 0) {
      h.qweight = q.packed;
      h.qscales = q.scales;
      h.qbits = q.bits;
      h.qgroup = q.group;
    } else {
      h.weight = w.fp16(WP + "embed_tokens.weight");  // tied
    }
    epi.push_back(h);
  }
  return plan;
}

// Field-for-field the plan PlanCudaEngine::build_vision_plan constructs, with weights resolved
// through WeightSource instead of a device-pointer map. Keep the two in lockstep until CUDA
// delegates here; the cross-backend soft-token gate is what catches drift meanwhile.
//
// Same RMSNorm note as the text plan: plain rmsnorm, no (1 + w) offset. The vision Attention op
// carries no scale of its own; a ScaleCopy pre-multiplies Q by sqrt(head_dim) so the kernel's
// 1/sqrt(head_dim) cancels to the tower's scaling of 1.0. Executors apply 1/sqrt(head_dim),
// not Op::scale, exactly as they do for text.
ModelPlan build_gemma4_vision_plan(const Gemma4VisionGeometry& g, const WeightSource& w) {
  ModelPlan plan;
  const int H = g.hidden;
  const int patch_dim = 3 * g.patch_size * g.patch_size;
  const std::string v = "model.vision_tower.";

  {  // prologue: pixels -> patch embeddings (+ the two-axis learned position table)
    Op o;
    o.kind = OpKind::PatchEmbed;
    o.out = Slot::X;
    o.cols = H;
    o.in_dim = patch_dim;
    o.rows = g.pos_table_size;
    o.weight = w.fp16(v + "patch_embedder.input_proj.weight");
    o.aux_ptr = w.fp16(v + "patch_embedder.position_embedding_table");
    plan.prologue.push_back(o);
  }
  plan.embed_ready = plan.prologue.size();

  plan.layers.assign(static_cast<std::size_t>(g.layers), LayerPlan{});
  for (int L = 0; L < g.layers; ++L) {
    const std::string p = v + "encoder.layers." + std::to_string(L) + ".";
    auto& ops = plan.layers[static_cast<std::size_t>(L)].ops;
    plan.layers[static_cast<std::size_t>(L)].layer_index = L;
    const int nq = g.heads, nkv = g.kv_heads, hd = g.head_dim;
    const int qdim = nq * hd, kvdim = nkv * hd;

    auto rms = [&](Slot in, Slot out, const void* wgt, int rows, int cols) {
      Op o;
      o.kind = OpKind::RmsNorm;
      o.in = in;
      o.out = out;
      o.weight = wgt;
      o.rows = rows;
      o.cols = cols;
      o.eps = g.rms_eps;
      ops.push_back(o);
    };
    auto gemv = [&](Slot in, Slot out, const char* t, int out_dim, int in_dim) {
      Op o;
      o.kind = OpKind::Gemv;
      o.in = in;
      o.out = out;
      o.cols = out_dim;
      o.in_dim = in_dim;
      o.weight = w.fp16(p + t);
      if (g.clipped_linears) {
        // "<proj>.linear.weight" -> "<proj>."; the bounds live beside the projection.
        const std::string tn(t);
        const std::string base = p + tn.substr(0, tn.find("linear.weight"));
        o.clip_in_min = w.scalar(base + "input_min");
        o.clip_in_max = w.scalar(base + "input_max");
        o.clip_out_min = w.scalar(base + "output_min");
        o.clip_out_max = w.scalar(base + "output_max");
      }
      ops.push_back(o);
    };
    auto rope2d = [&](Slot s, int heads) {
      Op o;
      o.kind = OpKind::Rope2D;
      o.in = s;
      o.out = s;
      o.heads = heads;
      o.head_dim = hd;
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

    // The same sandwich-norm shape as a Gemma text layer. The differences are that
    // attention is bidirectional and the positions are 2-D.
    rms(Slot::X, Slot::XNorm, w.fp16(p + "input_layernorm.weight"), 1, H);
    gemv(Slot::XNorm, Slot::Q, "self_attn.q_proj.linear.weight", qdim, H);
    gemv(Slot::XNorm, Slot::K, "self_attn.k_proj.linear.weight", kvdim, H);
    gemv(Slot::XNorm, Slot::V, "self_attn.v_proj.linear.weight", kvdim, H);
    rms(Slot::Q, Slot::Q, w.fp16(p + "self_attn.q_norm.weight"), nq, hd);
    rms(Slot::K, Slot::K, w.fp16(p + "self_attn.k_norm.weight"), nkv, hd);
    rms(Slot::V, Slot::V, nullptr, nkv, hd);  // weightless v-norm, as in the text tower
    rope2d(Slot::Q, nq);
    rope2d(Slot::K, nkv);
    {  // scaling = 1.0: pre-scale q by sqrt(hd) to cancel the kernel's 1/sqrt(hd)
      Op o;
      o.kind = OpKind::ScaleCopy;
      o.in = Slot::Q;
      o.out = Slot::Q;
      o.cols = qdim;
      o.scale = std::sqrt(static_cast<float>(hd));
      ops.push_back(o);
    }
    {
      Op o;
      o.kind = OpKind::Attention;
      o.in = Slot::Q;
      o.out = Slot::Att;
      o.heads = nq;
      o.kv_heads = nkv;
      o.head_dim = hd;
      o.full_attention = true;
      ops.push_back(o);
    }
    gemv(Slot::Att, Slot::Tmp, "self_attn.o_proj.linear.weight", H, qdim);
    rms(Slot::Tmp, Slot::Tmp, w.fp16(p + "post_attention_layernorm.weight"), 1, H);
    add_x(Slot::Tmp);

    rms(Slot::X, Slot::XNorm, w.fp16(p + "pre_feedforward_layernorm.weight"), 1, H);
    gemv(Slot::XNorm, Slot::Gate, "mlp.gate_proj.linear.weight", g.intermediate, H);
    gemv(Slot::XNorm, Slot::Up, "mlp.up_proj.linear.weight", g.intermediate, H);
    {
      Op o;
      o.kind = OpKind::GeluMul;
      o.in = Slot::Gate;
      o.in2 = Slot::Up;
      o.out = Slot::Inter;
      o.cols = g.intermediate;
      ops.push_back(o);
    }
    gemv(Slot::Inter, Slot::Tmp, "mlp.down_proj.linear.weight", H, g.intermediate);
    rms(Slot::Tmp, Slot::Tmp, w.fp16(p + "post_feedforward_layernorm.weight"), 1, H);
    add_x(Slot::Tmp);
  }

  // Epilogue: standardize, when the checkpoint has it (E2B does not). Pooling and projection
  // are not ops; pooling changes the token count, so the executor has to be re-entered with
  // the new count; see encode_image.
  if (g.standardize) {
    Op o;
    o.kind = OpKind::Standardize;
    o.in = Slot::X;
    o.out = Slot::X;
    o.cols = H;
    o.weight = w.fp16(v + "std_bias");
    o.aux_ptr = w.fp16(v + "std_scale");
    plan.epilogue.push_back(o);
  }
  return plan;
}

}  // namespace opplan
}  // namespace engine
