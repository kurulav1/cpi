#!/usr/bin/env python
"""Streaming HF golden trace for DeepSeek-V2-Lite: run the real model one layer AT A TIME (load that
layer's weights from the safetensors shards, forward, free) so a 16B model traces in <2GB RAM, and dump
the per-layer hidden states + final logits. This is the reference CPI's op-plan forward is verified
against, layer by layer (the "compare vs a real engine at every stage" plan).

Outputs artifacts/deepseek_trace.npz with: tokens, embed, layer_00..layer_26 (post-layer hidden),
final_norm, logits (last position). Prints the greedy top-5 next tokens so the run is sanity-checkable.
"""
import json
import os
import sys

import numpy as np
import torch

MODEL_DIR = os.path.join("artifacts", "hub", "deepseek-ai__DeepSeek-V2-Lite", "hf")


def coerce_rope_floats(cfg):
    for field in ("rope_scaling", "rope_parameters"):
        d = getattr(cfg, field, None)
        if isinstance(d, dict):
            d = dict(d)
            for k in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "rope_theta"):
                if k in d and d[k] is not None:
                    d[k] = float(d[k])
            setattr(cfg, field, d)


def shard_index():
    return json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]


def load_by_prefix(index, prefix, module):
    """Load tensors whose name starts with `prefix` into `module` (strip prefix). float32.

    crucial: transformers' DeepseekV2Moe stores experts fused (experts.gate_up_proj [E,2*MI,H],
    experts.down_proj [E,H,MI]) while the checkpoint ships per-expert tensors. from_pretrained fuses
    them; a manual load_state_dict does not; so we fuse here, or the routed experts stay at random
    init and the whole trace is silently wrong (learned this the hard way)."""
    import torch
    from safetensors import safe_open

    by_shard = {}
    for k in index:
        if k.startswith(prefix):
            by_shard.setdefault(index[k], []).append(k)
    raw = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            for k in keys:
                raw[k[len(prefix):]] = f.get_tensor(k).float()

    # Fuse per-expert gate/up/down into the module's expected layout.
    experts = {}
    for name in list(raw):
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "mlp" and parts[1] == "experts" and parts[2].isdigit():
            experts.setdefault(int(parts[2]), {})[parts[3]] = raw.pop(name)
    if experts:
        E = max(experts) + 1
        raw["mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], 0) for e in range(E)])
        raw["mlp.experts.down_proj"] = torch.stack([experts[e]["down_proj"] for e in range(E)])

    missing, unexpected = module.load_state_dict(raw, strict=False)
    hard = [m for m in missing if m.endswith((".weight", ".bias"))]
    if hard:
        print(f"  MISSING under {prefix}: {hard[:6]}", file=sys.stderr)
    return raw


def one_tensor(index, name):
    from safetensors import safe_open

    with safe_open(os.path.join(MODEL_DIR, index[name]), framework="pt") as f:
        return f.get_tensor(name).float()


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2DecoderLayer,
        DeepseekV2RMSNorm,
        DeepseekV2RotaryEmbedding,
    )

    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    coerce_rope_floats(cfg)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(prompt, return_tensors="pt").input_ids
    T = ids.shape[1]
    print(f"prompt={prompt!r}  tokens={ids.tolist()[0]}  T={T}")

    index = shard_index()
    out = {"tokens": ids.numpy()}

    # Embedding.
    embed_w = one_tensor(index, "model.embed_tokens.weight")
    hidden = torch.nn.functional.embedding(ids, embed_w)  # [1,T,H]
    out["embed"] = hidden[0].detach().numpy()

    # RoPE (freqs_cis) + causal mask, shared by every layer.
    rotary = DeepseekV2RotaryEmbedding(cfg)
    pos = torch.arange(T).unsqueeze(0)
    freqs_cis = rotary(hidden, pos)
    mask = torch.full((T, T), float("-inf")).triu(1).view(1, 1, T, T)

    with torch.no_grad():
        for L in range(cfg.num_hidden_layers):
            layer = DeepseekV2DecoderLayer(cfg, L).eval().float()
            load_by_prefix(index, f"model.layers.{L}.", layer)
            res = layer(hidden_states=hidden, attention_mask=mask, position_embeddings=freqs_cis)
            hidden = res[0] if isinstance(res, (tuple, list)) else res
            out[f"layer_{L:02d}"] = hidden[0].detach().numpy()
            del layer
            if L % 5 == 0:
                print(f"  layer {L:2d} done, hidden mean {float(hidden.mean()):+.4f}")

        # Final norm + lm_head.
        norm = DeepseekV2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).eval()
        norm.weight.data = one_tensor(index, "model.norm.weight")
        hn = norm(hidden)
        out["final_norm"] = hn[0].detach().numpy()
        lm_head = one_tensor(index, "lm_head.weight")  # [vocab, H]
        logits = torch.nn.functional.linear(hn[:, -1, :].float(), lm_head)  # [1, vocab]
        out["logits"] = logits[0].detach().numpy()

    top = torch.topk(logits[0], 5)
    print("greedy top-5 next tokens:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {i:6d}  {tok.decode([i])!r:16}  logit {p:+.3f}")

    os.makedirs("artifacts", exist_ok=True)
    np.savez(os.path.join("artifacts", "deepseek_trace.npz"), **out)
    print("wrote artifacts/deepseek_trace.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
