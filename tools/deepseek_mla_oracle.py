#!/usr/bin/env python
"""Reference oracle for DeepSeek-V2-Lite MLA: run one real layer's Multi-head Latent Attention through
transformers' built-in DeepseekV2Attention and dump {input, weights, rope, output} for the CPI test to
match (deepseek_mla_real_test).

We dump the rotary's inv_freq + attention_scaling (the YARN-derived frequencies + mscale) rather than
re-deriving YARN in C++, so the op consumes the exact frequencies; computing inv_freq from config is a
separate, simpler concern verified elsewhere. RoPE here is INTERLEAVED-complex: pairs (2p, 2p+1) rotate
by pos*inv_freq[p], scaled by attention_scaling.

Writes <out>.bin (flat little-endian fp32) + <out>.json (name -> [offset, shape]) + <out>.meta.json.
"""
import argparse
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


def dump_bundle(path, tensors):
    manifest, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        manifest[name] = [off, list(a.shape)]
        blob += a.tobytes()
        off += a.size
    open(path + ".bin", "wb").write(blob)
    json.dump(manifest, open(path + ".json", "w"), indent=2)
    print(f"wrote {path}.bin ({off * 4 / 1e6:.1f} MB, {len(tensors)} tensors)")


def load_layer_attn_weights(attn, layer):
    from safetensors import safe_open

    index = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))["weight_map"]
    prefix = f"model.layers.{layer}.self_attn."
    by_shard = {}
    for k in index:
        if k.startswith(prefix):
            by_shard.setdefault(index[k], []).append(k)
    state = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            for k in keys:
                state[k[len(prefix):]] = f.get_tensor(k).float()
    miss, unexp = attn.load_state_dict(state, strict=False)
    hard = [m for m in miss if m.endswith((".weight", ".bias"))]
    if hard:
        print("MISSING WEIGHTS:", hard, file=sys.stderr)
        sys.exit(2)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=6)
    ap.add_argument("--out", default=os.path.join("artifacts", "deepseek_mla_ref"))
    args = ap.parse_args()

    from transformers import AutoConfig
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2Attention,
        DeepseekV2RotaryEmbedding,
    )

    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    coerce_rope_floats(cfg)
    torch.manual_seed(0)
    attn = DeepseekV2Attention(cfg, layer_idx=args.layer).eval().float()
    if not args.smoke:
        load_layer_attn_weights(attn, args.layer)

    H, T = cfg.hidden_size, args.seq
    hidden = (torch.randn(1, T, H) * 0.2).float()
    pos = torch.arange(T).unsqueeze(0)
    rotary = DeepseekV2RotaryEmbedding(cfg)
    freqs_cis = rotary(hidden, pos)  # complex [1, T, qk_rope/2]; magnitude == attention_scaling
    mask = torch.full((T, T), float("-inf")).triu(1).view(1, 1, T, T)
    with torch.no_grad():
        attn_out, _ = attn(hidden_states=hidden, position_embeddings=freqs_cis, attention_mask=mask)

    scaling = float(attn.attention_scaling) if hasattr(attn, "attention_scaling") else 1.0
    print("attn_out", tuple(attn_out.shape), "mean", float(attn_out.mean()))
    print("rotary.attention_scaling (mscale) =", float(rotary.attention_scaling))
    print("softmax scaling =", attn.scaling)
    if args.smoke:
        print("SMOKE OK")
        return 0

    bundle = {
        "hidden": hidden[0].numpy(),
        "attn_out": attn_out[0].numpy(),
        "q_proj": attn.q_proj.weight.detach().numpy(),
        "kv_a_proj_with_mqa": attn.kv_a_proj_with_mqa.weight.detach().numpy(),
        "kv_a_layernorm": attn.kv_a_layernorm.weight.detach().numpy(),
        "kv_b_proj": attn.kv_b_proj.weight.detach().numpy(),
        "o_proj": attn.o_proj.weight.detach().numpy(),
        "inv_freq": rotary.inv_freq.detach().numpy(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump_bundle(args.out, bundle)
    meta = {
        "hidden_size": cfg.hidden_size,
        "num_heads": cfg.num_attention_heads,
        "qk_nope_head_dim": cfg.qk_nope_head_dim,
        "qk_rope_head_dim": cfg.qk_rope_head_dim,
        "v_head_dim": cfg.v_head_dim,
        "kv_lora_rank": cfg.kv_lora_rank,
        "seq": T,
        "softmax_scale": float(attn.scaling),
        "attention_scaling": float(rotary.attention_scaling),
        "rms_norm_eps": cfg.rms_norm_eps,
    }
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    # Plain whitespace scalars for the C++ test to read without a JSON parser. Tensors follow in .bin
    # in this fixed order: hidden, attn_out, q_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj,
    # o_proj, inv_freq -- the C++ side derives offsets from the dims below.
    with open(args.out + ".dims", "w") as f:
        f.write(f"{cfg.hidden_size} {cfg.num_attention_heads} {cfg.qk_nope_head_dim} "
                f"{cfg.qk_rope_head_dim} {cfg.v_head_dim} {cfg.kv_lora_rank} {T} "
                f"{float(attn.scaling):.9g} {float(rotary.attention_scaling):.9g} "
                f"{float(cfg.rms_norm_eps):.9g}\n")
    print("meta:", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
