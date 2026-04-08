#!/usr/bin/env python3
"""
Extract decoder tensors from HuggingFace safetensors into .bin files for LL2CUDA packing.

Supports LLaMA-2, LLaMA-3, Mistral, Mixtral, Phi-3/PhiMoE, and Qwen2 model families.
Model family is auto-detected from config.json (model_type field).
Architecture-specific metadata (rope_theta, sliding_window, QKV biases, etc.)
is extracted and saved to model_config.json for use by pack_ll2c.py.
"""

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

FAMILY_LLAMA2   = "llama2"
FAMILY_LLAMA3   = "llama3"
FAMILY_MISTRAL  = "mistral"
FAMILY_MIXTRAL  = "mixtral"
FAMILY_PHI3     = "phi3"
FAMILY_QWEN2    = "qwen2"
FAMILY_UNKNOWN  = "unknown"

# Canonical internal family ID (must match ModelFamily enum in llama_config.hpp)
FAMILY_ID = {
    FAMILY_UNKNOWN: 0,
    FAMILY_LLAMA2:  1,
    FAMILY_LLAMA3:  2,
    FAMILY_MISTRAL: 3,
    FAMILY_MIXTRAL: 6,
    FAMILY_PHI3:    4,
    FAMILY_QWEN2:   5,
}

# Default RoPE theta per family (matches default_rope_theta() in llama_config.hpp)
DEFAULT_ROPE_THETA = {
    FAMILY_LLAMA2:  10000.0,
    FAMILY_LLAMA3:  500000.0,
    FAMILY_MISTRAL: 10000.0,
    FAMILY_MIXTRAL: 10000.0,
    FAMILY_PHI3:    10000.0,
    FAMILY_QWEN2:   1000000.0,
}


def detect_family(cfg: dict) -> str:
    """Detect model family from HuggingFace config.json."""
    model_type = cfg.get("model_type", "").lower()

    if model_type in ("llama", "llama2"):
        # LLaMA-3 uses rope_scaling or has large vocab; LLaMA-2 has vocab=32000
        vocab = int(cfg.get("vocab_size", 32000))
        rope_theta = float(cfg.get("rope_theta", 10000.0))
        if vocab > 100000 or rope_theta > 100000:
            return FAMILY_LLAMA3
        return FAMILY_LLAMA2

    if model_type == "mixtral":
        return FAMILY_MIXTRAL

    if model_type == "mistral":
        # Sparse-MoE Mistral variants expose expert fields in config.
        if int(cfg.get("num_local_experts", 0) or 0) > 0:
            return FAMILY_MIXTRAL
        return FAMILY_MISTRAL

    if model_type in ("phi", "phi3", "phi-3", "phi-msft", "phimoe"):
        return FAMILY_PHI3

    if model_type == "qwen2":
        return FAMILY_QWEN2

    # Fallback heuristics
    if "qwen" in model_type:
        return FAMILY_QWEN2
    if "phi" in model_type:
        return FAMILY_PHI3
    if "mistral" in model_type:
        return FAMILY_MISTRAL
    if "mixtral" in model_type:
        return FAMILY_MIXTRAL

    print(f"[warn] Unknown model_type '{model_type}', defaulting to llama2")
    return FAMILY_LLAMA2


# ---------------------------------------------------------------------------
# Safetensors helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(hf_dir: Path) -> dict:
    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        return read_json(index_path)

    single_path = hf_dir / "model.safetensors"
    if not single_path.exists():
        raise FileNotFoundError(f"Missing {index_path} and {single_path}")

    with single_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

    weight_map = {k: "model.safetensors" for k in header if k != "__metadata__"}
    return {"weight_map": weight_map}


def load_hf_config(hf_dir: Path) -> dict:
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    return read_json(cfg_path)


def read_safetensor_blob(safetensor_path: Path, tensor_name: str):
    with safetensor_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

        if tensor_name not in header:
            raise KeyError(f"Tensor {tensor_name} not found in {safetensor_path}")

        meta = header[tensor_name]
        dtype = meta.get("dtype")
        start, end = meta["data_offsets"]
        f.seek(8 + header_len + start)
        raw = f.read(end - start)

        if dtype == "F16":
            return raw, meta.get("shape", [])
        if dtype == "BF16":
            if np is None:
                raise ValueError(f"Tensor {tensor_name} dtype BF16 requires numpy for conversion")
            bf16 = np.frombuffer(raw, dtype=np.uint16)
            f32_bits = bf16.astype(np.uint32) << np.uint32(16)
            f32 = f32_bits.view(np.float32)
            f16 = f32.astype(np.float16)
            return f16.tobytes(), meta.get("shape", [])
        if dtype == "F32":
            if np is None:
                raise ValueError(f"Tensor {tensor_name} dtype F32 requires numpy for conversion")
            f32 = np.frombuffer(raw, dtype=np.float32)
            f16 = f32.astype(np.float16)
            return f16.tobytes(), meta.get("shape", [])

        raise ValueError(f"Tensor {tensor_name} dtype {dtype}, expected F16/BF16/F32")


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def extract_model_config(hf_cfg: dict, family: str) -> dict:
    """Build the model_config.json that pack_ll2c.py consumes."""
    num_heads = int(hf_cfg["num_attention_heads"])
    num_kv_heads = int(hf_cfg.get("num_key_value_heads", num_heads))

    rope_theta = float(hf_cfg.get("rope_theta", DEFAULT_ROPE_THETA.get(family, 10000.0)))

    # Sliding-window attention (set when provided by checkpoint config).
    sliding_window = int(hf_cfg.get("sliding_window", 0) or 0)

    # Sparse-MoE metadata.
    num_local_experts = int(hf_cfg.get("num_local_experts", 0) or 0)
    num_experts_per_tok = int(hf_cfg.get("num_experts_per_tok", 0) or 0)
    expert_intermediate_size = int(hf_cfg.get("expert_intermediate_size", 0) or 0)
    if family == FAMILY_MIXTRAL and num_local_experts <= 0:
        num_local_experts = 8
    if num_local_experts > 0:
        if num_experts_per_tok <= 0:
            num_experts_per_tok = 2
        if expert_intermediate_size <= 0:
            # Most HF MoE checkpoints reuse intermediate_size as per-expert hidden dim.
            expert_intermediate_size = int(hf_cfg.get("intermediate_size", 0) or 0)

    # Tied word embeddings (some Phi-2 style models)
    tie_word_embeddings = bool(hf_cfg.get("tie_word_embeddings", False))

    # QKV biases (Qwen2 uses them)
    has_qkv_bias = (family == FAMILY_QWEN2) or bool(hf_cfg.get("attention_bias", False))

    # LayerNorm vs RMSNorm selection.
    # Prefer explicit normalization kind when present; otherwise infer from eps keys.
    norm_kind = str(hf_cfg.get("norm_type", hf_cfg.get("normalization", ""))).lower()
    has_rms_eps = "rms_norm_eps" in hf_cfg
    has_layer_eps = "layer_norm_eps" in hf_cfg
    use_layernorm = False
    if "layernorm" in norm_kind:
        use_layernorm = True
    elif "rmsnorm" in norm_kind:
        use_layernorm = False
    elif has_layer_eps and not has_rms_eps:
        use_layernorm = True
    norm_eps = float(hf_cfg.get("rms_norm_eps", hf_cfg.get("layer_norm_eps", 1e-5)))

    # Partial rotary factor (Phi-3; stored in config but not yet kernel-enforced)
    partial_rotary_factor = float(hf_cfg.get("partial_rotary_factor", 1.0))
    if partial_rotary_factor != 1.0:
        print(f"[info] partial_rotary_factor={partial_rotary_factor} detected (Phi-3). "
              "Full rotary will be used in the engine until partial RoPE is kernel-supported.")

    return {
        "model_family":        family,
        "model_family_id":     FAMILY_ID.get(family, 0),
        "vocab_size":          int(hf_cfg["vocab_size"]),
        "hidden_size":         int(hf_cfg["hidden_size"]),
        "intermediate_size":   int(hf_cfg["intermediate_size"]),
        "num_layers":          int(hf_cfg["num_hidden_layers"]),
        "num_heads":           num_heads,
        "num_kv_heads":        num_kv_heads,
        "max_seq_len":         int(hf_cfg.get("max_position_embeddings", 4096)),
        "rope_theta":          rope_theta,
        "norm_eps":            norm_eps,
        "sliding_window":      sliding_window,
        "tie_word_embeddings": tie_word_embeddings,
        "has_qkv_bias":        has_qkv_bias,
        "use_layernorm":       use_layernorm,
        "partial_rotary_factor": partial_rotary_factor,
        "num_local_experts":   num_local_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "expert_intermediate_size": expert_intermediate_size,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------

def build_mapping(family: str, num_layers: int, has_qkv_bias: bool, num_local_experts: int):
    """
    Map HuggingFace tensor names to canonical internal names used by LL2CUDA.

    Dense families share the standard decoder naming.
    MoE checkpoints use block_sparse_moe router + per-expert FFN weights.
    """
    items = [
        ("model.embed_tokens.weight", "tok_embeddings.weight", True),
        ("model.norm.weight",         "norm.weight",           True),
        ("model.norm.bias",           "norm.bias",             False),
        # lm_head.weight is optional when tie_word_embeddings=True
        ("lm_head.weight",            "output.weight",         False),
        ("lm_head.bias",              "output.bias",           False),
    ]

    for i in range(num_layers):
        layer = [
            (f"model.layers.{i}.input_layernorm.weight",
             f"layers.{i}.attention_norm.weight", True),
            (f"model.layers.{i}.input_layernorm.bias",
             f"layers.{i}.attention_norm.bias",   False),
            (f"model.layers.{i}.self_attn.q_proj.weight",
             f"layers.{i}.attention.wq",           True),
            (f"model.layers.{i}.self_attn.k_proj.weight",
             f"layers.{i}.attention.wk",           True),
            (f"model.layers.{i}.self_attn.v_proj.weight",
             f"layers.{i}.attention.wv",           True),
            (f"model.layers.{i}.self_attn.o_proj.weight",
             f"layers.{i}.attention.wo",           True),
            (f"model.layers.{i}.self_attn.o_proj.bias",
             f"layers.{i}.attention.bo",           False),
            (f"model.layers.{i}.post_attention_layernorm.weight",
             f"layers.{i}.ffn_norm.weight",        True),
            (f"model.layers.{i}.post_attention_layernorm.bias",
             f"layers.{i}.ffn_norm.bias",          False),
        ]

        if num_local_experts > 0:
            layer.append(
                (f"model.layers.{i}.block_sparse_moe.gate.weight",
                 f"layers.{i}.feed_forward.router", True)
            )
            for e in range(num_local_experts):
                layer.extend([
                    (f"model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight",
                     f"layers.{i}.feed_forward.experts.{e}.w1", True),
                    (f"model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight",
                     f"layers.{i}.feed_forward.experts.{e}.w2", True),
                    (f"model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight",
                     f"layers.{i}.feed_forward.experts.{e}.w3", True),
                ])
        else:
            layer.extend([
                (f"model.layers.{i}.mlp.gate_proj.weight",
                 f"layers.{i}.feed_forward.w1",        True),
                (f"model.layers.{i}.mlp.down_proj.weight",
                 f"layers.{i}.feed_forward.w2",        True),
                (f"model.layers.{i}.mlp.up_proj.weight",
                 f"layers.{i}.feed_forward.w3",        True),
            ])
        if has_qkv_bias:
            # QKV biases are stored as separate tensors per projection.
            # They will be fused into bqkv (q||k||v) during extraction.
            layer += [
                (f"model.layers.{i}.self_attn.q_proj.bias",
                 f"layers.{i}.attention.bq",  False),
                (f"model.layers.{i}.self_attn.k_proj.bias",
                 f"layers.{i}.attention.bk",  False),
                (f"model.layers.{i}.self_attn.v_proj.bias",
                 f"layers.{i}.attention.bv",  False),
            ]
        items.extend(layer)

    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir",  required=True, help="Path to HuggingFace model directory")
    ap.add_argument("--out-dir", required=True, help="Output directory for .bin files")
    ap.add_argument("--family",  default=None,
                    help="Override auto-detected family: llama2|llama3|mistral|mixtral|phi3|phimoe|qwen2")
    args = ap.parse_args()

    hf_dir  = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_cfg  = load_hf_config(hf_dir)
    family  = args.family if args.family else detect_family(hf_cfg)
    if family == "phimoe":
        # Keep runtime behavior aligned with Phi templates/defaults while using
        # MoE tensor mapping selected via num_local_experts > 0.
        family = FAMILY_PHI3
    model_cfg = extract_model_config(hf_cfg, family)
    num_layers  = model_cfg["num_layers"]
    has_qkv_bias = model_cfg["has_qkv_bias"]
    num_local_experts = int(model_cfg.get("num_local_experts", 0) or 0)

    print(f"[info] family={family} layers={num_layers} hidden={model_cfg['hidden_size']}"
          f" vocab={model_cfg['vocab_size']} rope_theta={model_cfg['rope_theta']}"
          f" qkv_bias={has_qkv_bias}"
          f" experts={num_local_experts}")

    index      = load_index(hf_dir)
    weight_map = index.get("weight_map", {})
    mapping    = build_mapping(family, num_layers, has_qkv_bias, num_local_experts)
    mapped_sources = {src for src, _dst, _required in mapping}

    # Surface known-but-currently-ignored tensors to make conversion behavior explicit.
    unsupported_biases = []
    warned: set[str] = set()
    for src_name in sorted(weight_map.keys()):
        if src_name in mapped_sources:
            continue
        for suffix in unsupported_biases:
            if src_name.endswith(suffix) and suffix not in warned:
                print(f"[warn] Ignoring unsupported tensor family '{suffix}' (example: {src_name})")
                warned.add(suffix)
                break

    # Collect which source tensors live in which shard, tracking dst names
    by_shard: dict[str, list] = defaultdict(list)
    for src_name, dst_name, required in mapping:
        shard_rel = weight_map.get(src_name)
        if shard_rel is None:
            if required:
                raise KeyError(f"Missing required tensor: {src_name}")
            continue
        by_shard[shard_rel].append((src_name, dst_name))

    # Read each shard once and write all its tensors
    extracted: dict[str, bytes] = {}  # dst_name -> blob bytes
    for shard_rel, tensors in sorted(by_shard.items()):
        shard_path = hf_dir / shard_rel
        for src_name, dst_name in tensors:
            blob, shape = read_safetensor_blob(shard_path, src_name)
            extracted[dst_name] = blob
            print(f"  {src_name} -> {dst_name}  shape={shape}")

    # Fuse Q/K/V biases into a single bqkv tensor per layer
    if has_qkv_bias:
        for i in range(num_layers):
            bq_name = f"layers.{i}.attention.bq"
            bk_name = f"layers.{i}.attention.bk"
            bv_name = f"layers.{i}.attention.bv"
            if bq_name in extracted and bk_name in extracted and bv_name in extracted:
                fused = extracted.pop(bq_name) + extracted.pop(bk_name) + extracted.pop(bv_name)
                extracted[f"layers.{i}.attention.bqkv"] = fused
                print(f"  fused bq+bk+bv -> layers.{i}.attention.bqkv")
            else:
                missing = [n for n in (bq_name, bk_name, bv_name) if n not in extracted]
                if missing:
                    print(f"[warn] Missing QKV bias tensors for layer {i}: {missing}")

    # Handle tied embeddings: copy tok_embeddings.weight to output.weight if absent
    if model_cfg["tie_word_embeddings"] and "output.weight" not in extracted:
        if "tok_embeddings.weight" in extracted:
            extracted["output.weight"] = extracted["tok_embeddings.weight"]
            print("[info] tie_word_embeddings=True: copied tok_embeddings.weight -> output.weight")

    # Write .bin files
    for dst_name, blob in extracted.items():
        out_path = out_dir / f"{dst_name}.bin"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)

    # Write extended model_config.json
    (out_dir / "model_config.json").write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
    print(f"\n[done] Wrote {len(extracted)} tensors to {out_dir}")
    print(f"[done] Wrote model_config.json: family={family}")


if __name__ == "__main__":
    main()
