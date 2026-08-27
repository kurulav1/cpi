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
import sys
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
FAMILY_QWEN3    = "qwen3"
FAMILY_QWEN3_5  = "qwen3_5"
FAMILY_GEMMA    = "gemma"
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
    FAMILY_QWEN3_5: 7,
    FAMILY_QWEN3:   8,
    FAMILY_GEMMA:   9,
}

# Default RoPE theta per family (matches default_rope_theta() in llama_config.hpp)
DEFAULT_ROPE_THETA = {
    FAMILY_LLAMA2:  10000.0,
    FAMILY_LLAMA3:  500000.0,
    FAMILY_MISTRAL: 10000.0,
    FAMILY_MIXTRAL: 10000.0,
    FAMILY_PHI3:    10000.0,
    FAMILY_QWEN2:   1000000.0,
    FAMILY_QWEN3:   1000000.0,
    FAMILY_QWEN3_5: 1000000.0,
    FAMILY_GEMMA:   10000.0,
}


def detect_family(cfg: dict) -> str:
    """Detect model family from HuggingFace config.json."""
    text_cfg = cfg.get("text_config", cfg)
    model_type = text_cfg.get("model_type", cfg.get("model_type", "")).lower()

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
    # The text half of a Qwen3.5 checkpoint reports "qwen3_5_text"; the multimodal wrapper
    # reports "qwen3_5". Match both, and match them before the generic "qwen" fallback below,
    # which would otherwise classify this family as qwen2 and map the wrong tensor names.
    if model_type.startswith("qwen3_5"):
        return FAMILY_QWEN3_5
    # Qwen3 dense (model_type "qwen3"): standard decoder + per-head QK-norm, no
    # QKV bias. Distinct from the mixed linear/full-attention "qwen3_5".
    if model_type == "qwen3":
        return FAMILY_QWEN3

    # Gemma 1 (model_type "gemma"): GeGLU MLP, embedding scale, (1+w) RMSNorm.
    # gemma2/gemma3 add attention/logit softcap and are rejected below.
    if model_type == "gemma":
        return FAMILY_GEMMA

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


def unsupported_reason(cfg: dict) -> str:
    text_cfg = cfg.get("text_config", cfg)
    model_type = str(text_cfg.get("model_type", cfg.get("model_type", ""))).lower()
    layer_types = [str(value).lower() for value in text_cfg.get("layer_types", [])]
    has_linear_attention = (
        any("linear_attention" in value for value in layer_types) or
        int(text_cfg.get("linear_num_key_heads", 0) or 0) > 0 or
        int(text_cfg.get("linear_num_value_heads", 0) or 0) > 0
    )

    if "qwen3_5" in model_type:
        # Mixed linear/full attention is now expressible: the container records a per-layer
        # attention kind (v5) and the delta-net geometry (v6), and every op the block needs has a
        # kernel. What this conversion does not carry is the vision tower and the MTP head; both
        # are present in these checkpoints and both are skipped, so the result is the text model.
        if not has_linear_attention:
            return "Qwen3.5 is not supported by the native CPI engine yet."
        return ""

    # Linear attention is supported for qwen3_5 (handled above, which returns "" for it). Any
    # other family arriving here with linear-attention layers has a block shape this converter has
    # not been taught, so it is still refused rather than mapped by guesswork.
    if has_linear_attention:
        return ("This model uses linear-attention layers, which the native CPI "
                "engine does not support yet for this family.")

    if model_type in ("gemma2", "gemma3") or "gemma2" in model_type or "gemma3" in model_type:
        return ("Gemma 2/3 is not supported yet: it needs attention- and "
                "logit-softcapping (and Gemma 2's alternating attention).")

    return ""


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
        # Some repos ship a single shard under the sharded name (Qwen3.5-0.8B is
        # "model.safetensors-00001-of-00001.safetensors") and then omit the index, since there is
        # nothing to index. Accept exactly one such file; refuse several, because picking one of
        # a real multi-shard set would silently convert a fraction of the model.
        shards = sorted(hf_dir.glob("*.safetensors")) + sorted(
            hf_dir.glob("*.safetensors-*-of-*.safetensors"))
        shards = sorted({p for p in shards})
        if len(shards) == 1:
            single_path = shards[0]
        elif len(shards) > 1:
            raise FileNotFoundError(
                f"{index_path} is missing but {len(shards)} shards are present; an index is "
                f"required to know which tensor lives where")
        else:
            raise FileNotFoundError(f"Missing {index_path} and {single_path}")

    with single_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

    # Point at the file actually opened, not the canonical name; they differ whenever a repo
    # ships one shard under a sharded filename.
    weight_map = {k: single_path.name for k in header if k != "__metadata__"}
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

# NOTE on the lookups below: Python evaluates a .get() default eagerly, so
# `text_cfg.get(k, hf_cfg[k])` raises KeyError whenever k is absent from the top level; even
# when text_config supplies it. That is every nested-config model (Qwen3.5 keeps everything under
# text_config), and it read as "this model has no attention heads" rather than as a lookup bug.
# hf_cfg.get() keeps the fallback lazy.
def extract_model_config(hf_cfg: dict, family: str) -> dict:
    """Build the model_config.json that pack_ll2c.py consumes."""
    text_cfg = hf_cfg.get("text_config", hf_cfg)
    num_heads = int(text_cfg.get("num_attention_heads", hf_cfg.get("num_attention_heads")))
    num_kv_heads = int(text_cfg.get("num_key_value_heads", hf_cfg.get("num_key_value_heads", num_heads)))

    # Newer HF configs (Qwen3.5 and later) nest every rope knob under "rope_parameters" instead of
    # leaving them at the top level. Reading only the flat key silently yields the family default
    # theta and a full-rotary factor; a model that loads, runs, and is wrong.
    def rope_param(key, default):
        for scope in (text_cfg, hf_cfg):
            nested = scope.get("rope_parameters") or scope.get("rope_scaling") or {}
            if key in scope:
                return scope[key]
            if key in nested:
                return nested[key]
        return default

    rope_theta = float(rope_param("rope_theta", DEFAULT_ROPE_THETA.get(family, 10000.0)))

    # Vision geometry, when the checkpoint is multimodal. Empty for text-only models, which
    # leaves every field zero in the container; the engine reads depth == 0 as "no tower"
    # rather than needing a separate flag.
    vc = hf_cfg.get("vision_config") or {}
    vision_cfg = {}
    if vc:
        vision_cfg = {
            "vision_depth": int(vc.get("depth", 0) or 0),
            "vision_hidden_size": int(vc.get("hidden_size", 0) or 0),
            "vision_num_heads": int(vc.get("num_heads", 0) or 0),
            "vision_intermediate_size": int(vc.get("intermediate_size", 0) or 0),
            "vision_patch_size": int(vc.get("patch_size", 0) or 0),
            "vision_temporal_patch_size": int(vc.get("temporal_patch_size", 0) or 0),
            "vision_in_channels": int(vc.get("in_channels", 0) or 0),
            "vision_spatial_merge_size": int(vc.get("spatial_merge_size", 0) or 0),
            "vision_num_position_embeddings": int(vc.get("num_position_embeddings", 0) or 0),
            "vision_out_hidden_size": int(vc.get("out_hidden_size", 0) or 0),
        }

    # Sliding-window attention (set when provided by checkpoint config).
    # Honor use_sliding_window: models like Qwen2.5 advertise a large window but
    # keep it disabled, in which case attention is full and storing a window
    # would push the engine onto the slow non-full (sequential prefill) path.
    sliding_window = int(text_cfg.get("sliding_window", hf_cfg.get("sliding_window", 0)) or 0)
    use_sliding_window = text_cfg.get("use_sliding_window", hf_cfg.get("use_sliding_window", None))
    if use_sliding_window is False:
        sliding_window = 0

    # Sparse-MoE metadata.
    num_local_experts = int(text_cfg.get("num_local_experts", hf_cfg.get("num_local_experts", 0)) or 0)
    num_experts_per_tok = int(text_cfg.get("num_experts_per_tok", hf_cfg.get("num_experts_per_tok", 0)) or 0)
    expert_intermediate_size = int(text_cfg.get("expert_intermediate_size", hf_cfg.get("expert_intermediate_size", 0)) or 0)
    if family == FAMILY_MIXTRAL and num_local_experts <= 0:
        num_local_experts = 8
    if num_local_experts > 0:
        if num_experts_per_tok <= 0:
            num_experts_per_tok = 2
        if expert_intermediate_size <= 0:
            # Most HF MoE checkpoints reuse intermediate_size as per-expert hidden dim.
            expert_intermediate_size = int(text_cfg.get("intermediate_size", hf_cfg.get("intermediate_size", 0)) or 0)

    # Tied word embeddings (some Phi-2 style models). Gemma always ties (no
    # lm_head.weight in the checkpoint), and some configs omit the flag, so force it.
    tie_word_embeddings = bool(text_cfg.get("tie_word_embeddings", hf_cfg.get("tie_word_embeddings", False))) \
        or (family == FAMILY_GEMMA)

    # QKV biases (Qwen2 uses them; Qwen3 dropped them)
    has_qkv_bias = (family == FAMILY_QWEN2) or bool(text_cfg.get("attention_bias", hf_cfg.get("attention_bias", False)))

    # Per-head QK-norm (RMSNorm on Q and K after projection): Qwen3 dense.
    # Qwen3.5's full-attention layers carry q_norm/k_norm too. Missing it here does not fail the
    # conversion; it drops both tensors from the mapping, so the container is quietly short two
    # weights per attention layer and the engine skips a normalisation it needs.
    has_qk_norm = family in (FAMILY_QWEN3, FAMILY_QWEN3_5)

    # Gemma: GeGLU MLP (tanh GELU) + token-embedding scale by sqrt(hidden). Its
    # (1+w) RMSNorm is folded into the norm weights at extraction (below), so no
    # runtime flag is needed for that.
    mlp_gelu = (family == FAMILY_GEMMA)
    scale_embeddings = (family == FAMILY_GEMMA)

    # LayerNorm vs RMSNorm selection.
    # Prefer explicit normalization kind when present; otherwise infer from eps keys.
    norm_kind = str(text_cfg.get("norm_type", text_cfg.get("normalization", hf_cfg.get("norm_type", hf_cfg.get("normalization", ""))))).lower()
    has_rms_eps = "rms_norm_eps" in text_cfg or "rms_norm_eps" in hf_cfg
    has_layer_eps = "layer_norm_eps" in text_cfg or "layer_norm_eps" in hf_cfg
    use_layernorm = False
    if "layernorm" in norm_kind:
        use_layernorm = True
    elif "rmsnorm" in norm_kind:
        use_layernorm = False
    elif has_layer_eps and not has_rms_eps:
        use_layernorm = True
    norm_eps = float(text_cfg.get("rms_norm_eps", text_cfg.get("layer_norm_eps", hf_cfg.get("rms_norm_eps", hf_cfg.get("layer_norm_eps", 1e-5)))))

    # Partial rotary factor (Phi-3; stored in config but not yet kernel-enforced)
    partial_rotary_factor = float(rope_param("partial_rotary_factor", 1.0))
    if partial_rotary_factor != 1.0:
        print(f"[info] partial_rotary_factor={partial_rotary_factor}: only the leading "
              f"{partial_rotary_factor:.0%} of each head is rotated.")

    layer_types_raw = text_cfg.get("layer_types", [])
    layer_attention_types = []
    for value in layer_types_raw:
        lower = str(value).lower()
        if "linear_attention" in lower:
            layer_attention_types.append("linear")
        elif "sliding" in lower:
            layer_attention_types.append("sliding_window")
        else:
            layer_attention_types.append("full")

    linear_num_key_heads = int(text_cfg.get("linear_num_key_heads", 0) or 0)
    linear_num_value_heads = int(text_cfg.get("linear_num_value_heads", 0) or 0)

    return {
        "model_family":        family,
        "model_family_id":     FAMILY_ID.get(family, 0),
        "vocab_size":          int(text_cfg.get("vocab_size", hf_cfg.get("vocab_size"))),
        "hidden_size":         int(text_cfg.get("hidden_size", hf_cfg.get("hidden_size"))),
        "intermediate_size":   int(text_cfg.get("intermediate_size", hf_cfg.get("intermediate_size"))),
        "num_layers":          int(text_cfg.get("num_hidden_layers", hf_cfg.get("num_hidden_layers"))),
        "num_heads":           num_heads,
        "num_kv_heads":        num_kv_heads,
        "max_seq_len":         int(text_cfg.get("max_position_embeddings", hf_cfg.get("max_position_embeddings", 4096))),
        "rope_theta":          rope_theta,
        "norm_eps":            norm_eps,
        "sliding_window":      sliding_window,
        "tie_word_embeddings": tie_word_embeddings,
        "has_qkv_bias":        has_qkv_bias,
        "has_qk_norm":         has_qk_norm,
        "mlp_gelu":            mlp_gelu,
        "scale_embeddings":    scale_embeddings,
        "use_layernorm":       use_layernorm,
        "partial_rotary_factor": partial_rotary_factor,
        **vision_cfg,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        # q_proj emits [q|gate] per head when this is set, so it is twice as wide as
        # heads*head_dim. The container records it as a flag; without it the doubled projection
        # reads as a head_dim twice the real one.
        "attn_output_gate": bool(text_cfg.get("attn_output_gate", False)),
        "linear_key_head_dim": int(text_cfg.get("linear_key_head_dim", 0) or 0),
        "linear_value_head_dim": int(text_cfg.get("linear_value_head_dim", 0) or 0),
        "linear_conv_kernel_dim": int(text_cfg.get("linear_conv_kernel_dim", 0) or 0),
        "layer_attention_types": layer_attention_types,
        "num_local_experts":   num_local_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "expert_intermediate_size": expert_intermediate_size,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------

def build_qwen35_vision_mapping(depth: int):
    """The vision tower's tensors, under model.visual.*.

    Names are flattened to vision.* so the container's flat namespace stays readable and the
    engine can find them without knowing HuggingFace's nesting. patch_embed.proj is a Conv3d
    whose stride equals its kernel, which makes it a plain Linear; its [768,3,2,16,16] weight
    is already contiguous as [768,1536], so it is stored as-is and reinterpreted, not reshaped.
    """
    V = "model.visual."
    items = [
        (V + "patch_embed.proj.weight", "vision.patch_embed.weight", True),
        (V + "patch_embed.proj.bias", "vision.patch_embed.bias", True),
        (V + "pos_embed.weight", "vision.pos_embed.weight", True),
        (V + "merger.norm.weight", "vision.merger.norm.weight", True),
        (V + "merger.norm.bias", "vision.merger.norm.bias", True),
        (V + "merger.linear_fc1.weight", "vision.merger.fc1.weight", True),
        (V + "merger.linear_fc1.bias", "vision.merger.fc1.bias", True),
        (V + "merger.linear_fc2.weight", "vision.merger.fc2.weight", True),
        (V + "merger.linear_fc2.bias", "vision.merger.fc2.bias", True),
    ]
    for i in range(depth):
        B = f"{V}blocks.{i}."
        C = f"vision.blocks.{i}."
        for hf, own in [("norm1.weight", "norm1.weight"), ("norm1.bias", "norm1.bias"),
                        ("norm2.weight", "norm2.weight"), ("norm2.bias", "norm2.bias"),
                        ("attn.qkv.weight", "attn.qkv.weight"), ("attn.qkv.bias", "attn.qkv.bias"),
                        ("attn.proj.weight", "attn.proj.weight"),
                        ("attn.proj.bias", "attn.proj.bias"),
                        ("mlp.linear_fc1.weight", "mlp.fc1.weight"),
                        ("mlp.linear_fc1.bias", "mlp.fc1.bias"),
                        ("mlp.linear_fc2.weight", "mlp.fc2.weight"),
                        ("mlp.linear_fc2.bias", "mlp.fc2.bias")]:
            items.append((B + hf, C + own, True))
    return items


def build_qwen35_mapping(num_layers: int, layer_types: list, has_qk_norm: bool,
                         vision_depth: int = 0):
    """Qwen3.5: tensors sit under model.language_model, and each layer is EITHER delta-net or
    full attention; `layer_types` says which, and the two carry disjoint tensor sets.

    The vision tower is included when vision_depth > 0 (see build_qwen35_vision_mapping). The
    multi-token-prediction head (mtp.*) is still deliberately skipped, and the caller reports it
    as skipped rather than letting it look absent.
    """
    P = "model.language_model."
    items = [
        (P + "embed_tokens.weight", "tok_embeddings.weight", True),
        (P + "norm.weight", "norm.weight", True),
        ("lm_head.weight", "output.weight", False),  # tied in the released checkpoints
    ]
    for i in range(num_layers):
        kind = layer_types[i] if i < len(layer_types) else "full_attention"
        items.append((f"{P}layers.{i}.post_attention_layernorm.weight",
                      f"layers.{i}.ffn_norm.weight", True))
        # Both layer kinds have input_layernorm; the delta-net block normalises its input the
        # same way the attention block does. linear_attn.norm is a second, narrower norm inside
        # the block (value_head_dim wide, gated by z), not a replacement for this one. Emitting
        # this only for attention layers leaves the delta-net block reading raw residual: the
        # container loads, the model runs, and the block is orthogonal to the reference.
        items.append((f"{P}layers.{i}.input_layernorm.weight",
                      f"layers.{i}.attention_norm.weight", True))
        items.extend([
            (f"{P}layers.{i}.mlp.gate_proj.weight", f"layers.{i}.feed_forward.w1", True),
            (f"{P}layers.{i}.mlp.down_proj.weight", f"layers.{i}.feed_forward.w2", True),
            (f"{P}layers.{i}.mlp.up_proj.weight", f"layers.{i}.feed_forward.w3", True),
        ])
        if "linear" in kind:
            # Delta-net. linear_attn.norm is the output norm inside the block (value_head_dim
            # wide, gated by z); it is in addition to the input_layernorm emitted above.
            items.extend([
                (f"{P}layers.{i}.linear_attn.in_proj_qkv.weight",
                 f"layers.{i}.linear_attn.in_proj_qkv", True),
                (f"{P}layers.{i}.linear_attn.in_proj_z.weight",
                 f"layers.{i}.linear_attn.in_proj_z", True),
                (f"{P}layers.{i}.linear_attn.in_proj_a.weight",
                 f"layers.{i}.linear_attn.in_proj_a", True),
                (f"{P}layers.{i}.linear_attn.in_proj_b.weight",
                 f"layers.{i}.linear_attn.in_proj_b", True),
                (f"{P}layers.{i}.linear_attn.out_proj.weight",
                 f"layers.{i}.linear_attn.out_proj", True),
                (f"{P}layers.{i}.linear_attn.conv1d.weight",
                 f"layers.{i}.linear_attn.conv1d", True),
                (f"{P}layers.{i}.linear_attn.dt_bias",
                 f"layers.{i}.linear_attn.dt_bias", True),
                (f"{P}layers.{i}.linear_attn.A_log",
                 f"layers.{i}.linear_attn.a_log", True),
                (f"{P}layers.{i}.linear_attn.norm.weight",
                 f"layers.{i}.linear_attn.norm", True),
            ])
        else:
            items.extend([
                (f"{P}layers.{i}.self_attn.q_proj.weight", f"layers.{i}.attention.wq", True),
                (f"{P}layers.{i}.self_attn.k_proj.weight", f"layers.{i}.attention.wk", True),
                (f"{P}layers.{i}.self_attn.v_proj.weight", f"layers.{i}.attention.wv", True),
                (f"{P}layers.{i}.self_attn.o_proj.weight", f"layers.{i}.attention.wo", True),
            ])
            if has_qk_norm:
                items.extend([
                    (f"{P}layers.{i}.self_attn.q_norm.weight",
                     f"layers.{i}.attention.q_norm", True),
                    (f"{P}layers.{i}.self_attn.k_norm.weight",
                     f"layers.{i}.attention.k_norm", True),
                ])
    if vision_depth > 0:
        items.extend(build_qwen35_vision_mapping(vision_depth))
    return items


def build_mapping(family: str, num_layers: int, has_qkv_bias: bool, num_local_experts: int,
                  has_qk_norm: bool = False, layer_types: list = None,
                  vision_depth: int = 0):
    """
    Map HuggingFace tensor names to canonical internal names used by LL2CUDA.

    Dense families share the standard decoder naming.
    MoE checkpoints use block_sparse_moe router + per-expert FFN weights.
    """
    if family == FAMILY_QWEN3_5:
        return build_qwen35_mapping(num_layers, layer_types or [], has_qk_norm, vision_depth)

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
        if has_qk_norm:
            # Qwen3 per-head RMSNorm on Q and K (head_dim-sized), applied after the
            # projection and before RoPE.
            layer += [
                (f"model.layers.{i}.self_attn.q_norm.weight",
                 f"layers.{i}.attention.q_norm",  True),
                (f"model.layers.{i}.self_attn.k_norm.weight",
                 f"layers.{i}.attention.k_norm",  True),
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
    reason = unsupported_reason(hf_cfg)
    if reason:
        raise RuntimeError(reason)
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
    mapping    = build_mapping(family, num_layers, has_qkv_bias, num_local_experts,
                               bool(model_cfg.get("has_qk_norm", False)),
                               model_cfg.get("layer_attention_types", []),
                               int(model_cfg.get("vision_depth", 0) or 0))
    mapped_sources = {src for src, _dst, _required in mapping}

    if family == FAMILY_QWEN3_5:
        # Say plainly what is being left behind. These are not missing tensors; they are parts of
        # the model this conversion does not represent, and silently dropping 168 of 488 tensors
        # is the kind of thing that reads later as a corrupt checkpoint.
        n_vis = sum(1 for k in weight_map if k.startswith("model.visual"))
        n_mtp = sum(1 for k in weight_map if k.startswith("mtp."))
        kinds = model_cfg.get("layer_attention_types", [])
        n_lin = sum(1 for k in kinds if "linear" in str(k))
        print(f"[info] qwen3.5: {n_lin} linear-attention + {len(kinds) - n_lin} full-attention "
              f"layers")
        if n_vis or n_mtp:
            print(f"[warn] skipping {n_vis} vision-tower and {n_mtp} MTP tensors: this "
                  f"conversion produces the TEXT model only")

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

    # Gemma stores RMSNorm weights `w` and applies `(1 + w)`. Fold the +1 in here
    # so the standard (scale = w) RMSNorm kernel reproduces Gemma's scaling.
    if family == FAMILY_GEMMA:
        folded = 0
        for dst_name in list(extracted.keys()):
            if dst_name.endswith("_norm.weight") or dst_name == "norm.weight":
                arr = np.frombuffer(extracted[dst_name], dtype=np.float16).astype(np.float32) + 1.0
                extracted[dst_name] = arr.astype(np.float16).tobytes()
                folded += 1
        print(f"[info] gemma: folded +1 into {folded} RMSNorm weight tensors")

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
    try:
        main()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
