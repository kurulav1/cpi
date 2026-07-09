#!/usr/bin/env python3
"""Convert a HuggingFace Gemma 4 checkpoint to a CPI Gemma4 weight file.

Gemma 4 is a dedicated engine fork (see memory: cpi-gemma4-arch), not a
LlamaEngine absorb. This extracts the TEXT tower only (ignores vision_tower /
audio_tower), converts bf16 -> fp16, and writes a safetensors-format blob
(same container as .ll2c: 8-byte LE header length, JSON directory, raw data)
plus an embedded __metadata__ config block the engine reads.

Unlike Gemma 1/2/3, Gemma4RMSNorm uses scale = w (weight init to ones), so we
do NOT fold +1 into norm weights. Embedding scale (sqrt(hidden)) and per-layer
scales are applied at runtime by the engine.

Usage:
  python tools/convert_gemma4.py \
      --model artifacts/hub/google__gemma-4-E2B-it/hf \
      --out   artifacts/hub/google__gemma-4-E2B-it/gemma4-e2b.cpi
"""
import argparse, json, os, struct, sys
import numpy as np

PREFIX = "model.language_model."


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
        data_start = 8 + n
    return hdr, data_start


def load_tensor(path, hdr, data_start, name):
    meta = hdr[name]
    dt = meta["dtype"]
    shape = meta["shape"]
    a, b = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + a)
        raw = f.read(b - a)
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        f32 = (u16 << 16).view(np.float32)
    elif dt == "F16":
        f32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == "F32":
        f32 = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"unhandled dtype {dt} for {name}")
    return f32.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir with config.json + model.safetensors")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_all = json.load(open(os.path.join(args.model, "config.json")))
    tc = cfg_all.get("text_config", cfg_all)
    st = os.path.join(args.model, "model.safetensors")
    hdr, data_start = read_safetensors_header(st)

    nl = tc["num_hidden_layers"]
    layer_types = tc["layer_types"]
    cfg = {
        "family": "gemma4",
        "num_layers": nl,
        "hidden": tc["hidden_size"],
        "num_heads": tc["num_attention_heads"],
        "num_kv_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "global_head_dim": tc.get("global_head_dim") or tc["head_dim"],
        "intermediate": tc["intermediate_size"],
        "vocab": tc["vocab_size"],
        "rms_eps": tc["rms_norm_eps"],
        "sliding_window": tc["sliding_window"],
        "final_logit_softcapping": tc.get("final_logit_softcapping"),
        "hidden_size_per_layer_input": tc["hidden_size_per_layer_input"],
        "vocab_size_per_layer_input": tc["vocab_size_per_layer_input"],
        "num_kv_shared_layers": tc.get("num_kv_shared_layers", 0),
        "use_double_wide_mlp": bool(tc.get("use_double_wide_mlp", False)),
        "rope_theta_full": tc["rope_parameters"]["full_attention"]["rope_theta"],
        "rope_theta_sliding": tc["rope_parameters"]["sliding_attention"]["rope_theta"],
        "partial_rotary_full": tc["rope_parameters"]["full_attention"].get("partial_rotary_factor", 1.0),
        "layer_types": layer_types,
        "bos_token_id": tc.get("bos_token_id", 2),
        "eos_token_id": tc.get("eos_token_id", 1),
        "tie_word_embeddings": bool(tc.get("tie_word_embeddings", True)),
    }

    # tensors to export: name in output <- HF name (with PREFIX)
    exports = {
        "embed_tokens.weight": "embed_tokens.weight",
        "embed_tokens_per_layer.weight": "embed_tokens_per_layer.weight",
        "per_layer_model_projection.weight": "per_layer_model_projection.weight",
        "per_layer_projection_norm.weight": "per_layer_projection_norm.weight",
        "norm.weight": "norm.weight",
    }
    per_layer = [
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "per_layer_input_gate.weight", "per_layer_projection.weight",
        "post_per_layer_input_norm.weight", "layer_scalar",
    ]
    for L in range(nl):
        for t in per_layer:
            exports[f"layers.{L}.{t}"] = f"layers.{L}.{t}"

    # compute KV-share source mapping for the engine (which layer each shared layer reuses)
    first_shared = nl - cfg["num_kv_shared_layers"] if cfg["num_kv_shared_layers"] > 0 else nl
    prev = layer_types[:first_shared]
    kv_source = []  # kv_source[L] = layer whose K/V layer L uses (== L if own)
    for L in range(nl):
        if L >= first_shared > 0:
            t = layer_types[L]
            src = max(i for i in range(first_shared) if prev[i] == t)
            kv_source.append(src)
        else:
            kv_source.append(L)
    cfg["kv_source"] = kv_source
    cfg["first_shared_layer"] = first_shared

    # gather, convert to fp16, build blob
    out_hdr = {}
    blobs = []
    offset = 0
    missing = []
    for out_name, hf_suffix in exports.items():
        hf_name = PREFIX + hf_suffix
        if hf_name not in hdr:
            missing.append(hf_name)
            continue
        arr = load_tensor(st, hdr, data_start, hf_name).astype(np.float16)
        raw = np.ascontiguousarray(arr).tobytes()
        out_hdr[out_name] = {"dtype": "F16", "shape": list(arr.shape),
                             "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)

    # shared layers' k/v_proj are dead weights in HF; we still export them (harmless),
    # but the engine uses kv_source to decide. Report any genuinely-missing tensors.
    if missing:
        print("WARNING missing tensors (first 10):", missing[:10], f"... {len(missing)} total")

    out_hdr["__metadata__"] = {k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                               for k, v in cfg.items()}
    hdr_json = json.dumps(out_hdr).encode("utf-8")
    pad = (8 - len(hdr_json) % 8) % 8
    hdr_json += b" " * pad

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_json)))
        f.write(hdr_json)
        for b in blobs:
            f.write(b)
    total = offset + 8 + len(hdr_json)

    # Robust, dependency-free sidecar the C++ engine parses (avoids JSON parsing
    # in the loader). data_start = 8 + len(hdr_json); tensor offsets are relative
    # to data_start. One TENSOR line per tensor: name dtype d0,d1,... offset bytes.
    data_start = 8 + len(hdr_json)
    man = os.path.splitext(args.out)[0] + ".manifest"
    with open(man, "w", encoding="utf-8") as f:
        f.write(f"CPI_GEMMA4_MANIFEST 1\n")
        f.write(f"DATA_START {data_start}\n")
        for k, v in cfg.items():
            if isinstance(v, (list, dict)):
                f.write(f"CFGJSON {k} {json.dumps(v)}\n")
            else:
                f.write(f"CFG {k} {v}\n")
        for name, meta in out_hdr.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            shp = ",".join(str(s) for s in meta["shape"])
            f.write(f"TENSOR {name} {meta['dtype']} {shp} {a} {b - a}\n")
    print(f"wrote {args.out}  ({total/1e9:.2f} GB, {len(out_hdr)-1} tensors)")
    print(f"wrote {man}")
    print("config:", json.dumps({k: cfg[k] for k in cfg if k not in ('layer_types','kv_source')}, indent=1))
    print("kv_source:", kv_source)


if __name__ == "__main__":
    main()
