#!/usr/bin/env python3
"""Stream HuggingFace safetensors directly into an int4-packed .ll2c.

Why this exists: convert_hf_to_bins.py + pack_ll2c.py each buffer the *entire*
model in RAM (the `extracted` / `blobs_by_name` dicts) and write a full fp16
intermediate. For a 32B model that is ~65 GB of RAM and ~65 GB of extra disk --
impossible on a 32 GB / disk-tight machine. This tool processes one tensor at a
time and writes the final container in a single pass, so peak RAM is ~one
tensor and peak disk is HF + the (smaller) output, with no fp16 intermediate.

It reuses the *exact* tensor mapping, dtype conversion, container format and
int4 quantizer from convert_hf_to_bins.py / pack_ll2c.py so output is identical
to running those two tools with --emit-streaming-int4 --omit-fp16-layer-tensors.

MLP weights (feed_forward.w1/w2/w3) are packed to int4 + per-row fp32 scales;
the fp16 copies are omitted. Everything else (embeddings, attention, norms,
lm_head, biases) is stored fp16, exactly as the engine's int4 streaming path
expects -- it quantizes the fp16 attention projections at load time.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_hf_to_bins import (  # noqa: E402
    FAMILY_PHI3,
    build_mapping,
    detect_family,
    extract_model_config,
    load_hf_config,
    load_index,
    read_safetensor_blob,
    unsupported_reason,
)
from pack_ll2c import (  # noqa: E402
    ATTENTION_KIND_ID,
    ENTRY_FMT,
    HEADER_FMT,
    MAGIC,
    VERSION,
    infer_packable_shape,
    is_packable_streaming_tensor,
    pad_name,
    quantize_fp16_blob_to_int4,
)


def shard_headers(hf_dir: Path, weight_map: dict) -> dict:
    """Map src tensor name -> (shape, num_elements) by reading shard headers only."""
    shards = set(weight_map.values())
    meta_by_name = {}
    for shard_rel in shards:
        path = hf_dir / shard_rel
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = meta.get("shape", [])
            n = 1
            for d in shape:
                n *= int(d)
            meta_by_name[name] = (shape, n)
    return meta_by_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--family", default=None)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hf_cfg = load_hf_config(hf_dir)
    reason = unsupported_reason(hf_cfg)
    if reason:
        raise RuntimeError(reason)
    family = args.family or detect_family(hf_cfg)
    if family == "phimoe":
        family = FAMILY_PHI3

    cfg = extract_model_config(hf_cfg, family)
    cfg["tensor_parallel"] = 1
    num_layers = cfg["num_layers"]
    num_local_experts = int(cfg.get("num_local_experts", 0) or 0)

    index = load_index(hf_dir)
    weight_map = index.get("weight_map", {})
    mapping = build_mapping(family, num_layers, cfg["has_qkv_bias"], num_local_experts)
    meta = shard_headers(hf_dir, weight_map)

    print(f"[info] family={family} layers={num_layers} hidden={cfg['hidden_size']} "
          f"vocab={cfg['vocab_size']} qkv_bias={cfg['has_qkv_bias']}", flush=True)

    # ---- Build the ordered output plan (no tensor data loaded yet) ----------
    # Each plan entry: dict(kind, name, size, ...source info).
    plan = []
    # Per-layer accumulators for the fused bqkv bias tensor.
    bias_parts = {}  # layer_idx -> {"bq": src, "bk": src, "bv": src}

    def fp16_size(src_name):
        return meta[src_name][1] * 2

    for src_name, dst_name, required in mapping:
        shard_rel = weight_map.get(src_name)
        if shard_rel is None:
            if required:
                raise KeyError(f"Missing required tensor: {src_name}")
            continue

        # Collect QKV bias parts to fuse after the layer's weights.
        for part in ("bq", "bk", "bv"):
            if dst_name.endswith(f".attention.{part}"):
                li = int(dst_name.split(".")[1])
                bias_parts.setdefault(li, {})[part] = (src_name, shard_rel)
                break
        else:
            if is_packable_streaming_tensor(dst_name):
                rows, cols = infer_packable_shape(dst_name, cfg)
                if rows * cols != meta[src_name][1]:
                    raise ValueError(f"{dst_name}: shape {rows}x{cols} != {meta[src_name][1]} elems")
                int4_size = rows * ((cols + 1) // 2)
                scale_size = rows * 4
                plan.append({"kind": "int4", "name": f"{dst_name}.int4", "size": int4_size,
                             "src": src_name, "shard": shard_rel, "rows": rows, "cols": cols})
                plan.append({"kind": "scale", "name": f"{dst_name}.scale", "size": scale_size})
            else:
                plan.append({"kind": "fp16", "name": dst_name, "size": fp16_size(src_name),
                             "src": src_name, "shard": shard_rel})
            continue

        # After emitting all of a layer's mapped tensors, flush its fused bias.
        # build_mapping emits q/k/v/o weights then (later) the biases, so by the
        # time we see bv we have all three; emit bqkv right then.
        if dst_name.endswith(".attention.bv"):
            li = int(dst_name.split(".")[1])
            parts = bias_parts.get(li, {})
            if {"bq", "bk", "bv"} <= parts.keys():
                size = sum(meta[parts[p][0]][1] * 2 for p in ("bq", "bk", "bv"))
                plan.append({"kind": "bqkv", "name": f"layers.{li}.attention.bqkv",
                             "size": size, "parts": [parts[p] for p in ("bq", "bk", "bv")]})

    tensor_count = len(plan)

    # ---- Attention-type table (empty for plain Qwen2/Llama) -----------------
    attn_ids = []
    for value in cfg.get("layer_attention_types", []) or []:
        if value not in ATTENTION_KIND_ID:
            raise ValueError(f"Unsupported layer attention type: {value}")
        attn_ids.append(ATTENTION_KIND_ID[value])
    if attn_ids and len(attn_ids) != num_layers:
        raise ValueError("layer_attention_types length must match num_layers")

    header_size = struct.calcsize(HEADER_FMT)
    attn_blob = struct.pack("<" + "i" * len(attn_ids), *attn_ids) if attn_ids else b""
    attn_offset = header_size if attn_ids else 0
    table_offset = header_size + len(attn_blob)
    table_size = tensor_count * struct.calcsize(ENTRY_FMT)

    cursor = table_offset + table_size
    for e in plan:
        e["offset"] = cursor
        cursor += e["size"]
    total_size = cursor

    flags = 0
    if cfg.get("tie_word_embeddings"):
        flags |= 1
    if cfg.get("has_qkv_bias"):
        flags |= 2
    if cfg.get("use_layernorm"):
        flags |= 4

    print(f"[info] tensors={tensor_count} output_size={total_size/1e9:.2f} GB", flush=True)

    # ---- Single streaming write pass ---------------------------------------
    written = 0
    with out_path.open("wb") as f:
        f.write(struct.pack(
            HEADER_FMT, MAGIC, VERSION,
            cfg["vocab_size"], cfg["hidden_size"], cfg["intermediate_size"],
            cfg["num_layers"], cfg["num_heads"], cfg["num_kv_heads"],
            cfg["max_seq_len"], cfg["tensor_parallel"], tensor_count, table_offset,
            cfg["rope_theta"], cfg["norm_eps"], cfg["sliding_window"], flags,
            cfg["model_family_id"], cfg["num_local_experts"], cfg["num_experts_per_tok"],
            cfg["expert_intermediate_size"], cfg["partial_rotary_factor"],
            cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
            len(attn_ids), attn_offset,
        ))
        if attn_blob:
            f.write(attn_blob)
        for e in plan:
            f.write(struct.pack(ENTRY_FMT, pad_name(e["name"]), e["offset"], e["size"]))

        pending_scale = None  # bytes for the .scale entry following an .int4 one
        for i, e in enumerate(plan):
            assert f.tell() == e["offset"], f"offset drift at {e['name']}"
            if e["kind"] == "fp16":
                blob, _ = read_safetensor_blob(hf_dir / e["shard"], e["src"])
                f.write(blob)
            elif e["kind"] == "bqkv":
                for src_name, shard_rel in e["parts"]:
                    blob, _ = read_safetensor_blob(hf_dir / shard_rel, src_name)
                    f.write(blob)
            elif e["kind"] == "int4":
                blob, _ = read_safetensor_blob(hf_dir / e["shard"], e["src"])
                packed, scales = quantize_fp16_blob_to_int4(blob, e["rows"], e["cols"])
                f.write(packed)
                pending_scale = scales
            elif e["kind"] == "scale":
                if pending_scale is None:
                    raise RuntimeError(f"scale entry {e['name']} without preceding int4")
                f.write(pending_scale)
                pending_scale = None
            if e["size"] != (f.tell() - e["offset"]):
                raise RuntimeError(f"size mismatch writing {e['name']}")
            written += 1
            if written % 100 == 0 or written == tensor_count:
                print(f"  [{written}/{tensor_count}] {e['name']}", flush=True)

    actual = out_path.stat().st_size
    if actual != total_size:
        raise RuntimeError(f"final size {actual} != planned {total_size}")
    print(f"[done] wrote {out_path} ({actual/1e9:.2f} GB, {tensor_count} tensors)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
