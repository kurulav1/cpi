#!/usr/bin/env python3
"""
Pack tensor .bin files into LL2CUDA mmap-friendly container.

All *.bin files inside input-dir are included.
Tensor name is derived from filename stem, e.g.:
  layers.0.attention.wq.bin -> layers.0.attention.wq
"""

import argparse
import json
import re
import struct
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

MAGIC = b"LL2CUDA\x00"
VERSION = 4

# HeaderV4 layout (matches HeaderV4 struct in weight_loader.cpp):
#   magic[8], version, vocab, hidden, inter, layers, heads, kv_heads,
#   max_seq, tp, tensor_count, table_offset (Q=uint64),
#   rope_theta (f), norm_eps (f), sliding_window (i), flags (i), model_family_id (i)
#   num_local_experts (i), num_experts_per_tok (i), expert_intermediate_size (i)
#
# flags bit layout: bit 0 = tie_word_embeddings, bit 1 = has_qkv_bias,
# bit 2 = use_layernorm
HEADER_FMT = "<8siiiiiiiiiiQffiiiiii"
ENTRY_FMT = "<64sqq"


def pad_name(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) >= 64:
        raise ValueError(f"Tensor name too long: {name}")
    return raw + b"\x00" * (64 - len(raw))


def load_config(src: Path, args) -> dict:
    cfg_path = src / "model_config.json"
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # V2 core fields
    result = {
        "vocab_size":       args.vocab_size       if args.vocab_size       is not None else cfg.get("vocab_size", 32000),
        "hidden_size":      args.hidden_size      if args.hidden_size      is not None else cfg.get("hidden_size", 4096),
        "intermediate_size": args.intermediate_size if args.intermediate_size is not None else cfg.get("intermediate_size", 11008),
        "num_layers":       args.num_layers       if args.num_layers       is not None else cfg.get("num_layers", 32),
        "num_heads":        args.num_heads        if args.num_heads        is not None else cfg.get("num_heads", 32),
        "num_kv_heads":     args.num_kv_heads     if args.num_kv_heads     is not None else cfg.get("num_kv_heads", cfg.get("num_heads", 32)),
        "max_seq_len":      args.max_seq_len      if args.max_seq_len      is not None else cfg.get("max_seq_len", 4096),
        "tensor_parallel":  args.tensor_parallel,
    }

    # V3 extended fields
    result["rope_theta"]          = float(cfg.get("rope_theta", 0.0))
    result["norm_eps"]            = float(cfg.get("norm_eps", 1e-5))
    result["sliding_window"]      = int(cfg.get("sliding_window", 0))
    result["tie_word_embeddings"] = bool(cfg.get("tie_word_embeddings", False))
    result["has_qkv_bias"]        = bool(cfg.get("has_qkv_bias", False))
    result["use_layernorm"]       = bool(cfg.get("use_layernorm", False))
    result["model_family_id"]     = int(cfg.get("model_family_id", 0))
    result["num_local_experts"] = int(cfg.get("num_local_experts", 0) or 0)
    result["num_experts_per_tok"] = int(cfg.get("num_experts_per_tok", 0) or 0)
    result["expert_intermediate_size"] = int(cfg.get("expert_intermediate_size", 0) or 0)

    return result


def is_packable_streaming_tensor(name: str) -> bool:
    if re.fullmatch(r"layers\.\d+\.feed_forward\.w[123]", name):
        return True
    if re.fullmatch(r"layers\.\d+\.feed_forward\.experts\.\d+\.w[123]", name):
        return True
    return False


def infer_packable_shape(name: str, cfg: dict) -> tuple[int, int]:
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    expert_inter = int(cfg.get("expert_intermediate_size", 0) or 0)
    if expert_inter <= 0:
        expert_inter = inter

    if re.fullmatch(r"layers\.\d+\.feed_forward\.w[13]", name):
        return inter, hidden
    if re.fullmatch(r"layers\.\d+\.feed_forward\.w2", name):
        return hidden, inter
    if re.fullmatch(r"layers\.\d+\.feed_forward\.experts\.\d+\.w[13]", name):
        return expert_inter, hidden
    if re.fullmatch(r"layers\.\d+\.feed_forward\.experts\.\d+\.w2", name):
        return hidden, expert_inter
    raise ValueError(f"Unsupported packable tensor shape inference for {name}")


def quantize_fp16_blob_to_int8(blob: bytes, rows: int, cols: int) -> tuple[bytes, bytes]:
    if np is None:
        raise RuntimeError("--emit-streaming-int8 requires numpy")

    values = np.frombuffer(blob, dtype=np.float16)
    if values.size != rows * cols:
        raise ValueError(f"{rows}x{cols} shape does not match fp16 blob length for packed tensor")
    if values.size == 0:
        quantized = np.empty(0, dtype=np.int8)
        scales = np.empty(rows, dtype=np.float32)
    else:
        values_f32 = values.astype(np.float32).reshape(rows, cols)
        max_abs = np.max(np.abs(values_f32), axis=1)
        scales = np.maximum(max_abs / 127.0, 1.0e-8).astype(np.float32)
        quantized = np.rint(np.clip(values_f32 / scales[:, None], -127.0, 127.0)).astype(np.int8)

    return quantized.tobytes(), scales.tobytes()


def quantize_fp16_blob_to_int4(blob: bytes, rows: int, cols: int) -> tuple[bytes, bytes]:
    if np is None:
        raise RuntimeError("--emit-streaming-int4 requires numpy")

    values = np.frombuffer(blob, dtype=np.float16)
    if values.size != rows * cols:
        raise ValueError(f"{rows}x{cols} shape does not match fp16 blob length for packed tensor")

    if values.size == 0:
        packed = np.empty(0, dtype=np.uint8)
        scales = np.empty(rows, dtype=np.float32)
    else:
        values_f32 = values.astype(np.float32).reshape(rows, cols)
        max_abs = np.max(np.abs(values_f32), axis=1)
        scales = np.maximum(max_abs / 7.0, 1.0e-8).astype(np.float32)
        quantized = np.rint(np.clip(values_f32 / scales[:, None], -7.0, 7.0)).astype(np.int8)

        packed_cols = (cols + 1) // 2
        packed = np.zeros((rows, packed_cols), dtype=np.uint8)
        even = quantized[:, 0::2].astype(np.int16)
        odd = np.zeros_like(even, dtype=np.int16)
        odd_src = quantized[:, 1::2].astype(np.int16)
        odd[:, : odd_src.shape[1]] = odd_src
        packed[:, :] = ((even & 0x0F) | ((odd & 0x0F) << 4)).astype(np.uint8)

    return packed.tobytes(), scales.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vocab-size", type=int)
    ap.add_argument("--hidden-size", type=int)
    ap.add_argument("--intermediate-size", type=int)
    ap.add_argument("--num-layers", type=int)
    ap.add_argument("--num-heads", type=int)
    ap.add_argument("--num-kv-heads", type=int)
    ap.add_argument("--max-seq-len", type=int)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--emit-streaming-int8", action="store_true",
                    help="Add packed .int8/.scale tensors for large streamed MLP weights.")
    ap.add_argument("--emit-streaming-int4", action="store_true",
                    help="Add packed .int4/.scale tensors for large streamed MLP weights.")
    ap.add_argument("--omit-fp16-layer-tensors", action="store_true",
                    help="Omit fp16 copies for tensors that were packed to low-bit form. Requires --emit-streaming-int8 or --emit-streaming-int4.")
    args = ap.parse_args()

    if args.omit_fp16_layer_tensors and not (args.emit_streaming_int8 or args.emit_streaming_int4):
        raise ValueError("--omit-fp16-layer-tensors requires --emit-streaming-int8 or --emit-streaming-int4")
    if args.emit_streaming_int8 and args.emit_streaming_int4:
        raise ValueError("Choose only one low-bit packing mode: --emit-streaming-int8 or --emit-streaming-int4")

    src = Path(args.input_dir)
    if not src.exists():
        raise FileNotFoundError(src)

    cfg = load_config(src, args)

    bins = sorted(src.glob("*.bin"))
    if not bins:
        raise RuntimeError(f"No .bin files found in {src}")

    blobs_by_name = {}
    packed_int8_count = 0
    packed_int4_count = 0
    omitted_fp16_count = 0
    for p in bins:
        name = p.stem
        blob = p.read_bytes()
        layer_tensor = is_packable_streaming_tensor(name)

        if not (args.omit_fp16_layer_tensors and layer_tensor):
            blobs_by_name[name] = blob
        else:
            omitted_fp16_count += 1

        if args.emit_streaming_int8 and layer_tensor:
            rows, cols = infer_packable_shape(name, cfg)
            qblob, sblob = quantize_fp16_blob_to_int8(blob, rows, cols)
            blobs_by_name[f"{name}.int8"] = qblob
            blobs_by_name[f"{name}.scale"] = sblob
            packed_int8_count += 1
        if args.emit_streaming_int4 and layer_tensor:
            rows, cols = infer_packable_shape(name, cfg)
            qblob, sblob = quantize_fp16_blob_to_int4(blob, rows, cols)
            blobs_by_name[f"{name}.int4"] = qblob
            blobs_by_name[f"{name}.scale"] = sblob
            packed_int4_count += 1

    tensor_names = list(blobs_by_name.keys())
    tensor_count = len(tensor_names)

    header_size = struct.calcsize(HEADER_FMT)
    table_offset = header_size
    table_size = tensor_count * struct.calcsize(ENTRY_FMT)
    cursor = table_offset + table_size

    entries = []
    blobs = []
    for name in tensor_names:
        blob = blobs_by_name[name]
        entries.append((pad_name(name), cursor, len(blob)))
        blobs.append(blob)
        cursor += len(blob)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flags = 0
    if cfg.get("tie_word_embeddings"):
        flags |= 1
    if cfg.get("has_qkv_bias"):
        flags |= 2
    if cfg.get("use_layernorm"):
        flags |= 4

    with out_path.open("wb") as f:
        f.write(struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            cfg["vocab_size"],
            cfg["hidden_size"],
            cfg["intermediate_size"],
            cfg["num_layers"],
            cfg["num_heads"],
            cfg["num_kv_heads"],
            cfg["max_seq_len"],
            cfg["tensor_parallel"],
            tensor_count,
            table_offset,
            cfg["rope_theta"],       # float
            cfg["norm_eps"],         # float
            cfg["sliding_window"],   # int32
            flags,                   # int32
            cfg["model_family_id"],  # int32
            cfg["num_local_experts"],       # int32
            cfg["num_experts_per_tok"],     # int32
            cfg["expert_intermediate_size"],# int32
        ))
        for e in entries:
            f.write(struct.pack(ENTRY_FMT, *e))
        for blob in blobs:
            f.write(blob)

    print(f"Packed tensors: {tensor_count}")
    if args.emit_streaming_int8:
        print(f"Packed int8 layer tensors: {packed_int8_count}")
    if args.emit_streaming_int4:
        print(f"Packed int4 layer tensors: {packed_int4_count}")
    if args.omit_fp16_layer_tensors:
        print(f"Omitted fp16 layer tensors: {omitted_fp16_count}")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
