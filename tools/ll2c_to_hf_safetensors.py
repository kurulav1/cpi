#!/usr/bin/env python3
"""Rebuild an HF-format safetensors file from a CPI .ll2c container.

Why this exists: to benchmark CPI against llama.cpp on a real-size model we need the SAME
weights in GGUF form, and the original HF safetensors had been deleted -- only the .ll2c
remained. Rather than re-download 16 GB, reconstruct the HF file from the .ll2c and hand it to
llama.cpp's own convert_hf_to_gguf.py.

Going through llama.cpp's converter (instead of writing GGUF directly) is the whole point: it
gets the tokenizer, the metadata, and -- critically -- the Q/K PERMUTATION right.

  ⚠ THE PERMUTATION. HF stores q_proj/k_proj permuted for its rotate_half RoPE; llama.cpp's
  converter permutes them back. That is only correct if the bytes we hand it are raw HF layout.
  CPI's own converter does NOT permute (checked), so the .ll2c holds exactly the HF bytes and
  the round-trip is sound. If CPI ever starts permuting at load time, this script silently
  produces a subtly wrong model -- the kind that still generates fluent text. The caller MUST
  sanity-check llama.cpp's generation before trusting any benchmark built on the output.

Streams tensor-by-tensor; never holds the model in RAM.
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

MAGIC = b"LL2CUDA\x00"
HDR_V4 = struct.Struct("<8siiiiiiiiiiQffiiiiii")
ENTRY = struct.Struct("<64sqq")


def hf_name(ll2c: str) -> str:
    """.ll2c tensor name -> HuggingFace Llama name."""
    if ll2c == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    if ll2c == "norm.weight":
        return "model.norm.weight"
    if ll2c == "output.weight":
        return "lm_head.weight"
    parts = ll2c.split(".")
    if parts[0] != "layers":
        raise ValueError(f"unmapped tensor: {ll2c}")
    i = parts[1]
    rest = ".".join(parts[2:])
    table = {
        "attention.wq": f"model.layers.{i}.self_attn.q_proj.weight",
        "attention.wk": f"model.layers.{i}.self_attn.k_proj.weight",
        "attention.wv": f"model.layers.{i}.self_attn.v_proj.weight",
        "attention.wo": f"model.layers.{i}.self_attn.o_proj.weight",
        "attention_norm.weight": f"model.layers.{i}.input_layernorm.weight",
        "feed_forward.w1": f"model.layers.{i}.mlp.gate_proj.weight",
        "feed_forward.w2": f"model.layers.{i}.mlp.down_proj.weight",
        "feed_forward.w3": f"model.layers.{i}.mlp.up_proj.weight",
        "ffn_norm.weight": f"model.layers.{i}.post_attention_layernorm.weight",
    }
    if rest not in table:
        raise ValueError(f"unmapped tensor: {ll2c}")
    return table[rest]


def shape_for(ll2c: str, cfg: dict) -> list:
    """Shape is NOT stored per tensor in the .ll2c -- derive it, then cross-check the byte count."""
    hidden = cfg["hidden"]
    inter = cfg["inter"]
    vocab = cfg["vocab"]
    kv_dim = cfg["kv_heads"] * (cfg["q_hidden"] // cfg["heads"])
    if ll2c in ("tok_embeddings.weight", "output.weight"):
        return [vocab, hidden]
    if ll2c == "norm.weight":
        return [hidden]
    rest = ".".join(ll2c.split(".")[2:])
    return {
        "attention.wq": [cfg["q_hidden"], hidden],
        "attention.wk": [kv_dim, hidden],
        "attention.wv": [kv_dim, hidden],
        "attention.wo": [hidden, cfg["q_hidden"]],
        "attention_norm.weight": [hidden],
        "feed_forward.w1": [inter, hidden],
        "feed_forward.w2": [hidden, inter],
        "feed_forward.w3": [inter, hidden],
        "ffn_norm.weight": [hidden],
    }[rest]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ll2c", required=True)
    ap.add_argument("--hf-src", required=True, help="dir with config.json / tokenizer.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src = Path(args.ll2c)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with src.open("rb") as f:
        v = HDR_V4.unpack(f.read(HDR_V4.size))
        if v[0] != MAGIC:
            raise ValueError("not an .ll2c file")
        # Field order per tools/validate_ll2c.py. Do NOT guess it: every field is an int, so a
        # wrong index yields a plausible-looking config (heads=8, kv_heads=1) that only blows up
        # much later, on a byte-count mismatch.
        (_magic, _version, vocab, hidden, inter, layers, heads, kv_heads, _max_seq, _tp,
         ntensors, table_off) = v[:12]
        cfg = dict(vocab=vocab, hidden=hidden, inter=inter, layers=layers, heads=heads,
                   kv_heads=kv_heads, q_hidden=hidden)
        # The tensor table lives at table_off -- NOT immediately after the header.
        f.seek(table_off)
        entries = []
        for _ in range(ntensors):
            nm, off, sz = ENTRY.unpack(f.read(ENTRY.size))
            entries.append((nm.rstrip(b"\x00").decode(), off, sz))

    print(f"[ll2c] vocab={cfg['vocab']} hidden={cfg['hidden']} inter={cfg['inter']} "
          f"layers={cfg['layers']} heads={cfg['heads']} kv_heads={cfg['kv_heads']} "
          f"tensors={ntensors}")

    # Build the safetensors header. Byte counts are CROSS-CHECKED against the derived shapes --
    # a shape mistake here would produce a model that loads and generates garbage.
    header = {}
    cursor = 0
    plan = []
    for nm, off, sz in entries:
        shape = shape_for(nm, cfg)
        expect = 1
        for d in shape:
            expect *= d
        expect *= 2  # fp16
        if expect != sz:
            raise ValueError(f"{nm}: derived shape {shape} = {expect} bytes, file says {sz}")
        h = hf_name(nm)
        header[h] = {"dtype": "F16", "shape": shape, "data_offsets": [cursor, cursor + sz]}
        plan.append((off, sz))
        cursor += sz

    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(blob) % 8)) % 8
    blob += b" " * pad

    dst = out / "model.safetensors"
    print(f"[write] {dst}  ({cursor / 1e9:.2f} GB)")
    with src.open("rb") as fi, dst.open("wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for i, (off, sz) in enumerate(plan):
            fi.seek(off)
            left = sz
            while left:
                chunk = fi.read(min(left, 64 << 20))
                if not chunk:
                    raise IOError("short read")
                fo.write(chunk)
                left -= len(chunk)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(plan)} tensors", flush=True)

    # config.json / tokenizer.json come along unchanged -- llama.cpp's converter needs them.
    hf = Path(args.hf_src)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "generation_config.json"):
        if (hf / f).exists():
            shutil.copy(hf / f, out / f)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
