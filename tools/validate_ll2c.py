#!/usr/bin/env python3
"""
Validate LL2CUDA (.ll2c) model container header and tensor byte sizes.
"""

import argparse
import re
import struct
from pathlib import Path

MAGIC = b"LL2CUDA\x00"
HDR_V1 = struct.Struct("<8siiiiiiiiiQ")
HDR_V2 = struct.Struct("<8siiiiiiiiiiQ")
HDR_V3 = struct.Struct("<8siiiiiiiiiiQffiii")
HDR_V4 = struct.Struct("<8siiiiiiiiiiQffiiiiii")
ENTRY = struct.Struct("<64sqq")


def expect(name: str, got: int, want: int) -> None:
    if got != want:
        raise ValueError(f"{name}: expected {want} bytes, got {got} bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    args = ap.parse_args()

    data = args.model.read_bytes()
    if len(data) < HDR_V1.size:
        raise ValueError("file too small for header")

    version = struct.unpack_from("<i", data, 8)[0]
    num_local_experts = 0
    num_experts_per_tok = 0
    expert_inter = 0
    if version >= 4:
        if len(data) < HDR_V4.size:
            raise ValueError("file too small for v4 header")
        (
            magic,
            version,
            vocab,
            hidden,
            inter,
            layers,
            heads,
            kv_heads,
            max_seq,
            tp,
            tensor_count,
            table_off,
            _rope_theta,
            _norm_eps,
            _sliding_window,
            _flags,
            _model_family_id,
            num_local_experts,
            num_experts_per_tok,
            expert_inter,
        ) = HDR_V4.unpack_from(data, 0)
    elif version >= 3:
        if len(data) < HDR_V3.size:
            raise ValueError("file too small for v3 header")
        (
            magic,
            version,
            vocab,
            hidden,
            inter,
            layers,
            heads,
            kv_heads,
            max_seq,
            tp,
            tensor_count,
            table_off,
            _rope_theta,
            _norm_eps,
            _sliding_window,
            _flags,
            _model_family_id,
        ) = HDR_V3.unpack_from(data, 0)
    elif version >= 2:
        if len(data) < HDR_V2.size:
            raise ValueError("file too small for v2 header")
        (magic, version, vocab, hidden, inter, layers, heads, kv_heads, max_seq, tp, tensor_count, table_off) = (
            HDR_V2.unpack_from(data, 0)
        )
    else:
        (magic, version, vocab, hidden, inter, layers, heads, max_seq, tp, tensor_count, table_off) = HDR_V1.unpack_from(
            data, 0
        )
        kv_heads = heads

    if magic != MAGIC:
        raise ValueError("bad magic, expected LL2CUDA")
    if hidden <= 0 or inter <= 0 or layers <= 0 or heads <= 0 or kv_heads <= 0:
        raise ValueError("invalid model config in header")
    if heads % kv_heads != 0:
        raise ValueError("invalid head topology")

    entries = {}
    offsets = {}
    for i in range(tensor_count):
        base = table_off + i * ENTRY.size
        if base + ENTRY.size > len(data):
            raise ValueError("tensor table out of bounds")
        raw_name, off, nbytes = ENTRY.unpack_from(data, base)
        name = raw_name.split(b"\x00", 1)[0].decode("utf-8")
        if off < 0 or nbytes < 0 or off + nbytes > len(data):
            raise ValueError(f"tensor {name} has invalid offset/size")
        entries[name] = nbytes
        offsets[name] = off

    hsz = 2  # fp16
    def infer_rows(name: str, cols: int) -> int:
        if cols <= 0:
            raise ValueError(f"invalid cols for {name}: {cols}")
        if name in entries:
            row_bytes = cols * hsz
            if row_bytes == 0 or (entries[name] % row_bytes) != 0:
                raise ValueError(f"{name}: invalid fp16 matrix byte size")
            return entries[name] // row_bytes
        q8name = f"{name}.int8"
        if q8name in entries:
            row_bytes = cols
            if row_bytes == 0 or (entries[q8name] % row_bytes) != 0:
                raise ValueError(f"{q8name}: invalid int8 matrix byte size")
            return entries[q8name] // row_bytes
        q4name = f"{name}.int4"
        if q4name in entries:
            packed_cols = (cols + 1) // 2
            if packed_cols == 0 or (entries[q4name] % packed_cols) != 0:
                raise ValueError(f"{q4name}: invalid int4 matrix byte size")
            return entries[q4name] // packed_cols
        raise ValueError(f"missing tensor: {name} (or packed int8/int4 alternative)")

    q_hidden = infer_rows("layers.0.attention.wq", hidden)
    wk_rows = infer_rows("layers.0.attention.wk", hidden)
    wv_rows = infer_rows("layers.0.attention.wv", hidden)
    if q_hidden <= 0 or (q_hidden % heads) != 0:
        raise ValueError("invalid q_proj shape: rows must be positive and divisible by heads")
    head_dim = q_hidden // heads
    kv_hidden = kv_heads * head_dim
    if wk_rows != kv_hidden or wv_rows != kv_hidden:
        raise ValueError("invalid k_proj/v_proj shape for inferred attention head_dim")

    packed_tensor_count = 0

    def req(name: str, want: int) -> None:
        if name not in entries:
            raise ValueError(f"missing tensor: {name}")
        expect(name, entries[name], want)

    def req_fp16_or_packed_lowbit(
        name: str, fp16_bytes: int, int8_bytes: int, int4_bytes: int, scale_bytes: int = 4
    ) -> None:
        nonlocal packed_tensor_count
        if name in entries:
            expect(name, entries[name], fp16_bytes)
            return

        q8name = f"{name}.int8"
        q4name = f"{name}.int4"
        sname = f"{name}.scale"
        has_q8 = q8name in entries and sname in entries
        has_q4 = q4name in entries and sname in entries
        if not has_q8 and not has_q4:
            raise ValueError(f"missing tensor: {name} (or packed int8/int4 alternative)")
        if has_q8:
            expect(q8name, entries[q8name], int8_bytes)
        else:
            expect(q4name, entries[q4name], int4_bytes)
        if entries[sname] not in (4, scale_bytes):
            raise ValueError(f"{sname}: expected 4 or {scale_bytes} bytes, got {entries[sname]} bytes")
        packed_tensor_count += 1

    req("tok_embeddings.weight", vocab * hidden * hsz)
    req("norm.weight", hidden * hsz)
    if "norm.bias" in entries:
        expect("norm.bias", entries["norm.bias"], hidden * hsz)
    if "output.weight" in entries:
        req("output.weight", vocab * hidden * hsz)
    if "output.bias" in entries:
        expect("output.bias", entries["output.bias"], vocab * hsz)

    for layer in range(layers):
        p = f"layers.{layer}"
        req_fp16_or_packed_lowbit(f"{p}.attention_norm.weight", hidden * hsz, hidden, (hidden + 1) // 2)
        if f"{p}.attention_norm.bias" in entries:
            expect(f"{p}.attention_norm.bias", entries[f"{p}.attention_norm.bias"], hidden * hsz)
        req_fp16_or_packed_lowbit(
            f"{p}.attention.wq", q_hidden * hidden * hsz, q_hidden * hidden, q_hidden * ((hidden + 1) // 2)
        )
        req_fp16_or_packed_lowbit(
            f"{p}.attention.wk", kv_hidden * hidden * hsz, kv_hidden * hidden, kv_hidden * ((hidden + 1) // 2)
        )
        req_fp16_or_packed_lowbit(
            f"{p}.attention.wv", kv_hidden * hidden * hsz, kv_hidden * hidden, kv_hidden * ((hidden + 1) // 2)
        )
        req_fp16_or_packed_lowbit(
            f"{p}.attention.wo", hidden * q_hidden * hsz, hidden * q_hidden, hidden * ((q_hidden + 1) // 2)
        )
        if f"{p}.attention.bqkv" in entries:
            expect(f"{p}.attention.bqkv", entries[f"{p}.attention.bqkv"], (q_hidden + 2 * kv_hidden) * hsz)
        if f"{p}.attention.bo" in entries:
            expect(f"{p}.attention.bo", entries[f"{p}.attention.bo"], hidden * hsz)
        req_fp16_or_packed_lowbit(f"{p}.ffn_norm.weight", hidden * hsz, hidden, (hidden + 1) // 2)
        if f"{p}.ffn_norm.bias" in entries:
            expect(f"{p}.ffn_norm.bias", entries[f"{p}.ffn_norm.bias"], hidden * hsz)

        dense_w1 = f"{p}.feed_forward.w1"
        dense_w2 = f"{p}.feed_forward.w2"
        dense_w3 = f"{p}.feed_forward.w3"
        has_dense = any(
            n in entries or f"{n}.int8" in entries or f"{n}.int4" in entries
            for n in (dense_w1, dense_w2, dense_w3)
        )

        expert_pat = re.compile(rf"^{re.escape(p)}\.feed_forward\.experts\.(\d+)\.w([123])$")
        expert_ids = set()
        for n in entries.keys():
            m = expert_pat.match(n)
            if m:
                expert_ids.add(int(m.group(1)))
                continue
            if n.endswith(".int8") or n.endswith(".int4"):
                b = n.rsplit(".", 1)[0]
                m = expert_pat.match(b)
                if m:
                    expert_ids.add(int(m.group(1)))

        has_moe = bool(expert_ids) or f"{p}.feed_forward.router" in entries

        if has_dense and has_moe:
            raise ValueError(f"layer {layer}: mixed dense and MoE feed-forward tensors are not allowed")

        if has_moe:
            e_inter = expert_inter if expert_inter > 0 else inter
            req_fp16_or_packed_lowbit(
                f"{p}.feed_forward.router",
                num_local_experts * hidden * hsz if num_local_experts > 0 else len(expert_ids) * hidden * hsz,
                num_local_experts * hidden if num_local_experts > 0 else len(expert_ids) * hidden,
                (num_local_experts * ((hidden + 1) // 2)) if num_local_experts > 0 else len(expert_ids) * ((hidden + 1) // 2),
                (num_local_experts if num_local_experts > 0 else len(expert_ids)) * 4,
            )
            if not expert_ids:
                raise ValueError(f"layer {layer}: MoE layer has no experts")
            if num_local_experts > 0 and len(expert_ids) != num_local_experts:
                raise ValueError(
                    f"layer {layer}: expected {num_local_experts} experts from header, found {len(expert_ids)}"
                )
            for e in sorted(expert_ids):
                b = f"{p}.feed_forward.experts.{e}"
                req_fp16_or_packed_lowbit(
                    f"{b}.w1", e_inter * hidden * hsz, e_inter * hidden, e_inter * ((hidden + 1) // 2), e_inter * 4
                )
                req_fp16_or_packed_lowbit(
                    f"{b}.w2", hidden * e_inter * hsz, hidden * e_inter, hidden * ((e_inter + 1) // 2), hidden * 4
                )
                req_fp16_or_packed_lowbit(
                    f"{b}.w3", e_inter * hidden * hsz, e_inter * hidden, e_inter * ((hidden + 1) // 2), e_inter * 4
                )
        else:
            req_fp16_or_packed_lowbit(
                dense_w1, inter * hidden * hsz, inter * hidden, inter * ((hidden + 1) // 2), inter * 4
            )
            req_fp16_or_packed_lowbit(
                dense_w2, hidden * inter * hsz, hidden * inter, hidden * ((inter + 1) // 2), hidden * 4
            )
            req_fp16_or_packed_lowbit(
                dense_w3, inter * hidden * hsz, inter * hidden, inter * ((hidden + 1) // 2), inter * 4
            )

    # Optional TurboQuant metadata/tensors.
    has_tq3_codebook = "tq3_codebook" in entries
    has_tq3_signs = "tq3_signs_hidden" in entries
    if has_tq3_codebook != has_tq3_signs:
        raise ValueError("incomplete TQ3 metadata: expected both tq3_codebook and tq3_signs_hidden")
    if has_tq3_codebook:
        expect("tq3_codebook", entries["tq3_codebook"], 8 * hsz)
        expect("tq3_signs_hidden", entries["tq3_signs_hidden"], hidden)

    tq_objective = 0
    if "tq_objective" in entries:
        expect("tq_objective", entries["tq_objective"], 4)
        tq_objective = struct.unpack_from("<i", data, offsets["tq_objective"])[0]

    qjl_dim = 0
    if tq_objective == 1:
        for n in ("tq_qjl_dim", "tq_qjl_seed", "tq_qjl_indices_hidden", "tq_qjl_signs_hidden"):
            if n not in entries:
                raise ValueError(f"missing tensor: {n} (required for tq_objective=prod)")
        expect("tq_qjl_dim", entries["tq_qjl_dim"], 4)
        expect("tq_qjl_seed", entries["tq_qjl_seed"], 4)
        # Read qjl_dim scalar from tensor bytes.
        qjl_dim = struct.unpack_from("<i", data, offsets["tq_qjl_dim"])[0]
        if qjl_dim <= 0 or qjl_dim > hidden:
            raise ValueError(f"invalid tq_qjl_dim={qjl_dim}, expected in [1, {hidden}]")
        expect("tq_qjl_indices_hidden", entries["tq_qjl_indices_hidden"], qjl_dim * 4)
        expect("tq_qjl_signs_hidden", entries["tq_qjl_signs_hidden"], qjl_dim)

    tq3_words = (hidden + 9) // 10
    qjl_words = (qjl_dim + 31) // 32 if qjl_dim > 0 else 0

    def req_tq_pair_if_present(base: str, out_rows: int) -> None:
        packed = f"{base}.tq3"
        scales = f"{base}.tq3s"
        has_p = packed in entries
        has_s = scales in entries
        if not has_p and not has_s:
            return
        if has_p != has_s:
            raise ValueError(f"incomplete TQ3 tensor pair: expected both {packed} and {scales}")
        expect(packed, entries[packed], out_rows * tq3_words * 4)
        expect(scales, entries[scales], out_rows * hsz)

        if tq_objective == 1:
            rb = f"{base}.tq3r"
            rs = f"{base}.tq3rs"
            if rb not in entries or rs not in entries:
                raise ValueError(f"incomplete Qprod residual pair: expected both {rb} and {rs}")
            expect(rb, entries[rb], out_rows * qjl_words * 4)
            expect(rs, entries[rs], out_rows * hsz)

    for layer in range(layers):
        p = f"layers.{layer}"
        req_tq_pair_if_present(f"{p}.attention.wq", q_hidden)
        req_tq_pair_if_present(f"{p}.attention.wk", kv_hidden)
        req_tq_pair_if_present(f"{p}.attention.wv", kv_hidden)
        req_tq_pair_if_present(f"{p}.attention.wo", hidden)
        req_tq_pair_if_present(f"{p}.feed_forward.w1", inter)
        req_tq_pair_if_present(f"{p}.feed_forward.w3", inter)

    print("OK")
    print(
        f"version={version} vocab={vocab} hidden={hidden} inter={inter} expert_inter={expert_inter} "
        f"layers={layers} heads={heads} q_hidden={q_hidden} kv_heads={kv_heads} max_seq={max_seq} tensors={tensor_count} "
        f"packed_layer_tensors={packed_tensor_count} experts={num_local_experts} topk={num_experts_per_tok}"
    )


if __name__ == "__main__":
    main()
