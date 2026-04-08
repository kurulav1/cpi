#!/usr/bin/env python3
"""
Add packed streaming low-bit tensors to an existing .ll2c model.

This tool keeps all original tensors and appends generated:
  - .int8 + .scale  (mode=int8)
  - .int4 + .scale  (mode=int4)
for MLP tensors:
  layers.{i}.feed_forward.w1/w2/w3

Output is a new .ll2c file.
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    print(json.dumps({"type": "error", "msg": f"numpy is required: {exc}"}), flush=True)
    sys.exit(1)

MAGIC = b"LL2CUDA\x00"
HDR_V1 = struct.Struct("<8siiiiiiiiiQ")
HDR_V2 = struct.Struct("<8siiiiiiiiiiQ")
HDR_V3 = struct.Struct("<8siiiiiiiiiiQffiii")
HDR_V4 = struct.Struct("<8siiiiiiiiiiQffiiiiii")
ENTRY = struct.Struct("<64sqq")


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def fail(msg: str) -> None:
    emit({"type": "error", "msg": msg})
    emit({"type": "status", "status": "error"})
    sys.exit(1)


def pad_name(name: str) -> bytes:
    raw = name.encode("utf-8")
    if len(raw) >= 64:
        raise ValueError(f"tensor name too long: {name}")
    return raw + b"\x00" * (64 - len(raw))


def decode_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8")


def parse_header(buf: bytes) -> dict:
    if len(buf) < HDR_V1.size:
        raise ValueError("file too small for LL2C header")
    version = struct.unpack_from("<i", buf, 8)[0]
    if version >= 4:
        fields = HDR_V4.unpack_from(buf, 0)
        return {
            "version": version,
            "struct": HDR_V4,
            "fields": list(fields),
            "header_size": HDR_V4.size,
            "hidden": int(fields[3]),
            "inter": int(fields[4]),
            "layers": int(fields[5]),
            "tensor_count": int(fields[10]),
            "table_offset": int(fields[11]),
            "expert_inter": int(fields[19]),
        }
    if version >= 3:
        fields = HDR_V3.unpack_from(buf, 0)
        return {
            "version": version,
            "struct": HDR_V3,
            "fields": list(fields),
            "header_size": HDR_V3.size,
            "hidden": int(fields[3]),
            "inter": int(fields[4]),
            "layers": int(fields[5]),
            "tensor_count": int(fields[10]),
            "table_offset": int(fields[11]),
            "expert_inter": 0,
        }
    if version >= 2:
        fields = HDR_V2.unpack_from(buf, 0)
        return {
            "version": version,
            "struct": HDR_V2,
            "fields": list(fields),
            "header_size": HDR_V2.size,
            "hidden": int(fields[3]),
            "inter": int(fields[4]),
            "layers": int(fields[5]),
            "tensor_count": int(fields[10]),
            "table_offset": int(fields[11]),
            "expert_inter": 0,
        }
    fields = HDR_V1.unpack_from(buf, 0)
    return {
        "version": version,
        "struct": HDR_V1,
        "fields": list(fields),
        "header_size": HDR_V1.size,
        "hidden": int(fields[3]),
        "inter": int(fields[4]),
        "layers": int(fields[5]),
        "tensor_count": int(fields[9]),
        "table_offset": int(fields[10]),
        "expert_inter": 0,
    }


def repack_header(header: dict, tensor_count: int, table_offset: int) -> bytes:
    fields = list(header["fields"])
    if header["version"] >= 2:
        fields[10] = int(tensor_count)
        fields[11] = int(table_offset)
    else:
        fields[9] = int(tensor_count)
        fields[10] = int(table_offset)
    return header["struct"].pack(*fields)


def is_packable_base(name: str) -> bool:
    if re.fullmatch(r"layers\.\d+\.feed_forward\.w[123]", name):
        return True
    if re.fullmatch(r"layers\.\d+\.feed_forward\.experts\.\d+\.w[123]", name):
        return True
    return False


def infer_shape(base: str, hidden: int, inter: int, expert_inter: int) -> tuple[int, int]:
    e_inter = expert_inter if expert_inter > 0 else inter
    if re.fullmatch(r"layers\.\d+\.feed_forward\.w[13]", base):
        return inter, hidden
    if re.fullmatch(r"layers\.\d+\.feed_forward\.w2", base):
        return hidden, inter
    if re.fullmatch(r"layers\.\d+\.feed_forward\.experts\.\d+\.w[13]", base):
        return e_inter, hidden
    if re.fullmatch(r"layers\.\d+\.feed_forward\.experts\.\d+\.w2", base):
        return hidden, e_inter
    raise ValueError(f"unsupported MLP tensor: {base}")


def quantize_int8(mm: mmap.mmap, off: int, rows: int, cols: int) -> tuple[bytes, bytes]:
    values = np.frombuffer(mm, dtype=np.float16, count=rows * cols, offset=off)
    f32 = values.astype(np.float32).reshape(rows, cols)
    max_abs = np.max(np.abs(f32), axis=1)
    scales = np.maximum(max_abs / 127.0, 1.0e-8).astype(np.float32)
    q = np.rint(np.clip(f32 / scales[:, None], -127.0, 127.0)).astype(np.int8)
    return q.tobytes(), scales.tobytes()


def quantize_int4(mm: mmap.mmap, off: int, rows: int, cols: int) -> tuple[bytes, bytes]:
    values = np.frombuffer(mm, dtype=np.float16, count=rows * cols, offset=off)
    f32 = values.astype(np.float32).reshape(rows, cols)
    max_abs = np.max(np.abs(f32), axis=1)
    scales = np.maximum(max_abs / 7.0, 1.0e-8).astype(np.float32)
    q = np.rint(np.clip(f32 / scales[:, None], -7.0, 7.0)).astype(np.int8)

    packed_cols = (cols + 1) // 2
    packed = np.zeros((rows, packed_cols), dtype=np.uint8)
    even = q[:, 0::2].astype(np.int16)
    odd = np.zeros_like(even, dtype=np.int16)
    odd_src = q[:, 1::2].astype(np.int16)
    odd[:, : odd_src.shape[1]] = odd_src
    packed[:, :] = ((even & 0x0F) | ((odd & 0x0F) << 4)).astype(np.uint8)
    return packed.tobytes(), scales.tobytes()


def copy_bytes(mm: mmap.mmap, dst, off: int, nbytes: int, chunk: int = 16 * 1024 * 1024) -> None:
    view = memoryview(mm)
    start = off
    end = off + nbytes
    while start < end:
        stop = min(start + chunk, end)
        dst.write(view[start:stop])
        start = stop


def main() -> None:
    ap = argparse.ArgumentParser(description="Add packed int8/int4 streaming tensors to a .ll2c file")
    ap.add_argument("--input", required=True, help="input .ll2c path")
    ap.add_argument("--output", default="", help="output .ll2c path (required unless --inplace)")
    ap.add_argument("--mode", choices=["int8", "int4"], required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--inplace", action="store_true", help="append tensors and rewrite table in input file")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()

    if not in_path.exists():
        fail(f"input not found: {in_path}")

    if args.inplace:
        if args.output:
            out_path = Path(args.output).resolve()
            if out_path != in_path:
                fail("--inplace requires --output to match --input (or omit --output)")
        out_path = in_path
    else:
        if not args.output:
            fail("--output is required unless --inplace is provided")
        out_path = Path(args.output).resolve()
        if in_path == out_path:
            fail("input and output must be different files (or use --inplace)")
        if out_path.exists() and not args.overwrite:
            fail(f"output exists: {out_path} (pass --overwrite to replace)")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    open_mode = "r+b" if args.inplace else "rb"
    with in_path.open(open_mode) as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            try:
                header = parse_header(mm)
            except Exception as exc:
                fail(f"invalid ll2c header: {exc}")
                return

            if mm[:8] != MAGIC:
                fail("invalid ll2c magic")

            hidden = header["hidden"]
            inter = header["inter"]
            layers = header["layers"]
            tensor_count = header["tensor_count"]
            table_offset = header["table_offset"]

            emit({
                "type": "log",
                "msg": f"model hidden={hidden} inter={inter} layers={layers} tensors={tensor_count}"
            })

            entries = []
            names = set()
            for i in range(tensor_count):
                base = table_offset + i * ENTRY.size
                if base + ENTRY.size > mm.size():
                    fail(f"tensor table out of bounds at index {i}")
                raw_name, off, nbytes = ENTRY.unpack_from(mm, base)
                name = decode_name(raw_name)
                if off < 0 or nbytes < 0 or off + nbytes > mm.size():
                    fail(f"invalid tensor span: {name}")
                entries.append({"name": name, "off": int(off), "nbytes": int(nbytes)})
                names.add(name)
            entry_by_name = {entry["name"]: entry for entry in entries}

            base_names = sorted(name for name in names if is_packable_base(name))
            if not base_names:
                fail("input does not contain any quantizable feed-forward tensors")

            gen_specs = []
            add_count = 0
            expert_inter = int(header.get("expert_inter", 0) or 0)
            for base_name in base_names:
                if base_name not in entry_by_name:
                    continue
                qname = f"{base_name}.{args.mode}"
                sname = f"{base_name}.scale"
                if qname in names and sname in names:
                    continue
                rows, cols = infer_shape(base_name, hidden, inter, expert_inter)
                qbytes = rows * cols if args.mode == "int8" else rows * ((cols + 1) // 2)
                sbytes = rows * 4
                gen_specs.append(
                    {
                        "base": base_name,
                        "rows": rows,
                        "cols": cols,
                        "qname": qname,
                        "sname": sname,
                        "qbytes": qbytes,
                        "sbytes": sbytes,
                    }
                )
                add_count += 2

            if add_count == 0:
                emit({"type": "log", "msg": f"nothing to do: {args.mode} tensors already present"})
                emit({"type": "done", "path": str(out_path), "copied": not args.inplace, "added_tensors": 0})
                emit({"type": "status", "status": "done"})
                if not args.inplace and out_path != in_path:
                    with out_path.open("wb") as dst:
                        copy_bytes(mm, dst, 0, mm.size())
                return

            if args.inplace:
                emit({"type": "log", "msg": f"appending {add_count} tensors in-place to {in_path.name}..."})
                total_gen = len(gen_specs)
                done_gen = 0
                generated_entries = []

                # Keep all original tensor spans; append generated low-bit tensors at EOF.
                f.seek(0, 2)
                for spec in gen_specs:
                    base_name = spec["base"]
                    src = entry_by_name[base_name]
                    if args.mode == "int8":
                        qblob, sblob = quantize_int8(mm, src["off"], spec["rows"], spec["cols"])
                    else:
                        qblob, sblob = quantize_int4(mm, src["off"], spec["rows"], spec["cols"])

                    qoff = f.tell()
                    f.write(qblob)
                    generated_entries.append((spec["qname"], qoff, spec["qbytes"]))

                    soff = f.tell()
                    f.write(sblob)
                    generated_entries.append((spec["sname"], soff, spec["sbytes"]))

                    done_gen += 1
                    emit(
                        {
                            "type": "progress",
                            "done": done_gen,
                            "total": total_gen,
                            "pct": round((100.0 * done_gen) / max(1, total_gen), 1),
                            "tensor": base_name,
                        }
                    )

                new_table_offset = f.tell()
                all_entries = [(e["name"], e["off"], e["nbytes"]) for e in entries] + generated_entries
                for name, off, nbytes in all_entries:
                    f.write(ENTRY.pack(pad_name(name), off, nbytes))

                out_header = repack_header(header, len(all_entries), new_table_offset)
                f.seek(0)
                f.write(out_header)
                f.flush()

                emit(
                    {
                        "type": "done",
                        "path": str(out_path),
                        "copied": False,
                        "added_tensors": add_count,
                        "mode": args.mode,
                        "inplace": True,
                    }
                )
                emit({"type": "status", "status": "done"})
                return

            # Preserve existing entries order and append generated q/scale pairs.
            out_entries = [{"name": e["name"], "kind": "copy", "off": e["off"], "nbytes": e["nbytes"]} for e in entries]
            for spec in gen_specs:
                out_entries.append({"name": spec["qname"], "kind": "gen_q", "spec": spec, "nbytes": spec["qbytes"]})
                out_entries.append({"name": spec["sname"], "kind": "gen_s", "spec": spec, "nbytes": spec["sbytes"]})

            header_size = header["header_size"]
            new_table_offset = header_size
            new_tensor_count = len(out_entries)
            data_cursor = new_table_offset + new_tensor_count * ENTRY.size

            packed_entries = []
            for item in out_entries:
                packed_entries.append((pad_name(item["name"]), data_cursor, item["nbytes"], item))
                data_cursor += item["nbytes"]

            out_header = repack_header(header, new_tensor_count, new_table_offset)

            emit({"type": "log", "msg": f"writing {out_path.name} (adding {add_count} tensors)..."})
            total_gen = len(gen_specs)
            done_gen = 0
            pending_scales: dict[str, bytes] = {}

            with out_path.open("wb") as out_f:
                out_f.write(out_header)
                for name_raw, off, nbytes, _item in packed_entries:
                    out_f.write(ENTRY.pack(name_raw, off, nbytes))

                for _name_raw, _off, _nbytes, item in packed_entries:
                    if item["kind"] == "copy":
                        copy_bytes(mm, out_f, item["off"], item["nbytes"])
                        continue

                    spec = item["spec"]
                    base_name = spec["base"]
                    src = entry_by_name[base_name]
                    if item["kind"] == "gen_q":
                        if args.mode == "int8":
                            qblob, sblob = quantize_int8(mm, src["off"], spec["rows"], spec["cols"])
                        else:
                            qblob, sblob = quantize_int4(mm, src["off"], spec["rows"], spec["cols"])
                        out_f.write(qblob)
                        pending_scales[base_name] = sblob
                    else:
                        sblob = pending_scales.pop(base_name, None)
                        if sblob is None:
                            if args.mode == "int8":
                                _qblob, sblob = quantize_int8(mm, src["off"], spec["rows"], spec["cols"])
                            else:
                                _qblob, sblob = quantize_int4(mm, src["off"], spec["rows"], spec["cols"])
                        out_f.write(sblob)
                        done_gen += 1
                        emit(
                            {
                                "type": "progress",
                                "done": done_gen,
                                "total": total_gen,
                                "pct": round((100.0 * done_gen) / max(1, total_gen), 1),
                                "tensor": base_name,
                            }
                        )

            emit(
                {
                    "type": "done",
                    "path": str(out_path),
                    "copied": False,
                    "added_tensors": add_count,
                    "mode": args.mode,
                }
            )
            emit({"type": "status", "status": "done"})


if __name__ == "__main__":
    main()
