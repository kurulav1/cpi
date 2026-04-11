#!/usr/bin/env python3
"""
Small CLI wrapper around Python `sentencepiece` for encode/decode using a
SentencePiece `.model` tokenizer. Used by the C++ runtime when the native
SentencePiece library is not compiled in and `spm_encode`/`spm_decode` are not
available.
"""

import argparse
from pathlib import Path

import sentencepiece as spm


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_processor(model: Path) -> spm.SentencePieceProcessor:
    proc = spm.SentencePieceProcessor()
    proc.load(str(model))
    return proc


def cmd_encode(args: argparse.Namespace) -> None:
    proc = load_processor(args.model)
    text = read_text(args.input)
    ids = list(proc.encode(text, out_type=int))
    if args.add_bos:
        bos_id = proc.bos_id()
        if bos_id >= 0 and (not ids or ids[0] != bos_id):
            ids = [bos_id] + ids
    write_text(args.output, " ".join(str(x) for x in ids))


def cmd_decode(args: argparse.Namespace) -> None:
    proc = load_processor(args.model)
    raw = read_text(args.input).strip()
    ids = [int(x) for x in raw.split()] if raw else []
    text = proc.decode(ids)
    write_text(args.output, text)


def cmd_special_ids(args: argparse.Namespace) -> None:
    proc = load_processor(args.model)
    bos_id = int(proc.bos_id())
    eos_id = int(proc.eos_id())
    unk_id = int(proc.unk_id())
    special = sorted({x for x in (bos_id, eos_id, unk_id) if x >= 0})
    lines = [
        f"bos_id {bos_id}",
        f"eos_id {eos_id}",
        f"unk_id {unk_id}",
        "special_ids " + " ".join(str(x) for x in special),
    ]
    write_text(args.output, "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("--model", type=Path, required=True)
    enc.add_argument("--input", type=Path, required=True)
    enc.add_argument("--output", type=Path, required=True)
    enc.add_argument("--add-bos", action="store_true")
    enc.set_defaults(fn=cmd_encode)

    dec = sub.add_parser("decode")
    dec.add_argument("--model", type=Path, required=True)
    dec.add_argument("--input", type=Path, required=True)
    dec.add_argument("--output", type=Path, required=True)
    dec.set_defaults(fn=cmd_decode)

    sid = sub.add_parser("special-ids")
    sid.add_argument("--model", type=Path, required=True)
    sid.add_argument("--output", type=Path, required=True)
    sid.set_defaults(fn=cmd_special_ids)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
