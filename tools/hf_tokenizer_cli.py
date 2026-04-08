#!/usr/bin/env python3
"""
Small CLI wrapper around Hugging Face `tokenizers` for encode/decode using tokenizer.json.
Used by the C++ runtime when SentencePiece is not the correct tokenizer backend.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def cmd_encode(args: argparse.Namespace) -> None:
    tok = Tokenizer.from_file(str(args.tokenizer_json))
    text = read_text(args.input)
    # We build chat prompts and BOS handling in C++.
    # Disable implicit tokenizer post-processing specials for deterministic IDs.
    out = tok.encode(text, add_special_tokens=args.add_special_tokens)
    ids = list(out.ids)
    if args.add_bos and (not ids or ids[0] != args.bos_id):
        ids = [args.bos_id] + ids
    write_text(args.output, " ".join(str(x) for x in ids))


def cmd_decode(args: argparse.Namespace) -> None:
    tok = Tokenizer.from_file(str(args.tokenizer_json))
    raw = read_text(args.input).strip()
    ids = [int(x) for x in raw.split()] if raw else []
    text = tok.decode(ids, skip_special_tokens=True)
    write_text(args.output, text)


def cmd_special_ids(args: argparse.Namespace) -> None:
    tok = Tokenizer.from_file(str(args.tokenizer_json))
    special = set()

    try:
        for idx, added in tok.get_added_tokens_decoder().items():
            if getattr(added, "special", False):
                special.add(int(idx))
    except Exception:
        pass

    def tok_id(piece: str) -> int:
        try:
            v = tok.token_to_id(piece)
            return -1 if v is None else int(v)
        except Exception:
            return -1

    bos_id = tok_id("<s>")
    eos_id = tok_id("</s>")
    unk_id = tok_id("<unk>")
    for x in (bos_id, eos_id, unk_id):
        if x >= 0:
            special.add(x)

    ids = sorted(special)
    lines = [
        f"bos_id {bos_id}",
        f"eos_id {eos_id}",
        f"unk_id {unk_id}",
        "special_ids " + " ".join(str(x) for x in ids),
    ]
    write_text(args.output, "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("--tokenizer-json", type=Path, required=True)
    enc.add_argument("--input", type=Path, required=True)
    enc.add_argument("--output", type=Path, required=True)
    enc.add_argument("--add-special-tokens", action="store_true")
    enc.add_argument("--add-bos", action="store_true")
    enc.add_argument("--bos-id", type=int, default=1)
    enc.set_defaults(fn=cmd_encode)

    dec = sub.add_parser("decode")
    dec.add_argument("--tokenizer-json", type=Path, required=True)
    dec.add_argument("--input", type=Path, required=True)
    dec.add_argument("--output", type=Path, required=True)
    dec.set_defaults(fn=cmd_decode)

    sid = sub.add_parser("special-ids")
    sid.add_argument("--tokenizer-json", type=Path, required=True)
    sid.add_argument("--output", type=Path, required=True)
    sid.set_defaults(fn=cmd_special_ids)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
