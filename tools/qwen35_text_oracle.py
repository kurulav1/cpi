#!/usr/bin/env python3
"""Text-only logits oracle for Qwen3.5, the control the multimodal gate needs.

FIRST_STEP_LOGITS showed the Metal port's multimodal logits differ from HuggingFace's by
mean_abs 0.64 at the very first generated step. That is far too large to be fp16 noise, but it
does not say where: the tower, the splice, M-RoPE and the text stack are all upstream of it.

This isolates the text stack. Same model, same engine, no image: no tower, no splice, and 1-D
rope instead of M-RoPE. If the port's text-only logits match this, the defect is in the
multimodal-specific path. If they drift the same way, vision is a red herring and the text stack
is wrong, which nothing would currently catch, because Qwen3.5 has no text gate in src/tests.

    python tools/qwen35_text_oracle.py --model ~/models/qwen35-0.8b-hf --out ~/models/q35_text

Dumps last-position logits for a FIXED token sequence (no tokenizer dependency, so the C++ side
can hardcode the same ids and cannot disagree about tokenization).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

# Fixed ids, deliberately not produced by a tokenizer: the C++ side hardcodes the same list, so
# a tokenizer difference cannot masquerade as an engine difference. Long enough to cross several
# of the 24 layers' delta-net/full-attention alternation (18 linear + 6 full).
TOKENS = [3838, 374, 419, 30, 358, 1079, 264, 4128, 1614, 13, 6771, 752, 3291, 498]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.float32).eval()

    ids = list(TOKENS)
    logits = model(input_ids=torch.tensor([ids])).logits[0, -1].float().contiguous()
    (out_dir / "text_first_step_logits.f32").write_bytes(
        logits.numpy().astype("<f4").tobytes())

    # Greedy continuation too, by re-running the full forward each step; no KV cache, so a
    # cache bug on either side cannot quietly align the two.
    stream, cur = [], list(ids)
    for _ in range(args.steps):
        lg = model(input_ids=torch.tensor([cur])).logits[0, -1].float()
        nxt = int(torch.argmax(lg))
        stream.append(nxt)
        cur.append(nxt)

    manifest = {
        "input_ids": ids,
        "generated_ids": stream,
        "vocab": int(logits.numel()),
        "logits_file": "text_first_step_logits.f32",
        "note": "text-only control for FIRST_STEP_LOGITS; no image, no splice, 1-D rope",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print("[text-oracle] %d prompt tokens, vocab %d" % (len(ids), logits.numel()))
    print("[text-oracle] greedy: %s" % stream)
    print("[text-oracle] wrote %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
