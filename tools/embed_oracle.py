#!/usr/bin/env python3
"""Reference embeddings for a BERT-family encoder, to gate cpi_embed.

cpi_embed produces 384 plausible-looking floats whether or not the encoder is right, so the only
thing that separates a working port from a broken one is a reference. This runs the same model
through HuggingFace in fp32, mean-pools over the attention mask and L2-normalises; which is
exactly what sentence-transformers' all-MiniLM-L6-v2 does, and what BertEmbedder/MetalBertEmbedder
implement.

    python tools/embed_oracle.py --model ~/models/all-MiniLM-L6-v2 --out ~/models/embed_oracle

Dumps, per sentence: HF's token ids (so a tokenizer disagreement can be told apart from an
encoder disagreement) and the final normalised vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Fixed and deliberately varied: two near-synonyms (a working encoder must rank them close), one
# unrelated technical phrase (must rank far), one short and one longer sentence so the pooling
# divisor and position embeddings are both exercised at different lengths.
SENTENCES = [
    "a photo of a cat",
    "a photo of a dog",
    "quantum chromodynamics",
    "hello",
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).eval()

    vectors, records = [], []
    for s in SENTENCES:
        enc = tok(s, return_tensors="pt")
        out = model(**enc).last_hidden_state[0]              # [L, H]
        mask = enc["attention_mask"][0].unsqueeze(-1).float()
        pooled = (out * mask).sum(0) / mask.sum(0)            # mean pool over real tokens
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
        vectors.append(pooled)
        records.append({"text": s, "token_ids": enc["input_ids"][0].tolist(),
                        "n_tokens": int(enc["attention_mask"][0].sum())})

    mat = torch.stack(vectors)                                # [N, H]
    (out_dir / "embeddings.f32").write_bytes(mat.numpy().astype("<f4").tobytes())

    # Pairwise cosine similarity, which is what the embedding is FOR. A port can match the
    # reference vectors poorly and still rank correctly, or match well and rank wrongly; both
    # numbers are worth having.
    sims = (mat @ mat.T).tolist()

    manifest = {
        "model": str(model_dir),
        "dim": int(mat.shape[1]),
        "count": int(mat.shape[0]),
        "sentences": records,
        "cosine": sims,
        "file": "embeddings.f32",
        "note": "fp32 HF reference; mean pooling over the attention mask, then L2 normalise",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print("[embed-oracle] %d sentences, dim %d" % (mat.shape[0], mat.shape[1]))
    for i, r in enumerate(records):
        print("  %-72s %2d tokens" % ('"' + r["text"] + '"', r["n_tokens"]))
    print("[embed-oracle] cos(cat,dog)=%.4f  cos(cat,qcd)=%.4f" % (sims[0][1], sims[0][2]))
    print("[embed-oracle] wrote %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
