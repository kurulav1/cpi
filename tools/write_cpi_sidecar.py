#!/usr/bin/env python3
"""Write a `cpi.json` sidecar (chat + reasoning descriptors) next to a model.

Descriptors SHIP with the model so the runtime carries no per-model knowledge; it just
reads them. A `.cpi` container carries them in its manifest (CFGJSON lines); a HuggingFace
safetensors directory needs this sidecar instead.

That gap was real: the Gemma 4 HF directories (the only container that still carries the
vision tower) had a chat descriptor but no reasoning descriptor, so the web UI reported
the vision-capable variant as unable to think and the .cpi variant as unable to see
the same model, split across two containers with half its capabilities each.

  python tools/write_cpi_sidecar.py --model artifacts/hub/google__gemma-4-E2B-it/hf
  python tools/write_cpi_sidecar.py --model <dir> --from-manifest path/to/model.manifest
"""

import argparse
import json
import os
import re

# Gemma 4. Thinking is enabled with a <|think|> system turn; the reasoning block is
# delimited by the SPECIAL tokens <|channel> … <channel|>, which the detokenizer must
# preserve (hence `markers`) so the stream splitter can find them.
GEMMA4_REASONING = {
    "mode": "optional",
    "enable": "<|turn>system\n<|think|><turn|>\n",
    "open": "<|channel>",
    "close": "<channel|>",
    "markers": ["<|channel>", "<channel|>"],
}
GEMMA4_CHAT = {
    "join": "",
    "addBos": True,
    "system": {"mode": "fold", "foldSeparator": "\n\n"},
    "user": {"prefix": "<|turn>user\n", "suffix": "<turn|>\n"},
    "assistant": {"prefix": "<|turn>model\n", "suffix": "<turn|>\n"},
    "generationPrompt": "<|turn>model\n",
}


def from_manifest(path):
    """Lift the descriptors straight out of a .cpi manifest, so the two containers can
    never drift apart."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^CFGJSON\s+(chat|reasoning)\s+(.+)$", line.strip())
            if m:
                out[m.group(1)] = json.loads(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model directory (or file) to sit beside")
    ap.add_argument("--from-manifest", default="", help="copy descriptors from a .cpi manifest")
    ap.add_argument("--family", default="gemma4", choices=["gemma4"])
    args = ap.parse_args()

    target = args.model if os.path.isdir(args.model) else os.path.dirname(args.model)
    desc = (
        from_manifest(args.from_manifest)
        if args.from_manifest
        else {"chat": GEMMA4_CHAT, "reasoning": GEMMA4_REASONING}
    )
    if "chat" not in desc or "reasoning" not in desc:
        raise SystemExit(f"missing descriptors: got {sorted(desc)}")

    path = os.path.join(target, "cpi.json")
    # Preserve anything already in the sidecar that we do not own.
    existing = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(desc)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    print(f"wrote {path}  ({', '.join(sorted(desc))})")


if __name__ == "__main__":
    main()
