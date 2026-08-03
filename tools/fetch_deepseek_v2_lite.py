#!/usr/bin/env python
"""Download DeepSeek-V2-Lite (the smallest native-MLA DeepSeek) into artifacts/hub for MLA TDD.

Reads the HF token from .env (never takes it on the command line). Pulls only the weights + configs +
tokenizer (safetensors, not any duplicate .bin/gguf). Idempotent -- resumes / skips already-present
files, so it is safe to re-run.
"""
import os
import re
import sys

REPO = "deepseek-ai/DeepSeek-V2-Lite"
DEST = os.path.join("artifacts", "hub", "deepseek-ai__DeepSeek-V2-Lite", "hf")


def load_token():
    for line in open(".env", encoding="utf-8", errors="ignore"):
        m = re.match(r'\s*(HF_TOKEN|HUGGINGFACE\w*)\s*=\s*["\']?([^"\'\r\n]+)', line, re.I)
        if m:
            return m.group(2).strip()
    return None


def main():
    tok = load_token()
    if not tok:
        print("no HF token found in .env", file=sys.stderr)
        return 1
    os.makedirs(DEST, exist_ok=True)
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO,
        local_dir=DEST,
        token=tok,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"],
        max_workers=8,
    )
    print("DONE:", path)
    total = 0
    for root, _, files in os.walk(DEST):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"on disk: {total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
