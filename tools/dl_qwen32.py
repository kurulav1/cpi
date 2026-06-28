"""One-off: download Qwen2.5-Coder-32B-Instruct weights + tokenizer only.

Skips the redundant .pth originals and any GGUF; we only need safetensors and
the tokenizer/config files for the convert->pack pipeline.
"""
import sys
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-Coder-32B-Instruct"
OUT = r"c:\Users\Väinö\Downloads\cpi\artifacts\hub\Qwen__Qwen2.5-Coder-32B-Instruct\hf"

path = snapshot_download(
    repo_id=REPO,
    local_dir=OUT,
    allow_patterns=[
        "*.safetensors",
        "*.json",
        "merges.txt",
        "vocab.json",
    ],
    max_workers=8,
)
print(f"DONE -> {path}", flush=True)
