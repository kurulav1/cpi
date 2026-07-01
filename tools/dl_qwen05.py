"""One-off: download Qwen2.5-0.5B-Instruct (draft model for spec decoding of the 7B)."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
_ARTIFACTS = Path(os.environ.get("CPI_ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
OUT = str(_ARTIFACTS / "hub" / "Qwen__Qwen2.5-0.5B-Instruct" / "hf")

path = snapshot_download(
    repo_id=REPO,
    local_dir=OUT,
    allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"],
    max_workers=8,
)
print(f"DONE -> {path}", flush=True)
