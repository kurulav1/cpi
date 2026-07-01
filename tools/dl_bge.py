"""One-off: download BAAI/bge-small-en-v1.5 (384-dim retrieval embedding model)."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "BAAI/bge-small-en-v1.5"
# Download into <repo>/artifacts/hub/... — derived from this file's location
# (tools/), not a hardcoded path. Override the artifacts root with CPI_ARTIFACTS_DIR.
_ARTIFACTS = Path(os.environ.get("CPI_ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
OUT = str(_ARTIFACTS / "hub" / "BAAI__bge-small-en-v1.5")

path = snapshot_download(
    repo_id=REPO,
    local_dir=OUT,
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "model.safetensors",
        "1_Pooling/config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
    ],
    max_workers=8,
)
print(f"DONE -> {path}", flush=True)
