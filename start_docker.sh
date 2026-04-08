#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./start_docker.sh [models_dir] [image_tag]

models_dir defaults to ./models
image_tag defaults to cpi-chat-ui
EOF
  exit 0
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${1:-${MODELS_DIR:-$REPO_DIR/models}}"
IMAGE_TAG="${2:-${IMAGE_TAG:-cpi-chat-ui}}"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "[start_docker] Models directory does not exist: $MODELS_DIR" >&2
  exit 1
fi

echo "[start_docker] Building Docker image $IMAGE_TAG..."
docker build -t "$IMAGE_TAG" "$REPO_DIR"

echo "[start_docker] Starting container on http://localhost:3001"
echo "[start_docker] Mounting host models from $MODELS_DIR"
exec docker run --rm -it -p 3001:3001 --gpus all \
  -e LLAMA_MODEL_DIRS=/models \
  -v "$MODELS_DIR:/models:ro" \
  "$IMAGE_TAG"
