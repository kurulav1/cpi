#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./start_local.sh

Starts the non-Docker package:
  1. Copies web/.env from web/.env.example if needed
  2. Builds build/llama_infer if missing
  3. Installs web dependencies if needed
  4. Builds the React UI
  5. Starts the local server on http://localhost:3001
EOF
  exit 0
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$REPO_DIR/web"
INFER_BIN="$REPO_DIR/build/llama_infer"

if [[ ! -f "$WEB_DIR/package.json" ]]; then
  echo "[start_local] Could not find web/package.json." >&2
  exit 1
fi

if [[ ! -f "$WEB_DIR/.env" ]]; then
  echo "[start_local] web/.env not found, copying from web/.env.example"
  cp "$WEB_DIR/.env.example" "$WEB_DIR/.env"
fi

if [[ ! -x "$INFER_BIN" ]]; then
  echo "[start_local] llama_infer is missing, building it now..."
  if [[ ! -f "$REPO_DIR/build/CMakeCache.txt" ]]; then
    cmake -S "$REPO_DIR" -B "$REPO_DIR/build" -DCMAKE_BUILD_TYPE=Release
  fi
  cmake --build "$REPO_DIR/build" --target llama_infer -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

cd "$WEB_DIR"

if [[ ! -d node_modules ]]; then
  echo "[start_local] Installing web dependencies..."
  npm install
fi

echo "[start_local] Building web UI..."
npm run build

echo "[start_local] Starting local package on http://localhost:3001"
echo "[start_local] The API launches llama_infer on demand for each chat request."
exec node server/index.mjs
