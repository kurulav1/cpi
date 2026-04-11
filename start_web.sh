#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./start_web.sh

Starts the web dev stack:
  1. Copies web/.env from web/.env.example if needed
  2. Copies web/config.json from web/config.example.json if needed
  3. Installs dependencies if node_modules is missing
  4. Runs npm run dev (API + Vite UI)
EOF
  exit 0
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$REPO_DIR/web"
INFER_BIN="$REPO_DIR/build/llama_infer"

if [[ ! -f "$WEB_DIR/package.json" ]]; then
  echo "[start_web] Could not find web/package.json." >&2
  exit 1
fi

if [[ ! -f "$WEB_DIR/.env" ]]; then
  if [[ -f "$WEB_DIR/.env.example" ]]; then
    echo "[start_web] web/.env not found, copying from web/.env.example"
    cp "$WEB_DIR/.env.example" "$WEB_DIR/.env"
  else
    echo "[start_web] web/.env and web/.env.example are missing."
    echo "[start_web] Create web/.env manually if your setup requires it."
  fi
fi

if [[ ! -f "$WEB_DIR/config.json" ]]; then
  if [[ -f "$WEB_DIR/config.example.json" ]]; then
    echo "[start_web] web/config.json not found, copying from web/config.example.json"
    cp "$WEB_DIR/config.example.json" "$WEB_DIR/config.json"
    echo "[start_web] Edit web/config.json and set modelPath/tokenizerPath before generating."
  else
    echo "[start_web] web/config.json and web/config.example.json are missing."
  fi
fi

if [[ ! -x "$INFER_BIN" ]]; then
  echo "[start_web] Warning: $INFER_BIN not found."
  echo "[start_web] API will start, but inference requests will fail until llama_infer is built."
fi

cd "$WEB_DIR"

if [[ ! -d node_modules ]]; then
  if [[ ! -f package-lock.json ]]; then
    echo "[start_web] package-lock.json is missing, cannot run npm ci." >&2
    exit 1
  fi
  echo "[start_web] Installing web dependencies with npm ci..."
  npm ci
fi

echo "[start_web] Starting dev server:"
echo "[start_web] API: http://localhost:3001"
echo "[start_web] UI : http://localhost:5173"
exec npm run dev
