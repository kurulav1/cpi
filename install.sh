#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./install.sh

Prepares the repo for first use:
  1. Installs Python dependencies from requirements.txt
  2. Installs web dependencies
  3. Creates web/.env if missing
  4. Creates web/config.json if missing
  5. Builds build/llama_infer if missing
EOF
  exit 0
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$REPO_DIR/web"
BUILD_DIR="$REPO_DIR/build"
INFER_BIN="$BUILD_DIR/llama_infer"

if [[ ! -f "$WEB_DIR/package.json" ]]; then
  echo "[install] Could not find web/package.json." >&2
  exit 1
fi

echo "[install] Installing Python and web dependencies..."
"$REPO_DIR/install_deps.sh"

if [[ ! -f "$WEB_DIR/.env" && -f "$WEB_DIR/.env.example" ]]; then
  echo "[install] Creating web/.env from web/.env.example"
  cp "$WEB_DIR/.env.example" "$WEB_DIR/.env"
fi

if [[ ! -f "$WEB_DIR/config.json" && -f "$WEB_DIR/config.example.json" ]]; then
  echo "[install] Creating web/config.json from web/config.example.json"
  cp "$WEB_DIR/config.example.json" "$WEB_DIR/config.json"
fi

if [[ ! -x "$INFER_BIN" ]]; then
  echo "[install] Building llama_infer..."
  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$REPO_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  fi
  cmake --build "$BUILD_DIR" --target llama_infer -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

echo "[install] Repo is prepared."
echo "[install] Next steps:"
echo "[install]   1. Put or convert models into artifacts/"
echo "[install]   2. Start dev UI with ./start_web.sh"
echo "[install]   3. Or run the local packaged app with ./start_local.sh"
