#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./install_deps.sh

Installs repo-managed dependencies:
  1. Python packages from requirements.txt
  2. Web packages from web/package-lock.json (npm ci)
EOF
  exit 0
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$REPO_DIR/web"
REQ_FILE="$REPO_DIR/requirements.txt"

if [[ ! -f "$WEB_DIR/package.json" ]]; then
  echo "[install_deps] Could not find web/package.json." >&2
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "[install_deps] Could not find requirements.txt." >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[install_deps] Python was not found on PATH." >&2
  exit 1
fi

echo "[install_deps] Installing Python dependencies..."
"$PYTHON_BIN" -m pip install -r "$REQ_FILE"

cd "$WEB_DIR"
if [[ ! -f package-lock.json ]]; then
  echo "[install_deps] package-lock.json is missing, cannot run npm ci." >&2
  exit 1
fi
echo "[install_deps] Installing web dependencies with npm ci..."
npm ci
