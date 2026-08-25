#!/usr/bin/env bash
# Portable launcher for macOS arm64 and DGX OS Linux aarch64.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${EMR_ANALYZER_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python non trovato. Attiva l'ambiente emr-analyzer o imposta EMR_ANALYZER_PYTHON." >&2
  exit 1
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/run.py"
