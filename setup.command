#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ then run this file again."
  exit 1
fi
uv sync --frozen
.venv/bin/python scripts/download_model.py
echo "Setup complete. Open Start JEV.command to launch."
