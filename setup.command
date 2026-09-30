#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ then run this file again."
  exit 1
fi
uv sync --frozen
if [[ "${1:-}" == "--with-qwen" ]]; then
  .venv/bin/python scripts/download_model.py
fi
echo "Setup complete. Train a Winward checkpoint using README.md, then open Start JEV.command."
echo "The optional Qwen demo can be installed with: .venv/bin/python scripts/download_model.py"
