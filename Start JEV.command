#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Please run setup.command first."
  exit 1
fi
exec .venv/bin/python scripts/server.py start
