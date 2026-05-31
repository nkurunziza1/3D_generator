#!/usr/bin/env bash
# Always use the project venv — do NOT pip install into Anaconda base.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/pip install -q -r requirements.txt
exec .venv/bin/uvicorn main:app --reload --host 127.0.0.1 --port 8000
