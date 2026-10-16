#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONO="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"
"$MONO/.venv/bin/python" -m pip install -r requirements.txt
echo OK
