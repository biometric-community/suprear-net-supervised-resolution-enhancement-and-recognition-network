#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -x "$(cd "$ROOT/../../.." && pwd)/.venv/bin/python" ]]; then
  export PATH="$(cd "$ROOT/../../.." && pwd)/.venv/bin:$PATH"
fi
CFG="${1:-configs/default.yaml}"
shift || true
python -m suprear.train --config "$CFG" "$@"
