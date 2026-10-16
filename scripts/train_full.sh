#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -x "$(cd "$ROOT/../../.." && pwd)/.venv/bin/python" ]]; then
  export PATH="$(cd "$ROOT/../../.." && pwd)/.venv/bin:$PATH"
fi
bash scripts/check_dataset_size.sh ../../datasets/celeba
bash scripts/train.sh configs/default.yaml
bash scripts/eval.sh configs/default.yaml
bash scripts/report.sh configs/default.yaml
