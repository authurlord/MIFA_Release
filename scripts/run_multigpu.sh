#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
python scripts/run_multigpu_replicas.py \
  --datasets "${1:-all}" --mechanism "${2:-hce}" \
  --injected-root "data/injected/${2:-hce}"
