#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
python scripts/run_release.py --datasets "${1:-all}" --mechanism hce --export-only --export-injected-root data/injected/hce

