#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_release.py --datasets "${1:-all}" --mechanism mcar --export-only --export-injected-root data/injected/mcar

