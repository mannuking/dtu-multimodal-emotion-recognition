#!/usr/bin/env bash
# train_all.sh — uv-native training pipeline for the DTU multimodal pipeline.
# Run via:
#   ./scripts/train_all.sh                # full pipeline
#   ./scripts/train_all.sh --quick        # 3-epoch smoke test
#
# Or just: uv run python uv_run_all.py --quick
#
# Prereqs: uv installed; CUDA-capable GPU; combined_ser_dataset/ extracted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Run via uv (creates .venv if needed, installs from uv.lock)
exec uv run python uv_run_all.py "$@"