#!/usr/bin/env bash
# eval_all.sh — Re-runs the integrity verification on the trained checkpoints.
# Should be run AFTER train_all.sh completes.
#
# Outputs:
#   reports/ser_verification.json
#   reports/ser_verification_log.txt
#   reports/MODEL_INTEGRITY_TEST_REPORT.pdf (rebuilt if render_pdf.py exists)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_USE_LEGACY_KERAS=1

mkdir -p reports

echo "[eval] Running SER verification on held-out test split..."
python3 verify_ser.py | tee reports/ser_verification_log.txt

if [[ -f reports/render_pdf.py ]]; then
    echo "[eval] Rebuilding integrity report PDF..."
    python3 reports/render_pdf.py
fi

echo "[eval] Done. Artifacts:"
ls -la reports/