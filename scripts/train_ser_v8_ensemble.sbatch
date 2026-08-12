#!/bin/bash
#SBATCH --job-name=dtu-ser-v8-ens
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=ser_v8_ens.%A.out
#SBATCH --error=ser_v8_ens.%A.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v8 — ensemble orchestrator
# Submits 15 jobs (5 folds × 3 seeds) and runs the ensemble eval after.
# Usage: bash scripts/train_ser_v8_ensemble.sbatch
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

git pull origin main || echo "  (git pull failed - continuing)"

# Submit the 15 array jobs (5 folds × 3 seeds)
ARRAY_JOBID=$(sbatch --array=0-14 scripts/train_ser_v8.sbatch | awk '{print $4}')
echo "============================================================"
echo "v8 ensemble orchestrator: array job $ARRAY_JOBID (15 sub-jobs)"
echo "  Fold 0-4 × Seed 42,43,44"
echo "  Each ~6h on A100, parallel = ~6h wall-clock total"
echo "============================================================"

# Wait for all 15 to finish
echo "Waiting for all 15 jobs to complete..."
while true; do
    PENDING=$(squeue -j "$ARRAY_JOBID" -h 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PENDING" -eq 0 ]; then
        break
    fi
    sleep 180
    echo "  $(date '+%H:%M:%S'): $PENDING jobs still pending/running"
done

echo "All 15 jobs finished at $(date)"

# Run ensemble eval
uv run python ensemble_evaluate_v8.py

echo "============================================================"
echo "Ensemble eval complete: $(date)"
echo "============================================================"
