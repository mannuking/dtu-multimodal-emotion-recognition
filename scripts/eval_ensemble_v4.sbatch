#!/bin/bash
#SBATCH --job-name=dtu-ser-v4-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=00:45:00
#SBATCH --output=ser_v4_eval.%J.out
#SBATCH --error=ser_v4_eval.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v4 — Ensemble evaluation (averages 3 v4 checkpoints)
# ============================================================
# This is INFERENCE on GPU, must go through sbatch (login-node rule).
# Assumes ser_v4_best_seed{42,43,44}.pt exist.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "v4 ensemble eval started: $(date)"
echo "JobID: $SLURM_JOB_ID"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

for s in 42 43 44; do
    if [ ! -f "model_checkpoints/ser_v4_best_seed${s}.pt" ]; then
        echo "ERROR: model_checkpoints/ser_v4_best_seed${s}.pt missing"
        exit 1
    fi
done

uv run python ensemble_evaluate_v4.py --seeds 42 43 44 --n-tta 5

echo "============================================================"
echo "v4 ensemble eval complete: $(date)"
echo "============================================================"