#!/bin/bash
#SBATCH --job-name=dtu-ser-ens-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=00:30:00
#SBATCH --output=ser_ens_eval.%J.out
#SBATCH --error=ser_ens_eval.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER — Ensemble evaluation (3-seed average, 5-pass TTA each)
# ============================================================
# This is INFERENCE on GPU, but we still go through sbatch to respect
# the "never compute on login node" rule.
#
# Assumes ser_best_seed{42,43,44}.pt already exist in model_checkpoints/.
#
# Usage (from HPC login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/eval_ensemble.sbatch
#   # Then:
#   tail -f ser_ens_eval.<JOBID>.out
#
# Expected runtime: ~5 min per seed × 3 seeds = ~15 min total.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Ensemble evaluation started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Verify all three checkpoints exist
for s in 42 43 44; do
    if [ ! -f "model_checkpoints/ser_best_seed${s}.pt" ]; then
        echo "ERROR: model_checkpoints/ser_best_seed${s}.pt missing"
        echo "Run scripts/train_ser_v2_ensemble.sbatch first, or sbatch scripts/train_ser_v2.sbatch <seed> 4"
        exit 1
    fi
done

echo "All 3 checkpoints present. Running ensemble_evaluate.py..."
uv run python ensemble_evaluate.py --seeds 42 43 44 --n-tta 5

echo "============================================================"
echo "Ensemble complete: $(date)"
echo "Summary saved to model_checkpoints/ser_ensemble_summary.json"
echo "============================================================"