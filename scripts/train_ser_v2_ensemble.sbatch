#!/bin/bash
#SBATCH --job-name=dtu-ser-v2-ens
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v2_ens.%J.out
#SBATCH --error=ser_v2_ens.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v2 — 3-seed ensemble (recommended way to reproduce + beat 68%)
# ============================================================
# Submits three v2 jobs (seeds 42/43/44, 4 unfrozen layers each) in parallel.
# Total wall-clock: ~3 hours (each job ~2.5h on a separate A100 node).
# Each saves its own checkpoint: ser_best_seed{42,43,44}.pt.
#
# After all three finish, run ensemble_evaluate.py:
#   uv run python ensemble_evaluate.py --seeds 42 43 44 --n-tta 5
#
# Expected result: ensemble test_acc 70-73% (vs 68.4% single seed).
#
# Usage (from HPC login node ONLY — never run compute on login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   bash scripts/train_ser_v2_ensemble.sbatch
#   squeue -u $USER
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "v2 ensemble orchestrator started: $(date)"
echo "Submitting 3 parallel jobs (seeds 42/43/44, 4 unfrozen layers)"
echo "============================================================"

# Submit three jobs. Each one is independent and lands on its own node.
JOB42=$(sbatch scripts/train_ser_v2.sbatch 42 4 | awk '{print $4}')
JOB43=$(sbatch scripts/train_ser_v2.sbatch 43 4 | awk '{print $4}')
JOB44=$(sbatch scripts/train_ser_v2.sbatch 44 4 | awk '{print $4}')

echo "Submitted:"
echo "  seed 42 -> job $JOB42"
echo "  seed 43 -> job $JOB43"
echo "  seed 44 -> job $JOB44"

# Save job IDs for later reference
cat > ser_v2_ensemble_jobs.txt <<EOF
seed42=$JOB42
seed43=$JOB43
seed44=$JOB44
submitted_at=$(date -Iseconds)
EOF

echo "Saved job IDs to ser_v2_ensemble_jobs.txt"
echo "============================================================"
echo "Waiting for all three to finish (max ~3h wall-clock)..."
echo "Tail individual logs with:"
echo "  tail -f ser_v2.$JOB42.out"
echo "  tail -f ser_v2.$JOB43.out"
echo "  tail -f ser_v2.$JOB44.out"
echo "============================================================"

# Wait for all three to reach a terminal state. squeue polling is safe on login.
while true; do
    PENDING=$(squeue -j "$JOB42,$JOB43,$JOB44" -h 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PENDING" -eq 0 ]; then
        break
    fi
    sleep 60
done

echo "All three v2 jobs finished at $(date)"
echo "============================================================"
echo "Running ensemble_evaluate.py to combine the 3 checkpoints..."
echo "============================================================"

uv run python ensemble_evaluate.py --seeds 42 43 44 --n-tta 5

echo "============================================================"
echo "Ensemble complete: $(date)"
echo "Summary: model_checkpoints/ser_ensemble_summary.json"
echo "============================================================"