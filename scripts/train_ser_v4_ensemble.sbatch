#!/bin/bash
#SBATCH --job-name=dtu-ser-v4-ens
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=ser_v4_ens.%J.out
#SBATCH --error=ser_v4_ens.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v4 — Ensemble orchestrator
# ============================================================
# Submits three v4 jobs (seeds 42/43/44) in parallel on separate
# A100 nodes (~5-6h wall-clock), then runs ensemble_evaluate_v4.
#
# Usage (from HPC login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   bash scripts/train_ser_v4_ensemble.sbatch
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "v4 ensemble orchestrator started: $(date)"
echo "Submitting 3 parallel jobs (seeds 42/43/44)"
echo "============================================================"

JOB42=$(sbatch scripts/train_ser_v4.sbatch 42 | awk '{print $4}')
JOB43=$(sbatch scripts/train_ser_v4.sbatch 43 | awk '{print $4}')
JOB44=$(sbatch scripts/train_ser_v4.sbatch 44 | awk '{print $4}')

echo "Submitted:"
echo "  seed 42 -> job $JOB42"
echo "  seed 43 -> job $JOB43"
echo "  seed 44 -> job $JOB44"

cat > ser_v4_ensemble_jobs.txt <<EOF
seed42=$JOB42
seed43=$JOB43
seed44=$JOB44
submitted_at=$(date -Iseconds)
EOF

echo "Saved job IDs to ser_v4_ensemble_jobs.txt"
echo "============================================================"
echo "Waiting for all three to finish..."
echo "Tail individual logs with:"
echo "  tail -f ser_v4.$JOB42.out"
echo "============================================================"

# Poll until all three jobs reach terminal state
while true; do
    PENDING=$(squeue -j "$JOB42,$JOB43,$JOB44" -h 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PENDING" -eq 0 ]; then
        break
    fi
    sleep 120
done

echo "All three v4 jobs finished at $(date)"
echo "============================================================"

# Run ensemble averaging (this is inference, also via sbatch)
EVAL_JOB=$(sbatch scripts/eval_ensemble_v4.sbatch | awk '{print $4}')
echo "Submitted ensemble eval: job $EVAL_JOB"

# Wait for eval to finish too
while true; do
    PENDING=$(squeue -j "$EVAL_JOB" -h 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PENDING" -eq 0 ]; then
        break
    fi
    sleep 30
done

echo "============================================================"
echo "Ensemble complete: $(date)"
echo "Summary: model_checkpoints/ser_v4_ensemble_summary.json"
echo "============================================================"