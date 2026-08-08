#!/bin/bash
#SBATCH --job-name=dtu-ser-v5-ens
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=ser_v5_ens.%J.out
#SBATCH --error=ser_v5_ens.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v5 - Submit 3 seeds in parallel + auto-run ensemble eval
# ============================================================
# ~24 hours wall-clock total (each seed ~8h in parallel on separate nodes).
# Then runs ensemble_evaluate_v5.py automatically.
#
# Usage (from HPC login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   bash scripts/train_ser_v5_ensemble.sbatch
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "v5 ensemble orchestrator started: $(date)"
echo "Submitting 3 parallel jobs (seeds 42/43/44)"
echo "============================================================"

JOB42=$(sbatch scripts/train_ser_v5.sbatch 42 | awk '{print $4}')
JOB43=$(sbatch scripts/train_ser_v5.sbatch 43 | awk '{print $4}')
JOB44=$(sbatch scripts/train_ser_v5.sbatch 44 | awk '{print $4}')

echo "Submitted:"
echo "  seed 42 -> job $JOB42"
echo "  seed 43 -> job $JOB43"
echo "  seed 44 -> job $JOB44"

cat > ser_v5_ensemble_jobs.txt <<EOF
seed42=$JOB42
seed43=$JOB43
seed44=$JOB44
submitted_at=$(date -Iseconds)
EOF
echo "Saved JOBIDs to ser_v5_ensemble_jobs.txt"

# Wait for all three
while true; do
    PENDING=$(squeue -j "$JOB42,$JOB43,$JOB44" -h 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PENDING" -eq 0 ]; then
        break
    fi
    sleep 120
done

echo "All three v5 jobs finished at $(date)"
echo "============================================================"
echo "Running ensemble averaging..."
echo "============================================================"

uv run python ensemble_evaluate_v5.py --seeds 42 43 44 --n-tta 8

echo "============================================================"
echo "Ensemble complete: $(date)"
echo "Summary: model_checkpoints/ser_v5_ensemble_summary.json"
echo "============================================================"