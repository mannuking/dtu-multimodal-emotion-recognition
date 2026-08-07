#!/bin/bash
#SBATCH --job-name=dtu-ser-train
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=ser_train.%J.out
#SBATCH --error=ser_train.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER enhanced training — SLURM wrapper
# ============================================================
# Survives SSH drops. Runs on a compute node with one A100.
# All output goes to ser_train.<JOBID>.{out,err} files you can
# tail from any shell session.
# Usage:
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   sbatch scripts/train_ser.sbatch
#   squeue -u $USER                      # see job id
#   tail -f ser_train.<JOBID>.out        # watch progress
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Working dir: $(pwd)"
echo "============================================================"

# Pull latest code (in case local edits weren't pushed)
git pull origin main || echo "  (git pull failed — continuing with local checkout)"

# Run with full visibility
HF_HUB_OFFLINE=1 uv run python -u train_ser_enhanced.py 2>&1 | tee train_ser_enhanced.log

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
