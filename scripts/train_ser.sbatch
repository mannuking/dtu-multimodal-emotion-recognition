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
# Jai prefers interactive `uv run | tee` flow on the login node.
# This script is here for unattended/long runs only; not the
# default path. See memory: user dislikes sbatch pain.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Working dir: $(pwd)"
echo "============================================================"

git pull origin main || echo "  (git pull failed — continuing with local checkout)"

HF_HUB_OFFLINE=1 uv run python -u train_ser_enhanced.py 2>&1 | tee train_ser_enhanced.log

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
