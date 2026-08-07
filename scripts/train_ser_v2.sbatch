#!/bin/bash
#SBATCH --job-name=dtu-ser-v2
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v2.%J.out
#SBATCH --error=ser_v2.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER enhanced v2 - SLURM batch job
# ============================================================
# 2026-08-07: switched from interactive login-node runs (HPC ops
# warned us) to proper SLURM submission. The job runs entirely on
# the compute node; you never log in to run anything.
#
# Usage (from HPC login node):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/train_ser_v2.sbatch
#
#   # Watch progress:
#   squeue -u $USER
#   tail -f ser_v2.<JOBID>.out
#
# Expected runtime: ~2-3 hours (70 epochs + live wav2vec2 fine-tune
# + TTA on test). 4-hour budget is safe.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

# Pull latest code (no-op if no remote changes)
git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Confirm wav2vec2 model is cached locally (avoids HF download on compute node)
if [ ! -f ~/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/pytorch_model.bin ]; then
    echo "WARNING: wav2vec2-base model not in HF cache. The job will try to download it."
    echo "If the compute node has no internet, the job will fail. Cancel and download on login node first."
fi

# Run training. NO HF_HUB_OFFLINE=1 - we want the model to load, but it should
# come from local cache. Only fails if cache is missing.
uv run python -u train_ser_enhanced.py 2>&1 | tee train_ser_v2.log

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
