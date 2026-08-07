#!/bin/bash
#SBATCH --job-name=dtu-ser-v3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v3.%J.out
#SBATCH --error=ser_v3.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER enhanced v3 - SLURM batch job (single seed)
# ============================================================
# v3 changes (vs v2):
# - All 12 wav2vec2 transformer layers unfrozen (was: last 4)
# - Layer-wise LR decay (ULMFiT-style): early layers get smaller LR
# - DualLoss = 0.5 * FocalLoss + 0.5 * LabelSmoothedCE
# - SpecAugment tuned: 96/128 dims, 3 masks each, p=0.6
# - 90 epochs cosine (was: 70)
# - Optional --seed argument for ensemble runs
#
# Usage (from HPC login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   sbatch scripts/train_ser_v3.sbatch 42
#   sbatch scripts/train_ser_v3.sbatch 43
#   sbatch scripts/train_ser_v3.sbatch 44
#
# Expected runtime: ~3 hours per seed on A100.
# 3 seeds in parallel (separate sbatch calls) = ~3 hours total wall-clock.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

SEED="${1:-42}"  # first arg, default 42

echo "============================================================"
echo "SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Seed: $SEED"
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

# Pull latest code
git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Confirm wav2vec2 model is cached
if [ ! -f ~/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/pytorch_model.bin ]; then
    echo "WARNING: wav2vec2-base model not in HF cache. Job may fail if compute node has no internet."
fi

# Run training with seed
uv run python -u train_ser_enhanced.py --seed "$SEED" 2>&1 | tee "train_ser_v3_seed${SEED}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
