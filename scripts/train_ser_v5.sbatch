#!/bin/bash
#SBATCH --job-name=dtu-ser-v5
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --output=ser_v5.%J.out
#SBATCH --error=ser_v5.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER enhanced v5 - Advanced pipeline (78-82% target)
# ============================================================
# Improvements over v4 (single-seed 0.6447 test acc):
#   - Last 8 transformer layers unfrozen (v4: 6)
#   - SupCon weight 1.0 (v4: 0.5)
#   - SpecAugment dialed back: time 64 / freq 128 / 2 masks / p=0.5
#   - Exponential Moving Average (EMA, decay 0.999) for inference
#   - Stochastic depth on conv blocks (p=0.1)
#   - 60 epochs with 10% warmup (v4: 50 epochs, 5% warmup)
#   - Higher head dropout 0.5 (v4: 0.4)
#   - 8-pass TTA on test (v4: 5)
#
# Expected runtime: ~7-8 hours per seed on 1x A100.
# 8-hour budget is safe.
#
# Usage (from HPC login, NEVER compute on login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/train_ser_v5.sbatch 42
#   sbatch scripts/train_ser_v5.sbatch 43
#   sbatch scripts/train_ser_v5.sbatch 44
#
# Then run ensemble:
#   sbatch scripts/eval_ensemble_v5.sbatch
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

SEED="${1:-42}"

echo "============================================================"
echo "v5 SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Seed: $SEED"
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Auto-download wav2vec2-large on compute node if needed
LARGE_PATH="$HOME/.cache/huggingface/hub/models--facebook--wav2vec2-large"
if [ ! -d "$LARGE_PATH" ]; then
    echo "wav2vec2-large not in HF cache. Downloading on compute node (~1.2 GB)..."
    uv run python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large')" || {
        echo "ERROR: download failed. Check compute-node outbound network."
        exit 1
    }
fi
echo "wav2vec2-large cache: OK"

uv run python -u train_ser_v5.py --seed "$SEED" 2>&1 | tee "train_ser_v5_seed${SEED}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"