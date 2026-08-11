#!/bin/bash
#SBATCH --job-name=dtu-ser-v6-multimodal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --output=ser_v6.%J.out
#SBATCH --error=ser_v6.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v6 - Multimodal SER fusion (audio + text + facial)
# ============================================================
# Based on the WORKING v5 sbatch template (got 71% test accuracy).
# Key fix: --account=dtuarp-acc MUST be set — without it, sbatch returns
# "Memory specification can not be satisfied" because SLURM can't resolve
# the QoS against an unspecified account.
#
# Architecture: 1.58M trainable fusion head over frozen
#   - audio: wav2vec2-large (1024-dim)
#   - text: MobileBERT (768-dim)
#   - facial: ResNet-50 (2048-dim)
# Loss: 1.0 * CE(label_smoothing=0.1) + 0.3 * SupCon(temperature=0.07)
#
# Usage:
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/train_ser_v6.sbatch
#   sbatch scripts/train_ser_v6.sbatch 43
#   sbatch scripts/train_ser_v6.sbatch 44
#
# Then run ensemble eval (separate sbatch).
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

SEED="${1:-42}"

echo "============================================================"
echo "v6 multimodal SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Seed: $SEED"
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Pre-flight: ensure dataset manifest exists
if [ ! -e combined_ser_dataset/manifest.csv ]; then
    if [ -e combined_ser_dataset/metadata.csv ]; then
        echo "Creating manifest.csv symlink to metadata.csv"
        ln -sf metadata.csv combined_ser_dataset/manifest.csv
    else
        echo "ERROR: combined_ser_dataset manifest missing"
        echo "Run: bash fetch_combined_dataset.sh first"
        exit 1
    fi
fi

# Pre-flight: ensure frozen encoder checkpoints exist
if [ ! -f model_checkpoints/ser_v5_best_seed42.pt ]; then
    echo "ERROR: model_checkpoints/ser_v5_best_seed42.pt missing"
    exit 1
fi
if [ ! -f model_checkpoints/ter_pytorch_best.pt ]; then
    echo "ERROR: model_checkpoints/ter_pytorch_best.pt missing"
    exit 1
fi

# Run the v6 multimodal training
uv run python -u train_ser_v6_multimodal.py --seed "$SEED" --epochs 30 2>&1 | tee "train_ser_v6_seed${SEED}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
