#!/usr/bin/env bash
# train_ser_v6.sbatch - Multimodal SER (audio + text + facial) fusion
#
# Submits a single A100 job to train the v6 multimodal fusion head.
# Encoders (audio v5, text MobileBERT, facial ResNet-50) are frozen.
# Only the 1.58M fusion head params are trained.
#
# Wall-clock: ~4 hours per seed (small fusion head, 30 epochs).
# Memory:      32 GB A100 sufficient (frozen encoders stay on CPU offload).
#
# Usage:
#   sbatch scripts/train_ser_v6.sbatch              # default seed=42
#   SEED=43 sbatch scripts/train_ser_v6.sbatch      # override seed
#   EPOCHS=20 sbatch scripts/train_ser_v6.sbatch    # override epochs

#SBATCH --job-name=dtu-ser-v6-multimodal
#SBATCH --partition=dgxnp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=ser_v6.%j.out
#SBATCH --error=ser_v6.%j.err

set -euo pipefail

SEED=${SEED:-42}
EPOCHS=${EPOCHS:-30}

cd ~/Research/dtu/dtu-multimodal-emotion-recognition

echo "=================================="
echo "v6 Multimodal SER fusion"
echo "Seed:    $SEED"
echo "Epochs:  $EPOCHS"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"
echo "=================================="

# Make sure the v5 audio checkpoint is in place
if [ ! -f model_checkpoints/ser_v5_best_seed42.pt ]; then
    echo "ERROR: model_checkpoints/ser_v5_best_seed42.pt missing"
    echo "Train v5 first (sbatch scripts/train_ser_v5_ensemble.sbatch)"
    exit 1
fi

# Make sure text encoder is in place
if [ ! -f model_checkpoints/ter_pytorch_best.pt ]; then
    echo "ERROR: model_checkpoints/ter_pytorch_best.pt missing"
    echo "Train text encoder first (sbatch scripts/train_ter.sbatch)"
    exit 1
fi

# Run the v6 multimodal training
uv run python -u train_ser_v6_multimodal.py \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size 16 \
    --lr 1e-4 \
    --audio-ckpt model_checkpoints/ser_v5_best_seed42.pt \
    --text-ckpt model_checkpoints/ter_pytorch_best.pt \
    --manifest combined_ser_dataset/manifest.csv

echo "=================================="
echo "Job finished: $(date)"
echo "=================================="
