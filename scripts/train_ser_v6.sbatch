#!/usr/bin/env bash
# train_ser_v6.sbatch - Multimodal SER (audio + text + facial) fusion
# DGX dgxnp, A100-SXM4 nomenclature, single-GPU via CUDA_VISIBLE_DEVICES=0
# Fixed 2026-08-11: corrected GPU string (A100-SXM4 not a100), added single-GPU,
#                   renamed output logs to standard ser_v6.%j pattern.

#SBATCH --job-name=dtu-ser-v6-multimodal
#SBATCH --partition=dgxnp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=ser_v6.%j.out
#SBATCH --error=ser_v6.%j.err

set -euo pipefail
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-30}
cd ~/Research/dtu/dtu-multimodal-emotion-recognition

# Ensure manifest.csv exists (build_combined_ser_dataset.py writes metadata.csv;
# v6 multimodal expects manifest.csv)
if [ ! -e combined_ser_dataset/manifest.csv ]; then
    if [ -e combined_ser_dataset/metadata.csv ]; then
        echo "Creating manifest.csv symlink to metadata.csv"
        ln -sf metadata.csv combined_ser_dataset/manifest.csv
    else
        echo "ERROR: neither manifest.csv nor metadata.csv found in combined_ser_dataset/"
        echo "Run: bash fetch_combined_dataset.sh first"
        exit 1
    fi
fi

echo "v6 multimodal | seed=$SEED epochs=$EPOCHS job=$SLURM_JOB_ID node=$SLURM_NODELIST start=$(date)"
if [ ! -f model_checkpoints/ser_v5_best_seed42.pt ]; then
    echo "ERROR: v5 audio checkpoint missing"
    exit 1
fi
if [ ! -f model_checkpoints/ter_pytorch_best.pt ]; then
    echo "ERROR: text encoder checkpoint missing"
    exit 1
fi

# Pin to single GPU even though whole-node gives 8 (PyTorch uses only 1)
export CUDA_VISIBLE_DEVICES=0
uv run python -u train_ser_v6_multimodal.py --seed "$SEED" --epochs "$EPOCHS" 2>&1 | tee "ser_v6_seed${SEED}.log"
