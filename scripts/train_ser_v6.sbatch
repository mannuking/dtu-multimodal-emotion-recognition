#!/usr/bin/env bash
# train_ser_v6.sbatch - Multimodal SER (audio + text + facial) fusion
# DGX-only — requires a QoS that permits dgxnp submission (maxnodeal+ / nltm-asr / similar)
# Single-A100 used via CUDA_VISIBLE_DEVICES=0, but gres requests whole-node to satisfy
# the nodeallocgpu partition constraint.

#SBATCH --job-name=dtu-ser-v6-multimodal
#SBATCH --partition=dgxnp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=ser_v6.%j.out
#SBATCH --error=ser_v6.%j.err

set -euo pipefail
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-30}
cd ~/Research/dtu/dtu-multimodal-emotion-recognition
echo "v6 multimodal DGX | seed=$SEED epochs=$EPOCHS job=$SLURM_JOB_ID node=$SLURM_NODELIST start=$(date)"
if [ ! -f model_checkpoints/ser_v5_best_seed42.pt ]; then
    echo "ERROR: v5 audio checkpoint missing"
    exit 1
fi
if [ ! -f model_checkpoints/ter_pytorch_best.pt ]; then
    echo "ERROR: text encoder checkpoint missing"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=0
uv run python -u train_ser_v6_multimodal.py --seed "$SEED" --epochs "$EPOCHS" 2>&1 | tee "ser_v6_seed${SEED}.log"
