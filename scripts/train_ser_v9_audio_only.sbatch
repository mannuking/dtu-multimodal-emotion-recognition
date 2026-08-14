#!/bin/bash
#SBATCH --job-name=dtu-ser-v9-audio
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v9_audio.%A_%a.out
#SBATCH --error=ser_v9_audio.%A_%a.err
#SBATCH --account=dtuarp-acc
set -e
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
echo "=== AUDIO job $SLURM_ARRAY_TASK_ID on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
[[ -f data/iemocap/manifest.csv ]] || { echo "FATAL: manifest missing at $(pwd)/data/iemocap/manifest.csv"; exit 1; }
[[ -d data/iemocap/Session1/dialog/wav ]] || { echo "FATAL: IEMOCAP audio dir missing"; exit 1; }
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((42 + SLURM_ARRAY_TASK_ID / 5))
echo "Fold=$FOLD Seed=$SEED"
uv run python -u train_ser_v9_audio_only.py --fold "$FOLD" --seed "$SEED" 2>&1
echo "=== AUDIO fold $FOLD seed $SEED done: $(date) ==="