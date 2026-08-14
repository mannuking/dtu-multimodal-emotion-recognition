#!/bin/bash
#SBATCH --job-name=dtu-ser-v9-text
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v9_text.%A_%a.out
#SBATCH --error=ser_v9_text.%A_%a.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v9 — Text-only baseline (BERT-base-uncased frozen + linear head)
# 3 seeds × 5 folds = 15 jobs. Submit with:
#   sbatch --array=0-14 scripts/train_ser_v9_text_only.sbatch
# where:
#   FOLD=$((SLURM_ARRAY_TASK_ID % 5))     # 0-4
#   SEED=$((42 + SLURM_ARRAY_TASK_ID / 5)) # 42, 43, 44
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

git pull origin main || echo "  (git pull failed - continuing)"

[[ -f data/iemocap/manifest.csv ]] || {
    echo "ERROR: data/iemocap/manifest.csv missing"
    exit 1
}

FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((42 + SLURM_ARRAY_TASK_ID / 5))

echo "============================================================"
echo "v9 TEXT-ONLY: ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "  Fold: $FOLD  Seed: $SEED  Host: $(hostname)"
echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Started: $(date)"
echo "============================================================"

uv run python -u train_ser_v9_text_only.py --fold "$FOLD" --seed "$SEED" 2>&1

echo "============================================================"
echo "TEXT fold $FOLD seed $SEED done: $(date)"
echo "============================================================"