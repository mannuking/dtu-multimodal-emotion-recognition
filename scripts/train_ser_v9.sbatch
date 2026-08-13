#!/bin/bash
#SBATCH --job-name=dtu-ser-v9
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --output=ser_v9.%A_%a.out
#SBATCH --error=ser_v9.%A_%a.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v9 — MemoCMT baseline reproduction (5-fold random CV, 3 seeds)
#
# Per-fold × per-seed job. Submit with:
#   sbatch --array=0-14 scripts/train_ser_v9.sbatch
# where the array index decodes as:
#   FOLD=$((SLURM_ARRAY_TASK_ID % 5))     # 0-4
#   SEED=$((42 + SLURM_ARRAY_TASK_ID / 5)) # 42, 43, 44
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

# Required env vars
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

# Pull latest from main
git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Required data + manifest must be present
[[ -f data/iemocap/manifest.csv ]] || {
    echo "ERROR: data/iemocap/manifest.csv missing. Run the manifest builder first."
    exit 1
}
[[ -d data/iemocap/Session1/dialog/wav ]] || {
    echo "ERROR: IEMOCAP audio missing under data/iemocap/."
    exit 1
}

# Job array index -> fold, seed
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((42 + SLURM_ARRAY_TASK_ID / 5))

echo "============================================================"
echo "v9 SLURM job: ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "  Fold: $FOLD  Seed: $SEED  Host: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Started: $(date)"
echo "============================================================"

uv run python -u train_ser_v9_baseline.py --fold "$FOLD" --seed "$SEED" 2>&1

echo "============================================================"
echo "Fold $FOLD seed $SEED done: $(date)"
echo "============================================================"
