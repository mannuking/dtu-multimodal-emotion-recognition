#!/bin/bash
#SBATCH --job-name=dtu-ser-v8
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --output=ser_v8.%A_%a.out
#SBATCH --error=ser_v8.%A_%a.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v8 — V/A-conditioned CMT + dialog context
# Per-fold × per-seed job. Submit with:
#   sbatch --array=0-19 scripts/train_ser_v8.sbatch
# where the array index 0-4 = fold, 5-9 = fold1+seed2, etc.
# Or use the wrapper scripts/train_ser_v8_ensemble.sbatch.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

# Auto-download BERT-base + wav2vec2-base on compute node if missing
python3 -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')" 2>/dev/null || \
    echo "BERT cache check failed — will download inside training script"

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
# Index 0-4:  fold 0-4, seed 42
# Index 5-9:  fold 0-4, seed 43
# Index 10-14: fold 0-4, seed 44
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((42 + SLURM_ARRAY_TASK_ID / 5))

echo "============================================================"
echo "v8 SLURM job: ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "  Fold: $FOLD  Seed: $SEED  Host: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Started: $(date)"
echo "============================================================"

uv run python -u train_ser_v8_cmt.py --fold "$FOLD" --seed "$SEED" 2>&1

echo "============================================================"
echo "Fold $FOLD seed $SEED done: $(date)"
echo "============================================================"
