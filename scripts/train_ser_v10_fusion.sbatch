#!/bin/bash
#SBATCH --job-name=dtu-ser-v10-fusion
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v10_fusion.%A_%a.out
#SBATCH --error=ser_v10_fusion.%A_%a.err
#SBATCH --account=dtuarp-acc
#SBATCH --array=0-29
# Array layout: 30 jobs = 15 (fold,seed) x 2 (class_weighted false/true).
# task_id % 15 = fold*3 + (seed-42), task_id // 15 = 0 (unif) or 1 (cw).
set -e
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  echo "=== V10 FUSION warmup: cache v9b encoders ==="
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  uv run python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('microsoft/wavlm-base-plus'); \
    AutoModel.from_pretrained('microsoft/deberta-v3-base'); \
    print('v9b encoders cached')"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "=== V10 FUSION job $SLURM_ARRAY_TASK_ID on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
[[ -f data/iemocap/manifest.csv ]] || { echo "FATAL: manifest missing"; exit 1; }
[[ -d data/iemocap/Session1/dialog/wav ]] || { echo "FATAL: IEMOCAP audio dir missing"; exit 1; }

TID=$SLURM_ARRAY_TASK_ID
WEIGHT_IDX=$((TID / 15))           # 0=uniform, 1=class_weighted
FS_IDX=$((TID % 15))               # 0..14: fold*3 + (seed-42)
FOLD=$((FS_IDX / 3))
SEED=$((42 + FS_IDX % 3))
if [ "$WEIGHT_IDX" = "0" ]; then
  CW="false"
else
  CW="true"
fi
echo "Fold=$FOLD Seed=$SEED class_weighted=$CW"
cd scripts/iemocap
uv run python -u train_ser_v10_fusion.py \
  --fold "$FOLD" --seed "$SEED" --class-weighted "$CW" 2>&1
echo "=== V10 FUSION fold $FOLD seed $SEED cw=$CW done: $(date) ==="
