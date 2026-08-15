#!/bin/bash
#SBATCH --job-name=dtu-ser-v9b-text
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v9b_text.%A_%a.out
#SBATCH --error=ser_v9b_text.%A_%a.err
#SBATCH --account=dtuarp-acc
set -e
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false

# Warmup (job 0 only): ensure sentencepiece + HF model cache.
# 1. Install sentencepiece if missing (DeBERTa-v3 tokenizer requires it).
# 2. Download microsoft/deberta-v3-base to the shared HF cache with HF online.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  echo "=== V9B TEXT warmup: ensure sentencepiece + DeBERTa-v3-base cached ==="
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  uv pip install sentencepiece 2>&1 | tail -3
  uv run python -c "from transformers import AutoModel, AutoTokenizer; \
    m = AutoModel.from_pretrained('microsoft/deberta-v3-base'); \
    t = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base'); \
    print('DeBERTa-v3-base cached:', t.vocab_size, 'tokens,', sum(p.numel() for p in m.parameters()), 'params')"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "=== V9B TEXT job $SLURM_ARRAY_TASK_ID on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
[[ -f data/iemocap/manifest.csv ]] || { echo "FATAL: manifest missing"; exit 1; }
[[ -d data/iemocap/Session1/dialog/wav ]] || { echo "FATAL: IEMOCAP audio dir missing"; exit 1; }
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((42 + SLURM_ARRAY_TASK_ID / 5))
echo "Fold=$FOLD Seed=$SEED"
uv run python -u train_ser_v9b_text_only.py --fold "$FOLD" --seed "$SEED" 2>&1
echo "=== V9B TEXT fold $FOLD seed $SEED done: $(date) ==="
