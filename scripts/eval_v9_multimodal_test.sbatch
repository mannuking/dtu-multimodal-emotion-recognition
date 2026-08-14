#!/bin/bash
#SBATCH --job-name=v9-mul-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:00:00
#SBATCH --output=v9_mul_eval.%j.out
#SBATCH --error=v9_mul_eval.%j.err
#SBATCH --account=dtuarp-acc
set -e
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
echo "=== EVAL on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
[[ -f data/iemocap/manifest.csv ]] || { echo "FATAL: manifest missing"; exit 1; }
uv run python -u ensemble_evaluate_v9_test.py \
  --manifest data/iemocap/manifest.csv \
  --out reports/v9_multimodal_test_results.json 2>&1
echo "=== EVAL done: $(date) ==="