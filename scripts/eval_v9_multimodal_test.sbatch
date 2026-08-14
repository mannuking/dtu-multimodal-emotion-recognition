#!/bin/bash
#SBATCH --job-name=v9-mul-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:00:00
#SBATCH --output=v9_mul_eval.%j.out
#SBATCH --error=v9_mul_eval.%j.err
#SBATCH --account=dtuarp-acc

# ============================================================
# v9 Multimodal (Text+Audio) test eval — 14 ckpts, per-seed
# Loads existing v9_baseline_fold*_seed*.pt and reports
# per-seed + per-fold + grand-mean accuracy on each ckpt's own
# random-split val set.
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

echo "============================================================"
echo "v9 multimodal eval: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Started: $(date)"
echo "============================================================"

uv run python -u ensemble_evaluate_v9_test.py \
  --manifest data/iemocap/manifest.csv \
  --out reports/v9_multimodal_test_results.json 2>&1

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"