#!/bin/bash
#SBATCH --job-name=dtu-ser-v10-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:00:00
#SBATCH --output=ser_v10_eval.%j.out
#SBATCH --error=ser_v10_eval.%j.err
#SBATCH --account=dtuarp-acc
set -e
cd "$SLURM_SUBMIT_DIR"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "=== V10 EVAL on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
[[ -f data/iemocap/manifest.csv ]] || { echo "FATAL: manifest missing"; exit 1; }
mkdir -p reports
cd scripts/iemocap
uv run python -u ensemble_evaluate_v10.py 2>&1
echo "=== V10 EVAL done: $(date) ==="
