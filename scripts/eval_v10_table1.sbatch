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
echo "unif ckpts: $(ls model_checkpoints/v10_fusion_unif_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "cw   ckpts: $(ls model_checkpoints/v10_fusion_cw_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "v9b text ckpts: $(ls model_checkpoints/v9b_text_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "v9b audio ckpts: $(ls model_checkpoints/v9b_audio_fold*_seed*.pt 2>/dev/null | wc -l)/15"
# NOTE: the eval script resolves repo-relative paths from __file__, but --out is
# relative to CWD. Run from the repo root so the JSON lands in reports/.
uv run python -u scripts/iemocap/ensemble_evaluate_v10.py \
  --manifest data/iemocap/manifest.csv \
  --out reports/v10_table1_results.json 2>&1
echo "=== V10 EVAL done: $(date) ==="
