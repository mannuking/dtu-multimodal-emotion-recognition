#!/bin/bash
#SBATCH --job-name=v9b-table1-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=00:30:00
#SBATCH --output=v9b_table1_eval.%j.out
#SBATCH --error=v9b_table1_eval.%j.err
#SBATCH --account=dtuarp-acc
set -e
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
echo "=== V9B TABLE 1 EVAL on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "text ckpts: $(ls model_checkpoints/v9b_text_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "audio ckpts: $(ls model_checkpoints/v9b_audio_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "fusion ckpts: $(ls model_checkpoints/v9_baseline_fold*_seed*.pt 2>/dev/null | wc -l)/15"
uv run python -u ensemble_evaluate_v9b.py \
  --manifest data/iemocap/manifest.csv \
  --out reports/v9b_table1.json 2>&1
echo "=== done: $(date) ==="
echo "=== v9b_table1.json ==="
cat reports/v9b_table1.json
