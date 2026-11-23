#!/bin/bash
#SBATCH --job-name=v11-table1-eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:00:00
#SBATCH --output=v11_table1_eval.%j.out
#SBATCH --error=v11_table1_eval.%j.err
#SBATCH --account=dtuarp-acc
set -e
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
echo "=== V11 TABLE 1 EVAL on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "unif ckpts: $(ls model_checkpoints/v11_va_unif_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "cw   ckpts: $(ls model_checkpoints/v11_va_cw_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "v9b text ckpts: $(ls model_checkpoints/v9b_text_fold*_seed*.pt 2>/dev/null | wc -l)/15"
echo "v9b audio ckpts: $(ls model_checkpoints/v9b_audio_fold*_seed*.pt 2>/dev/null | wc -l)/15"
mkdir -p reports
# Run from the repo root so --out lands in reports/ (relative to CWD).
uv run python -u scripts/iemocap/ensemble_evaluate_v11.py \
  --manifest data/iemocap/manifest.csv \
  --out reports/v11_table1_results.json 2>&1
echo "=== done: $(date) ==="
echo "=== v11_table1_results.json ==="
cat reports/v11_table1_results.json
