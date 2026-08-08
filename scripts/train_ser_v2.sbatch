#!/bin/bash
#SBATCH --job-name=dtu-ser-v2
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=ser_v2.%J.out
#SBATCH --error=ser_v2.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER enhanced v2 - SLURM batch job (v2 recipe, with seed + unfrozen-layers)
# ============================================================
# v2 recipe (the one that gave test_acc=0.6836 on Aug 7):
#   - last 4 wav2vec2 transformer layers unfrozen (NOT all 12)
#   - 70 epochs cosine (NOT 90)
#   - mixup + SpecAugment + class-balanced CE (NOT dual loss)
#   - 5-pass TTA on test
#
# Usage (from HPC login node, NEVER run compute on login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/train_ser_v2.sbatch           # default: seed=42, unfrozen=4
#   sbatch scripts/train_ser_v2.sbatch 43 4      # seed=43, unfrozen=4
#   sbatch scripts/train_ser_v2.sbatch 44 4      # seed=44, unfrozen=4
#
#   # Watch progress:
#   squeue -u $USER
#   tail -f ser_v2.<JOBID>.out
#
# For a 3-seed v2 ensemble (recommended), use train_ser_v2_ensemble.sbatch
# instead — it submits all three seeds in parallel for ~3h wall-clock.
#
# Expected runtime: ~2.5 hours (70 epochs + live wav2vec2 fine-tune + TTA).
# 4-hour budget is safe.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

# Positional args (defaults match the original 68% recipe)
SEED="${1:-42}"
UNFROZEN="${2:-4}"

echo "============================================================"
echo "SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Seed: $SEED  Unfrozen layers: $UNFROZEN"
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

# Pull latest code (no-op if no remote changes)
git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Confirm wav2vec2 model is cached locally (avoids HF download on compute node)
if [ ! -f ~/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/pytorch_model.bin ]; then
    echo "WARNING: wav2vec2-base model not in HF cache. The job will try to download it."
    echo "If the compute node has no internet, the job will fail. Cancel and download on login node first."
fi

# Run training with seed + unfrozen-layers (defaults preserve the 68% recipe).
# NO HF_HUB_OFFLINE=1 - we want the model to load, but it should come from
# local cache. Only fails if cache is missing.
uv run python -u train_ser_enhanced.py --seed "$SEED" --num-unfrozen-layers "$UNFROZEN" 2>&1 | tee "train_ser_v2_seed${SEED}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
