#!/bin/bash
#SBATCH --job-name=dtu-ser-v4
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --output=ser_v4.%J.out
#SBATCH --error=ser_v4.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER enhanced v4 - Phase 1 architecture for the 82% target
# ============================================================
# Changes vs v2 (the 68% baseline):
#   - wav2vec2-large (317M params, 24 layers, 1024-dim) instead of -base
#   - Last 6 layers unfrozen (vs 4 in v2)
#   - Bidirectional attention pooling head
#   - Supervised Contrastive (SupCon) auxiliary loss (Khosla et al. 2020)
#   - Gradient checkpointing enabled for memory
#   - 50 epochs cosine (vs 70 in v2 — large converges faster)
#
# Expected runtime: ~5-6 hours per seed on A100.
# 8-hour budget is safe.
#
# Usage (from HPC login node, NEVER run compute on login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/train_ser_v4.sbatch 42
#   sbatch scripts/train_ser_v4.sbatch 43
#   sbatch scripts/train_ser_v4.sbatch 44
#
#   # Then run ensemble evaluation:
#   sbatch scripts/eval_ensemble_v4.sbatch
#
# Pre-requisite: wav2vec2-large must be in HF cache. If not, download on
# login node first (~1.2GB):
#   python3 -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large')"
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

SEED="${1:-42}"

echo "============================================================"
echo "v4 SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Seed: $SEED"
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# Confirm wav2vec2-large is cached
LARGE_PATH="$HOME/.cache/huggingface/hub/models--facebook--wav2vec2-large"
if [ ! -d "$LARGE_PATH" ]; then
    echo "ERROR: wav2vec2-large not in HF cache ($LARGE_PATH)"
    echo "Download on login node first:"
    echo "  python3 -c \"from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large')\""
    exit 1
fi
echo "wav2vec2-large cache: OK"

# Run training
uv run python -u train_ser_v4.py --seed "$SEED" 2>&1 | tee "train_ser_v4_seed${SEED}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"