#!/bin/bash
#SBATCH --job-name=dtu-ser-v4
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
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
#   - Single GPU per job (3 GPUs total across seeds 42/43/44)
#   - 50 epochs cosine (vs 70 in v2 — large converges faster)
#
# Expected runtime: ~6-7 hours per seed on 1× A100.
# 8-hour budget is safe.
#
# Per Jai's spec (2026-08-08): 1 GPU per job, 3 GPUs total across the
# 3-seed ensemble. Earlier multi-GPU configs (4 GPU, then 2 GPU per job)
# got stuck in PD state — likely the dgxnp partition's `Priority` reason
# indicates a scheduling conflict with multi-GPU requests. Single-GPU
# submits reliably, even though each job takes longer.
#
# GPU budget per job: 1× A100-40GB. With grad ckpt + batch_size=8,
# peak memory ~32 GB, safe under 40 GB limit.
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
# Pre-requisite: wav2vec2-large is auto-downloaded by this script on the
# compute node (no manual download needed). ~1.2 GB, first run only.
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

# GPU visibility: SLURM assigns 1 GPU per job (per Jai's spec 2026-08-08).
# Earlier multi-GPU configs (2 GPU, 4 GPU) got stuck in PD state — likely
# the dgxnp partition's scheduler has trouble packing multi-GPU jobs from
# our account. Single-GPU per job submits reliably.
echo "Visible GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Confirm wav2vec2-large is cached. If not, download it HERE on the
# compute node (NOT on the login node — login-node downloads are banned).
# Compute nodes have outbound internet via the cluster's NAT.
LARGE_PATH="$HOME/.cache/huggingface/hub/models--facebook--wav2vec2-large"
if [ ! -d "$LARGE_PATH" ]; then
    echo "wav2vec2-large not in HF cache. Downloading on compute node (~1.2 GB)..."
    uv run python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large')" || {
        echo "ERROR: download failed. Check compute-node outbound network."
        exit 1
    }
fi
echo "wav2vec2-large cache: OK"

# Run training
uv run python -u train_ser_v4.py --seed "$SEED" 2>&1 | tee "train_ser_v4_seed${SEED}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"