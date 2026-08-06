#!/bin/bash
#SBATCH --job-name=dtu-train
#SBATCH -N 1
#SBATCH --partition=dgxnp
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=12:00:00
#SBATCH --output=train.%J.out
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU Multimodal Emotion Recognition — Full training pipeline
#
# Multi-GPU configuration (per PARAM Siddhi-AI Advisories):
#   - 16 cpus/GPU (Advisory 6)
#   - No --exclusive flag (Advisory 5)
#   - Max 8 running + 8 queued jobs/user (Advisory 4)
#   - partition=dgxnp (your normal partition) + account=dtuarp-acc
#
# Examples:
#   1 GPU (default): sbatch scripts/train.sbatch
#   2 GPUs:          sbatch --export=NGPUS=2,NTASKS_PER_NODE=32 \
#                          --gres=gpu:A100-SXM4:2 scripts/train.sbatch
#   4 GPUs:          sbatch --export=NGPUS=4,NTASKS_PER_NODE=64 \
#                          --gres=gpu:A100-SXM4:4 scripts/train.sbatch
#   8 GPUs (full):   sbatch --export=NGPUS=8,NTASKS_PER_NODE=128 \
#                          --gres=gpu:A100-SXM4:8 scripts/train.sbatch
#
# Stages:
#   1. Build leakage-free triplet manifest
#   2. Train SER (1D-CNN, MFCC + energy)         ~ MirroredStrategy
#   3. Train TER (MobileBERT + FGSM)             ~ nn.DataParallel
#   4. Train FER (VGG16 + ResNet50 ensemble)    ~ MirroredStrategy
#   5. Train meta-classifier (late fusion MLP)   ~ MirroredStrategy
#   6. Run integrity test
#   7. Print summary with measured accuracy vs paper claim
#
# Estimated runtime: 4-8 hrs on 1× A100, ~2-4 hrs on 4× A100.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

# Resolve NGPUS — set via --export=NGPUS=N on the sbatch command line.
NGPUS=${NGPUS:-1}
NTASKS_PER_NODE=${NTASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-$((16 * NGPUS))}}

echo "============================================================"
echo "Train job started: $(date)"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU(s): $(nvidia-smi -L 2>/dev/null | head -8)"
echo "NGPUS=$NGPUS  tasks/node=$NTASKS_PER_NODE"
echo "Working dir: $(pwd)"
echo "============================================================"

# Ensure uv is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "uv not found at ~/.local/bin/uv — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv location: $(which uv)"
echo "uv version: $(uv --version)"

# Sync deps (idempotent — fast if up-to-date, slow on first run)
echo "Syncing Python dependencies..."
uv sync --extra cuda 2>&1 | tee uv_sync.log || {
    echo "uv sync failed — likely stale venv. Recreating from scratch..."
    rm -rf .venv
    uv sync --extra cuda 2>&1 | tee uv_sync.log
}

# Verify dataset
if [[ ! -f "combined_ser_dataset/metadata.csv" ]]; then
    echo "ERROR: combined_ser_dataset/metadata.csv missing"
    echo "Run scripts/setup.sbatch first to download the dataset."
    exit 1
fi
WAV_COUNT=$(find combined_ser_dataset -name "*.wav" 2>/dev/null | wc -l)
echo "Dataset: $WAV_COUNT wav files"
echo ""

# GPU summary
nvidia-smi -L 2>&1 | head -10 || echo "nvidia-smi unavailable"

# Run pipeline
echo "Launching uv_run_all.py..."
echo "  (full pipeline: SER → TER → FER → meta-classifier → integrity test)"
echo ""

uv run python uv_run_all.py