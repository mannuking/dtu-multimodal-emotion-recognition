#!/bin/bash
#SBATCH --job-name=dtu-train
#SBATCH -N 1
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE:-256}
#SBATCH --gres=gpu:A100-SXM4:${NGPUS:-1}
#SBATCH --time=12:00:00
#SBATCH --output=train.%J.out
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU Multimodal Emotion Recognition — Full training pipeline
#
# Multi-GPU configuration (set NGPUS before sbatch):
#
#   1 GPU  : sbatch scripts/train.sbatch                    (--gres=A100:1)
#   2 GPUs : sbatch --export=NGPUS=2 scripts/train.sbatch   (--gres=A100:2, 64 cores)
#   4 GPUs : sbatch --export=NGPUS=4 scripts/train.sbatch   (--gres=A100:4, 128 cores)
#   8 GPUs : sbatch --export=NGPUS=8 scripts/train.sbatch   (--gres=A100:8, 256 cores)
#
# Memory per GPU is 40 GB on PARAM Siddhi-AI A100-SXM4; 8×40 = 320 GB GPU memory
# available on a full node. Our pipeline peak usage (SER + FER in sequence) is
# ~30 GB, so any GPU count works without changing the env.
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

# Resolve NGPUS (default 1) and CPU count (manual says 32 cores per GPU)
NGPUS=${NGPUS:-1}
NTASKS_PER_NODE=${NTASKS_PER_NODE:-$((32 * NGPUS))}

echo "============================================================"
echo "Train job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "GPUs requested: $NGPUS  (tasks/node: $NTASKS_PER_NODE)"
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

# GPU summary — confirm SLURM is giving us what we asked for
nvidia-smi -L 2>&1 | head -10 || echo "nvidia-smi unavailable"

# Run pipeline
echo "Launching uv_run_all.py..."
echo "  (full pipeline: SER → TER → FER → meta-classifier → integrity test)"
echo ""

uv run python uv_run_all.py