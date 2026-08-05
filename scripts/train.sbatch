#!/bin/bash
#SBATCH --job-name=dtu-train
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=12:00:00
#SBATCH --output=train.%J.out
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU Multimodal Emotion Recognition — Full training pipeline
#
# Assumes scripts/setup.sbatch has already run successfully
# (uv installed, .venv created, dataset extracted).
#
# Stages:
#   1. Build leakage-free triplet manifest
#   2. Train SER (1D-CNN, MFCC + energy features)
#   3. Train TER (MobileBERT + FGSM adversarial)
#   4. Train FER (VGG16 + ResNet50 ensemble)
#   5. Train meta-classifier (late fusion MLP)
#   6. Run integrity test
#   7. Print summary with measured accuracy vs paper claim
#
# Estimated runtime: 4-8 hours on A100.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Train job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Working dir: $(pwd)"
echo "============================================================"

# Ensure uv is on PATH (in case the SLURM env doesn't inherit it)
export PATH="$HOME/.local/bin:$PATH"

# Install uv if missing (idempotent — does nothing if already installed)
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

# ---- Verify dataset is in place ----
if [[ ! -f "combined_ser_dataset/metadata.csv" ]]; then
    echo "ERROR: combined_ser_dataset/metadata.csv missing"
    echo "Run scripts/setup.sbatch first to download the dataset."
    exit 1
fi
WAV_COUNT=$(find combined_ser_dataset -name "*.wav" 2>/dev/null | wc -l)
echo "Dataset: $WAV_COUNT wav files"
echo ""

# ---- Run pipeline ----
echo "Launching uv_run_all.py..."
echo "  (full pipeline: SER → TER → FER → meta-classifier → integrity test)"
echo ""

uv run python uv_run_all.py 2>&1 | tee "train.${SLURM_JOB_ID}.log"

echo ""
echo "============================================================"
echo "Train job finished: $(date)"
echo "Outputs:"
echo "  - model_checkpoints/  (trained .keras + .pt files)"
echo "  - reports/            (verification JSON + log)"
echo "  - train.${SLURM_JOB_ID}.log  (full stdout/stderr)"
echo "============================================================"