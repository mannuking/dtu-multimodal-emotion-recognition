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
# Submits to a PARAM Siddhi-AI compute node (login node has
# no internet; everything must run inside SLURM).
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Working dir: $(pwd)"
echo "============================================================"

# ---- Install uv if not present ----
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "[1/4] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ---- Create venv + install deps ----
echo "[2/4] Setting up Python 3.11 + deps via uv..."
uv venv --python 3.11
uv sync --extra cuda

# ---- Verify dataset ----
echo "[3/4] Verifying dataset..."
if [[ ! -f "combined_ser_dataset/metadata.csv" ]]; then
    echo "ERROR: combined_ser_dataset/metadata.csv missing"
    echo "Run scripts/install_dataset.sh first to download."
    exit 1
fi
WAV_COUNT=$(find combined_ser_dataset -name "*.wav" 2>/dev/null | wc -l)
echo "  Dataset: $WAV_COUNT wav files"

# ---- Run pipeline ----
echo "[4/4] Running training pipeline..."
uv run python uv_run_all.py 2>&1 | tee "train.${SLURM_JOB_ID}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"