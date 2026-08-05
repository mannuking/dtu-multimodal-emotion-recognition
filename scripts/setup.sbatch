#!/bin/bash
#SBATCH --job-name=dtu-setup
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=setup.%J.out
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU Multimodal Emotion Recognition — HPC setup
#
# What this does:
#   1. Allocates a compute node (login node has no internet)
#   2. Installs uv (single static binary, no sudo)
#   3. Pulls the latest repo
#   4. Creates Python 3.11 venv via uv
#   5. Installs all dependencies via `uv sync --extra cuda`
#   6. Downloads + extracts the dataset tarball from GitHub
#   7. Verifies torch/tensorflow/CUDA/triplet builder
#
# After this finishes successfully, run scripts/train.sbatch
# to launch the actual training.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Setup job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo '(no GPU visible)')"
echo "Working dir: $(pwd)"
echo "============================================================"

# ---- 1. Install uv ----
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo ""
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# ---- 2. Pull latest code ----
echo ""
echo "[2/5] Pulling latest code..."
git pull origin main || echo "  (pull skipped — using existing checkout)"

# ---- 3. Create venv + install deps ----
echo ""
echo "[3/5] Creating Python 3.11 venv + installing dependencies..."
echo "  (uv sync --extra cuda takes ~5-8 min)"
uv venv --python 3.11
uv sync --extra cuda

# ---- 4. Ensure dataset is present ----
echo ""
echo "[4/5] Checking dataset..."
if [[ ! -f "combined_ser_dataset/metadata.csv" ]]; then
    echo "  Dataset missing — downloading + extracting..."
    curl -fsSL https://raw.githubusercontent.com/mannuking/dtu-multimodal-emotion-recognition/main/scripts/install_dataset.sh | bash
else
    WAV_COUNT=$(find combined_ser_dataset -name "*.wav" 2>/dev/null | wc -l)
    echo "  Dataset present: $WAV_COUNT wav files"
fi

# ---- 5. Verify environment ----
echo ""
echo "[5/5] Verifying environment..."
uv run python -c "
import torch
import tensorflow as tf
print(f'  torch        : {torch.__version__}')
print(f'  torch.cuda   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU          : {torch.cuda.get_device_name(0)}')
print(f'  tensorflow   : {tf.__version__}')
print(f'  TF GPUs seen : {len(tf.config.list_physical_devices(\"GPU\"))}')
"

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Dataset location: $SLURM_SUBMIT_DIR/combined_ser_dataset/"
echo "Wav file count  : $(find combined_ser_dataset -name '*.wav' 2>/dev/null | wc -l)"
echo ""
echo "Next step — submit training:"
echo "  sbatch scripts/train.sbatch"
echo ""
echo "Or watch this setup job's tail first:"
echo "  tail -f setup.${SLURM_JOB_ID}.out"