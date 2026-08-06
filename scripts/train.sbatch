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
# Advisories honored:
#   - partition=dgxnp + account=dtuarp-acc
#   - 16 cpus/GPU
#   - no --exclusive
#   - max 8 running + 8 queued jobs
#
# GPU env vars MUST be set before any Python import — they tell the
# CUDA runtime which GPUs are visible and how to initialize. Setting
# these inside Python is too late (CUDA already initialized).
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

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

# ---- Critical: GPU env vars BEFORE any Python ----
# Tell CUDA which GPUs to use (SLURM allocates these; restrict visibility)
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Tell TF to use the async CUDA allocator (faster on A100)
export TF_GPU_ALLOCATOR="cuda_malloc_async"
# Disable XLA flag that breaks newer CUDA
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/hpcx"
# Tell HF datasets to use offline cache when possible
export HF_HUB_DISABLE_TELEMETRY=1
# Avoid TF import order issues — set Keras backend explicitly
export TF_USE_LEGACY_KERAS=1
export KERAS_BACKEND="tensorflow"

# ---- uv on PATH ----
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "uv not found at ~/.local/bin/uv — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ---- Sync deps ----
echo "Syncing Python dependencies..."
uv sync --extra cuda 2>&1 | tee uv_sync.log || {
    echo "uv sync failed — recreating venv from scratch..."
    rm -rf .venv
    uv sync --extra cuda 2>&1 | tee uv_sync.log
}

# ---- Verify dataset ----
if [[ ! -f "combined_ser_dataset/metadata.csv" ]]; then
    echo "ERROR: combined_ser_dataset/metadata.csv missing"
    exit 1
fi
WAV_COUNT=$(find combined_ser_dataset -name "*.wav" 2>/dev/null | wc -l)
echo "Dataset: $WAV_COUNT wav files"

# ---- GPU visibility check ----
echo "GPU visibility (nvidia-smi):"
nvidia-smi -L | head -8

echo ""
echo "Launching uv_run_all.py..."
uv run python uv_run_all.py