#!/usr/bin/env bash
# train_all.sh — End-to-end training pipeline for the DTU multimodal pipeline.
# Designed for HPC / CUDA workstation execution (recommended: 16+ GB VRAM).
#
# Usage:
#   chmod +x scripts/train_all.sh
#   ./scripts/train_all.sh             # full pipeline, all 4 SER variants
#   ./scripts/train_all.sh --quick     # 3-epoch smoke test
#
# Prereqs (HPC):
#   - Python 3.11+
#   - CUDA 12.x + cuDNN 8.x
#   - NVIDIA GPU with >=12 GB VRAM (RTX 3080 / 4070+ recommended)
#   - pip install -r requirements.txt
#
# Outputs:
#   - model_checkpoints/ser_best.keras
#   - model_checkpoints/{vgg16,resnet50}_{orig,bal}_best.keras
#   - model_checkpoints/ter_pytorch_best.pt
#   - model_checkpoints/meta_hybrid_best.keras
#   - All *.history.json files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Args ----
QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_USE_LEGACY_KERAS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "============================================================"
echo " DTU Multimodal Emotion Recognition — Training Pipeline"
echo " Project: $PROJECT_ROOT"
echo " Host:    $(hostname)"
echo " GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"
echo " Date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ---- 1. Sanity check ----
echo ""
echo "[1/5] Environment check..."
python3 -c "
import sys; print('Python:', sys.version.split()[0])
import tensorflow as tf; print('TensorFlow:', tf.__version__)
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  Device:', torch.cuda.get_device_name(0))
    print('  Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# ---- 2. Build triplet manifest ----
echo ""
echo "[2/5] Building leakage-free triplet manifest..."
python3 build_triplets_leakage_free.py \
    --audio-manifest combined_ser_dataset/metadata.csv \
    --output triplets_manifest.csv \
    --per-class 1000 \
    --seed 42

# ---- 3. Train unimodal models ----
echo ""
echo "[3/5] Training unimodal models..."
if [[ $QUICK -eq 1 ]]; then
    SER_EPOCHS=3
    FER_EPOCHS=3
    TER_EPOCHS=2
else
    SER_EPOCHS=50
    FER_EPOCHS=30
    TER_EPOCHS=12
fi

echo "  [SER] speech emotion recognition..."
python3 train_ser_tensorflow.py

echo "  [TER] text emotion recognition (PyTorch + MobileBERT)..."
python3 train_ter_pytorch.py

echo "  [FER] facial emotion recognition (4 models)..."
python3 train_fer_tensorflow.py

# ---- 4. Train meta-classifier (late fusion) ----
echo ""
echo "[4/5] Training meta-classifier (late fusion)..."
python3 train_meta_classifier.py

# ---- 5. Run integrity test ----
echo ""
echo "[5/5] Running model integrity test on held-out split..."
python3 verify_ser.py

echo ""
echo "============================================================"
echo " Pipeline complete."
echo " Reports:    reports/"
echo " Checkpoints: model_checkpoints/"
echo " Triplets:    triplets_manifest.csv"
echo "============================================================"