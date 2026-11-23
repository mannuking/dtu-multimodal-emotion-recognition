#!/bin/bash
#SBATCH --job-name=orig-pipeline
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=12:00:00
#SBATCH --output=orig_pipeline.%j.out
#SBATCH --error=orig_pipeline.%j.err
#SBATCH --account=dtuarp-acc
# ============================================================
# ORIGINAL (Oct-2025) DTU multimodal pipeline — full re-run.
#
# This is the ORIGINAL project: SER (1D-CNN on MFCC features) + FER
# (VGG16 / ResNet50 on FER2013) + TER (MobileBERT) + a late-fusion
# meta-classifier, over the COMBINED dataset
# (RAVDESS + CREMA-D + SAVEE + TESS, 7 classes, 11568 utterances).
#
# It is a different dataset and a different architecture family from the
# v9/v10/v11 IEMOCAP line. It is Paper 1. The v9+ line is Paper 2.
#
# ORDER MATTERS — each stage consumes the previous stage's checkpoint:
#   1. FER   -> vgg16_{orig,bal}_best.keras, resnet50_{orig,bal}_best.keras
#   2. SER   -> ser_best.keras
#   3. TER   -> ter_pytorch_best.pt + ter_pytorch_tokenizer/
#   4. META  -> meta_hybrid_best.keras   (fuses all three)
#
# The whole thing is resumable: each trainer skips itself if its
# checkpoint already exists, and combined_dtu.py records progress in
# model_checkpoints/master_training_status.json.
# ============================================================

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo " ORIGINAL PIPELINE — job started: $(date)"
echo " Node: $(hostname)   JobID: $SLURM_JOB_ID"
echo " GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed — continuing with local checkout)"

# ---------------- environment check ----------------
# A `uv sync` that fails can UNINSTALL the packages it would have added.
# Never start training on an environment that cannot import TensorFlow.
echo ""
echo "--- ENVIRONMENT ---"
uv run python - <<'PYEOF' || { echo "FATAL: the venv is missing required packages."; \
  echo "       Run: uv sync --extra cuda   (do NOT fall back to a bare uv sync)"; exit 1; }
import importlib, sys
need = ["tensorflow", "keras", "torch", "transformers", "librosa", "sklearn",
        "pandas", "numpy", "PIL", "soundfile", "datasets", "huggingface_hub"]
bad = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception:
        bad.append(m); print(f"  --  {m:16s} MISSING")
if bad:
    print(f"  MISSING: {', '.join(bad)}"); sys.exit(1)
import tensorflow as tf
print(f"  TF GPUs visible on this node: {len(tf.config.list_physical_devices('GPU'))}")
PYEOF

# ---------------- preflight ----------------
echo ""
echo "--- PREFLIGHT ---"
echo "repo:        $(pwd)"
echo "python:      $(uv run python -V 2>&1)"
echo "dataset dir: $(ls -d combined_ser_dataset 2>/dev/null || echo MISSING)"
echo "metadata:    $([ -f combined_ser_dataset/metadata.csv ] && echo present || echo MISSING)"
echo "wav count:   $(find combined_ser_dataset -name '*.wav' 2>/dev/null | wc -l)"
echo "feature npy: $([ -f ser_feature_output/features.npy ] && echo present || echo 'absent (will extract)')"
echo "labels npy:  $([ -f ser_feature_output/labels.npy ] && echo present || echo 'absent (will extract)')"
echo "fer2013:     $([ -d fer2013/train ] && echo present || echo 'absent (will download)')"
echo "ter csvs:    $([ -f ter_dataset/merged/text_train.csv ] && echo present || echo 'absent (will build)')"

# ---------------- stage 0: data preparation ----------------
# Everything below is CPU-only and cheap; it belongs in the same job so a
# fresh node can go from bare checkout to trained models unattended.
echo ""
echo "============================================================"
echo " STAGE 0 — data preparation (CPU)"
echo "============================================================"

echo "[0a] Building the text corpus for TER..."
uv run python -u prepare_ter_dataset.py || {
  echo "FATAL: TER dataset preparation failed"; exit 1; }

echo "[0b] Building the triplets manifest for the meta-classifier..."
uv run python -u build_triplets_leakage_free.py || {
  echo "FATAL: triplets manifest build failed"; exit 1; }

echo "[0c] Extracting SER audio features (this is the slow CPU step, ~30-60 min)..."
uv run python -u -c "
from feature_utils import ensure_features_exist
import os
f, l = ensure_features_exist(
    features_path=os.path.join('ser_feature_output','features.npy'),
    labels_path=os.path.join('ser_feature_output','labels.npy'),
    manifest_csv=os.path.join('combined_ser_dataset','metadata.csv'),
)
print('SER features ready:', f.shape, 'labels:', len(l))
" || { echo "FATAL: SER feature extraction failed"; exit 1; }

echo "[0d] Downloading FER2013..."
# fer2013_download.py skips the download if the train dir already has >= 7
# subdirectories. A partial or synthetic directory satisfies that check, so
# the download would be skipped and the trainer would silently train on
# garbage. Force the download and verify the real image count afterwards.
uv run python -u -c "
from fer2013_download import download_and_unpack
download_and_unpack(force=True)
" || echo "  WARNING: FER2013 download failed — do NOT accept a synthetic fallback"

# Verify FER2013 is REAL. The synthetic fallback writes 200 PNGs per class;
# the real dataset has ~28700 train images. 1500 images == synthetic.
uv run python -u -c "
import os, sys
n = sum(len(fs) for _, _, fs in os.walk('fer2013/train'))
print(f'  FER2013 train images: {n}')
if n < 5000:
    print('  FATAL: fer2013/train has too few images — this is the SYNTHETIC')
    print('         fallback, not the real dataset. FER results would be')
    print('         meaningless. Fix the download and re-run.')
    sys.exit(1)
print('  FER2013 looks real.')
" || { echo "FATAL: FER2013 is not the real dataset — stopping before training"; exit 1; }

# ---------------- stage 1: FER ----------------
echo ""
echo "============================================================"
echo " STAGE 1 — FER (VGG16 + ResNet50 on FER2013)"
echo "============================================================"
uv run python -u train_fer_tensorflow.py 2>&1 | tee -a orig_fer.log

# ---------------- stage 2: SER ----------------
echo ""
echo "============================================================"
echo " STAGE 2 — SER (1D-CNN on MFCC features)"
echo "============================================================"
uv run python -u train_ser_tensorflow.py 2>&1 | tee -a orig_ser.log

# ---------------- stage 3: TER ----------------
echo ""
echo "============================================================"
echo " STAGE 3 — TER (MobileBERT)"
echo "============================================================"
uv run python -u train_ter_pytorch.py 2>&1 | tee -a orig_ter.log

# ---------------- stage 4: meta-classifier ----------------
echo ""
echo "============================================================"
echo " STAGE 4 — META late-fusion classifier"
echo "============================================================"
uv run python -u train_meta_classifier.py 2>&1 | tee -a orig_meta.log

# ---------------- verification ----------------
echo ""
echo "============================================================"
echo " VERIFICATION — the numbers, and the integrity check"
echo "============================================================"
uv run python -u verify_original_pipeline.py 2>&1 | tee -a orig_verify.log

echo ""
echo "============================================================"
echo " ORIGINAL PIPELINE finished: $(date)"
echo "============================================================"
