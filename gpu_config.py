"""
gpu_config.py — Centralised configuration for the DTU multimodal pipeline.

This module was previously missing from the repo (referenced via
`from gpu_config import *` in train_*.py but never committed). It is now
generated from `config.py` and the actual constants used by each training
script. Keep this file in sync if you add new datasets or change paths.

Usage (per-platform):
  - Linux HPC + CUDA:  uv sync --extra cuda
  - macOS Apple Silicon: uv sync --extra macos
"""
import os

# ===== Numeric =====
NUM_CLASSES = 7
SEED = 42

# ===== Paths =====
CHECKPOINT_DIR = "model_checkpoints"
SER_FEATURES_DIR = "ser_feature_output"
SER_COMBINED_DIR = "combined_ser_dataset"

# Local MobileBERT copy for offline TER inference
LOCAL_MOBILEBERT_PYTORCH = "mobilebert_pytorch"
# Alias kept for compatibility with older code that referenced the wrong name
LOCAL_MOBILEBERT_PATH = LOCAL_MOBILEBERT_PYTORCH

# ===== Audio =====
TARGET_SR = 16000
OFFSET_S = 0.5
DUR_S = 3.0

# ===== Image =====
IMG_SIZE = (224, 224)

# ===== FER2013 =====
# Set FER_TRAIN_DIR and FER_TEST_DIR to your FER2013 download location.
# Default: relative to project root (expects fer2013/train and fer2013/test subdirs).
FER_TRAIN_DIR = os.environ.get("FER_TRAIN_DIR", "fer2013/train")
FER_TEST_DIR = os.environ.get("FER_TEST_DIR", "fer2013/test")

# ===== Text (DailyDialog / MELD / EmoryNLP / IEMOCAP) =====
# Set TEXT_TRAIN_CSV and TEXT_VAL_CSV to your preprocessed CSV file paths.
# Expected CSV columns: text,emotion (or any columns auto-detected by load_text_csv).
TEXT_TRAIN_CSV = os.environ.get("TEXT_TRAIN_CSV", "data/text_train.csv")
TEXT_VAL_CSV = os.environ.get("TEXT_VAL_CSV", "data/text_val.csv")

# ===== Triplet manifest =====
TRIPLETS_MANIFEST = os.environ.get("TRIPLETS_MANIFEST", "triplets_manifest.csv")

# ===== Emotion taxonomy =====
EMOTION_ORDER = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_ORDER_LOWER = [e.lower() for e in EMOTION_ORDER]