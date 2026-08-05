# config.py - Configuration matching your training setup
import os

# Model configuration
NUM_CLASSES = 7
EMOTION_ORDER = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Directories
CHECKPOINT_DIR = "model_checkpoints"
SER_FEATURES_DIR = "ser_feature_output"
SER_COMBINED_DIR = "combined_ser_dataset"

# Audio processing
TARGET_SR = 16000
OFFSET_S = 0.5  # Skip first 0.5 seconds
DUR_S = 3.0     # Use 3 seconds total

# Image processing
IMG_SIZE = (224, 224)

# Training
SEED = 42
TRIPLETS_MANIFEST = "triplets_manifest.csv"