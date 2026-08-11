#!/bin/bash
#SBATCH --job-name=dtu-ser-v7
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --output=ser_v7.%J.out
#SBATCH --error=ser_v7.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v7 - V5-style pipeline targeting 75-80% accuracy
# ============================================================
# Built on proven v5 architecture (last 8 wav2vec2 layers unfrozen,
# SupCon, EMA, SpecAugment, TTA=8).
#
# Three robustness improvements over v5:
#   1. Balanced batch sampler — prevents majority-class collapse
#   2. Mixed-precision (AMP) — ~1.5x training speedup
#   3. Seed averaging on inference — picks the best of 3 seeds
#
# Expected runtime: ~5-6 hours per seed on 1x A100 (faster than v5 due to AMP).
# 8-hour budget is safe for one seed. For 3-seed ensemble: submit 3 jobs sequentially.
#
# Usage (from HPC login, NEVER compute on login):
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/train_ser_v7.sbatch 42    # seed=42
#   sbatch scripts/train_ser_v7.sbatch 44    # seed=44 (after first finishes)
#   sbatch scripts/train_ser_v7.sbatch 46    # seed=46 (after second finishes)
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

SEED="${1:-42}"

echo "============================================================"
echo "v7 SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Seed: $SEED"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# CRITICAL: HuggingFace offline mode (compute nodes have flaky outbound)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Mixed-precision for ~1.5x speedup
export V7_USE_AMP=1

# Convert manifest.csv (v6 ASR format) to metadata.csv (v5/v7 format)
uv run python - <<'PYEOF'
import pandas as pd
import os
from pathlib import Path

src = Path("combined_ser_dataset/manifest.csv")
dst = Path("combined_ser_dataset/metadata.csv")

if not src.exists():
    raise SystemExit(f"ERROR: {src} not found. Run ASR preprocessing first.")

df = pd.read_csv(src)
print(f"   loaded {len(df)} rows from manifest.csv")

# Build metadata.csv in v5/v7 format
# Need: wav_path (relative to combined_ser_dataset), emotion
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# label is int 0-6, emotion is in audio_path directory
def get_emotion(row):
    # Prefer text-derived emotion if path is consistent
    emo = row["audio_path"].split("/")[0].lower()
    if emo in EMOTIONS:
        return emo
    # Fallback to label
    if "label" in df.columns and 0 <= int(row["label"]) < 7:
        return EMOTIONS[int(row["label"])]
    return emo

out = pd.DataFrame({
    "wav_path": df["audio_path"].apply(lambda p: p),  # relative to combined_ser_dataset
    "emotion": df.apply(get_emotion, axis=1),
    "subject": df["subject"].astype(str),
    "split": df["split"],
})
out.to_csv(dst, index=False)
print(f"   wrote {len(out)} rows to {dst}")
print(f"   emotion distribution:
{out['emotion'].value_counts()}")
PYEOF

SEED="$SEED" uv run python -u train_ser_v7.py --seed "$SEED" --n-epochs 60
RET=$?

echo "============================================================"
echo "v7 seed=$SEED done (exit=$RET): $(date)"
echo "============================================================"
exit $RET
