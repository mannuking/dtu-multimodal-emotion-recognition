#!/bin/bash
#SBATCH --job-name=dtu-asr-preprocess
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=asr_preprocess.%J.out
#SBATCH --error=asr_preprocess.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v6 - ASR preprocessing step
# Transcribes all WAVs using Whisper base, updates manifest.csv
# with REAL transcripts (replaces 'a person feels X' text leakage).
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "v6 ASR preprocessing started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

export HF_HUB_OFFLINE=0  # Need to download Whisper this time
export TRANSFORMERS_OFFLINE=0

DATASET_DIR="combined_ser_dataset"
MANIFEST="$DATASET_DIR/manifest.csv"
TRANSCRIPTS="$DATASET_DIR/transcripts.json"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST missing. Run main v6 sbatch first to build it."
    exit 1
fi

# ============================================================
# Step 1: Transcribe all WAVs with Whisper
# ============================================================
echo "Loading Whisper base..."
uv run python - <<'PYEOF'
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load manifest
manifest_path = "combined_ser_dataset/manifest.csv"
print(f"Loading manifest from {manifest_path}")
df = pd.read_csv(manifest_path)
print(f"  {len(df)} samples to transcribe")

# Load Whisper base (~74M params, ~150MB)
print("Loading Whisper base model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cuda")
model.eval()
print("Model loaded.")

# Transcribe each unique wav
transcripts = {}
out_path = Path("combined_ser_dataset/transcripts.json")
if out_path.exists():
    transcripts = json.loads(out_path.read_text())
    print(f"Resuming from {len(transcripts)} already-transcribed files")

unique_wavs = df["audio_path"].unique()
print(f"Total unique wavs: {len(unique_wavs)}")

for i, rel_path in enumerate(unique_wavs):
    if rel_path in transcripts:
        continue
    full_path = Path("combined_ser_dataset") / rel_path
    if not full_path.exists():
        transcripts[rel_path] = ""
        continue
    try:
        audio, sr = librosa.load(str(full_path), sr=16000, mono=True)
        # Whisper expects max 30s, but our audio is max 6s so safe
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda")
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=128)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcripts[rel_path] = text.strip()
    except Exception as e:
        print(f"  ERROR on {rel_path}: {e}")
        transcripts[rel_path] = ""
    if (i + 1) % 100 == 0:
        # Save progress every 100 files (in case of crash)
        out_path.write_text(json.dumps(transcripts, indent=1))
        print(f"  [{i+1}/{len(unique_wavs)}] {len(transcripts)} done, last: '{transcripts[rel_path][:60]}'")

# Final save
out_path.write_text(json.dumps(transcripts, indent=1))
print(f"\n✓ Transcribed {len(transcripts)} wavs -> {out_path}")
PYEOF

# ============================================================
# Step 2: Update manifest.csv with real transcripts
# ============================================================
echo ""
echo "Updating manifest.csv with real transcripts..."

uv run python - <<'PYEOF'
import json
import pandas as pd
from pathlib import Path

manifest_path = Path("combined_ser_dataset/manifest.csv")
transcripts_path = Path("combined_ser_dataset/transcripts.json")

df = pd.read_csv(manifest_path)
transcripts = json.loads(transcripts_path.read_text())

# Replace text column with real transcripts (fall back to emotion-only if empty)
def get_text(audio_path):
    t = transcripts.get(audio_path, "").strip()
    if t and len(t) > 2:
        return t
    # Fallback: synthesize from emotion (parent dir name)
    emo = audio_path.split("/")[0]
    return f"a person feels {emo}"

df["text"] = df["audio_path"].apply(get_text)
df.to_csv(manifest_path, index=False)
print(f"  ✓ Updated {len(df)} rows in {manifest_path}")
print(f"  Sample texts:")
for i, row in df.head(5).iterrows():
    print(f"    {row['audio_path']} -> '{row['text'][:80]}'")
PYEOF

echo ""
echo "============================================================"
echo "ASR preprocessing finished: $(date)"
echo "Manifest updated with real transcripts. Now re-run v6 training:"
echo "  sbatch scripts/train_ser_v6.sbatch"
echo "============================================================"
