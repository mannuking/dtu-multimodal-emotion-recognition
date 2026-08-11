"""
manifest_to_metadata.py - Convert ASR manifest.csv to v5/v7 metadata.csv.

Reads combined_ser_dataset/manifest.csv (v6 ASR format with columns:
audio_path, text, label, subject, split, facial_path) and writes
combined_ser_dataset/metadata.csv in v5/v7 format (wav_path, emotion,
subject, split).
"""
import pandas as pd
from pathlib import Path

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

src = Path("combined_ser_dataset/manifest.csv")
dst = Path("combined_ser_dataset/metadata.csv")

if not src.exists():
    raise SystemExit(f"ERROR: {src} not found. Run ASR preprocessing first.")

print(f"   loading {src}")
df = pd.read_csv(src)
print(f"   loaded {len(df)} rows from manifest.csv")

# Map audio_path -> emotion (parent directory)
def get_emotion(audio_path):
    emo = str(audio_path).split("/")[0].lower()
    return emo if emo in EMOTIONS else "unknown"

out = pd.DataFrame({
    "wav_path": df["audio_path"].astype(str),
    "emotion": df["audio_path"].apply(get_emotion),
    "subject": df["subject"].astype(str),
    "split": df["split"].astype(str),
})

# Filter out unknown
unknown_count = (out["emotion"] == "unknown").sum()
if unknown_count > 0:
    print(f"   WARNING: {unknown_count} rows filtered (unknown emotion)")
    out = out[out["emotion"] != "unknown"].reset_index(drop=True)

out.to_csv(dst, index=False)
print(f"   wrote {len(out)} rows to {dst}")
print(f"   emotion distribution:")
for emo, count in out["emotion"].value_counts().items():
    print(f"     {emo}: {count}")
print(f"   split distribution:")
for split, count in out["split"].value_counts().items():
    print(f"     {split}: {count}")
