"""
rebuild_manifest.py - Rebuild metadata.csv from filesystem by parsing filenames.

Each dataset has a distinct filename pattern:
- CREMA-D: <4digit>_<speaker>_<emotion>_<level>.wav (e.g., 1073_IOM_DIS_XX.wav)
- RAVDESS: <modality>-<channel>-<emotion>-<intensity>-<statement>-<rep>-<actor>.wav
- SAVEE: <letter><emotion_letter><num>.wav (e.g., d03.wav for disgust)
- TESS: <speaker>_<word>_<emotion>.wav (e.g., OAF_back_angry.wav)

This script scans combined_ser_dataset/<emotion>/ for wav files and assigns
the original_dataset label by pattern matching.
"""
import os
import re
import csv
from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path("combined_ser_dataset")
OUT_CSV = DATASET_DIR / "metadata.csv"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# RAVDESS emotion codes: 01=neutral, 02=calm(not used), 03=happy, 04=sad, 05=angry,
# 06=fear, 07=disgust, 08=surprise
RAVDESS_EMOTION = {
    "01": "neutral", "03": "happy", "04": "sad", "05": "angry",
    "06": "fear", "07": "disgust", "08": "surprise",
}

# CREMA-D emotion codes in filename: ANG, DIS, FEA, HAP, NEU, SAD
CREMA_EMOTION = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy",
    "NEU": "neutral", "SAD": "sad",
}

# SAVEE emotion letters: a=anger, d=disgust, f=fear, h=happy, n=neutral, sa=sad, su=surprise
SAVEE_EMOTION = {
    "a": "angry", "d": "disgust", "f": "fear", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprise",
}


def parse_filename(filename: str):
    """Return (dataset, emotion, subject_id) or None."""
    name = filename.replace(".wav", "")
    # RAVDESS: 03-01-05-01-01-01-01
    m = re.match(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$", name)
    if m:
        emotion_code = m.group(3)
        actor = m.group(7)
        emo = RAVDESS_EMOTION.get(emotion_code)
        if emo:
            return ("RAVDESS", emo, f"actor_{actor}")
    # CREMA-D: 1073_IOM_DIS_XX
    m = re.match(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})$", name)
    if m:
        speaker = m.group(2)
        emotion_code = m.group(3)
        emo = CREMA_EMOTION.get(emotion_code)
        if emo:
            return ("CREMA-D", emo, f"speaker_{speaker}")
    # SAVEE: DC_a01.wav, JE_d03.wav, JK_f08.wav, KL_h05.wav, DC_n06.wav, DC_sa01.wav, DC_su01.wav
    # Speaker prefix (2 letters) + emotion code
    m = re.match(r"^([A-Z]{2})_(a|d|f|h|n|sa|su)(\d+)$", name)
    if m:
        speaker = m.group(1)
        prefix = m.group(2)
        emo = SAVEE_EMOTION.get(prefix)
        if emo:
            return ("SAVEE", emo, f"SAVEE_{speaker}")
    # TESS: OAF_back_angry, YAF_date_disgust
    m = re.match(r"^(OAF|YAF)_(\w+)_(\w+)$", name)
    if m:
        speaker = m.group(1)
        emo = m.group(3).lower()
        if emo in EMOTIONS:
            return ("TESS", emo, speaker)
    return None


def main():
    rows = []
    skipped = 0
    for emo_dir in sorted(DATASET_DIR.iterdir()):
        if not emo_dir.is_dir():
            continue
        # Trust the directory name = emotion (since build_combined_ser_dataset used that)
        emo_from_dir = emo_dir.name.lower()
        if emo_from_dir not in EMOTIONS:
            continue
        for wav_path in sorted(emo_dir.glob("*.wav")):
            result = parse_filename(wav_path.name)
            if result is None:
                skipped += 1
                continue
            dataset, parsed_emo, subject_id = result
            # Use directory emotion as ground truth (it's how files were placed)
            final_emo = emo_from_dir
            gender = ""
            if dataset == "RAVDESS":
                # RAVDESS filename: 03-01-05-01-01-01-01 (last digit = actor number)
                actor = wav_path.stem.split("-")[-1]
                gender = "male" if int(actor) % 2 == 1 else "female"
            elif dataset == "TESS":
                gender = "female"  # both speakers are female
            elif dataset == "SAVEE":
                gender = "male"  # all SAVEE speakers are male
            elif dataset == "CREMA-D":
                speaker_code = wav_path.stem.split("_")[1]
                # CREMA speaker code: F=female, M=male, I=child
                gender_code = speaker_code[0]
                if gender_code == "F":
                    gender = "female"
                elif gender_code == "M":
                    gender = "male"
                else:
                    gender = "child"
            rows.append({
                "wav_path": str(wav_path),
                "emotion": final_emo,
                "dataset": dataset,
                "subject": subject_id,
                "gender": gender,
            })

    print(f"Found {len(rows)} wav files, skipped {skipped}")

    # Class distribution
    by_class = defaultdict(int)
    by_dataset = defaultdict(int)
    by_dataset_class = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_class[r["emotion"]] += 1
        by_dataset[r["dataset"]] += 1
        by_dataset_class[r["dataset"]][r["emotion"]] += 1

    print("\nClass distribution:")
    for emo in EMOTIONS:
        print(f"  {emo:9s}: {by_class.get(emo, 0)}")
    print("\nDataset distribution:")
    for ds, n in sorted(by_dataset.items()):
        print(f"  {ds:8s}: {n}")
    print("\nDataset x Emotion matrix:")
    print(f"  {'dataset':10s}  " + "  ".join(f"{e[:4]:>6s}" for e in EMOTIONS))
    for ds in sorted(by_dataset.keys()):
        line = f"  {ds:10s}  " + "  ".join(f"{by_dataset_class[ds].get(e, 0):>6d}" for e in EMOTIONS)
        print(line)

    # Write CSV
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "emotion", "dataset", "subject", "gender"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n\u2705 Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()