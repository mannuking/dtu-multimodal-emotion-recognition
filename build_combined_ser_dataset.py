"""
build_combined_ser_dataset.py — Build the unified SER dataset manifest.

Combines 4 emotion speech datasets into one labeled CSV manifest:
  - SAVEE, CREMA-D, TESS, RAVDESS

The paper's Sec 4.1 lists these as the 4 source datasets, ~13,000 samples total
after harmonization (we get a subset due to dataset availability).

Output: combined_ser_dataset/metadata.csv with columns:
    wav_path, emotion, dataset, subject, gender
"""

import os
import csv
from pathlib import Path
from collections import defaultdict


EMOTION_MAP = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "pleasant_surprise": "surprise",
    "pleasant_surprised": "surprise",
    "ps": "surprise",
    "neutral": "neutral",
    "calm": "neutral",
}

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_filename(filepath: str, source: str) -> dict | None:
    """Parse emotion + subject + gender from filename per dataset convention.

    Returns dict with keys: emotion, subject, gender — or None if not parseable.
    """
    fname = Path(filepath).stem
    f_lower = fname.lower()

    if source == "tess":
        # TESS: YAF_disgust / OAF_angry (folder name is most reliable)
        folder = Path(filepath).parent.name.lower()
        speaker = "YAF" if folder.startswith("yaf") else "OAF"
        # Emotion is everything after speaker prefix
        emo_part = folder.replace("yaf_", "").replace("oaf_", "")
        # Normalize
        emo_norm = EMOTION_MAP.get(emo_part, EMOTION_MAP.get(emo_part.replace("pleasant_surprised", "pleasant_surprise")))
        if emo_norm is None:
            return None
        return {
            "emotion": emo_norm,
            "subject": speaker,
            "gender": "female",
        }

    if source == "crema":
        # CREMA-D: 1001_IEO_ANG_HI.wav
        # format: <actor_id>_<sentence>_<emotion>_<intensity>
        parts = fname.split("_")
        if len(parts) < 4:
            return None
        actor_id = parts[0]
        emo_code = parts[2].lower()
        emo_norm = EMOTION_MAP.get(emo_code)
        if emo_norm is None:
            return None
        # CREMA actors: 1001-1080 male, 2001-2080 female (approx)
        try:
            actor_num = int(actor_id)
            gender = "female" if 2000 <= actor_num < 3000 else "male"
        except ValueError:
            gender = "unknown"
        return {
            "emotion": emo_norm,
            "subject": f"crema_{actor_id}",
            "gender": gender,
        }

    if source == "savee":
        # SAVEE: DC_a01.wav (a=anger, d=disgust, f=fear, h=happy, n=neutral, sa=sad, su=surprise)
        # Subject codes: DC, JE, JK, KL
        emo_codes = {
            "a": "angry", "d": "disgust", "f": "fear", "h": "happy",
            "n": "neutral", "sa": "sad", "su": "surprise",
        }
        # Format: <speaker>_<code><num>.wav  e.g. DC_a01, KL_sa15
        for sp in ["DC", "JE", "JK", "KL"]:
            if f_lower.startswith(sp.lower() + "_"):
                rest = f_lower[len(sp) + 1:]
                # strip leading numbers
                code = "".join(c for c in rest if c.isalpha())
                if code in emo_codes:
                    return {
                        "emotion": emo_codes[code],
                        "subject": f"savee_{sp}",
                        "gender": "male",
                    }
        return None

    if source == "ravdess":
        # RAVDESS: 03-01-01-01-01-01.wav
        # Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor
        parts = fname.split("-")
        if len(parts) < 7:
            return None
        emo_code = parts[2]
        emo_map = {
            "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fear", "07": "disgust", "08": "surprise",
        }
        emo_norm = EMOTION_MAP.get(emo_map.get(emo_code, ""))
        if emo_norm is None:
            return None
        actor_id = parts[6]
        gender = "female" if int(actor_id) % 2 == 0 else "male"
        return {
            "emotion": emo_norm,
            "subject": f"ravdess_{actor_id}",
            "gender": gender,
        }

    return None


def scan_directory(root: str, source: str) -> list[dict]:
    """Walk root directory, parse each wav file, return list of records."""
    rows = []
    for wav_path in Path(root).rglob("*.wav"):
        info = parse_filename(str(wav_path), source)
        if info is None:
            continue
        rows.append({
            "wav_path": str(wav_path.resolve()),
            "emotion": info["emotion"],
            "dataset": source,
            "subject": info["subject"],
            "gender": info["gender"],
        })
    return rows


def main():
    base = Path(__file__).parent
    out_csv = base / "combined_ser_dataset" / "metadata.csv"
    out_csv.parent.mkdir(exist_ok=True)

    all_rows = []
    print("Scanning datasets...")

    # TESS
    tess_root = base / "tess_wavs"
    if tess_root.exists():
        rows = scan_directory(str(tess_root), "tess")
        print(f"  TESS: {len(rows)} files")
        all_rows.extend(rows)

    # CREMA-D
    crema_root = base / "crema_dataset" / "AudioWAV"
    if crema_root.exists():
        rows = scan_directory(str(crema_root), "crema")
        print(f"  CREMA-D: {len(rows)} files")
        all_rows.extend(rows)

    # SAVEE
    savee_root = base / "savee_dataset"
    if savee_root.exists():
        rows = scan_directory(str(savee_root), "savee")
        print(f"  SAVEE: {len(rows)} files")
        all_rows.extend(rows)

    # RAVDESS
    ravdess_root = base / "ravdess_dataset"
    if ravdess_root.exists():
        # RAVDESS audio is in subdirs like "Actor_01"
        rows = scan_directory(str(ravdess_root), "ravdess")
        print(f"  RAVDESS: {len(rows)} files")
        all_rows.extend(rows)

    # Dedupe by wav_path
    seen = set()
    deduped = []
    for r in all_rows:
        if r["wav_path"] in seen:
            continue
        seen.add(r["wav_path"])
        deduped.append(r)
    print(f"\nTotal after dedup: {len(deduped)}")

    # Emotion distribution
    dist = defaultdict(int)
    for r in deduped:
        dist[r["emotion"]] += 1
    print("Class distribution:")
    for emo in EMOTIONS:
        print(f"  {emo}: {dist.get(emo, 0)}")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wav_path", "emotion", "dataset", "subject", "gender"])
        writer.writeheader()
        writer.writerows(deduped)
    print(f"\n✓ Wrote {out_csv}")


if __name__ == "__main__":
    main()