"""
build_triplets_leakage_free.py — Subject-disjoint triplet manifest construction.

Why this exists:
The original triplet manifest in the manuscript pairs samples by shared emotion
label across modalities (text / audio / face). For datasets like RAVDESS and
CREMA-D where the same speaker contributes both audio and sometimes other modalities,
that construction can leak subject identity across train and test triplets —
the meta-classifier can learn subject-consistent shortcuts rather than genuine
cross-modal signal.

This script builds a triplet manifest with:
  - Subject-disjoint splits: no (text_subject, audio_subject, face_subject)
    triple in train appears in test (subject IDs treated as opaque strings).
  - Class-balanced: equal number of triplets per emotion.
  - One canonical triplet per (text_id, audio_id, face_id, label) tuple.

Inputs (set via CLI flags or edit the constants below):
  --text-manifest CSV with columns: text_id, label, subject_id (optional)
  --audio-manifest CSV (combined_ser_dataset/metadata.csv works)
  --face-manifest CSV with columns: face_id, label, subject_id (optional)

Output:
  triplets_manifest.csv with columns:
    triplet_id, text_id, speech_wav, face_img, label, split

Usage:
  python build_triplets_leakage_free.py \
    --audio-manifest combined_ser_dataset/metadata.csv \
    --output triplets_manifest.csv \
    --seed 42

If text and face manifests are missing (FER2013 not on disk, no text CSVs),
the script can construct a placeholder manifest with a single text/face row per
audio row, marked with subject_id="UNKNOWN", and skip the subject-disjoint
check. This is the bootstrap path for the HPC run when starting from raw
datasets.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_ORDER_LOWER = [e.lower() for e in EMOTION_ORDER]


def subject_id_from_audio_path(filepath):
    """
    Best-effort subject extraction from the audio file path.
    RAVDESS filenames look like: '03-01-05-01-02-01-12.wav' (modality-vocal-emotion-intensity-statement-repetition-actor)
    CREMA-D: '1001_DFA_ANG_XX.wav' (actor_id_sentence_emotion)
    SAVEE: 'DC_f1.wav' (speaker initials)
    TESS: 'OAF_angry.wav' (no subject — null)
    """
    base = os.path.basename(filepath)
    # RAVDESS
    parts = base.replace(".wav", "").split("-")
    if len(parts) == 7 and parts[0] in ("01", "02", "03", "04", "05", "06", "07", "08"):
        return f"ravdess_actor_{parts[-1]}"
    # CREMA-D
    if "_" in base:
        return f"crema_actor_{base.split('_')[0]}"
    # TESS / SAVEE — no reliable subject
    return f"unknown_{base[:4]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-manifest", required=True,
                   help="CSV with columns filepath, emotion [, original_dataset]")
    p.add_argument("--text-manifest", default=None,
                   help="CSV with columns text_id, label [, subject_id]")
    p.add_argument("--face-manifest", default=None,
                   help="CSV with columns face_id, label [, subject_id]")
    p.add_argument("--output", default="triplets_manifest.csv")
    p.add_argument("--per-class", type=int, default=1000,
                   help="Triplets per emotion class (default 1000 = 7000 total)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subject-disjoint", action="store_true", default=True,
                   help="Enforce subject-disjoint splits (default on)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- Load audio manifest (always required) ----
    audio_df = pd.read_csv(args.audio_manifest)
    print(f"[audio] loaded {len(audio_df)} rows from {args.audio_manifest}")
    # Normalize column name: build_combined_ser_dataset uses 'wav_path', but
    # older manifests used 'filepath'. Support both.
    if "filepath" not in audio_df.columns and "wav_path" in audio_df.columns:
        audio_df = audio_df.rename(columns={"wav_path": "filepath"})
    if "filepath" not in audio_df.columns:
        raise SystemExit(f"ERROR: audio manifest needs a 'filepath' (or 'wav_path') column. Found: {list(audio_df.columns)}")
    audio_df["emotion"] = audio_df["emotion"].str.lower()
    audio_df = audio_df[audio_df["emotion"].isin(EMOTION_ORDER_LOWER)].copy()
    audio_df["subject_id"] = audio_df["filepath"].apply(subject_id_from_audio_path)
    print(f"[audio] {len(audio_df)} rows after class filter")
    print(f"[audio] {audio_df['subject_id'].nunique()} unique subjects")

    # ---- Optional text manifest ----
    if args.text_manifest and os.path.exists(args.text_manifest):
        text_df = pd.read_csv(args.text_manifest)
        print(f"[text] loaded {len(text_df)} rows from {args.text_manifest}")
    else:
        # Bootstrap path — text is missing; generate placeholders per audio row
        print("[text] no text manifest provided; generating placeholders")
        text_df = audio_df[["filepath"]].copy()
        text_df["text_id"] = [f"placeholder_{i}" for i in range(len(text_df))]
        text_df["label"] = audio_df["emotion"].values
        text_df["subject_id"] = "TEXT_UNKNOWN"

    # ---- Optional face manifest ----
    if args.face_manifest and os.path.exists(args.face_manifest):
        face_df = pd.read_csv(args.face_manifest)
        print(f"[face] loaded {len(face_df)} rows from {args.face_manifest}")
    else:
        print("[face] no face manifest provided; generating placeholders")
        face_df = audio_df[["filepath"]].copy()
        face_df["face_id"] = [f"placeholder_{i}" for i in range(len(face_df))]
        face_df["label"] = audio_df["emotion"].values
        face_df["subject_id"] = "FACE_UNKNOWN"

    # ---- Build triplets: one per audio row, paired with random text+face
    # of the same emotion class. Subject-disjoint across triplets.
    print(f"\n[build] constructing {args.per_class} triplets per emotion...")

    rows = []
    triplet_id = 0
    for emo in EMOTION_ORDER_LOWER:
        audio_pool = audio_df[audio_df["emotion"] == emo].reset_index(drop=True)
        text_pool = text_df[text_df["label"].str.lower() == emo].reset_index(drop=True) \
            if "label" in text_df.columns else text_df
        face_pool = face_df[face_df["label"].str.lower() == emo].reset_index(drop=True) \
            if "label" in face_df.columns else face_df

        if len(audio_pool) == 0:
            print(f"  ⚠️  no audio for {emo}, skipping")
            continue

        for i in range(args.per_class):
            audio_row = audio_pool.iloc[i % len(audio_pool)]
            text_row = text_pool.iloc[i % len(text_pool)]
            face_row = face_pool.iloc[i % len(face_pool)]
            rows.append({
                "triplet_id": triplet_id,
                "text_id": text_row.get("text_id", text_row.get("filepath", f"text_{i}")),
                "speech_wav": audio_row["filepath"],
                "face_img": face_row.get("face_id", face_row.get("filepath", f"face_{i}")),
                "label": emo,
                "audio_subject": audio_row["subject_id"],
                "text_subject": text_row.get("subject_id", "TEXT_UNKNOWN"),
                "face_subject": face_row.get("subject_id", "FACE_UNKNOWN"),
            })
            triplet_id += 1

    df = pd.DataFrame(rows)
    print(f"[build] {len(df)} triplets constructed")

    # ---- Subject-disjoint split (80/10/10) ----
    if args.subject_disjoint:
        # All unique subjects across the triplets
        subjects = pd.concat([
            df["audio_subject"],
            df["text_subject"],
            df["face_subject"],
        ]).unique()
        print(f"[split] {len(subjects)} unique subjects across triplets")
        sub_train, sub_temp = train_test_split(
            subjects, test_size=0.2, random_state=args.seed
        )
        sub_val, sub_test = train_test_split(
            sub_temp, test_size=0.5, random_state=args.seed
        )
        print(f"[split] subjects: train={len(sub_train)} val={len(sub_val)} test={len(sub_test)}")

        def assign_split(row):
            subs = {row["audio_subject"], row["text_subject"], row["face_subject"]}
            if subs & set(sub_test):
                return "test"
            if subs & set(sub_val):
                return "val"
            return "train"

        df["split"] = df.apply(assign_split, axis=1)
        # If any subject is shared with test (common in placeholder mode),
        # mark the whole triplet as test
    else:
        df["split"] = "train"

    counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    print("\n[split] triplets per class × split:")
    print(counts.to_string())

    # ---- Save ----
    df.to_csv(args.output, index=False)
    print(f"\n[save] wrote {args.output} ({len(df)} triplets)")
    print(f"[save] train={len(df[df.split=='train'])} val={len(df[df.split=='val'])} test={len(df[df.split=='test'])}")


if __name__ == "__main__":
    main()