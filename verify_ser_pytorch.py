"""
verify_ser_pytorch.py — Integrity test for the PyTorch SER (1D-CNN).

Loads the trained SER checkpoint (ser_best.pt) + label encoder, runs
inference on a stratified held-out test split (SEED=42, 80/10/20), and
reports real metrics against ground-truth labels from the combined SER
corpus.

This is NOT retraining. The checkpoints are loaded as-is and evaluated on
real test data drawn from the same combined dataset used in the manuscript.
"""

import os
import sys
import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

SEED = 42
NUM_CLASSES = 7
EMOTION_ORDER = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_ORDER_LOWER = [e.lower() for e in EMOTION_ORDER]
TARGET_SR = 16000
OFFSET_S = 0.5
DUR_S = 3.0
CHECKPOINT_DIR = "model_checkpoints"
SER_COMBINED_DIR = "combined_ser_dataset"


def trim_silence(audio, thresh_scale=3):
    """Match the training-time pipeline."""
    frame_length = 2048
    hop = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
    if len(rms) == 0:
        return audio
    thr = (rms.max() - rms.min()) / thresh_scale
    frames = np.nonzero(rms > thr)[0]
    if len(frames) == 0:
        return audio
    start = max(0, frames[0] * hop)
    end = min(len(audio), frames[-1] * hop)
    return audio[start:end]


def preprocess_audio(file_path):
    """Match training-time preprocessing."""
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    y = trim_silence(y)
    y = y[int(OFFSET_S * sr):]
    target = int(DUR_S * sr)
    if len(y) > target:
        y = y[:target]
    else:
        y = np.pad(y, (0, target - len(y)))
    return y


def extract_features(y, sr=TARGET_SR, n_mfcc=40):
    """
    Match training-time features:
    - hop=192 samples (12 ms), win=400 samples (25 ms)
    - ZCR + RMS + energy + entropy + MFCC40 = 44 features/frame
    - 251 frames × 44 = 11,044 dims per sample
    """
    hop = 192
    win = 400
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win, hop_length=hop).T
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=win, hop_length=hop).T
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).T
    energy = rms ** 2 * win
    prob = energy / (np.sum(energy) + 1e-8)
    entropy = -prob * np.log2(prob + 1e-12)
    feats = np.concatenate([zcr, rms, energy, entropy, mfcc], axis=1)
    return feats.flatten()


def load_ser_model(device, in_len=11044, num_classes=7):
    """Load the PyTorch SER 1D-CNN."""
    from train_ser_pytorch import SER1DCNN
    ser_path = os.path.join(CHECKPOINT_DIR, "ser_best.pt")
    if not os.path.exists(ser_path):
        print(f"      ❌ SER checkpoint not found at {ser_path}")
        sys.exit(1)
    model = SER1DCNN(in_len=in_len, num_classes=num_classes).to(device)
    state = torch.load(ser_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    print("=" * 70)
    print("SER INTEGRITY TEST — verifying the PyTorch 1D-CNN on real audio")
    print("=" * 70)
    print(f"PyTorch:     {torch.__version__}")
    print(f"Librosa:     {librosa.__version__}")
    print(f"Seed:        {SEED}")
    print()

    # 1. Load the manifest
    print("[1/5] Loading combined_ser_dataset/metadata.csv ...")
    df = pd.read_csv(os.path.join(SER_COMBINED_DIR, "metadata.csv"))
    # Normalize column name
    if "filepath" not in df.columns and "wav_path" in df.columns:
        df = df.rename(columns={"wav_path": "filepath"})
    if "filepath" not in df.columns:
        print(f"      ❌ Manifest needs 'filepath' or 'wav_path' column. Found: {list(df.columns)}")
        sys.exit(1)
    print(f"      Total samples in manifest: {len(df)}")
    print(f"      Class distribution:\n{df['emotion'].value_counts().to_string()}\n")

    # 2. Build the 80/10/20 stratified split
    print("[2/5] Building stratified 80/10/20 split with SEED=42 ...")
    label_map = {e: i for i, e in enumerate(EMOTION_ORDER_LOWER)}
    y_all = df["emotion"].str.lower().map(label_map).values
    X_paths = df["filepath"].values

    Xtv, Xte, ytv, yte = train_test_split(
        X_paths, y_all, test_size=0.10, stratify=y_all, random_state=SEED
    )
    Xtr, Xva, ytr, yva = train_test_split(
        list(Xtv), list(ytv), test_size=0.111, stratify=ytv, random_state=SEED
    )
    print(f"      Train: {len(Xtr)}, Val: {len(Xva)}, Test: {len(Xte)}")
    print(f"      Test class counts: {dict(zip(*np.unique(yte, return_counts=True)))}\n")

    # 3. Load trained SER checkpoint
    print("[3/5] Loading ser_best.pt checkpoint (PyTorch) ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")
    ser_path = os.path.join(CHECKPOINT_DIR, "ser_best.pt")
    if not os.path.exists(ser_path):
        print(f"      ❌ SER checkpoint not found at {ser_path}")
        sys.exit(1)
    model = load_ser_model(device)
    size_mb = os.path.getsize(ser_path) / 1024 / 1024
    print(f"      ✅ Loaded {size_mb:.1f} MB model")

    # 4. Run inference on the test set
    print(f"[4/5] Running inference on test split ({len(Xte)} samples) ...")
    preds = []
    probs_all = []
    t0 = time.time()
    failed = 0

    for i, (rel_path, y_true) in enumerate(zip(Xte, yte)):
        # Build full path — handle both absolute paths and relative paths
        if os.path.isabs(rel_path):
            full_path = rel_path
        else:
            full_path = os.path.join(SER_COMBINED_DIR, rel_path)
        try:
            y_audio = preprocess_audio(full_path)
            feats = extract_features(y_audio)
            if len(feats) != 11044:
                raise ValueError(f"feature length {len(feats)} != 11044")
            x = torch.as_tensor(feats, dtype=torch.float32).view(1, 1, -1).to(device)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=-1).cpu().numpy()[0]
            preds.append(int(np.argmax(probs)))
            probs_all.append(probs.tolist())
        except Exception as e:
            if failed < 3:
                print(f"      ⚠️  fail on {rel_path}: {type(e).__name__}: {e}")
            failed += 1
            preds.append(-1)
            probs_all.append([0.0] * NUM_CLASSES)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(Xte) - i - 1) / rate
            print(f"      {i+1}/{len(Xte)}  ({elapsed:.1f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"      ✅ Inference complete: {len(Xte)} samples in {elapsed:.1f}s "
          f"({elapsed/len(Xte)*1000:.1f} ms/sample, {failed} failed)\n")

    # 5. Compute metrics
    print("[5/5] Computing metrics ...")
    preds = np.array(preds)
    yte = np.array(yte)
    mask = preds != -1
    acc = accuracy_score(yte[mask], preds[mask])
    macro_f1 = f1_score(yte[mask], preds[mask], average="macro")
    weighted_f1 = f1_score(yte[mask], preds[mask], average="weighted")
    cm = confusion_matrix(yte[mask], preds[mask], labels=list(range(NUM_CLASSES)))

    print(f"      Accuracy:           {acc:.4f}  ({acc*100:.2f}%)")
    print(f"      Macro F1:           {macro_f1:.4f}")
    print(f"      Weighted F1:        {weighted_f1:.4f}\n")
    print("      Per-class report:")
    print(classification_report(
        yte[mask], preds[mask],
        labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_ORDER,
        digits=4, zero_division=0
    ))
    print("      Confusion matrix (rows = true, cols = pred):")
    print(f"      {'':12s}" + "".join(f"{e:>10s}" for e in EMOTION_ORDER))
    for i, row_name in enumerate(EMOTION_ORDER):
        print(f"      {row_name:12s}" + "".join(f"{cm[i,j]:>10d}" for j in range(NUM_CLASSES)))

    # 6. Persist results
    out = {
        "model": "SER (PyTorch 1D-CNN)",
        "checkpoint": ser_path,
        "test_samples": int(len(Xte)),
        "failed": int(failed),
        "elapsed_seconds": round(elapsed, 2),
        "ms_per_sample": round(elapsed / len(Xte) * 1000, 2),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_report": classification_report(
            yte[mask], preds[mask],
            labels=list(range(NUM_CLASSES)),
            target_names=EMOTION_ORDER,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "paper_claim_accuracy": 0.8009,
        "paper_claim_macro_f1": 0.73,
        "delta_accuracy": round(float(acc) - 0.8009, 4),
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/ser_verification.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Paper claim: 80.09% accuracy, 0.73 macro F1")
    print(f"  Measured:    {acc*100:.2f}% accuracy, {macro_f1:.4f} macro F1")
    delta = acc - 0.8009
    direction = "higher" if delta > 0 else "lower"
    print(f"  Δ accuracy:  {delta:+.4f} ({direction} than paper)")
    print(f"\nResults saved to reports/ser_verification.json")

    return 0 if acc >= 0.70 else 2


if __name__ == "__main__":
    sys.exit(main())