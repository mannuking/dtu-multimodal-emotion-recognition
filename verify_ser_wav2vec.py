"""
verify_ser_wav2vec.py \u2014 Integrity test for the wav2vec2 SER.

Loads ser_best.pt + label encoder, runs inference on a held-out test split,
reports real metrics against ground-truth labels.
"""
import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

SEED = 42
TARGET_SR = 16000
MAX_S = 6.0
CHECKPOINT_DIR = "model_checkpoints"
SER_COMBINED_DIR = "combined_ser_dataset"

# Import the model class from the training script
sys.path.insert(0, ".")
from train_ser_wav2vec import Wav2Vec2SER


def load_audio(path, max_seconds=MAX_S, sr=TARGET_SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_samples = int(max_seconds * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    elif len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)), mode="constant")
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max()
    return y.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ser_ckpt = os.path.join(CHECKPOINT_DIR, "ser_best.pt")
    ser_encoder = os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl")
    test_idx_path = os.path.join(CHECKPOINT_DIR, "ser_test_indices.pkl")

    if not os.path.exists(ser_ckpt):
        print(f"ERROR: {ser_ckpt} not found. Run train_ser_wav2vec.py first.")
        sys.exit(1)
    if not os.path.exists(ser_encoder):
        print(f"ERROR: {ser_encoder} not found.")
        sys.exit(1)
    if not os.path.exists(test_idx_path):
        print(f"ERROR: {test_idx_path} not found. Run training first to save test indices.")
        sys.exit(1)

    with open(ser_encoder, "rb") as f:
        label_encoder = pickle.load(f)
    with open(test_idx_path, "rb") as f:
        meta = pickle.load(f)
    test_idx = np.array(meta["test_indices"])
    classes = label_encoder.classes_
    num_classes = len(classes)
    print(f"   Classes: {list(classes)}")

    # Load manifest
    manifest_csv = os.path.join(SER_COMBINED_DIR, "metadata.csv")
    df = pd.read_csv(manifest_csv)
    if "wav_path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "wav_path"})
    if "emotion" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "emotion"})

    # Filter to test set
    df_test = df.iloc[test_idx].reset_index(drop=True)
    print(f"   Test set: {len(df_test)} samples")

    # Load model
    print(f"   Loading {ser_ckpt}...")
    model = Wav2Vec2SER(num_classes=num_classes).to(device)
    state = torch.load(ser_ckpt, map_location=device, weights_only=False)
    # Remove _orig_mod prefix if present (torch.compile)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    print(f"   Model loaded.")

    # Predict
    y_true = []
    y_pred = []
    times = []
    for i, row in df_test.iterrows():
        path = row["wav_path"]
        if not os.path.exists(path):
            continue
        x = load_audio(path)
        x_tensor = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            logits = model(x_tensor)
        times.append(time.time() - t0)
        pred = logits.argmax(1).item()
        # Map label string to int
        emo = str(row["emotion"]).lower()
        if emo in classes:
            y_true.append(label_encoder.transform([emo])[0])
            y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n\u2705 Test accuracy: {acc:.4f}")
    print(f"\u2705 Macro F1: {macro_f1:.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=[c for c in classes]))
    print(f"\nAvg inference time per sample: {1000 * np.mean(times):.2f} ms")

    # Write summary JSON
    summary = {
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "num_classes": int(num_classes),
        "classes": [str(c) for c in classes],
        "n_test_samples": int(len(y_true)),
        "avg_ms_per_sample": float(1000 * np.mean(times)),
        "model": "wav2vec2-base frozen + MLP head",
    }
    with open(os.path.join(CHECKPOINT_DIR, "ser_integrity_result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved integrity result to {CHECKPOINT_DIR}/ser_integrity_result.json")


if __name__ == "__main__":
    main()