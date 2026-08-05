"""
verify_ser.py — Integrity test for the SER (Speech Emotion Recognition) component.

Loads the trained SER checkpoint (ser_best.keras) + label encoder, applies the same
preprocessing used during training (MFCC-40 + ZCR + RMS + energy + entropy, 11,044
features per sample), runs inference on a stratified held-out test split that
mirrors the training protocol (SEED=42, 80/10/20 stratified), and reports real
metrics against ground-truth labels from the combined SER corpus.

This is NOT retraining. The checkpoints are loaded as-is and evaluated on real
test data drawn from the same combined dataset used in the manuscript.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ---- Configuration (matches train_ser_tensorflow.py + config.py) ----
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
    """Same silence-trimming function used in train_meta_classifier.py"""
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
    """Match the training-time pipeline exactly."""
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
    Reverse-engineered from the saved features-001.npy (11,044 dims per sample).
    Training-time params: hop=192 samples (12 ms), win=400 samples (25 ms).
    This yields 251 frames × 44 features (ZCR + RMS + energy + entropy + MFCC40) = 11,044.
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


def build_ser_1d_cnn(input_shape, num_classes=NUM_CLASSES):
    """Match train_ser_tensorflow.build_ser_1d_cnn exactly."""
    I = layers.Input(shape=input_shape)
    x = layers.Conv1D(256, 5, padding='same')(I)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(256, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(512, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3, 2, 'same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(I, out)


def main():
    print("=" * 70)
    print("SER INTEGRITY TEST — verifying the trained 1D-CNN on real audio")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Librosa:    {librosa.__version__}")
    print(f"Seed:       {SEED}")
    print()

    # 1. Load the manifest
    print("[1/5] Loading combined_ser_dataset/metadata.csv ...")
    df = pd.read_csv(os.path.join(SER_COMBINED_DIR, "metadata.csv"))
    print(f"      Total samples in manifest: {len(df)}")
    print(f"      Class distribution:\n{df['emotion'].value_counts().to_string()}\n")

    # 2. Build the 80/10/20 stratified split (matches train_ser_tensorflow.py exactly)
    print("[2/5] Building stratified 80/10/20 split with SEED=42 ...")
    label_map = {e: i for i, e in enumerate(EMOTION_ORDER_LOWER)}
    y_all = df['emotion'].str.lower().map(label_map).values
    X_paths = df['filepath'].values

    # First split: 90/10 (train+val / test)
    Xtv, Xte, ytv, yte = train_test_split(
        X_paths, y_all, test_size=0.10, stratify=y_all, random_state=SEED
    )
    # Second split: 80/10/20 of original (val = 11.1% of train+val = 10% of total)
    Xtr, Xva, ytr, yva = train_test_split(
        list(Xtv), list(ytv), test_size=0.111, stratify=ytv, random_state=SEED
    )
    print(f"      Train: {len(Xtr)}, Val: {len(Xva)}, Test: {len(Xte)}")
    print(f"      Test class counts: {dict(zip(*np.unique(yte, return_counts=True)))}\n")

    # 3. Load trained SER checkpoint
    print("[3/5] Loading ser_best.keras checkpoint ...")
    ser_path = os.path.join(CHECKPOINT_DIR, "ser_best.keras")
    enc_path = os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl")

    if not os.path.exists(ser_path):
        print(f"      ❌ SER checkpoint not found at {ser_path}")
        sys.exit(1)

    # Load architecture + weights from the .keras file.
    # NOTE: The file is HDF5 format saved with a .keras extension (legacy Keras).
    # load_model() works correctly with compile=False in that case; using
    # build()+load_weights() leaves BN moving stats un-initialised and the
    # model outputs random "surprise" for every input.
    model = tf.keras.models.load_model(ser_path, compile=False)
    print(f"      ✅ Loaded {os.path.getsize(ser_path)/1024/1024:.1f} MB model")
    print(f"      Model input shape: {model.input_shape}")
    print(f"      Model output shape: {model.output_shape}\n")

    # 4. Run inference on the test set
    print("[4/5] Running inference on test split ({} samples) ...".format(len(Xte)))
    preds = []
    probs_all = []
    t0 = time.time()
    failed = 0

    for i, (rel_path, y_true) in enumerate(zip(Xte, yte)):
        full_path = os.path.join(SER_COMBINED_DIR, rel_path)
        try:
            y_audio = preprocess_audio(full_path)
            feats = extract_features(y_audio).reshape(1, -1, 1).astype(np.float32)
            if feats.shape[1] != 11044:
                raise ValueError(f"feature shape {feats.shape} != (1, 11044, 1)")
            probs = model.predict(feats, verbose=0)[0]
            preds.append(int(np.argmax(probs)))
            probs_all.append(probs.tolist())
        except Exception as e:
            if failed < 3:
                print(f"      ⚠️  fail on {rel_path}: {type(e).__name__}: {e}")
            failed += 1
            preds.append(-1)
            probs_all.append([0.0] * NUM_CLASSES)

        if (i + 1) % 200 == 0:
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
    macro_f1 = f1_score(yte[mask], preds[mask], average='macro')
    weighted_f1 = f1_score(yte[mask], preds[mask], average='weighted')
    cm = confusion_matrix(yte[mask], preds[mask], labels=list(range(NUM_CLASSES)))

    print(f"      Accuracy:           {acc:.4f}  ({acc*100:.2f}%)")
    print(f"      Macro F1:           {macro_f1:.4f}")
    print(f"      Weighted F1:        {weighted_f1:.4f}")
    print()
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
        "model": "SER (1D-CNN)",
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