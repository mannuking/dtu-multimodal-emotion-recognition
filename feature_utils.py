"""
feature_utils.py — Self-contained audio feature extraction for SER training.

Reuses the exact reverse-engineered parameters from verify_ser.py so the
extracted features match the saved features-001.npy format:
  - hop = 192 samples (12 ms @ 16 kHz)
  - win = 400 samples (25 ms @ 16 kHz)
  - n_mfcc = 40
  - Frame-level concatenation: [ZCR(1), RMS(1), energy(1), entropy(1), MFCC40]
  - Total: 44 features × 251 frames = 11,044 dims per sample

This lets train_ser_tensorflow.py run end-to-end on a fresh HPC node
without needing pre-computed features.npy files.
"""
import os
import numpy as np
import pandas as pd
import librosa

# Match gpu_config.py exactly
TARGET_SR = 22050
OFFSET_S = 0.5
DUR_S = 2.5
EMOTION_ORDER_LOWER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

HOP = 192
WIN = 400
N_MFCC = 40


def trim_silence(audio, thresh_scale=3):
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
    """Load + trim + slice to fixed duration. Same as verify_ser."""
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    y = trim_silence(y)
    y = y[int(OFFSET_S * sr):]
    target = int(DUR_S * sr)
    if len(y) > target:
        y = y[:target]
    else:
        y = np.pad(y, (0, target - len(y)))
    return y


def extract_features(y, sr=TARGET_SR, n_mfcc=N_MFCC):
    """Returns flat 11044-dim vector (251 frames × 44 features)."""
    hop = HOP
    win = WIN
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win, hop_length=hop).T
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=win, hop_length=hop).T
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).T
    energy = rms ** 2 * win
    prob = energy / (np.sum(energy) + 1e-8)
    entropy = -prob * np.log2(prob + 1e-12)
    feats = np.concatenate([zcr, rms, energy, entropy, mfcc], axis=1)
    return feats.flatten()


def extract_features_batch(wav_paths, manifest_df=None, n_mfcc=N_MFCC, verbose=True):
    """
    Extract features for a list of wav paths.
    Returns (features: (N, 11044) np.float32 array, labels: np.ndarray of int).
    Skips files that fail to load.
    """
    feats_list = []
    labels_list = []
    skipped = 0
    total = len(wav_paths)
    for i, path in enumerate(wav_paths):
        try:
            y = preprocess_audio(path)
            if len(y) < int(0.5 * TARGET_SR):
                skipped += 1
                continue
            f = extract_features(y, sr=TARGET_SR, n_mfcc=n_mfcc)
            if len(f) != 251 * 44:
                skipped += 1
                continue
            feats_list.append(f)
            # Label from manifest_df if provided, else 0
            if manifest_df is not None and 'emotion' in manifest_df.columns:
                row = manifest_df.iloc[i]
                emo = str(row['emotion']).lower().strip()
                labels_list.append(EMOTION_ORDER_LOWER.index(emo) if emo in EMOTION_ORDER_LOWER else -1)
            else:
                labels_list.append(0)
            if verbose and (i + 1) % 500 == 0:
                print(f"  [features] {i+1}/{total} processed ({skipped} skipped)")
        except Exception as e:
            skipped += 1
            continue
    if verbose:
        print(f"  [features] done: {len(feats_list)} ok, {skipped} skipped")
    if not feats_list:
        return np.zeros((0, 251 * 44), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    feats = np.array(feats_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    return feats, labels


def ensure_features_exist(features_path="ser_feature_output/features.npy",
                          labels_path="ser_feature_output/labels.npy",
                          manifest_csv="combined_ser_dataset/metadata.csv",
                          overwrite=False):
    """
    If features.npy is missing on disk, compute it from the audio manifest.
    Caches result so re-runs are instant.
    """
    if os.path.exists(features_path) and os.path.exists(labels_path) and not overwrite:
        f = np.load(features_path, mmap_mode='r')
        l = np.load(labels_path)
        print(f"  [features] cached: {f.shape} features, {len(l)} labels at {features_path}")
        return f, l

    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    print(f"  [features] cache miss — extracting from {manifest_csv}")
    df = pd.read_csv(manifest_csv)
    wav_paths = [os.path.join("combined_ser_dataset", row['filepath']) for _, row in df.iterrows()]
    feats, labels = extract_features_batch(wav_paths, manifest_df=df, verbose=True)
    print(f"  [features] saving {feats.shape} → {features_path}")
    # Save as float16 to halve disk space (5.4 GB → 2.7 GB)
    np.save(features_path, feats.astype(np.float16))
    np.save(labels_path, labels)
    return feats, labels