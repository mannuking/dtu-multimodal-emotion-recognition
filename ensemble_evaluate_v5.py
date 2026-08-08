"""
ensemble_evaluate_v5.py - Average v5 (advanced wav2vec2-large + SupCon + EMA) checkpoints.

Loads N v5 checkpoints, runs 8-pass TTA on each via the EMA model, averages
softmax, reports ensemble accuracy/F1/per-class report.

Usage (must go through sbatch, never on login node):
  uv run python ensemble_evaluate_v5.py --seeds 42 43 44 --n-tta 8
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from train_ser_v5 import (
    StrongSERHead, Wav2Vec2LargeExtractor, WavSERDataset,
    TARGET_SR, MAX_S, EMOTIONS, NUM_CLASSES, ENCODER_HIDDEN,
    CHECKPOINT_DIR, SER_COMBINED_DIR,
)

warnings.filterwarnings("ignore")


def load_seed_state(seed: int):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"ser_v5_best_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state["model"], state["feature_extractor"]


def predict_tta(model, feature_extractor, x, n_passes=8):
    probs_sum = None
    for i in range(n_passes):
        with torch.no_grad():
            if i == 0:
                feats = feature_extractor(x)
            else:
                T = x.shape[1]
                crop_frac = float(torch.rand(1).item() * 0.2 + 0.8)
                crop_T = int(T * crop_frac)
                t0 = int(torch.randint(0, T - crop_T + 1, (1,)).item())
                x_crop = x[:, t0:t0 + crop_T]
                x_padded = F.pad(x_crop, (0, T - crop_T))
                feats = feature_extractor(x_padded)
            logits, _ = model(feats)
            probs = F.softmax(logits, dim=-1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / n_passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-tta", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds to ensemble: {args.seeds}")
    print(f"TTA passes: {args.n_tta}")

    torch.manual_seed(42)
    np.random.seed(42)

    manifest_csv = os.path.join(SER_COMBINED_DIR, "metadata.csv")
    df = pd.read_csv(manifest_csv)
    if "wav_path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "wav_path"})
    df["emotion"] = df["emotion"].astype(str).str.lower()
    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)
    le = LabelEncoder().fit(EMOTIONS)
    y_all = le.transform(df["emotion"].values)

    subjects = df["subject"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    idx_train_full, idx_temp = next(gss.split(np.arange(len(df)), y_all, groups=subjects))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    idx_val, idx_test = next(gss2.split(idx_temp, y_all[idx_temp], groups=subjects[idx_temp]))
    idx_val = idx_temp[idx_val]
    idx_test = idx_temp[idx_test]
    print(f"Test set: {len(idx_test)} samples (split_seed=42)")

    import librosa
    print("Loading raw audio for test split...")
    audios_test = []
    for i in idx_test:
        p = df["wav_path"].iloc[i]
        try:
            audio, _ = librosa.load(p, sr=TARGET_SR, mono=True, duration=MAX_S + 0.5)
            if len(audio) < 1600:
                raise ValueError("too short")
            max_samples = int(MAX_S * TARGET_SR)
            audio = audio[:max_samples] if len(audio) > max_samples else np.pad(audio, (0, max_samples - len(audio)))
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            audios_test.append(audio.astype(np.float32))
        except Exception:
            audios_test.append(np.zeros(int(MAX_S * TARGET_SR), dtype=np.float32))
    y_test = y_all[idx_test]
    print(f"Loaded {len(audios_test)} test audios")

    local_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--wav2vec2-large/snapshots"
    )
    model_path = "facebook/wav2vec2-large"
    if os.path.isdir(local_path):
        for snap in os.listdir(local_path):
            candidate = os.path.join(local_path, snap)
            if os.path.exists(os.path.join(candidate, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(candidate, "model.safetensors")):
                model_path = candidate
                break
    feature_extractor = Wav2Vec2LargeExtractor(model_path, num_unfrozen_layers=8).to(device).eval()
    model = StrongSERHead(in_channels=ENCODER_HIDDEN, num_classes=NUM_CLASSES).to(device).eval()

    all_probs_sum = None
    per_seed_accs = {}

    test_ds = WavSERDataset(audios_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    for seed in args.seeds:
        print(f"\n--- Loading seed {seed} v5 checkpoint ---")
        model_state, encoder_state = load_seed_state(seed)
        model.load_state_dict(model_state)
        feature_extractor.load_state_dict(encoder_state)
        model.eval()
        feature_extractor.eval()

        seed_probs, seed_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                probs = predict_tta(model, feature_extractor, x, n_passes=args.n_tta)
                seed_probs.append(probs.cpu().numpy())
                seed_labels.append(y.numpy())
        seed_probs = np.concatenate(seed_probs, axis=0)
        seed_labels = np.concatenate(seed_labels, axis=0)
        seed_preds = seed_probs.argmax(axis=1)
        seed_acc = accuracy_score(seed_labels, seed_preds)
        per_seed_accs[seed] = seed_acc
        print(f"  seed {seed} TTA acc: {seed_acc:.4f}")

        all_probs_sum = seed_probs if all_probs_sum is None else all_probs_sum + seed_probs

    ensemble_probs = all_probs_sum / len(args.seeds)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    y_true = seed_labels

    ensemble_acc = accuracy_score(y_true, ensemble_preds)
    ensemble_f1 = f1_score(y_true, ensemble_preds, average="macro")

    print(f"\n============================================================")
    print(f"V5 ENSEMBLE RESULTS ({len(args.seeds)} seeds, {args.n_tta}-pass TTA + EMA)")
    print(f"============================================================")
    print(f"Per-seed test accuracy:")
    for seed, acc in per_seed_accs.items():
        print(f"  seed {seed}: {acc:.4f}")
    print(f"  mean ± std: {np.mean(list(per_seed_accs.values())):.4f} ± {np.std(list(per_seed_accs.values())):.4f}")
    print(f"\nV5 ENSEMBLE TEST ACCURACY: {ensemble_acc:.4f}")
    print(f"V5 ENSEMBLE TEST MACRO-F1: {ensemble_f1:.4f}")
    print(f"\nEnsemble lift over best single seed: {ensemble_acc - max(per_seed_accs.values()):+.4f}")
    print(f"\nPer-class report (ensemble):")
    print(classification_report(y_true, ensemble_preds, target_names=list(le.classes_)))

    summary = {
        "method": f"v5 ensemble: wav2vec2-large + SupCon(1.0) + EMA + stochastic depth + bi-attn pool, {len(args.seeds)}-seed avg, {args.n_tta}-pass TTA",
        "seeds": args.seeds,
        "per_seed_test_acc": {str(k): float(v) for k, v in per_seed_accs.items()},
        "per_seed_mean_acc": float(np.mean(list(per_seed_accs.values()))),
        "per_seed_std_acc": float(np.std(list(per_seed_accs.values()))),
        "ensemble_test_acc": float(ensemble_acc),
        "ensemble_test_macro_f1": float(ensemble_f1),
        "ensemble_lift_over_best_seed": float(ensemble_acc - max(per_seed_accs.values())),
        "n_test": int(len(y_true)),
        "classes": list(EMOTIONS),
        "tta_passes_per_seed": args.n_tta,
    }
    summary_path = os.path.join(CHECKPOINT_DIR, "ser_v5_ensemble_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved v5 ensemble summary to {summary_path}")


if __name__ == "__main__":
    main()