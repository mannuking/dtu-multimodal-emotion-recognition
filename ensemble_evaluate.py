"""
ensemble_evaluate.py - Average predictions from multiple seed checkpoints.

Loads N checkpoints (one per seed), runs TTA on each, averages softmax
probabilities, and reports the ensemble accuracy / F1 / classification
report. This is the standard "deep ensemble" pattern from Lakshminarayanan
2017 — averaging softmax outputs reduces variance and typically adds 2-4pp
over the best single model.

The train script fixes split_seed=42 across all ensemble runs, so the test
set is identical across seeds (only model init / dropout / SpecAugment vary).
This script reproduces the same split to load the right test audios.

Usage (from HPC login or compute node):
  uv run python ensemble_evaluate.py --seeds 42 43 44 --n-tta 5
  uv run python ensemble_evaluate.py --seeds 42 43 44 --n-tta 10

Expected runtime: ~5 min per seed (TTA only, no training).
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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Reuse the model + training code
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_ser_enhanced import (
    EnhancedSER1DCNN, Wav2Vec2FeatureExtractor, WavSERDataset,
    TARGET_SR, MAX_S, EMOTIONS, CHECKPOINT_DIR, SER_COMBINED_DIR,
)

warnings.filterwarnings("ignore")


def load_seed_state(seed: int):
    """Load the (model, feature_extractor) state_dicts for a given seed."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"ser_best_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state["model"], state["feature_extractor"]


def predict_tta(model, feature_extractor, x, device, n_passes=5):
    """Average predictions over n_passes random time crops."""
    probs_sum = None
    for i in range(n_passes):
        with torch.no_grad():
            if i == 0:
                # full audio pass
                feats = feature_extractor(x)
            else:
                # random 80-100% crop, padded
                T = x.shape[1]
                crop_frac = float(torch.rand(1).item() * 0.2 + 0.8)
                crop_T = int(T * crop_frac)
                t0 = int(torch.randint(0, T - crop_T + 1, (1,)).item())
                x_crop = x[:, t0:t0 + crop_T]
                x_padded = F.pad(x_crop, (0, T - crop_T))
                feats = feature_extractor(x_padded)
            logits = model(feats)
            probs = F.softmax(logits, dim=-1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / n_passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Seeds whose checkpoints to ensemble")
    parser.add_argument("--n-tta", type=int, default=5,
                        help="Number of TTA passes per checkpoint (default: 5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds to ensemble: {args.seeds}")
    print(f"TTA passes: {args.n_tta}")
    print(f"Note: split_seed=42 is fixed in train_ser_enhanced.py, so all seeds share the same test set.")

    # ---- Reproduce the canonical split (split_seed=42) to get test indices ----
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

    # ---- Load raw audio (same as training) ----
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
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)))
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            audios_test.append(audio.astype(np.float32))
        except Exception:
            audios_test.append(np.zeros(int(MAX_S * TARGET_SR), dtype=np.float32))
    y_test = y_all[idx_test]
    print(f"Loaded {len(audios_test)} test audios")

    # ---- Load wav2vec2 + 1D-CNN architecture once (shared across seeds) ----
    # num_unfrozen_layers=12 because the v3-style architecture unfreezes all
    # layers and v2 (4 unfrozen) is a strict subset — load_state_dict copies
    # only matching keys, so either checkpoint loads cleanly into this
    # larger container.
    local_model = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
    )
    model_path = local_model if os.path.exists(os.path.join(local_model, "pytorch_model.bin")) else "facebook/wav2vec2-base"
    feature_extractor = Wav2Vec2FeatureExtractor(model_path, num_unfrozen_layers=12).to(device).eval()
    model = EnhancedSER1DCNN(in_channels=768, num_classes=len(EMOTIONS)).to(device).eval()

    # ---- Run each seed, accumulate softmax probabilities ----
    all_probs_sum = None
    per_seed_accs = {}

    from torch.utils.data import DataLoader
    test_ds = WavSERDataset(audios_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=24, shuffle=False, num_workers=2, pin_memory=True)

    for seed in args.seeds:
        print(f"\n--- Loading seed {seed} checkpoint ---")
        model_state, encoder_state = load_seed_state(seed)
        model.load_state_dict(model_state)
        feature_extractor.load_state_dict(encoder_state)
        model.eval()
        feature_extractor.eval()

        # Per-seed TTA predictions
        seed_probs = []
        seed_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                probs = predict_tta(model, feature_extractor, x, device, n_passes=args.n_tta)
                seed_probs.append(probs.cpu().numpy())
                seed_labels.append(y.numpy())
        seed_probs = np.concatenate(seed_probs, axis=0)
        seed_labels = np.concatenate(seed_labels, axis=0)
        seed_preds = seed_probs.argmax(axis=1)
        seed_acc = accuracy_score(seed_labels, seed_preds)
        per_seed_accs[seed] = seed_acc
        print(f"  seed {seed} TTA acc: {seed_acc:.4f}")

        all_probs_sum = seed_probs if all_probs_sum is None else all_probs_sum + seed_probs

    # ---- Ensemble prediction (average softmax across seeds) ----
    ensemble_probs = all_probs_sum / len(args.seeds)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    y_true = seed_labels  # all seeds predict on the same test set in same order

    ensemble_acc = accuracy_score(y_true, ensemble_preds)
    ensemble_f1 = f1_score(y_true, ensemble_preds, average="macro")

    print(f"\n============================================================")
    print(f"ENSEMBLE RESULTS (across {len(args.seeds)} seeds, {args.n_tta}-pass TTA each)")
    print(f"============================================================")
    print(f"Per-seed test accuracy:")
    for seed, acc in per_seed_accs.items():
        print(f"  seed {seed}: {acc:.4f}")
    print(f"  mean +/- std: {np.mean(list(per_seed_accs.values())):.4f} +/- {np.std(list(per_seed_accs.values())):.4f}")
    print(f"\nENSEMBLE TEST ACCURACY: {ensemble_acc:.4f}")
    print(f"ENSEMBLE TEST MACRO-F1: {ensemble_f1:.4f}")
    print(f"\nEnsemble lift over best single seed: {ensemble_acc - max(per_seed_accs.values()):+.4f}")
    print(f"\nPer-class report (ensemble):")
    print(classification_report(y_true, ensemble_preds, target_names=list(le.classes_)))

    # Save ensemble summary
    summary = {
        "method": f"Ensemble of {len(args.seeds)} wav2vec2-base fine-tuned models (different seeds), 5-pass TTA each",
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
    summary_path = os.path.join(CHECKPOINT_DIR, "ser_ensemble_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved ensemble summary to {summary_path}")


if __name__ == "__main__":
    main()
