"""
ensemble_evaluate_v10.py — Eval v10-B fusion ckpts on held-out test split.

Reads:
  model_checkpoints/v10_fusion_unif_fold{fold}_seed{seed}.pt       (15 ckpts)
  model_checkpoints/v10_fusion_cw_fold{fold}_seed{seed}.pt          (15 ckpts)
  model_checkpoints/v9b_audio_fold{fold}_seed{seed}.pt              (reuse)
  model_checkpoints/v9b_text_fold{fold}_seed{seed}.pt               (reuse)

Writes:
  reports/v10_table1_results.json  — same shape as ensemble_evaluate_v9b.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_random_kfold_splits, IDX_TO_EMOTION,
)
from train_ser_v10_fusion import (
    FrozenEncoder, CMFusionHead,
    WAVLM_NAME, DEBERTA_NAME, TARGET_SR, MAX_TEXT_LEN,
)

warnings.filterwarnings("ignore")


def compute_ua(y_true, y_pred):
    """Unweighted accuracy = per-class recall averaged."""
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average="macro")


def eval_one_fold_seed(fold: int, seed: int, device: str, tokenizer, manifest,
                       weight_tag: str, split):
    """Eval a single v10 fusion ckpt on the held-out test split."""
    train_idx, val_idx = split
    val_manifest = [manifest[i] for i in val_idx]
    loader = DataLoader(
        IEMOCAPDataset(val_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    text_ckpt = Path("model_checkpoints") / f"v9b_text_fold{fold}_seed{seed}.pt"
    audio_ckpt = Path("model_checkpoints") / f"v9b_audio_fold{fold}_seed{seed}.pt"
    fusion_ckpt = (Path("model_checkpoints")
                   / f"v10_fusion_{weight_tag}_fold{fold}_seed{seed}.pt")
    text_enc = FrozenEncoder(text_ckpt, kind="text").to(device)
    audio_enc = FrozenEncoder(audio_ckpt, kind="audio").to(device)
    head = CMFusionHead().to(device)
    state = torch.load(fusion_ckpt, map_location=device, weights_only=False)
    head.load_state_dict(state["head"])
    head.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            emo = batch["emotion"].to(device)
            text_emb = text_enc.extract(batch)
            audio_emb = audio_enc.extract(batch)
            logits = head(text_emb, audio_emb)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            labels.extend(emo.cpu().numpy().tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    ua = compute_ua(labels, preds)
    return acc, f1, ua


def main():
    parser = argparse.ArgumentParser()
    # Same repo-root resolution as train_ser_v10_fusion.py (cwd-relative paths
    # break because sbatch does `cd scripts/iemocap` first).
    REPO_ROOT = Path(__file__).parent.parent.parent
    default_manifest = str(REPO_ROOT / "data" / "iemocap" / "manifest.csv")
    parser.add_argument("--manifest", default=default_manifest)
    parser.add_argument("--out", default="reports/v10_table1_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    manifest = load_manifest(Path(args.manifest))
    print(f"Manifest: {len(manifest)} utterances")

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    out = {"v10": {"unif": {"per_ckpt": [], "grand_mean": {}},
                   "cw":   {"per_ckpt": [], "grand_mean": {}}}}

    for weight_tag in ("unif", "cw"):
        for fold in range(5):
            for seed in (42, 43, 44):
                split = make_random_kfold_splits(manifest, n_folds=5, seed=seed)[fold]
                acc, f1, ua = eval_one_fold_seed(
                    fold, seed, device, tokenizer, manifest, weight_tag, split)
                print(f"  v10 {weight_tag} fold{fold} seed{seed}: "
                      f"acc={acc:.4f} f1={f1:.4f} ua={ua:.4f}")
                out["v10"][weight_tag]["per_ckpt"].append({
                    "fold": fold, "seed": seed, "acc": acc, "f1": f1, "ua": ua,
                })

    # Grand means
    for weight_tag in ("unif", "cw"):
        per = out["v10"][weight_tag]["per_ckpt"]
        accs = [p["acc"] for p in per]
        f1s = [p["f1"] for p in per]
        uas = [p["ua"] for p in per]
        out["v10"][weight_tag]["grand_mean"] = {
            "acc": float(np.mean(accs)),
            "f1": float(np.mean(f1s)),
            "ua": float(np.mean(uas)),
            "n": len(per),
        }

    # Append v9b legs (reuse from existing reports if available)
    v9b_path = Path("reports/v9b_table1_results.json")
    if v9b_path.exists():
        with open(v9b_path) as f:
            out["v9b"] = json.load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== v10 TABLE 1 — grand means ===")
    for tag in ("unif", "cw"):
        m = out["v10"][tag]["grand_mean"]
        print(f"  v10-B {tag.upper():4s}: "
              f"W-Acc={m['acc']:.4f}  F1={m['f1']:.4f}  UA={m['ua']:.4f}  "
              f"(n={m['n']})")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
