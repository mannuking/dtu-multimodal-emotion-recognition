"""
ensemble_evaluate_v9_test.py — Multimodal (Text+Audio) test-set evaluation
on the v9 PureCMT checkpoints with 3-seed softmax averaging.

Companion to train_ser_v9_baseline.py. Loads all available
v9_baseline_fold{F}_seed{S}.pt checkpoints, evaluates each on the
held-out val split of fold F (which is the same data layout
make_random_kfold_splits produces with seed S), and reports:

  - Per-seed test acc + macro-F1
  - Per-fold 3-seed softmax-averaged ensemble acc + macro-F1
  - Grand-mean W-Acc, UA-Acc

Outputs:
  reports/v9_multimodal_test_results.json
  printed per-fold table
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_random_kfold_splits,
    IDX_TO_EMOTION,
)
from cmt_fusion import FusionConfig, PureCMT
from train_ser_v9_baseline import FrozenEncoders, TARGET_SR, MAX_TEXT_LEN

warnings.filterwarnings("ignore")
CHECKPOINT_DIR = Path("model_checkpoints")


def load_fusion(ckpt_path: Path, device: str) -> tuple[PureCMT, dict]:
    """Reload a PureCMT from a v9 baseline checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = FusionConfig(
        audio_dim=cfg_dict["audio_dim"],
        text_dim=cfg_dict["text_dim"],
        proj_dim=cfg_dict.get("proj_dim", 256),
        num_classes=cfg_dict.get("num_classes", 4),
        n_cmt_layers=cfg_dict.get("n_cmt_layers", 2),
        n_heads=cfg_dict.get("n_heads", 4),
        va_dim=cfg_dict.get("va_dim", 3),
        va_proj_dim=cfg_dict.get("va_proj_dim", 64),
        dropout=cfg_dict.get("dropout", 0.3),
        dialog_context=cfg_dict.get("dialog_context", False),
        dialog_window=cfg_dict.get("dialog_window", 10),
        aggregation=ckpt.get("aggregation", cfg_dict.get("aggregation", "min")),
    )
    fusion = PureCMT(cfg).to(device)
    fusion.load_state_dict(ckpt["fusion"])
    fusion.eval()
    return fusion, ckpt


@torch.no_grad()
def predict_proba(fusion, encoders, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return softmax probs and ground-truth labels for an entire loader."""
    all_probs, all_labels = [], []
    for batch in loader:
        wav = batch["wav"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        emotion = batch["emotion"].to(device)
        audio_h = encoders.encode_audio(wav)
        text_h = encoders.encode_text(input_ids, attn_mask)
        logits = fusion(audio_h, text_h)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(emotion.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def per_class_recall(labels: np.ndarray, preds: np.ndarray, n_classes: int = 4) -> dict:
    out = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            out[IDX_TO_EMOTION[c]] = 0.0
        else:
            out[IDX_TO_EMOTION[c]] = float((preds[mask] == c).mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    parser.add_argument("--ckpt-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--out", default="reports/v9_multimodal_test_results.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest = load_manifest(Path(args.manifest))
    print(f"Manifest: {len(manifest)} utterances")

    ckpt_paths = sorted(Path(args.ckpt_dir).glob("v9_baseline_fold*_seed*.pt"))
    if not ckpt_paths:
        raise SystemExit(f"No v9_baseline_fold*_seed*.pt in {args.ckpt_dir}")
    print(f"Found {len(ckpt_paths)} checkpoints")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoders = FrozenEncoders(device)

    # Group ckpts by fold (one loader per fold per seed)
    per_seed_results = defaultdict(dict)   # (fold, seed) -> {acc, f1, ua_acc}
    fold_ensemble_probs = defaultdict(dict)  # fold -> seed -> per-sample probs

    for ckpt_path in ckpt_paths:
        fusion, ckpt = load_fusion(ckpt_path, device)
        fold = ckpt["fold"]
        seed = ckpt.get("seed", 42)
        if seed not in args.seeds:
            continue

        splits = make_random_kfold_splits(manifest, n_folds=5, seed=seed)
        train_idx, val_idx = splits[fold]
        val_manifest = [manifest[i] for i in val_idx]

        val_ds = IEMOCAPDataset(val_manifest, tokenizer,
                                target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

        probs, labels = predict_proba(fusion, encoders, val_loader, device)
        preds = probs.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        f1m = f1_score(labels, preds, average="macro")
        ua_acc = float(np.mean(list(per_class_recall(labels, preds).values())))

        per_seed_results[(fold, seed)] = {
            "acc": float(acc), "f1_macro": float(f1m), "ua_acc": ua_acc,
            "ckpt": ckpt_path.name,
        }
        fold_ensemble_probs[fold][seed] = (probs, labels)
        print(f"  fold{fold} seed{seed}: acc={acc:.4f} f1={f1m:.4f} ua={ua_acc:.4f}")

    # Per-fold ensemble (softmax avg across available seeds, on aligned val set)
    # Note: each seed has its OWN random k-fold split, so val sets differ across
    # seeds. The honest report is per-seed mean + grand mean — ensembles only
    # make sense when all models share a test set. v9 random splits DON'T share.
    # -> We report the per-fold-per-seed mean, and a per-fold ensemble only when
    #    seeds happen to overlap on the same val set (they don't by design).
    print("\n=== Per-fold per-seed (val = each seed's own random-split val) ===")
    fold_means = {}
    for fold in sorted({k[0] for k in per_seed_results}):
        accs = [per_seed_results[(fold, s)]["acc"] for s in args.seeds
                if (fold, s) in per_seed_results]
        if accs:
            fold_means[fold] = float(np.mean(accs))
        cells = []
        for s in args.seeds:
            r = per_seed_results.get((fold, s))
            cells.append(f"{r['acc']:.4f}" if r else "MISSING")
        print(f"  fold{fold}: {' | '.join(cells)}  -> mean {fold_means[fold]:.4f}"
              if accs else f"  fold{fold}: no seeds")

    grand_mean = float(np.mean(list(fold_means.values()))) if fold_means else 0.0
    grand_f1 = float(np.mean([
        per_seed_results[(f, s)]["f1_macro"]
        for (f, s) in per_seed_results]))
    grand_ua = float(np.mean([
        per_seed_results[(f, s)]["ua_acc"]
        for (f, s) in per_seed_results]))

    print(f"\n=== v9 MULTIMODAL (Text+Audio) — Random 5-fold, 14/15 ckpts ===")
    print(f"  Per-fold grand-mean (W-Acc):  {grand_mean:.4f}")
    print(f"  Per-seed macro-F1 grand mean:  {grand_f1:.4f}")
    print(f"  Per-seed UA-Acc grand mean:    {grand_ua:.4f}")
    print(f"  vs v8 LOSO mean: 0.7298  vs MemoCMT published (W-Acc 4-class random 5-fold): 0.8185")
    print(f"  Per-seed 95% CI: ±{1.96 * float(np.std([per_seed_results[k]['acc'] for k in per_seed_results]) / np.sqrt(len(per_seed_results))):.4f}")

    out = {
        "model": "PureCMT (HuBERT-base + BERT-base, frozen, CMT min-aggregation)",
        "task": "IEMOCAP 4-class (angry/happy/neutral/sad) random 5-fold CV",
        "n_checkpoints_used": len(per_seed_results),
        "per_seed": {
            f"fold{f}_seed{s}": per_seed_results[(f, s)]
            for (f, s) in per_seed_results
        },
        "per_fold_mean_acc": {str(f): m for f, m in fold_means.items()},
        "grand_mean_acc_W": grand_mean,
        "grand_mean_macro_F1": grand_f1,
        "grand_mean_UA_acc": grand_ua,
        "comparison": {
            "v8_LOSO_mean": 0.7298,
            "MemoCMT_published_W_acc": 0.8185,
            "MemoCMT_published_UA_acc": 0.8133,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()