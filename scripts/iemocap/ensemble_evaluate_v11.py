"""
ensemble_evaluate_v11.py — Eval v11 V/A-conditioned fusion ckpts on the
held-out test split.

Reads:
  model_checkpoints/v11_va_unif_fold{fold}_seed{seed}.pt       (15 ckpts)
  model_checkpoints/v11_va_cw_fold{fold}_seed{seed}.pt         (15 ckpts)

Writes:
  reports/v11_table1_results.json  — same shape as ensemble_evaluate_v10.py,
  so build_table1_v10.py can read it the same way.

The v11 head is VAFusionHead (scripts/iemocap/train_ser_v11_va.py), which
carries two extra pieces of state in its state_dict: `class_va` and
`va_head`. Both are restored from the checkpoint, so the evaluated model is
bit-identical to the trained one.

The va_weight used at eval time is read from the checkpoint itself rather
than from a CLI flag — that way an eval can never silently disagree with
the training configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from iemocap_dataset import (  # noqa: E402
    IEMOCAPDataset, load_manifest, make_random_kfold_splits, IDX_TO_EMOTION,
)
from train_ser_v10_fusion import (  # noqa: E402
    FrozenEncoder, WAVLM_NAME, DEBERTA_NAME, TARGET_SR, MAX_TEXT_LEN,
)
from train_ser_v11_va import VAFusionHead  # noqa: E402

warnings.filterwarnings("ignore")


def compute_ua(y_true, y_pred):
    """Unweighted accuracy = per-class recall averaged (macro recall)."""
    return recall_score(y_true, y_pred, average="macro")


def eval_one_fold_seed(fold: int, seed: int, device: str, tokenizer, manifest,
                       weight_tag: str, split):
    """Eval a single v11 VA-fusion ckpt on the held-out test split."""
    train_idx, val_idx = split
    val_manifest = [manifest[i] for i in val_idx]
    loader = DataLoader(
        IEMOCAPDataset(val_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    CKPT_ROOT = Path(__file__).resolve().parent.parent.parent / "model_checkpoints"
    text_ckpt = CKPT_ROOT / f"v9b_text_fold{fold}_seed{seed}.pt"
    audio_ckpt = CKPT_ROOT / f"v9b_audio_fold{fold}_seed{seed}.pt"
    fusion_ckpt = CKPT_ROOT / f"v11_va_{weight_tag}_fold{fold}_seed{seed}.pt"

    text_enc = FrozenEncoder(text_ckpt, kind="text").to(device)
    audio_enc = FrozenEncoder(audio_ckpt, kind="audio").to(device)

    state = torch.load(fusion_ckpt, map_location=device, weights_only=False)
    # va_weight comes from the checkpoint so eval can never drift from training.
    va_weight = float(state.get("va_weight", 0.5))
    head = VAFusionHead(va_weight=va_weight).to(device)
    head.load_state_dict(state["head"])
    head.eval()

    preds, labels = [], []
    va_err = 0.0
    with torch.no_grad():
        for batch in loader:
            emo = batch["emotion"].to(device)
            text_emb = text_enc.extract(batch)
            audio_emb = audio_enc.extract(batch)
            logits, va_pred, _ = head(text_emb, audio_emb)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            labels.extend(emo.cpu().numpy().tolist())
            va_true = torch.stack([batch["valence"], batch["arousal"]], dim=1).to(device)
            va_err += torch.nn.functional.mse_loss(
                va_pred, va_true, reduction="sum").item()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    ua = compute_ua(labels, preds)
    return acc, f1, ua, va_err / max(1, len(labels) * 2), va_weight


def main():
    parser = argparse.ArgumentParser()
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    default_manifest = str(REPO_ROOT / "data" / "iemocap" / "manifest.csv")
    parser.add_argument("--manifest", default=default_manifest)
    parser.add_argument("--out", default="reports/v11_table1_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    manifest = load_manifest(Path(args.manifest))
    print(f"Manifest: {len(manifest)} utterances")

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    out = {"v11": {"unif": {"per_ckpt": [], "grand_mean": {}},
                   "cw":   {"per_ckpt": [], "grand_mean": {}}}}

    for weight_tag in ("unif", "cw"):
        for fold in range(5):
            for seed in (42, 43, 44):
                split = make_random_kfold_splits(manifest, n_folds=5, seed=seed)[fold]
                acc, f1, ua, vae, vw = eval_one_fold_seed(
                    fold, seed, device, tokenizer, manifest, weight_tag, split)
                print(f"  v11 {weight_tag} fold{fold} seed{seed}: "
                      f"acc={acc:.4f} f1={f1:.4f} ua={ua:.4f} va_mae={vae:.5f} va_w={vw}")
                out["v11"][weight_tag]["per_ckpt"].append({
                    "fold": fold, "seed": seed, "acc": acc, "f1": f1, "ua": ua,
                    "val_vae": vae, "va_weight": vw,
                })

    for weight_tag in ("unif", "cw"):
        per = out["v11"][weight_tag]["per_ckpt"]
        accs = [p["acc"] for p in per]
        f1s = [p["f1"] for p in per]
        uas = [p["ua"] for p in per]
        vaes = [p["val_vae"] for p in per]
        out["v11"][weight_tag]["grand_mean"] = {
            "acc": float(np.mean(accs)),
            "f1": float(np.mean(f1s)),
            "ua": float(np.mean(uas)),
            "va_mae": float(np.mean(vaes)),
            "n": len(per),
        }
        print(f"\n  v11 {weight_tag}: acc={np.mean(accs):.4f} "
              f"f1={np.mean(f1s):.4f} ua={np.mean(uas):.4f} "
              f"(n={len(per)})")

    # Carry the v9b + v10 reference legs so one table can show the whole story.
    ref_dir = REPO_ROOT / "reports"
    for cand in ("v9b_table1_results.json", "v9b_table1.json"):
        p = ref_dir / cand
        if p.exists():
            with open(p) as f:
                out["v9b"] = json.load(f)
            print(f"  loaded v9b reference from {cand}")
            break
    p10 = ref_dir / "v10_table1_results.json"
    if p10.exists():
        with open(p10) as f:
            out["v10"] = json.load(f)
        print("  loaded v10 reference")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
