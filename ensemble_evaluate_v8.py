"""
ensemble_evaluate_v8.py — Average the 15 v8 checkpoints (5 folds × 3 seeds)
via softmax averaging, report per-fold, per-seed, and ensemble test accuracy.

Usage (on HPC):
  uv run python ensemble_evaluate_v8.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_loso_splits,
)
from cmt_fusion import FusionConfig, VAAwareCMT, DialogContextLayer
from train_ser_v8_cmt import FrozenEncoders, _fusion_forward, DialogContextBuffer

warnings.filterwarnings("ignore")
TARGET_SR = 16000
MAX_TEXT_LEN = 64
CHECKPOINT_DIR = Path("model_checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(Path(args.manifest))
    splits = make_loso_splits(manifest)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoders = FrozenEncoders(device)

    # Collect all checkpoints
    ckpts = sorted(CHECKPOINT_DIR.glob("v8_fold*_seed*.pt"))
    if not ckpts:
        print(f"ERROR: no v8_fold*_seed*.pt checkpoints found in {CHECKPOINT_DIR}")
        return
    print(f"Found {len(ckpts)} checkpoints")

    loaded = []
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        loaded.append((ckpt_path.name, ckpt))
        print(f"  {ckpt_path.name}: fold={ckpt['fold']} seed={ckpt.get('seed','?')} "
              f"val_acc={ckpt['val_acc']:.4f} val_f1={ckpt['val_f1']:.4f}")

    print("\n=== Per-fold ensemble (3 seeds, softmax avg) ===")
    fold_results = {}
    for fold_idx, (train_sess, val_sess) in enumerate(splits):
        fold_ckpts = [(name, c) for name, c in loaded if c["fold"] == fold_idx]
        if not fold_ckpts:
            print(f"  Fold {fold_idx}: no checkpoints, skipping")
            continue
        val_manifest = [r for r in manifest if r["session"] in val_sess]
        val_ds = IEMOCAPDataset(val_manifest, tokenizer, target_sr=TARGET_SR,
                                max_text_len=MAX_TEXT_LEN)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        cfg = FusionConfig(audio_dim=encoders.audio_dim, text_dim=encoders.text_dim)
        models = []
        for name, ckpt in fold_ckpts:
            fusion = VAAwareCMT(cfg).to(device)
            fusion.load_state_dict(ckpt["fusion"])
            ctx_layer = DialogContextLayer(cfg, cfg.proj_dim * 2).to(device)
            ctx_layer.load_state_dict(ckpt["ctx_layer"])
            fusion.eval(); ctx_layer.eval()
            models.append((name, fusion, ctx_layer))

        # Per-seed accuracy
        per_seed_acc = []
        all_labels = None
        all_probs = None
        for name, fusion, ctx_layer in models:
            ctx_buffer = DialogContextBuffer(window=cfg.dialog_window,
                                             dim=cfg.proj_dim * 2)
            ctx_buffer.reset()
            probs_one = []
            labs_one = []
            with torch.no_grad():
                for batch in val_loader:
                    wav = batch["wav"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    va = torch.stack([batch["valence"], batch["arousal"],
                                       batch["dominance"]], dim=1).to(device)
                    emotion = batch["emotion"].to(device)
                    dialog = batch["dialog"][0]
                    audio_h = encoders.encode_audio(wav)
                    text_h = encoders.encode_text(input_ids, attn_mask)
                    base_logits, base_feat = _fusion_forward(fusion, audio_h, text_h, va)
                    ctx_buffer.append(dialog, base_feat.mean(dim=0).squeeze(0))
                    dialog_ctx = ctx_buffer.get_context(dialog).unsqueeze(0).to(device)
                    logits = ctx_layer(base_feat, dialog_ctx)
                    p = F.softmax(logits, dim=-1).cpu().numpy()
                    probs_one.append(p)
                    labs_one.append(emotion.item())
            probs_one = np.concatenate(probs_one, axis=0)
            labs_one = np.array(labs_one)
            preds_one = probs_one.argmax(axis=1)
            per_seed_acc.append(float(accuracy_score(labs_one, preds_one)))
            all_probs = probs_one if all_probs is None else all_probs + probs_one
            all_labels = labs_one
        ensemble_probs = all_probs / len(models)
        ensemble_preds = ensemble_probs.argmax(axis=1)
        acc = accuracy_score(all_labels, ensemble_preds)
        f1 = f1_score(all_labels, ensemble_preds, average="macro")
        held = list(val_sess)[0]
        print(f"  Fold {fold_idx} (held={held}, {len(val_manifest)} utts): "
              f"ensemble_acc={acc:.4f} ensemble_f1={f1:.4f}  "
              f"per_seed_acc={[f'{a:.4f}' for a in per_seed_acc]}")
        fold_results[fold_idx] = {
            "held_out_session": held,
            "n_val": len(val_manifest),
            "ensemble_acc": acc,
            "ensemble_f1": f1,
            "per_seed_acc": per_seed_acc,
        }

    accs = [r["ensemble_acc"] for r in fold_results.values()]
    f1s = [r["ensemble_f1"] for r in fold_results.values()]
    print(f"\n=== Aggregate across 5 folds (3-seed ensemble each) ===")
    print(f"  Mean acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Per-fold ensemble accs: {[f'{a:.4f}' for a in accs]}")

    summary = {
        "per_fold": {str(k): v for k, v in fold_results.items()},
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "n_checkpoints": len(loaded),
    }
    out = CHECKPOINT_DIR / "v8_ensemble_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved ensemble summary to {out}")


if __name__ == "__main__":
    main()
