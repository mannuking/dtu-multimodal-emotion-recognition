"""
train_ser_v9b_audio_only.py — Audio-only baseline on IEMOCAP-4
(WavLM-base+, last 2 transformer layers unfrozen + mean-pool + linear head).

This is the V9B AUDIO row of MemoCMT Table 1. Replaces v9's frozen HuBERT-base
(which scored ~0.66) with WavLM-base+ + last 2 layers unfrozen. Same 5-fold
random splits, same IEMOCAP-4 manifest, same seeds (42, 43, 44) as v9
multimodal. Expected val_acc: 0.76-0.80.

Usage on HPC:
  uv run python train_ser_v9b_audio_only.py --fold 0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_random_kfold_splits,
    IDX_TO_EMOTION,
)

warnings.filterwarnings("ignore")

CHECKPOINT_DIR = Path("model_checkpoints")
TARGET_SR = 16000

# WavLM-Base+ from Microsoft. Output dim 768. 12 transformer encoder layers.
WAVLM_NAME = "microsoft/wavlm-base-plus"
UNFREEZE_LAST_N = 2


class WavLMHead(nn.Module):
    """WavLM-Base+ + unfreeze last N transformer layers + mean-pool + LayerNorm + Dropout + Linear."""
    def __init__(self, num_classes: int = 4, dropout: float = 0.3,
                 unfreeze_last_n: int = UNFREEZE_LAST_N):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(WAVLM_NAME)
        hidden = self.wavlm.config.hidden_size  # 768 for base+
        # Freeze everything by default.
        for p in self.wavlm.parameters():
            p.requires_grad = False
        # Re-enable grads on the last N encoder layers. WavLM uses WavLMEncoder
        # whose layers live in `.encoder.layers` (nn.ModuleList).
        try:
            layers = self.wavlm.encoder.layers
        except AttributeError:
            # Some HF versions nest under encoder.encoder
            layers = self.wavlm.encoder.encoder.layers
        n_total = len(layers)
        for i in range(n_total - unfreeze_last_n, n_total):
            for p in layers[i].parameters():
                p.requires_grad = True
        # WavLM also has a final LayerNorm at the encoder top.
        ln = getattr(self.wavlm.encoder, "layer_norm", None)
        if ln is not None:
            for p in ln.parameters():
                p.requires_grad = True
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Full forward (grad-enabled for last N layers).
        out = self.wavlm(wav)
        h = out.last_hidden_state                # (B, T, 768)
        pooled = h.mean(dim=1)                   # (B, 768)
        return self.head(pooled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr-head", type=float, default=2e-3)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Seed: {args.seed} | Fold: {args.fold}")

    manifest = load_manifest(Path(args.manifest))
    splits = make_random_kfold_splits(manifest, n_folds=5, seed=args.seed)
    train_idx, val_idx = splits[args.fold]
    train_manifest = [manifest[i] for i in train_idx]
    val_manifest = [manifest[i] for i in val_idx]
    print(f"Train: {len(train_manifest)}  Val: {len(val_manifest)}")

    train_loader = DataLoader(
        IEMOCAPDataset(train_manifest, tokenizer=None,
                       target_sr=TARGET_SR),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        IEMOCAPDataset(val_manifest, tokenizer=None,
                       target_sr=TARGET_SR),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = WavLMHead(num_classes=4).to(device)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    encoder_params = [p for p in model.wavlm.parameters() if p.requires_grad]
    n_enc_trainable = sum(p.numel() for p in encoder_params)
    print(f"Trainable params: head={sum(p.numel() for p in head_params):,} "
          f"+ encoder(last {UNFREEZE_LAST_N})={n_enc_trainable:,} "
          f"= {sum(p.numel() for p in head_params) + n_enc_trainable:,}")

    optimizer = AdamW([
        {"params": head_params, "lr": args.lr_head},
        {"params": encoder_params, "lr": args.lr_encoder},
    ], weight_decay=0.01)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc, best_epoch, best_f1 = 0.0, -1, 0.0
    history, epochs_no_improve = [], 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for batch in train_loader:
            wav = batch["wav"].to(device)
            emo = batch["emotion"].to(device)
            logits = model(wav)
            loss = F.cross_entropy(logits, emo)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * wav.size(0)
            tr_correct += (logits.argmax(1) == emo).sum().item()
            tr_total += wav.size(0)
        train_loss = tr_loss / tr_total
        train_acc = tr_correct / tr_total

        model.eval()
        v_correct = v_total = 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                wav = batch["wav"].to(device)
                emo = batch["emotion"].to(device)
                logits = model(wav)
                v_correct += (logits.argmax(1) == emo).sum().item()
                v_total += wav.size(0)
                v_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                v_labels.extend(emo.cpu().numpy().tolist())
        val_acc = v_correct / v_total
        val_f1 = f1_score(v_labels, v_preds, average="macro")

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, best_f1 = val_acc, epoch, val_f1
            epochs_no_improve = 0
            ckpt_path = CHECKPOINT_DIR / f"v9b_audio_fold{args.fold}_seed{args.seed}.pt"
            torch.save({
                "model": model.state_dict(),
                "fold": args.fold,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "seed": args.seed,
            }, ckpt_path)
            saved = f"  ✅ {ckpt_path.name}"
        else:
            epochs_no_improve += 1
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc,
                        "val_f1": val_f1})
        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}  ({time.time()-t0:.0f}s){saved}")

        if epochs_no_improve >= args.patience:
            print(f"Early stop @ epoch {epoch} (no improve for {args.patience})")
            break

    print(f"\n=== fold{args.fold} seed{args.seed} done ===")
    print(f"Best val_acc={best_val_acc:.4f} val_f1={best_f1:.4f} @ epoch {best_epoch}")
    if v_labels:
        print(classification_report(v_labels, v_preds,
                                    target_names=[IDX_TO_EMOTION[i] for i in range(4)]))
    summary_path = CHECKPOINT_DIR / f"v9b_audio_fold{args.fold}_seed{args.seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": "WavLM-Base+ + unfreeze last 2 + mean-pool + Linear head",
            "fold": args.fold, "seed": args.seed,
            "n_train": len(train_manifest), "n_val": len(val_manifest),
            "best_val_acc": best_val_acc, "best_val_f1": best_f1,
            "best_epoch": best_epoch, "history": history,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
