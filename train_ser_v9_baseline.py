"""
train_ser_v9_baseline.py — MemoCMT baseline reproduction on IEMOCAP-4-class.

This is the floor. We start from MemoCMT's exact recipe before adding our
novel contributions (V/A bias, dialog context). Target: 81.85% W-Acc.

Reference: Khan et al. 2025, "MemoCMT: multimodal emotion recognition using
cross-modal transformer-based feature fusion", Scientific Reports.

Recipe (MemoCMT, IEMOCAP-4-class):
  - Audio: HuBERT (frozen, facebook/hubert-base-ls960)
  - Text:  BERT (frozen, bert-base-uncased)
  - CMT: bidirectional cross-attention, NO V/A bias
  - Aggregation: MIN (best on IEMOCAP per paper)
  - Optimizer: Adam, lr=1e-4, step LR × 0.1 every 30 epochs
  - Epochs: 100 (with patience-based early stopping)
  - Batch size: 1
  - 5-fold CV with random splits (NOT LOSO)

Usage (on HPC compute node):
  uv run python train_ser_v9_baseline.py --fold 0 --seed 42
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
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_random_kfold_splits,
    EMOTION_TO_IDX, IDX_TO_EMOTION,
)
from cmt_fusion import (
    FusionConfig, PureCMT,
)

warnings.filterwarnings("ignore")

CHECKPOINT_DIR = Path("model_checkpoints")
TARGET_SR = 16000
MAX_TEXT_LEN = 64


class FrozenEncoders(nn.Module):
    """HuBERT (audio) + BERT (text), both frozen. MemoCMT's exact setup."""
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.audio = AutoModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
        self.text = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
        for p in self.audio.parameters():
            p.requires_grad = False
        for p in self.text.parameters():
            p.requires_grad = False
        self.audio_dim = self.audio.config.hidden_size   # 768
        self.text_dim = self.text.config.hidden_size    # 768

    @torch.no_grad()
    def encode_audio(self, wav: torch.Tensor) -> torch.Tensor:
        return self.audio(wav).last_hidden_state

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.text(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    parser.add_argument("--fold", type=int, required=True,
                        help="Random 5-fold CV index 0-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience on val_acc")
    parser.add_argument("--aggregation", choices=["min", "mean", "max", "cls"],
                        default="min", help="MemoCMT tried all 4; min wins on IEMOCAP")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {args.seed}, Fold: {args.fold}, Aggregation: {args.aggregation}")

    # Load manifest
    manifest = load_manifest(Path(args.manifest))
    splits = make_random_kfold_splits(manifest, n_folds=5, seed=args.seed)
    train_idx, val_idx = splits[args.fold]
    train_manifest = [manifest[i] for i in train_idx]
    val_manifest = [manifest[i] for i in val_idx]
    print(f"Train: {len(train_manifest)} utts")
    print(f"Val:   {len(val_manifest)} utts")

    # Build datasets
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_ds = IEMOCAPDataset(train_manifest, tokenizer, target_sr=TARGET_SR,
                              max_text_len=MAX_TEXT_LEN)
    val_ds = IEMOCAPDataset(val_manifest, tokenizer, target_sr=TARGET_SR,
                            max_text_len=MAX_TEXT_LEN)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Models
    encoders = FrozenEncoders(device)
    cfg = FusionConfig(
        audio_dim=encoders.audio_dim,
        text_dim=encoders.text_dim,
        aggregation=args.aggregation,
    )
    fusion = PureCMT(cfg).to(device)
    optimizer = Adam(fusion.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    print(f"Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")

    # Training loop
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0
    best_epoch = -1
    best_val_f1 = 0.0
    history = []
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # ---- TRAIN ----
        fusion.train()
        train_loss = train_correct = train_total = 0
        for batch in train_loader:
            wav = batch["wav"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            emotion = batch["emotion"].to(device)

            with torch.no_grad():
                audio_h = encoders.encode_audio(wav)
                text_h = encoders.encode_text(input_ids, attn_mask)
            logits = fusion(audio_h, text_h)
            loss = F.cross_entropy(logits, emotion)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * wav.size(0)
            train_correct += (logits.argmax(1) == emotion).sum().item()
            train_total += wav.size(0)
        train_acc = train_correct / train_total
        train_loss /= train_total

        # ---- VAL ----
        fusion.eval()
        val_correct = val_total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                wav = batch["wav"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                emotion = batch["emotion"].to(device)
                audio_h = encoders.encode_audio(wav)
                text_h = encoders.encode_text(input_ids, attn_mask)
                logits = fusion(audio_h, text_h)
                val_correct += (logits.argmax(1) == emotion).sum().item()
                val_total += wav.size(0)
                val_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                val_labels.extend(emotion.cpu().numpy().tolist())
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        scheduler.step()
        elapsed = time.time() - t0
        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt = {
                "fusion": fusion.state_dict(),
                "fold": args.fold,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": cfg.__dict__,
                "aggregation": args.aggregation,
            }
            ckpt_path = CHECKPOINT_DIR / f"v9_baseline_fold{args.fold}_seed{args.seed}.pt"
            torch.save(ckpt, ckpt_path)
            saved = f"  ✅ saved {ckpt_path.name}"
        else:
            epochs_no_improve += 1
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_f1": val_f1,
        })
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}  "
              f"({elapsed:.0f}s){saved}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Final report
    print(f"\n=== Fold {args.fold} done ===")
    print(f"Best val_acc={best_val_acc:.4f} val_f1={best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Class report at best epoch:")
    print(classification_report(val_labels, val_preds,
                                target_names=[IDX_TO_EMOTION[i] for i in range(4)]))
    summary_path = CHECKPOINT_DIR / f"v9_baseline_fold{args.fold}_seed{args.seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "fold": args.fold,
            "n_train": len(train_manifest),
            "n_val": len(val_manifest),
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "seed": args.seed,
            "aggregation": args.aggregation,
            "history": history,
        }, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
