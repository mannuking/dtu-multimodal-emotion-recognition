"""
train_ser_v9_text_only.py — Text-only baseline on IEMOCAP-4
(BERT-base-uncased frozen + linear head).

Companion to v9 baseline. This is the TEXT row of MemoCMT Table 1:
BERT [CLS] → LayerNorm → Dropout → Linear(768, 4). Same 5-fold random
splits, same IEMOCAP-4 manifest, same seeds as v9 multimodal.

Usage on HPC:
  uv run python train_ser_v9_text_only.py --fold 0 --seed 42
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
    IDX_TO_EMOTION,
)

warnings.filterwarnings("ignore")

CHECKPOINT_DIR = Path("model_checkpoints")
TARGET_SR = 16000
MAX_TEXT_LEN = 64


class BERTHead(nn.Module):
    """BERT-base-uncased (frozen) → [CLS] → LayerNorm → Dropout → Linear(768, 4)."""
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.bert(input_ids=input_ids,
                         attention_mask=attn_mask).last_hidden_state[:, 0, :]  # CLS

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cls = self.encode(input_ids, attn_mask)
        return self.head(cls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
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

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_loader = DataLoader(
        IEMOCAPDataset(train_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        IEMOCAPDataset(val_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BERTHead(num_classes=4).to(device)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc, best_epoch, best_f1 = 0.0, -1, 0.0
    history, epochs_no_improve = [], 0
    v_labels, v_preds = [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        model.bert.eval()
        tr_loss = tr_correct = tr_total = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            emo = batch["emotion"].to(device)
            logits = model(ids, am)
            loss = F.cross_entropy(logits, emo)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * ids.size(0)
            tr_correct += (logits.argmax(1) == emo).sum().item()
            tr_total += ids.size(0)
        train_loss = tr_loss / tr_total
        train_acc = tr_correct / tr_total

        model.eval()
        v_correct = v_total = 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                am = batch["attention_mask"].to(device)
                emo = batch["emotion"].to(device)
                logits = model(ids, am)
                v_correct += (logits.argmax(1) == emo).sum().item()
                v_total += ids.size(0)
                v_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                v_labels.extend(emo.cpu().numpy().tolist())
        val_acc = v_correct / v_total
        val_f1 = f1_score(v_labels, v_preds, average="macro")
        scheduler.step()

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, best_f1 = val_acc, epoch, val_f1
            epochs_no_improve = 0
            ckpt_path = CHECKPOINT_DIR / f"v9_text_fold{args.fold}_seed{args.seed}.pt"
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
    summary_path = CHECKPOINT_DIR / f"v9_text_fold{args.fold}_seed{args.seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": "BERT-base-uncased (frozen) + Linear head",
            "fold": args.fold, "seed": args.seed,
            "n_train": len(train_manifest), "n_val": len(val_manifest),
            "best_val_acc": best_val_acc, "best_val_f1": best_f1,
            "best_epoch": best_epoch, "history": history,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()