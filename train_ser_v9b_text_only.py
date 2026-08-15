"""
train_ser_v9b_text_only.py — Text-only baseline on IEMOCAP-4
(DeBERTa-v3-base, last 2 transformer layers unfrozen + linear head).

This is the V9B TEXT row of MemoCMT Table 1. Replaces v9's frozen BERT-base
(which scored 0.5971) with DeBERTa-v3-base + last 2 layers unfrozen. Same
5-fold random splits, same IEMOCAP-4 manifest, same seeds (42, 43, 44)
as v9 multimodal. Expected val_acc: 0.74-0.78.

Usage on HPC:
  uv run python train_ser_v9b_text_only.py --fold 0 --seed 42
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

# DeBERTa-v3-base has 12 encoder layers; unfreeze last 2.
DEBERTA_NAME = "microsoft/deberta-v3-base"
UNFREEZE_LAST_N = 2


class DebertaV3Head(nn.Module):
    """DeBERTa-v3-base + unfreeze last N transformer layers + LayerNorm + Dropout + Linear."""
    def __init__(self, num_classes: int = 4, dropout: float = 0.3,
                 unfreeze_last_n: int = UNFREEZE_LAST_N):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(DEBERTA_NAME)
        hidden = self.deberta.config.hidden_size  # 768 for base
        # Freeze embeddings + most encoder layers, leave only the last N trainable.
        for p in self.deberta.parameters():
            p.requires_grad = False
        # The encoder is a DebertaV2Encoder (ModelOutput). Its layers live in
        # `.encoder.layer` (a nn.ModuleList). Re-enable grads on the last N.
        try:
            layers = self.deberta.encoder.layer
        except AttributeError:
            # Some HF versions nest under encoder.encoder
            layers = self.deberta.encoder.encoder.layer
        n_total = len(layers)
        for i in range(n_total - unfreeze_last_n, n_total):
            for p in layers[i].parameters():
                p.requires_grad = True
        # Also unfreeze the encoder's final LayerNorm (DebertaV2 has encoder.LayerNorm)
        for attr in ("LayerNorm", "layer_norm"):
            ln = getattr(self.deberta.encoder, attr, None) if hasattr(self.deberta, "encoder") else None
            if ln is not None:
                for p in ln.parameters():
                    p.requires_grad = True
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    @torch.no_grad()
    def encode_frozen(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.deberta(input_ids=input_ids,
                            attention_mask=attn_mask).last_hidden_state[:, 0, :]

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Full forward (grad-enabled for last N layers).
        out = self.deberta(input_ids=input_ids, attention_mask=attn_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-head", type=float, default=2e-3,
                        help="LR for classifier head (frozen encoder layers see no LR)")
    parser.add_argument("--lr-encoder", type=float, default=2e-5,
                        help="LR for the unfrozen last 2 encoder layers")
    parser.add_argument("--batch-size", type=int, default=16)
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

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    train_loader = DataLoader(
        IEMOCAPDataset(train_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        IEMOCAPDataset(val_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DebertaV3Head(num_classes=4).to(device)
    # Param groups: head vs last-2-encoder layers (vs frozen everything else gets no LR)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    encoder_params = [p for p in model.deberta.parameters() if p.requires_grad]
    n_enc_trainable = sum(p.numel() for p in encoder_params)
    print(f"Trainable params: head={sum(p.numel() for p in head_params):,} "
          f"+ encoder(last {UNFREEZE_LAST_N})={n_enc_trainable:,} "
          f"= {sum(p.numel() for p in head_params) + n_enc_trainable:,}")

    optimizer = AdamW([
        {"params": head_params, "lr": args.lr_head},
        {"params": encoder_params, "lr": args.lr_encoder},
    ], weight_decay=0.01)

    # Linear warmup over 2 epochs, then linear decay to 0.
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
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            emo = batch["emotion"].to(device)
            logits = model(ids, am)
            loss = F.cross_entropy(logits, emo)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
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

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, best_f1 = val_acc, epoch, val_f1
            epochs_no_improve = 0
            ckpt_path = CHECKPOINT_DIR / f"v9b_text_fold{args.fold}_seed{args.seed}.pt"
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
    summary_path = CHECKPOINT_DIR / f"v9b_text_fold{args.fold}_seed{args.seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": "DeBERTa-v3-base + unfreeze last 2 + Linear head",
            "fold": args.fold, "seed": args.seed,
            "n_train": len(train_manifest), "n_val": len(val_manifest),
            "best_val_acc": best_val_acc, "best_val_f1": best_f1,
            "best_epoch": best_epoch, "history": history,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
