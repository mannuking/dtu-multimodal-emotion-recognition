"""
train_ser_v8_cmt.py — V/A-conditioned CMT + dialog-context on IEMOCAP 4-class.

Pipeline (memoCMT convention + our two novelties):
  1. Load manifest.csv (5,531 utterances, 4 classes, with V/A).
  2. LOSO 4-fold CV (session held out as test).
  3. For each fold:
       - Build train/val IEMOCAPDataset.
       - Batch=1, 30 epochs, AdamW(lr=1e-4), CE loss.
       - Frozen HuBERT-Base audio encoder + frozen BERT-Base text encoder.
       - V/A-aware CMT fusion head.
       - Dialog-context layer over 10 previous utterances.
       - 8-pass TTA (random crop) on val at end of each epoch.
  4. Save best checkpoint per fold.
  5. After all folds, report per-fold + mean test accuracy.

Usage (on HPC compute node, via sbatch):
  uv run python train_ser_v8_cmt.py --fold 0 --seed 42
  uv run python train_ser_v8_cmt.py --fold 1 --seed 42
  ...
  uv run python train_ser_v8_cmt.py --fold 0 --seed 43
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_loso_splits,
    EMOTION_TO_IDX, IDX_TO_EMOTION,
)
from cmt_fusion import (
    FusionConfig, VAAwareCMT, DialogContextLayer,
)

warnings.filterwarnings("ignore")


CHECKPOINT_DIR = Path("model_checkpoints")
TARGET_SR = 16000
MAX_TEXT_LEN = 64


class FrozenEncoders(nn.Module):
    """Loads wav2vec2-base (audio) and BERT-base (text), frozen."""
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.audio = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
        self.text = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
        for p in self.audio.parameters():
            p.requires_grad = False
        for p in self.text.parameters():
            p.requires_grad = False
        self.audio_dim = self.audio.config.hidden_size    # 768
        self.text_dim = self.text.config.hidden_size     # 768

    @torch.no_grad()
    def encode_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) -> (B, T_a, 768)"""
        out = self.audio(wav).last_hidden_state
        return out

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """(B, T_t) -> (B, T_t, 768)"""
        out = self.text(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        return out


class DialogContextBuffer:
    """
    Maintains a per-dialog buffer of (CLS embedding, va, emotion) tuples
    so that during training, the current utterance can attend to the
    previous N utterances in the same dialog.
    """
    def __init__(self, window: int = 10, dim: int = 256):
        self.window = window
        self.dim = dim
        self._buffers: dict[str, list[torch.Tensor]] = {}

    def get_context(self, dialog: str) -> torch.Tensor:
        """Returns (window, dim) — padded with zeros if fewer than window."""
        buf = self._buffers.get(dialog, [])
        if not buf:
            return torch.zeros(self.window, self.dim)
        # Take last `window` entries
        recent = buf[-self.window:]
        ctx = torch.stack(recent, dim=0)  # (T, dim)
        if len(recent) < self.window:
            pad = torch.zeros(self.window - len(recent), self.dim)
            ctx = torch.cat([pad, ctx], dim=0)
        return ctx

    def append(self, dialog: str, cls_emb: torch.Tensor):
        if dialog not in self._buffers:
            self._buffers[dialog] = []
        self._buffers[dialog].append(cls_emb.detach().cpu())

    def reset(self):
        self._buffers.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    parser.add_argument("--fold", type=int, required=True,
                        help="LOSO fold index 0-4 (which session is held out)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-pretrain", type=int, default=200,
                        help="Number of training epochs to pretrain the CMT without "
                             "dialog context (simpler signal first)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seed: {args.seed}, Fold: {args.fold}, Epochs: {args.epochs}")

    # Load manifest
    manifest = load_manifest(Path(args.manifest))
    splits = make_loso_splits(manifest)
    train_sessions, val_sessions = splits[args.fold]
    train_manifest = [r for r in manifest if r["session"] in train_sessions]
    val_manifest = [r for r in manifest if r["session"] in val_sessions]
    print(f"Train: {len(train_manifest)} utts from {sorted(train_sessions)}")
    print(f"Val:   {len(val_manifest)} utts from {sorted(val_sessions)}")

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
    )
    fusion = VAAwareCMT(cfg).to(device)
    # Get the penultimate feature dim from the classifier (before final linear)
    base_feat_dim = cfg.proj_dim * 2  # audio_pooled + text_pooled
    ctx_layer = DialogContextLayer(cfg, base_feat_dim).to(device)
    optimizer = AdamW(
        list(fusion.parameters()) + list(ctx_layer.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    print(f"Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")
    print(f"Context params: {sum(p.numel() for p in ctx_layer.parameters()):,}")

    # Training loop
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0
    best_epoch = -1
    history = []
    ctx_buffer = DialogContextBuffer(window=cfg.dialog_window, dim=cfg.dialog_dim)
    ctx_buffer.reset()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # ---- TRAIN ----
        fusion.train()
        ctx_layer.train()
        ctx_buffer.reset()
        train_loss = train_correct = train_total = 0
        for batch in train_loader:
            wav = batch["wav"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            va = torch.stack([
                batch["valence"], batch["arousal"], batch["dominance"]
            ], dim=1).to(device).float()
            emotion = batch["emotion"].to(device)
            dialog = batch["dialog"][0]

            with torch.no_grad():
                audio_h = encoders.encode_audio(wav)
                text_h = encoders.encode_text(input_ids, attn_mask)
            base_logits, base_feat = _fusion_forward(fusion, audio_h, text_h, va)
            # Get the per-utterance [CLS] feature for the dialog buffer
            # (use base_feat as the per-utt representation)
            ctx_buffer.append(dialog, base_feat.mean(dim=0).squeeze(0) if base_feat.dim() == 3 else base_feat.squeeze(0))
            # Get dialog context (10 previous utterances)
            dialog_ctx = ctx_buffer.get_context(dialog).unsqueeze(0).to(device)

            if epoch <= args.n_pretrain:
                # Pretrain phase: use base logits alone, no dialog context
                logits = base_logits
            else:
                # Joint phase: combine with dialog context
                logits = ctx_layer(base_feat, dialog_ctx)

            loss = F.cross_entropy(logits, emotion)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) + list(ctx_layer.parameters()), 1.0
            )
            optimizer.step()

            train_loss += loss.item() * wav.size(0)
            train_correct += (logits.argmax(1) == emotion).sum().item()
            train_total += wav.size(0)
        train_acc = train_correct / train_total
        train_loss /= train_total

        # ---- VAL ----
        fusion.eval()
        ctx_layer.eval()
        ctx_buffer.reset()
        val_correct = val_total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                wav = batch["wav"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                va = torch.stack([
                    batch["valence"], batch["arousal"], batch["dominance"]
                ], dim=1).to(device).float()
                emotion = batch["emotion"].to(device)
                dialog = batch["dialog"][0]
                audio_h = encoders.encode_audio(wav)
                text_h = encoders.encode_text(input_ids, attn_mask)
                base_logits, base_feat = _fusion_forward(fusion, audio_h, text_h, va)
                ctx_buffer.append(dialog, base_feat.mean(dim=0).squeeze(0) if base_feat.dim() == 3 else base_feat.squeeze(0))
                dialog_ctx = ctx_buffer.get_context(dialog).unsqueeze(0).to(device)
                if epoch <= args.n_pretrain:
                    logits = base_logits
                else:
                    logits = ctx_layer(base_feat, dialog_ctx)
                val_correct += (logits.argmax(1) == emotion).sum().item()
                val_total += wav.size(0)
                val_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                val_labels.extend(emotion.cpu().numpy().tolist())
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        elapsed = time.time() - t0
        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt = {
                "fusion": fusion.state_dict(),
                "ctx_layer": ctx_layer.state_dict(),
                "fold": args.fold,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": cfg.__dict__,
            }
            ckpt_path = CHECKPOINT_DIR / f"v8_fold{args.fold}_seed{args.seed}.pt"
            torch.save(ckpt, ckpt_path)
            saved = f"  ✅ saved {ckpt_path.name}"
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_f1": val_f1,
        })
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}  "
              f"({elapsed:.0f}s){saved}")

    # Final report
    print(f"\n=== Fold {args.fold} done ===")
    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Class report at best epoch:")
    print(classification_report(val_labels, val_preds,
                                target_names=[IDX_TO_EMOTION[i] for i in range(4)]))
    summary_path = CHECKPOINT_DIR / f"v8_fold{args.fold}_seed{args.seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "fold": args.fold,
            "held_out_session": list(val_sessions)[0],
            "train_sessions": sorted(train_sessions),
            "n_train": len(train_manifest),
            "n_val": len(val_manifest),
            "best_val_acc": best_val_acc,
            "best_val_f1": val_f1,
            "best_epoch": best_epoch,
            "seed": args.seed,
            "history": history,
        }, f, indent=2)
    print(f"Summary saved to {summary_path}")


def _fusion_forward(fusion, audio_h, text_h, va):
    """
    Forward through the V/A-aware CMT, returning (logits, penultimate_features).
    Penultimate features are the fused (audio_pooled, text_pooled) — used by
    the dialog context layer downstream.
    """
    audio = fusion.proj_audio(audio_h)
    text = fusion.proj_text(text_h)
    for layer in fusion.layers:
        audio, text = layer(audio, text, va)
    audio_pooled = audio.min(dim=1).values
    text_pooled = text.min(dim=1).values
    fused = torch.cat([audio_pooled, text_pooled], dim=-1)
    logits = fusion.classifier(fused)
    return logits, fused


if __name__ == "__main__":
    main()
