"""
train_ser_v10_fusion.py — Cross-attention fusion of v9b text + audio ckpts.

Architecture (MemoCMT-inspired CMT block):
  1. Load frozen WavLM-base+ (audio) and DeBERTa-v3-base (text) from v9b ckpts.
  2. Extract per-utterance embeddings (mean-pool audio, CLS-pool text).
  3. Project both to d_model=512, add learned [MOD] tokens.
  4. 2x CrossAttentionBlock:
       text_out = text + MHA(Q=text, K=audio, V=audio)
       audio_out = audio + MHA(Q=audio, K=text, V=text)
       each followed by LayerNorm + residual FFN(512, 2048, 512).
  5. Tensor fusion: concat [text_out ; audio_out ; text_out*audio_out ; |text_out - audio_out|] -> (B, 2048).
  6. MLP head: Linear(2048, 512) -> GELU -> Dropout(0.3) -> Linear(512, 4).
  7. CE loss (uniform OR class-weighted via --class-weighted).

Expected val_acc: 0.80-0.83 (uniform), 0.81-0.84 (class-weighted).

Usage on HPC:
  uv run python train_ser_v10_fusion.py --fold 0 --seed 42 --class-weighted false
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

import sys
# Add working tree root to sys.path so `iemocap_dataset` resolves
# (the module lives at repo root, this script lives at scripts/iemocap/).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_random_kfold_splits,
    IDX_TO_EMOTION,
)

warnings.filterwarnings("ignore")

# Resolve checkpoint dir to repo root, not cwd - same reason as --manifest.
# sbatch does `cd scripts/iemocap`, so bare "model_checkpoints/" would
# resolve to scripts/iemocap/model_checkpoints/ which doesn't exist.
# The real ckpts live at the repo root: ~/Research/.../model_checkpoints/
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "model_checkpoints"
WAVLM_NAME = "microsoft/wavlm-base-plus"
DEBERTA_NAME = "microsoft/deberta-v3-base"
TARGET_SR = 16000
MAX_TEXT_LEN = 64

# Class weights calibrated for IEMOCAP-4 with angry under-represented.
# angry=0, happy=1, neutral=2, sad=3 in EMOTION_TO_IDX.
CLASS_WEIGHTS = torch.tensor([1.30, 1.00, 1.00, 1.15])  # +angry, +sad


class FrozenEncoder(nn.Module):
    """Loads a v9b checkpoint and exposes its forward as embedding extraction."""

    def __init__(self, ckpt_path: Path, kind: str):
        super().__init__()
        if kind == "audio":
            self.model = AutoModel.from_pretrained(WAVLM_NAME)
        else:
            self.model = AutoModel.from_pretrained(DEBERTA_NAME)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # v9b saves the wrapped model (DebertaV3Head or WavLMHead), so the
        # state_dict has both "deberta.*"/"wavlm.*" (encoder) and "head.*" (cls).
        # We only want the encoder weights - keep only keys with the right prefix,
        # then strip that prefix so the bare AutoModel can load them.
        
        prefix = "deberta." if kind == "text" else "wavlm."
        state_dict = {
            k.removeprefix(prefix): v
            for k, v in state["model"].items()
            if k.startswith(prefix)
        }
        self.model.load_state_dict(state_dict)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.kind = kind

    @torch.no_grad()
    def extract(self, batch: dict) -> torch.Tensor:
        if self.kind == "audio":
            wav = batch["wav"].to(next(self.model.parameters()).device)
            out = self.model(wav)
            return out.last_hidden_state.mean(dim=1)  # (B, 768) mean-pool
        else:
            ids = batch["input_ids"].to(next(self.model.parameters()).device)
            am = batch["attention_mask"].to(next(self.model.parameters()).device)
            out = self.model(input_ids=ids, attention_mask=am)
            return out.last_hidden_state[:, 0, :]  # (B, 768) CLS


class CrossAttentionBlock(nn.Module):
    """One CMT block: bidirectional cross-attention + FFN, with residuals."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, ffn_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.text_to_audio = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.audio_to_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_t1 = nn.LayerNorm(d_model)
        self.norm_a1 = nn.LayerNorm(d_model)
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, d_model),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, d_model),
        )
        self.norm_t2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        # Cross-attention
        t2a, _ = self.text_to_audio(text, audio, audio)
        a2t, _ = self.audio_to_text(audio, text, text)
        text = self.norm_t1(text + t2a)
        audio = self.norm_a1(audio + a2t)
        # FFN
        text = self.norm_t2(text + self.ffn_t(text))
        audio = self.norm_a2(audio + self.ffn_a(audio))
        return text, audio


class CMFusionHead(nn.Module):
    """Cross-modal transformer fusion head.

    Inputs: per-utterance text emb (B, 768) and audio emb (B, 768) from frozen
    v9b ckpts. Outputs: logits (B, 4).
    """

    def __init__(self, d_model: int = 512, n_blocks: int = 2,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.text_proj = nn.Linear(768, d_model)
        self.audio_proj = nn.Linear(768, d_model)
        self.text_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.audio_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Tensor fusion: concat [t ; a ; t*a ; |t-a|] -> 4*d_model
        fusion_dim = 4 * d_model
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor):
        # Project to d_model and add [MOD] token (sequence length 2)
        B = text_emb.size(0)
        t = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, d)
        a = self.audio_proj(audio_emb).unsqueeze(1)
        t = t + self.text_cls
        a = a + self.audio_cls
        for block in self.blocks:
            t, a = block(t, a)
        t = self.norm(t).squeeze(1)  # (B, d)
        a = self.norm(a).squeeze(1)
        # Tensor fusion
        fused = torch.cat([t, a, t * a, torch.abs(t - a)], dim=-1)
        return self.head(fused)


def main():
    parser = argparse.ArgumentParser()
    # Resolve manifest path relative to the working tree root (where this repo
    # lives), NOT relative to cwd — sbatch does `cd scripts/iemocap` before
    # invoking Python, so a bare "data/iemocap/manifest.csv" would resolve to
    # the wrong place. Path(__file__).parent.parent.parent = repo root.
    REPO_ROOT = Path(__file__).parent.parent.parent
    default_manifest = str(REPO_ROOT / "data" / "iemocap" / "manifest.csv")
    parser.add_argument("--manifest", default=default_manifest)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)  # was 32, lowered for memory + num_workers=0
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--class-weighted", type=str, default="false",
                        choices=["true", "false"])
    # ckpt dirs default to repo-root CHECKPOINT_DIR (already resolved above)
    parser.add_argument("--ckpt-text-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--ckpt-audio-dir", default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_class_weights = args.class_weighted.lower() == "true"
    print(f"Device: {device} | Seed: {args.seed} | Fold: {args.fold} | "
          f"class_weighted: {use_class_weights}")

    manifest = load_manifest(Path(args.manifest))
    splits = make_random_kfold_splits(manifest, n_folds=5, seed=args.seed)
    train_idx, val_idx = splits[args.fold]
    train_manifest = [manifest[i] for i in train_idx]
    val_manifest = [manifest[i] for i in val_idx]
    print(f"Train: {len(train_manifest)}  Val: {len(val_manifest)}")

    # Tokenizer needed for the dataset (frozen encoder reads input_ids/attn)
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    train_loader = DataLoader(
        IEMOCAPDataset(train_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(
        IEMOCAPDataset(val_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Load frozen v9b encoders (one ckpt per fold/seed)
    text_ckpt = (Path(args.ckpt_text_dir)
                 / f"v9b_text_fold{args.fold}_seed{args.seed}.pt")
    audio_ckpt = (Path(args.ckpt_audio_dir)
                  / f"v9b_audio_fold{args.fold}_seed{args.seed}.pt")
    if not text_ckpt.exists() or not audio_ckpt.exists():
        print(f"FATAL: missing v9b ckpts: {text_ckpt} or {audio_ckpt}")
        sys.exit(1)
    text_enc = FrozenEncoder(text_ckpt, kind="text").to(device)
    audio_enc = FrozenEncoder(audio_ckpt, kind="audio").to(device)

    # Trainable fusion head
    head = CMFusionHead().to(device)
    head_params = [p for p in head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in head_params)
    print(f"Trainable params (fusion head only): {n_trainable:,}")

    optimizer = AdamW(head_params, lr=args.lr_head, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda)

    if use_class_weights:
        class_weights = CLASS_WEIGHTS.to(device)
    else:
        class_weights = None

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    weight_tag = "cw" if use_class_weights else "unif"
    best_val_acc, best_epoch, best_f1 = 0.0, -1, 0.0
    history, epochs_no_improve = [], 0

    v_preds, v_labels = [], []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        head.train()
        tr_loss = tr_correct = tr_total = 0
        for batch in train_loader:
            emo = batch["emotion"].to(device)
            with torch.no_grad():
                text_emb = text_enc.extract(batch)
                audio_emb = audio_enc.extract(batch)
            logits = head(text_emb, audio_emb)
            loss = F.cross_entropy(logits, emo, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * emo.size(0)
            tr_correct += (logits.argmax(1) == emo).sum().item()
            tr_total += emo.size(0)
        train_loss = tr_loss / tr_total
        train_acc = tr_correct / tr_total

        head.eval()
        v_correct = v_total = 0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                emo = batch["emotion"].to(device)
                text_emb = text_enc.extract(batch)
                audio_emb = audio_enc.extract(batch)
                logits = head(text_emb, audio_emb)
                v_correct += (logits.argmax(1) == emo).sum().item()
                v_total += emo.size(0)
                v_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                v_labels.extend(emo.cpu().numpy().tolist())
        val_acc = v_correct / v_total
        val_f1 = f1_score(v_labels, v_preds, average="macro")
        # Unweighted accuracy = W-Acc; class-balanced = UA-Acc
        val_ua = accuracy_score(v_labels, v_preds)

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, best_f1 = val_acc, epoch, val_f1
            epochs_no_improve = 0
            ckpt_path = (CHECKPOINT_DIR
                         / f"v10_fusion_{weight_tag}_fold{args.fold}"
                         f"_seed{args.seed}.pt")
            torch.save({
                "head": head.state_dict(),
                "fold": args.fold,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "seed": args.seed,
                "class_weighted": use_class_weights,
            }, ckpt_path)
            saved = f"  ✅ {ckpt_path.name}"
        else:
            epochs_no_improve += 1
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc,
                        "val_f1": val_f1, "val_ua": val_ua})
        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_ua={val_ua:.4f}  "
              f"({time.time()-t0:.0f}s){saved}")

        if epochs_no_improve >= args.patience:
            print(f"Early stop @ epoch {epoch} (no improve for {args.patience})")
            break

    print(f"\n=== fold{args.fold} seed{args.seed} ({weight_tag}) done ===")
    print(f"Best val_acc={best_val_acc:.4f} val_f1={best_f1:.4f} @ epoch {best_epoch}")
    if v_labels is not None and len(v_labels) > 0:
        print(classification_report(v_labels, v_preds,
                                    target_names=[IDX_TO_EMOTION[i] for i in range(4)]))
    summary_path = (CHECKPOINT_DIR
                    / f"v10_fusion_{weight_tag}_fold{args.fold}"
                    f"_seed{args.seed}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": "v10-B CMT fusion head on frozen v9b WavLM-base+/DeBERTa-v3-base",
            "fold": args.fold, "seed": args.seed,
            "class_weighted": use_class_weights,
            "n_train": len(train_manifest), "n_val": len(val_manifest),
            "best_val_acc": best_val_acc, "best_val_f1": best_f1,
            "best_epoch": best_epoch, "history": history,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
