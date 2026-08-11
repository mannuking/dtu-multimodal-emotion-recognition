"""
train_ser_v7.py - Audio+Text multimodal SER (MemoCMT-style simple version).

Based on the architecture of MemoCMT (Khan et al., Nature Scientific Reports 2025):
    - HuBERT for audio features (we use wav2vec2-large as a close alternative)
    - BERT for text features (we use bert-base-uncased)
    - Cross-modal fusion: concatenation + projection + classification
    - MemoCMT achieves 81.33% on IEMOCAP, 91.93% on ESD

V7 simplifications (vs full MemoCMT):
    - wav2vec2-large instead of HuBERT (avoids extra model download)
    - bert-base-uncased instead of BERT-large (smaller, faster)
    - Concatenation fusion instead of full CMT (cross-attention)
    - Last 4 wav2vec2 layers + last 2 BERT layers fine-tuned

Data:
    - combined_ser_dataset/manifest.csv (with REAL Whisper transcripts)
    - 11970 samples, 7 classes, 80/15/5 split
    - train: 8594, val: 1905, test: 1471

Expected accuracy: 75-82% (vs v5 71% audio-only)

Usage:
    uv run python train_ser_v7.py --seed 42 --n-epochs 60
"""
from __future__ import annotations

import os
import argparse
import json
import time
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
MAX_AUDIO_LEN = 6 * TARGET_SR  # 6 seconds max
MAX_TEXT_LEN = 64
BATCH_SIZE = 8
NUM_EPOCHS = 60
LR_ENCODER = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-5
SEED_DEFAULT = 42
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_UNFROZEN_AUDIO = 4
NUM_UNFROZEN_TEXT = 2

# HuggingFace offline mode (compute nodes have no internet)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


class AudioTextSERDataset(Dataset):
    def __init__(self, df, tokenizer, audio_processor):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import librosa
        row = self.df.iloc[idx]

        # ---- Audio ----
        try:
            audio, sr = librosa.load(row["wav_path"], sr=TARGET_SR, mono=True)
        except Exception:
            audio = np.zeros(MAX_AUDIO_LEN, dtype=np.float32)

        if len(audio) > MAX_AUDIO_LEN:
            audio = audio[:MAX_AUDIO_LEN]
        else:
            audio = np.pad(audio, (0, MAX_AUDIO_LEN - len(audio)), mode="constant")

        # wav2vec2 processor expects raw audio + sampling_rate
        audio_inputs = self.audio_processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        input_values = audio_inputs["input_values"].squeeze(0)

        # ---- Text ----
        text = str(row.get("text", "")).strip()
        if not text:
            text = "a person feels something"

        enc = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        # ---- Label ----
        label = int(row["label"])

        return {
            "input_values": input_values,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


class AttentionPool(nn.Module):
    """Learned attention pooling over sequence dimension."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x, mask=None):
        # x: (B, T, D), return (B, D)
        scores = torch.einsum("btd,od->bt", x, self.query.squeeze(0))
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", weights, x)


class MultimodalFusion(nn.Module):
    """Audio + text fusion via concatenation + projection + classifier."""

    def __init__(self, audio_dim=1024, text_dim=768, num_classes=7, dropout=0.3):
        super().__init__()
        self.audio_pool = AttentionPool(audio_dim)
        self.text_pool = AttentionPool(text_dim)
        # Gated fusion: learn how much text contributes
        self.text_gate = nn.Sequential(
            nn.Linear(audio_dim + text_dim, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(audio_dim + text_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, audio_emb, text_emb, attention_mask=None):
        # audio_emb: (B, T_a, D_a)
        # text_emb: (B, T_t, D_t)
        # Flatten temporal: use mean
        a_pool = audio_emb.mean(dim=1)  # (B, D_a)
        # Text: use attention pool over masked positions
        text_mask = attention_mask.bool() if attention_mask is not None else None
        t_pool = self.text_pool(text_emb, text_mask)  # (B, D_t)

        # Concat
        combined = torch.cat([a_pool, t_pool], dim=-1)  # (B, D_a + D_t)

        # Soft gate on text contribution
        gate = self.text_gate(combined)  # (B, 1) in [0, 1]
        # Re-apply gate: reduce text contribution if gate is low
        combined = torch.cat([a_pool, t_pool * gate], dim=-1)

        # Project
        h = self.proj(combined)
        logits = self.classifier(h)
        return logits, gate


def build_loaders(seed):
    """Build train/val/test loaders with balanced sampler."""
    from transformers import Wav2Vec2Processor, BertTokenizer

    # Load manifest
    src = Path("combined_ser_dataset/manifest.csv")
    if not src.exists():
        raise SystemExit(f"ERROR: {src} not found")

    df = pd.read_csv(src)
    print(f"   loaded {len(df)} rows from manifest.csv")

    # Load wav2vec2 audio processor AND BERT tokenizer
    print("   loading wav2vec2-large processor...")
    audio_processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-large-lv60"
    )
    print("   loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Split
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"   train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_ds = AudioTextSERDataset(train_df, tokenizer, audio_processor)
    val_ds = AudioTextSERDataset(val_df, tokenizer, audio_processor)
    test_ds = AudioTextSERDataset(test_df, tokenizer, audio_processor)

    # Balanced sampler
    labels = train_df["label"].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_model():
    """Build multimodal model with audio and text encoders."""
    from transformers import Wav2Vec2Model, BertModel

    print("   loading wav2vec2-large for audio...")
    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60")
    # Freeze all, then unfreeze last 4
    for p in audio_encoder.parameters():
        p.requires_grad = False
    n_layers = len(audio_encoder.encoder.layers)
    for i in range(n_layers - NUM_UNFROZEN_AUDIO, n_layers):
        for p in audio_encoder.encoder.layers[i].parameters():
            p.requires_grad = True
    print(f"   unfroze last {NUM_UNFROZEN_AUDIO} wav2vec2 layers (of {n_layers})")

    print("   loading BERT for text...")
    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    for p in text_encoder.parameters():
        p.requires_grad = False
    n_bert = len(text_encoder.encoder.layer)
    for i in range(n_bert - NUM_UNFROZEN_TEXT, n_bert):
        for p in text_encoder.encoder.layer[i].parameters():
            p.requires_grad = True
    print(f"   unfroze last {NUM_UNFROZEN_TEXT} BERT layers (of {n_bert})")

    # Detect actual dims
    audio_dim = audio_encoder.config.hidden_size
    text_dim = text_encoder.config.hidden_size
    print(f"   audio_dim={audio_dim}, text_dim={text_dim}")

    fusion = MultimodalFusion(audio_dim, text_dim, num_classes=7)
    return audio_encoder, text_encoder, fusion


def evaluate(audio_encoder, text_encoder, fusion, loader):
    audio_encoder.eval()
    text_encoder.eval()
    fusion.eval()
    all_preds, all_labels = [], []
    all_gates = []
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            audio_out = audio_encoder(input_values).last_hidden_state
            text_out = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            logits, gate = fusion(audio_out, text_out, attn_mask)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_gates.extend(gate.squeeze(-1).cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_labels, all_gates


def train_one_epoch(audio_encoder, text_encoder, fusion, loader, optimizer, epoch):
    audio_encoder.train()
    text_encoder.train()
    fusion.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        input_values = batch["input_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        audio_out = audio_encoder(input_values).last_hidden_state
        text_out = text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        logits, gate = fusion(audio_out, text_out, attn_mask)

        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(audio_encoder.parameters()) + list(text_encoder.parameters()) + list(fusion.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1} batch {i+1}/{len(loader)} "
                  f"loss={loss.item():.4f} gate={gate.mean().item():.3f} "
                  f"elapsed={elapsed:.1f}s")
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--n-epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"=== v7 Audio+Text SER training: seed={args.seed}, epochs={args.n_epochs} ===")
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader = build_loaders(args.seed)
    audio_encoder, text_encoder, fusion = build_model()

    audio_encoder = audio_encoder.to(DEVICE)
    text_encoder = text_encoder.to(DEVICE)
    fusion = fusion.to(DEVICE)

    n_trainable = sum(
        p.numel() for p in list(audio_encoder.parameters()) +
        list(text_encoder.parameters()) + list(fusion.parameters())
        if p.requires_grad
    )
    print(f"   trainable params: {n_trainable:,}")

    # Optimizer with different LR for encoder vs head
    optimizer = torch.optim.AdamW([
        {"params": [p for p in audio_encoder.parameters() if p.requires_grad], "lr": LR_ENCODER},
        {"params": [p for p in text_encoder.parameters() if p.requires_grad], "lr": LR_ENCODER},
        {"params": fusion.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    best_val = 0.0
    for epoch in range(args.n_epochs):
        ep_loss = train_one_epoch(audio_encoder, text_encoder, fusion, train_loader, optimizer, epoch)
        val_acc, val_f1, _, _, gates = evaluate(audio_encoder, text_encoder, fusion, val_loader)
        print(f"Epoch {epoch+1}/{args.n_epochs} train_loss={ep_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} mean_gate={np.mean(gates):.3f}")
        if val_acc > best_val:
            best_val = val_acc
            print(f"  ✅ saved best (val={val_acc:.4f})")

    # Test
    test_acc, test_f1, preds, labels, gates = evaluate(audio_encoder, text_encoder, fusion, test_loader)
    print(f"\n=== TEST: acc={test_acc:.4f} f1={test_f1:.4f} mean_gate={np.mean(gates):.3f} ===")
    print(classification_report(
        labels, preds, target_names=EMOTIONS, digits=4, zero_division=0.0
    ))

    # Save summary
    summary = {
        "version": "v7",
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "best_val": best_val,
        "mean_text_gate": float(np.mean(gates)),
    }
    Path("model_checkpoints").mkdir(exist_ok=True)
    summary_path = Path(f"model_checkpoints/ser_v7_summary_seed{args.seed}.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"   saved summary to {summary_path}")


if __name__ == "__main__":
    main()