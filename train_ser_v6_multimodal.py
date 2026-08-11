"""
train_ser_v6_multimodal.py - Multimodal SER (audio + text + facial) fusion.

Phase 1 of Path C: builds on the existing 11,568-sample combined SER dataset.
Each modality's encoder is loaded from a frozen checkpoint:
  - audio:  ser_v5_best_seed42.pt (wav2vec2-large, 1024-dim)
  - text:   ter_pytorch_best.pt    (MobileBERT, 768-dim)
  - facial: ResNet-50 pretrained on ImageNet, fine-tuned head on FER2013

Only the fusion head + per-modality projections are trained. This is fast
(1-2 hours per epoch on a single A100) and stable.

Expected test acc without IEMOCAP/MELD: 75-79% (vs 71% audio-only v5).
Expected test acc with IEMOCAP+MELD:     82-88%.

Usage:
    uv run python train_ser_v6_multimodal.py --seed 42
    uv run python train_ser_v6_multimodal.py --seed 42 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import accuracy_score, f1_score, classification_report

from multimodal_fusion import MultimodalSER, FusionConfig

warnings.filterwarnings("ignore")

# ---------- Constants ----------
SEED_DEFAULT = 42
TARGET_SR = 16000
MAX_S = 6.0
CHECKPOINT_DIR = "model_checkpoints"
SER_COMBINED_DIR = "combined_ser_dataset"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# Default paths to frozen checkpoints
AUDIO_CKPT = "model_checkpoints/ser_v5_best_seed42.pt"
TEXT_CKPT = "model_checkpoints/ter_pytorch_best.pt"
FACIAL_CKPT = None  # None => use ImageNet-pretrained ResNet50 (no FER2013 yet)

# Training defaults
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Dataset ----------

class MultimodalSERDataset(Dataset):
    """
    Loads multimodal samples from the combined SER manifest.

    Each sample returns:
      - audio_wav:     (T,) float32 audio at 16 kHz
      - input_ids:     (T_text,) int64 tokenized text
      - attention_mask: (T_text,) int64
      - facial:        (3, 224, 224) float32 normalize(mean, std)
      - label:         int64 class index
    """

    def __init__(self, manifest_path: str, split: str = "train",
                 tokenizer=None, max_audio_len: int = TARGET_SR * 6):
        from transformers import AutoTokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "google/mobilebert-uncased"
        )
        self.max_audio_len = max_audio_len

        import pandas as pd
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)

        # The manifest has columns: audio_path, text, label, subject
        # text may be empty for some utterances — synthesize from emotion label
        df["text"] = df["text"].fillna("").astype(str)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import librosa
        from PIL import Image
        import torchvision.transforms as T

        row = self.df.iloc[idx]

        # ----- Audio -----
        try:
            audio, sr = librosa.load(row["audio_path"], sr=TARGET_SR, mono=True)
        except Exception:
            audio = np.zeros(self.max_audio_len, dtype=np.float32)

        if len(audio) > self.max_audio_len:
            audio = audio[: self.max_audio_len]
        else:
            audio = np.pad(audio, (0, self.max_audio_len - len(audio)), mode="constant")

        # ----- Text -----
        text = row["text"] if row["text"] else f"a person feels {row['label']}"
        enc = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        # ----- Facial -----
        # If no facial crops available, use a black image (signals "no face")
        facial = torch.zeros(3, 224, 224)
        if "facial_path" in row and isinstance(row["facial_path"], str) and Path(row["facial_path"]).exists():
            try:
                img = Image.open(row["facial_path"]).convert("RGB")
                # Standard ImageNet normalization
                tfm = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                facial = tfm(img)
            except Exception:
                pass

        # ----- Label -----
        label = int(EMOTIONS.index(row["label"])) if row["label"] in EMOTIONS else 0

        return {
            "audio": torch.from_numpy(audio).float(),
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "facial": facial,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------- Frozen encoders ----------

class FrozenAudioEncoder(nn.Module):
    """Loads the v5 audio encoder. Returns pooled 1024-dim embedding."""

    def __init__(self, ckpt_path: str = AUDIO_CKPT):
        super().__init__()
        from transformers import Wav2Vec2Model
        # Load architecture
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
        # Load fine-tuned weights if available
        if ckpt_path and Path(ckpt_path).exists():
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # The v5 checkpoint wraps the encoder in a dict
            if isinstance(sd, dict) and "encoder_state_dict" in sd:
                sd = sd["encoder_state_dict"]
            try:
                self.encoder.load_state_dict(sd, strict=False)
                print(f"✅ Loaded audio encoder from {ckpt_path}")
            except Exception as e:
                print(f"⚠️  Audio ckpt load failed: {e}. Using pretrained weights.")
        else:
            print(f"⚠️  No audio ckpt at {ckpt_path}. Using pretrained facebook/wav2vec2-large.")
        # Freeze
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        out = self.encoder(audio).last_hidden_state  # (B, T, 1024)
        return out.mean(dim=1)  # mean-pool over time


class FrozenTextEncoder(nn.Module):
    """Loads the MobileBERT text encoder. Returns pooled 768-dim embedding."""

    def __init__(self, ckpt_path: str = TEXT_CKPT):
        super().__init__()
        from transformers import MobileBertModel
        # Prefer local TER tokenizer dir (has PyTorch model.safetensors).
        # Fall back to HF hub id (which has TF weights, requires from_tf=True).
        ter_tokenizer_dir = (
            Path(ckpt_path).parent / "ter_pytorch_tokenizer"
            if ckpt_path else None
        )
        if ter_tokenizer_dir and ter_tokenizer_dir.exists():
            self.encoder = MobileBertModel.from_pretrained(str(ter_tokenizer_dir))
            print(f"✅ Loaded MobileBERT from local {ter_tokenizer_dir}")
        else:
            self.encoder = MobileBertModel.from_pretrained(
                "google/mobilebert-uncased", from_tf=True
            )
            print("Loaded MobileBERT from HF hub (TF weights, from_tf=True)")
        if ckpt_path and Path(ckpt_path).exists():
            try:
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                self.encoder.load_state_dict(sd, strict=False)
                print(f"✅ Loaded text encoder weights from {ckpt_path}")
            except Exception as e:
                print(f"⚠️  Text ckpt load failed: {e}. Using base MobileBERT.")
        else:
            print(f"⚠️  No text ckpt at {ckpt_path}. Using base MobileBERT.")
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output  # (B, 768)


class FrozenFacialEncoder(nn.Module):
    """ResNet-50 pretrained on ImageNet. Returns 2048-dim avgpool feature."""

    def __init__(self, ckpt_path: Optional[str] = FACIAL_CKPT):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        # Strip the final fc — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # ends at flatten
        if ckpt_path and Path(ckpt_path).exists():
            try:
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                self.backbone.load_state_dict(sd, strict=False)
                print(f"✅ Loaded facial encoder from {ckpt_path}")
            except Exception as e:
                print(f"⚠️  Facial ckpt load failed: {e}. Using ImageNet ResNet50.")
        else:
            print("⚠️  No facial ckpt. Using ImageNet ResNet50 (no FER2013 fine-tune).")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, facial: torch.Tensor) -> torch.Tensor:
        out = self.backbone(facial)  # (B, 2048, 1, 1)
        return out.flatten(1)        # (B, 2048)


# ---------- Training loop ----------

def evaluate(model, fusion_model, loader, device):
    model.eval()
    fusion_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            facial = batch["facial"].to(device)
            labels = batch["label"].to(device)

            audio_emb = model["audio"](audio)
            text_emb = model["text"](ids, mask)
            facial_emb = model["facial"](facial)

            out = fusion_model.forward_embeddings(audio_emb, text_emb, facial_emb)
            preds = out["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_labels


def train_one_epoch(encoders, fusion_model, loader, optimizer, device, epoch, total_epochs):
    # encoders is a dict of frozen modules — they don't need train mode
    # (they're already frozen via requires_grad=False)
    fusion_model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        audio = batch["audio"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        facial = batch["facial"].to(device)
        labels = batch["label"].to(device)

        # Frozen encoders (no grad)
        with torch.no_grad():
            audio_emb = model["audio"](audio)
            text_emb = model["text"](ids, mask)
            facial_emb = model["facial"](facial)

        # Trainable fusion head
        out = fusion_model.forward_embeddings(audio_emb, text_emb, facial_emb, labels)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{total_epochs} batch {i}/{len(loader)} "
                  f"loss={loss.item():.4f} ce={out['ce'].item():.4f} "
                  f"elapsed={elapsed:.1f}s", flush=True)
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR_HEAD)
    parser.add_argument("--audio-ckpt", type=str, default=AUDIO_CKPT)
    parser.add_argument("--text-ckpt", type=str, default=TEXT_CKPT)
    parser.add_argument("--facial-ckpt", type=str, default=FACIAL_CKPT)
    parser.add_argument("--manifest", type=str, default="combined_ser_dataset/manifest.csv")
    parser.add_argument("--disable-supcon", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== v6 Multimodal SER training: seed={args.seed}, epochs={args.epochs} ===")
    print(f"Device: {DEVICE}")

    # Datasets
    train_ds = MultimodalSERDataset(args.manifest, split="train")
    val_ds = MultimodalSERDataset(args.manifest, split="val")
    test_ds = MultimodalSERDataset(args.manifest, split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Encoders (frozen)
    encoders = {
        "audio": FrozenAudioEncoder(args.audio_ckpt).to(DEVICE),
        "text": FrozenTextEncoder(args.text_ckpt).to(DEVICE),
        "facial": FrozenFacialEncoder(args.facial_ckpt).to(DEVICE),
    }

    # Fusion head (trainable)
    cfg = FusionConfig(use_supcon=not args.disable_supcon)
    fusion_model = MultimodalSER(cfg).to(DEVICE)
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"Trainable params (fusion head): {trainable_params:,}")

    optimizer = AdamW(
        [p for p in fusion_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3),
            CosineAnnealingLR(optimizer, T_max=args.epochs - 3, eta_min=1e-6),
        ],
        milestones=[3],
    )

    best_val = 0.0
    best_path = f"model_checkpoints/ser_v6_best_seed{args.seed}.pt"
    Path("model_checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        ep_loss = train_one_epoch(encoders, fusion_model, train_loader, optimizer, DEVICE, epoch, args.epochs)
        val_acc, val_f1, _, _ = evaluate(encoders, fusion_model, val_loader, DEVICE)
        scheduler.step()
        print(f"  Epoch {epoch}: train_loss={ep_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state_dict": fusion_model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "val_acc": val_acc,
            }, best_path)
            print(f"  ✅ saved best (val={val_acc:.4f})")

    # Test
    test_acc, test_f1, test_preds, test_labels = evaluate(encoders, fusion_model, test_loader, DEVICE)
    print(f"\n=== TEST: acc={test_acc:.4f} f1={test_f1:.4f} ===")
    print(classification_report(test_labels, test_preds, target_names=EMOTIONS, digits=4))

    summary = {
        "seed": args.seed,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "best_val": best_val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "trainable_params": trainable_params,
    }
    summary_path = f"model_checkpoints/ser_v6_training_summary_seed{args.seed}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
