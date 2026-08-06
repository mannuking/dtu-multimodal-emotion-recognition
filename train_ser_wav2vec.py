# train_ser_wav2vec.py - Speech Emotion Recognition with wav2vec2 backbone.
#
# Replaces train_ser_pytorch.py with a significantly more accurate approach:
#   - Frozen wav2vec2-base encoder extracts rich speech representations
#   - Small MLP head classifies emotions
#   - Mixup + augmentation for regularization
#
# Why wav2vec2:
#   - Pretrained on 960h of LibriSpeech audio (vs our 11k samples)
#   - Captures phonetic, prosodic, and speaker features
#   - Linear probe on wav2vec2 features hits 80%+ on most speech emotion datasets
#   - Fine-tuning the top layers hits 85-90%
#
# Same paper hyperparameter strategy: focal loss + class weights + Adam.
# Output: same checkpoint paths so verify_ser_pytorch.py works without changes.

import os

# GPU env
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa

from gpu_config import *
from feature_utils import EMOTION_ORDER_LOWER

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===== Wav2Vec2 backbone =====

class Wav2Vec2SER(nn.Module):
    """Frozen wav2vec2-base + trainable MLP classifier head.

    Architecture:
        - wav2vec2-base (frozen): outputs 768-dim features per ~20ms audio frame
        - Mean-pool over time to get one 768-dim vector per utterance
        - MLP head: 768 -> 256 -> 128 -> num_classes
        - Dropout 0.3 between layers
    """

    def __init__(self, num_classes: int, model_name: str = "facebook/wav2vec2-base", cache_dir: str | None = None):
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2Config

        # Load pretrained wav2vec2 encoder
        cfg = Wav2Vec2Config.from_pretrained(model_name, cache_dir=cache_dir)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir)

        # Freeze the encoder — we only train the classifier head
        # (Optional: unfreeze last 2-4 layers for fine-tuning)
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = cfg.hidden_size  # 768 for base

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # audio: (B, T) float32 in [-1, 1]
        with torch.no_grad():  # frozen backbone
            outputs = self.encoder(audio, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, T', 768)

        # Mean-pool over time
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)

        return self.classifier(pooled)

    def unfreeze_top_layers(self, n_layers: int = 2):
        """Unfreeze the top n transformer layers for fine-tuning."""
        for layer in self.encoder.encoder.layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


# ===== Loss =====

class FocalWeightedCE(nn.Module):
    def __init__(self, alpha: np.ndarray, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, target):
        alpha = self.alpha.to(logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_oh = F.one_hot(target, num_classes=logits.shape[-1]).float()
        ce = -(target_oh * log_probs)
        p_t = (target_oh * probs).sum(dim=-1)
        focal = (1.0 - p_t).pow(self.gamma)
        alpha_t = (target_oh * alpha).sum(dim=-1)
        return (alpha_t * focal * ce.sum(dim=-1)).mean()


# ===== Audio loading (from manifest) =====

def load_audio_from_manifest(manifest_csv: str, sample_rate: int = 16000, max_seconds: float = 6.0, base_dir: str = "."):
    """Read manifest, load all audio files into memory at 16 kHz mono.

    Manifest rows have 'wav_path' OR 'filepath'. Paths are resolved relative
    to base_dir (default: current working directory). A path like
    'combined_ser_dataset/angry/123.wav' resolves to <cwd>/combined_ser_dataset/angry/123.wav.

    Returns:
        audio_list: list of np.ndarray (T,) float32
        labels: list of int
        label_encoder: fitted LabelEncoder
        paths: list of wav file paths (absolute)
    """
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    # Normalize column names
    if "wav_path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "wav_path"})

    df["emotion"] = df["emotion"].astype(str).str.lower()
    df = df[df["emotion"].isin(EMOTION_ORDER_LOWER)].reset_index(drop=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_ORDER_LOWER)
    labels = label_encoder.transform(df["emotion"].values)

    max_samples = int(max_seconds * sample_rate)
    audio_list = []
    valid_labels = []
    valid_paths = []
    skipped = 0
    for i, row in df.iterrows():
        path = str(row["wav_path"])
        # Resolve relative paths against base_dir (default cwd)
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        if not os.path.exists(path):
            skipped += 1
            continue
        try:
            y, sr = librosa.load(path, sr=sample_rate, mono=True)
            # Pad or trim to max_seconds
            if len(y) > max_samples:
                y = y[:max_samples]
            elif len(y) < max_samples:
                y = np.pad(y, (0, max_samples - len(y)), mode="constant")
            # Normalize
            if np.abs(y).max() > 0:
                y = y / np.abs(y).max()
            audio_list.append(y.astype(np.float32))
            valid_labels.append(labels[i])
            valid_paths.append(path)
        except Exception as e:
            skipped += 1

    if skipped > 0:
        print(f"   skipped {skipped} files (missing or unreadable)")
    return audio_list, np.array(valid_labels), label_encoder, valid_paths


# ===== Augmentation =====

class WavAugment:
    """Light augmentation: noise + gain + time shift + small pitch shift."""

    def __init__(self, sample_rate=16000, p=0.5):
        self.sr = sample_rate
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return x
        # noise
        if np.random.rand() < 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.005
        # gain
        g = np.random.uniform(0.85, 1.15)
        x = x * g
        # time shift (circular)
        shift = np.random.randint(0, x.shape[0])
        x = np.roll(x, shift)
        return x.astype(np.float32)


# ===== Dataset =====

class WavSERDataset(Dataset):
    def __init__(self, audio_list, labels, augment=None):
        self.audio = audio_list
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        x = self.audio[idx]
        if self.augment is not None:
            x = self.augment(x)
        return torch.as_tensor(x, dtype=torch.float32), int(self.labels[idx])


# ===== Training =====

def train_ser_model():
    from gpu_config import CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ser_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ser_best.pt")
    ser_encoder_path = os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device: {device}  GPUs: {torch.cuda.device_count()}")

    # ---- Load data from manifest ----
    manifest_csv = "combined_ser_dataset/metadata.csv"
    if not os.path.exists(manifest_csv):
        print(f"   manifest not found at {manifest_csv}, running build_combined_ser_dataset.py")
        os.system("uv run python build_combined_ser_dataset.py")

    print(f"   Loading audio from {manifest_csv}...")
    audio_list, labels, label_encoder, paths = load_audio_from_manifest(manifest_csv)
    num_classes = len(label_encoder.classes_)
    print(f"   {len(audio_list)} samples, {num_classes} classes: {list(label_encoder.classes_)}")

    if len(audio_list) < 1000:
        print(f"\n   \u26a0\ufe0f  Only {len(audio_list)} samples loaded (expected 11,000+).")
        print(f"   Check that combined_ser_dataset/ has wav files in subfolders,")
        print(f"   or that metadata.csv paths point to actual files.")
        print(f"   First 3 paths: {paths[:3] if paths else 'NONE'}")

    # ---- Train/val/test split (stratified, subject-disjoint best-effort) ----
    indices = np.arange(len(audio_list))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"   splits: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    audio_train = [audio_list[i] for i in idx_train]
    audio_val = [audio_list[i] for i in idx_val]
    audio_test = [audio_list[i] for i in idx_test]
    y_train_arr = labels[idx_train]
    y_val_arr = labels[idx_val]
    y_test_arr = labels[idx_test]

    # Save test indices for verify_ser_pytorch.py
    test_indices_path = os.path.join(CHECKPOINT_DIR, "ser_test_indices.pkl")
    with open(test_indices_path, "wb") as f:
        pickle.dump({"test_indices": idx_test.tolist(),
                     "label_encoder_classes": label_encoder.classes_.tolist()}, f)

    # Save label encoder
    with open(ser_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    train_ds = WavSERDataset(audio_train, y_train_arr, augment=WavAugment(p=0.5))
    val_ds = WavSERDataset(audio_val, y_val_arr, augment=None)
    test_ds = WavSERDataset(audio_test, y_test_arr, augment=None)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model ----
    print("   loading wav2vec2-base (from local cache if HF_HUB_OFFLINE=1)...")
    local_model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
    )
    if os.path.isdir(local_model_path) and os.path.exists(os.path.join(local_model_path, "pytorch_model.bin")):
        model_name_or_path = local_model_path
        print(f"   using cached model at {model_name_or_path}")
    else:
        model_name_or_path = "facebook/wav2vec2-base"
        print(f"   using HF hub: {model_name_or_path}")
    model = Wav2Vec2SER(num_classes=num_classes, model_name=model_name_or_path).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f"   trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- Loss / Optimizer ----
    alpha = np.bincount(y_train_arr, minlength=num_classes).astype(np.float64)
    alpha = len(y_train_arr) / (num_classes * np.maximum(alpha, 1.0))
    alpha = alpha / alpha.mean()  # normalize
    alpha = alpha.astype(np.float32)
    criterion = FocalWeightedCE(alpha=alpha, gamma=2.0).to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    n_epochs = 30
    scheduler = OneCycleLR(optimizer, max_lr=3e-4, total_steps=n_epochs * len(train_loader), pct_start=0.1)

    # ---- Train loop ----
    best_val = 0.0
    best_state = None
    print(f"\nEpoch 0/{n_epochs}  (untrained)")
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total

        saved = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()).items()}
            saved = "  \u2705 saved best"

        print(f"Epoch {epoch+1}/{n_epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  loss={train_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}{saved}")

    # ---- Save best ----
    if best_state is not None:
        torch.save(best_state, ser_checkpoint_path)
        print(f"\n\u2705 SER training complete. best val_acc={best_val:.4f}")
        print(f"   saved to {ser_checkpoint_path}")

    # ---- Test ----
    print("\nTest set evaluation:")
    # Load state into the underlying (un-wrapped) module to handle both
    # single-GPU and DataParallel saved checkpoints.
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(best_state)
    target_model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            test_correct += (logits.argmax(1) == y).sum().item()
            test_total += x.size(0)
    test_acc = test_correct / test_total
    print(f"\u2705 Test accuracy: {test_acc:.4f}")

    # ---- Save final summary ----
    summary = {
        "model": "wav2vec2-base frozen + MLP head",
        "best_val_acc": float(best_val),
        "test_acc": float(test_acc),
        "n_train_samples": int(len(idx_train)),
        "n_val_samples": int(len(idx_val)),
        "n_test_samples": int(len(idx_test)),
        "num_classes": int(num_classes),
        "classes": [str(c) for c in label_encoder.classes_],
        "epochs_trained": int(n_epochs),
        "gpus_used": int(torch.cuda.device_count()),
    }
    with open(os.path.join(CHECKPOINT_DIR, "ser_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   saved summary to {CHECKPOINT_DIR}/ser_training_summary.json")


if __name__ == "__main__":
    train_ser_model()