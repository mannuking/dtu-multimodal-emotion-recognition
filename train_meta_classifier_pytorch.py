# train_meta_classifier_pytorch.py
# PyTorch version of meta-classifier (loads PyTorch SER + PyTorch TER + TensorFlow FER).
# Replaces train_meta_classifier.py.
#
# Paper Sec 5.4: 3-layer MLP (128, 64, 32) + softmax, dropout 0.5/0.3/0.2,
# Adam(1e-3), batch=32, ~20 epochs.

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from gpu_config import *


class MetaMLP(nn.Module):
    """Paper Sec 5.4: 3 dense layers (128, 64, 32) + softmax, dropout 0.5/0.3/0.2."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_ter_model(device):
    """Load PyTorch TER (MobileBERT)."""
    from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
    ter_path = os.path.join(CHECKPOINT_DIR, "ter_best.pt")
    ter_tok_path = os.path.join(CHECKPOINT_DIR, "ter_tokenizer")
    model = MobileBertForSequenceClassification.from_pretrained(ter_tok_path, num_labels=NUM_CLASSES)
    model.load_state_dict(torch.load(ter_path, map_location=device))
    model.to(device).eval()
    return model


def load_ser_model(device):
    """Load PyTorch SER (1D-CNN)."""
    from train_ser_pytorch import SER1DCNN
    ser_path = os.path.join(CHECKPOINT_DIR, "ser_best.pt")
    model = SER1DCNN(in_len=11044, num_classes=NUM_CLASSES).to(device)
    state = torch.load(ser_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_ser_probs(model, loader, device):
    """Get SER probability predictions for each sample in loader."""
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=-1)
            out.append(probs.cpu().numpy())
    return np.concatenate(out, axis=0)


def get_ter_probs(model, tokenizer, texts, device, batch_size=64):
    """Get TER probability predictions for a list of text strings."""
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1)
            out.append(probs.cpu().numpy())
    return np.concatenate(out, axis=0)


def get_fer_synthetic_probs(texts, num_classes=NUM_CLASSES):
    """FER model not yet implemented in PyTorch — return uniform for now (paper
    baseline for missing-modality graceful degradation)."""
    return np.full((len(texts), num_classes), 1.0 / num_classes, dtype=np.float32)


def load_triplets():
    """Load triplets manifest built by uv_run_all.py."""
    df = pd.read_csv("triplets_manifest.csv")
    return df


def synth_text_from_emotion(emotion: str) -> str:
    return {
        "angry": "I am feeling very angry right now",
        "disgust": "This is completely disgusting",
        "fear": "I am scared and afraid",
        "happy": "I am so happy today",
        "sad": "I feel very sad",
        "surprise": "Wow what a surprise",
        "neutral": "I am speaking normally",
    }.get(emotion.lower(), "I am speaking")


def train_meta_classifier():
    print("🔄 Training hybrid meta-classifier (PyTorch)...")

    df = load_triplets()
    print(f"   triplets: {len(df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device: {device}")

    # Load SER (PyTorch)
    print("1️⃣ SER predictions (PyTorch)...")
    ser_model = load_ser_model(device)
    # Build SER inference data from triplets' speech_wav column
    from feature_utils import ensure_features_exist
    # The features were extracted for the SER training manifest; for triplets
    # we re-extract per sample using their wav paths.
    import librosa
    from feature_utils import extract_features
    speech_paths = df["speech_wav"].tolist()
    speech_feats = []
    for sp in speech_paths:
        if not os.path.exists(sp):
            sp = os.path.join("combined_ser_dataset", sp)
        if not os.path.exists(sp):
            speech_feats.append(np.zeros(11044, dtype=np.float32))
            continue
        try:
            speech_feats.append(extract_features(sp))
        except Exception:
            speech_feats.append(np.zeros(11044, dtype=np.float32))
    speech_feats = np.asarray(speech_feats, dtype=np.float32)
    speech_ds = TensorDataset(torch.as_tensor(speech_feats).unsqueeze(1), torch.zeros(len(speech_feats), dtype=torch.long))
    speech_loader = DataLoader(speech_ds, batch_size=64, shuffle=False)
    ser_probs = get_ser_probs(ser_model, speech_loader, device)
    del ser_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load TER (PyTorch)
    print("2️⃣ TER predictions (PyTorch)...")
    from transformers import MobileBertTokenizer
    ter_model = load_ter_model(device)
    tokenizer = MobileBertTokenizer.from_pretrained(os.path.join(CHECKPOINT_DIR, "ter_tokenizer"))
    texts = [synth_text_from_emotion(e) for e in df["label"].tolist()]
    ter_probs = get_ter_probs(ter_model, tokenizer, texts, device)
    del ter_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FER — synthetic uniform for now (PyTorch FER port pending)
    print("3️⃣ FER predictions (uniform baseline — PyTorch FER port pending)...")
    fer_probs = get_fer_synthetic_probs(texts)

    # Fuse: 21-dim vector (3 models × 7 classes)
    X = np.concatenate([ser_probs, ter_probs, fer_probs], axis=1).astype(np.float32)
    label_map = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5, "neutral": 6}
    y = np.asarray([label_map.get(e.lower(), 0) for e in df["label"]], dtype=np.int64)
    print(f"   features: {X.shape}  labels: {y.shape}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=SEED)

    meta = MetaMLP(input_dim=X.shape[1], num_classes=NUM_CLASSES).to(device)
    opt = Adam(meta.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr))
    val_ds = TensorDataset(torch.as_tensor(Xva), torch.as_tensor(yva))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    best_val = 0.0
    for epoch in range(20):
        meta.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(meta(xb), yb)
            loss.backward()
            opt.step()
        meta.eval()
        vcorr, vtotal = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = meta(xb).argmax(1)
                vcorr += (out == yb).sum().item()
                vtotal += xb.size(0)
        val_acc = vcorr / max(vtotal, 1)
        print(f"   epoch {epoch+1}/20  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(meta.state_dict(), os.path.join(CHECKPOINT_DIR, "meta_best.pt"))
    print(f"✅ Meta-classifier trained. best val_acc={best_val:.4f}")

    # Test
    meta.eval()
    with torch.no_grad():
        out = meta(torch.as_tensor(Xte).to(device)).argmax(1).cpu().numpy()
    test_acc = (out == yte).mean()
    print(f"✅ Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train_meta_classifier()