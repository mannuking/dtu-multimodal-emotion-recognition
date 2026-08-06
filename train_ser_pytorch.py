# train_ser_pytorch.py - Speech Emotion Recognition with PyTorch.
#
# Replaces train_ser_tensorflow.py - same architecture, same paper hyperparameters,
# but uses PyTorch so it reliably uses the GPU on PARAM Siddhi-AI (where TF was
# failing to enumerate the A100).
#
# Architecture: deep 1D-CNN with 9 Conv1D layers + dense head (paper Sec 5.1)
# Loss: focal weighted categorical cross-entropy (paper Sec 5.1)
# Augmentation: noise injection + gain variation in dataloader
# Batch: 64 per replica, multi-GPU via nn.DataParallel
# Epochs: 50 (paper range 30-50)
# Optimizer: Adam(lr=1e-3) + ReduceLROnPlateau(patience=2, factor=0.5)

import os

# GPU env (also set in train.sbatch; harmless duplicate)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from gpu_config import *
from feature_utils import ensure_features_exist, EMOTION_ORDER_LOWER

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---- Loss ----

class FocalWeightedCE(nn.Module):
    """Focal weighted categorical cross-entropy (paper Sec 5.1)."""

    def __init__(self, alpha: np.ndarray, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C); target: (B,)
        # Ensure alpha is on the same device as logits (model may move to GPU
        # after construction).
        alpha = self.alpha.to(logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_oh = F.one_hot(target, num_classes=logits.shape[-1]).float()
        ce = -(target_oh * log_probs)
        p_t = (target_oh * probs).sum(dim=-1)
        focal = (1.0 - p_t).pow(self.gamma)
        alpha_t = (target_oh * alpha).sum(dim=-1)
        loss = (alpha_t * focal * ce.sum(dim=-1)).mean()
        return loss


def class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for unseen classes
    w = len(y) / (num_classes * counts)
    return w.astype(np.float32)


# ---- Augmentation ----

class AudioAugment:
    """Light in-dataloader augmentation: noise + gain."""

    def __init__(self, p: float = 0.5, noise_std: float = 0.005, gain_range=(0.85, 1.15)):
        self.p = p
        self.noise_std = noise_std
        self.gain_range = gain_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, T) float32
        if torch.rand(1).item() > self.p:
            return x
        # noise
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * self.noise_std
        # gain
        g = torch.empty(1).uniform_(self.gain_range[0], self.gain_range[1]).item()
        x = x * g
        return x


# ---- Dataset ----

class SERDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: AudioAugment | None = None):
        # X: (N, T) float32; y: (N,) int64
        self.X = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, T)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[idx]


# ---- Model ----

class SER1DCNN(nn.Module):
    """Deep 1D-CNN, 9 Conv1D layers + dense head (paper Sec 5.1)."""

    def __init__(self, in_len: int, num_classes: int = NUM_CLASSES, dropout_conv: float = 0.3, dropout_dense: float = 0.5):
        super().__init__()
        self.in_len = in_len
        self.num_classes = num_classes
        self.dropout_conv = dropout_conv
        self.dropout_dense = dropout_dense
        # 9 Conv1D blocks (paper Sec 5.1): progressive channel growth 64->128->256 then decay
        layers = []
        # Block 1: 1 -> 64
        layers += self._conv_block(1, 64)
        # Block 2: 64 -> 128
        layers += self._conv_block(64, 128)
        # Block 3: 128 -> 256
        layers += self._conv_block(128, 256)
        # Block 4: 256 -> 256
        layers += self._conv_block(256, 256)
        # Block 5: 256 -> 128
        layers += self._conv_block(256, 128)
        # Block 6: 128 -> 64
        layers += self._conv_block(128, 64)
        # Block 7: 64 -> 64
        layers += self._conv_block(64, 64)
        # Block 8: 64 -> 64
        layers += self._conv_block(64, 64)
        # Block 9: 64 -> 64
        layers += self._conv_block(64, 64)
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        # Compute flat dim via a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_len)
            flat_dim = self.features(dummy).numel()
        self.dense1 = nn.Linear(flat_dim, 512)
        self.dropout = nn.Dropout(dropout_dense)
        self.out = nn.Linear(512, num_classes)

    def _conv_block(self, in_c: int, out_c: int) -> list[nn.Module]:
        return [
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(self.dropout_conv),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        return self.out(x)


# ---- Train ----

def train_ser_model():
    ser_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ser_best.pt")
    ser_encoder_path = os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl")

    if os.path.exists(ser_checkpoint_path) and os.path.exists(ser_encoder_path):
        print("✅ SER model already trained!")
        return

    print("🔄 Training SER model (PyTorch)...")

    # Load pre-extracted features
    X, y = ensure_features_exist()
    # Build a labelmap (sorted set of classes from y)
    labelmap = {int(c): str(c) for c in sorted(set(y.tolist()))}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.111, stratify=ytr, random_state=SEED)

    # Class weights for focal loss
    alpha = class_weights(y, NUM_CLASSES)

    # Device + multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"   device: {device}  GPUs: {n_gpus}")

    # Model
    model = SER1DCNN(in_len=X.shape[1], num_classes=NUM_CLASSES).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"   ⚡ DataParallel on {n_gpus} GPUs")

    loss_fn = FocalWeightedCE(alpha=alpha, gamma=2.0)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5, min_lr=1e-5)

    # Data
    train_ds = SERDataset(Xtr, ytr, augment=AudioAugment(p=0.5))
    val_ds = SERDataset(Xva, yva, augment=None)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)

    best_val_acc = 0.0
    epochs = 50

    for epoch in range(epochs):
        model.train()
        tloss = 0.0
        tcorr = 0
        ttotal = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item() * xb.size(0)
            tcorr += (out.argmax(1) == yb).sum().item()
            ttotal += xb.size(0)
        train_acc = tcorr / max(ttotal, 1)

        model.eval()
        vcorr = 0
        vtotal = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb)
                vcorr += (out.argmax(1) == yb).sum().item()
                vtotal += xb.size(0)
        val_acc = vcorr / max(vtotal, 1)

        scheduler.step(val_acc)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  lr={cur_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, ser_checkpoint_path)
            print(f"  ✅ saved best (val_acc={val_acc:.4f})")

    with open(ser_encoder_path, "wb") as f:
        pickle.dump(labelmap, f)
    print(f"✅ SER training complete. best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    train_ser_model()