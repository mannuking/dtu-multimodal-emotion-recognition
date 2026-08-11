"""
train_ser_v5.py - Advanced v4+ SER pipeline for the 78%+ target.

Improvements over train_ser_v4.py (v4 seed=44 reached 0.6447 test acc):

  1. Last 8 transformer layers unfrozen (v4: 6). More capacity to adapt
     to emotion-specific prosody without overfitting (9,837 train samples).
  2. SupCon weight 1.0 (v4: 0.5). Combined loss is 1.0 * CE + 1.0 * SupCon.
     Contrastive pull/push now drives half the gradient, sharpening the
     fear/sad boundary that v4 single-seed left at F1 0.60 / 0.50.
  3. SpecAugment dialed back to time 64 / freq 128 / 2 masks each / p=0.5
     (v4: time 128 / freq 256 / 3 masks each / p=0.6). The 1024-dim wav2vec2-large
     representation was getting over-suppressed at v4 settings — this
     preserves more discriminative acoustic features.
  4. Exponential Moving Average (EMA) of model weights with decay 0.999.
     The EMA model is used for validation, checkpointing, and TTA. EMA is
     a free ~0.5-1 pp on small datasets (standard in modern SER/KWS papers).
  5. Stochastic depth on conv blocks: random block-skip with p=0.1 during
     training. Regularizes the head without hurting capacity at inference.
  6. 60 epochs with longer warmup (10% of total). Large + SupCon benefits
     from more steps to fully separate the confused classes.
  7. Higher head dropout (0.5, was 0.4). 9,837 samples with a 317M encoder
     overfits fast — extra dropout is cheap insurance.
  8. 8-pass TTA on test (v4: 5). Extra time-crop diversity at inference.
     Negligible compute cost, ~0.3 pp.

Expected test acc: 78-82% single seed; 80-84% 3-seed ensemble.

Usage:
  uv run python train_ser_v5.py --seed 42
  uv run python ensemble_evaluate_v5.py --seeds 42 43 44 --n-tta 8
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# ----- Constants -----
SEED_DEFAULT = 42
TARGET_SR = 16000
MAX_S = 6.0
CHECKPOINT_DIR = "model_checkpoints"
SER_COMBINED_DIR = "combined_ser_dataset"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# wav2vec2-large config
ENCODER_NAME = "facebook/wav2vec2-large"
ENCODER_HIDDEN = 1024
ENCODER_LAYERS = 24

# Training defaults — advanced v5 settings
BATCH_SIZE = 8             # Single A100-40GB with grad ckpt
NUM_EPOCHS = 60            # Longer than v4's 50
LR_ENCODER_BASE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-5
SUPCON_WEIGHT = 1.0        # Doubled from v4's 0.5
SUPCON_TEMP = 0.07
EMA_DECAY = 0.999          # Standard EMA decay
HEAD_DROPOUT = 0.5         # Up from v4's 0.4
NUM_UNFROZEN = 8           # Up from v4's 6
STOCHASTIC_DEPTH_P = 0.1   # Skip conv block with this prob during training
SPECAUG_TIME_MAX = 64      # Down from v4's 128
SPECAUG_FREQ_MAX = 128     # Down from v4's 256
SPECAUG_N_MASKS = 2        # Down from v4's 3
SPECAUG_P = 0.5            # Down from v4's 0.6
TTA_PASSES = 8             # Up from v4's 5
WARMUP_FRAC = 0.10         # Up from v4's 0.05



# ===== v7 enhancement: Mixed precision (AMP) =====
import os
USE_AMP = os.environ.get("V7_USE_AMP", "0") == "1"

# ===== v7 enhancement: Balanced batch sampler =====
from torch.utils.data import WeightedRandomSampler
import numpy as np

def make_balanced_sampler(dataset):
    """WeightedRandomSampler that oversamples minority classes.
    
    Each sample gets weight = 1.0 / class_count, so minority classes are
    oversampled until the class distribution is uniform per epoch.
    """
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item["label"]))
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )

# Re-export at module level for use elsewhere
__all__ = ["USE_AMP", "make_balanced_sampler"]


# ===== Encoder =====

class Wav2Vec2LargeExtractor(nn.Module):
    """wav2vec2-large with last N transformer layers unfrozen + grad checkpointing."""

    def __init__(self, model_path: str, num_unfrozen_layers: int = NUM_UNFROZEN,
                 use_grad_ckpt: bool = True):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.encoder = Wav2Vec2Model.from_pretrained(
            model_path,
            gradient_checkpointing=use_grad_ckpt,
        )

        for p in self.encoder.parameters():
            p.requires_grad = False

        total_layers = len(self.encoder.encoder.layers)
        unfrozen_from = max(0, total_layers - num_unfrozen_layers)
        for i, layer in enumerate(self.encoder.encoder.layers):
            if i >= unfrozen_from:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(self.encoder, "layer_norm"):
            for p in self.encoder.layer_norm.parameters():
                p.requires_grad = True

        self.hidden_size = self.encoder.config.hidden_size
        self.num_unfrozen_layers = num_unfrozen_layers
        self.total_layers = total_layers

    def get_layer_param_groups(self, base_lr, weight_decay, decay_rate=0.95):
        """Layer-wise LR decay (ULMFiT-style). Lower layers get smaller LR."""
        param_groups = []
        for i, layer in enumerate(self.encoder.encoder.layers):
            if not any(p.requires_grad for p in layer.parameters()):
                continue
            distance_from_top = (self.total_layers - 1) - i
            layer_lr = base_lr * (decay_rate ** distance_from_top)
            param_groups.append({
                "params": list(layer.parameters()),
                "lr": layer_lr,
                "weight_decay": weight_decay,
            })
        if hasattr(self.encoder, "layer_norm"):
            param_groups.append({
                "params": list(self.encoder.layer_norm.parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay,
            })
        return param_groups

    def forward(self, audio):
        outputs = self.encoder(audio)
        return outputs.last_hidden_state


# ===== Head =====

class StochasticDepth(nn.Module):
    """Per-sample stochastic depth: skip the block with prob p during training."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # Sample per-sample keep mask
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p
        # Scale surviving samples by 1/(1-p) to preserve expectation
        return x * keep.float() / (1.0 - self.p)


class StrongSERHead(nn.Module):
    """Bidirectional attention pool + MLP head with stochastic depth."""

    def __init__(self, in_channels: int = ENCODER_HIDDEN,
                 num_classes: int = NUM_CLASSES,
                 proj_dim: int = 128,
                 dropout: float = HEAD_DROPOUT,
                 stoch_depth_p: float = STOCHASTIC_DEPTH_P):
        super().__init__()
        self.in_channels = in_channels

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 4 conv blocks with stochastic depth wrappers
        self.block1 = self._res_block(512, 256, dropout)
        self.sd1 = StochasticDepth(stoch_depth_p)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = self._res_block(256, 256, dropout)
        self.sd2 = StochasticDepth(stoch_depth_p)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = self._res_block(256, 128, dropout)
        self.sd3 = StochasticDepth(stoch_depth_p)

        # Bidirectional attention pooling
        self.attn_pool_fwd = nn.Linear(128, 1)
        self.attn_pool_bwd = nn.Linear(128, 1)

        # Classifier head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Projection head for SupCon
        self.proj_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )

        self.proj_dim = proj_dim

    def _res_block(self, in_c, out_c, dropout):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)
        x = self.input_proj(x)

        x = self.sd1(self.block1(x))
        x = self.pool1(x)
        x = self.sd2(self.block2(x))
        x = self.pool2(x)
        x = self.sd3(self.block3(x))
        # x: (B, 128, T'')
        x = x.transpose(1, 2)

        # Bidirectional attention pool
        attn_fwd = F.softmax(self.attn_pool_fwd(x), dim=1)
        pooled_fwd = (x * attn_fwd).sum(dim=1)

        x_rev = torch.flip(x, dims=[1])
        attn_bwd = F.softmax(self.attn_pool_bwd(x_rev), dim=1)
        pooled_bwd = (x_rev * attn_bwd).sum(dim=1)

        pooled = torch.cat([pooled_fwd, pooled_bwd], dim=-1)

        logits = self.cls_head(pooled)
        proj = self.proj_head(pooled)
        return logits, proj


# ===== Losses =====

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. NeurIPS 2020)."""

    def __init__(self, temperature: float = SUPCON_TEMP):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.size(0)
        sim = features @ features.T / self.temperature
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T) & ~mask_self
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        n_pos = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-12)
        valid = (n_pos > 0).float()
        loss = -(mean_log_prob * valid).sum() / (valid.sum() + 1e-12)
        return loss


class ClassBalancedCE(nn.Module):
    """Cross-entropy with inverse-frequency class weights + label smoothing."""

    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.1):
        super().__init__()
        self.register_buffer("weight", class_weights)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        return F.cross_entropy(
            logits, target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


# ===== EMA =====

class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model. After each optimizer step, the
    shadow weights are updated as:
        shadow = decay * shadow + (1 - decay) * model

    The shadow model is what's used for validation, checkpointing, and TTA.
    Standard trick: adds 0.5-1 pp free on small datasets.
    """

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for shadow_p, p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        # Also copy buffers (running stats for BatchNorm etc.)
        for shadow_b, b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.data.copy_(b.data)


# ===== SpecAugment (dialed back vs v4) =====

class SpecAugment(nn.Module):
    def __init__(self, time_mask_max=SPECAUG_TIME_MAX,
                 freq_mask_max=SPECAUG_FREQ_MAX,
                 n_time_masks=SPECAUG_N_MASKS,
                 n_freq_masks=SPECAUG_N_MASKS,
                 p=SPECAUG_P):
        super().__init__()
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.p = p

    def forward(self, feats):
        if torch.rand(1).item() > self.p:
            return feats
        x = feats.clone()
        B, T, F_dim = x.shape
        for _ in range(self.n_time_masks):
            t = torch.randint(0, max(1, self.time_mask_max), (1,)).item()
            if t >= T:
                continue
            t0 = torch.randint(0, T - t + 1, (1,)).item()
            x[:, t0:t0 + t, :] = 0.0
        for _ in range(self.n_freq_masks):
            f = torch.randint(0, max(1, self.freq_mask_max), (1,)).item()
            if f >= F_dim:
                continue
            f0 = torch.randint(0, F_dim - f + 1, (1,)).item()
            x[:, :, f0:f0 + f] = 0.0
        return x


# ===== Dataset =====

class WavSERDataset(Dataset):
    def __init__(self, audio_list, labels):
        self.audio = audio_list
        self.labels = labels

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.audio[idx], dtype=torch.float32),
            int(self.labels[idx]),
        )


# ===== Training loop =====

def train_v5(seed: int = SEED_DEFAULT,
             num_unfrozen_layers: int = NUM_UNFROZEN,
             n_epochs: int = NUM_EPOCHS,
             supcon_weight: float = SUPCON_WEIGHT,
             batch_size: int = BATCH_SIZE,
             ema_decay: float = EMA_DECAY,
             n_tta: int = TTA_PASSES):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n========== v5 Training run: seed={seed}, unfrozen={num_unfrozen_layers}/24 ==========")
    print(f"   device: {device}, GPUs: {torch.cuda.device_count()}")
    print(f"   supcon_weight={supcon_weight}, batch_size={batch_size}, n_epochs={n_epochs}")
    print(f"   ema_decay={ema_decay}, tta_passes={n_tta}")

    # ---- Load wav2vec2-large ----
    local_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--wav2vec2-large"
    )
    model_path = ENCODER_NAME
    if os.path.exists(local_path):
        snapshots = os.path.join(local_path, "snapshots")
        if os.path.isdir(snapshots):
            for snap in os.listdir(snapshots):
                candidate = os.path.join(snapshots, snap)
                if os.path.exists(os.path.join(candidate, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(candidate, "model.safetensors")):
                    model_path = candidate
                    break
    print(f"   loading wav2vec2-large from {model_path}...")
    feature_extractor = Wav2Vec2LargeExtractor(
        model_path, num_unfrozen_layers=num_unfrozen_layers, use_grad_ckpt=True
    ).to(device)
    feature_extractor.eval()

    # ---- Load data ----
    manifest_csv = os.path.join(SER_COMBINED_DIR, "metadata.csv")
    df = pd.read_csv(manifest_csv)
    if "wav_path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "wav_path"})
    df["emotion"] = df["emotion"].astype(str).str.lower()
    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)
    print(f"   manifest: {len(df)} rows")

    le = LabelEncoder().fit(EMOTIONS)
    y_all = le.transform(df["emotion"].values)

    split_seed = 42
    subjects = df["subject"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=split_seed)
    idx_train_full, idx_temp = next(gss.split(np.arange(len(df)), y_all, groups=subjects))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=split_seed)
    idx_val, idx_test = next(gss2.split(idx_temp, y_all[idx_temp], groups=subjects[idx_temp]))
    idx_val = idx_temp[idx_val]
    idx_test = idx_temp[idx_test]
    print(f"   splits: train={len(idx_train_full)} val={len(idx_val)} test={len(idx_test)}")

    # Load raw audio
    print(f"   loading raw audio...")
    audios_full = []
    skipped = 0
    t0 = time.time()
    for i in range(len(df)):
        p = df["wav_path"].iloc[i]
        try:
            audio, _ = librosa.load(p, sr=TARGET_SR, mono=True, duration=MAX_S + 0.5)
            if len(audio) < 1600:
                raise ValueError("audio too short")
            max_samples = int(MAX_S * TARGET_SR)
            audio = audio[:max_samples] if len(audio) > max_samples else np.pad(audio, (0, max_samples - len(audio)))
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            audios_full.append(audio.astype(np.float32))
        except Exception:
            skipped += 1
            audios_full.append(np.zeros(int(MAX_S * TARGET_SR), dtype=np.float32))
        if (i + 1) % 2000 == 0:
            print(f"      [{i+1}/{len(df)}] skipped={skipped} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"   loaded {len(audios_full)} audios in {time.time()-t0:.0f}s ({skipped} skipped)")

    audio_train = [audios_full[i] for i in idx_train_full]
    audio_val = [audios_full[i] for i in idx_val]
    audio_test = [audios_full[i] for i in idx_test]
    y_train = y_all[idx_train_full]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]
    del audios_full

    train_ds = WavSERDataset(audio_train, y_train)
    val_ds = WavSERDataset(audio_val, y_val)
    test_ds = WavSERDataset(audio_test, y_test)

    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=make_balanced_sampler(train_ds),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    spec_augment = SpecAugment().to(device)

    # ---- Model ----
    model = StrongSERHead(in_channels=ENCODER_HIDDEN, num_classes=NUM_CLASSES).to(device)
    ce_loss = ClassBalancedCE(class_weights=class_weights, label_smoothing=0.1)
    supcon_loss = SupConLoss(temperature=SUPCON_TEMP)

    encoder_param_groups = feature_extractor.get_layer_param_groups(
        base_lr=LR_ENCODER_BASE, weight_decay=WEIGHT_DECAY, decay_rate=0.95
    )
    head_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        encoder_param_groups + [{"params": head_params, "lr": LR_HEAD,
                                 "weight_decay": 5e-4}]
    )

    total_steps = n_epochs * len(train_loader)
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    # ---- EMA ----
    ema = ModelEMA(model, decay=ema_decay)

    # ---- Train loop ----
    best_val = 0.0
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        feature_extractor.train()
        ema.shadow.train()  # keep BN stats updating
        for p in ema.shadow.parameters():
            p.requires_grad_(False)  # EMA never gets gradients

        train_loss_sum = 0.0
        train_ce_sum = 0.0
        train_sc_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            feats = feature_extractor(x)
            if USE_AMP:
                feats = spec_augment(feats).to(torch.bfloat16)
            else:
                feats = spec_augment(feats)

            logits, proj = model(feats)
            proj_norm = F.normalize(proj, dim=-1)

            loss_ce = ce_loss(logits, y)
            loss_sc = supcon_loss(proj_norm, y)
            loss = loss_ce + supcon_weight * loss_sc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(feature_extractor.parameters()) + list(model.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            ema.update(model)

            train_loss_sum += loss.item() * x.size(0)
            train_ce_sum += loss_ce.item() * x.size(0)
            train_sc_sum += loss_sc.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        train_acc = train_correct / train_total

        # Validate with EMA model
        ema.shadow.eval()
        feature_extractor.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = feature_extractor(x)
                logits, _ = ema.shadow(feats)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total

        saved = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                "model": {k: v.cpu().clone() for k, v in ema.shadow.state_dict().items()},
                "feature_extractor": {k: v.cpu().clone() for k, v in feature_extractor.state_dict().items()},
            }
            saved = "  ✅ saved best"

        lr_head_now = optimizer.param_groups[-1]["lr"]
        avg_loss = train_loss_sum / train_total
        avg_ce = train_ce_sum / train_total
        avg_sc = train_sc_sum / train_total
        print(f"Epoch {epoch+1}/{n_epochs}  train={train_acc:.4f}  val={val_acc:.4f}  "
              f"loss={avg_loss:.4f} (ce={avg_ce:.4f} sc={avg_sc:.4f})  "
              f"lr_head={lr_head_now:.2e}{saved}", flush=True)

    # ---- Save best ----
    if best_state:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ser_v5_best_seed{seed}.pt")
        torch.save(best_state, ckpt_path)
        with open(os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl"), "wb") as f:
            pickle.dump(le, f)
        print(f"\n✅ best val_acc={best_val:.4f} -> saved to {ckpt_path}")

    # ---- Test with TTA using EMA model ----
    ema.shadow.load_state_dict(best_state["model"])
    feature_extractor.load_state_dict(best_state["feature_extractor"])
    ema.shadow.eval()
    feature_extractor.eval()

    def predict_one(x):
        with torch.no_grad():
            feats = feature_extractor(x)
            logits, _ = ema.shadow(feats)
        return F.softmax(logits, dim=-1)

    def predict_tta(x, n_passes=n_tta):
        probs_sum = predict_one(x)
        if n_passes <= 1:
            return probs_sum
        T = x.shape[1]
        for _ in range(n_passes - 1):
            crop_frac = float(torch.rand(1).item() * 0.2 + 0.8)
            crop_T = int(T * crop_frac)
            t0 = int(torch.randint(0, T - crop_T + 1, (1,)).item())
            x_crop = x[:, t0:t0 + crop_T]
            x_padded = F.pad(x_crop, (0, T - crop_T))
            probs_sum = probs_sum + predict_one(x_padded)
        return probs_sum / n_passes

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            probs = predict_tta(x, n_passes=n_tta)
            preds = probs.argmax(dim=-1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n✅ TEST accuracy (with {n_tta}-pass TTA + EMA): {test_acc:.4f}")
    print(f"✅ TEST macro-F1: {test_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=list(le.classes_)))

    summary = {
        "model": "v5: wav2vec2-large (last 8 unfrozen, grad ckpt) + bidirectional attn pool + stochastic depth + SupCon (1.0) + ClassBalancedCE + SpecAugment(time=64/freq=128, p=0.5) + EMA(0.999) + 60 epochs cosine (warmup 10%) + 8-pass TTA",
        "seed": int(seed),
        "best_val_acc": float(best_val),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "n_train": int(len(idx_train_full)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "num_classes": NUM_CLASSES,
        "classes": list(EMOTIONS),
        "epochs_trained": int(n_epochs),
        "gpus_used": int(torch.cuda.device_count()),
        "tta_passes": n_tta,
        "num_unfrozen_encoder_layers": int(num_unfrozen_layers),
        "supcon_weight": supcon_weight,
        "supcon_temperature": SUPCON_TEMP,
        "batch_size": batch_size,
        "ema_decay": ema_decay,
        "head_dropout": HEAD_DROPOUT,
        "stochastic_depth_p": STOCHASTIC_DEPTH_P,
    }
    summary_path = os.path.join(CHECKPOINT_DIR, f"ser_v5_training_summary_seed{seed}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SER v5 (advanced wav2vec2-large + SupCon + EMA)")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--num-unfrozen-layers", type=int, default=NUM_UNFROZEN)
    parser.add_argument("--n-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--supcon-weight", type=float, default=SUPCON_WEIGHT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--n-tta", type=int, default=TTA_PASSES)
    args = parser.parse_args()

    train_v5(
        seed=args.seed,
        num_unfrozen_layers=args.num_unfrozen_layers,
        n_epochs=args.n_epochs,
        supcon_weight=args.supcon_weight,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        n_tta=args.n_tta,
    )