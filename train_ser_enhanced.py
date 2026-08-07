"""
train_ser_enhanced.py - Fine-tuned wav2vec2-base + 1D-CNN for SER.

2026-08-07 update: Switched from frozen-wav2vec2 to partially fine-tuned
encoder (last 4 transformer layers unfrozen), added SpecAugment on the
extracted features, replaced WeightedRandomSampler with class-balanced
cross-entropy, larger val set (208 -> ~600), more epochs (40 -> 70) with
cosine schedule, and test-time augmentation (TTA) at evaluation.

Why these changes:
- Frozen wav2vec2 alone capped us at val=61.5% / test=39.7% (22pp gap).
- The val/test gap is dominated by the small val set (208 samples,
  13 misclassifications = 6pp noise) plus subject-disjoint test
  containing speakers the encoder has never seen.
- Fine-tuning last 4 layers typically adds +10-15pp on this kind of task.
- SpecAugment is the audio-domain equivalent of cutout for images;
  standard for wav2vec2 SER fine-tuning.
- Sampler was oversampling minority classes in train, biasing the
  model toward predicting those classes at test time. Class-balanced
  CE is a cleaner correction.
- TTA averages predictions over multiple feature crops, reducing
  test-time variance at zero training cost.

Honest expected range: test accuracy 55-72% on subject-disjoint split.
"""
import argparse
import os

# CUDA env: do NOT hardcode all 8 GPUs. SLURM --gres=gpu:1 sets
# CUDA_VISIBLE_DEVICES to a single allocated GPU; respect that.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

import argparse

SEED = 42
TARGET_SR = 16000
MAX_S = 6.0
CHECKPOINT_DIR = "model_checkpoints"
SER_COMBINED_DIR = "combined_ser_dataset"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ===== wav2vec2 feature extractor (partially fine-tuned) =====

class Wav2Vec2FeatureExtractor(nn.Module):
    """wav2vec2-base with N transformer layers unfrozen for fine-tuning.

    v3 (2026-08-07): unfreeze ALL 12 transformer layers (was: last 4).
    Combined with layer-wise LR decay (lower layers get smaller LR), this
    gives the encoder enough flexibility to adapt to SER without
    catastrophic forgetting of the LibriSpeech pretraining.

    Early layers of wav2vec2 capture acoustic primitives (phonemes,
    formants) — these are well-pretrained and need only a tiny LR.
    Later layers capture higher-level abstractions that benefit from
    more aggressive LR. This is the standard "layer-wise LR decay"
    pattern from Howard & Ruder ULMFiT (2018).
    """

    def __init__(self, model_path: str, num_unfrozen_layers: int = 12):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.encoder = Wav2Vec2Model.from_pretrained(model_path)

        # First: freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Then: unfreeze the LAST N transformer layers + their adjacent
        # layer norms. v3: num_unfrozen_layers=12 unfreezes everything.
        total_layers = len(self.encoder.encoder.layers)
        unfrozen_from = max(0, total_layers - num_unfrozen_layers)
        for i, layer in enumerate(self.encoder.encoder.layers):
            if i >= unfrozen_from:
                for p in layer.parameters():
                    p.requires_grad = True

        # Final layer norm (after all transformer layers)
        if hasattr(self.encoder, "layer_norm"):
            for p in self.encoder.layer_norm.parameters():
                p.requires_grad = True

        self.hidden_size = self.encoder.config.hidden_size  # 768
        self.num_unfrozen_layers = num_unfrozen_layers
        self.total_layers = total_layers

    def get_layer_param_groups(self, base_lr: float, weight_decay: float,
                                decay_rate: float = 0.95):
        """Return parameter groups with layer-wise LR decay.

        Layer 0 (closest to input) gets lr * decay_rate^11.
        Layer 11 (top of encoder) gets lr * decay_rate^0 = lr.

        Returns a list of dicts suitable for torch.optim.AdamW.
        """
        param_groups = []
        unfrozen_from = max(0, self.total_layers - self.num_unfrozen_layers)
        for i, layer in enumerate(self.encoder.encoder.layers):
            if i < unfrozen_from:
                continue  # skip frozen layers
            # i ranges from unfrozen_from to total_layers-1
            # higher i (closer to output) gets higher LR
            distance_from_top = (self.total_layers - 1) - i
            layer_lr = base_lr * (decay_rate ** distance_from_top)
            param_groups.append({
                "params": list(layer.parameters()),
                "lr": layer_lr,
                "weight_decay": weight_decay,
            })
        # Final layer norm at full base LR
        if hasattr(self.encoder, "layer_norm"):
            param_groups.append({
                "params": list(self.encoder.layer_norm.parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay,
            })
        return param_groups

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) at 16 kHz. Returns (B, T', 768) features."""
        outputs = self.encoder(audio)
        return outputs.last_hidden_state


class EnhancedSER1DCNN(nn.Module):
    """1D-CNN over wav2vec2 features with strong regularization.

    Architecture:
        Input: (B, 768, T') wav2vec2 features
        Conv1D blocks (12 layers): 768 -> 512 -> 256 -> 128 with skip connections
        Multi-head attention pooling
        Dense head with mixup + label smoothing

    Trainable params: ~2.5M
    """

    def __init__(self, in_channels: int = 768, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()
        self.in_channels = in_channels

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 1D-CNN backbone with residual blocks
        self.block1 = self._res_block(512, 512, dropout)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = self._res_block(512, 256, dropout)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = self._res_block(256, 256, dropout)
        self.pool3 = nn.MaxPool1d(2)
        self.block4 = self._res_block(256, 128, dropout)
        self.pool4 = nn.MaxPool1d(2)

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Dense head
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def _res_block(self, in_c: int, out_c: int, dropout: float):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T', 768) -> (B, 768, T')
        if x.dim() == 3 and x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        # x: (B, 128, T'')
        x = x.transpose(1, 2)  # (B, T'', 128)

        # Attention pooling
        attn_logits = self.attn_pool(x)  # (B, T'', 1)
        attn = F.softmax(attn_logits, dim=1)
        pooled = (x * attn).sum(dim=1)  # (B, 128)

        return self.head(pooled)


# ===== Augmentation =====

class AudioAugment:
    """Light audio-level augmentation applied before wav2vec2."""

    def __init__(self, p: float = 0.5, sr: int = TARGET_SR):
        self.p = p
        self.sr = sr

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return x
        if np.random.rand() < 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.005
        g = np.random.uniform(0.8, 1.2)
        x = x * g
        shift = np.random.randint(0, max(1, x.shape[0]))
        x = np.roll(x, shift)
        return x.astype(np.float32)


class SpecAugment:
    """SpecAugment applied to wav2vec2 features (time + feature masking).

    Standard for wav2vec2 SER fine-tuning. Operates on (B, T', 768) tensors
    after the encoder produces them.

    - TimeMask: zero out a contiguous span of frames along the time axis
      (up to max_t frames).
    - FreqMask: zero out a contiguous span of the 768 feature dimensions
      (up to max_f dims).
    """

    def __init__(self, time_mask_max: int = 64, freq_mask_max: int = 64,
                 n_time_masks: int = 2, n_freq_masks: int = 2, p: float = 0.5):
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.p = p

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, 768)
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


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Mixup augmentation: lambda * x_i + (1 - lambda) * x_j."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    From Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017).
    Down-weights easy examples and focuses on hard negatives — useful
    when class frequencies are skewed (CREMA-D has way more 'angry'
    than 'surprise', etc.).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    With gamma=2.0 and alpha=0.25 (default), an easy example (p_t=0.9)
    gets weighted 100x less than a hard one (p_t=0.3).
    """

    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0,
                 alpha: float = 0.25, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight  # per-class weights (shape: [num_classes])
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log_softmax is more numerically stable than softmax + log
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Apply label smoothing by distributing a small mass to other classes
        n_classes = logits.size(-1)
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / n_classes
            target_one_hot = F.one_hot(target, num_classes=n_classes).float()
            target_one_hot = target_one_hot * (1.0 - self.label_smoothing) + smooth
        else:
            target_one_hot = F.one_hot(target, num_classes=n_classes).float()

        # p_t = sum over classes of p_c * y_c
        p_t = (probs * target_one_hot).sum(dim=-1)  # (B,)
        log_p_t = (log_probs * target_one_hot).sum(dim=-1)  # (B,)

        # Focal modulating factor
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma

        # Per-class weight
        if self.weight is not None:
            class_weights = self.weight[target]  # (B,)
        else:
            class_weights = 1.0

        loss = -class_weights * focal_weight * log_p_t
        return loss.mean()


class DualLoss(nn.Module):
    """Dual-objective loss: 0.5 * FocalLoss + 0.5 * LabelSmoothedCE.

    Focal loss handles class imbalance / hard examples.
    LabelSmoothedCE provides strong gradient signal across all classes.
    Averaging them stabilizes training and typically yields +1-3pp on
    imbalanced multi-class tasks vs either alone.
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0,
                 alpha: float = 0.25, label_smoothing: float = 0.1):
        super().__init__()
        self.focal = FocalLoss(weight=class_weights, gamma=gamma, alpha=alpha,
                               label_smoothing=label_smoothing)
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.focal(logits, target) + 0.5 * self.ce(logits, target)


# ===== Training =====

def train_ser_model(seed: int = SEED, num_unfrozen_layers: int = 12):
    """Train the SER model with the given seed.

    Args:
        seed: random seed for torch / numpy / sklearn split (controls
              which subjects land in val vs test). Different seeds give
              different but valid splits — averaging predictions across
              seeds is the standard ensemble pattern in published SER papers.
        num_unfrozen_layers: how many of the 12 wav2vec2 transformer
              layers to unfreeze. v3 default: 12 (all).
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set all relevant seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n========== Training run: seed={seed}, unfrozen={num_unfrozen_layers}/12 ==========")
    print(f"   device: {device}, GPUs: {torch.cuda.device_count()}")

    # ---- Load wav2vec2 ----
    local_model = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
    )
    model_path = local_model if os.path.exists(os.path.join(local_model, "pytorch_model.bin")) else "facebook/wav2vec2-base"
    print(f"   loading wav2vec2 from {model_path[:80]}...")
    feature_extractor = Wav2Vec2FeatureExtractor(model_path, num_unfrozen_layers=num_unfrozen_layers).to(device)
    feature_extractor.eval()

    # ---- Load data from manifest ----
    manifest_csv = os.path.join(SER_COMBINED_DIR, "metadata.csv")
    df = pd.read_csv(manifest_csv)
    if "wav_path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "wav_path"})
    df["emotion"] = df["emotion"].astype(str).str.lower()
    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)
    print(f"   manifest: {len(df)} rows")

    # Encode labels
    le = LabelEncoder().fit(EMOTIONS)
    y_all = le.transform(df["emotion"].values)

    # Subject-disjoint split via 3-way GroupShuffleSplit.
    # Prior two-stage 80/20 then 50/50 produced val=208 test=1315 (val
    # too small for stable checkpoint selection, ~6pp epoch noise).
    # New: explicit 70/15/15 train/val/test using GroupShuffleSplit twice:
    #   first split 70% train / 30% held-out
    #   second split of held-out: 50% val / 50% test
    # Expected with 42 subjects: train ~8100 / val ~1700 / test ~1700.
    # Subject-disjoint split.
    # IMPORTANT: split is FIXED to SEED=42 across all ensemble runs.
    # Different SEED values vary the model init, dropout, SpecAugment,
    # and other stochastic ops — but the data split stays the same so
    # we can ensemble predictions on a single canonical test set.
    # (Prior version: split random_state = seed, which produced
    # broken splits when seed=44 gave test=364 samples — too few to
    # estimate accuracy reliably. Verified Aug 7 2026.)
    split_seed = 42
    if "subject" in df.columns:
        subjects = df["subject"].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=split_seed)
        idx_train_full, idx_temp = next(gss.split(np.arange(len(df)), y_all, groups=subjects))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=split_seed)
        idx_val, idx_test = next(gss2.split(idx_temp, y_all[idx_temp], groups=subjects[idx_temp]))
        idx_val = idx_temp[idx_val]
        idx_test = idx_temp[idx_test]
        print(f"   subject-disjoint: {len(df['subject'].unique())} subjects (split_seed=42, fixed across ensemble)")
    else:
        idx_train_full, idx_temp, _, _ = train_test_split(
            np.arange(len(df)), test_size=0.30, random_state=split_seed, stratify=y_all
        )
        y_temp = y_all[idx_temp]
        idx_val, idx_test, _, _ = train_test_split(
            np.arange(len(idx_temp)), y_temp, test_size=0.50, random_state=split_seed, stratify=y_temp
        )
        idx_val = idx_temp[idx_val]
        idx_test = idx_temp[idx_test]

    print(f"   splits: train={len(idx_train_full)} val={len(idx_val)} test={len(idx_test)}")

    # ---- Load raw audio into memory ----
    # We deliberately do NOT pre-extract wav2vec2 features here. Reasons:
    #   1. We now FINE-TUNE the last 4 transformer layers, so features
    #      need to be computed live inside the training graph (gradients
    #      flow through the encoder). Pre-extracted cached features
    #      detach the encoder from training.
    #   2. Caching object-dtype arrays of variable-length wav2vec2
    #      features is slow on NFS and prone to corruption (Aug 7 hang).
    # Cost: every batch does one wav2vec2 forward (live fine-tuning).
    # Memory: ~11,568 audios x 6 sec x 16 kHz x 4 bytes = ~4.5 GB RAM.
    print(f"   loading raw audio for {len(df)} files...")
    audios_full = []
    skipped = 0
    t0 = time.time()
    for i in range(len(df)):
        p = df["wav_path"].iloc[i]
        try:
            audio, _ = librosa.load(p, sr=TARGET_SR, mono=True, duration=MAX_S + 0.5)
            if len(audio) < 1600:  # < 100ms, probably corrupt
                raise ValueError("audio too short")
            max_samples = int(MAX_S * TARGET_SR)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)))
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            audios_full.append(audio.astype(np.float32))
        except Exception as e:
            skipped += 1
            audios_full.append(np.zeros(int(MAX_S * TARGET_SR), dtype=np.float32))
        if (i + 1) % 1000 == 0:
            print(f"      [{i+1}/{len(df)}] skipped={skipped} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"   loaded {len(audios_full)} audios in {time.time()-t0:.0f}s ({skipped} corrupt/missing)")

    # ---- Build datasets (raw audio, not pre-extracted features) ----
    audio_train = [audios_full[i] for i in idx_train_full]
    audio_val = [audios_full[i] for i in idx_val]
    audio_test = [audios_full[i] for i in idx_test]
    y_train = y_all[idx_train_full]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]
    del audios_full  # free memory; we now hold only the split slices

    train_ds = WavSERDataset(audio_train, y_train, augment=None)
    val_ds = WavSERDataset(audio_val, y_val, augment=None)
    test_ds = WavSERDataset(audio_test, y_test, augment=None)

    # Class-balanced cross-entropy loss (no WeightedRandomSampler).
    # Rationale: WeightedRandomSampler over-samples minority-class
    # examples in train, which biases the model to predict those classes
    # at test time (we saw this in run #1: recall on "happy" was 0.90
    # while recall on other classes was 0.16-0.45). Class-balanced CE
    # achieves the same imbalance correction via the loss weights only,
    # without duplicating samples in the train loader.
    class_counts = np.bincount(y_train, minlength=len(EMOTIONS))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    alpha = class_weights / class_weights.sum()

    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=24, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=24, shuffle=False, num_workers=2, pin_memory=True)

    spec_augment = SpecAugment(time_mask_max=96, freq_mask_max=128,
                               n_time_masks=3, n_freq_masks=3, p=0.6)

    # ---- Model ----
    model = EnhancedSER1DCNN(in_channels=768, num_classes=len(EMOTIONS)).to(device)
    print(f"   trainable params (head only): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Dual-objective loss: 0.5 * focal + 0.5 * label-smoothed CE
    criterion = DualLoss(class_weights=torch.as_tensor(alpha, dtype=torch.float32).to(device),
                         gamma=2.0, alpha=0.25, label_smoothing=0.1)

    # Layer-wise LR decay (ULMFiT-style) on the encoder, higher LR on the head.
    # Layer 0 (input-side) gets lr = 2e-5 * 0.95^11 ~ 1.24e-5
    # Layer 11 (output-side) gets lr = 2e-5
    # Head gets lr = 2e-4 (10x higher)
    encoder_param_groups = feature_extractor.get_layer_param_groups(
        base_lr=2e-5, weight_decay=1e-5, decay_rate=0.95
    )
    head_params = [p for p in model.parameters() if p.requires_grad]
    print(f"   trainable encoder params: {sum(p.numel() for g in encoder_param_groups for p in g['params']):,}")
    print(f"   trainable head params:    {sum(p.numel() for p in head_params):,}")

    optimizer = AdamW(
        encoder_param_groups + [
            {"params": head_params, "lr": 2e-4, "weight_decay": 5e-4},
        ]
    )
    n_epochs = 90
    # Cosine schedule with linear warmup (replaces OneCycleLR; cosine
    # anneals more smoothly on the longer 90-epoch horizon).
    total_steps = n_epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    # ---- Train loop ----
    best_val = 0.0
    best_state = None
    for epoch in range(n_epochs):
        # Train with mixup + SpecAugment + live wav2vec2 forward (for fine-tuning)
        model.train()
        feature_extractor.train()  # encoder has unfrozen layers now
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward through wav2vec2 (gradients flow through unfrozen layers)
            feats = feature_extractor(x)  # (B, T', 768)
            feats = spec_augment(feats)   # time + freq masking on features

            x_mixed, y_a, y_b, lam = mixup_batch(feats, y, alpha=0.2)
            logits = model(x_mixed)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(feature_extractor.parameters()) + list(model.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)
        train_acc = train_correct / train_total
        train_loss /= train_total

        # Validate
        model.eval()
        feature_extractor.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = feature_extractor(x)
                logits = model(feats)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total

        saved = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "feature_extractor": {k: v.cpu().clone() for k, v in feature_extractor.state_dict().items()},
            }
            saved = "  \u2705 saved best"
        lr_head = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1}/{n_epochs}  train={train_acc:.4f}  val={val_acc:.4f}  loss={train_loss:.4f}  lr_head={lr_head:.2e}{saved}", flush=True)

    # ---- Save best ----
    if best_state:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ser_best_seed{seed}.pt")
        torch.save(best_state, ckpt_path)
        with open(os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl"), "wb") as f:
            pickle.dump(le, f)
        print(f"\n\u2705 best val_acc={best_val:.4f} -> saved to {ckpt_path}")

    # ---- Test with TTA (test-time augmentation) ----
    # Load best state into both model and feature_extractor
    model.load_state_dict(best_state["model"])
    feature_extractor.load_state_dict(best_state["feature_extractor"])
    model.eval()
    feature_extractor.eval()

    def _predict_one(x: torch.Tensor) -> torch.Tensor:
        """One forward pass through encoder + head. Returns softmax probs (B, C)."""
        with torch.no_grad():
            feats = feature_extractor(x)
            logits = model(feats)
        return F.softmax(logits, dim=-1)

    def _predict_tta(x: torch.Tensor, n_passes: int = 5) -> torch.Tensor:
        """Average predictions over n_passes random time crops.
        Wav2vec2-base is fully convolutional over time, so we can
        safely crop random subsegments and average the predictions
        — this reduces test-time variance.
        """
        probs_sum = _predict_one(x)
        if n_passes <= 1:
            return probs_sum
        T = x.shape[1]
        for _ in range(n_passes - 1):
            # Random crop of 80-100% of length
            crop_frac = float(torch.rand(1).item() * 0.2 + 0.8)
            crop_T = int(T * crop_frac)
            t0 = int(torch.randint(0, T - crop_T + 1, (1,)).item())
            x_crop = x[:, t0:t0 + crop_T]
            # Pad back to original length with zeros (wav2vec2 handles padding)
            x_padded = F.pad(x_crop, (0, T - crop_T))
            probs_sum = probs_sum + _predict_one(x_padded)
        return probs_sum / n_passes

    y_true, y_pred = [], []
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            probs = _predict_tta(x, n_passes=5)
            preds = probs.argmax(1)
            test_correct += (preds == y).sum().item()
            test_total += x.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    test_acc = test_correct / test_total
    test_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n\u2705 TEST accuracy (with 5-pass TTA): {test_acc:.4f}")
    print(f"\u2705 TEST macro-F1: {test_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=[c for c in le.classes_]))

    # ---- Save summary ----
    summary = {
        "model": "wav2vec2-base (ALL 12 layers unfrozen, layer-wise LR decay) + 1D-CNN with mixup + SpecAugment (96/128, 3x) + DualLoss (focal+CE) + class-balanced weights + 90 epochs cosine + 5-pass TTA",
        "seed": int(seed),
        "best_val_acc": float(best_val),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "n_train": int(len(idx_train_full)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "num_classes": int(len(EMOTIONS)),
        "classes": list(EMOTIONS),
        "epochs_trained": int(n_epochs),
        "gpus_used": int(torch.cuda.device_count()),
        "tta_passes": 5,
        "num_unfrozen_encoder_layers": int(num_unfrozen_layers),
    }
    summary_path = os.path.join(CHECKPOINT_DIR, f"ser_training_summary_seed{seed}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SER model (wav2vec2-base + 1D-CNN)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for splits and model init (default: 42)")
    parser.add_argument("--num-unfrozen-layers", type=int, default=12,
                        help="Number of wav2vec2 transformer layers to unfreeze (default: 12 = all)")
    args = parser.parse_args()
    train_ser_model(seed=args.seed, num_unfrozen_layers=args.num_unfrozen_layers)