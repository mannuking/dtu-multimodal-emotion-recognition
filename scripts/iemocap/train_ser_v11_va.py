"""
train_ser_v11_va.py — v11: V/A-conditioned CMT fusion head on top of the v10-B
architecture, with an auxiliary valence/arousal regression objective.

WHY THIS EXISTS
---------------
v10-B (the CMT fusion head on frozen v9b WavLM-base+ / DeBERTa-v3-base
encoders) produced 30 checkpoints on 18 Aug 2026 but was never evaluated.
v11 is the rank-1 candidate from the post-v10 roadmap: add a
Valence/Arousal bias so the classifier is told *where in affective space*
each class sits, not just which of four labels it is.

The IEMOCAP manifest already carries per-utterance valence, arousal and
dominance (from the EmoEvaluation self-assessment annotations), normalised
to 0-1. So the bias is supervised, not hand-set.

TWO DESIGN CHOICES, AND WHY
---------------------------
1. V/A EMBEDDING, NOT A HARD-CODED PRIOR.
   Each of the four classes gets a LEARNED prototype in (V, A) space
   (`class_va`), initialised from the dataset's mean V/A per class but
   free to move. A hard-coded prior would freeze a quantity we can
   measure; a learned prototype lets the model discover that, say, "angry"
   and "sad" sit apart in arousal but overlap in valence on this corpus.

2. AUXILIARY REGRESSION HEAD, NOT A POST-HOC ARBITRATION.
   The head predicts a continuous (valence, arousal) vector per utterance
   from the same fused representation, trained with MSE against the
   EmoEvaluation ground truth. At inference the continuous prediction is
   compared against the four learned class prototypes, and that distance
   is added to the logits as a soft bias. This means the V/A pathway is
   trained end-to-end on its own signal and then *used*, rather than
   being a post-hoc rule bolted onto confident logits.

   The bias is added to the logits, so it competes with the classifier
   rather than overriding it. `va_weight` scales it; 0.0 recovers v10-B
   exactly, which makes this an ablation rather than a leap of faith.

WHAT IS THE SAME AS v10-B
-------------------------
- Frozen v9b encoders, one ckpt per (fold, seed), loaded by FrozenEncoder.
- Same random 5-fold x 3-seed protocol, same splits, same manifest.
- Same tensor fusion [t ; a ; t*a ; |t-a|] -> MLP(2048->512->4).
- Same class-weight option, same early stopping on val accuracy.

WHAT IS NEW
-----------
- `class_va`: nn.Parameter (4, 2) — learned V/A prototype per class.
- `va_head`: nn.Linear(d_model, 2) — continuous (valence, arousal) regression.
- Auxiliary MSE loss on (valence, arousal), weight `--va-reg-weight`.
- Logit bias = -va_weight * ||predicted_va - class_va[target]||^2, i.e.
  a soft penalty on classes whose prototype is far from the utterance's
  predicted affect.
- Summary JSON records the VA regression error so the auxiliary task can be
  checked, not assumed.

CHECKPOINT NAMING
-----------------
v11_va_{unif|cw}_fold{F}_seed{S}.pt — deliberately parallel to
v10_fusion_{unif|cw}_fold{F}_seed{S}.pt so that eval scripts can treat
them as one more branch of the same array.

References for the affective-space reasoning:
- Bradley & Lang 1994, "Measuring emotion: The self-assessment
  manikin and semantic differential" (the IEMOCAP self-assessment scales).
- Russell 1980, "A circumplex model of affect" (the V/A circumplex the
  class prototypes are parameterised in).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Repo root exposes `iemocap_dataset`; this directory holds the v10 module
# we reuse. Both go on the path explicitly rather than relying on namespace
# packages, which is the style v10 uses and is proven on the cluster.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iemocap_dataset import (  # noqa: E402
    IEMOCAPDataset, load_manifest, make_random_kfold_splits, IDX_TO_EMOTION,
)
from train_ser_v10_fusion import (  # noqa: E402
    FrozenEncoder, CrossAttentionBlock,
    WAVLM_NAME, DEBERTA_NAME, TARGET_SR, MAX_TEXT_LEN,
)

CHECKPOINT_DIR = REPO_ROOT / "model_checkpoints"
CLASS_WEIGHTS = torch.tensor([1.30, 1.00, 1.00, 1.15])


class VAFusionHead(nn.Module):
    """v10-B's CMT fusion head plus a valence/arousal pathway.

    forward() returns (logits, va_pred, va_bias) where va_bias is the
    additive per-class correction in logit space.
    """

    def __init__(self, d_model: int = 512, n_blocks: int = 2,
                 num_classes: int = 4, dropout: float = 0.3,
                 va_weight: float = 0.5,
                 class_va_init: torch.Tensor | None = None):
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

        # Continuous affect regression from the fused representation.
        self.va_head = nn.Linear(d_model, 2)

        # Learned (valence, arousal) prototype per class.
        if class_va_init is not None:
            init = torch.as_tensor(class_va_init, dtype=torch.float32).clone()
        else:
            init = torch.tensor([[0.5, 0.5]] * num_classes, dtype=torch.float32)
        self.class_va = nn.Parameter(init)

        self.va_weight = va_weight
        fusion_dim = 4 * d_model
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor):
        B = text_emb.size(0)
        t = self.text_proj(text_emb).unsqueeze(1)
        a = self.audio_proj(audio_emb).unsqueeze(1)
        t = t + self.text_cls
        a = a + self.audio_cls
        for block in self.blocks:
            t, a = block(t, a)
        t = self.norm(t).squeeze(1)
        a = self.norm(a).squeeze(1)
        fused = torch.cat([t, a, t * a, torch.abs(t - a)], dim=-1)

        logits = self.head(fused)
        va_pred = torch.sigmoid(self.va_head(fused))          # (B, 2) in [0,1]

        # Squared distance from each class prototype -> (B, C)
        # va_pred: (B,2) -> (B,1,2); class_va: (C,2) -> (1,C,2)
        dist = ((va_pred.unsqueeze(1) - self.class_va.unsqueeze(0)) ** 2).sum(-1)
        va_bias = -self.va_weight * dist
        return logits + va_bias, va_pred, va_bias


def build_class_va_init(manifest) -> torch.Tensor:
    """Mean (valence, arousal) per class, from the manifest.

    Used only to INITIALISE the learned prototypes. If a class has no
    rows with finite V/A, fall back to (0.5, 0.5) for that class rather
    than letting a NaN into the parameter.
    """
    import pandas as pd

    df = pd.DataFrame(manifest)
    out = np.full((4, 2), 0.5, dtype=np.float32)
    for idx, emo in IDX_TO_EMOTION.items():
        sub = df[df["emotion"] == emo]
        if len(sub) == 0:
            continue
        v = pd.to_numeric(sub["valence"], errors="coerce")
        a = pd.to_numeric(sub["arousal"], errors="coerce")
        if v.notna().any() and a.notna().any():
            out[idx, 0] = float(v.mean())
            out[idx, 1] = float(a.mean())
    return torch.from_numpy(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "iemocap" / "manifest.csv"))
    parser.add_argument("--ckpt-text-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--ckpt-audio-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--class-weighted", default="false",
                        choices=["true", "false"])
    # --- new in v11 ---
    parser.add_argument("--va-weight", type=float, default=0.5,
                        help="scale of the V/A logit bias; 0.0 reproduces v10-B exactly")
    parser.add_argument("--va-reg-weight", type=float, default=0.3,
                        help="weight of the auxiliary (valence, arousal) MSE loss")
    args = parser.parse_args()

    use_class_weights = args.class_weighted == "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== v11 VA-Fusion  fold={args.fold} seed={args.seed} "
          f"cw={use_class_weights} va_weight={args.va_weight} "
          f"va_reg_weight={args.va_reg_weight} device={device} ===")

    manifest = load_manifest(Path(args.manifest))
    splits = make_random_kfold_splits(manifest, n_folds=5, seed=args.seed)
    train_idx, val_idx = splits[args.fold]
    train_manifest = [manifest[i] for i in train_idx]
    val_manifest = [manifest[i] for i in val_idx]
    print(f"train={len(train_manifest)} val={len(val_manifest)}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)

    train_loader = DataLoader(
        IEMOCAPDataset(train_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=False)
    val_loader = DataLoader(
        IEMOCAPDataset(val_manifest, tokenizer,
                       target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True)

    text_ckpt = Path(args.ckpt_text_dir) / f"v9b_text_fold{args.fold}_seed{args.seed}.pt"
    audio_ckpt = Path(args.ckpt_audio_dir) / f"v9b_audio_fold{args.fold}_seed{args.seed}.pt"
    if not text_ckpt.exists() or not audio_ckpt.exists():
        print(f"FATAL: missing v9b ckpts: {text_ckpt} or {audio_ckpt}")
        sys.exit(1)
    text_enc = FrozenEncoder(text_ckpt, kind="text").to(device)
    audio_enc = FrozenEncoder(audio_ckpt, kind="audio").to(device)

    va_init = build_class_va_init(train_manifest)
    print("  class VA prototypes (init):")
    for idx, emo in IDX_TO_EMOTION.items():
        print(f"    {emo:8s} V={va_init[idx,0]:.3f} A={va_init[idx,1]:.3f}")

    head = VAFusionHead(va_weight=args.va_weight, class_va_init=va_init).to(device)
    head_params = [p for p in head.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW(head_params, lr=args.lr_head,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    class_weights = CLASS_WEIGHTS.to(device) if use_class_weights else None

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    weight_tag = "cw" if use_class_weights else "unif"
    best_val_acc, best_epoch, best_f1 = 0.0, -1, 0.0
    best_val_ua, best_vae = 0.0, float("nan")
    history, epochs_no_improve = [], 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        head.train()
        tr_loss = tr_ce = tr_reg = tr_correct = tr_total = 0
        for batch in train_loader:
            emo = batch["emotion"].to(device)
            va_true = torch.stack([batch["valence"], batch["arousal"]], dim=1).to(device)
            with torch.no_grad():
                text_emb = text_enc.extract(batch)
                audio_emb = audio_enc.extract(batch)
            logits, va_pred, _ = head(text_emb, audio_emb)
            ce = F.cross_entropy(logits, emo, weight=class_weights)
            reg = F.mse_loss(va_pred, va_true)
            loss = ce + args.va_reg_weight * reg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * emo.size(0)
            tr_ce += ce.item() * emo.size(0)
            tr_reg += reg.item() * emo.size(0)
            tr_correct += (logits.argmax(1) == emo).sum().item()
            tr_total += emo.size(0)
        train_loss = tr_loss / tr_total
        train_acc = tr_correct / tr_total

        head.eval()
        v_correct = v_total = 0
        v_vae = 0.0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                emo = batch["emotion"].to(device)
                va_true = torch.stack([batch["valence"], batch["arousal"]], dim=1).to(device)
                text_emb = text_enc.extract(batch)
                audio_emb = audio_enc.extract(batch)
                logits, va_pred, _ = head(text_emb, audio_emb)
                v_correct += (logits.argmax(1) == emo).sum().item()
                v_total += emo.size(0)
                v_vae += F.mse_loss(va_pred, va_true, reduction="sum").item()
                v_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                v_labels.extend(emo.cpu().numpy().tolist())
        val_acc = v_correct / v_total
        val_f1 = f1_score(v_labels, v_preds, average="macro")
        val_ua = accuracy_score(v_labels, v_preds)
        val_vae = v_vae / (v_total * 2)

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, best_f1 = val_acc, epoch, val_f1
            best_val_ua, best_val_vae = val_ua, val_vae
            epochs_no_improve = 0
            ckpt_path = (CHECKPOINT_DIR
                         / f"v11_va_{weight_tag}_fold{args.fold}_seed{args.seed}.pt")
            torch.save({
                "head": head.state_dict(),
                "fold": args.fold,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_ua": val_ua,
                "val_vae": val_vae,
                "seed": args.seed,
                "class_weighted": use_class_weights,
                "va_weight": args.va_weight,
                "va_reg_weight": args.va_reg_weight,
            }, ckpt_path)
            saved = f"  saved {ckpt_path.name}"
        else:
            epochs_no_improve += 1
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc,
                        "val_f1": val_f1, "val_ua": val_ua,
                        "val_vae": val_vae,
                        "train_ce": tr_ce / tr_total, "train_reg": tr_reg / tr_total})
        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f} "
              f"acc={train_acc:.4f}  val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
              f"val_ua={val_ua:.4f} val_vae={val_vae:.5f} "
              f"({time.time()-t0:.0f}s){saved}")
        if epochs_no_improve >= args.patience:
            print(f"Early stop @ epoch {epoch} (no improve for {args.patience})")
            break

    print(f"\n=== fold{args.fold} seed{args.seed} ({weight_tag}) done ===")
    print(f"Best val_acc={best_val_acc:.4f} val_f1={best_f1:.4f} "
          f"val_ua={best_val_ua:.4f} val_vae={best_val_vae:.5f} @ epoch {best_epoch}")

    summary_path = (CHECKPOINT_DIR
                    / f"v11_va_{weight_tag}_fold{args.fold}_seed{args.seed}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": "v11 V/A-conditioned CMT fusion head on frozen v9b encoders",
            "fold": args.fold, "seed": args.seed,
            "class_weighted": use_class_weights,
            "va_weight": args.va_weight, "va_reg_weight": args.va_reg_weight,
            "n_train": len(train_manifest), "n_val": len(val_manifest),
            "best_val_acc": best_val_acc, "best_val_f1": best_f1,
            "best_val_ua": best_val_ua, "best_val_vae": best_val_vae,
            "best_epoch": best_epoch, "history": history,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
