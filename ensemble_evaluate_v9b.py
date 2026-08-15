"""
ensemble_evaluate_v9b.py — Build the v9b 3-row Table 1 from saved ckpts.

Loads all available ckpts:
  v9b_text_fold{F}_seed{S}.pt        -> text-only row
  v9b_audio_fold{F}_seed{S}.pt       -> audio-only row
  v9_baseline_fold{F}_seed{S}.pt     -> multimodal row (kept from v9 eval)

For each (fold, seed) combo, evaluates text-only ckpt, audio-only ckpt, and
multimodal ckpt on the same val split. Reports per-seed acc + macro-F1 +
UA-Acc, plus grand-mean W-Acc across folds.

Outputs:
  reports/v9b_table1.json   (consumable by build_v8_paper.py / Table 1 renderer)
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from iemocap_dataset import (
    IEMOCAPDataset, load_manifest, make_random_kfold_splits,
    IDX_TO_EMOTION,
)
from train_ser_v9b_text_only import DebertaV3Head, DEBERTA_NAME, MAX_TEXT_LEN
from train_ser_v9b_audio_only import WavLMHead, WAVLM_NAME
from train_ser_v9_baseline import FrozenEncoders, TARGET_SR
from cmt_fusion import FusionConfig, PureCMT

warnings.filterwarnings("ignore")
CHECKPOINT_DIR = Path("model_checkpoints")


def per_class_recall(labels: np.ndarray, preds: np.ndarray, n_classes: int = 4) -> dict:
    out = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            out[IDX_TO_EMOTION[c]] = 0.0
        else:
            out[IDX_TO_EMOTION[c]] = float((preds[mask] == c).mean())
    return out


@torch.no_grad()
def predict_text(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probs + labels for the DeBERTa text model."""
    model.eval()
    all_p, all_y = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        emo = batch["emotion"].to(device)
        logits = model(ids, am)
        all_p.append(F.softmax(logits, dim=-1).cpu().numpy())
        all_y.append(emo.cpu().numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


@torch.no_grad()
def predict_audio(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probs + labels for the WavLM audio model."""
    model.eval()
    all_p, all_y = [], []
    for batch in loader:
        wav = batch["wav"].to(device)
        emo = batch["emotion"].to(device)
        logits = model(wav)
        all_p.append(F.softmax(logits, dim=-1).cpu().numpy())
        all_y.append(emo.cpu().numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


def load_text_ckpt(ckpt_path: Path, device: str) -> DebertaV3Head:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DebertaV3Head(num_classes=4).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_audio_ckpt(ckpt_path: Path, device: str) -> WavLMHead:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WavLMHead(num_classes=4).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def load_fusion_ckpt(ckpt_path: Path, device: str) -> tuple[PureCMT, "FrozenEncoders"]:
    """Reload the v9 multimodal PureCMT (HuBERT+BERT frozen) and its encoders."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = FusionConfig(
        audio_dim=cfg_dict["audio_dim"],
        text_dim=cfg_dict["text_dim"],
        proj_dim=cfg_dict.get("proj_dim", 256),
        num_classes=cfg_dict.get("num_classes", 4),
        n_cmt_layers=cfg_dict.get("n_cmt_layers", 2),
        n_heads=cfg_dict.get("n_heads", 4),
        va_dim=cfg_dict.get("va_dim", 3),
        va_proj_dim=cfg_dict.get("va_proj_dim", 64),
        dropout=cfg_dict.get("dropout", 0.3),
        dialog_context=cfg_dict.get("dialog_context", False),
        dialog_window=cfg_dict.get("dialog_window", 10),
        aggregation=ckpt.get("aggregation", cfg_dict.get("aggregation", "min")),
    )
    fusion = PureCMT(cfg).to(device)
    fusion.load_state_dict(ckpt["fusion"])
    fusion.eval()
    encoders = FrozenEncoders(device)  # for inference only, weights frozen
    return fusion, encoders


@torch.no_grad()
def predict_fusion(fusion, encoders, loader, device) -> tuple[np.ndarray, np.ndarray]:
    all_p, all_y = [], []
    for batch in loader:
        wav = batch["wav"].to(device)
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        emo = batch["emotion"].to(device)
        a = encoders.encode_audio(wav)
        t = encoders.encode_text(ids, am)
        logits = fusion(a, t)
        all_p.append(F.softmax(logits, dim=-1).cpu().numpy())
        all_y.append(emo.cpu().numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/iemocap/manifest.csv")
    parser.add_argument("--ckpt-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--out", default="reports/v9b_table1.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest = load_manifest(Path(args.manifest))
    print(f"Manifest: {len(manifest)} utterances")

    ckpt_dir = Path(args.ckpt_dir)
    text_paths = sorted(ckpt_dir.glob("v9b_text_fold*_seed*.pt"))
    audio_paths = sorted(ckpt_dir.glob("v9b_audio_fold*_seed*.pt"))
    fusion_paths = sorted(ckpt_dir.glob("v9_baseline_fold*_seed*.pt"))
    print(f"Found text ckpts: {len(text_paths)} | audio ckpts: {len(audio_paths)} | fusion ckpts: {len(fusion_paths)}")

    # Tokenizer for text eval (DeBERTa-v3 — also valid for BERT in FrozenEncoders)
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)

    # Pre-build per-(fold, seed) manifest views
    splits_cache = {}
    for seed in args.seeds:
        splits_cache[seed] = make_random_kfold_splits(manifest, n_folds=5, seed=seed)

    results = {
        "text": defaultdict(dict),    # (fold, seed) -> {acc, f1, ua}
        "audio": defaultdict(dict),
        "fusion": defaultdict(dict),
    }

    # ---- Text evaluation ----
    for p in text_paths:
        import re
        m = re.search(r"fold(\d+)_seed(\d+)\.pt$", p.name)
        if not m:
            continue
        fold, seed = int(m.group(1)), int(m.group(2))
        if seed not in args.seeds:
            continue
        train_idx, val_idx = splits_cache[seed][fold]
        val_manifest = [manifest[i] for i in val_idx]
        ds = IEMOCAPDataset(val_manifest, tokenizer, target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
        model = load_text_ckpt(p, device)
        probs, labels = predict_text(model, loader, device)
        preds = probs.argmax(axis=1)
        acc = float(accuracy_score(labels, preds))
        f1m = float(f1_score(labels, preds, average="macro"))
        ua = float(np.mean(list(per_class_recall(labels, preds).values())))
        results["text"][(fold, seed)] = {"acc": acc, "f1": f1m, "ua": ua, "ckpt": p.name}
        print(f"  TEXT fold{fold} seed{seed}: acc={acc:.4f} f1={f1m:.4f} ua={ua:.4f}")
        del model
        torch.cuda.empty_cache()

    # ---- Audio evaluation ----
    for p in audio_paths:
        import re
        m = re.search(r"fold(\d+)_seed(\d+)\.pt$", p.name)
        if not m:
            continue
        fold, seed = int(m.group(1)), int(m.group(2))
        if seed not in args.seeds:
            continue
        train_idx, val_idx = splits_cache[seed][fold]
        val_manifest = [manifest[i] for i in val_idx]
        ds = IEMOCAPDataset(val_manifest, tokenizer, target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
        model = load_audio_ckpt(p, device)
        probs, labels = predict_audio(model, loader, device)
        preds = probs.argmax(axis=1)
        acc = float(accuracy_score(labels, preds))
        f1m = float(f1_score(labels, preds, average="macro"))
        ua = float(np.mean(list(per_class_recall(labels, preds).values())))
        results["audio"][(fold, seed)] = {"acc": acc, "f1": f1m, "ua": ua, "ckpt": p.name}
        print(f"  AUDIO fold{fold} seed{seed}: acc={acc:.4f} f1={f1m:.4f} ua={ua:.4f}")
        del model
        torch.cuda.empty_cache()

    # ---- Fusion (v9 multimodal) evaluation — kept from v9 ----
    for p in fusion_paths:
        import re
        m = re.search(r"fold(\d+)_seed(\d+)\.pt$", p.name)
        if not m:
            continue
        fold, seed = int(m.group(1)), int(m.group(2))
        if seed not in args.seeds:
            continue
        train_idx, val_idx = splits_cache[seed][fold]
        val_manifest = [manifest[i] for i in val_idx]
        ds = IEMOCAPDataset(val_manifest, tokenizer, target_sr=TARGET_SR, max_text_len=MAX_TEXT_LEN)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        fusion, encoders = load_fusion_ckpt(p, device)
        probs, labels = predict_fusion(fusion, encoders, loader, device)
        preds = probs.argmax(axis=1)
        acc = float(accuracy_score(labels, preds))
        f1m = float(f1_score(labels, preds, average="macro"))
        ua = float(np.mean(list(per_class_recall(labels, preds).values())))
        results["fusion"][(fold, seed)] = {"acc": acc, "f1": f1m, "ua": ua, "ckpt": p.name}
        print(f"  FUSION fold{fold} seed{seed}: acc={acc:.4f} f1={f1m:.4f} ua={ua:.4f}")
        del fusion, encoders
        torch.cuda.empty_cache()

    def grand_mean(d):
        if not d:
            return {"acc": 0.0, "f1": 0.0, "ua": 0.0, "n": 0}
        return {
            "acc": float(np.mean([v["acc"] for v in d.values()])),
            "f1": float(np.mean([v["f1"] for v in d.values()])),
            "ua": float(np.mean([v["ua"] for v in d.values()])),
            "n": len(d),
        }

    def per_fold(d):
        out = {}
        for (fold, seed), v in d.items():
            out.setdefault(fold, []).append(v["acc"])
        return {f: float(np.mean(a)) for f, a in out.items()}

    out = {
        "text": {
            "model": "DeBERTa-v3-base + unfreeze last 2 + Linear head",
            "per_seed": {f"fold{f}_seed{s}": results["text"][(f, s)] for (f, s) in results["text"]},
            "per_fold_mean_acc": per_fold(results["text"]),
            "grand_mean": grand_mean(results["text"]),
        },
        "audio": {
            "model": "WavLM-Base+ + unfreeze last 2 + mean-pool + Linear head",
            "per_seed": {f"fold{f}_seed{s}": results["audio"][(f, s)] for (f, s) in results["audio"]},
            "per_fold_mean_acc": per_fold(results["audio"]),
            "grand_mean": grand_mean(results["audio"]),
        },
        "fusion": {
            "model": "PureCMT (HuBERT-base + BERT-base, frozen, CMT min-aggregation) — kept from v9",
            "per_seed": {f"fold{f}_seed{s}": results["fusion"][(f, s)] for (f, s) in results["fusion"]},
            "per_fold_mean_acc": per_fold(results["fusion"]),
            "grand_mean": grand_mean(results["fusion"]),
        },
        "comparison": {
            "MemoCMT_published": {
                "text_W_acc": 0.7110, "text_UA_acc": 0.7112,
                "audio_W_acc": 0.6656, "audio_UA_acc": 0.6534,
                "fusion_W_acc": 0.7925, "fusion_UA_acc": 0.7892,
            },
            "v8_LOSO_mean": 0.7298,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== v9b TABLE 1 — grand means ===")
    print(f"  TEXT    : W-Acc={out['text']['grand_mean']['acc']:.4f}  F1={out['text']['grand_mean']['f1']:.4f}  UA={out['text']['grand_mean']['ua']:.4f}  (n={out['text']['grand_mean']['n']})")
    print(f"  AUDIO   : W-Acc={out['audio']['grand_mean']['acc']:.4f}  F1={out['audio']['grand_mean']['f1']:.4f}  UA={out['audio']['grand_mean']['ua']:.4f}  (n={out['audio']['grand_mean']['n']})")
    print(f"  FUSION  : W-Acc={out['fusion']['grand_mean']['acc']:.4f}  F1={out['fusion']['grand_mean']['f1']:.4f}  UA={out['fusion']['grand_mean']['ua']:.4f}  (n={out['fusion']['grand_mean']['n']})")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
