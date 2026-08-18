"""
iemocap_dataset.py — PyTorch Dataset for IEMOCAP 4-class with dialog context.

Returns per-item:
  - wav:    (T,) float32 at 16 kHz
  - input_ids:  (T_text,) int64 tokenized
  - attention_mask: (T_text,) int64
  - v: float — valence  [0, 1]
  - a: float — arousal  [0, 1]
  - d: float — dominance [0, 1]
  - emotion: int64 — class index
  - session: str
  - utterance_id: str

Each __getitem__ also returns the dialog_id and the position in the
dialog (for the 10-previous-utterances context window).

For LOSO 4-fold CV, the caller builds TrainDataset(manifest, train_sessions)
and ValDataset(manifest, held_out_session). Same loader, just different
manifest filter.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

EMOTION_TO_IDX = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


def load_manifest(path: Path) -> list[dict]:
    """Read the manifest.csv as a list of dicts."""
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_dict("records")


def slice_dialog_wav(wav_path: str, start: float, end: float,
                     target_sr: int = 16000) -> torch.Tensor:
    """
    Read a dialog-level wav and slice out the per-utterance audio
    between start and end (seconds). Pad to 6 seconds if shorter.
    Returns (T,) float32 tensor at 16 kHz.

    The manifest stores wav paths relative to the repo root
    (e.g. "data/iemocap/Session1/dialog/wav/Ses01F_impro01.wav"). When
    sbatch does "cd scripts/iemocap" before running Python, that relative
    path would resolve to scripts/iemocap/data/iemocap/... which does
    not exist. Convert to absolute at this entry point using the repo
    root (where this iemocap_dataset.py file lives, parent.parent.parent).
    Also uses soundfile (libsndfile) instead of torchaudio for HPC compat,
    and copies to /tmp first to dodge NFS file-handle staleness.
    """
    from pathlib import Path as _P
    # iemocap_dataset.py lives at repo root, so parent.parent is the repo root.
    # (NOT parent.parent.parent — that would go up one too many levels.)
    repo_root = _P(__file__).parent.parent
    if not _P(wav_path).is_absolute():
        wav_path = str(repo_root / wav_path)
    import soundfile as sf
    import shutil
    import tempfile
    # HPC compute nodes raise "System error" on libsndfile reads even though
    # the same files are readable on the login node. The cause is NFS file-handle
    # staleness on the compute node mount. Workaround: copy the wav to /tmp first
    # (forces a fresh NFS read), then read from /tmp.
    #
    # Falls back to silent tensor on any failure so a single bad file doesn't
    # kill the training run.
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            shutil.copy(wav_path, tmp.name)
            tmp_path = tmp.name
        try:
            wav, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        if not hasattr(slice_dialog_wav, "_bad_files"):
            slice_dialog_wav._bad_files = set()
        if wav_path not in slice_dialog_wav._bad_files:
            slice_dialog_wav._bad_files.add(wav_path)
            print(f"[slice_dialog_wav] WARN: cannot read {wav_path}: "
                  f"{type(e).__name__}: {e}. Using silent tensor instead.",
                  flush=True)
        return torch.zeros(target_sr * 6, dtype=torch.float32)
    wav = torch.from_numpy(wav)  # (T, C)
    if sr != target_sr:
        # Linear resample via torch (no torchaudio dependency).
        # For the small fraction of files not at 16 kHz (rare in IEMOCAP).
        ratio = target_sr / sr
        new_len = int(wav.shape[0] * ratio)
        wav = torch.nn.functional.interpolate(
            wav.transpose(0, 1).unsqueeze(0),  # (1, C, T)
            size=new_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0).transpose(0, 1)  # back to (T, C)
        sr = target_sr
    # Mono
    if wav.shape[1] > 1:
        wav = wav.mean(dim=1, keepdim=True)
    wav = wav.squeeze(1)  # (T,)
    s = int(start * sr)
    e = int(end * sr)
    s = max(0, min(s, len(wav)))
    e = max(s, min(e, len(wav)))
    seg = wav[s:e]
    target_len = target_sr * 6
    if len(seg) < target_len:
        seg = torch.nn.functional.pad(seg, (0, target_len - len(seg)))
    elif len(seg) > target_len:
        seg = seg[:target_len]
    return seg.float()


class IEMOCAPDataset(Dataset):
    """
    One item per utterance in the manifest.

    The dialog context (10 previous utterances' [CLS] vectors) is
    computed in __getitem__ by looking up the manifest entries for
    the same dialog with utt_id < current utt_id. The caller (training
    loop) accumulates these into a per-dialog tensor of shape
    (10, hidden_size).
    """

    def __init__(
        self,
        manifest: list[dict],
        tokenizer: AutoTokenizer,
        target_sr: int = 16000,
        max_text_len: int = 64,
    ):
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.max_text_len = max_text_len
        # Index by (dialog_id, position_in_dialog) for fast context lookup
        self._by_dialog: dict[str, list[dict]] = {}
        for row in manifest:
            self._by_dialog.setdefault(row["dialog"], []).append(row)
        # Sort each dialog by start_time
        for d in self._by_dialog:
            self._by_dialog[d].sort(key=lambda r: r["start_time"])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest[idx]
        wav = slice_dialog_wav(row["wav_path"], row["start_time"],
                                row["end_time"], self.target_sr)
        text = str(row.get("transcript", "") or "")
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attn_mask = enc["attention_mask"].squeeze(0)
        else:
            # Audio-only baseline: no tokenizer needed, return zero tensors so the
            # collate path doesn't crash. Models that ignore them never look.
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attn_mask = torch.zeros(self.max_text_len, dtype=torch.long)
        emo_idx = EMOTION_TO_IDX[row["emotion"]]
        item = {
            "wav": wav,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "valence": float(row["valence"]) if not pd_is_nan(row["valence"]) else 0.5,
            "arousal": float(row["arousal"]) if not pd_is_nan(row["arousal"]) else 0.5,
            "dominance": float(row["dominance"]) if not pd_is_nan(row["dominance"]) else 0.5,
            "emotion": torch.tensor(emo_idx, dtype=torch.long),
            "session": row["session"],
            "utterance_id": row["utterance_id"],
            "dialog": row["dialog"],
        }
        return item


def pd_is_nan(x) -> bool:
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False


def make_loso_splits(manifest: list[dict]) -> list[tuple[set, set]]:
    """
    Returns 5 (train_sessions, held_out_session) tuples for LOSO CV.
    Session1-5 each held out once; 4 sessions train + 1 session val.
    """
    sessions = sorted({r["session"] for r in manifest})
    splits = []
    for held in sessions:
        train = set(sessions) - {held}
        splits.append((train, {held}))
    return splits


def make_random_kfold_splits(manifest: list[dict], n_folds: int = 5,
                             seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """
    MemoCMT-style random k-fold CV: split utterances randomly into n_folds
    buckets, then for each fold k: train = all-but-bucket-k, val = bucket-k.

    Returns list of (train_indices, val_indices) tuples, one per fold.
    Index refers to position in the manifest list.
    """
    rng = np.random.RandomState(seed)
    n = len(manifest)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1
    folds = []
    cursor = 0
    for size in fold_sizes:
        folds.append(indices[cursor:cursor + size].tolist())
        cursor += size
    splits = []
    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = [i for j, f in enumerate(folds) if j != k for i in f]
        splits.append((train_idx, val_idx))
    return splits


# Tiny sanity test (run with `python iemocap_dataset.py`)
if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    if len(sys.argv) < 2:
        print("Usage: python iemocap_dataset.py <manifest.csv>")
        sys.exit(1)
    m = load_manifest(Path(sys.argv[1]))
    print(f"Loaded {len(m)} manifest rows")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = IEMOCAPDataset(m, tok)
    print(f"Dataset length: {len(ds)}")
    item = ds[0]
    print(f"Sample item keys: {list(item.keys())}")
    print(f"  wav shape: {item['wav'].shape}")
    print(f"  input_ids shape: {item['input_ids'].shape}")
    print(f"  emotion: {item['emotion']} ({IDX_TO_EMOTION[int(item['emotion'])]})")
    print(f"  valence/arousal/dominance: {item['valence']:.3f}/{item['arousal']:.3f}/{item['dominance']:.3f}")
    print(f"  utterance_id: {item['utterance_id']}")
    splits = make_loso_splits(m)
    print(f"\nLOSO splits:")
    for i, (train, held) in enumerate(splits):
        n_train = sum(1 for r in m if r["session"] in train)
        n_held = sum(1 for r in m if r["session"] in held)
        print(f"  Fold {i+1}: train={sorted(train)} ({n_train} utts), "
              f"val={sorted(held)} ({n_held} utts)")
