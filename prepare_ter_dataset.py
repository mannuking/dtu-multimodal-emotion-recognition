"""
prepare_ter_dataset.py — Build the TER text corpus for the ORIGINAL pipeline.

The original project trained a MobileBERT text-emotion classifier over a
merged text corpus (DailyDialog + MELD + EmoryNLP), written out as:

    ter_dataset/merged/text_train.csv
    ter_dataset/merged/text_val.csv
    ter_dataset/merged/text_test.csv

Those CSVs were never committed — they were rebuilt from the source
datasets on each machine, and the original HPC run fell back to the
SYNTHETIC path in train_ter_pytorch.py (7 hand-written sentences, one per
emotion). A classifier trained on 7 constant sentences learns nothing
about language; its accuracy is a measure of memorisation, not emotion
recognition. This script produces the REAL corpus so the TER leg of
Paper 1 is defensible.

Sources (all public, all downloadable from the compute node):
  - DailyDialog  — HF `DeepPavlov/daily_dialog`  (7 emotion classes)
  - MELD         — GitHub `declare-lab/MELD`     (7 emotion classes)
  - EmoryNLP     — GitHub `emorynlp/emotion-detection`

All three are mapped onto the project's canonical 7-class taxonomy:

    angry  disgust  fear  happy  sad  surprise  neutral

Mapping notes:
  - DailyDialog "no emotion"      -> neutral
  - DailyDialog "happiness"       -> happy
  - DailyDialog "sadness"         -> sad
  - DailyDialog "anger"           -> angry
  - MELD "Joy"                    -> happy
  - MELD "Neutral"                -> neutral
  - EmoryNLP labels are already the canonical names (case-insensitive)

Output CSVs have columns: `text,emotion` — the shape train_ter_pytorch.py's
load_text_csv() auto-detects.

If a source cannot be reached, the script reports it and continues with
whatever it has. It NEVER fabricates data and NEVER falls back to
synthetic sentences: if every source fails it exits non-zero so the
pipeline stops rather than producing a meaningless TER number.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from pathlib import Path

OUT_DIR = Path("ter_dataset/merged")
TRAIN = OUT_DIR / "text_train.csv"
VAL = OUT_DIR / "text_val.csv"
TEST = OUT_DIR / "text_test.csv"

CANON = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Source-specific label -> canonical label.
ALIASES = {
    "anger": "angry", "angry": "angry", "mad": "angry", "frustration": "angry",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "scared": "fear",
    "happiness": "happy", "happy": "happy", "joy": "happy",
    "sadness": "sad", "sad": "sad",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral", "no emotion": "neutral", "none": "neutral",
    "calm": "neutral", "peaceful": "neutral",
}


def canon(label: str) -> str | None:
    return ALIASES.get(str(label).strip().lower())


def write_split(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "emotion"])
        for t, e in rows:
            w.writerow([t, e])


def load_dailydialog():
    """DailyDialog from HuggingFace parquet. Returns list of (text, canon)."""
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print(f"  [dailydialog] skipped (import failed: {e})")
        return []

    out = []
    # The dataset is stored as parquet with list columns (dialog, emotion).
    # Reading it via `datasets` is the robust path.
    try:
        from datasets import load_dataset
        ds = load_dataset("DeepPavlov/daily_dialog")
        emo_names = ["no emotion", "anger", "disgust", "fear",
                     "happiness", "sadness", "surprise"]
        for split in ds:
            for row in ds[split]:
                utts = row.get("dialog") or []
                emos = row.get("emotion") or []
                for u, e in zip(utts, emos):
                    name = emo_names[int(e)] if isinstance(e, int) and int(e) < len(emo_names) else str(e)
                    c = canon(name)
                    if c and str(u).strip():
                        out.append((str(u).strip(), c))
        return out
    except Exception as e:
        print(f"  [dailydialog] `datasets` path failed ({e}); trying parquet directly")
    try:
        for split in ("train", "validation", "test"):
            p = hf_hub_download(repo_id="DeepPavlov/daily_dialog",
                                filename=f"default/{split}/0000.parquet",
                                repo_type="dataset")
            df = pd.read_parquet(p)
            emo_names = ["no emotion", "anger", "disgust", "fear",
                         "happiness", "sadness", "surprise"]
            for _, row in df.iterrows():
                utts = list(row["dialog"])
                emos = list(row["emotion"])
                for u, e in zip(utts, emos):
                    name = emo_names[int(e)] if int(e) < len(emo_names) else str(e)
                    c = canon(name)
                    if c and str(u).strip():
                        out.append((str(u).strip(), c))
    except Exception as e:
        print(f"  [dailydialog] parquet path failed: {e}")
    return out


def load_meld():
    """MELD from the declare-lab GitHub repo (train/val/test CSVs)."""
    import urllib.request
    import csv as _csv
    import io
    out = []
    base = ("https://raw.githubusercontent.com/declare-lab/MELD/master/data/"
            "MELD_{}.csv")
    for split in ("train", "dev", "test"):
        url = base.format(split)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [meld] {split} unavailable: {e}")
            continue
        rd = _csv.DictReader(io.StringIO(text))
        for row in rd:
            u = (row.get("utterance") or "").strip()
            c = canon(row.get("emotion") or "")
            if u and c:
                out.append((u, c))
    return out


def load_emorynlp():
    """EmoryNLP from GitHub."""
    import urllib.request
    import csv as _csv
    import io
    out = []
    base = ("https://raw.githubusercontent.com/emorynlp/emotion-detection/master/"
            "emorynlp/emorynlp_{}.csv")
    for split in ("train", "dev", "test"):
        try:
            with urllib.request.urlopen(base.format(split), timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [emorynlp] {split} unavailable: {e}")
            continue
        rd = _csv.DictReader(io.StringIO(text))
        for row in rd:
            u = (row.get("utterance") or row.get("text") or "").strip()
            c = canon(row.get("emotion") or "")
            if u and c:
                out.append((u, c))
    return out


def main():
    print("=" * 64)
    print("Building the TER text corpus")
    print("=" * 64)

    rows = []
    for name, fn in (("dailydialog", load_dailydialog),
                     ("meld", load_meld),
                     ("emorynlp", load_emorynlp)):
        print(f"[{name}] fetching...")
        got = fn()
        print(f"[{name}] {len(got)} utterances")
        rows.extend(got)

    if not rows:
        print("\nFATAL: no text data could be fetched from any source.")
        print("       Refusing to write a synthetic corpus — a TER model trained")
        print("       on constant sentences produces a meaningless accuracy.")
        print("       Re-run this job on a compute node with outbound network,")
        print("       or pre-stage the CSVs into ter_dataset/merged/ by hand.")
        sys.exit(1)

    # De-duplicate on (text, label) — the sources overlap heavily.
    seen = set()
    uniq = []
    for t, e in rows:
        k = (t.lower(), e)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((t, e))

    print(f"\ntotal: {len(rows)} raw -> {len(uniq)} unique")
    dist = Counter(e for _, e in uniq)
    print("class distribution:")
    for c in CANON:
        print(f"   {c:10s} {dist.get(c, 0):7d}  ({dist.get(c, 0) / max(1, len(uniq)):6.2%})")

    # Deterministic 80/10/10 split, stratified by class.
    import random
    rnd = random.Random(42)
    by_class = {c: [] for c in CANON}
    for t, e in uniq:
        by_class[e].append((t, e))
    train, val, test = [], [], []
    for c in CANON:
        items = by_class[c]
        rnd.shuffle(items)
        n = len(items)
        n_va = max(1, int(n * 0.10))
        n_te = max(1, int(n * 0.10))
        val.extend(items[:n_va])
        test.extend(items[n_va:n_va + n_te])
        train.extend(items[n_va + n_te:])
    rnd.shuffle(train); rnd.shuffle(val); rnd.shuffle(test)

    write_split(train, TRAIN)
    write_split(val, VAL)
    write_split(test, TEST)
    print(f"\nwrote:")
    print(f"   {TRAIN}  {len(train)}")
    print(f"   {VAL}  {len(val)}")
    print(f"   {TEST}  {len(test)}")
    print(f"\ncolumns: text,emotion   (train_ter_pytorch.py auto-detects these)")


if __name__ == "__main__":
    main()
