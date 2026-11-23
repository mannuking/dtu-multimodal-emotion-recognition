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


# EmoryNLP uses its own label names for the same Ekman emotions.
EMORY_ALIASES = {
    "joyful": "happy", "happy": "happy",
    "sad": "sad", "peaceful": "neutral", "neutral": "neutral",
    "mad": "angry", "angry": "angry",
    "disgusted": "disgust", "disgust": "disgust",
    "terrified": "fear", "fear": "fear",
    "surprised": "surprise", "surprise": "surprise",
}


def canon_emory(label: str) -> str | None:
    """Map an EmoryNLP label onto the canonical 7-class taxonomy."""
    key = str(label).strip().lower()
    if key in EMORY_ALIASES:
        return EMORY_ALIASES[key]
    return canon(key)


def write_split(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "emotion"])
        for t, e in rows:
            w.writerow([t, e])


DAILYDIALOG_FILES = {
    "train": "data/train-00000-of-00001-f151c79abb2c1fd5.parquet",
    "validation": "data/validation-00000-of-00001-2407eb323af19881.parquet",
    "test": "data/test-00000-of-00001-66dc7d981b70c918.parquet",
}

# DailyDialog emotion ids: 0=no emotion 1=anger 2=disgust 3=fear
#                         4=happiness 5=sadness 6=surprise
DAILYDIALOG_EMOS = ["no emotion", "anger", "disgust", "fear",
                    "happiness", "sadness", "surprise"]


def load_dailydialog():
    """
    DailyDialog from HuggingFace parquet. Returns list of (text, canon).

    Read the parquet DIRECTLY rather than through `datasets`.
    `load_dataset()` on the script-based mirrors needs a dataset script,
    which newer `datasets` releases refuse to execute -- and the installed
    version is pinned to 2.x, so that path is fragile.

    NOTE: DeepPavlov/daily_dialog is DIALOG-level (87170 rows, one label
    per dialogue) and is useless for utterance-level TER. OpenRL/daily_dialog
    is the utterance-level mirror: 11118 dialogs, each with a per-utterance
    `emotion` array aligned to the `dialog` array. Use that one.
    """
    try:
        import pandas as pd
    except Exception as e:
        print(f"  [dailydialog] skipped (no pandas: {e})")
        return []

    out = []
    for split, fname in DAILYDIALOG_FILES.items():
        url = f"https://huggingface.co/datasets/OpenRL/daily_dialog/resolve/main/{fname}"
        try:
            import io
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                blob = resp.read()
            df = pd.read_parquet(io.BytesIO(blob))
        except Exception as e:
            print(f"  [dailydialog] {split} failed: {e}")
            continue

        if "dialog" not in df.columns or "emotion" not in df.columns:
            print(f"  [dailydialog] {split} has unexpected columns {list(df.columns)}")
            continue

        n = 0
        for _, row in df.iterrows():
            try:
                utts = list(row["dialog"])
                emos = list(row["emotion"])
            except Exception:
                continue
            if len(utts) != len(emos):
                continue
            for u, e in zip(utts, emos):
                name = DAILYDIALOG_EMOS[int(e)] if 0 <= int(e) < len(DAILYDIALOG_EMOS) else str(e)
                c = canon(name)
                if c and str(u).strip():
                    out.append((str(u).strip(), c))
                    n += 1
        print(f"  [dailydialog] {split}: {n} utterances")
    return out


def load_meld():
    """
    MELD (Friends) from the declare-lab GitHub repo.

    The CSVs live under data/MELD/, NOT at the repo root — the earlier
    path guessed data/MELD_{split}.csv at the top level and 404'd on all
    three splits.
    """
    import csv as _csv
    import io
    import urllib.request
    out = []
    base = ("https://raw.githubusercontent.com/declare-lab/MELD/master/data/"
            "MELD/{f}")
    for split, fn in (("train", "train_sent_emo.csv"),
                      ("dev", "dev_sent_emo.csv"),
                      ("test", "test_sent_emo.csv")):
        url = base.format(f=fn)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [meld] {split} unavailable: {e}")
            continue
        n = 0
        for row in _csv.DictReader(io.StringIO(text)):
            # The MELD CSVs use CAPITALISED headers: 'Utterance', 'Emotion'.
            # Looking for 'utterance'/'emotion' matched nothing and the
            # split silently contributed 0 rows.
            u = (row.get("Utterance") or row.get("utterance") or "").strip()
            c = canon(row.get("Emotion") or row.get("emotion") or "")
            if u and c:
                out.append((u, c))
                n += 1
        print(f"  [meld] {split}: {n} utterances")
    return out


def load_emorynlp():
    """EmoryNLP from the declare-lab/MELD repo (data/emorynlp/)."""
    import csv as _csv
    import io
    import urllib.request
    out = []
    base = ("https://raw.githubusercontent.com/declare-lab/MELD/master/data/"
            "emorynlp/{f}")
    for split, fn in (("train", "emorynlp_train_final.csv"),
                      ("dev", "emorynlp_dev_final.csv"),
                      ("test", "emorynlp_test_final.csv")):
        url = base.format(f=fn)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  [emorynlp] {split} unavailable: {e}")
            continue
        n = 0
        for row in _csv.DictReader(io.StringIO(text)):
            # EmoryNLP headers are 'Utterance'/'Emotion', and its emotion
            # vocabulary is its own ('Joyful', 'Sad', 'Mad', ...), which
            # needs mapping onto the canonical 7 classes.
            u = (row.get("Utterance") or row.get("utterance") or "").strip()
            c = canon_emory(row.get("Emotion") or row.get("emotion") or "")
            if u and c:
                out.append((u, c))
                n += 1
        print(f"  [emorynlp] {split}: {n} utterances")
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
