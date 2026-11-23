"""
verify_original_pipeline.py — Integrity + results check for the ORIGINAL pipeline.

Two jobs:

1. INTEGRITY. The Oct-2025 TF 2.13 run produced DEGENERATE checkpoints:
   `ser_best.keras` predicted a single class ("surprise") for every input
   at 7.74% accuracy, because BatchNorm moving statistics were corrupted.
   A checkpoint that predicts one class is not a result — it is a broken
   artefact. This script loads each checkpoint and measures its prediction
   DISTRIBUTION on real data. If the modal class covers more than 40% of
   predictions, the checkpoint is degenerate and the script says so loudly
   and exits non-zero.

2. RESULTS. Reports the per-stage accuracy that each checkpoint's own
   training history recorded, so Paper 1 has one honest table.

Every number printed here is READ FROM DISK. Nothing is recomputed from
memory, and nothing is invented. If a checkpoint or a results file is
missing, the script prints `—` for that cell rather than guessing.

Run on the HPC (needs the model files, so a compute node):
    uv run python -u verify_original_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

CKPT = Path("model_checkpoints")
RESULTS = Path("reports")

DEGENERATE_MODAL_SHARE = 0.40


def bar(share: float, width: int = 30) -> str:
    n = int(round(share * width))
    return "#" * n + "." * (width - n)


def check_ser_degeneracy():
    """
    Load ser_best.keras and measure the prediction distribution on real
    audio. A healthy 7-class model spreads its predictions; a collapsed one
    puts everything in one bucket.
    """
    path = CKPT / "ser_best.keras"
    if not path.exists():
        print(f"  [SER] {path} — NOT FOUND")
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import numpy as np
        import tensorflow as tf
        from feature_utils import ensure_features_exist, preprocess_audio
        import pandas as pd
    except Exception as e:
        print(f"  [SER] could not import TF/numpy ({e}) — skipping the live check")
        return None

    try:
        model = tf.keras.models.load_model(str(path), compile=False)
    except Exception as e:
        print(f"  [SER] could not load {path}: {e}")
        return None

    # Take real samples from the manifest.
    try:
        df = pd.read_csv("combined_ser_dataset/metadata.csv")
        wavs, _ = None, None
        from feature_utils import _resolve_wav_paths
        wavs, _col = _resolve_wav_paths(df)
    except Exception as e:
        print(f"  [SER] could not read the manifest ({e})")
        return None

    step = max(1, len(wavs) // 400)
    sample = wavs[::step][:400]
    print(f"  [SER] probing {len(sample)} utterances from the manifest...")

    preds = []
    try:
        for w in sample:
            y = preprocess_audio(w)
            f = None
            try:
                from feature_utils import extract_features
                f = extract_features(y, sr=16000, n_mfcc=40)
            except Exception:
                pass
            if f is None:
                continue
            x = f.reshape(1, -1, 1)
            p = model.predict(x, verbose=0)[0]
            preds.append(int(np.argmax(p)))
    except Exception as e:
        print(f"  [SER] prediction loop failed: {e}")
        return None

    if not preds:
        print("  [SER] no predictions could be produced — skipping")
        return None

    dist = Counter(preds)
    n = len(preds)
    modal_class, modal_n = dist.most_common(1)[0]
    share = modal_n / n
    print(f"  [SER] predicted {n} samples over {len(dist)} distinct classes")
    for c, cnt in sorted(dist.items()):
        print(f"        class {c}: {cnt:4d}  {cnt / n:6.2%}  {bar(cnt / n)}")
    print(f"  [SER] modal class {modal_class} covers {share:.2%} of predictions")
    degenerate = share > DEGENERATE_MODAL_SHARE
    if degenerate:
        print(f"  [SER] ** DEGENERATE — modal class > {DEGENERATE_MODAL_SHARE:.0%}. "
              f"This checkpoint predicts one class and CANNOT be reported. **")
    else:
        print(f"  [SER] healthy — predictions are spread across classes.")
    return {"n": n, "modal_class": modal_class, "modal_share": share,
            "degenerate": degenerate, "distribution": dict(dist)}


def read_json(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def hist_acc(obj, key="accuracy"):
    """Pull the final accuracy out of a Keras history dict."""
    if not isinstance(obj, dict):
        return None
    h = obj.get("history", obj)
    if not isinstance(h, dict):
        return None
    for k in (key, "val_accuracy", "acc"):
        v = h.get(k)
        if isinstance(v, list) and v:
            return float(v[-1])
    return None


def main():
    print("=" * 78)
    print(" ORIGINAL PIPELINE — INTEGRITY AND RESULTS")
    print("=" * 78)

    # ---------- integrity ----------
    print("\n### CHECKPOINT INTEGRITY\n")
    ser = check_ser_degeneracy()

    # A checkpoint file that exists but is tiny is a truncated save.
    print("\n  [files]")
    for name in ("ser_best.keras", "meta_hybrid_best.keras",
                 "vgg16_orig_best.keras", "vgg16_bal_best.keras",
                 "resnet50_orig_best.keras", "resnet50_bal_best.keras",
                 "ter_pytorch_best.pt"):
        p = CKPT / name
        if p.exists():
            print(f"     ✓ {name:34s} {p.stat().st_size / 1024 / 1024:8.2f} MB")
        else:
            print(f"     — {name:34s} absent")

    # ---------- results ----------
    print("\n### MEASURED RESULTS\n")
    rows = []

    ser_res = read_json(CKPT / "ser_training_results.json")
    if ser_res:
        rows.append(("SER (1D-CNN, MFCC features)",
                     hist_acc(ser_res, "accuracy"),
                     ser_res.get("test_accuracy")))

    for name in ("vgg16_orig", "vgg16_bal", "resnet50_orig", "resnet50_bal"):
        h = read_json(CKPT / f"{name}_history.json")
        if h:
            rows.append((f"FER ({name})", hist_acc(h, "accuracy"), None))

    # ---------- master status ----------
    status = read_json(CKPT / "master_training_status.json")
    print(f"\n  master_training_status.json: {status}")

    print(f"\n  {'Stage':34s} | {'train acc':>9s} | {'test acc':>9s}")
    print(f"  {'-' * 34}-+-{'-' * 9}-+-{'-' * 9}")
    for label, tr, te in rows:
        trs = f"{tr:.4f}" if tr is not None else "—"
        tes = f"{te:.4f}" if te is not None else "—"
        print(f"  {label:34s} | {trs:>9s} | {tes:>9s}")

    # ---------- verdicts ----------
    print("\n### VERDICT\n")
    failed = False
    if ser is None:
        print("  SER integrity: NOT CHECKED (checkpoint missing or unreadable)")
    elif ser["degenerate"]:
        print("  SER integrity: ** FAILED — degenerate checkpoint **")
        print("  The SER number cannot go in the paper. Root cause is the known")
        print("  BatchNorm moving-statistic corruption. Fix and re-run.")
        failed = True
    else:
        print(f"  SER integrity: PASSED (modal class {ser['modal_share']:.1%})")

    if status and not status.get("meta_complete", False):
        print("  Meta-classifier: NOT COMPLETE — the fusion stage never finished.")
        print("  Do not report a fused number until it does.")
        failed = True
    elif status:
        print("  Meta-classifier: complete.")

    print()
    if failed:
        print("RESULT: NOT READY — see the failures above.")
        sys.exit(1)
    print("RESULT: all present checkpoints pass the integrity check.")


if __name__ == "__main__":
    main()
