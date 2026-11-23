#!/bin/bash
#SBATCH --job-name=orig-pipeline
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=12:00:00
#SBATCH --output=orig_pipeline.%j.out
#SBATCH --error=orig_pipeline.%j.err
#SBATCH --account=dtuarp-acc
# ============================================================
# ORIGINAL (Oct-2025) DTU multimodal pipeline — full re-run.
#
# This is the ORIGINAL project: SER (1D-CNN on MFCC features) + FER
# (VGG16 / ResNet50 on FER2013) + TER (MobileBERT) + a late-fusion
# meta-classifier, over the COMBINED dataset
# (RAVDESS + CREMA-D + SAVEE + TESS, 7 classes, 11568 utterances).
#
# It is a different dataset and a different architecture family from the
# v9/v10/v11 IEMOCAP line. It is Paper 1. The v9+ line is Paper 2.
#
# ORDER MATTERS — each stage consumes the previous stage's checkpoint:
#   1. FER   -> vgg16_{orig,bal}_best.keras, resnet50_{orig,bal}_best.keras
#   2. SER   -> ser_best.keras
#   3. TER   -> ter_pytorch_best.pt + ter_pytorch_tokenizer/
#   4. META  -> meta_hybrid_best.keras   (fuses all three)
#
# The whole thing is resumable: each trainer skips itself if its
# checkpoint already exists, and combined_dtu.py records progress in
# model_checkpoints/master_training_status.json.
# ============================================================

set -euo pipefail

# ---------------- login-node guard ----------------
# The cluster's login node is for compilation and job submission ONLY.
# Running compute there risks silent kills and an account-suspension
# report. This script therefore REFUSES to run outside a SLURM job.
# The check is on the environment, not on a flag: nothing can bypass it
# by accident, and no one can run this on the login node by mistake.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo ""
  echo "FATAL: this script must be launched with sbatch."
  echo "       The login node is for compilation and job submission only."
  echo "       Running compute here risks silent kills and an account-suspension report."
  echo ""
  echo "       Use:  sbatch $(basename "$0")"
  exit 1
fi
if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "FATAL: SLURM_JOB_NODELIST is empty — refusing to run outside a compute node."
  exit 1
fi
echo "  node: $(hostname)   job: ${SLURM_JOB_ID}"

# Only now is it safe to touch SLURM_SUBMIT_DIR.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}"


echo "============================================================"
echo " ORIGINAL PIPELINE — job started: $(date)"
echo " Node: $(hostname)   JobID: $SLURM_JOB_ID"
echo " GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed — continuing with local checkout)"

# ---------------- environment check ----------------
# A `uv sync` that fails can UNINSTALL the packages it would have added.
# Never start training on an environment that cannot import TensorFlow.
echo ""
echo "--- ENVIRONMENT ---"
uv run python - <<'PYEOF' || { echo "FATAL: the venv is missing required packages."; \
  echo "       Run: uv sync --extra cuda   (do NOT fall back to a bare uv sync)"; exit 1; }
import importlib, sys
need = ["tensorflow", "keras", "torch", "transformers", "librosa", "sklearn",
        "pandas", "numpy", "PIL", "soundfile", "datasets", "huggingface_hub"]
bad = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception:
        bad.append(m); print(f"  --  {m:16s} MISSING")
if bad:
    print(f"  MISSING: {', '.join(bad)}"); sys.exit(1)
import tensorflow as tf
ngpu = len(tf.config.list_physical_devices("GPU"))
print(f"  TF GPUs visible on this node: {ngpu}")
# A CPU-only run would take days and would be a policy risk on this cluster.
# Refuse rather than silently degrade.
if ngpu == 0:
    print("FATAL: no GPU visible on this node.")
    print("       This job MUST run on a GPU node. Do not run it on CPU.")
    sys.exit(1)
PYEOF

# ---------------- preflight ----------------
echo ""
echo "--- PREFLIGHT ---"
echo "repo:        $(pwd)"
echo "python:      $(uv run python -V 2>&1)"
echo "dataset dir: $(ls -d combined_ser_dataset 2>/dev/null || echo MISSING)"
echo "metadata:    $([ -f combined_ser_dataset/metadata.csv ] && echo present || echo MISSING)"
echo "wav count:   $(find combined_ser_dataset -name '*.wav' 2>/dev/null | wc -l)"
echo "feature npy: $([ -f ser_feature_output/features.npy ] && echo present || echo 'absent (will extract)')"
echo "labels npy:  $([ -f ser_feature_output/labels.npy ] && echo present || echo 'absent (will extract)')"
echo "fer2013:     $([ -d fer2013/train ] && echo present || echo 'absent (will download)')"
echo "ter csvs:    $([ -f ter_dataset/merged/text_train.csv ] && echo present || echo 'absent (will build)')"

# ---------------- pre-flight: the data contract ----------------
# Everything below checks a contract that, if violated, produces a
# plausible-looking but meaningless number rather than an error.
echo ""
echo "--- PRE-FLIGHT ---"
uv run python -u - <<'PYEOF' || { echo "FATAL: pre-flight failed — refusing to train"; exit 1; }
import os, sys, csv, traceback
import numpy as np
sys.path.insert(0, os.getcwd())
ok = True

def check(label, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        ok = False

def section(name):
    print(f"  [{name}]")

# 1. The manifest and the audio must agree.
mp = "combined_ser_dataset/metadata.csv"
rows = []
if os.path.exists(mp):
    with open(mp, newline="") as f:
        rows = list(csv.DictReader(f))
    check("manifest readable", len(rows) > 0, f"{len(rows)} rows")
    col = next((c for c in ("wav_path", "filepath", "audio_path") if rows and c in rows[0]), None)
    check("manifest has a path column", col is not None, f"'{col}'")
    if col:
        try:
            from feature_utils import _resolve_wav_paths
            import pandas as pd
            df = pd.read_csv(mp)
            paths, _ = _resolve_wav_paths(df)
            missing = sum(1 for p in paths[:400] if not os.path.exists(p))
            check("manifest audio files resolve", missing == 0,
                  f"{400 - missing}/400 present")
        except Exception as e:
            check("manifest audio files resolve", False, f"{type(e).__name__}: {e}"[:70])
else:
    check("manifest exists", False, mp)

# 2. The SER feature cache must match the manifest, or be regenerated.
fs = "ser_feature_output/features.npy"
ls_ = "ser_feature_output/labels.npy"
if os.path.exists(fs) and os.path.exists(ls_):
    try:
        l = np.load(ls_, allow_pickle=True)
        n_exp = len(rows) if 'rows' in dir() else 0
        check("SER label cache is numeric", l.dtype.kind in "iuf",
              f"dtype={l.dtype}")
        check("SER label cache row count matches manifest",
              (n_exp == 0) or (len(l) == n_exp), f"{len(l)} vs {n_exp}")
    except Exception as e:
        check("SER label cache readable", False, f"{type(e).__name__}: {e}"[:70])
else:
    print("  INFO  no SER feature cache — it will be extracted from the audio")

# 3. The triplet manifest's audio paths must resolve, or the meta-classifier
#    trains on zero vectors.
tp = "triplets_manifest.csv"
if os.path.exists(tp):
    with open(tp, newline="") as f:
        trows = list(csv.DictReader(f))
    bad = 0
    for r in trows[:300]:
        w = r.get("speech_wav", "")
        cands = [w, os.path.join("combined_ser_dataset", w),
                 os.path.join("combined_ser_dataset", "combined_ser_dataset", w)]
        if not any(c and os.path.exists(c) for c in cands):
            bad += 1
    check("triplet audio paths resolve", bad == 0, f"{300 - bad}/300 present")
else:
    print("  INFO  no triplets manifest — it will be built")

# 4. fer2013 must be the REAL dataset (28k images), not the synthetic one.
n_imgs = sum(len(fs_) for _, _, fs_ in os.walk("fer2013/train")) if os.path.isdir("fer2013/train") else 0
if n_imgs:
    check("fer2013 is the real dataset", n_imgs > 20000, f"{n_imgs} images")
else:
    print("  INFO  fer2013 not unpacked — it will be downloaded")

print()
print("  PRE-FLIGHT:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PYEOF

# ---------------- stage 0: data preparation ----------------
# Everything below is CPU-only and cheap; it belongs in the same job so a
# fresh node can go from bare checkout to trained models unattended.
echo ""
echo "============================================================"
echo " STAGE 0 — data preparation (CPU)"
echo "============================================================"

echo "[0a] Building the text corpus for TER..."
uv run python -u prepare_ter_dataset.py || {
  echo "FATAL: TER dataset preparation failed"; exit 1; }

# The corpus must be the real one. The synthetic fallback is 7 sentences
# x 7 classes; the real corpus is >80000 utterances. A corpus smaller
# than 20000 rows means the download failed and the builder is about to
# be bypassed -- refuse rather than train on 7 constant sentences.
uv run python -u -c "
import csv, os, sys
p = 'ter_dataset/merged/text_train.csv'
if not os.path.exists(p):
    print('FATAL: no TER training corpus at', p); sys.exit(1)
with open(p, newline='') as f:
    n = sum(1 for _ in csv.reader(f)) - 1
print(f'  TER corpus rows: {n}')
if n < 20000:
    print('FATAL: TER corpus is too small — this is the synthetic fallback,')
    print('       not real text. TER results would be meaningless.')
    sys.exit(1)
print('  TER corpus looks real.')
" || { echo "FATAL: TER corpus is not the real dataset — stopping"; exit 1; }

# The Oct-2025 TER checkpoint was trained on the SYNTHETIC corpus. It must
# not survive into this run, or train_ter_pytorch.py will skip Stage 3
# ("TER model already trained!") and the meta-classifier will fuse in a
# model that never saw real text.
if [ -f model_checkpoints/ter_pytorch_best.pt ]; then
  echo "  removing the stale synthetic-corpus TER checkpoint"
  rm -f model_checkpoints/ter_pytorch_best.pt
  rm -rf model_checkpoints/ter_pytorch_tokenizer/
fi

echo "[0b] Building the triplets manifest for the meta-classifier..."
uv run python -u build_triplets_leakage_free.py || {
  echo "FATAL: triplets manifest build failed"; exit 1; }

echo "[0c] Extracting SER audio features (this is the slow CPU step, ~30-60 min)..."
uv run python -u -c "
from feature_utils import ensure_features_exist
import os
f, l = ensure_features_exist(
    features_path=os.path.join('ser_feature_output','features.npy'),
    labels_path=os.path.join('ser_feature_output','labels.npy'),
    manifest_csv=os.path.join('combined_ser_dataset','metadata.csv'),
)
print('SER features ready:', f.shape, 'labels:', len(l))
" || { echo "FATAL: SER feature extraction failed"; exit 1; }

echo "[0d] Downloading FER2013..."
# fer2013_download.py skips the download if the train dir already has >= 7
# subdirectories. A partial or synthetic directory satisfies that check, so
# the download would be skipped and the trainer would silently train on
# garbage. Force the download and verify the real image count afterwards.
uv run python -u -c "
from fer2013_download import download_and_unpack
download_and_unpack(force=True)
" || echo "  WARNING: FER2013 download failed — do NOT accept a synthetic fallback"

# Verify FER2013 is REAL. The synthetic fallback writes 200 PNGs per class;
# the real dataset has ~28700 train images. 1500 images == synthetic.
uv run python -u -c "
import os, sys
n = sum(len(fs) for _, _, fs in os.walk('fer2013/train'))
print(f'  FER2013 train images: {n}')
if n < 5000:
    print('  FATAL: fer2013/train has too few images — this is the SYNTHETIC')
    print('         fallback, not the real dataset. FER results would be')
    print('         meaningless. Fix the download and re-run.')
    sys.exit(1)
print('  FER2013 looks real.')
" || { echo "FATAL: FER2013 is not the real dataset — stopping before training"; exit 1; }

# ---------------- stage 1: FER ----------------
echo ""
echo "============================================================"
echo " STAGE 1 — FER (VGG16 + ResNet50 on FER2013)"
echo "============================================================"
uv run python -u train_fer_tensorflow.py 2>&1 | tee -a orig_fer.log

# ---------------- stage 2: SER ----------------
echo ""
echo "============================================================"
echo " STAGE 2 — SER (1D-CNN on MFCC features)"
echo "============================================================"
uv run python -u train_ser_tensorflow.py 2>&1 | tee -a orig_ser.log

# ---------------- stage 3: TER ----------------
echo ""
echo "============================================================"
echo " STAGE 3 — TER (MobileBERT)"
echo "============================================================"
uv run python -u train_ter_pytorch.py 2>&1 | tee -a orig_ter.log

# ---------------- stage 4: meta-classifier ----------------
echo ""
echo "============================================================"
echo " STAGE 4 — META late-fusion classifier"
echo "============================================================"

# The visual channel may still be empty. That is a reportable state, not
# a reason to fail — but it MUST be reported, and the meta-classifier now
# prints the coverage itself.
if [ -f triplets_manifest.csv ]; then
  # tail -n +2 drops the header row, so the counts are real data rows.
  n_rows=$(($(tail -n +2 triplets_manifest.csv | wc -l)))
  n_real=$(tail -n +2 triplets_manifest.csv | grep -c -v "placeholder" || true)
  n_real=${n_real:-0}
  echo "  triplets rows: $n_rows   with a real face_img: $n_real"
  if [ "$n_real" -lt $((n_rows / 2)) ]; then
    echo "  ⚠️  visual channel under 50% — the fusion is mostly audio+text."
    echo "     Run scripts/extract_fer_frames.sbatch and re-run this stage"
    echo "     to get a genuine three-modality fusion."
  fi
else
  echo "  ⚠️  no triplets manifest — the meta stage will build it."
fi
uv run python -u train_meta_classifier.py 2>&1 | tee -a orig_meta.log

# ---------------- verification ----------------
echo ""
echo "============================================================"
echo " VERIFICATION — the numbers, and the integrity check"
echo "============================================================"
uv run python -u verify_original_pipeline.py 2>&1 | tee -a orig_verify.log

echo ""
echo "============================================================"
echo " ORIGINAL PIPELINE finished: $(date)"
echo "============================================================"
