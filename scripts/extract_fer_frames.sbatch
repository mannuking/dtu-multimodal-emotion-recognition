#!/bin/bash
#SBATCH --job-name=orig-fer-frames
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=04:00:00
#SBATCH --output=orig_fer_frames.%j.out
#SBATCH --error=orig_fer_frames.%j.err
#SBATCH --account=dtuarp-acc
# ============================================================
# Extract the REAL visual channel for the ORIGINAL pipeline.
#
# The triplets manifest shipped with face_img = placeholder_N for all
# 7000 rows, and there are no face directories anywhere on disk. The
# meta-classifier therefore fused a ZERO vector for the visual
# modality — the "multimodal" fusion was audio+text wearing a
# multimodal label.
#
# CREMA-D (7441 of the 11568 combined-corpus utterances) ships FLV
# video alongside the WAV on the official GitLab mirror. This job
# downloads it through the git-lfs batch API, decodes frames with
# ffmpeg, and rewrites both manifests so face_img points at a real
# frame.
#
# This is CPU + network bound; it asks for a GPU only so it can share
# the node allocation. It does NOT touch a checkpoint.
# ============================================================

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo " FER FRAME EXTRACTION — job started: $(date)"
echo " Node: $(hostname)  JobID: $SLURM_JOB_ID"
echo "============================================================"

git pull origin main || echo "  (git pull failed — continuing with local checkout)"

# ---------------- preflight ----------------
echo ""
echo "--- PREFLIGHT ---"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "  ffmpeg not found. Installing a static build..."
  # conda-forge ffmpeg is the most reliable source on a cluster with conda
  if command -v conda >/dev/null 2>&1; then
    conda install -y -c conda-forge ffmpeg || echo "  (conda install failed)"
  fi
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "  FATAL: ffmpeg is required to decode the .flv clips."
  echo "         Install it (conda install -c conda-forge ffmpeg) and resubmit."
  exit 1
fi
echo "  ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"

uv run python -u - <<'PYEOF' || { echo "FATAL: environment check failed"; exit 1; }
import importlib, sys
for m in ("requests", "PIL", "numpy", "pandas"):
    try:
        importlib.import_module(m); print(f"  OK  {m}")
    except Exception as e:
        print(f"  --  {m} MISSING"); sys.exit(1)
PYEOF

# ---------------- extraction ----------------
echo ""
echo "--- EXTRACTING ---"
# Smoke test first (4 clips) so a broken network or a missing ffmpeg is
# caught in seconds rather than after an hour of downloads.
uv run python -u extract_fer_frames.py --limit 4 --workers 4 || {
  echo "FATAL: the smoke test failed — not running the full extraction."
  exit 1
}

echo ""
echo "--- FULL EXTRACTION ---"
uv run python -u extract_fer_frames.py --workers 8 2>&1 | tee -a orig_fer_frames.log

# ---------------- verification ----------------
echo ""
echo "--- VERIFICATION ---"
uv run python -u - <<'PYEOF' || { echo "FATAL: frame verification failed"; exit 1; }
import csv, os, sys
from collections import Counter

mp = "fer_frames/manifest.csv"
if not os.path.exists(mp):
    print("FATAL: no fer_frames/manifest.csv"); sys.exit(1)
rows = list(csv.DictReader(open(mp)))
by_sample = Counter(r["sample_id"] for r in rows)
print(f"  clips with frames: {len(by_sample)}")
print(f"  total frames:      {len(rows)}")
if not by_sample:
    print("FATAL: no frames were extracted"); sys.exit(1)

# every referenced frame must exist on disk
missing = sum(1 for r in rows[:2000] if not os.path.exists(r["frame_path"]))
print(f"  frame paths resolve: {2000 - missing}/2000 sampled")
if missing:
    print("FATAL: extracted frame paths do not exist"); sys.exit(1)

# the corpus manifest must now carry face_img for a real share of rows
cm = "combined_ser_dataset/metadata.csv"
crows = list(csv.DictReader(open(cm)))
withface = sum(1 for r in crows if r.get("face_img"))
cov = withface / len(crows) * 100
print(f"  corpus rows with a face_img: {withface}/{len(crows)} ({cov:.1f}%)")

tm = "triplets_manifest.csv"
trows = list(csv.DictReader(open(tm)))
tface = sum(1 for r in trows if r.get("face_img") and not r["face_img"].startswith("placeholder"))
print(f"  triplets rows with a real face_img: {tface}/{len(trows)}")

print()
if cov < 10:
    print("RESULT: visual channel is still effectively empty.")
    sys.exit(1)
print(f"RESULT: visual channel is real ({cov:.1f}% of the corpus).")
PYEOF

echo ""
echo "============================================================"
echo " FER FRAME EXTRACTION finished: $(date)"
echo "============================================================"
