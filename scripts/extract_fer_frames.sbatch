#!/bin/bash
#SBATCH --job-name=orig-fer-frames
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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
# This is CPU + network bound, so it asks for NO GPU. That matters: the
# dgxnp queue is backed up (the training job sat in PD for 25 minutes),
# and a GPU request would put this job BEHIND that queue for no reason.
# Running it without a GPU lets it start immediately in parallel.
# It does NOT touch a checkpoint and does NOT touch the project venv.
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
FFMPEG_BIN=""
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_BIN="$(command -v ffmpeg)"
  echo "  ffmpeg: system binary at $FFMPEG_BIN"
else
  # Fallback 1: imageio-ffmpeg ships a self-contained static ffmpeg. Run it
  # with --no-project so it lands in a THROWAWAY environment: it cannot
  # touch the project venv the training job is using.
  echo "  ffmpeg: not on PATH — trying imageio-ffmpeg (isolated env)..."
  FFMPEG_BIN="$(uv run --no-project --with imageio-ffmpeg python -c \
      'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  if [ -n "$FFMPEG_BIN" ] && [ -x "$FFMPEG_BIN" ]; then
    echo "  ffmpeg: static binary at $FFMPEG_BIN"
  else
    FFMPEG_BIN=""
  fi
fi
if [ -z "$FFMPEG_BIN" ]; then
  # Fallback 2: conda-forge. Last resort because conda writes to a shared
  # filesystem and can be slow/contended.
  if command -v conda >/dev/null 2>&1; then
    echo "  ffmpeg: trying conda-forge..."
    conda install -y -c conda-forge ffmpeg || echo "  (conda install failed)"
    if command -v ffmpeg >/dev/null 2>&1; then
      FFMPEG_BIN="$(command -v ffmpeg)"
    fi
  fi
fi
if [ -z "$FFMPEG_BIN" ]; then
  echo "  FATAL: ffmpeg is required to decode the .flv clips."
  echo "         Tried: system PATH, imageio-ffmpeg, conda-forge."
  exit 1
fi
export FFMPEG_BIN
echo "  ffmpeg version: $("$FFMPEG_BIN" -version 2>/dev/null | head -1)"

# The extractor is PURE STDLIB + the ffmpeg binary. It deliberately does NOT
# use `uv run`: that would share the project venv with the training job, and
# a `uv run` sync can prune packages out of that venv mid-training (the
# exact failure that cost us TensorFlow once already). Use `uv run --no-sync`
# so the venv is read but never modified.
uv run --no-sync python -u - <<'PYEOF' || { echo "FATAL: environment check failed"; exit 1; }
import sys
assert sys.version_info >= (3, 11), f"need 3.11+, got {sys.version_info}"
for m in ("urllib.request", "json", "csv", "subprocess", "concurrent.futures"):
    __import__(m)
print(f"  OK  python {sys.version.split()[0]} (stdlib only)")
print("  OK  venv untouched (--no-sync)")
PYEOF

# ---------------- extraction ----------------
echo ""
echo "--- EXTRACTING ---"
# Smoke test first (4 clips) so a broken network or a missing ffmpeg is
# caught in seconds rather than after an hour of downloads.
uv run --no-sync python -u extract_fer_frames.py --limit 4 --workers 4 || {
  echo "FATAL: the smoke test failed — not running the full extraction."
  exit 1
}

echo ""
echo "--- FULL EXTRACTION ---"
uv run --no-sync python -u extract_fer_frames.py --workers 8 2>&1 | tee -a orig_fer_frames.log

# ---------------- verification ----------------
echo ""
echo "--- VERIFICATION ---"
uv run --no-sync python -u - <<'PYEOF' || { echo "FATAL: frame verification failed"; exit 1; }
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
