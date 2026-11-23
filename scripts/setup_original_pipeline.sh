#!/bin/bash
# ============================================================
# setup_original_pipeline.sh — prepare the ORIGINAL (Paper 1) run on the HPC.
#
# Run this on the LOGIN node. It does no compute — it only:
#   1. syncs the repo
#   2. installs the uv environment (CPU-light, but needs network)
#   3. verifies the dataset is in place
#   4. prints exactly what to submit next
#
# The login node has no GPU and must not train. Training happens via
# scripts/train_original_pipeline.sbatch.
# ============================================================
set -euo pipefail

REPO="$HOME/Research/dtu/dtu-multimodal-emotion-recognition"
cd "$REPO"

echo "============================================================"
echo " ORIGINAL PIPELINE SETUP — $(date)"
echo "============================================================"

# ---- 1. sync ----
echo ""
echo "[1/5] Syncing the repo..."
git pull origin main
git log --oneline -1

# ---- 2. environment ----
echo ""
echo "[2/5] Installing uv (if absent)..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

echo ""
echo "[3/5] Syncing the Python environment (this can take a few minutes)..."
# The cuda extra pulls tensorflow 2.15.1 + the CUDA wheels. On the login node
# this is a download + unpack only; no CUDA runtime is touched.
#
# Do NOT add --extra docs here. The docs extra pulls weasyprint -> fonttools
# -> brotli, and brotli's wheel frequently fails to download behind the
# cluster's HTTP proxy ("tunnel error: unsuccessful"). A failed `uv sync`
# with an extra UNINSTALLS the packages it would have added -- so the
# fallback path silently removed TensorFlow, keras and tf_keras from the
# venv, leaving an environment that cannot train anything.
#
# The base env has no TensorFlow at all, so a fallback to plain `uv sync`
# is worse than failing loudly. Install the cuda extra and verify.
if ! uv sync --extra cuda; then
  echo ""
  echo "FATAL: 'uv sync --extra cuda' failed."
  echo "       Do NOT fall back to a bare 'uv sync' — that removes TensorFlow."
  echo "       If the failure is a download error, retry; if it is a proxy"
  echo "       error, check the cluster's egress rules for files.pythonhosted.org."
  exit 1
fi
uv run python -V

# Verify the packages the pipeline actually imports are present.
echo ""
echo "  verifying the runtime dependencies..."
uv run python - <<'PYEOF' || { echo "FATAL: required packages missing from the venv"; exit 1; }
import importlib, sys
need = ["tensorflow", "keras", "torch", "transformers", "librosa", "sklearn",
        "pandas", "numpy", "PIL", "soundfile"]
bad = []
for m in need:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"    OK  {m:14s} {v}")
    except Exception as e:
        bad.append(m)
        print(f"    --  {m:14s} MISSING ({type(e).__name__})")
if bad:
    print(f"\n  MISSING: {', '.join(bad)}")
    sys.exit(1)
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"    tensorflow sees {len(gpus)} GPU(s) on the login node")
print("    (0 on the login node is expected — GPUs appear on compute nodes)")
PYEOF

# ---- 3. dataset ----
echo ""
echo "[4/5] Checking the dataset..."
echo "  repo dir:      $(pwd)"
if [ -d combined_ser_dataset ]; then
  echo "  ✓ combined_ser_dataset/ present"
  echo "    wav files:   $(find combined_ser_dataset -name '*.wav' | wc -l)"
  echo "    metadata:    $([ -f combined_ser_dataset/metadata.csv ] && echo present || echo MISSING)"
else
  echo "  ✗ combined_ser_dataset/ MISSING in the repo"
  echo ""
  echo "  The dataset lives in one of two places on this cluster:"
  echo "    A) ~/Research/dtu/combined_ser_dataset/   (sibling of the repo)"
  echo "    B) the tarball  ~/Research/dtu/dtu_ser_dataset_v1.tar.gz"
  echo ""
  echo "  Link it in with:"
  echo "    ln -s ~/Research/dtu/combined_ser_dataset combined_ser_dataset"
  echo ""
  echo "  Or extract the tarball:"
  echo "    tar -xzf ~/Research/dtu/dtu_ser_dataset_v1.tar.gz -C ."
fi

# The tarball may hold the dataset under a different root name.
if [ ! -d combined_ser_dataset ] && [ -f dtu_ser_dataset_v1.tar.gz ]; then
  echo "  attempting tarball extraction..."
  tar -xzf dtu_ser_dataset_v1.tar.gz -C . || true
fi

if [ -d combined_ser_dataset ]; then
  echo "  ✓ after setup: $(find combined_ser_dataset -name '*.wav' | wc -l) wav files"
else
  echo "  ✗ still missing — run the ln -s or tar command above, then re-run this script"
fi

# ---- 4. what is already there ----
echo ""
echo "[5/5] Current state of the artefacts this pipeline produces..."
for f in \
  ser_feature_output/features.npy \
  ser_feature_output/labels.npy \
  model_checkpoints/ser_best.keras \
  model_checkpoints/vgg16_orig_best.keras \
  model_checkpoints/vgg16_bal_best.keras \
  model_checkpoints/resnet50_orig_best.keras \
  model_checkpoints/resnet50_bal_best.keras \
  model_checkpoints/ter_pytorch_best.pt \
  model_checkpoints/meta_hybrid_best.keras \
  fer2013/train \
  ter_dataset/merged/text_train.csv
do
  if [ -e "$f" ]; then
    printf "    ✓ %-44s %s\n" "$f" "$(du -sh "$f" 2>/dev/null | cut -f1)"
  else
    printf "    — %-44s (will be built)\n" "$f"
  fi
done

echo ""
echo "============================================================"
echo " SETUP COMPLETE"
echo "============================================================"
echo ""
echo "NEXT — submit the pipeline (this is the only command that trains):"
echo ""
echo "    cd ~/Research/dtu/dtu-multimodal-emotion-recognition"
echo "    sbatch scripts/train_original_pipeline.sbatch"
echo ""
echo "Then watch it:"
echo ""
echo "    tail -f orig_pipeline.<JOBID>.out"
echo ""
echo "NOTE: the run has four ordered stages — FER, then SER, then TER, then"
echo "the META fusion — and each needs the previous one's checkpoint. The"
echo "whole thing is 6-12 GPU-hours; the SER feature extraction is CPU-bound"
echo "and is the slowest single step."
echo ""
echo "If it fails, paste me the tail of orig_pipeline.<JOBID>.err and the"
echo "last 40 lines of orig_pipeline.<JOBID>.out."
