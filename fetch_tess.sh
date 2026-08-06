#!/bin/bash
# fetch_tess.sh - Download TESS dataset chunks from the repo via git pull.
#
# The repo has 5 chunks (~89 MB each, ~428 MB total) at tess_chunks/tess_part_*
# Git pull from ssh.github.com:443 works on HPC login (the only blocker was
# releases/download URLs blocked by squid).
#
# Usage:
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   bash fetch_tess.sh
set -e

EXPECTED_SIZE=448572034

# Step 1: git pull (fetches tess_chunks/tess_part_*)
echo "Pulling TESS chunks from repo (git fetch)..."
git fetch origin main 2>&1 | tail -3 || true
git checkout origin/main -- tess_chunks/ 2>&1 | tail -3 || true

# Step 2: Verify all 5 chunks present
for chunk in tess_chunks/tess_part_aa tess_chunks/tess_part_ab tess_chunks/tess_part_ac tess_chunks/tess_part_ad tess_chunks/tess_part_ae; do
  if [[ ! -f "$chunk" ]]; then
    echo "ERROR: missing chunk $chunk"
    exit 1
  fi
done
echo "✓ All 5 chunks present"

# Step 3: Reassemble
echo "Reassembling tess_kaggle.zip..."
cat tess_chunks/tess_part_aa tess_chunks/tess_part_ab tess_chunks/tess_part_ac \
    tess_chunks/tess_part_ad tess_chunks/tess_part_ae \
    > tess_kaggle.zip

actual_size=$(stat -f%z tess_kaggle.zip 2>/dev/null || stat -c%s tess_kaggle.zip)
if [[ "${actual_size}" != "${EXPECTED_SIZE}" ]]; then
  echo "ERROR: tess_kaggle.zip size ${actual_size} != expected ${EXPECTED_SIZE}"
  exit 1
fi
echo "✓ tess_kaggle.zip verified (${actual_size} bytes)"

# Step 4: Extract
echo "Extracting..."
unzip -q tess_kaggle.zip -d tess_extracted/

# Step 5: Move wav files into the structure build_combined_ser_dataset.py expects
mkdir -p tess_wavs
if [[ -d "tess_extracted/TESS Toronto emotional speech set data" ]]; then
  cp -r "tess_extracted/TESS Toronto emotional speech set data/"* tess_wavs/
else
  cp -r tess_extracted/*/ tess_wavs/ 2>/dev/null || {
    echo "ERROR: could not find TESS data folder inside tess_extracted/"
    exit 1
  }
fi

# Cleanup intermediate (keep tess_chunks/ for future re-runs)
rm -rf tess_extracted

echo "✓ TESS data ready in tess_wavs/"
echo ""
echo "Now run: uv run python build_combined_ser_dataset.py"