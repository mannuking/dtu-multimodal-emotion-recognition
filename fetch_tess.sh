#!/bin/bash
# fetch_tess.sh - Download and reassemble TESS dataset on HPC using gh CLI.
#
# The HPC login node's squid proxy blocks GitHub release download URLs
# (/releases/download/) but the GitHub API (api.github.com) is reachable.
# Solution: use `gh release download` which uses the API.
#
# Usage:
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   bash fetch_tess.sh
set -e

REPO="mannuking/dtu-multimodal-emotion-recognition"
TAG="v1.0-tess-dataset"
EXPECTED_SIZE=448572034

# Use gh if available; else try curl with API URL pattern
if command -v gh &>/dev/null; then
  echo "Using gh CLI to download release assets..."
  gh release download "$TAG" --repo "$REPO" --pattern 'tess_part_*' --dir tess_chunks
else
  echo "gh CLI not found — falling back to API direct download"
  # Get asset URLs via API
  mkdir -p tess_chunks
  cd tess_chunks
  for i in aa ab ac ad ae af ag ah ai; do
    fname="tess_part_${i}"
    if [[ -f "${fname}" ]] && [[ $(stat -f%z "${fname}" 2>/dev/null || stat -c%s "${fname}") -gt 0 ]]; then
      echo "✓ ${fname} already present"
      continue
    fi
    # Get asset URL from API
    url=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" \
      | python3 -c "import sys,json; r=json.load(sys.stdin); [print(a['browser_download_url']) for a in r['assets'] if a['name']=='${fname}']")
    if [[ -z "$url" ]]; then
      echo "ERROR: could not find download URL for ${fname}"
      exit 1
    fi
    echo "↓ downloading ${fname}..."
    curl -fsSL --retry 5 --retry-delay 3 --max-time 300 -o "${fname}" "$url"
  done
  cd ..
fi

echo "Reassembling tess_kaggle.zip..."
cat tess_chunks/tess_part_aa tess_chunks/tess_part_ab tess_chunks/tess_part_ac \
    tess_chunks/tess_part_ad tess_chunks/tess_part_ae tess_chunks/tess_part_af \
    tess_chunks/tess_part_ag tess_chunks/tess_part_ah tess_chunks/tess_part_ai \
    > tess_kaggle.zip

# Verify
actual_size=$(stat -f%z tess_kaggle.zip 2>/dev/null || stat -c%s tess_kaggle.zip)
if [[ "${actual_size}" != "${EXPECTED_SIZE}" ]]; then
  echo "ERROR: tess_kaggle.zip size ${actual_size} != expected ${EXPECTED_SIZE}"
  exit 1
fi
echo "✓ tess_kaggle.zip verified (${actual_size} bytes)"

# Extract
echo "Extracting..."
unzip -q tess_kaggle.zip -d tess_extracted/

# Move wav files into the structure build_combined_ser_dataset.py expects
mkdir -p tess_wavs
if [[ -d "tess_extracted/TESS Toronto emotional speech set data" ]]; then
  cp -r "tess_extracted/TESS Toronto emotional speech set data/"* tess_wavs/
else
  cp -r tess_extracted/*/ tess_wavs/ 2>/dev/null || {
    echo "ERROR: could not find TESS data folder inside tess_extracted/"
    exit 1
  }
fi

# Cleanup intermediate
rm -rf tess_extracted

echo "✓ TESS data ready in tess_wavs/"
echo ""
echo "Now run: uv run python build_combined_ser_dataset.py"