#!/bin/bash
# fetch_tess.sh - Download and reassemble TESS dataset on HPC.
#
# GitHub Releases sometimes 403 from login node proxies. Workaround:
#   - Download in 9 chunks of 50 MB each via curl with retries
#   - Reassemble with cat
#   - Verify checksum against expected size
#
# Usage:
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   bash fetch_tess.sh
set -e

REPO="mannuking/dtu-multimodal-emotion-recognition"
TAG="v1.0-tess-dataset"
CHUNK_BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
TOTAL_PARTS=9
EXPECTED_SIZE=448572034

mkdir -p tess_chunks
cd tess_chunks

# Download chunks (with retries)
for i in aa ab ac ad ae af ag ah ai; do
  fname="tess_part_${i}"
  if [[ -f "${fname}" ]] && [[ $(stat -f%z "${fname}" 2>/dev/null || stat -c%s "${fname}") -gt 0 ]]; then
    echo "✓ ${fname} already present"
    continue
  fi
  echo "↓ downloading ${fname}..."
  curl -fsSL --retry 5 --retry-delay 3 --max-time 300 \
    -o "${fname}" \
    "${CHUNK_BASE_URL}/${fname}"
  echo "  → size: $(stat -f%z "${fname}" 2>/dev/null || stat -c%s "${fname}") bytes"
done

cd ..

# Reassemble
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
  # Fallback: maybe the directory has a slightly different name
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
