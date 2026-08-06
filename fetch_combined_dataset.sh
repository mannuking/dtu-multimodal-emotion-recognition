#!/bin/bash
# fetch_combined_dataset.sh - Reassemble the full combined SER dataset from
# git-tracked chunks. Restores RAVDESS + CREMA + SAVEE + TESS (11,970 samples).
#
# Use this on HPC to recover the original 11k-sample dataset if combined_ser_dataset/
# was wiped (e.g. by an older build_combined_ser_dataset.py that didn't merge).
#
# Usage:
#   bash fetch_combined_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHUNKS_DIR="$SCRIPT_DIR/dataset_chunks"
TARGET_DIR="$SCRIPT_DIR/combined_ser_dataset"
TMP_TARBALL="/tmp/dtu_combined_ser_dataset.tar.gz"

# Verify all chunks are present
expected_chunks=(aa ab ac ad ae af ag ah ai aj ak)
for c in "${expected_chunks[@]}"; do
    if [ ! -f "$CHUNKS_DIR/dataset_chunk_$c" ]; then
        echo "ERROR: missing chunk $c. Run 'git pull origin main' first."
        exit 1
    fi
done

# Reassemble
echo "Reassembling combined_ser_dataset.tar.gz from $((${#expected_chunks[@]})) chunks..."
cat "$CHUNKS_DIR"/dataset_chunk_* > "$TMP_TARBALL"
size=$(stat -c%s "$TMP_TARBALL" 2>/dev/null || stat -f%z "$TMP_TARBALL")
echo "  \u2705 reassembled ($size bytes)"

# Backup any existing combined_ser_dataset
if [ -d "$TARGET_DIR" ]; then
    echo "Backing up existing combined_ser_dataset to combined_ser_dataset.bak..."
    mv "$TARGET_DIR" "$TARGET_DIR.bak.$(date +%s)"
fi

# Extract
echo "Extracting to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"
tar xzf "$TMP_TARBALL" -C "$SCRIPT_DIR/"
rm "$TMP_TARBALL"

# Verify
n=$(find "$TARGET_DIR" -name "*.wav" | wc -l | tr -d ' ')
echo "  \u2705 extracted $n wav files"

if [ "$n" -ge 11900 ]; then
    echo "  \u2705 dataset restored (expected ~11,970 samples)"
else
    echo "  WARNING: only $n files extracted (expected 11,970)"
fi

echo
echo "Now rebuild the manifest and retrain:"
echo "  uv run python build_combined_ser_dataset.py    # TESS-only manifest"
echo "  uv run python train_ser_wav2vec.py             # wav2vec2 SER on combined"