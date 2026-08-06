#!/bin/bash
# fetch_wav2vec2.sh — Reassemble wav2vec2-base model from repo chunks.
#
# The HPC login node's proxy blocks huggingface.co. We bundle wav2vec2-base
# into git-friendly chunks (~95MB each) and reassemble locally. Run BEFORE
# train_ser_wav2vec.py so the model loads from a local directory.
#
# Usage:
#   bash fetch_wav2vec2.sh

set -euo pipefail

CACHE_DIR="$HOME/.cache/huggingface/hub/models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
CHUNKS_DIR="wav2vec2_chunks"
mkdir -p "$CACHE_DIR"

# Reassemble pytorch_model.bin from chunks
if [ ! -f "$CACHE_DIR/pytorch_model.bin" ]; then
    echo "Reassembling pytorch_model.bin from chunks..."
    cat "$CHUNKS_DIR"/pytorch_chunk_* > "$CACHE_DIR/pytorch_model.bin"
    # Sanity check size
    size=$(stat -c%s "$CACHE_DIR/pytorch_model.bin" 2>/dev/null || stat -f%z "$CACHE_DIR/pytorch_model.bin")
    if [ "$size" -ne 380220182 ]; then
        echo "WARNING: reassembled size $size != expected 380220182"
    else
        echo "  \u2705 pytorch_model.bin reassembled ($size bytes)"
    fi
fi

# Copy config.json
if [ ! -f "$CACHE_DIR/config.json" ]; then
    cp "$CHUNKS_DIR/config.json" "$CACHE_DIR/config.json"
    echo "  \u2705 config.json copied"
fi

# Set HF_HUB_OFFLINE so transformers uses the cached version
echo
echo "wav2vec2-base ready at: $CACHE_DIR"
echo "Now run with HF_HUB_OFFLINE=1 to use the local cache:"
echo "  HF_HUB_OFFLINE=1 uv run python train_ser_wav2vec.py"