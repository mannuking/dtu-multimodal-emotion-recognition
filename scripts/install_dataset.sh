#!/usr/bin/env bash
# install_dataset.sh — One-command dataset install for the HPC.
#
# Downloads dtu_ser_dataset_v1.tar.gz from the GitHub Release, verifies SHA256,
# extracts to ./combined_ser_dataset/, and runs a sanity check (counts wav files,
# checks the manifest is parseable).
#
# Usage on the HPC:
#   curl -fsSL https://raw.githubusercontent.com/mannuking/dtu-multimodal-emotion-recognition/main/scripts/install_dataset.sh | bash
#
# Or with a custom download URL (e.g. from Google Drive):
#   ./scripts/install_dataset.sh https://example.com/dtu_ser_dataset_v1.tar.gz
#
# Environment overrides:
#   DATASET_DIR=/path/to/extract    # default: ./combined_ser_dataset
#   DATASET_SHA256=<expected-sha>   # default: embedded below
set -euo pipefail

# ---- Config ----
DEFAULT_URL="https://github.com/mannuking/dtu-multimodal-emotion-recognition/releases/download/v1.0.0-data/dtu_ser_dataset_v1.tar.gz"
EXPECTED_SHA256="3df9907ffd1298c1eb28b5fcc7c4ffdd7f7a9a14b67650b2461e934e102f4346"
TARBALL="${DATASET_TARBALL:-dtu_ser_dataset_v1.tar.gz}"
EXTRACT_DIR="${DATASET_DIR:-combined_ser_dataset}"

URL="${1:-$DEFAULT_URL}"

# ---- Header ----
echo "============================================================"
echo " DTU Multimodal Dataset Installer"
echo " Source:  $URL"
echo " Target:  $EXTRACT_DIR/"
echo " Expected SHA256: $EXPECTED_SHA256"
echo "============================================================"

# ---- Pre-flight ----
mkdir -p "$EXTRACT_DIR"
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 1; }
command -v tar   >/dev/null 2>&1 || { echo "ERROR: tar not found"; exit 1; }
command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 \
    || { echo "ERROR: sha256sum/shasum not found"; exit 1; }

# ---- 1. Download ----
echo ""
echo "[1/4] Downloading dataset..."
if [[ "$URL" =~ ^https?:// ]]; then
    curl -fSL --retry 3 --connect-timeout 30 -o "$TARBALL" "$URL"
else
    # Local file path
    cp "$URL" "$TARBALL"
fi
SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  ✓ downloaded $TARBALL ($SIZE)"

# ---- 2. Verify checksum ----
echo ""
echo "[2/4] Verifying SHA256..."
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA=$(sha256sum "$TARBALL" | awk '{print $1}')
else
    ACTUAL_SHA=$(shasum -a 256 "$TARBALL" | awk '{print $1}')
fi
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA256" ]]; then
    echo "  ✗ SHA256 mismatch"
    echo "    expected: $EXPECTED_SHA256"
    echo "    actual:   $ACTUAL_SHA"
    echo "  The download is corrupted or tampered with. Aborting."
    exit 1
fi
echo "  ✓ SHA256 verified"

# ---- 3. Extract ----
echo ""
echo "[3/4] Extracting to $EXTRACT_DIR/..."
tar -xzf "$TARBALL" -C .
WAV_COUNT=$(find "$EXTRACT_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
echo "  ✓ extracted $WAV_COUNT wav files"

# ---- 4. Sanity check ----
echo ""
echo "[4/4] Running sanity checks..."
if [[ ! -f "$EXTRACT_DIR/metadata.csv" ]]; then
    echo "  ✗ metadata.csv missing"
    exit 1
fi
ROW_COUNT=$(wc -l < "$EXTRACT_DIR/metadata.csv" | tr -d ' ')
echo "  ✓ metadata.csv present ($ROW_COUNT rows including header)"

if command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import csv, os
with open("$EXTRACT_DIR/metadata.csv") as f:
    rows = list(csv.DictReader(f))
classes = {}
for r in rows:
    classes[r["emotion"]] = classes.get(r["emotion"], 0) + 1
print(f"  ✓ manifest parseable, {len(rows)} data rows")
print(f"  ✓ class distribution:")
for cls in sorted(classes):
    print(f"      {cls:10s} {classes[cls]:>5d}")
missing = sum(1 for r in rows if not os.path.exists(os.path.join("$EXTRACT_DIR", r["filepath"])))
if missing > 0:
    print(f"  ⚠ {missing} manifest entries point to missing files")
else:
    print(f"  ✓ all manifest entries point to existing files")
PY
fi

# ---- Cleanup ----
echo ""
echo "Cleaning up tarball..."
rm -f "$TARBALL"

echo ""
echo "============================================================"
echo " Dataset ready at $EXTRACT_DIR/"
echo ""
echo " Next steps:"
echo "   uv sync --extra cuda"
echo "   uv run python uv_run_all.py"
echo "============================================================"