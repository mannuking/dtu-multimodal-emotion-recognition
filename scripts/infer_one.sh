#!/usr/bin/env bash
# infer_one.sh — Single-sample inference through the full SER+FER+TER pipeline.
# Used by the Gradio app backend and by ad-hoc test scripts.
#
# Usage:
#   ./scripts/infer_one.sh audio.wav [text.txt] [face.jpg]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_USE_LEGACY_KERAS=1

AUDIO="${1:-}"
TEXT="${2:-}"
FACE="${3:-}"

if [[ -z "$AUDIO" ]]; then
    echo "Usage: $0 <audio.wav> [text.txt] [face.jpg]"
    exit 1
fi

python3 - <<PY
import os, sys
sys.path.insert(0, "$PROJECT_ROOT")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
from test_system import run_inference
result = run_inference(
    audio_path="$AUDIO",
    text="$([ -n "$TEXT" ] && cat "$TEXT" || '')",
    face_path="$FACE" if "$FACE" else None,
)
print(result)
PY