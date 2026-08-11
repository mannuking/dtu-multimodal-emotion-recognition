#!/bin/bash
#SBATCH --job-name=dtu-ser-v6-multimodal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --output=ser_v6.%J.out
#SBATCH --error=ser_v6.%J.err
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU SER v6 - Multimodal SER fusion (audio + text + facial)
# Uses v5 sbatch template (71% baseline).
# Builds manifest.csv directly from the WAV directory tree.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

SEED="${1:-42}"

echo "============================================================"
echo "v6 multimodal SLURM job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "Seed: $SEED"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

git pull origin main || echo "  (git pull failed - continuing with local checkout)"

# ============================================================
# Step 1: Build manifest.csv directly from the WAV directory tree
# (the metadata.csv that came from the tarball is unreliable)
# ============================================================
DATASET_DIR="combined_ser_dataset"
MANIFEST="$DATASET_DIR/manifest.csv"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: $DATASET_DIR missing. Run: bash fetch_combined_dataset.sh"
    exit 1
fi

echo "Building manifest.csv from WAV directory tree..."

# Remove any stale metadata.csv that may have been clobbered
rm -f "$DATASET_DIR/metadata.csv"

# Use find to enumerate all WAVs, parse emotion from parent dir name
# and subject from filename prefix. Use awk for the deterministic split.
{
    echo "audio_path,text,label,subject,split,facial_path"
    # Exclude macOS resource fork files (._*) that came from tarball extraction
    find "$DATASET_DIR" -name '*.wav' -type f ! -name '._*' | awk -F/ '
    {
        n = NF
        parent = $(n-1)
        fname = $n
        # Subject = filename prefix before first underscore
        split(fname, parts, "_")
        subject = parts[1]

        # Deterministic split: pure awk hash (no external commands)
        h = 0
        for (i = 1; i <= length(subject); i++) {
            c = substr(subject, i, 1)
            v = index("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_", c)
            if (v == 0) v = 1
            h = (h * 31 + v) % 1000000
        }
        bucket = h % 100

        if (bucket >= 80)      split_name = "test"
        else if (bucket >= 65) split_name = "val"
        else                   split_name = "train"

        # emotion -> label
        emo = parent
        label = 6  # default neutral
        if      (emo == "angry")    label = 0
        else if (emo == "disgust")  label = 1
        else if (emo == "fear")     label = 2
        else if (emo == "happy")    label = 3
        else if (emo == "sad")      label = 4
        else if (emo == "surprise") label = 5
        else if (emo == "neutral")  label = 6

        # Path relative to project root (for v6 to find wavs)
        rel = $0
        sub(/^.*combined_ser_dataset\//, "", rel)

        # text synthesis
        text = "a person feels " emo

        print rel "," text "," label "," subject "," split_name ","
    }'
} > "$MANIFEST"

echo "  ✓ manifest.csv built"
echo "  total rows (incl. header): $(wc -l < $MANIFEST)"
echo "  by split:"
awk -F, 'NR>1 {c[$5]++} END {for (k in c) printf "    %s: %d\n", k, c[k]}' "$MANIFEST"

# ============================================================
# Step 2: Pre-flight
# ============================================================
if [ ! -f model_checkpoints/ser_v5_best_seed42.pt ]; then
    echo "ERROR: model_checkpoints/ser_v5_best_seed42.pt missing"
    exit 1
fi
if [ ! -f model_checkpoints/ter_pytorch_best.pt ]; then
    echo "ERROR: model_checkpoints/ter_pytorch_best.pt missing"
    exit 1
fi

# ============================================================
# Step 3: Run training
# ============================================================
uv run python -u train_ser_v6_multimodal.py --seed "$SEED" --epochs 30 2>&1 | tee "train_ser_v6_seed${SEED}.log"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
