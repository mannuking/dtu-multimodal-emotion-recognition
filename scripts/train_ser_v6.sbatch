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
# ============================================================
# Uses v5 sbatch template (the working one, 71% baseline).
# Key: --account=dtuarp-acc + --gres=gpu:A100-SXM4:1
#
# Build manifest.csv with proper schema BEFORE training.
# Metadata: filepath,emotion,original_dataset
# Manifest: audio_path,text,label,subject,split,facial_path
#
# Usage:
#   cd ~/Research/dtu/dtu-multimodal-emotion-recognition
#   git pull origin main
#   sbatch scripts/train_ser_v6.sbatch
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
# Step 1: Build proper manifest.csv from metadata.csv (login-safe awk)
# ============================================================
META="combined_ser_dataset/metadata.csv"
MANIFEST="combined_ser_dataset/manifest.csv"

if [ ! -f "$META" ]; then
    echo "ERROR: $META missing. Run: bash fetch_combined_dataset.sh"
    exit 1
fi

echo "Building manifest.csv from $META..."

# Use awk with a stable hash via the POSIX-grade approach:
# bucket = (md5 hash of subject) % 100 -> train/val/test split
# emotion label from string via lookup table written in BEGIN block.

awk -F, '
BEGIN {
    # emotion -> label
    E2I["angry"]    = 0
    E2I["disgust"]  = 1
    E2I["fear"]     = 2
    E2I["happy"]    = 3
    E2I["sad"]      = 4
    E2I["surprise"] = 5
    E2I["neutral"]  = 6
    OFS = ","
    print "audio_path,text,label,subject,split,facial_path"
}
NR == 1 && $1 == "filepath" { next }
{
    fp = $1
    n = split(fp, p, "/")
    emo = tolower(p[1])
    fname = p[n]
    label = (emo in E2I) ? E2I[emo] : 6

    # subject = filename prefix before first underscore
    split(fname, parts, "_")
    subject = parts[1]

    # text synthesis
    text = "a person feels " emo

    # deterministic split: hex sum of md5(subject) bytes % 100
    cmd = "printf %s " subject " | md5sum | cut -c1-8"
    cmd | getline hash_hex
    close(cmd)
    # Convert hex to int (works for 0-ffffffff range)
    bucket = 0
    for (i = 1; i <= length(hash_hex); i++) {
        c = substr(hash_hex, i, 1)
        v = index("0123456789abcdef", tolower(c)) - 1
        bucket = (bucket * 16 + v) % 100
    }
    if (bucket >= 80)      split_name = "test"
    else if (bucket >= 65) split_name = "val"
    else                   split_name = "train"

    facial_path = ""
    print fp "," text "," label "," subject "," split_name "," facial_path
}
' "$META" > "$MANIFEST"

echo "  ✓ manifest.csv built"
echo "  total rows: $(wc -l < $MANIFEST) (incl. header)"
echo "  by split:"
awk -F, 'NR>1 {c[$5]++} END {for (k in c) printf "    %s: %d\n", k, c[k]}' "$MANIFEST"

# ============================================================
# Step 2: Pre-flight checks for frozen encoders
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
