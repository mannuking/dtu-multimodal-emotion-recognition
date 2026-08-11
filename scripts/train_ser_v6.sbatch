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
# Build manifest.csv with proper schema BEFORE training:
#   metadata.csv has: filepath,emotion,original_dataset
#   v6 expects:       audio_path,text,label,subject,split,facial_path
# We transform via awk (login-node safe) into combined_ser_dataset/manifest.csv
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

# Map emotion string -> integer label (0-6)
EMOTION2ID=$(mktemp)
cat > "$EMOTION2ID" <<'EOF'
angry 0
disgust 1
fear 2
happy 3
sad 4
surprise 5
neutral 6
EOF

# Subject-disjoint split: use last 20% of unique subjects as test, next 15% as val
# Simpler heuristic: deterministic hash of subject % 100 -> split
#   test  = subject_hash in [80,99]
#   val   = subject_hash in [65,79]
#   train = otherwise
awk -F, -v emo2id="$EMOTION2ID" -v out="$MANIFEST" '
BEGIN {
    while ((getline line < emo2id) > 0) {
        split(line, parts, " ")
        E2I[parts[1]] = parts[2]
    }
    close(emo2id)
    OFS = ","
    print "audio_path,text,label,subject,split,facial_path"
}
NR==1 && $1=="filepath" { next }
{
    # $1 = filepath like "angry/1022_ITS_ANG_XX.wav"
    # $2 = emotion
    # $3 = original_dataset (CREMA-D, TESS, SAVEE, RAVDESS)
    fp = $1
    n = split(fp, p, "/")
    emo = tolower(p[1])
    fname = p[n]
    label = E2I[emo]
    if (label == "") label = 6  # default to neutral for unknown

    # Subject = filename prefix before first underscore (e.g. "1022_ITS_ANG_XX.wav" -> "1022")
    split(fname, parts, "_")
    subject = parts[1]

    # Synthesize text from emotion
    text = "a person feels " emo

    # Deterministic split by hashing subject
    cmd = "echo -n " subject " | cksum | awk "{print \$1}""
    cmd | getline h
    close(cmd)
    bucket = (h % 100)
    if (bucket >= 80)      split_name = "test"
    else if (bucket >= 65) split_name = "val"
    else                   split_name = "train"

    facial_path = ""  # no facial crops in v6 (v7 would add)
    print fp "," text "," label "," subject "," split_name "," facial_path
}' "$META" > "$MANIFEST"

rm -f "$EMOTION2ID"

echo "  ✓ manifest.csv built"
echo "  total rows: $(wc -l < $MANIFEST)"
echo "  by split:"
awk -F, 'NR>1 {c[$5]++} END {for (k in c) print "    "k": "c[k]}' "$MANIFEST"

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
