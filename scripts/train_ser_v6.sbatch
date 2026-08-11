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
# Uses v5 sbatch template (the working one, 71% baseline).
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
# Step 1: Build proper manifest.csv from metadata.csv
# Use a self-contained awk script (no external getline calls — those can
# silently fail on some compute nodes). Hash via in-process byte summing.
# ============================================================
META="combined_ser_dataset/metadata.csv"
MANIFEST="combined_ser_dataset/manifest.csv"

if [ ! -f "$META" ]; then
    echo "ERROR: $META missing. Run: bash fetch_combined_dataset.sh"
    exit 1
fi

echo "Building manifest.csv from $META..."

awk -F, '
function hash_bucket(s,   i, c, v, h) {
    # Simple deterministic hash: sum of (char * position) mod 100
    h = 0
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        v = 0
        # crude char->int: use ord via sprintf trick
        # Actually use awk ord: split s into chars, get numeric via index()
        # Simpler: use the byte as the index into a lookup table
        n = index("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_", c)
        if (n == 0) n = 1
        h = (h * 31 + n) % 1000000
    }
    return h % 100
}
BEGIN {
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

    split(fname, parts, "_")
    subject = parts[1]
    text = "a person feels " emo

    bucket = hash_bucket(subject)
    if (bucket >= 80)      split_name = "test"
    else if (bucket >= 65) split_name = "val"
    else                   split_name = "train"

    print fp, text, label, subject, split_name, ""
}
' "$META" > "$MANIFEST"

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
