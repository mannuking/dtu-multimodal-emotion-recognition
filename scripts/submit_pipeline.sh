#!/usr/bin/env bash
# submit_pipeline.sh — Submit the full multimodal training pipeline to a
# PARAM Siddhi-AI compute node via SLURM.
#
# Usage:
#   ./submit_pipeline.sh               # full training
#   ./submit_pipeline.sh --quick       # 3-epoch smoke test
#   ./submit_pipeline.sh --gpus=2      # request 2 GPUs
#   ./submit_pipeline.sh --time=24:00:00 --partition=boost_q  # custom time + partition
#
# Defaults assume:
#   - 1 GPU on the boost_q partition
#   - 24-hour walltime
#   - Default project = dtuarp-acc
#   - Output to logs/slurm-<jobid>.out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

QUICK=0
GPUS=1
PARTITION="boost_q"
TIME="24:00:00"
PROJECT="dtuarp-acc"
NODES=1
NTASKS_PER_NODE=1
CPUS_PER_TASK=8
MEM="64G"

for arg in "$@"; do
    case "$arg" in
        --quick)                QUICK=1 ;;
        --gpus=*)               GPUS="${arg#*=}" ;;
        --partition=*)          PARTITION="${arg#*=}" ;;
        --time=*)               TIME="${arg#*=}" ;;
        --project=*)            PROJECT="${arg#*=}" ;;
        --nodes=*)              NODES="${arg#*=}" ;;
        --cpus-per-task=*)      CPUS_PER_TASK="${arg#*=}" ;;
        --mem=*)                MEM="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ---- Probe what's available ----
echo "============================================================"
echo " Probing available partitions and resources..."
echo "============================================================"
sinfo -o "%20P %10L %5D %10T" 2>&1 | head -20 || true
echo ""
squeue -u "$USER" 2>&1 | head -5 || true
echo ""

# ---- Build the sbatch script dynamically ----
mkdir -p "$PROJECT_ROOT/logs"
SBATCH_FILE="$PROJECT_ROOT/logs/pipeline_$(date +%Y%m%d_%H%M%S).sbatch"

cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dtu-multimodal
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${PROJECT_ROOT}/logs/slurm-%j.out
#SBATCH --error=${PROJECT_ROOT}/logs/slurm-%j.err
#SBATCH --account=${PROJECT}

set -euo pipefail

echo "============================================================"
echo " Job started: \$(date)"
echo " Node: \$(hostname)"
echo " GPUs visible: \$(nvidia-smi -L 2>/dev/null || echo 'none')"
echo " Working dir: ${PROJECT_ROOT}"
echo "============================================================"

# ---- Module + proxy setup (C-DAC PARAM Siddhi-AI conventions) ----
# Modules aren't always needed — uv brings its own Python
# Uncomment if your cluster requires modules:
# module load python/3.11 cuda/12.1 cudnn/8.9

export http_proxy="http://proxy-10g.10g.siddhi.param:9090"
export https_proxy="http://proxy-10g.10g.siddhi.param:9090"
export ftp_proxy="http://proxy-10g.10g.siddhi.param:9090"
export no_proxy="localhost,127.0.0.1,*.npsf.cdac.in,login.npsf.cdac.in"

# ---- Ensure uv is on PATH ----
export PATH="\$HOME/.local/bin:\$HOME/tools/gh/bin:\$PATH"

# ---- Verify dataset is in place ----
if [[ ! -f "combined_ser_dataset/metadata.csv" ]]; then
    echo "ERROR: combined_ser_dataset/metadata.csv missing"
    echo "Run install_dataset.sh first:"
    echo "  curl -fsSL https://raw.githubusercontent.com/mannuking/dtu-multimodal-emotion-recognition/main/scripts/install_dataset.sh | bash"
    exit 1
fi

WAV_COUNT=\$(find combined_ser_dataset -name '*.wav' | wc -l)
echo "Dataset: \${WAV_COUNT} wav files"

# ---- Run the pipeline ----
cd "${PROJECT_ROOT}"
EOF

if [[ $QUICK -eq 1 ]]; then
    echo "echo 'Running smoke test (3 epochs each)...'" >> "$SBATCH_FILE"
    echo "uv run python uv_run_all.py --quick 2>&1 | tee logs/run_quick.log" >> "$SBATCH_FILE"
else
    echo "echo 'Running full training pipeline...'" >> "$SBATCH_FILE"
    echo "uv run python uv_run_all.py 2>&1 | tee logs/run_full.log" >> "$SBATCH_FILE"
fi

cat >> "$SBATCH_FILE" <<EOF

echo "============================================================"
echo " Job finished: \$(date)"
echo " Exit code: \$?"
echo "============================================================"
EOF

chmod +x "$SBATCH_FILE"
echo ""
echo "Generated SBATCH script: $SBATCH_FILE"
echo ""
cat "$SBATCH_FILE"
echo ""
echo "============================================================"
echo " Submitting with: sbatch $SBATCH_FILE"
echo "============================================================"

# ---- Submit ----
JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $4}')
echo ""
echo "Submitted job ID: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f $PROJECT_ROOT/logs/slurm-$JOB_ID.out"
echo ""
echo "Cancel with:"
echo "  scancel $JOB_ID"