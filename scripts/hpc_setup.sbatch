#!/bin/bash
#SBATCH --job-name=dtu-setup
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --output=setup.%J.out
#SBATCH --account=dtuarp-acc

# ============================================================
# DTU Multimodal Emotion Recognition — HPC setup script (uv-native)
# ============================================================
# Runs on a compute node (login node has no internet access).
# Installs: uv, Python 3.11, all deps via uv sync, dataset via curl.
# Skips Conda entirely — uv brings its own Python interpreter.
# ============================================================

set -e
cd "$SLURM_SUBMIT_DIR"

echo "============================================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Home: $HOME"
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "============================================================"

# ---- 1. Install uv (no sudo, single static binary) ----
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[1/5] uv already installed: $(uv --version)"
fi
export PATH="$HOME/.local/bin:$PATH"

# ---- 2. Clone the repo (private — uses HTTPS via PAT or SSH key) ----
echo "[2/5] Cloning the repo..."
if [[ ! -d "dtu-multimodal-emotion-recognition" ]]; then
    # Try HTTPS first (relies on PAT being embedded in URL or git credential helper).
    # If the bare clone fails with auth error, the user needs to run
    #   git clone https://<TOKEN>@github.com/mannuking/dtu-multimodal-emotion-recognition.git
    # manually on a compute node once.
    if ! git clone https://github.com/mannuking/dtu-multimodal-emotion-recognition.git 2>&1; then
        echo ""
        echo "=== AUTHENTICATION REQUIRED ==="
        echo "The bare 'git clone' failed — repo is private."
        echo "You have two options:"
        echo "  (a) Embed a GitHub PAT in the URL:"
        echo "      git clone https://<YOUR_PAT>@github.com/mannuking/dtu-multimodal-emotion-recognition.git"
        echo "  (b) Set up SSH keys (recommended):"
        echo "      ssh-keygen -t ed25519  (no passphrase)"
        echo "      cat ~/.ssh/id_ed25519.pub  (copy this)"
        echo "      Paste it at https://github.com/settings/keys"
        echo "      git clone git@github.com:mannuking/dtu-multimodal-emotion-recognition.git"
        exit 1
    fi
fi
cd dtu-multimodal-emotion-recognition
git pull origin main || echo "  (pull failed — using existing checkout)"

# ---- 3. Install dependencies via uv ----
echo "[3/5] Installing Python 3.11 + dependencies via uv (this takes ~5-10 min)..."
uv venv --python 3.11
uv sync --extra cuda

# ---- 4. Download dataset (if not already present) ----
echo "[4/5] Downloading dataset (if needed)..."
if [[ ! -f "combined_ser_dataset/metadata.csv" ]]; then
    curl -fsSL https://raw.githubusercontent.com/mannuking/dtu-multimodal-emotion-recognition/main/scripts/install_dataset.sh | bash
else
    WAV_COUNT=$(find combined_ser_dataset -name "*.wav" 2>/dev/null | wc -l)
    echo "  Dataset already present ($WAV_COUNT wav files)"
fi

# ---- 5. Verify everything ----
echo "[5/5] Verification..."
uv run python -c "
import torch, tensorflow as tf
print('  torch:', torch.__version__)
print('  torch.cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU:', torch.cuda.get_device_name(0))
print('  tensorflow:', tf.__version__)
print('  gpus visible to tf:', len(tf.config.list_physical_devices('GPU')))
import librosa, sklearn, transformers
print('  librosa:', librosa.__version__)
print('  sklearn:', sklearn.__version__)
print('  transformers:', transformers.__version__)
"

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Dataset:"
ls combined_ser_dataset/metadata.csv
find combined_ser_dataset -name "*.wav" | wc -l
echo ""
echo "Next step — start training interactively:"
echo "  srun --nodes=1 --ntasks-per-node=32 --gres=gpu:A100-SXM4:1 \\"
echo "       --time=12:00:00 --account=dtuarp-acc --pty /bin/bash"
echo ""
echo "Then on the compute node:"
echo "  export PATH=\$HOME/.local/bin:\$PATH"
echo "  cd ~/Research/dtu/dtu-multimodal-emotion-recognition"
echo "  uv run python uv_run_all.py 2>&1 | tee training.log"
echo ""
echo "Or submit training as another batch job (see scripts/submit_pipeline.sh in the repo)."