"""
uv_run_all.py — Single-command entrypoint for the DTU multimodal pipeline.

Run with:
    uv run python uv_run_all.py

On the HPC after `git clone` + `tar xzf` of the dataset archive:
    uv sync --extra cuda            # one-time setup
    tar xzf ~/dtu_ser_dataset_v1.tar.gz
    uv run python uv_run_all.py     # full pipeline

This script:
  1. Validates the environment (GPU + dataset presence).
  2. Builds the leakage-free triplet manifest.
  3. Trains SER (1D-CNN).
  4. Trains TER (MobileBERT + FGSM).
  5. Trains FER (VGG16 + ResNet50 ensemble).
  6. Trains the meta-classifier (late fusion).
  7. Runs the integrity test on the held-out SER split.
  8. Prints a final summary with the test accuracy vs. paper claim.

All sub-stages are checkpoint-guarded: re-running resumes from where it
left off (existing checkpoints are not retrained).
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Force GPU env vars BEFORE importing torch/TF (which loads CUDA)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/opt/hpcx")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# TF logs get noisy with mixed precision + XLA
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Use the new Keras 3 backend (TF_USE_LEGACY_KERAS=1 breaks modern optimizer/model API)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n {title}\n{line}")


def run(cmd: list[str], name: str) -> bool:
    banner(f"▶ {name}")
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, env=os.environ).returncode
    dt = time.time() - t0
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  → {status} in {dt:.1f}s")
    return rc == 0


def check_environment() -> dict:
    banner("Environment check")
    info = {}
    # Python
    info["python"] = sys.version.split()[0]
    print(f"Python: {info['python']}")
    # PyTorch CUDA
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            print(f"PyTorch: {torch.__version__} | CUDA: yes | {info['cuda_device']} ({info['cuda_vram_gb']} GB)")
        else:
            print(f"PyTorch: {torch.__version__} | CUDA: NO (CPU only — will be slow)")
    except ImportError:
        print("PyTorch not installed")
        info["cuda_available"] = False
    # TensorFlow
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["tf_gpus"] = len(gpus)
        print(f"TensorFlow: {tf.__version__} | GPUs: {len(gpus)}")
    except ImportError:
        print("TensorFlow not installed")
        info["tensorflow"] = None
    # Dataset presence
    audio_dir = PROJECT_ROOT / "combined_ser_dataset"
    info["audio_dir"] = str(audio_dir)
    info["audio_exists"] = audio_dir.exists() and any(audio_dir.glob("**/*.wav"))
    info["audio_count"] = sum(1 for _ in audio_dir.rglob("*.wav")) if info["audio_exists"] else 0
    manifest = audio_dir / "metadata.csv" if audio_dir.exists() else None
    info["manifest_exists"] = manifest is not None and manifest.exists()
    print(f"Dataset dir: {audio_dir} | exists: {info['audio_exists']} | wavs: {info['audio_count']}")
    print(f"Manifest: {manifest} | exists: {info['manifest_exists']}")
    return info


def main():
    print(f"DTU Multimodal Emotion Recognition — full pipeline entrypoint")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    info = check_environment()
    if not info["audio_exists"]:
        print("\nERROR: Dataset not found at combined_ser_dataset/")
        print("  Expected after extracting dtu_ser_dataset_v1.tar.gz")
        print("  Get the archive from Google Drive (see README.md) and run:")
        print("    tar xzf dtu_ser_dataset_v1.tar.gz")
        sys.exit(1)

    # 1. Leakage-free triplet manifest
    if not run(
        [sys.executable, "build_triplets_leakage_free.py",
         "--audio-manifest", "combined_ser_dataset/metadata.csv",
         "--output", "triplets_manifest.csv",
         "--per-class", "1000",
         "--seed", "42"],
        "Build leakage-free triplet manifest",
    ):
        sys.exit(1)

    # 2. Train SER
    run([sys.executable, "train_ser_wav2vec.py"], "Train SER (wav2vec2-base frozen + MLP head, PyTorch)")

    # 3. Train TER (PyTorch + MobileBERT)
    run([sys.executable, "train_ter_pytorch.py"], "Train TER (MobileBERT + FGSM adversarial)")

    # 4. Train FER
    run([sys.executable, "train_fer_tensorflow.py"], "Train FER (VGG16 + ResNet50 ensemble)")

    # 5. Train meta-classifier
    run([sys.executable, "train_meta_classifier_pytorch.py"], "Train meta-classifier (late fusion MLP, PyTorch)")

    # 6. Integrity test
    ser_verified = run([sys.executable, "verify_ser_wav2vec.py"], "Run integrity test on SER (wav2vec2)")

    # 7. Summary
    banner("Pipeline summary")
    summary_path = PROJECT_ROOT / "reports" / "ser_verification.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        print(f"  SER test accuracy: {data.get('accuracy', '?')}")
        print(f"  SER macro F1:      {data.get('macro_f1', '?')}")
        print(f"  Paper claim:       {data.get('paper_claim_accuracy', '?')} / {data.get('paper_claim_macro_f1', '?')}")
        print(f"  Delta accuracy:    {data.get('delta_accuracy', '?')}")
        print(f"  ms per sample:     {data.get('ms_per_sample', '?')}")
    else:
        print("  No ser_verification.json found — integrity test may have failed")

    print("\nDone.")
    sys.exit(0 if ser_verified else 2)


if __name__ == "__main__":
    main()