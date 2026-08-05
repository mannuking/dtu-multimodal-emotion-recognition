# DTU Multimodal Emotion Recognition — Training & Inference Pipeline

A multi-modal deep-learning system for emotion recognition across speech (SER), text (TER), and facial expression (FER), with late-fusion meta-classifier for downstream psychological-trait inference.

This repository contains the **complete source code, configuration, and entrypoints** for retraining and evaluating the system end-to-end. The dataset is distributed separately (see "Dataset" below). Trained checkpoints are available as a GitHub Release.

---

## TL;DR for the HPC

```bash
git clone https://github.com/mannuking/dtu-multimodal-emotion-recognition.git
cd dtu-multimodal-emotion-recognition

# One-time setup — installs everything into .venv via uv
uv sync --extra cuda

# Get the dataset (upload to Google Drive once, then download here)
# See "Dataset" section below.

# Run the full pipeline with one command
uv run python uv_run_all.py
```

That's the entire workflow. `uv sync` creates `.venv/` and installs all dependencies from `uv.lock`. `uv run python uv_run_all.py` activates the venv, validates the environment, builds the leakage-free triplet manifest, trains all unimodal models, trains the meta-classifier, and runs the integrity test — all checkpoint-guarded.

---

## Dataset

The audio corpus (`combined_ser_dataset/`) contains 11,970 labeled WAV files across 7 emotion classes, drawn from CREMA-D, RAVDESS, SAVEE, and TESS. It is distributed as a single tarball for easy upload to Google Drive.

**Download:** [Google Drive — `dtu_ser_dataset_v1.tar.gz` (965 MB)](https://drive.google.com/) — *paste your actual Google Drive link here*

After downloading:

```bash
tar xzf dtu_ser_dataset_v1.tar.gz
ls combined_ser_dataset/metadata.csv   # 11,970 rows, manifest
ls combined_ser_dataset/angry/ | wc -l # 1,923 wav files
```

The manifest (`metadata.csv`) has columns: `filepath, emotion, original_dataset`.

### Class distribution (after dedup)

| Emotion  | Samples |
|----------|---------|
| Angry    | 1,923   |
| Disgust  | 1,923   |
| Fear     | 1,923   |
| Happy    | 1,923   |
| Sad      | 1,923   |
| Neutral  | 1,703   |
| Surprise |   652   |
| **Total**| **11,970** |

### Source datasets (per `original_dataset`)

| Source     | Samples |
|------------|---------|
| CREMA-D    | 7,442 (note: manifest dedup leaves fewer unique files per class) |
| TESS       | 5,600   |
| RAVDESS    | 2,496   |
| SAVEE      | 480     |

---

## What this repo contains

```
dtu-multimodal-emotion-recognition/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml             (uv-managed; single source of truth for deps)
├── uv.lock                    (deterministic lockfile — committed)
│
├── uv_run_all.py              (ONE-COMMAND entrypoint — does everything)
│
├── train_ser_tensorflow.py    (SER — deep 1D-CNN over MFCC + energy features)
├── train_fer_tensorflow.py    (FER — VGG16 + ResNet50 ensemble)
├── train_ter_pytorch.py       (TER — MobileBERT with FGSM adversarial training)
├── train_meta_classifier.py   (Late-fusion meta-classifier MLP)
│
├── build_triplets_leakage_free.py  (subject-disjoint triplet manifest)
├── verify_ser.py              (integrity test for the SER model)
├── multi_page_app.py          (Gradio multi-page UI for demos)
├── test_system.py             (inference smoke-test harness)
├── test_environment.py        (Python / TF / Torch environment check)
├── combined_dtu.py            (original end-to-end pipeline driver)
├── config.py                  (single source of truth for paths, hyperparams)
│
├── scripts/
│   ├── train_all.sh           (bash equivalent of uv_run_all.py)
│   ├── eval_all.sh            (run integrity tests after training)
│   ├── infer_one.sh           (single-sample inference helper)
│   └── git-hooks/pre-commit   (secret scanner)
│
├── reports/                   (integrity test reports)
│   ├── MODEL_INTEGRITY_TEST_REPORT.pdf
│   ├── MODEL_INTEGRITY_TEST_REPORT.md
│   ├── IMPLEMENTATION_READINESS_REPORT.md
│   ├── ser_verification.json
│   └── render_pdf.py
│
├── model_checkpoints/         (populated locally after training)
│   └── *.json                 (training histories — TRACKED)
│
├── combined_ser_dataset/      (populated from tarball; not in git)
│   ├── metadata.csv
│   └── {angry,disgust,fear,happy,sad,neutral,surprise}/*.wav
│
├── triplets_manifest.csv      (OUTPUT — built by build_triplets_leakage_free.py)
│
└── data/
    └── README.md              (dataset acquisition guide)
```

---

## What this repo does NOT contain

| Asset | Size | Where it lives | Why |
|---|---|---|---|
| Trained model checkpoints | ~1.3 GB | GitHub Release `v1.0.0` | Too large for git history |
| `combined_ser_dataset/*.wav` (11,970 audio files) | 1.45 GB raw / 965 MB tar.gz | Google Drive | Datasets are separately licensed |
| FER2013 images | ~300 MB | https://www.kaggle.com/datasets/msambare/fer2013 | Kaggle |
| DailyDialog / MELD / EmoryNLP / IEMOCAP text CSVs | varies | HuggingFace / dataset authors | Per dataset license |

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12 GB | 24 GB (RTX 4090, A5000) |
| RAM | 32 GB | 64 GB+ |
| Disk | 15 GB | 50 GB (incl. raw datasets) |
| CUDA | 11.8 | 12.1+ |
| cuDNN | 8.x | 8.9+ |
| Python | 3.11 | 3.11 (pinned by pyproject.toml) |
| uv | 0.4+ | latest |

NVIDIA RTX 3070 6 GB is **insufficient** for the VGG16 + ResNet50 FER ensemble. Use RTX 3080/4070 or higher.

---

## Why `uv`?

This project uses [uv](https://github.com/astral-sh/uv) as the exclusive Python package manager. Benefits:

- **Fast**: 10-100x faster than pip
- **Deterministic**: `uv.lock` pins exact versions for reproducibility
- **Self-contained**: no system Python pollution
- **Single source of truth**: `pyproject.toml` + `uv.lock` replace `requirements.txt` + `setup.py` + `pip` + `python -m venv`

The `.venv/` directory is uv-managed. There is no `requirements.txt` and there are no manual `pip install` commands anywhere. To add a dependency: `uv add <package>`. To install everything from the lockfile: `uv sync`. To run a script: `uv run python script.py`.

### Extra dependency groups

| Extra | Adds | When to use |
|---|---|---|
| `--extra cuda` | `tensorflow[and-cuda]` | Linux + NVIDIA GPU (HPC) |
| `--extra macos` | `tensorflow-macos`, `tensorflow-metal` | macOS Apple Silicon dev |
| `--extra docs` | `weasyprint`, `python-docx`, `pymupdf` | regenerating the integrity report PDF |

Install multiple: `uv sync --extra cuda --extra docs`.

---

## Architectural summary

| Modality | Architecture | Input | Output |
|---|---|---|---|
| **SER** | 9× Conv1D + BatchNorm + ReLU + MaxPool, Dense(512) → Softmax(7) | 3-second audio at 16 kHz → MFCC-40 + ZCR + RMS + energy + entropy (11,044 dims) | 7-dim softmax over emotions |
| **FER** | VGG16 + ResNet50 ensemble (both ImageNet-pretrained, top-8/top-10 layers unfrozen) | 224×224 RGB face image | 7-dim softmax per model, averaged |
| **TER** | MobileBERT-base + classification head, FGSM adversarial training on word embeddings | Tokenized text (max 128 tokens) | 7-dim softmax over emotions |
| **Meta-classifier** | Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.2) → Softmax(7) | Concatenated 21-dim vector (7 + 7 + 7) of unimodal probabilities | 7-dim fused softmax |

Training: focal-weighted cross-entropy (SER), class-weighted cross-entropy (TER, FER), Adam optimizer with ReduceLROnPlateau and EarlyStopping.

---

## Triplet manifest construction (leakage-free)

`build_triplets_leakage_free.py` constructs a balanced triplet manifest with **subject-disjoint splits**:

1. Subject ID is extracted per audio file (RAVDESS actor ID, CREMA-D speaker ID, etc.).
2. Triplets are constructed by pairing audio with random same-class text + face samples.
3. Subjects are split 80/10/10 into train/val/test; any triplet containing a test subject is assigned to the test split (and similarly for val).
4. This prevents the meta-classifier from learning subject-consistency shortcuts that could inflate test accuracy.

This is the standard fix for the leakage pattern flagged during manuscript review.

---

## Single-sample inference

After training completes, you can run a single-sample inference:

```bash
uv run python scripts/infer_one.py audio.wav "some text" face.jpg
```

Or interactively via the Gradio app:

```bash
cp .env.example .env          # fill in XAI_API_KEY
uv run python multi_page_app.py
```

---

## License

MIT License. See `LICENSE`. Dataset licenses apply separately per the upstream dataset providers.