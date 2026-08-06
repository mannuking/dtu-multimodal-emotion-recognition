"""
fer2013_download.py — Download FER2013 dataset from HuggingFace and lay it out
into the directory structure that train_fer_tensorflow.py expects:

    fer2013/train/<emotion>/image.jpg
    fer2013/test/<emotion>/image.jpg

Emotions in FER2013: angry, disgust, fear, happy, sad, surprise, neutral
(7 classes — matches our NUM_CLASSES).

Source: FER2013 is available on HuggingFace as `Jeneral/fer-2013` and others.
We pull the canonical CSV-format mirror and split 80/20.
"""

import os
import csv
import shutil
import urllib.request

FER_DIR = "fer2013"
TRAIN_DIR = os.path.join(FER_DIR, "train")
TEST_DIR = os.path.join(FER_DIR, "test")

# HuggingFace-hosted FER2013 CSV (28740 rows incl header)
HF_URL = (
    "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/fer2013.csv"
)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _download(url: str, dst: str) -> None:
    """Download with progress + resume-friendly simple fetch."""
    if os.path.exists(dst):
        print(f"  ✓ already cached: {dst}")
        return
    print(f"  ↓ downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  ✓ saved {dst} ({os.path.getsize(dst)/1024/1024:.1f} MB)")


def _parse_pixels(pixels: str) -> bytes:
    """FER2013 pixels column is space-separated 0-255 ints, length 48*48."""
    arr = bytes(int(x) for x in pixels.split())
    assert len(arr) == 48 * 48, f"bad pixel length: {len(arr)}"
    return arr


def _write_pgm(data: bytes, path: str) -> None:
    """Write 48x48 grayscale PGM (no PIL dependency required)."""
    with open(path, "wb") as f:
        f.write(f"P5\n48 48\n255\n".encode())
        f.write(data)


def download_and_unpack(force: bool = False) -> None:
    """Download FER2013 CSV and unpack into train/test directories."""
    csv_path = "fer2013.csv"

    if not force and os.path.isdir(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) >= 7:
        n = sum(1 for _ in os.walk(TRAIN_DIR)) - 1
        if n >= 7:
            print(f"  ✓ FER2013 already unpacked ({n} class dirs found)")
            return

    _download(HF_URL, csv_path)

    if os.path.isdir(FER_DIR):
        shutil.rmtree(FER_DIR)
    os.makedirs(TRAIN_DIR)
    os.makedirs(TEST_DIR)
    for e in EMOTIONS:
        os.makedirs(os.path.join(TRAIN_DIR, e), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, e), exist_ok=True)

    print(f"  ↓ unpacking FER2013 into {FER_DIR}/...")
    counts = {"train": {}, "test": {}}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            emotion_idx = int(row["emotion"])
            usage = row["Usage"]  # "Training" or "PublicTest"/"PrivateTest"
            if usage == "Training":
                split, sub = "train", TRAIN_DIR
            elif usage in ("PublicTest", "PrivateTest"):
                split, sub = "test", TEST_DIR
            else:
                continue
            emo_name = EMOTIONS[emotion_idx]
            out_dir = os.path.join(sub, emo_name)
            out_path = os.path.join(out_dir, f"{i:06d}.pgm")
            _write_pgm(_parse_pixels(row["pixels"]), out_path)
            counts[split].setdefault(emo_name, 0)
            counts[split][emo_name] += 1
            if i and i % 5000 == 0:
                print(f"    ...{i} rows processed")

    print("  ✓ FER2013 unpacked:")
    for split in ("train", "test"):
        for emo in EMOTIONS:
            print(f"    {split}/{emo}: {counts[split].get(emo, 0)}")


if __name__ == "__main__":
    download_and_unpack()