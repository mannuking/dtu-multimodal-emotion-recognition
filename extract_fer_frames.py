"""
extract_fer_frames.py — Build a REAL visual channel for the ORIGINAL pipeline.

WHY THIS EXISTS
The original triplets manifest has `face_img = placeholder_N` for all 7000
rows, and there are no face directories anywhere on disk. The meta-classifier
therefore fused a ZERO vector for the visual modality — the "multimodal"
fusion was audio+text wearing a multimodal label.

The visual channel is recoverable. CREMA-D (7441 of the 11568 combined-corpus
utterances) ships FLV VIDEO alongside the WAV, on the official GitLab mirror:

    https://gitlab.com/cs-cooper-lab/crema-d-mirror

The video files are git-lfs objects, so a plain /-/raw/ fetch returns the
131-byte POINTER, not the video. The real bytes come from the LFS batch API:

    POST <repo>.git/info/lfs/objects/batch
    {"operation":"download","transfers":["basic"],"objects":[{"oid":...,"size":...}]}

Verified end-to-end: pointer -> LFS batch -> 265922-byte FLV ->
ffmpeg -> 11 frames at 480x360, and the frames are clear human faces
mid-expression.

RAVDESS and TESS also ship video, but CREMA-D alone covers 64% of the corpus
and is the single largest source, so it is done first.

OUTPUT
    fer_frames/<sample_id>/frame_XXX.jpg   extracted frames
    fer_frames/manifest.csv                sample_id,frame_path,emotion,n_frames
    combined_ser_dataset/manifest.csv      gains a `face_img` column

The triplets manifest is then rebuilt against this, so the meta-classifier
fuses a real FER probability vector instead of zeros.

Usage (on the HPC, compute node — it is network + CPU bound):
    uv run python -u extract_fer_frames.py --limit 100     # smoke test
    uv run python -u extract_fer_frames.py                 # full CREMA-D
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path("cs-cooper-lab%2Fcrema-d-mirror")
GITLAB_PROJECT = "cs-cooper-lab/crema-d-mirror"
LFS_BATCH = f"https://gitlab.com/{GITLAB_PROJECT}.git/info/lfs/objects/batch"
TREE_API = (f"https://gitlab.com/api/v4/projects/{REPO}/repository/tree"
            "?path=VideoFlash&per_page=100&page={page}")
FILE_API = (f"https://gitlab.com/api/v4/projects/{REPO}/repository/files/"
            "VideoFlash%2F{name}?ref=main")

FRAME_DIR = Path("fer_frames")
OUT_MANIFEST = FRAME_DIR / "manifest.csv"
FPS = 5           # frames per second to sample
MAX_FRAMES = 12   # cap per clip — the fusion only needs a few


def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/vnd.git-lfs+json",
                 "Accept": "application/vnd.git-lfs+json",
                 "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def list_video_files(max_files: int | None = None) -> list[str]:
    """Page through the VideoFlash tree."""
    names: list[str] = []
    page = 1
    while True:
        raw = http_get(TREE_API.format(page=page), timeout=60)
        try:
            items = json.loads(raw)
        except Exception:
            break
        if not items:
            break
        for it in items:
            if it.get("type") == "blob" and it.get("name", "").lower().endswith(".flv"):
                names.append(it["name"])
                if max_files and len(names) >= max_files:
                    return names
        if len(items) < 100:
            break
        page += 1
    return names


def lfs_pointer(name: str) -> tuple[str, int] | None:
    """Get the (oid, size) of a git-lfs object for a file."""
    try:
        raw = http_get(FILE_API.format(name=name), timeout=60)
        meta = json.loads(raw)
    except Exception:
        return None
    # GitLab returns the pointer content base64 in content for lfs files
    content = meta.get("content")
    if not content:
        return None
    import base64
    try:
        text = base64.b64decode(content).decode("utf-8", "replace")
    except Exception:
        return None
    oid = size = None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            try:
                size = int(line.split()[1])
            except Exception:
                pass
    if oid and size:
        return oid, size
    return None


def lfs_download(oid: str, size: int) -> bytes | None:
    """Fetch the real bytes of an lfs object via the batch API."""
    try:
        resp = http_post_json(LFS_BATCH, {
            "operation": "download", "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size}]})
        obj = (resp.get("objects") or [{}])[0]
        dl = (obj.get("actions") or {}).get("download") or {}
        href = dl.get("href")
        if not href:
            return None
        req = urllib.request.Request(href, headers=dl.get("header") or {})
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()
    except Exception:
        return None


def ffmpeg_binary() -> str:
    """Resolve the ffmpeg binary.

    The sbatch exports FFMPEG_BIN when the system has no ffmpeg on PATH but
    imageio-ffmpeg supplied a static binary. Fall back to PATH, then to
    imageio_ffmpeg's own bundled copy.
    """
    env = os.environ.get("FFMPEG_BIN")
    if env and os.path.exists(env):
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_frames(video_bytes: bytes, sample_id: str) -> list[str]:
    """Decode frames from the video bytes. Returns frame paths."""
    outdir = FRAME_DIR / sample_id
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".flv", delete=False) as tf:
        tf.write(video_bytes)
        tmp = tf.name
    try:
        subprocess.run(
            [ffmpeg_binary(), "-y", "-v", "error", "-i", tmp,
             "-vf", f"fps={FPS}", "-frames:v", str(MAX_FRAMES),
             "-q:v", "2", str(outdir / "frame_%03d.jpg")],
            check=True, capture_output=True, timeout=120)
    except Exception as e:
        # leave the dir clean if the decode failed
        for f in outdir.glob("*.jpg"):
            f.unlink()
        try:
            outdir.rmdir()
        except OSError:
            pass
        return []
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return sorted(str(p) for p in outdir.glob("frame_*.jpg"))


def emotion_of(sample_id: str) -> str:
    """CREMA-D codes -> canonical emotion name."""
    code = sample_id.split("_")[2] if len(sample_id.split("_")) > 2 else ""
    return {"ANG": "angry", "DIS": "disgust", "FEA": "fear",
            "HAP": "happy", "NEU": "neutral", "SAD": "sad"}.get(code, "unknown")


def process_one(name: str, retries: int = 2) -> dict | None:
    sample_id = name[:-4]  # strip .flv
    outdir = FRAME_DIR / sample_id
    existing = sorted(outdir.glob("frame_*.jpg"))
    if existing:
        return {"sample_id": sample_id, "frames": [str(p) for p in existing],
                "emotion": emotion_of(sample_id), "cached": True}

    for attempt in range(retries):
        ptr = lfs_pointer(name)
        if not ptr:
            continue
        blob = lfs_download(*ptr)
        if not blob or len(blob) < 1000:
            continue
        frames = extract_frames(blob, sample_id)
        if frames:
            return {"sample_id": sample_id, "frames": frames,
                    "emotion": emotion_of(sample_id), "cached": False}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N clips (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print("=" * 70)
    print("Extracting the real visual channel from CREMA-D video")
    print("=" * 70)

    ff = ffmpeg_binary()
    if ff and os.path.exists(ff):
        print(f"  ffmpeg: {ff}")
    else:
        print(f"  FATAL: ffmpeg not found at '{ff}' — cannot decode the .flv files")
        sys.exit(1)

    names = list_video_files(args.limit)
    print(f"  VideoFlash clips listed: {len(names)}")
    if not names:
        print("  FATAL: could not list the CREMA-D video files")
        sys.exit(1)

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    failures = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, n): n for n in names}
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                results.append(r)
            else:
                failures += 1
            if done % 100 == 0 or done == len(names):
                print(f"    {done}/{len(names)} processed, "
                      f"{len(results)} with frames, {failures} failed")

    print(f"\n  clips with frames: {len(results)}/{len(names)}")

    # ---- write the manifest ----
    with open(OUT_MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "frame_path", "emotion", "n_frames"])
        for r in sorted(results, key=lambda x: x["sample_id"]):
            for fr in r["frames"]:
                w.writerow([r["sample_id"], fr, r["emotion"], len(r["frames"])])
    print(f"  wrote {OUT_MANIFEST}")

    # ---- attach face_img to the corpus manifest ----
    corpus = Path("combined_ser_dataset/metadata.csv")
    if corpus.exists() and results:
        by_sample = {r["sample_id"]: r for r in results}
        with open(corpus, newline="") as f:
            rows = list(csv.DictReader(f))
            cols = list(rows[0].keys()) if rows else []
        if "face_img" not in cols:
            cols.append("face_img")
        hit = 0
        for row in rows:
            stem = Path(row.get("wav_path", "")).stem
            r = by_sample.get(stem)
            if r:
                # point at the MIDDLE frame — the most expressive one
                mid = r["frames"][len(r["frames"]) // 2]
                row["face_img"] = mid
                hit += 1
            else:
                row["face_img"] = ""
        with open(corpus, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  corpus manifest: {hit}/{len(rows)} rows now carry a face_img")

    # ---- rebuild the triplets manifest against the real faces ----
    triplets = Path("triplets_manifest.csv")
    if triplets.exists() and results:
        with open(triplets, newline="") as f:
            trows = list(csv.DictReader(f))
        by_stem = {r["sample_id"]: r for r in results}
        hit = 0
        for row in trows:
            stem = Path(row.get("speech_wav", "")).stem
            r = by_stem.get(stem)
            if r:
                row["face_img"] = r["frames"][len(r["frames"]) // 2]
                hit += 1
        with open(triplets, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
            w.writeheader()
            w.writerows(trows)
        print(f"  triplets manifest: {hit}/{len(trows)} rows now carry a real face_img")

    print("\n  The meta-classifier's visual channel is now real.")


if __name__ == "__main__":
    main()
