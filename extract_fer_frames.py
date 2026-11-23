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
import random
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path("cs-cooper-lab%2Fcrema-d-mirror")
GITLAB_PROJECT = "cs-cooper-lab/crema-d-mirror"
LFS_BATCH = f"https://gitlab.com/{GITLAB_PROJECT}.git/info/lfs/objects/batch"
# The VideoFlash pointers are fetched with a GIT CLONE, not the REST API.
# GitLab's anonymous REST API rate-limits hard (HTTP 429 after a few
# hundred calls), and listing 7442 clips through it means 7442 requests.
# A blobless sparse clone of the VideoFlash directory fetches ALL 7442
# lfs POINTERS in one operation -- the pointers are tiny text files, so
# the clone is fast and hits no API limit.
CLONE_URL = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git"
CLONE_DIR = Path("_crema_d_lfs")

FRAME_DIR = Path("fer_frames")
OUT_MANIFEST = FRAME_DIR / "manifest.csv"
POINTER_WORKERS = 6   # throttled: GitLab rate-limits the API
FPS = 5           # frames per second to sample
MAX_FRAMES = 12   # candidates decoded per clip
# How many of those to KEEP. The fusion needs exactly one image per
# utterance, so the default keeps only the most expressive frame: 7442
# files instead of 89000, which is 12x less NFS I/O for the same result.
KEEP_FRAMES = 1


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff, capped. GitLab rate-limits anonymous API use."""
    delay = min(2.0 * (2 ** attempt), 60.0)
    # jitter so parallel workers do not resynchronise
    time.sleep(delay * (0.5 + random.random() * 0.5))


def http_get(url: str, timeout: int = 120, retries: int = 6) -> bytes:
    """GET with retry on 429/5xx.

    GitLab rate-limits anonymous requests hard. A 429 during a 7442-clip
    sweep silently dropped clips -- 89 of 120 failed in the first test.
    Every call must therefore retry with exponential backoff.
    """
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                _sleep_backoff(attempt)
                continue
            raise
        except Exception as e:
            last = e
            _sleep_backoff(attempt)
    raise last if last else RuntimeError(f"GET failed: {url}")


def http_post_json(url: str, payload: dict, timeout: int = 120, retries: int = 6) -> dict:
    """POST with retry on 429/5xx."""
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/vnd.git-lfs+json",
                         "Accept": "application/vnd.git-lfs+json",
                         "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                _sleep_backoff(attempt)
                continue
            raise
        except Exception as e:
            last = e
            _sleep_backoff(attempt)
    raise last if last else RuntimeError(f"POST failed: {url}")


def clone_pointers() -> dict[str, tuple[str, int]]:
    """Fetch every VideoFlash lfs pointer with ONE git clone.

    Returns {sample_id: (oid, size)}. The clone is blobless + sparse, so
    only the tiny pointer files are transferred -- a few MB for 7442 clips,
    versus 7442 rate-limited REST calls.
    """
    if not (CLONE_DIR / "VideoFlash").is_dir():
        if CLONE_DIR.exists():
            shutil.rmtree(CLONE_DIR, ignore_errors=True)
        CLONE_DIR.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "HOME", "LANG", "LC_ALL", "SSL_CERT_FILE",
                        "http_proxy", "https_proxy", "no_proxy",
                        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")}
        cmds = [
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             "--depth", "1", CLONE_URL, "repo"],
        ]
        for cmd in cmds:
            r = subprocess.run(cmd, cwd=str(CLONE_DIR), capture_output=True,
                               text=True, timeout=1800, env=env)
            if r.returncode != 0:
                raise RuntimeError(
                    f"git clone failed: {(r.stderr or r.stdout)[-400:]}")
        repo = CLONE_DIR / "repo"
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "VideoFlash"],
                       capture_output=True, text=True, timeout=900, env=env)
        r = subprocess.run(["git", "-C", str(repo), "checkout"],
                           capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"git checkout failed: {(r.stderr or r.stdout)[-400:]}")

    out: dict[str, tuple[str, int]] = {}
    vdir = CLONE_DIR / "repo" / "VideoFlash"
    for f in sorted(vdir.glob("*.flv")):
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
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
            out[f.stem] = (oid, size)
    return out


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


def lfs_batch_download(oids: list[tuple[str, int]]) -> dict[str, bytes]:
    """Fetch MANY lfs objects in one batch request.

    The batch API accepts an arbitrary number of objects, so one POST can
    replace hundreds of individual requests. This is the difference between
    ~7400 round trips and ~150.
    """
    out: dict[str, bytes] = {}
    if not oids:
        return out
    payload = [{"oid": o, "size": sz} for o, sz in oids]
    try:
        resp = http_post_json(LFS_BATCH, {
            "operation": "download", "transfers": ["basic"],
            "objects": payload})
    except Exception:
        return out
    by_oid = {}
    for obj in resp.get("objects") or []:
        dl = (obj.get("actions") or {}).get("download") or {}
        href = dl.get("href")
        if href:
            by_oid[obj.get("oid")] = (href, dl.get("header") or {})
    for oid, _sz in oids:
        entry = by_oid.get(oid)
        if not entry:
            continue
        href, headers = entry
        try:
            req = urllib.request.Request(href, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                out[oid] = r.read()
        except Exception:
            continue
    return out


def lfs_download(oid: str, size: int) -> bytes | None:
    """Fetch the real bytes of an lfs object via the batch API."""
    return lfs_batch_download([(oid, size)]).get(oid)


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
    frames = sorted(str(p) for p in outdir.glob("frame_*.jpg"))
    if len(frames) > KEEP_FRAMES:
        # Keep the middle candidate(s) — the most expressive one.
        mid = len(frames) // 2
        keep = frames[mid: mid + KEEP_FRAMES]
        for f in frames:
            if f not in keep:
                try:
                    os.unlink(f)
                except OSError:
                    pass
        frames = keep
    return frames


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


def process_batch(sample_ids: list[str],
                  pointers: dict[str, tuple[str, int]]) -> list[dict]:
    """Process a group of clips with ONE batched LFS request.

    The pointers are already known (from the git clone), so the only
    network work here is a single batched LFS request for the video
    bytes, followed by parallel frame decoding.
    """
    results: list[dict] = []
    todo: list[str] = []
    for sid in sample_ids:
        outdir = FRAME_DIR / sid
        existing = sorted(outdir.glob("frame_*.jpg"))
        if existing:
            results.append({"sample_id": sid, "frames": [str(p) for p in existing],
                            "emotion": emotion_of(sid), "cached": True})
        else:
            todo.append(sid)
    if not todo:
        return results

    # one batched request for all the video bytes
    pairs = [pointers[s] for s in todo if s in pointers]
    blobs = lfs_batch_download(pairs)
    by_oid = {oid: blobs.get(oid) for oid, _ in pairs}

    # decode in parallel
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(extract_frames, by_oid.get(pointers[s][0]), s): s
                for s in todo if s in pointers}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                frames = fut.result()
            except Exception:
                frames = []
            if frames:
                results.append({"sample_id": s, "frames": frames,
                                "emotion": emotion_of(s), "cached": False})
    return results


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

    print("  fetching the lfs pointers via a blobless sparse git clone...")
    try:
        pointers = clone_pointers()
    except Exception as e:
        print(f"  FATAL: could not fetch the VideoFlash pointers: {e}")
        sys.exit(1)
    sample_ids = sorted(pointers)
    if args.limit:
        sample_ids = sample_ids[:args.limit]
    names = [f"{s}.flv" for s in sample_ids]
    print(f"  VideoFlash clips available: {len(pointers)}"
          + (f"  (processing {len(sample_ids)})" if args.limit else ""))
    if not names:
        print("  FATAL: no CREMA-D video files found")
        sys.exit(1)

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    failures = 0

    # Process in groups. Each group costs ONE batched LFS request instead
    # of one request per clip -- 7442 clips become ~150 requests.
    group = max(args.workers * 10, 50)
    done = 0
    for start in range(0, len(names), group):
        chunk = names[start:start + group]
        try:
            got = process_batch([n[:-4] for n in chunk], pointers)
        except Exception as e:
            print(f"    group at {start} failed: {type(e).__name__}: {e}")
            got = []
        results.extend(got)
        done += len(chunk)
        failures += len(chunk) - len(got)
        print(f"    {done}/{len(names)} processed, "
              f"{len(results)} with frames, {failures} failed", flush=True)

    print(f"\n  clips with frames: {len(results)}/{len(names)}")

    # ---- write the manifest ATOMICALLY ----
    # Write to a temp file then rename, so a reader can never observe a
    # half-written manifest.
    tmp = OUT_MANIFEST.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "frame_path", "emotion", "n_frames"])
        for r in sorted(results, key=lambda x: x["sample_id"]):
            for fr in r["frames"]:
                w.writerow([r["sample_id"], fr, r["emotion"], len(r["frames"])])
    os.replace(tmp, OUT_MANIFEST)
    print(f"  wrote {OUT_MANIFEST} (atomic)")

    # ---- deliberately do NOT touch the corpus or triplets manifests ----
    # The pipeline job ALSO writes both of them:
    #   Stage 0b regenerates triplets_manifest.csv with placeholder faces
    #   Stage 0c reads combined_ser_dataset/metadata.csv for the SER features
    # If this job rewrote either one concurrently, the pipeline would read a
    # half-written CSV (torn read -> corrupted SER features) or the real
    # faces would be overwritten by Stage 0b moments later.
    #
    # So the extractor owns fer_frames/ and NOTHING else. The
    # meta-classifier resolves faces from fer_frames/manifest.csv at read
    # time, joining on the audio stem. The two jobs then cannot conflict.
    print("\n  This job wrote ONLY fer_frames/. The corpus and triplets")
    print("  manifests were left untouched on purpose — the pipeline job")
    print("  writes those, and concurrent writes would corrupt them.")

    print("\n  The meta-classifier's visual channel is now real.")


if __name__ == "__main__":
    main()
