"""
build_iemocap_manifest.py — Build the 4-class IEMOCAP manifest from the dialog
transcripts + EmoEvaluation V/A scores.

Output schema (data/iemocap/manifest.csv):
    utterance_id, session, speaker, dialog, emotion, valence, arousal,
    dominance, wav_path, start_time, end_time, transcript

4-class filter (matches MemoCMT's published convention):
    angry, happy (= happy + excited merged), neutral, sad

Usage (on HPC):
    uv run python scripts/iemocap/build_iemocap_manifest.py \
        --iemocap-root data/iemocap \
        --out data/iemocap/manifest.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# 4-class mapping per MemoCMT's published convention
EMOTION_MAP = {
    "angry": "angry",
    "anger": "angry",
    "happy": "happy",
    "happiness": "happy",
    "excited": "happy",       # merged with happy
    "excitement": "happy",     # merged with happy
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    # Dropped classes (not in 4-class benchmark)
    # fear, surprise, disgust, frustration, other
}

# Parses the dialog-level transcript line format:
#   Ses01F_impro01_F000 [006.0600 - 009.5600]:  Excuse me.       Anger
DIALOG_LINE_RE = re.compile(
    r"^(?P<utt>Ses\d{2}[FM]_(?:impro|script)\w*_\w+)\s+"
    r"\[(?P<start>\d+\.\d+)\s*-\s*(?P<end>\d+\.\d+)\]:\s*"
    r"(?P<text>.*?)\s+(?P<emo>[A-Za-z]+)\s*$"
)

# EmoEvaluation per-utterance format:
#   Ses01F_impro01_F000:  V 2.00  A 3.00  D 2.50
EMO_LINE_RE = re.compile(
    r"^(?P<utt>Ses\d{2}[FM]_(?:impro|script)\w*_\w+):\s+"
    r"V\s+(?P<v>\d+\.\d+)\s+"
    r"A\s+(?P<a>\d+\.\d+)\s+"
    r"D\s+(?P<d>\d+\.\d+)\s*$"
)


def parse_transcripts(session_dir: Path) -> dict:
    """Returns {utterance_id: (text, emotion, start, end, dialog_id)}"""
    rows = {}
    transcripts_dir = session_dir / "dialog" / "transcriptions"
    for txt in sorted(transcripts_dir.glob("Ses*.txt")):
        dialog_id = txt.stem
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            m = DIALOG_LINE_RE.match(line)
            if not m:
                continue
            utt = m.group("utt")
            start = float(m.group("start"))
            end = float(m.group("end"))
            text = m.group("text").strip()
            emo_raw = m.group("emo").strip().lower()
            rows[utt] = {
                "text": text,
                "emotion_raw": emo_raw,
                "emotion": EMOTION_MAP.get(emo_raw, None),
                "start": start,
                "end": end,
                "dialog": dialog_id,
            }
    return rows


def parse_va_categorical(session_dir: Path) -> dict:
    """
    Returns {utterance_id: (valence, arousal, dominance)} averaged across annotators.
    Each annotator has a separate file; we average all available.
    """
    emo_dir = session_dir / "dialog" / "EmoEvaluation" / "Categorical"
    if not emo_dir.exists():
        return {}
    by_utt = defaultdict(list)
    for f in sorted(emo_dir.glob("*_cat.txt")):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "Ses01F_impro01_F000: angry; V 2; A 3; D 2"
            # Or: "Ses01F_impro01_F000: angry excited 2.5 3.0 2.0"
            # Be lenient — parse what we can.
            m = re.match(r"^(\S+):\s*(.*)$", line)
            if not m:
                continue
            utt = m.group(1)
            rest = m.group(2)
            v = a = d = None
            # Try to find V, A, D values
            for vm in re.finditer(r"\bV\s*([0-9.]+)", rest):
                v = float(vm.group(1))
                break
            for am in re.finditer(r"\bA\s*([0-9.]+)", rest):
                a = float(am.group(1))
                break
            for dm in re.finditer(r"\bD\s*([0-9.]+)", rest):
                d = float(dm.group(1))
                break
            # If we didn't find structured V/A/D, try the legacy format
            # (last 3 numbers in the line)
            if v is None or a is None or d is None:
                nums = re.findall(r"[-+]?\d+\.\d+", rest)
                if len(nums) >= 3:
                    try:
                        v = float(nums[-3])
                        a = float(nums[-2])
                        d = float(nums[-1])
                    except ValueError:
                        continue
            if v is not None and a is not None and d is not None:
                by_utt[utt].append((v, a, d))
    out = {}
    for utt, vals in by_utt.items():
        vs = [v[0] for v in vals]
        aas = [v[1] for v in vals]
        ds = [v[2] for v in vals]
        # IEMOCAP V/A range is 1-5; normalize to [0, 1] for the model
        out[utt] = (sum(vs) / len(vs), sum(aas) / len(aas), sum(ds) / len(ds))
    return out


def session_id_from_utt(utt: str) -> str:
    m = re.match(r"^Ses(\d{2})[FM]_", utt)
    if not m:
        return "?"
    return f"Session{int(m.group(1))}"


def speaker_from_utt(utt: str) -> str:
    m = re.match(r"^Ses\d{2}([FM])_", utt)
    return m.group(1) if m else "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    sessions = sorted(args.iemocap_root.glob("Session*"))
    if not sessions:
        print(f"ERROR: no Session* directories found under {args.iemocap_root}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sessions)} sessions: {[s.name for s in sessions]}")

    all_transcripts = {}
    all_va = {}
    for session_dir in sessions:
        t = parse_transcripts(session_dir)
        v = parse_va_categorical(session_dir)
        print(f"  {session_dir.name}: {len(t)} transcripts, {len(v)} V/A annotations")
        all_transcripts.update(t)
        all_va.update(v)

    # Join: for each utterance, get transcript + V/A if available
    rows = []
    for utt, t in all_transcripts.items():
        if t["emotion"] is None:
            continue  # skip utterances not in the 4-class set
        session = session_id_from_utt(utt)
        speaker = speaker_from_utt(utt)
        wav_path = f"{args.iemocap_root}/{session}/dialog/wav/{t['dialog']}.wav"
        if not Path(wav_path).exists():
            # skip if wav missing
            continue
        va = all_va.get(utt, (None, None, None))
        v, a, d = va
        # Normalize V/A from 1-5 to 0-1 if present
        if v is not None:
            v = (v - 1.0) / 4.0
        if a is not None:
            a = (a - 1.0) / 4.0
        if d is not None:
            d = (d - 1.0) / 4.0
        rows.append({
            "utterance_id": utt,
            "session": session,
            "speaker": speaker,
            "dialog": t["dialog"],
            "emotion": t["emotion"],
            "valence": v,
            "arousal": a,
            "dominance": d,
            "wav_path": wav_path,
            "start_time": t["start"],
            "end_time": t["end"],
            "transcript": t["text"],
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} utterances to {args.out}")
    print(f"\nClass distribution:")
    print(df["emotion"].value_counts().to_string())
    print(f"\nV/A coverage: {df['valence'].notna().sum()} / {len(df)} utterances have V/A")
    print(f"\nSession distribution:")
    print(df["session"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
