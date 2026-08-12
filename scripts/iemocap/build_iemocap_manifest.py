"""
build_iemocap_manifest.py — Build the 4-class IEMOCAP manifest.

Source files (verified against the IEMOCAP_full_release distribution):
  - dialog/transcriptions/Ses<N><F|M>_impro<script><digits>.txt
        Per-utterance text + timecode, no emotion label.
        Format: "Ses01F_impro01_F000 [6.2901-8.2357]: Excuse me."
  - dialog/EmoEvaluation/Ses<N><F|M>_impro<script><digits>.txt
        Per-utterance CONSENSUS categorical (3-letter code) + 3D V/A + per-annotator
        breakdown. THIS is the gold file — one file per dialog, all data inside.
        Format per utterance (8 lines):
            [6.2901-8.2357]       Ses01F_impro01_F000     neu     [2.5000, 2.5000, 2.5000]
            C-E2:   Neutral;        ()
            C-E3:   Neutral;        ()
            C-E4:   Neutral;        ()
            C-F1:   Neutral;        (curious)
            A-E3:   val 3; act 2; dom  2;   ()
            A-E4:   val 2; act 3; dom  3;   (mildly aggravated but staying polite, attitude)
            A-F1:   val 3; act 2; dom  1;   ()

4-class benchmark (matches MemoCMT, 2025):
  angry, happy (happy + excited merged), neutral, sad
  Everything else is dropped.

Output: data/iemocap/manifest.csv
  Columns: utterance_id, session, speaker, dialog, emotion, valence, arousal,
           dominance, wav_path, start_time, end_time, transcript
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# 3-letter consensus code -> 4-class emotion label
CONSENSUS_TO_4CLASS = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",       # merge: excited → happy
    "sad": "sad",
    "neu": "neutral",
    # Dropped (not in 4-class):
    "fru": None,
    "sur": None,
    "fea": None,
    "dis": None,
    "oth": None,
    "xxx": None,           # no consensus
}


# Parses the gold per-dialog file line:
#   [6.2901-8.2357]       Ses01F_impro01_F000     neu     [2.5000, 2.5000, 2.5000]
GOLD_LINE_RE = re.compile(
    r"^\[\s*(?P<start>\d+\.\d+)\s*-\s*(?P<end>\d+\.\d+)\s*\]\s+"
    r"(?P<utt>Ses\d{2}[FM]_(?:impro|script)\w*_\w+)\s+"
    r"(?P<code>ang|hap|exc|sad|neu|fru|sur|fea|dis|oth|xxx)\s+"
    r"\[(?P<v>\d+\.\d+),\s*(?P<a>\d+\.\d+),\s*(?P<d>\d+\.\d+)\s*\]\s*$"
)


def parse_gold_file(gold_path: Path) -> dict:
    """
    Returns {utterance_id: {start, end, code, v, a, d}}.
    Drops utterances whose consensus code is not in the 4-class set.
    """
    rows = {}
    for line in gold_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.rstrip()
        m = GOLD_LINE_RE.match(line)
        if not m:
            continue
        code = m.group("code")
        if CONSENSUS_TO_4CLASS.get(code) is None:
            continue
        utt = m.group("utt")
        # If duplicate (shouldn't happen, but be safe), keep first.
        if utt in rows:
            continue
        rows[utt] = {
            "start": float(m.group("start")),
            "end": float(m.group("end")),
            "code": code,
            "emotion": CONSENSUS_TO_4CLASS[code],
            "v": float(m.group("v")),
            "a": float(m.group("a")),
            "d": float(m.group("d")),
            "dialog": gold_path.stem,
        }
    return rows


# Parses the per-utterance transcript line:
#   Ses01F_impro01_F000 [6.2901-8.2357]: Excuse me.
TRANSCRIPT_LINE_RE = re.compile(
    r"^(?P<utt>Ses\d{2}[FM]_(?:impro|script)\w*_\w+)\s+"
    r"\[(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)\]:\s*"
    r"(?P<text>.*?)\s*$"
)


def parse_transcript_file(transcript_path: Path) -> dict:
    """Returns {utterance_id: text}."""
    rows = {}
    for line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.rstrip()
        m = TRANSCRIPT_LINE_RE.match(line)
        if not m:
            continue
        rows[m.group("utt")] = m.group("text").strip()
    return rows


def session_id_from_utt(utt: str) -> str:
    m = re.match(r"^Ses(\d{2})[FM]_", utt)
    return f"Session{int(m.group(1))}" if m else "?"


def speaker_from_utt(utt: str) -> str:
    m = re.match(r"^Ses\d{2}([FM])_", utt)
    return m.group(1) if m else "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--debug", action="store_true",
                        help="Print per-file parse statistics")
    args = parser.parse_args()

    sessions = sorted(args.iemocap_root.glob("Session*"))
    if not sessions:
        print(f"ERROR: no Session* directories under {args.iemocap_root}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sessions)} sessions: {[s.name for s in sessions]}")
    all_gold = {}
    all_text = {}
    for session_dir in sessions:
        emo_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"
        if not emo_dir.exists():
            print(f"  {session_dir.name}: EmoEvaluation missing, skipping")
            continue
        if not trans_dir.exists():
            print(f"  {session_dir.name}: transcriptions missing, skipping")
            continue
        n_gold_files = 0
        n_gold_utts = 0
        n_4class_utts = 0
        n_trans_utts = 0
        for gold in sorted(emo_dir.glob("Ses*.txt")):
            if "_cat" in gold.stem or "_atr" in gold.stem:
                continue   # skip per-annotator files
            n_gold_files += 1
            parsed = parse_gold_file(gold)
            n_gold_utts += len(parsed)
            n_4class_utts += sum(1 for v in parsed.values()
                                 if v["emotion"] in ("angry", "happy", "neutral", "sad"))
            all_gold.update(parsed)
        for trans in sorted(trans_dir.glob("Ses*.txt")):
            parsed = parse_transcript_file(trans)
            n_trans_utts += len(parsed)
            all_text.update(parsed)
        if args.debug:
            print(f"  {session_dir.name}: {n_gold_files} gold files, "
                  f"{n_gold_utts} total utts, {n_4class_utts} in 4-class, "
                  f"{n_trans_utts} transcripts parsed")
        else:
            print(f"  {session_dir.name}: {n_4class_utts} 4-class utts, "
                  f"{n_trans_utts} transcripts")

    # Inner join on utterance_id
    rows = []
    matched_text = 0
    missing_wav = 0
    for utt, g in all_gold.items():
        session = session_id_from_utt(utt)
        speaker = speaker_from_utt(utt)
        wav_path = f"{args.iemocap_root}/{session}/dialog/wav/{g['dialog']}.wav"
        if not Path(wav_path).exists():
            missing_wav += 1
            continue
        text = all_text.get(utt, "")
        if text:
            matched_text += 1
        # V/A range: 1.0-5.0; normalize to 0.0-1.0 for the model
        v_norm = (g["v"] - 1.0) / 4.0
        a_norm = (g["a"] - 1.0) / 4.0
        d_norm = (g["d"] - 1.0) / 4.0
        rows.append({
            "utterance_id": utt,
            "session": session,
            "speaker": speaker,
            "dialog": g["dialog"],
            "emotion": g["emotion"],
            "valence": round(v_norm, 4),
            "arousal": round(a_norm, 4),
            "dominance": round(d_norm, 4),
            "wav_path": wav_path,
            "start_time": g["start"],
            "end_time": g["end"],
            "transcript": text,
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} utterances to {args.out}")
    print(f"  {matched_text} have transcripts matched")
    print(f"  {missing_wav} skipped (wav missing)")
    print()
    print("Class distribution:")
    print(df["emotion"].value_counts().to_string())
    print()
    print("Per-session distribution:")
    print(df["session"].value_counts().sort_index().to_string())
    print()
    print("V/A statistics (normalized 0-1):")
    print(df[["valence", "arousal", "dominance"]].describe().to_string())


if __name__ == "__main__":
    main()
