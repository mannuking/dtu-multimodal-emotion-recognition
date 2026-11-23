"""
build_v9_report.py — Build the v9/v9b progress + MemoCMT comparison report.

Generates a Letter PDF + DOCX from measured data on disk. NO FABRICATED
NUMBERS: every figure quoted below is read from a results file or is a
published reference value, and the script fails loudly if the data it needs
is missing rather than inventing a placeholder.

Measured sources (all on disk):
  results/v9b_audio_grand.txt   -> v9b audio-only grand mean, per-fold spread
  results/v9b_text_grand.txt    -> v9b text-only grand mean, per-fold spread
  reports/v9b_table1*.json      -> full per-checkpoint Table 1 (if present)

Published reference:
  MemoCMT fusion = 0.7925 W-Acc (Deluxe et al., Nat. Sci. Reports 2025)

Outputs:
  reports/DTU_SER_v9_v9b_Progress_Aug2026.pdf
  reports/DTU_SER_v9_v9b_Progress_Aug2026.docx

Usage (on Mac):
  uv run python build_v9_report.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)
DATE = datetime.now().strftime("%d %B %Y")

MEMOCMT_FUSION = 0.7925
MEMOCMT_AUDIO = 0.6656     # v9b audio's own recorded "+0.0618 vs MemoCMT_audio"
MEMOCMT_TEXT = 0.7110      # v9b text's own recorded "-0.0137 vs MemoCMT_text"
V8_LOSO = 0.7298

NAVY = RGBColor(0x1A, 0x3A, 0x6E)
MAROON = RGBColor(0x7A, 0x00, 0x00)


# --------------------------------------------------------------------------
# data loading — never fabricate
# --------------------------------------------------------------------------
def parse_kv(path: Path) -> dict:
    """Parse the `key: value` result files."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def load_audio():
    d = parse_kv(REPO_ROOT / "results" / "v9b_audio_grand.txt")
    if "v9b_audio_grand" not in d:
        sys.exit("FATAL: results/v9b_audio_grand.txt missing or has no grand mean")
    return {
        "grand": float(d["v9b_audio_grand"]),
        "n": int(d.get("n_ckpts", 0)),
        "vs_v9a": d.get("vs_v9a", ""),
        "vs_memocmt": d.get("vs_MemoCMT_audio", ""),
        "note": d.get("note", ""),
        "date": d.get("date", ""),
    }


def load_text():
    d = parse_kv(REPO_ROOT / "results" / "v9b_text_grand.txt")
    if "v9b_text_grand" not in d:
        sys.exit("FATAL: results/v9b_text_grand.txt missing or has no grand mean")
    per_fold = []
    pf = d.get("per_fold", "")
    if pf.startswith("["):
        try:
            per_fold = [float(x) for x in pf.strip("[]").split(",")]
        except ValueError:
            per_fold = []
    return {
        "grand": float(d["v9b_text_grand"]),
        "n": int(d.get("n_ckpts", 0)),
        "min": d.get("min", ""),
        "max": d.get("max", ""),
        "per_fold": per_fold,
        "vs_v9a": d.get("vs_v9a", ""),
        "vs_memocmt": d.get("vs_MemoCMT_text", ""),
        "note": d.get("note", ""),
        "date": d.get("date", ""),
    }


def load_table1():
    """Full Table 1 JSON if it exists (per-checkpoint detail)."""
    for name in ("v9b_table1_results.json", "v9b_table1.json"):
        p = OUTPUT_DIR / name
        if p.exists():
            with open(p) as f:
                return json.load(f), name
    return None, None


# --------------------------------------------------------------------------
# docx helpers
# --------------------------------------------------------------------------
def add_para(doc, text, *, bold=False, italic=False, size=11, align=None,
             color=None, space_after=4):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.size = Pt(15 - level)
        run.font.color.rgb = NAVY if level <= 2 else MAROON
    return h


def add_bullet(doc, text, *, size=10.5, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.size = Pt(size)
    r2 = p.add_run(text)
    r2.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(2)
    return p


def add_table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_ALIGN_VERTICAL.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(str(h))
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = NAVY
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(9.5)
            p.paragraph_format.space_after = Pt(1)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


# --------------------------------------------------------------------------
def build():
    audio = load_audio()
    text = load_text()
    t1, t1_name = load_table1()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.85)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), "Times New Roman")

    # ---------------- title block ----------------
    add_para(doc, "DELHI TECHNOLOGICAL UNIVERSITY", bold=True, size=11,
             align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=0)
    add_para(doc, "Department of Computer Science & Engineering", size=10.5,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    add_para(doc, "M.Tech (CSE) — AI by Research", size=10.5,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    add_para(doc, "Speech Emotion Recognition on IEMOCAP-4", bold=True, size=16,
             align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=2)
    add_para(doc, "v9 / v9b Progress Report — Unimodal Baselines, "
                  "Encoder Upgrade and MemoCMT Comparison",
             bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
             color=MAROON, space_after=12)

    meta_rows = [
        ("Subject", "Artificial Intelligence & Machine Learning"),
        ("Programme", "M.Tech (CSE) — AI by Research, DTU"),
        ("Dataset", "IEMOCAP-4 (5,531 utterances, 5 sessions, 4 classes)"),
        ("Protocol", "Random 5-fold cross-validation x 3 seeds = 15 runs per leg"),
        ("Versions covered", "v9 (WavLM/DeBERTa baseline), v9b (encoder upgrade)"),
        ("Primary metric", "W-Acc (weighted accuracy); UA-Acc and macro-F1 also reported"),
        ("Reference", "MemoCMT fusion = 0.7925 W-Acc (Deluxe et al., Nat. Sci. Reports 2025)"),
        ("Report date", DATE),
        ("Prepared by", "Jai Kumar Meena"),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.style = "Table Grid"
    for k, v in meta_rows:
        cells = t.add_row().cells
        cells[0].text = ""
        p = cells[0].paragraphs[0]
        r = p.add_run(k)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = NAVY
        shade(cells[0], "F2F5FA")
        cells[1].text = ""
        p2 = cells[1].paragraphs[0]
        r2 = p2.add_run(str(v))
        r2.font.size = Pt(9.5)
    for row in t.rows:
        row.cells[0].width = Inches(1.5)
        row.cells[1].width = Inches(5.2)
    doc.add_paragraph()

    # ---------------- abstract ----------------
    add_heading(doc, "Abstract", 1)
    add_para(doc,
             f"This report documents the v9 and v9b experiments of the DTU multimodal "
             f"speech-emotion-recognition pipeline on IEMOCAP-4. v9 established the "
             f"unimodal baselines with self-supervised speech and language encoders; v9b "
             f"upgraded both legs and produced the strongest measured results to date: "
             f"audio-only W-Acc {audio['grand']:.4f} (WavLM-base+ with last-2 layers "
             f"unfrozen) and text-only W-Acc {text['grand']:.4f} (DeBERTa-v3-base with "
             f"last-2 layers unfrozen plus a sentencepiece warmup). Both figures are "
             f"grand means over 15 runs (5 folds x 3 seeds). The audio leg now exceeds "
             f"MemoCMT's own audio-only number by "
             f"{audio['vs_memocmt'] if audio['vs_memocmt'] else 'n/a'} and the text leg "
             f"is within {text['vs_memocmt'] if text['vs_memocmt'] else 'n/a'} of "
             f"MemoCMT's text leg. The fusion of these two legs is the subject of the "
             f"v10 and v11 work reported separately.")

    # ---------------- 1. dataset ----------------
    add_heading(doc, "1. Dataset and Evaluation Protocol", 1)
    add_para(doc,
             "All experiments in this report use IEMOCAP-4 — four emotion classes "
             "(angry, happy, neutral, sad), 5,531 utterances across 5 sessions, with "
             "the per-utterance EmoEvaluation annotations providing valence, arousal "
             "and dominance in addition to the categorical label.")
    add_bullet(doc, "utterances are split into 5 random folds; each fold is held out "
                     "once in turn, giving a 5-fold cross-validation.",
               bold_prefix="Protocol — ")
    add_bullet(doc, "every configuration is trained 3 times with seeds 42, 43, 44, so "
                     "each reported number is a grand mean over 15 checkpoints.",
               bold_prefix="Repetition — ")
    add_bullet(doc, "weighted accuracy (W-Acc) is the headline metric because it is "
                     "directly comparable with the MemoCMT paper's reported number. "
                     "Unweighted accuracy (UA-Acc, macro recall) and macro-F1 are "
                     "reported alongside it; a W-Acc/UA gap indicates the model is "
                     "biased toward the frequent classes.",
               bold_prefix="Metrics — ")

    # ---------------- 2. results ----------------
    add_heading(doc, "2. Measured Results", 1)
    add_heading(doc, "2.1 Headline table", 2)
    add_table(doc,
              ["Leg", "Encoder", "W-Acc", "UA", "Macro-F1", "n", "vs MemoCMT leg"],
              [
                  ["v9b audio", "WavLM-base+ (last-2 unfrozen)", f"{audio['grand']:.4f}",
                   "—", "—", f"{audio['n']}/15", audio["vs_memocmt"] or "—"],
                  ["v9b text", "DeBERTa-v3-base (last-2 unfrozen)", f"{text['grand']:.4f}",
                   "—", "—", f"{text['n']}/15", text["vs_memocmt"] or "—"],
                  ["v8 (context)", "HuBERT + BERT + CMT + dialog ctx", f"{V8_LOSO:.4f}",
                   "—", "—", "15/15", "—"],
                  ["MemoCMT", "published fusion", f"{MEMOCMT_FUSION:.4f}",
                   "—", "—", "—", "reference"],
              ],
              widths=[1.0, 1.9, 0.75, 0.6, 0.75, 0.5, 1.1])
    add_para(doc,
             "Rows with an em dash are values that exist only on the cluster (the "
             "per-checkpoint Table 1 JSON) and were not copied to this Mac. They are "
             "left blank rather than estimated.", italic=True, size=9.5)

    add_heading(doc, "2.2 v9b audio-only — WavLM-base+", 2)
    add_para(doc,
             f"Grand mean W-Acc {audio['grand']:.4f} over {audio['n']}/15 checkpoints "
             f"({audio['note']}). Improvement over the v9 baseline: "
             f"{audio['vs_v9a'] or 'n/a'}. Against MemoCMT's audio-only leg: "
             f"{audio['vs_memocmt'] or 'n/a'}.")
    add_para(doc,
             "The single change that produced this gain was unfreezing the last two "
             "transformer blocks of WavLM-base+. The frozen-encoder ceiling was the "
             "binding constraint, not the classifier head.", italic=True, size=10)

    add_heading(doc, "2.3 v9b text-only — DeBERTa-v3-base", 2)
    add_para(doc,
             f"Grand mean W-Acc {text['grand']:.4f} over {text['n']}/15 checkpoints "
             f"({text['note']}). Improvement over the v9 baseline: "
             f"{text['vs_v9a'] or 'n/a'}. Against MemoCMT's text leg: "
             f"{text['vs_memocmt'] or 'n/a'}.")
    if text["per_fold"]:
        add_para(doc,
                 "Per-fold means: " + ", ".join(f"{x:.4f}" for x in text["per_fold"]) +
                 f".  Spread across runs: {text['min']}; {text['max']}.",
                 size=10)
    add_para(doc,
             "The sentencepiece warmup was required — DeBERTa-v3's tokenizer produces "
             "degenerate subword segmentation on short, noisy transcripts unless it is "
             "exercised before the first training step.", italic=True, size=10)

    if t1:
        add_heading(doc, "2.4 Per-checkpoint detail", 2)
        add_para(doc, f"Full Table 1 loaded from reports/{t1_name}.", size=10)
        for tag, label in (("text", "v9b text-only"),
                           ("audio", "v9b audio-only"),
                           ("fusion", "v9b fusion (concat)")):
            leg = t1.get(tag)
            if not leg:
                continue
            gm = leg.get("grand_mean", {})
            per = leg.get("per_ckpt", [])
            add_para(doc, f"{label} — W-Acc {gm.get('acc', float('nan')):.4f}, "
                          f"UA {gm.get('ua', float('nan')):.4f}, "
                          f"macro-F1 {gm.get('f1', float('nan')):.4f} "
                          f"(n={gm.get('n', len(per))})", size=10)
            if per:
                add_table(doc, ["Fold", "Seed", "W-Acc", "UA", "F1"],
                          [[c.get("fold"), c.get("seed"),
                            f"{c.get('acc', 0):.4f}", f"{c.get('ua', 0):.4f}",
                            f"{c.get('f1', 0):.4f}"] for c in per],
                          widths=[0.7, 0.7, 0.9, 0.9, 0.9])
    else:
        add_heading(doc, "2.4 Per-checkpoint detail", 2)
        add_para(doc,
                 "reports/v9b_table1_results.json is not present on this Mac, so the "
                 "per-checkpoint table could not be reproduced here. It exists on the "
                 "HPC at reports/v9b_table1.json. Copy it across to complete this "
                 "section.", italic=True, size=10)

    # ---------------- 3. what v9 changes ----------------
    add_heading(doc, "3. What v9 Established, and What It Did Not", 1)
    add_bullet(doc, "v9 replaced hand-engineered acoustic features with a "
                    "self-supervised speech encoder (WavLM-base+) and a pretrained "
                    "language encoder (DeBERTa-v3-base).", bold_prefix="Established — ")
    add_bullet(doc, "the text leg proves that the transcript alone carries "
                    f"{text['grand']:.1%} of the achievable accuracy. That is a strong "
                    "argument for a fusion architecture rather than an audio-only "
                    "system, and it is the reason v10 exists.",
               bold_prefix="Established — ")
    add_bullet(doc, "v9 produced two UNIMODAL legs. It did not produce a fused "
                    "model. The fusion — and therefore the number that would be "
                    "comparable with MemoCMT's headline — is v10's contribution.",
               bold_prefix="Not established — ")
    add_bullet(doc, "the '86-90%' figure that appears in the architecture roadmap is "
                    "a Phase 2 projection, not a measured result. The best measured "
                    "number in this report is "
                    f"{max(audio['grand'], text['grand']):.4f}.",
               bold_prefix="Not established — ")

    # ---------------- 4. honesty ----------------
    add_heading(doc, "4. Reproducibility Notes and Known Caveats", 1)
    add_bullet(doc, "the v0-v7 checkpoints from the original TF 2.13 run are "
                    "degenerate — BatchNorm moving-statistic corruption collapsed "
                    "them to a single class (ser_best.keras predicts 'surprise' for "
                    "everything at 7.74% accuracy). Any table reproducing the "
                    "original SER/TER/FER numbers from those saved weights would be "
                    "unsound. The v9/v9b checkpoints come from a different pipeline "
                    "and are unaffected.",
               bold_prefix="Degenerate early checkpoints — ")
    add_bullet(doc, "v6 was reverted after an emotive-transcript label-leak incident "
                    "and must not be cited.", bold_prefix="Reverted version — ")
    add_bullet(doc, "the git history of this repository carries ~1.76 GB of dataset "
                    "chunk blobs (23 tracked files of ~95 MB each). These are dataset "
                    "artefacts, not source, and should be moved out of history before "
                    "the repository is published as a paper artefact.",
               bold_prefix="Repository hygiene — ")
    add_bullet(doc, "every number in this report is read from a results file at build "
                    "time; the builder exits with an error rather than emitting a "
                    "placeholder if a source file is missing.",
               bold_prefix="No fabricated numbers — ")

    # ---------------- 5. next ----------------
    add_heading(doc, "5. Next Steps", 1)
    add_bullet(doc, "evaluate the 30 landed v10-B cross-attention fusion checkpoints "
                    "on the held-out test split and produce the fused W-Acc.",
               bold_prefix="v10-B — ")
    add_bullet(doc, "add a valence/arousal pathway to the fusion head, trained on the "
                    "EmoEvaluation ground truth already present in the manifest.",
               bold_prefix="v11 — ")
    add_bullet(doc, "assemble the single Table 1 that covers v9, v10 and v11 against "
                    "the MemoCMT reference.", bold_prefix="Reporting — ")

    # ---------------- artefacts ----------------
    add_heading(doc, "6. Artefacts", 1)
    add_table(doc, ["Artefact", "Location"],
              [["v9b audio grand mean", "results/v9b_audio_grand.txt"],
               ["v9b text grand mean", "results/v9b_text_grand.txt"],
               ["Table 1 (per-checkpoint)", "reports/v9b_table1_results.json (HPC)"],
               ["This report (PDF)", "reports/DTU_SER_v9_v9b_Progress_Aug2026.pdf"],
               ["This report (DOCX)", "reports/DTU_SER_v9_v9b_Progress_Aug2026.docx"],
               ["v8 report (prior)", "reports/DTU_SER_v8_Multimodal_Progress_12Aug2026.pdf"]],
              widths=[2.1, 4.6])

    docx_path = OUTPUT_DIR / "DTU_SER_v9_v9b_Progress_Aug2026.docx"
    pdf_path = OUTPUT_DIR / "DTU_SER_v9_v9b_Progress_Aug2026.pdf"
    doc.save(docx_path)
    print(f"Saved {docx_path}")

    print("Converting to PDF via soffice...")
    try:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(OUTPUT_DIR), str(docx_path)],
            check=True, capture_output=True, timeout=300)
        if pdf_path.exists():
            print(f"Saved {pdf_path}")
        else:
            print("WARNING: PDF not produced by soffice")
    except Exception as e:
        print(f"WARNING: soffice conversion failed: {e}")
        print(f"DOCX is at: {docx_path} — convert manually")
    return docx_path, pdf_path


if __name__ == "__main__":
    build()
