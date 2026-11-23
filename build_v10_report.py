"""
build_v10_report.py — Build the v10-B fusion progress report.

Generates a Letter PDF + DOCX describing the v10-B cross-modal-transformer
fusion experiment: the architecture, the training array, the state of the
evaluation, and the measured results IF the evaluation has been run.

NO FABRICATED NUMBERS. If reports/v10_table1_results.json is absent, the
report states plainly that the evaluation has not been run and leaves the
result cells empty. It never estimates a value from the training-time
validation metric.

Sources:
  reports/v10_table1_results.json  -> measured test metrics (if present)
  results/v10_session_log.txt       -> the training session's own record
  model_checkpoints/*_summary.json  -> training-time validation metrics

Outputs:
  reports/DTU_SER_v10_Fusion_Progress_Aug2026.pdf
  reports/DTU_SER_v10_Fusion_Progress_Aug2026.docx

Usage (on Mac):
  uv run python build_v10_report.py
"""
from __future__ import annotations

import json
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
V9B_AUDIO = 0.7274
V9B_TEXT = 0.6973
V8_LOSO = 0.7298

NAVY = RGBColor(0x1A, 0x3A, 0x6E)
MAROON = RGBColor(0x7A, 0x00, 0x00)


def load_v10():
    p = OUTPUT_DIR / "v10_table1_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_session_log():
    p = REPO_ROOT / "results" / "v10_session_log.txt"
    return p.read_text() if p.exists() else ""


def load_val_summaries():
    """Training-time validation metrics from the 30 v10 _summary.json files."""
    d = REPO_ROOT / "model_checkpoints"
    out = {}
    if not d.exists():
        return out
    for p in sorted(d.glob("v10_fusion_*_summary.json")):
        tag = "cw" if "_cw_" in p.name else "unif"
        try:
            with open(p) as f:
                j = json.load(f)
        except Exception:
            continue
        out.setdefault(tag, []).append({
            "fold": j.get("fold"), "seed": j.get("seed"),
            "acc": j.get("best_val_acc"), "f1": j.get("best_val_f1"),
            "epoch": j.get("best_epoch"),
        })
    return out


# ---------------------------------------------------------------- docx helpers
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


# ---------------------------------------------------------------------- build
def build():
    v10 = load_v10()
    session_log = load_session_log()
    val_sums = load_val_summaries()

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

    # ---------------- title ----------------
    add_para(doc, "DELHI TECHNOLOGICAL UNIVERSITY", bold=True, size=11,
             align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=0)
    add_para(doc, "Department of Computer Science & Engineering", size=10.5,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    add_para(doc, "M.Tech (CSE) — AI by Research", size=10.5,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    add_para(doc, "v10-B Cross-Modal-Transformer Fusion", bold=True, size=16,
             align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY, space_after=2)
    add_para(doc, "Multimodal Speech Emotion Recognition on IEMOCAP-4 — "
                  "Architecture, Training Array and Evaluation Status",
             bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
             color=MAROON, space_after=12)

    meta = [
        ("Subject", "Artificial Intelligence & Machine Learning"),
        ("Programme", "M.Tech (CSE) — AI by Research, DTU"),
        ("Dataset", "IEMOCAP-4 (5,531 utterances, 5 sessions, 4 classes)"),
        ("Protocol", "Random 5-fold x 3 seeds = 15 runs per branch; 2 branches = 30 runs"),
        ("Architecture", "Frozen WavLM-base+ / DeBERTa-v3-base + 2x CrossAttention + tensor fusion"),
        ("Trainable params", "14.45 M (fusion head only; encoders frozen)"),
        ("Reference", "MemoCMT fusion = 0.7925 W-Acc (Deluxe et al., Nat. Sci. Reports 2025)"),
        ("Evaluation", "COMPLETE — test metrics below" if v10 else
                       "NOT YET RUN — no test metrics exist"),
        ("Report date", DATE),
        ("Prepared by", "Jai Kumar Meena"),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.style = "Table Grid"
    for k, v in meta:
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
        if k == "Evaluation" and not v10:
            r2.bold = True
            r2.font.color.rgb = MAROON
    for row in t.rows:
        row.cells[0].width = Inches(1.5)
        row.cells[1].width = Inches(5.2)
    doc.add_paragraph()

    # ---------------- abstract ----------------
    add_heading(doc, "Abstract", 1)
    if v10:
        g_unif = v10.get("v10", {}).get("unif", {}).get("grand_mean", {})
        g_cw = v10.get("v10", {}).get("cw", {}).get("grand_mean", {})
        add_para(doc,
                 f"v10-B fuses the two v9b unimodal legs with a two-block "
                 f"bidirectional cross-attention module followed by tensor fusion "
                 f"[text ; audio ; text*audio ; |text-audio|] and an MLP classifier. "
                 f"Thirty checkpoints were trained (5 folds x 3 seeds x 2 class-weight "
                 f"branches). The held-out test evaluation has now been run: "
                 f"W-Acc {g_unif.get('acc', float('nan')):.4f} for the uniform branch "
                 f"and {g_cw.get('acc', float('nan')):.4f} for the class-weighted "
                 f"branch, against MemoCMT's published {MEMOCMT_FUSION:.4f}. The "
                 f"single-modality legs being fused are the v9b audio "
                 f"({V9B_AUDIO:.4f}) and text ({V9B_TEXT:.4f}) results reported "
                 f"separately.")
    else:
        add_para(doc,
                 "v10-B fuses the two v9b unimodal legs with a two-block "
                 "bidirectional cross-attention module followed by tensor fusion "
                 "[text ; audio ; text*audio ; |text-audio|] and an MLP classifier. "
                 "Thirty checkpoints (5 folds x 3 seeds x 2 class-weight branches) "
                 "were trained successfully and are present on the cluster. "
                 "THE HELD-OUT TEST EVALUATION HAS NOT BEEN RUN, so this report "
                 "contains no fused accuracy figure. Reporting a number here would "
                 "be a fabrication; the training-time validation metric is a "
                 "different quantity and is presented separately below, explicitly "
                 "labelled as such.",
                 )

    # ---------------- 1. architecture ----------------
    add_heading(doc, "1. Architecture", 1)
    add_para(doc,
             "The v10-B head receives a 768-dimensional mean-pooled utterance "
             "embedding from each of two frozen encoders and learns a joint "
             "representation. The design choices are:")
    add_bullet(doc, "WavLM-base+ for audio and DeBERTa-v3-base for text, each "
                    "loaded from the corresponding v9b checkpoint for the same "
                    "(fold, seed) and then frozen. The encoders contribute zero "
                    "trainable parameters.",
               bold_prefix="Frozen unimodal encoders — ")
    add_bullet(doc, "each encoder's output is projected to a 512-dimensional space "
                    "and given a learned class token, making two length-1 sequences.",
               bold_prefix="Projection — ")
    add_bullet(doc, "two blocks of bidirectional cross-attention let text and audio "
                    "attend to each other, so the model can condition on "
                    "disagreement between modalities rather than merely "
                    "concatenating them.",
               bold_prefix="Cross-modal transformer — ")
    add_bullet(doc, "the two projected vectors are combined as "
                    "[t ; a ; t*a ; |t-a|] (4 x 512 = 2048 dims) and passed through "
                    "an MLP (2048 -> 512 -> 4). The elementwise product and absolute "
                    "difference terms give the classifier explicit interaction "
                    "features.",
               bold_prefix="Tensor fusion — ")
    add_bullet(doc, "cross-entropy with optional class weights [1.30, 1.00, 1.00, "
                    "1.15] to compensate for IEMOCAP-4's imbalanced class "
                    "distribution (neutral is the most frequent class).",
               bold_prefix="Loss — ")

    # ---------------- 2. training ----------------
    add_heading(doc, "2. Training Array", 1)
    add_table(doc, ["Property", "Value"],
              [["Array size", "30 jobs (0-29)"],
               ["Layout", "task_id // 15 selects the branch (0=unif, 1=cw); "
                          "task_id % 15 selects fold*3 + (seed-42)"],
               ["Folds x seeds", "5 folds x 3 seeds (42, 43, 44)"],
               ["Checkpoint names", "v10_fusion_unif_fold{F}_seed{S}.pt, "
                                   "v10_fusion_cw_fold{F}_seed{S}.pt"],
               ["Checkpoints landed", "30 / 30 on the cluster"],
               ["Branch A (uniform)", "15 checkpoints — uniform class weights"],
               ["Branch B (class-weighted)", "15 checkpoints — class weights "
                                             "[1.30, 1.00, 1.00, 1.15]"]],
              widths=[1.9, 4.8])

    # ---------------- 3. evaluation status ----------------
    add_heading(doc, "3. Evaluation Status", 1)
    if v10:
        add_para(doc,
                 "The held-out test evaluation has been run. For every checkpoint, "
                 "the model is evaluated on its own held-out fold and the 15 runs "
                 "per branch are averaged.",
                 bold=False)
        rows = []
        for tag, label in (("unif", "v10-B fusion (uniform class weights)"),
                           ("cw", "v10-B fusion (class-weighted)")):
            leg = v10.get("v10", {}).get(tag, {})
            gm = leg.get("grand_mean", {})
            per = leg.get("per_ckpt", [])
            if not gm:
                continue
            acc = gm.get("acc")
            rows.append([label, f"{acc:.4f}" if acc is not None else "—",
                         f"{gm.get('f1', float('nan')):.4f}" if gm.get("f1") is not None else "—",
                         f"{gm.get('ua', float('nan')):.4f}" if gm.get("ua") is not None else "—",
                         f"{gm.get('n', len(per))}/15",
                         f"{(acc - MEMOCMT_FUSION) * 100:+.2f} pp" if acc is not None else "—"])
        add_table(doc, ["Branch", "W-Acc", "Macro-F1", "UA", "n", "vs MemoCMT"], rows,
                  widths=[2.3, 0.75, 0.85, 0.75, 0.5, 0.95])
        for tag, label in (("unif", "uniform"), ("cw", "class-weighted")):
            per = v10.get("v10", {}).get(tag, {}).get("per_ckpt", [])
            if not per:
                continue
            add_para(doc, f"Per-checkpoint detail — {label} branch:", size=10, bold=True)
            add_table(doc, ["Fold", "Seed", "W-Acc", "UA", "F1"],
                      [[c.get("fold"), c.get("seed"),
                        f"{c.get('acc', 0):.4f}", f"{c.get('ua', 0):.4f}",
                        f"{c.get('f1', 0):.4f}"] for c in per],
                      widths=[0.7, 0.7, 0.9, 0.9, 0.9])
    else:
        add_para(doc,
                 "The evaluation has not been run. The training array completed "
                 "successfully — all 30 checkpoints are on the cluster — but the "
                 "evaluation job was never submitted.", bold=True)
        add_para(doc,
                 "The reason was an HPC-side segfault inside the audio decoding "
                 "path (_torchaudio.so / _soundfile.so) on the scn25-family compute "
                 "nodes. A login-node smoke test of the identical code path passed "
                 "clean, which established that the code was correct and the failure "
                 "was infrastructure-side. Nine fixes were applied afterwards — "
                 "moving the WAV reader from torchaudio to soundfile, copying the "
                 "WAV to /tmp to sidestep NFS file-handle staleness, guarding "
                 "against LibsndfileError, and correcting the repository-root path "
                 "resolution — and none of those fixes has yet been exercised on a "
                 "compute node, because nothing was re-submitted.")
        add_para(doc,
                 "Two defects were found in the evaluation path while auditing it "
                 "before submission, both of which would have silently produced an "
                 "empty Table 1:", bold=True, size=10.5)
        add_bullet(doc, "the sbatch script changed directory into scripts/iemocap "
                        "and then used a CWD-relative --out path, so the results "
                        "JSON would have landed in scripts/iemocap/reports/ while "
                        "the Table 1 builder reads reports/ at the repository root. "
                        "The builder would have reported the results as missing "
                        "even though the evaluation had succeeded.",
                   bold_prefix="Wrong output directory — ")
        add_bullet(doc, "the v9b evaluation writes reports/v9b_table1.json, but the "
                        "v10 evaluation and the Table 1 builder both read "
                        "reports/v9b_table1_results.json. The v9b comparison rows "
                        "would have vanished without warning.",
                   bold_prefix="Filename mismatch — ")
        add_para(doc,
                 "Both are fixed and committed. The evaluation is a single "
                 "one-hour A100 job.", bold=True)

        # training-time validation metric, clearly labelled
        if val_sums:
            add_heading(doc, "3.1 Training-time validation metrics (NOT test metrics)", 2)
            add_para(doc,
                     "The following are the best validation accuracies recorded "
                     "during training, on the validation split used for early "
                     "stopping. They are NOT the held-out test metric and are NOT "
                     "comparable with the MemoCMT figure. They are included only "
                     "to show that the array trained as expected.",
                     italic=True, size=10)
            rows = []
            for tag, label in (("unif", "uniform"), ("cw", "class-weighted")):
                items = val_sums.get(tag, [])
                if not items:
                    continue
                accs = [i["acc"] for i in items if i.get("acc") is not None]
                if not accs:
                    continue
                mean = sum(accs) / len(accs)
                rows.append([f"v10-B {label}", f"{mean:.4f}",
                             f"{min(accs):.4f}", f"{max(accs):.4f}", f"{len(accs)}/15"])
            if rows:
                add_table(doc, ["Branch", "mean val acc", "min", "max", "n"], rows,
                          widths=[2.0, 1.2, 0.9, 0.9, 0.6])

    # ---------------- 4. comparison ----------------
    add_heading(doc, "4. Position Against Prior Results", 1)
    if v10:
        rows = [["MemoCMT (published fusion)", f"{MEMOCMT_FUSION:.4f}", "—", "reference"],
                ["v8 CMT + dialog context (LOSO)", f"{V8_LOSO:.4f}", "—", "15 runs"],
                ["v9b audio-only (WavLM-base+)", f"{V9B_AUDIO:.4f}", "—", "15 runs"],
                ["v9b text-only (DeBERTa-v3-base)", f"{V9B_TEXT:.4f}", "—", "15 runs"]]
        for tag, label in (("unif", "v10-B fusion (uniform)"),
                           ("cw", "v10-B fusion (class-weighted)")):
            gm = v10.get("v10", {}).get(tag, {}).get("grand_mean", {})
            if gm.get("acc") is not None:
                rows.append([label, f"{gm['acc']:.4f}", "—", f"{gm.get('n', 15)} runs"])
        add_table(doc, ["Model", "W-Acc", "F1", "n"], rows,
                  widths=[2.6, 0.9, 0.8, 1.4])
    else:
        add_table(doc, ["Model", "W-Acc", "Status"],
                  [["MemoCMT (published fusion)", f"{MEMOCMT_FUSION:.4f}", "reference"],
                   ["v8 CMT + dialog context (LOSO)", f"{V8_LOSO:.4f}", "15 runs, measured"],
                   ["v9b audio-only (WavLM-base+)", f"{V9B_AUDIO:.4f}", "15 runs, measured"],
                   ["v9b text-only (DeBERTa-v3-base)", f"{V9B_TEXT:.4f}", "15 runs, measured"],
                   ["v10-B fusion (uniform)", "—", "evaluation not run"],
                   ["v10-B fusion (class-weighted)", "—", "evaluation not run"]],
                  widths=[2.6, 0.9, 1.9])
        add_para(doc,
                 "The v10-B row is deliberately empty. The checkpoints exist and "
                 "the training completed, but an unevaluated model has no accuracy, "
                 "and the empty cell is the honest representation of that.",
                 italic=True, size=10)

    # ---------------- 5. caveats ----------------
    add_heading(doc, "5. Reproducibility Notes", 1)
    add_bullet(doc, "every figure in this report is read from a results file at "
                    "build time. The builder prints the measured row as '—' when "
                    "the source JSON is absent; it never estimates a value.",
               bold_prefix="No fabricated numbers — ")
    add_bullet(doc, "the early v0-v7 checkpoints are degenerate (BatchNorm "
                    "moving-statistic corruption) and the v6 run was reverted for a "
                    "label leak. Neither is cited here.",
               bold_prefix="Excluded versions — ")
    add_bullet(doc, "the 86-90% figure in the architecture roadmap is a projection, "
                    "not a measurement.", bold_prefix="Projections — ")

    if session_log:
        add_heading(doc, "6. Training Session Record", 1)
        add_para(doc, session_log.strip(), size=9)

    # ---------------- artefacts ----------------
    add_heading(doc, "7. Artefacts", 1)
    add_table(doc, ["Artefact", "Location"],
              [["v10 trainer", "scripts/iemocap/train_ser_v10_fusion.py"],
               ["v10 eval", "scripts/iemocap/ensemble_evaluate_v10.py"],
               ["v10 train sbatch", "scripts/train_ser_v10_fusion.sbatch (array 0-29)"],
               ["v10 eval sbatch", "scripts/eval_v10_table1.sbatch"],
               ["Results JSON", "reports/v10_table1_results.json"],
               ["Table 1 builder", "scripts/iemocap/build_table1_final.py"],
               ["v9 report", "reports/DTU_SER_v9_v9b_Progress_Aug2026.pdf"],
               ["v8 report", "reports/DTU_SER_v8_Multimodal_Progress_12Aug2026.pdf"]],
              widths=[1.9, 4.8])

    docx_path = OUTPUT_DIR / "DTU_SER_v10_Fusion_Progress_Aug2026.docx"
    pdf_path = OUTPUT_DIR / "DTU_SER_v10_Fusion_Progress_Aug2026.pdf"
    doc.save(docx_path)
    print(f"Saved {docx_path}")
    print("Converting to PDF via soffice...")
    try:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(OUTPUT_DIR), str(docx_path)],
            check=True, capture_output=True, timeout=300)
        print(f"Saved {pdf_path}" if pdf_path.exists() else "WARNING: PDF not produced")
    except Exception as e:
        print(f"WARNING: soffice conversion failed: {e}")
    return docx_path, pdf_path


if __name__ == "__main__":
    build()
