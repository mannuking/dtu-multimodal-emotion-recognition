"""
build_table1_final.py — THE master results table for the DTU SER paper.

Reads whichever of the three result files exist and prints one table
covering the whole v9 -> v10 -> v11 progression:

  reports/v9b_table1_results.json   (or v9b_table1.json)  -> v9 / v9b legs
  reports/v10_table1_results.json                          -> v10-B fusion
  reports/v11_table1_results.json                          -> v11 V/A fusion

Missing legs are skipped with an explicit note, never silently omitted.

Each row reports W-Acc (weighted accuracy = plain accuracy), macro-F1 and
UA-Acc (unweighted accuracy = macro recall). The two accuracy measures
disagree exactly when the model is biased toward the frequent classes, so
reporting both is what makes the comparison with MemoCMT (0.7925 W-Acc)
honest rather than flattering.

Usage:
    python scripts/iemocap/build_table1_final.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = REPO_ROOT / "reports"

MEMOCMT = 0.7925
V8_LOSO = 0.7298


def load(*candidates):
    """First JSON that exists, searched at repo root and in CWD."""
    for cand in candidates:
        for base in (REPORTS, Path.cwd()):
            p = base / cand
            if p.exists():
                with open(p) as f:
                    return json.load(f), cand
    return None, None


def row(label, m, note=""):
    if m is None:
        return None
    acc = m.get("acc")
    f1 = m.get("f1")
    ua = m.get("ua")
    if acc is None:
        return None
    delta = (acc - MEMOCMT) * 100
    cols = [f"{label:38s}", f"{acc:.4f}",
            f"{f1:.4f}" if f1 is not None else "  —   ",
            f"{ua:.4f}" if ua is not None else "  —   ",
            f"{delta:+.2f} pp",
            f"{note:12s}" if note else " " * 12]
    return "| " + " | ".join(cols) + " |"


def main():
    print("\n" + "=" * 108)
    print("TABLE 1 — DTU SER on IEMOCAP-4  (random 5-fold x 3 seeds = 15 runs per row)")
    print("=" * 108)
    print(f"{'Model':38s} | {'W-Acc':6s} | {'F1':6s} | {'UA':6s} | {'vs MemoCMT':11s} | {'note':12s}")
    print(f"{'-' * 38}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 11}-+-{'-' * 12}")

    v9b, v9b_name = load("v9b_table1_results.json", "v9b_table1.json")
    v10, v10_name = load("v10_table1_results.json")
    v11, v11_name = load("v11_table1_results.json")

    printed = 0

    # ---- reference / published numbers -------------------------------
    print(f"| {'MemoCMT (Deluxe et al. 2025) [ref]':38s} | {MEMOCMT:.4f} |   —    |   —    |  +0.00 pp | {'published':12s} |")
    print(f"| {'v8 CMT + dialog context (LOSO)':38s} | {V8_LOSO:.4f} |   —    |   —    | "
          f"{(V8_LOSO - MEMOCMT) * 100:+.2f} pp | {'LOSO, 15 runs':12s} |")

    # ---- v9 / v9b unimodal legs --------------------------------------
    if v9b:
        for tag, label in (
            ("text", "v9b text-only (DeBERTa-v3-base)"),
            ("audio", "v9b audio-only (WavLM-base+)"),
            ("fusion", "v9b fusion (concat baseline)"),
        ):
            leg = v9b.get(tag)
            if not leg:
                continue
            r = row(label, leg.get("grand_mean"))
            if r:
                print(r)
                printed += 1
    else:
        print(f"| {'[missing] v9b unimodal legs (no table-1 JSON)':38s} |    —    |   —    |   —    |     —     | {'':12s} |")

    # ---- v10 --------------------------------------------------------
    if v10:
        for tag, label in (
            ("unif", "v10-B fusion (uniform class weights)"),
            ("cw", "v10-B fusion (class-weighted)"),
        ):
            leg = v10.get("v10", {}).get(tag)
            if not leg:
                continue
            r = row(label, leg.get("grand_mean"))
            if r:
                print(r)
                printed += 1
    else:
        print(f"| {'[missing] v10-B fusion (eval not run yet)':38s} |    —    |   —    |   —    |     —     | {'':12s} |")

    # ---- v11 --------------------------------------------------------
    if v11:
        for tag, label in (
            ("unif", "v11 V/A-conditioned fusion (uniform)"),
            ("cw", "v11 V/A-conditioned fusion (class-weighted)"),
        ):
            leg = v11.get("v11", {}).get(tag)
            if not leg:
                continue
            gm = leg.get("grand_mean") or {}
            note = ""
            if gm.get("va_mae") is not None:
                note = f"VA-MAE {gm['va_mae']:.4f}"
            r = row(label, gm, note)
            if r:
                print(r)
                printed += 1
    else:
        print(f"| {'[missing] v11 V/A fusion (not trained yet)':38s} |    —    |   —    |   —    |     —     | {'':12s} |")

    print(f"{'-' * 38}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 11}")
    print(f"\nReference: MemoCMT fusion = {MEMOCMT:.4f} W-Acc (Deluxe et al., "
          f"Nat. Sci. Reports 2025).")
    print(f"{printed} measured row(s) above. Rows marked [missing] have no "
          f"results JSON on disk yet.")

    # Provenance
    print("\nSource files:")
    for name, found in (("v9b", v9b_name), ("v10", v10_name), ("v11", v11_name)):
        print(f"  {name}: {found or 'NOT FOUND'}")

    # Sanity: if we have v10 and v11, state the delta explicitly.
    if v10 and v11:
        def gm(d, tag, root):
            leg = d.get(root, {}).get(tag, {})
            return leg.get("grand_mean", {}).get("acc")
        a10, a11 = gm(v10, "unif", "v10"), gm(v11, "unif", "v11")
        if a10 is not None and a11 is not None:
            print(f"\nv11 (uniform) vs v10-B (uniform): {(a11 - a10) * 100:+.2f} pp")
        a10c, a11c = gm(v10, "cw", "v10"), gm(v11, "cw", "v11")
        if a10c is not None and a11c is not None:
            print(f"v11 (class-weighted) vs v10-B (class-weighted): {(a11c - a10c) * 100:+.2f} pp")

    print()


if __name__ == "__main__":
    main()
