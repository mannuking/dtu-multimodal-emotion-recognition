"""
build_table1_v10.py — Print the final v10 Table 1 in human-readable form.

Reads:
  reports/v9b_table1_results.json
  reports/v10_table1_results.json

Prints:
  Markdown table comparing v9b audio/text/fusion vs v10-B (unif/cw) vs MemoCMT.
"""
from __future__ import annotations

import json
from pathlib import Path


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    v9b = load("reports/v9b_table1_results.json")
    v10 = load("reports/v10_table1_results.json")
    if v10 is None:
        print("ERROR: reports/v10_table1_results.json missing — run eval first.")
        return

    print("\n# Table 1 — v10-B CMT Fusion on IEMOCAP-4 (Random 5-Fold, 15 ckpts each)\n")
    print("| Model              |   W-Acc |      F1 |      UA |  vs MemoCMT (0.7925) |")
    print("| ------------------ | ------- | ------- | ------- | -------------------- |")
    if v9b is not None:
        for tag, label in (("text", "v9b text (DeBERTa-v3-base)"),
                            ("audio", "v9b audio (WavLM-base+)"),
                            ("fusion", "v9b fusion (concat)")):
            if tag in v9b:
                m = v9b[tag]["grand_mean"]
                delta = (m["acc"] - 0.7925) * 100
                print(f"| {label:18s} | {m['acc']:.4f}  | {m['f1']:.4f}  | "
                      f"{m['ua']:.4f}  |  {delta:+.2f} pp           |")
    for tag, label in (("unif", "v10-B fusion (uniform)"),
                        ("cw",   "v10-B fusion (class-weighted)")):
        m = v10["v10"][tag]["grand_mean"]
        delta = (m["acc"] - 0.7925) * 100
        print(f"| {label:18s} | {m['acc']:.4f}  | {m['f1']:.4f}  | "
              f"{m['ua']:.4f}  |  {delta:+.2f} pp           |")
    print("\nReference: MemoCMT (Deluxe et al., Nat. Sci. Reports 2025) fusion = 0.7925\n")


if __name__ == "__main__":
    main()
