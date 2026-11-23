"""
aggregate_v10_summaries.py — CPU-only aggregation of the v10-B fusion array.

Reads the 30 paired `_summary.json` files written by train_ser_v10_fusion.py
and computes mean ± std of the best validation accuracy / F1 across the
5 folds × 3 seeds, separately for the `unif` and `cw` branches.

This is the cheap path to a number: it needs NO GPU, NO model loading and NO
dataset read, so it is safe to run on the login node (it is a few hundred
kilobytes of JSON). It reports the *validation* metric each checkpoint was
selected on, which is the number the trainer used for early stopping.

It is NOT the same as the Table 1 test metric — the eval script
(ensemble_evaluate_v10.py) re-runs every checkpoint on the held-out test
split and ensembles the 15 runs per branch. Use this aggregator to sanity
check that the array trained as expected, and to know the expected ballpark
before spending a GPU-hour on the full eval.

Writes: reports/v10_val_aggregate.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = REPO_ROOT / "model_checkpoints"
OUT = REPO_ROOT / "reports" / "v10_val_aggregate.json"

BRANCHES = ("unif", "cw")
FOLDS = range(5)
SEEDS = range(3)


def mean_std(xs):
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    m = sum(xs) / len(xs)
    # sample std (ddof=1); fall back to 0.0 for a single observation
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": round(m, 6), "std": round(s, 6), "n": len(xs)}


def main():
    out = {"source": "v10 fusion _summary.json (best validation metric per ckpt)",
           "branches": {}}
    missing = []

    for branch in BRANCHES:
        accs, f1s, uas, rows = [], [], [], []
        for fold in FOLDS:
            for seed in SEEDS:
                p = CKPT_DIR / f"v10_fusion_{branch}_fold{fold}_seed{seed}_summary.json"
                if not p.exists():
                    missing.append(p.name)
                    continue
                with open(p) as f:
                    d = json.load(f)
                hist = d.get("history") or []
                # best UA over the run, matching the epoch the checkpoint was
                # selected on (best_val_acc); fall back to the last epoch.
                best_ua = None
                if hist:
                    best_acc = max(h.get("val_acc", -1) for h in hist)
                    same = [h for h in hist if abs(h.get("val_acc", -1) - best_acc) < 1e-12]
                    best_ua = max((h.get("val_ua") for h in same if h.get("val_ua") is not None),
                                  default=None)
                acc = d.get("best_val_acc")
                f1 = d.get("best_val_f1")
                if acc is not None:
                    accs.append(float(acc))
                if f1 is not None:
                    f1s.append(float(f1))
                if best_ua is not None:
                    uas.append(float(best_ua))
                rows.append({"fold": fold, "seed": seed,
                             "best_val_acc": acc, "best_val_f1": f1,
                             "best_val_ua": best_ua, "best_epoch": d.get("best_epoch"),
                             "n_train": d.get("n_train"), "n_val": d.get("n_val")})
        out["branches"][branch] = {
            "val_acc": mean_std(accs),
            "val_f1": mean_std(f1s),
            "val_ua": mean_std(uas),
            "per_ckpt": sorted(rows, key=lambda r: (r["fold"], r["seed"])),
        }
        print(f"v10-B {branch:4s}: val_acc={out['branches'][branch]['val_acc']['mean']}"
              f" ± {out['branches'][branch]['val_acc']['std']}"
              f"  (n={out['branches'][branch]['val_acc']['n']}/15)")

    out["missing_summaries"] = missing
    if missing:
        print(f"\nWARNING: {len(missing)} summary file(s) missing:")
        for m in missing[:10]:
            print("   ", m)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}")
    print("NOTE: these are VALIDATION metrics (early-stopping selection).")
    print("      The paper's Table 1 uses held-out TEST metrics from")
    print("      scripts/iemocap/ensemble_evaluate_v10.py (needs a GPU).")


if __name__ == "__main__":
    sys.exit(main())
