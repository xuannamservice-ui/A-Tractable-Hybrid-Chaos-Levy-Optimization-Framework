"""Summary statistics for block 05, computed IN-BAND and OUT-OF-BAND separately.

The shipped block 05 carried no summary at all, and the only aggregate a reader
could form from it -- the median relative difference over every row -- mixes
three populations that mean different things:

  * rows inside the admissible band, where the manuscript claims Eq. (22) holds
    and where the comparison is a real test of it;
  * rows outside it, which the manuscript's own guard is specified to reject
    rather than score, and which this sweep additionally evaluates at a
    truncation order the fidelity ladder would not have chosen (see
    eq22_ladder_check.py);
  * rows at high SNR where the *reference* has fallen below the round-off floor
    of the 16-fold FFT, so neither side of the comparison is meaningful.

This script writes every one of those populations, and the unscoped figure, to
data/05_eq22_validation/summary.json.  It selects nothing: all four populations
are reported whatever they say, and the row counts are printed so a reader can
see how much of the box each covers.

Usage:  python code/eq22_summary.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
BLOCK = os.path.join(PKG, "data", "05_eq22_validation")
CSV = os.path.join(BLOCK, "eq22_vs_reference.csv")
OUT = os.path.join(BLOCK, "summary.json")


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def describe(rows, label):
    v = sorted(abs(fl(r["rel_diff_percent"])) for r in rows
               if math.isfinite(fl(r["rel_diff_percent"])))
    d = {"label": label, "rows": len(rows), "rows_scored": len(v),
         "rows_unscoreable": len(rows) - len(v)}
    if v:
        d.update({
            "median_abs_rel_diff_percent": statistics.median(v),
            "p90_abs_rel_diff_percent": v[min(int(0.9 * len(v)), len(v) - 1)],
            "max_abs_rel_diff_percent": v[-1],
            "min_abs_rel_diff_percent": v[0],
        })
    return d


def main() -> int:
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    inb = [r for r in rows if r["admissible"] == "1"]
    oob = [r for r in rows if r["admissible"] == "0"]
    res = [r for r in rows if r["ref_resolved"] == "1"]
    val = [r for r in rows if r["comparison_valid"] == "1"]

    pops = [
        describe(rows, "all rows, no scoping"),
        describe(oob, "out-of-band (admissible = 0)"),
        describe(inb, "in-band (admissible = 1)"),
        describe(val, "in-band AND reference resolved (comparison_valid = 1)"),
        describe([r for r in res if r["admissible"] == "0"],
                 "out-of-band but reference resolved"),
    ]

    by_snr = {}
    for g in sorted({int(r["snr_db"]) for r in rows}):
        sub = [r for r in rows if int(r["snr_db"]) == g]
        sv = [r for r in sub if r["comparison_valid"] == "1"]
        zs = [fl(r["z"]) for r in sub]
        by_snr[str(g)] = {
            "rows": len(sub),
            "z_min": min(zs), "z_max": max(zs),
            "in_band": sum(1 for r in sub if r["admissible"] == "1"),
            "comparison_valid": len(sv),
            "median_abs_rel_diff_percent_on_valid":
                statistics.median(sorted(abs(fl(r["rel_diff_percent"]))
                                         for r in sv)) if sv else None,
        }

    summary = {
        "source": "data/05_eq22_validation/eq22_vs_reference.csv",
        "generated_by": "code/eq22_summary.py",
        "total_rows": len(rows),
        "admissibility_predicate":
            "admissible = (ladder admits z at THIS row's SNR) and "
            "(K used >= ladder-selected K). Identical to the predicate block 01 "
            "applies; z = sqrt(2)*alpha*beta/(A_0*sqrt(gbar)), ladder "
            "z<=0.5 -> K=5, z<=2 -> K=10, z<=8 -> K=20, beyond -> inadmissible.",
        "resolved_predicate":
            "ref_resolved = the two independent branch-density constructions "
            "(ref_quad, ref_logdomain) agree to better than 1%. Below the FFT "
            "round-off floor they do not, and neither reference is meaningful.",
        "scoping_justification":
            "The in-band scoping is the manuscript's own, not a choice made "
            "here: Fig. odt_validation plots the surrogate only inside the "
            "admissible band, and Sec. III-C declares candidates beyond z=8 "
            "inadmissible and has the guard reject them rather than score them.",
        "caveat":
            "The out-of-band population is NOT evidence that Eq. (22) fails "
            "there. This sweep fixes one truncation order K per configuration "
            "from z at the highest SNR and reuses it at every lower SNR, which "
            "the deployed per-candidate fidelity ladder never does. "
            "code/eq22_ladder_check.py re-evaluates at the ladder-selected "
            "order and recovers agreement to ~1e-3 percent at SNRs where this "
            "sweep reports errors of 10^2 to 10^7 percent. What this dataset "
            "establishes is the in-band figure; it does not establish an "
            "out-of-band one either way.",
        "populations": pops,
        "by_snr": by_snr,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("wrote %s\n" % OUT)
    print("%-52s %6s %6s %14s %14s" % ("population", "rows", "score",
                                       "median |rel|%", "max |rel|%"))
    print("-" * 96)
    for p in pops:
        print("%-52s %6d %6d %14s %14s"
              % (p["label"], p["rows"], p["rows_scored"],
                 "%.4g" % p["median_abs_rel_diff_percent"]
                 if "median_abs_rel_diff_percent" in p else "-",
                 "%.4g" % p["max_abs_rel_diff_percent"]
                 if "max_abs_rel_diff_percent" in p else "-"))
    print("\n%-8s %6s %10s %10s %8s %8s %s"
          % ("SNR", "rows", "z_min", "z_max", "in_band", "valid",
             "median |rel|% on valid"))
    print("-" * 96)
    for g, d in by_snr.items():
        print("%-8s %6d %10.4f %10.4f %8d %8d %s"
              % (g + " dB", d["rows"], d["z_min"], d["z_max"], d["in_band"],
                 d["comparison_valid"],
                 "%.4g" % d["median_abs_rel_diff_percent_on_valid"]
                 if d["median_abs_rel_diff_percent_on_valid"] is not None else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
