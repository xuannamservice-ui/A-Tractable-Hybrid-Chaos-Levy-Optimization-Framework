"""Check data/06_system_aber against an arbiter that builds no density at all.

Block 06 carries two constructions of the branch density -- `aber_system`
(log-domain) and `aber_system_quad` (quadrature over the pointing law) -- and
their disagreement in `ref_spread_percent`.  A disagreement says one of them is
wrong; it does not say which.  This script asks a third method that shares no
code with either:

    system_metric.system_aber(..., method='mc')

samples the 16-branch sum directly, using the fact that a unit-mean
gamma-gamma is a product of two unit-mean gamma variates and that
h_p = A_0 U^(1/xi^2) for uniform U.  No density is constructed, no grid is
chosen, and no convolution is performed, so it is independent of every
discretisation decision the other two paths make.

Monte Carlo can only resolve a probability it actually observes: with n
samples the relative standard error of an ABER p is about sqrt((1-p)/(n p)),
so this script refuses rows below `--min-aber` rather than reporting noise as
a verdict.  That threshold is a property of the arbiter, not a choice about
which rows look good, and every row above it in the requested selection is
reported -- including the ones where the shipped column loses.

Usage:
    python code/verify_block06.py                  # worst-spread rows
    python code/verify_block06.py --all            # every resolvable row
    python code/verify_block06.py --samples 16000000
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import system_metric as sm                                       # noqa: E402

CSV = os.path.join(PKG, "data", "06_system_aber", "system_aber_curves.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--samples", type=int, default=4_000_000)
    ap.add_argument("--min-aber", type=float, default=1e-5,
                    help="skip rows the arbiter cannot resolve at this sample count")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--tight-rse", type=float, default=0.02,
                    help="MC relative standard error below which a ratio "
                         "against the arbiter is treated as meaningful")
    ap.add_argument("--all", action="store_true",
                    help="check every resolvable row, not just the worst spreads")
    a = ap.parse_args()

    with open(a.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    cand = [r for r in rows if fl(r["aber_system"]) >= a.min_aber]
    skipped = len(rows) - len(cand)
    if not a.all:
        cand.sort(key=lambda r: -(fl(r["ref_spread_percent"])
                                  if math.isfinite(fl(r["ref_spread_percent"]))
                                  else -1.0))
        cand = cand[: a.rows]

    print("Arbitrating data/06_system_aber against Monte Carlo (%d samples/row)"
          % a.samples)
    print("%d of %d rows are below --min-aber=%.0e and cannot be arbitrated "
          "at this sample count.\n" % (skipped, len(rows), a.min_aber))
    print("%-9s %-6s %-7s %-5s %-12s %-12s %-12s %-9s %-9s %-8s %s"
          % ("regime", "sig", "xi", "dB", "fast(shipped)", "quad", "MC",
             "fast/MC", "quad/MC", "sigma", "verdict"))
    print("-" * 126)

    # A ratio against Monte Carlo is only informative when the Monte Carlo
    # itself is precise.  At p = 1e-5 and n = 2e6 the relative standard error
    # is ~22%, so a 5% "deviation" there is noise, not evidence.  Deviations
    # are therefore reported BOTH as a ratio and in units of the MC standard
    # error, and the headline ratio is taken only over rows where the arbiter
    # is precise to better than `--tight-rse`.
    tight, zs_fast, zs_quad = [], [], []
    worst_quad = 0.0
    bad = 0
    t0 = time.time()
    for r in cand:
        alpha, beta = sm.REGIMES[r["regime"]]
        xi, A0 = fl(r["xi"]), fl(r["A0"])
        gbar = 10.0 ** (fl(r["snr_db"]) / 10.0)
        vf, vq = fl(r["aber_system"]), fl(r["aber_system_quad"])
        vm = sm.system_aber(alpha, beta, xi, A0, gbar, method="mc",
                            mc_samples=a.samples)
        if vm <= 0:
            continue
        rse = math.sqrt(max(1.0 - vm, 0.0) / (a.samples * vm))
        ef, eq = vf / vm - 1.0, vq / vm - 1.0
        zf, zq = (ef / rse if rse else float("nan")), (eq / rse if rse else float("nan"))
        zs_fast.append(abs(zf))
        zs_quad.append(abs(zq))
        worst_quad = max(worst_quad, abs(eq))
        if rse <= a.tight_rse:
            tight.append((abs(ef), abs(eq), r))
        verdict = "ok" if abs(zf) <= 5.0 else "SHIPPED COLUMN OFF"
        if verdict != "ok":
            bad += 1
        print("%-9s %-6s %-7s %-5s %-12.6e %-12.6e %-12.6e %-9.4f %-9.4f "
              "%-8s %s"
              % (r["regime"], r["sigma_s"], r["xi"], r["snr_db"], vf, vq, vm,
                 vf / vm, vq / vm, "%.1f" % zf, verdict))

    print("\nchecked %d rows in %.0f s" % (len(cand), time.time() - t0))
    print("deviation from the arbiter, in units of the MC standard error:")
    print("   shipped `aber_system` : worst %.1f sigma"
          % (max(zs_fast) if zs_fast else float("nan")))
    print("   `aber_system_quad`    : worst %.1f sigma"
          % (max(zs_quad) if zs_quad else float("nan")))
    if tight:
        print("on the %d row(s) where the arbiter is precise to better than "
              "%.1f%% (so a ratio is meaningful):" % (len(tight), 100 * a.tight_rse))
        print("   shipped `aber_system` : worst %.3f%%"
              % (100 * max(t[0] for t in tight)))
        print("   `aber_system_quad`    : worst %.3f%%"
              % (100 * max(t[1] for t in tight)))
    else:
        print("no row was resolved tightly enough for a ratio to be meaningful; "
              "raise --samples or --min-aber.")
    print("worst `aber_system_quad` ratio over ALL checked rows: %.3f%%"
          % (100 * worst_quad))
    if bad:
        print("\n%d row(s) where the SHIPPED column disagrees with the arbiter "
              "by more than 5 sigma." % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
