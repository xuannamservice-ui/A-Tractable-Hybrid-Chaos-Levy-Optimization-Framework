"""Show that block 04's error column measures something, and what it measures.

The previous release of `data/04_offgrid_error/offgrid_error.csv` carried
`abs_err_interp_free = 0.000e+00` on every row.  That was not a result about
the deployed kernel.  Both sides of the subtraction were the SAME mpmath
function -- `rtodt.Pe_series` at dps 200 and at dps 90 -- and both were cast to
float64 before differencing.  float64 carries ~16 significant digits, so
rounding a 200-digit and a 90-digit value of the same quantity gives the
identical double, and their difference is exactly zero by construction.  The
column measured mpmath's round-off against itself; it exercised no float64
kernel and no interpolation, in a block advertised as doing exactly that.

This script replays BOTH comparisons on the same sampled points, taken from the
shipped CSV so the rows are the real ones:

    OLD   float(Pe_series @ dps 200)  -  float(Pe_series @ dps 90)
    NEW   float(Pe_series @ dps 200)  -  rtodt_fast.pe_series_f64(...)

and prints how many rows come out exactly 0.0 under each.  It asserts nothing
about what the answer should be; it reports what the two comparisons do.

It then checks the shipped file against its own columns: `abs_err` versus the
eq. (27) round-off floor `eta_f64` recorded on the same row, and the relation
between the large errors and the `a_k` poles at xi^2 = beta+k.  Both are
properties a reader can verify from the file alone -- this script just does the
arithmetic.

Usage:
    python code/verify_block04.py                    # replay 200 rows
    python code/verify_block04.py --rows 2000
    python code/verify_block04.py --csv path/to.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import mpmath as mp                                              # noqa: E402
import numpy as np                                               # noqa: E402

from rtodt import Pe_series, REGIMES, db                         # noqa: E402
from rtodt_fast import pe_series_f64                             # noqa: E402

CSV = os.path.join(PKG, "data", "04_offgrid_error", "offgrid_error.csv")


def pole_distance(regime: str, xi: float, K: int) -> float:
    """min_k |xi^2 - (beta+k)| and |xi^2 - (alpha+k)|, the a_k poles of eq. (16)."""
    A, B = (float(x) for x in REGIMES[regime])
    x2 = xi * xi
    return min(min(abs(x2 - (B + k)) for k in range(K + 1)),
               min(abs(x2 - (A + k)) for k in range(K + 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--rows", type=int, default=200,
                    help="how many rows to replay through both comparisons")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the subsample of rows to replay")
    a = ap.parse_args()

    with open(a.csv, newline="", encoding="utf-8") as f:
        allrows = list(csv.DictReader(f))
    if not allrows:
        print("empty file: %s" % a.csv)
        return 1

    # ---------------------------------------------------------------- replay
    rng = random.Random(a.seed)
    sample = (allrows if len(allrows) <= a.rows
              else rng.sample(allrows, a.rows))

    old_zero = new_zero = 0
    old_max = 0.0
    new_errs = []
    for r in sample:
        A, B = REGIMES[r["regime"]]
        xi, A0 = mp.mpf(r["xi"]), mp.mpf(r["A0"])
        g, K = db(int(r["snr_db"])), int(r["K"])

        mp.mp.dps = 200
        ref = float(Pe_series(A, B, xi, A0, g, K))
        mp.mp.dps = 90
        dep_old = float(Pe_series(A, B, xi, A0, g, K))
        e_old = abs(dep_old - ref)
        old_zero += (e_old == 0.0)
        old_max = max(old_max, e_old)

        fast = float(pe_series_f64(float(A), float(B), float(xi),
                                   float(A0), float(g), K)[0])
        e_new = abs(fast - ref) if np.isfinite(fast) else float("inf")
        new_zero += (e_new == 0.0)
        new_errs.append(e_new)

    fin = [e for e in new_errs if math.isfinite(e)]
    n = len(sample)
    print("Replaying both comparisons on %d row(s) of %s\n"
          % (n, os.path.relpath(a.csv, PKG)))
    print("  OLD  float(Pe_series@200) - float(Pe_series@90)   "
          "[the shipped comparison]")
    print("     rows with abs_err exactly 0.000e+00 : %d / %d" % (old_zero, n))
    print("     max abs_err                        : %.3e" % old_max)
    print("  NEW  float(Pe_series@200) - pe_series_f64(...)    "
          "[deployed float64 kernel]")
    print("     rows with abs_err exactly 0.000e+00 : %d / %d" % (new_zero, n))
    if fin:
        print("     min / median / max abs_err         : %.3e / %.3e / %.3e"
              % (min(fin), statistics.median(fin), max(fin)))
    if len(fin) != n:
        print("     non-finite (kernel overflow)       : %d" % (n - len(fin)))

    # ------------------------------------------------- checks over the file
    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    ae = [fl(r["abs_err_interp_free"]) for r in allrows]
    re_ = [fl(r["rel_err_interp_free"]) for r in allrows]
    eta = [fl(r["eta_f64"]) for r in allrows]
    fa = [x for x in ae if math.isfinite(x)]
    fr = [x for x in re_ if math.isfinite(x)]

    print("\nThe shipped file, %d rows:" % len(allrows))
    print("  abs_err exactly 0.000e+00 : %d (%.2f%%)"
          % (sum(1 for x in ae if x == 0.0),
             100.0 * sum(1 for x in ae if x == 0.0) / len(ae)))
    if fa:
        fa_s = sorted(fa)
        print("  abs_err  median / p99 / max : %.3e / %.3e / %.3e"
              % (statistics.median(fa_s), fa_s[int(0.99 * len(fa_s))], fa_s[-1]))
    if fr:
        fr_s = sorted(fr)
        print("  rel_err  median / p99 / max : %.3e / %.3e / %.3e"
              % (statistics.median(fr_s), fr_s[int(0.99 * len(fr_s))], fr_s[-1]))
    print("  guard test (ii), 0 <= Pe <= 1/2, violations: "
          "reference %d, float64 %d"
          % (sum(1 for r in allrows if r["ref_in_range"] == "0"),
             sum(1 for r in allrows if r["f64_in_range"] == "0")))

    # eq. (27) is a PER-TERM estimate; the evaluation sums 2(K+1) terms, so the
    # committed error is expected to exceed it by roughly that count.  Reported,
    # not asserted.
    ratio = sorted(x / e for x, e in zip(ae, eta)
                   if math.isfinite(x) and math.isfinite(e) and e > 0)
    if ratio:
        over = sum(1 for x in ratio if x > 1.0)
        print("\n  abs_err / eta_f64 (eq. 27 per-term floor):")
        print("     median %.3f   p99 %.2f   max %.2f"
              % (statistics.median(ratio), ratio[int(0.99 * len(ratio))], ratio[-1]))
        print("     rows where the committed error EXCEEDS the floor: "
              "%d / %d (%.1f%%)" % (over, len(ratio), 100.0 * over / len(ratio)))

    pd_all = [pole_distance(r["regime"], fl(r["xi"]), int(r["K"]))
              for r in allrows]
    bad = [r for r in allrows if fl(r["rel_err_interp_free"]) > 1e-10]
    print("\n  distance to the nearest a_k pole (xi^2 = beta+k / alpha+k):")
    print("     median over all rows              : %.4f"
          % statistics.median(pd_all))
    if bad:
        pd_bad = [pole_distance(r["regime"], fl(r["xi"]), int(r["K"]))
                  for r in bad]
        print("     median over rows with rel_err>1e-10 (%d, %.2f%%): %.4f"
              % (len(bad), 100.0 * len(bad) / len(allrows),
                 statistics.median(pd_bad)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
