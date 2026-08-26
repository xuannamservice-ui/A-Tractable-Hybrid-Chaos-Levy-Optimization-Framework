"""Is Eq. (22) wrong out of band, or is the sweep's K-selection wrong?

data/05_eq22_validation shows Eq. (22) missing an independent reference by up
to 165 orders of magnitude on most of its rows.  Before that is read as
"Eq. (22) fails across the parameter box", the sweep's own protocol has to be
accounted for, because it is not the deployed one:

    block_eq22 picks ONE truncation order K per configuration, from the
    conditioning parameter z at the HIGHEST SNR of the sweep, and then reuses
    that K at every lower SNR.

Since z = sqrt(2) alpha beta / (A_0 sqrt(gbar)) grows as gbar falls, a K that
the ladder admits at 40 dB is too small at 20 dB by up to a factor of ten in z.
The deployed kernel never does this: the fidelity ladder is per candidate and
keyed on that candidate's own z (Sec. III-C), so it would either select a
larger K or declare the candidate inadmissible and refuse to score it.

This script separates the two explanations by re-evaluating the SAME
configurations at the SAME SNRs with the ladder-selected order, against the
same reference.  It asserts nothing in advance; it prints both columns.

Usage:  python code/eq22_ladder_check.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)

import mpmath as mp                                              # noqa: E402
from rtodt import REGIMES, A0_for, z_param, db                   # noqa: E402
import system_metric as sm                                       # noqa: E402
from generate import eq22_series, ladder_K                       # noqa: E402

SNRS = [20, 24, 28, 32, 36, 40]

# The manuscript's own single validation point comes first (Sec. III-D quotes
# 4.51e-3, 2.24e-5, 6.20e-7, 9.04e-11 at 20, 28, 32, 40 dB for it), then two
# configurations that fail hardest in the shipped sweep.
CONFIGS = [
    ("strong", "0.05", "1.967"),
    ("weak", "0.05", "1.548"),
    ("weak", "0.05", "1.967"),
    ("moderate", "0.1", "2.511"),
]


def main() -> None:
    mp.mp.dps = 260
    print("Eq. (22): sweep-protocol K vs ladder-selected K, same reference\n")
    print("The sweep fixes K from z at %d dB and reuses it; the ladder picks K"
          % max(SNRS))
    print("from z at each SNR.  'inadm' means z > 8 there, where the manuscript")
    print("declares the candidate inadmissible and the guard refuses it.\n")

    for rname, ss, xs in CONFIGS:
        A, B = REGIMES[rname]
        xi, s = mp.mpf(xs), mp.mpf(ss)
        A0 = A0_for(xi, s)
        if A0 is None:
            continue
        K_sweep = ladder_K(float(z_param(A, B, A0, db(max(SNRS)))))
        print("=" * 104)
        print("%s  sigma_s=%s  xi=%s   A_0=%.6e   sweep K=%s (from z at %d dB)"
              % (rname, ss, xs, float(A0), K_sweep, max(SNRS)))
        print("=" * 104)
        print("  %-6s %-9s %-6s %-15s %-13s %-6s %-15s %-13s"
              % ("SNR", "z", "K_swp", "eq22(K_swp)", "rel_swp", "K_lad",
                 "eq22(K_lad)", "rel_lad"))
        print("  " + "-" * 100)

        # one series build per distinct order actually needed
        needed = {K_sweep} | {ladder_K(float(z_param(A, B, A0, db(g))))
                              for g in SNRS}
        needed = {k for k in needed if k is not None}
        series = {}
        for K in sorted(needed):
            t0 = time.time()
            series[K], nexp = eq22_series(A, B, xi, A0, K, SNRS)
            print("  [built K=%-3d %6d exponents in %5.1f s]"
                  % (K, nexp, time.time() - t0))

        for g in SNRS:
            gbar = float(db(g))
            z = float(z_param(A, B, A0, db(g)))
            K_lad = ladder_K(z)
            ref = sm.system_aber(float(A), float(B), float(xi), float(A0),
                                 gbar, method="quad")

            def rel(K):
                if K is None or K not in series or not ref:
                    return None
                return (float(series[K][g]) - ref) / ref * 100.0

            r_s, r_l = rel(K_sweep), rel(K_lad)
            print("  %-6s %-9.4f %-6s %-15s %-13s %-6s %-15s %-13s"
                  % ("%d dB" % g, z, K_sweep,
                     "%.6e" % float(series[K_sweep][g]) if K_sweep in series else "-",
                     "%+.4f%%" % r_s if r_s is not None else "-",
                     K_lad if K_lad is not None else "inadm",
                     "%.6e" % float(series[K_lad][g]) if K_lad in series else "-",
                     "%+.4f%%" % r_l if r_l is not None else "-"))
        print()

    print("Read the last two columns together with the two before them: where")
    print("rel_swp is enormous and rel_lad is small, the shipped sweep's row is")
    print("measuring its own K-selection, not the accuracy of Eq. (22).")


if __name__ == "__main__":
    main()
