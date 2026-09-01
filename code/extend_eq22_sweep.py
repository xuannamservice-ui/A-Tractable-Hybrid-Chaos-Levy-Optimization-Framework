"""Extend the Eq. (22) system-level sweep to the wider-jitter half of the
decision box, sigma_s in {0.2, 0.3} m.

Why this exists (reviewer-driven data gap):
    Section III-D / IX-C of the manuscript state that the system-level
    validation of Eq. (24) covers sigma_s in {0.05, 0.1} m only, leaving the
    wider-jitter half of the decision box unchecked.  Section IX-D (ii) lists
    extending the sweep to sigma_s in {0.2, 0.3} m as outstanding work.  This
    script closes that gap with the same machinery, the same 17-column schema
    and the same per-row flags as generate.py's block 05, writing to a NEW
    file so the shipped dataset is never overwritten.

The sweep protocol is identical to block_eq22: one truncation order K per
configuration chosen from z at the highest swept SNR, applied at every SNR;
`admissible` applies the ladder test row by row; `ref_resolved` requires the
two independent reference constructions to agree to better than 1%; rows
failing either flag are written with comparison_valid=0, never filtered.

Usage:  python code/extend_eq22_sweep.py [--deadline "YYYY-MM-DD HH:MM"]
"""
import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(HERE, "code")
sys.path.insert(0, HERE)   # generate.py lives at the repo root
sys.path.insert(0, CODE)

from generate import eq22_series, ladder_K, z_param, ZMAX_FOR_K, save_csv  # noqa: E402
from rtodt import REGIMES, A0_for, db                                     # noqa: E402
import system_metric as sm                                                # noqa: E402

OUT = os.path.join(HERE, "data", "05_eq22_validation",
                   "eq22_vs_reference_sigma0p2_0p3.csv")

XIS = ["1.548", "1.967", "2.511", "3.104"]
SIGS = ["0.2", "0.3"]                 # the half of the box block 05 does not cover
SNRS = [20, 24, 28, 32, 36, 40]

HDR = ["regime", "sigma_s", "xi", "K", "n_exponents", "snr_db",
       "z", "z_max_for_K", "ladder_K_at_snr", "admissible",
       "eq22", "ref_quad", "ref_logdomain", "ref_spread_percent",
       "ref_resolved", "comparison_valid", "rel_diff_percent"]


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default=None,
                    help="stop point 'YYYY-MM-DD HH:MM' (default: no limit)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    deadline = (datetime.strptime(a.deadline, "%Y-%m-%d %H:%M").timestamp()
                if a.deadline else float("inf"))
    mp.mp.dps = 260
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    n_done = n_skipped = 0
    for rname, (A, B) in REGIMES.items():
        for ss in SIGS:
            for xs in XIS:
                if time.time() > deadline:
                    save_csv(a.out, HDR, rows)
                    log("deadline reached: %d rows written (partial)" % len(rows))
                    return 1
                xi, s = mp.mpf(xs), mp.mpf(ss)
                A0 = A0_for(xi, s)
                if A0 is None:
                    log("  skip %s sigma=%s xi=%s (A0 None)" % (rname, ss, xs))
                    n_skipped += 1
                    continue
                K = ladder_K(float(z_param(A, B, A0, db(max(SNRS)))))
                if K is None:
                    log("  skip %s sigma=%s xi=%s (K=None at max SNR)"
                        % (rname, ss, xs))
                    n_skipped += 1
                    continue
                t0 = time.time()
                try:
                    ser, nexp = eq22_series(A, B, xi, A0, K, SNRS)
                    for gdb in SNRS:
                        if time.time() > deadline:
                            save_csv(a.out, HDR, rows)
                            log("deadline reached: %d rows (partial)" % len(rows))
                            return 1
                        gbar = float(db(gdb))
                        z = float(z_param(A, B, A0, db(gdb)))
                        Kl = ladder_K(z)
                        adm = int(Kl is not None and K >= Kl)
                        rq = sm.system_aber(float(A), float(B), float(xi),
                                            float(A0), gbar, method="quad")
                        rf = sm.system_aber(float(A), float(B), float(xi),
                                            float(A0), gbar, method="fast")
                        spread = (abs(rq - rf) / abs(rq) * 100.0
                                  if rq else float("nan"))
                        resolved = int(rq > 0.0 and np.isfinite(spread)
                                       and spread < 1.0)
                        v = float(ser[gdb])
                        rel = (v - rq) / rq * 100 if rq else float("nan")
                        rows.append([rname, ss, xs, K, nexp, gdb,
                                     "%.4f" % z, ZMAX_FOR_K[K],
                                     Kl if Kl is not None else -1, adm,
                                     "%.8e" % v, "%.8e" % rq, "%.8e" % rf,
                                     "%.6f" % spread, resolved,
                                     int(adm and resolved), "%+.4f" % rel])
                    n_done += 1
                    log("  eq22: %s sigma=%s xi=%s K=%d (%d configs done, %.0fs)"
                        % (rname, ss, xs, K, n_done, time.time() - t0))
                except Exception as e:
                    log("  eq22 config failed %s/%s/%s: %s" % (rname, ss, xs, e))
                save_csv(a.out, HDR, rows)

    save_csv(a.out, HDR, rows)
    log("done: %d rows (%d configs, %d skipped) -> %s"
        % (len(rows), n_done, n_skipped, a.out))

    # in-band statistics, same scoping language as eq22_summary
    import csv
    with open(a.out, encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    ib = [r for r in rd if r["admissible"] == "1" and r["comparison_valid"] == "1"]
    if ib:
        rels = sorted(abs(float(r["rel_diff_percent"])) for r in ib)
        med = rels[len(rels) // 2]
        log("in-band valid rows: %d/%d; median |rel%%| = %.4f, max = %.4f"
            % (len(ib), len(rd), med, rels[-1]))
    else:
        log("NO in-band valid rows at sigma_s in {0.2, 0.3} — out-of-band "
            "coverage is exactly what this dataset documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
