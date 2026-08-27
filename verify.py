"""Reviewer entry point. Runs the checks that can pass or fail deterministically.

Design rule: every check here either regenerates a published quantity from the printed
equations, or compares two INDEPENDENT computations of the same quantity. Nothing here
takes a published number as an input and then reports that arithmetic on it closes -- that
is a transcription check, not a reproduction, and the two scripts in this package that do
it are named `check_*` rather than `reproduce_*` for exactly that reason.

A check that fails prints FAIL. It is not softened, and no tolerance here was chosen after
seeing the number it had to admit.

Usage:
    python verify.py            # tier 1, a couple of minutes
    python verify.py --full     # adds the slower measurements
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
DATA = os.path.join(HERE, "data")

ROWS = []


def record(name, ok, detail, tier=1):
    ROWS.append((tier, name, ok, detail))
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


# ------------------------------------------------------------------ tier 1
def t1_model_faithful():
    """Geometry and coefficients rebuilt from eqs. (3), (18), (20) against printed values."""
    r = subprocess.run([sys.executable, "validate_model.py"], cwd=CODE,
                       capture_output=True, text=True, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0 and "FAIL" not in out.upper().replace("FAILURE MODES", "")
    n = out.upper().count("OK")
    record("model rebuilt from the printed equations", ok,
           "%d checks, exit %d" % (n, r.returncode))


def t1_kernel_vs_arbitrary_precision():
    """Deployed float64 kernel against the same expressions in 200-digit arithmetic."""
    sys.path.insert(0, CODE)
    import numpy as np
    from rtodt_fast import pe_series_f64
    import rtodt as ref

    worst = 0.0
    for (A, B) in ((4.2, 3.0), (2.1, 1.5), (1.2, 1.1)):
        for K in (5, 10, 20):
            for xi in (0.9, 1.967, 3.1, 4.7):
                A0, g = 0.129, 10 ** 3.8
                a = float(pe_series_f64(A, B, np.array([xi]), np.array([A0]), g, K)[0])
                b = float(ref.Pe_series(A, B, ref.mp.mpf(xi), ref.mp.mpf(A0),
                                        ref.mp.mpf(g), K))
                if b != 0 and np.isfinite(a):
                    worst = max(worst, abs(a - b) / abs(b))
    record("float64 kernel vs 200-digit reference", worst < 1e-11,
           "worst relative %.2e (float64 floor ~1e-16 x cancellation)" % worst)


def t1_offgrid_dataset():
    """The released off-grid file: does it still say what the paper says it says?"""
    import csv, math
    p = os.path.join(DATA, "04_offgrid_error", "offgrid_error.csv")
    if not os.path.exists(p):
        record("off-grid error dataset", False, "file missing", 1)
        return
    rows = list(csv.DictReader(io.open(p, encoding="utf-8")))
    e, inr = [], 0
    for r in rows:
        try:
            v = float(r["abs_err_interp_free"])
            if math.isfinite(v):
                e.append(abs(v))
        except (KeyError, ValueError):
            pass
        if r.get("f64_in_range") in ("1", "True", "true"):
            inr += 1
    e.sort()
    ok = bool(e) and e[-1] < 1e-6 and inr == len(rows)
    record("off-grid: every value a probability, error in budget", ok,
           "n=%d  max %.3e (%.0fx inside 1e-6)  in-range %d/%d"
           % (len(rows), e[-1], 1e-6 / e[-1], inr, len(rows)))


def t1_eq22_in_band():
    """eq. (22) against an independent convolution reference, separated by band."""
    import csv, math
    p = os.path.join(DATA, "05_eq22_validation", "eq22_vs_reference.csv")
    if not os.path.exists(p):
        record("eq. (22) in-band agreement", False, "file missing")
        return
    rows = list(csv.DictReader(io.open(p, encoding="utf-8")))
    v = []
    for r in rows:
        if r.get("admissible") not in ("1", "True", "true"):
            continue
        if r.get("comparison_valid") not in ("1", "True", "true"):
            continue          # reference itself unconverged: not a comparison
        try:
            x = abs(float(r["rel_diff_percent"]))
            if math.isfinite(x):
                v.append(x)
        except (KeyError, ValueError):
            pass
    v.sort()
    ok = bool(v) and v[len(v) // 2] < 0.05 and v[-1] < 0.2
    record("eq. (22) vs independent reference, in band", ok,
           "n=%d  median %.4f%%  max %.4f%%" % (len(v), v[len(v) // 2], v[-1]) if v
           else "no comparable rows")


def t1_admissibility_bounds():
    """Table 7 regenerated from eqs. (18), (20), (26) rather than transcribed."""
    r = subprocess.run([sys.executable, "admissibility_bounds.py"], cwd=CODE,
                       capture_output=True, text=True, timeout=1800)
    out = (r.stdout or "") + (r.stderr or "")
    record("Table 7 bounds regenerated from the equations", r.returncode == 0,
           "exit %d, %d lines of comparison" % (r.returncode, len(out.splitlines())))


def t1_surrogate_monotone():
    """The paper assumes the per-branch surrogate ranks like the system metric. Test it."""
    sys.path.insert(0, CODE)
    import numpy as np
    from scipy.stats import spearmanr
    from system_metric import BeamConfig, aber_of
    from channel import beam_geometry
    from rtodt_fast import pe_series_f64, z_of
    from hclpso_ga import ladder_order

    A, B, g, s = 1.2, 1.1, 10 ** 3.8, 0.05
    W = np.linspace(0.055, 3.0, 90)
    A0, weq = beam_geometry(W)
    z = z_of(A, B, A0, g)
    pb = pe_series_f64(A, B, weq / (2 * s), A0, g, ladder_order(z))
    sy = np.array([aber_of(BeamConfig(regime="strong", w_z=float(w),
                                      sigma_s=s, r_d=0.0), 38.0) for w in W])
    m = np.isfinite(pb) & np.isfinite(sy) & (pb >= 0) & (pb <= 0.5)
    rho = float(spearmanr(pb[m], sy[m]).statistic)
    record("surrogate ranks like the system metric", rho > 0.99,
           "Spearman %.4f over %d admissible beams" % (rho, int(m.sum())))



# --------------------------------------------- manuscript-vs-dataset agreement
def t1_manuscript_matches_data():
    """Does the manuscript cite the datasets it ships?

    This check exists because the failure it detects actually happened: the off-grid file
    was regenerated by a script that overwrites rather than appends, the row count changed
    under a manuscript that had already quoted the old one, and nothing else in the package
    would have noticed. A paper that cites n = X for a released file containing n = Y is
    caught by the first reviewer who opens the file, and the cost of that is out of all
    proportion to the cost of this check.
    """
    import csv, math, re
    tex = os.path.abspath(os.path.join(HERE, "..", "access.tex"))
    csvp = os.path.join(DATA, "04_offgrid_error", "offgrid_error.csv")
    if not (os.path.exists(tex) and os.path.exists(csvp)):
        record("manuscript cites the dataset it ships", False, "tex or csv missing")
        return

    rows = list(csv.DictReader(io.open(csvp, encoding="utf-8")))
    e = sorted(abs(float(r["abs_err_interp_free"])) for r in rows
               if r.get("abs_err_interp_free") not in (None, "", "nan")
               and math.isfinite(float(r["abs_err_interp_free"])))
    n_actual, max_actual = len(rows), e[-1]
    margin_actual = 1e-6 / max_actual

    t = io.open(tex, encoding="utf-8").read()
    # the manuscript writes counts with a thin space: 41\,674
    # The manuscript separates thousands with a LaTeX thin space, 34\,864. Building the
    # pattern from chr(92) rather than writing a backslash literal keeps it correct
    # whatever mangles the source on its way here -- an earlier version of this line used
    # "\," inside the regex, where the escape is a no-op, so it matched "34,864" and never
    # the form the manuscript actually uses. The check then passed by never looking.
    bs = chr(92)
    pat = r"(\d{1,3}(?:" + re.escape(bs) + r",\d{3})+)"
    cited_n = set(int(m.replace(bs + ",", "").replace(",", ""))
                  for m in re.findall(pat, t))
    plausible = {c for c in cited_n if 1000 <= c <= 10 ** 7}
    n_ok = n_actual in plausible

    det = "file has n=%d, max %.3e (%.0fx inside 1e-6)" % (n_actual, max_actual, margin_actual)
    if not n_ok and plausible:
        det += "; manuscript cites %s" % sorted(plausible)
    record("manuscript cites the dataset it ships", n_ok, det)


# ------------------------------------------------------------------ tier 2
def t2_feasibility_ceiling():
    """Solver-free: what SNR does the best beam in the box need, per (regime, jitter)?"""
    sys.path.insert(0, CODE)
    import numpy as np
    from system_metric import BeamConfig, aber_of
    W = np.linspace(0.055, 3.0, 40)
    feasible = 0
    for regime in ("weak", "moderate", "strong"):
        for s in (0.05, 0.10, 0.20, 0.30):
            best = np.inf
            for w in W:
                try:
                    v = aber_of(BeamConfig(regime=regime, w_z=float(w),
                                           sigma_s=s, r_d=0.0), 38.0)
                    if np.isfinite(v):
                        best = min(best, float(v))
                except Exception:
                    pass
            if best <= 1e-6:
                feasible += 1
    record("feasibility at 38 dB is link-budget bound", feasible <= 6,
           "%d of 12 (regime, jitter) cells admit any passing beam" % feasible, tier=2)


def t2_platform_timings():
    """Both platform records present, and both report pinning that took effect."""
    ok, det = True, []
    for f in ("portable_PC-i5-14600KF.json", "portable_Jetson-TX2-pinned.json"):
        p = os.path.join(DATA, "10_platform", f)
        if not os.path.exists(p):
            ok = False; det.append(f + " missing"); continue
        d = json.load(io.open(p, encoding="utf-8"))
        eff = d.get("pinning", {}).get("effective")
        det.append("%s pinned=%s" % (d.get("label", f), eff))
        ok = ok and bool(eff)
    record("two-platform kernel timing, pinning verified", ok, "; ".join(det), tier=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    print("\nTIER 1  deterministic: regenerated from the equations, or two independent")
    print("        computations of the same quantity. These pass or fail.\n")
    for fn in (t1_model_faithful, t1_kernel_vs_arbitrary_precision, t1_offgrid_dataset,
               t1_eq22_in_band, t1_surrogate_monotone, t1_manuscript_matches_data,
               t1_admissibility_bounds):
        try:
            fn()
        except Exception as e:
            record(fn.__name__, False, "raised %s: %s" % (type(e).__name__, e))

    if a.full:
        print("\nTIER 2  measured on this machine. Reproducible in kind; the numbers")
        print("        depend on the hardware they were taken on.\n")
        for fn in (t2_feasibility_ceiling, t2_platform_timings):
            try:
                fn()
            except Exception as e:
                record(fn.__name__, False, "raised %s: %s" % (type(e).__name__, e), 2)

    n_ok = sum(1 for _, _, ok, _ in ROWS if ok)
    print("\n" + "-" * 74)
    print("  %d/%d checks passed in %.0f s" % (n_ok, len(ROWS), time.time() - t0))
    print("""
TIER 3  what this package CANNOT reproduce, stated so no reader has to discover it:
    - the closed-loop campaign driver as originally deployed is not in the release, so
      the published optimization-success and latency figures cannot be re-executed; what
      the release supports is the bound, not the reproduction
    - the 200 us TCN INT8 inference figure is a design target on an AGX-class device. No
      engine was built for the Jetson TX2 available here and nothing measured it
    - no physical testbed or steering mirror was involved anywhere in this work
""")
    return 0 if n_ok == len(ROWS) else 1


if __name__ == "__main__":
    sys.exit(main())
