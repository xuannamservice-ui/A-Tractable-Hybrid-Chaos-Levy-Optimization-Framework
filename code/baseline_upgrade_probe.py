"""Three baselines the comparison table was missing, on the deployed protocol.

The optimizer table pits H-CLPSO-GA against PSO, CMA-ES, SLSQP and uniform
random sampling.  A reviewer can fairly object that none of these is the
obvious engineering answer to the problem as deployed, and that the SLSQP arm
is handicapped rather than beaten:

  xi_table   An offline lookup.  A designer who knows the turbulence regime and
             the jitter level precomputes the optimal divergence once and
             commands it every cycle.  No search at all, one evaluation.
  xi_table_mismatch
             The same lookup, but commanded from the NOMINAL sigma_s = 0.10 m
             entry in every cell.  A table is only as good as the condition it
             was tabulated for, and a deployed link does not get told when the
             jitter level has moved; this is what that costs.
  golden     Golden-section on the scalar divergence.  Section V-A already
             notes the per-stage cost is unimodal; this is what that observation
             implies operationally, run per cycle inside the same budget.
  sqp_smooth SLSQP as the released arm runs it, except that infeasible points
             return a finite penalty instead of +inf/NaN.  A quasi-Newton
             method cannot descend a surface that returns NaN, so the released
             arm measures the guard's return convention as much as the solver.

All three publish a w_z and are scored by `measure_all.system_success`, the
same post-EGC test at the same gbar_op on the same paired draws and seeds as
the released campaign, so the counts are directly comparable to the table.

Usage:  python baseline_upgrade_probe.py [--trials 200]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from measure_all import (GBAR_OP_DB, N_P, SIGMAS, STRONG, TAU_O, T_ITER,
                         _budgeted, _make_problem, clopper_pearson,
                         mcnemar_exact, system_success)

NEW_METHODS = ("xi_table", "xi_table_mismatch", "golden", "sqp_smooth")
# Run on the same draws so the comparison is paired rather than tabulated from
# a different campaign: `sqp` is the released arm the smoothing is meant to fix,
# `hclpso_ga` the proposed solver.
REF_METHODS = ("sqp", "hclpso_ga")
ALL_METHODS = NEW_METHODS + REF_METHODS
PENALTY = 1e3          # finite stand-in for an infeasible evaluation
NOMINAL_SIGMA = 0.10   # the cell a single-entry table would be tabulated for


def _uniform_traj(w_z, lo, hi, blocks):
    """A trajectory holding divergence at w_z, everything else at box centre."""
    x = 0.5 * (lo + hi)
    s, e = blocks[0]
    x[s:e] = w_z
    return x


def offline_table(sigma_s, seed=0):
    """The designer's precomputed divergence for this jitter level.

    Chosen once, offline, at boresight and on the same surrogate objective the
    solver ranks by -- not per trial and not using that trial's r_d, which is
    exactly what makes it a lookup rather than a search.
    """
    m, f, lo, hi, blocks, repair = _make_problem(sigma_s, 0.0, seed)
    s, e = blocks[0]
    grid = np.linspace(lo[s], hi[s], 400)
    X = np.array([_uniform_traj(w, lo, hi, blocks) for w in grid])
    v = f(repair(np.clip(X, lo, hi)))
    v = np.where(np.isfinite(v), v, np.inf)
    return float(grid[int(np.argmin(v))])


def run_method(method, f, lo, hi, blocks, repair, seed, table_wz):
    """Returns (published w_z, evaluations used)."""
    s, e = blocks[0]
    deadline = time.perf_counter() + TAU_O

    def score(w):
        X = repair(np.clip(_uniform_traj(w, lo, hi, blocks)[None, :], lo, hi))
        v = float(f(X)[0])
        return v if np.isfinite(v) else np.inf

    if method in ("xi_table", "xi_table_mismatch"):
        return float(np.clip(table_wz, lo[s], hi[s])), 1

    if method == "golden":
        a, b = float(lo[s]), float(hi[s])
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = score(c), score(d)
        n = 2
        while (b - a) > 1e-5 and n < 4 * T_ITER and time.perf_counter() < deadline:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = score(c); n += 1
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = score(d); n += 1
        return float(0.5 * (a + b)), n

    if method == "sqp_smooth":
        from scipy.optimize import minimize
        rng = np.random.default_rng(seed)
        d_ = lo.size
        x0 = lo + rng.random(d_) * (hi - lo)
        calls = {"n": 0}

        def g(x):
            calls["n"] += 1
            if time.perf_counter() > deadline:
                raise StopIteration
            X = repair(np.clip(np.atleast_2d(x), lo, hi))
            v = float(f(X)[0])
            if not np.isfinite(v):
                # Finite, and increasing in how far the point had to be moved to
                # become feasible, so the surface the solver sees is descendable.
                return PENALTY + float(np.linalg.norm(X[0] - x))
            return v
        best = None
        try:
            r = minimize(g, x0, method="SLSQP", bounds=list(zip(lo, hi)),
                         options=dict(maxiter=T_ITER))
            best = r.x
        except StopIteration:
            pass
        except Exception:
            pass
        if best is None:
            return None, calls["n"]
        X = repair(np.clip(np.atleast_2d(best), lo, hi))
        return float(X[0][s]), calls["n"]

    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    print("offline divergence table (per jitter level, computed once):")
    table = {}
    for s in SIGMAS:
        table[s] = offline_table(s)
        print(f"    sigma_s={s:.2f} m -> w_z* = {table[s]:.4f} m")

    from channel import SwayProcess
    per_cell = max(1, args.trials // len(SIGMAS))
    ind = {m: [] for m in ALL_METHODS}
    evals = {m: [] for m in ALL_METHODS}
    cells = []
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)      # same order as measure_all

    t0 = time.time()
    for s, k in order:
        seed = 900000 + int(s * 1000) * 1000 + k        # same seeds as measure_all
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        cells.append(s)
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        for meth in NEW_METHODS:
            tw = table[NOMINAL_SIGMA] if meth == "xi_table_mismatch" else table[s]
            w, n = run_method(meth, f, lo, hi, blocks, repair, seed, tw)
            ok, _ = system_success(w, s, r_d)
            ind[meth].append(bool(ok))
            evals[meth].append(n)
        for meth in REF_METHODS:
            bx, n = _budgeted(f, lo, hi, seed, meth, blocks, repair)
            w = float(bx[0]) if bx is not None else None
            ok, _ = system_success(w, s, r_d)
            ind[meth].append(bool(ok))
            evals[meth].append(n)
    print(f"  {len(order)} trials done ({time.time() - t0:.0f}s)")

    out = {"generated_by": "code/baseline_upgrade_probe.py",
           "gbar_op_db": GBAR_OP_DB, "tau_o_s": TAU_O, "trials": len(order),
           "offline_table": {f"{k:.2f}": v for k, v in table.items()},
           "results": {}}
    print(f"\n{'method':>12} {'k/n':>10} {'rate':>8} {'CP95':>18} "
          f"{'median evals':>13} {'b':>4} {'c':>4} {'McNemar p':>9}")
    ref = np.array(ind['hclpso_ga'], bool)
    for meth in ALL_METHODS:
        a = np.array(ind[meth], bool)
        kk, nn = int(a.sum()), a.size
        lo_, hi_ = clopper_pearson(kk, nn)
        med = float(np.median(evals[meth]))
        b = int(np.sum(ref & ~a)); c = int(np.sum(~ref & a))
        p = mcnemar_exact(b, c) if meth != 'hclpso_ga' else float('nan')
        out["results"][meth] = {"k": kk, "n": nn, "rate": kk / nn,
                                "ci95": [lo_, hi_], "median_evals": med,
                                "b_vs_proposed": b, "c_vs_proposed": c,
                                "p_mcnemar": p,
                                "per_sigma": {f"{sv:.2f}": int(a[np.array(cells) == sv].sum())
                                              for sv in SIGMAS}}
        print(f"{meth:>12} {kk:>4}/{nn:<5} {kk / nn:>8.3f} "
              f"[{lo_:.3f},{hi_:.3f}]{'':>3} {med:>13.1f} "
              f"{b:>4} {c:>4} {p:>9.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "21_baseline_upgrade")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "baseline_upgrade.json")
    with open(path, "w", encoding="utf-8") as f_:
        json.dump(out, f_, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
