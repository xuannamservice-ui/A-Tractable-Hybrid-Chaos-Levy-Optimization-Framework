"""Map where the heavy-tailed step does and does not pay, instead of testing one point.

WHY A SWEEP RATHER THAN ANOTHER SINGLE COMPARISON

levy_mechanism_probe.py compared Levy against Gaussian at the deployed settings and found
nothing: p = 0.72 gated, p = 0.997 ungated. That is one cell of a parameter space, and
Lemma 2 is explicit about which coordinate decides the answer. Its escape-probability ratio

    p_L / p_G  ->  infinity   as   r_tilde = r / sigma  ->  infinity

says the heavy tail wins when the basin is WIDE relative to the step scale. A null measured
at one sigma therefore says nothing about the operator in general; it says the deployed
sigma is not in the winning region, or that something else is preventing the tail from
acting. This script maps the region.

THE AXES, AND WHY EACH IS HERE

  jump_scale (sigma)   the direct r_tilde axis. The deployed value is 0.02 of the decision
                       box. Sweeping it down raises r_tilde, which is where Lemma 2 says
                       the advantage lives; sweeping it up should destroy the advantage,
                       and observing that destruction is what makes the result a
                       measurement rather than a coincidence.

  levy_lambda          the tail exponent, deployed at 1.5. Smaller lambda is a heavier
                       tail and a stronger predicted effect. If the advantage is real it
                       must vary monotonically with this, and if it does not, any single
                       significant cell is noise.

  repair               THE MECHANISTIC SUSPECT. Every candidate is projected onto the
                       slew-feasible set before evaluation. A projection is precisely an
                       operation that truncates large excursions, so it may be removing
                       the tail exactly where the tail is supposed to act, which would
                       make the operator's failure an artefact of the constraint handling
                       rather than a property of the search. Running with the projection
                       disabled is diagnostic only: those trajectories are not physically
                       realisable and must never be reported as performance.

  rank_stages          1 gives the unimodal per-stage cost, 20 the coupled trajectory
                       where the manuscript locates the multimodality. An escape operator
                       has nothing to escape in the first.

WHAT COUNTS AS A RESULT

Levy and Gaussian are run on identical paired draws with identical seeds and an identical
trigger, so the only difference is the step distribution. Each cell reports the paired
Wilcoxon signed-rank p and the median log10 ABER difference. A cell is only interesting if
it is significant AND its neighbours trend the same way: an isolated significant cell in a
grid this size is what multiple comparisons produce for free, and it is reported with the
Holm-adjusted threshold alongside the raw one so the distinction is visible.

WHAT THIS CANNOT DO

It cannot make the deployed configuration benefit from an advantage found at some other
sigma. If a winning region exists away from the deployed point, the honest report is that
the operator has an operating envelope and the deployed setting sits outside it. That is a
finding about the design, not a defence of it.

Usage:
    python levy_envelope.py [--trials 200] [--iters 60] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from measure_all import (N_P, SIGMAS, GBAR_OP_DB, _make_problem, system_success,
                         pin_and_prioritise)
from ablation_continuous import wilcoxon_signed_rank


def _stamp(obj, script, argv=None):
    """Record what produced this artefact, in the two fields build_manifest.py reads.

    generated_by must be a bare path: the manifest validates it with os.path.basename and
    requires the result to exist under code/, so a value carrying arguments is rejected.
    The full invocation goes in command, which is what reproduces this particular run.
    """
    import sys as _sys
    args = " ".join(argv if argv is not None else _sys.argv[1:])
    out = {"generated_by": "code/%s" % script,
           "command": ("python code/%s %s" % (script, args)).rstrip()}
    out.update(obj)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, float)
    running = 0.0
    for rank, i in enumerate(idx):
        v = (m - rank) * pvals[i]
        running = max(running, v)
        adj[i] = min(1.0, running)
    return adj


def one_cell(cfgkw, order, iters, rank_stages, use_repair):
    """Levy vs Gaussian on identical paired draws. Returns the paired log10 differences."""
    from hclpso_ga import HCLPSOGA, SolverConfig
    from channel import SwayProcess
    import measure_all as _ma
    _ma.RANK_STAGES = rank_stages

    L, G = [], []
    for s, k in order:
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        rep = repair if use_repair else None
        for tag, store in (("levy", L), ("gauss", G)):
            cfg = SolverConfig(n_particles=N_P, max_iters=iters,
                               use_levy=(tag == "levy"), **cfgkw)
            sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks, repair=rep)
            r = sol.minimise(lambda X: (f(X), {}))
            w = float(r.best_x[0]) if r.best_x is not None else None
            _ok, v = system_success(w, s, r_d)
            store.append(v)
    L = np.array(L, float)
    G = np.array(G, float)
    ok = np.isfinite(L) & np.isfinite(G) & (L > 0) & (G > 0)
    return np.log10(L[ok]) - np.log10(G[ok])   # negative => Levy better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--quick", action="store_true",
                    help="coarse grid, for a first look")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "12_continuous"))
    a = ap.parse_args()

    if a.quick:
        SCALES = [0.002, 0.02, 0.2]
        LAMBDAS = [1.2, 1.5]
        RANKS = [20]
        REPAIRS = [True, False]
    else:
        SCALES = [0.001, 0.005, 0.02, 0.05, 0.15, 0.40]
        LAMBDAS = [1.1, 1.3, 1.5, 1.7]
        RANKS = [1, 20]
        REPAIRS = [True, False]

    print("  pinning %s" % pin_and_prioritise())
    grid = list(itertools.product(SCALES, LAMBDAS, RANKS, REPAIRS))
    print("  %d cells x %d trials x 2 arms, T_iter = %d"
          % (len(grid), a.trials, a.iters))
    print("  deployed point is jump_scale=0.02, lambda=1.5, repair=on")
    print()

    per_cell = max(1, a.trials // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)

    rows = []
    t0 = time.time()
    print("  %8s %7s %6s %7s %11s %9s %7s %7s"
          % ("sigma", "lambda", "rank", "repair", "med dlog10", "p", "L+", "L-"))
    print("  " + "-" * 74)
    for sc, lam, rk, rp in grid:
        d = one_cell(dict(jump_scale=sc, levy_lambda=lam), order, a.iters, rk, rp)
        w = wilcoxon_signed_rank(d)
        med = float(np.median(d)) if d.size else float("nan")
        rows.append(dict(jump_scale=sc, levy_lambda=lam, rank_stages=rk,
                         repair=bool(rp), n=int(d.size), median_log10=med,
                         p=float(w["p"]), levy_better=int((d < 0).sum()),
                         levy_worse=int((d > 0).sum()), tie=int((d == 0).sum())))
        star = "*" if (w["p"] == w["p"] and w["p"] < 0.05) else " "
        print("  %8.3f %7.1f %6d %7s %+11.5f %8.4f%s %6d %6d"
              % (sc, lam, rk, "on" if rp else "OFF", med, w["p"], star,
                 int((d < 0).sum()), int((d > 0).sum())))

    print()
    print("    %d cells in %.0fs" % (len(rows), time.time() - t0))

    ps = np.array([r["p"] for r in rows], float)
    fin = np.isfinite(ps)
    adj = np.full(len(rows), np.nan)
    if fin.any():
        adj[fin] = holm(ps[fin])
    for r, av in zip(rows, adj):
        r["p_holm"] = None if not np.isfinite(av) else float(av)

    raw = [r for r in rows if np.isfinite(r["p"]) and r["p"] < 0.05]
    survivors = [r for r in rows if r["p_holm"] is not None and r["p_holm"] < 0.05]
    levywin = [r for r in survivors if r["median_log10"] < 0]

    print()
    print("  significant before correction : %d / %d" % (len(raw), len(rows)))
    print("  surviving Holm correction     : %d" % len(survivors))
    print("  of those, favouring Levy      : %d" % len(levywin))
    if levywin:
        print()
        print("  CELLS WHERE THE HEAVY TAIL WINS:")
        for r in sorted(levywin, key=lambda r: r["median_log10"]):
            print("    sigma=%.3f lambda=%.1f rank=%d repair=%s  "
                  "%.1f%% better ABER  p_holm=%.4g"
                  % (r["jump_scale"], r["levy_lambda"], r["rank_stages"],
                     "on" if r["repair"] else "OFF",
                     (1 - 10 ** r["median_log10"]) * 100, r["p_holm"]))
    else:
        print()
        print("  No cell favours Levy after correction.")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "levy_envelope.json"), "w") as fh:
        json.dump(_stamp({"iters": a.iters, "trials": len(order), "cells": rows},
                          "levy_envelope.py"), fh, indent=1)
    print()
    print("  wrote levy_envelope.json")
    print()
    print("  A repair=OFF cell is DIAGNOSTIC ONLY. Those trajectories violate the slew")
    print("  constraint and are not deliverable; they say whether the projection is what")
    print("  removes the tail, and must never be quoted as achieved performance.")


if __name__ == "__main__":
    main()
