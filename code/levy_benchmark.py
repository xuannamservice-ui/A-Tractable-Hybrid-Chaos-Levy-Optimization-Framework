"""Controlled multimodal benchmark: when does the heavy tail pay, and why.

The FSO problem cannot answer this question -- its per-stage landscape is
unimodal (landscape_probe.json: block0_local_minima = 1), so an escape
operator has nothing to escape, and the rank-20 trajectory coupling is too
weak to exploit.  That is a finding about the PROBLEM, not about the
operator.  This script builds a controlled problem with the SAME decision
structure as the FSO controller (60-D trajectory, 3 physical blocks, box +
slew tube, forward-sweep repair) but an objective that demands escape:

    cost = +inf   for w_z block-mean in the WALL          (like the z>8 guard)
           (m_w - c_near)^2                for m_w < wall  (local attractor)
           (m_w - c_far)^2 - DEPTH         for m_w > wall  (global, deeper)

A solver sitting in the near well cannot reach the far well by local moves:
the wall scores +inf (unscoreable, exactly as an inadmissible candidate in
the FSO kernel), so every finite-cost direction pulls back toward the near
well.  The ONLY way out is a jump that TELEPORTS across the wall in one
step.  That is precisely the situation Lemma 2 describes, made measurable.

  * per_dim jumps move the block mean by the AVERAGE of T i.i.d. steps --
    the central limit thins the tail by sqrt(T) before the repair even sees
    it, and the forward-sweep repair deletes the per-stage content on top.
    Measured: p99 tail ratio collapses 8.9x -> 2.3x.  The heavy tail is
    observationally dead, so per_dim-Levy escapes no more than Gaussian.
  * feas_shift jumps move the whole block by ONE heavy-tailed scalar: the
    tail survives (3.6x -> 3.4x) and the block mean can cross the wall into
    the global well.

Escape rate (fraction of trials whose final w_z block-mean lands in the far
well) is the headline number, with paired Wilcoxon on final cost.

Diagnostic experiment, not a paper claim by itself: the FSO problem reports
its own honest null, and this benchmark scopes WHERE the operator earns its
place.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from hclpso_ga import HCLPSOGA, SolverConfig
from ablation_continuous import wilcoxon_signed_rank

T = 20
BLOCKS = [(0, T), (T, 2 * T), (2 * T, 3 * T)]
# box per block: (lo, hi)
BOX = [(0.05, 3.0), (-0.01, 0.01), (-0.01, 0.01)]
SLEW = [0.05, 5e-4, 5e-4]          # per-stage slew limit per block
# w_z block-mean landscape
WALL = (0.80, 1.60)                # infeasible wall, like the z > 8 guard
C_NEAR, C_FAR = 0.30, 2.20         # local / global well centres
DEPTH = 2.0                        # how much deeper the far well is
SHAPE_PEN = 0.01                   # penalty on per-stage scatter


def _lo_hi():
    lo = np.array([l for (l, h) in BOX] * T, dtype=float)
    hi = np.array([h for (l, h) in BOX] * T, dtype=float)
    return lo, hi


def repair(X, lo, hi):
    """Box clip + forward-sweep slew repair, exactly the FSO structure."""
    X = np.clip(X, lo, hi).copy()
    for (s, e), lim in zip(BLOCKS, SLEW):
        for k in range(s + 1, e):
            X[:, k] = np.clip(X[:, k], X[:, k - 1] - lim, X[:, k - 1] + lim)
    return X


def objective(X, wall_mode="mean"):
    """Vectorised cost.  Returns (cost, {}).  Wall -> +inf, like the guard.

    wall_mode="mean"  : the wall blocks the w_z block MEAN -- escape needs a
                        jump that moves the trajectory's average across.  Both
                        per_dim (whose surviving stage-0 pulls the repaired
                        trajectory along) and feas_shift can do this.
    wall_mode="stage" : the wall blocks ANY stage of the w_z block -- escape
                        needs the WHOLE trajectory across.  per_dim cannot:
                        its jump moves stage 0 far, but the forward-sweep
                        repair then ramps the other stages back at the slew
                        limit, stranding part of the trajectory inside the
                        wall, and the candidate scores +inf.  feas_shift moves
                        all T stages together, so no stage enters the wall.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if wall_mode == "mean":
        m_w = X[:, 0:T].mean(axis=1)
        cost = np.where(m_w < WALL[0], (m_w - C_NEAR) ** 2,
                        np.where(m_w > WALL[1], (m_w - C_FAR) ** 2 - DEPTH,
                                 np.inf))
    else:
        wz = X[:, 0:T]
        any_in_wall = ((wz >= WALL[0]) & (wz <= WALL[1])).any(axis=1)
        m_w = wz.mean(axis=1)
        cost = np.where(any_in_wall, np.inf,
                        np.where(m_w < WALL[0], (m_w - C_NEAR) ** 2,
                                 (m_w - C_FAR) ** 2 - DEPTH))
    m_az = X[:, T:2 * T].mean(axis=1)
    m_el = X[:, 2 * T:3 * T].mean(axis=1)
    cost += 0.5 * (m_az ** 2 + m_el ** 2) / (0.01 ** 2)
    cost += SHAPE_PEN * (X[:, 0:T].std(axis=1)
                         + X[:, T:2 * T].std(axis=1)
                         + X[:, 2 * T:3 * T].std(axis=1))
    return cost, {}


def one_trial(seed, cfgkw, iters, wall_mode="mean"):
    """One solver run.  Returns (best_f, escaped, final_m_w).

    The swarm warm-starts from a flat trajectory at the LOCAL well (the
    deployed MPC re-uses the previous cycle's solution as its anchor), so no
    particle starts past the wall: the far well is reachable ONLY by a jump
    that teleports across it.
    """
    lo, hi = _lo_hi()
    centre = np.concatenate([np.full(T, C_NEAR),
                             np.zeros(2 * T)])
    cfg = SolverConfig(n_particles=30, max_iters=iters,
                       init_centre=centre, init_spread=0.02, **cfgkw)
    sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=BLOCKS,
                   repair=lambda X: repair(X, lo, hi),
                   block_slew=SLEW)
    r = sol.minimise(lambda X: objective(X, wall_mode))
    if r.best_x is None:
        return float("inf"), False, float("nan")
    m_w = float(r.best_x[0:T].mean())
    escaped = bool(m_w > WALL[1])
    return float(r.best_f), escaped, m_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--wall-mode", choices=("mean", "stage"), default="mean")
    ap.add_argument("--jump-scale", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "13_levy_benchmark"))
    a = ap.parse_args()

    arms = {
        "per_dim_gauss": dict(use_levy=False, jump_mode="per_dim",
                              jump_scale=a.jump_scale),
        "per_dim_levy": dict(use_levy=True, jump_mode="per_dim",
                             jump_scale=a.jump_scale),
        "feas_shift_gauss": dict(use_levy=False, jump_mode="feas_shift",
                                 jump_scale=a.jump_scale),
        "feas_shift_levy": dict(use_levy=True, jump_mode="feas_shift",
                                jump_scale=a.jump_scale),
    }
    seeds = [1000 + k for k in range(a.trials)]
    res = {name: {"f": [], "esc": [], "mw": []} for name in arms}
    t0 = time.time()
    for j, seed in enumerate(seeds):
        for name, kw in arms.items():
            fv, esc, mw = one_trial(seed, kw, a.iters, a.wall_mode)
            res[name]["f"].append(fv)
            res[name]["esc"].append(esc)
            res[name]["mw"].append(mw)
        if (j + 1) % 20 == 0:
            print("  %d/%d trials (%.0fs)" % (j + 1, a.trials, time.time() - t0),
                  flush=True)

    print("\nControlled multimodal benchmark: %d trials, T_iter=%d, jump_scale=%.2f\n"
          % (a.trials, a.iters, a.jump_scale))
    print("  infeasible wall on w_z block-mean: [%.2f, %.2f] (like the z>8 guard)"
          % WALL)
    print("  local well at %.2f (cost (m-%.2f)^2); global well at %.2f, %.1f deeper\n"
          % (C_NEAR, C_NEAR, C_FAR, DEPTH))
    print("  %-16s %12s %12s %12s" % ("arm", "median f", "escape rate", "final m_w"))
    print("  " + "-" * 56)
    med = {}
    for name, r in res.items():
        f = np.array(r["f"], float)
        ok = np.isfinite(f)
        med[name] = float(np.median(f[ok])) if ok.any() else float("nan")
        esc = float(np.mean(r["esc"]))
        mw = float(np.median(r["mw"]))
        print("  %-16s %12.5f %12.3f %12.3f" % (name, med[name], esc, mw))

    print("\n  paired McNemar (escape) and Wilcoxon (final cost, raw diff):")
    names = list(arms)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            ea = np.array(res[na]["esc"], bool)
            eb = np.array(res[nb]["esc"], bool)
            b = int(np.sum(ea & ~eb)); c = int(np.sum(~ea & eb))
            fa = np.array(res[na]["f"], float)
            fb = np.array(res[nb]["f"], float)
            m = np.isfinite(fa) & np.isfinite(fb)
            d = fa[m] - fb[m]                      # negative => row better
            w = wilcoxon_signed_rank(d) if m.sum() >= 10 else {"p": float("nan")}
            print("    %-16s vs %-16s  esc %d/%d  p_mc=%.4f  med d=%.4f  p_w=%.4f%s"
                  % (na, nb, b, c,
                     (2.0 * min(1.0, _mcnemar(b, c))),
                     float(np.median(d)) if m.sum() else float("nan"),
                     w["p"], "*" if w["p"] == w["p"] and w["p"] < 0.05 else ""))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "levy_benchmark.json")
    with open(path, "w") as fh:
        json.dump({
            "generated_by": "code/levy_benchmark.py",
            "command": "python code/levy_benchmark.py --trials %d --iters %d "
                       "--jump-scale %.2f" % (a.trials, a.iters, a.jump_scale),
            "structure": "60-D trajectory, 3 blocks, box+slew tube, forward-sweep "
                         "repair, infeasible wall like the z>8 guard",
            "wall": WALL, "wells": {"near": C_NEAR, "far": C_FAR, "depth": DEPTH},
            "trials": a.trials, "iters": a.iters, "jump_scale": a.jump_scale,
            "median_f": {k: med[k] for k in names},
            "escape_rate": {k: float(np.mean(res[k]["esc"])) for k in names},
        }, fh, indent=1)
    print("\n  wrote %s" % path)


def _mcnemar(b, c):
    """Two-sided exact McNemar p on discordant counts (b, c)."""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    return float(2.0 * min(1.0, binom.sf(k - 1, n, 0.5)))


if __name__ == "__main__":
    main()
