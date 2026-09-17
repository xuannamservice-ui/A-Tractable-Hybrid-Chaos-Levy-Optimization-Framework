"""Why does the exact Euclidean (Dykstra) projection give a landscape that a
single pattern-search descent cannot descend, when both projections have the
SAME reachable set (both are idempotent on U_feas and onto it, so the composed
objectives f(Pi(.)) share a range and a global minimum)?

Hypothesis: the answer is local sensitivity of the composed map, not the
attainable optimum.  A coordinate poll perturbs one ambient coordinate by delta
and sees the objective move only as far as the REPAIRED point moves.  This
script measures exactly that: the per-coordinate displacement gain

    g_i(x) = || Pi(x + delta e_i) - Pi(x) || / delta

for both repair operators, over random ambient points of the manuscript's 60-D
decision box (variant E of landscape_probe.py).  A gain near 0 means the poll
is invisible to the objective and the descent stalls; a gain near 1 means the
poll is transmitted.

Usage:  python projection_sensitivity_probe.py [--n-points 200] [--delta 1e-4]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mpc_loop import BeamSteeringMPC
import landscape_probe as lp
from dykstra_repair_probe import DykstraMPC


def gains(mpc, rng, n_points, delta):
    lo, hi = np.asarray(mpc.lower(), float), np.asarray(mpc.upper(), float)
    d = lo.size
    out = []
    for _ in range(n_points):
        x = lo + rng.random(d) * (hi - lo)
        base = mpc.repair(x[None, :])[0]
        pert = np.repeat(x[None, :], d, axis=0)
        pert[np.arange(d), np.arange(d)] += delta
        pert = np.clip(pert, lo, hi)
        rep = mpc.repair(pert)
        out.append(np.linalg.norm(rep - base[None, :], axis=1) / delta)
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--delta", type=float, default=1e-4)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "13_dykstra_probe")
    os.makedirs(out_dir, exist_ok=True)

    label, kw, _ = lp.VARIANTS[-1]          # "E. D + two-axis steering"
    assert label.startswith("E."), label
    gbar = 10 ** (lp.GBAR_DB / 10)

    o = {"generated_by": "code/projection_sensitivity_probe.py",
         "n_points": args.n_points, "delta": args.delta, "variant": label}
    for name, cls in (("causal_clip", BeamSteeringMPC), ("dykstra", DykstraMPC)):
        mpc = cls(lp.ALPHA, lp.BETA, lp.SIGMA_S, gbar, horizon=lp.HORIZON, seed=7, **kw)
        lp.prime_predictor(mpc, lp.ALPHA, lp.BETA)
        g = gains(mpc, np.random.default_rng(20260917), args.n_points, args.delta)
        o[name] = {
            "median_gain": float(np.median(g)),
            "mean_gain": float(g.mean()),
            "p90_gain": float(np.percentile(g, 90)),
            "p99_gain": float(np.percentile(g, 99)),
            "max_gain": float(g.max()),
            "frac_gain_below_0.01": float((g < 0.01).mean()),
            "frac_gain_above_0.5": float((g > 0.5).mean()),
            "frac_gain_above_1": float((g > 1.0).mean()),
        }
        print(name, json.dumps(o[name], indent=2))

    path = os.path.join(out_dir, "projection_sensitivity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
