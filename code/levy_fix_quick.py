"""Quick check: does the feas_shift jump geometry convert into an ABER win?

levy_envelope.py established that with the deployed per_dim jump the heavy
tail never separates from Gaussian.  levy_feasible_jump.py established the
kinematic reason: per_dim is a low-pass-filtered jump (tail deleted), while
feas_shift preserves ~95% of the tail ratio.  This script asks the remaining
question: on the ACTUAL objective, paired draws, same seeds, does Levy now
beat Gaussian under feas_shift where it did not under per_dim?

Small trials on purpose -- this is a first look, not the shipped measurement.
"""
from __future__ import annotations

import argparse
import numpy as np

from measure_all import N_P, SIGMAS, _make_problem, system_success
from ablation_continuous import wilcoxon_signed_rank


def one_cell(cfgkw, order, iters, rank_stages, jump_mode):
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
        for tag, store in (("levy", L), ("gauss", G)):
            cfg = SolverConfig(n_particles=N_P, max_iters=iters,
                               use_levy=(tag == "levy"), jump_mode=jump_mode,
                               **cfgkw)
            sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks,
                           repair=repair, block_slew=m.block_slew())
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
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--scales", default="0.02")
    ap.add_argument("--lambdas", default="1.5")
    a = ap.parse_args()

    scales = [float(x) for x in a.scales.split(",")]
    lambdas = [float(x) for x in a.lambdas.split(",")]

    per_cell = max(1, a.trials // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260901).shuffle(order)

    print("Levy vs Gaussian on system ABER, paired draws, %d trials/cell, "
          "T_iter=%d\n" % (len(order), a.iters))
    print("  %-10s %-7s %-5s %-11s %-9s %8s %7s"
          % ("jump_mode", "sigma", "lam", "rank", "med dlog10", "p", "n"))
    print("  " + "-" * 64)
    for mode in ("per_dim", "feas_shift"):
        for sc in scales:
            for lam in lambdas:
                for rk in (1, 20):
                    d = one_cell(dict(jump_scale=sc, levy_lambda=lam),
                                 order, a.iters, rk, mode)
                    w = wilcoxon_signed_rank(d)
                    med = float(np.median(d)) if d.size else float("nan")
                    star = "*" if (w["p"] == w["p"] and w["p"] < 0.05) else " "
                    print("  %-10s %-7.3f %-5.1f %-11d %+9.4f %7.4f%s %5d"
                          % (mode, sc, lam, rk, med, w["p"], star, d.size))
    print("\n  * = p < 0.05 raw; negative med dlog10 => Levy better ABER")


if __name__ == "__main__":
    main()
