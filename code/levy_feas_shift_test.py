"""
Genuine test: does the feas_shift jump geometry let Levy flight separate from a
Gaussian control on the continuous ABER quality metric, at the coupled-trajectory
ranking (rank_stages=None) where the manuscript's own argument says the
multimodality lives?

Mirrors ablation_continuous.py's methodology exactly (same seeds, same
system_success evaluator, same T_ITER=25 at a budget that admits the full
iteration count, same paired Wilcoxon test), but adds two arms that were not
in the original ARMS dict:

    full_feas_shift_levy   jump_mode="feas_shift", use_levy=True
    full_feas_shift_gauss  jump_mode="feas_shift", use_levy=False

and passes block_slew explicitly (the original ablation_continuous.py never
does, so feas_shift silently falls back to per_dim there -- HCLPSOGA checks
`self.block_slew is not None` before using it).

This is a diagnostic script, not a release artefact. It reports whatever it
finds, honestly, in either direction.
"""
import sys
sys.path.insert(0, r"R:\code")

import time
import numpy as np

from measure_all import (N_P, SIGMAS, GBAR_OP_DB, TARGET, _make_problem,
                          system_success)
import measure_all as _ma
_ma.RANK_STAGES = None   # coupled trajectory: the fair test for Levy, per the manuscript

from hclpso_ga import HCLPSOGA, SolverConfig

T_ITER = 25
TAU_US = 20000.0   # 20 ms: admits the full T_ITER budget, as ablation_continuous.py's
                   # own comment specifies for testing whether components are inert
                   # only because the deployed checkpoint stops them after one iteration

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 500

ARMS = {
    "full_per_dim_levy":     dict(jump_mode="per_dim",   use_levy=True),
    "full_per_dim_gauss":    dict(jump_mode="per_dim",   use_levy=False),
    "full_feas_shift_levy":  dict(jump_mode="feas_shift", use_levy=True),
    "full_feas_shift_gauss": dict(jump_mode="feas_shift", use_levy=False),
}


def wilcoxon_signed_rank(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0.0]
    n = nz.size
    if n == 0:
        return dict(n=0, p=float("nan"))
    from scipy.stats import wilcoxon
    st, p = wilcoxon(nz, alternative="two-sided", mode="exact" if n <= 25 else "approx")
    return dict(n=int(n), W=float(st), p=float(p))


def main():
    per_cell = max(1, N_TRIALS // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)

    aber = {arm: [] for arm in ARMS}
    iters = {arm: [] for arm in ARMS}

    t0 = time.time()
    for i, (s, k) in enumerate(order):
        seed = 700000 + int(s * 1000) * 1000 + k
        from channel import SwayProcess
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        block_slew = m.block_slew()
        for arm, over in ARMS.items():
            cfg = SolverConfig(n_particles=N_P, max_iters=T_ITER, **over)
            sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks, repair=repair,
                            block_slew=block_slew)
            dl = time.perf_counter() + TAU_US * 1e-6
            r = sol.minimise(lambda X: (f(X), {}),
                             checkpoint=lambda it, bf: time.perf_counter() > dl)
            w = float(r.best_x[0]) if r.best_x is not None else None
            ok, v = system_success(w, s, r_d)
            aber[arm].append(v)
            iters[arm].append(int(r.iterations))
        if (i + 1) % 50 == 0:
            print("  %d/%d (%.0fs)" % (i + 1, len(order), time.time() - t0), flush=True)

    print("\n%d trials, T_ITER=%d, tau=%.0fus, rank_stages=%s\n"
          % (len(order), T_ITER, TAU_US, _ma.RANK_STAGES))

    ref = np.array(aber["full_per_dim_levy"], float)
    print("%-24s %11s %9s %8s %8s %8s %8s" %
          ("arm", "median ABER", "med iters", "vs REF", "Wilcox p", "better", "worse"))
    print("-" * 90)
    for arm in ARMS:
        v = np.array(aber[arm], float)
        fin = np.isfinite(v)
        med = float(np.median(v[fin])) if fin.any() else float("nan")
        medit = float(np.median(iters[arm]))
        if arm == "full_per_dim_levy":
            print("%-24s %11.3e %9.1f %8s %8s %8s %8s" % (arm, med, medit, "-", "-", "-", "-"))
        else:
            pair = np.isfinite(v) & np.isfinite(ref) & (v > 0) & (ref > 0)
            d = np.log10(v[pair]) - np.log10(ref[pair])
            w = wilcoxon_signed_rank(d)
            better = int(np.sum(d < 0)); worse = int(np.sum(d > 0))
            print("%-24s %11.3e %9.1f %+8.4f %8.4f %8d %8d"
                  % (arm, med, medit, np.median(d) if d.size else float("nan"),
                     w["p"], better, worse))

    print("\nTHE KEY COMPARISON: feas_shift levy vs feas_shift gauss")
    vL = np.array(aber["full_feas_shift_levy"], float)
    vG = np.array(aber["full_feas_shift_gauss"], float)
    pair = np.isfinite(vL) & np.isfinite(vG) & (vL > 0) & (vG > 0)
    d = np.log10(vL[pair]) - np.log10(vG[pair])
    w = wilcoxon_signed_rank(d)
    print("  n_paired=%d  median log10 delta (levy-gauss)=%+.4f  Wilcoxon p=%.4g"
          % (pair.sum(), np.median(d) if d.size else float("nan"), w["p"]))
    print("  levy better than gauss on %d, worse on %d, tie on %d"
          % (int(np.sum(d < 0)), int(np.sum(d > 0)), int(np.sum(d == 0))))

    np.savez_compressed("feas_shift_levy_test.npz",
                        sigma_s=np.array([s for s, k in order]),
                        **{arm: np.array(aber[arm], float) for arm in ARMS})
    print("\nwrote feas_shift_levy_test.npz")


if __name__ == "__main__":
    main()
