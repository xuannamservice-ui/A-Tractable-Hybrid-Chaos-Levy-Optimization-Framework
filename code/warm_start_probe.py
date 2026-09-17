"""
Real measurement of the MPC warm-start mechanism (Sec. limitations: "specified
and not yet deployed" until this script and the matching switch added to
BeamSteeringMPC.step -- see `warm_start=True` in mpc_loop.py).

WHAT THIS MEASURES
-------------------
Under the interpreted implementation only ONE iteration of H-CLPSO-GA
completes before the tau_O=600 us anytime checkpoint fires (measured
separately, Table `tab:tail_mitigation`). With one iteration, the achieved
cost is set almost entirely by where the swarm was INITIALISED, not by
PSO/GA refinement. A cold cycle draws that initial swarm uniformly (chaotic
map) over the whole decision box; a warm cycle anchors it on last cycle's
solved trajectory, shifted one stage forward. This script runs both arms,
paired, over a closed-loop channel-and-sway trace, and reports the achieved
ranking cost each cycle actually reached at the SAME checkpoint -- not an
assumption that warm-starting helps, a measurement of whether it does here.

Nothing here is synthesised: every reported cost is `BeamSteeringMPC.step`'s
own `res.best_f`, the identical ranking objective `_objective` returns and
`system_metric`-scored elsewhere in this release; every channel/sway sample
comes from `channel.GammaGammaAR1` / `channel.SwayProcess`, the same
generators every other closed-loop script in this release uses.

USAGE
    python warm_start_probe.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from channel import GammaGammaAR1, SwayProcess
from mpc_loop import BeamSteeringMPC

ALPHA, BETA = 1.2, 1.1          # strong turbulence, the paper's main operating point
GBAR_DB = 38.0
GBAR = 10.0 ** (GBAR_DB / 10.0)
SIGMA_S = 0.10
HORIZON = 20
N_SEEDS = 20
CYCLES_PER_SEED = 100
BURN_IN = 200


def _run_arm(seed, warm_start, theta_trace, h_trace):
    mpc = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=seed,
                          warm_start=warm_start)
    mpc.u_prev = np.zeros(2)
    T = mpc.horizon
    costs = np.empty(len(h_trace))
    iters = np.empty(len(h_trace), dtype=int)
    n_none = 0
    for i, (theta, h) in enumerate(zip(theta_trace, h_trace)):
        res = mpc.step(theta, h_meas=h)
        if res.best_x is None:
            n_none += 1
            costs[i] = np.nan
            iters[i] = res.iterations
            continue
        costs[i] = res.best_f
        iters[i] = res.iterations
        mpc.u_prev = np.array([res.best_x[T], res.best_x[2 * T]])
    return costs, iters, n_none


def run():
    np.seterr(all="ignore")
    cold_all, warm_all = [], []
    cold_iters, warm_iters = [], []
    for s in range(N_SEEDS):
        sway = SwayProcess(SIGMA_S, seed=3000 + s)
        for _ in range(BURN_IN):
            sway.step()
        theta_trace = [sway.step() for _ in range(CYCLES_PER_SEED)]

        ch = GammaGammaAR1(ALPHA, BETA, rho_a=0.98, seed=4000 + s, calibrate=False)
        h_trace = [ch.step() - 1.0 for _ in range(CYCLES_PER_SEED)]

        # cold and warm arms replay the IDENTICAL theta/h trace (paired), and
        # the SAME solver seed, so the only difference between them is the
        # swarm's initial placement.
        c_costs, c_iters, c_none = _run_arm(5000 + s, False, theta_trace, h_trace)
        w_costs, w_iters, w_none = _run_arm(5000 + s, True, theta_trace, h_trace)
        cold_all.append(c_costs); warm_all.append(w_costs)
        cold_iters.append(c_iters); warm_iters.append(w_iters)

    cold = np.concatenate(cold_all)
    warm = np.concatenate(warm_all)
    ci = np.concatenate(cold_iters)
    wi = np.concatenate(warm_iters)

    ok = np.isfinite(cold) & np.isfinite(warm)
    diff = cold[ok] - warm[ok]   # positive: warm start reached a LOWER (better) cost

    print(f"cycles: {cold.size}  (both arms admissible: {int(ok.sum())}, "
          f"cold no-candidate: {int(np.sum(~np.isfinite(cold)))}, "
          f"warm no-candidate: {int(np.sum(~np.isfinite(warm)))})")
    print(f"iterations completed per cycle -- cold: median {np.median(ci):.1f} "
          f"(min {ci.min()}, max {ci.max()})  warm: median {np.median(wi):.1f} "
          f"(min {wi.min()}, max {wi.max()})")
    print(f"achieved ranking cost -- cold median {np.median(cold[ok]):.6e}  "
          f"warm median {np.median(warm[ok]):.6e}")
    print(f"paired diff (cold - warm), positive = warm better: "
          f"mean {diff.mean():.6e}  median {np.median(diff):.6e}")
    try:
        p = wilcoxon(diff).pvalue
    except ValueError:
        p = float("nan")
    print(f"Wilcoxon signed-rank p = {p:.3e}")
    frac_warm_better = float(np.mean(diff > 0))
    frac_tied = float(np.mean(diff == 0))
    print(f"fraction of cycles warm strictly better: {frac_warm_better:.4f}  "
          f"tied: {frac_tied:.4f}")


if __name__ == "__main__":
    run()
