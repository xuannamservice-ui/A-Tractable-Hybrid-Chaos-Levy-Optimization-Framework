"""Is the allocation sweep measuring the allocation, or measuring which basin one
sequential trajectory happened to fall into?

THE SUSPICION

`levy_mixture_probe.py` and `levy_warmstart_probe.py` each run ONE sequential closed loop
per arm and then apply a Wilcoxon signed-rank test across its 400 cycles. The cycles are
not independent: cycle t warm-starts from cycle t-1's solution, so a single bad command
propagates, and the arm's median is a property of the trajectory as much as of the
setting under test. The unit of analysis should be the trajectory, not the cycle.

The sweep's own output is what raises this. Within one law and one cell the median ABER
moves by up to two decades between adjacent allocations, non-monotonically, in both laws:
10% Gaussian particles reads as catastrophic while 25% reads as fine and 50% as bad
again. No allocation mechanism produces that. Trajectory-level luck does.

THE TEST, WHICH SETTLES IT EITHER WAY

Hold the setting completely fixed and vary only the seed. If between-SEED spread at a
fixed allocation is comparable to, or larger than, the between-ALLOCATION spread the
sweep reports, then the sweep resolves nothing and its p-values are inflated by serial
dependence. If between-seed spread is small, the sweep's differences are real and the
allocation curve can be read as it stands.

Three settings are carried so the comparison is like-for-like: the deployed control
(f = 0) and the two Levy allocations whose reported medians differ by two decades
(f = 0.10 and f = 0.25) in the tuned cell. Same cell, same cycle count, same everything
but the seed, which drives both the disturbance traces and the solver's own draws.

Usage: python trajectory_variance_check.py [--seeds 6] [--cycles 400]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_all as _ma
_ma.RANK_STAGES = None
from measure_all import GBAR_OP_DB, system_success
from mpc_loop import BeamSteeringMPC
from mpc_fast import install
from channel import SwayProcess, GammaGammaAR1

STRONG = (1.2, 1.1)
T_ITER = 25
SIGMA_S, SPREAD = 0.05, 0.00125          # the tuned cell, rho = 0.5
SETTINGS = (("levy", 0.00), ("levy", 0.10), ("levy", 0.25))


def one_run(sigma_s, spread, law, frac, seed, cycles):
    gbar = 10.0 ** (GBAR_OP_DB / 10.0)
    sway = SwayProcess(sigma_s, seed=1000 + seed)
    for _ in range(500):
        sway.step()
    thetas = np.array([sway.step() for _ in range(cycles)])
    ch = GammaGammaAR1(*STRONG, rho_a=0.98, seed=2000 + seed, calibrate=False)
    hs = np.array([ch.step() for _ in range(cycles)])

    m = BeamSteeringMPC(*STRONG, sigma_s, gbar, horizon=20, seed=seed, rank_stages=None)
    install(m)
    m.tau_o = None
    m.cfg.max_iters = T_ITER
    m.cfg.init_law = law
    m.cfg.init_spread = spread
    m.cfg.init_levy_fraction = frac
    m.u_prev = np.zeros(2)
    T = m.horizon
    anchor, out = None, []
    for th, h in zip(thetas, hs):
        m.cfg.init_centre = anchor
        r = m.step(th, h)
        if r.best_x is not None:
            anchor = r.best_x.copy()
            m.u_prev = np.array([r.best_x[T], r.best_x[2 * T]])
            w = float(r.best_x[0])
        else:
            w = None
        out.append(system_success(w, sigma_s, m.L * float(np.linalg.norm(th)))[1])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    a = ap.parse_args()
    seeds = [4242 + 1000 * i for i in range(a.seeds)]

    print("  sigma_s=%.2f spread=%.6g (rho=0.5), %d cycles, %d independent seeds per setting"
          % (SIGMA_S, SPREAD, a.cycles, a.seeds))
    print("  each entry is one trajectory's median system ABER\n")
    print("  %-12s %s" % ("setting", "  ".join("seed %d" % s for s in seeds)))

    res, t0 = {}, time.time()
    for law, frac in SETTINGS:
        meds = []
        for s in seeds:
            v = one_run(SIGMA_S, SPREAD, law, frac, s, a.cycles)
            meds.append(float(np.nanmedian(v)))
        lg = np.log10(np.array(meds))
        res["%s_f%.2f" % (law, frac)] = dict(
            law=law, fraction=frac, seeds=seeds, medians=meds,
            log10_mean=float(np.mean(lg)), log10_sd=float(np.std(lg, ddof=1)),
            log10_range=float(lg.max() - lg.min()))
        print("  %-12s %s" % ("%s f=%.2f" % (law, frac),
                              "  ".join("%.2e" % m for m in meds)))
        print("  %-12s log10 mean %+.3f, SD %.3f, range %.3f decades  (%.0fs)"
              % ("", np.mean(lg), np.std(lg, ddof=1), lg.max() - lg.min(), time.time() - t0),
              flush=True)

    print("\n  --- the comparison that decides whether the sweep resolves anything ---")
    between_seed = max(v["log10_range"] for v in res.values())
    a0 = res["levy_f0.00"]["log10_mean"]
    between_alloc = max(abs(v["log10_mean"] - a0) for k, v in res.items() if k != "levy_f0.00")
    print("  largest between-SEED range at a fixed allocation : %.3f decades" % between_seed)
    print("  largest between-ALLOCATION shift in seed-mean     : %.3f decades" % between_alloc)
    if between_seed >= between_alloc:
        print("\n  The sweep does NOT resolve the allocation. Trajectory-to-trajectory")
        print("  variation at a FIXED setting is as large as the differences it reports")
        print("  between settings, so its single-trajectory p-values are inflated by serial")
        print("  dependence and cannot be read as evidence in either direction.")
    else:
        print("\n  Between-allocation shift exceeds between-seed spread, so the allocation")
        print("  curve carries signal; it should still be re-estimated with the trajectory")
        print("  as the unit of analysis rather than the cycle.")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "trajectory_variance_check.json"), "w") as fh:
        json.dump(dict(what="between-seed spread of a sequential closed loop's median ABER at a "
                            "FIXED allocation, against the between-allocation differences the "
                            "mixture sweep reports; decides whether one trajectory per arm can "
                            "resolve the allocation at all",
                       sigma_s=SIGMA_S, init_spread=SPREAD, cycles=a.cycles, t_iter=T_ITER,
                       between_seed_range_decades=between_seed,
                       between_allocation_shift_decades=between_alloc,
                       results=res), fh, indent=1)
    print("  wrote trajectory_variance_check.json")


if __name__ == "__main__":
    main()
