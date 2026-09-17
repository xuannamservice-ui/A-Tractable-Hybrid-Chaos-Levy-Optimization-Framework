"""Does adapting the divergence every cycle beat commanding a tabulated one?

Every single-shot success rate in this paper ties: at 38, 38.5, 39 and 40 dB
the full kernel, its ablations, plain PSO and uniform random sampling return
the same count, and a one-evaluation offline lookup ties them too.  That
criterion is a threshold on one cycle at a fixed reference SNR, so it cannot
see the thing a predictive controller is for -- tracking a channel that moves.

This measures the thing it cannot see.  Three controllers see the IDENTICAL
channel realisation, the IDENTICAL sway draws and the IDENTICAL steering law,
so the ONLY difference between them is the commanded divergence:

  solver   w_z from BeamSteeringMPC.step() each cycle, which ranks candidates
           at the per-stage reference SNR gbar_k = gbar h_hat_k^2 and therefore
           re-optimises as the Kalman predictor tracks the fading level
  table    w_z fixed at the offline optimum for this (regime, sigma_s) cell --
           the baseline that ties the solver on every success rate here
  safe     w_z fixed at the xi_safe fallback beam
  oracle   w_z re-optimised every cycle against the TRUE realised level, an
           upper bound on what any amount of per-cycle divergence adaptation
           could buy.  If the oracle barely beats the table, adaptation is
           worth nothing here; if it beats it and the solver does not, the
           deficit is in the stage cost, not in the idea.

Because the steering command is common to all three, the pointing state r_d(t)
is identical across arms and the comparison isolates divergence adaptation.
Scoring is the real post-EGC system ABER (`system_metric.aber_of`) at the
realised (w_z, r_d), not the search's own ranking cost, and the statistic is
the paired per-cycle difference over the run.

Usage:  python closed_loop_tracking_probe.py [--cycles 300] [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon

from channel import GammaGammaAR1
from mpc_loop import BeamSteeringMPC
from system_metric import BeamConfig, aber_of

ALPHA, BETA = 1.2, 1.1
LINK_LENGTH = 2000.0
U_SLEW = 50e-3 * 1e-3
U_MAX = 10e-3
TRACK_GAIN = 0.8          # the proportional boresight law, common to all arms


def offline_optimum(mpc, gbar_db, n=400):
    """The table entry: best fixed w_z at the nominal state, chosen once."""
    grid = np.linspace(mpc.wz_lo, mpc.wz_hi, n)
    vals = [aber_of(BeamConfig("strong", float(w), mpc.sigma_s, 0.0), gbar_db)
            for w in grid]
    vals = np.where(np.isfinite(vals), vals, np.inf)
    return float(grid[int(np.argmin(vals))])


def run_seed(seed, cycles, sigma_s, gbar_db, safe_wz, tau_o=None):
    gbar = 10 ** (gbar_db / 10.0)
    kw = {} if tau_o is None else {"tau_o": tau_o}
    mpc = BeamSteeringMPC(ALPHA, BETA, sigma_s, gbar, horizon=20, seed=seed, **kw)
    # Same solver, same seed, with the stage cost's h_hat^2 rescaling of the
    # reference SNR disabled.  Section IV-C notes that rescaling composes a
    # point forecast with a metric that has already marginalised the fading
    # law, so the turbulence factor enters twice; this arm measures what that
    # costs rather than arguing about it.
    mpc_ns = BeamSteeringMPC(ALPHA, BETA, sigma_s, gbar, horizon=20, seed=seed, **kw)
    mpc_ns._stage_gbar = lambda h_pred: np.full(mpc_ns.horizon, mpc_ns.gbar)
    ch = GammaGammaAR1(ALPHA, BETA, rho_a=0.98, seed=20260918 + seed,
                       calibrate=False)
    for _ in range(200):
        v = ch.step() - 1.0
        mpc.kf.update(v)
        mpc_ns.kf.update(v)

    table_wz = offline_optimum(mpc, gbar_db)
    theta = np.array([3.0e-5, 2.0e-5])
    mpc.u_prev = np.zeros(2)
    mpc_ns.u_prev = np.zeros(2)
    rng = np.random.default_rng(5000 + seed)

    pe = {"solver": [], "solver_nohscale": [], "table": [], "safe": [], "oracle": []}
    w_hist = []
    for c in range(cycles):
        h_true = ch.step()
        h_meas = h_true - 1.0
        res = mpc.step(theta.copy(), h_meas=h_meas)
        w_solver = float(res.best_x[0]) if res.best_x is not None else mpc.wz_lo
        w_solver = float(np.clip(w_solver, mpc.wz_lo, mpc.wz_hi))
        res_ns = mpc_ns.step(theta.copy(), h_meas=h_meas)
        w_ns = float(res_ns.best_x[0]) if res_ns.best_x is not None else mpc_ns.wz_lo
        w_ns = float(np.clip(w_ns, mpc_ns.wz_lo, mpc_ns.wz_hi))

        # one steering law for every arm, so r_d is common and only w_z differs
        u_cmd = np.clip(TRACK_GAIN * theta, -U_MAX, U_MAX)
        u_out = np.clip(u_cmd, mpc.u_prev - U_SLEW, mpc.u_prev + U_SLEW)
        mpc.u_prev = u_out
        theta = theta - u_out + rng.normal(0.0, 5.59e-6, 2)
        r_d = LINK_LENGTH * float(np.linalg.norm(theta))

        # Score CONDITIONALLY on the level actually realised this cycle:
        # gamma = gbar h^2 of eq. (16), the same convention
        # predictor_ablation.py uses.  Scoring at the fixed reference SNR
        # instead would be the marginal ABER, whose minimiser is by
        # construction the fixed table value -- that test cannot be lost by a
        # table and cannot be won by a controller that conditions on h.
        gbar_true_db = gbar_db + 20.0 * np.log10(max(h_true, 1e-6))
        r = minimize_scalar(
            lambda w: np.log10(max(aber_of(BeamConfig("strong", float(w), sigma_s,
                                                      r_d), gbar_true_db), 1e-18)),
            bounds=(mpc.wz_lo, mpc.wz_hi), method="bounded",
            options=dict(xatol=1e-4))
        w_oracle = float(r.x)
        for name, w in (("solver", w_solver), ("solver_nohscale", w_ns),
                        ("table", table_wz), ("safe", safe_wz),
                        ("oracle", w_oracle)):
            pe[name].append(aber_of(BeamConfig("strong", w, sigma_s, r_d),
                                    gbar_true_db))
        w_hist.append(w_solver)

    return {k: np.asarray(v, float) for k, v in pe.items()}, table_wz, np.asarray(w_hist)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sigma-s", type=float, default=0.10)
    ap.add_argument("--gbar-db", type=float, default=38.0)
    ap.add_argument("--safe-wz", type=float, default=0.157)
    ap.add_argument("--tau-o", type=float, default=None,
                    help="solver budget in seconds; the deployed 600 us lets the\n"
                         "interpreted solver complete one iteration, so a larger\n"
                         "value separates the algorithm from the dispatch cost")
    args = ap.parse_args()

    all_pe = {"solver": [], "solver_nohscale": [], "table": [], "safe": [], "oracle": []}
    tables, spreads = [], []
    for s in range(args.seeds):
        pe, twz, wh = run_seed(s, args.cycles, args.sigma_s, args.gbar_db,
                               args.safe_wz, tau_o=args.tau_o)
        for k in all_pe:
            all_pe[k].append(pe[k])
        tables.append(twz)
        spreads.append((float(wh.min()), float(np.median(wh)), float(wh.max())))
        print(f"  seed {s}: table w_z={twz:.4f} m | solver w_z spans "
              f"{wh.min():.4f}-{wh.max():.4f} (median {np.median(wh):.4f})")

    pe = {k: np.concatenate(v) for k, v in all_pe.items()}
    n = pe["solver"].size
    out = {"generated_by": "code/closed_loop_tracking_probe.py",
           "cycles": args.cycles, "seeds": args.seeds, "n_paired": int(n),
           "sigma_s": args.sigma_s, "gbar_db": args.gbar_db,
           "tau_o": args.tau_o,
           "table_wz": tables, "solver_wz_min_med_max": spreads, "arms": {}}

    print(f"\n{'arm':>8} {'mean ABER':>13} {'median ABER':>13} {'cycles below 1e-6':>19}")
    for k in ("solver", "solver_nohscale", "table", "safe", "oracle"):
        v = pe[k]
        out["arms"][k] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                          "frac_below_target": float(np.mean(v <= 1e-6))}
        print(f"{k:>8} {np.mean(v):>13.4e} {np.median(v):>13.4e} "
              f"{100 * np.mean(v <= 1e-6):>18.2f}%")

    print(f"\npaired per-cycle comparison against the solver (n={n})")
    for k in ("solver_nohscale", "table", "safe", "oracle"):
        d = pe[k] - pe["solver"]          # >0 means the solver is better
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = float("nan")
        better = float(np.mean(d > 0))
        out["arms"][k].update({"mean_paired_diff_vs_solver": float(np.mean(d)),
                               "frac_cycles_solver_better": better,
                               "p_wilcoxon": float(p)})
        print(f"  {k:>6}: mean diff {np.mean(d):+.4e}   solver better on "
              f"{100 * better:5.1f}% of cycles   Wilcoxon p={p:.3e}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "22_closed_loop_tracking")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "closed_loop_tracking.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
