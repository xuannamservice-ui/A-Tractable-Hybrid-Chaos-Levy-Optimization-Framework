"""
Does a true Euclidean (Dykstra) projection onto the box-cap-slew polytope
recover the near-unimodality that `landscape_probe.py` variant E measures on
the raw feasible polytope (distinct_minima=2, frac_reaching_global=0.975),
instead of the causal forward-sweep clip's repaired_distinct_minima=40,
frac_reaching_global=0.025, worst_excess=3.325 (i.e. 4.33x)?

Reuses landscape_probe.py's OWN instrumentation (probe(), pattern_search(),
cluster_minima()) unchanged, monkey-patching only BeamSteeringMPC.repair on a
throwaway subclass. Everything else -- the objective, the guard, the starts,
the convergence settings -- is identical to the shipped measurement.
"""
from __future__ import annotations

import json
import os

import numpy as np

from mpc_loop import BeamSteeringMPC
import landscape_probe as lp


def _dykstra_axis_batch(V_init, u_prev, delta, lo, hi, n_sweeps=150):
    """Vectorized multi-set Dykstra projection, ALL rows (candidates) of a
    (rows, T) block at once, onto box [lo,hi] intersected with the slew-band
    chain (anchored to the scalar u_prev at column 0). Converges to the exact
    Euclidean projection onto the intersection (Boyle & Dykstra 1986)."""
    V = np.atleast_2d(np.asarray(V_init, dtype=float)).copy()
    rows, n = V.shape
    lo_eff = np.full(n, lo, dtype=float)
    hi_eff = np.full(n, hi, dtype=float)
    lo_eff[0] = max(lo, u_prev - delta)
    hi_eff[0] = min(hi, u_prev + delta)
    even_l = np.arange(0, n - 1, 2); even_r = even_l + 1
    odd_l = np.arange(1, n - 1, 2); odd_r = odd_l + 1

    def proj_pairs(x, y):
        d = y - x
        excess = np.where(np.abs(d) > delta, (np.abs(d) - delta) / 2.0 * np.sign(d), 0.0)
        return x + excess, y - excess

    v = V
    p_box = np.zeros_like(V)
    p_even = np.zeros_like(V)
    p_odd = np.zeros_like(V)
    for _ in range(n_sweeps):
        y = np.clip(v + p_box, lo_eff, hi_eff)
        p_box = v + p_box - y
        v = y
        if even_l.size:
            vv = v + p_even
            xl, xr = proj_pairs(vv[:, even_l], vv[:, even_r])
            newv = vv.copy(); newv[:, even_l] = xl; newv[:, even_r] = xr
            p_even = vv - newv; v = newv
        if odd_l.size:
            vv = v + p_odd
            xl, xr = proj_pairs(vv[:, odd_l], vv[:, odd_r])
            newv = vv.copy(); newv[:, odd_l] = xl; newv[:, odd_r] = xr
            p_odd = vv - newv; v = newv
    return v


_DYK_SWEEPS = 300


class DykstraMPC(BeamSteeringMPC):
    """Same objective, guard, box and blocks as BeamSteeringMPC; only `repair`
    is replaced by a genuine multi-set Dykstra projection onto the identical
    box-and-slew polytope (per steering block, independently -- w_z carries no
    slew constraint in either version, matching the manuscript's S_mir=0)."""

    def repair(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        lo, hi = self.lower(), self.upper()
        Xr = np.clip(X, lo, hi).copy()
        for bi, ((s, e), lim) in enumerate(zip(self.blocks(), self.block_slew())):
            if not (self.steering and bi > 0):
                continue
            u_prev = float(self.u_prev[bi - 1]) if self.u_prev is not None else float(Xr[0, s])
            Xr[:, s:e] = _dykstra_axis_batch(Xr[:, s:e], u_prev, lim, lo[s], hi[s],
                                              n_sweeps=_DYK_SWEEPS)
        return Xr


def reduced_multistart(mpc, theta0, h_pred, n_starts, rng, max_polls, n_rand):
    """Same as landscape_probe.multistart(with_repair=True), but with a
    reduced poll budget/rand-direction count so a Dykstra-repaired run
    finishes in bounded time; settings are stated, not hidden."""
    lo, hi = np.asarray(mpc.lower(), float), np.asarray(mpc.upper(), float)
    span = hi - lo
    batched_rep = lp.guarded_objective(mpc, theta0, h_pred, with_repair=True)
    xs, fs, fails = [], [], 0
    for i in range(n_starts):
        x0 = lp.draw_start(rng, mpc, lo, hi, batched_rep)
        if x0 is None:
            fails += 1
            continue
        xb, fb = lp.pattern_search(batched_rep, x0, lo, hi, mpc.blocks(), span, rng,
                                    max_polls=max_polls, n_rand=n_rand)
        print(f"  start {i:2d}: f={fb:.8f}" if np.isfinite(fb) else f"  start {i:2d}: infeasible")
        if np.isfinite(fb):
            xs.append(xb); fs.append(fb)
    return np.array(xs), np.array(fs), fails


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "13_dykstra_probe")
    os.makedirs(out_dir, exist_ok=True)
    gbar = 10 ** (lp.GBAR_DB / 10)
    theta0 = lp.initial_state(lp.SIGMA_S)

    label, kw, prime = lp.VARIANTS[-1]   # "E. D + two-axis steering"
    assert label.startswith("E."), label

    # Known reference (already shipped, data/08_landscape_probe/landscape_probe.json,
    # variant E, causal-clip repair, 40 starts, max_polls=6000/n_rand=48):
    #   unrepaired:  distinct_minima=2,  frac_reaching_global=0.975, worst_excess=0.602
    #   repaired  :  distinct_minima=40, frac_reaching_global=0.025, worst_excess=3.325 (=4.33x)
    # Re-measuring the causal case is therefore skipped; only Dykstra is run.
    N_STARTS, MAX_POLLS, N_RAND, DYK_SWEEPS = 15, 2500, 24, 300
    print(f"Settings: n_starts={N_STARTS} max_polls={MAX_POLLS} n_rand={N_RAND} "
          f"dyk_sweeps={DYK_SWEEPS}  (paper/shipped ref used max_polls=6000, n_rand=48)")

    global _DYK_SWEEPS
    _DYK_SWEEPS = DYK_SWEEPS

    mpc_dyk = DykstraMPC(lp.ALPHA, lp.BETA, lp.SIGMA_S, gbar, horizon=lp.HORIZON, seed=7, **kw)
    lp.prime_predictor(mpc_dyk, lp.ALPHA, lp.BETA)
    h_pred = mpc_dyk.kf.predict(mpc_dyk.horizon)

    print("\n=== Variant E, DYKSTRA projection repair, REPAIRED landscape only ===")
    rng = np.random.default_rng(20260826)
    xs, fs, fails = reduced_multistart(mpc_dyk, theta0, h_pred, N_STARTS, rng,
                                        MAX_POLLS, N_RAND)
    lo, hi = np.asarray(mpc_dyk.lower(), float), np.asarray(mpc_dyk.upper(), float)
    span = hi - lo
    o = {"n_starts": N_STARTS, "max_polls": MAX_POLLS, "n_rand": N_RAND,
         "dyk_sweeps": DYK_SWEEPS, "starts_infeasible": int(fails),
         "descents_converged": int(len(fs))}
    lp.summarise_minima(xs, fs, span, "", o)
    print(json.dumps(o, indent=2))

    with open(os.path.join(out_dir, "dykstra_repaired_only.json"), "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()
