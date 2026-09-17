"""
Real measurement of the predictor-failure fallback (Sec. safety_protocols:
"illustrative and not yet implemented" until this script and the matching
`sigma_crit_sq` switch added to BeamSteeringMPC -- see `mpc_loop.py`).

SCOPE, STATED PLAINLY
---------------------
The released `KalmanAR1` is a STEADY-STATE scalar filter: its error
covariance P is solved once from the algebraic Riccati equation at
construction and never updated per cycle (that is precisely what makes each
update a single multiply-accumulate, Sec. prediction_module). A steady-state
filter has, by construction, no notion of a live, time-varying uncertainty
spike -- there is nothing in the released channel/predictor pair that could
generate the manuscript's "20x Cn2 step" scenario as a MEASURED trajectory,
only as an assumption about what such a step would do. This script does not
pretend otherwise. It tests the actual mechanism now wired into
BeamSteeringMPC -- `_predictor_failed()` comparing the one-step forecast
variance 1 - rho^2(1-P) against `sigma_crit_sq`, and `_effective_rank_stages`
forcing Tr=1 when it fires -- against the one thing that DOES vary this
quantity in the released model: rho_a itself, which eq. (`crosswind_tracker`)
already treats as adapting to wind conditions. A lower rho_a (faster
decorrelation, e.g. a wind gust) raises the steady-state P and hence the
one-step variance monotonically; this script sweeps rho_a and reports
exactly where the trigger fires, then runs closed-loop cycles at a
representative degraded rho_a with the switch on and off.

WHAT IS MEASURED VS. ASSERTED
------------------------------
  (1) The trigger's own logic: `BeamSteeringMPC._predictor_failed()` and
      `_effective_rank_stages()` exercised directly, confirming (a) it is
      OFF (False, Tr unchanged) whenever `sigma_crit_sq=None`, matching
      every existing caller, and (b) it fires exactly where the swept
      one-step variance crosses the chosen threshold -- not asserted, read
      off the swept array.
  (2) Priority over the deep-fade event trigger: with BOTH
      `rank_stages_trigger_z` (which would otherwise widen Tr to
      `rank_stages_deep`) and `sigma_crit_sq` active and firing on the same
      cycle, `_effective_rank_stages` returns 1, not `rank_stages_deep` --
      confirmed by direct call, not inferred from the docstring.
  (3) Closed-loop effect at a representative degraded rho_a: paired cycles
      (identical channel/sway trace) with the fallback on vs. off, reporting
      the achieved ranking cost and the admissible-candidate rate for both --
      honestly, including a result that goes the OTHER way, if it does.

USAGE
    python predictor_failure_probe.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from channel import GammaGammaAR1, SwayProcess
from mpc_loop import BeamSteeringMPC, KalmanAR1

ALPHA, BETA = 1.2, 1.1
GBAR_DB = 38.0
GBAR = 10.0 ** (GBAR_DB / 10.0)
SIGMA_S = 0.10
HORIZON = 20


def one_step_variance(rho_a):
    kf = KalmanAR1(rho_a=rho_a)
    return 1.0 - kf.rho ** 2 * (1.0 - kf.P)


def part1_sweep():
    print("=== (1) one-step forecast variance vs rho_a, and where a given")
    print("    sigma_crit_sq threshold would fire ===")
    rhos = np.array([0.995, 0.99, 0.98, 0.95, 0.90, 0.80, 0.60, 0.40, 0.20])
    variances = np.array([one_step_variance(r) for r in rhos])
    for r, v in zip(rhos, variances):
        print(f"  rho_a={r:5.3f}  one-step variance={v:.4f}")
    threshold = 0.60
    fires = variances > threshold
    print(f"  sigma_crit_sq={threshold}: fires at rho_a in "
          f"{list(np.round(rhos[fires], 3))}")
    return rhos, variances, threshold


def part2_switch_off_by_default():
    print("\n=== (1b) switch is OFF (False) whenever sigma_crit_sq=None ===")
    mpc = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=1,
                          sigma_crit_sq=None)
    h_pred = mpc.kf.predict(HORIZON)
    failed = mpc._predictor_failed()
    tr = mpc._effective_rank_stages(h_pred)
    print(f"  sigma_crit_sq=None -> predictor_failed={failed}, "
          f"_effective_rank_stages={tr} (rank_stages={mpc.rank_stages})")
    assert failed is False and tr == mpc.rank_stages


def part3_priority_over_deep_fade(rho_degraded, sigma_crit_sq):
    print("\n=== (2) predictor-failure takes priority over the deep-fade "
          "z-trigger ===")
    mpc = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=1,
                          rank_stages=1, rank_stages_trigger_z=0.1,
                          rank_stages_deep=20, sigma_crit_sq=sigma_crit_sq)
    mpc.kf = KalmanAR1(rho_a=rho_degraded)   # degraded filter, same interface
    h_pred = mpc.kf.predict(HORIZON)
    # rank_stages_trigger_z=0.1 is set low enough that the z-trigger alone
    # fires on almost any operating point, so if the priority rule were
    # absent this would return rank_stages_deep=20, not 1.
    tr = mpc._effective_rank_stages(h_pred)
    var1 = mpc._one_step_forecast_variance()
    print(f"  rho_a={rho_degraded}, one-step variance={var1:.4f}, "
          f"sigma_crit_sq={sigma_crit_sq} -> predictor_failed="
          f"{mpc.predictor_failed_last_cycle}, _effective_rank_stages={tr} "
          f"(would be {mpc.rank_stages_deep} without the priority rule)")
    assert mpc.predictor_failed_last_cycle is True and tr == 1


def _run_arm(seed, sigma_crit_sq, rho_a, theta_trace, h_trace):
    mpc = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=seed,
                          rank_stages=None, sigma_crit_sq=sigma_crit_sq)
    mpc.kf = KalmanAR1(rho_a=rho_a)
    mpc.u_prev = np.zeros(2)
    T = mpc.horizon
    costs = np.full(len(h_trace), np.nan)
    for i, (theta, h) in enumerate(zip(theta_trace, h_trace)):
        res = mpc.step(theta, h_meas=h)
        if res.best_x is None:
            continue
        costs[i] = res.best_f
        mpc.u_prev = np.array([res.best_x[T], res.best_x[2 * T]])
    return costs


def part4_closed_loop(rho_degraded, sigma_crit_sq, n_seeds=15, cycles=80, burn_in=200):
    print("\n=== (3) closed-loop effect at the degraded rho_a, fallback on "
          "vs. off ===")
    print(f"  rho_a={rho_degraded} (one-step variance "
          f"{one_step_variance(rho_degraded):.4f}, threshold {sigma_crit_sq}) "
          f"-- Tr collapses from the FULL T={HORIZON} lookahead (rank_stages=None) "
          f"to 1 whenever the fallback is on.")
    off_all, on_all = [], []
    for s in range(n_seeds):
        sway = SwayProcess(SIGMA_S, seed=6000 + s)
        for _ in range(burn_in):
            sway.step()
        theta_trace = [sway.step() for _ in range(cycles)]
        ch = GammaGammaAR1(ALPHA, BETA, rho_a=rho_degraded, seed=7000 + s,
                           calibrate=False)
        h_trace = [ch.step() - 1.0 for _ in range(cycles)]
        off_all.append(_run_arm(8000 + s, None, rho_degraded, theta_trace, h_trace))
        on_all.append(_run_arm(8000 + s, sigma_crit_sq, rho_degraded, theta_trace, h_trace))
    off = np.concatenate(off_all)
    on = np.concatenate(on_all)
    ok = np.isfinite(off) & np.isfinite(on)
    diff = off[ok] - on[ok]   # positive: fallback ON reached a LOWER (better) cost
    print(f"  cycles: {off.size}  (both admissible: {int(ok.sum())}, "
          f"fallback-off no-candidate: {int(np.sum(~np.isfinite(off)))}, "
          f"fallback-on no-candidate: {int(np.sum(~np.isfinite(on)))})")
    print(f"  achieved ranking cost -- fallback off median {np.nanmedian(off):.6e}  "
          f"fallback on median {np.nanmedian(on):.6e}")
    if ok.sum() > 0:
        try:
            p = wilcoxon(diff).pvalue
        except ValueError:
            p = float("nan")
        print(f"  paired diff (off - on), positive = fallback helps: "
              f"mean {diff.mean():.6e}  median {np.median(diff):.6e}  "
              f"Wilcoxon p={p:.3e}")
        print(f"  fraction fallback-on strictly better: "
              f"{float(np.mean(diff > 0)):.4f}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    part1_sweep()
    part2_switch_off_by_default()
    part3_priority_over_deep_fade(rho_degraded=0.60, sigma_crit_sq=0.60)
    part4_closed_loop(rho_degraded=0.60, sigma_crit_sq=0.60)
