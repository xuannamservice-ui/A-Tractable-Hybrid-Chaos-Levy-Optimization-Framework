"""
Closed-loop demonstration of the event-triggered stage horizon
(BeamSteeringMPC.rank_stages_trigger_z / rank_stages_deep, added to mpc_loop.py).

Scenario: a large pointing excursion (a wind-gust-scale sway jump) at cycle 0,
recovered over several 1 ms cycles under the hard slew-rate limit -- the
"deep-fade recovery" regime the manuscript's Levy-flight argument targets.
Three controllers see the IDENTICAL disturbance sequence (fixed seed) and are
scored on the REAL post-EGC system ABER (system_metric.aber_of), not the
search's own per-branch ranking cost:

  A. rank_stages=1        (the deployed default: single-stage ranking)
  B. rank_stages=None     (full T=20-stage ranking every cycle: expensive)
  C. rank_stages=1 + event trigger -> rank_stages_deep during the excursion

Each controller uses the SAME solver seed and SAME channel/sway realisation,
so differences are attributable to the horizon mechanism alone.
"""
from __future__ import annotations

import numpy as np

from mpc_loop import BeamSteeringMPC
from channel import GammaGammaAR1
from system_metric import BeamConfig, aber_of

ALPHA, BETA = 1.2, 1.1
SIGMA_S = 0.10
GBAR_DB = 38.0
LINK_LENGTH = 2000.0
U_SLEW = 50e-3 * 1e-3          # U_DOT_MAX * T_U, matches mpc_loop.U_SLEW
U_MAX = 10e-3
N_CYCLES = 18
EXCURSION = np.array([5.0e-4, 4.0e-4])   # rad, a large sway-jump wind gust


def run(label, rank_stages, trigger_z=None, trigger_theta=None, trigger_theta_off=None,
        deep=None, seed=7):
    gbar = 10 ** (GBAR_DB / 10)
    mpc = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, gbar, horizon=20, seed=seed,
                          rank_stages=rank_stages,
                          rank_stages_trigger_z=trigger_z,
                          rank_stages_trigger_theta=trigger_theta,
                          rank_stages_trigger_theta_off=trigger_theta_off,
                          rank_stages_deep=deep)
    # prime the Kalman predictor exactly as landscape_probe.prime_predictor does
    ch = GammaGammaAR1(ALPHA, BETA, rho_a=0.98, seed=20260826, calibrate=False)
    for _ in range(200):
        mpc.kf.update(ch.step() - 1.0)

    theta = EXCURSION.copy()          # the disturbance: start already displaced
    mpc.u_prev = np.zeros(2)

    trace = []
    for c in range(N_CYCLES):
        h_meas = ch.step() - 1.0
        result = mpc.step(theta.copy(), h_meas=h_meas)
        Tr_used = mpc._effective_rank_stages(mpc.kf.predict(mpc.horizon))
        T = mpc.horizon
        if result.best_x is None:
            w_cmd, u_cmd = mpc.wz_lo, mpc.u_prev.copy()
        else:
            w_cmd = float(result.best_x[0])
            # layout is [w_z(T) | theta_az(T) | theta_el(T)]; stage 0 of each
            # steering block is index T (theta_az) and 2T (theta_el).
            u_cmd = np.array([result.best_x[T], result.best_x[2 * T]])
        # publish under the actuator's own slew/range clamp (bench_cycle.py's
        # convention), then advance the TRUE pointing state and u_prev
        u_out = np.clip(u_cmd, mpc.u_prev - U_SLEW, mpc.u_prev + U_SLEW)
        u_out = np.clip(u_out, -U_MAX, U_MAX)
        mpc.u_prev = u_out

        # true pointing recursion, eq. (5): Theta(k+1) = Theta(k) + delta_sway - u
        delta_sway = np.random.default_rng(1000 + c).normal(0, 5.59e-6, 2)
        theta = theta - u_out + delta_sway

        r_d = LINK_LENGTH * float(np.linalg.norm(theta))
        cfg = BeamConfig("strong", max(w_cmd, mpc.wz_lo), SIGMA_S, r_d)
        pe = aber_of(cfg, gbar_db=GBAR_DB)
        trace.append(dict(cycle=c, Tr=int(Tr_used) if Tr_used else 20,
                          w_cmd=w_cmd, r_d=r_d, pe_sys=pe,
                          theta_norm=float(np.linalg.norm(theta))))
    return trace


def summarise(label, trace):
    pes = np.array([t["pe_sys"] for t in trace])
    trs = [t["Tr"] for t in trace]
    print(f"--- {label} ---")
    print("  Tr per cycle       :", trs)
    print("  r_d per cycle (mm) :", [f"{t['r_d']*1e3:.3f}" for t in trace])
    print("  P_e,sys per cycle  :", [f"{p:.3e}" for p in pes])
    print(f"  cumulative P_e,sys : {pes.sum():.6e}   (sum over {len(pes)} cycles)")
    below = np.where(pes <= 1e-6)[0]
    print(f"  first cycle P_e,sys<=1e-6 : {int(below[0]) if below.size else 'never'}")


if __name__ == "__main__":
    np.seterr(all="ignore")

    tA = run("A. rank_stages=1 (deployed)", rank_stages=1)
    tB = run("B. rank_stages=None (full 20-stage)", rank_stages=None)
    tC = run("C. rank_stages=1 + theta-trigger->deep=8, no hysteresis", rank_stages=1,
             trigger_theta=1.5e-4, deep=8)
    tD = run("D. rank_stages=1 + theta-trigger->deep=8, WITH hysteresis", rank_stages=1,
             trigger_theta=1.5e-4, trigger_theta_off=3.0e-5, deep=8)

    summarise("A. rank_stages=1 (deployed)", tA)
    summarise("B. rank_stages=None (full horizon)", tB)
    summarise("C. event-triggered, single threshold (chatters)", tC)
    summarise("D. event-triggered, WITH hysteresis (engage 150urad / release 30urad)", tD)
