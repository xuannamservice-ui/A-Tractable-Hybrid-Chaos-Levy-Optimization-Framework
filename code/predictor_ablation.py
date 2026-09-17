"""
Re-run of Table `prediction_ablation` (Predictive MPC vs Reactive Control),
using the REAL Eq. (25) Gauss-Hermite predictor (gh_predictor.GaussHermiteKF)
for Arm A instead of the released KalmanAR1 linear approximation, so the
headline $p<10^{-5}$ claim in the Introduction ("Predictive MPC ... an
edge-executable Kalman state predictor") is backed by the algorithm actually
described in Section IV-B, not a cruder stand-in.

No released script implements this table's exact campaign (same situation as
the other "campaign driver not released" tables); this is the reference
mechanism the manuscript states, run three ways for a fair three-armed
paired comparison:

  Arm A_old : predictive, forecast = released `1 + mu_eff` (no MNLT, no GH)
  Arm A_new : predictive, forecast = real Eq. (25) 3-point Gauss-Hermite
  Arm B     : reactive, forecast = current estimate with NO delay projection

Mechanism (matches the manuscript's description exactly): a scalar Kalman
filter tracks the latent AR(1) state at cycle rate T_u; sub-cycle actuator
lag tau_act is realized by projecting the CONTINUOUS-time latent process
forward by tau_act (correlation rho_eff = rho_a^(tau_act/T_u)) before the
command takes effect. Each arm chooses the SNR-optimal divergence w_z given
its own forecast of h_a at t+tau_act (Arm B uses no projection at all, i.e.
its forecast is the stationary estimate evaluated at zero lookahead); the
TRUE post-EGC system ABER is then scored against the actually-realized
h_a(t+tau_act), using system_metric.aber_of -- the identical evaluator the
manuscript's own construction uses for every other reported ABER number.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon

from mpc_loop import KalmanAR1, manuscript_wz_box
from gh_predictor import build_mnlt, GH_NODES, GH_WEIGHTS
from system_metric import BeamConfig, aber_of, XI_MAX
from channel import xi_floor, beam_geometry
from rtodt_fast import pe_series_f64, z_of
from hclpso_ga import ladder_order

ALPHA, BETA = 1.2, 1.1          # strong turbulence, Table 1
RHO_A = 0.98
GBAR_OP_DB = 38.0
SIGMA_S = 0.10                   # nominal jitter, matches the paper's headline cell
T_U = 1e-3
N_PAIRS = 1000
SEED = 20260826


_WZ_LO, _WZ_HI = manuscript_wz_box(SIGMA_S)


def _pe_branch_of_wz(w_z, gbar_lin):
    """RT-ODT per-branch surrogate (rtodt_fast.pe_series_f64), the identical
    ranking kernel the deployed solver uses -- justified as a ranking proxy
    for the system-level ABER by Proposition 1's order-preservation, not a
    shortcut invented for this script."""
    A0, w_zeq = beam_geometry(np.array([w_z]))
    xi = w_zeq / (2.0 * SIGMA_S)
    z = z_of(ALPHA, BETA, A0, gbar_lin)
    K = ladder_order(z)
    pe = pe_series_f64(ALPHA, BETA, xi, A0, gbar_lin, K)
    return float(np.asarray(pe).reshape(-1)[0])


def best_wz_for_forecast(h_hat: float, gbar_db: float) -> float:
    """The SNR-optimal beam waist given an assumed h_a forecast: minimises
    the RT-ODT per-branch ranking surrogate at gbar_assumed = gbar_op *
    h_hat^2 -- exactly the 1-D unimodal search Section V-A(i) establishes
    suffices per stage, and the identical kernel the deployed solver ranks
    candidates with (fast: ~1 us/call, vs ~50 ms/call for the system-level
    evaluator used only for the final score below). Returns w_z directly,
    avoiding a redundant xi<->w_z round trip through BeamConfig's brentq."""
    gbar_assumed_lin = 10 ** ((gbar_db + 20.0 * np.log10(max(h_hat, 1e-6))) / 10.0)

    def neg_log_pe(w_z):
        return np.log10(max(_pe_branch_of_wz(w_z, gbar_assumed_lin), 1e-300))

    r = minimize_scalar(neg_log_pe, bounds=(_WZ_LO, _WZ_HI), method="bounded",
                        options=dict(xatol=1e-6))
    return float(r.x)


def run(tau_act_us: float, n_pairs: int = N_PAIRS, seed: int = SEED):
    rng = np.random.default_rng(seed)
    T, _T_inv = build_mnlt(ALPHA, BETA)

    kf_ref = KalmanAR1(rho_a=RHO_A)
    P = kf_ref.P
    rho_eff = RHO_A ** (tau_act_us * 1e-6 / T_U)

    var_ghat = max(0.0, 1.0 - P)     # Var(g_hat(t|t)) at steady state
    diff_old, diff_gh = [], []

    for _ in range(n_pairs):
        g_hat = rng.normal(0.0, np.sqrt(var_ghat))
        e = rng.normal(0.0, np.sqrt(P))         # independent estimation error
        g_true_now = g_hat + e
        eps = rng.normal()
        g_true_future = rho_eff * g_true_now + np.sqrt(max(0.0, 1 - rho_eff ** 2)) * eps

        h_true = float(T(g_true_future))

        # --- Arm B: reactive, no delay projection at all ---
        h_hat_B = float(T(g_hat))

        # --- Arm A_old: released linear approximation, projected by rho_eff ---
        mu_eff = rho_eff * g_hat
        h_hat_A_old = 1.0 + mu_eff

        # --- Arm A_new: real Eq. (25) Gauss-Hermite, projected by rho_eff ---
        sigma_eff = np.sqrt(max(0.0, 1.0 - rho_eff ** 2 * (1.0 - P)))
        nodes = mu_eff + sigma_eff * GH_NODES
        h_hat_A_new = float(np.dot(GH_WEIGHTS, T(nodes)))

        wz_B = best_wz_for_forecast(h_hat_B, GBAR_OP_DB)
        wz_A_old = best_wz_for_forecast(max(h_hat_A_old, 1e-3), GBAR_OP_DB)
        wz_A_new = best_wz_for_forecast(h_hat_A_new, GBAR_OP_DB)

        gbar_true_db = GBAR_OP_DB + 20.0 * np.log10(max(h_true, 1e-6))

        def true_pe(w_z):
            return aber_of(BeamConfig("strong", w_z, SIGMA_S), gbar_db=gbar_true_db)

        pe_B = true_pe(wz_B)
        pe_A_old = true_pe(wz_A_old)
        pe_A_new = true_pe(wz_A_new)

        diff_old.append(pe_B - pe_A_old)
        diff_gh.append(pe_B - pe_A_new)

    diff_old = np.array(diff_old)
    diff_gh = np.array(diff_gh)

    def summarize(d):
        mean = d.mean()
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = float("nan")
        return mean, ci, p

    return summarize(diff_old), summarize(diff_gh)


if __name__ == "__main__":
    np.seterr(all="ignore")
    print(f"{'tau_act':>9} | {'Arm A_old (released 1+mu) vs B':>42} | {'Arm A_new (real Eq.25 GH) vs B':>42}")
    print(f"{'(us)':>9} | {'mean diff':>14} {'95% CI':>18} {'p':>8} | {'mean diff':>14} {'95% CI':>18} {'p':>8}")
    for tau_act in (200, 500, 1000):
        (m_old, ci_old, p_old), (m_gh, ci_gh, p_gh) = run(tau_act)
        print(f"{tau_act:9d} | {m_old:14.6e} [{m_old-ci_old:.3e},{m_old+ci_old:.3e}] {p_old:8.2e} | "
              f"{m_gh:14.6e} [{m_gh-ci_gh:.3e},{m_gh+ci_gh:.3e}] {p_gh:8.2e}")
