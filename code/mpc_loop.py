"""
Closed-loop predictive beam-steering controller.

Assembles the architecture of Sections IV and VI:

    sense -> predict -> optimise -> safety-check -> actuate

  predictor      AR(1) + steady-state Kalman filter on the zero-mean latent
                 scintillation state, giving h_a(t+k) over the horizon.  The
                 TCN branch of the manuscript is represented by its fusion
                 rule (eq. 30); on an AR(1) channel the Kalman filter is
                 MMSE-optimal and the inverse-variance weight collapses onto
                 it (omega -> 1), which is what the campaign observes.

  optimiser      H-CLPSO-GA over the receding-horizon trajectory, stopped at
                 the anytime checkpoint tau_O.

  safety filter  the three-part envelope guard of Section VI-C:
                     (i)  z(u) <= z_max          admissibility
                     (ii) 0 <= Pe(u) <= 1/2      range
                     (iii) Pe(u) < eps_safe      threshold
                 A command failing any test is replaced by the anytime
                 incumbent, or by the offline xi_safe override.

This is a reference implementation written from the specification in the paper.
It is not the campaign driver that produced the tabulated results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channel import beam_geometry, xi_effective
from hclpso_ga import HCLPSOGA, SolverConfig, ladder_order
from rtodt_fast import pe_series_f64, z_of

Z_MAX = 8.0
EPS_SAFE = 1e-3
T_U = 1e-3
TAU_O = 600e-6


# --------------------------------------------------------------- predictor
class KalmanAR1:
    """Steady-state scalar Kalman filter on the AR(1) latent state (Sec. IV-B)."""

    def __init__(self, rho_a=0.98, q=None, r=1e-3):
        self.rho = rho_a
        self.q = (1 - rho_a ** 2) if q is None else q
        self.r = r
        p = self.q
        for _ in range(500):                       # iterate the Riccati recursion
            p_pred = self.rho ** 2 * p + self.q
            p = p_pred * self.r / (p_pred + self.r)
        self.P = p
        self.K = (self.rho ** 2 * p + self.q) / (self.rho ** 2 * p + self.q + self.r)
        self.x = 0.0

    def update(self, z_meas: float) -> float:
        x_pred = self.rho * self.x
        self.x = x_pred + self.K * (z_meas - x_pred)
        return self.x

    def predict(self, horizon: int) -> np.ndarray:
        return np.array([self.x * self.rho ** k for k in range(1, horizon + 1)])

    def innovation_variance(self) -> float:
        return self.rho ** 2 * self.P + self.q + self.r


def fuse(kf_pred, tcn_pred, var_kf, var_tcn):
    """Inverse-variance fusion, eq. (30). On a Markovian channel var_kf is the
    smaller, so omega -> 1 and the rule returns the Kalman branch."""
    omega = var_tcn / (var_kf + var_tcn)
    return omega * kf_pred + (1.0 - omega) * tcn_pred, omega


# ------------------------------------------------------------ safety filter
@dataclass
class GuardReport:
    admissible: np.ndarray
    n_z: int
    n_range: int
    n_threshold: int


def envelope_guard(z, pe, z_max=Z_MAX, three_part=True):
    """Section VI-C, restricted to the two tests that are per-branch quantities.

    Test (i)  z <= z_max          admissibility of the surrogate
    Test (ii) 0 <= Pe <= 1/2      the evaluation must be a probability

    Test (iii), Pe < eps_safe, is a POST-EGC system-level threshold: eps_safe =
    1e-3 while the per-branch surrogate the swarm ranks by is of order 1e-1 at
    the operating SNR, so the two are not comparable and applying (iii) inside
    the swarm loop would reject every candidate.  It is therefore applied once,
    to the selected command, against the combined ABER (see `egc_system.py`).

    With three_part=False only test (ii) is retained, which is the
    threshold-only form the reported campaign was executed under.
    """
    finite = np.isfinite(pe)
    t_range = (pe >= 0.0) & (pe <= 0.5)
    if not three_part:
        ok = finite & t_range
        return GuardReport(ok, 0, int(np.sum(finite & ~t_range)), 0)
    t_z = z <= z_max
    ok = finite & t_z & t_range
    return GuardReport(ok, int(np.sum(~t_z)), int(np.sum(finite & ~t_range)), 0)


# --------------------------------------------------------------- controller
class BeamSteeringMPC:
    """One receding-horizon controller instance."""

    def __init__(self, alpha, beta, sigma_s, gbar, horizon=20, seed=0,
                 three_part_guard=True, config: SolverConfig = None,
                 slew_limit=0.05, lambda_u=2.0):
        self.alpha, self.beta = alpha, beta
        self.sigma_s, self.gbar = sigma_s, gbar
        self.horizon = horizon
        self.three_part = three_part_guard
        self.kf = KalmanAR1()
        self.cfg = config or SolverConfig()
        self.rng = np.random.default_rng(seed)
        self.wz_lo, self.wz_hi = 0.055, 3.0
        self.slew_limit = slew_limit          # |w_z(k) - w_z(k-1)| per cycle
        self.lambda_u = lambda_u              # slew penalty weight, eq. (13)
        self.guard_stats = dict(z=0, range=0, threshold=0)

    # -- objective ----------------------------------------------------
    def _objective(self, X, r_d, h_pred):
        """Receding-horizon cost over the divergence trajectory.

        The per-stage ABER is unimodal in w_z; what makes the landscape rugged
        is the slew-rate coupling between consecutive stages (Section V-A),
        which is why the decision variable is the whole trajectory rather than
        a single divergence.
        """
        n, T = X.shape
        flat = X.reshape(-1)
        A0f, weqf = beam_geometry(flat)
        xif = weqf / (2.0 * self.sigma_s)
        xi_eff = xi_effective(xif, r_d, self.sigma_s)
        zf = z_of(self.alpha, self.beta, A0f, self.gbar)
        Kf = (ladder_order(zf) if self.cfg.use_fidelity_ladder
              else np.full(zf.shape, self.cfg.fixed_order))
        pef = pe_series_f64(self.alpha, self.beta, xi_eff, A0f, self.gbar, Kf)

        pe = pef.reshape(n, T)
        z = zf.reshape(n, T)

        # stage cost weighted by the predicted scintillation state
        w = np.asarray(h_pred[:T], dtype=float)
        w = 1.0 + 0.5 * (w - w.mean()) / (w.std() + 1e-12)
        cost = np.nansum(pe * w[None, :], axis=1) / T

        # slew penalty, and a hard slew-rate violation, eqs. (13)-(14)
        d = np.abs(np.diff(X, axis=1))
        cost = cost + self.lambda_u * np.sum(d ** 2, axis=1) / max(T - 1, 1)
        cost = np.where(np.any(d > self.slew_limit, axis=1), np.inf, cost)

        return cost, dict(z=z[:, 0], pe_first=pe[:, 0])

    def step(self, r_d: float):
        """Run one control cycle over the horizon; returns the solver result."""
        h_pred = self.kf.predict(self.horizon)
        solver = HCLPSOGA([self.wz_lo] * self.horizon,
                          [self.wz_hi] * self.horizon, self.cfg,
                          seed=int(self.rng.integers(1 << 31)))

        def obj(X):
            return self._objective(X, r_d, h_pred)

        def guard(X, f, aux):
            # the guard vets the PUBLISHED command, i.e. the first stage
            rep = envelope_guard(aux["z"], aux["pe_first"], three_part=self.three_part)
            self.guard_stats["z"] += rep.n_z
            self.guard_stats["range"] += rep.n_range
            self.guard_stats["threshold"] += rep.n_threshold
            return rep.admissible

        res = solver.minimise(obj, guard=guard)
        return res
