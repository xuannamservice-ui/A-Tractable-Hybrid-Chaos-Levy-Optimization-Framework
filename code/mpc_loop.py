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

DECISION VECTOR
    With `steering=True` (the default) the decision vector is the manuscript's
    60-dimensional trajectory of Sec. IV-C,

        u_{t:t+T-1} = { [ xi_{t+k} ; u_ptr,{t+k} ] }_{k=0}^{T-1},   T = 20,

    laid out flat as [ w_z(T) | theta_az(T) | theta_el(T) ].  The divergence is
    carried by its physical generator w_z rather than by xi itself, because
    A_0 and w_{z,eq} are functions of w_z (eq. 3) and the RT-ODT kernel must
    stay interpolation-free; w_z ranges over the exact image of the
    manuscript's xi box on the beam-broadening branch, so the two
    parametrisations describe the same feasible set.

    With `steering=False` the vector collapses to the divergence trajectory
    alone (20-D) and the pointing state is frozen at its measured value.  That
    is the earlier form of this file, retained so that `landscape_probe.py` can
    measure both objectives under identical instrumentation and attribute the
    difference.

This is a reference implementation written from the specification in the paper.
It is not the campaign driver that produced the tabulated results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channel import beam_geometry, branch_min_wz, xi_effective
from hclpso_ga import HCLPSOGA, SolverConfig, ladder_order
from rtodt_fast import pe_series_f64, z_of

Z_MAX = 8.0
EPS_SAFE = 1e-3
T_U = 1e-3
TAU_O = 600e-6

# Actuator specification, Table `tab:actuator_specs`.
U_MAX = 10e-3                 # +/- 10 mrad max deflection, in radians
U_DOT_MAX = 50e-3             # 50 mrad/s peak slew rate, in rad/s
U_SLEW = U_DOT_MAX * T_U      # 0.05 mrad per 1 ms cycle, in radians
TAU_ACT = 200e-6              # FSM dead time
LINK_LENGTH = 2000.0          # L = 2 km

# Manuscript decision box for the divergence variable (Sec. VII-A):
#   xi in [ max(0.5, xi_min(sigma_s)), xi_max ],  xi_min = 0.0877/(2 sigma_s),
#   xi_max = 4.888.
XI_MAX = 4.888
XI_FLOOR_NUM = 0.0877         # min of w_{z,eq} over w_z, at a = 0.05 m
XI_HARD_FLOOR = 0.5


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


# ------------------------------------------------------------- geometry aid
def _wzeq(w_z):
    return beam_geometry(np.atleast_1d(np.asarray(w_z, float)))[1]


def wz_for_xi(xi: float, sigma_s: float, lo=1e-3, hi=60.0, iters=200) -> float:
    """Invert xi = w_{z,eq}/(2 sigma_s) on the beam-broadening branch.

    w_{z,eq} is non-monotonic in w_z with an interior minimum near w_z = 0.055 m;
    the upper (increasing) branch is the physically meaningful one, on which A_0
    decreases as the beam is widened.  Bisection on that branch.
    """
    target = 2.0 * sigma_s * xi
    # The interior minimum of w_zeq -- the boundary of the beam-broadening
    # branch -- now comes from channel.branch_min_wz rather than from a private
    # golden section here. That search evaluated eq. (3) on BOTH sides of the
    # boundary, which the guarded beam_geometry now flags as NaN; a NaN
    # comparison is silently False and would have walked the search onto the
    # wrong branch. One routine locates the boundary, and it is entitled to.
    w_argmin, weq_argmin = branch_min_wz()
    if target <= weq_argmin:
        return float(w_argmin)          # xi below the attainable floor
    lo, hi = w_argmin, hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _wzeq(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def manuscript_wz_box(sigma_s: float):
    """The image of the manuscript's xi box on the beam-broadening branch."""
    xi_lo = max(XI_HARD_FLOOR, XI_FLOOR_NUM / (2.0 * sigma_s))
    return wz_for_xi(xi_lo, sigma_s), wz_for_xi(XI_MAX, sigma_s)


# --------------------------------------------------------------- controller
class BeamSteeringMPC:
    """One receding-horizon controller instance."""

    def __init__(self, alpha, beta, sigma_s, gbar, horizon=20, seed=0,
                 three_part_guard=True, config: SolverConfig = None,
                 slew_limit=0.05, lambda_u=2.0,
                 steering=True, manuscript_box=True, strict_admissibility=True,
                 h_in_aber=True, link_length=LINK_LENGTH, rank_stages=None):
        self.alpha, self.beta = alpha, beta
        self.sigma_s, self.gbar = sigma_s, gbar
        self.horizon = horizon
        self.three_part = three_part_guard
        self.kf = KalmanAR1()
        self.cfg = config or SolverConfig()
        self.rng = np.random.default_rng(seed)
        self.L = link_length
        # How many horizon stages the RANKING surrogate evaluates per candidate.
        #   None -> all T stages, the literal receding-horizon sum of eq. (12).
        #   1    -> the published stage only, which is the cost model the
        #           manuscript states: "the per-candidate ranking inside the
        #           solver uses the per-branch surrogate ... and is what the
        #           O(T_iter N_p K) cost refers to" (Sec. VI-C). That expression
        #           carries no factor T, so evaluating all T stages does 20x the
        #           work the paper specifies.
        # This is not only a speed switch. With rank_stages=1 the trajectory
        # beyond the published stage is shaped by the slew penalty and the
        # feasibility test alone, not by its own ABER, so the two settings
        # optimise different objectives and their optima need not coincide.
        # Both are measured in the release rather than one being assumed.
        self.rank_stages = rank_stages

        # --- faithfulness switches ------------------------------------
        # Each isolates one manuscript feature so `landscape_probe.py` can
        # attribute a landscape change to a specific specification item.
        self.steering = bool(steering)
        self.manuscript_box = bool(manuscript_box)
        self.strict_admissibility = bool(strict_admissibility)
        self.h_in_aber = bool(h_in_aber)

        if self.manuscript_box:
            self.wz_lo, self.wz_hi = manuscript_wz_box(sigma_s)
        else:
            self.wz_lo, self.wz_hi = 0.055, 3.0

        self.slew_limit = slew_limit          # |w_z(k) - w_z(k-1)| per cycle
        self.u_slew = U_SLEW                  # |u(k) - u(k-1)| per cycle, rad
        self.u_max = U_MAX
        self.lambda_u = lambda_u              # control penalty weight, eq. (13)
        # tau_act = 200 us = 0.2 T_u.  The manuscript gives no rule for a
        # sub-sample dead time in a 1 ms discrete prediction model; 0.2 samples
        # rounds to zero, so the horizon dynamics carry no integer delay.  The
        # choice is recorded here rather than buried.
        self.act_delay_samples = int(round(TAU_ACT / T_U))
        self.guard_stats = dict(z=0, range=0, threshold=0)
        self.theta0 = np.zeros(2)

    # -- decision-vector layout ---------------------------------------
    @property
    def decision_dim(self) -> int:
        return 3 * self.horizon if self.steering else self.horizon

    def blocks(self):
        """(start, end) index pairs of each physical variable's stage block."""
        T = self.horizon
        return [(0, T), (T, 2 * T), (2 * T, 3 * T)] if self.steering else [(0, T)]

    def block_names(self):
        return ["w_z", "theta_az", "theta_el"] if self.steering else ["w_z"]

    def lower(self):
        T = self.horizon
        if not self.steering:
            return np.full(T, self.wz_lo)
        return np.concatenate([np.full(T, self.wz_lo), np.full(2 * T, -self.u_max)])

    def upper(self):
        T = self.horizon
        if not self.steering:
            return np.full(T, self.wz_hi)
        return np.concatenate([np.full(T, self.wz_hi), np.full(2 * T, self.u_max)])

    def centre(self):
        return 0.5 * (self.lower() + self.upper())

    def block_slew(self):
        """Per-block hard slew limit, in each block's own units."""
        return ([self.slew_limit, self.u_slew, self.u_slew] if self.steering
                else [self.slew_limit])

    # -- feasibility repair --------------------------------------------
    def repair(self, X):
        """Project a candidate onto the slew-rate tube, eq. (14).

        The slew bound |u_{k} - u_{k-1}| <= u_dot_max T_u couples consecutive
        stages, so the feasible set is a polytope, not a box, and clipping to
        the box leaves almost every candidate infeasible.  The manuscript
        prescribes reflection back into the feasible set but states no
        projection order or sweep direction; a forward sweep -- the causal
        rate-limiter, each stage pulled to within one slew step of the stage
        before it -- is the standard realisation and is applied here.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X = np.clip(X, self.lower(), self.upper()).copy()
        # after the box clip every stage is inside the box, and pulling a stage
        # toward its in-box predecessor keeps it there, so the sweep preserves
        # both constraints simultaneously.
        for (s, e), lim in zip(self.blocks(), self.block_slew()):
            for k in range(s + 1, e):
                X[:, k] = np.clip(X[:, k], X[:, k - 1] - lim, X[:, k - 1] + lim)
        return X

    # -- state handling ------------------------------------------------
    @staticmethod
    def _as_theta(state, L=LINK_LENGTH):
        """Accept either the pointing 2-vector Theta (rad) or a scalar radial
        offset r_d (m); the latter is placed on the azimuth axis."""
        s = np.atleast_1d(np.asarray(state, dtype=float))
        return s.astype(float) if s.size == 2 else np.array([float(s[0]) / L, 0.0])

    def _stage_rd(self, X):
        """Radial offset r_d at every horizon stage, from the pointing recursion
        Theta(t+k+1) = Theta(t+k) + Delta_sway - u_ptr(t+k), eq. (5).

        The MPC equality constraint is written x_{t+k+1} = f(x_{t+k}, u_{t+k}, 0),
        i.e. the disturbance argument is zeroed over the horizon, so Delta_sway
        is taken as zero inside the prediction and the recursion is a running
        subtraction of the steering commands from the measured Theta(t).
        """
        n, T = X.shape[0], self.horizon
        if not self.steering:
            return np.full((n, T), self.L * np.linalg.norm(self.theta0))
        U = np.stack([X[:, T:2 * T], X[:, 2 * T:3 * T]], axis=2)   # (n, T, 2)
        if self.act_delay_samples:
            U = np.concatenate([np.zeros((n, self.act_delay_samples, 2)),
                                U[:, :-self.act_delay_samples]], axis=1)
        theta = self.theta0[None, None, :] - np.cumsum(U, axis=1)
        theta = np.concatenate([np.tile(self.theta0, (n, 1, 1)), theta[:, :-1]], axis=1)
        return self.L * np.linalg.norm(theta, axis=2)              # (n, T)

    def _stage_gbar(self, h_pred):
        """Per-stage reference SNR.

        The manuscript's stage cost is Pe(h_hat_{t+k}, xi_{t+k}, Theta_{t+k}) and
        eq. (`eq:snr_def`) gives gamma = gbar h^2, so the predicted scintillation
        state enters as a per-stage rescaling of the reference SNR.  The Kalman
        state is a zero-mean latent and the turbulence factor is normalised to
        unit mean (the manuscript states sigma_crit^2 is expressed relative to
        the unit-mean turbulence factor), hence h_hat = 1 + x_hat.
        """
        h = 1.0 + np.asarray(h_pred[:self.horizon], dtype=float)
        h = np.clip(h, 1e-3, None)
        return self.gbar * h ** 2

    # -- objective ----------------------------------------------------
    def _objective(self, X, state, h_pred):
        """Receding-horizon cost, eq. (`eq:mpc_problem`):

            min sum_k [ Pe(h_hat_{t+k}, xi_{t+k}, Theta_{t+k})
                        + lambda_u ||Delta u_{t+k}||^2 ]

        The per-stage ABER is unimodal in w_z; the manuscript locates the
        multimodality in the trajectory space, where the slew-rate constraint
        chains stages together and xi_eff depends nonlinearly on the two-axis
        steering error through eq. (7).  Both couplings are live only when
        `steering=True`.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, T = X.shape[0], self.horizon
        self.theta0 = self._as_theta(state, self.L)

        W = X[:, :T]                                   # divergence trajectory
        r_d = self._stage_rd(X)                        # (n, T)

        # Restrict the RANKING surrogate to the first `rank_stages` stages before
        # any geometry is computed, not after: beam_geometry, xi_effective and
        # the kernel then all see n*Tr elements instead of n*T. Slicing after the
        # geometry would leave the dominant cost untouched.
        Tr = T if self.rank_stages is None else min(int(self.rank_stages), T)
        Wr = W[:, :Tr]
        rdr = r_d[:, :Tr]

        A0f, weqf = beam_geometry(Wr.reshape(-1))
        xif = weqf / (2.0 * self.sigma_s)
        xi_eff = xi_effective(xif, rdr.reshape(-1), self.sigma_s)

        if self.h_in_aber:
            gb = np.tile(self._stage_gbar(h_pred)[:Tr], (n, 1)).reshape(-1)
        else:
            gb = np.full(n * Tr, self.gbar)

        # Restrict the ranking to the first `rank_stages` stages when asked. The
        # slew penalty and the feasibility test below still see the whole
        # trajectory -- they are numpy diffs and cost nothing -- so only the
        # ABER evaluations are reduced.
        # the kernel takes a scalar gbar, so evaluate one stage-group at a time
        pef = np.empty(n * Tr)
        zf = np.empty(n * Tr)
        for g in np.unique(gb):
            m = gb == g
            zf[m] = z_of(self.alpha, self.beta, A0f[m], float(g))
            Km = (ladder_order(zf[m]) if self.cfg.use_fidelity_ladder
                  else np.where(zf[m] <= Z_MAX, self.cfg.fixed_order, -1))
            pef[m] = pe_series_f64(self.alpha, self.beta, xi_eff[m], A0f[m],
                                   float(g), Km)

        pe = pef.reshape(n, Tr)
        z = zf.reshape(n, Tr)

        if self.strict_admissibility:
            # An inadmissible stage makes the whole trajectory inadmissible.
            # Summing with np.nansum instead would score a stage that the
            # fidelity ladder rejected as ZERO cost, i.e. cheaper than any
            # admissible stage, and the optimum would be the trajectory that
            # maximises the number of rejected stages.
            cost = np.sum(pe, axis=1) / Tr
        else:
            cost = np.nansum(pe, axis=1) / Tr

        # control penalty and the hard slew-rate constraint, eqs. (13)-(14)
        pen = np.zeros(n)
        viol = np.zeros(n, dtype=bool)
        for (s, e), lim in zip(self.blocks(), self.block_slew()):
            d = np.abs(np.diff(X[:, s:e], axis=1))
            pen = pen + np.sum(d ** 2, axis=1) / max(T - 1, 1)
            viol |= np.any(d > lim, axis=1)
        cost = cost + self.lambda_u * pen
        cost = np.where(viol, np.inf, cost)

        return cost, dict(z=z[:, 0], pe_first=pe[:, 0])

    # -- diagnostics ---------------------------------------------------
    # Read-only measurements used by `landscape_probe.py`.  They report on the
    # objective; they do not alter it.
    def per_stage_kernel(self, w_grid, state, h_pred, stage=0, r_d=None):
        """Pe of a single stage as a function of w_z, at that stage's own
        pointing offset and predicted SNR.  `r_d` overrides the offset."""
        self.theta0 = self._as_theta(state, self.L)
        w = np.asarray(w_grid, float)
        A0, weq = beam_geometry(w)
        if r_d is None:
            r_d = self.L * float(np.linalg.norm(self.theta0))
        xi_eff = xi_effective(weq / (2.0 * self.sigma_s), r_d, self.sigma_s)
        g = float(self._stage_gbar(h_pred)[stage]) if self.h_in_aber else self.gbar
        z = z_of(self.alpha, self.beta, A0, g)
        K = (ladder_order(z) if self.cfg.use_fidelity_ladder
             else np.where(z <= Z_MAX, self.cfg.fixed_order, -1))
        return pe_series_f64(self.alpha, self.beta, xi_eff, A0, g, K), z

    def admissible_fraction(self, w_grid, state, h_pred=None):
        h_pred = np.zeros(self.horizon) if h_pred is None else h_pred
        pe, z = self.per_stage_kernel(w_grid, state, h_pred)
        ok = np.isfinite(pe) & (z <= Z_MAX)
        w = np.asarray(w_grid, float)
        edge = float(w[ok].max()) if ok.any() else None
        return dict(fraction=float(ok.mean()), upper_edge=edge,
                    box=[float(w.min()), float(w.max())])

    def per_stage_argmins(self, w_grid, state, h_pred):
        """argmin over w_z of each stage's own cost -- if these coincide, every
        stage wants the same divergence, the chain coupling is inactive at the
        optimum, and the trajectory problem is a 1-D problem in disguise."""
        out = []
        w = np.asarray(w_grid, float)
        for k in range(self.horizon):
            pe, z = self.per_stage_kernel(w, state, h_pred, stage=k)
            v = np.where(np.isfinite(pe) & (z <= Z_MAX), pe, np.inf)
            out.append(w[int(np.argmin(v))] if np.isfinite(v).any() else np.nan)
        return np.array(out)

    def degeneracy_audit(self, state, h_pred, n_level=400, n_slope=201):
        """Does the objective reward pushing stages OUT of the admissible band?

        Scans the (level, slope) family of trajectories -- the family the hard
        slew limit actually permits -- and compares the best trajectory whose
        stages are ALL admissible against the best trajectory that merely passes
        the guard (which inspects the published first stage only).
        """
        T = self.horizon
        lo, hi = self.wz_lo, self.wz_hi
        lim = self.slew_limit
        lv = np.linspace(lo, hi, n_level)
        sl = np.linspace(-lim * 0.999, lim * 0.999, n_slope)
        L, S = np.meshgrid(lv, sl, indexing="ij")
        W = np.clip(L.ravel()[:, None] + S.ravel()[:, None] * np.arange(T)[None, :], lo, hi)
        if self.steering:
            X = np.hstack([W, np.zeros((W.shape[0], 2 * T))])
        else:
            X = W
        c, aux = self._objective(X, state, h_pred)
        rep = envelope_guard(aux["z"], aux["pe_first"], three_part=self.three_part)
        cg = np.where(np.isfinite(c) & rep.admissible, c, np.inf)

        # stage-level admissibility of every scanned trajectory
        self.theta0 = self._as_theta(state, self.L)
        r_d = self._stage_rd(X)
        A0f, weqf = beam_geometry(W.reshape(-1))
        xe = xi_effective(weqf / (2.0 * self.sigma_s), r_d.reshape(-1), self.sigma_s)
        gb = (np.tile(self._stage_gbar(h_pred), (W.shape[0], 1)).reshape(-1)
              if self.h_in_aber else np.full(W.size, self.gbar))
        pef = np.empty(W.size)
        for g in np.unique(gb):
            m = gb == g
            zz = z_of(self.alpha, self.beta, A0f[m], float(g))
            Km = (ladder_order(zz) if self.cfg.use_fidelity_ladder
                  else np.where(zz <= Z_MAX, self.cfg.fixed_order, -1))
            pef[m] = pe_series_f64(self.alpha, self.beta, xe[m], A0f[m], float(g), Km)
        nan_stages = np.sum(~np.isfinite(pef.reshape(W.shape[0], T)), axis=1)

        all_adm = np.where(nan_stages == 0, cg, np.inf)
        i_all = int(np.argmin(all_adm))
        i_any = int(np.argmin(cg))
        best_adm = float(all_adm[i_all])
        best_any = float(cg[i_any])
        gain = (best_adm / best_any) if np.isfinite(best_any) and best_any > 0 else float("nan")
        return dict(best_admissible=best_adm, best_overall=best_any,
                    nan_stages_at_best=int(nan_stages[i_any]),
                    exploit_gain=float(gain),
                    scanned=int(W.shape[0]))

    # -- one control cycle ---------------------------------------------
    def step(self, state, h_meas: float = None):
        """Run one control cycle over the horizon; returns the solver result.

        `state` is the measured pointing state: the 2-vector Theta(t) in rad, or
        a scalar radial offset r_d(t) in m.  `h_meas` is the measured latent
        scintillation state; supplying it drives the Kalman predictor, without
        which the horizon forecast is identically zero and every stage of the
        objective sees the same reference SNR.
        """
        if h_meas is not None:
            self.kf.update(float(h_meas))
        h_pred = self.kf.predict(self.horizon)
        self.theta0 = self._as_theta(state, self.L)
        solver = HCLPSOGA(self.lower(), self.upper(), self.cfg,
                          seed=int(self.rng.integers(1 << 31)),
                          blocks=self.blocks(), repair=self.repair)

        def obj(X):
            return self._objective(X, state, h_pred)

        def guard(X, f, aux):
            # the guard vets the PUBLISHED command, i.e. the first stage
            rep = envelope_guard(aux["z"], aux["pe_first"], three_part=self.three_part)
            self.guard_stats["z"] += rep.n_z
            self.guard_stats["range"] += rep.n_range
            self.guard_stats["threshold"] += rep.n_threshold
            return rep.admissible

        return solver.minimise(obj, guard=guard)
