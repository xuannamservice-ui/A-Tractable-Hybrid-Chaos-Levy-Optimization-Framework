"""
Hybrid Chaos-Enhanced Levy-Flight PSO-GA (H-CLPSO-GA).

Reference implementation of the solver specified in Section V and Table 4:

  chaotic initialisation   logistic map x_{n+1} = 4 x_n (1 - x_n), spreading the
                           swarm more evenly than uniform sampling
  Levy flight jumps        Mantegna's algorithm at lambda = 1.5, giving the
                           heavy-tailed steps that escape the traps a Gaussian
                           perturbation cannot (Lemma 2)
  PSO core                 inertia + cognitive + social update
  GA refinement            arithmetic crossover among the top eta_e = 20%
  anytime operation        a monotone incumbent is maintained at all times, so
                           the search can be stopped at any iteration and still
                           return its best feasible candidate
  fidelity ladder          the series order K is selected per candidate from the
                           conditioning parameter z, not fixed globally

Defaults follow Table 4: N_p = 30, T_iter = 25, lambda = 1.5, eta_e = 20%.

This is a reference implementation written from the specification in the paper.
It is not the campaign driver that produced the tabulated results, and the
numbers it produces are its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.special import gamma as G

LADDER = ((0.5, 5), (2.0, 10), (8.0, 20))     # z threshold -> series order


def ladder_order(z: np.ndarray):
    """Per-candidate series order; None (encoded as -1) means inadmissible."""
    K = np.full(np.shape(z), -1, dtype=int)
    for zt, k in reversed(LADDER):
        K = np.where(z <= zt, k, K)
    return K


def logistic_chaos(n: int, seed_value: float) -> np.ndarray:
    """Logistic map at r = 4, the fully chaotic regime (Section V-B1)."""
    x = np.empty(n)
    v = float(seed_value)
    for i in range(n):
        v = 4.0 * v * (1.0 - v)
        # the map has fixed points at 0 and 0.75; nudge away if we land on one
        if v in (0.0, 0.25, 0.5, 0.75, 1.0):
            v = (v + 0.123456789) % 1.0
        x[i] = v
    return x


def levy(rng: np.random.Generator, n: int, lam: float = 1.5) -> np.ndarray:
    """Mantegna's algorithm for symmetric Levy-stable steps of index lam."""
    num = G(1 + lam) * np.sin(np.pi * lam / 2.0)
    den = G((1 + lam) / 2.0) * lam * 2 ** ((lam - 1) / 2.0)
    sigma = (num / den) ** (1.0 / lam)
    u = rng.normal(0.0, sigma, n)
    v = np.abs(rng.normal(0.0, 1.0, n))
    return u / v ** (1.0 / lam)


@dataclass
class SolverConfig:
    n_particles: int = 30            # N_p          (Table 4)
    max_iters: int = 25              # T_iter       (Table 4)
    levy_lambda: float = 1.5         # lambda       (Table 4)
    elite_fraction: float = 0.20     # eta_e        (Table 4)
    inertia: float = 0.70
    cognitive: float = 1.5
    social: float = 1.5
    jump_probability: float = 0.25
    jump_scale: float = 0.02
    # Stagnation gate. OFF by default: with stagnation_gated False the jump fires
    # unconditionally at jump_probability, which is what every published measurement in
    # this repository was taken under. Turning it on reproduces the mechanism the
    # manuscript describes, Var(J_gbest) < epsilon_s over a short window, so that the
    # described operator can be measured rather than only asserted.
    stagnation_gated: bool = False
    stagnation_window: int = 5       # W: incumbents inspected
    stagnation_eps: float = 1e-6     # epsilon_s, RELATIVE variance of the window
    # Jump geometry. "decoupled_opt" is the physical decoupled mechanism:
    # heavy-tailed Levy flight on the optical divergence block (w_z) unconstrained
    # by mechanical slew rate, plus smooth Gaussian perturbation on mechanical
    # steering angles (theta_az, theta_el).
    # "per_dim" is the legacy Cartesian mechanism; "feas_shift" is block-common shift.
    jump_mode: str = "decoupled_opt"       # "decoupled_opt" | "feas_shift" | "per_dim"
    # Warm start.  The deployed controller re-uses the previous cycle's
    # solution as the swarm anchor (the docstring notes the alternative,
    # per-stage seeding, violates eq. (14) almost surely).  In the solver the
    # same physics applies to any receding-horizon problem: if set, the swarm
    # is initialised as a cloud around `init_centre` with spread
    # `init_spread` (box fraction) instead of uniformly over the box.
    init_centre: Optional[np.ndarray] = None
    init_spread: float = 0.02
    # Law of the warm-start cloud around `init_centre`. "chaotic" is the
    # deployed form: the logistic map on [0,1] mapped to [-1,1], so every
    # particle lands within `init_spread` of the anchor and the cloud CANNOT
    # reach further however the plant moves. "levy" replaces that bounded draw
    # with a Mantegna step of the same scale: the bulk stays as close to the
    # anchor, but a minority of particles are thrown far. The two laws are
    # indistinguishable while the previous solution remains near-optimal and
    # differ only when the optimum has moved outside the bounded cloud, which
    # is exactly the case a rate-limited pointing loop meets after a large sway
    # excursion. "gauss" is the light-tailed control at matched scale: it
    # isolates tail weight from mere unboundedness. "smooth" is the one law
    # actually safe for a MULTI-STAGE trajectory anchor such as
    # BeamSteeringMPC's: the three laws above jitter every stage
    # independently, which for a tight per-stage slew tube (e.g. the
    # divergence block's 0.05-unit/cycle limit against a box width of order
    # 1) reliably produces an all-infeasible swarm; "smooth" instead shifts
    # each block's LEVEL by one chaotic scalar per particle, preserving the
    # anchor's own (already slew-feasible) stage-to-stage differences.
    # Ignored unless `init_centre` is set; "chaotic" leaves the deployed
    # code path untouched.
    init_law: str = "chaotic"        # "chaotic" | "levy" | "gauss" | "smooth"
    # Fraction of the swarm drawn from `init_law`'s cloud; the remainder keeps the
    # deployed bounded chaotic one. This is what makes the initialiser a HYBRID
    # rather than a contest between pure strategies, and the two questions are not
    # the same: an operator can lose every pure-arm comparison and still pay at a
    # small allocation. PSO's social term is why. A far-flung particle that finds
    # nothing never becomes gbest and never moves the swarm, so it costs only its
    # own evaluation; one that finds something better pulls the whole swarm to it.
    # The payoff is therefore asymmetric in the explorer's favour, which is the
    # classical condition under which a small exploration allocation dominates both
    # pure exploitation and pure exploration. 1.0 applies `init_law` to the whole
    # swarm (the pure arm); 0.0 reduces exactly to the deployed bounded cloud for
    # every law, so it is the shared control point of any mixture sweep.
    init_levy_fraction: float = 1.0
    use_chaos: bool = True
    use_levy: bool = True
    use_ga: bool = True
    use_fidelity_ladder: bool = True
    fixed_order: int = 10             # used when the ladder is disabled
    smooth_span: float = 0.01        # trajectory init spread, box fraction


@dataclass
class SolverResult:
    best_x: Optional[np.ndarray]
    best_f: float
    iterations: int
    evaluations: int
    incumbent_trace: list = field(default_factory=list)
    rejected_by_guard: int = 0


class HCLPSOGA:
    """Anytime hybrid solver over a box-constrained decision vector."""

    def __init__(self, lower, upper, config: SolverConfig = SolverConfig(), seed: int = 0,
                 blocks=None, repair: Optional[Callable] = None,
                 block_slew: Optional[list] = None):
        self.lo = np.atleast_1d(np.asarray(lower, dtype=float))
        self.hi = np.atleast_1d(np.asarray(upper, dtype=float))
        self.dim = self.lo.size
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        # `blocks` lists the (start, end) index range of each physical variable's
        # stage block, so the swarm is seeded one level per variable rather than
        # one level for the whole concatenated vector -- the blocks of a 60-D
        # trajectory carry different units and box widths.
        self.blocks = list(blocks) if blocks is not None else [(0, self.dim)]
        # Per-block slew limit (|x_k - x_{k-1}| per cycle) for the feas_shift
        # jump mode; None keeps the deployed per_dim geometry.
        self.block_slew = list(block_slew) if block_slew is not None else None
        # `repair` maps an arbitrary point to a feasible one.  Section V-B2:
        # "When a jump generates a coordinate exceeding the physical slew-rate
        # or angular limits, the particle is not simply truncated; instead, it
        # is reflected back into the feasible search space."  Without it the
        # swarm spends its whole budget on candidates the objective returns
        # +inf for, and no component of the kernel can differentiate.
        self.repair = repair

    # ------------------------------------------------------------------
    def _feasible(self, x: np.ndarray) -> np.ndarray:
        """Reflect into the box, then project onto the slew-rate tube."""
        span = self.hi - self.lo
        t = np.abs((x - self.lo) % (2.0 * span) )
        x = self.lo + np.where(t > span, 2.0 * span - t, t)
        x = np.clip(x, self.lo, self.hi)          # guards against fp overshoot
        return self.repair(x) if self.repair is not None else x

    # ------------------------------------------------------------------
    def _initialise(self) -> np.ndarray:
        """Chaotic initialisation.

        For a trajectory decision vector the swarm is seeded with *smooth*
        trajectories: a chaotically-spread base level per block per particle
        plus a small per-stage variation bounded by `smooth_span`. Seeding each
        stage independently would violate the slew-rate constraint of eq. (14)
        almost surely and leave the whole swarm infeasible, which is also why
        the deployed controller warm-starts from the previous cycle's solution.
        """
        n, d = self.cfg.n_particles, self.dim
        span = self.hi - self.lo

        # Heavy- or light-tailed warm start. Placed BEFORE the chaotic draw so
        # that `init_law == "chaotic"` reaches the original code below with the
        # original random-number consumption: every measurement taken before
        # this option existed is reproduced bit for bit.
        law = self.cfg.init_law
        frac = float(self.cfg.init_levy_fraction)
        warm = self.cfg.init_centre is not None

        if warm and law == "smooth":
            # Warm start for a TRAJECTORY decision vector: perturb each
            # block's LEVEL by one chaotic scalar per particle, not every
            # stage independently. A uniform per-block shift leaves every
            # stage-to-stage difference of `init_centre` unchanged, so a
            # centre that already respects the slew-rate constraint (e.g. a
            # previously solved, repaired trajectory) stays slew-feasible
            # after the shift for every particle -- unlike the
            # "chaotic"/"levy"/"gauss" warm laws below, which jitter every
            # stage independently and can blow a tight slew tube (the
            # divergence block's 0.05-unit/cycle limit is far smaller than a
            # box-fraction jitter) even at a small `init_spread`, leaving an
            # all-infeasible swarm with no admissible incumbent at all.
            c = np.asarray(self.cfg.init_centre, dtype=float)
            nb = len(self.blocks)
            shift_draw = (logistic_chaos(n * nb, self.rng.uniform(0.1, 0.9)).reshape(n, nb)
                          if self.cfg.use_chaos else self.rng.random((n, nb)))
            x = np.tile(c[None, :], (n, 1))
            for bi, (s, e) in enumerate(self.blocks):
                level_shift = (shift_draw[:, bi:bi + 1] - 0.5) * 2.0 * self.cfg.init_spread * span[s]
                x[:, s:e] += level_shift
            return self._feasible(x)

        def _tail(m):
            """m rows of the unbounded cloud's step, in the requested law."""
            if law == "levy":
                return levy(self.rng, m * d, self.cfg.levy_lambda).reshape(m, d)
            if law == "gauss":
                return self.rng.normal(size=(m, d))
            raise ValueError("init_law must be 'chaotic', 'levy' or 'gauss', "
                             "not %r" % (law,))

        if warm and law != "chaotic" and frac >= 1.0:
            c = np.asarray(self.cfg.init_centre, dtype=float)
            return self._feasible(c[None, :] + _tail(n) * self.cfg.init_spread * span)

        draw = (logistic_chaos(n * d, self.rng.uniform(0.1, 0.9)).reshape(n, d)
                if self.cfg.use_chaos else self.rng.random((n, d)))

        if warm:
            # warm start: cloud around the anchor, spread as a box fraction
            c = np.asarray(self.cfg.init_centre, dtype=float)
            x = (c[None, :]
                 + (draw - 0.5) * 2.0 * self.cfg.init_spread * span)
            if law != "chaotic" and frac > 0.0:
                # Hybrid initialiser: the last n_h particles trade the bounded
                # cloud for the unbounded one, so the swarm carries exploiters and
                # explorers at once instead of being one or the other. At frac = 0
                # this block does not run and the draw is bit-identical to the
                # deployed bounded cloud, which is what makes frac = 0 a usable
                # shared control rather than a fourth arm.
                n_h = int(round(frac * n))
                if n_h > 0:
                    x[n - n_h:] = (c[None, :]
                                   + _tail(n_h) * self.cfg.init_spread * span)
            return self._feasible(x)

        if d == 1 or self.cfg.smooth_span is None:
            return self._feasible(self.lo + draw * span)

        x = np.empty((n, d))
        for (s, e) in self.blocks:
            base = self.lo[s] + draw[:, s:s + 1] * span[s]     # one level per block
            jitter = (draw[:, s:e] - 0.5) * 2.0 * self.cfg.smooth_span
            x[:, s:e] = base + jitter * span[s:e]
        return self._feasible(x)

    # ------------------------------------------------------------------
    def minimise(self, objective: Callable, guard: Optional[Callable] = None,
                 checkpoint: Optional[Callable] = None) -> SolverResult:
        """`objective(X) -> (f, aux)`; `guard(X, f, aux) -> bool mask of admissible`."""
        cfg = self.cfg
        x = self._initialise()
        v = np.zeros_like(x)
        n_elite = max(2, int(cfg.elite_fraction * cfg.n_particles))

        pbest_x = x.copy()
        pbest_f = np.full(cfg.n_particles, np.inf)
        best_x, best_f = None, np.inf
        evals, rejected = 0, 0
        trace = []
        self._jump_fired = 0
        self._gate_open_iters = 0

        for it in range(cfg.max_iters):
            x = self._feasible(x)
            f, aux = objective(x)
            evals += cfg.n_particles

            ok = np.isfinite(f)
            if guard is not None:
                admissible = guard(x, f, aux)
                rejected += int(np.sum(~admissible))
                ok &= admissible
            fw = np.where(ok, f, np.inf)

            improved = fw < pbest_f
            pbest_f[improved] = fw[improved]
            pbest_x[improved] = x[improved]

            i = int(np.argmin(fw))
            if fw[i] < best_f:                       # monotone incumbent
                best_f, best_x = float(fw[i]), x[i].copy()
            trace.append(best_f)

            if checkpoint is not None and checkpoint(it, best_f):
                return SolverResult(best_x, best_f, it + 1, evals, trace, rejected)

            # --- PSO core -------------------------------------------------
            r1 = self.rng.random((cfg.n_particles, self.dim))
            r2 = self.rng.random((cfg.n_particles, self.dim))
            g = best_x if best_x is not None else x[i]
            v = (cfg.inertia * v
                 + cfg.cognitive * r1 * (pbest_x - x)
                 + cfg.social * r2 * (g - x))
            x = x + v

            # --- heavy-tailed exploration ---------------------------------
            # The gate, when enabled, is the manuscript's Var(J_gbest) < epsilon_s. The
            # incumbent is monotone here, so a flat window means no improvement landed in
            # it; the variance is taken relative to the window mean so the threshold does
            # not depend on the absolute scale of J, which spans orders of magnitude
            # across jitter strata. With the gate off this is unconditional, exactly as
            # every measurement already in this repository was taken.
            gate_open = True
            if cfg.stagnation_gated:
                w = trace[-cfg.stagnation_window:]
                if len(w) < cfg.stagnation_window:
                    gate_open = False          # not enough history to call it stagnant
                else:
                    aw = np.asarray(w, float)
                    aw = aw[np.isfinite(aw)]
                    if aw.size < 2:
                        gate_open = False
                    else:
                        scale = max(abs(float(np.mean(aw))), 1e-300)
                        gate_open = bool(np.var(aw) / (scale ** 2) < cfg.stagnation_eps)
            jump = ((self.rng.random(cfg.n_particles) < cfg.jump_probability)
                    if gate_open else np.zeros(cfg.n_particles, bool))
            k = int(jump.sum())
            self._jump_fired += k
            self._gate_open_iters += int(gate_open)
            if k:
                span = (self.hi - self.lo)
                if cfg.jump_mode == "decoupled_opt" and len(self.blocks) >= 2:
                    # Decoupled Optical-Mechanical Operator:
                    # Block 0: Optical beam divergence w_z (unconstrained by mechanical slew rate).
                    # Preserves 100% of heavy-tailed Levy flight reach across multimodal basins / z > 8 wall.
                    s0, e0 = self.blocks[0]
                    span0 = span[s0:e0]
                    d0 = e0 - s0
                    if cfg.use_levy:
                        steps_opt = levy(self.rng, k * d0, cfg.levy_lambda).reshape(k, d0)
                    else:                                  # ablation: Gaussian
                        steps_opt = self.rng.normal(size=(k, d0))
                    x[jump, s0:e0] += cfg.jump_scale * steps_opt * span0

                    # Blocks 1 and 2: Mechanical steering angles (theta_az, theta_el).
                    # Smooth Gaussian perturbation respecting physical actuator slew rate.
                    for bi in range(1, len(self.blocks)):
                        (sb, eb) = self.blocks[bi]
                        sl = self.block_slew[bi] if (self.block_slew is not None and bi < len(self.block_slew)) else 0.05e-3
                        x[jump, sb:eb] += self.rng.normal(size=(k, eb - sb)) * (0.3 * sl)
                elif cfg.jump_mode == "feas_shift" and self.block_slew is not None:
                    # One heavy-tailed scalar per physical block, shifting all
                    # T stages of that block together.  Stage-to-stage
                    # differences are unchanged, so the slew tube is preserved
                    # by construction; only the box can clip.  A small jitter
                    # at 0.3x the block slew limit adds shape without leaving
                    # the tube.  This is the jump geometry under which the
                    # heavy tail survives the feasibility repair; the per_dim
                    # geometry deletes it (forward sweep = low-pass filter).
                    nb = len(self.blocks)
                    if cfg.use_levy:
                        steps = levy(self.rng, k * nb, cfg.levy_lambda).reshape(k, nb)
                    else:                                  # ablation: Gaussian
                        steps = self.rng.normal(size=(k, nb))
                    for bi, ((s, e), sl) in enumerate(
                            zip(self.blocks, self.block_slew)):
                        spanb = float(np.max(span[s:e]))
                        x[jump, s:e] += cfg.jump_scale * steps[:, bi, None] * spanb
                        x[jump, s:e] += self.rng.normal(size=(k, e - s)) * (0.3 * sl)
                else:
                    if cfg.use_levy:
                        steps = levy(self.rng, k * self.dim, cfg.levy_lambda).reshape(k, self.dim)
                    else:                                  # ablation: Gaussian instead
                        steps = self.rng.normal(size=(k, self.dim))
                    x[jump] += cfg.jump_scale * steps * span

            # --- GA refinement on the elite -------------------------------
            if cfg.use_ga:
                order = np.argsort(pbest_f)
                elite = pbest_x[order[:n_elite]]
                if np.all(np.isfinite(pbest_f[order[:n_elite]])):
                    m = cfg.n_particles // 3
                    ia = self.rng.integers(0, n_elite, m)
                    ib = self.rng.integers(0, n_elite, m)
                    w = self.rng.random((m, self.dim))
                    x[order[-m:]] = w * elite[ia] + (1 - w) * elite[ib]

        return SolverResult(best_x, best_f, cfg.max_iters, evals, trace, rejected)
