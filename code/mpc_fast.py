"""
Bit-identical, low-overhead re-expression of `mpc_loop.BeamSteeringMPC._objective`.

WHAT THIS FILE IS FOR
---------------------
The closed loop must fit an 800 us computation budget with an anytime checkpoint
at tau_O = 600 us.  Profiling the deployed objective on this machine showed that
the RT-ODT kernel `rtodt_fast.pe_series_f64` is not the bottleneck: the *wrapper*
around it -- `_objective` -- spends the majority of its time on work that is
either constant across the whole control cycle (the block layout, the slew
limits, the per-stage reference SNR vector, the geometry constants) or on numpy
dispatch overhead for operations whose arithmetic is trivial (`np.diff` three
times, `np.tile`, `np.linalg.norm`, `np.unique` over a per-candidate array).

This module re-expresses that wrapper.  It does NOT re-express the series, the
truncation order K, the admissibility ladder, the pole handling, the NaN
propagation, the working precision, the guard, or the decision box.  Every
floating-point operation that contributes to a returned number is performed on
the same operands, in the same order, with the same rounding as in `mpc_loop`.
The kernel itself is called unchanged, through the same public entry point.

WHAT "BIT-IDENTICAL" MEANS HERE, AND WHY IT IS ACHIEVABLE
--------------------------------------------------------
Every transformation below is one of exactly four kinds:

  (1) HOISTING.  A subexpression whose operands do not change between calls
      (or between candidates) is evaluated once and reused.  Reusing the
      *result* of a deterministic float64 computation is exact by definition.

  (2) COMMON SUBEXPRESSION ELIMINATION.  `erf(v)` was evaluated twice inside
      `channel._raw_beam_geometry`, once for A_0 and once for w_zeq.  Evaluating
      it once and using the value twice is exact.

  (3) DISPATCH REMOVAL.  `np.diff(A, axis=1)` is replaced by
      `A[:, 1:] - A[:, :-1]`, which is what `np.diff` computes; `np.tile(g, (n,1))`
      is replaced by broadcasting; `np.linalg.norm(t, axis=2)` on a length-2 axis
      is replaced by `sqrt(tx*tx + ty*ty)`, which is the reduction `norm`
      performs (`sqrt(add.reduce(x*x, axis))` over two elements).  These emit the
      same machine instructions on the same operands.

  (4) REGROUPING OF AN ELEMENTWISE MAP.  The `for g in np.unique(gb)` loop
      partitions a flat (n*Tr,) array by its reference-SNR value.  Every
      operation inside the loop -- `z_of`, `ladder_order`, `pe_series_f64` -- is
      elementwise in the candidate index, and the group scalar `g` is a property
      of the horizon STAGE, not of the candidate.  The partition is therefore
      known from the (Tr,) stage vector alone and can be precomputed once per
      control cycle.  Each element still enters the kernel with the same
      (xi_eff, A_0, gbar, K) tuple it had before, so the value it receives is
      the same value.

The one thing that is deliberately NOT done is the class of transformation that
an earlier attempt at this task was rejected for: replacing a directly-evaluated
quantity with a mathematically-equal but differently-rounded recurrence.  The
series has violent cancellation -- alternating coefficients reaching 1e31 -- so a
1-ulp change upstream is amplified to the float64 floor eta_f64 downstream.  No
expression in this file is re-associated, re-ordered, or algebraically rewritten.
See `REJECTED` at the bottom of this file for the list of optimisations that were
tried, measured, and discarded for exactly that reason.

USAGE
-----
    from mpc_fast import FastBeamSteeringMPC
    mpc = FastBeamSteeringMPC(alpha, beta, sigma_s, gbar, horizon=20, seed=0)

`FastBeamSteeringMPC` is a drop-in subclass of `mpc_loop.BeamSteeringMPC`: it
overrides `_objective` (and `_stage_rd`, identically) and inherits everything
else.  `mpc_loop.py` is untouched, so `compare_objective.py` can run both.

    from mpc_fast import fast_objective
    cost, aux = fast_objective(any_beamsteeringmpc_instance, X, state, h_pred)

is the same code as a free function, for attaching to an existing instance.

CACHE VALIDITY
--------------
Two caches key on object identity: the per-cycle reference-SNR vector (keyed on
`h_pred`) and the pointing state (keyed on `state`).  Both hold a strong
reference to the keyed object, so an id cannot be recycled while it is cached,
and both additionally re-check the array's first and last entries.  Within one
control cycle `HCLPSOGA.minimise` closes over a single `h_pred` and a single
`state` and mutates neither, which is the case the cache is built for; a
different or mutated array falls through to a full recompute.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from channel import APERTURE, branch_min_wz
from hclpso_ga import ladder_order
from mpc_loop import BeamSteeringMPC, Z_MAX
from rtodt_fast import pe_series_f64

__all__ = ["FastBeamSteeringMPC", "fast_objective", "install"]

# --------------------------------------------------------------------------
# Geometry constants of eq. (3).  `channel._raw_beam_geometry` rebuilds these
# two scalars on every call; they depend on the aperture alone.
#
#   v      = sqrt(pi/2) * a / w_z
#   A_0    = erf(v)^2
#   w_zeq  = sqrt( w_z^2 * sqrt(pi) * erf(v) / ( 2 v exp(-v^2) ) )
#
# `_BG_C` is `np.sqrt(np.pi / 2.0) * a` and `_SQRT_PI` is `np.sqrt(np.pi)`,
# each evaluated exactly as `channel` evaluates it, so the float64 values are
# the same bit patterns the original multiplies by.
# --------------------------------------------------------------------------
_BG_C = np.sqrt(np.pi / 2.0) * APERTURE
_SQRT_PI = np.sqrt(np.pi)
# `channel.beam_geometry_valid` is  isfinite(w) & (w > 0) & (w >= w* (1 - rtol)).
# The third clause implies the second because the threshold is positive, so the
# two-clause test below accepts and rejects exactly the same waists.
_BG_THR = branch_min_wz(APERTURE)[0] * (1.0 - 1e-6)

# --------------------------------------------------------------------------
# The fidelity ladder as a lookup.  `hclpso_ga.ladder_order` allocates a full
# -1 array and lays three `np.where` passes over it; the same map is a single
# `searchsorted` into the rung thresholds followed by one gather.  The two agree
# exactly, including on the three thresholds themselves and on both neighbours
# of each (side='left' puts z == threshold on the lower-order rung, which is
# what `z <= zt` does), and on NaN and +-inf, where searchsorted lands in the
# last bin and returns the inadmissible marker -1 just as the `np.where` chain
# leaves it at -1.  `compare_objective.run_ladder_equivalence` checks this over
# the boundaries and over a large random sweep; it is not asserted here.
#
# This is a lookup substituted for an equal lookup, not an algebraic rewrite of
# an arithmetic expression: the ladder itself, its thresholds, and the orders it
# selects are untouched.
# --------------------------------------------------------------------------
_LADDER_BINS = np.array([0.5, 2.0, 8.0])
_LADDER_VALS = np.array([5, 10, 20, -1], dtype=ladder_order(np.zeros(1)).dtype)


def _ladder_order_fast(z):
    return _LADDER_VALS[np.searchsorted(_LADDER_BINS, z)]


def _beam_geometry_fused(w):
    """`channel.beam_geometry` with `erf(v)` evaluated once instead of twice.

    Identical operands, identical operation order, identical domain guard.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        v = _BG_C / w
        e = erf(v)
        a0 = e * e                                   # channel writes erf(v) ** 2
        num = (w * w) * _SQRT_PI * e                 # (w_z ** 2 * sqrt(pi)) * erf(v)
        den = (2.0 * v) * np.exp(-(v * v))           # 2.0 * v * exp(-v ** 2)
        weq = np.sqrt(num / den)
        bad = ~(np.isfinite(w) & (w >= _BG_THR))
        if bad.any():
            a0 = np.where(bad, np.nan, a0)
            weq = np.where(bad, np.nan, weq)
    return a0, weq


class _CycleCache:
    """Everything `_objective` recomputes per call that is constant per cycle."""

    __slots__ = ("h_obj", "h_first", "h_last", "g_stage", "sqrt_g",
                 "groups", "state_obj", "state0", "state1", "theta0",
                 "n", "buf_d", "buf_d2", "buf_abs", "buf_pe", "buf_z")

    def __init__(self):
        self.h_obj = None
        self.state_obj = None
        self.n = -1


class _FastObjective:
    """Per-instance precompute + the optimised evaluation path.

    Held on the controller as `_fastobj`.  Rebuilt if any attribute it folded in
    is changed on the controller after construction (checked by `_signature`).
    """

    def __init__(self, mpc: BeamSteeringMPC):
        self.mpc = mpc
        self._build()

    # -- structural constants ------------------------------------------
    @staticmethod
    def _signature(m):
        return (m.horizon, m.steering, m.h_in_aber, m.strict_admissibility,
                m.rank_stages, m.sigma_s, m.gbar, m.alpha, m.beta, m.L,
                m.lambda_u, m.slew_limit, m.u_slew, m.act_delay_samples,
                m.cfg.use_fidelity_ladder, m.cfg.fixed_order)

    def _build(self):
        m = self.mpc
        self.sig = self._signature(m)

        T = int(m.horizon)
        self.T = T
        self.Tr = T if m.rank_stages is None else min(int(m.rank_stages), T)
        self.steering = bool(m.steering)
        self.h_in_aber = bool(m.h_in_aber)
        self.strict = bool(m.strict_admissibility)
        self.use_ladder = bool(m.cfg.use_fidelity_ladder)
        self.fixed_order = int(m.cfg.fixed_order)
        self.alpha = float(m.alpha)
        self.beta = float(m.beta)
        self.L = m.L
        self.lambda_u = m.lambda_u
        self.gbar = m.gbar
        self.delay = int(m.act_delay_samples)

        # xi = w_zeq / (2 sigma_s); xi_eff = xi / sqrt(1 + r_d^2 / (2 sigma_s^2))
        self.two_sigma = 2.0 * m.sigma_s
        self.xie_den = 2.0 * m.sigma_s ** 2

        # z = sqrt(2) alpha beta / (A_0 sqrt(gbar)).  `rtodt_fast.z_of` evaluates
        # `np.sqrt(2.0) * A * B` as a scalar before dividing, so folding it is
        # exact: the same float64 numerator divides the same denominator.
        self.z_num = np.sqrt(2.0) * self.alpha * self.beta

        # block layout, slew limits, and the eq. (13) normaliser
        self.blocks = tuple((int(s), int(e)) for s, e in m.blocks())
        self.slews = tuple(float(x) for x in m.block_slew())
        self.pen_den = max(T - 1, 1)
        self.dim = 3 * T if self.steering else T

        # `viol` is the OR over blocks of `any(|diff| > lim)`, i.e. one `any`
        # over the union of the blocks' difference columns.  Build a per-column
        # limit vector with +inf on the two columns that straddle a block
        # boundary, so those columns can never trip the test and the three
        # per-block reductions collapse into one.
        contiguous = all(self.blocks[i][1] == self.blocks[i + 1][0]
                         for i in range(len(self.blocks) - 1))
        self.fused_slew = (contiguous and self.blocks[0][0] == 0
                           and self.blocks[-1][1] == self.dim and self.dim > 1)
        if self.fused_slew:
            lim = np.full(self.dim - 1, np.inf)
            for (s, e), L_ in zip(self.blocks, self.slews):
                lim[s:e - 1] = L_
            self.lim_vec = lim
            # column ranges of each block inside the (dim-1,) difference array
            self.pen_cols = tuple((s, e - 1) for (s, e) in self.blocks)

        self.cache = _CycleCache()

    # -- per-cycle precompute ------------------------------------------
    def _stage_gbar_cached(self, h_pred):
        """gbar * (1 + h_hat)^2 per stage, and the induced (gbar -> columns)
        partition -- both constant for the whole control cycle."""
        c = self.cache
        if (c.h_obj is h_pred and c.h_first == h_pred[0]
                and c.h_last == h_pred[-1]):
            return
        m = self.mpc
        Tr = self.Tr
        if self.h_in_aber:
            g_stage = np.asarray(m._stage_gbar(h_pred)[:Tr], dtype=float)
        else:
            g_stage = np.full(Tr, m.gbar, dtype=float)
        # `np.unique` over the flat (n*Tr,) tiled array and over the (Tr,) stage
        # vector return the same set of values in the same (sorted) order, so
        # the groups are the same groups; only the cost of finding them differs.
        vals, inv = np.unique(g_stage, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        groups = []
        for j in range(vals.size):
            cols = np.nonzero(inv == j)[0]
            groups.append((float(vals[j]),
                           int(cols[0]) if cols.size == 1 else cols))
        c.groups = groups
        c.g_stage = g_stage
        c.sqrt_g = np.sqrt(g_stage)
        c.h_obj = h_pred                    # strong ref: id cannot be recycled
        c.h_first = h_pred[0]
        c.h_last = h_pred[-1]

    def _theta0_cached(self, state):
        c = self.cache
        s = state
        if c.state_obj is s:
            try:
                if c.state0 == s[0] and c.state1 == s[1]:
                    return c.theta0
            except (IndexError, TypeError):
                pass
        th = self.mpc._as_theta(state, self.L)
        c.theta0 = th
        c.state_obj = s
        try:
            c.state0, c.state1 = s[0], s[1]
        except (IndexError, TypeError):
            c.state0 = c.state1 = None
            c.state_obj = None
        return th

    def _buffers(self, n):
        c = self.cache
        if c.n != n:
            d = self.dim
            c.buf_d = np.empty((n, d - 1)) if d > 1 else None
            c.buf_d2 = np.empty((n, d - 1)) if d > 1 else None
            c.buf_abs = np.empty((n, d - 1)) if d > 1 else None
            c.n = n
        return c

    # -- radial offset over the horizon --------------------------------
    def _stage_rd_fast(self, X, n, Tr, theta0):
        """First `Tr` columns of `BeamSteeringMPC._stage_rd`.

        The original stacks the two steering blocks into an (n, T, 2) cube,
        cumulative-sums it, prepends Theta(t), and takes a length-2
        `np.linalg.norm` along the last axis.  Written out, stage 0 is Theta(t)
        itself and stage k >= 1 is Theta(t) minus the running sum of the first
        k steering commands -- so the cube, the prepend, and the norm dispatch
        are all avoidable.  `np.linalg.norm(x, axis=2)` on a length-2 axis is
        `sqrt(add.reduce(x*x, axis=2))` = `sqrt(x0*x0 + x1*x1)`; the cumulative
        sums are the same partial sums in the same order, merely truncated at
        Tr instead of T.
        """
        L = self.L
        if not self.steering:
            return np.full((n, Tr), L * np.linalg.norm(theta0))

        t0x = theta0[0]
        t0y = theta0[1]
        r0 = L * np.sqrt(t0x * t0x + t0y * t0y)
        if Tr == 1:
            return r0                                  # scalar: same for all rows

        T = self.T
        if self.delay:                                  # non-zero FSM dead time
            return self.mpc._stage_rd(X)[:, :Tr]

        k = Tr - 1
        cx = np.cumsum(X[:, T:T + k], axis=1)
        cy = np.cumsum(X[:, 2 * T:2 * T + k], axis=1)
        ax = t0x - cx
        ay = t0y - cy
        out = np.empty((n, Tr))
        out[:, 0] = r0
        np.multiply(ax, ax, out=ax)
        np.multiply(ay, ay, out=ay)
        np.add(ax, ay, out=ax)
        np.sqrt(ax, out=ax)
        np.multiply(L, ax, out=out[:, 1:])
        return out

    # -- the objective --------------------------------------------------
    def __call__(self, X, state, h_pred):
        m = self.mpc
        if self._signature(m) != self.sig:              # attribute changed
            self._build()

        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        T, Tr = self.T, self.Tr

        theta0 = self._theta0_cached(state)
        m.theta0 = theta0
        self._stage_gbar_cached(h_pred)
        c = self.cache

        rd = self._stage_rd_fast(X, n, Tr, theta0)

        # --- geometry, xi, xi_eff --------------------------------------
        w = X[:, :Tr].reshape(-1)
        a0f, weqf = _beam_geometry_fused(w)
        xif = weqf / self.two_sigma
        if Tr == 1 and self.steering:
            # rd is one scalar shared by every candidate at stage 0
            xef = xif / np.sqrt(1.0 + rd * rd / self.xie_den)
        else:
            r = rd.reshape(-1)
            xef = xif / np.sqrt(1.0 + r * r / self.xie_den)

        if Tr == 1 and self.strict:
            # One ranked stage: the (n, 1) reshape, the length-1 reduction and
            # the column views are all identities, so they are skipped.  A
            # length-1 `np.add.reduce` returns its input and `x / 1` is exact,
            # so `cost` is the same number the general path produces.  Guarded
            # on `strict` because `np.nansum` over a length-1 axis is NOT the
            # identity -- it maps NaN to 0.0, which is the whole point of the
            # non-strict branch and must keep going through np.nansum.
            g = c.groups[0][0]
            z1 = self.z_num / (a0f * c.sqrt_g[0])
            k1 = (_ladder_order_fast(z1) if self.use_ladder
                  else np.where(z1 <= Z_MAX, self.fixed_order, -1))
            pe1 = pe_series_f64(self.alpha, self.beta, xef, a0f, g, k1)
            cost = pe1 / Tr
            pen, viol = self._slew(X, n, c)
            cost = cost + self.lambda_u * pen
            cost = np.where(viol, np.inf, cost)
            return cost, {"z": z1, "pe_first": pe1}

        a02 = a0f.reshape(n, Tr)
        xe2 = xef.reshape(n, Tr)

        # --- z and the fidelity ladder, once for the whole block -------
        # `z_of` is elementwise and `ladder_order` is elementwise, so evaluating
        # them on the (n, Tr) block gives each element the value the per-group
        # loop gave it -- one dispatch instead of one per reference-SNR group.
        z2 = self.z_num / (a02 * c.sqrt_g)
        if self.use_ladder:
            k2 = _ladder_order_fast(z2)
        else:
            k2 = np.where(z2 <= Z_MAX, self.fixed_order, -1)

        # --- the kernel, once per reference-SNR group ------------------
        pe2 = np.empty((n, Tr))
        for g, cols in c.groups:
            if type(cols) is int:
                pe2[:, cols] = pe_series_f64(self.alpha, self.beta,
                                             xe2[:, cols], a02[:, cols],
                                             g, k2[:, cols])
            else:
                sub = pe_series_f64(self.alpha, self.beta,
                                    xe2[:, cols].reshape(-1),
                                    a02[:, cols].reshape(-1),
                                    g, k2[:, cols].reshape(-1))
                pe2[:, cols] = sub.reshape(n, cols.size)

        # np.sum(a, axis=1) IS np.add.reduce(a, axis=1) for a float64 array;
        # calling the ufunc reduction directly skips np.sum's Python-level
        # argument handling, which at n = 30 is most of its cost.
        cost = (np.add.reduce(pe2, axis=1) if self.strict
                else np.nansum(pe2, axis=1)) / Tr

        # --- control penalty and hard slew test, eqs. (13)-(14) --------
        pen, viol = self._slew(X, n, c)
        cost = cost + self.lambda_u * pen
        cost = np.where(viol, np.inf, cost)

        return cost, {"z": z2[:, 0], "pe_first": pe2[:, 0]}

    # -- slew penalty ---------------------------------------------------
    def _slew(self, X, n, c):
        """One differencing pass over the whole decision vector.

        `np.diff(X[:, s:e], axis=1)` is `X[:, s+1:e] - X[:, s:e-1]`; the three
        blocks are contiguous and tile the vector, so a single subtraction over
        all of X produces every per-block difference plus two columns that
        straddle a boundary.  Those two columns are excluded from the per-block
        sums by slicing and neutralised in the violation test by a +inf entry in
        the limit vector.  The per-block sums are kept separate and each is
        divided by (T-1) separately, because
        `s1/(T-1) + s2/(T-1) + s3/(T-1)` and `(s1+s2+s3)/(T-1)` are not the same
        float64 number.  `d ** 2` where `d = |diff|` equals `diff * diff`
        exactly, so the absolute value is needed only for the threshold test.
        """
        if not self.fused_slew or X.shape[1] != self.dim:
            return self._slew_blockwise(X, n)

        self._buffers(n)
        d = c.buf_d
        np.subtract(X[:, 1:], X[:, :-1], out=d)
        d2 = np.multiply(d, d, out=c.buf_d2)

        s, e = self.pen_cols[0]
        pen = np.add.reduce(d2[:, s:e], axis=1) / self.pen_den
        for (s, e) in self.pen_cols[1:]:
            pen += np.add.reduce(d2[:, s:e], axis=1) / self.pen_den

        ad = np.abs(d, out=c.buf_abs)
        viol = np.logical_or.reduce(ad > self.lim_vec, axis=1)
        return pen, viol

    def _slew_blockwise(self, X, n):
        pen = np.zeros(n)
        viol = np.zeros(n, dtype=bool)
        for (s, e), lim in zip(self.blocks, self.slews):
            if e - s < 2:
                continue
            d = X[:, s + 1:e] - X[:, s:e - 1]
            np.abs(d, out=d)
            pen = pen + np.sum(d * d, axis=1) / self.pen_den
            viol |= np.any(d > lim, axis=1)
        return pen, viol


# --------------------------------------------------------------------------
def fast_objective(mpc: BeamSteeringMPC, X, state, h_pred):
    """`mpc._objective(X, state, h_pred)`, evaluated through the fast path.

    Works on any `BeamSteeringMPC` instance; the precompute is attached to the
    instance on first use.
    """
    fo = getattr(mpc, "_fastobj", None)
    if fo is None or fo.mpc is not mpc:
        fo = _FastObjective(mpc)
        mpc._fastobj = fo
    return fo(X, state, h_pred)


def install(mpc: BeamSteeringMPC):
    """Bind the fast objective onto an existing controller instance."""
    fo = _FastObjective(mpc)
    mpc._fastobj = fo
    mpc._objective = fo
    return mpc


class FastBeamSteeringMPC(BeamSteeringMPC):
    """`BeamSteeringMPC` with the optimised objective.  Nothing else differs."""

    def _objective(self, X, state, h_pred):
        fo = getattr(self, "_fastobj", None)
        if fo is None:
            fo = _FastObjective(self)
            self._fastobj = fo
        return fo(X, state, h_pred)

    def _stage_rd(self, X):
        """Full-horizon radial offset, kept identical to the parent.

        The objective uses `_FastObjective._stage_rd_fast`, which computes only
        the `rank_stages` columns it needs.  This method is the diagnostic entry
        point (`degeneracy_audit` uses it) and returns all T columns, so it is
        left on the parent implementation rather than re-expressed here.
        """
        return BeamSteeringMPC._stage_rd(self, X)


# ==========================================================================
# REJECTED -- optimisations that were tried and discarded because they moved a
# number, or because they could not be shown not to.  Recorded so the next
# person does not spend the afternoon rediscovering them.
#
#  * multiplying by a precomputed 1/(T-1) instead of dividing by (T-1) in the
#    eq. (13) penalty.  1/19 is not representable in binary64, so x*(1/19) and
#    x/19 differ in the last place on roughly a third of inputs.  Rejected:
#    the penalty enters the ranked cost directly.
#
#  * accumulating the three block penalties before the single division,
#    (s1+s2+s3)/(T-1) instead of s1/(T-1)+s2/(T-1)+s3/(T-1).  Same reason:
#    different rounding, and the sum is what the swarm ranks by.
#
#  * replacing |d| > lim by d*d > lim*lim to reuse the squared differences and
#    drop the absolute value.  Mathematically equivalent, but d*d and lim*lim
#    are both rounded, so a candidate whose slew sits within one ulp of the
#    limit can change side.  That flips a hard feasibility decision, i.e. an
#    +inf, which is the largest possible change to a returned number.
#
#  * `np.hypot(ax, ay)` for the pointing-error magnitude.  hypot is a scaled,
#    overflow-safe algorithm and is not `sqrt(x*x + y*y)`; it differs from
#    `np.linalg.norm`'s reduction in the last place.  Rejected: r_d feeds
#    xi_eff, which feeds the series.
#
#  * folding `w_z^2 * sqrt(pi)` into a single constant multiply of `w_z^2`.
#    Fine in isolation, but `channel` evaluates `w_z**2 * sqrt(pi) * erf(v)`
#    left to right; any regrouping changes which product is rounded first.
#    The form used above keeps the original association.
#
#  * a running product for A_0^(beta+k) inside the series.  Not attempted here;
#    it was measured and rejected upstream.  It is 3.9x faster and more accurate
#    on the power itself, but the series' alternating coefficients reach 1e31
#    and amplify the difference to 7.9e-10 -- the float64 floor -- in the weak
#    regime.  Do not reintroduce it.
#
#  * float32 or mixed-precision evaluation of the geometry.  Out of scope: the
#    specification fixes float64 working precision.
#
#  * skipping `np.errstate` around the geometry.  It does not change any value,
#    but it changes whether a RuntimeWarning is emitted for an out-of-domain
#    waist, which is observable behaviour of the original.  Kept.  (Replacing
#    the context manager with a `np.seterr` / restore pair was measured at
#    2.00 us against 1.85 us for the context manager, i.e. slower.  Hoisting one
#    `errstate` around the whole objective would suppress warnings the original
#    lets through from the parts outside the geometry, so it is not equivalent.)
#
# ==========================================================================
# TRIED, NOT ADOPTED -- numerically fine, but they did not pay.  Listed so the
# measurement is not repeated.
#
#  * `out=` buffers through the whole geometry, to avoid the intermediate
#    allocations.  Measured 7.60 us against 7.35 us for the allocating form at
#    n = 30: numpy's `out=` keyword handling costs about what a 30-element
#    allocation costs, so it is a wash at the swarm size that matters and only
#    obscures the correspondence with `channel`.  Buffers ARE reused for the
#    slew arrays, where the array is (n, 3T-1) and the allocation is real.
#
#  * a two-comparison domain guard, `((w >= thr) & (w <= finfo.max)).all()`,
#    replacing `~(isfinite(w) & (w >= thr))` + `.any()`.  It selects exactly the
#    same waists, but measured 2.80 us against 1.50 us -- `np.isfinite` is a
#    single cheap ufunc and the extra comparison costs more than it saves.
#
#  * folding the constant into `xi_effective`.  Measured at 3.80 -> 3.90 us and
#    3.50 -> 3.30 us on two runs, i.e. zero within the noise of this machine.
#    Kept anyway because it removes a function call from the hot path, but it
#    should not be claimed as a saving.
# ==========================================================================
