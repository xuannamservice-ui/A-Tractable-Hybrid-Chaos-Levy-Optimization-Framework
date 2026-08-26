"""Batched RT-ODT kernel -- `rtodt_fast.pe_series_f64` restructured, not rewritten.

WHY THIS EXISTS
---------------
Inside `mpc_loop._objective` the kernel is called on very small arrays.  With the
fidelity ladder active the per-candidate order array takes three distinct values
-- measured {5: 8, 10: 10, 20: 12} over a 30-candidate swarm at the published
stage -- so `pe_series_f64` runs its group loop three times over ~10 elements
each and pays the whole numpy dispatch cost every time.  Measured on this
machine (i5-14600KF, Windows, Python 3.14, numpy 2.5, P-cores pinned), one group
at N = 30 spends

    np.unique(K)                  5.1 us     pure bookkeeping
    5 x round() + 3 dict lookups  3.3 us     cache keys, rebuilt on every call
    a0**(B+k), twice             22.0 us     the actual arithmetic
    everything else              ~30   us    ~25 separate numpy calls on <=630 elts

so the arithmetic is a minority of the cost and the rest is per-operation
overhead multiplied by the number of groups.  This module removes the overhead
and leaves the arithmetic exactly alone.

WHAT WAS CHANGED, AND WHAT EACH CHANGE BOUGHT
---------------------------------------------
Measured cumulatively on the arguments `_objective` actually passes, min over
150 interleaved batches, N = 30 with the three ladder rungs live:

    rtodt_fast.pe_series_f64                              91.8 us
  + interned cache keys and set(K.tolist())               83.8 us   (-8.0)
  + the two Bessel families fused onto one tensor         61.5 us  (-22.3)
  + the residue hoisted out of the group loop             46.0 us  (-15.5)

  1. The two Bessel families (the alpha<->beta pair of eq. 19) are formed in ONE
     pass on a (2, m, K+1) tensor instead of two independent (m, K+1)
     broadcasts.  Every elementwise operation runs once instead of twice, on an
     array twice as long.  The two sums are still taken separately, over the
     trailing contiguous length-(K+1) axis, one row per candidate per family, so
     the pairwise summation blocks identically and the rounding is untouched.

  2. The residue -- the pointing-error term D * C(xi^2, gbar) -- carries no
     series order, so it is the same expression on every rung.  `pe_series_f64`
     rebuilds it inside each group; here it is evaluated once for the whole call
     and the groups add into it.  Candidates the ladder refused are skipped, as
     they are in the reference.

  3. The per-regime constants (the xi-free coefficients Kc, the exponent rows
     B+k and A+k, Gamma(alpha), Gamma(beta), alpha*beta) and the moment rows
     C(B+k, gbar), C(A+k, gbar) are hoisted into caches keyed on the exact
     floats, so `round(A, 10)` and its four siblings leave the hot path.  The
     caches are FILLED by calling `rtodt_fast._kc` and `rtodt_fast.c_moment`:
     the numbers stored are the reference's own.

  4. `np.unique(K)` becomes `set(K.tolist())` for the small arrays the objective
     passes -- 0.6 us against 5.1 us at N = 30.  The loop visits the same set of
     orders; the visiting order is unobservable because every output element
     belongs to one order and is written once.

  5. When a single order covers every candidate the masking is skipped and the
     group is evaluated on the input arrays directly.

  6. `pe_series_stage_batch` additionally takes a per-STAGE table of reference
     SNRs, so the caller can collapse `_objective`'s `for g in np.unique(gb)`
     loop -- measured 20 calls of 30 candidates, 45 (gbar, order) groups -- into
     one call with at most one group per rung.  Measured 1747 us -> 253 us.

WHAT WAS NOT CHANGED
--------------------
The series, its truncation order K, the admissibility ladder and its thresholds,
the pole handling, the NaN propagation for inadmissible candidates, the float64
working precision.  Every floating-point operation is performed on the same
operands, in the same association, in the same order as in `rtodt_fast`.

Two shortcuts that would have collapsed the group loop were measured and
rejected, and are NOT applied here:

  * evaluating every candidate at max(K) and dropping the group loop moves the
    returned Pe by up to 2.2e-6 relative -- 2800x the float64 floor eta_f64 of
    eq. (27).  `_promotion_check()` prints that measurement.

  * building the terms to max(K), zeroing those beyond each candidate's own
    order and summing the padded row keeps the same terms but lengthens the
    reduction, which re-blocks the pairwise summation: 26% of results change,
    by up to 1.8e-13 relative.  Below the floor, but not bit-identical, so it
    fails the bar this module is held to.

Nor does this module replace A_0^(B+k) by a running product.  That substitution
is faster and more accurate on the power itself, but the series coefficients
alternate in sign and reach 1e31, and the extra error amplifies through that
cancellation to the float64 floor in the weak regime.  It was rejected before
and is not reintroduced.

`python rtodt_batch.py` checks this module against `rtodt_fast.pe_series_f64`
over the admissible box, prints the measured maximum relative difference and the
bit-identical fraction, prints the promotion measurement above, and times both.
It asserts no pre-agreed figure.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as sp_gamma

from rtodt_fast import _kc, c_moment

__all__ = ["pe_series_batch", "pe_series_stage_batch", "clear_caches"]

# c_moment() evaluates np.sqrt(np.pi) on every call; the value is a constant.
_SQRT_PI = np.sqrt(np.pi)

_CONST: dict = {}       # (A, B, order)        -> _Regime
_CM: dict = {}          # (A, B, order, gbar)  -> (2, 1, K+1) moment rows
_CM_LIMIT = 8192        # same bound rtodt_fast puts on its own moment cache


class _Regime:
    """Everything in eq. (19) that does not depend on the candidate.

    Filled once per (alpha, beta, K).  The arrays are read-only: they are handed
    to numpy as broadcast operands and must never be written through.
    """

    __slots__ = ("order", "kk", "BA", "BAk", "KC", "ABv", "AB", "gA", "gB", "s")

    def __init__(self, A: float, B: float, order: int):
        k = np.arange(order + 1)                       # int64, as in rtodt_fast
        self.order = order
        self.kk = k[None, None, :]                     # (1, 1, K+1)
        # family 0 carries B (the alpha<->beta pair of eq. 19), family 1 carries A
        self.BA = np.array([B, A])[:, None, None]      # (2, 1, 1)
        s = np.stack([B + k, A + k])                   # (2, K+1), == the c_moment argument
        self.s = s
        self.BAk = s[:, None, :]                       # (2, 1, K+1) exponent rows
        self.KC = np.stack([_kc(A, B, order), _kc(B, A, order)])[:, None, :]
        self.ABv = np.array([A, B])[:, None]           # (2, 1) for Gamma(A-xi^2), Gamma(B-xi^2)
        self.AB = A * B
        self.gA = sp_gamma(A)
        self.gB = sp_gamma(B)
        for a in (self.kk, self.BA, self.BAk, self.KC, self.ABv, self.s):
            a.flags.writeable = False


def _regime(A: float, B: float, order: int) -> _Regime:
    key = (A, B, order)
    c = _CONST.get(key)
    if c is None:
        c = _CONST[key] = _Regime(A, B, order)
    return c


def _moments(A: float, B: float, order: int, gbar: float, c: _Regime):
    """The two moment rows C(B+k, gbar), C(A+k, gbar) as one (2, 1, K+1) block.

    Filled by calling `rtodt_fast.c_moment`, so the stored numbers are the
    reference's.  The key is the exact float, not `round(gbar, 10)`: rounding
    the key cost 1.5 us per group and could only ever change the result by
    silently serving the moments of a *different* gbar.
    """
    key = (A, B, order, gbar)
    cm = _CM.get(key)
    if cm is None:
        if len(_CM) > _CM_LIMIT:       # gbar varies per horizon stage; bound it
            _CM.clear()
        cm = np.stack([c_moment(c.s[0], gbar), c_moment(c.s[1], gbar)])[:, None, :]
        cm.flags.writeable = False
        _CM[key] = cm
    return cm


def clear_caches():
    _CONST.clear()
    _CM.clear()


def _distinct_orders(K: np.ndarray):
    """The set of series orders present.

    `np.unique` sorts; nothing here needs sorted output, because each output
    element belongs to exactly one order and is written exactly once, so the
    visiting order is unobservable.  Measured 0.6 us against 5.1 us at N = 30.
    """
    return set(K.tolist()) if K.size <= 4096 else set(np.unique(K).tolist())


def _residue(c: _Regime, x2, a0, inv2g):
    """The pointing-error residue D * C(xi^2, gbar) of eq. (19).

    It carries no series order, so it is the same expression for every rung of
    the fidelity ladder.  `pe_series_f64` recomputes it inside each group; here
    it is evaluated once for the whole call and the groups index into it.  The
    expression, its association and its operands are unchanged -- only the
    length of the arrays the ufuncs run over.

    Gamma poles at xi^2 = A+k, B+k give NaN, which the range test then rejects;
    that is intended, and the errstate scope is the same one `pe_series_f64`
    puts around this block.
    """
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # Gamma(A - xi^2) and Gamma(B - xi^2) in one call, rows contiguous
        gAB = sp_gamma(c.ABv - x2)
        D = (x2 * c.AB ** x2 * gAB[0] * gAB[1]
             / (a0 ** x2 * c.gA * c.gB))
        # c_moment(xi^2, gbar), inlined: sqrt(pi) and 2/gbar are loop-invariant
        D *= sp_gamma((x2 + 1.0) / 2.0) / (2.0 * x2 * _SQRT_PI) * inv2g ** (x2 / 2.0)
    return D


def _series(c: _Regime, x2, a0, cm):
    """The two hypergeometric families of eq. (19), summed, in one pass.

    rtodt_fast, per family:
        t = kc[None,:] * x2[:,None] / ((x2[:,None] - B - k[None,:])
                                       * a0[:,None] ** (B+k)[None,:])
        (t * C[None,:]).sum(1)
    Here the alpha<->beta pair is carried on a leading axis of length 2, so each
    ufunc runs once instead of twice.  The reduction is still taken over the
    trailing, contiguous, length-(K+1) axis, one row per candidate per family,
    exactly as before -- the pairwise summation therefore blocks identically and
    the rounding is unchanged.
    """
    X2 = x2[None, :, None]
    den = (X2 - c.BA) - c.kk
    den *= a0[None, :, None] ** c.BAk
    t = (c.KC * X2) / den
    t *= cm
    s = t.sum(2)                                   # (2, m)
    return s[0] + s[1]


def pe_series_batch(A: float, B: float, xi, A0, gbar: float, K):
    """Per-branch ABER, eq. (19).  Drop-in for `rtodt_fast.pe_series_f64`.

    `K` may be a scalar or a per-candidate array (the fidelity ladder); entries
    with K < 0 are inadmissible and return NaN.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    A0 = np.atleast_1d(np.asarray(A0, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=int))
    shape = xi.shape
    if K.size == 1:
        K = np.full(shape, int(K[0]))
    # the reference reaches 1-D through its boolean masking; flat views are free
    xi, A0, K = xi.ravel(), A0.ravel(), K.ravel()

    present = _distinct_orders(K)
    orders = [o for o in present if o >= 0]       # K < 0 is the ladder's refusal
    if not orders:
        return np.full(shape, np.nan)             # every candidate inadmissible

    c0 = _regime(A, B, orders[0])
    x2 = xi * xi
    inv2g = 2.0 / gbar

    # The residue carries no series order, so it is the term every rung shares:
    # evaluate it once for the whole call and let the groups add into it.  Only
    # the candidates the ladder admitted are evaluated -- the rest stay NaN, as
    # in the reference, and evaluating them would be wasted work.
    if len(orders) == len(present):
        out = _residue(c0, x2, A0, inv2g)
        if len(orders) == 1:
            out += _series(c0, x2, A0, _moments(A, B, orders[0], gbar, c0))
            return out.reshape(shape)
    else:
        adm = K >= 0
        out = np.full(K.shape, np.nan)
        out[adm] = _residue(c0, x2[adm], A0[adm], inv2g)

    for order in orders:
        m = K == order
        c = _regime(A, B, order)
        out[m] = _series(c, x2[m], A0[m], _moments(A, B, order, gbar, c)) + out[m]
    return out.reshape(shape)


def pe_series_stage_batch(A: float, B: float, xi, A0, gbar_stage, K, stage_index):
    """Eq. (19) with a per-candidate reference SNR, drawn from a small table.

    `gbar_stage` is the table of distinct reference SNRs (in `_objective` it is
    the per-horizon-stage gbar, length T) and `stage_index` says which entry each
    candidate uses.  This exists so the caller can replace

        for g in np.unique(gb):  pe[m] = pe_series_f64(..., float(g), K[m])

    -- 20 calls of 30 candidates, 45 (gbar, order) groups measured -- by one call
    with at most one group per distinct order.  The moment rows are still the
    reference's own per-(gbar, order) vectors, taken from the same cache; they are
    gathered per candidate instead of being recomputed per group, so every
    candidate is multiplied by exactly the numbers `pe_series_f64` would have
    multiplied it by.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    A0 = np.atleast_1d(np.asarray(A0, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=int))
    shape = xi.shape
    if K.size == 1:
        K = np.full(shape, int(K[0]))
    xi, A0, K = xi.ravel(), A0.ravel(), K.ravel()
    g = np.asarray(gbar_stage, dtype=float).ravel()
    sidx = np.asarray(stage_index).ravel()
    inv2g_tab = 2.0 / g

    present = _distinct_orders(K)
    orders = [o for o in present if o >= 0]
    if not orders:
        return np.full(shape, np.nan)

    x2 = xi * xi
    c0 = _regime(A, B, orders[0])
    if len(orders) == len(present):
        out = _residue(c0, x2, A0, inv2g_tab[sidx])     # order-free: once
    else:
        adm = K >= 0
        out = np.full(K.shape, np.nan)
        out[adm] = _residue(c0, x2[adm], A0[adm], inv2g_tab[sidx[adm]])

    for order in orders:
        m = K == order
        si = sidx[m]
        c = _regime(A, B, order)
        # per-(gbar, order) moment rows, taken from the same cache the scalar
        # path uses, then gathered per candidate: (nu, 2, K+1) -> (2, m, K+1)
        tab = np.stack([_moments(A, B, order, float(gv), c)[:, 0, :] for gv in g])
        cm = np.ascontiguousarray(np.transpose(tab[si], (1, 0, 2)))
        out[m] = _series(c, x2[m], A0[m], cm) + out[m]
    return out.reshape(shape)


# ---------------------------------------------------------------- self-check
def _cases(rng):
    """Argument sets spanning the admissible box, the three regimes, every rung
    of the ladder, and the array shapes the objective actually passes."""
    from channel import beam_geometry
    from hclpso_ga import ladder_order
    from rtodt_fast import z_of

    REG = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
    for name, (A, B) in REG.items():
        for gdb in (26.0, 30.0, 34.0, 38.0, 42.0, 46.0):
            gbar = 10.0 ** (gdb / 10.0)
            for N in (1, 2, 7, 30, 31, 200, 600, 4001):
                w = rng.uniform(0.0549, 3.0, N)
                A0, weq = beam_geometry(w)
                x = weq / (2.0 * 0.10)
                bad = ~np.isfinite(A0)
                A0 = np.where(bad, 0.5, A0)
                x = np.where(bad, 1.0, x)
                z = z_of(A, B, A0, gbar)
                for tag, K in (("ladder", ladder_order(z)),
                               ("K=5", 5), ("K=10", 10), ("K=20", 20),
                               ("mixed", rng.choice([-1, 5, 10, 20], N)),
                               ("all -1", np.full(N, -1)),
                               ("one rung + -1",
                                np.where(rng.random(N) < 0.3, -1, 10))):
                    yield name, gdb, N, tag, A, B, x, A0, gbar, K


def _self_check(verbose=True):
    """Measure this module against `rtodt_fast.pe_series_f64`.

    Not a tolerance assertion.  It reports (a) the worst relative difference and
    (b) the fraction of returned doubles that are bit-for-bit equal, over the
    admissible box in all three regimes, at every rung of the ladder, on the
    mixed-order arrays the fidelity ladder actually produces, and on the
    all-inadmissible and 2-D edge cases.
    """
    from channel import beam_geometry
    from hclpso_ga import ladder_order
    from rtodt_fast import pe_series_f64, z_of

    rng = np.random.default_rng(12345)
    worst, nbits, ntot, ncase = 0.0, 0, 0, 0
    offenders = []

    for name, gdb, N, tag, A, B, x, A0, gbar, K in _cases(rng):
        r = pe_series_f64(A, B, x, A0, gbar, K)
        t = pe_series_batch(A, B, x, A0, gbar, K)
        ncase += 1
        if r.shape != t.shape:
            offenders.append((name, gdb, N, tag, float("inf")))
            continue
        same = (r.view(np.uint64) == t.view(np.uint64)) | (np.isnan(r) & np.isnan(t))
        nbits += int(np.sum(same))
        ntot += r.size
        f = np.isfinite(r) & np.isfinite(t) & (r != 0.0)
        d = float(np.max(np.abs(t[f] - r[f]) / np.abs(r[f]))) if f.any() else 0.0
        if d > 0.0 or not same.all():
            offenders.append((name, gdb, N, tag, d))
        worst = max(worst, d)

    # 2-D input: the reference returns an array of xi's shape
    A, B, gbar = 1.2, 1.1, 10.0 ** 3.8
    w = rng.uniform(0.0549, 3.0, (30, 20))
    A0, weq = beam_geometry(w)
    x = weq / 0.20
    K = ladder_order(z_of(A, B, A0, gbar))
    r = pe_series_f64(A, B, x, A0, gbar, K)
    t = pe_series_batch(A, B, x, A0, gbar, K)
    same2d = (r.shape == t.shape
              and bool(((r.view(np.uint64) == t.view(np.uint64))
                        | (np.isnan(r) & np.isnan(t))).all()))
    nbits += int(same2d) * r.size
    ntot += r.size
    ncase += 1

    # the stagewise entry point, against the per-gbar loop it is meant to replace
    T, n = 20, 30
    stage_ok = True
    for A, B in ((4.2, 3.0), (2.1, 1.5), (1.2, 1.1)):
        g0 = 10.0 ** 3.8
        gstage = g0 * (1.0 + 0.35 * 0.98 ** np.arange(1, T + 1)) ** 2
        w = rng.uniform(0.0549, 3.0, n * T)
        A0, weq = beam_geometry(w)
        x = weq / 0.20
        sidx = np.tile(np.arange(T), n)
        Kall = np.empty(n * T, dtype=int)
        ref = np.empty(n * T)
        for j in range(T):
            m = sidx == j
            Kall[m] = ladder_order(z_of(A, B, A0[m], float(gstage[j])))
            ref[m] = pe_series_f64(A, B, x[m], A0[m], float(gstage[j]), Kall[m])
        got = pe_series_stage_batch(A, B, x, A0, gstage, Kall, sidx)
        same = ((ref.view(np.uint64) == got.view(np.uint64))
                | (np.isnan(ref) & np.isnan(got)))
        stage_ok &= bool(same.all())
        nbits += int(np.sum(same))
        ntot += ref.size
        ncase += 1
        f = np.isfinite(ref) & np.isfinite(got) & (ref != 0.0)
        if f.any():
            worst = max(worst, float(np.max(np.abs(got[f] - ref[f]) / np.abs(ref[f]))))

    if verbose:
        print("rtodt_batch.py vs rtodt_fast.pe_series_f64, over the admissible box")
        print("  argument sets compared      : %d" % ncase)
        print("  doubles compared            : %d" % ntot)
        print("  sets that differed anywhere : %d" % len(offenders))
        for o in offenders[:20]:
            print("    %-9s gbar=%.0fdB N=%-5d %-14s %.3e" % o)
        print("  2-D input agrees            : %s" % same2d)
        print("  stagewise agrees            : %s" % stage_ok)
        print("")
        print("  maximum relative difference : %.3e" % worst)
        print("  bit-identical fraction      : %.6f (%d / %d)"
              % (nbits / ntot, nbits, ntot))
        print("  float64 floor eta_f64 the deployed kernel already carries: ~7.9e-10")
    return worst, nbits / ntot


def _promotion_check(verbose=True):
    """What it would cost to collapse the ladder's group loop by promotion.

    Evaluating every candidate at max(K) would leave one group instead of three.
    It is not in scope -- it changes the series order the ladder selected -- and
    this measures by how much, so the rejection is a number rather than an
    assertion.
    """
    from channel import beam_geometry
    from hclpso_ga import ladder_order
    from rtodt_fast import pe_series_f64, z_of

    rng = np.random.default_rng(7)
    worst = 0.0
    rows = []
    for name, (A, B) in (("weak", (4.2, 3.0)), ("moderate", (2.1, 1.5)),
                         ("strong", (1.2, 1.1))):
        for gdb in (30.0, 38.0, 46.0):
            gbar = 10.0 ** (gdb / 10.0)
            w = rng.uniform(0.0549, 3.0, 4000)
            A0, weq = beam_geometry(w)
            x = weq / 0.20
            ok = np.isfinite(A0)
            A0, x = A0[ok], x[ok]
            K = ladder_order(z_of(A, B, A0, gbar))
            live = K >= 0
            A0, x, K = A0[live], x[live], K[live]
            if K.size == 0 or K.max() == K.min():
                continue
            kmax = int(K.max())
            ref = pe_series_f64(A, B, x, A0, gbar, K)
            pro = pe_series_f64(A, B, x, A0, gbar, np.full(K.shape, kmax))
            sel = (K < kmax) & np.isfinite(ref) & np.isfinite(pro) & (ref != 0.0)
            if not sel.any():
                continue
            rel = np.abs(pro[sel] - ref[sel]) / np.abs(ref[sel])
            worst = max(worst, float(rel.max()))
            rows.append((name, gdb, int(sel.sum()), float(np.median(rel)),
                         float(rel.max())))
    if verbose:
        print("promotion to max(K): relative change in the returned Pe")
        print("  %-9s %-8s %-10s %14s %14s"
              % ("regime", "gbar dB", "n promoted", "median rel", "max rel"))
        for r in rows:
            print("  %-9s %-8.0f %-10d %14.3e %14.3e" % r)
        print("  worst : %.3e   against the float64 floor 7.9e-10 -- %.0fx over"
              % (worst, worst / 7.9e-10))
        print("  NOT APPLIED: it changes the order the fidelity ladder selected.")
    return worst


def _timing(verbose=True):
    """Time this module against `rtodt_fast` on the shapes `_objective` passes.

    Timings on this machine are noisy enough that a median over independent
    calls is not monotonic in the work done.  The two candidates are therefore
    timed alternately inside one loop -- so any drift in clock or load hits both
    -- and the figure reported is the minimum over batches, the sample least
    contaminated by preemption.  The median is printed beside it so the spread
    stays visible.
    """
    import time

    from channel import beam_geometry
    from hclpso_ga import ladder_order
    from rtodt_fast import pe_series_f64, z_of

    def pair(fa, fb, inner, batches=120, warm=200):
        for _ in range(warm):
            fa()
            fb()
        ta, tb = np.empty(batches), np.empty(batches)
        for j in range(batches):
            t0 = time.perf_counter_ns()
            for _ in range(inner):
                fa()
            t1 = time.perf_counter_ns()
            for _ in range(inner):
                fb()
            t2 = time.perf_counter_ns()
            ta[j] = (t1 - t0) / inner / 1e3
            tb[j] = (t2 - t1) / inner / 1e3
        return float(ta.min()), float(np.median(ta)), float(tb.min()), float(np.median(tb))

    A, B, gbar = 1.2, 1.1, 10.0 ** 3.8
    rng = np.random.default_rng(3)
    print("us per call, min / median, interleaved")
    print("  %-38s %16s %18s" % ("", "rtodt_fast", "rtodt_batch"))
    for N in (30, 200, 600):
        w = rng.uniform(0.0549, 3.0, N)
        A0, weq = beam_geometry(w)
        x = weq / 0.20
        ok = np.isfinite(A0)
        A0, x = np.where(ok, A0, 0.5), np.where(ok, x, 1.0)
        Kl = ladder_order(z_of(A, B, A0, gbar))
        cases = (("ladder, %d rungs" % len(set(Kl.tolist())), Kl),
                 ("K=5", 5), ("K=10", 10), ("K=20", 20))
        for tag, K in cases:
            am, ad, bm, bd = pair(
                lambda K=K: pe_series_f64(A, B, x, A0, gbar, K),
                lambda K=K: pe_series_batch(A, B, x, A0, gbar, K),
                inner=40 if N == 30 else 8)
            print("  N=%-5d %-30s %7.1f /%7.1f  %7.1f /%7.1f  (%.2fx)"
                  % (N, tag, am, ad, bm, bd, am / bm))

    # the per-gbar loop `_objective` runs today, against one stagewise call
    T, n = 20, 30
    gstage = gbar * (1.0 + 0.35 * 0.98 ** np.arange(1, T + 1)) ** 2
    w = rng.uniform(0.0549, 3.0, n * T)
    A0, weq = beam_geometry(w)
    x = weq / 0.20
    ok = np.isfinite(A0)
    A0, x = np.where(ok, A0, 0.5), np.where(ok, x, 1.0)
    sidx = np.tile(np.arange(T), n)
    Kall = np.empty(n * T, dtype=int)
    for j in range(T):
        m = sidx == j
        Kall[m] = ladder_order(z_of(A, B, A0[m], float(gstage[j])))
    masks = [(float(gstage[j]), sidx == j) for j in range(T)]

    def ref_loop():
        o = np.empty(n * T)
        for gv, m in masks:
            o[m] = pe_series_f64(A, B, x[m], A0[m], gv, Kall[m])
        return o

    def stage_one():
        return pe_series_stage_batch(A, B, x, A0, gstage, Kall, sidx)

    am, ad, bm, bd = pair(ref_loop, stage_one, inner=4)
    print("")
    print("  %-38s %7.1f /%7.1f  %7.1f /%7.1f  (%.2fx)"
          % ("20 gbar groups x 30 candidates", am, ad, bm, bd, am / bm))
    print("     left: 20 calls to pe_series_f64;  right: one pe_series_stage_batch")


if __name__ == "__main__":
    _self_check()
    print("")
    _promotion_check()
    print("")
    _timing()
