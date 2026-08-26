"""
Vectorised float64 RT-ODT kernel -- the interpolation-free evaluator.

The xi-dependence of the series coefficients factorises (eq. 21):

    a_k(A,B,xi) = Kc_k(A,B) * xi^2 / ( (xi^2 - B - k) * A_0^{B+k} )

with Kc_k independent of xi.  The constants are computed once per regime in
extended precision and cached; the xi-dependent factor is then evaluated in
closed form per candidate, so no interpolation in xi occurs anywhere.

This is the evaluator the manuscript reports results on.  `rtodt.py` holds the
same expressions in arbitrary precision and is the reference against which this
module is checked.

Run `python rtodt_fast.py` to perform that check.  It does not assert a
pre-agreed figure; it samples the admissible band and prints the worst relative
disagreement it finds, so the number in the log is a measurement.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as sp_gamma

_KC_CACHE: dict = {}
_C_CACHE: dict = {}


def _kc(A: float, B: float, K: int) -> np.ndarray:
    """xi-free part of a_k, in extended precision, cached per (A, B, K)."""
    key = (round(A, 10), round(B, 10), K)
    if key in _KC_CACHE:
        return _KC_CACHE[key]
    import mpmath as mp
    with mp.workdps(120):
        a, b = mp.mpf(A), mp.mpf(B)
        out = np.array([float((-1) ** k * (a * b) ** (b + k) * mp.gamma(a - b - k)
                              / (mp.factorial(k) * mp.gamma(a) * mp.gamma(b)))
                        for k in range(K + 1)])
    _KC_CACHE[key] = out
    return out


def c_moment(s, gbar: float):
    """Power moment C(s, gbar), eq. (20)."""
    s = np.asarray(s, dtype=float)
    return sp_gamma((s + 1.0) / 2.0) / (2.0 * s * np.sqrt(np.pi)) * (2.0 / gbar) ** (s / 2.0)


def z_of(A: float, B: float, A0, gbar: float):
    """Conditioning parameter z = sqrt(2) alpha beta / (A_0 sqrt(gbar))."""
    return np.sqrt(2.0) * A * B / (np.asarray(A0, dtype=float) * np.sqrt(gbar))


def pe_series_f64(A: float, B: float, xi, A0, gbar: float, K):
    """Per-branch ABER, eq. (19), evaluated in closed form.

    `K` may be a scalar or a per-candidate array (the fidelity ladder); entries
    with K < 0 are inadmissible and return NaN.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    A0 = np.atleast_1d(np.asarray(A0, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=int))
    if K.size == 1:
        K = np.full(xi.shape, int(K[0]))

    out = np.full(xi.shape, np.nan)
    for order in np.unique(K):
        if order < 0:
            continue                      # inadmissible: left as NaN
        m = K == order
        x, a0 = xi[m], A0[m]
        x2 = x * x
        k = np.arange(order + 1)

        kcAB, kcBA = _kc(A, B, order), _kc(B, A, order)
        key = (round(A, 10), round(B, 10), order, round(gbar, 10))
        if len(_C_CACHE) > 8192:      # gbar varies per horizon stage; bound the cache
            _C_CACHE.clear()
        if key not in _C_CACHE:
            _C_CACHE[key] = (c_moment(B + k, gbar), c_moment(A + k, gbar))
        CB, CA = _C_CACHE[key]

        t1 = kcAB[None, :] * x2[:, None] / ((x2[:, None] - B - k[None, :])
                                            * a0[:, None] ** (B + k)[None, :])
        t2 = kcBA[None, :] * x2[:, None] / ((x2[:, None] - A - k[None, :])
                                            * a0[:, None] ** (A + k)[None, :])
        total = (t1 * CB[None, :]).sum(1) + (t2 * CA[None, :]).sum(1)

        # pointing-error residue (Gamma poles at xi^2 = A+k, B+k give NaN,
        # which the range test then rejects -- this is intended)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            D = (x2 * (A * B) ** x2 * sp_gamma(A - x2) * sp_gamma(B - x2)
                 / (a0 ** x2 * sp_gamma(A) * sp_gamma(B)))
            out[m] = total + D * c_moment(x2, gbar)
    return out


def _self_check(n=60, seed=0):
    """Measure this module against the arbitrary-precision rtodt.py.

    Sampled inside the band each ladder rung actually serves (z <= 0.5 for
    K=5, z <= 2 for K=10, z <= 8 for K=20), because outside its own band the
    float64 evaluation is expected to be dominated by the round-off floor
    eta_f64 of eq. (27) -- that is the property the fidelity ladder exists to
    manage, not a defect of this kernel.
    """
    import mpmath as mp
    from rtodt import REGIMES, A0_for, Pe_series, z_param, db

    rng = np.random.default_rng(seed)
    gbar = db(38.0)
    print("rtodt_fast.py vs rtodt.py (90-digit), gbar = 38 dB, sigma_s = 0.1 m")
    print("  %-9s %-6s %-6s %-14s %s" % ("regime", "K", "n", "worst rel", "at xi"))
    print("  " + "-" * 54)
    overall = 0.0
    for reg in ("weak", "moderate", "strong"):
        A, B = REGIMES[reg]
        for K, zmax in ((5, 0.5), (10, 2.0), (20, 8.0)):
            worst, at, got = 0.0, None, 0
            tries = 0
            while got < n and tries < 200 * n:
                tries += 1
                xi = mp.mpf("0.5") + mp.mpf("4.388") * mp.mpf(float(rng.random()))
                a0 = A0_for(xi, mp.mpf("0.1"))
                if a0 is None or float(z_param(A, B, a0, gbar)) > zmax:
                    continue
                got += 1
                ref = Pe_series(A, B, xi, a0, gbar, K)
                if ref == 0:
                    continue
                fast = float(pe_series_f64(float(A), float(B), float(xi),
                                           float(a0), float(gbar), K)[0])
                rel = abs(fast - float(ref)) / abs(float(ref))
                if rel > worst:
                    worst, at = rel, float(xi)
            if got:
                overall = max(overall, worst)
                print("  %-9s %-6d %-6d %-14.3e %s"
                      % (reg, K, got, worst, "%.4f" % at if at else "-"))
    print("\n  worst relative disagreement over all rungs and regimes: %.3e" % overall)
    return overall


if __name__ == "__main__":
    _self_check()
