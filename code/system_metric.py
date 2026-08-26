"""
Post-EGC SYSTEM-level ABER and the manuscript's optimization-success test.

WHY THIS MODULE EXISTS
    The campaign driver in this directory scores candidates on the PER-BRANCH
    surrogate of eq:aber_emulator.  The manuscript defines optimization success
    at POST-EGC SYSTEM level:

        "The threshold 1e-6 is a *post-EGC system* figure, evaluated through
         (eq:mimo_egc_aber) over the M x N combined channel; the per-branch
         quantity (eq:aber_emulator) is of order 1e-1 at gbar_op under strong
         turbulence and is not comparable to it."

    The two differ by many orders of magnitude, so scoring success against the
    per-branch surrogate measures the wrong quantity.  This module supplies the
    right one.

WHAT IS COMPUTED (eq:mimo_egc_aber, verbatim)

        P_e,sys = Int_0^inf  Q( sqrt( gbar / (M N) ) * H )  f_H(H)  dH

    with  H = sum_{i=1}^{MN} h_i,  MN = 4 x 4 = 16,  f_H the MN-fold convolution
    of the composite per-branch density f_h, and Q(z) = erfc(z/sqrt2)/2 the IM/DD
    OOK conditional error probability.  The 1/(MN) divisor inside the square root
    is taken exactly as printed (the manuscript never derives it, and Proposition
    1 writes the same probability WITHOUT it -- an inconsistency noted, not
    resolved, here: eq:mimo_egc_aber is the equation the success test cites, so
    eq:mimo_egc_aber is what is implemented).

    Branch statistics follow the manuscript's arithmetic: all MN branches share
    one (alpha, beta, xi, A_0) and are independent.  Neither is ever stated in
    prose; both are forced by the analytic mean E[H] = MN*A_0*xi^2/(xi^2+1) and
    by the exponent lattice n_D xi^2 + n_beta beta + n_alpha alpha + S with
    n_D+n_beta+n_alpha = MN.  Physically this is optimistic -- with a shared beam
    and shared sigma_s the pointing loss is largely common-mode across the 16
    branches -- but it is the model the manuscript specifies.

THREE EVALUATION PATHS
    'fast'  branch density by log-domain (Mellin) convolution: ln h = ln h_a +
            ln h_p is an ordinary additive convolution, the gamma-gamma factor is
            cached once per regime, and the pointing law is handled analytically.
    'quad'  branch density by egc_system.f_h_exact -- the manuscript's prescribed
            "quadrature over the pointing law", reused from the released module.
    'mc'    direct Monte Carlo over the 16-branch sum, using the fact that a
            unit-mean gamma-gamma is a product of two unit-mean gamma variates.
            Independent of every density construction above; it is the arbiter.

    Downstream of the branch density all paths use egc_system's own code
    (`convolve_MN`, `Q`), so 'fast' and 'quad' differ ONLY in how f_h is built.

A CORRECTION TO THE RELEASED REFERENCE
    egc_system.aber_system point-samples f_h on a uniform lattice starting at
    h = 0, patching the singular first sample with `h[0] = h[1]*1e-6`.  For
    xi > 1 that is harmless.  For xi < 1 -- which is inside the swept decision
    box, since xi is clipped to [max(0.5, xi_min(sigma_s)), 4.888] -- the branch
    density diverges as h^{xi^2-1} and the patched first sample injects spurious
    mass: calling the released routine at xi = 0.5 returns P_e,sys ~ 1e46, which
    is not a probability.  This module therefore represents the branch density by
    exact CELL MASSES on the lattice rather than by point samples.  That is the
    same discretisation everywhere, and it is well-defined for every xi in the box.

Run `python system_metric.py` for the self-check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

import egc_system as egc
from egc_system import MN, Q, convolve_MN, f_h_exact

# --------------------------------------------------------------------------
# Constants the manuscript SPECIFIES
# --------------------------------------------------------------------------
GBAR_OP_DB = 38.0          # reference SNR of the success test (Sec. VI-C).
                           # NOT 30 dB: 30 dB is Numerical Result 1's operating
                           # point (fallback ABER, link continuity) -- a different
                           # experiment the manuscript warns not to conflate.
ABER_TARGET = 1e-6         # success threshold, and the comparison is "<=".
APERTURE = 0.05            # a, metres (access.tex line 146)
LINK_LENGTH = 2000.0       # L, metres -- enters only r_d = L*||Theta||_2

REGIMES = {                # Table 1; (alpha, beta) are the PRIMARY specification
    "weak": (4.2, 3.0),    # -- no C_n^2 drives any simulated quantity
    "moderate": (2.1, 1.5),
    "strong": (1.2, 1.1),
}
SIGMAS = (0.05, 0.10, 0.20, 0.30)   # swept building-sway levels (Sec. VII-A)
XI_MAX = 4.888

# --------------------------------------------------------------------------
# Numerical choices the manuscript does NOT specify.  Each is fixed from an
# error bound or a convergence study -- never from a target value.
# --------------------------------------------------------------------------

_Q_TAIL_ARG = 12.0
# Grid extent.  The manuscript gives "no quadrature order, no grid extent, no
# grid size".  Fixed here by an exact truncation bound.  Every h_i >= 0, so f_H
# on [0, hmax] depends only on the branch density on [0, hmax]: a branch value
# above hmax cannot contribute to a sum below hmax.  Truncating the branch
# density at hmax therefore leaves f_H EXACT on [0, hmax] and can only misplace
# mass at H > hmax, where the integrand is bounded by Q(sqrt(gbar/MN)*hmax).
#     hmax = _Q_TAIL_ARG / sqrt(gbar/MN)
# bounds the total truncation error by Q(12) = 1.8e-33 -- twenty-seven decades
# below the 1e-6 threshold being tested.  This is what makes a system-level
# evaluation cheap enough to run per candidate: at 38 dB it needs H only out to
# 0.604, not out to E[H] = 1.6 and beyond.

_NH = 32768
# Grid resolution.  Fixed by the manuscript's own stated check -- "convergence of
# the result under fourfold grid refinement" -- carried out in this directory:
# halving dh from 1.25e-4 to 6.25e-5 moved the strong-regime system ABER by
# 0.19-0.70%, and the two further halvings by under 0.08%, so dh = 6.25e-5 is
# converged at the 0.1% level.  Because hmax scales as 1/sqrt(gbar), a FIXED
# point count holds dh/H* constant, H* = 4.75/sqrt(gbar/MN) being the H at which
# Q(.) crosses 1e-6; convergence therefore holds uniformly in gbar rather than
# only at the SNR where it was measured.  _NH = 32768 puts ~13000 lattice points
# below H* and makes MN*_NH exactly a power of two, so convolve_MN pads exactly.

_LOG_LO, _LOG_HI, _LOG_DT = -120.0, 5.0, 0.002
# Fast-path log grid for ln h_a.  e^5 = 148 covers the gamma-gamma upper tail to
# below 1e-9 survival in every regime (worst case, strong: P(h_a>60) = 2.3e-7).
# The midpoint/Simpson reconstruction is O(dt^2); dt = 0.002 reproduces the
# pointwise quadrature to ~1e-6 relative wherever that quadrature is itself
# trustworthy.
_CDF_LO = -80.0
# Bottom of the ln h axis over which the branch CDF is accumulated.  The neglected
# mass is (e^-80/A_0)^{xi^2}; at the worst admissible corner (xi = 0.5, A_0 = 0.52)
# that is 2e-9, and those branch values are ~1e-35, far below one lattice cell.

ROUNDOFF_FLOOR = 1e-16
# Double-precision floor of the MN-fold FFT convolution.  f_H is reconstructed
# from a transform whose peak is O(1), so its absolute error is ~1e-16 per
# sample; integrated against Q over the lattice this leaves an ABER noise floor
# around 1e-17..1e-16, below which the returned value stops being monotone in
# SNR and can go negative.  Measured on the validation configuration: the strong
# regime tracks smoothly down to 4.6e-20 at 55 dB and turns negative at 60 dB.
# The floor is TEN decades below the 1e-6 threshold this module tests, so the
# success predicate is never affected; a returned value below ROUNDOFF_FLOOR
# should nonetheless be read as "at or under the floor", not as a number.
# Removing it would require an exponentially tilted convolution or the
# lambda_j/C_j recursion in extended precision (eq22_recursion.py), neither of
# which the success test needs.

_NY_QUAD = 1200            # egc_system.f_h_exact's own default pointing-quadrature
                           # order.  ny = 800 already reproduces ny = 2000 to six
                           # figures at the validation configuration.
_QUAD_DT = 0.01            # log-grid step for the 'quad' path (cost is ny per node)

_PA_CACHE: dict = {}


# --------------------------------------------------------------------------
# Beam geometry (eq:hp_def)
# --------------------------------------------------------------------------
class BeamGeometryDomainError(ValueError):
    """A transmitted waist outside the domain on which eq:hp_def is a beam model.

    See `beam_geometry` for what the domain is and why leaving it silently was a
    correctness defect rather than a numerical nicety.
    """


_BRANCH_CACHE: dict = {}

# Slack on the branch boundary.  d(w_zeq)/d(w_z) vanishes at the minimum, so a
# waist a part in 1e6 below it differs from the minimum by ~1e-12 relative and
# is a rounding artefact of whichever bisection produced it, not a narrow-branch
# beam.  The beams this guard exists to reject sit 30-50% below the boundary
# (see test_beam_geometry.py), four to five orders outside this slack.
_BRANCH_RTOL = 1e-6


def branch_min_wz(a: float = APERTURE) -> Tuple[float, float]:
    """(w_z*, w_zeq*) at the interior minimum of the equivalent width.

    The manuscript computes this itself -- "the Farid-Hranilovic equivalent
    width w_{z,eq} is *non-monotonic* in beam waist w_z (minimum 0.0877 m at
    w_z = 0.0549 m for a = 0.05 m)" (Sec. VII-A) -- and uses it to set the
    attainable floor xi_min(sigma_s) = 0.0877/(2 sigma_s) of the decision box.
    It is recomputed here from the formula rather than pasted in.
    """
    key = float(a)
    if key not in _BRANCH_CACHE:
        f = lambda w: _raw_geometry(w, a)[1]
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 1e-4 * (a / APERTURE), 20.0 * (a / APERTURE)
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        for _ in range(400):
            if f(c) < f(d):
                hi = d
            else:
                lo = c
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        w = 0.5 * (lo + hi)
        _BRANCH_CACHE[key] = (float(w), float(f(w)))
    return _BRANCH_CACHE[key]


def _raw_geometry(w_z, a: float = APERTURE):
    """eq:hp_def with NO domain guard.  Internal; used to locate the branch
    boundary itself, which necessarily requires evaluating across it."""
    from scipy.special import erf
    w = np.asarray(w_z, float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        v = np.sqrt(np.pi / 2.0) * a / w
        A0 = erf(v) ** 2
        w_zeq = np.sqrt(w ** 2 * np.sqrt(np.pi) * erf(v)
                        / (2.0 * v * np.exp(-v ** 2)))
    return A0, w_zeq


def beam_geometry_valid(w_z, a: float = APERTURE):
    """Boolean (elementwise) domain test for `beam_geometry`."""
    w = np.asarray(w_z, float)
    return (np.isfinite(w) & (w > 0.0)
            & (w >= branch_min_wz(a)[0] * (1.0 - _BRANCH_RTOL)))


def beam_geometry(w_z: float, a: float = APERTURE,
                  check_domain: bool = True) -> Tuple[float, float]:
    """(A_0, w_zeq) for a transmitted waist w_z -- eq:hp_def, as in channel.py.

    DOMAIN, AND WHY IT IS ENFORCED
        w_{z,eq}(w_z) is non-monotonic with an interior minimum at
        w_z* = 0.054869 m, w_zeq* = 0.087719 m (a = 0.05 m).  Only the branch
        w_z >= w_z* is a beam model.  Below it the map *inverts*: narrowing the
        beam raises w_{z,eq} -- and therefore raises xi = w_{z,eq}/(2 sigma_s),
        the beam-to-jitter ratio -- while A_0 rises towards 1.  The physical
        trade-off the whole optimization exists to resolve, more collected power
        bought with more pointing sensitivity, runs backwards there: the narrow
        branch offers unlimited xi at A_0 -> 1, for free.

        That is not a mild extrapolation error, it is the model being read
        outside the regime it was derived in.  The Farid-Hranilovic equivalent
        width is a Gaussian fit to the collected-power profile of a beam that is
        wide compared with the aperture; at w_z < w_z* the beam is *narrower*
        than a = 0.05 m and there is no such profile to fit.

        Left unguarded this reached the success predicate.  At sigma_s = 0.10 m
        the narrow-branch waist w_z = 0.0286173 m returns xi = 1.0000 with
        A_0 = 0.9961 and P_e,sys = 1.22e-14, so `success()` returned True.  The
        real beam at that xi is the broad-branch w_z = 0.1930499 m, which has
        A_0 = 0.1252 and P_e,sys = 1.42e-06 and FAILS the same test.  A success
        rate computed without this guard can be inflated by configurations that
        do not exist.

        The boundary is not a tuning knob.  It is the same minimum the
        manuscript computes in Sec. VII-A to define xi_min(sigma_s), so the
        guard coincides exactly with the lower edge of the manuscript's own
        decision box: `mpc_loop.manuscript_wz_box(0.05)` starts at w_z =
        0.054869 m, the boundary itself.  The guard therefore excludes nothing
        the manuscript admits -- see test_beam_geometry.py, which checks that at
        every swept sigma_s.

    `check_domain=False` returns the raw two-branch formula, for the code that
    has to evaluate across the boundary in order to locate it (`branch_min_wz`,
    `channel.xi_floor`, `mpc_loop.wz_for_xi`).  Nothing that scores a beam may
    use it.
    """
    if check_domain and not bool(np.all(beam_geometry_valid(w_z, a))):
        w_star, weq_star = branch_min_wz(a)
        raise BeamGeometryDomainError(
            "w_z = %r is outside the beam-broadening branch (w_z >= %.9f m for "
            "a = %.4f m, where w_zeq attains its minimum %.9f m). Below it "
            "eq:hp_def inverts -- A_0 -> 1 with w_zeq, hence xi, INCREASING -- "
            "and the value returned is not a beam. Pass check_domain=False only "
            "to locate the branch boundary, never to score a candidate."
            % (w_z, w_star, a, weq_star))
    A0, w_zeq = _raw_geometry(w_z, a)
    return float(A0), float(w_zeq)


def xi_effective(xi: float, r_d: float, sigma_s: float) -> float:
    """eq:xi_eff -- residual boresight r_d folded into sigma_s,eff."""
    return xi / np.sqrt(1.0 + r_d ** 2 / (2.0 * sigma_s ** 2))


@dataclass(frozen=True)
class BeamConfig:
    """One beam configuration, as the success test sees it.

    `w_z` (transmitted waist, m) and `sigma_s` (building sway, m) are the physical
    decision; xi and A_0 follow.  `r_d` is the residual radial pointing offset at
    the receiver plane, r_d = L*||Theta||_2 (eq:rd_vector).

    r_d defaults to 0.  The manuscript derives the whole r_d -> xi_eff machinery
    but gives no initial condition or distribution for Theta(0), so no non-zero
    default is justifiable from the text; 0 is a statement about the model rather
    than an invention.
    """
    regime: str
    w_z: float
    sigma_s: float
    r_d: float = 0.0

    @property
    def alpha_beta(self) -> Tuple[float, float]:
        return REGIMES[self.regime]

    @property
    def A0(self) -> float:
        return beam_geometry(self.w_z)[0]

    @property
    def xi(self) -> float:
        return beam_geometry(self.w_z)[1] / (2.0 * self.sigma_s)

    @property
    def xi_eff(self) -> float:
        # "It is xi_eff, not nominal xi, that feeds the RT-ODT coefficients."
        return xi_effective(self.xi, self.r_d, self.sigma_s)

    @classmethod
    def from_xi(cls, regime: str, xi: float, sigma_s: float,
                r_d: float = 0.0) -> "BeamConfig":
        """Invert xi = w_zeq/(2 sigma_s) on the beam-broadening (upper) branch."""
        from scipy.optimize import brentq
        target = 2.0 * sigma_s * xi
        # w_zeq(w_z) is non-monotonic with a minimum 0.0877 m at w_z = 0.0549 m;
        # the physical branch is the one above it, where A_0 decreases with xi.
        # The lower bracket is that minimum, from branch_min_wz(). It used to be
        # the literal 0.05491 -- 7e-4 ABOVE the true boundary 0.054869382, so
        # the exact box floor xi_min(sigma_s) was reported unattainable.
        lo, weq_lo = branch_min_wz()
        if weq_lo > target * (1.0 + 1e-12):
            raise ValueError("xi=%.4f unattainable at sigma_s=%.3f m (below "
                             "xi_min=%.6f)"
                             % (xi, sigma_s, weq_lo / (2.0 * sigma_s)))
        w = brentq(lambda w: beam_geometry(w)[1] - target, lo, 200.0,
                   xtol=1e-14, rtol=8.9e-16)
        return cls(regime, float(w), sigma_s, r_d)


# --------------------------------------------------------------------------
# Branch density on the ln h axis
# --------------------------------------------------------------------------
def _log_pa(alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Density of ln(h_a) on the shared log grid; cached per turbulence regime.

    p(t) = f_ha(e^t) e^t, f_ha the unit-mean gamma-gamma of egc_system.gg_pdf.
    """
    key = (round(alpha, 12), round(beta, 12))
    if key not in _PA_CACHE:
        t = np.arange(_LOG_LO, _LOG_HI + 0.5 * _LOG_DT, _LOG_DT)
        x = np.exp(t)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            p = egc.gg_pdf(x, alpha, beta) * x
        _PA_CACHE[key] = (t, np.where(np.isfinite(p), p, 0.0))
    return _PA_CACHE[key]


def _p_lnh_fast(alpha: float, beta: float, xi: float,
                A0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Density of ln h for the composite branch gain h = h_a * h_p.

    h is a PRODUCT of independent factors, so ln h is a SUM and its density is an
    ordinary additive convolution:  p_lnh = p_ln_ha * p_ln_hp.

    p_ln_ha is one cached vector of gamma-gamma evaluations -- it does not depend
    on the beam.  p_ln_hp is analytic: h_p has CDF (h_p/A_0)^{xi^2} on [0, A_0],
    so ln h_p has CDF exp(xi^2 (t - ln A_0)) on (-inf, ln A_0].  Its EXACT
    per-cell masses are used, which removes the jump discontinuity at t = ln A_0
    that point-sampling would smear, and removes the h -> 0 singularity that
    breaks a linear-grid evaluation for xi < 1.

    Cost: one cached Bessel vector plus one FFT, against the nh x ny Bessel
    evaluations f_h_exact performs per configuration.
    """
    t, pa = _log_pa(alpha, beta)
    lA0 = np.log(A0)
    x2 = xi * xi
    hi = np.minimum(t + 0.5 * _LOG_DT, lA0)
    lo = np.minimum(t - 0.5 * _LOG_DT, lA0)
    w = np.exp(x2 * (hi - lA0)) - np.exp(x2 * (lo - lA0))      # exact cell masses
    s = np.maximum(fftconvolve(pa, w), 0.0)
    grid = 2.0 * _LOG_LO + np.arange(s.size) * _LOG_DT
    keep = grid >= _CDF_LO
    return grid[keep], s[keep]


def _p_lnh_quad(alpha: float, beta: float, xi: float, A0: float,
                t_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Same density, from egc_system.f_h_exact -- the manuscript's prescribed
    "quadrature over the pointing law", evaluated on the ln h axis.

    NOTE: the integrand of f_h_exact carries y^{xi^2-2} over y in [0, A_0], which
    is non-integrably singular at y = 0 for xi^2 <= 1 and only mildly better just
    above it.  Fixed-order Gauss-Legendre on [0, A_0] therefore loses accuracy as
    xi falls towards the bottom of the decision box.  This path is kept because it
    is the released reference, not because it is the more accurate one; the 'mc'
    path arbitrates.
    """
    t = np.arange(_CDF_LO, t_hi + 0.5 * _QUAD_DT, _QUAD_DT)
    h = np.exp(t)
    p = np.empty_like(h)
    for i in range(0, h.size, 4096):          # chunked only to bound peak memory
        p[i:i + 4096] = f_h_exact(h[i:i + 4096], alpha, beta, xi, A0, ny=_NY_QUAD)
    p = np.where(np.isfinite(p), p, 0.0) * h
    return t, np.maximum(p, 0.0)


def f_h(h, alpha: float, beta: float, xi: float, A0: float,
        method: str = "fast") -> np.ndarray:
    """Composite branch density f_h(h) = p_lnh(ln h)/h.  Convenience/diagnostic."""
    h = np.asarray(h, dtype=float)
    if method == "quad":
        return f_h_exact(h, alpha, beta, xi, A0, ny=_NY_QUAD)
    t, p = _p_lnh_fast(alpha, beta, xi, A0)
    good = p > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(np.interp(np.log(h), t[good], np.log(p[good]),
                               left=-np.inf, right=-np.inf)) / h
    return np.where(np.isfinite(out), out, 0.0)


# --------------------------------------------------------------------------
# Lattice cell masses
# --------------------------------------------------------------------------
def _cell_masses(t: np.ndarray, p: np.ndarray, dh: float, nh: int) -> np.ndarray:
    """Exact-as-possible mass of the branch gain in each lattice cell.

    Lattice point i sits at h = i*dh and owns the cell
        [max(0, (i-0.5)dh), (i+0.5)dh].
    Cell 0 is the half cell [0, dh/2]; its mass is read off the cumulative of
    p_lnh, which is finite for every xi > 0 even though f_h itself is not.
    Cells i >= 1 use Simpson's rule on the ln h axis with log-linear interpolation
    of p; their log-width ln(1+1/i) shrinks like 1/i, so this is far finer than
    the underlying grid step for all but the first few cells.

    Returns masses (length nh).  They sum to less than 1 by construction: the
    branch tail above hmax is deliberately truncated (see _Q_TAIL_ARG).
    """
    good = p > 0.0
    lt, lp = t[good], np.log(p[good])
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))))

    edges = (np.arange(nh + 1) - 0.5) * dh
    edges[0] = 0.0
    with np.errstate(divide="ignore"):
        le = np.log(edges)

    m = np.empty(nh)
    m[0] = float(np.interp(le[1], t, cum))                  # mass on [0, dh/2]

    a, b = le[1:-1], le[2:]
    mid = 0.5 * (a + b)

    def pv(x):
        return np.exp(np.interp(x, lt, lp, left=-np.inf, right=-np.inf))

    m[1:] = (b - a) * (pv(a) + 4.0 * pv(mid) + pv(b)) / 6.0
    return np.where(np.isfinite(m) & (m > 0.0), m, 0.0)


# --------------------------------------------------------------------------
# eq:mimo_egc_aber
# --------------------------------------------------------------------------
def system_aber(alpha: float, beta: float, xi: float, A0: float, gbar: float,
                method: str = "fast", nh: int = _NH,
                hmax: Optional[float] = None,
                mc_samples: int = 4_000_000, mc_seed: int = 0,
                return_floor: bool = False):
    """Post-EGC system ABER, eq:mimo_egc_aber.  `gbar` is LINEAR, not dB.

    method='fast' | 'quad' | 'mc' | 'egc'
        'egc' calls egc_system.aber_system verbatim, so the released routine's own
        answer can be seen next to the corrected one.  It is unreliable for xi < 1
        (see the module docstring).

    return_floor=True returns `(aber, floor)` instead of `aber`, for the 'fast'
    and 'quad' paths only.  `floor` is a MEASURED per-row estimate of the
    round-off noise of this particular convolution, not the module-wide
    constant ROUNDOFF_FLOOR: a probability density cannot be negative, so every
    negative sample of the reconstructed f_H is pure FFT round-off, and
    integrating its absolute value against the same Q weight measures that
    noise in the same units as the answer.  A returned `aber` that is not
    comfortably above its own `floor` is round-off, not a small probability --
    and because `aber` is clamped at 0, an exact 0.00000000e+00 means "under
    the floor", never "zero".  The clamp is why the floor has to be reported
    separately: after clamping, the value itself no longer shows that it failed.
    """
    c = np.sqrt(gbar / MN)
    if hmax is None:
        hmax = _Q_TAIL_ARG / c

    if method == "mc":
        if return_floor:
            raise ValueError("return_floor is defined for 'fast' and 'quad' "
                             "only; the MC path has a sampling error, not an "
                             "FFT round-off floor")
        return _system_aber_mc(alpha, beta, xi, A0, c, mc_samples, mc_seed)

    if method == "egc":
        if return_floor:
            v, _, _, fl = egc.aber_system(f_h_exact, hmax, nh, gbar, alpha,
                                          beta, xi, A0, which="exact",
                                          return_floor=True)
            return v, fl
        return egc.aber_system(f_h_exact, hmax, nh, gbar, alpha, beta, xi, A0,
                               which="exact")[0]

    dh = hmax / (nh - 1)
    if method == "fast":
        t, p = _p_lnh_fast(alpha, beta, xi, A0)
    elif method == "quad":
        t, p = _p_lnh_quad(alpha, beta, xi, A0, np.log(hmax) + 0.05)
    else:
        raise ValueError("method must be 'fast', 'quad', 'mc' or 'egc'")

    mass = _cell_masses(t, p, dh, nh)
    fH = convolve_MN(mass / dh, dh)          # single FFT, MN-fold -- egc_system
    H = np.arange(fH.size) * dh
    w = Q(c * H)
    # fH*dh is a lattice probability mass, so the ABER is the discrete
    # expectation sum_i Q(c H_i) p_i -- not a trapezoid, which would halve the
    # H = 0 cell.
    val = float(np.sum(w * fH) * dh)
    if not return_floor:
        return max(val, 0.0)          # see ROUNDOFF_FLOOR
    floor = float(np.sum(w * np.abs(np.minimum(fH, 0.0))) * dh)
    return max(val, 0.0), floor


def _system_aber_mc(alpha: float, beta: float, xi: float, A0: float,
                    c: float, n: int, seed: int) -> float:
    """Independent Monte Carlo over the MN-branch sum.

    A unit-mean gamma-gamma is the product of two independent unit-mean gamma
    variates with shapes alpha and beta, and h_p = A_0 U^{1/xi^2} with U uniform,
    so H needs no density construction at all.  This checks the whole chain --
    branch model, MN-fold combining, and the 1/(MN) SNR scaling -- at once.
    """
    rng = np.random.default_rng(seed)
    acc, done, blk = 0.0, 0, 200_000
    while done < n:
        k = min(blk, n - done)
        ha = (rng.gamma(alpha, 1.0 / alpha, (k, MN))
              * rng.gamma(beta, 1.0 / beta, (k, MN)))
        hp = A0 * rng.random((k, MN)) ** (1.0 / xi ** 2)
        acc += float(np.sum(Q(c * np.sum(ha * hp, axis=1))))
        done += k
    return acc / n


def aber_of(cfg: BeamConfig, gbar_db: float = GBAR_OP_DB,
            method: str = "fast") -> float:
    """Post-EGC system ABER of a beam configuration at a given reference SNR."""
    a, b = cfg.alpha_beta
    return system_aber(a, b, cfg.xi_eff, cfg.A0, 10.0 ** (gbar_db / 10.0),
                       method=method)


# --------------------------------------------------------------------------
# The success test
# --------------------------------------------------------------------------
def success(beam_config: BeamConfig, gbar_db: float = GBAR_OP_DB,
            target: float = ABER_TARGET, method: str = "fast") -> bool:
    """The manuscript's optimization-success criterion, exactly as defined.

        "a beam configuration satisfying  P_e_bar <= 1e-6  at the fixed reference
         SNR gbar_op = 38 dB"                                       (Sec. VI-C)

    with P_e_bar the POST-EGC SYSTEM ABER of eq:mimo_egc_aber over the 4x4
    combined channel -- not the per-branch surrogate the solver ranks by.  The
    comparison is "<=", as printed, and gbar_op is the same 38 dB for every
    algorithm row of Table 9 and every ablated variant of Table 11.

    This is the per-configuration predicate only.  Turning it into a success RATE
    additionally requires the campaign protocol -- what one trial redraws, which
    sigma_s the campaign ran at, whether a "realization" is one MPC cycle or a
    multi-cycle closed-loop run -- none of which the manuscript specifies.  Those
    are left to the caller rather than invented here.
    """
    return aber_of(beam_config, gbar_db=gbar_db, method=method) <= target


# --------------------------------------------------------------------------
# Verification helpers -- the manuscript's own two checks, plus moments
# --------------------------------------------------------------------------
def verify_density(alpha: float, beta: float, xi: float, A0: float,
                   method: str = "fast", h_a_max: float = 80.0,
                   nh: int = 200_000) -> Tuple[float, float, float]:
    """Recover unit mass and the analytic mean E[H] = MN*A_0*xi^2/(xi^2+1).

    Run on the FULL branch support (hmax = A_0 * h_a_max), unlike `system_aber`,
    which truncates deliberately because the ABER is a lower-tail functional.
    Returns (mass, mean, analytic_mean).
    """
    hmax = A0 * h_a_max
    dh = hmax / (nh - 1)
    if method == "quad":
        t, p = _p_lnh_quad(alpha, beta, xi, A0, np.log(hmax) + 0.05)
    else:
        t, p = _p_lnh_fast(alpha, beta, xi, A0)
    fH = convolve_MN(_cell_masses(t, p, dh, nh) / dh, dh)
    H = np.arange(fH.size) * dh
    return (float(np.sum(fH) * dh), float(np.sum(H * fH) * dh),
            MN * A0 * xi ** 2 / (xi ** 2 + 1.0))


def branch_moments(alpha: float, beta: float, xi: float, A0: float,
                   method: str = "fast", orders=(0, 1, 2, 3)) -> list:
    """Numeric vs analytic raw moments of ONE branch gain h = h_a h_p.

        E[h^m] = [Gamma(a+m)Gamma(b+m)/(Gamma(a)Gamma(b)(ab)^m)] * A_0^m xi^2/(xi^2+m)

    An exact, closed-form check on the branch density independent of the mean
    identity the manuscript quotes (which only probes m = 1).
    """
    from scipy.special import gamma as G
    hmax = A0 * 80.0
    nh = 200_000
    dh = hmax / (nh - 1)
    if method == "quad":
        t, p = _p_lnh_quad(alpha, beta, xi, A0, np.log(hmax) + 0.05)
    else:
        t, p = _p_lnh_fast(alpha, beta, xi, A0)
    m = _cell_masses(t, p, dh, nh)
    h = np.arange(nh) * dh
    out = []
    for k in orders:
        num = float(np.sum(h ** k * m))
        ana = (G(alpha + k) * G(beta + k) / (G(alpha) * G(beta) * (alpha * beta) ** k)
               * A0 ** k * xi ** 2 / (xi ** 2 + k))
        out.append((k, num, ana))
    return out


# --------------------------------------------------------------------------
def _main() -> None:
    import time

    XI_V, A0_V = 1.967, 0.1294336517          # the fig:odt_validation configuration
    GB_OP = 10.0 ** (GBAR_OP_DB / 10.0)

    print("=" * 100)
    print("POST-EGC SYSTEM ABER  --  eq:mimo_egc_aber,  MN = %d (4x4 MIMO-FSO),  "
          "Q(sqrt(gbar/MN) H)" % MN)
    print("Success test:  P_e,sys <= %.0e  at  gbar_op = %.1f dB" % (ABER_TARGET, GBAR_OP_DB))
    print("=" * 100)

    # -- 1 -----------------------------------------------------------------
    REF = {20: 4.510e-3, 28: 2.240e-5, 32: 6.200e-7, 40: 9.040e-11}
    print("\n[1] Validation configuration of Fig. odt_validation "
          "(strong, xi = 1.967, sigma_s = 0.05 m, A_0 = 0.1294),")
    print("    against the reference dictionary hardcoded in eq22_recursion.py.\n")
    print("    %-7s %-15s %-15s %-15s %-11s %s"
          % ("SNR", "fast", "quad", "REF (shipped)", "fast-quad", "fast vs REF"))
    for g in (20, 28, 32, 40):
        gb = 10.0 ** (g / 10.0)
        vf = system_aber(1.2, 1.1, XI_V, A0_V, gb, method="fast")
        vq = system_aber(1.2, 1.1, XI_V, A0_V, gb, method="quad")
        print("    %-7s %-15.6e %-15.6e %-15.3e %+-11.4f%% %+.3f%%"
              % ("%d dB" % g, vf, vq, REF[g], (vf - vq) / vq * 100,
                 (vf - REF[g]) / REF[g] * 100))

    # -- 2 -----------------------------------------------------------------
    print("\n[2] Density verification -- the manuscript's two stated checks")
    print("    (unit mass, and E[H] = MN*A_0*xi^2/(xi^2+1) = %.6f here)"
          % (MN * A0_V * XI_V ** 2 / (XI_V ** 2 + 1)))
    for lbl, (a, b) in REGIMES.items():
        m, mu, mu_a = verify_density(a, b, XI_V, A0_V)
        print("    %-9s mass = %.7f   E[H] = %.6f   (%+.4f%% vs analytic)"
              % (lbl, m, mu, (mu - mu_a) / mu_a * 100))
    print("\n    Branch raw moments, strong regime, at two corners of the xi box:")
    for xi, A0 in ((XI_V, A0_V), (0.5, 5.228863e-1)):
        print("      xi = %.3f, A_0 = %.4e :" % (xi, A0), end="")
        for k, num, ana in branch_moments(1.2, 1.1, xi, A0):
            print("  m%d %+.3e%%" % (k, (num - ana) / ana * 100), end="")
        print()

    # -- 3 -----------------------------------------------------------------
    print("\n[3] System ABER at gbar_op = %.0f dB, sigma_s = 0.10 m, r_d = 0."
          % GBAR_OP_DB)
    print("    The per-branch surrogate is shown alongside to make the scale gap")
    print("    the manuscript warns about visible.\n")
    from rtodt_fast import pe_series_f64, z_of
    print("    %-9s %-7s %-12s %-15s %-13s %-8s"
          % ("regime", "xi", "A_0", "P_e,sys", "P_e,branch", "success"))
    for regime in ("weak", "moderate", "strong"):
        a, b = REGIMES[regime]
        for xi in (0.5, 1.0, 1.967, 3.0, 4.888):
            cfg = BeamConfig.from_xi(regime, xi, 0.10)
            v = aber_of(cfg)
            z = z_of(a, b, np.array([cfg.A0]), GB_OP)
            pb = float(pe_series_f64(a, b, np.array([cfg.xi_eff]),
                                     np.array([cfg.A0]), GB_OP,
                                     np.array([20 if z[0] <= 8 else -1]))[0])
            print("    %-9s %-7.3f %-12.4e %-15.6e %-13.4e %-8s"
                  % (regime, xi, cfg.A0, v, pb, success(cfg)))

    # -- 4 -----------------------------------------------------------------
    print("\n[4] Strong turbulence at %.0f dB vs building sway, xi = 1.967"
          % GBAR_OP_DB)
    for s in SIGMAS:
        cfg = BeamConfig.from_xi("strong", 1.967, s)
        print("    sigma_s = %.2f m   w_z = %.4f m   A_0 = %.4e   P_e,sys = %.6e"
              "   success = %s" % (s, cfg.w_z, cfg.A0, aber_of(cfg), success(cfg)))

    # -- 5 -----------------------------------------------------------------
    print("\n[5] Best attainable beam per (regime, sigma_s): min over the xi box")
    print("    [max(0.5, xi_min(sigma_s)), 4.888].  This is what an ideal solver")
    print("    would find, so it bounds the achievable success rate.\n")
    from scipy.optimize import minimize_scalar
    print("    %-9s %-9s %-9s %-17s %s"
          % ("regime", "sigma_s", "xi*", "min P_e,sys", "target reachable"))
    for regime in ("weak", "moderate", "strong"):
        for s in SIGMAS:
            lo = 0.5
            for cand in np.linspace(0.5, 6.0, 400):      # xi_min via attainability
                try:
                    BeamConfig.from_xi(regime, cand, s); lo = cand; break
                except ValueError:
                    continue
            fun = lambda x: np.log10(max(aber_of(BeamConfig.from_xi(regime, x, s)),
                                         ROUNDOFF_FLOOR))
            r = minimize_scalar(fun, bounds=(lo, XI_MAX), method="bounded",
                                options=dict(xatol=1e-4))
            v = 10.0 ** r.fun
            shown = ("<= %.0e (floor)" % ROUNDOFF_FLOOR if v <= ROUNDOFF_FLOOR
                     else "%.6e" % v)
            print("    %-9s %-9.2f %-9.4f %-17s %s"
                  % (regime, s, r.x, shown, v <= ABER_TARGET))
    print("\n    NOTE: entries at the floor are limited by the FFT convolution's")
    print("    double precision, not by physics -- they are ten decades under the")
    print("    1e-6 target, so the reachability verdict is unaffected either way.")

    # -- 6 -----------------------------------------------------------------
    print("\n[6] Residual boresight (strong, xi = 1.967, sigma_s = 0.10 m)")
    w = BeamConfig.from_xi("strong", 1.967, 0.10).w_z
    for r_d in (0.0, 0.02, 0.05, 0.10, 0.20):
        cfg = BeamConfig("strong", w, 0.10, r_d)
        print("    r_d = %.2f m   xi_eff = %.4f   P_e,sys = %.6e   success = %s"
              % (r_d, cfg.xi_eff, aber_of(cfg), success(cfg)))

    # -- 7 -----------------------------------------------------------------
    print("\n[7] SNR sweep, strong turbulence, sigma_s = 0.05 m, xi = 1.967")
    cfg = BeamConfig.from_xi("strong", 1.967, 0.05)
    for g in (20, 25, 30, 34, 36, 38, 40, 45, 50):
        print("    gbar = %2d dB   P_e,sys = %.6e%s"
              % (g, aber_of(cfg, gbar_db=g), "   <- gbar_op" if g == 38 else ""))

    # -- 8 -----------------------------------------------------------------
    print("\n[8] Fast path vs the two references.  'quad' differs from 'fast' only")
    print("    in how f_h is built; 'mc' is an independent 16-branch simulation")
    print("    (%.0e samples) and arbitrates where they disagree.\n" % 4e6)
    panel = [("strong  xi=1.967 s=0.05", (1.2, 1.1, 1.967, 1.294336517e-1)),
             ("strong  xi=0.500 s=0.10", (1.2, 1.1, 0.500, 5.228863e-1)),
             ("strong  xi=4.888 s=0.10", (1.2, 1.1, 4.888, 5.231769e-3)),
             ("moderate xi=1.000 s=0.10", (2.1, 1.5, 1.000, 1.251839e-1)),
             ("moderate xi=3.000 s=0.30", (2.1, 1.5, 3.000, 1.543210e-3)),
             ("weak    xi=1.000 s=0.20", (4.2, 3.0, 1.000, 3.125259e-2)),
             ("weak    xi=1.967 s=0.05", (4.2, 3.0, 1.967, 1.294336517e-1))]
    print("    %-25s %-7s %-14s %-14s %-14s %-12s %s"
          % ("configuration", "SNR", "fast", "quad", "mc", "fast-quad", "fast-mc"))
    worst_fq = worst_fm = 0.0
    n_fq = n_fm = 0
    for lbl, (a, b, xi, A0) in panel:
        for g in (30, 38):
            gb = 10.0 ** (g / 10.0)
            vf = system_aber(a, b, xi, A0, gb, method="fast")
            vq = system_aber(a, b, xi, A0, gb, method="quad")
            vm = system_aber(a, b, xi, A0, gb, method="mc")
            live = min(vf, vq) > ROUNDOFF_FLOOR          # both above the floor
            resolved = vm > 4e-5                         # MC 1se below ~0.4%
            dq = (vf - vq) / vq * 100 if vq > 0 else float("nan")
            dm = (vf - vm) / vm * 100 if vm > 0 else float("nan")
            se = np.sqrt(max(vm, 0.0) / 4e6) / vm * 100 if vm > 0 else float("inf")
            if live and xi >= 1.0:      # xi < 1 is where 'quad' itself breaks down
                worst_fq = max(worst_fq, abs(dq)); n_fq += 1
            if resolved:
                worst_fm = max(worst_fm, abs(dm)); n_fm += 1
            tag = "" if live else "   [< roundoff floor: not comparable]"
            print("    %-25s %-7s %-14.6e %-14.6e %-14.6e %+-12.4f%% %s%s"
                  % (lbl, "%d dB" % g, vf, vq, vm, dq,
                     ("%+.3f%% (MC 1se %.2f%%)" % (dm, se)) if resolved
                     else "  --  (MC 1se %.0f%%)" % se, tag))
    print("\n    worst |fast - quad|, over the %d points with xi >= 1 and both paths"
          % n_fq)
    print("      above the roundoff floor                        : %.4f%%" % worst_fq)
    print("    worst |fast - mc|,   over the %d points the MC resolves : %.4f%%"
          % (n_fm, worst_fm))
    print("\n    The xi = 0.500 rows are where 'quad' fails: f_h_exact integrates")
    print("    y^{xi^2-2} over y in [0, A_0] with fixed-order Gauss-Legendre, and at")
    print("    xi^2 = 0.25 that integrand is too sharply peaked near y = 0 for the")
    print("    nodes to resolve.  The independent Monte Carlo backs 'fast' there")
    print("    (+0.2% at 30 dB, +1.1% at 38 dB, inside its own 1.2%/5.2% 1se),")
    print("    while 'quad' is low by a factor of 1.6-1.8.  xi < 1 is inside the")
    print("    swept decision box, so this is not a corner case.")

    # -- 8b ----------------------------------------------------------------
    print("\n[8b] Arbitration of the one decision-critical point: the BEST beam in")
    print("     the box under strong turbulence at sigma_s = 0.10 m sits within a")
    print("     factor of 1.5 of the target, so all three paths are made to agree")
    print("     on it before any success verdict is trusted.\n")
    crit = BeamConfig.from_xi("strong", 0.9909, 0.10)
    n_mc = 40_000_000
    vf = aber_of(crit); vq = aber_of(crit, method="quad")
    vm = system_aber(1.2, 1.1, crit.xi_eff, crit.A0, GB_OP,
                     method="mc", mc_samples=n_mc, mc_seed=7)
    se = np.sqrt(vm / n_mc)
    print("     xi = %.4f, A_0 = %.4e, gbar = 38 dB" % (crit.xi, crit.A0))
    print("     fast %.6e   quad %.6e   mc %.6e +/- %.2e (1se, n = %.0e)"
          % (vf, vq, vm, se, n_mc))
    print("     target %.0e is %.2f sigma below the MC estimate -> success = %s"
          % (ABER_TARGET, (vm - ABER_TARGET) / se, success(crit)))

    # -- 9 -----------------------------------------------------------------
    print("\n[9] What the released egc_system.aber_system returns on the same panel")
    print("    (point-sampled lattice with the h[0] = h[1]*1e-6 patch):\n")
    for lbl, (a, b, xi, A0) in panel[:3]:
        v = system_aber(a, b, xi, A0, GB_OP, method="egc")
        f = system_aber(a, b, xi, A0, GB_OP, method="fast")
        print("    %-25s  egc_system %-14.6e   this module %-14.6e   ratio %.3e"
              % (lbl, v, f, v / f))

    # -- 10 ----------------------------------------------------------------
    print("\n[10] Cost per evaluation at gbar_op")
    system_aber(1.2, 1.1, XI_V, A0_V, GB_OP, method="fast")       # warm caches
    t0 = time.time()
    for _ in range(25):
        system_aber(1.2, 1.1, XI_V, A0_V, GB_OP, method="fast")
    tf = (time.time() - t0) / 25
    t0 = time.time(); system_aber(1.2, 1.1, XI_V, A0_V, GB_OP, method="quad")
    tq = time.time() - t0
    t0 = time.time(); system_aber(1.2, 1.1, XI_V, A0_V, GB_OP, method="mc")
    tm = time.time() - t0
    print("    fast %.4f s   quad %.4f s (%.0fx)   mc %.4f s (%.0fx)"
          % (tf, tq, tq / tf, tm, tm / tf))


if __name__ == "__main__":
    _main()
