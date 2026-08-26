"""
System-level (post-EGC) ABER by explicit 16-fold convolution.

The manuscript obtains the combined density by convolving branch PDFs into
lambda_j / C_j via the recursion of [b13].  That recursion is an analytical
shortcut for a convolution, so the convolution itself can be done directly and
serves as an INDEPENDENT check on the combined result:

    H = sum_{i=1}^{MN} h_i ,  MN = 16
    f_H = f_h^{*16}                       (one FFT, 16th power, one inverse FFT)
    ABER_sys = Int Q( sqrt(gbar/MN) H ) f_H(H) dH

Two branch densities are convolved, giving two system curves:
  EXACT  f_h from the composite h = h_a h_p, by quadrature over the pointing law
  SERIES f_h from eq:pdf_series at order K  (the RT-ODT surrogate density)

Comparing them answers the question Section III-C leaves open: does the
truncation error amplify through the 16-fold convolution?
"""
import numpy as np
from scipy.special import kv, gamma as G, erfc

MN = 16


def gg_pdf(x, a, b):
    """gamma-gamma density, unit mean."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x > 0
    c = 2.0 * (a * b) ** ((a + b) / 2.0) / (G(a) * G(b))
    out[m] = c * x[m] ** ((a + b) / 2.0 - 1.0) * kv(a - b, 2.0 * np.sqrt(a * b * x[m]))
    return np.where(np.isfinite(out), out, 0.0)


def f_h_exact(h, a, b, xi, A0, ny=1200):
    """f_h(h) = Int_0^{A0} f_ha(h/y) * (xi^2 / A0^{xi^2}) y^{xi^2-2} dy."""
    gu, gw = np.polynomial.legendre.leggauss(ny)
    y = 0.5 * (gu + 1.0) * A0            # (ny,)
    w = gw * 0.5 * A0
    coef = xi ** 2 / A0 ** (xi ** 2)
    wy = w * coef * y ** (xi ** 2 - 2.0)
    arg = h[:, None] / y[None, :]
    return (gg_pdf(arg, a, b) * wy[None, :]).sum(axis=1)


def f_h_series(h, a, b, xi, A0, K):
    """eq:pdf_series -- D h^{xi^2-1} + sum_k [a_k(a,b) h^{b+k-1} + a_k(b,a) h^{a+k-1}]."""
    from rtodt import a_k, D_coef
    import mpmath as mp
    x = mp.mpf(xi); A = mp.mpf(a); B = mp.mpf(b); A0m = mp.mpf(A0)
    D = float(D_coef(A, B, x, A0m))
    ak1 = np.array([float(a_k(A, B, x, A0m, k)) for k in range(K + 1)])
    ak2 = np.array([float(a_k(B, A, x, A0m, k)) for k in range(K + 1)])
    out = D * h ** (xi ** 2 - 1.0)
    k = np.arange(K + 1)
    out = out + (ak1[None, :] * h[:, None] ** (b + k - 1.0)[None, :]).sum(1)
    out = out + (ak2[None, :] * h[:, None] ** (a + k - 1.0)[None, :]).sum(1)
    return out


def convolve_MN(f, dh, n=MN):
    """n-fold self-convolution of a density sampled on a uniform grid from 0.

    `f * dh` is the per-cell mass vector; pass dh = 1.0 if `f` already holds
    masses rather than densities.
    """
    L = len(f)
    N = 1
    while N < n * L:
        N *= 2
    F = np.fft.rfft(f * dh, N)
    FH = F ** n
    g = np.fft.irfft(FH, N) / dh
    return g[: n * L]


def Q(z):
    return 0.5 * erfc(z / np.sqrt(2.0))


def aber_system(f_branch, hmax, nh, gbar, a, b, xi, A0, which, K=10,
                return_floor=False):
    """Post-EGC ABER by explicit MN-fold convolution.

    Returns (aber, recovered_mass, mean), or (aber, mass, mean, floor) when
    `return_floor=True`. READ THE NOTE ON THE ROUND-OFF FLOOR BELOW BEFORE
    USING THE FIRST ELEMENT AT HIGH SNR.

    `hmax` and `nh` are the BRANCH grid, not the combined one: f_h is sampled
    on [0, hmax] with nh points and the convolution then spans [0, MN*hmax].
    The result is only as good as that grid, and the grid has to be chosen
    against the branch scale, not against 1: the branch mean is

        E[h] = E[h_a] E[h_p] = 1 * A_0 xi^2/(xi^2+1)  <  A_0,

    which for the configurations in the paper is of order 0.01-0.1. Passing a
    grid sized for a unit-mean variable badly under-resolves the rise of f_h
    near zero -- e.g. (hmax=4, nh=4000) at xi=1.967, sigma_s=0.05, A_0=0.129
    recovers only 1.072 of the unit mass, a 7% error that swamps anything the
    result is being used to measure.

    The convention used throughout this package, and by generate.py, is

        hmax = 40 * A_0,  nh = 60000

    which recovers 0.988-1.000 of the mass across the swept box. The recovered
    mass is returned as the second element precisely so the caller can check it
    rather than assume it; treat a mass far from 1 as a signal that the grid,
    not the model, needs attention.

    THE SINGULARITY AT h = 0.  f_h(h) -> C h^{xi^2 - 1} as h -> 0 (eq. 4), so
    for xi < 1 the density diverges, integrably, at the origin. This routine
    used to sample f on a grid that included h = 0 and then patch the first
    node to h[0] = h[1] * 1e-6 -- an arbitrary offset with no derivation --
    before multiplying it by the full cell width dh. For xi^2 < 1 that assigns
    the first cell a mass of order (1e-6)^{xi^2-1} times what it should have,
    and the error is unbounded: at xi = 0.500 (xi^2 = 0.25), strong turbulence,
    sigma_s = 0.1 m, the patched version recovered a mass of 2.4e27 and
    returned an "ABER" of 9.9e26. At xi = 0.992 it recovered 1.43. Both are
    inside the swept decision box, so this was not a corner case.

    The grid is now cell-MIDPOINT sampled, so h = 0 is never evaluated and no
    arbitrary offset appears. When the density actually diverges -- xi^2 < 1 --
    the first cell's mass is integrated analytically against the leading power
    law instead of being taken from the midpoint value:

        Int_0^dh C h^{x2-1} dh = C dh^{x2} / x2,   C = f(dh/2) (dh/2)^{1-x2}
        =>  m_0 = f(dh/2) * dh * 2^{x2-1} / x2

    which is exact for a pure power law, reduces to f*dh at x2 = 1, and is
    derived from eq. (4) rather than fitted to anything.

    The correction is applied ONLY for x2 < 1, and that restriction is not a
    tuning knob: for x2 >= 1 the density is bounded on [0, dh] and the plain
    midpoint rule is already second-order there, while the extrapolation
    constant 2^{x2-1} grows without bound -- at xi = 4.888 (x2 = 23.9) it is
    1.6e5, and applying it multiplies the round-off in f(dh/2) by that factor,
    driving the recovered branch mass to 726. Measured branch mass on the
    strong-turbulence panel, hmax = 40 A_0, nh = 60000:

        xi     x2      old (endpoint + 1e-6 patch)   this routine
        0.500  0.250   53.04                         0.9974
        0.789  0.623    1.991                        0.9988
        0.992  0.984    1.023                        0.9993
        1.548  2.396    1.00006                      1.000005
        1.967  3.869    0.999993                     1.000011
        4.888 23.893    0.999966                     1.000009

    The branch mass matters more than it looks: the MN = 16 fold convolution
    raises it to the 16th power, so a 0.25% branch error is a 4% system error.

    The residual 0.26% at xi = 0.500 is not this routine's: it is the inner
    Gauss-Legendre in `f_h_exact`, whose fixed-order rule under-resolves
    y^{xi^2-2} near y = 0 when xi < 1. That limit is inherited, and is the
    reason `system_metric.py` carries an independent log-domain path.

    THE ROUND-OFF FLOOR, AND WHY THE ANSWER CAN GO NEGATIVE.  The convolution
    is done by raising a double-precision FFT to the 16th power, so f_H carries
    an absolute round-off noise of order eps_mach * max|f_H|, spread over the
    whole H axis including the far tail where the true density has long since
    underflowed. Integrating Q() against that noise gives a floor on what this
    routine can resolve. Below the floor the returned value is the noise, and
    because the noise is signed the value can come out NEGATIVE -- which is
    what a reader sees if they take it at face value as a probability.

    This is not hypothetical. At weak turbulence, xi = 0.992, sigma_s = 0.05,
    the routine returns -4.6e-19 at 40 dB and -1.7e-19 at 48 dB. The released
    dataset data/06_system_aber/system_aber_curves.csv carries those negative
    entries in a column named `aber_system_exact`.

    `return_floor=True` returns a measured estimate of that floor: the mass
    sitting in the negative excursions of f_H, integrated against the same Q
    weight. Compare the returned ABER against it. If |aber| is not comfortably
    above the floor, the value is round-off and this method cannot resolve the
    point -- use the log-domain path in `system_metric.py`, which does not
    convolve in the linear domain and has no such floor.
    """
    x2 = float(xi) ** 2
    dh = float(hmax) / nh
    h = (np.arange(nh) + 0.5) * dh          # cell midpoints; h = 0 never sampled
    f = f_branch(h, a, b, xi, A0, K) if which == "series" else f_branch(h, a, b, xi, A0)
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.clip(f, -1e300, 1e300)

    m = f * dh                              # per-cell mass, midpoint rule
    if x2 < 1.0:                            # integrable divergence at h -> 0
        m[0] = f[0] * dh * 2.0 ** (x2 - 1.0) / x2

    fH = convolve_MN(m, 1.0) / dh           # convolve masses, return a density
    H = (np.arange(len(fH)) + 0.5 * MN) * dh      # MN midpoints sum to MN*dh/2
    mass = float(fH.sum() * dh)
    mean = float((H * fH).sum() * dh)
    w = Q(np.sqrt(gbar / MN) * H)
    val = float((w * fH).sum() * dh)
    if not return_floor:
        return val, mass, mean
    # A density cannot be negative; every negative sample of f_H is pure FFT
    # round-off, so integrating |negative part| against the same weight
    # measures the noise in the same units as `val`.
    floor = float((w * np.abs(np.minimum(fH, 0.0))).sum() * dh)
    return val, mass, mean, floor
