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
    """n-fold self-convolution of a density sampled on a uniform grid from 0."""
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


def aber_system(f_branch, hmax, nh, gbar, a, b, xi, A0, which, K=10):
    h = np.linspace(0.0, hmax, nh)
    dh = h[1] - h[0]
    h[0] = h[1] * 1e-6                      # avoid the 0^{negative} singularity
    f = f_branch(h, a, b, xi, A0, K) if which == "series" else f_branch(h, a, b, xi, A0)
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.clip(f, -1e300, 1e300)
    fH = convolve_MN(f, dh)
    H = np.arange(len(fH)) * dh
    mass = float(np.trapezoid(fH, H))
    mean = float(np.trapezoid(H * fH, H))
    val = float(np.trapezoid(Q(np.sqrt(gbar / MN) * H) * fH, H))
    return val, mass, mean
