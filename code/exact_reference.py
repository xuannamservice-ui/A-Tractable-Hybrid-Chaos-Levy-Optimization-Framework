"""
Independent 'exact' ABER reference, derived from first principles rather than
from the truncated series, so that the validation figure compares two genuinely
different computations.

    h = h_a * h_p
    h_a ~ gamma-gamma(alpha, beta), unit mean:
        f(x) = 2 (ab)^{(a+b)/2} / (G(a)G(b)) * x^{(a+b)/2 - 1} * K_{a-b}(2 sqrt(ab x))
    h_p : pointing loss with pdf  xi^2 / A0^{xi^2} * y^{xi^2-1}  on [0, A0]
          equivalently  h_p = A0 * U^{1/xi^2},  U ~ Uniform(0,1)

    ABER = E[ Q(sqrt(gbar) * h_a * h_p) ]
         = Int_0^1 Int_0^inf Q(sqrt(gbar) x A0 u^{1/xi^2}) f(x) dx du

Both integrals by Gauss-Legendre; the inner one on a log-x grid.
"""
import numpy as np
from scipy.special import kv, gamma as G, erfc

_GLX, _GLW = np.polynomial.legendre.leggauss(400)


def gg_pdf(x, a, b):
    c = 2.0 * (a * b) ** ((a + b) / 2.0) / (G(a) * G(b))
    return c * x ** ((a + b) / 2.0 - 1.0) * kv(a - b, 2.0 * np.sqrt(a * b * x))


def _logx_nodes(lo=-18.0, hi=6.0):
    t = 0.5 * (_GLX + 1.0) * (hi - lo) + lo
    w = _GLW * 0.5 * (hi - lo)
    x = np.exp(t)
    return x, w * x            # dx = x dt


def Q(z):
    return 0.5 * erfc(z / np.sqrt(2.0))


def aber_exact(gbar, a, b, xi, A0, nu=200):
    """Double Gauss-Legendre. Returns E[Q(sqrt(gbar) h)]."""
    x, wx = _logx_nodes()
    fx = gg_pdf(x, a, b)
    fx = np.where(np.isfinite(fx), fx, 0.0)

    gu, gw = np.polynomial.legendre.leggauss(nu)
    u = 0.5 * (gu + 1.0)
    wu = gw * 0.5
    hp = A0 * u ** (1.0 / xi ** 2)                       # (nu,)

    arg = np.sqrt(gbar) * np.outer(hp, x)                # (nu, nx)
    val = Q(arg) * (fx * wx)[None, :]
    return float((val.sum(axis=1) * wu).sum())


def sanity():
    for (a, b) in [(4.2, 3.0), (2.1, 1.5), (1.2, 1.1)]:
        x, wx = _logx_nodes()
        f = gg_pdf(x, a, b)
        f = np.where(np.isfinite(f), f, 0.0)
        m0 = float((f * wx).sum())
        m1 = float((x * f * wx).sum())
        print("  gamma-gamma(%.1f,%.1f):  integral = %.10f   mean = %.10f"
              % (a, b, m0, m1))


if __name__ == "__main__":
    print("Sanity of the gamma-gamma density (both should be 1.0):")
    sanity()
