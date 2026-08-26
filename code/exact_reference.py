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

Note on the outer variable.  Integrating in u directly is what the expression
above literally says, but u^{1/xi^2} has an infinite derivative at u = 0 for
every xi > 1, and Gauss-Legendre converges algebraically rather than
spectrally against such an endpoint.  Substituting  u = t^{xi^2}  removes it:

    h_p = A0 * t,   t ~ pdf  xi^2 t^{xi^2 - 1}  on [0, 1]

which is the pointing law written directly in the loss variable (eq. 4 with
h_p = A0 t).  The integrand is then smooth on the closed interval and the
outer rule converges spectrally.  This is a change of variable, not a change
of model: the two forms are the same integral.

Measured effect at xi=1.967, sigma_s=0.05, weak turbulence, against the K=10
series inside its admissible band (z <= 2): the u-form disagrees by 6.8e-5 at
40 dB and 8.7e-4 at 50 dB and is still drifting at nu = 200; the t-form agrees
to 5.6e-10 at 40 dB and 2.5e-14 at 50 dB and is converged by nt = 200.
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
    """Double Gauss-Legendre. Returns E[Q(sqrt(gbar) h)].

    `nu` is the order of the outer (pointing) rule, taken in the smooth
    variable t = u^{1/xi^2}; see the module docstring. 200 nodes is already
    converged to float64 round-off for the configurations used in the paper --
    it is not a tuned value, the rule simply stops improving.
    """
    x, wx = _logx_nodes()
    fx = gg_pdf(x, a, b)
    fx = np.where(np.isfinite(fx), fx, 0.0)

    gt, gw = np.polynomial.legendre.leggauss(nu)
    t = 0.5 * (gt + 1.0)
    x2 = xi ** 2
    wt = gw * 0.5 * x2 * t ** (x2 - 1.0)                 # pdf of t, times dt
    hp = A0 * t                                          # (nu,)

    arg = np.sqrt(gbar) * np.outer(hp, x)                # (nu, nx)
    val = Q(arg) * (fx * wx)[None, :]
    return float((val.sum(axis=1) * wt).sum())


def sanity():
    for (a, b) in [(4.2, 3.0), (2.1, 1.5), (1.2, 1.1)]:
        x, wx = _logx_nodes()
        f = gg_pdf(x, a, b)
        f = np.where(np.isfinite(f), f, 0.0)
        m0 = float((f * wx).sum())
        m1 = float((x * f * wx).sum())
        print("  gamma-gamma(%.1f,%.1f):  integral = %.10f   mean = %.10f"
              % (a, b, m0, m1))


def convergence():
    """Show that the outer rule is converged: doubling nu must not move it."""
    from rtodt import REGIMES, A0_for, db
    import mpmath as mp
    xi, sigma = mp.mpf("1.967"), mp.mpf("0.05")
    A0 = float(A0_for(xi, sigma))
    print("\nOuter-rule convergence at xi=1.967, sigma_s=0.05, A_0=%.4f:" % A0)
    print("  %-9s %-6s %-22s %-22s %s"
          % ("regime", "SNR", "nu=100", "nu=200", "nu=400"))
    for reg in ("weak", "moderate", "strong"):
        A, B = REGIMES[reg]
        for gdb in (30, 40, 50):
            vals = [aber_exact(float(db(float(gdb))), float(A), float(B),
                               float(xi), A0, nu=n) for n in (100, 200, 400)]
            print("  %-9s %-6s %-22.14e %-22.14e %.14e"
                  % (reg, "%d dB" % gdb, vals[0], vals[1], vals[2]))


if __name__ == "__main__":
    print("Sanity of the gamma-gamma density (both should be 1.0):")
    sanity()
    convergence()
