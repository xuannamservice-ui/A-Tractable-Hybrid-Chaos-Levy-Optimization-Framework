"""
EXPERIMENT 3 -- is the exact-at-runtime kernel still real-time?

The manuscript's headline is speed: ~0.6 us/particle amortized, SIMD-vectorized
swarm, tau_O = 600 us for T_iter=25 x N_p=30 = 750 evaluations.

Vectorized float64 implementation of the proposed kernel:
    a_k part : Kc_k (xi-free, precomputed) * xi^2 / ((xi^2-B-k) * A0^(B+k))
    D part   : two gamma evaluations per candidate (scipy.special.gamma)
Timed over a whole swarm at once, exactly as the paper's vectorized kernel is.
"""
import time
import numpy as np
from scipy.special import gamma as sp_gamma, erf

K = 10
ALPHA, BETA = 1.2, 1.1          # strong regime, the campaign's operating point
NP_SWARM, T_ITER = 30, 25
A_APER = 0.05


def precompute(A, B, K):
    """xi-free coefficients; computed once, offline."""
    from mpmath import mpf, gamma as mg, factorial as mfac
    KcAB, KcBA = [], []
    for k in range(K + 1):
        a, b = mpf(A), mpf(B)
        KcAB.append(float((-1)**k * (a*b)**(b+k) * mg(a-b-k) / (mfac(k)*mg(a)*mg(b))))
        KcBA.append(float((-1)**k * (b*a)**(a+k) * mg(b-a-k) / (mfac(k)*mg(b)*mg(a))))
    return np.array(KcAB), np.array(KcBA)


def C_moments(A, B, gbar, K):
    from mpmath import mpf, gamma as mg, sqrt as msq, pi as mpi
    def C(s):
        s = mpf(s)
        return float(mg((s+1)/2) / (2*s*msq(mpi)) * (2/mpf(gbar))**(s/2))
    return np.array([C(B+k) for k in range(K+1)]), np.array([C(A+k) for k in range(K+1)])


def kernel_exact(xi, A0, A, B, gbar, KcAB, KcBA, CB, CA):
    """Vectorised over a swarm of candidates. Pure float64."""
    x2 = xi * xi
    k = np.arange(K + 1)
    # a_k families, broadcast (n_cand, K+1)
    powB = A0[:, None] ** (B + k)[None, :]
    powA = A0[:, None] ** (A + k)[None, :]
    t1 = KcAB[None, :] * x2[:, None] / ((x2[:, None] - B - k[None, :]) * powB)
    t2 = KcBA[None, :] * x2[:, None] / ((x2[:, None] - A - k[None, :]) * powA)
    tot = (t1 * CB[None, :]).sum(1) + (t2 * CA[None, :]).sum(1)
    # residue: two gamma calls per candidate
    D = (x2 * (A*B)**x2 * sp_gamma(A - x2) * sp_gamma(B - x2)
         / (A0**x2 * sp_gamma(A) * sp_gamma(B)))
    Cx = sp_gamma((x2 + 1) / 2) / (2 * x2 * np.sqrt(np.pi)) * (2 / gbar) ** (x2 / 2)
    return tot + D * Cx


def kernel_interp(xi, node_tab):
    """Cost model of the CURRENT scheme: bracket search + 2*(K+1)+2 lerps + dot."""
    nodes, ak1, ak2, Dv, Cv, CB, CA = node_tab
    j = np.clip(np.searchsorted(nodes, xi) - 1, 0, len(nodes) - 2)
    t = (xi - nodes[j]) / (nodes[j+1] - nodes[j])
    a1 = ak1[j] + t[:, None] * (ak1[j+1] - ak1[j])
    a2 = ak2[j] + t[:, None] * (ak2[j+1] - ak2[j])
    D = Dv[j] + t * (Dv[j+1] - Dv[j])
    C = Cv[j] + t * (Cv[j+1] - Cv[j])
    return (a1 * CB).sum(1) + (a2 * CA).sum(1) + D * C


# ---- setup
gbar = 10 ** (38 / 10)
KcAB, KcBA = precompute(ALPHA, BETA, K)
CB, CA = C_moments(ALPHA, BETA, gbar, K)

rng = np.random.default_rng(0)
n = NP_SWARM
wz = rng.uniform(0.06, 0.5, n)
v = np.sqrt(np.pi / 2) * A_APER / wz
A0 = erf(v) ** 2
wzeq = np.sqrt(wz**2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v**2)))
xi = wzeq / (2 * 0.1)

# Node tables for the interpolated cost model.
#
# An earlier version of this file filled these with invented numbers
# (A0n = linspace(0.52, 0.005), Cv = linspace(1e-3, 1e-9), plus a dead line
# commented "placeholder A0 per node"). The timing is insensitive to the
# VALUES -- the two kernels do the same count of float64 operations whatever
# is in the tables -- but shipping invented tables in a released script
# invites the reader to wonder what else is invented, so they are now built
# from the same geometry the rest of the package uses.
nodes = np.array([0.500, 0.628, 0.789, 0.992, 1.266,
                  1.548, 1.967, 2.511, 3.104, 3.912, 4.888])
SIGMA_S = 0.1


def _wz_for_xi(xi_target, sigma_s, a=A_APER):
    """Beam waist giving this xi at this jitter, upper (beam-broadening) branch."""
    lo, hi = 0.0549, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        vv = np.sqrt(np.pi / 2) * a / mid
        wq = np.sqrt(mid ** 2 * np.sqrt(np.pi) * erf(vv) / (2 * vv * np.exp(-vv ** 2)))
        if wq / (2 * sigma_s) < xi_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_wzn = np.array([_wz_for_xi(x, SIGMA_S) for x in nodes])
_vn = np.sqrt(np.pi / 2) * A_APER / _wzn
A0n = erf(_vn) ** 2
_x2n = nodes ** 2
_kk = np.arange(K + 1)
ak1 = KcAB[None, :] * _x2n[:, None] / ((_x2n[:, None] - BETA - _kk[None, :])
                                       * A0n[:, None] ** (BETA + _kk)[None, :])
ak2 = KcBA[None, :] * _x2n[:, None] / ((_x2n[:, None] - ALPHA - _kk[None, :])
                                       * A0n[:, None] ** (ALPHA + _kk)[None, :])
Dv = (_x2n * (ALPHA * BETA) ** _x2n * sp_gamma(ALPHA - _x2n) * sp_gamma(BETA - _x2n)
      / (A0n ** _x2n * sp_gamma(ALPHA) * sp_gamma(BETA)))
Cv = sp_gamma((_x2n + 1) / 2) / (2 * _x2n * np.sqrt(np.pi)) * (2 / gbar) ** (_x2n / 2)
node_tab = (nodes, ak1, ak2, Dv, Cv, CB, CA)

# ---- warm up
for _ in range(50):
    kernel_exact(xi, A0, ALPHA, BETA, gbar, KcAB, KcBA, CB, CA)
    kernel_interp(xi, node_tab)

REP = 4000
t0 = time.perf_counter()
for _ in range(REP):
    kernel_exact(xi, A0, ALPHA, BETA, gbar, KcAB, KcBA, CB, CA)
t_exact = (time.perf_counter() - t0) / REP

t0 = time.perf_counter()
for _ in range(REP):
    kernel_interp(xi, node_tab)
t_interp = (time.perf_counter() - t0) / REP

print("Vectorised over a swarm of N_p = %d, K = %d, float64, single thread" % (n, K))
print()
print("  exact-at-runtime : %8.2f us / swarm   = %6.3f us / particle" % (t_exact*1e6, t_exact*1e6/n))
print("  interpolated     : %8.2f us / swarm   = %6.3f us / particle" % (t_interp*1e6, t_interp*1e6/n))
print("  ratio            : %.2f x" % (t_exact / t_interp))
print()
print("  Full optimization phase, T_iter=%d x N_p=%d = %d evaluations:" % (T_ITER, n, T_ITER*n))
print("      exact-at-runtime : %7.1f us      (paper's tau_O budget = 600 us)"
      % (t_exact * T_ITER * 1e6))
print("      interpolated     : %7.1f us" % (t_interp * T_ITER * 1e6))
print()
print("  NOTE: Python/NumPy figures. The paper's 0.6 us/particle refers to its own")
print("        optimized kernel; what matters here is the RATIO between the two")
print("        schemes measured under identical conditions.")
