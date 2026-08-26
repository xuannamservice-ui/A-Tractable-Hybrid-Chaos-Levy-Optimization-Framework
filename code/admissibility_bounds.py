"""
Regenerate the RT-ODT truncation / admissibility bounds of Table 7.

Everything is computed from the printed equations alone -- no pre-computed
tensor, no stored intermediate:

    eq. (16)  a_k(A,B,xi) = xi^2/(G(A)G(B)) * (-1)^k (AB)^{B+k} G(A-B-k)
                            / ( k! (xi^2-B-k) A_0^{B+k} )
    eq. (20)  C(s,gbar)   = G((s+1)/2) / (2 s sqrt(pi)) * (2/gbar)^{s/2}
    eq. (26)  eps_trunc(K) = sum_{k>K} [ |a_k(A,B)| C(B+k) + |a_k(B,A)| C(A+k) ]
    eq. (27)  eta_f64      = max_k |a_k C| * eps_mach

The tabulated entry for a rung is  max [ eps_trunc(K) + eta_f64 ]  over the
176-point (xi, gbar) grid and the four swept jitter levels, restricted to the
band that rung serves:  z <= 0.5 (K=5), z <= 2 (K=10), z <= 8 (K=20), where
z = sqrt(2) alpha beta / (A_0(xi) sqrt(gbar)).

Usage:  python admissibility_bounds.py
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtodt import NODES, REGIMES, SIGMAS, A0_for, a_k, C_moment, z_param, db

mp.mp.dps = 80

SNR_GRID = list(range(20, 51, 2))            # 16 points, 20-50 dB
RUNGS = ((5, mp.mpf("0.5")), (10, mp.mpf(2)), (20, mp.mpf(8)))
EPS64 = mp.mpf(2) ** -52
KTAIL = 260                                  # tail summation cut-off

PUBLISHED = {                                # Table 7, for comparison
    (5, "weak"): 1.82e-9, (5, "moderate"): 8.75e-8, (5, "strong"): 4.44e-7,
    (10, "weak"): 3.98e-9, (10, "moderate"): 5.49e-10, (10, "strong"): 7.87e-10,
    (20, "weak"): 1.56e-10, (20, "moderate"): 4.83e-13, (20, "strong"): 4.55e-13,
}


def eps_trunc(A, B, xi, A0, gbar, K):
    """Tail of eq. (26), summed until the remaining terms stop contributing."""
    tot = mp.mpf(0)
    for k in range(K + 1, KTAIL):
        t = (abs(a_k(A, B, xi, A0, k)) * C_moment(B + k, gbar)
             + abs(a_k(B, A, xi, A0, k)) * C_moment(A + k, gbar))
        tot += t
        if k > K + 25 and t < tot * mp.mpf("1e-30"):
            break
    return tot


def eta_f64(A, B, xi, A0, gbar, K):
    """eq. (27): the float64 round-off floor of the retained sum."""
    m = mp.mpf(0)
    for k in range(K + 1):
        m = max(m, abs(a_k(A, B, xi, A0, k)) * C_moment(B + k, gbar),
                abs(a_k(B, A, xi, A0, k)) * C_moment(A + k, gbar))
    return m * EPS64


def rung_bound(regime, K, zmax, verbose=False):
    A, B = REGIMES[regime]
    worst, arg = mp.mpf(0), None
    n_in = 0
    for s in SIGMAS:
        for xi in NODES:
            A0 = A0_for(xi, s)
            if A0 is None:
                continue
            for gdb in SNR_GRID:
                g = db(gdb)
                if z_param(A, B, A0, g) > zmax:
                    continue
                n_in += 1
                b = eps_trunc(A, B, xi, A0, g, K) + eta_f64(A, B, xi, A0, g, K)
                if b > worst:
                    worst, arg = b, (float(s), float(xi), gdb)
    return worst, arg, n_in


if __name__ == "__main__":
    print("RT-ODT admissibility bounds, regenerated from Eqs. (16), (20), (26), (27)")
    print("grid: %d xi-nodes x %d SNR points x %d jitter levels\n"
          % (len(NODES), len(SNR_GRID), len(SIGMAS)))
    print("  %-5s %-9s %-13s %-13s %-8s %s"
          % ("K", "regime", "regenerated", "Table 7", "ratio", "argmax (sigma_s, xi, dB)"))
    print("  " + "-" * 82)
    for K, zmax in RUNGS:
        for regime in ("weak", "moderate", "strong"):
            b, arg, n = rung_bound(regime, K, zmax)
            pub = PUBLISHED[(K, regime)]
            r = float(b) / pub if pub else float("nan")
            print("  %-5d %-9s %-13.3e %-13.3e %-8.2f %s   [%d admissible pts]"
                  % (K, regime, float(b), pub, r, arg, n))
        print()
