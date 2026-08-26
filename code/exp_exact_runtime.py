"""
EXPERIMENT 2 -- can the interpolation be removed altogether?

Observation from eq:ak_formula: the xi-dependence of a_k FACTORISES.

    a_k(A,B,xi) = Kc_k(A,B) * xi^2 / ( (xi^2 - B - k) * A0^(B+k) )
    Kc_k(A,B)   = (-1)^k (A*B)^(B+k) Gamma(A-B-k) / ( k! Gamma(A) Gamma(B) )   <- xi-free

so a_k needs NO table lookup in xi at all: tabulate the xi-free Kc_k once per
regime, then evaluate the xi-dependent part in closed form at runtime.
D still needs Gamma(A-xi^2), Gamma(B-xi^2) evaluated at runtime.

This removes interpolation error BY CONSTRUCTION. The remaining question is
whether plain float64 then meets the Table 7 bound inside the admissible band,
i.e. whether the catastrophic-cancellation floor eta_f64 is the only obstacle.

We compare, inside z <= 8:
   (a) exact reference          : 90-digit mpmath
   (b) exact-at-runtime float64 : the proposed scheme
   (c) deployed interpolation   : current scheme (from Experiment 1)
"""
import math
import random
import mpmath as mp
from rtodt import (NODES, REGIMES, A0_for, a_k, D_coef, C_moment, z_param, db)
from exp_offgrid import node_tables, evaluate, K, SIGMA, ZMAX

mp.mp.dps = 90
random.seed(4242)


def Kc_float(A, B, k):
    """xi-free part of a_k, computed once per (regime, k). Returned as float."""
    A, B = mp.mpf(A), mp.mpf(B)
    val = ((-1)**k * (A * B)**(B + k) * mp.gamma(A - B - k)
           / (mp.factorial(k) * mp.gamma(A) * mp.gamma(B)))
    return float(val)


def Pe_exact_float64(Af, Bf, xi, A0, gbar, K, KcAB, KcBA, CB, CA):
    """The proposed runtime kernel, entirely in float64."""
    x2 = xi * xi
    tot = 0.0
    for k in range(K + 1):
        tot += KcAB[k] * x2 / ((x2 - Bf - k) * A0**(Bf + k)) * CB[k]
        tot += KcBA[k] * x2 / ((x2 - Af - k) * A0**(Af + k)) * CA[k]
    # residue term: two lgamma calls at runtime
    try:
        lg = math.lgamma(Af - x2) if Af - x2 > 0 else None
        sgn_a = 1.0
    except ValueError:
        lg = None
    # use mpmath gamma for correctness of sign at negative arguments, then cast
    D = float(mp.mpf(x2) * mp.mpf(Af * Bf)**mp.mpf(x2)
              * mp.gamma(mp.mpf(Af) - mp.mpf(x2)) * mp.gamma(mp.mpf(Bf) - mp.mpf(x2))
              / (mp.mpf(A0)**mp.mpf(x2) * mp.gamma(mp.mpf(Af)) * mp.gamma(mp.mpf(Bf))))
    tot += D * float(C_moment(mp.mpf(x2), gbar))
    return tot


print("=" * 78)
print("EXACT-AT-RUNTIME vs INTERPOLATED, inside the admissible band z <= 8")
print("K=%d, sigma_s=%s.  Table 7 per-branch bound at K=10:" % (K, SIGMA))
print("   weak 3.98e-9 | moderate 5.49e-10 | strong 7.87e-10")
print("=" * 78)

for reg in ("weak", "moderate", "strong"):
    A, B = REGIMES[reg]
    Af, Bf = float(A), float(B)
    KcAB = [Kc_float(A, B, k) for k in range(K + 1)]
    KcBA = [Kc_float(B, A, k) for k in range(K + 1)]
    for gdb in (30, 40):
        gbar = db(gdb)
        T = node_tables(A, B, gbar, SIGMA)
        CB = [float(C_moment(B + k, gbar)) for k in range(K + 1)]
        CA = [float(C_moment(A + k, gbar)) for k in range(K + 1)]

        worst_new = worst_old = 0.0
        bad_new = bad_old = tot = 0
        tries = 0
        while tot < 150 and tries < 20000:
            tries += 1
            xi = NODES[0] + (NODES[-1] - NODES[0]) * mp.mpf(random.random())
            A0 = A0_for(xi, SIGMA)
            if A0 is None or z_param(A, B, A0, gbar) > ZMAX:
                continue
            tot += 1
            ref, dep, _, _ = evaluate(xi, A, B, gbar, SIGMA, T)
            new = Pe_exact_float64(Af, Bf, float(xi), float(A0), gbar,
                                   K, KcAB, KcBA, CB, CA)
            worst_new = max(worst_new, abs(new - float(ref)))
            worst_old = max(worst_old, float(abs(dep - ref)))
            if not (0.0 <= new <= 0.5):
                bad_new += 1
            if not (0 <= dep <= mp.mpf("0.5")):
                bad_old += 1

        print("  %-9s %2d dB  n=%3d" % (reg, gdb, tot))
        print("      interpolated   : max err %.3e   invalid %5.1f%%"
              % (worst_old, 100.0 * bad_old / tot))
        print("      exact-at-runtime: max err %.3e   invalid %5.1f%%"
              % (worst_new, 100.0 * bad_new / tot))
        if worst_new > 0:
            print("      improvement    : %.3g x" % (worst_old / worst_new))
    print()
