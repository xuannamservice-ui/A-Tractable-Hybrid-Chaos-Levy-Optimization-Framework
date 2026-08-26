"""Sanity-check the off-grid result before believing it.

(1) single-point breakdown at the worst xi
(2) is the EXACT value a sane probability there?
(3) what fraction of admissible off-grid points does the deployed scheme push
    outside [0, 1/2]  (i.e. would be caught by guard range-test (ii))?
(4) does the error vanish AT the nodes (it must, by construction)?
"""
import mpmath as mp
import random
from rtodt import (NODES, REGIMES, A0_for, a_k, D_coef, C_moment, z_param, db)
from exp_offgrid import node_tables, evaluate, bracket, lerp, K, SIGMA, ZMAX

mp.mp.dps = 90
random.seed(7)

print("=" * 78)
print("(1)+(2) SINGLE-POINT BREAKDOWN, weak regime, 40 dB, xi = 2.2433")
print("=" * 78)
A, B = REGIMES["weak"]
g = db(40)
T = node_tables(A, B, g, SIGMA)
xi = mp.mpf("2.2433")
A0 = A0_for(xi, SIGMA)
ex, dep, fx, parts = evaluate(xi, A, B, g, SIGMA, T)
j, t = bracket(xi)
print("  bracketing nodes      : [%.3f, %.3f]  t=%.4f" % (float(NODES[j]), float(NODES[j+1]), float(t)))
print("  A_0                   : %.6g" % float(A0))
print("  z (admissible if <=8) : %.4f" % float(z_param(A, B, A0, g)))
print("  EXACT Pe              : %s" % mp.nstr(ex, 10))
print("  DEPLOYED (interp)     : %s" % mp.nstr(dep, 10))
print("  FIXED (product D*C)   : %s" % mp.nstr(fx, 10))
print("  exact in [0,0.5]?     : %s" % (0 <= ex <= mp.mpf("0.5")))
print("  deployed in [0,0.5]?  : %s" % (0 <= dep <= mp.mpf("0.5")))
print()
print("  poles of a_k inside the bracketing interval:")
poles = sorted({float(mp.sqrt(v + k)) for k in range(K + 1) for v in (A, B)
                if NODES[j] < mp.sqrt(v + k) < NODES[j + 1]})
print("    ", [round(p, 4) for p in poles] or "none")

print()
print("=" * 78)
print("(4) ERROR AT THE NODES THEMSELVES (must be ~0 by construction)")
print("=" * 78)
for jj in (3, 5, 7):
    xn = NODES[jj]
    exn, depn, _, _ = evaluate(xn, A, B, g, SIGMA, T)
    print("  xi = %-6s  |deployed - exact| = %.3e" % (float(xn), float(abs(depn - exn))))

print()
print("=" * 78)
print("(3) FRACTION OF ADMISSIBLE OFF-GRID POINTS RETURNING A NON-PROBABILITY")
print("=" * 78)
print("  (guard range-test (ii) requires 0 <= Pe <= 1/2)")
for reg in ("weak", "moderate", "strong"):
    A, B = REGIMES[reg]
    for gdb in (30, 40):
        gb = db(gdb)
        T = node_tables(A, B, gb, SIGMA)
        bad_dep = bad_fix = tot = 0
        exact_bad = 0
        tries = 0
        while tot < 200 and tries < 20000:
            tries += 1
            x = NODES[0] + (NODES[-1] - NODES[0]) * mp.mpf(random.random())
            A0x = A0_for(x, SIGMA)
            if A0x is None or z_param(A, B, A0x, gb) > ZMAX:
                continue
            tot += 1
            e, d, f, _ = evaluate(x, A, B, gb, SIGMA, T)
            if not (0 <= d <= mp.mpf("0.5")):
                bad_dep += 1
            if not (0 <= f <= mp.mpf("0.5")):
                bad_fix += 1
            if not (0 <= e <= mp.mpf("0.5")):
                exact_bad += 1
        if tot:
            print("  %-9s %2d dB  n=%3d : deployed %5.1f%% invalid | product-fix %5.1f%% | exact %5.1f%%"
                  % (reg, gdb, tot, 100*bad_dep/tot, 100*bad_fix/tot, 100*exact_bad/tot))
