"""Reproduce numbers already printed in access.tex, to prove the
re-implementation is faithful before using it for anything new.

Every block prints `got` beside `want` and labels the row OK or MISMATCH with
the relative gap. Nothing here is tuned to agree: where a row mismatches it is
left mismatching and the percentage is printed.
"""
import mpmath as mp
from rtodt import (REGIMES, SIGMAS, wzeq_min, A0_for, a_k, z_param, db)

TOL = 0.02          # rows are quoted to 3 significant figures in the paper


def chk(label, got, want, tol=TOL):
    try:
        g = float(got)
    except (TypeError, ValueError):
        print("  %-46s got=%s  want=%s   ??" % (label, got, want)); return
    rel = abs(g - want) / abs(want) if want else abs(g)
    print("  %-46s got=%-13.4g want=%-13.4g %s"
          % (label, g, want, "OK" if rel <= tol else "MISMATCH (%.1f%%)" % (100*rel)))


print("=" * 78)
print("A. GEOMETRY  (manuscript line 938)")
print("=" * 78)
w, weq = wzeq_min()
chk("w_zeq minimum value [m]", weq, 0.0877)
chk("argmin w_z [m]", w, 0.0549)
for s, want in zip(SIGMAS, [0.877, 0.439, 0.219, 0.146]):
    chk("xi_min at sigma_s=%s" % s, weq / (2 * s), want)

print()
print("=" * 78)
print("B. A_0 AT xi=0.992 ACROSS JITTER  (manuscript line 273)")
print("=" * 78)
for s, want in zip(SIGMAS, [0.533, 0.127, 0.0318, 0.0141]):
    chk("A_0(xi=0.992, sigma_s=%s)" % s, A0_for(mp.mpf("0.992"), s), want)

print()
print("=" * 78)
print("C. COEFFICIENT DYNAMIC RANGE, weak regime  (manuscript line 374/380)")
print("=" * 78)
# The manuscript quotes max_k |a_k| at K=10 in the weak regime at the two
# extreme nodes. Two parameters have to be pinned down to make that a single
# number, and neither is stated on the same line:
#
#   sigma_s : taken as 0.1 m, because line 381 -- the sentence that fixes the
#             A_0 range these coefficients are built from -- is explicitly at
#             sigma_s = 0.1 m (block E below reproduces it).
#   family  : eq. (16) defines a_k(alpha, beta, xi); the emulator also uses the
#             argument-swapped a_k(beta, alpha, xi). Both are printed here
#             rather than silently maximised over, because they differ by
#             one to two decades and only one of them can be the tabulated one.
#
# An earlier version of this script maximised over BOTH families and over all
# four sigma_s and K in {10, 20}, and then carried a 35% tolerance. That is not
# the quantity the manuscript defines: the maximum over that larger set is
# 1.8e17 and 9.9e67, i.e. 12 and 36 decades above the printed values, and no
# tolerance makes those rows meaningful. Both are now reported at the default
# tolerance.
A, B = REGIMES["weak"]
S_DYN = mp.mpf("0.1")
K_DYN = 10
for xi, want in [(mp.mpf("0.500"), 5.9e5), (mp.mpf("4.888"), 8.2e31)]:
    A0 = A0_for(xi, S_DYN)
    m_ab = max(abs(a_k(A, B, xi, A0, k)) for k in range(K_DYN + 1))
    m_ba = max(abs(a_k(B, A, xi, A0, k)) for k in range(K_DYN + 1))
    chk("max_k|a_k(alpha,beta)| xi=%s K=10 s=0.1" % xi, m_ab, want)
    print("      argument-swapped family max_k|a_k(beta,alpha)| = %.4g"
          % float(m_ba))

print()
print("=" * 78)
print("D. FLOAT64 ROUND-OFF FLOOR eta_f64  (manuscript line 374)")
print("=" * 78)
# NOTE ON SCOPE: this block multiplies the manuscript's own printed max_k|a_k|
# by eps_mach. It checks that eq. (27) was applied consistently to the numbers
# in the text; it does NOT check this implementation, because no quantity
# computed by this package enters it. The independent check of the same
# quantity is `admissibility_bounds.py`, which recomputes eta_f64 from the
# coefficients rather than from the printed maxima.
eps = mp.mpf(2)**-52
chk("5.9e5 * eps_mach   (paper's own arithmetic)", mp.mpf("5.9e5") * eps, 1.3e-10)
chk("8.2e31 * eps_mach  (paper's own arithmetic)", mp.mpf("8.2e31") * eps, 1.8e16)
print("      ...and from the value block C actually computed:")
A0hi = A0_for(mp.mpf("4.888"), S_DYN)
m_hi = max(abs(a_k(A, B, mp.mpf("4.888"), A0hi, k)) for k in range(K_DYN + 1))
chk("recomputed max_k|a_k| * eps_mach", m_hi * eps, 1.8e16)

print()
print("=" * 78)
print("E. A_0 RANGE AT sigma_s=0.1 m  (manuscript line 381: 0.523 -> 5.2e-3)")
print("=" * 78)
s = mp.mpf("0.1")
chk("A_0 at smallest node xi=0.500", A0_for(mp.mpf("0.500"), s), 0.523)
chk("A_0 at largest node xi=4.888", A0_for(mp.mpf("4.888"), s), 5.2e-3, tol=0.05)

print()
print("=" * 78)
print("F. z AT THE CAMPAIGN OPERATING POINT  (eq:z_worst_campaign)")
print("=" * 78)
A, B = REGIMES["strong"]
A0w = A0_for(mp.mpf("4.888"), mp.mpf("0.1"))
gop = db(38)
chk("z_worst (strong, widest beam, 38 dB)", z_param(A, B, A0w, gop), 4.52, tol=0.05)
# break-even SNR at which z = 8
gbe = (mp.sqrt(2) * A * B / (A0w * 8))**2
chk("break-even SNR for z=8 [dB]", 10 * mp.log10(gbe), 33.0, tol=0.05)
