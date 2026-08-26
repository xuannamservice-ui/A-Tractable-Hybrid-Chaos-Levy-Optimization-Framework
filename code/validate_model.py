"""Reproduce numbers already printed in access.tex, to prove the
re-implementation is faithful before using it for anything new."""
import mpmath as mp
from rtodt import (APERTURE, NODES, REGIMES, SIGMAS, wzeq_min, wz_for_xi,
                   A0_for, A0_of, max_abs_ak, Pe_series, z_param, db)

def chk(label, got, want, tol=0.02):
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
A, B = REGIMES["weak"]
for xi, want in [(mp.mpf("0.500"), 5.9e5), (mp.mpf("4.888"), 8.2e31)]:
    best = None
    for s in SIGMAS:
        A0 = A0_for(xi, s)
        if A0 is None:
            continue
        for K in (10, 20):
            m = max_abs_ak(A, B, xi, A0, K)
            if best is None or m > best[0]:
                best = (m, s, K)
    chk("max_k|a_k| at xi=%s (over sigma_s,K)" % xi, best[0], want, tol=0.35)
    print("      attained at sigma_s=%s, K=%d" % (best[1], best[2]))

print()
print("=" * 78)
print("D. FLOAT64 ROUND-OFF FLOOR eta_f64  (manuscript line 374)")
print("=" * 78)
eps = mp.mpf(2)**-52
chk("5.9e5 * eps_mach", mp.mpf("5.9e5") * eps, 1.3e-10)
chk("8.2e31 * eps_mach", mp.mpf("8.2e31") * eps, 1.8e16)

print()
print("=" * 78)
print("E. A_0 RANGE AT sigma_s=0.1 m  (manuscript line 381: 0.523 -> 5.2e-3)")
print("=" * 78)
s = mp.mpf("0.1")
chk("A_0 at smallest node xi=0.500", A0_for(mp.mpf("0.500"), s), 0.523)
chk("A_0 at largest node xi=4.888", A0_for(mp.mpf("4.888"), s), 5.2e-3, tol=0.05)

print()
print("=" * 78)
print("F. z AT THE CAMPAIGN OPERATING POINT (my inserted eq:z_worst_campaign)")
print("=" * 78)
A, B = REGIMES["strong"]
A0w = A0_for(mp.mpf("4.888"), mp.mpf("0.1"))
gop = db(38)
chk("z_worst (strong, widest beam, 38 dB)", z_param(A, B, A0w, gop), 4.52, tol=0.05)
# break-even SNR at which z = 8
gbe = (mp.sqrt(2) * A * B / (A0w * 8))**2
chk("break-even SNR for z=8 [dB]", 10 * mp.log10(gbe), 33.0, tol=0.05)
