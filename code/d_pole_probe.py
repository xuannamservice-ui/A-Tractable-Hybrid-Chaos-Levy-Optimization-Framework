"""
Do the UNCANCELLED poles of the pointing residue D fall inside the admitted
(xi, gbar) region the fidelity ladder actually serves?

THE MECHANISM
-------------
D(alpha,beta,xi) carries Gamma(alpha - xi^2) Gamma(beta - xi^2), whose poles sit
at xi^2 = alpha + k and xi^2 = beta + k for EVERY integer k >= 0 (unbounded in k).
In the untruncated series those poles are cancelled by the matching factor
(xi^2 - beta - k) in the denominator of a_k. The deployed kernel truncates at
K(z), so only k <= K has a partner: every pole with k > K survives in the
TRUNCATED expression.

If such a pole lies inside the region the ladder admits (z <= z_max at that K),
then the kernel has a genuine singularity in its own certified domain, and a
uniform-random off-grid sweep of 77,250 points can miss the narrow neighbourhood
around it while still reporting a small worst-case error.

WHAT THIS SCRIPT DOES
---------------------
Sweeps the physical decision variable w_z on a fine grid (xi and A_0 are both
functions of w_z, so sweeping xi directly would be wrong), forms z at every
gbar on the paper's grid, takes the ladder order K(z), and measures the distance
from xi^2 to the nearest UNCANCELLED pole (k > K). Reports the closest approach
per regime/jitter level, and whether any admitted point lands nearer than the
delta = 0.1 pole-clearance the offline node grid is built to respect.

USAGE
    python d_pole_probe.py
"""
from __future__ import annotations

import numpy as np

from channel import beam_geometry
from hclpso_ga import ladder_order
from mpc_loop import manuscript_wz_box
from rtodt_fast import z_of

REGIMES = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
SIGMAS = (0.05, 0.10, 0.20, 0.30)
GBAR_DB = (20, 24, 28, 30, 34, 38, 40, 44, 48, 50)
DELTA = 0.1          # the pole clearance the offline xi grid is built to


def nearest_uncancelled_pole(xi2, alpha, beta, K):
    """Distance from xi^2 to the closest pole of D that truncation at order K
    leaves without a cancelling partner, i.e. k > K in xi^2 = {alpha,beta} + k."""
    best = np.inf
    for base in (alpha, beta):
        k = xi2 - base                       # the k that would sit on the pole
        for kk in (np.floor(k), np.ceil(k)):
            if kk > K:                       # k <= K is cancelled by a_k
                best = min(best, abs(xi2 - (base + kk)))
    return best


def run():
    np.seterr(all="ignore")
    print(f"{'regime':>9} {'sigma_s':>8} {'K':>3} | {'closest admitted approach':>26} "
          f"| {'at xi':>7} {'gbar':>5} {'z':>6}")
    worst_overall = None
    for name, (alpha, beta) in REGIMES.items():
        for sig in SIGMAS:
            lo, hi = manuscript_wz_box(sig)
            wz = np.linspace(lo, hi, 20001)
            A0, weq = beam_geometry(wz)
            xi = weq / (2.0 * sig)
            xi2 = xi ** 2
            per_K = {}
            for gdb in GBAR_DB:
                g = 10.0 ** (gdb / 10.0)
                z = z_of(alpha, beta, A0, g)
                K = ladder_order(z)
                adm = K > 0                       # -1 encodes inadmissible (z>8)
                for i in np.flatnonzero(adm):
                    d = nearest_uncancelled_pole(float(xi2[i]), alpha, beta, int(K[i]))
                    rec = per_K.get(int(K[i]))
                    if rec is None or d < rec[0]:
                        per_K[int(K[i])] = (d, float(xi[i]), gdb, float(z[i]))
            for K in sorted(per_K):
                d, xi_at, gdb, z_at = per_K[K]
                flag = "  <-- INSIDE pole clearance" if d < DELTA else ""
                print(f"{name:>9} {sig:>8.2f} {K:>3} | {d:>26.4f} "
                      f"| {xi_at:>7.3f} {gdb:>5} {z_at:>6.3f}{flag}")
                if worst_overall is None or d < worst_overall[0]:
                    worst_overall = (d, name, sig, K, xi_at, gdb, z_at)

    d, name, sig, K, xi_at, gdb, z_at = worst_overall
    print(f"\nclosest approach anywhere in the admitted region: {d:.4f} in xi^2 units")
    print(f"  regime={name} sigma_s={sig} K={K} xi={xi_at:.4f} gbar={gdb} dB z={z_at:.3f}")
    print(f"  offline node grid is built to keep >= {DELTA}")
    if d < DELTA:
        print("  VERDICT: uncancelled D poles DO reach the admitted region.")
    else:
        print("  VERDICT: no uncancelled D pole reaches the admitted region;")
        print("           the ladder's z<=z_max test keeps them out as a side effect.")


def corrupt_width(K=10, sig=0.05, gdb=38, K_ref=40):
    """Stage 2: reaching the admitted region is not the same as doing damage.
    A pole is only a hazard over the band of xi where the truncated series is
    BOTH wrong by more than eps_req AND still inside the safety range test, so
    that nothing rejects it.  Measure that band.

    Everything here is 260-digit mpmath, so arithmetic noise is excluded and
    what is measured is the truncation's own singularity: `Pe_series` at the
    deployed order K against the same series at order K_ref, which carries the
    cancelling partner for the pole under test and is therefore regular across
    it.  The deployed float64 kernel is evaluated alongside, to record what the
    guard would actually have seen.
    """
    import mpmath as mp
    from rtodt import Pe_series, wz_for_xi, A0_of
    from rtodt_fast import pe_series_f64

    EPS_REQ = 1e-6
    offsets = [10.0 ** e for e in np.arange(-12.0, -0.4, 0.25)]
    g = 10.0 ** (gdb / 10.0)
    print(f"\nStage 2: hazard band around the first uncancelled pole "
          f"(K={K}, sigma_s={sig}, gbar={gdb} dB, reference order {K_ref})")
    print(f"{'regime':>9} {'pole xi^2':>10} | {'widest |d| wrong':>17} "
          f"| {'widest |d| wrong AND undetected':>32}")
    worst_undetected = 0.0
    for name, (alpha, beta) in REGIMES.items():
        pole = float(beta) + (K + 1)             # first beta-pole past the order
        wrong = undetected = 0.0
        for sgn in (-1.0, +1.0):
            for d in offsets:
                xi = float(np.sqrt(max(pole + sgn * d, 1e-12)))
                wz = wz_for_xi(mp.mpf(str(xi)), mp.mpf(str(sig)))
                if wz is None:
                    continue
                A0 = float(A0_of(wz))
                am, bm, xm, A0m, gm = (mp.mpf(str(v)) for v in (alpha, beta, xi, A0, g))
                got = Pe_series(am, bm, xm, A0m, gm, K)
                ref = Pe_series(am, bm, xm, A0m, gm, K_ref)
                if not mp.isfinite(got) or abs(got - ref) > EPS_REQ:
                    wrong = max(wrong, d)
                    f64 = float(np.atleast_1d(pe_series_f64(
                        alpha, beta, np.atleast_1d(xi), np.atleast_1d(A0), g, K))[0])
                    in_range = np.isfinite(f64) and 0.0 <= f64 <= 0.5
                    if in_range:                  # wrong, finite, and plausible
                        undetected = max(undetected, d)
        worst_undetected = max(worst_undetected, undetected)
        print(f"{name:>9} {pole:>10.3f} | {wrong:>17.2e} | {undetected:>32.2e}")
    print(f"\nwidest band that is wrong AND passes the range test: "
          f"{worst_undetected:.2e} in xi^2 units, against a decision box "
          f"spanning O(10) in xi^2")


if __name__ == "__main__":
    run()
    corrupt_width()
