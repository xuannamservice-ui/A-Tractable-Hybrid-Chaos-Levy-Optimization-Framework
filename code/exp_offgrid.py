"""
EXPERIMENT 1 -- off-grid interpolation error of the deployed RT-ODT lookup scheme.

Deployed scheme (Appendix B Step 3 + tensor layout of the Data Availability section):
  tabulate at the 11 pole-free nodes:  a_k(a,b,xi_j), a_k(b,a,xi_j), D(xi_j), C(xi_j^2,gbar)
  at runtime, linearly interpolate each in xi, then combine per eq:aber_emulator.
The Bessel weights C(b+k), C(a+k) are xi-independent and therefore exact.

We decompose the error into the Bessel part and the residue (D*C) part, to test the
mechanism claim, and we evaluate the proposed remedy: tabulate the PRODUCT D*C(xi^2)
as a single quantity instead of two factors.

Restricted to the admissible band z <= 8, which is where Table 7 claims to hold.
"""
import mpmath as mp
import random
from rtodt import (NODES, REGIMES, A0_for, a_k, D_coef, C_moment,
                   z_param, db)

mp.mp.dps = 90
random.seed(20260825)

SIGMA = mp.mpf("0.1")       # nominal jitter
K = 10                      # ladder rung serving z <= 2; also the Table 7 headline order
ZMAX = mp.mpf(8)


def node_tables(A, B, gbar, sigma):
    """Everything the deployed lookup stores, evaluated exactly at the nodes."""
    T = {"ak1": [], "ak2": [], "D": [], "Cx": [], "DC": [], "A0": []}
    for xj in NODES:
        A0 = A0_for(xj, sigma)
        T["A0"].append(A0)
        T["ak1"].append([a_k(A, B, xj, A0, k) for k in range(K + 1)])
        T["ak2"].append([a_k(B, A, xj, A0, k) for k in range(K + 1)])
        d = D_coef(A, B, xj, A0)
        c = C_moment(xj**2, gbar)
        T["D"].append(d)
        T["Cx"].append(c)
        T["DC"].append(d * c)
    return T


def lerp(lo, hi, t):
    return lo + t * (hi - lo)


def bracket(xi):
    for j in range(len(NODES) - 1):
        if NODES[j] <= xi <= NODES[j + 1]:
            t = (xi - NODES[j]) / (NODES[j + 1] - NODES[j])
            return j, t
    raise ValueError("outside grid")


def evaluate(xi, A, B, gbar, sigma, T):
    """Return (exact, deployed, fixed, parts) at off-grid xi."""
    A0 = A0_for(xi, sigma)
    j, t = bracket(xi)
    x2 = xi**2

    CB = [C_moment(B + k, gbar) for k in range(K + 1)]   # xi-independent -> exact
    CA = [C_moment(A + k, gbar) for k in range(K + 1)]

    # ---- exact
    bess_ex = sum(a_k(A, B, xi, A0, k) * CB[k] for k in range(K + 1)) \
            + sum(a_k(B, A, xi, A0, k) * CA[k] for k in range(K + 1))
    res_ex = D_coef(A, B, xi, A0) * C_moment(x2, gbar)
    exact = bess_ex + res_ex

    # ---- deployed: interpolate a_k, D, C(xi^2) separately
    bess_ip = sum(lerp(T["ak1"][j][k], T["ak1"][j + 1][k], t) * CB[k] for k in range(K + 1)) \
            + sum(lerp(T["ak2"][j][k], T["ak2"][j + 1][k], t) * CA[k] for k in range(K + 1))
    res_ip = lerp(T["D"][j], T["D"][j + 1], t) * lerp(T["Cx"][j], T["Cx"][j + 1], t)
    deployed = bess_ip + res_ip

    # ---- proposed fix: interpolate the product D*C as one tabulated quantity
    res_fx = lerp(T["DC"][j], T["DC"][j + 1], t)
    fixed = bess_ip + res_fx

    parts = {
        "bessel_err": abs(bess_ip - bess_ex),
        "residue_err_deployed": abs(res_ip - res_ex),
        "residue_err_fixed": abs(res_fx - res_ex),
        "exact_mag": abs(exact),
    }
    return exact, deployed, fixed, parts


def run(regime, gbar_db, n=120):
    A, B = REGIMES[regime]
    gbar = db(gbar_db)
    T = node_tables(A, B, gbar, SIGMA)

    samples = []
    tries = 0
    while len(samples) < n and tries < 60 * n:
        tries += 1
        xi = NODES[0] + (NODES[-1] - NODES[0]) * mp.mpf(random.random())
        A0 = A0_for(xi, SIGMA)
        if A0 is None or z_param(A, B, A0, gbar) > ZMAX:
            continue
        samples.append(xi)
    if not samples:
        print("  %-9s %5s dB : no admissible off-grid samples" % (regime, gbar_db))
        return

    worst_dep = worst_fix = mp.mpf(0)
    worst_bess = worst_res = mp.mpf(0)
    wx = None
    for xi in samples:
        ex, dep, fx, p = evaluate(xi, A, B, gbar, SIGMA, T)
        ed, ef = abs(dep - ex), abs(fx - ex)
        if ed > worst_dep:
            worst_dep, wx = ed, xi
        worst_fix = max(worst_fix, ef)
        worst_bess = max(worst_bess, p["bessel_err"])
        worst_res = max(worst_res, p["residue_err_deployed"])

    print("  %-9s %2d dB  n=%3d  xi in [%.3f, %.3f]" %
          (regime, gbar_db, len(samples), float(min(samples)), float(max(samples))))
    print("      max |err| deployed (interp D and C separately) : %.3e" % float(worst_dep))
    print("      max |err| fixed    (interp the product D*C)    : %.3e" % float(worst_fix))
    print("      ...decomposed: Bessel-part %.3e | residue-part %.3e"
          % (float(worst_bess), float(worst_res)))
    if worst_fix > 0:
        print("      improvement factor from the fix                : %.3g x"
              % float(worst_dep / worst_fix))
    print("      worst xi = %.4f" % float(wx))


def main():
    print("=" * 78)
    print("OFF-GRID INTERPOLATION ERROR, K=%d, sigma_s=%s, admissible band z<=%s"
          % (K, SIGMA, ZMAX))
    print("Table 7 certified per-branch bounds at K=10: 3.98e-9 / 5.49e-10 / 7.87e-10")
    print("=" * 78)
    for reg in ("weak", "moderate", "strong"):
        for g in (30, 40):
            run(reg, g)
        print()


# Guarded: exp_diagnose.py and exp_exact_runtime.py import node_tables/evaluate
# from this module. Before the guard existed, merely importing it re-ran the
# whole ~2-minute experiment and printed its output ahead of theirs.
if __name__ == "__main__":
    main()
