"""
Is the c = 0 assumption behind Table 11's p-values plausible?

Table 11 assumes ONE-SIDED discordance: every realization on which an ablated
variant succeeds is also one on which the full kernel succeeds, so c = 0 and
p = 2 * 2^-b.  For a stochastic solver a referee will doubt that.

This cannot reproduce the authors' (b, c) -- those belong to their campaign.
What it CAN do is settle the QUALITATIVE question on an equivalent solver:
run the full kernel and each ablated variant on the SAME paired channel draws
with the SAME seeds, and count the discordant pairs.

  b = # realizations where full succeeds and variant fails
  c = # realizations where variant succeeds and full fails

If c is reliably 0, the assumption is defensible. If c > 0, the tabulated
p-values are anti-conservative and must be recomputed from measured counts.

Usage:
    python ablation_bc.py [--realizations 1000] [--out ablation_success.npz]

The .npz it writes is the input `check_table11_statistics.py` consumes; the
two scripts together close the loop from paired indicators to the exact McNemar
and Clopper-Pearson statistics. That is a check on the statistics, not a
reproduction of the published counts -- see that file's docstring.
"""
import argparse

import numpy as np

from campaign import (geom, pe_exact, NODES, SIGMAS, ALPHA, BETA, GAMMA_OP,
                      NP_SWARM, T_ITER, ELITE_FRAC, Z_MAX, LADDER_K,
                      levy, _invert_xi, wzeq_min)

# Decision box: the manuscript's xi range at each jitter level. The lower edge
# is max(0.5, xi_min(sigma_s)) where xi_min = min_w w_zeq(w) / (2 sigma_s); the
# minimum is computed from the geometry rather than pasted in as 0.0877, which
# is what this line used to do.
_WEQ_MIN = wzeq_min()
BOX = {s: (_invert_xi(max(0.5, _WEQ_MIN / (2 * s)), s), _invert_xi(4.888, s))
       for s in SIGMAS}

VARIANTS = ["full", "no_chaos", "no_levy", "no_ga", "fixed_fidelity"]


def optimise(rng, sigma_s, variant):
    LO, HI = BOX[sigma_s]
    n_elite = max(2, int(ELITE_FRAC * NP_SWARM))

    if variant == "no_chaos":
        x = rng.uniform(LO, HI, NP_SWARM)          # uniform instead of logistic
    else:
        c = rng.uniform(0.1, 0.9)
        ch = np.empty(NP_SWARM)
        for i in range(NP_SWARM):
            c = 4.0 * c * (1 - c)
            ch[i] = c
        x = LO + ch * (HI - LO)

    v = np.zeros(NP_SWARM)
    pbest_x, pbest_f = x.copy(), np.full(NP_SWARM, np.inf)
    best_x, best_f = None, np.inf

    for _ in range(T_ITER):
        x = np.clip(x, LO, HI)
        A0, wzeq = geom(x)
        xi = np.clip(wzeq / (2 * sigma_s), NODES[0], NODES[-1])
        z = np.sqrt(2) * ALPHA * BETA / (A0 * np.sqrt(GAMMA_OP))
        # The fidelity-ladder ablation: 'fixed_fidelity' pins K=10 for every
        # candidate, every other arm selects K per candidate from z. Before
        # this branch existed the variant was scored with the same fixed K=10
        # as every other arm, so it was a copy of 'full' under a different
        # name -- it always returned b = c = 0 and was reported as an ablation
        # that had been tested.
        K = np.full(z.shape, 10, dtype=int) if variant == "fixed_fidelity" else LADDER_K(z)
        f = pe_exact(xi, A0, K)
        ok = np.isfinite(f) & (z <= Z_MAX) & (f >= 0.0) & (f <= 0.5)
        fw = np.where(ok, f, np.inf)

        imp = fw < pbest_f
        pbest_f[imp], pbest_x[imp] = fw[imp], x[imp]
        i = int(np.argmin(fw))
        if fw[i] < best_f:
            best_f, best_x = float(fw[i]), float(x[i])

        r1, r2 = rng.random(NP_SWARM), rng.random(NP_SWARM)
        gb = best_x if best_x is not None else x[i]
        v = 0.7 * v + 1.5 * r1 * (pbest_x - x) + 1.5 * r2 * (gb - x)
        x = x + v

        if variant != "no_levy":
            jump = rng.random(NP_SWARM) < 0.25
            k = int(jump.sum())
            if k:
                x[jump] += 0.02 * levy(rng, k)
        else:
            jump = rng.random(NP_SWARM) < 0.25       # Gaussian instead of Levy
            k = int(jump.sum())
            if k:
                x[jump] += 0.02 * rng.normal(0, 1, k)

        if variant != "no_ga":
            order = np.argsort(pbest_f)
            elite = pbest_x[order[:n_elite]]
            if len(elite) >= 2 and np.all(np.isfinite(elite)):
                m = NP_SWARM // 3
                pa, pb = rng.choice(elite, m), rng.choice(elite, m)
                w = rng.random(m)
                x[order[-m:]] = w * pa + (1 - w) * pb
    return best_f


def run(n=1000, seed0=90210, out=None):
    # a common target: median of the full kernel's achievable optimum
    probe = []
    for r in range(120):
        rng = np.random.default_rng(seed0 + r)
        s = SIGMAS[rng.integers(len(SIGMAS))]
        probe.append(optimise(rng, s, "full"))
    probe = np.array([p for p in probe if np.isfinite(p)])
    target = float(np.quantile(probe, 0.55))
    print("paired ablation, n=%d realizations, success threshold Pe <= %.4e"
          % (n, target))
    print("(threshold set so the full kernel succeeds on ~55% of draws, giving")
    print(" both b and c room to be non-zero -- the least favourable case for c=0)\n")

    succ = {v: np.zeros(n, dtype=bool) for v in VARIANTS}
    for r in range(n):
        base = seed0 + 1000 + r
        s = SIGMAS[np.random.default_rng(base).integers(len(SIGMAS))]
        for var in VARIANTS:
            rng = np.random.default_rng(base)      # SAME draw for every variant
            f = optimise(rng, s, var)
            succ[var][r] = np.isfinite(f) and f <= target

    print("  %-16s %7s %6s %6s   %s" % ("variant", "success", "b", "c", "verdict"))
    print("  " + "-" * 62)
    full = succ["full"]
    print("  %-16s %6.1f%%" % ("full kernel", 100 * full.mean()))
    any_c = False
    for var in VARIANTS[1:]:
        v = succ[var]
        b = int(np.sum(full & ~v))
        c = int(np.sum(~full & v))
        any_c |= c > 0
        verdict = "c=0 holds" if c == 0 else "c>0 -- assumption FAILS"
        print("  %-16s %6.1f%% %6d %6d   %s" % (var, 100 * v.mean(), b, c, verdict))
    print()
    print("  On this solver the c=0 assumption %s."
          % ("FAILS on at least one arm" if any_c else "holds on every arm"))

    if out:
        np.savez(out, **succ)
        print("  wrote %s  (feed it to check_table11_statistics.py)" % out)
    return succ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations", type=int, default=1000)
    ap.add_argument("--out", default=None,
                    help="write the paired indicators to this .npz")
    a = ap.parse_args()
    run(n=a.realizations, out=a.out)
