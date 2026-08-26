"""
CONTROLLED A/B ON THE QUANTITY THE OPTIMIZER ACTUALLY MINIMISES.

Scope, stated honestly.  The manuscript's 98% success rate is defined against a
POST-EGC system ABER of 1e-6 (line 969), and the system expression (22) is built
from the lambda_j / C_j recursion of ref. [b13], which is not reproduced in the
paper.  A faithful re-run of that success rate is therefore not possible from the
manuscript alone, and we do not attempt one.

What IS fully specified is the inner loop: line 555 states the optimizer's fitness
is "the microsecond-level ABER evaluation output from RT-ODT", i.e. eq (21).  So
the question "does the interpolated lookup corrupt the search?" is answerable
exactly, and that is what this measures.

Two arms share optimizer, seeds, channel draws and decision box, differing only in

  ARM A  as executed  : interpolated lookup           + threshold-only guard
  ARM B  as described : exact-at-runtime (eq. 68)     + z<=z_max and range test

Metric: the TRUE per-branch ABER (evaluated exactly, 90-digit-validated float64)
of the beam each arm actually selects.  Lower is better.  A kernel that reports
spurious minima will steer the optimizer to a worse beam while claiming a better
number, and that shows up as a gap between REPORTED and TRUE.
"""
import numpy as np
from campaign import (geom, pe_exact, levy, NODES, SIGMAS, ALPHA, BETA,
                      GAMMA_OP, NP_SWARM, T_ITER, ELITE_FRAC, Z_MAX, LUTS,
                      _invert_xi, wzeq_min)

# Per-jitter decision box spanning the manuscript's full xi range. The lower
# edge is max(0.5, min_w w_zeq(w) / (2 sigma_s)); the geometry minimum is
# computed rather than pasted in as the literal 0.0877 it used to be.
_WEQ_MIN = wzeq_min()
BOX = {s: (_invert_xi(max(0.5, _WEQ_MIN / (2 * s)), s), _invert_xi(4.888, s))
       for s in SIGMAS}


def optimise(rng, sigma_s, evaluate, guard_full):
    LO, HI = BOX[sigma_s]
    c = rng.uniform(0.1, 0.9)
    ch = np.empty(NP_SWARM)
    for i in range(NP_SWARM):
        c = 4.0 * c * (1 - c)
        ch[i] = c
    x = LO + ch * (HI - LO)
    v = np.zeros(NP_SWARM)
    n_elite = max(2, int(ELITE_FRAC * NP_SWARM))
    pbest_x, pbest_f = x.copy(), np.full(NP_SWARM, np.inf)
    best_x, best_f = None, np.inf
    saw_invalid = False

    for _ in range(T_ITER):
        x = np.clip(x, LO, HI)
        A0, wzeq = geom(x)
        xi = np.clip(wzeq / (2 * sigma_s), NODES[0], NODES[-1])
        f = evaluate(xi, A0)
        z = np.sqrt(2) * ALPHA * BETA / (A0 * np.sqrt(GAMMA_OP))

        ok = np.isfinite(f)
        if guard_full:
            ok = ok & (z <= Z_MAX) & (f >= 0.0) & (f <= 0.5)
        if np.any(~np.isfinite(f) | (f < 0.0)):
            saw_invalid = True
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
        jump = rng.random(NP_SWARM) < 0.25
        x[jump] += 0.02 * levy(rng, int(jump.sum()))
        order = np.argsort(pbest_f)
        elite = pbest_x[order[:n_elite]]
        if len(elite) >= 2 and np.all(np.isfinite(elite)):
            m = NP_SWARM // 3
            pa, pb = rng.choice(elite, m), rng.choice(elite, m)
            w = rng.random(m)
            x[order[-m:]] = w * pa + (1 - w) * pb
    return best_f, best_x, saw_invalid


def true_pe(wz, sigma_s):
    if wz is None:
        return np.nan
    A0, wzeq = geom(np.array([wz]))
    xi = np.clip(wzeq / (2 * sigma_s), NODES[0], NODES[-1])
    return float(pe_exact(xi, A0)[0])


def run(n_real=1000, seeds=(1, 2, 3, 4, 5)):
    out = {}
    for arm, (exact, guard) in (("A_as_executed", (False, False)),
                                ("B_as_described", (True, True))):
        rep, tru, inv = [], [], 0
        per = n_real // len(seeds)
        for sd in seeds:
            rng = np.random.default_rng(sd)
            for _ in range(per):
                s = SIGMAS[rng.integers(len(SIGMAS))]
                lut = LUTS[s]
                ev = (lambda xi, A0: pe_exact(xi, A0)) if exact else \
                     (lambda xi, A0, L=lut: L.pe(xi))
                f, wz, si = optimise(rng, s, ev, guard)
                rep.append(f)
                tru.append(true_pe(wz, s))
                inv += int(si)
        out[arm] = (np.array(rep), np.array(tru), inv, n_real)
    return out


if __name__ == "__main__":
    res = run()
    print("=" * 78)
    print("QUALITY OF THE BEAM ACTUALLY SELECTED  (per-branch ABER at 38 dB)")
    print("independent re-implementation; NOT a reproduction of the 98% figure")
    print("=" * 78)
    for arm, (rep, tru, inv, n) in res.items():
        good = np.isfinite(tru)
        print("\n  %s   (n=%d)" % (arm, n))
        print("     reported fitness : median %11.4e   min %11.4e"
              % (np.nanmedian(rep), np.nanmin(rep)))
        print("     TRUE ABER of it  : median %11.4e   best %11.4e"
              % (np.nanmedian(tru[good]), np.nanmin(tru[good])))
        print("     cycles where the kernel returned a negative/non-finite value: %.1f%%"
              % (100.0 * inv / n))
        neg = np.sum(rep < 0)
        print("     cycles whose REPORTED optimum was negative (impossible)     : %.1f%%"
              % (100.0 * neg / n))
    a_t = res["A_as_executed"][1]
    b_t = res["B_as_described"][1]
    ga, gb = np.isfinite(a_t), np.isfinite(b_t)
    ma, mb = float(np.median(a_t[ga])), float(np.median(b_t[gb]))
    print("\n" + "=" * 78)
    print("  median TRUE ABER, arm A (as executed) : %.4e" % ma)
    print("  median TRUE ABER, arm B (as described): %.4e" % mb)
    # The direction is READ OFF the two medians. It used to be hardcoded as
    # "arm A is <ratio>x WORSE", which printed "0.84x WORSE" whenever arm A
    # came out ahead -- i.e. the script announced its expected conclusion
    # while displaying the numbers that contradict it.
    ratio = ma / mb
    if ratio > 1.0:
        print("  arm A selects a %.2fx WORSE beam than arm B (lower ABER is better)"
              % ratio)
    elif ratio < 1.0:
        print("  arm A selects a %.2fx BETTER beam than arm B (lower ABER is better)"
              % (1.0 / ratio))
        print("  NOTE: this run does NOT support the hypothesis in the module")
        print("  docstring. The interpolated kernel did not steer the optimizer")
        print("  to a worse beam here; it reported spurious minima (see the")
        print("  negative-optimum line above) without that costing it beam")
        print("  quality on the median draw.")
    else:
        print("  the two arms select beams of identical median quality")
    print("=" * 78)
