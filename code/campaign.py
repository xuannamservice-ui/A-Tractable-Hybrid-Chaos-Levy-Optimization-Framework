"""
CONTROLLED RE-RUN OF THE OPTIMIZATION CAMPAIGN.

This is an INDEPENDENT re-implementation built from the manuscript's description
(Sections IV-VI, Tables 4 and 6), not the authors' source code.  It therefore
cannot reproduce their absolute 98%.  What it CAN do, and what it is for, is an
A/B comparison in which the two arms share the optimizer, the channel draws, the
seeds and the success criterion, and differ ONLY in:

    ARM A  "as executed"  : interpolated lookup kernel + range test only
    ARM B  "as described" : exact-at-runtime kernel (eq:ak_factorised) +
                            the per-branch tests of the Section VI-C guard

Everything else is held fixed, so the difference isolates the effect of the
kernel and the guard.

WHERE THE eps_safe THRESHOLD GOES
    An earlier version of this file applied test (iii) of the Section VI-C
    guard -- Pe < eps_safe = 1e-3 -- as an acceptance test INSIDE the swarm
    loop, against the per-branch surrogate.  The per-branch ABER at the
    operating SNR is of order 1e-1, so that test rejected every candidate in
    every cycle of both arms; the script ran to completion and printed a table
    of "0/1000  0.0%  nan", which is not a result, it is a script reporting
    that it rejected its own entire search space.

    eps_safe is a POST-EGC SYSTEM threshold and the surrogate is a per-branch
    quantity; the two are not comparable.  `mpc_loop.envelope_guard` already
    documents and implements that split.  This file now follows it: tests (i)
    and (ii) run in the loop, and eps_safe is evaluated once on the SELECTED
    command and reported.  Nothing has been loosened -- eps_safe is unchanged
    at 1e-3 and the attainment rate against it is printed explicitly, so the
    fact that the per-branch objective never reaches it is stated as a result
    rather than smuggled out as a row of zeros.

Optimizer: H-CLPSO-GA per Section V and Table 4 -- chaotic (logistic) init,
Levy-flight jumps (Mantegna, lambda=1.5), PSO core update, GA crossover on the
top 20% elite, N_p=30, T_iter=25.
Objective: per-branch ABER of eq:aber_emulator at gamma_op = 38 dB.
"""
import numpy as np
from scipy.special import gamma as sp_gamma, erf

# ----------------------------------------------------------------- constants
A_APER = 0.05
ALPHA, BETA = 1.2, 1.1                       # strong turbulence, Table 1
GAMMA_OP = 10 ** (38 / 10)                   # gamma_op = 38 dB, line 969
K = 10
NP_SWARM, T_ITER = 30, 25                    # Table 4
LEVY_LAMBDA = 1.5
ELITE_FRAC = 0.20
Z_MAX = 8.0
EPS_SAFE = 1e-3
NODES = np.array([0.500, 0.628, 0.789, 0.992, 1.266,
                  1.548, 1.967, 2.511, 3.104, 3.912, 4.888])
SIGMAS = [0.05, 0.1, 0.2, 0.3]

# z threshold -> series order, the fidelity ladder of Section V-B4.
LADDER = ((0.5, 5), (2.0, 10), (8.0, 20))


def LADDER_K(z):
    """Per-candidate series order from the conditioning parameter; -1 = reject."""
    out = np.full(np.shape(z), -1, dtype=int)
    for zt, k in reversed(LADDER):
        out = np.where(z <= zt, k, out)
    return out


# ----------------------------------------------------------------- geometry
def geom(wz):
    v = np.sqrt(np.pi / 2) * A_APER / wz
    A0 = erf(v) ** 2
    wzeq = np.sqrt(wz ** 2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v ** 2)))
    return A0, wzeq


def wzeq_min(a=A_APER):
    """Minimum of w_zeq over w_z. w_zeq is non-monotonic in w_z (Farid-
    Hranilovic), and this minimum sets the attainable floor on xi at any
    jitter level: xi >= wzeq_min / (2 sigma_s)."""
    lo, hi = 5e-3, 0.5
    gr = (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f = lambda w: geom(np.array([w]))[1][0]
    for _ in range(200):
        if f(c) < f(d):
            hi = d
        else:
            lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    return float(f(0.5 * (lo + hi)))


# --------------------------------------------------------- exact kernel (B)
def _Kc(A, B, K):
    k = np.arange(K + 1)
    # (-1)^k (AB)^(B+k) Gamma(A-B-k) / (k! Gamma(A) Gamma(B))   -- xi-free
    g = sp_gamma(A - B - k)
    return ((-1.0) ** k * (A * B) ** (B + k) * g
            / (sp_gamma(k + 1.0) * sp_gamma(A) * sp_gamma(B)))


def _Cmom(s, gbar):
    s = np.asarray(s, dtype=float)
    return sp_gamma((s + 1) / 2) / (2 * s * np.sqrt(np.pi)) * (2 / gbar) ** (s / 2)


_ORDER_CACHE = {}


def _order_tables(order):
    """(Kc_AB, Kc_BA, C_B, C_A, k) for one series order, cached."""
    if order not in _ORDER_CACHE:
        k = np.arange(order + 1)
        _ORDER_CACHE[order] = (_Kc(ALPHA, BETA, order), _Kc(BETA, ALPHA, order),
                               _Cmom(BETA + k, GAMMA_OP), _Cmom(ALPHA + k, GAMMA_OP), k)
    return _ORDER_CACHE[order]


# retained for callers that want the K=10 tables directly
KC_AB, KC_BA, C_B, C_A, _k = _order_tables(K)


def pe_exact(xi, A0, order=K):
    """eq:ak_factorised evaluated in closed form. No interpolation.

    `order` may be a scalar or a per-candidate integer array (the fidelity
    ladder); entries with order < 0 are inadmissible and return NaN.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    A0 = np.atleast_1d(np.asarray(A0, dtype=float))
    order = np.atleast_1d(np.asarray(order, dtype=int))
    if order.size == 1:
        order = np.full(xi.shape, int(order[0]))

    out = np.full(xi.shape, np.nan)
    for o in np.unique(order):
        if o < 0:
            continue
        m = order == o
        kcAB, kcBA, CB, CA, k = _order_tables(int(o))
        x2 = xi[m] * xi[m]
        a0 = A0[m]
        t1 = kcAB[None, :] * x2[:, None] / ((x2[:, None] - BETA - k[None, :])
                                            * a0[:, None] ** (BETA + k)[None, :])
        t2 = kcBA[None, :] * x2[:, None] / ((x2[:, None] - ALPHA - k[None, :])
                                            * a0[:, None] ** (ALPHA + k)[None, :])
        tot = (t1 * CB[None, :]).sum(1) + (t2 * CA[None, :]).sum(1)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            D = (x2 * (ALPHA * BETA) ** x2 * sp_gamma(ALPHA - x2) * sp_gamma(BETA - x2)
                 / (a0 ** x2 * sp_gamma(ALPHA) * sp_gamma(BETA)))
            out[m] = tot + D * _Cmom(x2, GAMMA_OP)
    return out


# -------------------------------------------------- interpolated kernel (A)
class Lookup:
    """The deployed tensor: a_k, D and C(xi^2) tabulated at the 11 nodes."""

    def __init__(self, sigma_s):
        wz = np.array([_invert_xi(x, sigma_s) for x in NODES])
        A0n, _ = geom(wz)
        self.A0n = A0n
        x2 = NODES ** 2
        self.ak1 = KC_AB[None, :] * x2[:, None] / (
            (x2[:, None] - BETA - _k[None, :]) * A0n[:, None] ** (BETA + _k)[None, :])
        self.ak2 = KC_BA[None, :] * x2[:, None] / (
            (x2[:, None] - ALPHA - _k[None, :]) * A0n[:, None] ** (ALPHA + _k)[None, :])
        self.D = (x2 * (ALPHA * BETA) ** x2 * sp_gamma(ALPHA - x2) * sp_gamma(BETA - x2)
                  / (A0n ** x2 * sp_gamma(ALPHA) * sp_gamma(BETA)))
        self.Cx = _Cmom(x2, GAMMA_OP)

    def pe(self, xi):
        j = np.clip(np.searchsorted(NODES, xi) - 1, 0, len(NODES) - 2)
        t = (xi - NODES[j]) / (NODES[j + 1] - NODES[j])
        a1 = self.ak1[j] + t[:, None] * (self.ak1[j + 1] - self.ak1[j])
        a2 = self.ak2[j] + t[:, None] * (self.ak2[j + 1] - self.ak2[j])
        D = self.D[j] + t * (self.D[j + 1] - self.D[j])
        C = self.Cx[j] + t * (self.Cx[j + 1] - self.Cx[j])
        return (a1 * C_B[None, :]).sum(1) + (a2 * C_A[None, :]).sum(1) + D * C


def _invert_xi(xi_target, sigma_s):
    """beam waist giving this xi at this jitter (upper branch)."""
    lo, hi = 0.0549, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if geom(mid)[1] / (2 * sigma_s) < xi_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- H-CLPSO-GA
def levy(rng, n, lam=LEVY_LAMBDA):
    sig = (sp_gamma(1 + lam) * np.sin(np.pi * lam / 2)
           / (sp_gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    u = rng.normal(0, sig, n)
    v = np.abs(rng.normal(0, 1, n))
    return u / v ** (1 / lam)


def optimise(rng, sigma_s, evaluate, guard_full):
    """One control cycle. Returns (best_reported_pe, best_wz, scored_invalid).

    `guard_full` selects the guard FORM, per mpc_loop.envelope_guard:
        True  -> tests (i) z <= z_max and (ii) 0 <= Pe <= 1/2
        False -> test (ii) alone
    Test (iii), Pe < eps_safe, is post-EGC and is applied by the caller to the
    selected command, not here; see the module docstring.
    """
    lo, hi = 0.055, 0.60                       # beam-waist box
    # chaotic (logistic) initialisation
    c = rng.uniform(0.1, 0.9)
    ch = np.empty(NP_SWARM)
    for i in range(NP_SWARM):
        c = 4.0 * c * (1 - c)
        ch[i] = c
    x = lo + ch * (hi - lo)
    v = np.zeros(NP_SWARM)

    best_x, best_f = None, np.inf
    scored_invalid = False
    n_elite = max(2, int(ELITE_FRAC * NP_SWARM))

    pbest_x, pbest_f = x.copy(), np.full(NP_SWARM, np.inf)
    for it in range(T_ITER):
        x = np.clip(x, lo, hi)
        A0, wzeq = geom(x)
        xi = wzeq / (2 * sigma_s)
        xi = np.clip(xi, NODES[0], NODES[-1])
        f = evaluate(xi, A0)

        z = np.sqrt(2) * ALPHA * BETA / (A0 * np.sqrt(GAMMA_OP))
        ok = np.isfinite(f)
        if guard_full:                          # tests (i) + (ii)
            ok &= (z <= Z_MAX) & (f >= 0.0) & (f <= 0.5)
        else:                                   # test (ii) only, as executed
            ok &= (f >= 0.0) & (f <= 0.5)
        fw = np.where(ok, f, np.inf)

        imp = fw < pbest_f
        pbest_f[imp], pbest_x[imp] = fw[imp], x[imp]
        i = int(np.argmin(fw))
        if fw[i] < best_f:
            best_f, best_x = float(fw[i]), float(x[i])
            if not (0.0 <= f[i] <= 0.5):
                scored_invalid = True

        # PSO core + Levy jumps + GA crossover on elites
        r1, r2 = rng.random(NP_SWARM), rng.random(NP_SWARM)
        gb = best_x if best_x is not None else x[i]
        v = 0.7 * v + 1.5 * r1 * (pbest_x - x) + 1.5 * r2 * (gb - x)
        x = x + v
        jump = rng.random(NP_SWARM) < 0.25
        x[jump] += 0.02 * levy(rng, int(jump.sum()))
        order = np.argsort(pbest_f)
        elite = pbest_x[order[:n_elite]]
        if len(elite) >= 2:
            pa = rng.choice(elite, NP_SWARM // 3)
            pb = rng.choice(elite, NP_SWARM // 3)
            w = rng.random(NP_SWARM // 3)
            x[order[-(NP_SWARM // 3):]] = w * pa + (1 - w) * pb
    return best_f, best_x, scored_invalid


# ------------------------------------------------------------------- run
def campaign(target, n_real=1000, seeds=(1, 2, 3, 4, 5), verbose=True):
    """`target` is the per-branch success threshold and is REQUIRED.

    It used to default to None, which made `f <= target` a TypeError waiting
    on any call that reached it -- masked only because the in-loop eps_safe
    test meant nothing ever did.
    """
    res = {}
    for arm, (use_exact, guard_full) in (("A_as_executed", (False, False)),
                                         ("B_as_described", (True, True))):
        succ = tot = invalid_scored = eps_safe_ok = 0
        true_pe_of_success = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            for r in range(n_real // len(seeds)):
                sigma_s = SIGMAS[rng.integers(len(SIGMAS))]
                lut = LUTS[sigma_s]
                ev = ((lambda xi, A0: pe_exact(xi, A0)) if use_exact else
                      (lambda xi, A0, L=lut: L.pe(xi)))
                f, wz, inv = optimise(rng, sigma_s, ev, guard_full)
                tot += 1
                if inv:
                    invalid_scored += 1
                if wz is not None and np.isfinite(f):
                    A0t, wzeqt = geom(np.array([wz]))
                    xit = np.clip(wzeqt / (2 * sigma_s), NODES[0], NODES[-1])
                    true_pe = float(pe_exact(xit, A0t)[0])
                    # test (iii) of the guard, applied where it belongs: once,
                    # to the selected command. Unchanged at 1e-3.
                    if true_pe < EPS_SAFE:
                        eps_safe_ok += 1
                    if f <= target:
                        succ += 1
                        true_pe_of_success.append(true_pe)
        res[arm] = dict(succ=succ, tot=tot, invalid=invalid_scored,
                        eps_safe=eps_safe_ok,
                        true=np.array(true_pe_of_success))
    return res


LUTS = {s: Lookup(s) for s in SIGMAS}

if __name__ == "__main__":
    # calibrate a per-branch target that the exact kernel can sometimes meet
    wz = np.linspace(0.055, 0.60, 4000)
    A0, wzeq = geom(wz)
    best = {}
    for s in SIGMAS:
        xi = np.clip(wzeq / (2 * s), NODES[0], NODES[-1])
        best[s] = float(np.nanmin(pe_exact(xi, A0)))
    print("Best achievable per-branch ABER at 38 dB, by jitter level:")
    for s in SIGMAS:
        print("   sigma_s=%-5s  min Pe = %.3e" % (s, best[s]))
    TARGET = float(np.median(list(best.values())))
    print("\nAdopted per-branch success threshold: %.3e" % TARGET)
    print("(the median over jitter levels of the best per-branch ABER the box")
    print(" contains, so roughly half the draws are winnable and the two arms")
    print(" are separated by search quality rather than by the threshold.)")
    print("(the manuscript's 1e-6 is a POST-EGC system figure; the per-branch")
    print(" objective cannot reach it, so the threshold is calibrated here.)\n")

    out = campaign(TARGET, n_real=1000, seeds=(1, 2, 3, 4, 5))
    print("=" * 88)
    print("%-16s %8s %10s %14s %14s %16s"
          % ("arm", "success", "rate", "scored on", "clears", "true Pe of"))
    print("%-16s %8s %10s %14s %14s %16s"
          % ("", "", "", "invalid val", "eps_safe=1e-3", "'successes'"))
    print("=" * 88)
    for arm, d in out.items():
        med = np.median(d["true"]) if len(d["true"]) else float("nan")
        print("%-16s %5d/%-4d %8.1f%% %11.1f%% %11.1f%% %16.3e"
              % (arm, d["succ"], d["tot"], 100 * d["succ"] / d["tot"],
                 100 * d["invalid"] / d["tot"], 100 * d["eps_safe"] / d["tot"], med))
    print("=" * 88)
    print("The eps_safe column is test (iii) of the Section VI-C guard applied")
    print("to the selected command. A 0.0% there is the real finding of this")
    print("script: the per-branch surrogate at 38 dB is of order 1e-1 and does")
    print("not approach a 1e-3 threshold, which is why that test cannot live")
    print("inside the swarm loop.")
