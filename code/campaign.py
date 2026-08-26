"""
CONTROLLED RE-RUN OF THE OPTIMIZATION CAMPAIGN.

This is an INDEPENDENT re-implementation built from the manuscript's description
(Sections IV-VI, Tables 4 and 6), not the authors' source code.  It therefore
cannot reproduce their absolute 98%.  What it CAN do, and what it is for, is an
A/B comparison in which the two arms share the optimizer, the channel draws, the
seeds and the success criterion, and differ ONLY in:

    ARM A  "as executed"  : interpolated lookup kernel + threshold-only guard
                            (Pe < eps_safe), i.e. the campaign of Section VII
    ARM B  "as described" : exact-at-runtime kernel (eq:ak_factorised) +
                            the full three-part guard of Section VI-C

Everything else is held fixed, so the difference isolates the effect of the
kernel and the guard.

Optimizer: H-CLPSO-GA per Section V and Table 4 -- chaotic (logistic) init,
Levy-flight jumps (Mantegna, lambda=1.5), PSO core update, GA crossover on the
top 20% elite, N_p=30, T_iter=25.
Objective: per-branch ABER of eq:aber_emulator at gamma_op = 38 dB.
"""
import numpy as np
from scipy.special import gamma as sp_gamma, erf, gammaln

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


# ----------------------------------------------------------------- geometry
def geom(wz):
    v = np.sqrt(np.pi / 2) * A_APER / wz
    A0 = erf(v) ** 2
    wzeq = np.sqrt(wz ** 2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v ** 2)))
    return A0, wzeq


# --------------------------------------------------------- exact kernel (B)
def _Kc(A, B, K):
    k = np.arange(K + 1)
    # (-1)^k (AB)^(B+k) Gamma(A-B-k) / (k! Gamma(A) Gamma(B))   -- xi-free
    g = sp_gamma(A - B - k)
    return ((-1.0) ** k * (A * B) ** (B + k) * g
            / (sp_gamma(k + 1.0) * sp_gamma(A) * sp_gamma(B)))


KC_AB = _Kc(ALPHA, BETA, K)
KC_BA = _Kc(BETA, ALPHA, K)
_k = np.arange(K + 1)


def _Cmom(s, gbar):
    s = np.asarray(s, dtype=float)
    return sp_gamma((s + 1) / 2) / (2 * s * np.sqrt(np.pi)) * (2 / gbar) ** (s / 2)


C_B = _Cmom(BETA + _k, GAMMA_OP)
C_A = _Cmom(ALPHA + _k, GAMMA_OP)


def pe_exact(xi, A0):
    """eq:ak_factorised evaluated in closed form. No interpolation."""
    x2 = xi * xi
    t1 = KC_AB[None, :] * x2[:, None] / ((x2[:, None] - BETA - _k[None, :])
                                         * A0[:, None] ** (BETA + _k)[None, :])
    t2 = KC_BA[None, :] * x2[:, None] / ((x2[:, None] - ALPHA - _k[None, :])
                                         * A0[:, None] ** (ALPHA + _k)[None, :])
    tot = (t1 * C_B[None, :]).sum(1) + (t2 * C_A[None, :]).sum(1)
    D = (x2 * (ALPHA * BETA) ** x2 * sp_gamma(ALPHA - x2) * sp_gamma(BETA - x2)
         / (A0 ** x2 * sp_gamma(ALPHA) * sp_gamma(BETA)))
    return tot + D * _Cmom(x2, GAMMA_OP)


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
    """One control cycle. Returns (best_reported_pe, best_wz, scored_invalid)."""
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
        if guard_full:                          # three-part guard
            ok &= (z <= Z_MAX) & (f >= 0.0) & (f <= 0.5) & (f < EPS_SAFE)
        else:                                   # threshold-only, as executed
            ok &= (f < EPS_SAFE)
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
def campaign(n_real=1000, seeds=(1, 2, 3, 4, 5), target=None, verbose=True):
    res = {}
    for arm, (use_exact, guard_full) in (("A_as_executed", (False, False)),
                                         ("B_as_described", (True, True))):
        succ = tot = invalid_scored = 0
        true_pe_of_success = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            for r in range(n_real // len(seeds)):
                sigma_s = SIGMAS[rng.integers(len(SIGMAS))]
                lut = LUTS[sigma_s]
                ev = (lambda xi, A0: pe_exact(xi, A0)) if use_exact else \
                     (lambda xi, A0: lut.pe(xi))
                f, wz, inv = optimise(rng, sigma_s, ev, guard_full)
                tot += 1
                if inv:
                    invalid_scored += 1
                if wz is not None and np.isfinite(f):
                    A0t, wzeqt = geom(np.array([wz]))
                    xit = np.clip(wzeqt / (2 * sigma_s), NODES[0], NODES[-1])
                    true_pe = float(pe_exact(xit, A0t)[0])
                    if f <= target:
                        succ += 1
                        true_pe_of_success.append(true_pe)
        res[arm] = dict(succ=succ, tot=tot, invalid=invalid_scored,
                        true=np.array(true_pe_of_success))
    return res


LUTS = {s: Lookup(s) for s in SIGMAS}

if __name__ == "__main__":
    # calibrate a per-branch target that the exact kernel can sometimes meet
    rng = np.random.default_rng(0)
    wz = np.linspace(0.055, 0.60, 4000)
    A0, wzeq = geom(wz)
    best = {}
    for s in SIGMAS:
        xi = np.clip(wzeq / (2 * s), NODES[0], NODES[-1])
        best[s] = float(np.nanmin(pe_exact(xi, A0)))
    print("Best achievable per-branch ABER at 38 dB, by jitter level:")
    for s in SIGMAS:
        print("   sigma_s=%-5s  min Pe = %.3e" % (s, best[s]))
    TARGET = 10 ** (np.floor(np.log10(np.median(list(best.values())))) + 1)
    print("\nAdopted per-branch success threshold: %.1e" % TARGET)
    print("(the manuscript's 1e-6 is a POST-EGC system figure; the per-branch")
    print(" objective cannot reach it, so the threshold is calibrated here.)\n")

    out = campaign(n_real=1000, seeds=(1, 2, 3, 4, 5), target=TARGET)
    print("=" * 76)
    print("%-16s %8s %10s %14s %16s" % ("arm", "success", "rate", "scored on", "true Pe of"))
    print("%-16s %8s %10s %14s %16s" % ("", "", "", "invalid val", "'successes'"))
    print("=" * 76)
    for arm, d in out.items():
        med = np.median(d["true"]) if len(d["true"]) else float("nan")
        print("%-16s %5d/%-4d %8.1f%% %11.1f%% %16.3e"
              % (arm, d["succ"], d["tot"], 100 * d["succ"] / d["tot"],
                 100 * d["invalid"] / d["tot"], med))
    print("=" * 76)
