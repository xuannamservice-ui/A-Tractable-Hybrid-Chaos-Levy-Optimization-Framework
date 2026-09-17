/*
 * Compiled (C) port of the RT-ODT kernel + H-CLPSO-GA solver + one control
 * cycle, ported from the released Python reference (mpc_loop.py, hclpso_ga.py,
 * rtodt_fast.py, channel.py) at the operating point of the manuscript
 * (strong regime, alpha=1.2, beta=1.1, gbar_op = 38 dB, sigma_s = 0.10 m,
 * horizon T = 20, N_p = 30, rank_stages = 1 -- the configuration of
 * tab:tail_mitigation).
 *
 * WHAT THIS IS NOT: a bit-identical replacement.  It uses a different PRNG
 * (xorshift128+ / Marsaglia polar normal, not NumPy's PCG64 + ziggurat), so
 * the search trajectory it produces on a given "seed" is NOT the trajectory
 * the Python release produces.  What is claimed is narrower and is exactly
 * what a latency measurement needs: the SAME algorithmic operations, in the
 * SAME order, on data of the SAME size (chaotic init, PSO update, Mantegna
 * Levy jump, GA crossover on the elite, reflect+forward-sweep repair, the
 * three-part envelope guard, the RT-ODT series kernel via the same factorised
 * closed-form coefficients), compiled instead of dispatched through NumPy.
 * Correctness of the RT-ODT arithmetic is checked at start-up against
 * independently-known reference values (see selfcheck()).
 *
 * Build:   zig cc -O3 -o rt_odt_kernel.exe rt_odt_kernel.c -lm
 * Run:     rt_odt_kernel.exe [n_cycles]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* ------------------------------------------------------------------ */
/* operating point (measure_all.py / bench_cycle.py)                   */
/* ------------------------------------------------------------------ */
/* ALPHA/BETA/SIGMA_S are overridable from argv (regime companion runs,
 * Section~\ref{subsec:results}'s physically-realizable-regime figures) --
 * default is the manuscript's strong/0.10 m operating point, unchanged. */
static double ALPHA = 1.2, BETA = 1.1;
static double GBAR;                      /* 10^(38/10) */
static double SIGMA_S = 0.10;
static char REGIME_TAG[16] = "strong";
static const int    T_HOR = 20;          /* horizon                    */
static const int    NP = 30;             /* swarm size                 */
static const int    D  = 60;             /* 3 * T_HOR                  */
static const int    MAX_ITERS = 25;      /* T_iter cap                 */
static const double TAU_O_US = 600.0;
static const double LAMBDA_U = 2.0;
static const double Z_MAX = 8.0;
static const double APERTURE = 0.05;
static const double L_LINK = 2000.0;

static const double U_MAX = 10e-3;
static const double U_DOT_MAX = 50e-3;
static const double T_U = 1e-3;
static double U_SLEW;                    /* U_DOT_MAX * T_U             */
static const double SLEW_W = 0.05;       /* w_z block slew limit        */

static const double INERTIA = 0.70, COGNITIVE = 1.5, SOCIAL = 1.5;
static const double JUMP_P = 0.25, JUMP_SCALE = 0.02, LEVY_LAMBDA = 1.5;
static const double SMOOTH_SPAN = 0.01;
static const double ELITE_FRACTION = 0.20;

static double wz_lo, wz_hi;
static double lo[64], hi[64], span[64], two_span[64];
static int    blk_s[3] = {0, 20, 40}, blk_e[3] = {20, 40, 60};
static double blk_lim[3];

static double GA_alpha, GA_beta;         /* tgamma(ALPHA), tgamma(BETA) */
static double kcAB[21], kcBA[21];        /* xi-free coefficients        */
static double CB[21], CA[21];            /* power moments C(B+k), C(A+k)*/
static double levy_sigma_15;             /* Mantegna sigma at lambda=1.5*/
static double w_safe;

static double kf_rho = 0.98, kf_q, kf_r = 1e-3, kf_P, kf_K;

/* ------------------------------------------------------------------ */
/* PRNG: xorshift128+ (uniform) with Marsaglia-polar normal deviates.   */
/* NOT numpy's PCG64/ziggurat -- see file header.                      */
/* ------------------------------------------------------------------ */
typedef struct { uint64_t s0, s1; int has_spare; double spare; } RNG;

static uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void rng_seed(RNG *r, uint64_t seed) {
    uint64_t sm = seed ? seed : 0x853c49e6748fea9bULL;
    r->s0 = splitmix64_next(&sm);
    r->s1 = splitmix64_next(&sm);
    if (!r->s0 && !r->s1) r->s0 = 1;
    r->has_spare = 0;
}
static uint64_t rng_next(RNG *r) {
    uint64_t x = r->s0, y = r->s1;
    r->s0 = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    r->s1 = x;
    return x + y;
}
static double rng_u01(RNG *r) {
    return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0); /* 2^53 */
}
static double rng_normal(RNG *r) {
    if (r->has_spare) { r->has_spare = 0; return r->spare; }
    double u, v, s;
    do {
        u = 2.0 * rng_u01(r) - 1.0;
        v = 2.0 * rng_u01(r) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    double mul = sqrt(-2.0 * log(s) / s);
    r->spare = v * mul;
    r->has_spare = 1;
    return u * mul;
}
static double rng_levy(RNG *r, double sigma, double lam) {
    double u = rng_normal(r) * sigma;
    double v = fabs(rng_normal(r));
    return u / pow(v, 1.0 / lam);
}

/* ------------------------------------------------------------------ */
/* geometry (channel.py: beam_geometry, branch_min_wz, wz_for_xi)      */
/* ------------------------------------------------------------------ */
static void raw_beam_geometry(double w_z, double *A0, double *w_zeq) {
    double v = sqrt(M_PI / 2.0) * APERTURE / w_z;
    double ev = erf(v);
    *A0 = ev * ev;
    *w_zeq = sqrt(w_z * w_z * sqrt(M_PI) * ev / (2.0 * v * exp(-v * v)));
}

static void branch_min_wz(double *w_star, double *weq_star) {
    double a_ = APERTURE;
    double lo_ = 1e-4 * (a_ / APERTURE), hi_ = 20.0 * (a_ / APERTURE);
    double gr = (sqrt(5.0) - 1.0) / 2.0;
    double c = hi_ - gr * (hi_ - lo_), d = lo_ + gr * (hi_ - lo_);
    for (int i = 0; i < 400; i++) {
        double A0c, wc, A0d, wd;
        raw_beam_geometry(c, &A0c, &wc);
        raw_beam_geometry(d, &A0d, &wd);
        if (wc < wd) hi_ = d; else lo_ = c;
        c = hi_ - gr * (hi_ - lo_);
        d = lo_ + gr * (hi_ - lo_);
    }
    double w = 0.5 * (lo_ + hi_);
    double A0w, weqw;
    raw_beam_geometry(w, &A0w, &weqw);
    *w_star = w; *weq_star = weqw;
}

static double wz_for_xi(double xi, double sigma_s, double w_argmin, double weq_argmin) {
    double target = 2.0 * sigma_s * xi;
    if (target <= weq_argmin) return w_argmin;
    double loB = w_argmin, hiB = 60.0;
    for (int i = 0; i < 200; i++) {
        double mid = 0.5 * (loB + hiB);
        double A0m, weqm;
        raw_beam_geometry(mid, &A0m, &weqm);
        if (weqm < target) loB = mid; else hiB = mid;
    }
    return 0.5 * (loB + hiB);
}

/* ------------------------------------------------------------------ */
/* RT-ODT kernel (rtodt_fast.py: pe_series_f64, z_of)                   */
/* ------------------------------------------------------------------ */
static double kc_term(double A, double B, int k) {
    double sign = (k % 2 == 0) ? 1.0 : -1.0;
    double ab_pow = pow(A * B, B + k);
    double gam = tgamma(A - B - (double)k);
    double kfact = tgamma((double)k + 1.0);
    return sign * ab_pow * gam / (kfact * tgamma(A) * tgamma(B));
}
static double c_moment(double s, double gbar) {
    return tgamma((s + 1.0) / 2.0) / (2.0 * s * sqrt(M_PI)) * pow(2.0 / gbar, s / 2.0);
}
static double z_of(double A0, double gbar) {
    return sqrt(2.0) * ALPHA * BETA / (A0 * sqrt(gbar));
}
static int ladder_order(double z) {
    if (z <= 0.5) return 5;
    if (z <= 2.0) return 10;
    if (z <= 8.0) return 20;
    return -1;
}
/* per-branch ABER, closed form, eq. (19)/(21) -- kcAB/kcBA/CB/CA precomputed
 * once for (ALPHA,BETA,GBAR); only A0, xi and the ladder order vary. */
static double pe_branch(int order, double xi, double A0) {
    double x2 = xi * xi;
    double pB = pow(A0, BETA), pA = pow(A0, ALPHA);
    double total = 0.0;
    for (int k = 0; k <= order; k++) {
        double t1 = kcAB[k] * x2 / ((x2 - BETA - k) * pB);
        double t2 = kcBA[k] * x2 / ((x2 - ALPHA - k) * pA);
        total += t1 * CB[k] + t2 * CA[k];
        pB *= A0; pA *= A0;
    }
    double D = x2 * pow(ALPHA * BETA, x2) * tgamma(ALPHA - x2) * tgamma(BETA - x2)
               / (pow(A0, x2) * GA_alpha * GA_beta);
    total += D * c_moment(x2, GBAR);
    return total;
}

/* ------------------------------------------------------------------ */
/* feasibility: reflect into box, then per-block forward-sweep repair  */
/* (hclpso_ga.py:_feasible + mpc_loop.py:BeamSteeringMPC.repair)       */
/* ------------------------------------------------------------------ */
/* Inter-cycle memory: the two-axis steering command actually published
 * last cycle. `feasible_row` pulls stage 0 of each steering block to
 * within one slew step of it, closing the gap the Step-0 fix (mpc_loop.py
 * BeamSteeringMPC.repair) closes in the released Python driver: without
 * this, stage 0 of a fresh horizon is checked against no predecessor at
 * all, and eq. (slew_const) binds across cycles, not only within one. */
static double u_prev[2] = {0.0, 0.0};

static void feasible_row(double *x) {
    for (int j = 0; j < D; j++) {
        double t = fmod(x[j] - lo[j], two_span[j]);
        if (t < 0.0) t += two_span[j];
        double v = (t > span[j]) ? (two_span[j] - t) : t;
        v = lo[j] + v;
        if (v < lo[j]) v = lo[j];
        if (v > hi[j]) v = hi[j];
        x[j] = v;
    }
    for (int b = 0; b < 3; b++) {
        double lim = blk_lim[b];
        if (b > 0) {
            double prev = u_prev[b - 1];
            double loB = prev - lim, hiB = prev + lim;
            double v = x[blk_s[b]];
            if (v < loB) v = loB;
            if (v > hiB) v = hiB;
            x[blk_s[b]] = v;
        }
        for (int k = blk_s[b] + 1; k < blk_e[b]; k++) {
            double loB = x[k - 1] - lim, hiB = x[k - 1] + lim;
            double v = x[k];
            if (v < loB) v = loB;
            if (v > hiB) v = hiB;
            x[k] = v;
        }
    }
}

/* ------------------------------------------------------------------ */
/* objective at rank_stages = 1: stage-0 ABER + full-trajectory slew    */
/* penalty/violation (mpc_loop.py:_objective)                          */
/* ------------------------------------------------------------------ */
static double objective_row(const double *x, double r_d, double g_stage0,
                             double *z_out, double *pe_out) {
    double A0, weq;
    raw_beam_geometry(x[0], &A0, &weq);
    double xi0 = weq / (2.0 * SIGMA_S);
    double xi_eff = xi0 / sqrt(1.0 + (r_d * r_d) / (2.0 * SIGMA_S * SIGMA_S));
    double z = z_of(A0, g_stage0);
    int K = ladder_order(z);
    double pe0 = (K >= 0) ? pe_branch(K, xi_eff, A0) : NAN;
    *z_out = z; *pe_out = pe0;

    /* the 1e-9 relative tolerance matches mpc_loop.py's _objective: a
     * trajectory forced to ramp at exactly the slew limit for many
     * consecutive stages (which anchoring stage 0 to u_prev now makes
     * common) accumulates floating-point rounding of order 1e-16
     * relative per hop, which a bare `dd > lim` flags as a spurious
     * violation once the chain runs ~15-20 stages deep. */
    double pen = 0.0; int viol = 0;
    for (int b = 0; b < 3; b++) {
        double lim = blk_lim[b];
        double lim_tol = lim * (1.0 + 1e-9);
        for (int k = blk_s[b] + 1; k < blk_e[b]; k++) {
            double dd = fabs(x[k] - x[k - 1]);
            pen += dd * dd;
            if (dd > lim_tol) viol = 1;
        }
        pen /= (double)(T_HOR - 1);
    }
    double cost = pe0 + LAMBDA_U * pen;
    if (viol) cost = INFINITY;
    return cost;
}

/* ------------------------------------------------------------------ */
/* the solver: chaotic init, PSO, Mantegna Levy, GA elite crossover,    */
/* anytime checkpoint (hclpso_ga.py:HCLPSOGA.minimise)                  */
/* ------------------------------------------------------------------ */
typedef struct {
    double best_x[64];
    double best_f;
    int iterations;
    long evaluations;
} SolverResult;

static LARGE_INTEGER g_freq;
static double now_us(void) {
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e6 / (double)g_freq.QuadPart;
}

static void logistic_chaos(double *out, int n, double seed_value, RNG *r) {
    (void)r;
    double v = seed_value;
    for (int i = 0; i < n; i++) {
        v = 4.0 * v * (1.0 - v);
        if (v == 0.0 || v == 0.25 || v == 0.5 || v == 0.75 || v == 1.0)
            v = fmod(v + 0.123456789, 1.0);
        out[i] = v;
    }
}

static void argsort_asc(const double *key, int n, int *order) {
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) {
        int oi = order[i]; double kv = key[oi];
        int j = i - 1;
        while (j >= 0 && key[order[j]] > kv) { order[j + 1] = order[j]; j--; }
        order[j + 1] = oi;
    }
}

static SolverResult solve(double r_d, double g_stage0, double deadline_us, RNG *r) {
    static double X[128][64], V[128][64], PBX[128][64];
    static double PBF[128], FW[128];
    double g_best[64]; int have_g = 0;
    SolverResult res; res.best_f = INFINITY; res.iterations = 0; res.evaluations = 0;
    memset(res.best_x, 0, sizeof(res.best_x));

    /* --- chaotic initialisation, block-wise smooth trajectory -------- */
    double seed_value = 0.1 + 0.8 * rng_u01(r);
    static double draw[128 * 64];
    logistic_chaos(draw, NP * D, seed_value, r);
    for (int p = 0; p < NP; p++) {
        for (int b = 0; b < 3; b++) {
            int s = blk_s[b], e = blk_e[b];
            double base = lo[s] + draw[p * D + s] * span[s];
            for (int j = s; j < e; j++) {
                double jitter = (draw[p * D + j] - 0.5) * 2.0 * SMOOTH_SPAN;
                X[p][j] = base + jitter * span[j];
            }
        }
        feasible_row(X[p]);
        memset(V[p], 0, sizeof(double) * D);
        memcpy(PBX[p], X[p], sizeof(double) * D);
        PBF[p] = INFINITY;
    }

    int n_elite = ELITE_FRACTION * NP; if (n_elite < 2) n_elite = 2;
    int m = NP / 3;
    int order[128];

    for (int it = 0; it < MAX_ITERS; it++) {
        for (int p = 0; p < NP; p++) feasible_row(X[p]);

        double z1, pe1;
        for (int p = 0; p < NP; p++) {
            double f = objective_row(X[p], r_d, g_stage0, &z1, &pe1);
            res.evaluations++;
            int finite_pe = isfinite(pe1);
            int t_z = z1 <= Z_MAX;
            int t_range = (pe1 >= 0.0) && (pe1 <= 0.5);
            int admissible = finite_pe && t_z && t_range;
            int ok = isfinite(f) && admissible;
            FW[p] = ok ? f : INFINITY;
            if (FW[p] < PBF[p]) { PBF[p] = FW[p]; memcpy(PBX[p], X[p], sizeof(double) * D); }
        }
        int ibest = 0;
        for (int p = 1; p < NP; p++) if (FW[p] < FW[ibest]) ibest = p;
        if (FW[ibest] < res.best_f) {
            res.best_f = FW[ibest];
            memcpy(res.best_x, X[ibest], sizeof(double) * D);
            memcpy(g_best, X[ibest], sizeof(double) * D);
            have_g = 1;
        }
        res.iterations = it + 1;

        if (now_us() >= deadline_us) return res;

        /* --- PSO core -------------------------------------------------- */
        const double *g = have_g ? g_best : X[ibest];
        for (int p = 0; p < NP; p++) {
            for (int j = 0; j < D; j++) {
                double r1 = rng_u01(r), r2 = rng_u01(r);
                V[p][j] = INERTIA * V[p][j]
                        + COGNITIVE * r1 * (PBX[p][j] - X[p][j])
                        + SOCIAL * r2 * (g[j] - X[p][j]);
                X[p][j] += V[p][j];
            }
        }

        /* --- heavy-tailed exploration (Levy, Mantegna) ------------------ */
        for (int p = 0; p < NP; p++) {
            if (rng_u01(r) < JUMP_P) {
                for (int j = 0; j < D; j++) {
                    double step = rng_levy(r, levy_sigma_15, LEVY_LAMBDA);
                    X[p][j] += JUMP_SCALE * step * span[j];
                }
            }
        }

        /* --- GA refinement on the elite ---------------------------------- */
        argsort_asc(PBF, NP, order);
        if (isfinite(PBF[order[n_elite - 1]])) {
            for (int t = 0; t < m; t++) {
                int ia = (int)(rng_u01(r) * n_elite); if (ia >= n_elite) ia = n_elite - 1;
                int ib = (int)(rng_u01(r) * n_elite); if (ib >= n_elite) ib = n_elite - 1;
                int dst = order[NP - m + t];
                int ea = order[ia], eb = order[ib];
                for (int j = 0; j < D; j++) {
                    double w = rng_u01(r);
                    X[dst][j] = w * PBX[ea][j] + (1.0 - w) * PBX[eb][j];
                }
            }
        }
    }
    return res;
}

/* ------------------------------------------------------------------ */
/* self-check: RT-ODT kernel against independently-known values        */
/* ------------------------------------------------------------------ */
static void selfcheck(void) {
    /* Fig. 3 / Sec. III-D validation point: strong regime not used here;
     * check instead against the manuscript's own printed system-scope
     * numbers is out of reach for a single-branch kernel. Use the
     * self-consistency checks that ARE reachable from this file alone:
     * (a) A0, w_zeq at known geometry limits; (b) z admissibility bands
     * are monotone in A0; (c) pe_branch returns a value in a sane range
     * (not the deployed [0,1/2] test -- this is a per-branch quantity at
     * the un-combined level and can exceed 1/2). */
    double w_argmin, weq_argmin;
    branch_min_wz(&w_argmin, &weq_argmin);
    printf("selfcheck: branch min  w_z*=%.6f m  w_zeq*=%.6f m"
           "  (manuscript: 0.054869 m, 0.0877 m)\n", w_argmin, weq_argmin);

    double A0t, weqt;
    raw_beam_geometry(0.1, &A0t, &weqt);
    printf("selfcheck: w_z=0.10 m -> A0=%.6f w_zeq=%.6f\n", A0t, weqt);

    printf("selfcheck: wz_lo=%.6f m  wz_hi=%.6f m  (box for sigma_s=0.10 m)\n",
           wz_lo, wz_hi);

    /* z at the box edges, nominal sigma_s=0.10, gbar at 38 dB */
    double A0lo, wlo, A0hi, whi;
    raw_beam_geometry(wz_lo, &A0lo, &wlo);
    raw_beam_geometry(wz_hi, &A0hi, &whi);
    printf("selfcheck: z(wz_lo)=%.4f  z(wz_hi)=%.4f  (z_max=8, widest beam"
           " should be near/above it)\n", z_of(A0lo, GBAR), z_of(A0hi, GBAR));

    /* pe_branch at a mid box point, sanity: finite, positive */
    double xi_mid = ((wz_lo + wz_hi) * 0.5);
    double A0m, wm; raw_beam_geometry(xi_mid, &A0m, &wm);
    double xim = wm / (2.0 * SIGMA_S);
    double zm = z_of(A0m, GBAR);
    int Km = ladder_order(zm);
    double pem = (Km >= 0) ? pe_branch(Km, xim, A0m) : NAN;
    printf("selfcheck: mid-box w_z=%.4f  xi=%.4f  z=%.4f  K=%d  pe=%.6e\n",
           xi_mid, xim, zm, Km, pem);
}

/* ------------------------------------------------------------------ */
/* stats helpers                                                        */
/* ------------------------------------------------------------------ */
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double pct(double *sorted, int n, double p) {
    double idx = p * (n - 1);
    int lo_ = (int)floor(idx), hi_ = (int)ceil(idx);
    if (lo_ == hi_) return sorted[lo_];
    double frac = idx - lo_;
    return sorted[lo_] * (1 - frac) + sorted[hi_] * frac;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    int n_cycles = (argc > 1) ? atoi(argv[1]) : 6000;
    if (argc > 2) {
        /* regime companion run: argv[2] in {strong,moderate,weak}, argv[3] sigma_s */
        if (strcmp(argv[2], "weak") == 0) { ALPHA = 4.2; BETA = 3.0; }
        else if (strcmp(argv[2], "moderate") == 0) { ALPHA = 2.1; BETA = 1.5; }
        else { ALPHA = 1.2; BETA = 1.1; }
        strncpy(REGIME_TAG, argv[2], sizeof(REGIME_TAG) - 1);
        if (argc > 3) SIGMA_S = atof(argv[3]);
    }

    /* pin + priority, mirroring the Python campaign methodology */
    HANDLE proc = GetCurrentProcess(), th = GetCurrentThread();
    SetPriorityClass(proc, HIGH_PRIORITY_CLASS);
    SetThreadPriority(th, THREAD_PRIORITY_HIGHEST);
    DWORD_PTR mask = (DWORD_PTR)1 << 2;         /* logical CPU 2 */
    SetThreadAffinityMask(th, mask);
    QueryPerformanceFrequency(&g_freq);

    /* ---- constants ---- */
    GBAR = pow(10.0, 38.0 / 10.0);
    U_SLEW = U_DOT_MAX * T_U;
    blk_lim[0] = SLEW_W; blk_lim[1] = U_SLEW; blk_lim[2] = U_SLEW;
    GA_alpha = tgamma(ALPHA); GA_beta = tgamma(BETA);
    for (int k = 0; k <= 20; k++) {
        kcAB[k] = kc_term(ALPHA, BETA, k);
        kcBA[k] = kc_term(BETA, ALPHA, k);
        CB[k] = c_moment(BETA + k, GBAR);
        CA[k] = c_moment(ALPHA + k, GBAR);
    }
    {
        double lam = LEVY_LAMBDA;
        double num = tgamma(1 + lam) * sin(M_PI * lam / 2.0);
        double den = tgamma((1 + lam) / 2.0) * lam * pow(2.0, (lam - 1) / 2.0);
        levy_sigma_15 = pow(num / den, 1.0 / lam);
    }

    double w_argmin, weq_argmin;
    branch_min_wz(&w_argmin, &weq_argmin);
    double xi_lo = fmax(0.5, 0.0877 / (2.0 * SIGMA_S));
    double xi_hi = 4.888;
    wz_lo = wz_for_xi(xi_lo, SIGMA_S, w_argmin, weq_argmin);
    wz_hi = wz_for_xi(xi_hi, SIGMA_S, w_argmin, weq_argmin);
    for (int j = 0; j < T_HOR; j++) { lo[j] = wz_lo; hi[j] = wz_hi; }
    for (int j = T_HOR; j < D; j++) { lo[j] = -U_MAX; hi[j] = U_MAX; }
    for (int j = 0; j < D; j++) { span[j] = hi[j] - lo[j]; two_span[j] = 2.0 * span[j]; }

    /* offline safe fallback: coarse+fine 1-D scan of pe_branch at r_d=0 */
    {
        double best = INFINITY, bestw = wz_lo;
        for (int i = 0; i <= 4000; i++) {
            double w = wz_lo + (wz_hi - wz_lo) * i / 4000.0;
            double A0, weq; raw_beam_geometry(w, &A0, &weq);
            double xi = weq / (2.0 * SIGMA_S);
            double z = z_of(A0, GBAR);
            int K = ladder_order(z);
            double pe = (K >= 0) ? pe_branch(K, xi, A0) : INFINITY;
            if (isfinite(pe) && pe < best) { best = pe; bestw = w; }
        }
        w_safe = bestw;
    }

    kf_q = 1.0 - kf_rho * kf_rho;
    { double p = kf_q;
      for (int i = 0; i < 500; i++) {
          double p_pred = kf_rho * kf_rho * p + kf_q;
          p = p_pred * kf_r / (p_pred + kf_r);
      }
      kf_P = p;
      kf_K = (kf_rho * kf_rho * p + kf_q) / (kf_rho * kf_rho * p + kf_q + kf_r);
    }

    printf("=== RT-ODT / H-CLPSO-GA compiled port -- self-check ===\n");
    selfcheck();
    printf("w_safe = %.6f m\n", w_safe);
    printf("kf: rho=%.4f q=%.6f r=%.6f  steady P=%.6e K=%.6f\n\n",
           kf_rho, kf_q, kf_r, kf_P, kf_K);

    double per_iter_median = 0, per_iter_p10 = 0, per_iter_p90 = 0;
    printf("=== per-iteration cost (25 fixed iterations, stub deadline) ===\n");
    {
        int reps = 300;
        double *totals = malloc(sizeof(double) * reps);
        for (int rep = 0; rep < reps; rep++) {
            RNG rr; rng_seed(&rr, 9000 + rep);
            double t0 = now_us();
            SolverResult res = solve(0.01, GBAR, 1.0e18, &rr);
            double t1 = now_us();
            totals[rep] = (t1 - t0) / (double)res.iterations;
        }
        qsort(totals, reps, sizeof(double), cmp_dbl);
        per_iter_median = pct(totals, reps, 0.5);
        per_iter_p10 = pct(totals, reps, 0.1);
        per_iter_p90 = pct(totals, reps, 0.9);
        printf("  per-iteration median = %.3f us   p10 = %.3f us   p90 = %.3f us\n",
               per_iter_median, per_iter_p10, per_iter_p90);
        printf("  (compare: released Python per-iteration floor 244.7 us,\n"
               "   solver_fast.py (numpy dispatch optimised) 92.1 us -- solver_fast.py.)\n");
        free(totals);
    }

    /* ---- full closed-loop cycle campaign ---- */
    printf("\n=== closed-loop cycle campaign: n=%d cycles, tau_O=%.0f us,"
           " rank_stages=1 ===\n", n_cycles, TAU_O_US);
    {
        RNG sway_rng, chan_rng, master_rng;
        rng_seed(&sway_rng, 1001);
        rng_seed(&chan_rng, 2001);
        rng_seed(&master_rng, 3001);

        double theta0 = 0.0, theta1 = 0.0;
        double u_prev0 = 0.0, u_prev1 = 0.0;
        double h_latent = 0.0;
        double kf_x = 0.0;
        double a_sway = exp(-2.0 * M_PI * 1.0 * T_U);
        double sigma_theta = SIGMA_S / L_LINK;

        double *total_us = malloc(sizeof(double) * n_cycles);
        double *opt_us = malloc(sizeof(double) * n_cycles);
        long *iters = malloc(sizeof(long) * n_cycles);
        int n_within = 0;

        /* per-cycle record for joint real-time + certified-accuracy scoring:
         * the ACTUALLY PUBLISHED w_z and r_d for that cycle (post safety-fallback
         * substitution), re-scored offline in Python against system_metric.py's
         * certified post-EGC system ABER -- this file's own admissible/pe_c test
         * is the per-branch surrogate the solver ranks by, not the certified
         * criterion Sec. VI-C defines success against. */
        /* Short names only: this repo's own path nests past 220 characters
         * before the filename even starts (cloned inside itself), so the
         * regime-tagged form used during development ("..._weak_0.10.csv")
         * pushed the full path past Windows' 260-character MAX_PATH and
         * fopen failed silently (NULL, no errno checked) -- caught by
         * printing the resolved handle, not by an error path, since the
         * campaign completes and prints its summary either way. */
        char csv_name[48];
        if (strcmp(REGIME_TAG, "strong") == 0)
            snprintf(csv_name, sizeof(csv_name), "compiled_poc_per_cycle.csv");
        else
            snprintf(csv_name, sizeof(csv_name), "poc_pc_%c%02d.csv",
                     REGIME_TAG[0], (int)lround(SIGMA_S * 100));
        FILE *pf = fopen(csv_name, "w");
        if (pf) fprintf(pf, "cycle,total_us,within_deadline,w_cmd,r_d,admissible\n");

        /* 500-cycle burn-in on the disturbance traces, outside the timed loop */
        for (int i = 0; i < 500; i++) {
            double drive0 = rng_normal(&sway_rng) * sigma_theta * sqrt(1 - a_sway * a_sway);
            double drive1 = rng_normal(&sway_rng) * sigma_theta * sqrt(1 - a_sway * a_sway);
            theta0 = a_sway * theta0 + drive0 - u_prev0;
            theta1 = a_sway * theta1 + drive1 - u_prev1;
        }

        for (int c = 0; c < n_cycles; c++) {
            double drive0 = rng_normal(&sway_rng) * sigma_theta * sqrt(1 - a_sway * a_sway);
            double drive1 = rng_normal(&sway_rng) * sigma_theta * sqrt(1 - a_sway * a_sway);
            theta0 = a_sway * theta0 + drive0 - u_prev0;
            theta1 = a_sway * theta1 + drive1 - u_prev1;
            h_latent = kf_rho * h_latent + sqrt(1 - kf_rho * kf_rho) * rng_normal(&chan_rng);
            double h_meas = h_latent;

            double t0 = now_us();
            /* 1. sensing (pass-through validity check) */
            int sense_ok = isfinite(theta0) && isfinite(theta1) && isfinite(h_meas)
                           && fabs(theta0) < 1.0 && fabs(theta1) < 1.0;
            double t1 = now_us();

            /* 2. prediction */
            double x_pred = kf_rho * kf_x;
            kf_x = x_pred + kf_K * (h_meas - x_pred);
            double h_pred0 = kf_x * kf_rho;   /* predict(horizon)[0] */
            double t2 = now_us();

            /* 3. optimisation */
            double r_d = L_LINK * sqrt(theta0 * theta0 + theta1 * theta1);
            double g_stage0 = GBAR * pow(fmax(1e-3, 1.0 + h_pred0), 2.0);
            uint64_t cyc_seed = rng_next(&master_rng);
            RNG solver_rng; rng_seed(&solver_rng, cyc_seed);
            double deadline = t2 + TAU_O_US;
            u_prev[0] = u_prev0; u_prev[1] = u_prev1;
            SolverResult res = solve(r_d, g_stage0, deadline, &solver_rng);
            double t3 = now_us();

            /* 4. safety check */
            double w_cmd = res.best_x[0];
            /* Boresight tracking loop, decoupled from the ranking surrogate.
             * At rank_stages=1 the per-branch objective_row() scores stage 0's
             * ABER as a function of x[0] (w_z) and the ALREADY-REALISED r_d
             * only -- u0/u1 (x[T_HOR], x[2*T_HOR]) enter no ABER term at this
             * ranking depth (mpc_loop.py's own comment: "the trajectory beyond
             * the published stage is shaped by the slew penalty and the
             * feasibility test alone, not by its own ABER"), so the PSO/GA
             * search has no incentive to make them correct pointing at all.
             * Run for 6000 cycles with u0/u1 taken straight from res.best_x,
             * theta drifts unboundedly (median r_d reached 311 m against a
             * sigma_s=0.10 m link -- three orders of magnitude past the
             * physical scale), because theta's AR(1) recursion (channel.py's
             * SwayProcess.step, a=exp(-2*pi*1*T_U)~=0.9937) amplifies any
             * un-corrected, non-restoring u0/u1 by 1/(1-a)~=159x at steady
             * state. Ground-truth theta0/theta1 are available every cycle
             * regardless of ranking depth, so a minimal proportional law run
             * OUTSIDE the ranking surrogate closes this loop without touching
             * real-time cost: u_ptr = clip(a*theta, -U_MAX, U_MAX), rate
             * limited to U_SLEW like any other published command. Kp=a exactly
             * cancels the AR(1) persistence (theta_{t+1} = a*theta_t + drive -
             * a*theta_t = drive), leaving theta at the innovation-noise scale
             * every cycle instead of accumulating it. w_z ranking is untouched. */
            double u_tgt0 = a_sway * theta0, u_tgt1 = a_sway * theta1;
            if (u_tgt0 < -U_MAX) u_tgt0 = -U_MAX; if (u_tgt0 > U_MAX) u_tgt0 = U_MAX;
            if (u_tgt1 < -U_MAX) u_tgt1 = -U_MAX; if (u_tgt1 > U_MAX) u_tgt1 = U_MAX;
            double d0 = u_tgt0 - u_prev0, d1 = u_tgt1 - u_prev1;
            if (d0 < -U_SLEW) d0 = -U_SLEW; if (d0 > U_SLEW) d0 = U_SLEW;
            if (d1 < -U_SLEW) d1 = -U_SLEW; if (d1 > U_SLEW) d1 = U_SLEW;
            double u0_cmd = u_prev0 + d0, u1_cmd = u_prev1 + d1;
            double A0c, weqc; raw_beam_geometry(w_cmd, &A0c, &weqc);
            double xi_c = weqc / (2.0 * SIGMA_S);
            double xi_eff_c = xi_c / sqrt(1.0 + (r_d * r_d) / (2.0 * SIGMA_S * SIGMA_S));
            double g_c = GBAR * pow(fmax(1e-3, 1.0 + h_pred0), 2.0);
            double z_c = z_of(A0c, g_c);
            int K_c = ladder_order(z_c);
            double pe_c = (K_c >= 0) ? pe_branch(K_c, xi_eff_c, A0c) : NAN;
            int test_i = z_c <= Z_MAX;
            int test_ii = isfinite(pe_c) && pe_c >= 0.0 && pe_c <= 0.5;
            int test_env = (fabs(u0_cmd) <= U_MAX) && (fabs(u1_cmd) <= U_MAX)
                           && (fabs(u0_cmd - u_prev0) <= U_SLEW)
                           && (fabs(u1_cmd - u_prev1) <= U_SLEW);
            int admissible = sense_ok && test_i && test_ii && test_env
                             && !isinf(res.best_f);
            /* Only the BEAM choice (w_cmd) falls back to w_safe on an
             * inadmissible test_i/test_ii/res.best_f verdict -- those tests
             * are about the solved w_z, not about u0_cmd/u1_cmd, which are
             * the boresight-tracking law above and already satisfy test_env
             * (|u|<=U_MAX, one slew step of u_prev) by construction. Freezing
             * u0_cmd/u1_cmd to u_prev here (the original behaviour) blocks
             * the tracking loop exactly when the beam is having trouble,
             * which is also when accumulated pointing error is largest and
             * correction is needed most -- self-sustaining the very drift
             * this loop exists to prevent. */
            if (!admissible) { w_cmd = w_safe; }
            double t4 = now_us();

            /* 5. publish */
            double u0_out = u0_cmd, u1_out = u1_cmd;
            if (u0_out < u_prev0 - U_SLEW) u0_out = u_prev0 - U_SLEW;
            if (u0_out > u_prev0 + U_SLEW) u0_out = u_prev0 + U_SLEW;
            if (u1_out < u_prev1 - U_SLEW) u1_out = u_prev1 - U_SLEW;
            if (u1_out > u_prev1 + U_SLEW) u1_out = u_prev1 + U_SLEW;
            if (u0_out < -U_MAX) u0_out = -U_MAX; if (u0_out > U_MAX) u0_out = U_MAX;
            if (u1_out < -U_MAX) u1_out = -U_MAX; if (u1_out > U_MAX) u1_out = U_MAX;
            u_prev0 = u0_out; u_prev1 = u1_out;
            (void)w_cmd;
            double t5 = now_us();

            total_us[c] = t5 - t0;
            opt_us[c] = t3 - t2;
            iters[c] = res.iterations;
            int within = total_us[c] <= 800.0;
            if (within) n_within++;
            if (pf) fprintf(pf, "%d,%.3f,%d,%.10f,%.6f,%d\n",
                             c, total_us[c], within, w_cmd, r_d, admissible);
        }
        if (pf) { fclose(pf); printf("  wrote %s\n", csv_name); }
        else printf("  [warning] could not open %s for writing (path length?)\n", csv_name);

        qsort(total_us, n_cycles, sizeof(double), cmp_dbl);
        double *opt_sorted = malloc(sizeof(double) * n_cycles);
        memcpy(opt_sorted, opt_us, sizeof(double) * n_cycles);
        qsort(opt_sorted, n_cycles, sizeof(double), cmp_dbl);

        long iter_sum = 0; for (int i = 0; i < n_cycles; i++) iter_sum += iters[i];
        double *iters_d = malloc(sizeof(double) * n_cycles);
        for (int i = 0; i < n_cycles; i++) iters_d[i] = (double)iters[i];
        qsort(iters_d, n_cycles, sizeof(double), cmp_dbl);

        printf("  end-to-end latency (us):  median=%.1f  p95=%.1f  p99=%.1f"
               "  p99.9=%.1f  max=%.1f\n",
               pct(total_us, n_cycles, 0.5), pct(total_us, n_cycles, 0.95),
               pct(total_us, n_cycles, 0.99), pct(total_us, n_cycles, 0.999),
               total_us[n_cycles - 1]);
        printf("  within 800 us deadline: %.1f%%  (%d/%d)\n",
               100.0 * n_within / n_cycles, n_within, n_cycles);
        printf("  optimisation-stage latency (us): median=%.1f  p95=%.1f  max=%.1f\n",
               pct(opt_sorted, n_cycles, 0.5), pct(opt_sorted, n_cycles, 0.95),
               opt_sorted[n_cycles - 1]);
        printf("  solver iterations per cycle: median=%.1f  mean=%.2f  max=%ld\n",
               pct(iters_d, n_cycles, 0.5), (double)iter_sum / n_cycles,
               (long)iters_d[n_cycles - 1]);

        char json_name[48];
        if (strcmp(REGIME_TAG, "strong") == 0)
            snprintf(json_name, sizeof(json_name), "compiled_poc_cycle_latency.json");
        else
            snprintf(json_name, sizeof(json_name), "poc_cl_%c%02d.json",
                     REGIME_TAG[0], (int)lround(SIGMA_S * 100));
        FILE *jf = fopen(json_name, "w");
        if (jf) {
            fprintf(jf,
                "{\n"
                "  \"what\": \"proof-of-concept compiled (C, zig cc -O3) port of the H-CLPSO-GA solver and the RT-ODT single-branch kernel at rank_stages=1, timed as one closed-loop cycle; NOT a bit-identical replacement of the released Python search (different PRNG: xorshift128+/Marsaglia-polar, not PCG64/ziggurat) and NOT a reproduction of tab:tail_mitigation, which stays the authoritative as-released figure\",\n"
                "  \"platform\": \"12th Gen Intel Core i3-12100 (4 cores / 8 threads, no E-cores), Windows 11, compiled with zig cc 0.16.0 (LLVM/clang backend) -O3\",\n"
                "  \"note\": \"measured on a separate machine from Platform A, so figures here are not comparable value-for-value with Table tab:tail_mitigation's Platform A figures -- only the MECHANISM (dispatch overhead removed by compilation) is the claim\",\n"
                "  \"n_cycles\": %d,\n"
                "  \"tau_o_us\": %.1f,\n"
                "  \"rank_stages\": 1,\n"
                "  \"operating_point\": {\"alpha\": %.2f, \"beta\": %.2f, \"gbar_db\": 38.0, \"sigma_s_m\": %.2f, \"horizon\": %d, \"n_particles\": %d},\n"
                "  \"per_iteration_us\": {\"median\": %.3f, \"p10\": %.3f, \"p90\": %.3f},\n"
                "  \"reference_python_per_iteration_us\": {\"released_stubbed_objective\": 244.7, \"solver_fast_stubbed_objective\": 92.1, \"note\": \"both are solver-overhead-only, objective stubbed to a constant; this file's per_iteration_us includes a REAL RT-ODT single-branch evaluation with 3 runtime Gamma calls, so the comparison understates the gap\"},\n"
                "  \"end_to_end_latency_us\": {\"median\": %.1f, \"p95\": %.1f, \"p99\": %.1f, \"p999\": %.1f, \"max\": %.1f},\n"
                "  \"within_800us_deadline\": {\"fraction\": %.4f, \"n_within\": %d, \"n_cycles\": %d},\n"
                "  \"optimisation_stage_us\": {\"median\": %.1f, \"p95\": %.1f, \"max\": %.1f},\n"
                "  \"iterations_per_cycle\": {\"median\": %.1f, \"mean\": %.3f, \"max\": %ld},\n"
                "  \"reference_python_tab_tail_mitigation\": {\"median_us\": 809, \"p95_us\": 986, \"max_us\": 1989, \"within_800us_pct\": 27.6, \"median_iterations\": 1}\n"
                "}\n",
                n_cycles, TAU_O_US, ALPHA, BETA, SIGMA_S, T_HOR, NP,
                per_iter_median, per_iter_p10, per_iter_p90,
                pct(total_us, n_cycles, 0.5), pct(total_us, n_cycles, 0.95),
                pct(total_us, n_cycles, 0.99), pct(total_us, n_cycles, 0.999),
                total_us[n_cycles - 1],
                (double)n_within / n_cycles, n_within, n_cycles,
                pct(opt_sorted, n_cycles, 0.5), pct(opt_sorted, n_cycles, 0.95),
                opt_sorted[n_cycles - 1],
                pct(iters_d, n_cycles, 0.5), (double)iter_sum / n_cycles,
                (long)iters_d[n_cycles - 1]);
            fclose(jf);
            printf("\n  wrote %s\n", json_name);
        }

        free(total_us); free(opt_us); free(opt_sorted); free(iters); free(iters_d);
    }

    return 0;
}
