/*
 * Single-shot trial batch for the optimization-success / ablation campaign
 * (measure_all.py part_ablation / _budgeted protocol), compiled.
 *
 * Reads trials from a text file, one per line:   sigma_s  r_d  seed
 * Runs one arm on every trial under the tau_O solver-time checkpoint and
 * writes:   sigma_s  r_d  seed  w_best  iterations  best_f
 * to the output file. System-level scoring (post-EGC ABER <= 1e-6) is done
 * afterwards in Python by system_metric.system_success, the paper's own
 * certified evaluator, so this file computes the SEARCH only.
 *
 * Faithful to measure_all.py in the following respects:
 *   - the objective is mpc_loop._objective at rank_stages=1 with theta0 =
 *     [r_d/L, 0], h_pred = 0 (fresh Kalman state), i.e. g_stage0 = GBAR;
 *   - NO envelope guard inside the search: part_ablation calls minimise()
 *     without guard=, and _budgeted's consider() keeps a candidate iff its
 *     cost is finite. Finiteness encodes both z-admissibility (K<0 -> NaN)
 *     and the hard slew constraint (violation -> inf);
 *   - no inter-cycle anchor: every trial is a fresh, single-cycle instance;
 *   - arms: full | no_chaos | no_levy | no_ga | no_ladder | random | pso,
 *     mirroring ARMS = {full, no_chaotic_init, no_levy_flight,
 *     no_ga_refinement, fixed_fidelity(K=10)} and METHODS random/pso.
 * Not bit-identical to NumPy's PCG64 stream (xorshift128+/Marsaglia here).
 *
 * Build: zig cc -O2 -o validation_success_kernel.exe validation_success_kernel.c -lm
 *        (the reported campaign was built at -O2; an -O3 build of this same
 *        source was refused by the host's Application Control policy, which
 *        was left in force rather than bypassed)
 * Run:   validation_success_kernel.exe trials.txt <arm> out.csv [tau_us]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static const double ALPHA = 1.2, BETA = 1.1;
static double GBAR;
static double SIGMA_S = 0.10;
static const int    T_HOR = 20, NP = 30, D = 60, MAX_ITERS = 25;
static const double LAMBDA_U = 2.0, Z_MAX = 8.0, APERTURE = 0.05, L_LINK = 2000.0;
static const double U_MAX = 10e-3, U_DOT_MAX = 50e-3, T_U = 1e-3;
static double U_SLEW;
static const double SLEW_W = 0.05;
static const double INERTIA = 0.70, COGNITIVE = 1.5, SOCIAL = 1.5;
static const double JUMP_P = 0.25, JUMP_SCALE = 0.02, LEVY_LAMBDA = 1.5;
static const double SMOOTH_SPAN = 0.01, ELITE_FRACTION = 0.20;

static int USE_CHAOS = 1, USE_LEVY = 1, USE_GA = 1, USE_LADDER = 1;
static const int FIXED_ORDER = 10;

static double wz_lo, wz_hi;
static double lo[64], hi[64], span[64], two_span[64];
static int    blk_s[3] = {0, 20, 40}, blk_e[3] = {20, 40, 60};
static double blk_lim[3];
static double GA_alpha, GA_beta, kcAB[21], kcBA[21], CB[21], CA[21], levy_sigma_15;

typedef struct { uint64_t s0, s1; int has_spare; double spare; } RNG;
static uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void rng_seed(RNG *r, uint64_t seed) {
    uint64_t sm = seed ? seed : 0x853c49e6748fea9bULL;
    r->s0 = splitmix64_next(&sm); r->s1 = splitmix64_next(&sm);
    if (!r->s0 && !r->s1) r->s0 = 1;
    r->has_spare = 0;
}
static uint64_t rng_next(RNG *r) {
    uint64_t x = r->s0, y = r->s1;
    r->s0 = y; x ^= x << 23; x ^= x >> 17; x ^= y ^ (y >> 26); r->s1 = x;
    return x + y;
}
static double rng_u01(RNG *r) { return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static double rng_normal(RNG *r) {
    if (r->has_spare) { r->has_spare = 0; return r->spare; }
    double u, v, s;
    do { u = 2.0 * rng_u01(r) - 1.0; v = 2.0 * rng_u01(r) - 1.0; s = u * u + v * v; }
    while (s >= 1.0 || s == 0.0);
    double mul = sqrt(-2.0 * log(s) / s);
    r->spare = v * mul; r->has_spare = 1;
    return u * mul;
}
static double rng_levy(RNG *r, double sigma, double lam) {
    double u = rng_normal(r) * sigma, v = fabs(rng_normal(r));
    return u / pow(v, 1.0 / lam);
}

static void raw_beam_geometry(double w_z, double *A0, double *w_zeq) {
    double v = sqrt(M_PI / 2.0) * APERTURE / w_z, ev = erf(v);
    *A0 = ev * ev;
    *w_zeq = sqrt(w_z * w_z * sqrt(M_PI) * ev / (2.0 * v * exp(-v * v)));
}
static void branch_min_wz(double *w_star, double *weq_star) {
    double lo_ = 1e-4, hi_ = 20.0, gr = (sqrt(5.0) - 1.0) / 2.0;
    double c = hi_ - gr * (hi_ - lo_), d = lo_ + gr * (hi_ - lo_);
    for (int i = 0; i < 400; i++) {
        double A0c, wc, A0d, wd;
        raw_beam_geometry(c, &A0c, &wc); raw_beam_geometry(d, &A0d, &wd);
        if (wc < wd) hi_ = d; else lo_ = c;
        c = hi_ - gr * (hi_ - lo_); d = lo_ + gr * (hi_ - lo_);
    }
    double w = 0.5 * (lo_ + hi_), A0w, weqw;
    raw_beam_geometry(w, &A0w, &weqw);
    *w_star = w; *weq_star = weqw;
}
static double wz_for_xi(double xi, double sigma_s, double w_argmin, double weq_argmin) {
    double target = 2.0 * sigma_s * xi;
    if (target <= weq_argmin) return w_argmin;
    double loB = w_argmin, hiB = 60.0;
    for (int i = 0; i < 200; i++) {
        double mid = 0.5 * (loB + hiB), A0m, weqm;
        raw_beam_geometry(mid, &A0m, &weqm);
        if (weqm < target) loB = mid; else hiB = mid;
    }
    return 0.5 * (loB + hiB);
}
static double g_w_argmin, g_weq_argmin;
static void compute_box(double sigma_s) {
    SIGMA_S = sigma_s;
    double xi_lo = fmax(0.5, 0.0877 / (2.0 * sigma_s)), xi_hi = 4.888;
    wz_lo = wz_for_xi(xi_lo, sigma_s, g_w_argmin, g_weq_argmin);
    wz_hi = wz_for_xi(xi_hi, sigma_s, g_w_argmin, g_weq_argmin);
    for (int j = 0; j < T_HOR; j++) { lo[j] = wz_lo; hi[j] = wz_hi; }
    for (int j = T_HOR; j < D; j++) { lo[j] = -U_MAX; hi[j] = U_MAX; }
    for (int j = 0; j < D; j++) { span[j] = hi[j] - lo[j]; two_span[j] = 2.0 * span[j]; }
}

static double kc_term(double A, double B, int k) {
    double sign = (k % 2 == 0) ? 1.0 : -1.0;
    return sign * pow(A * B, B + k) * tgamma(A - B - (double)k)
           / (tgamma((double)k + 1.0) * tgamma(A) * tgamma(B));
}
static double c_moment(double s, double gbar) {
    return tgamma((s + 1.0) / 2.0) / (2.0 * s * sqrt(M_PI)) * pow(2.0 / gbar, s / 2.0);
}
static double z_of(double A0, double gbar) { return sqrt(2.0) * ALPHA * BETA / (A0 * sqrt(gbar)); }
static int ladder_order(double z) {
    if (!USE_LADDER) return (z <= Z_MAX) ? FIXED_ORDER : -1;
    if (z <= 0.5) return 5;
    if (z <= 2.0) return 10;
    if (z <= 8.0) return 20;
    return -1;
}
static double pe_branch(int order, double xi, double A0) {
    double x2 = xi * xi, pB = pow(A0, BETA), pA = pow(A0, ALPHA), total = 0.0;
    for (int k = 0; k <= order; k++) {
        total += kcAB[k] * x2 / ((x2 - BETA - k) * pB) * CB[k]
               + kcBA[k] * x2 / ((x2 - ALPHA - k) * pA) * CA[k];
        pB *= A0; pA *= A0;
    }
    double Dr = x2 * pow(ALPHA * BETA, x2) * tgamma(ALPHA - x2) * tgamma(BETA - x2)
                / (pow(A0, x2) * GA_alpha * GA_beta);
    return total + Dr * c_moment(x2, GBAR);
}

/* repair: box clip + forward sweep (BeamSteeringMPC.repair, no anchor) */
static void repair_row(double *x) {
    for (int j = 0; j < D; j++) { if (x[j] < lo[j]) x[j] = lo[j]; if (x[j] > hi[j]) x[j] = hi[j]; }
    for (int b = 0; b < 3; b++) {
        double lim = blk_lim[b];
        for (int k = blk_s[b] + 1; k < blk_e[b]; k++) {
            double loB = x[k - 1] - lim, hiB = x[k - 1] + lim, v = x[k];
            if (v < loB) v = loB; if (v > hiB) v = hiB; x[k] = v;
        }
    }
}
/* _feasible: reflect into box, then repair (HCLPSOGA._feasible) */
static void feasible_row(double *x) {
    for (int j = 0; j < D; j++) {
        double t = fmod(x[j] - lo[j], two_span[j]);
        if (t < 0.0) t += two_span[j];
        double v = (t > span[j]) ? (two_span[j] - t) : t;
        x[j] = lo[j] + v;
    }
    repair_row(x);
}

static double objective_row(const double *x, double r_d, double g_stage0) {
    double A0, weq; raw_beam_geometry(x[0], &A0, &weq);
    double xi0 = weq / (2.0 * SIGMA_S);
    double xi_eff = xi0 / sqrt(1.0 + (r_d * r_d) / (2.0 * SIGMA_S * SIGMA_S));
    double z = z_of(A0, g_stage0);
    int K = ladder_order(z);
    double pe0 = (K >= 0) ? pe_branch(K, xi_eff, A0) : NAN;
    double pen = 0.0; int viol = 0;
    for (int b = 0; b < 3; b++) {
        double lim = blk_lim[b], lim_tol = lim * (1.0 + 1e-9);
        for (int k = blk_s[b] + 1; k < blk_e[b]; k++) {
            double dd = fabs(x[k] - x[k - 1]);
            pen += dd * dd; if (dd > lim_tol) viol = 1;
        }
        pen /= (double)(T_HOR - 1);
    }
    double cost = pe0 + LAMBDA_U * pen;
    if (viol) cost = INFINITY;
    return cost;
}

static LARGE_INTEGER g_freq;
static double now_us(void) { LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double)c.QuadPart * 1e6 / (double)g_freq.QuadPart; }

static void logistic_chaos(double *out, int n, double seed_value) {
    double v = seed_value;
    for (int i = 0; i < n; i++) {
        v = 4.0 * v * (1.0 - v);
        if (v == 0.0 || v == 0.25 || v == 0.5 || v == 0.75 || v == 1.0) v = fmod(v + 0.123456789, 1.0);
        out[i] = v;
    }
}
static void argsort_asc(const double *key, int n, int *order) {
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) {
        int oi = order[i]; double kv = key[oi]; int j = i - 1;
        while (j >= 0 && key[order[j]] > kv) { order[j + 1] = order[j]; j--; }
        order[j + 1] = oi;
    }
}

typedef struct { double best_x[64]; double best_f; int iterations; } SolverResult;

static SolverResult solve_hclpso(double r_d, double g0, double deadline_us, RNG *r) {
    static double X[64][64], V[64][64], PBX[64][64], PBF[64], FW[64], draw[64 * 64];
    double g_best[64]; int have_g = 0;
    SolverResult res; res.best_f = INFINITY; res.iterations = 0; memset(res.best_x, 0, sizeof res.best_x);

    if (USE_CHAOS) { double sv = 0.1 + 0.8 * rng_u01(r); logistic_chaos(draw, NP * D, sv); }
    else for (int i = 0; i < NP * D; i++) draw[i] = rng_u01(r);
    for (int p = 0; p < NP; p++) {
        for (int b = 0; b < 3; b++) {
            int s = blk_s[b], e = blk_e[b];
            double base = lo[s] + draw[p * D + s] * span[s];
            for (int j = s; j < e; j++) X[p][j] = base + (draw[p * D + j] - 0.5) * 2.0 * SMOOTH_SPAN * span[j];
        }
        feasible_row(X[p]);
        memset(V[p], 0, sizeof(double) * D); memcpy(PBX[p], X[p], sizeof(double) * D); PBF[p] = INFINITY;
    }
    int n_elite = (int)(ELITE_FRACTION * NP); if (n_elite < 2) n_elite = 2;
    int m = NP / 3, order[64];

    for (int it = 0; it < MAX_ITERS; it++) {
        for (int p = 0; p < NP; p++) feasible_row(X[p]);
        for (int p = 0; p < NP; p++) {
            double f = objective_row(X[p], r_d, g0);
            FW[p] = isfinite(f) ? f : INFINITY;        /* no guard: part_ablation passes none */
            if (FW[p] < PBF[p]) { PBF[p] = FW[p]; memcpy(PBX[p], X[p], sizeof(double) * D); }
        }
        int ib = 0; for (int p = 1; p < NP; p++) if (FW[p] < FW[ib]) ib = p;
        if (FW[ib] < res.best_f) { res.best_f = FW[ib]; memcpy(res.best_x, X[ib], sizeof(double) * D); memcpy(g_best, X[ib], sizeof(double) * D); have_g = 1; }
        res.iterations = it + 1;
        if (now_us() >= deadline_us) return res;

        const double *g = have_g ? g_best : X[ib];
        for (int p = 0; p < NP; p++) for (int j = 0; j < D; j++) {
            double r1 = rng_u01(r), r2 = rng_u01(r);
            V[p][j] = INERTIA * V[p][j] + COGNITIVE * r1 * (PBX[p][j] - X[p][j]) + SOCIAL * r2 * (g[j] - X[p][j]);
            X[p][j] += V[p][j];
        }
        for (int p = 0; p < NP; p++) if (rng_u01(r) < JUMP_P) {
            for (int j = 0; j < D; j++) {
                double step = USE_LEVY ? rng_levy(r, levy_sigma_15, LEVY_LAMBDA) : rng_normal(r);
                X[p][j] += JUMP_SCALE * step * span[j];
            }
        }
        if (USE_GA) {
            argsort_asc(PBF, NP, order);
            if (isfinite(PBF[order[n_elite - 1]])) {
                for (int t = 0; t < m; t++) {
                    int ia = (int)(rng_u01(r) * n_elite); if (ia >= n_elite) ia = n_elite - 1;
                    int ibx = (int)(rng_u01(r) * n_elite); if (ibx >= n_elite) ibx = n_elite - 1;
                    int dst = order[NP - m + t], ea = order[ia], eb = order[ibx];
                    for (int j = 0; j < D; j++) { double w = rng_u01(r); X[dst][j] = w * PBX[ea][j] + (1.0 - w) * PBX[eb][j]; }
                }
            }
        }
    }
    return res;
}

/* random sampling: NP uniform points per iteration, clip+repair, keep best finite */
static SolverResult solve_random(double r_d, double g0, double deadline_us, RNG *r) {
    static double X[64];
    SolverResult res; res.best_f = INFINITY; res.iterations = 0; memset(res.best_x, 0, sizeof res.best_x);
    for (int it = 0; it < MAX_ITERS; it++) {
        if (now_us() >= deadline_us) break;
        for (int p = 0; p < NP; p++) {
            for (int j = 0; j < D; j++) X[j] = lo[j] + rng_u01(r) * span[j];
            repair_row(X);
            double f = objective_row(X, r_d, g0);
            if (isfinite(f) && f < res.best_f) { res.best_f = f; memcpy(res.best_x, X, sizeof(double) * D); }
        }
        res.iterations = it + 1;
    }
    return res;
}

/* plain PSO baseline (measure_all._budgeted "pso"): uniform init, clip not reflect */
static SolverResult solve_pso(double r_d, double g0, double deadline_us, RNG *r) {
    static double X[64][64], V[64][64], PB[64][64], PBF[64], FV[64];
    SolverResult res; res.best_f = INFINITY; res.iterations = 0; memset(res.best_x, 0, sizeof res.best_x);
    for (int p = 0; p < NP; p++) { for (int j = 0; j < D; j++) { X[p][j] = lo[j] + rng_u01(r) * span[j]; V[p][j] = 0.0; } memcpy(PB[p], X[p], sizeof(double) * D); PBF[p] = INFINITY; }
    for (int it = 0; it < MAX_ITERS; it++) {
        if (now_us() >= deadline_us) break;
        for (int p = 0; p < NP; p++) repair_row(X[p]);
        int ib = 0;
        for (int p = 0; p < NP; p++) {
            FV[p] = objective_row(X[p], r_d, g0);
            if (FV[p] < PBF[p]) { PBF[p] = FV[p]; memcpy(PB[p], X[p], sizeof(double) * D); }
            if (isfinite(FV[p]) && FV[p] < res.best_f) { res.best_f = FV[p]; memcpy(res.best_x, X[p], sizeof(double) * D); }
            if (FV[p] < FV[ib]) ib = p;
        }
        res.iterations = it + 1;
        const double *g = isfinite(res.best_f) ? res.best_x : X[ib];
        for (int p = 0; p < NP; p++) for (int j = 0; j < D; j++) {
            double r1 = rng_u01(r), r2 = rng_u01(r);
            V[p][j] = 0.7 * V[p][j] + 1.5 * r1 * (PB[p][j] - X[p][j]) + 1.5 * r2 * (g[j] - X[p][j]);
            double v = X[p][j] + V[p][j]; if (v < lo[j]) v = lo[j]; if (v > hi[j]) v = hi[j]; X[p][j] = v;
        }
    }
    return res;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s trials.txt <arm> out.csv [tau_us] [gbar_db]\n", argv[0]); return 2; }
    const char *arm = argv[2];
    double tau_us = (argc > 4) ? atof(argv[4]) : 600.0;
    double gbar_db = (argc > 5) ? atof(argv[5]) : 38.0;   /* Step 3: alternative operating SNR */
    if (!strcmp(arm, "no_chaos")) USE_CHAOS = 0;
    else if (!strcmp(arm, "no_levy")) USE_LEVY = 0;
    else if (!strcmp(arm, "no_ga")) USE_GA = 0;
    else if (!strcmp(arm, "no_ladder")) USE_LADDER = 0;
    else if (strcmp(arm, "full") && strcmp(arm, "random") && strcmp(arm, "pso")) { fprintf(stderr, "unknown arm %s\n", arm); return 2; }

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 2);
    QueryPerformanceFrequency(&g_freq);

    GBAR = pow(10.0, gbar_db / 10.0); U_SLEW = U_DOT_MAX * T_U;
    blk_lim[0] = SLEW_W; blk_lim[1] = U_SLEW; blk_lim[2] = U_SLEW;
    GA_alpha = tgamma(ALPHA); GA_beta = tgamma(BETA);
    for (int k = 0; k <= 20; k++) { kcAB[k] = kc_term(ALPHA, BETA, k); kcBA[k] = kc_term(BETA, ALPHA, k); CB[k] = c_moment(BETA + k, GBAR); CA[k] = c_moment(ALPHA + k, GBAR); }
    { double lam = LEVY_LAMBDA; levy_sigma_15 = pow(tgamma(1 + lam) * sin(M_PI * lam / 2.0) / (tgamma((1 + lam) / 2.0) * lam * pow(2.0, (lam - 1) / 2.0)), 1.0 / lam); }
    branch_min_wz(&g_w_argmin, &g_weq_argmin);

    FILE *fin = fopen(argv[1], "r"); if (!fin) { perror("trials"); return 1; }
    FILE *fout = fopen(argv[3], "w"); if (!fout) { perror("out"); return 1; }
    fprintf(fout, "sigma_s,r_d,seed,w_best,iterations,best_f\n");
    double cur_sigma = -1.0; int n = 0;
    double s, rd; unsigned long long seed;
    while (fscanf(fin, "%lf %lf %llu", &s, &rd, &seed) == 3) {
        if (s != cur_sigma) { compute_box(s); cur_sigma = s; }
        RNG r; rng_seed(&r, (uint64_t)seed);
        double deadline = now_us() + tau_us;
        SolverResult res;
        if (!strcmp(arm, "random")) res = solve_random(rd, GBAR, deadline, &r);
        else if (!strcmp(arm, "pso")) res = solve_pso(rd, GBAR, deadline, &r);
        else res = solve_hclpso(rd, GBAR, deadline, &r);
        if (isfinite(res.best_f)) fprintf(fout, "%.6g,%.10g,%llu,%.17g,%d,%.17g\n", s, rd, seed, res.best_x[0], res.iterations, res.best_f);
        else fprintf(fout, "%.6g,%.10g,%llu,nan,%d,inf\n", s, rd, seed, res.iterations);
        n++;
    }
    fclose(fin); fclose(fout);
    fprintf(stderr, "arm=%s tau=%.0fus trials=%d done\n", arm, tau_us, n);
    return 0;
}
