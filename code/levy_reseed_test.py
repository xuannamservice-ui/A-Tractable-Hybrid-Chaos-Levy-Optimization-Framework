"""Step-2 direction (i): Levy at RE-INITIALISATION rather than as a mid-search jump.

Each iteration the worst 20% of the swarm (by pbest) is re-seeded the way the
chaotic initialiser seeds: one level per physical block, taken from the incumbent
gbest and displaced by a block-common heavy-tailed step (the feas_shift geometry,
which survives the slew projection by construction), plus the same small
per-stage jitter the initialiser uses, then projected. The control arm draws the
displacement from a Gaussian of the same nominal scale. Everything else in the
solver (chaotic init, PSO, GA) is unchanged and the standard per-dim jump is OFF
in both arms so the comparison isolates the re-seed operator.

Coupled-trajectory ranking (rank_stages=None), full T_iter=25 (tau=20 ms), the
same trial protocol and paired draws as ablation_continuous.py. Diagnostic;
reports what it finds in either direction.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_all as _ma
_ma.RANK_STAGES = None
from measure_all import N_P, SIGMAS, _make_problem, system_success
from hclpso_ga import HCLPSOGA, SolverConfig, levy
from channel import SwayProcess
from scipy.stats import wilcoxon

T_ITER, TAU = 25, 20e-3
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
RESEED_FRAC, RESEED_SCALE = 0.20, 0.02


def minimise_reseed(sol, objective, deadline, use_levy):
    cfg = sol.cfg; rng = sol.rng
    x = sol._initialise(); v = np.zeros_like(x)
    n = cfg.n_particles; d = sol.dim
    pbest_x = x.copy(); pbest_f = np.full(n, np.inf)
    best_x, best_f = None, np.inf
    n_elite = max(2, int(cfg.elite_fraction * n)); m = n // 3
    k_re = max(1, int(RESEED_FRAC * n))
    span = sol.hi - sol.lo
    for it in range(cfg.max_iters):
        x = sol._feasible(x)
        f = objective(x)
        fw = np.where(np.isfinite(f), f, np.inf)
        imp = fw < pbest_f; pbest_f[imp] = fw[imp]; pbest_x[imp] = x[imp]
        i = int(np.argmin(fw))
        if fw[i] < best_f:
            best_f, best_x = float(fw[i]), x[i].copy()
        if time.perf_counter() > deadline:
            return best_x, best_f, it + 1
        r1 = rng.random((n, d)); r2 = rng.random((n, d))
        g = best_x if best_x is not None else x[i]
        v = cfg.inertia * v + cfg.cognitive * r1 * (pbest_x - x) + cfg.social * r2 * (g - x)
        x = x + v
        # --- re-seed the worst k_re particles from gbest with a block-common step
        order = np.argsort(pbest_f)
        worst = order[-k_re:]
        for p in worst:
            for (s, e) in sol.blocks:
                spanb = float(np.max(span[s:e]))
                step = levy(rng, 1, cfg.levy_lambda)[0] if use_levy else rng.normal()
                base = g[s] + RESEED_SCALE * step * spanb
                jitter = (rng.random(e - s) - 0.5) * 2.0 * cfg.smooth_span
                x[p, s:e] = base + jitter * span[s:e]
        if cfg.use_ga:
            elite = pbest_x[order[:n_elite]]
            if np.all(np.isfinite(pbest_f[order[:n_elite]])):
                ia = rng.integers(0, n_elite, m); ib = rng.integers(0, n_elite, m)
                w = rng.random((m, d))
                x[order[-m:]] = w * elite[ia] + (1 - w) * elite[ib]
    return best_x, best_f, cfg.max_iters


def main():
    per_cell = max(1, N_TRIALS // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)
    aber = {"levy": [], "gauss": []}; cells = []
    t0 = time.time()
    for i, (s, k) in enumerate(order):
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5): sway.step()
        r_d = sway.radial()
        m_, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        for arm, use in (("levy", True), ("gauss", False)):
            cfg = SolverConfig(n_particles=N_P, max_iters=T_ITER, use_levy=False)  # per-dim jump OFF
            sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks, repair=repair)
            bx, bf, it = minimise_reseed(sol, f, time.perf_counter() + TAU, use)
            w = float(bx[0]) if bx is not None else None
            _ok, val = system_success(w, s, r_d)
            aber[arm].append(val)
        cells.append(s)
        if (i + 1) % 50 == 0: print("  %d/%d (%.0fs)" % (i + 1, len(order), time.time() - t0), flush=True)
    L = np.array(aber["levy"]); G = np.array(aber["gauss"]); cells = np.array(cells)
    pair = np.isfinite(L) & np.isfinite(G) & (L > 0) & (G > 0)
    d = np.log10(L[pair]) - np.log10(G[pair]); nz = d[d != 0]
    p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size else float("nan")
    print("\nLevy re-seed vs Gaussian re-seed, rank_stages=None, T_iter=%d, n=%d paired" % (T_ITER, pair.sum()))
    print("  median log10(ABER_levy/ABER_gauss) = %+.4f   Wilcoxon p = %.4g" % (np.median(d), p))
    print("  levy better on %d, worse on %d, tie on %d" % (np.sum(d < 0), np.sum(d > 0), np.sum(d == 0)))
    for s in SIGMAS:
        mm = cells == s; dd = np.log10(L[mm & pair]) - np.log10(G[mm & pair]); nzz = dd[dd != 0]
        ps = float(wilcoxon(nzz, alternative="two-sided").pvalue) if nzz.size >= 6 else float("nan")
        print("  sigma_s=%.2f: median dlog10=%+.4f  p=%.3g  better/worse=%d/%d" % (s, np.median(dd), ps, np.sum(dd < 0), np.sum(dd > 0)))
    D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "14_compiled_poc")
    np.savez_compressed(os.path.join(D, "levy_reseed_test.npz"), sigma_s=cells, levy=L, gauss=G)
    print("wrote levy_reseed_test.npz")


if __name__ == "__main__":
    main()
