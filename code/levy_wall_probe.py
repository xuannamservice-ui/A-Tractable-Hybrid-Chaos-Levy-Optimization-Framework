"""Does the Levy operator separate where the plant actually has a wall to jump?

THE HYPOTHESIS THIS TESTS

The positive control of Section VII-D6 shows the operator escaping 176/200
against Gaussian's 0/200 on a landscape carrying an infeasible wall "of the kind
the z > 8 guard creates". The deployed operating point has no such wall: at
sigma_s = 0.10 m the widest admissible beam gives z_worst ~ 4.52, comfortably
inside the guard, so nothing in the box is refused and there is nothing to jump
over. The manuscript states where the wall does appear -- "diverge only beyond
sigma_s ~ 0.1 m, where z_worst reaches 17.97 and 40.43 at sigma_s = 0.2 and
0.3 m".

Every previous probe missed the combination that matters:
  - the per-stratum split ran the DEPLOYED per-stage jump, whose tail the slew
    projection removes (realised/proposed p99 = 0.21) before it can reach any
    wall;
  - the feas_shift probes, whose tail does survive (0.93), POOLED all four
    jitter strata, diluting an effect confined to two of them.

So: feas_shift geometry, split by jitter, with the inadmissible fraction of each
stratum's box measured alongside, so the result can be read against whether a
wall is present at all. Levy and Gaussian arms are paired on identical seeds and
identical problems, differing only in the step distribution.

Usage: python levy_wall_probe.py [--trials 240] [--scale 0.10]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_all as _ma
_ma.RANK_STAGES = None
from measure_all import SIGMAS, GBAR_OP_DB, _make_problem, system_success
from hclpso_ga import HCLPSOGA, SolverConfig
from channel import SwayProcess, beam_geometry
from rtodt_fast import z_of
from mpc_loop import manuscript_wz_box, Z_MAX
from scipy.stats import wilcoxon

T_ITER, TAU = 25, 20e-3
ALPHA, BETA = 1.2, 1.1


def wall_fraction(sigma_s, n=2001):
    """Fraction of the divergence box the z > z_max guard refuses outright."""
    lo, hi = manuscript_wz_box(sigma_s)
    w = np.linspace(lo, hi, n)
    A0, _ = beam_geometry(w)
    z = z_of(ALPHA, BETA, A0, 10.0 ** (GBAR_OP_DB / 10.0))
    return float(np.mean(z > Z_MAX)), float(np.nanmax(z))


def run(lo, hi, blocks, repair, bslew, f, seed, n_p, scale, gate, use_levy):
    cfg = SolverConfig(n_particles=n_p, max_iters=T_ITER, use_levy=use_levy,
                       jump_scale=scale, jump_mode="feas_shift", stagnation_gated=gate)
    sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks, repair=repair, block_slew=bslew)
    dl = time.perf_counter() + TAU
    r = sol.minimise(lambda X: (f(X), {}), checkpoint=lambda it, bf: time.perf_counter() > dl)
    return (float(r.best_x[0]) if r.best_x is not None else None), int(r.rejected_by_guard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=240)
    ap.add_argument("--n-p", type=int, default=30)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.02, 0.10, 0.30])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    a = ap.parse_args()

    print("  inadmissible fraction of the divergence box, by jitter stratum:")
    walls = {}
    for s in SIGMAS:
        frac, zmax = wall_fraction(s)
        walls[str(s)] = dict(refused_fraction=frac, z_max_in_box=zmax)
        print("    sigma_s = %.2f m : %5.1f%% of the box refused (z_worst = %6.2f)" % (s, 100 * frac, zmax))

    per_cell = max(1, a.trials // len(SIGMAS))
    problems = {s: [] for s in SIGMAS}
    for s in SIGMAS:
        for k in range(per_cell):
            seed = 700000 + int(s * 1000) * 1000 + k
            sway = SwayProcess(s, seed=seed)
            for _ in range(5):
                sway.step()
            r_d = sway.radial()
            m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
            problems[s].append((r_d, seed, f, lo, hi, blocks, repair, m.block_slew()))

    print("\n  N_p = %d, feas_shift jumps, coupled-trajectory ranking, %d trials per stratum"
          % (a.n_p, per_cell))
    print("\n  %-7s %-6s %-6s %10s %10s %11s %9s %9s %9s" %
          ("sigma_s", "scale", "gate", "ABER levy", "ABER gauss", "med dlog10", "Wilcox p", "L better", "guard rej"))
    print("  " + "-" * 96)

    results = {}
    t0 = time.time()
    for s in SIGMAS:
        for scale in a.scales:
            for gate in (False, True):
                L, G, rejL = [], [], []
                for (r_d, seed, f, lo, hi, blocks, repair, bslew) in problems[s]:
                    wl, rl = run(lo, hi, blocks, repair, bslew, f, seed, a.n_p, scale, gate, True)
                    wg, _rg = run(lo, hi, blocks, repair, bslew, f, seed, a.n_p, scale, gate, False)
                    L.append(system_success(wl, s, r_d)[1])
                    G.append(system_success(wg, s, r_d)[1])
                    rejL.append(rl)
                L = np.array(L); G = np.array(G)
                pair = np.isfinite(L) & np.isfinite(G) & (L > 0) & (G > 0)
                d = np.log10(L[pair]) - np.log10(G[pair])
                nz = d[d != 0]
                p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size >= 6 else float("nan")
                key = "s%.2f_sc%.2f_%s" % (s, scale, "gated" if gate else "always")
                results[key] = dict(sigma_s=s, jump_scale=scale, gated=bool(gate),
                                    refused_fraction=walls[str(s)]["refused_fraction"],
                                    n_paired=int(pair.sum()), n_nonfinite=int((~pair).sum()),
                                    median_aber_levy=float(np.median(L[pair])) if pair.any() else float("nan"),
                                    median_aber_gauss=float(np.median(G[pair])) if pair.any() else float("nan"),
                                    median_log10_delta=float(np.median(d)) if d.size else float("nan"),
                                    wilcoxon_p=p, levy_better=int(np.sum(d < 0)), levy_worse=int(np.sum(d > 0)),
                                    tie=int(np.sum(d == 0)), mean_guard_rejections=float(np.mean(rejL)))
                e = results[key]
                print("  %-7.2f %-6.2f %-6s %10.3e %10.3e %+11.4f %9.4g %4d/%-4d %9.0f"
                      % (s, scale, "gated" if gate else "always", e["median_aber_levy"], e["median_aber_gauss"],
                         e["median_log10_delta"], p, e["levy_better"], e["levy_worse"],
                         e["mean_guard_rejections"]), flush=True)
    print("\n  %d configurations in %.0fs" % (len(results), time.time() - t0))

    fav = {k: v for k, v in results.items()
           if np.isfinite(v["wilcoxon_p"]) and v["wilcoxon_p"] < 0.05 and v["median_log10_delta"] < 0}
    adv = {k: v for k, v in results.items()
           if np.isfinite(v["wilcoxon_p"]) and v["wilcoxon_p"] < 0.05 and v["median_log10_delta"] > 0}
    print("  Levy-favourable at p<0.05: %d   Levy-unfavourable at p<0.05: %d" % (len(fav), len(adv)))
    for k, v in sorted(fav.items(), key=lambda kv: kv[1]["wilcoxon_p"]):
        print("    FAVOURABLE %-24s p=%.3g  median dlog10=%+.4f  better %d/%d  (box refused %.0f%%)"
              % (k, v["wilcoxon_p"], v["median_log10_delta"], v["levy_better"], v["n_paired"],
                 100 * v["refused_fraction"]))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "levy_wall_probe.json"), "w") as fh:
        json.dump(dict(what="Levy vs Gaussian, feas_shift geometry, split by jitter stratum, with the "
                            "inadmissible (z>z_max) fraction of each stratum's box reported alongside; "
                            "paired seeds, coupled-trajectory ranking, scored by system_metric",
                       t_iter=T_ITER, n_p=a.n_p, trials_per_stratum=per_cell,
                       walls=walls, results=results), fh, indent=1)
    print("  wrote levy_wall_probe.json")


if __name__ == "__main__":
    main()
