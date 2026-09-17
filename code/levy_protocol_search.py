"""Design the protocol AROUND the Levy operator, instead of testing the operator
inside a protocol designed without it.

WHY THIS EXISTS

Four independent probes found no Levy advantage on this plant (per-stage jump,
feas_shift jump, Levy re-seed, compiled run at 15 iterations per checkpoint).
The diagnosis those probes support is not "heavy tails do not work" -- the
positive control shows they do, 176/200 against 0/200, when a landscape demands
escape -- but "this swarm never gets trapped": 30 chaotically-initialised
particles on a box whose local minima lie within ~4x of each other, with PSO's
social term pulling every particle toward the incumbent, do not stagnate, and an
escape operator has nothing to escape from.

That is a statement about the CONFIGURATION, not about the operator. The
configuration is a design choice: N_p = 30 is Table 4's, not a law. This script
sweeps the choices that decide whether stagnation happens at all, and asks
whether there is a corner of that space where the heavy tail earns its place:

    N_p          4, 8, 15, 30    -- initial coverage vs. escape as the binding
                                   mechanism. At N_p = 4 the swarm cannot cover
                                   a 60-D box by initialisation and must move.
    jump_scale   0.02, 0.10, 0.30 -- the step must match the basin separation to
                                   escape it; 0.02 of the box is the deployed
                                   value and may simply be too small to leave a
                                   basin at all.
    gate         off / on        -- fire always (deployed) vs. only when the
                                   incumbent window is flat (the manuscript's
                                   own Var(J_gbest) < eps_s trigger). A gate
                                   spends jumps where they can pay.

Held fixed and chosen so the tail is not destroyed before it acts:
    jump_mode = feas_shift   the block-common geometry measured to pass the slew
                             projection intact (realised/proposed p99 = 0.93,
                             against 0.21 for the deployed per-stage geometry)
    rank_stages = None       the coupled-trajectory objective, where the
                             landscape probe locates the multimodality
    T_iter = 25, tau = 20 ms so every arm completes its budget

Levy and Gaussian arms are PAIRED: identical seed, identical trial, identical
everything but the step distribution. The comparison is therefore of tail
weight alone. Scoring is system_metric's certified post-EGC evaluator.

This reports what it finds in either direction, including "no corner helps".

Usage: python levy_protocol_search.py [--trials 200]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_all as _ma
_ma.RANK_STAGES = None                      # coupled trajectory: where the minima are
from measure_all import SIGMAS, _make_problem, system_success
from hclpso_ga import HCLPSOGA, SolverConfig
from channel import SwayProcess
from scipy.stats import wilcoxon

T_ITER, TAU = 25, 20e-3
NP_GRID = [4, 8, 15, 30]
SCALE_GRID = [0.02, 0.10, 0.30]
GATE_GRID = [False, True]


def run_arm(lo, hi, blocks, repair, block_slew, f, seed, n_p, scale, gate, use_levy):
    cfg = SolverConfig(n_particles=n_p, max_iters=T_ITER, use_levy=use_levy,
                       jump_scale=scale, jump_mode="feas_shift",
                       stagnation_gated=gate)
    sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks, repair=repair,
                   block_slew=block_slew)
    dl = time.perf_counter() + TAU
    r = sol.minimise(lambda X: (f(X), {}),
                     checkpoint=lambda it, bf: time.perf_counter() > dl)
    w = float(r.best_x[0]) if r.best_x is not None else None
    return w, int(r.iterations), int(getattr(sol, "_jump_fired", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    a = ap.parse_args()

    per_cell = max(1, a.trials // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)

    # pre-build every trial's problem once; all configurations share them
    problems = []
    for s, k in order:
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        problems.append((s, r_d, seed, f, lo, hi, blocks, repair, m.block_slew()))

    print("  %d trials, coupled-trajectory ranking, feas_shift jumps, T_iter=%d" % (len(problems), T_ITER))
    print("\n  %-4s %-6s %-6s %10s %10s %11s %8s %8s %7s" %
          ("N_p", "scale", "gate", "ABER levy", "ABER gauss", "med dlog10", "Wilcox p", "L better", "fires"))
    print("  " + "-" * 92)

    results = {}
    t0 = time.time()
    for n_p in NP_GRID:
        for scale in SCALE_GRID:
            for gate in GATE_GRID:
                L, G, fires = [], [], []
                for (s, r_d, seed, f, lo, hi, blocks, repair, bslew) in problems:
                    wl, _it, nf = run_arm(lo, hi, blocks, repair, bslew, f, seed, n_p, scale, gate, True)
                    wg, _it2, _ = run_arm(lo, hi, blocks, repair, bslew, f, seed, n_p, scale, gate, False)
                    L.append(system_success(wl, s, r_d)[1])
                    G.append(system_success(wg, s, r_d)[1])
                    fires.append(nf)
                L = np.array(L); G = np.array(G)
                pair = np.isfinite(L) & np.isfinite(G) & (L > 0) & (G > 0)
                d = np.log10(L[pair]) - np.log10(G[pair])      # < 0 : Levy BETTER
                nz = d[d != 0]
                p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size >= 6 else float("nan")
                key = "np%d_s%.2f_%s" % (n_p, scale, "gated" if gate else "always")
                results[key] = dict(n_p=n_p, jump_scale=scale, gated=bool(gate),
                                    n_paired=int(pair.sum()),
                                    median_aber_levy=float(np.median(L[pair])),
                                    median_aber_gauss=float(np.median(G[pair])),
                                    median_log10_delta=float(np.median(d)) if d.size else float("nan"),
                                    wilcoxon_p=p, levy_better=int(np.sum(d < 0)),
                                    levy_worse=int(np.sum(d > 0)), tie=int(np.sum(d == 0)),
                                    mean_jumps_fired=float(np.mean(fires)))
                e = results[key]
                print("  %-4d %-6.2f %-6s %10.3e %10.3e %+11.4f %8.4g %4d/%-4d %7.1f"
                      % (n_p, scale, "gated" if gate else "always", e["median_aber_levy"],
                         e["median_aber_gauss"], e["median_log10_delta"], p,
                         e["levy_better"], e["levy_worse"], e["mean_jumps_fired"]), flush=True)
    print("\n  swept %d configurations in %.0fs" % (len(results), time.time() - t0))

    sig = {k: v for k, v in results.items() if np.isfinite(v["wilcoxon_p"]) and v["wilcoxon_p"] < 0.05}
    fav = {k: v for k, v in sig.items() if v["median_log10_delta"] < 0}
    print("  configurations separating at p<0.05: %d, of which Levy-favourable: %d" % (len(sig), len(fav)))
    for k, v in sorted(fav.items(), key=lambda kv: kv[1]["wilcoxon_p"]):
        print("    %-22s p=%.3g  median dlog10=%+.4f  (Levy better on %d/%d)"
              % (k, v["wilcoxon_p"], v["median_log10_delta"], v["levy_better"], v["n_paired"]))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "levy_protocol_search.json"), "w") as fh:
        json.dump(dict(what="Levy vs Gaussian on paired draws across swarm size, jump scale and "
                            "stagnation gate, feas_shift geometry, coupled-trajectory ranking; "
                            "scored by system_metric.system_success",
                       t_iter=T_ITER, tau_s=TAU, n_trials=len(problems),
                       np_grid=NP_GRID, scale_grid=SCALE_GRID, results=results), fh, indent=1)
    print("  wrote levy_protocol_search.json")


if __name__ == "__main__":
    main()
