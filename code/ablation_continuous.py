"""Re-score the component ablation on a CONTINUOUS metric.

WHY THIS EXISTS

Table 11 of the manuscript reports all five ablation arms at 50/200 with discordant counts
b = c = 0 and an exact McNemar p of 1.000. That is a null, but it is not evidence that the
components do nothing. It is evidence that the SCORING RULE cannot see them: the criterion
is a binary indicator, post-EGC system ABER <= 1e-6 at 38 dB, and at that operating point
the link budget saturates it. One of the four jitter strata admits a passing beam and every
arm finds it; three admit none and every arm fails. A metric that returns the same answer
for every arm by construction has no power to separate them, and the manuscript says so.

The manuscript then identifies the fix -- "resolving component contributions requires a
continuous quality metric" -- and leaves it to future work. This script runs it.

The continuous metric was already being computed and discarded. In measure_all.py:

    ok, _v = system_success(w, s, r_d)

system_success returns (indicator, achieved post-EGC ABER). The ablation kept the indicator
and threw away the ABER. Nothing new has to be modelled: the same beam, the same channel
draw, the same evaluator, the same seeds. Only the number that gets recorded changes.

WHAT IS COMPARED

For each arm and each paired draw we record the achieved ABER, then compare each ablated
arm against the full kernel on the SAME draws:

  - median achieved ABER per arm, and the paired difference in log10 ABER
  - the Wilcoxon signed-rank test on those paired differences, which is the continuous
    analogue of the McNemar test used for the binary indicator
  - win/loss/tie counts, so a reader can see whether a significant result rests on many
    small differences or a few large ones

A draw is only usable if BOTH arms return a finite ABER, since a paired test needs pairs;
the count of unusable draws is reported rather than silently dropped.

WHAT THIS CANNOT SHOW

This measures the components against each other inside this solver, on this problem, at
this operating point. It does not establish that chaotic initialisation or Levy flight is
superior to some other exploration mechanism, and it does not transfer to a different
budget. The binary result stands as reported; this is a second, more sensitive reading of
the same campaign, not a replacement for it.

Usage:
    python ablation_continuous.py [--trials 400] [--out ../data/12_continuous]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from measure_all import (ARMS, N_P, SIGMAS, TAU_O, T_ITER, GBAR_OP_DB, TARGET,
                         _make_problem, system_success, pin_and_prioritise,
                         background_load)


def _stamp(obj, script, argv=None):
    """Record what produced this artefact, in the two fields build_manifest.py reads.

    `generated_by` must be a bare path: the manifest validates it with os.path.basename
    and requires the result to exist under code/, so anything carrying arguments is
    rejected. The full invocation goes in `command`, which is what a reader needs to
    reproduce this particular run rather than the script's defaults.
    """
    import sys as _sys
    args = " ".join(argv if argv is not None else _sys.argv[1:])
    out = {"generated_by": "code/%s" % script,
           "command": ("python code/%s %s" % (script, args)).rstrip()}
    out.update(obj)
    return out


def wilcoxon_signed_rank(d):
    """Exact-ish Wilcoxon signed-rank on paired differences d.

    Zeros are dropped (Wilcoxon's own convention), ties get average ranks. For n > 20 the
    normal approximation with a continuity correction is used, which is standard and is
    what scipy does; below that scipy's exact routine is used when available.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0.0]
    n = nz.size
    if n == 0:
        return dict(n=0, W=float("nan"), p=float("nan"), method="no non-zero pairs")
    try:
        from scipy.stats import wilcoxon
        st, p = wilcoxon(nz, alternative="two-sided",
                         mode="exact" if n <= 25 else "approx")
        return dict(n=int(n), W=float(st), p=float(p),
                    method="scipy exact" if n <= 25 else "scipy normal approx")
    except Exception:
        order = np.argsort(np.abs(nz))
        ranks = np.empty(n, float)
        ranks[order] = np.arange(1, n + 1)
        wp = float(ranks[nz > 0].sum())
        wm = float(ranks[nz < 0].sum())
        W = min(wp, wm)
        mu = n * (n + 1) / 4.0
        sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        from math import erfc, sqrt
        z = (abs(W - mu) - 0.5) / sd if sd > 0 else 0.0
        return dict(n=int(n), W=W, p=float(erfc(z / sqrt(2.0))),
                    method="normal approx, no scipy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    # The deployed budget lets the interpreted solver finish ONE iteration. Levy flight
    # fires on stagnation, GA refinement runs after the swarm has moved, and the fidelity
    # ladder only matters across repeated evaluations: none of them can act in a single
    # iteration, so at 600 us three of the five arms are bit-identical to the full kernel.
    # Running the same comparison at a budget that admits the full T_iter separates "the
    # component does nothing" from "the component never got to run".
    ap.add_argument("--tau-us", type=float, default=TAU_O * 1e6,
                    help="solver-time checkpoint in microseconds (default: the deployed "
                         "%.0f us; 20000 admits the full T_iter=%d)"
                         % (TAU_O * 1e6, T_ITER))
    # THE FAIR TEST FOR LEVY FLIGHT.
    #
    # measure_all.py sets RANK_STAGES = 1, and the manuscript's own argument for the Levy
    # operator says why that is the wrong setting to judge it on:
    #
    #   "the slew-rate coupling, not the per-stage P_e(xi) cost (which is unimodal),
    #    generates the multimodality"
    #
    # With a single ranked stage the objective is unimodal, so a heavy-tailed jump operator
    # whose whole purpose is escaping local minima has nothing to escape. Testing Levy
    # there measures nothing about Levy. Ranking over the coupled multi-stage trajectory is
    # where the manuscript predicts the multimodality lives, so that is where the operator
    # has to be judged. This flag makes that prediction falsifiable.
    ap.add_argument("--rank-stages", type=int, default=None,
                    help="stages ranked in the objective (default: measure_all's "
                         "RANK_STAGES; the manuscript locates the multimodality in the "
                         "coupled multi-stage trajectory, not the unimodal single stage)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "12_continuous"))
    a = ap.parse_args()
    tau = a.tau_us * 1e-6
    rank = a.rank_stages
    if rank is not None:
        import measure_all
        measure_all.RANK_STAGES = int(rank)


    from hclpso_ga import HCLPSOGA, SolverConfig
    from channel import SwayProcess

    pin = pin_and_prioritise()
    print("  pinning %s | background CPU %.1f%%" % (pin, background_load()))
    import measure_all as _ma
    print("  %d arms x %d trials, tau_O = %.0f us, criterion ABER at %.0f dB, "
          "rank_stages = %d"
          % (len(ARMS), a.trials, tau * 1e6, GBAR_OP_DB, _ma.RANK_STAGES))
    print()

    per_cell = max(1, a.trials // len(SIGMAS))
    aber = {arm: [] for arm in ARMS}
    okf = {arm: [] for arm in ARMS}
    iters = {arm: [] for arm in ARMS}
    cells = []

    # the SAME ordering and seeds as the binary ablation, so the two are comparable
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)

    t0 = time.time()
    for i, (s, k) in enumerate(order):
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        for arm, over in ARMS.items():
            cfg = SolverConfig(n_particles=N_P, max_iters=T_ITER, **over)
            sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks, repair=repair)
            dl = time.perf_counter() + tau
            r = sol.minimise(lambda X: (f(X), {}),
                             checkpoint=lambda it, bf: time.perf_counter() > dl)
            w = float(r.best_x[0]) if r.best_x is not None else None
            ok, v = system_success(w, s, r_d)
            aber[arm].append(v)
            okf[arm].append(bool(ok))
            iters[arm].append(int(r.iterations))
        cells.append(s)
        if (i + 1) % 50 == 0:
            print("    %d/%d  (%.0fs)" % (i + 1, len(order), time.time() - t0),
                  flush=True)

    print("    %d trials done (%.0fs)" % (len(order), time.time() - t0))
    print()

    cells = np.array(cells)
    full = np.array(aber["full"], float)
    out = {
        "criterion": ("achieved post-EGC system ABER at %.0f dB, strong turbulence, "
                      "all four sigma_s, paired draws and seeds identical to the "
                      "binary ablation" % GBAR_OP_DB),
        "binary_threshold_for_reference": TARGET,
        "tau_o_us": tau * 1e6,
        "rank_stages": int(_ma.RANK_STAGES),
        "trials_per_cell": per_cell,
        "n_trials": int(len(order)),
        "arms": {},
    }

    print("  %-18s %11s %11s %9s %8s %8s %8s" %
          ("arm", "median ABER", "vs full", "Wilcoxon p", "better", "worse", "tie"))
    print("  " + "-" * 78)

    for arm in ARMS:
        v = np.array(aber[arm], float)
        fin = np.isfinite(v)
        e = {
            "median_aber": float(np.median(v[fin])) if fin.any() else float("nan"),
            "mean_log10_aber": float(np.mean(np.log10(v[fin][v[fin] > 0])))
                               if (fin & (v > 0)).any() else float("nan"),
            "n_finite": int(fin.sum()),
            "n_nonfinite": int((~fin).sum()),
            "binary_rate": float(np.mean(okf[arm])),
            "median_iterations": float(np.median(iters[arm])),
        }
        if arm == "full":
            print("  %-18s %11.3e %11s %9s %8s %8s %8s"
                  % (arm, e["median_aber"], "-", "-", "-", "-", "-"))
        else:
            pair = np.isfinite(v) & np.isfinite(full) & (v > 0) & (full > 0)
            d = np.log10(v[pair]) - np.log10(full[pair])   # >0 means arm is WORSE
            w = wilcoxon_signed_rank(d)
            e.update({
                "n_paired": int(pair.sum()),
                "n_unpaired_dropped": int((~pair).sum()),
                "median_log10_delta_vs_full": float(np.median(d)) if d.size else float("nan"),
                "arm_better": int(np.sum(d < 0)),
                "arm_worse": int(np.sum(d > 0)),
                "tie": int(np.sum(d == 0)),
                "wilcoxon": w,
            })
            print("  %-18s %11.3e %+11.4f %9.4f %8d %8d %8d"
                  % (arm, e["median_aber"], e["median_log10_delta_vs_full"],
                     w["p"], e["arm_better"], e["arm_worse"], e["tie"]))
        out["arms"][arm] = e

    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "ablation_continuous_tau%.0f_rank%d.npz" % (tau*1e6, _ma.RANK_STAGES)),
                        sigma_s=cells,
                        **{arm: np.array(aber[arm], float) for arm in ARMS})
    with open(os.path.join(a.out, "ablation_continuous_tau%.0f_rank%d.json" % (tau*1e6, _ma.RANK_STAGES)), "w") as fh:
        json.dump(_stamp(out, "ablation_continuous.py"), fh, indent=1)

    print()
    print("  wrote %s" % os.path.join(a.out, "ablation_continuous_tau%.0f_rank%d.json" % (tau*1e6, _ma.RANK_STAGES)))
    print()
    print("  READ THE SIGN: median_log10_delta_vs_full > 0 means the ABLATED arm reached a")
    print("  WORSE (higher) ABER than the full kernel, i.e. the removed component helped.")
    print("  A p above 0.05 means this campaign does not separate that arm from the full")
    print("  kernel even on the continuous metric.")


if __name__ == "__main__":
    main()
