"""Does the Levy operator help when its own trigger condition is actually met?

WHY THIS IS NEEDED

The ablation returns a null for the Levy arm (p = 0.58 at tau_O = 20 ms with one ranked
stage, p = 0.084 over the coupled 20-stage trajectory, n = 2000 each). A null invites two
very different readings, and the paper must not pick one by assertion:

    (a) heavy-tailed exploration does not help on this problem, or
    (b) the operator never got into the regime where a heavy tail can help.

Instrumenting the solver settles it. Counting jumps over 25 iterations:

    deployed code, ungated     gate open 25.0/25 iterations, 185.6 jumps per solve
    as the manuscript describes, stagnation-gated   0.2/25 iterations, 1.3 jumps

Neither configuration tests the claim. The deployed one fires a jump on a quarter of the
particles every single iteration at 2% of the box: that is persistent small noise, and
against persistent small noise a heavy tail has no advantage over a Gaussian of the same
scale, which is exactly the measured null. The manuscript's own gated mechanism almost
never fires, because within T_iter = 25 the incumbent is still improving and the swarm
never stagnates.

So the claim in Lemma 2, a heavy-tailed advantage over Gaussian perturbation, has not been
tested by either configuration. This script tests it, by running the gated mechanism long
enough that stagnation actually occurs, and comparing Levy against Gaussian steps under an
identical trigger. The comparison is then purely one of step DISTRIBUTION.

WHAT IS AND IS NOT ESTABLISHED

This runs on the deployed objective at the deployed operating point, with the iteration cap
lifted. It answers "does the heavy tail pay off once the trigger fires", which is a
question about the operator. It does NOT show that the deployed configuration benefits,
and it must not be reported as if it did: at T_iter = 25 the trigger fires 0.2 times in 25
iterations, so whatever this measures is unavailable to the real-time loop as configured.
That gap is the finding, and it belongs in the paper next to this number.

Usage:
    python levy_mechanism_probe.py [--trials 300] [--iters 200]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from measure_all import (N_P, SIGMAS, GBAR_OP_DB, _make_problem, system_success,
                         pin_and_prioritise)
from ablation_continuous import wilcoxon_signed_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--iters", type=int, default=200,
                    help="iteration cap; the deployed value is 25, at which the "
                         "stagnation gate fires 0.2 times per solve")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "12_continuous"))
    a = ap.parse_args()

    from hclpso_ga import HCLPSOGA, SolverConfig
    from channel import SwayProcess

    print("  pinning %s" % pin_and_prioritise())
    print("  gated Levy vs gated Gaussian, %d trials, T_iter = %d, no wall-clock cap"
          % (a.trials, a.iters))
    print()

    per_cell = max(1, a.trials // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)

    ARMS = {
        "gated_levy": dict(stagnation_gated=True, use_levy=True),
        "gated_gaussian": dict(stagnation_gated=True, use_levy=False),
        "ungated_levy": dict(stagnation_gated=False, use_levy=True),
        "ungated_gaussian": dict(stagnation_gated=False, use_levy=False),
    }
    aber = {k: [] for k in ARMS}
    fired = {k: [] for k in ARMS}
    gopen = {k: [] for k in ARMS}

    t0 = time.time()
    for i, (s, k) in enumerate(order):
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        for arm, over in ARMS.items():
            cfg = SolverConfig(n_particles=N_P, max_iters=a.iters, **over)
            sol = HCLPSOGA(lo, hi, cfg, seed=seed, blocks=blocks, repair=repair)
            r = sol.minimise(lambda X: (f(X), {}))
            w = float(r.best_x[0]) if r.best_x is not None else None
            _ok, v = system_success(w, s, r_d)
            aber[arm].append(v)
            fired[arm].append(getattr(sol, "_jump_fired", 0))
            gopen[arm].append(getattr(sol, "_gate_open_iters", 0))
        if (i + 1) % 50 == 0:
            print("    %d/%d  (%.0fs)" % (i + 1, len(order), time.time() - t0),
                  flush=True)

    print("    %d trials done (%.0fs)" % (len(order), time.time() - t0))
    print()
    print("  %-18s %12s %12s %11s" %
          ("arm", "median ABER", "jumps/solve", "gate open"))
    print("  " + "-" * 58)
    for arm in ARMS:
        v = np.array(aber[arm], float)
        fin = np.isfinite(v)
        print("  %-18s %12.4e %12.1f %8.1f/%d"
              % (arm, np.median(v[fin]) if fin.any() else float("nan"),
                 np.mean(fired[arm]), np.mean(gopen[arm]), a.iters))

    out = {"iters": a.iters, "n_trials": len(order), "comparisons": {}}
    print()
    print("  PAIRED COMPARISONS, Levy against Gaussian under an identical trigger")
    print("  %-34s %11s %9s %7s %7s" % ("comparison", "median dlog10", "p", "levy+", "levy-"))
    print("  " + "-" * 74)
    for tag, la, ga in (("stagnation-gated", "gated_levy", "gated_gaussian"),
                        ("ungated (as deployed)", "ungated_levy", "ungated_gaussian")):
        L = np.array(aber[la], float)
        G = np.array(aber[ga], float)
        m_ = np.isfinite(L) & np.isfinite(G) & (L > 0) & (G > 0)
        d = np.log10(L[m_]) - np.log10(G[m_])      # negative => Levy BETTER
        w = wilcoxon_signed_rank(d)
        out["comparisons"][tag] = {
            "n_paired": int(m_.sum()),
            "median_log10_levy_minus_gaussian": float(np.median(d)) if d.size else float("nan"),
            "levy_better": int((d < 0).sum()), "levy_worse": int((d > 0).sum()),
            "tie": int((d == 0).sum()), "wilcoxon": w,
            "mean_jumps_per_solve": float(np.mean(fired[la])),
            "mean_gate_open_iters": float(np.mean(gopen[la])),
        }
        print("  %-34s %+11.5f %9.4f %7d %7d"
              % (tag, np.median(d) if d.size else float("nan"), w["p"],
                 int((d < 0).sum()), int((d > 0).sum())))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "levy_mechanism_probe.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print()
    print("  NEGATIVE median means Levy reached a LOWER (better) ABER than Gaussian.")
    print("  Whatever this shows, the deployed loop runs T_iter=25, where the gate opens")
    print("  0.2 times per solve, so a gated benefit measured here is not available to it.")


if __name__ == "__main__":
    main()
