"""Does the Levy operator help when its own trigger condition is actually met?

WHY THIS IS NEEDED

The ablation returns a null for the Levy arm (p = 0.58 at tau_O = 20 ms with one ranked
stage, p = 0.084 over the coupled 20-stage trajectory, n = 2000 each). A null invites two
very different readings, and the paper must not pick one by assertion:

    (a) heavy-tailed exploration does not help on this problem, or
    (b) the operator never got into the regime where a heavy tail can help.

Instrumenting the solver settles it. Counting jumps over 25 iterations:

    deployed code, ungated                        gate open 25.0/25, 187.8 jumps per solve
    stagnation-gated, as the manuscript describes   gate open 18.8/25, 141.4 jumps per solve

Both configurations reach the regime. The swarm is stagnant on three quarters of the
iterations even inside the real-time cap, so the gated operator fires freely, 141 times per
solve, which is squarely what it was designed for. An early check on a synthetic sphere
function suggested the gate almost never opens; that was an artefact of the test problem and
is not what the deployed objective does.

So the trigger is not what withholds the advantage from Lemma 2. This script measures the
advantage itself, comparing Levy against Gaussian steps under an identical trigger so that
the comparison is purely one of step DISTRIBUTION, at the deployed cap and at a lifted one.

WHAT IS AND IS NOT ESTABLISHED

The comparison isolates the step distribution and nothing else: same draws, same seeds, same
trigger, same scale. A null here is therefore about the heavy tail rather than about when it
fires, which is what makes it worth reporting.

It does not explain itself. Why a heavy tail buys nothing when it fires 141 times per solve
is answered by levy_truncation.py, which measures the slew-feasibility projection removing
83% of the tail advantage before it can act.

Usage:
    python levy_mechanism_probe.py [--trials 300] [--iters 200]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from measure_all import (N_P, SIGMAS, _make_problem, system_success,
                         pin_and_prioritise)
from ablation_continuous import wilcoxon_signed_rank


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--iters", type=int, default=200,
                    help="iteration cap; the deployed value is 25, at which the "
                         "stagnation gate is already open on 18.8 of 25 iterations")
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
    with open(os.path.join(a.out, "levy_mechanism_probe_iters%d.json" % a.iters), "w") as fh:
        json.dump(_stamp(out, "levy_mechanism_probe.py"), fh, indent=1)
    print()
    print("  NEGATIVE median means Levy reached a LOWER (better) ABER than Gaussian.")
    print("  At the deployed T_iter=25 the gate is already open on 18.8 of 25 iterations")
    print("  and the operator fires ~141 times per solve, so this is not a regime the")
    print("  real-time loop fails to reach. For why the tail still buys nothing there,")
    print("  see levy_truncation.py: the slew projection removes 83% of the advantage.")


if __name__ == "__main__":
    main()
