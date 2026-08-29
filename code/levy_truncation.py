"""How much of a Levy jump survives the slew-feasibility projection?

THE QUESTION THIS SETTLES

Every sweep so far returns the same answer: the heavy-tailed step does not separate from a
Gaussian one, at any sigma from 0.002 to 0.2, any lambda from 1.1 to 1.7, one ranked stage
or twenty, gated or ungated, 25 iterations or 200. A null that robust usually means the
mechanism is not reaching the thing it is supposed to act on.

Lemma 2's advantage is entirely a statement about the TAIL: the Gaussian and the Levy step
agree closely near the origin and diverge only for large excursions, which is why the ratio
p_L/p_G grows without bound in r_tilde. So if something removes large excursions before
they are evaluated, the two distributions become observationally identical and the measured
null is an artefact of that removal rather than a property of the search.

The solver applies `repair` to every candidate, projecting it onto the slew-feasible set.
A projection is exactly an operation that truncates large excursions. This script measures
the truncation directly rather than inferring it: for a large sample of jumps it records
the displacement the operator proposed and the displacement that survived projection, for
Levy and for Gaussian steps of the same nominal scale.

WHAT THE NUMBERS MEAN

  survival ratio        ||x_repaired - x_before|| / ||x_proposed - x_before||
                        1.0 means the jump passed through untouched, 0.0 means it was
                        entirely undone.

  tail survival         the same ratio restricted to the largest decile of proposed
                        jumps. This is the number that matters: Lemma 2's advantage lives
                        there and nowhere else.

  effective tail ratio  the ratio of the 99th percentile of REALISED displacement between
                        Levy and Gaussian. If the projection is doing what is suspected,
                        this collapses toward 1.0 even though the proposed distributions
                        differ by orders of magnitude.

If the proposed tails differ sharply and the realised tails do not, then the Levy operator
cannot possibly outperform the Gaussian one in this system, the null results are explained,
and the finding is about the constraint handling rather than about heavy-tailed search.
That is a statement worth making precisely, because it identifies what would have to change
for the operator to earn its place.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from measure_all import N_P, SIGMAS, _make_problem
from hclpso_ga import levy


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
    ap.add_argument("--draws", type=int, default=240,
                    help="channel draws; each contributes N_p jump proposals per arm")
    ap.add_argument("--jump-scale", type=float, default=0.02)
    ap.add_argument("--levy-lambda", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "12_continuous"))
    a = ap.parse_args()

    from channel import SwayProcess

    rng = np.random.default_rng(20260829)
    per_cell = max(1, a.draws // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]

    prop = {"levy": [], "gauss": []}     # proposed displacement magnitude
    real = {"levy": [], "gauss": []}     # displacement surviving projection

    for s, k in order:
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        span = hi - lo
        dim = lo.size

        # a swarm sitting mid-box, the state a jump would act from
        base = lo + 0.5 * span
        X = np.tile(base, (N_P, 1))

        for tag in ("levy", "gauss"):
            if tag == "levy":
                steps = levy(rng, N_P * dim, a.levy_lambda).reshape(N_P, dim)
            else:
                steps = rng.normal(size=(N_P, dim))
            Y = X + a.jump_scale * steps * span
            Yr = repair(Y.copy()) if repair is not None else Y
            Yr = np.asarray(Yr, float).reshape(Y.shape)
            prop[tag].append(np.linalg.norm(Y - X, axis=1))
            real[tag].append(np.linalg.norm(Yr - X, axis=1))

    out = {"jump_scale": a.jump_scale, "levy_lambda": a.levy_lambda,
           "n_proposals_per_arm": int(sum(x.size for x in prop["levy"])),
           "arms": {}}

    print("  jump_scale = %.3f, lambda = %.1f, %d proposals per arm"
          % (a.jump_scale, a.levy_lambda, out["n_proposals_per_arm"]))
    print()
    print("  %-7s %12s %12s %12s %12s"
          % ("arm", "prop p50", "prop p99", "real p50", "real p99"))
    print("  " + "-" * 60)

    for tag in ("levy", "gauss"):
        P = np.concatenate(prop[tag])
        R = np.concatenate(real[tag])
        ok = np.isfinite(P) & np.isfinite(R) & (P > 0)
        P, R = P[ok], R[ok]
        surv = R / P
        big = P >= np.quantile(P, 0.9)
        e = {
            "proposed_p50": float(np.quantile(P, 0.50)),
            "proposed_p99": float(np.quantile(P, 0.99)),
            "realised_p50": float(np.quantile(R, 0.50)),
            "realised_p99": float(np.quantile(R, 0.99)),
            "survival_median": float(np.median(surv)),
            "tail_survival_median": float(np.median(surv[big])),
        }
        out["arms"][tag] = e
        print("  %-7s %12.4g %12.4g %12.4g %12.4g"
              % (tag, e["proposed_p50"], e["proposed_p99"],
                 e["realised_p50"], e["realised_p99"]))

    L, G = out["arms"]["levy"], out["arms"]["gauss"]
    prop_ratio = L["proposed_p99"] / G["proposed_p99"] if G["proposed_p99"] else float("nan")
    real_ratio = L["realised_p99"] / G["realised_p99"] if G["realised_p99"] else float("nan")
    out["proposed_tail_ratio"] = float(prop_ratio)
    out["realised_tail_ratio"] = float(real_ratio)

    print()
    print("  survival of the whole jump   levy %.3f   gauss %.3f"
          % (L["survival_median"], G["survival_median"]))
    print("  survival of the LARGEST 10%%  levy %.3f   gauss %.3f"
          % (L["tail_survival_median"], G["tail_survival_median"]))
    print()
    print("  p99 tail ratio, Levy over Gaussian")
    print("    as proposed : %8.2f x" % prop_ratio)
    print("    as realised : %8.2f x" % real_ratio)
    print()
    if np.isfinite(prop_ratio) and np.isfinite(real_ratio) and prop_ratio > 2:
        collapse = 100.0 * (1 - (real_ratio - 1) / (prop_ratio - 1)) \
            if prop_ratio > 1 else float("nan")
        out["tail_advantage_removed_pct"] = float(collapse)
        print("    the projection removes %.1f%% of the tail advantage" % collapse)
        if real_ratio < 1.2:
            print("    the realised distributions are effectively indistinguishable,")
            print("    which is sufficient to explain every null measured so far")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "levy_truncation_s%s.json" % ("%g" % a.jump_scale).replace(".", "p")), "w") as fh:
        json.dump(_stamp(out, "levy_truncation.py"), fh, indent=1)
    print()
    print("  wrote levy_truncation_s%s.json" % ("%g" % a.jump_scale).replace(".", "p"))


if __name__ == "__main__":
    main()
