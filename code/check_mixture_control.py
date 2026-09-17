"""The mixture sweep's control must BE the deployed initialiser, not resemble it.

`levy_mixture_probe.py` compares every allocation against a single f = 0 arm and calls
that arm the deployed bounded chaotic cloud. If the restructured `_initialise` consumes
the random stream differently at f = 0 than `init_law = "chaotic"` does, the control is a
fourth arm wearing the control's name and every paired comparison in the sweep is against
the wrong baseline. That is cheap to check and expensive to get wrong, so it is checked.

Also verifies that a mixture at fraction f really places round(f * N_p) particles on the
unbounded cloud and leaves the rest on the bounded one, by counting how many initial
particles sit further from the anchor than the bounded cloud can reach.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hclpso_ga import HCLPSOGA, SolverConfig
from mpc_loop import BeamSteeringMPC
from measure_all import GBAR_OP_DB

STRONG = (1.2, 1.1)


def build(law, frac, anchor, seed, spread=0.00125):
    m = BeamSteeringMPC(*STRONG, 0.05, 10.0 ** (GBAR_OP_DB / 10.0), horizon=20,
                        seed=0, rank_stages=None)
    cfg = SolverConfig(n_particles=30, max_iters=25, init_centre=anchor,
                       init_spread=spread, init_law=law, init_levy_fraction=frac)
    s = HCLPSOGA(m.lower(), m.upper(), cfg, seed=seed, blocks=m.blocks(),
                 repair=m.repair, block_slew=m.block_slew())
    return s, np.asarray(m.lower(), float), np.asarray(m.upper(), float)


def main():
    m = BeamSteeringMPC(*STRONG, 0.05, 10.0 ** (GBAR_OP_DB / 10.0), horizon=20,
                        seed=0, rank_stages=None)
    lo, hi = np.asarray(m.lower(), float), np.asarray(m.upper(), float)
    rng = np.random.default_rng(7)
    anchor = lo + rng.random(lo.size) * (hi - lo)
    anchor = m.repair(anchor[None, :])[0]

    print("  control identity: init_law='chaotic' against every law at fraction 0")
    ref, _, _ = build("chaotic", 1.0, anchor, 123)
    x_ref = ref._initialise()
    ok = True
    for law in ("levy", "gauss", "chaotic"):
        s, _, _ = build(law, 0.0, anchor, 123)
        x = s._initialise()
        same = x.shape == x_ref.shape and np.array_equal(x, x_ref)
        print("    %-8s f=0 : bit-identical to deployed chaotic = %s" % (law, same))
        ok &= bool(same)

    print("\n  allocation actually applied (particles outside the bounded cloud's reach)")
    span = hi - lo
    for law in ("levy", "gauss"):
        for f in (0.0, 0.10, 0.25, 0.50, 1.00):
            s, _, _ = build(law, f, anchor, 321)
            x = s._initialise()
            # a bounded-cloud particle cannot exceed spread*span in ANY coordinate
            # before projection; count rows that do, as a lower bound on tail rows
            d = np.abs(x - anchor[None, :])
            outside = int(np.sum(np.any(d > (0.00125 * span)[None, :] * (1 + 1e-9), axis=1)))
            print("    %-6s f=%.2f : expected %2d tail rows, %2d of 30 rows exceed the "
                  "bounded radius after projection" % (law, f, int(round(f * 30)), outside))

    print("\n  %s" % ("control is sound" if ok else "CONTROL IS NOT THE DEPLOYED INITIALISER"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
