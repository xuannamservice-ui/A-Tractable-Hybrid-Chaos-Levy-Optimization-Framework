"""Is the heavy tail worth an ALLOCATION, even though it loses every pure-arm contest?

WHY THIS EXPERIMENT EXISTS, AND WHY THE EARLIER ONES DO NOT ANSWER IT

Seven measurements in this project have compared PURE strategies: a swarm drawn entirely
from a heavy-tailed law against one drawn entirely from a Gaussian or entirely from the
deployed bounded chaotic cloud. All seven found no heavy-tail advantage, and the most
recent found a large disadvantage (+0.348 decades of ABER at p = 5.9e-67 at the tuned
warm-start radius). None of them tested the architecture the paper actually describes,
which is a HYBRID: a swarm that is part exploiter and part explorer at the same time.

Those are different questions and the second does not follow from the first. In a PSO the
social term makes an explorer cheap in one direction and valuable in the other: a particle
thrown far that finds nothing never becomes gbest, never moves the swarm, and costs one
evaluation; a particle thrown far that finds something better pulls the entire swarm to
it. The payoff is asymmetric in the explorer's favour, which is exactly the condition
under which a small exploration allocation dominates BOTH pure exploitation and pure
exploration -- and it is why "levy loses at 100%" is logically compatible with "levy pays
at 10%".

So this sweeps the allocation rather than the side. `init_levy_fraction` is the fraction
of the swarm drawn from the unbounded cloud; the rest keeps the deployed bounded chaotic
one. At fraction 0 the draw is bit-identical to the deployed initialiser for every law,
so fraction 0 is a single shared control and not a fourth arm.

WHAT WOULD COUNT AS A RESULT, DECIDED BEFORE THE DATA

  supports the hybrid   ABER is minimised at an INTERIOR fraction, 0 < f* < 1, and the
                        improvement over f = 0 is significant on paired cycles. The tail
                        then earns its place as an allocation even though it loses as a
                        strategy, and the title's Levy is real.
  refutes it            ABER is monotone increasing in f. The tail costs what it is given
                        and there is no allocation at which it pays.
  half-result           an interior optimum exists for `gauss` but not for `levy`: the
                        gain is unboundedness, not tail weight, and the honest name for
                        the component is not Levy.

The Gaussian arm is carried at every fraction for exactly that third reading. Both laws
share the anchor, the radius, the traces, the budget and the deployed guard; only the
step distribution differs.

Cycles are additionally split by the size of the pointing excursion they had to absorb,
since an explorer can only pay on a cycle where the exploiter's anchor has gone stale.

Usage: python levy_mixture_probe.py [--cycles 400]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_all as _ma
_ma.RANK_STAGES = None
from measure_all import GBAR_OP_DB, system_success
from mpc_loop import BeamSteeringMPC, U_SLEW
from mpc_fast import install
from channel import SwayProcess, GammaGammaAR1
from scipy.stats import wilcoxon

STRONG = (1.2, 1.1)
T_ITER = 25
CELLS = ((0.05, 0.00125), (0.05, 0.02), (0.10, 0.00125), (0.10, 0.02))
FRACS = (0.10, 0.25, 0.50, 1.00)
LAWS = ("levy", "gauss")


def run_arm(sigma_s, thetas, hs, law, spread, frac, seed):
    """One sequential warm-started closed loop; returns per-cycle system ABER."""
    gbar = 10.0 ** (GBAR_OP_DB / 10.0)
    m = BeamSteeringMPC(*STRONG, sigma_s, gbar, horizon=20, seed=seed, rank_stages=None)
    install(m)
    m.tau_o = None
    m.cfg.max_iters = T_ITER
    m.cfg.init_law = law
    m.cfg.init_spread = spread
    m.cfg.init_levy_fraction = frac
    m.u_prev = np.zeros(2)
    T = m.horizon
    anchor, out = None, []
    for th, h in zip(thetas, hs):
        m.cfg.init_centre = anchor
        r = m.step(th, h)
        if r.best_x is not None:
            anchor = r.best_x.copy()
            m.u_prev = np.array([r.best_x[T], r.best_x[2 * T]])
            w = float(r.best_x[0])
        else:
            w = None
        out.append(system_success(w, sigma_s, m.L * float(np.linalg.norm(th)))[1])
    return np.array(out)


def paired(a, b):
    """Paired log10 comparison; negative favours `a`."""
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    d = np.log10(a[ok]) - np.log10(b[ok])
    nz = d[d != 0]
    p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size >= 6 else float("nan")
    return dict(n_paired=int(ok.sum()), median_log10_delta=float(np.median(d)) if d.size else float("nan"),
                wilcoxon_p=p, better=int(np.sum(d < 0)), worse=int(np.sum(d > 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    a = ap.parse_args()

    print("  allocation sweep: fraction of the swarm drawn from the unbounded cloud,")
    print("  remainder on the deployed bounded chaotic one. f=0 is the shared control.")
    print("  %d cycles per arm, T_iter=%d, U_slew=%.2e, deployed guard and ladder\n"
          % (a.cycles, T_ITER, U_SLEW))

    results, t0 = {}, time.time()
    for sigma_s, spread in CELLS:
        sway = SwayProcess(sigma_s, seed=1000 + a.seed)
        for _ in range(500):
            sway.step()
        thetas = np.array([sway.step() for _ in range(a.cycles)])
        ch = GammaGammaAR1(*STRONG, rho_a=0.98, seed=2000 + a.seed, calibrate=False)
        hs = np.array([ch.step() for _ in range(a.cycles)])
        exc = np.r_[0.0, np.linalg.norm(np.diff(thetas, axis=0), axis=1)]
        big = exc >= np.median(exc)

        m0 = BeamSteeringMPC(*STRONG, sigma_s, 10.0 ** (GBAR_OP_DB / 10.0),
                             horizon=20, seed=0, rank_stages=None)
        b0 = m0.blocks()[1][0]
        rho = spread * float(np.asarray(m0.upper())[b0] - np.asarray(m0.lower())[b0]) \
            / float(m0.block_slew()[1])

        ctrl = run_arm(sigma_s, thetas, hs, "levy", spread, 0.0, a.seed)
        print("  sigma_s=%.2f  spread=%-9.6g  rho=%.2f   control f=0: ABER %.4e"
              % (sigma_s, spread, rho, np.nanmedian(ctrl)), flush=True)
        print("    %-6s %5s %12s %12s %10s %10s %9s %11s"
              % ("law", "f", "ABER", "vs f=0", "Wilcox p", "better", "worse", "big-exc p"))
        cell = dict(sigma_s=sigma_s, init_spread=spread, rho=rho,
                    control_median_aber=float(np.nanmedian(ctrl)), arms={})
        for law in LAWS:
            for f in FRACS:
                v = run_arm(sigma_s, thetas, hs, law, spread, f, a.seed)
                st = paired(v, ctrl)
                stb = paired(v[big], ctrl[big])
                cell["arms"]["%s_f%.2f" % (law, f)] = dict(
                    law=law, fraction=f, median_aber=float(np.nanmedian(v)),
                    vs_control=st, vs_control_big_excursion=stb,
                    no_command_rate=float(np.mean(~np.isfinite(v))))
                print("    %-6s %5.2f %12.4e %+12.4f %10.3g %10d %9d %11.3g"
                      % (law, f, np.nanmedian(v), st["median_log10_delta"],
                         st["wilcoxon_p"], st["better"], st["worse"], stb["wilcoxon_p"]),
                      flush=True)
        results["s%.2f_sp%.6g" % (sigma_s, spread)] = cell

        # the decisive shape: is the optimum interior?
        for law in LAWS:
            curve = [(0.0, cell["control_median_aber"])] + \
                    [(f, cell["arms"]["%s_f%.2f" % (law, f)]["median_aber"]) for f in FRACS]
            fstar, best = min(curve, key=lambda t: t[1] if t[1] == t[1] else np.inf)
            shape = ("INTERIOR optimum" if 0.0 < fstar < 1.0 else
                     "no allocation helps" if fstar == 0.0 else "pure arm best")
            print("    -> %-5s best at f=%.2f (ABER %.4e): %s"
                  % (law, fstar, best, shape), flush=True)
        print(flush=True)

    print("  %d cells in %.0fs" % (len(results), time.time() - t0))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "levy_mixture_probe.json"), "w") as fh:
        json.dump(dict(what="allocation sweep for the hybrid initialiser: fraction of the swarm "
                            "drawn from an unbounded (Levy or Gaussian) cloud around the warm-start "
                            "anchor against the deployed bounded chaotic cloud, at the tuned and the "
                            "wide radius; f=0 is bit-identical to the deployed initialiser and is the "
                            "shared control; scored by system_metric.system_success on paired traces, "
                            "with a large-pointing-excursion stratum reported separately",
                       cycles=a.cycles, t_iter=T_ITER, u_slew=U_SLEW,
                       fractions=[0.0] + list(FRACS), laws=list(LAWS),
                       decision_rule="interior optimum in f supports the hybrid; monotone increase "
                                     "refutes it; an interior optimum for gauss but not levy means "
                                     "the gain is unboundedness rather than tail weight",
                       results=results), fh, indent=1)
    print("  wrote levy_mixture_probe.json")


if __name__ == "__main__":
    main()
