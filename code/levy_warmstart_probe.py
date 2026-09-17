"""When does a heavy-tailed warm start earn its place? A sweep across the one
ratio that decides it.

WHAT THE REACH CHECK ESTABLISHED

`warmstart_reach_check.py` measures, per block, the radius of the deployed
warm-start cloud against the radius of the set the actuator may reach in one
cycle. On the two steering axes -- the axes that carry the disturbance and where
the coupled-trajectory landscape is multimodal -- the deployed cloud is EIGHT
TIMES wider than the whole reachable set. A heavy tail there has nowhere to go:
whatever the draw, the slew projection of eq. (14) returns it inside a set the
bounded cloud already covered. On the divergence axis the ordering reverses
(cloud/reach = 0.17 to 1.05), so there a tail CAN reach further than the bounded
cloud, and the paper's usual answer -- that w_z is unimodal per stage, so there is
no second basin to escape to -- does not dispose of it. Unimodality says where the
optimum is, not whether a cloud of radius `spread` around a one-cycle-old anchor
can get to it. A bounded cloud on a unimodal axis still fails whenever the optimum
has moved further than `spread` since the anchor was computed. Reach, not modality,
is the binding question for an initialiser, and the two have been conflated
throughout this project's earlier probes.

That is a structural account of every null this paper has recorded, and it is
stated as a ratio of two design constants rather than as a p-value. It also
makes a prediction: the operator is inert BECAUSE

    rho = (init_spread * span) / (U_slew)        [ = 8 as deployed ]

is large, and a heavy tail should begin to pay once rho falls below one, when
the bounded cloud no longer covers the reachable set and the tail's projected
landing point does. rho is not a property of the plant -- init_spread is a
SOLVER parameter, free to choose -- so if the prediction holds, the paper gains
a deployable rule rather than a hardware wish.

This script tests that prediction. Four spreads spanning rho = 8 down to 0.125,
three initialisation laws at each, paired cycle by cycle on identical sway and
scintillation traces:

    chaotic   the deployed bounded cloud (logistic map on [-1,1] x spread)
    gauss     unbounded but light-tailed, matched scale -- isolates tail weight
              from mere unboundedness
    levy      Mantegna step, index 1.5, matched scale

All three run the deployed cycle `BeamSteeringMPC.step`, so the envelope guard,
the fidelity ladder, the inter-cycle anchor and the slew projection are the
released ones; only `cfg.init_law` and `cfg.init_spread` differ. Cycles are
scored on achieved post-EGC system ABER, and additionally split by the size of
the pointing excursion that cycle had to absorb, since the bounded cloud can
only fail when the optimum has moved outside it.

Budget is deterministic (T_iter = 25, no wall-clock cutoff) so that the
comparison is of search behaviour and not of this machine's timing noise.

Usage: python levy_warmstart_probe.py [--cycles 200]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_all as _ma
_ma.RANK_STAGES = None
from measure_all import SIGMAS, GBAR_OP_DB, system_success
from mpc_loop import BeamSteeringMPC, U_SLEW
from mpc_fast import install
from channel import SwayProcess, GammaGammaAR1
from scipy.stats import wilcoxon

STRONG = (1.2, 1.1)
T_ITER = 25
LAWS = ("chaotic", "gauss", "levy")
SPREADS = (0.02, 0.005, 0.00125, 0.0003125)      # rho = 8, 2, 0.5, 0.125
# sigma_s = 0.20 and 0.30 m are excluded, not omitted: at gamma_bar_op the
# envelope guard refuses EVERY candidate there, so the loop publishes no command
# on any cycle under any initialisation law and the comparison is empty. That is
# the operating-point finding of Table `tab:operating_point` showing up in the
# closed loop, not a failure of this probe; it is recorded in the output file.
SIGMAS_RUN = (0.05, 0.10)


def run_arm(sigma_s, thetas, hs, law, spread, seed):
    """One sequential warm-started closed loop. Returns per-cycle system ABER."""
    gbar = 10.0 ** (GBAR_OP_DB / 10.0)
    m = BeamSteeringMPC(*STRONG, sigma_s, gbar, horizon=20, seed=seed, rank_stages=None)
    install(m)
    m.tau_o = None                       # deterministic budget, not a wall clock
    m.cfg.max_iters = T_ITER
    m.cfg.init_law = law
    m.cfg.init_spread = spread
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


def compare(a, b):
    """Paired log10 comparison of two ABER series; negative favours `a`."""
    pair = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    d = np.log10(a[pair]) - np.log10(b[pair])
    nz = d[d != 0]
    p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size >= 6 else float("nan")
    return dict(n_paired=int(pair.sum()), median_log10_delta=float(np.median(d)) if d.size else float("nan"),
                wilcoxon_p=p, a_better=int(np.sum(d < 0)), a_worse=int(np.sum(d > 0)),
                tie=int(np.sum(d == 0))), pair, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    a = ap.parse_args()

    print("  deployed U_slew = %.2e rad/cycle; sweeping the warm-start cloud from rho = 8 to 0.125"
          % U_SLEW)
    print("  %d cycles per arm, T_iter = %d, deterministic budget, deployed guard and ladder\n"
          % (a.cycles, T_ITER))

    results, t0 = {}, time.time()
    print("  %-7s %-10s %6s %11s %11s %10s %10s %9s %13s" %
          ("sigma_s", "spread", "rho", "ABER levy", "ABER chaotic", "med dlog10",
           "Wilcox p", "L better", "no-cmd L/C"))
    print("  " + "-" * 108)

    for sigma_s in SIGMAS_RUN:
        sway = SwayProcess(sigma_s, seed=1000 + a.seed)
        for _ in range(500):
            sway.step()
        thetas = np.array([sway.step() for _ in range(a.cycles)])
        ch = GammaGammaAR1(*STRONG, rho_a=0.98, seed=2000 + a.seed, calibrate=False)
        hs = np.array([ch.step() for _ in range(a.cycles)])
        # how far the disturbance moved this cycle: the bounded cloud can only
        # fail on cycles where the optimum left it
        excursion = np.r_[0.0, np.linalg.norm(np.diff(thetas, axis=0), axis=1)]
        big = excursion >= np.median(excursion)

        for spread in SPREADS:
            m0 = BeamSteeringMPC(*STRONG, sigma_s, 10.0 ** (GBAR_OP_DB / 10.0),
                                 horizon=20, seed=0, rank_stages=None)
            span_st = float(np.asarray(m0.upper())[m0.blocks()[1][0]]
                            - np.asarray(m0.lower())[m0.blocks()[1][0]])
            rho = spread * span_st / float(m0.block_slew()[1])
            series = {law: run_arm(sigma_s, thetas, hs, law, spread, a.seed) for law in LAWS}
            key = "s%.2f_sp%.6g" % (sigma_s, spread)
            ent = dict(sigma_s=sigma_s, init_spread=spread, rho=rho,
                       median_aber={k: float(np.nanmedian(v)) for k, v in series.items()},
                       # a cycle with no admissible command is the failure the
                       # heavy tail is supposed to prevent, so it is scored
                       # separately rather than dropped by the paired test
                       no_command_rate={k: float(np.mean(~np.isfinite(v)))
                                        for k, v in series.items()})
            st, pair, d = compare(series["levy"], series["chaotic"])
            ent["levy_vs_chaotic"] = st
            ent["gauss_vs_chaotic"] = compare(series["gauss"], series["chaotic"])[0]
            ent["levy_vs_gauss"] = compare(series["levy"], series["gauss"])[0]
            ent["levy_vs_chaotic_big_excursion"] = compare(series["levy"][big],
                                                           series["chaotic"][big])[0]
            results[key] = ent
            print("  %-7.2f %-10.6g %6.2f %11.3e %11.3e %+10.4f %10.4g %4d/%-4d %6.1f%%/%-5.1f%%"
                  % (sigma_s, spread, rho, ent["median_aber"]["levy"],
                     ent["median_aber"]["chaotic"], st["median_log10_delta"],
                     st["wilcoxon_p"], st["a_better"], st["a_worse"],
                     100 * ent["no_command_rate"]["levy"],
                     100 * ent["no_command_rate"]["chaotic"]), flush=True)

    print("\n  %d configurations in %.0fs" % (len(results), time.time() - t0))
    fav = {k: v for k, v in results.items()
           if np.isfinite(v["levy_vs_chaotic"]["wilcoxon_p"])
           and v["levy_vs_chaotic"]["wilcoxon_p"] < 0.05
           and v["levy_vs_chaotic"]["median_log10_delta"] < 0}
    adv = {k: v for k, v in results.items()
           if np.isfinite(v["levy_vs_chaotic"]["wilcoxon_p"])
           and v["levy_vs_chaotic"]["wilcoxon_p"] < 0.05
           and v["levy_vs_chaotic"]["median_log10_delta"] > 0}
    print("  Levy-favourable at p<0.05: %d   Levy-unfavourable: %d" % (len(fav), len(adv)))
    for k, v in sorted(fav.items(), key=lambda kv: kv[1]["levy_vs_chaotic"]["wilcoxon_p"]):
        s = v["levy_vs_chaotic"]
        print("    FAVOURABLE %-20s rho=%.3f  p=%.3g  dlog10=%+.4f  better %d/%d"
              % (k, v["rho"], s["wilcoxon_p"], s["median_log10_delta"], s["a_better"], s["n_paired"]))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "levy_warmstart_probe.json"), "w") as fh:
        json.dump(dict(what="initialisation-law sweep for the warm-started receding-horizon loop: "
                            "heavy-tailed vs Gaussian vs the deployed bounded chaotic cloud, across "
                            "cloud radii spanning rho = (cloud radius)/(one-cycle reach) from 8 to "
                            "0.125; deployed guard, ladder and slew projection; scored by "
                            "system_metric.system_success on paired traces",
                       cycles=a.cycles, t_iter=T_ITER, u_slew=U_SLEW,
                       spreads=list(SPREADS), laws=list(LAWS),
                       sigmas_run=list(SIGMAS_RUN),
                       sigmas_excluded=[s for s in SIGMAS if s not in SIGMAS_RUN],
                       exclusion_reason="at gamma_bar_op the envelope guard refuses every "
                                        "candidate at these jitter levels, so no arm publishes "
                                        "a command on any cycle and the paired comparison is "
                                        "empty; this is the operating-point result, not a "
                                        "property of the initialisation law",
                       results=results), fh, indent=1)
    print("  wrote levy_warmstart_probe.json")


if __name__ == "__main__":
    main()
