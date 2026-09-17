"""The allocation question, asked with the trajectory lottery removed.

WHY THE PREVIOUS TWO SWEEPS CANNOT ANSWER IT

`levy_warmstart_probe.py` and `levy_mixture_probe.py` each ran ONE sequential closed loop
of 400 cycles per arm and applied a Wilcoxon test across its cycles. Cycle t warm-starts
from cycle t-1's solution, so the cycles are not independent observations of the setting;
they are 400 correlated observations of one trajectory. `trajectory_variance_check.py`
measured what that costs: holding the setting completely fixed and changing only the seed
moves the median system ABER by 2.447 decades (log10 SD 0.855 over six seeds). The largest
difference either sweep reported between allocations was 1.88 decades. The noise exceeded
the signal, so neither sweep resolved anything and both sets of p-values were inflated by
serial dependence -- including the ones that ran against the Levy operator.

Sharing the disturbance traces across arms, which both sweeps did, does not fix this. The
arms share the sway and scintillation but not the closed-loop state: a different initial
cloud gives a different first solution, hence a different anchor, and from there the two
trajectories occupy different basins and never re-converge.

THE CORRECTED DESIGN

Many SHORT independent episodes instead of one long run. Each episode draws its own sway
and scintillation traces and its own solver seed, runs C cycles from a cold start, and
contributes ONE number per arm: the median log10 system ABER over its warm-started cycles.
The unit of analysis is the episode, episodes are independent by construction, and arms
are paired within an episode so the episode's own difficulty cancels.

The null this can produce is only worth reporting if it comes with a detectable effect
size, so the script computes one: with E episodes and observed paired SD s, the smallest
true shift detectable at 80% power and alpha = 0.05 is about 2.80 * s / sqrt(E). A null
is then reported as "no effect larger than X decades", which is a measurement, rather than
as "no effect", which is not.

Usage: python levy_episodic_probe.py [--episodes 150] [--cycles 25]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_all as _ma
_ma.RANK_STAGES = None
from measure_all import GBAR_OP_DB, system_success
from mpc_loop import BeamSteeringMPC
from mpc_fast import install
from channel import SwayProcess, GammaGammaAR1
from scipy.stats import wilcoxon

STRONG = (1.2, 1.1)
T_ITER = 25
SIGMA_S, SPREAD = 0.05, 0.00125           # the tuned cell, rho = 0.5
ARMS = [("chaotic", 0.00)] + [(law, f) for law in ("levy", "gauss")
                              for f in (0.10, 0.25, 0.50)]


def episode(sigma_s, spread, law, frac, seed, cycles):
    """One independent episode. Returns median log10 ABER over its warm-started cycles."""
    gbar = 10.0 ** (GBAR_OP_DB / 10.0)
    sway = SwayProcess(sigma_s, seed=1000 + seed)
    for _ in range(500):
        sway.step()
    thetas = np.array([sway.step() for _ in range(cycles)])
    ch = GammaGammaAR1(*STRONG, rho_a=0.98, seed=2000 + seed, calibrate=False)
    hs = np.array([ch.step() for _ in range(cycles)])

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
    for i, (th, h) in enumerate(zip(thetas, hs)):
        m.cfg.init_centre = anchor
        r = m.step(th, h)
        if r.best_x is not None:
            anchor = r.best_x.copy()
            m.u_prev = np.array([r.best_x[T], r.best_x[2 * T]])
            w = float(r.best_x[0])
        else:
            w = None
        if i > 0:                     # cycle 0 is a cold start and identical in every arm
            out.append(system_success(w, sigma_s, m.L * float(np.linalg.norm(th)))[1])
    v = np.array(out)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(np.log10(v))) if v.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--cycles", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    a = ap.parse_args()
    seeds = [90000 + 137 * i for i in range(a.episodes)]

    print("  sigma_s=%.2f, spread=%.6g (rho=0.5), %d INDEPENDENT episodes of %d cycles"
          % (SIGMA_S, SPREAD, a.episodes, a.cycles))
    print("  unit of analysis is the episode; arms paired within an episode\n")

    series, t0 = {}, time.time()
    for law, f in ARMS:
        key = "%s_f%.2f" % (law, f)
        series[key] = np.array([episode(SIGMA_S, SPREAD, law, f, s, a.cycles) for s in seeds])
        ok = np.isfinite(series[key])
        print("    %-14s episode-median log10 ABER: mean %+.3f, SD %.3f, n=%d  (%.0fs)"
              % (key, np.nanmean(series[key]), np.nanstd(series[key], ddof=1), ok.sum(),
                 time.time() - t0), flush=True)

    ctrl = series["chaotic_f0.00"]
    print("\n  paired against the deployed initialiser (f=0), episode by episode")
    print("  %-14s %11s %10s %9s %9s %14s" %
          ("arm", "mean dlog10", "Wilcox p", "better", "worse", "min detectable"))
    print("  " + "-" * 76)
    res = {}
    for law, f in ARMS:
        key = "%s_f%.2f" % (law, f)
        if key == "chaotic_f0.00":
            continue
        v = series[key]
        ok = np.isfinite(v) & np.isfinite(ctrl)
        d = v[ok] - ctrl[ok]
        nz = d[d != 0]
        p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size >= 6 else float("nan")
        s = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
        mde = 2.80 * s / np.sqrt(max(d.size, 1))
        res[key] = dict(law=law, fraction=f, n_episodes=int(d.size),
                        mean_log10_delta=float(np.mean(d)), median_log10_delta=float(np.median(d)),
                        paired_sd=s, wilcoxon_p=p, min_detectable_effect_decades=float(mde),
                        better=int(np.sum(d < 0)), worse=int(np.sum(d > 0)))
        e = res[key]
        print("  %-14s %+11.4f %10.3g %9d %9d %14.4f"
              % (key, e["mean_log10_delta"], p, e["better"], e["worse"], mde))

    print("\n  --- reading ---")
    sig = {k: v for k, v in res.items() if v["wilcoxon_p"] == v["wilcoxon_p"] and v["wilcoxon_p"] < 0.05}
    fav = {k: v for k, v in sig.items() if v["mean_log10_delta"] < 0}
    if not sig:
        worst_mde = max(v["min_detectable_effect_decades"] for v in res.values())
        print("  No allocation of either law separates from the deployed initialiser.")
        print("  With %d episodes this excludes any true effect larger than about %.3f"
              % (a.episodes, worst_mde))
        print("  decades in either direction; it does not exclude smaller ones.")
    else:
        for k, v in sorted(sig.items(), key=lambda kv: kv[1]["wilcoxon_p"]):
            print("  %-14s %s by %+.4f decades, p=%.3g (detectable floor %.4f)"
                  % (k, "BETTER" if v["mean_log10_delta"] < 0 else "worse",
                     v["mean_log10_delta"], v["wilcoxon_p"], v["min_detectable_effect_decades"]))
        if fav:
            lv = [v for k, v in fav.items() if v["law"] == "levy"]
            gs = [v for k, v in fav.items() if v["law"] == "gauss"]
            print("\n  Levy allocations that help: %d;  Gaussian allocations that help: %d."
                  % (len(lv), len(gs)))
            print("  A Levy gain that a Gaussian of the same allocation also delivers is a")
            print("  gain from unboundedness, not from tail weight, and must be named as such.")

    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "levy_episodic_probe.npz"), seeds=np.array(seeds),
                        **{k: v for k, v in series.items()})
    with open(os.path.join(a.out, "levy_episodic_probe.json"), "w") as fh:
        json.dump(dict(what="allocation of the swarm between the deployed bounded chaotic warm-start "
                            "cloud and an unbounded (Levy or Gaussian) one, measured over independent "
                            "short episodes so that the unit of analysis is the episode rather than a "
                            "cycle of one long correlated trajectory; supersedes levy_warmstart_probe "
                            "and levy_mixture_probe, whose single-trajectory p-values are inflated by "
                            "serial dependence (see trajectory_variance_check.json)",
                       sigma_s=SIGMA_S, init_spread=SPREAD, episodes=a.episodes,
                       cycles_per_episode=a.cycles, t_iter=T_ITER,
                       power_note="min_detectable_effect_decades is 2.80*SD/sqrt(n), the shift "
                                  "detectable at 80% power and alpha=0.05; a null is reported "
                                  "against it rather than as an unqualified absence of effect",
                       results=res), fh, indent=1)
    print("\n  wrote levy_episodic_probe.json")


if __name__ == "__main__":
    main()
