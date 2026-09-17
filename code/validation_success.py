"""Optimization-success and ablation campaign on the compiled search kernel.

Reproduces the trial protocol of measure_all.py's part_ablation / part_baselines
exactly (same SIGMAS sweep, same per-cell count, same shuffle seed 20260827,
same per-trial seed 700000 + int(s*1000)*1000 + k, same SwayProcess warm-up
giving r_d), delegates only the SEARCH to validation_success_kernel.exe, and
scores every returned beam with system_metric.system_success -- the paper's own
certified post-EGC evaluator -- so the success criterion is unchanged.

The point of the campaign: on the interpreted solver the tau_O = 600 us
checkpoint fires after a median of ONE iteration, so three ablation arms were
bit-identical to the full kernel and every population method tied random
sampling (Tables alg_summary / ablation). The compiled search completes ~15
iterations inside the same budget. This measures what that buys.

Usage: python validation_success.py --exe <validation_success_kernel.exe> [--trials 200] [--tau-us 600] [--out DIR]
       (build the exe with: zig cc -O2 -o validation_success_kernel.exe validation_success_kernel.c -lm)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure_all import SIGMAS, GBAR_OP_DB, TARGET, system_success, clopper_pearson, mcnemar_exact
from channel import SwayProcess

ARMS = ["full", "no_chaos", "no_levy", "no_ga", "no_ladder", "random", "pso"]
LABEL = dict(full="full kernel", no_chaos="no chaotic init", no_levy="no Levy flight",
             no_ga="no GA refinement", no_ladder="fixed fidelity K=10",
             random="random sampling", pso="PSO")


def wilcoxon_p(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; nz = d[d != 0.0]
    if nz.size == 0:
        return float("nan"), 0
    from scipy.stats import wilcoxon
    return float(wilcoxon(nz, alternative="two-sided").pvalue), int(nz.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--tau-us", type=float, default=600.0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    ap.add_argument("--exe", default=os.path.join(HERE, "validation_success_kernel.exe"))
    a = ap.parse_args()

    per_cell = max(1, a.trials // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]
    np.random.default_rng(20260827).shuffle(order)

    trials = []
    for s, k in order:
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        trials.append((s, float(sway.radial()), seed))

    os.makedirs(a.out, exist_ok=True)
    tfile = os.path.join(a.out, "validation_trials.txt")
    with open(tfile, "w") as f:
        for s, rd, seed in trials:
            f.write("%.6g %.17g %d\n" % (s, rd, seed))

    ind, aber, iters = {}, {}, {}
    t0 = time.time()
    for arm in ARMS:
        ofile = os.path.join(a.out, "validation_%s_tau%.0f.csv" % (arm, a.tau_us))
        subprocess.run([a.exe, tfile, arm, ofile, str(a.tau_us)], check=True)
        rows = np.genfromtxt(ofile, delimiter=",", names=True)
        ok_l, ab_l, it_l = [], [], []
        for (s, rd, seed), w, it in zip(trials, rows["w_best"], rows["iterations"]):
            wv = None if not np.isfinite(w) else float(w)
            ok, v = system_success(wv, s, rd)
            ok_l.append(bool(ok)); ab_l.append(float(v)); it_l.append(int(it))
        ind[arm] = np.array(ok_l, bool); aber[arm] = np.array(ab_l, float); iters[arm] = np.array(it_l)
        print("  %-20s k/n = %3d/%d  median iters = %4.1f   (%.0fs)"
              % (LABEL[arm], ind[arm].sum(), ind[arm].size, np.median(iters[arm]), time.time() - t0), flush=True)

    cells = np.array([s for s, _, _ in trials])
    full = ind["full"]; fa = aber["full"]
    res = {}
    print("\n  %-20s %8s %-14s %7s  %4s %4s %9s   %11s %9s" %
          ("arm", "k/n", "CP95 (%)", "med it", "b", "c", "McNemar p", "med dlog10", "Wilcox p"))
    print("  " + "-" * 100)
    for arm in ARMS:
        av = ind[arm]; kk, nn = int(av.sum()), av.size
        lo_, hi_ = clopper_pearson(kk, nn)
        e = dict(k=kk, n=nn, rate=kk / nn, ci95=[lo_, hi_], median_iterations=float(np.median(iters[arm])),
                 per_sigma={str(s): int(av[cells == s].sum()) for s in SIGMAS})
        if arm != "full":
            b = int(np.sum(full & ~av)); c = int(np.sum(~full & av))
            v = aber[arm]; pair = np.isfinite(v) & np.isfinite(fa) & (v > 0) & (fa > 0)
            d = np.log10(v[pair]) - np.log10(fa[pair])       # > 0: arm worse than full
            wp, nnz = wilcoxon_p(d)
            e.update(b=b, c=c, p_mcnemar=mcnemar_exact(b, c),
                     median_log10_delta_vs_full=float(np.median(d)) if d.size else float("nan"),
                     arm_better=int(np.sum(d < 0)), arm_worse=int(np.sum(d > 0)), tie=int(np.sum(d == 0)),
                     wilcoxon_p=wp, wilcoxon_n=nnz)
            print("  %-20s %3d/%-4d [%4.1f,%4.1f]    %5.1f  %4d %4d %9.4f   %+11.4f %9.4g"
                  % (LABEL[arm], kk, nn, 100 * lo_, 100 * hi_, e["median_iterations"], b, c, e["p_mcnemar"],
                     e["median_log10_delta_vs_full"], wp))
        else:
            print("  %-20s %3d/%-4d [%4.1f,%4.1f]    %5.1f" % (LABEL[arm], kk, nn, 100 * lo_, 100 * hi_, e["median_iterations"]))
        res[arm] = e

    out = dict(what="optimization success and component ablation on the COMPILED search kernel "
                    "(validation_success_kernel.c), scored by system_metric.system_success; same trial "
                    "protocol, seeds and paired draws as measure_all.py part_ablation/part_baselines",
               criterion="post-EGC system ABER <= %g at %.0f dB, strong turbulence, all four sigma_s" % (TARGET, GBAR_OP_DB),
               tau_o_us=a.tau_us, rank_stages=1, trials_per_cell=per_cell, n_trials=len(trials),
               reference_interpreted=dict(full="50/200", pso="47/200", random="50/200",
                                          note="Tables alg_summary/ablation: every population arm 50/200 at both budgets, "
                                               "median 1 iteration at tau_O=600us"),
               results=res)
    jfile = os.path.join(a.out, "validation_success_tau%.0f.json" % a.tau_us)
    with open(jfile, "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(a.out, "validation_success_tau%.0f.npz" % a.tau_us),
                        sigma_s=cells, **{"%s_ok" % k: ind[k] for k in ARMS}, **{"%s_aber" % k: aber[k] for k in ARMS},
                        **{"%s_iters" % k: iters[k] for k in ARMS})
    print("\n  wrote", jfile)


if __name__ == "__main__":
    main()
