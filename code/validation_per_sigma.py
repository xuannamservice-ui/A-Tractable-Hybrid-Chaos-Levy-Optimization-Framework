"""Per-sigma_s breakdown of the compiled-kernel ablation (Step 2, third direction):
does any component separate on the continuous metric WITHIN a jitter stratum,
in particular at sigma_s = 0.05 m where actuator authority is 17.9x the sway?"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_all import SIGMAS
from scipy.stats import wilcoxon

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "14_compiled_poc")
out = {}
for tau in (600, 20000):
    z = np.load(os.path.join(D, "validation_success_tau%d.npz" % tau))
    cells = z["sigma_s"]; full = z["full_aber"]
    print("=== tau_O = %d us ===" % tau)
    print("  %-10s %-20s %6s %10s %12s %8s %8s" % ("sigma_s", "arm", "n", "med dlog10", "Wilcoxon p", "better", "worse"))
    out[str(tau)] = {}
    for s in SIGMAS:
        m = cells == s
        out[str(tau)][str(s)] = {}
        for arm in ("no_chaos", "no_levy", "no_ga", "no_ladder", "random", "pso"):
            v = z["%s_aber" % arm][m]; f = full[m]
            pair = np.isfinite(v) & np.isfinite(f) & (v > 0) & (f > 0)
            d = np.log10(v[pair]) - np.log10(f[pair]); nz = d[d != 0]
            p = float(wilcoxon(nz, alternative="two-sided").pvalue) if nz.size >= 6 else float("nan")
            e = dict(n=int(pair.sum()), n_nonzero=int(nz.size), median_dlog10=float(np.median(d)) if d.size else float("nan"),
                     p=p, better=int(np.sum(d < 0)), worse=int(np.sum(d > 0)),
                     success_full=int(z["full_ok"][m].sum()), success_arm=int(z["%s_ok" % arm][m].sum()))
            out[str(tau)][str(s)][arm] = e
            print("  %-10s %-20s %6d %+10.4f %12.4g %8d %8d" % (s, arm, e["n"], e["median_dlog10"], p, e["better"], e["worse"]))
with open(os.path.join(D, "validation_per_sigma.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote validation_per_sigma.json")
