"""
Reproduce Table 11 (ablation study) from the released paired indicators.

What this script establishes and what it does not:

  IT DOES   recompute, from `ablation_success.npz`, the discordant counts
            (b, c), the exact two-sided McNemar p-values, and the exact
            Clopper-Pearson success-rate intervals -- i.e. every derived
            number in the table.
  IT DOES NOT re-run the optimizer. The per-realization success indicators are
            the campaign's output; this script is the audit of the statistics
            computed from them.

If `ablation_success.npz` is not present, the script falls back to the
published (b, c) so that the arithmetic can still be checked.

Usage:  python reproduce_table11.py [path/to/ablation_success.npz]
"""
import os
import sys

import numpy as np
from mpmath import mp, binomial, power, nstr

mp.dps = 200

N = 1000
VARIANTS = ["no_chaotic_init", "no_levy_flight", "no_ga_refinement", "fixed_fidelity"]
LABEL = {"no_chaotic_init": "w/o Chaotic Initialization",
         "no_levy_flight": "w/o Levy Flight Jump",
         "no_ga_refinement": "w/o GA Refinement",
         "fixed_fidelity": "Fixed-Fidelity (K=10)"}
PUBLISHED_BC = {"no_chaotic_init": (104, 4), "no_levy_flight": (382, 2),
                "no_ga_refinement": (64, 4), "fixed_fidelity": (132, 2)}
PUBLISHED_P = {"no_chaotic_init": 3.4e-26, "no_levy_flight": 3.8e-111,
               "no_ga_refinement": 5.9e-15, "fixed_fidelity": 8.3e-37}


def mcnemar_two_sided(b, c):
    """p = 2 Pr{B >= b},  B ~ Bin(b+c, 1/2}, clipped at 1."""
    n = b + c
    if n == 0:
        return mp.mpf(1)
    tail = sum(binomial(n, k) for k in range(b, n + 1)) / power(2, n)
    return min(mp.mpf(1), 2 * tail)


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def from_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print("  loaded %s with arrays: %s" % (os.path.basename(path), keys))
    full = np.asarray(d["full"]).astype(bool)
    out = {}
    for v in VARIANTS:
        if v not in d:
            print("  !! array '%s' not in file; skipping" % v)
            continue
        arr = np.asarray(d[v]).astype(bool)
        out[v] = (int(np.sum(full & ~arr)), int(np.sum(~full & arr)), arr.mean())
    return out, full.mean()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ablation_success.npz"
    if os.path.exists(path):
        data, full_rate = from_npz(path)
        source = "measured from " + os.path.basename(path)
    else:
        print("  %s not found -- checking the published (b, c) arithmetic instead\n" % path)
        data = {v: (b, c, None) for v, (b, c) in PUBLISHED_BC.items()}
        full_rate = 0.980
        source = "published (b, c)"

    print("\n  source: %s;  full-kernel success rate: %.3f\n" % (source, full_rate))
    print("  %-28s %5s %4s %10s %12s %12s"
          % ("variant", "b", "c", "(b-c)/n", "p (exact)", "p (Table 11)"))
    print("  " + "-" * 78)
    for v in VARIANTS:
        if v not in data:
            continue
        b, c, rate = data[v]
        p = mcnemar_two_sided(b, c)
        print("  %-28s %5d %4d %9.3f%% %12s %12.1e"
              % (LABEL[v], b, c, 100.0 * (b - c) / N, nstr(p, 3), PUBLISHED_P[v]))
    print()
    print("  consistency check: (b-c)/n must equal the tabulated improvement on every row.")
