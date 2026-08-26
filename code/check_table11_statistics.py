"""
Check the STATISTICS of Table 11 (ablation study).  Not a reproduction of it.

WHY THE NAME CHANGED
    This file was called `reproduce_table11.py` and was listed under "Scripts
    that reproduce published tables".  It does not reproduce Table 11 and cannot:
    Table 11's content is four pairs of discordant counts (b, c) measured by the
    closed-loop campaign, and that campaign is not part of this release.  What
    the script does is take paired success indicators and compute the exact
    McNemar p-values and Clopper-Pearson intervals from them.

    Run on the campaign's own indicators it would be a reproduction.  Run on the
    PUBLISHED (b, c) -- which is what it used to do whenever no indicator file
    was present -- it consumes the published numbers as its input and verifies
    that arithmetic on them closes.  That is a transcription check.  It can
    catch a typo in a p-value.  It cannot say anything whatever about whether
    the counts are right, which is the only thing a reader wants to know.

WHAT THIS SCRIPT ESTABLISHES

  IT DOES   recompute, from an `ablation_success.npz` of per-realization paired
            success indicators, the discordant counts (b, c), the exact
            two-sided McNemar p-values, and the exact Clopper-Pearson
            success-rate intervals.
  IT DOES NOT re-run the optimizer, and does not regenerate the published
            counts.  Fed the re-implementation's indicators it produces the
            re-implementation's statistics, which are its own numbers.

WHERE THE INDICATOR FILE COMES FROM
    No measured indicator file ships with this package.  `ablation_bc.py`
    produces one from the re-implemented solver:

        python ablation_bc.py --realizations 1000 --out ablation_success.npz
        python check_table11_statistics.py ablation_success.npz

    Those indicators are the re-implementation's own and will NOT reproduce the
    published (b, c).

    A MISSING FILE IS NOW A HARD ERROR.  It used to fall back, silently and by
    default, to the published (b, c).  Silently substituting the answer for the
    input is the worst failure mode a reproduction script has: the run prints a
    full, well-formatted, entirely self-consistent table, and a reader skimming a
    long log sees a reproduction.  The transcription check is still available,
    but only by asking for it in as many words:

        python check_table11_statistics.py --published-arithmetic-only

Usage:  python check_table11_statistics.py path/to/ablation_success.npz
        python check_table11_statistics.py --published-arithmetic-only
"""
import os
import sys

import numpy as np
from mpmath import mp, binomial, power, nstr

mp.dps = 200

N = 1000
VARIANTS = ["no_chaotic_init", "no_levy_flight", "no_ga_refinement", "fixed_fidelity"]
# `ablation_bc.py` names its arms after the component removed, the table names
# them after the ablation. Both spellings are accepted: before this alias table
# existed, a file written by ablation_bc.py made this script print a one-line
# "array not in file" note and then silently drop three of the four rows, which
# in a long log reads exactly like a table that reproduced.
ALIASES = {"no_chaotic_init": ("no_chaotic_init", "no_chaos"),
           "no_levy_flight": ("no_levy_flight", "no_levy"),
           "no_ga_refinement": ("no_ga_refinement", "no_ga"),
           "fixed_fidelity": ("fixed_fidelity", "fixed_K", "no_ladder")}
LABEL = {"no_chaotic_init": "w/o Chaotic Initialization",
         "no_levy_flight": "w/o Levy Flight Jump",
         "no_ga_refinement": "w/o GA Refinement",
         "fixed_fidelity": "Fixed-Fidelity (K=10)"}
PUBLISHED_BC = {"no_chaotic_init": (104, 4), "no_levy_flight": (382, 2),
                "no_ga_refinement": (64, 4), "fixed_fidelity": (132, 2)}
PUBLISHED_P = {"no_chaotic_init": 3.4e-26, "no_levy_flight": 3.8e-111,
               "no_ga_refinement": 5.9e-15, "fixed_fidelity": 8.3e-37}
PUBLISHED_IMPROVEMENT = {"no_chaotic_init": 10.0, "no_levy_flight": 38.0,
                         "no_ga_refinement": 6.0, "fixed_fidelity": 13.0}
PUBLISHED_FULL_RATE = 0.980

_USAGE = """
  usage:  python check_table11_statistics.py path/to/ablation_success.npz
          python check_table11_statistics.py --published-arithmetic-only
"""


def mcnemar_two_sided(b, c):
    """p = 2 Pr{B >= b},  B ~ Bin(b+c, 1/2), clipped at 1."""
    n = b + c
    if n == 0:
        return mp.mpf(1)
    tail = sum(binomial(n, k) for k in range(b, n + 1)) / power(2, n)
    return min(mp.mpf(1), 2 * tail)


def clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def from_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())
    print("  loaded %s with arrays: %s" % (os.path.basename(path), sorted(keys)))
    if "full" not in keys:
        raise SystemExit("  ERROR: %s has no 'full' array; nothing to pair against."
                         % path)
    full = np.asarray(d["full"]).astype(bool)
    out, missing = {}, []
    for v in VARIANTS:
        hit = next((a for a in ALIASES[v] if a in keys), None)
        if hit is None:
            missing.append("%s (tried %s)" % (v, "/".join(ALIASES[v])))
            continue
        arr = np.asarray(d[hit]).astype(bool)
        if arr.shape != full.shape:
            raise SystemExit("  ERROR: '%s' has shape %s but 'full' has %s."
                             % (hit, arr.shape, full.shape))
        out[v] = (int(np.sum(full & ~arr)), int(np.sum(~full & arr)), arr.mean())
    if missing:
        # Hard failure, not a note: a partially populated table is worse than
        # no table, because it still looks like a reproduction.
        raise SystemExit("  ERROR: %s is missing %d of the %d ablation arms:\n    %s"
                         % (path, len(missing), len(VARIANTS), "\n    ".join(missing)))
    return out, full.mean(), int(full.size)


def _die_no_input(path):
    raise SystemExit(
        "\n" + "=" * 78 +
        "\n  ERROR: no paired-indicator file.\n" + "=" * 78 +
        "\n  looked for: %s\n"
        "\n  This script computes McNemar p-values and Clopper-Pearson intervals"
        "\n  FROM per-realization paired success indicators. Without them there is"
        "\n  nothing to compute them from."
        "\n"
        "\n  It does NOT fall back to the published (b, c). It used to, silently,"
        "\n  and then printed a complete table -- which reads, in a log, exactly"
        "\n  like a reproduction, while in fact having consumed the published"
        "\n  answer as its input."
        "\n"
        "\n  To produce indicators from the re-implemented solver (they will NOT"
        "\n  reproduce the published counts, and are not meant to):"
        "\n"
        "\n      python ablation_bc.py --realizations 1000 --out ablation_success.npz"
        "\n      python check_table11_statistics.py ablation_success.npz"
        "\n"
        "\n  To check only that the tabulated p-values follow from the tabulated"
        "\n  counts -- a transcription check on the printed table, which says"
        "\n  nothing about whether the counts are right -- ask for it explicitly:"
        "\n"
        "\n      python check_table11_statistics.py --published-arithmetic-only"
        "\n" % path)


def _banner_published():
    print("=" * 78)
    print("  TRANSCRIPTION CHECK ONLY -- THIS IS NOT A REPRODUCTION")
    print("=" * 78)
    print("  Input:  the PUBLISHED (b, c) of Table 11, typed into this file.")
    print("  Output: the p-values and intervals those counts imply.")
    print()
    print("  This closes a loop from the published table back onto itself. It")
    print("  can catch a typo in a tabulated p-value. It CANNOT say anything")
    print("  about whether the counts are right -- they are the input, not a")
    print("  result -- and it does not touch the optimizer, the channel, or any")
    print("  measurement. Nothing below is evidence for Table 11.")
    print("=" * 78)


def main(argv):
    args = [a for a in argv[1:]]
    published_only = "--published-arithmetic-only" in args
    if published_only:
        args.remove("--published-arithmetic-only")
    if "-h" in args or "--help" in args:
        print(__doc__)
        print(_USAGE)
        return 0
    if len(args) > 1:
        raise SystemExit("  ERROR: expected at most one path.\n" + _USAGE)

    if published_only:
        if args:
            raise SystemExit("  ERROR: --published-arithmetic-only takes no path; "
                             "it reads the constants in this file.\n" + _USAGE)
        _banner_published()
        data = {v: (b, c, None) for v, (b, c) in PUBLISHED_BC.items()}
        full_rate, n_obs = PUBLISHED_FULL_RATE, N
        source = "published (b, c) -- TRANSCRIPTION CHECK, not a measurement"
    else:
        path = args[0] if args else "ablation_success.npz"
        if not os.path.exists(path):
            _die_no_input(os.path.abspath(path))
        data, full_rate, n_obs = from_npz(path)
        source = "measured from " + os.path.basename(path)

    print("\n  source: %s;  n = %d;  full-kernel success rate: %.3f\n"
          % (source, n_obs, full_rate))
    print("  %-28s %5s %4s %10s %12s %12s"
          % ("variant", "b", "c", "(b-c)/n", "p (exact)", "p (Table 11)"))
    print("  " + "-" * 78)
    rows = []
    for v in VARIANTS:
        if v not in data:
            continue
        b, c, rate = data[v]
        p = mcnemar_two_sided(b, c)
        imp = 100.0 * (b - c) / n_obs
        rows.append((v, b, c, rate, imp))
        print("  %-28s %5d %4d %9.3f%% %12s %12.1e"
              % (LABEL[v], b, c, imp, nstr(p, 3), PUBLISHED_P[v]))

    # --- exact success-rate intervals ------------------------------------
    print("\n  Exact (Clopper-Pearson) 95% success-rate intervals:")
    print("  %-28s %10s %22s" % ("variant", "rate", "95% CI"))
    print("  " + "-" * 62)
    k_full = int(round(full_rate * n_obs))
    lo, hi = clopper_pearson(k_full, n_obs)
    print("  %-28s %9.3f  [%.4f, %.4f]" % ("full kernel", full_rate, lo, hi))
    for v, b, c, rate, _ in rows:
        if rate is None:
            # the transcription path has counts but no per-realization
            # indicators, so the variant's own success count is recoverable from
            # the pairing identity  k_variant = k_full - b + c
            k = k_full - b + c
            note = "  (from k_full - b + c)"
        else:
            k, note = int(round(rate * n_obs)), ""
        lo, hi = clopper_pearson(k, n_obs)
        print("  %-28s %9.3f  [%.4f, %.4f]%s"
              % (LABEL[v], k / n_obs, lo, hi, note))

    # --- the consistency check, actually performed ------------------------
    # Previously this section printed the sentence "(b-c)/n must equal the
    # tabulated improvement on every row" and then did not check it.
    print("\n  Consistency check: (b-c)/n must equal the tabulated improvement.")
    print("  %-28s %12s %12s %s" % ("variant", "(b-c)/n", "Table 11", "verdict"))
    print("  " + "-" * 66)
    n_bad = 0
    for v, b, c, rate, imp in rows:
        want = PUBLISHED_IMPROVEMENT[v]
        # counts are integers out of n, so the derived percentage is exact to
        # 0.1 points; anything larger is a real disagreement, not rounding
        ok = abs(imp - want) <= 0.05
        n_bad += 0 if ok else 1
        print("  %-28s %11.3f%% %11.1f%% %s"
              % (LABEL[v], imp, want, "OK" if ok else "MISMATCH"))
    print("\n  %d of %d rows disagree with Table 11." % (n_bad, len(rows)))
    if published_only:
        print("\n  Reminder: every number above descends from the four (b, c)")
        print("  pairs typed into this file. Agreement here is transcription,")
        print("  not reproduction.")
    elif n_bad:
        print("  Expected: these indicators are the re-implementation's, not the")
        print("  campaign's. The check is here to confirm the arithmetic path,")
        print("  not to claim the counts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
