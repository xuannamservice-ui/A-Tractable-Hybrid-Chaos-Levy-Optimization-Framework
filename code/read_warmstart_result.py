"""Read the warm-start sweep's controlled comparison, not its headline one.

The sweep's console output prints levy against the deployed CHAOTIC COLD START, which
confounds two things at once: the warm start itself (anchor versus no anchor) and the law
of the cloud around it. Only one of those is a statement about tail weight.

This reader prints the three comparisons side by side so the confound is visible rather
than assumed away:

    levy  vs chaotic    warm start AND heavy tail, against no warm start   (confounded)
    gauss vs chaotic    warm start AND light tail, against no warm start   (the confound alone)
    levy  vs gauss      identical anchor, identical scale, tail weight ONLY (the result)

If levy-vs-chaotic and gauss-vs-chaotic move together while levy-vs-gauss stays at zero,
the effect is the warm start and nothing in it belongs to the Levy operator.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "data", "14_compiled_poc", "levy_warmstart_probe.json")


def stars(p):
    if p != p:
        return "  -- "
    return "***" if p < 1e-3 else (" **" if p < 0.01 else ("  *" if p < 0.05 else "  ns"))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PATH
    with open(path) as fh:
        doc = json.load(fh)
    print("  %s\n  cycles=%d  T_iter=%d  U_slew=%.2e  strata run: %s"
          % (doc["what"], doc["cycles"], doc["t_iter"], doc["u_slew"],
             ", ".join(str(s) for s in doc.get("sigmas_run", []))))
    if doc.get("sigmas_excluded"):
        print("  excluded %s: %s" % (doc["sigmas_excluded"], doc["exclusion_reason"]))

    print("\n  %-7s %-10s %6s | %10s %9s | %10s %9s | %10s %9s %11s"
          % ("sigma_s", "spread", "rho",
             "L-vs-chaos", "p", "G-vs-chaos", "p", "LEVY-vs-GAUSS", "p", "L better"))
    print("  " + "-" * 116)
    verdict = []
    for key, e in doc["results"].items():
        lc, gc, lg = e["levy_vs_chaotic"], e["gauss_vs_chaotic"], e["levy_vs_gauss"]
        print("  %-7.2f %-10.6g %6.2f | %+10.4f %9.2g%s | %+10.4f %9.2g%s | %+10.4f %9.2g%s %5d/%-5d"
              % (e["sigma_s"], e["init_spread"], e["rho"],
                 lc["median_log10_delta"], lc["wilcoxon_p"], stars(lc["wilcoxon_p"]),
                 gc["median_log10_delta"], gc["wilcoxon_p"], stars(gc["wilcoxon_p"]),
                 lg["median_log10_delta"], lg["wilcoxon_p"], stars(lg["wilcoxon_p"]),
                 lg["a_better"], lg["a_worse"]))
        if lg["wilcoxon_p"] == lg["wilcoxon_p"] and lg["wilcoxon_p"] < 0.05:
            verdict.append((key, e, lg))

    print("\n  --- absolute quality, so the sweep is read on outcomes and not only on contrasts ---")
    print("  %-7s %-10s %6s %12s %12s %12s %10s"
          % ("sigma_s", "spread", "rho", "ABER levy", "ABER gauss", "ABER chaotic", "no-cmd L"))
    best = None
    for key, e in doc["results"].items():
        m, n = e["median_aber"], e["no_command_rate"]
        print("  %-7.2f %-10.6g %6.2f %12.4e %12.4e %12.4e %9.1f%%"
              % (e["sigma_s"], e["init_spread"], e["rho"], m["levy"], m["gauss"], m["chaotic"],
                 100 * n["levy"]))
        for law in ("levy", "gauss", "chaotic"):
            if m[law] == m[law] and (best is None or m[law] < best[0]):
                best = (m[law], law, e["sigma_s"], e["init_spread"])
    if best:
        print("\n  best configuration measured anywhere in the sweep: %s at sigma_s=%.2f, "
              "spread=%.6g, ABER = %.4e" % (best[1], best[2], best[3], best[0]))

    print("\n  --- the controlled question ---")
    if not verdict:
        print("  Levy does NOT separate from a Gaussian cloud of the same scale in any")
        print("  configuration. Whatever the levy-vs-chaotic column shows is the warm start,")
        print("  not the tail.")
    else:
        fav = [(k, e, l) for k, e, l in verdict if l["median_log10_delta"] < 0]
        adv = [(k, e, l) for k, e, l in verdict if l["median_log10_delta"] > 0]
        print("  Levy separates from a matched Gaussian in %d of %d configurations: "
              "%d favourable, %d unfavourable." % (len(verdict), len(doc["results"]), len(fav), len(adv)))
        for k, e, l in sorted(fav, key=lambda t: t[2]["wilcoxon_p"]):
            print("    FAVOURABLE  sigma_s=%.2f spread=%-9.6g rho=%5.2f  dlog10=%+.4f  p=%.3g  %d/%d"
                  % (e["sigma_s"], e["init_spread"], e["rho"], l["median_log10_delta"],
                     l["wilcoxon_p"], l["a_better"], l["a_worse"]))
        for k, e, l in sorted(adv, key=lambda t: t[2]["wilcoxon_p"]):
            print("    AGAINST     sigma_s=%.2f spread=%-9.6g rho=%5.2f  dlog10=%+.4f  p=%.3g  %d/%d"
                  % (e["sigma_s"], e["init_spread"], e["rho"], l["median_log10_delta"],
                     l["wilcoxon_p"], l["a_better"], l["a_worse"]))
        print("\n  A favourable result is only usable if it also sits at or near the best")
        print("  ABER in the sweep: an operator that wins inside a badly-tuned configuration")
        print("  has not earned a place in the deployed one.")


if __name__ == "__main__":
    main()
