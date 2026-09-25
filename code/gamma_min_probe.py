"""Regenerate the feasibility column gamma_min of Table `success_feasibility`,
and the two required-SNR figures behind the reported divergence gain.

gamma_min(regime, sigma_s) is the lowest reference SNR at which SOME beam in
the decision box meets the 1e-6 post-EGC target at boresight (r_d = 0).  The
manuscript obtains it on a finite xi mesh, which can only overestimate it.
Here the inner search is `feasibility_at_gbar.best_at` (a bounded continuous
minimisation over xi, the same certified `system_metric.aber_of` underneath)
and the outer search is a bisection on gbar to 1e-3 dB, so the column comes
out as a continuous optimum rather than a mesh upper bound.

The second half recomputes the two endpoints of the reported required-SNR
reduction on the same evaluator: the boundary static beam (widened just until
its pointing floor clears 1e-6) and the interior divergence optimum.

NOTE on sigma_s: Table `success_feasibility` labels its rows by the raw
building-sway level (Sec. VII-A) for cross-reference but reports gamma_min
computed at the corresponding TRACKED residual (its "tracked, mm" column),
since that is the displacement the beam actually sees once the steering loop
has closed. `SIGMAS_TRACKED` below is that column, in metres
(6.7/13.3/26.6/40.2 mm), and is what this script sweeps; `gamma_req`'s
default `sigma_s` is likewise the tracked nominal (13.3 mm), matching the
jitter eq:snr_reduction's two endpoints are scored under. Verified directly:
at the three tracked cells whose published crossing exceeds 15 dB (where the
bisection bound below did not yet mask the answer), this reproduces the
table to within 0.07 dB (moderate/40.2mm: 17.71 vs 17.64; strong/26.6mm:
17.176 vs 17.18; strong/40.2mm: 21.174 vs 21.11).

TWO caveats this fix carries:
(i) `lo` below was 15.0 dB, calibrated for the raw-sway sweep, where every
    crossing sits above it. At the tracked residual six of the twelve cells
    cross below 15 dB (as low as 6.98 dB), and a plain bisection with lo
    above the true crossing silently returns ~lo instead of raising an
    error. Confirmed directly: with lo=15.0, six tracked cells returned
    15.0008 regardless of their true (lower) crossing. `lo` is widened to
    0.0 below to clear every published crossing (min. 6.98 dB) with margin.
(ii) At the smallest tracked residual (6.7 mm = 0.0067 m), the geometric
    floor xi_min(sigma_s) = 0.087719/(2*sigma_s) = 6.546 EXCEEDS XI_MAX =
    4.888 (system_metric.py): under the currently released XI_MAX, no beam
    width satisfies xi <= XI_MAX at this sigma_s, so the decision box is
    empty regardless of reference SNR -- yet Table `success_feasibility`
    reports this as the EASIEST (lowest gamma_min) cell in every regime.
    This is an unresolved inconsistency between the published table and the
    released XI_MAX, not something this script invents a number to paper
    over: the three affected cells (one per regime, at sigma_s=0.0067 m)
    are reported as BOX EMPTY rather than silently wrong or crashing.

Usage:  python gamma_min_probe.py [--tol 1e-3]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scipy.optimize import minimize_scalar

from feasibility_at_gbar import best_at
from system_metric import REGIMES, BeamConfig, aber_of, ABER_TARGET, XI_MAX, branch_min_wz

W_STATIC = 0.123          # manuscript's boundary static beam
W_OPT = 0.0558            # manuscript's interior optimum (was 0.192, an earlier value superseded before publication)

# Table `success_feasibility`'s "sigma_s (tracked, mm)" column, in metres.
SIGMAS_TRACKED = (0.0067, 0.0133, 0.0266, 0.0402)
SIGMA_TRACKED_NOMINAL = 0.0133      # the jitter eq:snr_reduction's endpoints use


def xi_min_geometric(sigma_s):
    """Geometric floor of the decision box (Sec. VII-A): the narrowest
    admissible beam (the branch minimum w_z*) still yields this xi at the
    given sigma_s. If it exceeds XI_MAX, no beam width in the released
    geometric domain satisfies xi <= XI_MAX here -- see caveat (ii) above."""
    _, weq_floor = branch_min_wz()
    return weq_floor / (2.0 * sigma_s)


def gamma_min(regime, sigma_s, lo=0.0, hi=65.0, tol=1e-3):
    if xi_min_geometric(sigma_s) > XI_MAX:
        return None            # decision box empty at this sigma_s -- see caveat (ii)
    if best_at(regime, sigma_s, hi) > ABER_TARGET:
        return None                       # not feasible anywhere below `hi`
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if best_at(regime, sigma_s, mid) <= ABER_TARGET:
            hi = mid
        else:
            lo = mid
    return hi


def gamma_req(w_z, regime="strong", sigma_s=SIGMA_TRACKED_NOMINAL, lo=0.0, hi=65.0, tol=1e-3):
    f = lambda g: aber_of(BeamConfig(regime=regime, w_z=w_z, sigma_s=sigma_s, r_d=0.0), g)
    if f(hi) > ABER_TARGET:
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) <= ABER_TARGET:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=1e-3, help="bisection tolerance, dB")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "17_gamma_min")
    os.makedirs(out_dir, exist_ok=True)

    o = {"generated_by": "code/gamma_min_probe.py",
         "tol_db": args.tol, "target": ABER_TARGET, "gamma_min": {}, "box_empty": {}}
    print(f"{'regime':>9} {'sigma_s (tracked, m)':>21} {'gamma_min (dB)':>15}")
    for regime in REGIMES:
        for s in SIGMAS_TRACKED:
            key = "%s_%.4f" % (regime, s)
            xi_m = xi_min_geometric(s)
            if xi_m > XI_MAX:
                o["gamma_min"][key] = None
                o["box_empty"][key] = xi_m
                print(f"{regime:>9} {s:>21.4f} {'BOX EMPTY (xi_min=%.3f>XI_MAX=%.3f)' % (xi_m, XI_MAX):>15}")
                continue
            g = gamma_min(regime, s, tol=args.tol)
            o["gamma_min"][key] = g
            print(f"{regime:>9} {s:>21.4f} {('---' if g is None else '%.4f' % g):>15}")

    # The same comparison in every regime, holding the static beam fixed so the
    # baseline is one piece of hardware rather than a per-regime refit: the
    # divergence gain is then a property of the turbulence, not of the baseline.
    # The optimised endpoint is gamma_min at the same jitter, already computed
    # above -- that IS min over w_z of gamma_req, so re-deriving it by minimising
    # ABER at some fixed gbar would only introduce an arbitrary reference SNR.
    print(f"\n{'regime':>9} {'static greq':>12} {'opt greq':>10} {'delta dB':>10}")
    o["gain_by_regime"] = {}
    for regime in REGIMES:
        g_s = gamma_req(W_STATIC, regime=regime, tol=args.tol)
        g_o = o["gamma_min"].get("%s_%.4f" % (regime, SIGMA_TRACKED_NOMINAL))
        d = None if (g_s is None or g_o is None) else g_s - g_o
        o["gain_by_regime"][regime] = {"gamma_req_static_db": g_s,
                                       "gamma_req_opt_db": g_o, "delta_db": d,
                                       "opt_endpoint": "gamma_min at sigma_s=%.4f" % SIGMA_TRACKED_NOMINAL}
        print(f"{regime:>9} {g_s:>12.4f} {g_o:>10.4f} {d:>10.4f}")

    # Both calls below use gamma_req's default sigma_s (the tracked nominal,
    # see the module NOTE above), matching how eq:snr_reduction's two
    # endpoints are scored in the manuscript.
    gs = gamma_req(W_STATIC, tol=args.tol)
    go = gamma_req(W_OPT, tol=args.tol)
    # Local slope over the crossing region, dB per decade of ABER: the SNR
    # interval between an anchor 1 dB below the 1e-6 crossing and the crossing
    # itself, divided by the decades of ABER covered over that interval.
    anchor_s, anchor_o = gs - 1.0, go - 1.0
    a_s = aber_of(BeamConfig(regime="strong", w_z=W_STATIC, sigma_s=SIGMA_TRACKED_NOMINAL, r_d=0.0), anchor_s)
    a_o = aber_of(BeamConfig(regime="strong", w_z=W_OPT, sigma_s=SIGMA_TRACKED_NOMINAL, r_d=0.0), anchor_o)
    slope_s = (gs - anchor_s) / np.log10(a_s / ABER_TARGET)
    slope_o = (go - anchor_o) / np.log10(a_o / ABER_TARGET)
    o["required_snr"] = {
        "w_static": W_STATIC, "gamma_req_static_db": gs,
        "w_opt": W_OPT, "gamma_req_opt_db": go,
        "delta_db": None if (gs is None or go is None) else gs - go,
        "anchor_static_db": anchor_s, "aber_static_at_anchor": a_s,
        "anchor_opt_db": anchor_o, "aber_opt_at_anchor": a_o,
        "slope_static_db_per_decade": slope_s,
        "slope_opt_db_per_decade": slope_o,
    }
    print(f"\nboundary static  w_z={W_STATIC} m: gamma_req = {gs:.4f} dB "
          f"(ABER at {anchor_s:.2f} dB = {a_s:.4e}, local slope {slope_s:.2f} dB/decade)")
    print(f"interior optimum w_z={W_OPT} m: gamma_req = {go:.4f} dB "
          f"(ABER at {anchor_o:.2f} dB = {a_o:.4e}, local slope {slope_o:.2f} dB/decade)")
    print(f"required-SNR reduction = {gs - go:.4f} dB")

    path = os.path.join(out_dir, "gamma_min.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
