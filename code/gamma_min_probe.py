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

NOTE on sigma_s: `SIGMAS`, imported from `system_metric`, are the raw
building-sway sweep levels (Sec. VII-A), not the steering loop's tracked
pointing residual. The published Table `success_feasibility` reports
gamma_min at the TRACKED residual (sigma_s = 6.7/13.3/26.6/40.2 mm), so the
rows this script prints under the raw SIGMAS values (0.05/0.10/0.20/0.30 m)
do not reproduce that table as released; a caller wanting the published
figures must substitute the tracked-residual values for `sigma_s` explicitly
(the same applies to `gamma_req`'s default `sigma_s=0.10`, also raw).

Usage:  python gamma_min_probe.py [--tol 1e-3]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scipy.optimize import minimize_scalar

from feasibility_at_gbar import best_at
from system_metric import REGIMES, SIGMAS, BeamConfig, aber_of, ABER_TARGET

W_STATIC = 0.123          # manuscript's boundary static beam
W_OPT = 0.0558            # manuscript's interior optimum (was 0.192, an earlier value superseded before publication)


def gamma_min(regime, sigma_s, lo=15.0, hi=65.0, tol=1e-3):
    if best_at(regime, sigma_s, hi) > ABER_TARGET:
        return None                       # not feasible anywhere below `hi`
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if best_at(regime, sigma_s, mid) <= ABER_TARGET:
            hi = mid
        else:
            lo = mid
    return hi


def gamma_req(w_z, regime="strong", sigma_s=0.10, lo=15.0, hi=65.0, tol=1e-3):
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
         "tol_db": args.tol, "target": ABER_TARGET, "gamma_min": {}}
    print(f"{'regime':>9} {'sigma_s':>8} {'gamma_min (dB)':>15}")
    for regime in REGIMES:
        for s in SIGMAS:
            g = gamma_min(regime, s, tol=args.tol)
            o["gamma_min"]["%s_%.2f" % (regime, s)] = g
            print(f"{regime:>9} {s:>8.2f} {('---' if g is None else '%.4f' % g):>15}")

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
        g_o = o["gamma_min"].get("%s_%.2f" % (regime, 0.10))
        d = None if (g_s is None or g_o is None) else g_s - g_o
        o["gain_by_regime"][regime] = {"gamma_req_static_db": g_s,
                                       "gamma_req_opt_db": g_o, "delta_db": d,
                                       "opt_endpoint": "gamma_min at sigma_s=0.10"}
        print(f"{regime:>9} {g_s:>12.4f} {g_o:>10.4f} {d:>10.4f}")

    # Both calls below use gamma_req's default sigma_s=0.10 (raw sway, see the
    # module NOTE above). W_OPT is sized for the TRACKED residual, so scoring
    # it at raw sway understates it relative to W_STATIC and this section's
    # gs - go does NOT reproduce the manuscript's reported required-SNR gain.
    gs = gamma_req(W_STATIC, tol=args.tol)
    go = gamma_req(W_OPT, tol=args.tol)
    a_s41 = aber_of(BeamConfig(regime="strong", w_z=W_STATIC, sigma_s=0.10, r_d=0.0), 41.0)
    a_o38 = aber_of(BeamConfig(regime="strong", w_z=W_OPT, sigma_s=0.10, r_d=0.0), 38.0)
    # Local slope over the crossing region, dB per decade of ABER: the SNR
    # interval between the quoted anchor and the 1e-6 crossing, divided by the
    # decades of ABER covered over the same interval.
    slope_s = (gs - 41.0) / np.log10(a_s41 / ABER_TARGET)
    slope_o = (go - 38.0) / np.log10(a_o38 / ABER_TARGET)
    o["required_snr"] = {
        "w_static": W_STATIC, "gamma_req_static_db": gs,
        "w_opt": W_OPT, "gamma_req_opt_db": go,
        "delta_db": None if (gs is None or go is None) else gs - go,
        "aber_static_at_41db": a_s41, "aber_opt_at_38db": a_o38,
        "slope_static_db_per_decade": slope_s,
        "slope_opt_db_per_decade": slope_o,
    }
    print(f"\nboundary static  w_z={W_STATIC} m: gamma_req = {gs:.4f} dB "
          f"(ABER at 41.0 dB = {a_s41:.4e}, local slope {slope_s:.2f} dB/decade)")
    print(f"interior optimum w_z={W_OPT} m: gamma_req = {go:.4f} dB "
          f"(ABER at 38.0 dB = {a_o38:.4e}, local slope {slope_o:.2f} dB/decade)")
    print(f"required-SNR reduction = {gs - go:.4f} dB")

    path = os.path.join(out_dir, "gamma_min.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
