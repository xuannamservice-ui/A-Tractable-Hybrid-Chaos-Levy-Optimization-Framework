"""How much of the reported performance rests on the branch-independence
assumption, and what happens as it is relaxed?

The manuscript combines MN = 16 branches as i.i.d. draws of the composite
h = h_a h_p, which is what licenses the MN-fold convolution behind Theorem 1.
One fast-steering mirror pair sets Theta(t) for the whole array, though, so the
boresight component of the pointing error is COMMON to every branch.  The
question is not whether the pointing error is independent -- it is partly both
-- but how the total jitter divides, and how much the answer moves with that
division.

The split is parameterised rather than guessed.  Write

    sigma_s^2 = rho sigma_s^2 + (1 - rho) sigma_s^2 = sigma_com^2 + sigma_br^2,

so the per-axis displacement of branch i is r_i = r_com + r_br,i with r_com
shared and r_br,i independent.  Because both terms are zero-mean Gaussian, the
MARGINAL per-branch pointing law is identical for every rho -- only the
correlation across branches changes -- so rho moves nothing except the
diversity the array gets against pointing, which is exactly the quantity in
question.

No new numerics are needed.  Conditioned on r_com the branches ARE independent,
with per-branch jitter sigma_br and a common boresight offset r_com, and that
is precisely the configuration `system_metric.aber_of` already evaluates
through the equivalent-Beckmann fold of eq. (15).  So

    Pe(rho) = E_{r_com ~ Rayleigh(sigma_com)} [ aber_of(w_z, sigma_br, r_com) ],

a Gauss-Legendre average over r_com of the certified evaluator.  At rho = 0
this returns the deployed number identically (sigma_br = sigma_s, r_com = 0);
as rho -> 1 the pointing tail loses its MN-fold averaging.

Usage:  python branch_model_probe.py [--w-z 0.192] [--nodes 24]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from channel import beam_geometry
from system_metric import BeamConfig, aber_of

REGIMES = ("weak", "moderate", "strong")
TARGET = 1e-6
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)


def aber_split(regime, w_z, sigma_s, rho, gbar_db, nodes=24):
    """Post-EGC ABER with a fraction `rho` of the jitter variance common to all
    branches and the rest independent per branch."""
    if rho <= 0.0:
        return aber_of(BeamConfig(regime=regime, w_z=w_z, sigma_s=sigma_s, r_d=0.0),
                       gbar_db)
    s_com = sigma_s * np.sqrt(rho)
    s_br = sigma_s * np.sqrt(1.0 - rho)
    # r_com is 2-D Gaussian per axis with scale s_com, so its radius is
    # Rayleigh(s_com): f(r) = r/s^2 exp(-r^2/2s^2).  Integrate to 6 sigma.
    gu, gw = np.polynomial.legendre.leggauss(nodes)
    hi = 6.0 * s_com
    r = 0.5 * (gu + 1.0) * hi
    w = gw * 0.5 * hi
    pdf = r / s_com ** 2 * np.exp(-r ** 2 / (2.0 * s_com ** 2))
    vals = np.array([aber_of(BeamConfig(regime=regime, w_z=w_z, sigma_s=s_br,
                                        r_d=float(rr)), gbar_db) for rr in r])
    norm = float((w * pdf).sum())          # < 1 by the 6-sigma truncation
    return float((w * pdf * vals).sum() / norm)


def gamma_req(fn, lo=10.0, hi=90.0, tol=2e-3):
    if fn(hi) > TARGET:
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if fn(mid) <= TARGET:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w-z", type=float, default=0.192,
                    help="beam waist; default is the interior optimum")
    ap.add_argument("--sigma-s", type=float, default=0.10)
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    args = ap.parse_args()

    A0, weq = beam_geometry(args.w_z)
    xi = float(weq) / (2.0 * args.sigma_s)
    print(f"w_z={args.w_z} m  A0={float(A0):.4f}  xi={xi:.4f}  "
          f"sigma_s={args.sigma_s} m  ({args.nodes}-node Rayleigh average)\n")

    out = {"generated_by": "code/branch_model_probe.py", "w_z": args.w_z,
           "sigma_s": args.sigma_s, "A0": float(A0), "xi": xi,
           "nodes": args.nodes, "regimes": {}}

    print(f"{'regime':>9} {'rho':>6} {'sigma_br':>9} {'gamma_req (dB)':>15} {'penalty':>9}")
    for regime in args.regimes:
        rows, base = [], None
        for rho in RHOS:
            g = gamma_req(lambda gdb: aber_split(regime, args.w_z, args.sigma_s,
                                                 rho, gdb, nodes=args.nodes))
            if rho == 0.0:
                base = g
            pen = None if (g is None or base is None) else g - base
            rows.append({"rho": rho, "sigma_br": args.sigma_s * np.sqrt(1 - rho),
                         "gamma_req_db": g, "penalty_db": pen})
            sg = "---" if g is None else f"{g:15.3f}"
            sp = "---" if pen is None else f"{pen:9.3f}"
            print(f"{regime:>9} {rho:>6.2f} {args.sigma_s * np.sqrt(1 - rho):>9.4f} {sg} {sp}")
        out["regimes"][regime] = rows

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "20_branch_model")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"branch_model_wz{args.w_z}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
