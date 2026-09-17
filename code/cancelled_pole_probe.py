"""Near a pole that truncation DOES cancel, is the residual a round-off floor
or a cancellation that grows without bound?

`d_pole_probe.py` handles the poles truncation leaves UNCANCELLED (k > K).
This script asks the complementary question about the cancelled ones (k <= K).
Eq. (eq:ak_factorised) carries the factor xi^2 / ((xi^2 - beta - k) A_0^{beta+k}):
at xi^2 -> beta + k the individual term diverges, and the series stays finite
only because a matching divergence cancels it. In float64 that cancellation is
catastrophic, and -- unlike a round-off floor, which is a constant -- its error
grows as the distance to the pole shrinks.

This matters because the manuscript's off-grid campaign samples xi UNIFORMLY,
which lands near a pole only with probability proportional to the band width,
while the optimizer descends in xi and can stop anywhere. The worst draw of
that campaign is itself a near-pole hit: xi = 2.646 in the weak regime gives
xi^2 = 7.0013 against beta + 4 = 7.000, a distance of 1.3e-3 -- 77x closer than
the delta = 0.1 clearance eq:pole_free enforces at the tabulated nodes.

Measured here: float64 kernel vs the same expression at 200 digits, walking
xi^2 toward a cancelled pole. If the error scales like 1/distance the residual
is cancellation, not round-off, and no fixed floor bounds it.

Usage:  python cancelled_pole_probe.py [--regime weak] [--k 4]
"""
from __future__ import annotations

import argparse
import json
import os

import mpmath as mp
import numpy as np

from rtodt import Pe_series, wz_for_xi, A0_of, z_param
from rtodt_fast import pe_series_f64
from hclpso_ga import ladder_order

REGIMES = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}


def measure(alpha, beta, k, sigma_s, gbar_db, offsets):
    """Walk xi^2 toward the cancelled pole xi^2 = beta + k and compare the
    deployed float64 kernel against 260-digit arithmetic at the same order."""
    g = 10.0 ** (gbar_db / 10.0)
    pole = float(beta) + k
    rows = []
    for d in offsets:
        for sgn in (-1.0, +1.0):
            xi2 = pole + sgn * d
            if xi2 <= 0:
                continue
            xi = float(np.sqrt(xi2))
            wz = wz_for_xi(mp.mpf(str(xi)), mp.mpf(str(sigma_s)))
            if wz is None:
                continue
            A0 = float(A0_of(wz))
            z = float(z_param(*(mp.mpf(str(v)) for v in (alpha, beta, A0, g))))
            K = int(np.atleast_1d(ladder_order(np.atleast_1d(z)))[0])
            if K < 0 or k > K:
                continue                      # inadmissible, or not cancelled
            got = float(np.atleast_1d(pe_series_f64(
                alpha, beta, np.atleast_1d(xi), np.atleast_1d(A0), g, K))[0])
            ref = float(Pe_series(*(mp.mpf(str(v)) for v in (alpha, beta, xi, A0, g)), K))
            rows.append({"signed_offset": sgn * d, "xi": xi, "A0": A0, "z": z,
                         "K": K, "f64": got, "ref260": ref,
                         "abs_err": abs(got - ref),
                         "rel_err": abs(got - ref) / abs(ref) if ref else float("nan")})
    return pole, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", default="weak", choices=sorted(REGIMES))
    ap.add_argument("--k", type=int, default=4, help="pole index; cancelled iff k <= K")
    ap.add_argument("--sigma-s", type=float, default=0.05)
    ap.add_argument("--gbar-db", type=float, default=30.0)
    args = ap.parse_args()

    alpha, beta = REGIMES[args.regime]
    offsets = [10.0 ** e for e in np.arange(-12.0, -0.9, 0.5)]
    pole, rows = measure(alpha, beta, args.k, args.sigma_s, args.gbar_db, offsets)

    print(f"regime={args.regime} (alpha,beta)=({alpha},{beta})  cancelled pole "
          f"xi^2 = beta+{args.k} = {pole}  sigma_s={args.sigma_s} gbar={args.gbar_db} dB")
    print(f"{'d(xi^2)':>11} {'xi':>8} {'K':>3} {'z':>7} {'f64':>13} "
          f"{'abs err':>11} {'rel err':>11} {'abs*d':>11}")
    for r in sorted(rows, key=lambda r: abs(r["signed_offset"])):
        print(f"{r['signed_offset']:>11.1e} {r['xi']:>8.4f} {r['K']:>3} {r['z']:>7.3f} "
              f"{r['f64']:>13.5e} {r['abs_err']:>11.2e} {r['rel_err']:>11.2e} "
              f"{r['abs_err'] * abs(r['signed_offset']):>11.2e}")

    finite = [r for r in rows if np.isfinite(r["abs_err"])]
    if finite:
        worst = max(finite, key=lambda r: r["abs_err"])
        print(f"\nworst absolute error {worst['abs_err']:.2e} at offset "
              f"{worst['signed_offset']:.1e} (rel {worst['rel_err']:.2e})")
        print("A constant `abs*d` column means the error scales as 1/distance, i.e.")
        print("cancellation rather than a fixed round-off floor.")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "18_cancelled_pole")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cancelled_pole_{args.regime}_k{args.k}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"generated_by": "code/cancelled_pole_probe.py",
                   "regime": args.regime, "alpha": alpha, "beta": beta,
                   "k": args.k, "pole_xi2": pole, "sigma_s": args.sigma_s,
                   "gbar_db": args.gbar_db, "rows": rows}, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
