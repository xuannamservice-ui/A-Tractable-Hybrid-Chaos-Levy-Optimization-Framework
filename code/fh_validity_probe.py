"""How much does the Farid-Hranilovic pointing-loss approximation cost at the
beam widths this paper actually commands?

The manuscript's eq:hp_def uses the standard approximation

    h_p(r) ~= A0 exp(-2 r^2 / w_zeq^2),   A0 = erf(v)^2,  v = sqrt(pi/2) a / w_z,
    w_zeq^2 = w_z^2 sqrt(pi) erf(v) / (2 v exp(-v^2)),

which Farid and Hranilovic derive for a wide beam and quote as accurate for
w_z/a >~ 6.  This link runs narrower than that: a = 0.05 m and the commanded
waists sit between the guard floor 0.0549 m (w_z/a = 1.10) and the interior
optimum 0.192 m (w_z/a = 3.84).  So the question is not whether the
approximation is formally in range -- it is not, at the operating point -- but
how much it moves the numbers that are reported.

Two things are measured here, both against the exact overlap integral

    h_p^exact(r) = (2 / (pi w_z^2)) Int_{|x - r e_1| <= a} exp(-2 |x|^2 / w_z^2) dx

evaluated in polar coordinates about the aperture centre:

  (1) the relative error in h_p(r) itself, over the displacement range the
      sway law actually visits (r <= 3 sigma_s);
  (2) the relative error in the per-branch ABER it induces, by pushing both
      h_p laws through the same Rayleigh-r pointing model, the same
      gamma-gamma turbulence and the same Q(.) integration -- i.e. the
      quantity eq:aber_emulator approximates.

Usage:  python fh_validity_probe.py [--a 0.05] [--n-r 400]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.special import erf, kv, gamma as G, erfc

A_APERTURE = 0.05


def A0_and_wzeq(w_z, a=A_APERTURE):
    v = np.sqrt(np.pi / 2.0) * a / w_z
    A0 = erf(v) ** 2
    wzeq2 = w_z ** 2 * np.sqrt(np.pi) * erf(v) / (2.0 * v * np.exp(-v ** 2))
    return A0, np.sqrt(wzeq2)


def hp_approx(r, w_z, a=A_APERTURE):
    A0, wzeq = A0_and_wzeq(w_z, a)
    return A0 * np.exp(-2.0 * r ** 2 / wzeq ** 2)


def hp_exact(r, w_z, a=A_APERTURE, n_rho=600, n_phi=600):
    """Fraction of a unit-power Gaussian beam of waist w_z collected by a
    circular aperture of radius a whose centre is r from the beam axis.

    Polar grid about the APERTURE centre: rho in [0, a], phi in [0, 2pi).
    The beam-axis distance of a point at (rho, phi) is
    |x|^2 = r^2 + rho^2 - 2 r rho cos(phi) (law of cosines), so the integrand
    is smooth and the Gauss-Legendre product rule converges quickly.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    gu, gw = np.polynomial.legendre.leggauss(n_rho)
    rho = 0.5 * (gu + 1.0) * a
    wrho = gw * 0.5 * a * rho                      # rho d rho
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    wphi = 2.0 * np.pi / n_phi                     # midpoint rule, periodic
    d2 = (r[:, None, None] ** 2 + rho[None, :, None] ** 2
          - 2.0 * r[:, None, None] * rho[None, :, None] * np.cos(phi[None, None, :]))
    integrand = np.exp(-2.0 * d2 / w_z ** 2)
    val = (integrand.sum(axis=2) * wphi * wrho[None, :]).sum(axis=1)
    return (2.0 / (np.pi * w_z ** 2)) * val


def gg_pdf(x, alpha, beta):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x > 0
    c = 2.0 * (alpha * beta) ** ((alpha + beta) / 2.0) / (G(alpha) * G(beta))
    out[m] = c * x[m] ** ((alpha + beta) / 2.0 - 1.0) * kv(alpha - beta,
                                                           2.0 * np.sqrt(alpha * beta * x[m]))
    return np.where(np.isfinite(out), out, 0.0)


def aber_branch(hp_of_r, w_z, sigma_s, alpha, beta, gbar, n_r=400, n_ha=2000):
    """E_r E_ha [ Q( sqrt(2 gbar) h_a h_p(r) ) ] with r ~ Rayleigh(sigma_s).

    One expectation over the pointing displacement and one over turbulence --
    the same two the manuscript's branch ABER takes, differing only in which
    h_p(r) law is used, so the ratio isolates the approximation.
    """
    r = np.linspace(0.0, 6.0 * sigma_s, n_r)
    f_r = (r / sigma_s ** 2) * np.exp(-r ** 2 / (2.0 * sigma_s ** 2))
    hp = hp_of_r(r, w_z)
    ha = np.linspace(1e-6, 12.0, n_ha)
    f_ha = gg_pdf(ha, alpha, beta)
    h = ha[None, :] * hp[:, None]
    q = 0.5 * erfc(np.sqrt(gbar) * h)              # Q(sqrt(2 gbar) h)
    inner = np.trapezoid(q * f_ha[None, :], ha, axis=1)
    return float(np.trapezoid(inner * f_r, r))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=float, default=A_APERTURE)
    ap.add_argument("--n-r", type=int, default=400)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "15_fh_validity")
    os.makedirs(out_dir, exist_ok=True)

    # The waists this paper commands, plus the ratio Farid-Hranilovic quote.
    # "interior optimum" was 0.192 (an earlier value superseded before publication);
    # now 0.0558 m to match the certified optimum reported in the manuscript.
    cases = [("guard floor", 0.054869), ("boundary static", 0.123),
             ("xi_safe fallback", 0.157), ("interior optimum", 0.0558),
             ("F-H nominal", 6.0 * args.a)]
    sigma_s = 0.10
    regimes = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
    gbar = 10 ** (38.0 / 10.0)

    o = {"generated_by": "code/fh_validity_probe.py",
         "a": args.a, "sigma_s": sigma_s, "gbar_db": 38.0, "cases": []}
    print(f"{'case':18s} {'w_z':>8s} {'w_z/a':>6s} {'hp err @0':>10s} "
          f"{'hp err(sig)':>11s} " + " ".join(f"{'ratio ' + k:>14s}" for k in regimes))
    for name, w_z in cases:
        r = np.linspace(0.0, 3.0 * sigma_s, args.n_r)
        ha_ = hp_approx(r, w_z, args.a)
        he_ = hp_exact(r, w_z, args.a)
        rel = np.abs(ha_ - he_) / np.maximum(he_, 1e-300)
        # Pointwise relative error is unbounded far out in the tail, where h_p
        # itself is numerically irrelevant; restrict it to where the exact loss
        # still carries a thousandth of its on-axis value.
        A0, _ = A0_and_wzeq(w_z, args.a)
        sig = he_ >= 1e-3 * A0
        row = {"name": name, "w_z": w_z, "w_z_over_a": w_z / args.a,
               "rel_err_hp_at_zero": float(rel[0]),
               "max_rel_err_hp": float(rel.max()),
               "max_rel_err_hp_significant": float(rel[sig].max()) if sig.any() else 0.0,
               "aber_ratio": {}}
        for rname, (alpha, beta) in regimes.items():
            a_ap = aber_branch(lambda rr, ww: hp_approx(rr, ww, args.a), w_z,
                               sigma_s, alpha, beta, gbar)
            a_ex = aber_branch(lambda rr, ww: hp_exact(rr, ww, args.a), w_z,
                               sigma_s, alpha, beta, gbar)
            row["aber_ratio"][rname] = a_ap / a_ex if a_ex else float("nan")
        o["cases"].append(row)
        print(f"{name:18s} {w_z:8.4f} {w_z/args.a:6.2f} {rel[0]:10.4f} "
              f"{row['max_rel_err_hp_significant']:11.4f} "
              + " ".join(f"{row['aber_ratio'][k]:14.4f}" for k in regimes))

    path = os.path.join(out_dir, "fh_validity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
