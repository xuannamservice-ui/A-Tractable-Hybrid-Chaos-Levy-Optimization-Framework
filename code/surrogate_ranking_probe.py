"""What actually justifies ranking candidates on the per-branch surrogate?

Proposition 1 of the manuscript is a conditional: IF two candidates' branch
irradiance laws are ordered by first-order stochastic dominance, THEN the
post-EGC system ABER inherits that order.  First-order dominance is only a
PARTIAL order, so the proposition is silent on incomparable pairs and does not
by itself give an argmin equivalence.  This script measures both halves of
that gap:

  (A) DOMINANCE.  For the divergence settings this paper actually compares,
      are the branch CDFs ordered at all?  F_h(x) = E_{h_p}[F_{h_a}(x/h_p)]
      with h_p = A0 U^{1/xi^2} (U uniform on [0,1]) and h_a gamma-gamma.  A
      sign change in F_1 - F_2 over the support means the pair is
      INCOMPARABLE and Proposition 1 says nothing about it.

  (B) RANK AGREEMENT.  Spearman rho and Kendall tau between the per-branch
      surrogate and the post-EGC system metric over the admissible beams of
      the decision box -- the empirical statement that carries the surrogate
      reduction in place of dominance.

Usage:  python surrogate_ranking_probe.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr, kendalltau

from fh_validity_probe import A0_and_wzeq, gg_pdf

ALPHA, BETA = 1.2, 1.1          # strong turbulence
SIGMA_S = 0.10
GBAR_DB = 38.0


def _gg_cdf_grid(alpha, beta):
    ha = np.concatenate([np.logspace(-9, -3, 3000),
                         np.linspace(1.001e-3, 40.0, 200000)])
    f = gg_pdf(ha, alpha, beta)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ha))])
    return ha, F


def branch_cdf(w_z, x, ha, F_ha, sigma_s=SIGMA_S, n_u=4000):
    """F_h(x) for h = h_a h_p, by averaging F_{h_a}(x/h_p) over the pointing law."""
    A0, wzeq = A0_and_wzeq(w_z)
    xi = wzeq / (2.0 * sigma_s)
    gu, gw = np.polynomial.legendre.leggauss(n_u)
    u = 0.5 * (gu + 1.0)
    w = gw * 0.5
    hp = A0 * u ** (1.0 / xi ** 2)
    arg = x[:, None] / hp[None, :]
    return (np.interp(arg, ha, F_ha, left=0.0, right=1.0) * w[None, :]).sum(axis=1), xi, A0


def dominance_check(pairs, tol=1e-4):
    ha, F_ha = _gg_cdf_grid(ALPHA, BETA)
    x = np.logspace(-6, 1.7, 4000)
    rows = []
    for w1, w2 in pairs:
        F1, xi1, A01 = branch_cdf(w1, x, ha, F_ha)
        F2, xi2, A02 = branch_cdf(w2, x, ha, F_ha)
        d = F1 - F2
        m = ((F1 > 1e-4) & (F1 < 1 - 1e-4)) | ((F2 > 1e-4) & (F2 < 1 - 1e-4))
        dm, xm = d[m], x[m]
        rows.append({"w_z_1": w1, "w_z_2": w2, "xi_1": xi1, "xi_2": xi2,
                     "A0_1": A01, "A0_2": A02,
                     "max_gap": float(dm.max()), "h_at_max": float(xm[np.argmax(dm)]),
                     "min_gap": float(dm.min()), "h_at_min": float(xm[np.argmin(dm)]),
                     "cdfs_cross": bool(dm.max() > tol and dm.min() < -tol)})
        r = rows[-1]
        print(f"  w_z={w1:.4f} (xi={xi1:.3f}) vs {w2:.4f} (xi={xi2:.3f}): "
              f"max(F1-F2)={r['max_gap']:+.5f} @h={r['h_at_max']:.4g}, "
              f"min={r['min_gap']:+.5f} @h={r['h_at_min']:.4g} -> "
              + ("INCOMPARABLE" if r["cdfs_cross"] else "ordered"))
    return rows


def rank_agreement():
    """Same construction as verify.py's tier-1 surrogate-monotonicity check,
    with Kendall tau added alongside the Spearman rho it already reports."""
    from system_metric import BeamConfig, aber_of
    from channel import beam_geometry
    from rtodt_fast import pe_series_f64, z_of
    from hclpso_ga import ladder_order

    s = 0.05                       # verify.py's configuration, kept identical
    g = 10 ** (GBAR_DB / 10.0)
    W = np.linspace(0.055, 3.0, 90)
    A0, weq = beam_geometry(W)
    z = z_of(ALPHA, BETA, A0, g)
    pb = pe_series_f64(ALPHA, BETA, weq / (2 * s), A0, g, ladder_order(z))
    sy = np.array([aber_of(BeamConfig(regime="strong", w_z=float(w), sigma_s=s,
                                      r_d=0.0), GBAR_DB) for w in W])
    m = np.isfinite(pb) & np.isfinite(sy) & (pb >= 0) & (pb <= 0.5)
    rho = float(spearmanr(pb[m], sy[m]).statistic)
    tau = float(kendalltau(pb[m], sy[m]).statistic)
    print(f"  n={int(m.sum())} admissible beams: Spearman rho={rho:.4f}, "
          f"Kendall tau={tau:.4f}")
    return {"n_admissible": int(m.sum()), "sigma_s": s, "gbar_db": GBAR_DB,
            "spearman_rho": rho, "kendall_tau": tau}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "16_surrogate_ranking")
    os.makedirs(out_dir, exist_ok=True)

    print("(A) first-order dominance between the manuscript's own divergence settings")
    pairs = [(0.123, 0.192), (0.157, 0.192), (0.054869, 0.192), (0.123, 0.157)]
    rows = dominance_check(pairs)

    print("\n(B) rank agreement, per-branch surrogate vs post-EGC system metric")
    ranks = rank_agreement()

    o = {"alpha": ALPHA, "beta": BETA, "sigma_s_dominance": SIGMA_S,
         "dominance": rows, "rank_agreement": ranks}
    path = os.path.join(out_dir, "surrogate_ranking.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
