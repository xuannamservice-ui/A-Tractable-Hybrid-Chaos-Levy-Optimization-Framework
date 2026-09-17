"""
Real implementation of Section IV-B's GH-NLS-KF, closing the gap between the
manuscript's eq. (23)-(25) and the released `mpc_loop.KalmanAR1`, which tracks
the latent state correctly but returns mu_k (the LATENT prediction) with no
MNLT transform and no Gauss-Hermite correction applied at all -- `_stage_gbar`
then uses `1 + mu_k` directly as h_a(t+k), which is not even the "naive
certainty-equivalence predictor" T(mu_k) the manuscript names and measures as
biased (17.8%-42.8% error): it skips the transform T entirely.

This module supplies:
  * the missing forward/inverse MNLT maps, built on channel.py's own
    gamma-gamma CDF table (the same table GammaGammaAR1 uses to generate the
    channel, so predictor and channel agree on what T is);
  * GaussHermiteKF(KalmanAR1): same Riccati/gain machinery, but
      - update() takes a PHYSICAL h_a measurement and maps it through the
        forward MNLT to the latent g before the scalar KF update (the
        released KalmanAR1.update feeds h_a-1 in directly, silently treating
        the physical zero-mean perturbation as if it were already the latent
        Gaussian state);
      - predict() returns eq. (25)'s 3-point Gauss-Hermite quadrature through
        T at each horizon step, using sigma_k^2 = 1 - rho^{2k}(1-P) (the
        predictive-dispersion formula stated just above eq. (25));
  * naive_ce(): T(mu_k) alone, the manuscript's own named baseline, for the
    three-way accuracy comparison the manuscript's prose describes but no
    released script measures.

Validates the manuscript's own claim ("bounds prediction error below 0.45%
... verified against 150,000-sample Monte Carlo simulations") against an
independent Monte-Carlo ground truth, for the three predictors side by side.
"""
from __future__ import annotations

import numpy as np

from channel import _gg_cdf_table
from mpc_loop import KalmanAR1


def build_mnlt(alpha, beta, n=4000, xmax=None):
    """Forward/inverse MNLT maps T^{-1} (physical->latent) and T (latent->
    physical), built on the identical table channel.GammaGammaAR1 uses, so the
    predictor and the channel generator agree on what T is."""
    x, cdf = _gg_cdf_table(alpha, beta, n=n, xmax=xmax)

    def T(g):
        """Latent Gaussian g -> physical h_a = F_GG^{-1}(Phi(g))."""
        from scipy.stats import norm
        u = norm.cdf(np.asarray(g, dtype=float))
        return np.interp(u, cdf, x)

    def T_inv(h):
        """Physical h_a -> latent Gaussian g = Phi^{-1}(F_GG(h))."""
        from scipy.stats import norm
        u = np.interp(np.asarray(h, dtype=float), x, cdf)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return norm.ppf(u)

    return T, T_inv


GH_NODES = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
GH_WEIGHTS = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])


class GaussHermiteKF(KalmanAR1):
    """KalmanAR1 with (i) the update step operating on the true latent g,
    reached via the forward MNLT from a physical h_a measurement, and (ii)
    predict() returning eq. (25)'s 3-point Gauss-Hermite forecast instead of
    the bare latent mean mu_k."""

    def __init__(self, alpha, beta, rho_a=0.98, q=None, r=1e-3):
        super().__init__(rho_a=rho_a, q=q, r=r)
        self.T, self.T_inv = build_mnlt(alpha, beta)

    def update_physical(self, h_a_meas: float) -> float:
        """Accepts a PHYSICAL h_a measurement (unit-mean turbulence factor),
        maps it through the forward MNLT to the latent g, then runs the
        ordinary scalar KF update on g. Returns the updated latent estimate."""
        g_meas = float(self.T_inv(h_a_meas))
        return self.update(g_meas)

    def predict(self, horizon: int) -> np.ndarray:
        """eq. (25): 3-point Gauss-Hermite quadrature through T at each
        horizon step, using sigma_k^2 = 1 - rho^{2k} (1 - P)."""
        out = np.empty(horizon)
        for i, k in enumerate(range(1, horizon + 1)):
            mu_k = self.x * self.rho ** k
            sig_k = np.sqrt(max(0.0, 1.0 - self.rho ** (2 * k) * (1.0 - self.P)))
            nodes = mu_k + sig_k * GH_NODES
            out[i] = float(np.dot(GH_WEIGHTS, self.T(nodes)))
        return out

    def naive_ce(self, horizon: int) -> np.ndarray:
        """The manuscript's own named baseline, T(mu_k) with no dispersion
        correction -- for the three-way comparison against the released
        `1 + mu_k` and the true eq. (25) quadrature above."""
        k = np.arange(1, horizon + 1)
        mu_k = self.x * self.rho ** k
        return self.T(mu_k)


if __name__ == "__main__":
    import time

    np.seterr(all="ignore")
    rng = np.random.default_rng(0)

    for label, alpha, beta in [("weak", 4.2, 3.0), ("moderate", 2.1, 1.5), ("strong", 1.2, 1.1)]:
        rho_a = 0.98
        kf = GaussHermiteKF(alpha, beta, rho_a=rho_a)
        T, T_inv = kf.T, kf.T_inv

        # burn in on a real MNLT channel realisation, exactly as
        # landscape_probe.prime_predictor does, but feeding the PHYSICAL h_a
        # through update_physical (the forward-MNLT-corrected path) instead
        # of h_a - 1 through the raw update().
        from channel import GammaGammaAR1
        ch = GammaGammaAR1(alpha, beta, rho_a=rho_a, seed=20260826, calibrate=False)
        for _ in range(200):
            kf.update_physical(ch.step())

        g_hat = kf.x
        P = kf.P
        rho = kf.rho
        print(f"\n=== {label} (alpha={alpha}, beta={beta}) ===  g_hat={g_hat:.4f}  P={P:.4f}  rho={rho}")

        horizon = 20
        gh_forecast = kf.predict(horizon)
        naive_forecast = kf.naive_ce(horizon)
        linear_forecast = 1.0 + np.array([g_hat * rho ** k for k in range(1, horizon + 1)])

        # independent Monte Carlo ground truth, exactly matching the
        # manuscript's own validation description (150,000 samples), for
        # E[h_a(t+k) | Y_t] at each horizon step.
        n_mc = 150_000
        eps = rng.normal(size=n_mc)
        print(f"{'k':>3} {'mu_k':>8} {'sigma_k':>8} {'MC truth':>10} "
              f"{'GH (eq.25)':>10} {'naive CE':>10} {'released 1+mu_k':>16}  "
              f"{'GH err%':>8} {'CE err%':>8} {'lin err%':>9}")
        errs_gh, errs_ce, errs_lin = [], [], []
        for i, k in enumerate(range(1, horizon + 1)):
            mu_k = g_hat * rho ** k
            sig_k = np.sqrt(max(0.0, 1.0 - rho ** (2 * k) * (1.0 - P)))
            mc_truth = float(np.mean(T(mu_k + sig_k * eps)))
            gh = gh_forecast[i]
            ce = naive_forecast[i]
            lin = linear_forecast[i]
            e_gh = 100 * abs(gh - mc_truth) / mc_truth
            e_ce = 100 * abs(ce - mc_truth) / mc_truth
            e_lin = 100 * abs(lin - mc_truth) / mc_truth
            errs_gh.append(e_gh); errs_ce.append(e_ce); errs_lin.append(e_lin)
            print(f"{k:3d} {mu_k:8.4f} {sig_k:8.4f} {mc_truth:10.4f} "
                  f"{gh:10.4f} {ce:10.4f} {lin:16.4f}  "
                  f"{e_gh:8.3f} {e_ce:8.3f} {e_lin:9.3f}")
        print(f"  worst/mean abs%% error -- GH (eq.25): {max(errs_gh):.3f} / {np.mean(errs_gh):.3f}   "
              f"naive CE: {max(errs_ce):.3f} / {np.mean(errs_ce):.3f}   "
              f"released 1+mu_k: {max(errs_lin):.3f} / {np.mean(errs_lin):.3f}")
