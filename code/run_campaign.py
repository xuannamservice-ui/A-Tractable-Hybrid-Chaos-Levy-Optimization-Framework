"""
Closed-loop campaign driver for the reference implementation.

WHAT THIS IS
    A runnable assembly of the architecture described in the manuscript:
    the MNLT channel of Appendix A, the AR(1)/Kalman predictor of Section IV-B,
    the H-CLPSO-GA solver of Section V, the interpolation-free RT-ODT kernel of
    eq. (21), and the three-part envelope guard of Section VI-C.

WHAT THIS IS NOT
    It is NOT the campaign driver that produced Tables 9-12 of the manuscript.
    That driver is not part of this release. The rates this script reports are
    the reference implementation's own, obtained on its own channel draws with
    its own solver seeds, and they should be read as such -- they characterise
    the algorithm as specified, not the published campaign.

    Concretely: the published optimization-success rate is defined against a
    post-EGC system ABER target, whereas this script scores the per-branch
    surrogate the solver actually ranks by. The two are different quantities
    and their numerical values are not comparable.

WHAT IT IS USEFUL FOR
    (a) exercising the algorithm end to end;
    (b) A/B experiments in which everything is held fixed except one component,
        which is how the ablation and guard comparisons below are constructed.

Usage:
    python run_campaign.py [--realizations 300] [--out ../data/07_reference_campaign]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from channel import beam_geometry, xi_floor, SwayProcess, GammaGammaAR1
from hclpso_ga import SolverConfig
from mpc_loop import BeamSteeringMPC, envelope_guard
from rtodt_fast import pe_series_f64, z_of

REGIMES = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
SIGMAS = [0.05, 0.1, 0.2, 0.3]
GBAR_DB = 38.0

ABLATIONS = {
    "full":            dict(),
    "no_chaotic_init": dict(use_chaos=False),
    "no_levy_flight":  dict(use_levy=False),
    "no_ga_refinement": dict(use_ga=False),
    "fixed_fidelity":  dict(use_fidelity_ladder=False, fixed_order=10),
}


def true_quality(alpha, beta, w_z, sigma_s, gbar):
    """Per-branch ABER of a selected beam, evaluated exactly (no guard)."""
    if w_z is None:
        return np.nan
    A0, w_zeq = beam_geometry(np.array([float(w_z)]))
    xi = w_zeq / (2.0 * sigma_s)
    z = z_of(alpha, beta, A0, gbar)
    K = 20 if z[0] <= 8 else -1
    v = pe_series_f64(alpha, beta, xi, A0, gbar, K)
    return float(v[0])


def run(n_real: int, out_dir: str, seed0: int = 20260826):
    os.makedirs(out_dir, exist_ok=True)
    gbar = 10 ** (GBAR_DB / 10)
    rows, guard_rows = [], []
    t0 = time.time()

    for regime, (A, B) in REGIMES.items():
        for name, over in ABLATIONS.items():
            for three_part in (True, False):
                cfg = SolverConfig(**over)
                sel, rej, invalid = [], 0, 0
                for r in range(n_real):
                    rng = np.random.default_rng(seed0 + r)
                    sigma_s = SIGMAS[rng.integers(len(SIGMAS))]
                    sway = SwayProcess(sigma_s, seed=seed0 + r)
                    for _ in range(5):                 # let the sway settle
                        sway.step()
                    r_d = sway.radial()

                    mpc = BeamSteeringMPC(A, B, sigma_s, gbar, seed=seed0 + r,
                                          three_part_guard=three_part, config=cfg)
                    res = mpc.step(r_d)
                    rej += res.rejected_by_guard
                    q = true_quality(A, B, res.best_x[0] if res.best_x is not None else None,
                                     sigma_s, gbar)
                    sel.append(q)
                    if np.isfinite(res.best_f) and not (0.0 <= res.best_f <= 0.5):
                        invalid += 1

                sel = np.array(sel, dtype=float)
                ok = np.isfinite(sel)
                rows.append(dict(
                    regime=regime, variant=name,
                    guard="three_part" if three_part else "threshold_only",
                    n=n_real,
                    median_selected_aber=float(np.nanmedian(sel[ok])) if ok.any() else None,
                    best_selected_aber=float(np.nanmin(sel[ok])) if ok.any() else None,
                    cycles_with_invalid_optimum=invalid,
                    pct_invalid=100.0 * invalid / n_real,
                    candidates_rejected_by_guard=int(rej)))
                print("  %-9s %-17s %-14s median %.4e  invalid %.1f%%  rejected %d"
                      % (regime, name, "3-part" if three_part else "threshold",
                         rows[-1]["median_selected_aber"] or float("nan"),
                         rows[-1]["pct_invalid"], rej), flush=True)

    with open(os.path.join(out_dir, "reference_campaign.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            note=("Reference-implementation output. NOT a reproduction of Tables 9-12. "
                  "The published rates are defined against a post-EGC system target, "
                  "these against the per-branch surrogate the solver ranks by, so the "
                  "values are not comparable. Note also that this implementation does "
                  "NOT reproduce the ablation ordering of Table 11: several components "
                  "make little difference here, and removing chaotic initialisation can "
                  "even help. That is most likely a property of the choices this "
                  "implementation had to make where the paper does not specify them "
                  "(slew limit, penalty weight lambda_u, decision box, stage weighting) "
                  "rather than evidence about the published campaign."),
            realizations=n_real, gbar_db=GBAR_DB, sigma_s_swept=SIGMAS,
            seconds=round(time.time() - t0, 1), results=rows), f, indent=2)

    hdr = list(rows[0].keys())
    with open(os.path.join(out_dir, "reference_campaign.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in hdr) + "\n")
    print("\n  wrote %s  (%.1f s)" % (out_dir, time.time() - t0))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "data", "07_reference_campaign"))
    a = ap.parse_args()
    print("Reference-implementation campaign: %d realizations per cell\n" % a.realizations)
    run(a.realizations, os.path.abspath(a.out))
