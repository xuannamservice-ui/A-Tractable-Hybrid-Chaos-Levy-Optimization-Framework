"""Which jitter belongs in the pointing law once the loop is closed?

Section II-B gives the pointing-loss density in terms of sigma_s, the RMS
displacement about the COMMANDED direction, and folds any mean offset in
separately as r_d through eq. (15).  Section II-B also drives the sway process
with the same numerical sigma_s as its stationary amplitude.  Those are two
different quantities, and the steering loop is what separates them: it tracks
the sway, so the displacement that remains about boresight is not the
stationary sway amplitude but the residual the tracker settles to.

This script measures the residual with the loop's own sway process and its own
proportional law, and then scores the nominal cell both ways.  It is short
because the point is arithmetic, not modelling: the same disturbance cannot be
charged once to the jitter the beam must tolerate and again to the offset the
mirror has already removed.

Usage:  python jitter_accounting_probe.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from channel import SwayProcess
from system_metric import BeamConfig, aber_of

L = 2000.0
U_SLEW = 50e-3 * 1e-3
U_MAX = 10e-3
GAIN = 0.8
F_C, T_U = 1.0, 1e-3
SIGMAS = (0.05, 0.10, 0.20, 0.30)
GBAR_DB = 38.0
W_OPT = 0.192


def tracked_residual(sigma_s, cycles=400, burn=100, seed=11):
    """Median |Theta| the released proportional law settles to, in metres at L."""
    sig_delta = (sigma_s / L) * np.sqrt(1.0 - np.exp(-4.0 * np.pi * F_C * T_U))
    rng = np.random.default_rng(seed)
    theta = np.zeros(2)
    u_prev = np.zeros(2)
    rs = []
    for c in range(cycles):
        u = np.clip(GAIN * theta, -U_MAX, U_MAX)
        u = np.clip(u, u_prev - U_SLEW, u_prev + U_SLEW)
        u_prev = u
        theta = theta - u + rng.normal(0.0, sig_delta, 2)
        if c >= burn:
            rs.append(float(np.linalg.norm(theta)))
    return float(np.median(rs))


def main():
    out = {"generated_by": "code/jitter_accounting_probe.py", "gbar_db": GBAR_DB,
           "w_z": W_OPT, "cells": []}
    print(f"{'sigma_s':>8} {'stationary':>12} {'tracked resid':>14} "
          f"{'ABER as scored':>16} {'ABER coherent':>16}")
    for s in SIGMAS:
        r = tracked_residual(s)
        r_m = r * L
        as_scored = aber_of(BeamConfig("strong", W_OPT, s, r_m), GBAR_DB)
        coherent = aber_of(BeamConfig("strong", W_OPT, max(r_m, 1e-4), 0.0), GBAR_DB)
        out["cells"].append({"sigma_s": s, "stationary_urad": 1e6 * s / L,
                             "tracked_residual_urad": 1e6 * r,
                             "tracked_residual_m": r_m,
                             "aber_as_scored": as_scored,
                             "aber_coherent": coherent})
        print(f"{s:>8.2f} {1e6 * s / L:>10.1f}ur {1e6 * r:>12.2f}ur "
              f"{as_scored:>16.4e} {coherent:>16.4e}")

    print("\nThe released scorer charges the stationary sway as the jitter the beam "
          "must\ntolerate AND the tracker's residual as the offset, so the same "
          "disturbance is\ncounted twice. The tracked residual reproduces the "
          "6.5 us median the closed-loop\ncampaign reports, which is what makes it "
          "the right quantity for eq. (4).")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "23_jitter_accounting")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "jitter_accounting.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
