"""Does a runtime pole-clearance test close the near-pole hazard, and what
does it cost?

`cancelled_pole_probe.py` shows the residual near a CANCELLED pole scales as
1/distance rather than sitting on a float64 floor, so no fixed bound covers it,
and close enough in the wrong value still passes the range test 0 <= Pe <= 1/2.
`d_pole_probe.py` shows the UNCANCELLED poles (k > K) reach the admitted region
too. `rtodt_fast.pe_series_f64(..., pole_clearance=delta)` screens both by
returning NaN inside delta of any pole, which the existing guard then rejects.

Three things are measured here:

  1. COVERAGE -- walking xi^2 toward a pole, the widest offset at which the
     float64 value is wrong by more than eps_req AND still inside [0, 1/2],
     with the guard off and with it on.
  2. FIRING RATE -- how often the guard would reject over the same uniform
     off-grid draw the manuscript's certification campaign uses, so the cost in
     admitted candidates is stated rather than assumed.
  3. COST -- per-candidate overhead at the deployed swarm size.

Usage:  python pole_guard_probe.py [--clearance 1e-6] [--draws 200000]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mpmath as mp
import numpy as np

from channel import beam_geometry
from hclpso_ga import ladder_order
from mpc_loop import manuscript_wz_box
from rtodt import Pe_series
from rtodt_fast import pe_series_f64, pole_distance, z_of

REGIMES = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
EPS_REQ = 1e-6


def coverage(alpha, beta, k_pole, sigma_s, gbar_db, clearance):
    """Widest offset from the pole at which the kernel is wrong and undetected."""
    from rtodt import wz_for_xi, A0_of
    g = 10.0 ** (gbar_db / 10.0)
    pole = float(beta) + k_pole
    worst_off, worst_on = 0.0, 0.0
    for e in np.arange(-12.0, -0.4, 0.25):
        d = 10.0 ** e
        for sgn in (-1.0, +1.0):
            xi = float(np.sqrt(max(pole + sgn * d, 1e-12)))
            wz = wz_for_xi(mp.mpf(str(xi)), mp.mpf(str(sigma_s)))
            if wz is None:
                continue
            A0 = float(A0_of(wz))
            K = int(np.atleast_1d(ladder_order(np.atleast_1d(z_of(alpha, beta, A0, g))))[0])
            if K < 0:
                continue
            ref = float(Pe_series(*(mp.mpf(str(v)) for v in (alpha, beta, xi, A0, g)), K))
            for clr, tag in ((0.0, "off"), (clearance, "on")):
                v = float(np.atleast_1d(pe_series_f64(
                    alpha, beta, np.atleast_1d(xi), np.atleast_1d(A0), g, K,
                    pole_clearance=clr))[0])
                undetected = np.isfinite(v) and 0.0 <= v <= 0.5 and abs(v - ref) > EPS_REQ
                if undetected:
                    if tag == "off":
                        worst_off = max(worst_off, d)
                    else:
                        worst_on = max(worst_on, d)
    return pole, worst_off, worst_on


def firing_rate(clearance, draws, seed=20260918):
    """Fraction of uniform off-grid draws the clearance test would reject,
    over the same (regime, sigma_s, gbar) sweep the certification campaign uses."""
    rng = np.random.default_rng(seed)
    n_total = n_fire = 0
    per = {}
    for name, (alpha, beta) in REGIMES.items():
        for sigma_s in (0.05, 0.10, 0.20, 0.30):
            lo, hi = manuscript_wz_box(sigma_s)
            wz = float(lo) + rng.random(draws) * (float(hi) - float(lo))
            A0, weq = beam_geometry(wz)
            xi = np.asarray(weq, dtype=float) / (2.0 * sigma_s)
            for gdb in (30, 34, 38, 40, 44):
                g = 10.0 ** (gdb / 10.0)
                K = ladder_order(z_of(alpha, beta, np.asarray(A0, float), g))
                adm = K > 0
                if not adm.any():
                    continue
                d = pole_distance(alpha, beta, xi[adm], int(K[adm].max()))
                n_total += int(adm.sum())
                n_fire += int((d < clearance).sum())
            per[f"{name}_{sigma_s:.2f}"] = None
    return n_total, n_fire


def cost(clearance, n_p=30, reps=4000):
    """Two placements: inside the fitness evaluation (every candidate, every
    iteration) versus inside the envelope guard (the published command only)."""
    from mpc_loop import envelope_guard
    alpha, beta = REGIMES["strong"]
    g = 10.0 ** 3.8
    lo, hi = manuscript_wz_box(0.10)
    wz = np.linspace(float(lo), float(hi), n_p)
    A0, weq = beam_geometry(wz)
    xi = np.asarray(weq, float) / 0.2
    A0 = np.asarray(A0, float)

    out = {}
    for clr, tag in ((0.0, "in_loop_off"), (clearance, "in_loop_on")):
        best = np.inf
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            pe_series_f64(alpha, beta, xi, A0, g, 10, pole_clearance=clr)
            best = min(best, (time.perf_counter_ns() - t0) / 1e3 / n_p)
        out[tag] = best

    one_xi, one_z, one_pe = xi[:1], np.array([0.77]), np.array([0.21])
    b_off = b_on = np.inf
    for _ in range(reps * 4):
        t0 = time.perf_counter_ns()
        envelope_guard(one_z, one_pe, three_part=True)
        b_off = min(b_off, (time.perf_counter_ns() - t0) / 1e3)
        t0 = time.perf_counter_ns()
        d = pole_distance(alpha, beta, one_xi, 10)
        envelope_guard(one_z, one_pe, three_part=True, pole_d=d,
                       pole_clearance=clearance)
        b_on = min(b_on, (time.perf_counter_ns() - t0) / 1e3)
    out["guard_off_us_per_cycle"] = b_off
    out["guard_on_us_per_cycle"] = b_on
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clearance", type=float, default=1e-6)
    ap.add_argument("--draws", type=int, default=200000)
    args = ap.parse_args()

    o = {"generated_by": "code/pole_guard_probe.py", "clearance": args.clearance,
         "eps_req": EPS_REQ, "coverage": [], }

    print(f"clearance delta = {args.clearance:g} in xi^2 units\n")
    print("(1) widest offset at which the value is wrong AND passes the range test")
    print(f"{'regime':>9} {'pole xi^2':>10} {'guard off':>12} {'guard on':>12}")
    for name, (alpha, beta) in REGIMES.items():
        pole, off, on = coverage(alpha, beta, 4, 0.05, 30.0, args.clearance)
        o["coverage"].append({"regime": name, "pole_xi2": pole,
                              "widest_undetected_off": off, "widest_undetected_on": on})
        print(f"{name:>9} {pole:>10.3f} {off:>12.2e} {on:>12.2e}")

    print("\n(2) firing rate over the certification sweep")
    n_total, n_fire = firing_rate(args.clearance, args.draws)
    o["firing"] = {"admitted_draws": n_total, "rejected": n_fire,
                   "rate": n_fire / n_total if n_total else float("nan")}
    print(f"    {n_fire}/{n_total} admitted draws rejected "
          f"({100.0 * n_fire / max(n_total, 1):.2e}%)")

    print("\n(3) cost of the two placements")
    c = cost(args.clearance)
    o["cost"] = c
    print(f"    in fitness  {c['in_loop_off']:.4f} -> {c['in_loop_on']:.4f} us/candidate "
          f"({100.0 * (c['in_loop_on'] / c['in_loop_off'] - 1.0):+.1f}%)")
    print(f"    in guard    {c['guard_off_us_per_cycle']:.4f} -> "
          f"{c['guard_on_us_per_cycle']:.4f} us/published command "
          f"(+{c['guard_on_us_per_cycle'] - c['guard_off_us_per_cycle']:.1f} us/cycle)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "18_cancelled_pole")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pole_guard.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
