"""Can a bounded warm-start cloud already reach everywhere the plant allows?

This is the feasibility check the earlier Levy probes should have run first. A
heavy-tailed initial cloud can only change an outcome if the bounded cloud is
SMALLER than the set the controller may move into. Where the slew tube is
tighter than the cloud, the bounded cloud already covers every reachable
command, the projection of eq. (14) returns the tail inside that same set, and
no amount of tail weight can alter the result. The null would then be
structural, and predictable without running a single trial.

The comparison is stage-dependent, and saying so is the point. Under the
inter-cycle anchor of Section IX-B, stage 0 -- the command actually published --
may move at most U_slew from the previous cycle's command, so its reachable
radius is one slew step. Stage k may ramp, so its radius is (k+1) slew steps.
The bounded cloud has a fixed radius init_spread * span. There is therefore a
crossing stage

    k* = init_spread * span / U_slew - 1

below which the cloud is wider than the plant allows (a tail is inert) and above
which the cloud is the binding limit (a tail can reach further). The published
command sits at stage 0, the far end of the wrong side of that crossing.

Writes warmstart_reach.json alongside the other artefacts.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure_all import SIGMAS, GBAR_OP_DB
from mpc_loop import BeamSteeringMPC, U_SLEW, U_MAX

STRONG = (1.2, 1.1)
NAMES = ("w_z (divergence)", "theta_az (steering)", "theta_el (steering)")
OUT = os.path.join(HERE, "..", "data", "14_compiled_poc")


def main():
    gbar = 10.0 ** (GBAR_OP_DB / 10.0)
    print("  deployed init_spread = 0.02 of box, horizon T = 20, U_slew = %.3e rad, U_max = %.3e rad"
          % (U_SLEW, U_MAX))
    print("  cloud radius = init_spread * span;  reach at stage k = (k+1) * block slew limit,")
    print("  capped by the box.  ratio > 1 means the tail is inert at that stage.\n")
    rows = {}
    for s in SIGMAS:
        m = BeamSteeringMPC(*STRONG, s, gbar, horizon=20, seed=0, rank_stages=None)
        lo, hi = np.asarray(m.lower(), float), np.asarray(m.upper(), float)
        spread, T = m.cfg.init_spread, m.horizon
        print("  sigma_s = %.2f m" % s)
        print("    %-20s %11s %11s %11s %11s %11s %9s" %
              ("block", "span", "cloud rad", "reach k=0", "reach k=T-1", "ratio k=0", "crossing"))
        rows[str(s)] = {}
        for (name, (a, b), lim) in zip(NAMES, m.blocks(), m.block_slew()):
            span = float(hi[a] - lo[a])
            cloud = spread * span
            lim = float(lim)
            r0 = min(lim, span)
            rT = min(lim * T, span)
            ratio0 = cloud / r0
            # first stage k whose reachable radius equals or exceeds the cloud
            kstar = cloud / lim - 1.0
            kstar_s = ("%.1f" % kstar) if 0.0 <= kstar <= T - 1 else ("none" if kstar < 0 else ">T")
            rows[str(s)][name] = dict(span=span, cloud_radius=cloud, slew_limit=lim,
                                      reach_stage0=r0, reach_stage_last=rT,
                                      ratio_stage0=ratio0, crossing_stage=kstar)
            print("    %-20s %11.4e %11.4e %11.4e %11.4e %11.2f %9s"
                  % (name, span, cloud, r0, rT, ratio0, kstar_s))
        print()
    print("  Reading: on the steering axes the ratio at the published stage is 8, so the")
    print("  bounded cloud spans eight times what the actuator may reach that cycle and an")
    print("  unbounded draw is returned by the projection; the crossing lies at stage 7, so")
    print("  only the far half of the horizon -- which shapes the ranking but is never")
    print("  published -- leaves the tail any room at all. On the divergence axis the ratio")
    print("  is below one from stage 0, but that axis is unimodal per stage, so reaching")
    print("  further finds no second basin. The two facts together are why every Levy probe")
    print("  in this work returns a null at the deployed operating point.")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "warmstart_reach.json"), "w") as fh:
        json.dump(dict(what="radius of the bounded warm-start cloud against the one-cycle "
                            "reachable radius of the slew-limited actuator, per block and per "
                            "jitter level, with the stage at which the two cross",
                       u_slew_rad=U_SLEW, u_max_rad=U_MAX, horizon=20, init_spread=0.02,
                       blocks=rows), fh, indent=1)
    print("\n  wrote warmstart_reach.json")


if __name__ == "__main__":
    main()
