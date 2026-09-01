"""Probe: can a heavy-tailed jump survive the slew projection at all?

The deployed jump adds an i.i.d. Levy/Gaussian step to EVERY stage of the
60-D trajectory (hclpso_ga.py, L239-243), and the forward-sweep rate-limiter
(repair, mpc_loop.py L282-301) then pulls each stage back toward its
predecessor. A forward sweep is a low-pass filter: it keeps the low-frequency
content of the jump (stage-0 move + slow ramp) and deletes the high-frequency
content -- and the Levy tail lives precisely in the high frequencies.

The slew-feasible directions that carry a trajectory FAR are the block-wise
common shifts (constant over all T stages: stage-to-stage differences are
unchanged, so the slew tube is preserved by construction, and only the box
clips).  This probe measures, for both mechanisms:

    survival ratio       ||x_repaired - x_before|| / ||x_proposed - x_before||
    stage-0 survival     |(x_repaired - x_before)[0]| / |(x_proposed - x_before)[0]|
                         -- stage 0 is the published command (rank_stages=1),
                         so this is the displacement that actually matters
    p99 tail ratio       Levy/Gauss of proposed and of realised displacement

Variant "per_dim"   : the deployed mechanism (i.i.d. per stage + repair)
Variant "feas_shift": common heavy-tailed shift per block + small per-stage
                       jitter at 0.3 x slew limit (feasible by construction),
                       then box clip + repair (repair now has nothing to cut)

Diagnostic only -- nothing here is a performance claim.
"""
from __future__ import annotations

import argparse
import numpy as np

from channel import SwayProcess
from hclpso_ga import levy
from measure_all import N_P, SIGMAS, _make_problem


def repair_then(X, repair):
    X = np.clip(X, 0.0, 1.0) if repair is None else np.asarray(repair(X), float)
    return X


def jump_per_dim(x, rng, n_jump, dim, scale, lam, use_levy, span):
    steps = (levy(rng, n_jump * dim, lam).reshape(n_jump, dim) if use_levy
             else rng.normal(size=(n_jump, dim)))
    return x + scale * steps * span


def jump_feas_shift(x, rng, n_jump, blocks, scale, lam, use_levy, span, slew):
    """Common heavy-tailed shift per block + small feasible per-stage jitter.

    The shift is one scalar per physical block, applied to all T stages of
    that block: stage-to-stage differences are unchanged, so the slew tube is
    preserved exactly; only the box can clip.  The jitter is drawn at 0.3x the
    per-stage slew limit, so it is feasible with overwhelming probability and
    the forward-sweep has (almost) nothing to repair.
    """
    out = x.copy()
    for j in range(n_jump):
        for (s, e) in blocks:
            nb = e - s
            spanb = float(np.max(span[s:e]))
            if use_levy:
                d = levy(rng, 1, lam)[0]
            else:
                d = rng.normal()
            shift = scale * d * spanb
            out[j, s:e] += shift
            # small jitter, bounded by the block's slew limit
            jit = rng.normal(size=nb) * (0.3 * slew[(s, e)])
            out[j, s:e] += jit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--jump-scale", type=float, default=0.02)
    ap.add_argument("--levy-lambda", type=float, default=1.5)
    a = ap.parse_args()

    rng = np.random.default_rng(20260901)
    per_cell = max(1, a.draws // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]

    stats = {v: {"levy": {"prop": [], "real": [], "s0prop": [], "s0real": []},
                 "gauss": {"prop": [], "real": [], "s0prop": [], "s0real": []}}
             for v in ("per_dim", "feas_shift")}

    for s, k in order:
        seed = 700000 + int(s * 1000) * 1000 + k
        sway = SwayProcess(s, seed=seed)
        for _ in range(5):
            sway.step()
        r_d = sway.radial()
        m, f, lo, hi, blocks, repair = _make_problem(s, r_d, seed)
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        span = hi - lo
        dim = lo.size
        slew = {b: (m.block_slew()[i]) for i, b in enumerate(blocks)}
        base = lo + 0.5 * span
        X = np.tile(base, (N_P, 1))

        for tag in ("levy", "gauss"):
            use = tag == "levy"
            for variant in ("per_dim", "feas_shift"):
                Y = (jump_per_dim(X, rng, N_P, dim, a.jump_scale, a.levy_lambda,
                                  use, span)
                     if variant == "per_dim" else
                     jump_feas_shift(X, rng, N_P, blocks, a.jump_scale,
                                     a.levy_lambda, use, span, slew))
                Yr = repair_then(Y.copy(), repair)
                d0 = np.linalg.norm(Y - X, axis=1)
                d1 = np.linalg.norm(Yr - X, axis=1)
                s0p = np.abs(Y - X)[:, 0]
                s0r = np.abs(Yr - X)[:, 0]
                st = stats[variant][tag]
                st["prop"].append(d0); st["real"].append(d1)
                st["s0prop"].append(s0p); st["s0real"].append(s0r)

    print("jump_scale=%.3f  lambda=%.1f  draws=%d x N_p=%d"
          % (a.jump_scale, a.levy_lambda, len(order), N_P))
    print("\n%-11s %-6s %-8s %-10s %-10s %-10s %-10s"
          % ("variant", "arm", "surv", "s0 surv", "prop p99", "real p99",
             "real/prop"))
    print("-" * 78)
    for variant in ("per_dim", "feas_shift"):
        L, G = stats[variant]["levy"], stats[variant]["gauss"]
        for tag, st in (("levy", L), ("gauss", G)):
            P = np.concatenate(st["prop"]); R = np.concatenate(st["real"])
            S0P = np.concatenate(st["s0prop"]); S0R = np.concatenate(st["s0real"])
            ok = np.isfinite(P) & np.isfinite(R) & (P > 0)
            surv = np.median(R[ok] / P[ok])
            ok0 = np.isfinite(S0P) & np.isfinite(S0R) & (S0P > 0)
            s0 = np.median(S0R[ok0] / S0P[ok0])
            q = lambda v: float(np.quantile(v, 0.99))
            print("%-11s %-6s %-8.3f %-10.3f %-10.4g %-10.4g %-10.3f"
                  % (variant, tag, surv, s0, q(P), q(R),
                     q(R) / q(P) if q(P) else float("nan")))
        pr = q(np.concatenate(L["prop"])) / q(np.concatenate(G["prop"]))
        rr = q(np.concatenate(L["real"])) / q(np.concatenate(G["real"]))
        s0pr = (q(np.concatenate(L["s0prop"]))
                / q(np.concatenate(G["s0prop"])))
        s0rr = (q(np.concatenate(L["s0real"]))
                / q(np.concatenate(G["s0real"])))
        print("%-11s %-6s %-8s %-10s %-10.2f %-10.2f   (p99 L/G)"
              % (variant, "ratio", "", "", pr, rr))
        print("%-11s %-6s %-8s %-10s %-10.2f %-10.2f   (stage-0 p99 L/G)"
              % (variant, "ratio", "", "", s0pr, s0rr))
        print()


if __name__ == "__main__":
    main()
