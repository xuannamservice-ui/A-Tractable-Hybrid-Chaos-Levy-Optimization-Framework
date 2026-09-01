"""Measure the SECOND killer: the admissibility guard.

The feasibility repair is not the only gate that deletes large excursions.
After repair, every candidate is scored through the RT-ODT kernel with the
fidelity ladder: a candidate whose z = sqrt(2) alpha beta / (A0 sqrt(gbar))
exceeds 8 is inadmissible (K = -1) and scores NaN, and NaN candidates are
rejected by the solver's finiteness test.  For the w_z block, a large jump
broadens the beam, A0 collapses and z blows up -- so a far jump in w_z is
killed by the GUARD even if the repair let it through.

This probe counts, per jump mechanism, the fraction of proposed jumps that
the guard rejects, and the realised stage-0 displacement of the survivors.
"""
from __future__ import annotations

import argparse
import numpy as np

from channel import SwayProcess
from hclpso_ga import levy, ladder_order
from rtodt_fast import z_of
from measure_all import N_P, SIGMAS, _make_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--jump-scale", type=float, default=0.02)
    ap.add_argument("--levy-lambda", type=float, default=1.5)
    a = ap.parse_args()

    rng = np.random.default_rng(20260903)
    per_cell = max(1, a.draws // len(SIGMAS))
    order = [(s, k) for s in SIGMAS for k in range(per_cell)]

    gbar = 10 ** (38.0 / 10.0)
    stats = {}
    for variant in ("per_dim", "feas_shift"):
        for tag in ("levy", "gauss"):
            stats[(variant, tag)] = dict(rejected=0, n=0,
                                         surv_wz=[], surv_s0=[])

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
        slew = m.block_slew()
        base = lo + 0.5 * span
        X = np.tile(base, (N_P, 1))
        # w_z block of this problem (steering=True -> block 0)
        wz_lo, wz_hi = lo[0], hi[0]

        for tag in ("levy", "gauss"):
            use = tag == "levy"
            for variant in ("per_dim", "feas_shift"):
                st = stats[(variant, tag)]
                if variant == "per_dim":
                    steps = (levy(rng, N_P * dim, a.levy_lambda).reshape(N_P, dim)
                             if use else rng.normal(size=(N_P, dim)))
                    Y = X + a.jump_scale * steps * span
                else:
                    Y = X.copy()
                    nb = len(blocks)
                    steps = (levy(rng, N_P * nb, a.levy_lambda).reshape(N_P, nb)
                             if use else rng.normal(size=(N_P, nb)))
                    for bi, ((bs, be), sl) in enumerate(zip(blocks, slew)):
                        spanb = float(np.max(span[bs:be]))
                        Y[:, bs:be] += a.jump_scale * steps[:, bi, None] * spanb
                        Y[:, bs:be] += rng.normal(size=(N_P, be - bs)) * (0.3 * sl)
                Yr = np.asarray(repair(Y.copy()), float)
                A0, _weq = _raw_A0_wz(Yr[:, 0], wz_lo, wz_hi)
                z = z_of(*m_ab(), A0, gbar)
                K = ladder_order(z)
                adm = K >= 0
                st["n"] += int(adm.sum())
                st["rejected"] += int((~adm).sum())
                ok = adm
                if ok.any():
                    st["surv_s0"].extend(np.abs(Yr[ok, 0] - X[ok, 0])
                                         / np.maximum(np.abs(Y[ok, 0] - X[ok, 0]), 1e-300))
                    st["surv_wz"].extend(
                        np.abs(Yr[ok, 0] - X[ok, 0]) / np.maximum(
                            np.abs(Y[ok, 0] - X[ok, 0]), 1e-300))

    print("guard rejection (z > 8 after repair), jump_scale=%.3f lambda=%.1f"
          % (a.jump_scale, a.levy_lambda))
    print("  %-11s %-6s %-10s %-10s %-10s" % ("variant", "arm", "rejected",
                                               "n adm", "rej rate"))
    print("  " + "-" * 50)
    for variant in ("per_dim", "feas_shift"):
        for tag in ("levy", "gauss"):
            st = stats[(variant, tag)]
            rate = st["rejected"] / (st["rejected"] + st["n"]) if st["n"] else 1.0
            print("  %-11s %-6s %-10d %-10d %-10.3f"
                  % (variant, tag, st["rejected"], st["n"], rate))
        print()


def m_ab():
    return (1.2, 1.1)  # strong turbulence


def _raw_A0_wz(wz, wz_lo, wz_hi):
    """A0 for the w_z block without importing the guarded geometry."""
    import numpy as _np
    from channel import _raw_beam_geometry
    A0, weq = _raw_beam_geometry(_np.asarray(wz, float))
    return A0, weq


if __name__ == "__main__":
    main()
