"""
Per-change attribution: what each replacement in `mpc_fast` is worth, alone.

The whole-objective A/B in `compare_objective.py` says how much the wrapper as a
whole got cheaper.  This file says where that came from, by timing each replaced
piece against the piece it replaced, on the same inputs, interleaved, under the
same min-of-block estimator (this machine runs other benchmark processes, so
interference can only add time and the block minimum is the estimator of the
uncontended cost).

Run at both ends of the `rank_stages` range, because the pieces do not scale the
same way: the pointing recursion and the geometry grow with the number of ranked
stages, the slew penalty and the decision-vector handling do not, and the
reference-SNR grouping grows fastest of all.

    python ablate_objective.py
"""
from __future__ import annotations

import time

import numpy as np

from channel import beam_geometry, xi_effective
from hclpso_ga import ladder_order
from mpc_fast import (FastBeamSteeringMPC, _FastObjective,
                      _beam_geometry_fused, _ladder_order_fast)
from mpc_loop import BeamSteeringMPC
from rtodt_fast import z_of

A, B = 1.2, 1.1
SIGMA_S = 0.10
GBAR = 10.0 ** 3.8
HORIZON = 20
NP_SWARM = 30


def tmin(fn, blocks=25, reps=250, warm=600):
    for _ in range(warm):
        fn()
    ms = np.empty(blocks)
    for k in range(blocks):
        best = 1 << 62
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            fn()
            dt = time.perf_counter_ns() - t0
            if dt < best:
                best = dt
        ms[k] = best
    return float(np.median(ms)) / 1000.0


def ab(old, new, **kw):
    """Interleaved so drift lands on both."""
    a1 = tmin(old, **kw)
    b1 = tmin(new, **kw)
    b2 = tmin(new, **kw)
    a2 = tmin(old, **kw)
    return min(a1, a2), min(b1, b2)


def bench(rank_stages):
    m0 = BeamSteeringMPC(A, B, SIGMA_S, GBAR, horizon=HORIZON, seed=7,
                         rank_stages=rank_stages)
    m1 = FastBeamSteeringMPC(A, B, SIGMA_S, GBAR, horizon=HORIZON, seed=7,
                             rank_stages=rank_stages)
    fo = _FastObjective(m1)
    m1._fastobj = fo
    rng = np.random.default_rng(4)
    lo, hi = m0.lower(), m0.upper()
    X = np.ascontiguousarray(m0.repair(lo + rng.random((NP_SWARM, lo.size))
                                       * (hi - lo)))
    st = np.array([1.2e-5, -0.9e-5])
    m0.kf.update(0.03)
    hp = m0.kf.predict(HORIZON)
    m0._objective(X, st, hp)
    m1._objective(X, st, hp)

    n, T = NP_SWARM, HORIZON
    Tr = fo.Tr
    c = fo.cache
    theta0 = m0._as_theta(st, m0.L)
    m0.theta0 = theta0

    w = X[:, :Tr].reshape(-1)
    a0f, weqf = _beam_geometry_fused(w)
    xif = weqf / fo.two_sigma
    rd = fo._stage_rd_fast(X, n, Tr, theta0)
    rflat = np.full(n, rd) if np.ndim(rd) == 0 else rd.reshape(-1)
    xef = xif / np.sqrt(1.0 + rflat * rflat / fo.xie_den)
    a02 = a0f.reshape(n, Tr)
    z2 = fo.z_num / (a02 * c.sqrt_g)
    gb = np.tile(m0._stage_gbar(hp)[:Tr], (n, 1)).reshape(-1)
    uq = np.unique(gb)

    rows = []

    # 1 -- pointing recursion
    rows.append(("_stage_rd: stack/cumsum/tile/linalg.norm over all T"
                 " -> truncated cumsum + sqrt(x*x+y*y) over Tr",
                 *ab(lambda: m0._stage_rd(X)[:, :Tr],
                     lambda: fo._stage_rd_fast(X, n, Tr, theta0))))

    # 2 -- slew penalty and feasibility
    def base_slew():
        pen = np.zeros(n)
        viol = np.zeros(n, dtype=bool)
        for (s, e), lim in zip(m0.blocks(), m0.block_slew()):
            d = np.abs(np.diff(X[:, s:e], axis=1))
            pen = pen + np.sum(d ** 2, axis=1) / max(T - 1, 1)
            viol |= np.any(d > lim, axis=1)
        return pen, viol

    rows.append(("slew penalty + feasibility: 3x np.diff, 3x np.sum, 3x np.any"
                 " -> one differencing pass, 3 slice reductions, one np.any",
                 *ab(base_slew, lambda: fo._slew(X, n, c))))

    # 3 -- per-stage reference SNR
    rows.append(("stage gbar: _stage_gbar + np.tile every call"
                 " -> computed once per control cycle",
                 *ab(lambda: np.tile(m0._stage_gbar(hp)[:Tr], (n, 1)).reshape(-1),
                     lambda: fo._stage_gbar_cached(hp))))

    # 4 -- grouping
    def base_group():
        out = []
        for g in np.unique(gb):
            out.append((float(g), gb == g))
        return out

    rows.append(("SNR grouping: np.unique over the (n*Tr,) array + a boolean"
                 " mask per group -> precomputed stage -> column map",
                 *ab(base_group, lambda: c.groups)))

    # 5 -- geometry
    rows.append(("beam_geometry: erf(v) evaluated twice -> evaluated once",
                 *ab(lambda: beam_geometry(w),
                     lambda: _beam_geometry_fused(w))))

    # 6 -- fidelity ladder
    def base_ladder():
        out = []
        for g in uq:
            mm = gb == g
            out.append(ladder_order(z_of(A, B, a0f[mm], float(g))))
        return out

    rows.append(("z and ladder: z_of + ladder_order once per SNR group"
                 " -> one broadcast z, one searchsorted ladder",
                 *ab(base_ladder,
                     lambda: _ladder_order_fast(fo.z_num / (a02 * c.sqrt_g)))))

    # 7 -- reductions
    d2 = np.abs(X[:, 1:] - X[:, :-1]) ** 2
    rows.append(("reductions: np.sum / np.any -> np.add.reduce /"
                 " np.logical_or.reduce (same ufunc, no Python arg handling)",
                 *ab(lambda: (np.sum(d2[:, 0:T - 1], axis=1),
                              np.sum(d2[:, T:2 * T - 1], axis=1),
                              np.sum(d2[:, 2 * T:3 * T - 1], axis=1),
                              np.any(d2 > 1.0, axis=1)),
                     lambda: (np.add.reduce(d2[:, 0:T - 1], axis=1),
                              np.add.reduce(d2[:, T:2 * T - 1], axis=1),
                              np.add.reduce(d2[:, 2 * T:3 * T - 1], axis=1),
                              np.logical_or.reduce(d2 > 1.0, axis=1)))))

    # 8 -- state handling
    rows.append(("pointing state: _as_theta every call -> cached per cycle",
                 *ab(lambda: m0._as_theta(st, m0.L),
                     lambda: fo._theta0_cached(st))))

    # 9 -- xi and xi_eff
    rows.append(("xi / xi_eff: xi_effective with the 2 sigma_s^2 constant"
                 " rebuilt each call -> folded constant",
                 *ab(lambda: xi_effective(weqf / (2.0 * SIGMA_S), rflat, SIGMA_S),
                     lambda: (weqf / fo.two_sigma)
                     / np.sqrt(1.0 + rflat * rflat / fo.xie_den))))

    return rows


if __name__ == "__main__":
    for rs in (1, None):
        print()
        print("=" * 96)
        print("PER-CHANGE ATTRIBUTION   rank_stages = %s   (N_p = %d, T = %d)"
              % (rs, NP_SWARM, HORIZON))
        print("=" * 96)
        rows = bench(rs)
        tot = 0.0
        for what, old, new in rows:
            tot += old - new
            head, _, tail = what.partition(" -> ")
            print("  %-72s %7.2f -> %6.2f us   saves %6.2f"
                  % (head[:72], old, new, old - new))
            print("      %s" % tail[:88])
        print("  " + "-" * 92)
        print("  %-72s                        total  %6.2f us" % ("", tot))
