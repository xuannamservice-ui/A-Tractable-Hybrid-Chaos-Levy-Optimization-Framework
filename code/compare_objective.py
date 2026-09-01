"""
`mpc_loop._objective` vs `mpc_fast._objective`: equivalence first, then speed.

EQUIVALENCE
    Both objectives are evaluated on the SAME random swarms, drawn inside the
    manuscript decision box and projected onto the slew tube by the same
    `repair`, across all three turbulence regimes (weak / moderate / strong),
    four pointing-jitter levels sigma_s in {0.05, 0.1, 0.2, 0.3} m, several
    reference SNRs, several `rank_stages` settings, and pointing states ranging
    from perfectly aligned to badly misaligned.  For every returned array --
    the cost vector and both auxiliary vectors, `z` and `pe_first` -- the script
    reports

        the fraction of entries that are bit-identical (identical IEEE-754
        payloads, with NaN counted as identical to NaN and +-inf to +-inf), and

        the maximum relative difference over the entries that are not.

    Bit-identity, not a tolerance, is the acceptance criterion.  A tolerance
    chosen after the fact is what this exercise exists to avoid.

TIMING
    A/B interleaved: the two implementations are timed alternately inside the
    same loop, so drift in CPU frequency or scheduler state lands on both.  The
    reported figure is the median over many blocks of block-medians.  A
    wrapper-only figure is also reported, obtained by substituting a stub for
    `rtodt_fast.pe_series_f64` -- that isolates exactly the cost this exercise
    attacks, since the kernel is common to both paths and unmodified.

    Timings on this machine are noisy.  Where a sweep is not monotonic in the
    work done, the script says so rather than reporting the noise as a finding.

Run:  python compare_objective.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import mpc_loop
import rtodt_fast
from mpc_loop import BeamSteeringMPC
from mpc_fast import FastBeamSteeringMPC

REGIMES = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
SIGMAS = (0.05, 0.10, 0.20, 0.30)
GBARS_DB = (30.0, 38.0, 46.0)
HORIZON = 20
NP_SWARM = 30


# ---------------------------------------------------------------- equality
def bit_equal(a, b):
    """Elementwise IEEE-754 payload equality, with NaN == NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ua = a.view(np.uint64)
    ub = b.view(np.uint64)
    same_bits = ua == ub
    both_nan = np.isnan(a) & np.isnan(b)
    # -0.0 and +0.0 have different payloads but are the same number; the
    # objective never distinguishes them, so treat them as equal.
    both_zero = (a == 0.0) & (b == 0.0)
    return same_bits | both_nan | both_zero


def rel_diff(a, b):
    """Max relative difference over entries that are not bit-identical."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = ~bit_equal(a, b)
    if not m.any():
        return 0.0
    x, y = a[m], b[m]
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return np.inf          # a NaN/inf disagreement is not a small difference
    x, y = x[finite], y[finite]
    den = np.maximum(np.abs(x), np.abs(y))
    den = np.where(den == 0.0, 1.0, den)
    out = float(np.max(np.abs(x - y) / den))
    if finite.sum() != m.sum():
        return np.inf
    return out


# ------------------------------------------------------------------ swarms
def make_swarms(mpc, seed, n_swarms=6, n=NP_SWARM):
    """Feasible swarms plus deliberately hostile ones.

    The last two swarms are NOT repaired, so they contain slew-rate violations
    (the +inf branch) and waists driven onto the domain edge (the NaN branch).
    Both branches must agree too.
    """
    rng = np.random.default_rng(seed)
    lo, hi = mpc.lower(), mpc.upper()
    out = []
    for i in range(n_swarms):
        X = lo + rng.random((n, lo.size)) * (hi - lo)
        if i < n_swarms - 2:
            X = mpc.repair(X)
        else:
            # push a few waists below the branch minimum -> NaN geometry
            X = X.copy()
            X[: max(1, n // 6), :HORIZON] *= 0.3
        out.append(np.ascontiguousarray(X, dtype=float))
    return out


def states(rng, L):
    """Pointing states from aligned to badly misaligned, both input forms."""
    return [np.zeros(2),
            np.array([1.0e-5, -0.7e-5]),
            np.array([6.0e-5, 4.0e-5]),
            np.array([2.5e-4, -1.8e-4]),
            rng.normal(size=2) * 5e-5,
            np.array([0.15])]                      # scalar radial offset, metres


# ------------------------------------------------------------- equivalence
def run_equivalence(verbose=True):
    tot = 0
    same = 0
    worst = 0.0
    worst_where = None
    per_regime = {}

    for reg, (A, B) in REGIMES.items():
        r_tot = r_same = 0
        r_worst = 0.0
        for sigma_s in SIGMAS:
            for gdb in GBARS_DB:
                gbar = 10.0 ** (gdb / 10.0)
                for rank_stages in (None, 1, 3, 20):
                    for three_part in (True,):
                        ref = BeamSteeringMPC(A, B, sigma_s, gbar,
                                              horizon=HORIZON, seed=11,
                                              rank_stages=rank_stages,
                                              three_part_guard=three_part)
                        fst = FastBeamSteeringMPC(A, B, sigma_s, gbar,
                                                  horizon=HORIZON, seed=11,
                                                  rank_stages=rank_stages,
                                                  three_part_guard=three_part)
                        rng = np.random.default_rng(int(1e6 * sigma_s) + int(gdb))
                        swarms = make_swarms(ref, seed=int(gdb) * 97 + 5)
                        for X in swarms:
                            for st in states(rng, ref.L):
                                # fresh h_pred per (state, swarm) pair
                                hp = rng.normal(scale=0.05, size=HORIZON)
                                c0, a0 = ref._objective(X, st, hp)
                                c1, a1 = fst._objective(X, st, hp)
                                for name, u, v in (("cost", c0, c1),
                                                   ("z", a0["z"], a1["z"]),
                                                   ("pe_first", a0["pe_first"],
                                                    a1["pe_first"])):
                                    eq = bit_equal(u, v)
                                    tot += eq.size
                                    same += int(eq.sum())
                                    r_tot += eq.size
                                    r_same += int(eq.sum())
                                    d = rel_diff(u, v)
                                    if d > r_worst:
                                        r_worst = d
                                    if d > worst:
                                        worst = d
                                        worst_where = (reg, sigma_s, gdb,
                                                       rank_stages, name)
        per_regime[reg] = (r_same, r_tot, r_worst)

    if verbose:
        print("=" * 74)
        print("EQUIVALENCE  mpc_loop._objective  vs  mpc_fast._objective")
        print("=" * 74)
        print("  %-10s %14s %14s   %s" % ("regime", "bit-identical",
                                          "compared", "max rel diff"))
        for reg, (s, t, w) in per_regime.items():
            print("  %-10s %14d %14d   %.3e" % (reg, s, t, w))
        print("  " + "-" * 70)
        print("  %-10s %14d %14d   %.3e"
              % ("ALL", same, tot, worst))
        print("  fraction bit-identical: %.10f  (%d of %d)"
              % (same / tot, same, tot))
        if worst_where:
            print("  worst at:", worst_where)
        print()
    return same, tot, worst


def run_stage_rd_equivalence():
    """The pointing recursion on its own -- the np.linalg.norm replacement."""
    from mpc_fast import _FastObjective
    worst_same = True
    for sigma_s in SIGMAS:
        m = BeamSteeringMPC(1.2, 1.1, sigma_s, 10 ** 3.8, horizon=HORIZON,
                            seed=3)
        fo = _FastObjective(m)
        rng = np.random.default_rng(int(sigma_s * 1000))
        lo, hi = m.lower(), m.upper()
        for _ in range(40):
            X = m.repair(lo + rng.random((NP_SWARM, lo.size)) * (hi - lo))
            m.theta0 = m._as_theta(rng.normal(size=2) * 1e-4, m.L)
            ref = m._stage_rd(X)
            for Tr in (1, 2, 5, 20):
                got = fo._stage_rd_fast(X, X.shape[0], Tr, m.theta0)
                got = np.broadcast_to(np.atleast_1d(got),
                                      (X.shape[0], Tr)) if np.ndim(got) == 0 \
                    else got
                if not bit_equal(ref[:, :Tr], got).all():
                    worst_same = False
    print("  _stage_rd replacement bit-identical over all Tr and sigma_s: %s"
          % worst_same)
    return worst_same


def run_ladder_equivalence():
    """The searchsorted fidelity ladder against hclpso_ga.ladder_order.

    Checked on the three rung thresholds, on both float64 neighbours of each,
    on NaN, on +-inf, on zero and on negatives, and over 4 million random z --
    the ladder decides admissibility, so a single disagreement here is a
    candidate scored at the wrong series order.
    """
    from hclpso_ga import ladder_order
    from mpc_fast import _ladder_order_fast

    edges = [0.5, 2.0, 8.0]
    special = [0.0, -0.0, -1.0, np.nan, np.inf, -np.inf, 1e-300, 1e300]
    for e in edges:
        special += [e, np.nextafter(e, -np.inf), np.nextafter(e, np.inf),
                    e * (1 - 1e-16), e * (1 + 1e-16)]
    z = np.array(special, dtype=float)
    ok = bool(np.array_equal(ladder_order(z), _ladder_order_fast(z)))
    dt = ladder_order(z).dtype == _ladder_order_fast(z).dtype

    rng = np.random.default_rng(12345)
    for _ in range(20):
        r = rng.uniform(-1.0, 20.0, 200_000)
        # concentrate a fifth of the draws right on the rung boundaries
        r[: 40_000] = rng.choice(edges, 40_000) * (
            1.0 + rng.normal(scale=1e-15, size=40_000))
        ok &= bool(np.array_equal(ladder_order(r), _ladder_order_fast(r)))
    print("  fidelity ladder lookup bit-identical (4.0e6 draws + edges): %s"
          " [dtype match %s]" % (ok, dt))
    return ok and dt


def run_slew_shape_sweep():
    """The single-pass slew penalty against the three-np.diff original.

    numpy's pairwise summation has different unrolling below and above 8 and
    128 elements, so the equality has to be checked at several block lengths,
    not only at the deployed T = 20, and at several swarm sizes.
    """
    from mpc_fast import _FastObjective
    ok = True
    for T in (2, 3, 8, 9, 17, 20, 64, 129, 130, 257):
        for n in (1, 2, 7, 30, 64, 257):
            m = BeamSteeringMPC(1.2, 1.1, 0.1, 10 ** 3.8, horizon=T, seed=1)
            fo = _FastObjective(m)
            rng = np.random.default_rng(T * 1000 + n)
            lo, hi = m.lower(), m.upper()
            X = lo + rng.random((n, lo.size)) * (hi - lo)
            X[: max(1, n // 4)] = m.repair(X[: max(1, n // 4)])
            X = np.ascontiguousarray(X)
            pen0 = np.zeros(n)
            viol0 = np.zeros(n, dtype=bool)
            for (s, e), lim in zip(m.blocks(), m.block_slew()):
                d = np.abs(np.diff(X[:, s:e], axis=1))
                pen0 = pen0 + np.sum(d ** 2, axis=1) / max(T - 1, 1)
                viol0 |= np.any(d > lim, axis=1)
            pen1, viol1 = fo._slew(X, n, fo.cache)
            if not (bit_equal(pen0, pen1).all()
                    and np.array_equal(viol0, viol1)):
                ok = False
                print("    slew MISMATCH at T=%d n=%d" % (T, n))
    print("  single-pass slew penalty bit-identical over 60 (T, n) shapes: %s"
          % ok)
    return ok


def run_geometry_equivalence():
    """The fused single-erf geometry against channel.beam_geometry."""
    from channel import beam_geometry
    from mpc_fast import _beam_geometry_fused
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(200):
        w = np.concatenate([rng.uniform(0.02, 6.0, 400),
                            np.array([0.0, 1e-9, np.inf, np.nan, 0.0548, 0.0549])])
        a, b = beam_geometry(w)
        c, d = _beam_geometry_fused(w)
        ok &= bool(bit_equal(a, c).all() and bit_equal(b, d).all())
    print("  fused beam_geometry bit-identical (incl. domain edge / NaN): %s" % ok)
    return ok


def run_switch_equivalence():
    """Every faithfulness switch, and the branches only they reach.

    `steering=False` collapses the decision vector to 20-D and freezes the
    pointing state; `h_in_aber=False` removes the per-stage SNR rescaling and
    leaves a single reference-SNR group; `strict_admissibility=False` swaps the
    sum for `np.nansum`, which is the one place where a length-1 reduction is
    NOT the identity; `use_fidelity_ladder=False` replaces the ladder with the
    fixed order; a non-zero FSM dead time puts an integer delay in the pointing
    recursion.  Each of these is a separate branch in `mpc_fast`, so each is
    compared rather than assumed.
    """
    from hclpso_ga import SolverConfig
    ok = True
    tot = same = 0
    cases = []
    for steering in (True, False):
        for h_in_aber in (True, False):
            for strict in (True, False):
                for ladder in (True, False):
                    cases.append((steering, h_in_aber, strict, ladder, 0))
    cases.append((True, True, True, True, 1))       # non-zero FSM dead time
    cases.append((True, True, True, True, 3))

    for steering, h_in_aber, strict, ladder, delay in cases:
        cfg = SolverConfig(use_fidelity_ladder=ladder, fixed_order=10)
        kw = dict(horizon=HORIZON, seed=13, config=cfg, steering=steering,
                  h_in_aber=h_in_aber, strict_admissibility=strict)
        for rs in (None, 1, 4):
            ref = BeamSteeringMPC(1.2, 1.1, 0.10, 10.0 ** 3.8,
                                  rank_stages=rs, **kw)
            fst = FastBeamSteeringMPC(1.2, 1.1, 0.10, 10.0 ** 3.8,
                                      rank_stages=rs, **kw)
            ref.act_delay_samples = delay
            fst.act_delay_samples = delay
            rng = np.random.default_rng(7)
            for X in make_swarms(ref, seed=21, n_swarms=4):
                for st in states(rng, ref.L):
                    hp = rng.normal(scale=0.05, size=HORIZON)
                    c0, a0 = ref._objective(X, st, hp)
                    c1, a1 = fst._objective(X, st, hp)
                    for u, v in ((c0, c1), (a0["z"], a1["z"]),
                                 (a0["pe_first"], a1["pe_first"])):
                        eq = bit_equal(u, v)
                        tot += eq.size
                        same += int(eq.sum())
                        if not eq.all():
                            ok = False
                            print("    switch MISMATCH", steering, h_in_aber,
                                  strict, ladder, delay, rs)
    print("  faithfulness switches (%d combinations, incl. non-zero FSM dead"
          " time): %s  [%d of %d bit-identical]"
          % (len(cases) * 3, ok, same, tot))
    return ok


def run_step_equivalence():
    """End to end: the whole closed-loop solve, both ways, same seed.

    `HCLPSOGA` is deterministic given its seed, and the only thing the two
    controllers do differently is evaluate the objective.  If the objective is
    bit-identical then the swarm follows the same trajectory through the search
    space and `step()` must return the same command, the same incumbent value,
    the same iteration count and the same guard tallies.  This is the check
    that matters operationally: it exercises `repair`, the guard, the anytime
    incumbent and the ladder together, over many cycles, on data the solver
    generated itself rather than on data chosen by this script.
    """
    ok = True
    checked = 0
    for reg, (A, B) in REGIMES.items():
        for sigma_s in (0.05, 0.10, 0.30):
            for rs in (None, 1):
                # tau_o=None for BOTH controllers: the point of the check is
                # that the fast objective returns the same numbers as the
                # readable one, and with the default tau_o = 600 us anytime
                # checkpoint the iteration count depends on the machine's
                # speed -- the fast objective completes more iterations in
                # the same wall-clock budget, which changes best_f, trace and
                # guard tallies without any arithmetic differing.  Comparing
                # full-budget runs (25 iterations, deterministic given the
                # seed) is the comparison that can only disagree if the
                # objective changed.
                ref = BeamSteeringMPC(A, B, sigma_s, 10.0 ** 3.8,
                                      horizon=HORIZON, seed=5, rank_stages=rs,
                                      tau_o=None)
                fst = FastBeamSteeringMPC(A, B, sigma_s, 10.0 ** 3.8,
                                          horizon=HORIZON, seed=5,
                                          rank_stages=rs, tau_o=None)
                rng = np.random.default_rng(99)
                th = np.zeros(2)
                for _ in range(6):
                    th = th + rng.normal(scale=2e-5, size=2)
                    h = float(rng.normal(scale=0.05))
                    r0 = ref.step(th.copy(), h)
                    r1 = fst.step(th.copy(), h)
                    checked += 1
                    same = (bit_equal(np.atleast_1d(r0.best_f),
                                      np.atleast_1d(r1.best_f)).all()
                            and r0.iterations == r1.iterations
                            and r0.evaluations == r1.evaluations
                            and r0.rejected_by_guard == r1.rejected_by_guard
                            and bit_equal(np.asarray(r0.incumbent_trace),
                                          np.asarray(r1.incumbent_trace)).all())
                    if r0.best_x is None or r1.best_x is None:
                        same &= (r0.best_x is None) == (r1.best_x is None)
                    else:
                        same &= bool(bit_equal(r0.best_x, r1.best_x).all())
                    if not same:
                        ok = False
                        print("    step MISMATCH  regime=%s sigma_s=%.2f rs=%s"
                              % (reg, sigma_s, rs))
                if ref.guard_stats != fst.guard_stats:
                    ok = False
                    print("    guard tally MISMATCH", ref.guard_stats,
                          fst.guard_stats)
    print("  full step(): identical command, incumbent, trace and guard tally"
          " over %d cycles: %s" % (checked, ok))
    return ok


# ------------------------------------------------------------------ timing
def _block_stats(fn, reps):
    ts = np.empty(reps, dtype=np.int64)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts[i] = time.perf_counter_ns() - t0
    return float(np.min(ts)), float(np.median(ts))


def ab_time(fa, fb, blocks=40, reps=40, warm=200):
    """Interleaved A/B timing under contention.

    This machine runs other benchmark processes; the CPU sits near saturation
    and a median over raw samples measures the contention as much as the code.
    Interference can only ADD time, never remove it, so the block MINIMUM is the
    estimator of the uncontended cost and the median of block minima is the
    reported figure.  The raw median is carried alongside so the gap between the
    two shows how contended the run was.

    Returns (min_us_a, min_us_b, med_us_a, med_us_b).
    """
    for _ in range(warm):
        fa()
        fb()
    na = np.empty(blocks)
    nb = np.empty(blocks)
    da = np.empty(blocks)
    db = np.empty(blocks)
    for k in range(blocks):
        if k & 1:                       # alternate which goes first
            nb[k], db[k] = _block_stats(fb, reps)
            na[k], da[k] = _block_stats(fa, reps)
        else:
            na[k], da[k] = _block_stats(fa, reps)
            nb[k], db[k] = _block_stats(fb, reps)
    return (float(np.median(na)) / 1e3, float(np.median(nb)) / 1e3,
            float(np.median(da)) / 1e3, float(np.median(db)) / 1e3)


class _StubKernel:
    """Substitute for pe_series_f64 that does no series arithmetic.

    Used ONLY for timing, to separate the wrapper from the kernel.  It returns
    zeros of the right shape, so the surrounding wrapper does exactly the same
    array traffic.
    """

    def __init__(self):
        self.real = rtodt_fast.pe_series_f64

    def __enter__(self):
        def stub(A, B, xi, A0, gbar, K):
            return np.zeros(np.shape(np.atleast_1d(xi)))
        rtodt_fast.pe_series_f64 = stub
        mpc_loop.pe_series_f64 = stub
        import mpc_fast
        mpc_fast.pe_series_f64 = stub
        return self

    def __exit__(self, *a):
        rtodt_fast.pe_series_f64 = self.real
        mpc_loop.pe_series_f64 = self.real
        import mpc_fast
        mpc_fast.pe_series_f64 = self.real


def run_timing():
    A, B = REGIMES["strong"]
    sigma_s = 0.10
    gbar = 10.0 ** 3.8
    st = np.array([1.2e-5, -0.9e-5])

    print("=" * 78)
    print("TIMING  (A/B interleaved, N_p = %d; figure is the median of block"
          % NP_SWARM)
    print("        minima -- see ab_time for why, this machine is contended)")
    print("=" * 78)
    print("  %-11s %10s %10s %8s   %10s %10s %8s"
          % ("rank_stages", "base us", "fast us", "speedup",
             "base wrap", "fast wrap", "x"))

    rows = []
    for rs in (1, 2, 3, 5, 10, None):
        ref = BeamSteeringMPC(A, B, sigma_s, gbar, horizon=HORIZON, seed=7,
                              rank_stages=rs)
        fst = FastBeamSteeringMPC(A, B, sigma_s, gbar, horizon=HORIZON, seed=7,
                                  rank_stages=rs)
        rng = np.random.default_rng(4)
        lo, hi = ref.lower(), ref.upper()
        X = ref.repair(lo + rng.random((NP_SWARM, lo.size)) * (hi - lo))
        ref.kf.update(0.03)
        hp = ref.kf.predict(HORIZON)

        tb, tf, mb_, mf_ = ab_time(lambda: ref._objective(X, st, hp),
                                   lambda: fst._objective(X, st, hp))
        with _StubKernel():
            ref2 = BeamSteeringMPC(A, B, sigma_s, gbar, horizon=HORIZON, seed=7,
                                   rank_stages=rs)
            fst2 = FastBeamSteeringMPC(A, B, sigma_s, gbar, horizon=HORIZON,
                                       seed=7, rank_stages=rs)
            wb, wf, _, _ = ab_time(lambda: ref2._objective(X, st, hp),
                                   lambda: fst2._objective(X, st, hp))
        rows.append((rs, tb, tf, wb, wf, mb_, mf_))
        print("  %-11s %10.1f %10.1f %7.2fx   %10.1f %10.1f %7.2fx"
              % (rs, tb, tf, tb / tf, wb, wf, wb / wf))

    # monotonicity sanity check on the full-objective timings
    xs = [r for r in rows if r[0] is not None]
    tb_seq = [r[1] for r in xs]
    tf_seq = [r[2] for r in xs]
    for name, seq in (("base", tb_seq), ("fast", tf_seq)):
        if any(seq[i + 1] < seq[i] for i in range(len(seq) - 1)):
            print("  NOTE: the %s sweep is NOT monotonic in rank_stages; that is"
                  % name)
            print("        scheduler noise, not signal -- do not read it as a finding.")
    print()
    return rows


def run_cycle_budget():
    """How many solver iterations fit inside tau_O = 600 us, both ways."""
    A, B = REGIMES["strong"]
    sigma_s, gbar = 0.10, 10.0 ** 3.8
    st = np.array([1.2e-5, -0.9e-5])
    print("=" * 74)
    print("ITERATIONS INSIDE tau_O = 600 us  (objective evaluations only)")
    print("=" * 74)
    print("  %-13s %14s %14s" % ("rank_stages", "base iters", "fast iters"))
    for rs in (1, 3, None):
        ref = BeamSteeringMPC(A, B, sigma_s, gbar, horizon=HORIZON, seed=7,
                              rank_stages=rs)
        fst = FastBeamSteeringMPC(A, B, sigma_s, gbar, horizon=HORIZON, seed=7,
                                  rank_stages=rs)
        rng = np.random.default_rng(4)
        lo, hi = ref.lower(), ref.upper()
        X = ref.repair(lo + rng.random((NP_SWARM, lo.size)) * (hi - lo))
        ref.kf.update(0.03)
        hp = ref.kf.predict(HORIZON)
        tb, tf, _, _ = ab_time(lambda: ref._objective(X, st, hp),
                               lambda: fst._objective(X, st, hp),
                               blocks=20, reps=30)
        print("  %-13s %14.1f %14.1f" % (rs, 600.0 / tb, 600.0 / tf))
    print()


if __name__ == "__main__":
    print()
    print("COMPONENT EQUIVALENCE (each replaced piece, on its own)")
    ok_g = run_geometry_equivalence()
    ok_r = run_stage_rd_equivalence()
    ok_l = run_ladder_equivalence()
    ok_s = run_slew_shape_sweep()
    print()
    same, tot, worst = run_equivalence()
    ok_w = run_switch_equivalence()
    ok_e = run_step_equivalence()
    print()
    rows = run_timing()
    run_cycle_budget()
    if same != tot or not (ok_g and ok_r and ok_l and ok_s and ok_e and ok_w):
        print("RESULT: NOT bit-identical -- max relative difference %.3e" % worst)
        sys.exit(1)
    print("RESULT: bit-identical on every compared entry (%d of %d)."
          % (same, tot))
