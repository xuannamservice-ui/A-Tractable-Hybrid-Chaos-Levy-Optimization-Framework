"""
Tests for the domain guard on the beam geometry of eq. (3).

WHAT IS BEING TESTED, AND WHY IT IS A CORRECTNESS TEST AND NOT A STYLE TEST

    The Farid-Hranilovic equivalent width w_{z,eq}(w_z) is non-monotonic, with
    an interior minimum at w_z* = 0.054869 m for a = 0.05 m.  The manuscript
    computes this minimum itself (Sec. VII-A) and uses it to set the attainable
    floor xi_min(sigma_s) = 0.0877/(2 sigma_s) of the decision box, so the
    boundary is the manuscript's own, not one invented here.

    Below w_z* the map INVERTS: narrowing the beam raises w_{z,eq}, hence raises
    xi = w_{z,eq}/(2 sigma_s), while A_0 rises towards 1.  A beam on that branch
    reports full collected power AND unlimited immunity to pointing jitter --
    the trade-off the entire optimization exists to resolve, running backwards.
    It is not a beam; it is eq. (3) being read outside the regime it was derived
    in, since at w_z < w_z* the beam is narrower than the aperture itself.

    Before the guard, `system_metric.success()` returned True for such
    configurations.  test_impossible_beam_no_longer_passes_success pins that.

Run:  python test_beam_geometry.py      (or: python -m pytest test_beam_geometry.py)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import channel as ch
import mpc_loop as ml
import system_metric as sm

SIGMAS = sm.SIGMAS
TOL = 1e-9


# ---------------------------------------------------------------- boundary
def test_branch_minimum_matches_the_manuscript():
    """The guard boundary is the minimum the manuscript prints, recomputed."""
    w_star, weq_star = sm.branch_min_wz()
    # Sec. VII-A: "minimum 0.0877 m at w_z = 0.0549 m for a = 0.05 m"
    assert abs(w_star - 0.0549) < 5e-5, w_star
    assert abs(weq_star - 0.0877) < 5e-5, weq_star
    # and it really is a stationary point of w_zeq, found independently
    r = minimize_scalar(lambda w: sm.beam_geometry(w, check_domain=False)[1],
                        bounds=(1e-3, 1.0), method="bounded",
                        options=dict(xatol=1e-14))
    assert abs(r.x - w_star) < 1e-8, (r.x, w_star)
    # the two implementations of the same boundary agree
    assert abs(ch.branch_min_wz()[0] - w_star) < 1e-12
    assert abs(ch.branch_min_wz()[1] - weq_star) < 1e-12
    return dict(w_star=w_star, weq_star=weq_star, scipy_argmin=float(r.x))


def test_boundary_scales_with_aperture():
    """v = sqrt(pi/2) a / w_z, so the whole geometry is a function of a/w_z and
    the boundary must scale linearly in a.  A hardcoded 0.0549 would not."""
    w0, q0 = sm.branch_min_wz(0.05)
    out = {}
    for a in (0.02, 0.05, 0.10, 0.25):
        w, q = sm.branch_min_wz(a)
        out[a] = (w, q, w / a, q / a)
        assert abs(w / a - w0 / 0.05) < 1e-7, (a, w / a, w0 / 0.05)
        assert abs(q / a - q0 / 0.05) < 1e-7, (a, q / a, q0 / 0.05)
    return out


def test_map_really_does_invert_below_the_boundary():
    """The premise of the guard, measured rather than asserted."""
    w_star, _ = sm.branch_min_wz()
    below = np.array([0.30, 0.50, 0.70, 0.90]) * w_star
    above = np.array([1.10, 2.0, 5.0, 20.0]) * w_star
    rec = {"below": [], "above": []}
    for w in below:
        A0, q = sm.beam_geometry(w, check_domain=False)
        rec["below"].append((float(w), float(A0), float(q)))
    for w in above:
        A0, q = sm.beam_geometry(w, check_domain=False)
        rec["above"].append((float(w), float(A0), float(q)))
    # The statement being tested is about the SIGN of dA_0/dw_zeq, i.e. whether
    # more collected power is bought with more pointing sensitivity or given
    # away free. A_0 falls monotonically with w_z on both branches; what flips
    # is the direction w_zeq moves.
    qb = [r[2] for r in rec["below"]]
    ab = [r[1] for r in rec["below"]]
    qa = [r[2] for r in rec["above"]]
    aa = [r[1] for r in rec["above"]]
    assert all(ab[i] > ab[i + 1] for i in range(len(ab) - 1)), ab
    assert all(aa[i] > aa[i + 1] for i in range(len(aa) - 1)), aa
    # narrow branch: w_zeq falls WITH A_0 -- higher xi comes with MORE power
    assert all(qb[i] > qb[i + 1] for i in range(len(qb) - 1)), qb
    # broad branch: w_zeq rises AGAINST A_0 -- the physical trade-off
    assert all(qa[i] < qa[i + 1] for i in range(len(qa) - 1)), qa
    # stated as the slope itself, so the inversion is a number and not a claim
    slope_below = np.diff(ab) / np.diff(qb)
    slope_above = np.diff(aa) / np.diff(qa)
    assert (slope_below > 0).all(), slope_below
    assert (slope_above < 0).all(), slope_above
    rec["dA0_dwzeq_below"] = slope_below.tolist()
    rec["dA0_dwzeq_above"] = slope_above.tolist()
    return rec


# ------------------------------------------------------------------- guard
def test_guard_raises_on_the_narrow_branch():
    w_star, _ = sm.branch_min_wz()
    raised = []
    for w in (1e-6, 0.005, 0.01, 0.02, 0.0286173, 0.04, w_star * (1 - 1e-3)):
        try:
            sm.beam_geometry(w)
        except sm.BeamGeometryDomainError:
            raised.append(w)
    assert len(raised) == 7, raised
    for w in (0.0, -0.1, float("nan"), float("inf")):
        try:
            sm.beam_geometry(w)
            raise AssertionError("no raise at w_z = %r" % w)
        except sm.BeamGeometryDomainError:
            pass
    return dict(rejected=raised)


def test_guard_admits_the_whole_manuscript_decision_box():
    """The guard must exclude NOTHING the manuscript admits.

    This is the test that keeps the guard from being a convenient way to drop
    inconvenient candidates: the manuscript's box is
    xi in [max(0.5, xi_min(sigma_s)), 4.888] at every swept sigma_s, and every
    beam in it -- both edges included -- must pass.
    """
    out = {}
    for s in SIGMAS:
        lo, hi = ml.manuscript_wz_box(s)
        grid = np.linspace(lo, hi, 2001)
        ok = ch.beam_geometry_valid(grid, ch.APERTURE)
        assert ok.all(), (s, grid[~ok][:5])
        # the scalar entry point must not raise anywhere in the box either
        for w in (lo, 0.5 * (lo + hi), hi):
            sm.beam_geometry(float(w))
        A0lo, qlo = sm.beam_geometry(float(lo))
        out["%.2f" % s] = dict(w_lo=float(lo), w_hi=float(hi),
                               xi_lo=float(qlo / (2 * s)),
                               admitted=int(ok.sum()), n=int(ok.size))
    # the box's lower edge at sigma_s = 0.05 m IS the boundary: the guard sits
    # exactly on the manuscript's own floor, with nothing to spare on either side
    w_star, _ = sm.branch_min_wz()
    assert abs(ml.manuscript_wz_box(0.05)[0] - w_star) < 1e-9
    return out


def test_channel_flags_instead_of_raising():
    """The vectorised evaluator the solver calls per candidate must flag, not
    raise, so one bad particle cannot abort a control cycle."""
    w = np.array([0.02, 0.0286173, 0.054869382, 0.1, 1.0])
    A0, q = ch.beam_geometry(w)
    assert np.isnan(A0[:2]).all() and np.isnan(q[:2]).all(), (A0, q)
    assert np.isfinite(A0[2:]).all() and np.isfinite(q[2:]).all(), (A0, q)
    # and the flag survives into the fitness the guard inspects
    from rtodt_fast import pe_series_f64, z_of
    xi = q / (2 * 0.10)
    z = z_of(1.2, 1.1, A0, 10 ** 3.8)
    pe = pe_series_f64(1.2, 1.1, xi, A0, 10 ** 3.8,
                       np.where(np.isfinite(z) & (z <= 8), 20, -1))
    assert not np.isfinite(pe[0]) and not np.isfinite(pe[1]), pe
    from mpc_loop import envelope_guard
    rep = envelope_guard(z, pe, three_part=True)
    assert not rep.admissible[0] and not rep.admissible[1], rep.admissible
    return dict(A0=A0.tolist(), w_zeq=q.tolist(), pe=pe.tolist(),
                admissible=rep.admissible.tolist())


# ------------------------------------------------------- the actual defect
def test_impossible_beam_no_longer_passes_success():
    """THE REGRESSION.  A narrow-branch beam and the real beam at the same xi.

    Before the guard, `success()` returned True for the first and False for the
    second, at identical xi.  The first does not exist.
    """
    w_star, _ = sm.branch_min_wz()
    target_weq = 0.20                       # xi = 1.0 at sigma_s = 0.10 m
    w_narrow = brentq(lambda w: sm.beam_geometry(w, check_domain=False)[1] - target_weq,
                      1e-4, w_star, xtol=1e-15)
    w_broad = brentq(lambda w: sm.beam_geometry(w, check_domain=False)[1] - target_weq,
                     w_star, 200.0, xtol=1e-14)
    A0n, qn = sm.beam_geometry(w_narrow, check_domain=False)
    A0b, qb = sm.beam_geometry(w_broad, check_domain=False)
    assert abs(qn - qb) < 1e-6                       # same xi
    assert A0n > 0.99 and A0b < 0.2                  # free-lunch vs real

    # the real beam: scoreable, and it FAILS the 1e-6 system target
    cfg_broad = sm.BeamConfig("strong", w_broad, 0.10)
    pe_broad = sm.aber_of(cfg_broad)
    assert sm.success(cfg_broad) is False, pe_broad

    # the impossible beam: what it WOULD have scored, and what it does now
    pe_narrow_unguarded = sm.system_aber(
        1.2, 1.1, float(qn / 0.20), float(A0n), 10.0 ** 3.8, method="fast")
    assert pe_narrow_unguarded <= sm.ABER_TARGET, pe_narrow_unguarded

    cfg_narrow = sm.BeamConfig("strong", w_narrow, 0.10)
    for probe in (lambda: cfg_narrow.A0, lambda: cfg_narrow.xi,
                  lambda: sm.aber_of(cfg_narrow), lambda: sm.success(cfg_narrow)):
        try:
            probe()
            raise AssertionError("an impossible beam was still scoreable")
        except sm.BeamGeometryDomainError:
            pass
    return dict(w_narrow=w_narrow, A0_narrow=float(A0n),
                pe_narrow_unguarded=float(pe_narrow_unguarded),
                w_broad=w_broad, A0_broad=float(A0b), pe_broad=float(pe_broad),
                xi=float(qb / 0.20))


def test_from_xi_stays_on_the_physical_branch():
    """BeamConfig.from_xi must return the broad-branch root at every xi in the
    box, so the two-branch ambiguity never reaches a scored configuration."""
    out = []
    w_star, _ = sm.branch_min_wz()
    for regime in ("weak", "moderate", "strong"):
        for s in SIGMAS:
            xi_lo = max(0.5, ch.xi_floor(s))
            for xi in np.linspace(xi_lo, sm.XI_MAX, 9):
                cfg = sm.BeamConfig.from_xi(regime, float(xi), s)
                assert cfg.w_z >= w_star * (1 - 1e-9), (regime, s, xi, cfg.w_z)
                assert abs(cfg.xi - xi) < 1e-9, (cfg.xi, xi)
                out.append((regime, s, float(xi), cfg.w_z))
    return dict(checked=len(out), min_w_z=min(o[3] for o in out))


TESTS = [test_branch_minimum_matches_the_manuscript,
         test_boundary_scales_with_aperture,
         test_map_really_does_invert_below_the_boundary,
         test_guard_raises_on_the_narrow_branch,
         test_guard_admits_the_whole_manuscript_decision_box,
         test_channel_flags_instead_of_raising,
         test_impossible_beam_no_longer_passes_success,
         test_from_xi_stays_on_the_physical_branch]


if __name__ == "__main__":
    import traceback

    print("=" * 78)
    print("DOMAIN GUARD ON eq. (3) -- test_beam_geometry.py")
    print("=" * 78)
    n_fail = 0
    for t in TESTS:
        try:
            info = t()
            print("\n  PASS  %s" % t.__name__)
            if isinstance(info, dict):
                for k, v in info.items():
                    print("          %-22s %s" % (k, v))
        except Exception:
            n_fail += 1
            print("\n  FAIL  %s" % t.__name__)
            traceback.print_exc()
    print("\n" + "=" * 78)
    print("  %d of %d passed" % (len(TESTS) - n_fail, len(TESTS)))
    print("=" * 78)
    raise SystemExit(1 if n_fail else 0)
