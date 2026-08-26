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
    post-EGC system ABER target, whereas the solver here ranks candidates by the
    per-branch surrogate. The two are different quantities and their numerical
    values are not comparable.

    Since the landscape diagnosis this script reports BOTH:
      * the per-branch columns it always reported (median/best_selected_aber,
        median_objective) -- the quantity the solver optimises;
      * the system columns (system_success_rate, median_system_aber, ...) --
        the quantity the manuscript's success criterion DEFINES, computed by
        system_metric.py through eq:mimo_egc_aber over the 4x4 combined channel
        at gbar_op = 38 dB.
    The system columns are still not a reproduction of Table 9 or Table 11: the
    campaign PROTOCOL (what one trial redraws, which sigma_s the published
    campaign ran at, whether a realization is one MPC cycle or a warm-started
    multi-cycle run) is not specified in the manuscript, and this driver's
    protocol -- one cold-started cycle, sigma_s drawn uniformly from the four
    swept levels -- was chosen by this release, not by the paper. The per-sigma
    breakdown is reported alongside the pooled rate precisely because the pooled
    rate is a property of that draw.

WHAT IT IS USEFUL FOR
    (a) exercising the algorithm end to end;
    (b) A/B experiments in which everything is held fixed except one component,
        which is how the ablation and guard comparisons below are constructed.

THE GUARD A/B, AND WHAT IT USED TO BE
    This driver used to sweep `three_part=True/False`, which selected between

        arm A : (i) z <= z_max  AND  (ii) 0 <= Pe_branch <= 1/2
        arm B :                       (ii) 0 <= Pe_branch <= 1/2

    applied per candidate inside the swarm loop.  Two things were wrong with it.

    NEITHER ARM WAS A CONFIGURATION THE MANUSCRIPT DESCRIBES.  Sec. VI-C's guard
    is three tests, and test (iii), Pe < eps_safe = 1e-3, was in neither arm.
    The manuscript's *campaign* form is test (iii) alone -- "the campaign ran
    with the guard in its threshold-only form (Pe_bar < eps_safe)" (Sec. VII-B)
    -- and its *full* form is (i)+(ii)+(iii).  The old sweep offered (i)+(ii)
    against (ii), which is neither of those.

    AND THE TWO ARMS WERE THE SAME EXPERIMENT.  Over the whole swept decision
    box -- 3 regimes x 4 sigma_s x 4000 waists -- the number of candidates
    rejected by test (i) that test (ii) would have admitted is exactly 0.  The
    reason is structural: the fidelity ladder returns order -1 for z > z_max and
    the kernel returns NaN for order -1, so "z exceeds z_max" and "the fitness
    is not a number" are the same event.  The manuscript says as much: test (i)
    "is the same quantity that selects the order K on the adaptive-fidelity
    ladder" (Sec. VI-C).  `guard_test_overlap()` below measures this, and it is
    printed on every run so the equality is never mistaken for a null result
    about guard design.

    WHAT IS IMPLEMENTED NOW.  The manuscript's own two forms, applied where the
    manuscript applies them -- "to the command about to be published, within the
    safety-check stage tau_C" (Sec. VI-C):

        three_part      (i) z <= z_max, (ii) 0 <= Pe_branch <= 1/2,
                        (iii) Pe_sys < eps_safe
        threshold_only  (iii) Pe_sys < eps_safe

    Test (iii) is evaluated at SYSTEM level, because the manuscript puts it
    there: "The success test, the guard threshold eps_safe and the fallback
    figure of Numerical Result 1 are therefore all evaluated at system level"
    (Sec. VII-A).  A command failing its arm's guard is replaced by the offline
    xi_safe override, and BOTH the published beam and the actuated beam are
    scored, so the guard's effect on the reported rate is visible rather than
    folded in.

    The per-candidate filtering is identical in both arms, because the
    manuscript makes it identical: test (i) is the only test it applies inside
    the fitness evaluation, and in this implementation that is the ladder.

WHAT STILL CANNOT BE REPRODUCED FROM WHAT SHIPPED
    The manuscript's headline guard experiment is two kernels x two guard forms
    (Sec. III-B): the tabulate-and-interpolate kernel against the
    interpolation-free one.  This driver has only the interpolation-free kernel
    (`rtodt_fast.py`) plumbed into the MPC loop, so the kernel axis is absent
    here.  It is exercised separately, on a 1-D solver, by `campaign2.py`.
    Nothing below measures the interpolated kernel.

Usage:
    python run_campaign.py [--realizations 300] [--out ../data/07_reference_campaign]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from channel import beam_geometry, xi_floor, SwayProcess, GammaGammaAR1, xi_effective
from hclpso_ga import SolverConfig, ladder_order
from mpc_loop import (BeamSteeringMPC, envelope_guard, EPS_SAFE, LINK_LENGTH,
                      manuscript_wz_box, wz_for_xi, XI_MAX, Z_MAX)
from rtodt_fast import pe_series_f64, z_of
import system_metric as sm

REGIMES = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
SIGMAS = [0.05, 0.1, 0.2, 0.3]
GBAR_DB = 38.0

# Sec. VI-C envelope guard. Both forms the manuscript describes.
GUARD_FORMS = ("three_part", "threshold_only")

# The offline safe state of Numerical Result 1: "an intrinsic optimum
# xi_safe ~= 0.83 ... corresponds to the physical beam w_z ~= 0.157 m". The
# manuscript offers two realizations of the override (a fixed PHYSICAL beam, or
# a command in xi clipped per sigma_s) and says the second "is safer still". The
# first is used here because it is the one Numerical Result 1 is stated for; it
# is clipped into each sigma_s's box, which only ever makes it wider.
XI_SAFE = 0.83
SIGMA_S_NOM = 0.10

# The manuscript's optimization-success criterion (Sec. VI-C): a beam
# configuration counts as a success iff its POST-EGC SYSTEM ABER satisfies
# P_e_bar <= 1e-6 at the fixed reference SNR gbar_op = 38 dB.  Both constants
# are taken from system_metric.py rather than restated here, so there is one
# definition of the test in the release.
ABER_TARGET = sm.ABER_TARGET
GBAR_OP_DB = sm.GBAR_OP_DB

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


_SYS_CACHE: dict = {}


def system_quality(alpha, beta, w_z, sigma_s, r_d, gbar_db=GBAR_OP_DB):
    """POST-EGC SYSTEM ABER of the published command -- the quantity the
    manuscript's success criterion is defined against (eq:mimo_egc_aber).

    WHICH CONFIGURATION IS SCORED.  The MPC publishes only its first horizon
    stage, and mpc_loop's envelope guard vets exactly that stage (`pe_first`).
    So the success test is applied to the same stage-0 configuration the guard
    admitted:

        w_z  = best_x[0]                      the published divergence command
        r_d  = L * ||Theta(t)||_2             the MEASURED pointing offset

    Stage 0 of the horizon recursion Theta(t+k+1) = Theta(t+k) - u_ptr(t+k)
    still carries Theta(t): a steering command issued at time t moves the beam
    for stage 1 onward, so the beam published at t experiences the offset that
    was measured at t.  Using the post-command offset instead would credit the
    controller with a correction the published beam never saw.

    WHY xi_eff AND NOT xi.  The manuscript states that it is xi_eff, not the
    nominal xi, that feeds the kernel (eq:xi_eff), and mpc_loop's own stage cost
    uses xi_eff.  Note this differs from `true_quality` below, which is the
    pre-existing per-branch column and evaluates nominal xi; that column is left
    exactly as it was so the previously reported numbers do not move.

    WHY gbar_op AND NOT THE PREDICTED SNR.  The criterion is written "at the
    fixed reference SNR gbar_op = 38 dB".  It is a property of the beam, not of
    the instantaneous channel, and it is the same 38 dB for every algorithm row
    and every ablated variant.  The per-stage gbar_k = gbar*h_hat_k^2 that
    mpc_loop uses inside the objective is a forecasting device for RANKING
    candidates and does not enter the success test.
    """
    if w_z is None or not np.isfinite(w_z):
        return np.nan
    key = (alpha, beta, float(w_z), float(sigma_s), float(r_d), float(gbar_db))
    v = _SYS_CACHE.get(key)
    if v is None:
        A0, w_zeq = sm.beam_geometry(float(w_z))
        xi = w_zeq / (2.0 * sigma_s)
        xi_eff = xi_effective(xi, float(r_d), sigma_s)
        v = sm.system_aber(alpha, beta, xi_eff, A0, 10.0 ** (gbar_db / 10.0),
                           method="fast")
        _SYS_CACHE[key] = v
    return v


# --------------------------------------------------------------------------
# The Section VI-C envelope guard, in the two forms the manuscript describes
# --------------------------------------------------------------------------
def xi_safe_wz(sigma_s: float) -> float:
    """The offline override beam, as a transmitted waist, clipped into the box."""
    lo, hi = manuscript_wz_box(sigma_s)
    w = wz_for_xi(XI_SAFE, SIGMA_S_NOM)      # the physical beam of Num. Result 1
    return float(np.clip(w, lo, hi))


def z_worst_in_box(alpha, beta, sigma_s, gbar_db=GBAR_DB):
    """z of the widest beam the box allows -- the manuscript's eq:z_worst.

    Printed against the manuscript's own figures because it is the quantity that
    decides whether guard test (i) can bind at all at a given sway level. The
    manuscript computes 1.12 / 4.52 / 17.97 / 40.43 at sigma_s = 0.05 / 0.1 /
    0.2 / 0.3 m (strong, 38 dB) and concludes: "beyond sigma_s ~ 0.1 m the
    widest beams are genuinely inadmissible and the two guard forms do differ".
    """
    hi = manuscript_wz_box(sigma_s)[1]
    A0, _ = beam_geometry(np.array([hi]))
    return float(z_of(alpha, beta, A0, 10.0 ** (gbar_db / 10.0))[0]), float(hi)


def command_guard(form, alpha, beta, w_z, sigma_s, r_d, gbar_db=GBAR_OP_DB):
    """Sec. VI-C, applied to the command about to be published.

    Returns (admitted, binding, diag). `binding` is the list of test labels the
    command failed, so a run can report WHICH test did the work rather than only
    that something did.

        (i)   z(u) <= z_max                per-branch conditioning
        (ii)  0 <= Pe_branch(u) <= 1/2     per-branch range
        (iii) Pe_sys(u) < eps_safe         POST-EGC system threshold

    `three_part` applies all three; `threshold_only` applies (iii) alone, which
    is the form the manuscript states the reported campaign ran under.
    """
    if form not in GUARD_FORMS:
        raise ValueError("unknown guard form %r; expected one of %r"
                         % (form, GUARD_FORMS))
    if w_z is None or not np.isfinite(w_z):
        return False, ["no_command"], dict(z=None, pe_branch=None, pe_sys=None)

    A0, w_zeq = beam_geometry(np.array([float(w_z)]))
    xi_eff = xi_effective(w_zeq / (2.0 * sigma_s), float(r_d), sigma_s)
    gbar = 10.0 ** (gbar_db / 10.0)
    z = z_of(alpha, beta, A0, gbar)
    pe_b = float(pe_series_f64(alpha, beta, xi_eff, A0, gbar, ladder_order(z))[0])
    pe_s = system_quality(alpha, beta, w_z, sigma_s, r_d, gbar_db)
    zf = float(z[0])
    diag = dict(z=zf, pe_branch=pe_b, pe_sys=float(pe_s))

    binding = []
    if form == "three_part":
        if not (zf <= Z_MAX):
            binding.append("i_admissibility")
        if not (np.isfinite(pe_b) and 0.0 <= pe_b <= 0.5):
            binding.append("ii_range")
    if not (np.isfinite(pe_s) and pe_s < EPS_SAFE):
        binding.append("iii_threshold")
    return (not binding), binding, diag


def guard_test_overlap(n_grid: int = 4000, gbar_db: float = GBAR_DB):
    """Do in-loop tests (i) and (ii) reject the same candidates?

    This is the measurement that explains why the PREVIOUS three_part /
    threshold_only A/B returned identical numbers in every cell. It scans the
    manuscript's box directly, without a solver, and counts candidates that
    test (i) rejects and test (ii) would have admitted. If that count is zero
    the two in-loop forms are the same filter, and any A/B built on the
    difference between them is measuring nothing.
    """
    gbar = 10.0 ** (gbar_db / 10.0)
    rows, total_only_i, total = [], 0, 0
    for regime, (a, b) in REGIMES.items():
        for s in SIGMAS:
            lo, hi = manuscript_wz_box(s)
            w = np.linspace(lo, hi, n_grid)
            A0, weq = beam_geometry(w)
            z = z_of(a, b, A0, gbar)
            pe = pe_series_f64(a, b, weq / (2.0 * s), A0, gbar, ladder_order(z))
            fail_i = z > Z_MAX
            fail_ii = ~np.isfinite(pe) | ~((pe >= 0.0) & (pe <= 0.5))
            only_i = int(np.sum(fail_i & ~fail_ii))
            only_ii = int(np.sum(fail_ii & ~fail_i))
            total_only_i += only_i
            total += int(w.size)
            rows.append(dict(regime=regime, sigma_s=s, n=int(w.size),
                             fail_i=int(fail_i.sum()), fail_ii=int(fail_ii.sum()),
                             rejected_only_by_i=only_i,
                             rejected_only_by_ii=only_ii))
    return dict(rows=rows, scanned=total, rejected_only_by_i=total_only_i,
                tests_are_equivalent=bool(total_only_i == 0))


# --------------------------------------------------------------------------
# What the success rate is, under each way of restricting the sweep
# --------------------------------------------------------------------------
# Every entry names the manuscript sentence that does or does not license the
# restriction. A scoping belongs here because the manuscript scopes something
# that way, never because of the number it produces -- so the list is fixed
# before any of them is evaluated, and all of them are printed.
SCOPINGS = [
    dict(key="as_swept",
         label="all regimes, all four sigma_s",
         regimes=None, sigmas=None,
         support="LICENSED. Sec. VII-A: 'Building sway is swept over sigma_s in "
                 "[0.05,0.1,0.2,0.3] m.' Sec. III-B, explicitly about this "
                 "metric: 'the optimization success rate, however, is swept "
                 "across all four jitter levels'. This is the manuscript's own "
                 "scoping of the success rate."),
    dict(key="strong_all_sigma",
         label="strong turbulence, all four sigma_s",
         regimes=("strong",), sigmas=None,
         support="LICENSED. Table 9's caption reads 'Strong Turbulence', and "
                 "Lemma 2's discussion calls the figure 'the strong-turbulence "
                 "optimization success rate'. Restricting the REGIME is "
                 "supported; it does not restrict sigma_s."),
    dict(key="envelope_all_regimes",
         label="all regimes, sigma_s <= 0.1 m",
         regimes=None, sigmas=(0.05, 0.10),
         support="NOT LICENSED FOR THIS METRIC. The sigma_s <~ 0.1 m envelope "
                 "is real and scopes other claims (Sec. VI-C, link continuity; "
                 "Numerical Result 1, the fallback bound). Sec. III-B draws the "
                 "distinction itself and puts the success rate on the other "
                 "side of it: 'The link-continuity observation ... is "
                 "unaffected; the optimization success rate, however, is swept "
                 "across all four jitter levels.' Reported because it is the "
                 "scoping a reader will ask about, not because it is defensible."),
    dict(key="strong_envelope",
         label="strong turbulence, sigma_s <= 0.1 m",
         regimes=("strong",), sigmas=(0.05, 0.10),
         support="NOT LICENSED FOR THIS METRIC, for the same reason as above; "
                 "the regime half is licensed, the sigma_s half is not."),
    dict(key="strong_quiet_only",
         label="strong turbulence, sigma_s = 0.05 m only",
         regimes=("strong",), sigmas=(0.05,),
         support="NOT LICENSED. No sentence in the manuscript restricts any "
                 "reported figure to the quietest swept level alone. Included "
                 "because it is the only strong-regime cell in which the target "
                 "is reachable at all, so it shows what the headline number "
                 "would require."),
]


def success_by_scoping(per_real, ceiling):
    """Pooled success rate under each entry of SCOPINGS, plus its ceiling.

    `per_real` maps (regime, guard_form) -> (sigma_s array, hit array) for the
    full-kernel arm. The ceiling is the brute-force feasibility bound, counted
    over the same subset, so a low rate can be read as "the solver missed it"
    or "no such beam exists" without leaving the table.
    """
    guards = sorted({g for _, g in per_real})
    out = {}
    for sc in SCOPINGS:
        regs = sc["regimes"] or tuple(REGIMES)
        sgs = sc["sigmas"] or tuple(SIGMAS)
        cells = [ceiling["%s|%.2f" % (rg, s)] for rg in regs for s in sgs
                 if "%s|%.2f" % (rg, s) in ceiling]
        reach = sum(bool(c["target_reachable"]) for c in cells)
        per_guard = {}
        for g in guards:
            k = n = 0
            for rg in regs:
                sig, hit = per_real.get((rg, g), (None, None))
                if sig is None:
                    continue
                m = np.isin(sig, np.array(sgs, dtype=float))
                k += int(hit[m].sum())
                n += int(m.sum())
            lo, hi = clopper_pearson(k, n)
            per_guard[g] = dict(successes=k, n=n,
                                rate=(k / n) if n else None,
                                ci95_lo=lo, ci95_hi=hi)
        out[sc["key"]] = dict(label=sc["label"], manuscript_support=sc["support"],
                              cells_reachable=reach, cells_total=len(cells),
                              ceiling_rate=(reach / len(cells)) if cells else None,
                              by_guard=per_guard)
    return out


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    """Exact (Clopper-Pearson) two-sided binomial interval for k of n.

    Used rather than a Wald interval because the rates here land at 0 and 1,
    where Wald reports a width of exactly zero.
    """
    from scipy.stats import beta as _beta
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def feasibility_ceiling(sigmas, thetas, n_grid: int = 60):
    """Best system ABER attainable by ANY beam in the manuscript's box, at the
    median r_d each sway level actually produced.

    This is the interpretive key for the system_success_rate columns. A rate of
    0% has two completely different causes -- the solver failed to find a beam
    that exists, or no such beam exists -- and only this table distinguishes
    them. It is a property of the channel and the decision box, computed by
    brute-force scan without reference to any solver, so it is an upper bound on
    what any algorithm row could report under this driver's protocol.
    """
    sg = np.array(sigmas, dtype=float)
    rd = np.array([np.linalg.norm(t) for t in thetas], dtype=float) * LINK_LENGTH
    out = {}
    for regime, (A, B) in REGIMES.items():
        for s in SIGMAS:
            m = sg == s
            if not m.any():
                continue
            r = float(np.median(rd[m]))
            lo, hi = manuscript_wz_box(s)
            best, arg = np.inf, None
            for w in np.linspace(lo, hi, n_grid):
                v = system_quality(A, B, float(w), s, r)
                if v < best:
                    best, arg = v, float(w)
            out["%s|%.2f" % (regime, s)] = dict(
                regime=regime, sigma_s=s, median_r_d_m=r, best_w_z=arg,
                best_system_aber=float(best), target_reachable=bool(best <= ABER_TARGET))
    return out


def objective_kwargs(mode: str):
    """Which objective the arms compete on.

    `legacy`   the divergence-only objective this driver used before the
               landscape diagnosis: a 20-D w_z trajectory, a decision box three
               times wider in xi than the manuscript's, and np.nansum over the
               stage costs.
    `faithful` the manuscript's 60-D trajectory of Sec. IV-C: [xi; u_ptr] over
               T = 20 stages, the xi box of Sec. VII-A, and a stage cost that
               rejects rather than discounts an inadmissible stage.

    `landscape_probe.py` measures both.  The arms are near-identical on the
    first because that objective has exactly one local minimum, so no
    exploration mechanism has anything to explore.
    """
    if mode == "legacy":
        return dict(steering=False, manuscript_box=False,
                    strict_admissibility=False, h_in_aber=False)
    return dict(steering=True, manuscript_box=True,
                strict_admissibility=True, h_in_aber=True)


# Burn-in applied to SwayProcess before Theta(0) is read off.
#
# The manuscript gives no initial condition for Theta(0), so the neutral choice
# is a draw from the process's own STATIONARY distribution. The driver used to
# take 5 steps under a comment reading "let the sway settle", but the sway pole
# is a = exp(-2 pi f_lo T_u) = 0.993737 with a relaxation time of 160 samples:
# after 5 steps the state still carries a variance deficit of a^10 = 0.939,
# i.e. only 6% of its stationary variance, and the mean radial offset comes out
# roughly 4x too small (0.0153 m instead of 0.0627 m at sigma_s = 0.05 m). That
# understates r_d, hence understates xi_eff degradation, hence flatters every
# ABER column. 2000 samples is 12.5 relaxation times and leaves a variance
# deficit of a^4000 = 1.2e-11. Chosen from the pole, not from any outcome; the
# correction moves the reported numbers in the PESSIMISTIC direction.
SWAY_BURN_IN = 2000


def draw_realizations(n_real: int, seed0: int):
    """Per-realization channel and sway draws, computed ONCE and shared by every
    ablation arm and both guard settings.

    This is what makes the A/B experiments paired. sigma_s was already common
    across cells (its rng is seeded per realization), but the scintillation
    measurement was not: the driver built one AR(1) generator per regime and
    called .step() n_real times inside EVERY cell, so cell 0 saw samples
    0..n-1, cell 1 saw n..2n-1, and so on. Arms were therefore compared on
    different channel draws, which adds between-arm variance that has nothing
    to do with the arms. Realization r now means the same channel state and the
    same sway state in all 30 cells.
    """
    sigmas, thetas = [], []
    for r in range(n_real):
        rng = np.random.default_rng(seed0 + r)
        s = SIGMAS[rng.integers(len(SIGMAS))]
        sway = SwayProcess(s, seed=seed0 + r)
        for _ in range(SWAY_BURN_IN):
            sway.step()
        sigmas.append(s)
        thetas.append(sway.theta.copy())
    # One MNLT scintillation generator per regime, built once: the AR(1)
    # correlation calibration in its constructor is the expensive part and it
    # does not depend on the realization. The marginal has unit mean by
    # construction (Appendix A), so the sensed latent state is centred.
    #
    # This module used to import GammaGammaAR1 and never call it, while the
    # docstring above listed "the MNLT channel of Appendix A" as one of the
    # components the assembly exercises. Nothing sensed the channel, so the
    # Kalman predictor inside BeamSteeringMPC ran from its initial state
    # forever and every horizon forecast was identically zero.
    hmeas = {}
    for reg, (A, B) in REGIMES.items():
        ch = GammaGammaAR1(A, B, seed=seed0)
        hmeas[reg] = [ch.step() - 1.0 for _ in range(n_real)]
    return sigmas, thetas, hmeas


def run(n_real: int, out_dir: str, seed0: int = 20260826, mode: str = "faithful"):
    os.makedirs(out_dir, exist_ok=True)
    gbar = 10 ** (GBAR_DB / 10)
    okw = objective_kwargs(mode)
    rows = []
    t0 = time.time()

    per_real: dict = {}      # (regime, guard_form) -> (sigma_s[], hit[]) for "full"

    sigmas, thetas, hmeas = draw_realizations(n_real, seed0)

    ceiling = feasibility_ceiling(sigmas, thetas)
    print("  Feasibility ceiling -- best system ABER attainable by ANY beam in "
          "the box,\n  at the median r_d each sway level produced. This bounds "
          "every rate below.")
    for k in sorted(ceiling):
        c = ceiling[k]
        print("    %-9s sigma_s=%.2f  r_d=%.4f m  best P_e,sys %.4e  reachable %s"
              % (c["regime"], c["sigma_s"], c["median_r_d_m"],
                 c["best_system_aber"], c["target_reachable"]), flush=True)
    reach = sum(c["target_reachable"] for c in ceiling.values())
    print("  -> %d of %d (regime, sigma_s) cells admit ANY passing beam.\n"
          % (reach, len(ceiling)), flush=True)

    # ---- the two things that make the guard A/B readable -----------------
    overlap = guard_test_overlap()
    print("  In-loop guard tests (i) and (ii), scanned over the whole box "
          "(%d waists):" % overlap["scanned"])
    print("    candidates rejected by (i) that (ii) would have admitted: %d"
          % overlap["rejected_only_by_i"])
    if overlap["tests_are_equivalent"]:
        print("    -> (i) and (ii) are the SAME filter here: the ladder returns")
        print("       order -1 for z > z_max and the kernel returns NaN for order")
        print("       -1. The manuscript agrees -- test (i) 'is the same quantity")
        print("       that selects the order K on the adaptive-fidelity ladder'.")
        print("       An A/B on (i) alone therefore cannot resolve anything, which")
        print("       is why the guard forms below differ in test (iii) instead.")
    print()

    print("  eq:z_worst -- z of the widest beam the box allows, %s regime, %.0f dB."
          % ("strong", GBAR_DB))
    print("    The manuscript prints 1.12 / 4.52 / 17.97 / 40.43 at sigma_s = "
          "0.05 / 0.1 / 0.2 / 0.3 m")
    print("    and concludes that beyond sigma_s ~ 0.1 m the two guard forms "
          "differ.")
    zw = {}
    for s in SIGMAS:
        z, w = z_worst_in_box(*REGIMES["strong"], s)
        zw["%.2f" % s] = dict(z_worst=z, w_z_max=w, admissible=bool(z <= Z_MAX))
        print("    sigma_s=%.2f  w_z_max=%.4f m  z_worst=%7.2f  admissible=%s"
              % (s, w, z, z <= Z_MAX))
    print()

    for regime, (A, B) in REGIMES.items():
        for name, over in ABLATIONS.items():
            for guard_form in GUARD_FORMS:
                cfg = SolverConfig(**over)
                sel, best_f, rej, invalid = [], [], 0, 0
                n_z_binding = 0
                sys_aber, sig_of = [], []       # system-level column, per realization
                act_aber = []                   # after the command-level guard
                n_override = 0
                binding_counts: dict = {}
                for r in range(n_real):
                    sigma_s = sigmas[r]
                    theta0 = thetas[r]
                    r_d = float(np.linalg.norm(theta0)) * LINK_LENGTH
                    h_meas = hmeas[regime][r]

                    # The IN-LOOP filter is identical in both arms, because the
                    # manuscript makes it identical: test (i) is the only test it
                    # applies per candidate, and in this implementation test (i)
                    # is the fidelity ladder (see guard_test_overlap above). The
                    # arms differ at the COMMAND level, below, which is where
                    # Sec. VI-C says the guard is applied.
                    mpc = BeamSteeringMPC(A, B, sigma_s, gbar, seed=seed0 + r,
                                          three_part_guard=True, config=cfg,
                                          **okw)
                    # pass the full pointing state, not just its radius: with
                    # steering enabled the horizon propagates Theta itself
                    res = mpc.step(theta0.copy(), h_meas=h_meas)
                    rej += res.rejected_by_guard
                    n_z_binding += mpc.guard_stats["z"]
                    w_pub = res.best_x[0] if res.best_x is not None else None
                    q = true_quality(A, B, w_pub, sigma_s, gbar)
                    sel.append(q)
                    # the manuscript's own quantity, on the same published beam
                    sys_aber.append(system_quality(A, B, w_pub, sigma_s, r_d))
                    sig_of.append(sigma_s)
                    best_f.append(res.best_f)
                    if np.isfinite(res.best_f) and not (0.0 <= res.best_f <= 0.5):
                        invalid += 1

                    # ---- Sec. VI-C guard, on the command about to be published
                    adm, binding, _diag = command_guard(
                        guard_form, A, B, w_pub, sigma_s, r_d)
                    for t in binding:
                        binding_counts[t] = binding_counts.get(t, 0) + 1
                    if adm:
                        act_aber.append(sys_aber[-1])
                    else:
                        n_override += 1
                        act_aber.append(system_quality(
                            A, B, xi_safe_wz(sigma_s), sigma_s, r_d))

                sel = np.array(sel, dtype=float)
                bf = np.array(best_f, dtype=float)
                bok = np.isfinite(bf)
                ok = np.isfinite(sel)
                # np.nanmedian/np.nanmin on an all-NaN slice warn and return
                # NaN; guard the call rather than let a RuntimeWarning leak to
                # stderr in the middle of the results table.
                med = float(np.median(sel[ok])) if ok.any() else None
                bst = float(np.min(sel[ok])) if ok.any() else None

                # ---- the manuscript's success criterion -------------------
                sa = np.array(sys_aber, dtype=float)
                sg = np.array(sig_of, dtype=float)
                sok = np.isfinite(sa)
                # A cycle that produced no admissible command cannot satisfy a
                # target, so it counts as a failure rather than being dropped:
                # dropping it would score the arms on different denominators.
                hit = sok & (sa <= ABER_TARGET)
                k_hit, n_tot = int(hit.sum()), int(sa.size)
                ci_lo, ci_hi = clopper_pearson(k_hit, n_tot)
                if name == "full":
                    per_real[(regime, guard_form)] = (sg.copy(), hit.copy())
                # per-sigma breakdown. The pooled rate mixes four sway levels
                # drawn uniformly, and the criterion is close to a step function
                # in sigma_s, so the pooled number is largely a property of the
                # draw; this column is what actually resolves.
                by_sigma = {}
                for s in SIGMAS:
                    m = sg == s
                    by_sigma["%.2f" % s] = dict(
                        n=int(m.sum()),
                        successes=int((hit & m).sum()),
                        rate=(float(np.mean(hit[m])) if m.any() else None),
                        median_system_aber=(float(np.median(sa[m & sok]))
                                            if (m & sok).any() else None))

                # ---- the actuated beam: after the command-level guard -------
                # The published command is what the OPTIMIZER found; the actuated
                # beam is what the loop would really emit, which is the published
                # command when its arm's guard admits it and the xi_safe override
                # otherwise. Both are reported. The manuscript's criterion says
                # "the optimizer discovered a beam configuration satisfying ...",
                # which is the published column; the actuated column is what the
                # link would actually have seen, and the guard can only move a
                # cycle from the first to the second, never back.
                aa = np.array(act_aber, dtype=float)
                aok = np.isfinite(aa)
                hit_act = aok & (aa <= ABER_TARGET)
                k_act = int(hit_act.sum())
                ci_a_lo, ci_a_hi = clopper_pearson(k_act, n_tot)

                rows.append(dict(
                    regime=regime, variant=name,
                    guard=guard_form,
                    n=n_real,
                    median_selected_aber=med,
                    best_selected_aber=bst,
                    # The quantity the ablation arms actually compete on: the
                    # objective value the solver reached. median_selected_aber
                    # is a post-hoc re-scoring of only the first stage, so two
                    # arms that found different trajectories can still tie on
                    # it. If the arms differ at all, they differ here first.
                    median_objective=float(np.median(bf[bok])) if bok.any() else None,
                    mean_objective=float(np.mean(bf[bok])) if bok.any() else None,
                    solved_fraction=float(np.mean(bok)),
                    cycles_with_invalid_optimum=invalid,
                    pct_invalid=100.0 * invalid / n_real,
                    candidates_rejected_by_guard=int(rej),
                    # How often the in-loop test (i), z <= z_max, was the test
                    # that did the rejecting. Identical across guard arms by
                    # construction now: the in-loop filter is common to both
                    # (see the module docstring), and guard_test_overlap in the
                    # run header shows why an A/B on it could resolve nothing.
                    candidates_rejected_by_z_test=int(n_z_binding),
                    # ---- COMMAND-LEVEL GUARD, Sec. VI-C -------------------
                    # The arms differ here and only here.
                    guard_form=guard_form,
                    guard_eps_safe=EPS_SAFE,
                    guard_z_max=Z_MAX,
                    commands_overridden=int(n_override),
                    commands_overridden_rate=n_override / float(n_real),
                    guard_binding_test_counts=dict(binding_counts),
                    # ---- SYSTEM LEVEL: the quantity the manuscript defines --
                    # post-EGC ABER of the published beam through
                    # eq:mimo_egc_aber over the 4x4 combined channel, at the
                    # fixed reference SNR gbar_op. NOT comparable to the
                    # per-branch columns above, which are ~1e-1 at this SNR.
                    system_target=ABER_TARGET,
                    system_gbar_op_db=GBAR_OP_DB,
                    system_successes=k_hit,
                    system_success_rate=(k_hit / n_tot) if n_tot else None,
                    system_success_ci95_lo=ci_lo,
                    system_success_ci95_hi=ci_hi,
                    median_system_aber=(float(np.median(sa[sok])) if sok.any() else None),
                    best_system_aber=(float(np.min(sa[sok])) if sok.any() else None),
                    system_by_sigma_s=by_sigma,
                    # the same criterion applied to the beam actually actuated
                    actuated_successes=k_act,
                    actuated_success_rate=(k_act / n_tot) if n_tot else None,
                    actuated_success_ci95_lo=ci_a_lo,
                    actuated_success_ci95_hi=ci_a_hi,
                    median_actuated_aber=(float(np.median(aa[aok]))
                                          if aok.any() else None)))
                print("  %-9s %-17s %-14s obj %s  branch %s  SYS %s  succ %5.1f%% "
                      "[%.1f,%.1f]  act %5.1f%%  ovr %5.1f%%  solved %.2f"
                      % (regime, name, guard_form,
                         "%.6e" % rows[-1]["median_objective"]
                         if rows[-1]["median_objective"] is not None else "    n/a     ",
                         "%.3e" % med if med is not None else "   n/a   ",
                         "%.3e" % rows[-1]["median_system_aber"]
                         if rows[-1]["median_system_aber"] is not None else "   n/a   ",
                         100.0 * k_hit / n_tot, 100.0 * ci_lo, 100.0 * ci_hi,
                         100.0 * k_act / n_tot, 100.0 * n_override / n_real,
                         rows[-1]["solved_fraction"]), flush=True)

    # ---- the success rate under every scoping, including the bad ones ----
    scopings = success_by_scoping(per_real, ceiling)
    print("\n  SUCCESS RATE BY SCOPING (full kernel). Every scoping evaluated is")
    print("  listed, with the manuscript sentence that does or does not license")
    print("  it and with the brute-force ceiling over the same cells.")
    print("  %-42s %9s %9s %s"
          % ("scoping", "ceiling", "measured", "licensed by the manuscript?"))
    print("  " + "-" * 96)
    def _pct(v):
        # a scoping whose cells were never drawn has no rate. Printing 0.0%
        # there would read as "measured zero", which is a different statement.
        return "   n/a" if v is None else "%5.1f%%" % (100.0 * v)

    for sc in SCOPINGS:
        e = scopings[sc["key"]]
        g = e["by_guard"].get("three_part") or list(e["by_guard"].values())[0]
        print("  %-42s %8s  %8s   %s"
              % (e["label"], _pct(e["ceiling_rate"]), _pct(g["rate"]),
                 e["manuscript_support"].split(".")[0]))
        if g["n"] == 0:
            print("      no realization was drawn in these cells -- nothing measured")
            continue
        print("      %d/%d cells admit any passing beam;  %d/%d cycles succeeded, "
              "95%% CI [%.1f, %.1f]%%"
              % (e["cells_reachable"], e["cells_total"], g["successes"], g["n"],
                 100.0 * g["ci95_lo"], 100.0 * g["ci95_hi"]))
    print()

    with open(os.path.join(out_dir, "reference_campaign.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            note=("Reference-implementation output. NOT a reproduction of Tables 9-12. "
                  "TWO different quantities are reported per row. The per-branch "
                  "columns (median_selected_aber, best_selected_aber, "
                  "median_objective) are what the solver ranks by; they are of "
                  "order 1e-1 at this SNR and are NOT comparable to a 1e-6 target. "
                  "The system_* columns are the quantity the manuscript's success "
                  "criterion defines: post-EGC system ABER of the published "
                  "stage-0 beam through eq:mimo_egc_aber over the 4x4 combined "
                  "channel, at gbar_op, with xi_eff carrying the measured "
                  "pointing offset; computed by system_metric.py. Even so, "
                  "system_success_rate is NOT the published rate: the campaign "
                  "protocol is unspecified in the manuscript and the protocol "
                  "used here (one cold-started MPC cycle per realization, "
                  "sigma_s drawn uniformly from the four swept levels, "
                  "Theta(0) from the settled sway process) was chosen by this "
                  "release. The criterion is close to a step function in "
                  "sigma_s, so the pooled rate is largely a property of that "
                  "uniform draw -- read system_by_sigma_s, not the pooled "
                  "number. Cycles that produced no admissible command are "
                  "counted as failures, not dropped. Note also that this implementation does "
                  "NOT reproduce the ablation ordering of Table 11: several components "
                  "make little difference here, and removing chaotic initialisation can "
                  "even help. That is most likely a property of the choices this "
                  "implementation had to make where the paper does not specify them "
                  "(slew limit, penalty weight lambda_u, decision box, stage weighting) "
                  "rather than evidence about the published campaign. "
                  "GUARD A/B. The two guard rows are the two forms Sec. VI-C "
                  "describes -- three_part = (i) z<=z_max, (ii) 0<=Pe_branch<=1/2, "
                  "(iii) Pe_sys<eps_safe; threshold_only = (iii) alone, which is "
                  "the form the manuscript states the reported campaign ran "
                  "under -- applied to the command about to be published, which "
                  "is where Sec. VI-C applies them. Test (iii) is evaluated at "
                  "SYSTEM level because the manuscript places it there. The "
                  "in-loop filter is common to both arms, as it is in the "
                  "manuscript. A previous version swept (i)+(ii) against (ii) "
                  "inside the swarm loop: neither of those is a configuration "
                  "the manuscript describes, and guard_test_overlap shows they "
                  "are the same filter, so that A/B measured nothing. "
                  "NOT REPRODUCIBLE FROM THIS RELEASE: the manuscript's guard "
                  "experiment is two kernels x two guard forms; the interpolated "
                  "kernel is not plumbed into this MPC loop, so the kernel axis "
                  "is absent here and is exercised separately by campaign2.py."),
            objective_mode=mode, objective_kwargs=okw,
            realizations=n_real, gbar_db=GBAR_DB, sigma_s_swept=SIGMAS,
            sway_burn_in=SWAY_BURN_IN,
            mean_r_d_m=float(np.mean([np.linalg.norm(t) for t in thetas])
                             * LINK_LENGTH),
            paired_draws=("sigma_s, Theta(0) and the sensed scintillation "
                          "state are identical across all 30 cells, so the "
                          "arms are compared on the same realizations"),
            feasibility_ceiling=ceiling,
            feasibility_note=("best system ABER attainable by ANY beam in the "
                              "manuscript box at the median r_d of each sway "
                              "level, by brute-force scan. Read every "
                              "system_success_rate against this: where "
                              "target_reachable is false, no algorithm can "
                              "score a success and 0% is a statement about the "
                              "channel, not about the solver."),
            guard_test_overlap=overlap,
            z_worst_strong=zw,
            success_rate_by_scoping=scopings,
            scoping_note=("The success rate under each way of restricting the "
                          "sweep, with the manuscript sentence that does or "
                          "does not license the restriction. Reported in full, "
                          "including the scopings that come out badly. A "
                          "scoping is legitimate only if the manuscript "
                          "independently justifies it, never because of the "
                          "number it produces."),
            seconds=round(time.time() - t0, 1), results=rows), f, indent=2)

    # CSV: flatten the per-sigma breakdown into scalar columns. A dict written
    # through str() embeds commas and silently corrupts the row.
    def flat(r):
        d = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for s, b in r["system_by_sigma_s"].items():
            d["sys_n_s%s" % s] = b["n"]
            d["sys_succ_s%s" % s] = b["successes"]
            d["sys_rate_s%s" % s] = b["rate"]
            d["sys_med_aber_s%s" % s] = b["median_system_aber"]
        return d

    frows = [flat(r) for r in rows]
    hdr = list(frows[0].keys())
    with open(os.path.join(out_dir, "reference_campaign.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(hdr) + "\n")
        for r in frows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in hdr) + "\n")
    print("\n  wrote %s  (%.1f s)" % (out_dir, time.time() - t0))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations", type=int, default=300)
    ap.add_argument("--objective", choices=("faithful", "legacy"), default="faithful",
                    help="which objective the ablation arms compete on; see "
                         "objective_kwargs() and landscape_probe.py")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "data", "07_reference_campaign"))
    a = ap.parse_args()
    print("Reference-implementation campaign: %d realizations per cell, "
          "objective=%s\n" % (a.realizations, a.objective))
    run(a.realizations, os.path.abspath(a.out), mode=a.objective)
