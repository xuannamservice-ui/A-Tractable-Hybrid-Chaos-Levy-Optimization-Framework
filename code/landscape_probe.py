"""
Landscape diagnostic for the MPC objective the ablation arms are searching.

WHY THIS EXISTS
    The ablation arms of `run_campaign.py` (chaos / Levy / GA / fidelity) come
    out nearly identical.  Before touching any solver constant, this script asks
    the prior question: *what landscape are those arms searching?*  Chaotic
    initialisation buys ergodic coverage of distant basins; Levy flight buys
    escape from traps; GA refinement buys local polish.  If the objective has
    one basin, none of those three mechanisms has anything to act on and every
    arm must return the same point -- a fact about the objective, not about the
    algorithm, and one no amount of parameter tuning can change.

    The manuscript is explicit about where the multimodality is supposed to come
    from (Sec. II, restated in the Discussion):

        "The coupling establishes an interior optimum and a per-stage cost
         unimodal in xi; the multi-modality motivating a population-based solver
         instead arises in the 60-dimensional receding-horizon trajectory space,
         where the slew-rate constraint chains stages together and xi_eff depends
         nonlinearly on the two-axis steering error."

    So the paper concedes the per-stage cost is unimodal and names TWO coupling
    mechanisms that are supposed to supply the ruggedness.  This script measures
    whether either is live in the implemented objective.

WHAT IT MEASURES (no tuning; no published value is referenced anywhere)
    1. admissibility of the decision box: what fraction the fidelity ladder
       accepts (z <= z_max) and where the cliff sits
    2. the per-stage cost on a dense 1-D grid: strict local minima, argmin
    3. whether the predicted-state stage weight can move the per-stage argmin
       (if it cannot, every stage wants the same value, the chain coupling is
       inactive at the optimum, and the trajectory problem collapses to 1-D)
    4. degeneracy audit: does the objective reward driving stages OUT of the
       admissible band?  (an inadmissible stage must not be cheaper than an
       admissible one)
    5. the full guarded trajectory objective: multistart pattern-search descent
       from many feasible starts, minima clustered to count distinct basins
    6. slew activity at the located optimum, and the per-stage spread of the
       optimal trajectory (a constant trajectory means the horizon is inert)
    7. random-line ruggedness: strict local minima along random feasible chords

    The same code measures the divergence-only objective and, once present, the
    manuscript's 60-dimensional (xi, theta_az, theta_el) objective, so the two
    landscapes are compared under identical instrumentation.

Usage:
    python landscape_probe.py [--starts 120] [--out ../data/08_landscape_probe]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mpc_loop import BeamSteeringMPC, envelope_guard

# ----------------------------------------------------------------- settings
#
# Operating point, taken from the manuscript and not chosen to produce an effect:
#   * strong turbulence (alpha, beta) = (1.2, 1.1) -- Table 1, and the regime
#     Tables 9 and 11 are reported at;
#   * gbar_op = 38 dB -- the reference SNR the success criterion is defined at,
#     stated to be identical for every algorithm and every ablated variant;
#   * sigma_s = 0.1 m -- sigma_{s,nom}, the only jitter level the manuscript
#     calls nominal.  The manuscript never states which of the four swept levels
#     Tables 9/11 used, so the sweep below repeats the headline basin count at
#     all four levels and the conclusion does not rest on this choice.
ALPHA, BETA = 1.2, 1.1
GBAR_DB = 38.0
SIGMA_S = 0.1
SIGMAS_ALL = [0.05, 0.1, 0.2, 0.3]
HORIZON = 20

# "Are these two located minima the same minimum?"  Tolerances are set from
# numerical conditioning, not from a desired basin count: F_TOL is far above the
# double-precision floor of the cost and far below any cost difference that
# would matter; X_TOL is 1e-3 of the box diagonal, far above a pattern-search
# convergence radius and far below any separation that is physically distinct.
F_TOL_REL = 1e-8
X_TOL_FRAC = 1e-3


# ------------------------------------------------------------------ helpers
def guarded_objective(mpc: BeamSteeringMPC, state, h_pred, with_repair: bool):
    """The objective, guard-rejected and non-finite candidates mapped to +inf
    exactly as `HCLPSOGA.minimise` does.

    Two landscapes are distinguished and BOTH are reported, because they answer
    different questions and conflating them would let an artefact masquerade as
    ruggedness:

      with_repair=False -- the objective ON THE FEASIBLE POLYTOPE.  Infeasible
        points are +inf walls.  Minima counted here are minima of the
        constrained problem the manuscript states, independent of how any
        solver chooses to restore feasibility.

      with_repair=True -- the objective COMPOSED WITH the feasibility repair,
        i.e. what the swarm actually traverses.  The repair is a many-to-one
        projection, so it can fold distinct box points onto the same feasible
        point and create local minima that belong to the projection rather than
        to the objective.  Reported separately for exactly that reason.
    """

    def batched(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if with_repair:
            X = mpc.repair(X)
        c, aux = mpc._objective(X, state, h_pred)
        rep = envelope_guard(aux["z"], aux["pe_first"], three_part=mpc.three_part)
        return np.where(np.isfinite(c) & rep.admissible, c, np.inf)

    return batched


def count_local_minima(y) -> int:
    """Strict interior local minima of a 1-D sampled curve (+inf/NaN ignored)."""
    y = np.asarray(y, dtype=float)
    idx = np.where(np.isfinite(y))[0]
    if idx.size < 3:
        return 0
    v = y[idx]
    return int(np.sum((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:])))


def poll_directions(dim, blocks):
    """Pattern-search poll set: every coordinate, plus the all-ones direction
    of each variable block.

    The block directions are not a heuristic tweak -- the hard slew limit makes
    the feasible set a narrow tube around constant trajectories, so a purely
    coordinate-wise poll can only crawl.  Adding the block direction lets the
    poll translate a whole trajectory, which is the move the tube actually
    permits.  Including it can only find MORE distinct minima, never fewer.
    """
    D = []
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        D.append(e)
    for (s, e) in blocks:
        b = np.zeros(dim)
        b[s:e] = 1.0
        D.append(b)
    return np.array(D)


def pattern_search(f, x0, lo, hi, blocks, scale, rng, h0=0.25, hmin=1e-13,
                   max_polls=6000, n_rand=48):
    """Generalised pattern search: robust to the +inf walls the guard creates.

    Quasi-Newton methods cannot descend through a +inf wall and Nelder-Mead
    collapses on one, so a poll-based method is the appropriate local optimiser.
    The poll set is the coordinate directions, the per-block all-ones directions
    (see `poll_directions`), and `n_rand` fresh random unit directions each poll.

    The random directions are not cosmetic.  With the fixed set alone, a run in
    60 dimensions terminates at points that random feasible perturbations can
    still improve -- i.e. at poll-set stalls, not at local minima -- which would
    inflate any basin count.  Redrawing directions makes the terminal point a
    local minimum with respect to a direction set that is dense in the limit.

    The defaults were set by a convergence study, NOT by tuning toward any
    outcome: on the 60-D objective, (n_rand, max_polls) = (24, 3000) leaves
    36/1200 random perturbations of the terminal point still improving and a
    residual spread of 9e-6 across starts, while (48, 6000) leaves 1/1200 and a
    spread of 6e-8.  The measurement is reported at the converged setting
    because an under-converged descent manufactures spurious basins.
    """
    x = np.clip(np.asarray(x0, float), lo, hi)
    fx = float(f(x[None, :])[0])
    Dfix = poll_directions(x.size, blocks)
    h = h0
    polls = 0
    while h > hmin and polls < max_polls:
        R = rng.normal(size=(n_rand, x.size))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        D = np.vstack([Dfix, R])
        step = h * scale
        cand = np.clip(np.vstack([x + step * D, x - step * D]), lo, hi)
        fc = f(cand)
        j = int(np.argmin(fc))
        polls += 1
        if np.isfinite(fc[j]) and fc[j] < fx:
            x, fx = cand[j], float(fc[j])
        else:
            h *= 0.5
    return x, fx


def cluster_minima(xs, fs, span):
    order = np.argsort(fs)
    xs, fs = xs[order], fs[order]
    xtol = X_TOL_FRAC * np.linalg.norm(span)
    rx, rf, cnt = [], [], []
    for x, fv in zip(xs, fs):
        hit = -1
        for j in range(len(rx)):
            if (np.linalg.norm((x - rx[j]) / span) <= X_TOL_FRAC * np.sqrt(x.size)
                    or abs(fv - rf[j]) <= F_TOL_REL * max(abs(rf[j]), 1e-300)):
                hit = j
                break
        if hit >= 0:
            cnt[hit] += 1
        else:
            rx.append(x)
            rf.append(fv)
            cnt.append(1)
    return np.array(rx), np.array(rf), np.array(cnt)


def draw_start(rng, mpc, lo, hi, batched, n_try=400):
    """A feasible start covering the whole slew tube, not just its axis.

    Uniform draws over the raw box are essentially never feasible: the slew
    limit makes the feasible set a thin tube.  But the tube is NOT just a
    neighbourhood of constant trajectories -- over T = 20 stages a trajectory
    may travel 19 slew steps, which for the divergence block spans the entire
    decision box.  Seeding only near-constant trajectories (an earlier version
    of this function) samples a vanishing slice of the feasible set and can
    miss whole basins: it reported a single minimum on a landscape where a
    ramping trajectory was 17% better.

    Three families are therefore mixed in equal proportion -- constant plus
    ripple, linear ramp at a random slope, and a bounded random walk -- and the
    result is projected onto the feasible set.
    """
    dim = mpc.decision_dim
    for _ in range(n_try):
        x = np.empty(dim)
        for (s, e), lim in zip(mpc.blocks(), mpc.block_slew()):
            T = e - s
            level = rng.uniform(lo[s], hi[s])
            kind = rng.integers(3)
            if kind == 0:                                   # constant + ripple
                x[s:e] = level + rng.uniform(-1, 1, T) * 1e-3 * (hi[s] - lo[s])
            elif kind == 1:                                 # linear ramp
                x[s:e] = level + rng.uniform(-lim, lim) * np.arange(T)
            else:                                           # bounded random walk
                x[s:e] = level + np.concatenate(
                    [[0.0], np.cumsum(rng.uniform(-lim, lim, T - 1))])
        x = mpc.repair(x[None, :])[0]
        if np.isfinite(batched(x[None, :])[0]):
            return x
    return None


# --------------------------------------------------------------------- probe
def multistart(batched, mpc, lo, hi, span, n_starts, rng):
    """Multistart pattern-search descent; returns located minima and stats."""
    xs, fs, fails = [], [], 0
    for _ in range(n_starts):
        x0 = draw_start(rng, mpc, lo, hi, batched)
        if x0 is None:
            fails += 1
            continue
        xb, fb = pattern_search(batched, x0, lo, hi, mpc.blocks(), span, rng)
        if np.isfinite(fb):
            xs.append(xb)
            fs.append(fb)
    return np.array(xs), np.array(fs), fails


def summarise_minima(xs, fs, span, prefix, o):
    if len(fs) == 0:
        return None
    rx, rf, cnt = cluster_minima(xs, fs, span)
    rel = (fs - rf[0]) / max(abs(rf[0]), 1e-300)
    o[prefix + "distinct_minima"] = int(len(rf))
    # minima that differ by less than 0.1% of the optimum are not distinguishable
    # by any search mechanism at this cost scale; counted separately so the
    # headline number cannot be inflated by convergence dust.
    o[prefix + "distinct_minima_1e-3"] = int(np.sum(
        (rf - rf[0]) / max(abs(rf[0]), 1e-300) > 1e-3) + 1)
    o[prefix + "global_min"] = float(rf[0])
    o[prefix + "minima_values_top5"] = [float(v) for v in rf[:5]]
    o[prefix + "minima_counts_top5"] = [int(v) for v in cnt[:5]]
    o[prefix + "frac_reaching_global"] = float(np.mean(rel <= 1e-6))
    o[prefix + "frac_within_1pct"] = float(np.mean(rel <= 1e-2))
    o[prefix + "worst_excess_over_global"] = float(rel.max())
    lb = o.get("separable_lower_bound")
    if lb is not None:
        o[prefix + "below_separable_bound"] = bool(rf[0] < lb * (1 - 1e-6))
    return xs[int(np.argmin(fs))]


def probe(mpc: BeamSteeringMPC, state, n_starts, label, rng):
    h_pred = mpc.kf.predict(mpc.horizon)
    batched = guarded_objective(mpc, state, h_pred, with_repair=False)
    batched_rep = guarded_objective(mpc, state, h_pred, with_repair=True)
    lo, hi = np.asarray(mpc.lower(), float), np.asarray(mpc.upper(), float)
    span = hi - lo
    blocks = mpc.blocks()
    o = {"label": label, "dim": int(mpc.decision_dim), "n_blocks": len(blocks)}

    # -- 1/2. admissibility and the per-stage cost along the first block ---
    ns = 4001
    grid = np.linspace(lo[0], hi[0], ns)
    base = np.asarray(mpc.centre(), float)
    Xg = np.tile(base, (ns, 1))
    s0, e0 = blocks[0]
    Xg[:, s0:e0] = grid[:, None]
    cg = batched(Xg)
    fin = np.isfinite(cg)
    o["block0_grid_points"] = ns
    o["block0_finite_frac"] = float(fin.mean())
    o["block0_local_minima"] = count_local_minima(cg)
    if fin.any():
        o["block0_argmin"] = float(grid[np.argmin(np.where(fin, cg, np.inf))])
        o["block0_min"] = float(np.min(np.where(fin, cg, np.inf)))
        o["block0_max_finite"] = float(np.max(cg[fin]))
        o["block0_dynamic_range"] = float(o["block0_max_finite"] / max(o["block0_min"], 1e-300))

    # admissibility of the box itself, measured on the raw per-stage kernel
    adm = mpc.admissible_fraction(grid, state)
    o["box_admissible_fraction"] = float(adm["fraction"])
    o["box_admissible_upper_edge"] = adm["upper_edge"]

    # -- 3. can the predicted-state stage weight move the argmin? ----------
    o["h_pred_first5"] = [float(v) for v in np.asarray(h_pred)[:5]]
    o["h_pred_std"] = float(np.std(h_pred))
    w = np.asarray(h_pred[:mpc.horizon], float)
    w = 1.0 + 0.5 * (w - w.mean()) / (w.std() + 1e-12)
    o["stage_weight_range"] = [float(w.min()), float(w.max())]
    o["stage_weight_uniform"] = bool(np.allclose(w, w[0]))
    o["h_enters_aber_argument"] = bool(getattr(mpc, "h_in_aber", False))
    # argmin of the first stage's own cost at each stage weight actually used
    args = mpc.per_stage_argmins(grid, state, h_pred)
    o["per_stage_argmins_range"] = [float(np.nanmin(args)), float(np.nanmax(args))]
    o["per_stage_argmins_spread"] = float(np.nanmax(args) - np.nanmin(args))

    # -- 4. degeneracy audit: is inadmissibility profitable? ---------------
    o["degeneracy"] = mpc.degeneracy_audit(state, h_pred)

    # -- 4b. separable lower bound -----------------------------------------
    # Each stage's cost depends on its own w_z, and the control penalty is
    # non-negative, so mean_k min_w Pe_k(w) lower-bounds the whole trajectory
    # objective.  Any located minimum below it is not a landscape feature but a
    # bad evaluation -- exactly the failure the manuscript warns about when a
    # truncated series is read outside its band and a one-sided test accepts the
    # result unopposed.  Reported so it cannot pass unnoticed.
    #
    # Pe is monotone increasing in the residual offset r_d (measured: strictly
    # increasing, 0 sign changes over 400 samples of r_d in [0, 2] m), so the
    # bound must be taken at the SMALLEST offset the decision vector can reach.
    # Without steering that is the measured offset, which is fixed; with
    # steering the optimiser can null the offset, so the bound is taken at
    # r_d = 0.  Using the measured offset in the steering case would produce a
    # bound the true optimum legitimately sits below.
    wg = np.linspace(lo[0], hi[0], 20001)
    rd_min = 0.0 if mpc.steering else None
    stage_mins = []
    for k in range(mpc.horizon):
        pe_k, z_k = mpc.per_stage_kernel(wg, state, h_pred, stage=k, r_d=rd_min)
        vk = np.where(np.isfinite(pe_k) & (z_k <= 8.0), pe_k, np.inf)
        stage_mins.append(float(vk.min()))
    o["separable_lower_bound"] = float(np.mean(stage_mins))

    # -- 5/6. multistart pattern-search descent ----------------------------
    # (i) on the feasible polytope: the constrained problem itself
    xs, fs, fails = multistart(batched, mpc, lo, hi, span, n_starts, rng)
    o["starts_requested"] = n_starts
    o["starts_infeasible"] = int(fails)
    o["descents_converged"] = int(len(fs))
    best = summarise_minima(xs, fs, span, "", o)

    # (ii) composed with the feasibility repair: what the swarm traverses
    xs2, fs2, _ = multistart(batched_rep, mpc, lo, hi, span, n_starts, rng)
    summarise_minima(xs2, fs2, span, "repaired_", o)

    if best is not None:
        # report the FEASIBLE trajectory, i.e. after repair, so the slew figures
        # describe a command the actuator could actually execute
        bf = mpc.repair(best[None, :])[0]
        o["optimum_blockwise"] = []
        for i, (s, e) in enumerate(blocks):
            d = np.abs(np.diff(bf[s:e]))
            lim = mpc.block_slew()[i]
            o["optimum_blockwise"].append(dict(
                block=i, name=mpc.block_names()[i],
                lo=float(bf[s:e].min()), hi=float(bf[s:e].max()),
                stage_spread=float(bf[s:e].max() - bf[s:e].min()),
                max_abs_step=float(d.max()) if d.size else 0.0,
                slew_limit=float(lim),
                slew_utilisation=float(d.max() / lim) if d.size and lim > 0 else 0.0))

    # -- 7. random-line ruggedness (on the feasible landscape) --------------
    n_lines, n_pts, per = 200, 801, []
    for _ in range(n_lines):
        a = draw_start(rng, mpc, lo, hi, batched)
        b = draw_start(rng, mpc, lo, hi, batched)
        if a is None or b is None:
            continue
        t = np.linspace(0, 1, n_pts)[:, None]
        per.append(count_local_minima(batched(a[None, :] * (1 - t) + b[None, :] * t)))
    if per:
        per = np.array(per)
        o["random_lines"] = int(per.size)
        o["random_line_points"] = n_pts
        o["random_line_minima_total"] = int(per.sum())
        o["random_line_minima_mean"] = float(per.mean())
        o["random_line_minima_max"] = int(per.max())
        o["random_line_frac_unimodal"] = float(np.mean(per <= 1))
    return o


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


def report(o):
    print("  decision dimension / blocks   : %d  (%d block(s))" % (o["dim"], o["n_blocks"]))
    print("  --- decision box admissibility (fidelity ladder z <= z_max) ---")
    print("  admissible fraction of box    : %.4f" % o["box_admissible_fraction"])
    print("  admissible upper edge         : %s" % o["box_admissible_upper_edge"])
    print("  --- predicted state ---")
    print("  h_pred[:5]                    : %s" % ["%.4g" % v for v in o["h_pred_first5"]])
    print("  h_pred std                    : %.4e" % o["h_pred_std"])
    print("  stage weight range            : [%.4f, %.4f]   uniform=%s"
          % (o["stage_weight_range"][0], o["stage_weight_range"][1], o["stage_weight_uniform"]))
    print("  h enters the ABER argument    : %s" % o["h_enters_aber_argument"])
    print("  per-stage argmin spread       : %.4e  (range %.6f .. %.6f)"
          % (o["per_stage_argmins_spread"], o["per_stage_argmins_range"][0],
             o["per_stage_argmins_range"][1]))
    print("  --- per-stage (1-D) cost, %d grid points ---" % o["block0_grid_points"])
    print("  strict local minima           : %d" % o["block0_local_minima"])
    print("  finite fraction of the line   : %.4f" % o["block0_finite_frac"])
    if "block0_argmin" in o:
        print("  argmin / min / max            : %.6f / %.6e / %.6e"
              % (o["block0_argmin"], o["block0_min"], o["block0_max_finite"]))
        print("  best-to-worst ratio           : %.3f" % o["block0_dynamic_range"])
    d = o["degeneracy"]
    print("  --- degeneracy audit ---")
    print("  cost of best all-admissible traj : %.6e" % d["best_admissible"])
    print("  cost of best guard-passing traj  : %.6e  (with %d/%d inadmissible stages)"
          % (d["best_overall"], d["nan_stages_at_best"], o["dim"] // max(o["n_blocks"], 1)))
    print("  inadmissibility pays by a factor : %.3f   -> %s"
          % (d["exploit_gain"], "DEGENERATE" if d["exploit_gain"] > 1.01 else "clean"))
    print("  separable lower bound            : %.10e" % o["separable_lower_bound"])
    print("  --- multistart pattern-search descent ---")
    print("  starts / infeasible / converged: %d / %d / %d"
          % (o["starts_requested"], o["starts_infeasible"], o["descents_converged"]))
    for pre, tag in (("", "ON THE FEASIBLE POLYTOPE (the constrained problem)"),
                     ("repaired_", "COMPOSED WITH THE REPAIR (what the swarm traverses)")):
        if pre + "distinct_minima" not in o:
            continue
        print("  %s" % tag)
        print("      distinct local minima     : %d   (separated by >0.1%%: %d)"
              % (o[pre + "distinct_minima"], o[pre + "distinct_minima_1e-3"]))
        print("      best value                : %.10e" % o[pre + "global_min"])
        print("      top-5 minima values       : %s"
              % ["%.6e" % v for v in o[pre + "minima_values_top5"]])
        print("      top-5 basin hit counts    : %s" % (o[pre + "minima_counts_top5"],))
        print("      frac of descents at global: %.3f" % o[pre + "frac_reaching_global"])
        print("      frac within 1%% of global  : %.3f" % o[pre + "frac_within_1pct"])
        print("      worst excess over global  : %.3e" % o[pre + "worst_excess_over_global"])
        print("      below separable bound?    : %s%s"
              % (o.get(pre + "below_separable_bound"),
                 "   <-- BAD EVALUATION, not a basin"
                 if o.get(pre + "below_separable_bound") else ""))
    if "optimum_blockwise" in o:
        print("  at the optimum (after repair), per block:")
        for b in o["optimum_blockwise"]:
            print("      %-10s stage-spread %.4e   slew %.4e/%.4e = %.1f%% of limit"
                  % (b["name"], b["stage_spread"], b["max_abs_step"], b["slew_limit"],
                     100.0 * b["slew_utilisation"]))
    if "random_lines" in o:
        print("  --- random-line ruggedness (%d lines x %d points) ---"
              % (o["random_lines"], o["random_line_points"]))
        print("  total / mean / max minima     : %d / %.3f / %d"
              % (o["random_line_minima_total"], o["random_line_minima_mean"],
                 o["random_line_minima_max"]))
        print("  fraction of lines with <= 1   : %.3f" % o["random_line_frac_unimodal"])


def initial_state(sigma_s, seed=20260826, settle=500):
    """Steady-state pointing state of the reference sway process.

    The manuscript gives no initial condition or distribution for Theta(0)
    (Sec. II-B defines the recursion but never initialises it), so the state is
    taken as the settled output of the sway model already in `channel.py`.
    """
    from channel import SwayProcess
    sw = SwayProcess(sigma_s, seed=seed)
    for _ in range(settle):
        sw.step()
    return sw.theta.copy()


def prime_predictor(mpc, alpha, beta, seed=20260826, burn=200):
    """Drive the Kalman predictor with the MNLT channel so its horizon forecast
    is not identically zero.

    `BeamSteeringMPC.step` never calls `kf.update` unless a measurement is
    supplied, and `KalmanAR1.x` initialises at 0, so an unprimed predictor
    returns h_hat(t+k) = 0 for every k.  The filter tracks the zero-mean latent
    scintillation state, so the measurement is h_a - 1 about the unit-mean
    turbulence factor.
    """
    from channel import GammaGammaAR1
    ch = GammaGammaAR1(alpha, beta, rho_a=0.98, seed=seed, calibrate=False)
    for _ in range(burn):
        mpc.kf.update(ch.step() - 1.0)
    return mpc


# The attribution ladder.  Each rung turns on exactly one specification item
# that the divergence-only objective was missing, so any change in the measured
# landscape is attributable to that item rather than to a bundle of edits.
VARIANTS = [
    ("A. as implemented: divergence-only, wide box, nansum",
     dict(steering=False, manuscript_box=False, strict_admissibility=False,
          h_in_aber=False), False),
    ("B. A + strict admissibility (an inadmissible stage is not free)",
     dict(steering=False, manuscript_box=False, strict_admissibility=True,
          h_in_aber=False), False),
    ("C. B + manuscript xi box  [max(0.5, xi_min), 4.888]",
     dict(steering=False, manuscript_box=True, strict_admissibility=True,
          h_in_aber=False), False),
    ("D. C + predicted state inside the ABER (gbar_k = gbar h_hat_k^2)",
     dict(steering=False, manuscript_box=True, strict_admissibility=True,
          h_in_aber=True), True),
    ("E. D + two-axis steering: the manuscript's 60-D trajectory",
     dict(steering=True, manuscript_box=True, strict_admissibility=True,
          h_in_aber=True), True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=120)
    ap.add_argument("--sweep-sigma", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "data", "08_landscape_probe"))
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    gbar = 10 ** (GBAR_DB / 10)
    rng = np.random.default_rng(20260826)
    theta0 = initial_state(SIGMA_S)
    results = []

    for label, kw, prime in VARIANTS:
        mpc = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, gbar, horizon=HORIZON,
                              seed=7, **kw)
        if prime:
            prime_predictor(mpc, ALPHA, BETA)
        banner(label)
        o = probe(mpc, theta0, a.starts, label, rng)
        report(o)
        results.append(o)

    banner("SUMMARY")
    print("  %-46s %4s %7s %7s %7s %7s %7s"
          % ("variant", "dim", "minima", ">0.1%", "at-glob", "adm.fr", "exploit"))
    for o in results:
        print("  %-46s %4d %7s %7s %7s %7.3f %7.3f"
              % (o["label"][:46], o["dim"], o.get("distinct_minima", "-"),
                 o.get("distinct_minima_1e-3", "-"),
                 "%.3f" % o["frac_reaching_global"] if "frac_reaching_global" in o else "-",
                 o["box_admissible_fraction"], o["degeneracy"]["exploit_gain"]))
    print("\n  'minima'  distinct local minima of the objective ON THE FEASIBLE SET")
    print("  '>0.1%'   of those, how many are more than 0.1%% worse than the best")
    print("  'at-glob' fraction of independent descents that reach the best value")
    print("  'exploit' factor by which driving stages OUT of the admissible band pays")

    if a.sweep_sigma:
        banner("basin count across the four swept jitter levels")
        sweep = []
        for sg in SIGMAS_ALL:
            th = initial_state(sg)
            for label, kw, prime in VARIANTS:
                mpc = BeamSteeringMPC(ALPHA, BETA, sg, gbar, horizon=HORIZON,
                                      seed=7, **kw)
                if prime:
                    prime_predictor(mpc, ALPHA, BETA)
                o = probe(mpc, th, max(30, a.starts // 4), label, rng)
                sweep.append(dict(sigma_s=sg, label=label,
                                  distinct_minima=o.get("distinct_minima"),
                                  frac_global=o.get("frac_reaching_global"),
                                  admissible_frac=o["box_admissible_fraction"],
                                  exploit_gain=o["degeneracy"]["exploit_gain"],
                                  line_minima_mean=o.get("random_line_minima_mean")))
                print("  sigma_s=%.2f  %-46s minima=%-4s frac_global=%-7s adm=%.3f exploit=%.2f"
                      % (sg, label[:46], sweep[-1]["distinct_minima"],
                         ("%.3f" % sweep[-1]["frac_global"]) if sweep[-1]["frac_global"] is not None else "-",
                         sweep[-1]["admissible_frac"], sweep[-1]["exploit_gain"]))
        results.append({"sigma_sweep": sweep})

    with open(os.path.join(out_dir, "landscape_probe.json"), "w", encoding="utf-8") as f:
        json.dump(dict(alpha=ALPHA, beta=BETA, gbar_db=GBAR_DB, sigma_s=SIGMA_S,
                       horizon=HORIZON, theta0=[float(v) for v in theta0],
                       results=results), f, indent=2)
    print("\n  wrote %s" % out_dir)


if __name__ == "__main__":
    main()
