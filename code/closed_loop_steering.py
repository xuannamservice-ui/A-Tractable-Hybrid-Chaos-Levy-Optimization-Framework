"""Does the beam actually get steered? The physical feedback loop, closed for the
first time in this project.

THE GAP THIS CLOSES, AND THE ONE IT DOES NOT

`bench_cycle.py`'s own docstring (line 444) is explicit and correct about why its
`make_traces` draws the pointing disturbance OPEN LOOP, before any cycle runs: "The
plant is deliberately OUTSIDE the timed window ... inventing a closed loop would change
what is being timed." That is a sound reason for a LATENCY measurement, and it is
disclosed. It is not a reason that generalises to every other measurement in this
project, and it has been read as one by omission: a full grep of this codebase finds no
script, anywhere -- not `bench_cycle.py`, not `measure_all.py`, not `run_campaign.py`,
not any of the Levy probes -- that ever calls `SwayProcess.step(u_ptr=...)` with a
nonzero argument. Eq. (5)'s recursion,

    Theta(t+1) = Theta(t) + Delta_sway(t) - u_ptr(t),

the plant dynamics every stage cost in this paper is written against, has therefore
never been run for real anywhere in the release: `u_ptr` defaults to zero and nothing
overrides it. Theorem 2, the paper's own delay-compensated stability certificate, is
consequently pure mathematics here -- nothing in the release has exercised the closed
recursion it describes, and no measurement in this paper can currently distinguish "the
controller computes plausible steering commands" from "the controller computes
commands that, if applied, would actually reduce pointing error over time." Those are
different claims and only the first has been tested.

This script tests the second, using the OFFICIAL, UNMODIFIED `bench_cycle.CycleRunner`
so the guard, fallback and publish logic under test is bit-for-bit what the flagship
latency tables are built on -- nothing about the controller is reimplemented or
approximated; only the plant's response to what it publishes is added.

TWO ARMS, PAIRED ON IDENTICAL SEEDS

    open     u_ptr = 0 always, fed to the plant. This reproduces exactly what every
             existing measurement in the project has actually been running: the
             controller computes, guards and publishes a command every cycle, but the
             physical disturbance evolves as if it never had. Its stationary spread is
             exactly sigma_theta = sigma_s / L, `SwayProcess`'s own uncontrolled figure.
    closed   u_ptr = the command `CycleRunner` actually publishes (post-guard,
             post-fallback, post-slew-clamp), fed back into `SwayProcess.step()` for
             the NEXT cycle. This is eq. (5), run.

Both arms share the CycleRunner RNG seed and the scintillation stream; only the plant's
response to the published command differs. If steering does what its name says, the
closed arm's stationary ||Theta|| should fall below the open arm's, and the residual
system ABER should fall with it. If it does not, that is the honest result and is
reported as such.

WHAT IS SCORED, AND UNDER WHAT CONVENTION

Cycle t's system ABER is scored against r_d(t) = L*||Theta(t)||, the offset BEFORE
cycle t's freshly-computed command lands -- what a receiver at that instant actually
sees, since eq. (5) writes the correction into Theta(t+1), not Theta(t). This is the
same one-cycle convention already implicit in the closed-loop dynamics of Sec. IV-A
(the command computed from the sensed state acts on the NEXT state).

CONFIGURATION

The primary run is at the manuscript's own nominal operating point, unmodified:
sigma_s = 0.10 m, strong turbulence, gbar_op = 38 dB -- `bench_cycle.CycleRunner`'s
hardcoded configuration, so no monkey-patching of its module constants is needed and
fidelity to the official evaluator is exact.

The anytime wall-clock checkpoint is OFF by default (`CycleRunner(anytime=False)`),
deliberately, for the same reason `levy_warmstart_probe.py` and `levy_episodic_probe.py`
run a deterministic T_iter budget: whether steering reduces pointing error is a question
about the algorithm, and tying it to this machine's real-time scheduling noise would
confound search quality with timing jitter that has nothing to do with the question
asked. This is a tracking-quality experiment, not a latency one; Section VI's real-time
claim is untouched by it and is not what this script measures. Pass --anytime to instead
run under the real-time checkpoint if the timing-coupled version is wanted.

Usage: python closed_loop_steering.py [--cycles 2000] [--burn-in 500] [--seeds 8] [--anytime]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_all as _ma
from bench_cycle import CycleRunner, SIGMA_S, ALPHA, BETA, T_U
from channel import SwayProcess, GammaGammaAR1
from measure_all import system_success
from mpc_loop import U_SLEW, U_MAX

ARMS = ("open", "closed")


def run_episode(seed, cycles, burn_in, arm, anytime=False):
    """One independent episode of the given arm. Returns per-cycle arrays.

    `run` (CycleRunner) supplies the controller side (identical construction and RNG
    seed for both arms). `sway` supplies the plant side; only whether its `u_ptr`
    argument carries the real published command or a hardwired zero differs between
    arms, which is the entire experimental manipulation.
    """
    sway = SwayProcess(SIGMA_S, T_u=T_U, seed=1000 + seed)
    ch = GammaGammaAR1(ALPHA, BETA, rho_a=0.98, seed=2000 + seed, calibrate=False)
    run = CycleRunner(mpc_seed=seed, anytime=anytime)

    theta = np.zeros(2)
    u_applied = np.zeros(2)          # what actually reaches the plant this step
    for _ in range(burn_in):
        theta = sway.step(u_ptr=(u_applied if arm == "closed" else np.zeros(2)))
        ch.step()

    r_d_hist, w_hist, aber_hist, admissible_hist, theta_norm_hist = [], [], [], [], []
    for _ in range(cycles):
        h = float(ch.step()) - 1.0
        r_d = float(np.linalg.norm(theta)) * 2000.0        # L = 2 km, matches LINK_LENGTH
        _, _, diag = run.cycle(theta.copy(), h)
        _iters, _evals, admissible, _best_f, _z_cmd, _pe_cmd = diag
        w_cmd = float(run.out_buf[0])
        u_published = np.array([run.out_buf[1], run.out_buf[2]], dtype=np.float64)

        r_d_hist.append(r_d)
        w_hist.append(w_cmd)
        theta_norm_hist.append(float(np.linalg.norm(theta)))
        admissible_hist.append(bool(admissible))
        aber_hist.append(system_success(w_cmd, SIGMA_S, r_d)[1])

        u_applied = u_published if arm == "closed" else np.zeros(2)
        theta = sway.step(u_ptr=u_applied)

    return dict(r_d=np.array(r_d_hist), w=np.array(w_hist),
               theta_norm=np.array(theta_norm_hist),
               admissible=np.array(admissible_hist), aber=np.array(aber_hist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=2000)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--anytime", action="store_true",
                    help="run under the real-time wall-clock checkpoint instead of a "
                         "deterministic iteration budget")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "14_compiled_poc"))
    a = ap.parse_args()

    sigma_theta_uncontrolled = SIGMA_S / 2000.0
    print("  sigma_s=%.2f m, strong turbulence, gbar_op=38 dB (bench_cycle.CycleRunner,"
          " unmodified)" % SIGMA_S)
    print("  uncontrolled stationary std of ||Theta_axis|| = sigma_s/L = %.4e rad/axis"
          % sigma_theta_uncontrolled)
    print("  %d episodes x %d cycles (%d-cycle burn-in), paired open/closed on identical seeds\n"
          % (a.seeds, a.cycles, a.burn_in))

    t0 = time.time()
    episodes = {arm: [] for arm in ARMS}
    for s in range(a.seeds):
        for arm in ARMS:
            episodes[arm].append(run_episode(90000 + 137 * s, a.cycles, a.burn_in, arm,
                                             anytime=a.anytime))
        print("  seed %d/%d done (%.0fs)" % (s + 1, a.seeds, time.time() - t0), flush=True)

    print("\n  %-8s %14s %14s %12s %10s %10s" %
          ("arm", "mean ||Theta||", "SD ||Theta||", "median ABER", "admiss.%", "guard-fb%"))
    summary = {}
    for arm in ARMS:
        tn = np.concatenate([e["theta_norm"] for e in episodes[arm]])
        ab = np.concatenate([e["aber"] for e in episodes[arm]])
        ad = np.concatenate([e["admissible"] for e in episodes[arm]])
        ab_ok = ab[np.isfinite(ab) & (ab > 0)]
        summary[arm] = dict(mean_theta_norm=float(np.mean(tn)), sd_theta_norm=float(np.std(tn)),
                            median_aber=float(np.median(ab_ok)) if ab_ok.size else float("nan"),
                            admissible_rate=float(np.mean(ad)),
                            n_cycles=int(tn.size))
        print("  %-8s %14.4e %14.4e %12.4e %9.1f%% %9.1f%%"
              % (arm, summary[arm]["mean_theta_norm"], summary[arm]["sd_theta_norm"],
                 summary[arm]["median_aber"], 100 * summary[arm]["admissible_rate"],
                 100 * (1 - summary[arm]["admissible_rate"])))

    reduction = 1.0 - summary["closed"]["mean_theta_norm"] / summary["open"]["mean_theta_norm"]
    aber_ratio = summary["closed"]["median_aber"] / summary["open"]["median_aber"]
    print("\n  closed-loop mean ||Theta|| vs open-loop: %+.1f%% (negative = steering shrinks it)"
          % (100 * -reduction if reduction >= 0 else 100 * -reduction))
    print("  closed/open median system-ABER ratio: %.4g" % aber_ratio)

    # paired, per-episode summary for a proper significance test
    from scipy.stats import wilcoxon
    med_open = np.array([np.median(e["theta_norm"]) for e in episodes["open"]])
    med_closed = np.array([np.median(e["theta_norm"]) for e in episodes["closed"]])
    d = med_closed - med_open
    p = float(wilcoxon(d, alternative="two-sided").pvalue) if a.seeds >= 6 else float("nan")
    print("\n  paired per-episode median ||Theta||: closed - open, Wilcoxon p = %.3g" % p)
    print("  closed lower in %d/%d episodes" % (int(np.sum(d < 0)), a.seeds))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "closed_loop_steering.json"), "w") as fh:
        json.dump(dict(what="first physical closure of eq.(5)'s pointing recursion in this "
                            "project: the command bench_cycle.CycleRunner actually publishes "
                            "(post-guard, post-fallback, post-slew-clamp) fed back into "
                            "SwayProcess.step(u_ptr=...), against an open-loop control arm "
                            "where the identical controller runs but its command never reaches "
                            "the plant -- which is what every prior measurement in this project "
                            "has actually been doing",
                       sigma_s=SIGMA_S, gbar_op_db=38.0, cycles=a.cycles, burn_in=a.burn_in,
                       seeds=a.seeds, sigma_theta_uncontrolled=sigma_theta_uncontrolled,
                       summary=summary, mean_theta_norm_reduction=float(reduction),
                       median_aber_ratio_closed_over_open=float(aber_ratio),
                       paired_episode_wilcoxon_p=p,
                       episode_medians=dict(open=med_open.tolist(), closed=med_closed.tolist())),
                 fh, indent=1)
    print("\n  wrote closed_loop_steering.json")


if __name__ == "__main__":
    main()
