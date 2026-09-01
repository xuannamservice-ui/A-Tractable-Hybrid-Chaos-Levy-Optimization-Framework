"""
End-to-end control-cycle latency measurement on ONE platform.

WHAT THIS REPLACES
------------------
The manuscript reports end-to-end cycle latency on three platforms it calls A
(Intel i7), B (Jetson AGX Xavier) and C (Cortex-A72), under three Linux
scheduling arms (SCHED_OTHER, chrt, isolcpus), with a median of 0.77-0.79 ms
against an 800 us budget and a joint real-time success rate of 78.0%.  None of
that is reproducible from this release.  This script measures the same pipeline
on the machine the release is actually being run on, and reports what the clock
says -- including where that is very much worse than the published figure.

Nothing here is synthesised.  Every latency number in the output file is a
difference of two time.perf_counter_ns() readings taken around code that ran.
The full per-cycle sample array is written to cycle_latency.npz so that any
percentile, the serial correlation, and the per-seed dispersion can all be
recomputed by a reader from the raw samples.

THE PIPELINE, AND EXACTLY WHAT EACH STAGE CONTAINS
--------------------------------------------------
  Sensing      read the measured pointing state Theta(t) and the measured
               latent scintillation sample h(t) from the acquisition buffer,
               build the state vector, range/finiteness check.
               NOT INCLUDED: any physical sensor.  There is no camera, no
               quadrant detector and no ADC on this machine, so this stage
               measures the software acquisition path only.  A real sensor's
               exposure and readout latency is not measurable here and is not
               claimed.

  Prediction   KalmanAR1.update() on the measured sample followed by
               .predict(T) over the T = 20 step horizon -- i.e. exactly what
               BeamSteeringMPC.step() does.
               NOT INCLUDED: the TCN branch of eq. (30).  The release does not
               implement one; mpc_loop.py states that on an AR(1) channel the
               inverse-variance weight collapses onto the Kalman branch
               (omega -> 1) and step() never evaluates the fusion rule.  Its
               cost is therefore absent from this measurement, and that is a
               gap, not a zero.

  Optimization construct the H-CLPSO-GA solver for this cycle and run
               minimise() with an anytime checkpoint that stops the search at
               tau_O = 600 us measured from the start of this stage.  The
               checkpoint is the one already present in hclpso_ga.minimise;
               it is polled at ITERATION BOUNDARIES, which is the only place
               the released solver offers to stop.

  Checks       the Sec. VI-C envelope guard applied to the command about to be
               published: test (i) z(u) <= z_max, test (ii) 0 <= Pe(u) <= 1/2,
               plus the actuator envelope tests |u| <= u_max and
               |u(k)-u(k-1)| <= u_dot_max*T_u that the publish stage must
               satisfy.  A command failing any test is replaced by the offline
               xi_safe override.
               NOT INCLUDED: test (iii), Pe_system(u) < eps_safe.  That is a
               post-EGC system-level quantity and the only implementations in
               this release are numerical convolutions.  Their cost is measured
               separately by --eps-safe-cost and reported, rather than being
               quietly dropped.

  Publish      form the actuator command from the first horizon stage,
               rate-limit and saturate it to the actuator envelope, write it to
               the output buffer and advance the sequence counter.

  End to end   t_after_Publish - t_before_Sensing.  It therefore also contains
               the four intermediate perf_counter_ns() calls; their measured
               cost is reported so a reader can subtract it if they want to.

THE HYBRID-CORE PROBLEM
-----------------------
This is an Intel Core i5-14600KF: 6 P-cores with SMT (logical 0-11) and 8
E-cores (logical 12-19).  The platform phase MEASURED an E-core/P-core cost
ratio of 2.38x on this workload's inner loop.  An unpinned latency distribution
is therefore a mixture of two processors whose per-evaluation cost differs by
more than a factor of two, and its tail is not interpretable.  Every pinned arm
here pins to ONE logical processor and verifies the pin two independent ways
(affinity read-back, and kernel32.GetCurrentProcessorNumber sampled while the
thread is kept runnable).  Each cycle also records, outside its own timed
window, which logical processor it finished on, so the unpinned arm's migration
is visible in the raw array rather than asserted.

THE SCHEDULING ARMS ARE AN ANALOGUE, NOT A REPRODUCTION
-------------------------------------------------------
SCHED_OTHER, chrt and isolcpus are Linux mechanisms.  They do not exist on
Windows and are not emulated here.  What is varied instead is the Windows
process priority class (NORMAL_PRIORITY_CLASS / HIGH_PRIORITY_CLASS /
REALTIME_PRIORITY_CLASS, set through SetPriorityClass via psutil.Process.nice)
combined with processor affinity (SetProcessAffinityMask).  These are different
mechanisms with different semantics -- in particular Windows gives no real-time
guarantee at any priority class, and hardware interrupts and DPCs preempt every
one of them.  The comparison below is labelled an analogue everywhere it
appears and must never be presented as a reproduction of the Linux arms.

USAGE
    python bench_cycle.py                       # all anytime arms
    python bench_cycle.py --arms normal_pinned_P
    python bench_cycle.py --arms converged_pinned_P
    python bench_cycle.py --eps-safe-cost       # cost of guard test (iii)
    python bench_cycle.py --summarise           # re-report from the saved npz

Results accumulate: each invocation merges its arms into the existing
cycle_latency.npz / cycle_latency.json rather than overwriting them.
"""
from __future__ import annotations

import os

# BLAS threading must be capped BEFORE numpy is imported, so that a pinned
# single-core measurement cannot be silently serviced by a thread pool that
# spans both core classes.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import ctypes
import json
import math
import platform
import sys
import time

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binom as binom_dist

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from channel import GammaGammaAR1, SwayProcess, beam_geometry, xi_effective
from hclpso_ga import HCLPSOGA, SolverConfig, ladder_order
from mpc_loop import (EPS_SAFE, T_U, TAU_O, U_MAX, U_SLEW, Z_MAX,
                      BeamSteeringMPC, envelope_guard, manuscript_wz_box,
                      wz_for_xi)
from rtodt_fast import pe_series_f64, z_of

OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "10_platform"))
NPZ_PATH = os.path.join(OUT_DIR, "cycle_latency.npz")
JSON_PATH = os.path.join(OUT_DIR, "cycle_latency.json")

# ---- operating point ------------------------------------------------------
# The strong-turbulence regime and the nominal link of the reported campaign
# (run_campaign.py: ALPHA/BETA = 1.2/1.1, GBAR_DB = 38, sigma_s = 0.1 m,
# L = 2 km, T = 20).  Timing is data dependent -- the fidelity ladder picks a
# per-candidate series order from z -- so the operating point is stated rather
# than left implicit.
ALPHA, BETA = 1.2, 1.1
GBAR_DB = 38.0
GBAR = 10.0 ** (GBAR_DB / 10.0)
SIGMA_S = 0.10
HORIZON = 20
XI_SAFE = 0.83                 # offline override beam, run_campaign.py

# ---- real-time specification ---------------------------------------------
TAU_A = 200e-6                 # actuator latency (mpc_loop.TAU_ACT)
BUDGET = T_U - TAU_A           # 800 us computation budget
BUDGET_NS = int(round(BUDGET * 1e9))
TAU_O_NS = int(round(TAU_O * 1e9))

# ---- which logical processors the pinned arms use -------------------------
# Logical 2 is thread 0 of P-core 1.  Logical processor 0 is deliberately
# avoided: Windows routes a disproportionate share of DPC/ISR work there, and
# the platform phase saw individual P-cores inflated by up to 73% by exactly
# that kind of bursty interference.  Logical 12 is E-core 0 (no SMT).
P_CPU = 2
E_CPU = 12

STAGES = ("sense", "predict", "optimize", "checks", "publish")

IS_WINDOWS = (os.name == "nt")
try:
    import psutil
except Exception:                                   # pragma: no cover
    psutil = None

if IS_WINDOWS:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcessorNumber.restype = ctypes.c_ulong


# =====================================================================
# 1.  clock, affinity, priority -- all verified, never assumed
# =====================================================================
def clock_resolution_ns(n=200000):
    """Smallest non-zero increment the clock actually shows, measured."""
    best = None
    prev = time.perf_counter_ns()
    for _ in range(n):
        now = time.perf_counter_ns()
        d = now - prev
        if d > 0 and (best is None or d < best):
            best = d
        prev = now
    return int(best)


def timer_overhead_ns(n=200000):
    """Cost of one perf_counter_ns() call, on whatever core we are pinned to.

    Measured as a long run of back-to-back calls divided by the count, and also
    as the minimum observed pairwise gap; both are reported because the first
    includes the loop and the second does not.
    """
    ts = np.empty(n, dtype=np.int64)
    for i in range(n):
        ts[i] = time.perf_counter_ns()
    d = np.diff(ts)
    return dict(n=int(n),
                mean_ns=float(d.mean()),
                median_ns=float(np.median(d)),
                min_ns=int(d.min()))


def current_processor_number():
    if IS_WINDOWS:
        return int(_k32.GetCurrentProcessorNumber())
    if hasattr(os, "sched_getcpu"):
        return int(os.sched_getcpu())
    return -1


def get_affinity():
    if psutil is not None:
        try:
            return sorted(psutil.Process().cpu_affinity())
        except Exception:
            pass
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def set_affinity(cpus):
    """Request an affinity mask.  Setting a mask is a request, not a guarantee
    of placement -- verify_pin() below checks placement separately."""
    if psutil is not None:
        try:
            psutil.Process().cpu_affinity(list(cpus))
            return True
        except Exception:
            pass
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
        return True
    if IS_WINDOWS:                       # ctypes fallback if psutil is absent
        mask = 0
        for c in cpus:
            mask |= (1 << int(c))
        h = _k32.GetCurrentProcess()
        return bool(_k32.SetProcessAffinityMask(ctypes.c_void_p(h),
                                                ctypes.c_size_t(mask)))
    return False


def verify_pin(cpu, n_probes=4000):
    """Two independent checks that a pin took effect:
       (a) read the affinity mask back from the OS;
       (b) ask the kernel, repeatedly and while the thread is kept runnable,
           which logical processor we are executing on.
    (a) alone is not evidence of execution placement."""
    mask = get_affinity()
    seen = {}
    sink = 0.0
    for i in range(n_probes):
        c = current_processor_number()
        seen[c] = seen.get(c, 0) + 1
        sink += i * 1.000001
    return dict(requested_cpu=int(cpu),
                affinity_mask_readback=mask,
                mask_matches_request=(mask == [int(cpu)]),
                observed_processor_numbers={str(k): int(v)
                                            for k, v in sorted(seen.items())},
                all_probes_on_requested_cpu=(set(seen) == {int(cpu)}),
                n_probes=int(n_probes),
                _sink=float(sink))


def background_load(sample_s=1.0):
    """What else was running on the machine while we measured.

    A latency number is only interpretable next to the load the machine was
    under, and this measurement was NOT taken on an idle box.  Interference can
    only ever inflate a measured cost, never deflate it, so the minimum of a
    long run is the contention-proof statistic and is reported alongside the
    percentiles.
    """
    if psutil is None:
        return dict(available=False)
    me = os.getpid()
    psutil.cpu_percent(interval=None)
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            c = p.cpu_percent(interval=None)
        except Exception:
            continue
        procs.append((p.info["pid"], p.info["name"]))
    total = psutil.cpu_percent(interval=sample_s)
    busy = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            c = p.cpu_percent(interval=None)
        except Exception:
            continue
        if c >= 20.0 and p.info["pid"] != me:
            busy.append(dict(pid=int(p.info["pid"]), name=p.info["name"],
                             cpu_percent=float(c)))
    busy.sort(key=lambda d: -d["cpu_percent"])
    return dict(
        available=True,
        sample_seconds=sample_s,
        system_cpu_percent=float(total),
        n_logical_processors=int(psutil.cpu_count(logical=True)),
        other_processes_above_20pct=busy[:15],
        n_other_processes_above_20pct=len(busy),
        note=("system_cpu_percent is the whole-machine utilisation measured "
              "over the sample window while this benchmark was the caller. "
              "Anything well above the ~5%% one pinned thread contributes is "
              "other work sharing L3, memory bandwidth and the package power "
              "budget with the measurement."))


_PRIORITY_REQUEST = {}
if IS_WINDOWS and psutil is not None:
    _PRIORITY_REQUEST = {
        "normal": psutil.NORMAL_PRIORITY_CLASS,
        "high": psutil.HIGH_PRIORITY_CLASS,
        "realtime": psutil.REALTIME_PRIORITY_CLASS,
    }
_PRIORITY_NAME = {32: "NORMAL_PRIORITY_CLASS", 128: "HIGH_PRIORITY_CLASS",
                  256: "REALTIME_PRIORITY_CLASS", 64: "IDLE_PRIORITY_CLASS",
                  16384: "BELOW_NORMAL_PRIORITY_CLASS",
                  32768: "ABOVE_NORMAL_PRIORITY_CLASS"}


def set_priority(name):
    """Set the Windows process priority class and READ IT BACK.

    REALTIME_PRIORITY_CLASS needs SeIncreaseBasePriorityPrivilege; without it
    Windows silently substitutes HIGH_PRIORITY_CLASS.  The read-back value is
    what gets reported, so an arm that did not get the class it asked for is
    labelled with the class it actually ran at.
    """
    if not IS_WINDOWS or psutil is None or name not in _PRIORITY_REQUEST:
        return dict(mechanism="none", requested=name, granted=None,
                    granted_name=None, granted_as_requested=False,
                    note="priority class not settable in this environment")
    p = psutil.Process()
    req = _PRIORITY_REQUEST[name]
    try:
        p.nice(req)
    except Exception as exc:
        return dict(mechanism="SetPriorityClass (psutil.Process.nice)",
                    requested=name, granted=int(p.nice()),
                    granted_name=_PRIORITY_NAME.get(int(p.nice())),
                    granted_as_requested=False, error=repr(exc))
    got = int(p.nice())
    return dict(mechanism="SetPriorityClass (psutil.Process.nice)",
                requested=name, requested_value=int(req), granted=got,
                granted_name=_PRIORITY_NAME.get(got, str(got)),
                granted_as_requested=(got == int(req)))


# =====================================================================
# 2.  statistics -- order statistics only, no interpolation, no rounding
# =====================================================================
def order_stat(sorted_ns, p):
    """Nearest-rank empirical quantile: the smallest sample at or above rank
    ceil(p*n).  No interpolation between order statistics: interpolating is a
    smoothing operation and this file does not smooth anything."""
    n = len(sorted_ns)
    if n == 0:
        return float("nan")
    k = int(math.ceil(p * n)) - 1
    return float(sorted_ns[min(max(k, 0), n - 1)])


def quantile_order_ci(sorted_ns, p, conf=0.95):
    """Distribution-free CI for the p-quantile from the binomial order-statistic
    method.  This is what justifies the sample size: it says directly how much
    the reported P99.9 could move under resampling."""
    n = len(sorted_ns)
    if n == 0:
        return None
    a = (1.0 - conf) / 2.0
    lo_rank = int(binom_dist.ppf(a, n, p))
    hi_rank = int(binom_dist.ppf(1.0 - a, n, p)) + 1
    lo_rank = min(max(lo_rank, 1), n)
    hi_rank = min(max(hi_rank, 1), n)
    return dict(n_above=int(n - int(math.ceil(p * n))),
                lo_rank=lo_rank, hi_rank=hi_rank,
                lo_ns=float(sorted_ns[lo_rank - 1]),
                hi_ns=float(sorted_ns[hi_rank - 1]))


def clopper_pearson(k, n, conf=0.95):
    """Exact (Clopper-Pearson) binomial interval."""
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - a / 2.0, k + 1, n - k))
    return lo, hi


def lag1_autocorr(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den == 0.0:
        return float("nan")
    return float(np.dot(x[1:], x[:-1]) / den)


def summarise_ns(ns):
    s = np.sort(np.asarray(ns, dtype=np.int64))
    n = int(s.size)
    out = dict(n=n,
               min_us=float(s[0]) / 1e3,
               median_us=order_stat(s, 0.50) / 1e3,
               mean_us=float(s.mean()) / 1e3,
               std_us=float(s.std(ddof=1)) / 1e3 if n > 1 else 0.0,
               p95_us=order_stat(s, 0.95) / 1e3,
               p99_us=order_stat(s, 0.99) / 1e3,
               p999_us=order_stat(s, 0.999) / 1e3,
               max_us=float(s[-1]) / 1e3)
    ci = quantile_order_ci(s, 0.999)
    if ci is not None:
        out["p999_ci95_us"] = [ci["lo_ns"] / 1e3, ci["hi_ns"] / 1e3]
        out["p999_ci95_ranks"] = [ci["lo_rank"], ci["hi_rank"]]
        out["n_samples_above_p999"] = ci["n_above"]
    ci99 = quantile_order_ci(s, 0.99)
    if ci99 is not None:
        out["p99_ci95_us"] = [ci99["lo_ns"] / 1e3, ci99["hi_ns"] / 1e3]
    return out


# =====================================================================
# 3.  the plant realisation -- generated ONCE, replayed by every arm
# =====================================================================
def make_traces(n_seeds, cycles_per_seed, burn_in=500):
    """Pre-record the disturbance realisation each seed block replays.

    The plant is deliberately OUTSIDE the timed window and is not closed around
    the controller: the released reference implementation is a single-shot
    controller, not a closed-loop simulator, and inventing a closed loop would
    change what is being timed.  Each seed block gets its own sway realisation
    (Ornstein-Uhlenbeck, channel.SwayProcess, 500-step burn-in) and its own
    scintillation realisation (channel.GammaGammaAR1 at rho_a = 0.98,
    calibrate=False as in landscape_probe.py), and h_meas = h - 1 is the
    zero-mean latent the Kalman filter is written against.

    Every arm replays the SAME traces, so arms differ only in how the OS
    scheduled them, not in what they were asked to compute.
    """
    thetas, hs = [], []
    for s in range(n_seeds):
        sw = SwayProcess(SIGMA_S, T_u=T_U, seed=1000 + s)
        for _ in range(burn_in):
            sw.step()
        thetas.append(np.array([sw.step() for _ in range(cycles_per_seed)],
                               dtype=np.float64))
        ch = GammaGammaAR1(ALPHA, BETA, rho_a=0.98, seed=2000 + s,
                           calibrate=False)
        hs.append(np.array([ch.step() for _ in range(cycles_per_seed)],
                           dtype=np.float64) - 1.0)
    return thetas, hs


# =====================================================================
# 4.  one instrumented control cycle
# =====================================================================
class CycleRunner:
    """Runs BeamSteeringMPC's control cycle with a timer around every stage.

    The body is BeamSteeringMPC.step() taken apart line by line so the five
    stages can be timed separately.  `check_equivalence()` proves the take-apart
    did not change the computation: with the anytime checkpoint disabled it must
    return bit-identical results to the untouched step() from the same seed.
    """

    def __init__(self, mpc_seed, anytime=True):
        self.mpc = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON,
                                   seed=mpc_seed)
        self.anytime = bool(anytime)
        self.T = self.mpc.horizon
        lo, hi = manuscript_wz_box(SIGMA_S)
        self.w_safe = float(np.clip(wz_for_xi(XI_SAFE, SIGMA_S), lo, hi))
        self.u_prev = np.zeros(2)
        self.out_buf = np.zeros(3)
        self.out_seq = 0

    # ---------------------------------------------------------------
    def cycle(self, theta_row, h_meas):
        """One control cycle.  Returns (stage_ns tuple, total_ns, diag)."""
        mpc = self.mpc
        T = self.T

        t0 = time.perf_counter_ns()

        # ---------------- 1. Sensing --------------------------------
        state = np.empty(2)
        state[0] = theta_row[0]
        state[1] = theta_row[1]
        h = float(h_meas)
        sense_ok = bool(np.isfinite(state).all() and math.isfinite(h)
                        and abs(state[0]) < 1.0 and abs(state[1]) < 1.0)

        t1 = time.perf_counter_ns()

        # ---------------- 2. Prediction -----------------------------
        mpc.kf.update(h)
        h_pred = mpc.kf.predict(T)

        t2 = time.perf_counter_ns()

        # ---------------- 3. Optimization (anytime, stopped at tau_O)
        mpc.theta0 = mpc._as_theta(state, mpc.L)
        solver = HCLPSOGA(mpc.lower(), mpc.upper(), mpc.cfg,
                          seed=int(mpc.rng.integers(1 << 31)),
                          blocks=mpc.blocks(), repair=mpc.repair)
        deadline = t2 + TAU_O_NS

        def obj(X):
            return mpc._objective(X, state, h_pred)

        def guard(X, f, aux):
            rep = envelope_guard(aux["z"], aux["pe_first"],
                                 three_part=mpc.three_part)
            mpc.guard_stats["z"] += rep.n_z
            mpc.guard_stats["range"] += rep.n_range
            mpc.guard_stats["threshold"] += rep.n_threshold
            return rep.admissible

        if self.anytime:
            def ckpt(it, best_f):
                return time.perf_counter_ns() >= deadline
            res = solver.minimise(obj, guard=guard, checkpoint=ckpt)
        else:
            res = solver.minimise(obj, guard=guard)

        t3 = time.perf_counter_ns()

        # ---------------- 4. safety Checks --------------------------
        bx = res.best_x
        if bx is None:
            admissible = False
            w_cmd = self.w_safe
            u_cmd = self.u_prev.copy()
            pe_cmd = float("nan")
            z_cmd = float("nan")
        else:
            w_cmd = float(bx[0])
            u_cmd = np.array([bx[T], bx[2 * T]], dtype=np.float64)
            A0c, weqc = beam_geometry(np.array([w_cmd]))
            xi_c = weqc / (2.0 * SIGMA_S)
            r_d = mpc.L * float(np.linalg.norm(mpc.theta0))
            xi_eff = xi_effective(xi_c, np.array([r_d]), SIGMA_S)
            g_c = GBAR * max(1e-3, 1.0 + float(h_pred[0])) ** 2
            zc = z_of(ALPHA, BETA, A0c, g_c)
            Kc = ladder_order(zc)
            pec = pe_series_f64(ALPHA, BETA, xi_eff, A0c, g_c, Kc)
            z_cmd = float(zc[0])
            pe_cmd = float(pec[0])
            # test (i) admissibility, test (ii) range
            test_i = z_cmd <= Z_MAX
            test_ii = math.isfinite(pe_cmd) and (0.0 <= pe_cmd <= 0.5)
            # actuator envelope, Table tab:actuator_specs
            test_env = bool(np.all(np.abs(u_cmd) <= U_MAX)
                            and np.all(np.abs(u_cmd - self.u_prev) <= U_SLEW))
            admissible = bool(sense_ok and test_i and test_ii and test_env)
            if not admissible:
                w_cmd = self.w_safe
                u_cmd = self.u_prev.copy()

        t4 = time.perf_counter_ns()

        # ---------------- 5. Publish --------------------------------
        u_out = np.clip(u_cmd, self.u_prev - U_SLEW, self.u_prev + U_SLEW)
        u_out = np.clip(u_out, -U_MAX, U_MAX)
        self.out_buf[0] = w_cmd
        self.out_buf[1] = u_out[0]
        self.out_buf[2] = u_out[1]
        self.out_seq += 1
        self.u_prev = u_out

        t5 = time.perf_counter_ns()

        return ((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4), t5 - t0,
                (res.iterations, res.evaluations, admissible,
                 res.best_f, z_cmd, pe_cmd))


def check_equivalence():
    """The instrumented take-apart must compute exactly what step() computes.

    Both objects are constructed with the same seed and driven with the same
    input; with the anytime checkpoint disabled the RNG consumption and the
    arithmetic are identical, so best_f must match bit for bit.
    """
    theta = np.array([2.0e-4, 1.0e-4])
    h = 0.0137
    # tau_o=None so the reference runs its FULL iteration budget: the point of
    # the check is that the instrumented take-apart computes exactly what
    # step() computes, and with the default tau_o checkpoint the wall-clock
    # cut-off makes the two runs a comparison of different iteration counts
    # (1 on a slow host vs 25), which no correct take-apart can match.
    ref = BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=4242,
                          tau_o=None)
    r_ref = ref.step(theta.copy(), h_meas=h)
    run = CycleRunner(4242, anytime=False)
    _, _, diag = run.cycle(theta.copy(), h)
    ok = (np.isclose(diag[3], r_ref.best_f, rtol=0.0, atol=0.0)
          and diag[1] == r_ref.evaluations and diag[0] == r_ref.iterations)
    return dict(instrumented_best_f=float(diag[3]),
                step_best_f=float(r_ref.best_f),
                instrumented_evaluations=int(diag[1]),
                step_evaluations=int(r_ref.evaluations),
                instrumented_iterations=int(diag[0]),
                step_iterations=int(r_ref.iterations),
                bit_identical=bool(ok))


# =====================================================================
# 5.  arm definitions and the measurement loop
# =====================================================================
ARMS = {
    # anytime arms: optimizer stopped at the tau_O = 600 us checkpoint
    "normal_unpinned": dict(
        priority="normal", cpu=None, anytime=True, seeds=10, per_seed=3000,
        label="NORMAL_PRIORITY_CLASS, no affinity restriction (all 20 logical "
              "processors, scheduler free to migrate between P- and E-cores)"),
    "normal_pinned_P": dict(
        priority="normal", cpu=P_CPU, anytime=True, seeds=10, per_seed=3000,
        label="NORMAL_PRIORITY_CLASS, affinity pinned to logical processor %d "
              "(P-core)" % P_CPU),
    "high_pinned_P": dict(
        priority="high", cpu=P_CPU, anytime=True, seeds=10, per_seed=3000,
        label="HIGH_PRIORITY_CLASS, affinity pinned to logical processor %d "
              "(P-core)" % P_CPU),
    "realtime_pinned_P": dict(
        priority="realtime", cpu=P_CPU, anytime=True, seeds=10, per_seed=3000,
        label="REALTIME_PRIORITY_CLASS, affinity pinned to logical processor "
              "%d (P-core)" % P_CPU),
    "normal_pinned_E": dict(
        priority="normal", cpu=E_CPU, anytime=True, seeds=10, per_seed=3000,
        label="NORMAL_PRIORITY_CLASS, affinity pinned to logical processor %d "
              "(E-core)" % E_CPU),
    # repeatability arm: the primary arm run again, in a separate process, at
    # a different time, so the reader can see how much of the primary arm's
    # numbers is the machine's mood on the day.
    "normal_pinned_P_repeat": dict(
        priority="normal", cpu=P_CPU, anytime=True, seeds=10, per_seed=3000,
        label="repeat of normal_pinned_P in a separate process at a later "
              "time -- reproducibility check, not an independent arm"),
    # control arm: the same loop with the anytime checkpoint REMOVED, i.e. the
    # solver run to Table 4's T_iter = 25.  Fewer cycles because each one is
    # ~20x longer; the sample size is reported with the result.
    "converged_pinned_P": dict(
        priority="normal", cpu=P_CPU, anytime=False, seeds=5, per_seed=200,
        label="NORMAL_PRIORITY_CLASS, pinned to logical processor %d (P-core), "
              "anytime checkpoint REMOVED: solver runs the full T_iter = 25"
              % P_CPU),
}

DEFAULT_ARMS = ["normal_pinned_P", "high_pinned_P", "realtime_pinned_P",
                "normal_unpinned", "normal_pinned_E"]


def run_arm(name, spec, traces, warmup=200, verbose=True):
    thetas, hs = traces
    n_seeds = int(spec["seeds"])
    per_seed = int(spec["per_seed"])
    n = n_seeds * per_seed

    prev_aff = get_affinity()
    prev_prio = int(psutil.Process().nice()) if (IS_WINDOWS and psutil) else None

    pin_report = None
    if spec["cpu"] is not None:
        set_affinity([spec["cpu"]])
        pin_report = verify_pin(spec["cpu"])
    else:
        # explicitly widen to every logical processor, so "unpinned" means
        # unpinned and not "whatever the previous arm left behind"
        allcpus = list(range(psutil.cpu_count(logical=True))) if psutil else None
        if allcpus:
            set_affinity(allcpus)
        pin_report = dict(requested_cpu=None,
                          affinity_mask_readback=get_affinity(),
                          mask_matches_request=None,
                          all_probes_on_requested_cpu=None,
                          note="arm is intentionally unpinned")

    prio_report = set_priority(spec["priority"])
    tover = timer_overhead_ns(100000)
    load_before = background_load()

    stage_ns = {s: np.zeros(n, dtype=np.int64) for s in STAGES}
    total_ns = np.zeros(n, dtype=np.int64)
    seed_id = np.zeros(n, dtype=np.int32)
    iters = np.zeros(n, dtype=np.int32)
    evals = np.zeros(n, dtype=np.int32)
    adm = np.zeros(n, dtype=np.int8)
    cpu_of = np.zeros(n, dtype=np.int16)

    try:
        # ---- warm-up: imports, first-call numpy dispatch, the mpmath-backed
        # coefficient cache in rtodt_fast, the branch predictors.  Untimed.
        warm = CycleRunner(999, anytime=spec["anytime"])
        for i in range(warmup):
            warm.cycle(thetas[0][i % per_seed], hs[0][i % per_seed])

        k = 0
        for s in range(n_seeds):
            run = CycleRunner(10000 + s, anytime=spec["anytime"])
            th, hh = thetas[s], hs[s]
            for i in range(per_seed):
                st, tot, diag = run.cycle(th[i], hh[i])
                # sampled OUTSIDE the timed window
                cpu_of[k] = current_processor_number()
                total_ns[k] = tot
                stage_ns["sense"][k] = st[0]
                stage_ns["predict"][k] = st[1]
                stage_ns["optimize"][k] = st[2]
                stage_ns["checks"][k] = st[3]
                stage_ns["publish"][k] = st[4]
                seed_id[k] = s
                iters[k] = diag[0]
                evals[k] = diag[1]
                adm[k] = 1 if diag[2] else 0
                k += 1
            if verbose:
                blk = total_ns[s * per_seed:(s + 1) * per_seed]
                print("    seed %2d/%d  median %8.1f us  max %9.1f us"
                      % (s + 1, n_seeds, np.median(blk) / 1e3, blk.max() / 1e3))
    finally:
        if IS_WINDOWS and psutil is not None and prev_prio is not None:
            try:
                psutil.Process().nice(prev_prio)
            except Exception:
                pass
        if prev_aff:
            try:
                set_affinity(prev_aff)
            except Exception:
                pass

    arrays = {"%s__total_ns" % name: total_ns,
              "%s__seed_id" % name: seed_id,
              "%s__iters" % name: iters,
              "%s__evals" % name: evals,
              "%s__admissible" % name: adm,
              "%s__cpu" % name: cpu_of}
    for s in STAGES:
        arrays["%s__%s_ns" % (name, s)] = stage_ns[s]

    summary = analyse_arm(name, spec, total_ns, stage_ns, seed_id, iters,
                          evals, adm, cpu_of, per_seed, n_seeds)
    summary["pinning"] = pin_report
    summary["priority"] = prio_report
    summary["timer_overhead_on_this_core"] = tover
    summary["warmup_cycles_untimed"] = int(warmup)
    summary["machine_load_before_arm"] = load_before
    summary["machine_load_after_arm"] = background_load()
    return arrays, summary


def analyse_arm(name, spec, total_ns, stage_ns, seed_id, iters, evals, adm,
                cpu_of, per_seed, n_seeds):
    n = int(total_ns.size)

    # ---- deadline success, exact binomial interval ------------------
    hits = int(np.sum(total_ns <= BUDGET_NS))
    lo, hi = clopper_pearson(hits, n)
    hits_tu = int(np.sum(total_ns <= int(round(T_U * 1e9))))
    lo_tu, hi_tu = clopper_pearson(hits_tu, n)

    # ---- per-seed dispersion of the deadline indicator --------------
    # The forensic signature that condemned the released traces was UNDER-
    # binomial per-seed dispersion.  Report the ratio here whatever it is.
    per_seed_hits = np.array([int(np.sum(total_ns[seed_id == s] <= BUDGET_NS))
                              for s in range(n_seeds)], dtype=float)
    p_hat = per_seed_hits.sum() / float(n)
    binom_var = per_seed_hits.size and p_hat * (1 - p_hat) * per_seed
    obs_var = float(np.var(per_seed_hits, ddof=1)) if n_seeds > 1 else float("nan")
    disp = (obs_var / binom_var) if binom_var else float("nan")

    per_seed_median = np.array([float(np.median(total_ns[seed_id == s]))
                                for s in range(n_seeds)])
    per_seed_p99 = np.array([order_stat(np.sort(total_ns[seed_id == s]), 0.99)
                             for s in range(n_seeds)])

    # ---- serial correlation -----------------------------------------
    lag1_full = lag1_autocorr(total_ns)
    lag1_blocks = [lag1_autocorr(total_ns[seed_id == s]) for s in range(n_seeds)]
    # lag 1..10 of the full series, so a reader can see whether structure
    # persists or is a one-lag artefact
    acf = []
    x = total_ns.astype(np.float64)
    x = x - x.mean()
    den = float(np.dot(x, x))
    for L in range(1, 11):
        acf.append(float(np.dot(x[L:], x[:-L]) / den) if den else float("nan"))

    # ---- where did the cycles actually run --------------------------
    vals, counts = np.unique(cpu_of, return_counts=True)
    cpu_hist = {int(v): int(c) for v, c in zip(vals, counts)}

    out = dict(
        arm=name,
        label=spec["label"],
        anytime_checkpoint=bool(spec["anytime"]),
        tau_O_us=(TAU_O * 1e6) if spec["anytime"] else None,
        n_cycles=n, n_seed_blocks=int(n_seeds), cycles_per_seed=int(per_seed),
        end_to_end=summarise_ns(total_ns),
        stages={s: summarise_ns(stage_ns[s]) for s in STAGES},
        deadline=dict(
            budget_us=BUDGET * 1e6,
            budget_definition="T_u - tau_A = 1000 us - 200 us",
            n_meeting_budget=hits, n_cycles=n,
            success_rate=hits / float(n),
            clopper_pearson_95=[lo, hi],
            n_meeting_T_u=hits_tu, success_rate_T_u=hits_tu / float(n),
            clopper_pearson_95_T_u=[lo_tu, hi_tu]),
        budget_that_would_be_met=dict(
            us_for_50pct=order_stat(np.sort(total_ns), 0.50) / 1e3,
            us_for_95pct=order_stat(np.sort(total_ns), 0.95) / 1e3,
            us_for_99pct=order_stat(np.sort(total_ns), 0.99) / 1e3,
            us_for_999pct=order_stat(np.sort(total_ns), 0.999) / 1e3,
            note="smallest computation budget this arm would have met at each "
                 "rate, read straight off the measured order statistics"),
        per_seed_dispersion=dict(
            per_seed_deadline_hits=[int(v) for v in per_seed_hits],
            pooled_p=float(p_hat),
            observed_variance_of_hits=obs_var,
            binomial_variance_of_hits=float(binom_var) if binom_var else None,
            dispersion_ratio=float(disp) if binom_var else None,
            per_seed_median_us=[v / 1e3 for v in per_seed_median],
            spread_of_per_seed_medians_us=float(
                per_seed_median.max() - per_seed_median.min()) / 1e3,
            per_seed_p99_us=[v / 1e3 for v in per_seed_p99],
            note="dispersion_ratio is observed/binomial variance of the "
                 "per-seed deadline-hit count; 1.0 is exactly binomial, "
                 "below 1 is the under-dispersion signature that condemned "
                 "the released traces, above 1 means slow drift between "
                 "blocks. It is undefined when the rate is 0 or 1."),
        serial_correlation=dict(
            lag1_full_series=lag1_full,
            lag1_within_seed_blocks=[float(v) for v in lag1_blocks],
            lag1_within_seed_mean=float(np.nanmean(lag1_blocks)),
            acf_lag_1_to_10=acf),
        solver=dict(
            iterations_completed_median=float(np.median(iters)),
            iterations_completed_min=int(iters.min()),
            iterations_completed_max=int(iters.max()),
            iterations_histogram={int(v): int(c) for v, c in
                                  zip(*np.unique(iters, return_counts=True))},
            objective_evaluations_median=float(np.median(evals)),
            fraction_command_admissible=float(adm.mean())),
        executed_on_logical_processors=cpu_hist,
        n_distinct_logical_processors=len(cpu_hist),
    )
    # which core CLASS each cycle finished on -- the hybrid-CPU mixture,
    # sampled outside the timed window
    on_e = cpu_of >= 12
    out["core_class_mixture"] = dict(
        fraction_finishing_on_E_core=float(on_e.mean()),
        fraction_finishing_on_P_core=float((~on_e).mean()),
        median_us_when_finished_on_P=(float(np.median(total_ns[~on_e])) / 1e3
                                      if (~on_e).any() else None),
        median_us_when_finished_on_E=(float(np.median(total_ns[on_e])) / 1e3
                                      if on_e.any() else None),
        note=("logical 0-11 are P-core threads, 12-19 are E-cores. The core "
              "number is sampled once per cycle AFTER the timed window "
              "closes, so it is where the cycle ended, not necessarily where "
              "all of it ran -- which is exactly the problem with an unpinned "
              "measurement."))
    return out


# =====================================================================
# 6.  cost of guard test (iii), measured rather than assumed
# =====================================================================
def eps_safe_cost(reps_fast=5):
    """How long the post-EGC eps_safe test would take if it ran in-cycle.

    mpc_loop.envelope_guard explicitly defers test (iii), Pe_system < eps_safe,
    to a system-level evaluation.  The release provides three implementations of
    that evaluation (system_metric.system_aber, methods 'fast', 'quad', 'egc').
    None of them is a lookup; all are numerical convolutions.  This measures
    them once so the Checks stage's omission is quantified instead of hidden.
    """
    import system_metric as sm
    xi, A0 = 1.967, 0.129
    out = {}
    sm.system_aber(ALPHA, BETA, xi, A0, GBAR, method="fast")     # warm
    for method, reps in (("fast", reps_fast), ("quad", 1), ("egc", 1)):
        ts = []
        val = None
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            val = sm.system_aber(ALPHA, BETA, xi, A0, GBAR, method=method)
            ts.append(time.perf_counter_ns() - t0)
        ts = np.array(ts, dtype=np.int64)
        out[method] = dict(reps=int(reps), median_us=float(np.median(ts)) / 1e3,
                           min_us=float(ts.min()) / 1e3,
                           value=float(np.asarray(val).ravel()[0]),
                           times_the_800us_budget=float(np.median(ts)) / 1e3 / (BUDGET * 1e6))
    out["eps_safe"] = EPS_SAFE
    out["note"] = ("Measured cost of ONE evaluation of guard test (iii). The "
                   "cheapest available implementation is compared against the "
                   "whole 800 us computation budget, not against the Checks "
                   "stage alone.")
    return out


# =====================================================================
# 7.  I/O -- results accumulate across invocations
# =====================================================================
def load_existing():
    arrays, meta = {}, {}
    if os.path.exists(NPZ_PATH):
        with np.load(NPZ_PATH, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as fh:
            meta = json.load(fh)
    return arrays, meta


def save(arrays, meta):
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    np.savez_compressed(NPZ_PATH, **arrays)
    with open(JSON_PATH, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=False)
    print("\nwrote %s  (%.2f MB)" % (NPZ_PATH, os.path.getsize(NPZ_PATH) / 1e6))
    print("wrote %s" % JSON_PATH)


def platform_block():
    blk = dict(
        node=platform.node(),
        platform=platform.platform(),
        processor=platform.processor(),
        python=sys.version,
        numpy=np.__version__,
        logical_processors=(psutil.cpu_count(logical=True) if psutil else None),
        physical_cores=(psutil.cpu_count(logical=False) if psutil else None),
        blas_thread_caps={v: os.environ.get(v) for v in
                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS")},
        gc_enabled=True,
        gc_note=("the garbage collector is left ENABLED. A collection pause is "
                 "a real source of tail latency in this loop and belongs "
                 "inside the measurement, not outside it."),
        p_core_logical_processors="0-11 (6 P-cores, SMT)",
        e_core_logical_processors="12-19 (8 E-cores, no SMT)",
        measured_E_over_P_cost_ratio=2.38,
        hybrid_note=("MEASURED in the platform phase on the same inner-loop "
                     "shape: an E-core costs 2.38x a P-core. An unpinned "
                     "latency distribution on this CPU is a mixture of two "
                     "processors and its tail is not interpretable."),
    )
    return blk


def scheduling_analogue_block():
    return dict(
        status="ANALOGUE -- NOT a reproduction of the manuscript's Linux arms",
        manuscript_arms=["SCHED_OTHER", "chrt", "isolcpus"],
        reproducible_here=False,
        why=("SCHED_OTHER, chrt and isolcpus are Linux kernel mechanisms. They "
             "do not exist on Windows 11 and are not emulated by anything in "
             "this script. No number in this file is a measurement of any of "
             "them."),
        mechanism_used_here=[
            "SetPriorityClass (NORMAL_PRIORITY_CLASS = 32, "
            "HIGH_PRIORITY_CLASS = 128, REALTIME_PRIORITY_CLASS = 256), "
            "invoked through psutil.Process().nice()",
            "SetProcessAffinityMask, invoked through "
            "psutil.Process().cpu_affinity(); ctypes fallback included",
        ],
        semantic_differences=[
            "Windows priority classes schedule a PROCESS against other "
            "processes; SCHED_OTHER/SCHED_FIFO are per-thread scheduling "
            "POLICIES with different preemption rules.",
            "isolcpus removes a core from the general scheduler's runqueue "
            "entirely. Windows processor affinity does the opposite: it "
            "constrains this process to a core, while leaving every other "
            "thread on the system free to run there.",
            "Windows offers no real-time guarantee at any priority class. "
            "Hardware interrupts and DPCs preempt REALTIME_PRIORITY_CLASS.",
        ],
    )


def print_arm(summary):
    e = summary["end_to_end"]
    d = summary["deadline"]
    print("\n  %s" % summary["arm"])
    print("    %s" % summary["label"])
    pr = summary.get("priority", {})
    print("    priority granted : %s" % pr.get("granted_name"))
    pin = summary.get("pinning", {})
    print("    pin verified     : mask=%s  all-probes-on-cpu=%s"
          % (pin.get("mask_matches_request"),
             pin.get("all_probes_on_requested_cpu")))
    print("    ran on logical   : %s" % summary["executed_on_logical_processors"])
    print("    n cycles         : %d  (%d seed blocks x %d)"
          % (summary["n_cycles"], summary["n_seed_blocks"],
             summary["cycles_per_seed"]))
    print("    end-to-end us    : median %9.1f  P95 %9.1f  P99 %9.1f  "
          "P99.9 %9.1f  max %10.1f"
          % (e["median_us"], e["p95_us"], e["p99_us"], e["p999_us"],
             e["max_us"]))
    print("    P99.9 95%% CI     : [%.1f, %.1f] us over ranks %s"
          % (e["p999_ci95_us"][0], e["p999_ci95_us"][1],
             e["p999_ci95_ranks"]))
    for s in STAGES:
        q = summary["stages"][s]
        print("      %-9s us   median %9.3f  P95 %9.3f  P99 %9.3f  "
              "P99.9 %9.3f  max %10.3f"
              % (s, q["median_us"], q["p95_us"], q["p99_us"], q["p999_us"],
                 q["max_us"]))
    print("    deadline 800 us  : %d/%d = %.4f%%  Clopper-Pearson 95%% "
          "[%.6f, %.6f]"
          % (d["n_meeting_budget"], d["n_cycles"], 100.0 * d["success_rate"],
             d["clopper_pearson_95"][0], d["clopper_pearson_95"][1]))
    sc = summary["serial_correlation"]
    print("    lag-1 autocorr   : %.4f (full series), %.4f (mean within seed "
          "block)" % (sc["lag1_full_series"], sc["lag1_within_seed_mean"]))
    pd = summary["per_seed_dispersion"]
    print("    per-seed spread  : medians span %.1f us; dispersion ratio %s"
          % (pd["spread_of_per_seed_medians_us"],
             ("%.3f" % pd["dispersion_ratio"])
             if pd["dispersion_ratio"] is not None else "undefined (rate 0 or 1)"))
    sv = summary["solver"]
    print("    solver iters     : median %.1f  min %d  max %d  hist %s"
          % (sv["iterations_completed_median"], sv["iterations_completed_min"],
             sv["iterations_completed_max"], sv["iterations_histogram"]))


# =====================================================================
# 6b. which safety test rejects the published command, and how often
# =====================================================================
def guard_audit(n_cycles=2000, seed_block=0, anytime=True):
    """The timed runs report that almost every published command is replaced by
    the offline override.  A latency measurement is not the place to leave that
    unexplained, so this repeats the Checks stage UNTIMED and records which of
    the tests bound.  It changes no timing number in this file.
    """
    thetas, hs = make_traces(1, n_cycles)
    th, hh = thetas[0], hs[0]
    run = CycleRunner(10000 + seed_block, anytime=anytime)
    mpc = run.mpc
    T = run.T
    counts = dict(n=0, admissible=0, fail_test_i_z=0, fail_test_ii_range=0,
                  fail_actuator_abs=0, fail_actuator_slew=0,
                  no_feasible_candidate=0)
    for i in range(n_cycles):
        mpc.kf.update(float(hh[i]))
        h_pred = mpc.kf.predict(T)
        state = np.array(th[i], dtype=float)
        mpc.theta0 = mpc._as_theta(state, mpc.L)
        solver = HCLPSOGA(mpc.lower(), mpc.upper(), mpc.cfg,
                          seed=int(mpc.rng.integers(1 << 31)),
                          blocks=mpc.blocks(), repair=mpc.repair)
        t0 = time.perf_counter_ns()
        dl = t0 + TAU_O_NS
        res = solver.minimise(
            lambda X: mpc._objective(X, state, h_pred),
            guard=lambda X, f, aux: envelope_guard(
                aux["z"], aux["pe_first"], three_part=mpc.three_part).admissible,
            checkpoint=(lambda it, bf: time.perf_counter_ns() >= dl)
            if anytime else None)
        counts["n"] += 1
        if res.best_x is None:
            counts["no_feasible_candidate"] += 1
            continue
        bx = res.best_x
        w_cmd = float(bx[0])
        u_cmd = np.array([bx[T], bx[2 * T]])
        A0c, weqc = beam_geometry(np.array([w_cmd]))
        r_d = mpc.L * float(np.linalg.norm(mpc.theta0))
        xe = xi_effective(weqc / (2.0 * SIGMA_S), np.array([r_d]), SIGMA_S)
        g_c = GBAR * max(1e-3, 1.0 + float(h_pred[0])) ** 2
        zc = z_of(ALPHA, BETA, A0c, g_c)
        pec = pe_series_f64(ALPHA, BETA, xe, A0c, g_c, ladder_order(zc))
        ok = True
        if not (float(zc[0]) <= Z_MAX):
            counts["fail_test_i_z"] += 1
            ok = False
        if not (math.isfinite(float(pec[0])) and 0.0 <= float(pec[0]) <= 0.5):
            counts["fail_test_ii_range"] += 1
            ok = False
        if not np.all(np.abs(u_cmd) <= U_MAX):
            counts["fail_actuator_abs"] += 1
            ok = False
        if not np.all(np.abs(u_cmd - run.u_prev) <= U_SLEW):
            counts["fail_actuator_slew"] += 1
            ok = False
        if ok:
            counts["admissible"] += 1
            run.u_prev = u_cmd
    counts["rates"] = {k: v / float(counts["n"])
                       for k, v in counts.items() if k != "n"}
    counts["note"] = (
        "UNTIMED diagnostic. u_dot_max*T_u = %.2e rad per cycle while the "
        "decision box for the pointing command is +/- %.2e rad, and the "
        "released solver applies the slew constraint only BETWEEN horizon "
        "stages within one cycle, never between the command published this "
        "cycle and the one published last cycle. Whether that is what binds "
        "is the question this audit answers." % (U_SLEW, U_MAX))
    return counts


# =====================================================================
# 7b. self-forensics: apply to OUR trace the tests that condemned theirs
# =====================================================================
def forensics(arrays, meta, arm="normal_pinned_P"):
    """The released latency traces were rejected because they carried the
    arithmetic signature of construction: order statistics landing on exact
    integers of a 0.1 us lattice, lag-1 autocorrelation of ~0, under-binomial
    per-seed dispersion, and a density plateau starting exactly at the 800 us
    deadline.  A replacement trace has no standing unless the same tests are
    turned on it.  They are, here, and the answers are printed whatever they
    are.

    One result needs stating up front so it is not mistaken for the signature
    it resembles: EVERY sample in this file is an exact multiple of 100 ns,
    because 100 ns is the MEASURED resolution of time.perf_counter_ns on this
    machine.  That is the clock, not a lattice imposed on the numbers.  The
    discriminating test is the next one down -- whether the samples also pile
    onto coarser round values, which a real clock has no reason to do.
    """
    key = "%s__total_ns" % arm
    if key not in arrays:
        return None
    x = np.asarray(arrays[key], dtype=np.int64)
    n = int(x.size)
    seed_id = np.asarray(arrays.get("%s__seed_id" % arm,
                                    np.zeros(n, dtype=np.int32)))
    res = int(meta.get("clock", {}).get("measured_resolution_ns", 100))

    out = dict(arm=arm, n=n, clock_resolution_ns=res)

    # -- lattice tests -------------------------------------------------
    out["lattice"] = dict(
        frac_multiple_of_clock_resolution=float(np.mean(x % res == 0)),
        frac_multiple_of_1us=float(np.mean(x % 1000 == 0)),
        frac_multiple_of_10us=float(np.mean(x % 10000 == 0)),
        frac_multiple_of_100us=float(np.mean(x % 100000 == 0)),
        expected_by_chance_1us=res / 1000.0,
        expected_by_chance_10us=res / 10000.0,
        distinct_values=int(np.unique(x).size),
        distinct_fraction=float(np.unique(x).size) / n,
        note=("frac_multiple_of_clock_resolution is 1.0 by construction: the "
              "clock ticks at that resolution. The 1 us / 10 us / 100 us rows "
              "are the real test and should sit at the by-chance rate."))
    # the 15 reported order statistics, in the same units the released trace
    # was audited in (0.1 us lattice units)
    s = np.sort(x)
    stats15 = {}
    for lbl, p in (("median", 0.50), ("p95", 0.95), ("p99", 0.99),
                   ("p999", 0.999)):
        stats15["%s_us" % lbl] = order_stat(s, p) / 1e3
    stats15["max_us"] = float(s[-1]) / 1e3
    n_int_us = sum(1 for v in stats15.values() if float(v).is_integer())
    out["order_statistics_on_integer_microseconds"] = dict(
        values_us=stats15, n_landing_on_integer_us=int(n_int_us),
        n_reported=len(stats15),
        note="the released traces had 14 of 15 order statistics on exact "
             "integers; anything much above chance here would be the same "
             "problem in a new file")

    # -- serial structure ----------------------------------------------
    z = x.astype(np.float64)
    z = z - z.mean()
    den = float(np.dot(z, z))
    out["acf_lag_1_to_20"] = [float(np.dot(z[L:], z[:-L]) / den)
                              for L in range(1, 21)]

    # -- per-seed dispersion of the location, not just of a rate -------
    meds = np.array([float(np.median(x[seed_id == s_])) for s_ in
                     np.unique(seed_id)])
    out["per_seed_medians_us"] = [v / 1e3 for v in meds]
    out["per_seed_median_cv"] = float(meds.std(ddof=1) / meds.mean())

    # -- warm-up transient ---------------------------------------------
    # Not an artefact to be trimmed away: a deployed controller also boots
    # cold, and this loop's mpmath-backed coefficient cache (rtodt_fast._KC_
    # CACHE / _C_CACHE) fills over thousands of distinct per-stage gbar
    # values. Report the block profile and let a reader see where it settles.
    nb = 30
    if n >= nb * 100:
        blk = x[:(n // nb) * nb].reshape(nb, -1)
        bm = np.median(blk, axis=1) / 1e3
        settled = float(np.median(bm[nb // 2:]))
        idx = [i for i, v in enumerate(bm) if v <= 1.05 * settled]
        out["warm_up"] = dict(
            block_size=int(blk.shape[1]),
            block_medians_us=[float(v) for v in bm],
            settled_median_us=settled,
            first_block_within_5pct_of_settled=(idx[0] if idx else None),
            cycles_to_settle=(idx[0] * blk.shape[1] if idx else None),
            note="block medians over equal blocks of the series as measured, "
                 "in order. A rising-then-flat profile is cache warm-up; the "
                 "raw array is saved so a reader can cut it themselves.")

    # -- density near the deadline -------------------------------------
    # The released trace had a density plateau beginning exactly at 800 us.
    # Report the measured mass in 100 us bins around the budget so a reader
    # can see whether anything special happens there.
    edges = np.arange(0, max(int(s[-1]) + 100000, 1200000), 100000)
    h, _ = np.histogram(x, bins=edges)
    nz = np.nonzero(h)[0]
    out["histogram_100us_bins"] = dict(
        first_nonempty_bin_us=float(edges[nz[0]]) / 1e3 if nz.size else None,
        counts_around_800us={
            "%d-%d us" % (edges[i] / 1000, edges[i + 1] / 1000): int(h[i])
            for i in range(max(0, 5), min(len(h), 12))},
        n_below_budget=int(np.sum(x <= BUDGET_NS)))
    return out


# =====================================================================
# 8.  cross-arm report and the manuscript table
# =====================================================================
def cross_arm_report(meta):
    arms = meta.get("arms", {})
    if not arms:
        print("no arms in %s" % JSON_PATH)
        return None

    canon = ("normal_pinned_P", "high_pinned_P", "realtime_pinned_P",
             "normal_unpinned", "normal_pinned_E", "normal_pinned_P_repeat",
             "converged_pinned_P")
    order = ([k for k in canon if k in arms]
             + [k for k in arms if k not in canon])

    print("\n" + "=" * 108)
    print("CROSS-ARM COMPARISON -- end-to-end control-cycle latency, "
          "microseconds, %g us computation budget" % (BUDGET * 1e6))
    print("=" * 108)
    print("%-24s %7s %9s %10s %10s %10s %10s %11s %9s %7s"
          % ("arm", "n", "min", "median", "P95", "P99", "P99.9", "max",
             "<=800us", "lag1"))
    for k in order:
        s = arms[k]
        e, d = s["end_to_end"], s["deadline"]
        print("%-24s %7d %9.1f %10.1f %10.1f %10.1f %10.1f %11.1f %8.3f%% %7.4f"
              % (k, s["n_cycles"], e["min_us"], e["median_us"], e["p95_us"],
                 e["p99_us"], e["p999_us"], e["max_us"],
                 100.0 * d["success_rate"],
                 s["serial_correlation"]["lag1_full_series"]))
    print("\nmachine load while measuring (this box was NOT idle):")
    for k in order:
        lb = arms[k].get("machine_load_before_arm") or {}
        if lb.get("available"):
            print("  %-24s system CPU %5.1f%% of %d logical processors, "
                  "%d other processes above 20%%"
                  % (k, lb["system_cpu_percent"], lb["n_logical_processors"],
                     lb["n_other_processes_above_20pct"]))

    comp = {}
    if "normal_pinned_P" in arms and "normal_pinned_E" in arms:
        p, e = arms["normal_pinned_P"], arms["normal_pinned_E"]
        comp["E_over_P_same_priority"] = {
            m: e["end_to_end"][m] / p["end_to_end"][m]
            for m in ("median_us", "p95_us", "p99_us", "p999_us", "max_us")}
    if "normal_pinned_P" in arms and "normal_unpinned" in arms:
        p, u = arms["normal_pinned_P"], arms["normal_unpinned"]
        comp["unpinned_over_pinned_P"] = {
            m: u["end_to_end"][m] / p["end_to_end"][m]
            for m in ("median_us", "p95_us", "p99_us", "p999_us", "max_us")}
        comp["unpinned_logical_processors_touched"] = \
            u["n_distinct_logical_processors"]
    for hi in ("high_pinned_P", "realtime_pinned_P"):
        if hi in arms and "normal_pinned_P" in arms:
            p = arms["normal_pinned_P"]
            comp["%s_over_normal_pinned_P" % hi] = {
                m: arms[hi]["end_to_end"][m] / p["end_to_end"][m]
                for m in ("median_us", "p95_us", "p99_us", "p999_us",
                          "max_us")}
    if "converged_pinned_P" in arms and "normal_pinned_P" in arms:
        comp["converged_over_anytime_median"] = (
            arms["converged_pinned_P"]["end_to_end"]["median_us"]
            / arms["normal_pinned_P"]["end_to_end"]["median_us"])

    print("\nratios (the point of the comparison is the TAIL, not the median):")
    for k, v in comp.items():
        if isinstance(v, dict):
            print("  %-34s %s" % (k, "  ".join("%s x%.3f" % (m.replace("_us", ""), r)
                                               for m, r in v.items())))
        else:
            print("  %-34s %s" % (k, v))
    return comp


def latex_table(meta):
    arms = meta.get("arms", {})
    rows = [k for k in ("normal_pinned_P", "high_pinned_P",
                        "realtime_pinned_P", "normal_unpinned",
                        "normal_pinned_E") if k in arms]
    # The row label reports the priority class the OS actually GRANTED, read
    # back from the process, not the one that was requested.  On this machine
    # REALTIME_PRIORITY_CLASS was requested without SeIncreaseBasePriority-
    # Privilege and Windows silently substituted HIGH_PRIORITY_CLASS; labelling
    # that row "realtime" would be a false statement about the measurement.
    short = {32: "Normal", 128: "High", 256: "Realtime"}

    def row_label(k):
        s = arms[k]
        got = s.get("priority", {}).get("granted")
        req = s.get("priority", {}).get("requested", "?")
        cls = short.get(got, str(got))
        if not s.get("priority", {}).get("granted_as_requested", True):
            cls = r"%s\textsuperscript{$\dagger$} (%s requested)" % (
                cls, req)
        where = (r"\emph{unpinned}" if s["pinning"].get("requested_cpu") is None
                 else r"pinned cpu\,%d (%s-core)"
                 % (s["pinning"]["requested_cpu"],
                    "P" if s["pinning"]["requested_cpu"] < 12 else "E"))
        return "%s priority, %s" % (cls, where)

    pretty = {k: row_label(k) for k in rows}
    L = []
    A = L.append
    A(r"% Measured on a single platform: Intel Core i5-14600KF, Windows 11 Pro")
    A(r"% build 26200, CPython 3.14.6, numpy 2.5.0 / scipy-openblas 0.3.33 (1 thread).")
    A(r"% Clock: time.perf_counter_ns, measured resolution 100 ns.")
    A(r"% The three rows below are a WINDOWS ANALOGUE (SetPriorityClass +")
    A(r"% SetProcessAffinityMask). They are NOT SCHED_OTHER / chrt / isolcpus,")
    A(r"% which are Linux mechanisms and cannot be run on this machine.")
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{Measured end-to-end control-cycle latency on the single "
      r"reference platform (Intel Core i5-14600KF, Windows~11, CPython~3.14.6). "
      r"The optimiser is stopped at the anytime checkpoint $\tau_O=600\,\mu$s; "
      r"the computation budget is $T_u-\tau_A = 800\,\mu$s. Percentiles are "
      r"empirical order statistics of $n$ measured cycles, not interpolated. "
      r"The priority/affinity rows are a Windows analogue of an OS-tuning "
      r"sweep and are not a reproduction of the Linux "
      r"\texttt{SCHED\_OTHER}/\texttt{chrt}/\texttt{isolcpus} arms.}")
    A(r"\label{tab:cycle_latency_measured}")
    A(r"\begin{tabular}{lrrrrrrr}")
    A(r"\toprule")
    A(r"Configuration & $n$ & Median & P95 & P99 & P99.9 & Max & "
      r"$\Pr[\le 800\,\mu\mathrm{s}]$\\")
    A(r" & & ($\mu$s) & ($\mu$s) & ($\mu$s) & ($\mu$s) & ($\mu$s) & \\")
    A(r"\midrule")
    for k in rows:
        s = arms[k]
        e, d = s["end_to_end"], s["deadline"]
        A(r"%s & %d & %.0f & %.0f & %.0f & %.0f & %.0f & %s\\"
          % (pretty[k], s["n_cycles"], e["median_us"], e["p95_us"],
             e["p99_us"], e["p999_us"], e["max_us"],
             (r"$%.3f$" % d["success_rate"]) if d["success_rate"] > 0
             else r"$0$ (0/%d)" % d["n_cycles"]))
    A(r"\bottomrule")
    A(r"\end{tabular}")
    dagger = any(not arms[k].get("priority", {}).get("granted_as_requested",
                                                     True) for k in rows)
    if dagger:
        A(r"\\[2pt]\footnotesize $\dagger$ \texttt{REALTIME\_PRIORITY\_CLASS} "
          r"was requested but not granted: the measurement process did not "
          r"hold \texttt{SeIncreaseBasePriorityPrivilege}, and Windows "
          r"substituted \texttt{HIGH\_PRIORITY\_CLASS}. The read-back value is "
          r"what is reported. No measurement in this table was taken at "
          r"realtime priority.")
    rep = arms.get("normal_pinned_P_repeat")
    if rep:
        e = rep["end_to_end"]
        A(r"\\[2pt]\footnotesize The measurement host was not idle: a "
          r"concurrent load of ten compute-bound processes held it near 50\%% "
          r"utilisation throughout. A repeat of row~1 taken later under "
          r"heavier interference gave median %.0f, P99.9 %.0f and a single "
          r"%.1f\,ms outlier ($n=%d$, again $0$ cycles inside the budget). "
          r"Contention can only lengthen a measured duration, so the "
          r"contention-proof statement is the minimum: the fastest of "
          r"$60\,000$ pinned P-core cycles took %.0f\,$\mu$s, still "
          r"$2.4\times$ the budget."
          % (e["median_us"], e["p999_us"], e["max_us"] / 1e3,
             rep["n_cycles"], min(arms["normal_pinned_P"]["end_to_end"]["min_us"],
                                  e["min_us"])))
    A(r"\end{table}")

    if "normal_pinned_P" in arms:
        s = arms["normal_pinned_P"]
        A("")
        A(r"\begin{table}[t]")
        A(r"\centering")
        A(r"\caption{Measured per-stage latency of the five-stage control "
          r"pipeline, normal priority pinned to one P-core, $n=%d$ cycles. "
          r"Sensing is the software acquisition path only: no physical sensor "
          r"is present on the measurement platform. Prediction is the "
          r"Kalman branch alone; the TCN branch of eq.~(30) is not implemented "
          r"in the release and its cost is absent. Checks omits the post-EGC "
          r"$\epsilon_{\mathrm{safe}}$ test, whose cheapest available "
          r"implementation is measured separately.}" % s["n_cycles"])
        A(r"\label{tab:stage_latency_measured}")
        A(r"\begin{tabular}{lrrrrr}")
        A(r"\toprule")
        A(r"Stage & Median & P95 & P99 & P99.9 & Max\\")
        A(r" & ($\mu$s) & ($\mu$s) & ($\mu$s) & ($\mu$s) & ($\mu$s)\\")
        A(r"\midrule")
        nice = dict(sense="Sensing", predict="Prediction",
                    optimize=r"Optimisation (anytime, $\tau_O$)",
                    checks="Safety checks", publish="Publish")
        for st in STAGES:
            q = s["stages"][st]
            A(r"%s & %.2f & %.2f & %.2f & %.2f & %.2f\\"
              % (nice[st], q["median_us"], q["p95_us"], q["p99_us"],
                 q["p999_us"], q["max_us"]))
        A(r"\midrule")
        e = s["end_to_end"]
        A(r"\textbf{End to end} & %.1f & %.1f & %.1f & %.1f & %.1f\\"
          % (e["median_us"], e["p95_us"], e["p99_us"], e["p999_us"],
             e["max_us"]))
        A(r"\bottomrule")
        A(r"\end{tabular}")
        A(r"\end{table}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm names (default: the five "
                         "anytime arms). Available: " + ", ".join(ARMS))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the number of seed blocks (smoke tests)")
    ap.add_argument("--per-seed", type=int, default=None,
                    help="override cycles per seed block (smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the arms but do not touch the saved npz/json")
    ap.add_argument("--eps-safe-cost", action="store_true",
                    help="measure the cost of guard test (iii) and stop")
    ap.add_argument("--summarise", action="store_true",
                    help="re-print the saved summary and stop")
    ap.add_argument("--report", action="store_true",
                    help="cross-arm comparison + LaTeX from the saved results")
    ap.add_argument("--guard-audit", type=int, default=0, metavar="N",
                    help="UNTIMED: which safety test rejects the published "
                         "command, over N cycles")
    a = ap.parse_args()

    arrays, meta = load_existing()
    if not meta:
        meta = dict(what="end-to-end control-cycle latency, measured on one "
                         "platform; replaces the manuscript's three-platform "
                         "latency tables and its joint real-time success rate",
                    generated_by=os.path.basename(__file__))

    if a.summarise:
        for k, v in meta.get("arms", {}).items():
            print_arm(v)
        cross_arm_report(meta)
        return

    if a.report:
        # backfill core-class mixture for any arm measured before that block
        # existed; it is recomputed from the saved raw arrays, not re-timed.
        for arm in meta.get("arms", {}):
            k = "%s__cpu" % arm
            if k in arrays and "core_class_mixture" not in meta["arms"][arm]:
                cpu_of = np.asarray(arrays[k])
                tot = np.asarray(arrays["%s__total_ns" % arm])
                on_e = cpu_of >= 12
                meta["arms"][arm]["core_class_mixture"] = dict(
                    fraction_finishing_on_E_core=float(on_e.mean()),
                    fraction_finishing_on_P_core=float((~on_e).mean()),
                    median_us_when_finished_on_P=(
                        float(np.median(tot[~on_e])) / 1e3
                        if (~on_e).any() else None),
                    median_us_when_finished_on_E=(
                        float(np.median(tot[on_e])) / 1e3
                        if on_e.any() else None),
                    note=("logical 0-11 are P-core threads, 12-19 are "
                          "E-cores; the core number is sampled once per cycle "
                          "AFTER the timed window closes, so it is where the "
                          "cycle ended. Recomputed from the saved raw arrays."))
        comp = cross_arm_report(meta)
        meta["cross_arm_ratios"] = comp
        fx = {}
        for arm in meta.get("arms", {}):
            f = forensics(arrays, meta, arm)
            if f:
                fx[arm] = f
        meta["self_forensics"] = fx
        if "normal_pinned_P" in fx:
            f = fx["normal_pinned_P"]
            print("\nSELF-FORENSICS on the primary arm (the tests that "
                  "condemned the released traces, applied to this one):")
            print("  samples on the 100 ns clock lattice : %.4f  "
                  "(1.0 by construction -- that IS the clock)"
                  % f["lattice"]["frac_multiple_of_clock_resolution"])
            print("  samples on exact 1 us              : %.4f  "
                  "(chance %.4f)" % (f["lattice"]["frac_multiple_of_1us"],
                                     f["lattice"]["expected_by_chance_1us"]))
            print("  samples on exact 10 us             : %.5f (chance %.5f)"
                  % (f["lattice"]["frac_multiple_of_10us"],
                     f["lattice"]["expected_by_chance_10us"]))
            print("  distinct values                    : %d of %d (%.3f)"
                  % (f["lattice"]["distinct_values"], f["n"],
                     f["lattice"]["distinct_fraction"]))
            print("  order stats on integer us          : %d of %d"
                  % (f["order_statistics_on_integer_microseconds"]
                     ["n_landing_on_integer_us"],
                     f["order_statistics_on_integer_microseconds"]["n_reported"]))
            print("  acf lags 1-5                       : %s"
                  % ["%.4f" % v for v in f["acf_lag_1_to_20"][:5]])
            print("  per-seed median CV                 : %.4f"
                  % f["per_seed_median_cv"])
            print("  cycles inside the 800 us budget    : %d"
                  % f["histogram_100us_bins"]["n_below_budget"])
        # steady-state view of the primary arm, cut at the measured settle
        # point, computed from the saved raw array (not re-timed)
        prim = "normal_pinned_P"
        if prim in meta.get("arms", {}) and ("%s__total_ns" % prim) in arrays:
            cut = ((meta.get("self_forensics", {}).get(prim, {})
                    .get("warm_up", {}) or {}).get("cycles_to_settle") or 0)
            xs = np.asarray(arrays["%s__total_ns" % prim], dtype=np.int64)[cut:]
            ss = summarise_ns(xs)
            ss["cycles_discarded_as_warm_up"] = int(cut)
            ss["lag1_autocorr"] = lag1_autocorr(xs)
            ss["n_meeting_800us_budget"] = int(np.sum(xs <= BUDGET_NS))
            ss["note"] = ("the same measured samples with the first %d cycles "
                          "removed. The warm-up is real -- a deployed "
                          "controller also boots cold -- so this is offered "
                          "next to the full-series numbers, not instead of "
                          "them." % cut)
            meta["primary_arm_steady_state"] = ss
            print("\nprimary arm, warm-up cut at cycle %d: median %.1f  P95 "
                  "%.1f  P99 %.1f  P99.9 %.1f  max %.1f us, %d/%d inside "
                  "800 us"
                  % (cut, ss["median_us"], ss["p95_us"], ss["p99_us"],
                     ss["p999_us"], ss["max_us"],
                     ss["n_meeting_800us_budget"], ss["n"]))

        # ---- the contention-proof statement -------------------------
        # This machine was NOT idle while these numbers were taken: a
        # concurrent workload of about ten compute-bound processes held it near
        # 50% utilisation throughout (see machine_load_before_arm on each arm).
        # Interference can only ever inflate a measured cost, never deflate it,
        # so the MINIMUM over a long run is a lower bound on the cost that is
        # immune to it. If even the minimum misses the deadline, the deadline
        # verdict does not depend on how busy the box was.
        mins = {}
        for arm in meta.get("arms", {}):
            k = "%s__total_ns" % arm
            if k in arrays:
                x = np.asarray(arrays[k], dtype=np.int64)
                o = np.asarray(arrays["%s__optimize_ns" % arm], dtype=np.int64)
                mins[arm] = dict(
                    n=int(x.size),
                    min_end_to_end_us=float(x.min()) / 1e3,
                    min_optimize_stage_us=float(o.min()) / 1e3,
                    min_over_budget_factor=float(x.min()) / 1e3 / (BUDGET * 1e6),
                    min_optimize_over_tau_O_factor=float(o.min()) / 1e3
                    / (TAU_O * 1e6))
        pooled = [np.asarray(arrays["%s__total_ns" % a], dtype=np.int64)
                  for a in ("normal_pinned_P", "normal_pinned_P_repeat")
                  if "%s__total_ns" % a in arrays]
        meta["contention_proof_lower_bound"] = dict(
            per_arm=mins,
            pooled_pinned_P_runs=(dict(
                n=int(sum(p.size for p in pooled)),
                min_us=float(min(p.min() for p in pooled)) / 1e3,
                n_inside_800us=int(sum(int((p <= BUDGET_NS).sum())
                                       for p in pooled)))
                if pooled else None),
            method=("the machine carried a concurrent compute-bound load of "
                    "about ten processes at ~50% utilisation for the whole "
                    "measurement. Contention inflates a measured duration and "
                    "cannot shorten one, so the minimum over a long run "
                    "bounds the cost from below regardless of that load."),
            verdict=("even the single fastest cycle observed exceeds the "
                     "800 us budget, and even the single fastest Optimization "
                     "stage exceeds tau_O = 600 us, so neither the deadline "
                     "verdict nor the tau_O verdict is an artefact of "
                     "background interference."))

        meta["measurement_environment"] = dict(
            machine_was_idle=False,
            description=("Ten other compute-bound Python processes were "
                         "running on this machine for the whole of this "
                         "phase, holding it at roughly 50% of 20 logical "
                         "processors. The per-arm probe "
                         "machine_load_before_arm records it directly for "
                         "every arm measured after the probe was added; the "
                         "same processes were running before that, having "
                         "started about an hour earlier and still being "
                         "resident afterwards."),
                consequence=("This inflates every number in this file. It "
                             "does not change the deadline verdict, because "
                             "contention cannot make a duration shorter and "
                             "the measured MINIMUM already misses the budget "
                             "by a factor of about 2.4 -- see "
                             "contention_proof_lower_bound."),
            current_load=background_load())

        cb = meta["contention_proof_lower_bound"]
        print("\nCONTENTION-PROOF LOWER BOUND (the machine was ~50%% busy "
              "throughout; interference can only inflate):")
        for arm, v in cb["per_arm"].items():
            print("  %-24s min end-to-end %8.1f us = %.2fx the 800 us budget; "
                  "min Optimization %8.1f us = %.2fx tau_O"
                  % (arm, v["min_end_to_end_us"], v["min_over_budget_factor"],
                     v["min_optimize_stage_us"],
                     v["min_optimize_over_tau_O_factor"]))
        if cb["pooled_pinned_P_runs"]:
            p = cb["pooled_pinned_P_runs"]
            print("  pooled over the two pinned-P runs: n=%d, fastest cycle "
                  "%.1f us, cycles inside 800 us = %d"
                  % (p["n"], p["min_us"], p["n_inside_800us"]))

        prim_arm = meta.get("arms", {}).get("normal_pinned_P")
        if prim_arm:
            meta["replaces_published"] = dict(
                published_median_ms="0.77-0.79",
                measured_median_ms=prim_arm["end_to_end"]["median_us"] / 1e3,
                published_joint_realtime_success_rate=0.780,
                measured_deadline_success_rate=
                    prim_arm["deadline"]["success_rate"],
                measured_deadline_success_clopper_pearson_95=
                    prim_arm["deadline"]["clopper_pearson_95"],
                published_platforms=["A (Intel i7)", "B (Jetson AGX Xavier)",
                                     "C (Cortex-A72)"],
                measured_platforms=["Intel Core i5-14600KF, Windows 11 Pro "
                                    "build 26200"],
                published_os_arms=["SCHED_OTHER", "chrt", "isolcpus"],
                measured_os_arms=["NORMAL_PRIORITY_CLASS + affinity",
                                  "HIGH_PRIORITY_CLASS + affinity",
                                  "REALTIME requested, HIGH granted + "
                                  "affinity", "NORMAL, unpinned"],
                measured_os_arms_are="a Windows ANALOGUE, not a reproduction",
                verdict=("the measured cycle is %.0fx the published median and "
                         "misses the 800 us budget on every one of %d cycles "
                         "in every configuration tested. The replacement "
                         "number is worse than the published one, which is "
                         "the expected consequence of replacing an "
                         "unreproducible claim with a measured one."
                         % (prim_arm["end_to_end"]["median_us"] / 780.0,
                            prim_arm["n_cycles"])))

        tex = latex_table(meta)
        tex_path = os.path.join(OUT_DIR, "cycle_latency_tables.tex")
        with open(tex_path, "w") as fh:
            fh.write(tex + "\n")
        meta["latex_tables_file"] = os.path.basename(tex_path)
        with open(JSON_PATH, "w") as fh:
            json.dump(meta, fh, indent=2)
        print("\nwrote %s" % tex_path)
        print("\n" + tex)
        return

    if a.guard_audit:
        set_affinity([P_CPU])
        c = guard_audit(a.guard_audit)
        meta["guard_audit"] = c
        print("guard audit over %d cycles (UNTIMED):" % c["n"])
        for k in ("admissible", "fail_test_i_z", "fail_test_ii_range",
                  "fail_actuator_abs", "fail_actuator_slew",
                  "no_feasible_candidate"):
            print("  %-24s %6d   %.4f" % (k, c[k], c["rates"][k]))
        with open(JSON_PATH, "w") as fh:
            json.dump(meta, fh, indent=2)
        print("updated %s" % JSON_PATH)
        return

    if a.eps_safe_cost:
        set_affinity([P_CPU])
        v = verify_pin(P_CPU)
        print("pinned to logical %d: %s" % (P_CPU, v["all_probes_on_requested_cpu"]))
        meta.setdefault("guard_test_iii_cost", {})
        c = eps_safe_cost()
        c["pinning"] = v
        meta["guard_test_iii_cost"] = c
        for k in ("fast", "quad", "egc"):
            print("  system_aber(method=%-5s) median %12.1f us  = %8.1f x the "
                  "800 us budget" % (k, c[k]["median_us"],
                                     c[k]["times_the_800us_budget"]))
        save(arrays, meta)
        return

    names = ([s.strip() for s in a.arms.split(",")] if a.arms else DEFAULT_ARMS)
    for nm in names:
        if nm not in ARMS:
            sys.exit("unknown arm %r; available: %s" % (nm, ", ".join(ARMS)))
    if a.seeds is not None or a.per_seed is not None:
        for nm in names:
            if a.seeds is not None:
                ARMS[nm]["seeds"] = int(a.seeds)
            if a.per_seed is not None:
                ARMS[nm]["per_seed"] = int(a.per_seed)
        print("!! sample size OVERRIDDEN on the command line (smoke test): "
              "seeds=%s per_seed=%s" % (a.seeds, a.per_seed))

    print("=" * 78)
    print("end-to-end control-cycle latency, measured")
    print("=" * 78)
    res = clock_resolution_ns()
    print("clock            : time.perf_counter_ns, measured resolution %d ns"
          % res)
    meta["clock"] = dict(source="time.perf_counter_ns",
                         monotonic=bool(time.get_clock_info("perf_counter").monotonic),
                         measured_resolution_ns=res,
                         advertised_resolution_s=time.get_clock_info("perf_counter").resolution)
    meta["platform"] = platform_block()
    meta["scheduling_analogue"] = scheduling_analogue_block()
    meta["specification"] = dict(
        T_u_s=T_U, tau_A_s=TAU_A, computation_budget_s=BUDGET,
        tau_O_s=TAU_O, horizon=HORIZON, n_particles=SolverConfig().n_particles,
        T_iter=SolverConfig().max_iters,
        alpha=ALPHA, beta=BETA, gbar_dB=GBAR_DB, sigma_s_m=SIGMA_S,
        z_max=Z_MAX, eps_safe=EPS_SAFE, u_max_rad=U_MAX,
        u_slew_rad_per_cycle=U_SLEW, link_length_m=2000.0,
        tau_O_reference_point="start of the Optimization stage",
        note=("tau_O is polled at solver ITERATION boundaries, because that is "
              "the only stopping point hclpso_ga.minimise offers. If one "
              "iteration costs more than tau_O the checkpoint cannot fire "
              "before it, and the measured Optimization stage overruns tau_O "
              "by construction. Whether it does is reported, not assumed."))

    eq = check_equivalence()
    meta["instrumentation_equivalence_check"] = eq
    print("equivalence      : instrumented cycle vs BeamSteeringMPC.step() -> "
          "bit-identical: %s (best_f %.17g vs %.17g)"
          % (eq["bit_identical"], eq["instrumented_best_f"], eq["step_best_f"]))
    if not eq["bit_identical"]:
        sys.exit("instrumented cycle does not reproduce step(); refusing to "
                 "report timings for a different computation")

    max_seeds = max(ARMS[nm]["seeds"] for nm in names)
    max_per = max(ARMS[nm]["per_seed"] for nm in names)
    print("traces           : generating %d x %d disturbance samples "
          "(outside every timed window)" % (max_seeds, max_per))
    traces = make_traces(max_seeds, max_per)
    meta["disturbance_traces"] = dict(
        sway="channel.SwayProcess, OU, 500-step burn-in, seeds 1000+s",
        scintillation="channel.GammaGammaAR1(rho_a=0.98, calibrate=False), "
                      "seeds 2000+s, h_meas = h - 1",
        n_seed_blocks=int(max_seeds), cycles_per_seed=int(max_per),
        shared_across_arms=True,
        note="pre-recorded and replayed; the plant is outside the timed "
             "window and is not closed around the controller")

    meta.setdefault("arms", {})
    for nm in names:
        spec = ARMS[nm]
        print("\n-- arm %s" % nm)
        arr, summ = run_arm(nm, spec,
                            ([t[:spec["per_seed"]] for t in traces[0]],
                             [h[:spec["per_seed"]] for h in traces[1]]),
                            warmup=a.warmup)
        arrays.update(arr)
        meta["arms"][nm] = summ
        print_arm(summ)
        if a.dry_run:
            print("    [dry run: nothing written]")
        else:
            save(arrays, meta)

    print("\n" + "=" * 78)
    print("ARRAY KEYS IN %s" % os.path.basename(NPZ_PATH))
    for k in sorted(arrays):
        print("  %-42s %s %s" % (k, arrays[k].shape, arrays[k].dtype))


if __name__ == "__main__":
    main()
