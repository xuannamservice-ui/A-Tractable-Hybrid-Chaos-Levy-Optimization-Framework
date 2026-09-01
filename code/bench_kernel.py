"""
bench_kernel.py -- measured per-evaluation cost of the RT-ODT kernel and of the
double-precision and arbitrary-precision baselines it claims a speedup over.

WHAT THIS REPLACES
  Table 7 (tab:K_error_runtime) runtime column: 14.2 / 23.6 / 45.9 us per evaluation at
  K = 5 / 10 / 20 and 45000 us for the exact Meijer-G baseline; the "approximately 64x
  faster per evaluation than optimized double-precision quadrature" claim; and the
  0.6 us/particle SIMD figure.  None of those were reproducible from the release.  This
  script measures all of them on ONE machine and writes what it measured.

  Every number this script emits comes from time.perf_counter_ns().  Nothing is
  synthesised, smoothed or rounded to a nicer value.  The full per-sample arrays are
  written to a .npz beside the JSON so any percentile can be recomputed and the serial
  correlation checked independently.

WHAT IS MEASURED
  1. rtodt_fast.pe_series_f64 -- the deployed evaluator, exactly as released -- at
     K = 5, 10, 20, both single-candidate and vectorised across a swarm of N_p = 30.
  2. A double-precision quadrature baseline of the SAME per-branch ABER, in two forms:
       B1  nested scipy.integrate.quad (adaptive Gauss-Kronrod) with scipy.special.kv,
           which is the baseline the manuscript describes in prose;
       B2  the cheapest fixed-order tensor rule found in a searched family that reaches
           a stated accuracy target -- an "optimized" baseline in the sense that matters,
           i.e. the fastest one we could build that is still as accurate as the series is
           certified to be.  Using the FASTEST fair baseline makes the reported speedup
           the smallest defensible one rather than the largest.
  3. The arbitrary-precision Meijer-G reference: mpmath integration of the exact
     Farid-Hranilovic composite density against Q(sqrt(gbar) h).
  4. The speedup as the ratio of two measured medians, both of which are printed.

THE HYBRID-CORE PROBLEM
  This is an Intel i5-14600KF: 6 P-cores with SMT (logical 0-11) and 8 E-cores without
  (logical 12-19).  An unpinned measurement is a mixture of two processors whose
  per-evaluation cost differs by more than 2x, and its tail means nothing.  Every timed
  block here is pinned to one logical processor and the pin is verified twice: the
  affinity mask is read back, and GetCurrentProcessorNumber() is sampled and must return
  the requested processor every time.  The P-core/E-core difference for THIS kernel is
  measured and reported so a reader can see the size of what pinning removes.

  A second, less obvious effect is measured here too and it dominates our tail: SMT
  sibling contention.  Pinning to logical processor 2 does not stop Windows scheduling
  other work on logical processor 3, which is the other thread of the same physical
  P-core.  When that happens this kernel costs about twice as much.  The bimodal
  distribution reported below is that effect, not measurement error, and the script
  demonstrates it with a controlled loaded-sibling arm.

WHAT CANNOT BE REPRODUCED HERE
  The manuscript's three OS-tuning arms (SCHED_OTHER, chrt, isolcpus) are Linux
  mechanisms and do not exist on Windows.  This script runs a priority-class arm
  (NORMAL / HIGH / REALTIME_PRIORITY_CLASS with fixed affinity) and labels it an
  ANALOGUE.  It is not a reproduction of the Linux arms and must never be presented as
  one; Windows is a general-purpose scheduler with no real-time guarantee.

USAGE
  python bench_kernel.py [--cpu 2] [--ecore 12] [--reps 20000] [--out PATH]
"""
from __future__ import annotations

# BLAS threading must be capped BEFORE numpy is imported, so that a measurement pinned
# to one logical processor is not silently serviced by a thread pool spanning both core
# classes.
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import ctypes
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc, gamma as sp_gamma, kv, roots_jacobi

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import mpmath as mp                                    # noqa: E402
from rtodt import A0_for, REGIMES, Pe_series, db, z_param   # noqa: E402
from rtodt_fast import pe_series_f64                   # noqa: E402

IS_WINDOWS = (os.name == "nt")
SQ2 = math.sqrt(2.0)

# ---------------------------------------------------------------- configuration
ORDERS = (5, 10, 20)
N_SWARM = 30                       # N_p of the manuscript
GBAR_DB = 38.0                     # gbar_op, the campaign's reference SNR
SIGMA_S = "0.05"                   # jitter of the Fig. odt_validation configuration
XI_REF = "1.967"                   # the configuration Fig. odt_validation is drawn at
# Accuracy-check points.  These are three of the eleven pole-free nodes of the
# manuscript; off-node xi would put Gamma(alpha - xi^2) on a pole for some regimes and
# the comparison would be about the pole, not about the quadrature.
XI_CHECK = ("1.266", "1.967", "2.511")
# Accuracy targets the double-precision baseline is required to reach.  eps_req is the
# manuscript's own requirement; 1e-9 is the scale of the Table 7 certified per-branch
# bound at K=10 (3.98e-9 / 5.49e-10 / 7.87e-10); 1e-12 is three decades tighter and near
# the float64 floor for a quantity of order 1e-1.
TARGETS = (1e-6, 1e-9, 1e-12)
HEADLINE_TARGET = 1e-9


# =====================================================================
# clock
# =====================================================================
def measure_clock(n_samples=400_000, n_calls=200_000):
    """Measure the clock rather than assume it."""
    pc = time.perf_counter_ns
    best = None
    deltas_seen = 0
    prev = pc()
    for _ in range(n_samples):
        cur = pc()
        d = cur - prev
        if d > 0:
            deltas_seen += 1
            if best is None or d < best:
                best = d
            prev = cur
    t0 = pc()
    for _ in range(n_calls):
        pc()
    t1 = pc()
    info = time.get_clock_info("perf_counter")
    return {
        "clock": "time.perf_counter_ns",
        "monotonic": bool(info.monotonic),
        "implementation": info.implementation,
        "reported_resolution_s": info.resolution,
        "measured_min_nonzero_increment_ns": best,
        "n_probe_calls": n_samples,
        "fraction_of_calls_returning_same_value": round(1.0 - deltas_seen / n_samples, 6),
        "measured_call_cost_ns": round((t1 - t0) / n_calls, 3),
    }


# =====================================================================
# pinning: set, then VERIFY (twice)
# =====================================================================
try:
    import psutil
except Exception:
    psutil = None

if IS_WINDOWS:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcessorNumber.restype = ctypes.c_ulong


def current_processor_number():
    if IS_WINDOWS:
        return int(_k32.GetCurrentProcessorNumber())
    if hasattr(os, "sched_getcpu"):
        return os.sched_getcpu()
    return None


def get_affinity():
    if psutil is not None:
        return sorted(psutil.Process().cpu_affinity())
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def set_affinity(cpus):
    cpus = list(cpus)
    if psutil is not None:
        psutil.Process().cpu_affinity(cpus)
        return True
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
        return True
    if IS_WINDOWS:                      # ctypes fallback, no psutil required
        mask = 0
        for c in cpus:
            mask |= (1 << c)
        h = _k32.GetCurrentProcess()
        return bool(_k32.SetProcessAffinityMask(ctypes.c_void_p(h),
                                                ctypes.c_size_t(mask)))
    return False


def pin_and_verify(cpu, n_probes=2000):
    """Pin to one logical processor and prove it took effect.

    Setting an affinity mask is a request.  Check (a) reads the mask back from the OS;
    check (b) asks the kernel, repeatedly and while the thread is kept runnable, which
    logical processor is actually executing us.  Both must agree with the request.
    """
    ok = set_affinity([cpu])
    mask = get_affinity()
    seen = {}
    sink = 0.0
    for i in range(n_probes):
        c = current_processor_number()
        seen[c] = seen.get(c, 0) + 1
        sink += i * 1.000001
    return {
        "requested_cpu": cpu,
        "affinity_api_ok": bool(ok),
        "affinity_mask_readback": mask,
        "mask_matches_request": (mask == [cpu]),
        "observed_processor_numbers": {str(k): v for k, v in sorted(seen.items())},
        "all_probes_on_requested_cpu": (set(seen) == {cpu}),
        "n_probes": n_probes,
        "_sink": sink,
    }


def set_priority(name):
    """Windows priority class / POSIX nice.  Returns what the OS reports back."""
    if psutil is None:
        return {"requested": name, "applied": False, "reason": "psutil unavailable"}
    p = psutil.Process()
    try:
        if IS_WINDOWS:
            cls = {"NORMAL": psutil.NORMAL_PRIORITY_CLASS,
                   "HIGH": psutil.HIGH_PRIORITY_CLASS,
                   "REALTIME": psutil.REALTIME_PRIORITY_CLASS}[name]
        else:
            cls = {"NORMAL": 0, "HIGH": -10, "REALTIME": -20}[name]
        p.nice(cls)
        return {"requested": name, "applied": True, "readback_nice": int(p.nice())}
    except Exception as exc:
        return {"requested": name, "applied": False, "reason": repr(exc)}


# =====================================================================
# timing harness
# =====================================================================
def time_calls(fn, reps, warmup, check_cpu=None):
    """One timed sample per call.

    Every call of every kernel timed here is long compared with the measured clock
    resolution (100 ns) and with the cost of one perf_counter_ns call (~35 ns on a
    P-core), so no inner repetition loop is needed and each element of the returned
    array is a genuine single-evaluation sample.  That is what makes the serial
    correlation of the array meaningful.
    """
    for _ in range(warmup):
        fn()
    out = np.empty(reps, dtype=np.float64)
    pc = time.perf_counter_ns
    for i in range(reps):
        t0 = pc()
        fn()
        out[i] = pc() - t0
    cpus = None
    if check_cpu is not None:
        cpus = sorted({current_processor_number() for _ in range(64)})
    return out, cpus


def time_calls_interleaved(fns, reps, chunk=250, warmup=2000, check_cpu=None):
    """Time several configurations ROUND-ROBIN rather than one after another.

    Background load on this machine is bursty and block-structured -- a single
    contiguous block can sit entirely inside a quiet window or entirely inside a noisy
    one, which makes the medians of consecutively-timed configurations
    incomparable.  Cycling through the configurations in small chunks spreads every
    configuration over the whole measurement window, so they all see the same ambient
    load.  Samples are stored in acquisition order, so the serial correlation of each
    array is still the serial correlation of a real time series (of its own chunks).

    `fns` is an ordered dict {name: callable}.
    """
    names = list(fns)
    for nm in names:
        f = fns[nm]
        for _ in range(warmup):
            f()
    bufs = {nm: np.empty(reps, dtype=np.float64) for nm in names}
    pc = time.perf_counter_ns
    done = 0
    while done < reps:
        c = min(chunk, reps - done)
        for nm in names:
            f = fns[nm]
            b = bufs[nm]
            for i in range(done, done + c):
                t0 = pc()
                f()
                b[i] = pc() - t0
        done += c
    cpus = None
    if check_cpu is not None:
        cpus = sorted({current_processor_number() for _ in range(64)})
    return bufs, cpus


def summarize(ns, per=1):
    """Distribution of per-evaluation cost in microseconds.  `per` divides a swarm
    total into an amortised per-candidate figure."""
    a = np.asarray(ns, dtype=np.float64) / (1e3 * per)
    s = np.sort(a)
    n = len(s)

    def pct(p):
        return float(np.percentile(s, p))

    lag1 = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if n > 2 and a.std() > 0 else None
    return {
        "n": int(n),
        "unit": "us_per_evaluation",
        "min": float(s[0]),
        "p05": pct(5), "p25": pct(25),
        "median": float(np.median(s)),
        "mean": float(a.mean()), "std": float(a.std(ddof=1)),
        "p75": pct(75), "p95": pct(95), "p99": pct(99), "p999": pct(99.9),
        "max": float(s[-1]),
        "iqr": pct(75) - pct(25),
        "lag1_autocorrelation": lag1,
        "n_distinct_values": int(np.unique(a).size),
        # Objective handle on the bimodality: on this part the slow mode is an SMT-
        # contended physical core and costs about twice the fast mode, so anything
        # above 1.5x the observed floor is in the contended mode.
        "fraction_above_1p5x_min": float((a > 1.5 * s[0]).mean()),
        "median_fast_mode": (float(np.median(a[a <= 1.5 * s[0]]))
                             if np.any(a <= 1.5 * s[0]) else None),
        "median_slow_mode": (float(np.median(a[a > 1.5 * s[0]]))
                             if np.any(a > 1.5 * s[0]) else None),
    }


# =====================================================================
# the geometry / operating points
# =====================================================================
def build_points():
    mp.mp.dps = 90
    sig = mp.mpf(SIGMA_S)
    gbar = db(GBAR_DB)
    pts = {}
    for xs in XI_CHECK:
        xi = mp.mpf(xs)
        a0 = A0_for(xi, sig)
        pts[xs] = {
            "xi": float(xi), "A0": float(a0),
            "z": {r: float(z_param(*REGIMES[r], a0, gbar)) for r in REGIMES},
        }
    # the swarm: N_p candidates spread over the beam-width decision axis, with A_0
    # taken from the same geometry the rest of the package uses, not invented.
    xi_sw = np.linspace(1.0, 2.4, N_SWARM)
    a0_sw = np.array([float(A0_for(mp.mpf(float(x)), sig)) for x in xi_sw])
    return pts, xi_sw, a0_sw, float(gbar)


def high_precision_reference(regime, xi_str, a0, gbar_mp, K=60):
    """mpmath evaluation of eq. (19) at 90 digits and K=60.  Cross-checked below
    against an independent Meijer-G integration, so it is not self-referential."""
    mp.mp.dps = 90
    A, B = REGIMES[regime]
    return float(Pe_series(A, B, mp.mpf(xi_str), mp.mpf(a0), gbar_mp, K))


# =====================================================================
# baseline B0: arbitrary-precision Meijer-G
# =====================================================================
def meijerg_aber_factory(regime, xi, A0, gbar, dps=15):
    """Exact ABER by arbitrary-precision integration of the Farid-Hranilovic composite
    density, which is a Meijer-G:

        f_h(h) = alpha beta xi^2 / (A_0 G(alpha) G(beta))
                 * G^{3,0}_{1,3}( alpha beta h / A_0 | xi^2 ; xi^2-1, alpha-1, beta-1 )

        ABER   = Int_0^inf Q(sqrt(gbar) h) f_h(h) dh

    The upper limit is cut at h_max = 15/sqrt(gbar), beyond which Q(sqrt(gbar) h) <
    Q(15) = 3.7e-51 and the density integrates to at most 1, so the discarded tail is
    below 4e-51 -- forty decades under the 15 significant digits requested.  This is a
    bound, not an assumption.
    """
    A, B = REGIMES[regime]

    def run():
        mp.mp.dps = dps
        a, b = mp.mpf(A), mp.mpf(B)
        x2 = mp.mpf(xi) ** 2
        a0 = mp.mpf(A0)
        g = mp.mpf(gbar)
        sg = mp.sqrt(g)
        pref = a * b * x2 / (a0 * mp.gamma(a) * mp.gamma(b))

        def integrand(h):
            fh = pref * mp.meijerg([[], [x2]], [[x2 - 1, a - 1, b - 1], []],
                                   a * b * h / a0, zeroprec=100000, maxterms=10 ** 6)
            return mp.erfc(sg * h / mp.sqrt(2)) / 2 * fh

        return mp.quad(integrand, [0, 1 / sg, 5 / sg, 15 / sg])

    return run


# =====================================================================
# baseline B1: nested adaptive Gauss-Kronrod (scipy.integrate.quad)
# =====================================================================
def _gg_pdf_scalar(x, a, b):
    c = 2.0 * (a * b) ** ((a + b) / 2.0) / (sp_gamma(a) * sp_gamma(b))
    return c * x ** ((a + b) / 2.0 - 1.0) * float(kv(a - b, 2.0 * math.sqrt(a * b * x)))


def quad_aber_factory(regime, xi, A0, gbar, tol, lo=-40.0, hi=16.0, limit=200):
    """Nested adaptive Gauss-Kronrod of the same per-branch ABER.

        Pe = Int_0^1 [ Int Q(sqrt(gbar) A_0 t x) f_gg(x) dx ] xi^2 t^(xi^2-1) dt

    The inner integral is taken in log x, which removes the x^{(a+b)/2-1}K_{a-b} endpoint
    singularity of the gamma-gamma density; leaving it in would make the baseline slower
    and the speedup larger, so it is removed.  Bessel functions are scipy.special.kv in
    double precision, matching the baseline the manuscript describes in prose.
    """
    a, b = float(REGIMES[regime][0]), float(REGIMES[regime][1])
    x2 = xi * xi
    sg = math.sqrt(gbar)

    def run():
        def inner(t):
            c = sg * A0 * t

            def g(u):
                x = math.exp(u)
                return 0.5 * math.erfc(c * x / SQ2) * _gg_pdf_scalar(x, a, b) * x

            return quad(g, lo, hi, epsabs=tol * 0.1, epsrel=tol * 0.1, limit=limit)[0]

        def outer(t):
            return inner(t) * x2 * t ** (x2 - 1.0)

        return quad(outer, 0.0, 1.0, epsabs=tol, epsrel=tol, limit=limit)[0]

    return run


# =====================================================================
# baseline B2: fixed-order tensor rules (the "optimized" double-precision baseline)
# =====================================================================
class TensorQuad:
    """Product rule for the same double integral.

    Inner (turbulence) dimension: Gauss-Legendre in log x.  Everything on that axis
    depends only on (alpha, beta) -- the turbulence regime, which is a per-cycle channel
    estimate, not a per-candidate quantity -- so the Bessel evaluations and weights are
    precomputed in __init__ and are NOT charged to the per-evaluation cost.  That is
    deliberately generous to the baseline.

    Outer (pointing) dimension, two variants:
      'GL'  Gauss-Legendre in t with the explicit weight xi^2 t^(xi^2-1).  Nodes are
            xi-free so they are precomputed; only the weight vector is per-candidate.
            Converges algebraically because t^(xi^2-1) has an unbounded derivative at
            t=0 whenever xi^2 < 2.
      'GJ'  Gauss-Jacobi with (alpha_J, beta_J) = (0, xi^2-1), which absorbs that
            singularity into the weight function and converges spectrally.  Its nodes
            depend on xi, so roots_jacobi IS charged to the per-evaluation cost -- a
            candidate's xi is what the optimizer varies.

    Only per-candidate work is inside the timed region in both variants.
    """

    def __init__(self, regime, rule, nt, nx, lo=-30.0, hi=10.0):
        a, b = float(REGIMES[regime][0]), float(REGIMES[regime][1])
        self.a, self.b, self.rule, self.nt, self.nx = a, b, rule, nt, nx
        gx, wx = np.polynomial.legendre.leggauss(nx)
        u = 0.5 * (gx + 1.0) * (hi - lo) + lo
        x = np.exp(u)
        c = 2.0 * (a * b) ** ((a + b) / 2.0) / (sp_gamma(a) * sp_gamma(b))
        f = c * x ** ((a + b) / 2.0 - 1.0) * kv(a - b, 2.0 * np.sqrt(a * b * x))
        f = np.where(np.isfinite(f), f, 0.0)
        self.x = x
        self.fw = f * wx * 0.5 * (hi - lo) * x        # density * dx, precomputed
        if rule == "GL":
            gt, wt = np.polynomial.legendre.leggauss(nt)
            self.t = 0.5 * (gt + 1.0)
            self.wg = wt * 0.5

    def _outer(self, xi):
        x2 = xi * xi
        if self.rule == "GL":
            return self.t, self.wg * x2 * self.t ** (x2 - 1.0)
        s, w = roots_jacobi(self.nt, 0.0, x2 - 1.0)
        return 0.5 * (s + 1.0), w * x2 * 0.5 ** x2

    def __call__(self, xi, A0, gbar):
        t, w = self._outer(xi)
        arg = (math.sqrt(gbar) * A0 / SQ2) * np.outer(t, self.x)
        return float(w @ (0.5 * erfc(arg) @ self.fw))

    def swarm(self, xi_arr, A0_arr, gbar, max_elems=2_000_000):
        """Vectorised across a swarm.  For 'GL' the outer nodes are shared so the whole
        swarm becomes one chunked 3-D tensor contraction; for 'GJ' the nodes differ per
        candidate and no such sharing exists, which is itself a finding about how far a
        quadrature baseline can be SIMD-amortised."""
        n = len(xi_arr)
        out = np.empty(n)
        if self.rule == "GL":
            chunk = max(1, int(max_elems // (self.nt * self.nx)))
            k = math.sqrt(gbar) / SQ2
            for i in range(0, n, chunk):
                xs = xi_arr[i:i + chunk]
                x2 = xs * xs
                w = self.wg[None, :] * x2[:, None] * self.t[None, :] ** (x2[:, None] - 1.0)
                arg = (k * A0_arr[i:i + chunk])[:, None, None] * \
                    (self.t[:, None] * self.x[None, :])[None, :, :]
                v = 0.5 * erfc(arg) @ self.fw
                out[i:i + chunk] = (w * v).sum(1)
            return out
        for i in range(n):
            out[i] = self(float(xi_arr[i]), float(A0_arr[i]), gbar)
        return out


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", type=int, default=2,
                    help="logical processor to pin to (default 2: the first P-core "
                         "logical processor that is not cpu0, chosen a priori)")
    ap.add_argument("--ecore", type=int, default=12,
                    help="an E-core logical processor, for the core-class arm")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--reps-quad", type=int, default=40)
    ap.add_argument("--reps-meijer", type=int, default=25)
    ap.add_argument("--reps-tensor", type=int, default=2000)
    ap.add_argument("--reps-headline", type=int, default=400,
                    help="repetitions in the interleaved series-vs-B2 block")
    ap.add_argument("--reps-slow", type=int, default=25,
                    help="repetitions in the interleaved series-vs-B1-vs-Meijer-G block")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "10_platform",
                                                  "kernel_timing.json"))
    ap.add_argument("--skip-smt-arm", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    out_json = os.path.abspath(args.out)
    out_npz = os.path.splitext(out_json)[0] + "_raw.npz"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    raw = {}                      # name -> ns sample array, written to the .npz
    rec = {
        "schema": "rtodt.kernel_timing/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": os.path.basename(__file__),
        "purpose": ("Measured replacement for the Table 7 runtime column, the 64x "
                    "double-precision-quadrature speedup claim and the 0.6 us/particle "
                    "SIMD figure.  Single platform, pinned and verified."),
        "honesty_note": ("Every timing figure in this file is an output of "
                         "time.perf_counter_ns on this machine.  No value was "
                         "synthesised, smoothed, rounded to a lattice or otherwise "
                         "constructed.  Full per-sample arrays are in %s."
                         % os.path.basename(out_npz)),
    }

    original_affinity = get_affinity()
    print("[i] original process affinity: %s" % original_affinity)

    # ---------------------------------------------------------------- environment
    rec["environment"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
        "python": sys.version,
        "python_impl": platform.python_implementation(),
        "numpy": np.__version__,
        "mpmath": mp.__version__,
        "psutil": (psutil.__version__ if psutil else None),
        "blas_threads_env": {k: os.environ.get(k) for k in
                             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                              "VECLIB_MAXIMUM_THREADS")},
    }
    try:
        import scipy
        rec["environment"]["scipy"] = scipy.__version__
    except Exception:
        pass

    # ---------------------------------------------------------------- pin + clock
    pin = pin_and_verify(args.cpu)
    pin.pop("_sink", None)
    print("[i] pinned to cpu %d: mask=%s verified=%s"
          % (args.cpu, pin["affinity_mask_readback"], pin["all_probes_on_requested_cpu"]))
    if not (pin["mask_matches_request"] and pin["all_probes_on_requested_cpu"]):
        print("[!] PIN NOT VERIFIED -- every timing below is a mixture of processors "
              "and its tail is not interpretable.")
    prio = set_priority("NORMAL")
    print("[i] measuring clock ...")
    clk = measure_clock()
    print("    perf_counter_ns: min non-zero increment %s ns, call cost %.1f ns"
          % (clk["measured_min_nonzero_increment_ns"], clk["measured_call_cost_ns"]))

    rec["core_pinning"] = {
        "mechanism": ("Windows processor affinity via SetProcessAffinityMask, invoked "
                      "through psutil.Process().cpu_affinity([i]); a direct ctypes "
                      "SetProcessAffinityMask fallback and an os.sched_setaffinity path "
                      "are both present so the script runs without psutil and on Linux."),
        "verification": ("Two independent checks: the affinity mask is read back from "
                         "the OS and compared against the request, and "
                         "kernel32.GetCurrentProcessorNumber() is sampled 2000 times "
                         "while the thread is kept runnable and must return the "
                         "requested logical processor every time.  Setting a mask is a "
                         "request, not a guarantee of placement."),
        "timing_cpu": args.cpu,
        "timing_cpu_class": "P-core (performance), logical processors 0-11 on this part",
        "cpu_choice_rule": ("fixed a priori before any timing was taken: the first "
                            "P-core logical processor other than cpu0, because cpu0 "
                            "fields the bulk of Windows DPC/interrupt work.  No core "
                            "was selected by looking at its measured cost."),
        "verify_record": pin,
        "process_priority": prio,
    }
    rec["clock"] = clk

    # ---------------------------------------------------------------- geometry
    print("[i] building operating points ...")
    pts, xi_sw, a0_sw, gbar = build_points()
    gbar_mp = db(GBAR_DB)
    rec["operating_point"] = {
        "gbar_dB": GBAR_DB, "gbar_linear": gbar,
        "sigma_s_m": float(SIGMA_S), "aperture_a_m": 0.05,
        "regimes": {k: [float(v[0]), float(v[1])] for k, v in REGIMES.items()},
        "accuracy_check_points": pts,
        "swarm": {"N_p": N_SWARM,
                  "xi": xi_sw.tolist(), "A0": a0_sw.tolist(),
                  "note": ("xi spread linearly over the beam-width decision axis; A_0 "
                           "computed from the same Farid-Hranilovic geometry the rest "
                           "of the package uses (rtodt.A0_for), not invented.")},
    }

    # ---------------------------------------------------------------- references
    print("[i] high-precision references (mpmath, 90 digits, K=60) ...")
    refs = {}
    for reg in REGIMES:
        for xs in XI_CHECK:
            refs[(reg, xs)] = high_precision_reference(reg, xs, pts[xs]["A0"], gbar_mp)

    print("[i] cross-checking the reference against an independent Meijer-G "
          "integration ...")
    xcheck = []
    for reg in REGIMES:
        for xs in XI_CHECK:
            f = meijerg_aber_factory(reg, pts[xs]["xi"], pts[xs]["A0"], gbar, dps=25)
            try:
                v = float(f())
                r = refs[(reg, xs)]
                xcheck.append({"regime": reg, "xi": xs, "series_K60_dps90": r,
                               "meijerg_dps25": v,
                               "abs_diff": abs(v - r),
                               "rel_diff": abs(v - r) / abs(r) if r else None})
            except Exception as exc:
                xcheck.append({"regime": reg, "xi": xs, "error": repr(exc)})
            finally:
                mp.mp.dps = 90
    worst_x = max((c["rel_diff"] for c in xcheck if c.get("rel_diff") is not None),
                  default=None)
    print("    worst relative disagreement series-vs-MeijerG: %s" % worst_x)
    rec["reference_validation"] = {
        "what": ("The accuracy of every kernel and every baseline below is measured "
                 "against a 90-digit mpmath evaluation of eq. (19) at K=60.  That "
                 "reference is itself validated here against a completely independent "
                 "construction -- arbitrary-precision integration of the "
                 "Farid-Hranilovic Meijer-G composite density -- so the accuracy claims "
                 "are not circular."),
        "comparisons": xcheck,
        "worst_relative_disagreement": worst_x,
    }

    # ---------------------------------------------------------------- 1. THE KERNEL
    print("[i] timing the deployed kernel (rtodt_fast.pe_series_f64) ...")
    A_s, B_s = (float(REGIMES["strong"][0]), float(REGIMES["strong"][1]))
    xi1 = np.array([pts[XI_REF]["xi"]])
    a01 = np.array([pts[XI_REF]["A0"]])

    # Warm the coefficient caches.  _KC_CACHE holds the xi-free K_k(alpha,beta) of
    # eq. (21), computed once per (regime, K) at 120 digits; _C_CACHE holds the power
    # moments C(s, gbar).  Both are offline quantities in the deployed system, so their
    # construction is deliberately outside the timed region -- but they are built here,
    # by this process, not shipped as an artefact.
    kernel_fns, kernel_per = {}, {}
    for reg in REGIMES:
        A_r, B_r = float(REGIMES[reg][0]), float(REGIMES[reg][1])
        for K in ORDERS:
            for tag, xx, aa, per in (("single", xi1, a01, 1),
                                     ("swarm%d" % N_SWARM, xi_sw, a0_sw, N_SWARM)):
                key = "kernel_%s_K%d_%s" % (reg, K, tag)
                kernel_fns[key] = (lambda A=A_r, B=B_r, x=xx, a=aa, k=K:
                                   pe_series_f64(A, B, x, a, gbar, k))
                kernel_per[key] = per
    bufs, cpus = time_calls_interleaved(kernel_fns, args.reps, chunk=200,
                                        warmup=2000, check_cpu=args.cpu)
    kernel_res = {}
    for key, ns in bufs.items():
        raw[key] = ns
        d = summarize(ns, per=kernel_per[key])
        d["per"] = kernel_per[key]
        d["cpus_observed_after_block"] = cpus
        kernel_res[key] = d
    for reg in REGIMES:
        for K in ORDERS:
            ks = kernel_res["kernel_%s_K%d_single" % (reg, K)]
            kw = kernel_res["kernel_%s_K%d_swarm%d" % (reg, K, N_SWARM)]
            print("    %-9s K=%2d  single median %7.2f (min %6.2f) us | swarm%d "
                  "median %7.2f us -> %6.3f us/particle (min %6.3f)"
                  % (reg, K, ks["median"], ks["min"], N_SWARM,
                     kw["median"] * N_SWARM, kw["median"], kw["min"]))

    # what the kernel actually returns at those points, and its accuracy
    acc = {}
    for reg in REGIMES:
        A_r, B_r = float(REGIMES[reg][0]), float(REGIMES[reg][1])
        for K in ORDERS:
            errs = {}
            for xs in XI_CHECK:
                v = float(pe_series_f64(A_r, B_r, pts[xs]["xi"], pts[xs]["A0"],
                                        gbar, K)[0])
                r = refs[(reg, xs)]
                errs[xs] = {"value": v, "reference": r, "abs_err": abs(v - r),
                            "rel_err": abs(v - r) / abs(r) if r else None,
                            "z": pts[xs]["z"][reg]}
            acc["%s_K%d" % (reg, K)] = errs
    rec["kernel_accuracy_vs_90digit_reference"] = acc
    sw = pe_series_f64(A_s, B_s, xi_sw, a0_sw, gbar, 10)
    rec["kernel_swarm_output_check"] = {
        "regime": "strong", "K": 10,
        "n_finite": int(np.isfinite(sw).sum()),
        "n_in_unit_half_interval": int(np.sum((sw >= 0.0) & (sw <= 0.5))),
        "n_total": int(sw.size),
        "note": ("candidates outside [0, 1/2] or non-finite are what the manuscript's "
                 "range test and envelope guard exist to reject; they are reported, not "
                 "hidden, and they do not change the cost of an evaluation."),
    }
    rec["kernel"] = kernel_res

    # ------------------------------------------------- 1b. CONTIGUOUS REFERENCE BLOCK
    # The arrays above were acquired ROUND-ROBIN, which deliberately destroys short-range
    # serial correlation: consecutive elements of one array are 18 configurations apart in
    # wall-clock time.  Their lag-1 autocorrelation is therefore a property of the
    # acquisition schedule, not of the machine, and must not be read as evidence about
    # either.  This block re-times ONE configuration contiguously so that a genuine
    # serial-correlation figure exists in the record.
    print("[i] contiguous reference block (for serial correlation) ...")
    fn_c = lambda: pe_series_f64(A_s, B_s, xi1, a01, gbar, 10)
    ns_c, _ = time_calls(fn_c, args.reps, warmup=2000)
    raw["contiguous_strong_K10_single"] = ns_c
    a_c = ns_c / 1e3
    thr = 1.5 * a_c.min()
    mode = (a_c > thr).astype(np.float64)
    rank = np.argsort(np.argsort(a_c)).astype(np.float64)
    trimmed = a_c[a_c <= np.percentile(a_c, 99.5)]

    def _ac(x, L):
        if len(x) <= L + 2 or x.std() == 0:
            return None
        return float(np.corrcoef(x[:-L], x[L:])[0, 1])

    acf = {}
    for L in (1, 2, 5, 10, 50, 200, 1000):
        acf["pearson_lag%d" % L] = _ac(a_c, L)
        acf["rank_lag%d" % L] = _ac(rank, L)
        acf["mode_indicator_lag%d" % L] = _ac(mode, L)
    acf["pearson_lag1_trimmed_at_p99p5"] = _ac(trimmed, 1)
    acf["interpretation"] = (
        "The raw Pearson lag-1 is small only because the variance of this array is "
        "dominated by a handful of millisecond-scale preemption spikes; Pearson "
        "correlation is not robust to those.  Trim the top 0.5%% and it rises to the "
        "rank statistic's value.  The rank and mode-indicator lag-1 figures show what is "
        "actually happening: the cost is strongly serially correlated, because it sits "
        "in one contention mode for tens to hundreds of consecutive evaluations at a "
        "time.  All three are reported so nobody has to take one of them on trust.")
    switches = int(np.sum(np.abs(np.diff(mode))))
    rec["contiguous_reference_block"] = {
        "what": ("The same strong-regime K=10 single-candidate evaluation, timed as one "
                 "uninterrupted run of %d calls rather than round-robin, so that the "
                 "serial correlation reported here is the machine's and not the "
                 "schedule's." % args.reps),
        "config": "strong regime, K=10, single candidate, cpu %d" % args.cpu,
        "stats": summarize(ns_c),
        "autocorrelation": acf,
        "mode_run_structure": {
            "threshold_us": float(thr),
            "n_mode_switches": switches,
            "mean_run_length_samples": (len(mode) / switches) if switches else len(mode),
            "note": ("the cost alternates between an uncontended and an SMT-contended "
                     "mode in long runs, not sample by sample; that block structure is "
                     "what produces the positive lag-1 autocorrelation"),
        },
        "why_this_matters": ("A latency trace with lag-1 autocorrelation of about zero "
                             "is one of the signatures that condemned the three released "
                             "trace files.  The interleaved arrays in this file also have "
                             "near-zero lag-1, but for a stated and checkable reason -- "
                             "the acquisition is chunked round-robin -- and this "
                             "contiguous block is supplied so the machine's actual serial "
                             "correlation is on the record."),
    }
    print("    lag-1 autocorrelation (contiguous) = %.4f, %d mode switches in %d samples"
          % (acf.get("lag1", float("nan")), switches, len(mode)))

    # ---------------------------------------------------------------- 2. CORE CLASS
    print("[i] core-class arm: same kernel across P-cores and E-cores ...")
    # Every P-core primary SMT thread plus two E-cores, timed ROUND-ROBIN with a re-pin
    # before each chunk, so no core is measured only during a quiet or a noisy window.
    sweep_cpus = [c for c in (0, 2, 4, 6, 8, 10) if c < (os.cpu_count() or 20)]
    sweep_cpus += [c for c in (args.ecore, args.ecore + 1) if c < (os.cpu_count() or 20)]
    fn_k10 = lambda: pe_series_f64(A_s, B_s, xi1, a01, gbar, 10)
    core_arm, verifies = {}, {}
    reps_c = min(4000, max(1000, args.reps // 4))
    bufs_c = {c: np.empty(reps_c) for c in sweep_cpus}
    for c in sweep_cpus:
        v = pin_and_verify(c, n_probes=400)
        v.pop("_sink", None)
        verifies[c] = v
        for _ in range(1000):
            fn_k10()
    pc = time.perf_counter_ns
    done, chunk = 0, 200
    while done < reps_c:
        n_c = min(chunk, reps_c - done)
        for c in sweep_cpus:
            set_affinity([c])
            b = bufs_c[c]
            for i in range(done, done + n_c):
                t0 = pc()
                fn_k10()
                b[i] = pc() - t0
        done += n_c
    for c in sweep_cpus:
        raw["corearm_cpu%d_K10_single" % c] = bufs_c[c]
        core_arm["cpu%d" % c] = {
            "label": "P-core" if c <= 11 else "E-core",
            "verify": verifies[c], "stats": summarize(bufs_c[c])}
        print("    %s cpu%-2d: median %7.2f us  min %7.2f us  p99 %8.2f"
              % (core_arm["cpu%d" % c]["label"], c,
                 core_arm["cpu%d" % c]["stats"]["median"],
                 core_arm["cpu%d" % c]["stats"]["min"],
                 core_arm["cpu%d" % c]["stats"]["p99"]))
    pin_and_verify(args.cpu)
    p_mins = [core_arm["cpu%d" % c]["stats"]["min"] for c in sweep_cpus if c <= 11]
    e_mins = [core_arm["cpu%d" % c]["stats"]["min"] for c in sweep_cpus if c > 11]
    rec["core_class_arm"] = {
        "what": ("The identical single-candidate K=10 evaluation, pinned and verified on "
                 "each P-core primary SMT thread and on two E-cores, timed round-robin "
                 "so no core is measured only in a quiet or only in a noisy window. "
                 "This is why an unpinned latency distribution on this part is not "
                 "interpretable: it is a mixture of these two processors."),
        "cpus_swept": sweep_cpus,
        "per_cpu": core_arm,
        "P_core_min_us": {"values": p_mins, "min": min(p_mins), "max": max(p_mins),
                          "spread_pct": 100.0 * (max(p_mins) - min(p_mins)) / min(p_mins)},
    }
    if e_mins:
        rec["core_class_arm"]["E_core_min_us"] = {"values": e_mins, "min": min(e_mins),
                                                  "max": max(e_mins)}
        rec["core_class_arm"]["E_over_P_ratio_on_group_minima"] = min(e_mins) / min(p_mins)
        print("    P-core minima %.2f-%.2f us | E-core minima %.2f-%.2f us | E/P = %.3f"
              % (min(p_mins), max(p_mins), min(e_mins), max(e_mins),
                 rec["core_class_arm"]["E_over_P_ratio_on_group_minima"]))
    else:
        rec["core_class_arm"]["E_core_min_us"] = None
        rec["core_class_arm"]["E_over_P_ratio_on_group_minima"] = None
        rec["core_class_arm"]["no_E_cores_note"] = (
            "This host exposes no logical processors above 11 (the E-core half of the "
            "i5-14600KF topology): it reports %d uniform vCPUs, so no E-cores exist to "
            "measure.  The E/P ratio and the 'unpinned distribution is a mixture of two "
            "processors' argument are therefore NOT APPLICABLE on this host; the P-core "
            "column above is the per-vCPU cost on a homogeneous set." % os.cpu_count())
        print("    P-core minima %.2f-%.2f us | NO E-cores on this host "
              "(uniform vCPUs) - E/P ratio not applicable"
              % (min(p_mins), max(p_mins)))

    # ---------------------------------------------------------------- 3. SMT ARM
    if not args.skip_smt_arm and IS_WINDOWS and psutil is not None:
        print("[i] SMT arm: sibling logical processor idle vs loaded ...")
        sibling = args.cpu ^ 1          # the other SMT thread of the same physical core
        fn = lambda: pe_series_f64(A_s, B_s, xi1, a01, gbar, 10)
        smt = {}
        reps_smt = min(20000, max(6000, args.reps // 2))
        ns_idle, _ = time_calls(fn, reps_smt, warmup=2000)
        raw["smtarm_sibling_idle"] = ns_idle
        smt["sibling_ambient"] = summarize(ns_idle)
        # The child must actually BE spinning on the sibling before the second arm is
        # timed; an unsynchronised sleep is not proof of that.  It announces itself on
        # stdout after pinning, and its CPU consumption is then sampled and recorded, so
        # the arm carries evidence that the load was applied rather than an assumption.
        code = ("import psutil, sys, time\n"
                "psutil.Process().cpu_affinity([%d])\n"
                "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
                "x = 0.0\nt = time.time()\n"
                "while time.time() - t < 90.0: x += 1.0000001\n" % sibling)
        proc = subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE, text=True)
        try:
            ready = proc.stdout.readline().strip()
            child = psutil.Process(proc.pid)
            c0 = sum(child.cpu_times()[:2])
            w0 = time.perf_counter()
            time.sleep(0.5)
            ns_busy, _ = time_calls(fn, reps_smt, warmup=2000)
            c1 = sum(child.cpu_times()[:2])
            w1 = time.perf_counter()
            raw["smtarm_sibling_loaded"] = ns_busy
            smt["sibling_loaded"] = summarize(ns_busy)
            smt["loader_evidence"] = {
                "child_handshake": ready,
                "child_affinity": list(child.cpu_affinity()),
                "child_cpu_seconds_during_arm": c1 - c0,
                "wall_seconds_during_arm": w1 - w0,
                "child_cpu_utilisation": (c1 - c0) / (w1 - w0) if w1 > w0 else None,
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        time.sleep(1.0)
        rec["smt_sibling_arm"] = {
            "what": ("Pinning to logical processor %d does not stop Windows scheduling "
                     "other work on logical processor %d, the other SMT thread of the "
                     "same physical P-core.  Here a spinner subprocess is pinned to "
                     "that sibling and the same kernel is re-timed.  This is the "
                     "mechanism behind the bimodality in the main distributions above: "
                     "the fast mode is an uncontended physical core, the slow mode is a "
                     "shared one." % (args.cpu, sibling)),
            "timing_cpu": args.cpu, "loaded_sibling_cpu": sibling,
            "arms": smt,
            "loaded_over_ambient_ratio_min": (smt["sibling_loaded"]["min"] /
                                              smt["sibling_ambient"]["min"]),
            "loaded_over_ambient_ratio_median": (smt["sibling_loaded"]["median"] /
                                                 smt["sibling_ambient"]["median"]),
            "caveat": ("The 'ambient' arm is not a guaranteed-idle sibling -- nothing on "
                       "a general-purpose desktop can guarantee that.  It is the sibling "
                       "under whatever background load the machine had, which is exactly "
                       "why its own distribution is bimodal."),
        }
        print("    ambient min %.2f us -> loaded min %.2f us (x%.2f); loader used "
              "%.2f CPU-s over %.2f wall-s"
              % (smt["sibling_ambient"]["min"], smt["sibling_loaded"]["min"],
                 rec["smt_sibling_arm"]["loaded_over_ambient_ratio_min"],
                 smt["loader_evidence"]["child_cpu_seconds_during_arm"],
                 smt["loader_evidence"]["wall_seconds_during_arm"]))

    # ---------------------------------------------------------------- 4. PRIORITY
    print("[i] priority-class arm (WINDOWS ANALOGUE, not the Linux arms) ...")
    prio_arm = {}
    fn = lambda: pe_series_f64(A_s, B_s, xi1, a01, gbar, 10)
    PRIOS = ("NORMAL", "HIGH", "REALTIME")
    reps_p = min(8000, max(2000, args.reps // 2))
    bufs_p = {nm: np.empty(reps_p) for nm in PRIOS}
    states = {}
    for _ in range(2000):
        fn()
    pc = time.perf_counter_ns
    done, chunk = 0, 200
    while done < reps_p:                      # round-robin over priority classes
        n_c = min(chunk, reps_p - done)
        for nm in PRIOS:
            states[nm] = set_priority(nm)
            b = bufs_p[nm]
            for i in range(done, done + n_c):
                t0 = pc()
                fn()
                b[i] = pc() - t0
        done += n_c
    set_priority("NORMAL")
    for nm in PRIOS:
        raw["prioarm_%s" % nm] = bufs_p[nm]
        prio_arm[nm] = {"priority": states[nm], "stats": summarize(bufs_p[nm])}
        print("    %-9s median %7.2f  min %6.2f  p99 %8.2f  max %9.2f  "
              "frac_contended %.3f"
              % (nm, prio_arm[nm]["stats"]["median"], prio_arm[nm]["stats"]["min"],
                 prio_arm[nm]["stats"]["p99"], prio_arm[nm]["stats"]["max"],
                 prio_arm[nm]["stats"]["fraction_above_1p5x_min"]))
    rec["os_tuning_analogue"] = {
        "IS_NOT_A_REPRODUCTION": True,
        "what_the_manuscript_reports": ["SCHED_OTHER", "chrt", "isolcpus"],
        "why_not_reproducible": ("SCHED_OTHER, chrt and isolcpus are Linux scheduler "
                                 "mechanisms.  They do not exist on Windows.  Nothing "
                                 "in this file reproduces them and nothing here should "
                                 "be presented as reproducing them."),
        "mechanism_actually_used": ("Windows priority classes -- "
                                    "NORMAL_PRIORITY_CLASS / HIGH_PRIORITY_CLASS / "
                                    "REALTIME_PRIORITY_CLASS, set via "
                                    "psutil.Process().nice() which calls "
                                    "SetPriorityClass -- combined with the fixed, "
                                    "verified processor affinity above.  Windows is a "
                                    "general-purpose scheduler and offers no real-time "
                                    "guarantee under any of these classes."),
        "arms": prio_arm,
        "all_other_measurements_in_this_file_ran_at": "NORMAL_PRIORITY_CLASS",
    }

    # ---------------------------------------------------------------- 5. B2 SEARCH
    print("[i] searching the fixed-order quadrature family for the cheapest rule that "
          "reaches each accuracy target ...")
    GRID = ([("GJ", nt, nx) for nt in (12, 16, 24, 32, 48, 64, 96) for nx in (100, 200, 400)] +
            [("GL", nt, nx) for nt in (64, 128, 256, 512, 1024) for nx in (100, 200, 400)])
    err_grid = {}
    for rule, nt, nx in GRID:
        worst = 0.0
        for reg in REGIMES:
            q = TensorQuad(reg, rule, nt, nx)
            for xs in XI_CHECK:
                v = q(pts[xs]["xi"], pts[xs]["A0"], gbar)
                worst = max(worst, abs(v - refs[(reg, xs)]))
        err_grid[(rule, nt, nx)] = worst

    def stable_ok(rule, nt, nx, target):
        """Meets the target here and at every larger nt at the same nx -- fixed-order
        rules can meet a target accidentally at low order."""
        cand = sorted([k for k in err_grid if k[0] == rule and k[2] == nx],
                      key=lambda k: k[1])
        hit = False
        for k in cand:
            if k[1] < nt:
                continue
            if err_grid[k] > target:
                return False
            if k[1] == nt:
                hit = True
        return hit

    # Time every rule that qualifies for the LOOSEST target once, and reuse; a rule's
    # cost does not depend on which target it is being considered for.
    all_qual = sorted({k for t in TARGETS for k in err_grid
                       if stable_ok(k[0], k[1], k[2], t)})
    cost_single, cost_swarm = {}, {}
    for rule, nt, nx in all_qual:
        q = TensorQuad("strong", rule, nt, nx)
        f = lambda q=q: q(pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar)
        ns, _ = time_calls(f, 120, warmup=30)
        # Screen on the MINIMUM, not the median.  Roughly half of all samples on this
        # machine land in an SMT-contended mode that costs about twice as much, which
        # makes a short block's median a coin flip and would let the search pick a rule
        # that merely happened to be measured in a quiet window.  Interference can only
        # inflate a cost, so the minimum ranks the rules by their actual work.
        cost_single[(rule, nt, nx)] = float(np.min(ns))
        fs = lambda q=q: q.swarm(xi_sw, a0_sw, gbar)
        # Give every rule the same screening WALL TIME (~0.4 s) rather than the same
        # repetition count, so a cheap rule is not screened over a window too short to
        # contain a quiet stretch while an expensive one is screened over many.
        reps_sw = int(np.clip(4.0e8 / max(1.0, cost_single[(rule, nt, nx)] * N_SWARM),
                              12, 120))
        nss, _ = time_calls(fs, reps_sw, warmup=6)
        cost_swarm[(rule, nt, nx)] = float(np.min(nss)) / N_SWARM

    chosen, chosen_swarm = {}, {}
    for target in TARGETS:
        qual = [k for k in err_grid if stable_ok(k[0], k[1], k[2], target)]
        if not qual:
            chosen[target] = chosen_swarm[target] = None
            print("    target %.0e: NOT REACHED anywhere in the searched family" % target)
            continue
        chosen[target] = min(qual, key=lambda k: cost_single[k])
        chosen_swarm[target] = min(qual, key=lambda k: cost_swarm[k])
        print("    target %.0e -> scalar-cheapest %s nt=%d nx=%d (err %.3e); "
              "swarm-cheapest %s nt=%d nx=%d"
              % ((target,) + chosen[target] + (err_grid[chosen[target]],)
                 + chosen_swarm[target]))

    rec["b2_rule_search"] = {
        "what": ("The 'optimized double-precision quadrature' baseline is not asserted, "
                 "it is searched for.  Every rule in the family below was evaluated for "
                 "accuracy against the 90-digit reference at three xi in three regimes, "
                 "then every rule meeting a target was timed, and the CHEAPEST was "
                 "selected.  Choosing the fastest fair baseline makes the reported "
                 "speedup the smallest defensible one; a slower baseline would inflate "
                 "it."),
        "family": ("outer rule in {Gauss-Legendre in t with explicit weight "
                   "xi^2 t^(xi^2-1), Gauss-Jacobi with (0, xi^2-1)} x outer order nt x "
                   "inner Gauss-Legendre order nx in log x over [-30, 10]"),
        "accuracy_grid": {"%s_nt%d_nx%d" % k: v for k, v in err_grid.items()},
        "stability_rule": ("a rule qualifies for a target only if it and every larger nt "
                           "at the same nx meet it, so no rule qualifies by an accidental "
                           "sign cancellation at low order"),
        "screening_cost_us_minimum": {"%s_nt%d_nx%d" % k: {"single": cost_single[k] / 1e3,
                                                           "per_candidate": cost_swarm[k] / 1e3}
                                      for k in cost_single},
        "screening_statistic": ("minimum over the screening block, because about half of "
                                "all samples on this machine land in an SMT-contended "
                                "mode costing ~1.8x and a short block's median is "
                                "therefore unstable; the minimum ranks rules by work "
                                "done rather than by when they were measured"),
        "selected": {("%.0e" % t): (None if chosen[t] is None else
                                    {"scalar_cheapest":
                                        {"rule": chosen[t][0], "nt": chosen[t][1],
                                         "nx": chosen[t][2],
                                         "worst_abs_err": err_grid[chosen[t]]},
                                     "swarm_cheapest":
                                        {"rule": chosen_swarm[t][0],
                                         "nt": chosen_swarm[t][1],
                                         "nx": chosen_swarm[t][2],
                                         "worst_abs_err": err_grid[chosen_swarm[t]]}})
                     for t in TARGETS},
        "two_selections": ("The rule that is cheapest for one candidate is not "
                           "necessarily the rule that is cheapest amortised over a "
                           "swarm: the Gauss-Jacobi outer nodes depend on the "
                           "candidate's own xi and so cannot be shared across a swarm, "
                           "while the Gauss-Legendre outer nodes can.  Both selections "
                           "are made and both are timed, so neither comparison is "
                           "handicapped by a rule chosen for the other."),
        "generous_to_the_baseline": ("All (alpha, beta)-dependent work -- the "
                                     "scipy.special.kv Bessel evaluations, the "
                                     "gamma-gamma density and the inner weights -- is "
                                     "precomputed and NOT charged to the per-evaluation "
                                     "cost, because the turbulence regime is a per-cycle "
                                     "channel estimate rather than a per-candidate "
                                     "quantity.  Only xi- and A_0-dependent work is "
                                     "timed.  For the Gauss-Jacobi rule the node "
                                     "computation IS charged, because its nodes depend "
                                     "on the candidate's own xi."),
    }

    # ---------------------------------------------------------------- 6. B2 TIMING
    print("[i] timing baseline B2 (optimized fixed-order quadrature) ...")
    b2 = {}
    for target in TARGETS:
        if chosen[target] is None:
            continue
        rule, nt, nx = chosen[target]
        rs, nts, nxs = chosen_swarm[target]
        for reg in REGIMES:
            q = TensorQuad(reg, rule, nt, nx)
            key = "b2_%s_%s_nt%d_nx%d_single" % (reg, rule, nt, nx)
            f = lambda q=q: q(pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar)
            ns, _ = time_calls(f, args.reps_tensor, warmup=200)
            raw[key] = ns
            v = q(pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar)
            r = refs[(reg, XI_REF)]
            b2["%.0e|%s|single" % (target, reg)] = {
                "rule": rule, "nt": nt, "nx": nx, "target_abs_err": target,
                "worst_abs_err_over_check_points": err_grid[(rule, nt, nx)],
                "value_at_check_point": v, "reference": r,
                "abs_err_at_check_point": abs(v - r),
                "stats": summarize(ns), "raw_key": key,
            }
            qs = TensorQuad(reg, rs, nts, nxs)
            keys = "b2_%s_%s_nt%d_nx%d_swarm%d" % (reg, rs, nts, nxs, N_SWARM)
            fs = lambda q=qs: q.swarm(xi_sw, a0_sw, gbar)
            nss, _ = time_calls(fs, max(200, args.reps_tensor // 5), warmup=20)
            raw[keys] = nss
            b2["%.0e|%s|swarm" % (target, reg)] = {
                "rule": rs, "nt": nts, "nx": nxs,
                "worst_abs_err_over_check_points": err_grid[(rs, nts, nxs)],
                "stats_total": summarize(nss),
                "stats_per_candidate": summarize(nss, per=N_SWARM),
                "raw_key": keys,
            }
        s = b2["%.0e|strong|single" % target]["stats"]
        p = b2["%.0e|strong|swarm" % target]["stats_per_candidate"]
        print("    target %.0e strong: single median %8.2f us | swarm %8.3f us/candidate"
              % (target, s["median"], p["median"]))
    rec["baseline_B2_optimized_double_precision_quadrature"] = b2

    # ---------------------------------------------------------------- 7. B1 TIMING
    print("[i] timing baseline B1 (nested scipy.integrate.quad, adaptive "
          "Gauss-Kronrod + scipy.special.kv) ...")
    b1 = {}
    # Same tolerance ladder as B2, so B1 is not handicapped by being run tighter than
    # the comparison needs.
    for tol in TARGETS:
        for reg in REGIMES:
            f = quad_aber_factory(reg, pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar, tol)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = f()
                ns, _ = time_calls(f, args.reps_quad, warmup=3)
            key = "b1_%s_tol%.0e_single" % (reg, tol)
            raw[key] = ns
            r = refs[(reg, XI_REF)]
            b1["%.0e|%s" % (tol, reg)] = {
                "tolerance_epsabs_epsrel_outer": tol,
                "tolerance_inner": tol * 0.1,
                "value": v, "reference": r, "abs_err": abs(v - r),
                "rel_err": abs(v - r) / abs(r) if r else None,
                "stats": summarize(ns), "raw_key": key,
            }
        e = b1["%.0e|strong" % tol]
        print("    tol %.0e strong: median %9.1f us   abs err reached %.3e"
              % (tol, e["stats"]["median"], e["abs_err"]))
    rec["baseline_B1_adaptive_gauss_kronrod"] = {
        "what": ("Nested scipy.integrate.quad over the pointing variable t and, inside "
                 "it, over log x for the gamma-gamma density, with scipy.special.kv in "
                 "double precision.  This is the baseline the manuscript describes in "
                 "prose ('adaptive Gauss-Kronrod quadrature of the exact ABER integral "
                 "with double-precision modified Bessel functions')."),
        "why_it_is_not_the_headline_baseline": ("It is far slower than the fixed-order "
                                                "rule B2 at the same accuracy, mostly "
                                                "because every integrand evaluation is a "
                                                "Python callback.  Using it for the "
                                                "headline would inflate the speedup, so "
                                                "B2 is used instead and B1 is reported "
                                                "for comparison with the published "
                                                "figure."),
        "tolerance_ladder": ("run at the same three absolute-error targets as B2, so it "
                             "is not handicapped by being asked for more accuracy than "
                             "the comparison needs; the accuracy each tolerance actually "
                             "reaches against the 90-digit reference is recorded"),
        "per_tolerance_and_regime": b1,
    }

    # ---------------------------------------------------------------- 8. MEIJER-G
    print("[i] timing the arbitrary-precision Meijer-G reference ...")
    b0 = {}
    for reg in REGIMES:
        f = meijerg_aber_factory(reg, pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar, dps=15)
        v = float(f())
        mp.mp.dps = 90
        ns, _ = time_calls(f, args.reps_meijer, warmup=2)
        mp.mp.dps = 90
        raw["b0_meijerg_%s" % reg] = ns
        r = refs[(reg, XI_REF)]
        b0[reg] = {"dps": 15, "value": v, "reference": r,
                   "abs_err": abs(v - r), "rel_err": abs(v - r) / abs(r) if r else None,
                   "stats": summarize(ns), "raw_key": "b0_meijerg_%s" % reg}
        print("    %-9s median %9.1f us (%.2f ms)  rel err %.2e"
              % (reg, b0[reg]["stats"]["median"], b0[reg]["stats"]["median"] / 1e3,
                 b0[reg]["rel_err"]))
    rec["baseline_B0_arbitrary_precision_meijerg"] = {
        "what": ("mpmath integration at 15 significant digits of "
                 "Q(sqrt(gbar) h) f_h(h), with f_h the Farid-Hranilovic composite "
                 "density written as G^{3,0}_{1,3}."),
        "tail_bound": ("integration cut at h = 15/sqrt(gbar), beyond which "
                       "Q(sqrt(gbar) h) < Q(15) = 3.7e-51 and the density integrates to "
                       "at most 1, so the discarded tail is below 4e-51"),
        "per_regime": b0,
    }

    # ---------------------------------------------------------------- 9. SPEEDUPS
    tgt = HEADLINE_TARGET
    # The speedup is a ratio of two medians, so the two endpoints must see the same
    # ambient load or the ratio is partly a measurement of when each was taken.  Both
    # endpoints of every headline ratio are therefore re-timed HERE, round-robin against
    # each other, rather than lifted from blocks measured minutes apart.
    print("[i] headline block A: series K=10 vs B2, interleaved ...")
    hl = {}
    fnsA = {}
    for reg in REGIMES:
        A_r, B_r = float(REGIMES[reg][0]), float(REGIMES[reg][1])
        fnsA["hlA_series_single_%s" % reg] = (
            lambda A=A_r, B=B_r: pe_series_f64(A, B, xi1, a01, gbar, 10))
        fnsA["hlA_series_swarm_%s" % reg] = (
            lambda A=A_r, B=B_r: pe_series_f64(A, B, xi_sw, a0_sw, gbar, 10))
        if chosen[tgt] is not None:
            q = TensorQuad(reg, *chosen[tgt])
            qs = TensorQuad(reg, *chosen_swarm[tgt])
            fnsA["hlA_B2_single_%s" % reg] = (
                lambda q=q: q(pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar))
            fnsA["hlA_B2_swarm_%s" % reg] = (
                lambda q=qs: q.swarm(xi_sw, a0_sw, gbar))
    bA, _ = time_calls_interleaved(fnsA, args.reps_headline, chunk=20, warmup=200)
    for k, v in bA.items():
        raw[k] = v
        hl[k] = summarize(v, per=(N_SWARM if "swarm" in k else 1))

    print("[i] headline block B: series K=10 vs B1 and Meijer-G, interleaved ...")
    # B1 and B0 cost 10-100 ms per call, the series ~30 us.  Interleaving them one call
    # at a time would time the series with every cache line it needs evicted by 100 ms of
    # unrelated mpmath work, which is not the condition it runs in: the optimizer calls
    # this kernel T_iter x N_p = 750 times back to back inside one cycle.  So the series
    # endpoint here is a BURST of BURST_N back-to-back calls divided by BURST_N -- warm,
    # as deployed -- while B1 and B0 are timed one call per sample.
    BURST_N = 200
    fnsB, perB = {}, {}
    for reg in REGIMES:
        A_r, B_r = float(REGIMES[reg][0]), float(REGIMES[reg][1])

        def burst(A=A_r, B=B_r):
            for _ in range(BURST_N):
                pe_series_f64(A, B, xi1, a01, gbar, 10)

        fnsB["hlB_seriesburst%d_%s" % (BURST_N, reg)] = burst
        perB["hlB_seriesburst%d_%s" % (BURST_N, reg)] = BURST_N
        fnsB["hlB_B1_%s" % reg] = quad_aber_factory(
            reg, pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar, tgt)
        perB["hlB_B1_%s" % reg] = 1
        fnsB["hlB_B0_%s" % reg] = meijerg_aber_factory(
            reg, pts[XI_REF]["xi"], pts[XI_REF]["A0"], gbar, dps=15)
        perB["hlB_B0_%s" % reg] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bB, _ = time_calls_interleaved(fnsB, args.reps_slow, chunk=1, warmup=2)
    mp.mp.dps = 90
    for k, v in bB.items():
        raw[k] = v
        hl[k] = summarize(v, per=perB[k])
        hl[k]["per"] = perB[k]
    rec["headline_interleaved_block"] = {
        "what": ("Both endpoints of every headline speedup, re-timed round-robin against "
                 "each other so the ratio is not partly a record of which block happened "
                 "to run during a quiet window.  Block A holds the series and the "
                 "fixed-order quadrature B2; block B holds the series, the adaptive "
                 "Gauss-Kronrod baseline B1 and the arbitrary-precision Meijer-G "
                 "reference B0, at fewer repetitions because each of those calls costs "
                 "tens of milliseconds."),
        "block_B_series_is_a_burst": ("In block B the series endpoint is a burst of %d "
                                      "back-to-back calls divided by %d, because "
                                      "interleaving a 30 us call one-at-a-time against "
                                      "100 ms mpmath calls would time it with a cold "
                                      "cache every sample.  That is not the condition "
                                      "the kernel runs in: the optimizer issues "
                                      "T_iter x N_p = 750 evaluations back to back "
                                      "inside a cycle." % (BURST_N, BURST_N)),
        "stats": hl,
    }

    print("[i] speedups (ratios of measured medians, from the interleaved block) ...")
    sp = {}
    for reg in REGIMES:
        ks = hl["hlA_series_single_%s" % reg]
        kw = hl["hlA_series_swarm_%s" % reg]
        entry = {
            "series_K10_single_median_us": ks["median"],
            "series_K10_single_min_us": ks["min"],
            "series_K10_per_candidate_median_us": kw["median"],
            "series_K10_per_candidate_min_us": kw["min"],
        }
        if chosen[tgt] is not None:
            b2s = hl["hlA_B2_single_%s" % reg]
            b2p = hl["hlA_B2_swarm_%s" % reg]
            entry["B2_single_median_us"] = b2s["median"]
            entry["B2_single_min_us"] = b2s["min"]
            entry["B2_per_candidate_median_us"] = b2p["median"]
            entry["B2_per_candidate_min_us"] = b2p["min"]
            entry["speedup_vs_B2_scalar_medians"] = b2s["median"] / ks["median"]
            entry["speedup_vs_B2_scalar_minima"] = b2s["min"] / ks["min"]
            entry["speedup_vs_B2_per_candidate_medians"] = b2p["median"] / kw["median"]
            entry["speedup_vs_B2_per_candidate_minima"] = b2p["min"] / kw["min"]
            # Both endpoints taken in the SAME, well-defined machine state: one P-core
            # with its SMT sibling idle.  The ambient median of either endpoint is a
            # knife-edge statistic of a bimodal distribution and moves between runs; the
            # mode-resolved figure does not.
            if b2s["median_fast_mode"] and ks["median_fast_mode"]:
                entry["speedup_vs_B2_scalar_fast_mode"] = (b2s["median_fast_mode"] /
                                                           ks["median_fast_mode"])
            if b2p["median_fast_mode"] and kw["median_fast_mode"]:
                entry["speedup_vs_B2_per_candidate_fast_mode"] = (b2p["median_fast_mode"] /
                                                                  kw["median_fast_mode"])
        ksB = hl["hlB_seriesburst%d_%s" % (BURST_N, reg)]
        b1e = hl["hlB_B1_%s" % reg]
        entry["B1_median_us"] = b1e["median"]
        entry["speedup_vs_B1_scalar_medians"] = b1e["median"] / ksB["median"]
        b0e = hl["hlB_B0_%s" % reg]
        entry["B0_meijerg_median_us"] = b0e["median"]
        entry["speedup_vs_B0_scalar_medians"] = b0e["median"] / ksB["median"]
        entry["speedup_vs_B0_per_candidate_medians"] = b0e["median"] / kw["median"]
        sp[reg] = entry
        print("    %-9s vs B2 %.1fx scalar / %.1fx per-candidate | vs B1 %.0fx | "
              "vs Meijer-G %.0fx"
              % (reg, entry.get("speedup_vs_B2_scalar_medians", float("nan")),
                 entry.get("speedup_vs_B2_per_candidate_medians", float("nan")),
                 entry["speedup_vs_B1_scalar_medians"],
                 entry["speedup_vs_B0_scalar_medians"]))

    rec["speedups"] = {
        "definition": ("ratio of two measured medians, both taken on this machine, "
                       "pinned to the same verified logical processor, in the same "
                       "interpreter, with BLAS capped to one thread.  Both endpoints of "
                       "every ratio are printed in this file."),
        "accuracy_matched_at_abs_err": tgt,
        "why_that_target": ("1e-9 is the scale of the Table 7 certified worst-case "
                            "per-branch bound at K=10 (3.98e-9 / 5.49e-10 / 7.87e-10). "
                            "Requiring the quadrature to match the series' MEASURED "
                            "error, which is at the float64 round-off floor, would be a "
                            "target no quadrature can reach and would not be a fair "
                            "comparison."),
        "per_regime": sp,
        "like_for_like_note": ("'scalar' compares one single-candidate series evaluation "
                              "against one quadrature evaluation.  'per-candidate' "
                              "compares both methods vectorised across the same swarm of "
                              "N_p = %d, which is the honest comparison for the "
                              "manuscript's SIMD claim: the published 2.5e3 figure "
                              "divided a SIMD-amortised series cost by a scalar "
                              "quadrature cost." % N_SWARM),
        "which_ratio_to_quote": ("The scalar ratio UNDERSTATES the algorithmic gap on "
                                 "this stack: a single-candidate series evaluation costs "
                                 "only ~100 float64 operations and its measured cost is "
                                 "almost entirely CPython/NumPy dispatch overhead, while "
                                 "the quadrature's cost is real arithmetic.  The "
                                 "per-candidate ratio, where that fixed overhead is "
                                 "amortised over N_p = %d candidates on both sides, is "
                                 "the one that reflects the algorithm rather than the "
                                 "interpreter.  Both are reported and neither is "
                                 "presented alone." % N_SWARM),
        "medians_vs_minima": ("Ratios are given on medians (what the machine delivers "
                              "under its ambient load) and on minima (the "
                              "interference-free floor).  Interference can only inflate "
                              "a measured cost, so the two bracket the truth."),
    }

    # ---------------------------------------------------------------- 10. VS PUBLISHED
    published = {"K5_us": 14.2, "K10_us": 23.6, "K20_us": 45.9,
                 "meijerg_us": 45000.0, "per_particle_us": 0.6,
                 "speedup_vs_double_precision_quadrature": 64.0,
                 "speedup_vs_meijerg": 1900.0}
    comp = {}
    for K in ORDERS:
        st = rec["kernel"]["kernel_strong_K%d_single" % K]
        p = published["K%d_us" % K]
        comp["K%d_single" % K] = {
            "published_us": p, "measured_median_us": st["median"],
            "measured_min_us": st["min"], "measured_p99_us": st["p99"],
            "measured_over_published_on_median": st["median"] / p,
            "measured_over_published_on_min": st["min"] / p,
            "verdict": "SLOWER here" if st["median"] > p else "faster here"}
    swst = rec["kernel"]["kernel_strong_K10_swarm%d" % N_SWARM]
    comp["per_particle_K10"] = {
        "published_us": published["per_particle_us"],
        "measured_median_us": swst["median"], "measured_min_us": swst["min"],
        "measured_over_published_on_median": swst["median"] / published["per_particle_us"],
        "measured_over_published_on_min": swst["min"] / published["per_particle_us"],
        "verdict": "SLOWER here" if swst["median"] > published["per_particle_us"]
                   else "faster here"}
    comp["meijerg"] = {"published_us": published["meijerg_us"],
                       "measured_median_us": b0["strong"]["stats"]["median"],
                       "measured_over_published":
                           b0["strong"]["stats"]["median"] / published["meijerg_us"],
                       "verdict": "SLOWER here"
                                  if b0["strong"]["stats"]["median"] > published["meijerg_us"]
                                  else "faster here"}
    if chosen[tgt] is not None:
        comp["speedup_vs_double_precision_quadrature"] = {
            "published": 64.0,
            "measured_scalar_medians": sp["strong"]["speedup_vs_B2_scalar_medians"],
            "measured_scalar_minima": sp["strong"]["speedup_vs_B2_scalar_minima"],
            "measured_per_candidate_medians":
                sp["strong"]["speedup_vs_B2_per_candidate_medians"],
            "measured_vs_B1_the_prose_baseline":
                sp["strong"]["speedup_vs_B1_scalar_medians"],
            "note": ("the published 64x came from 1.5 ms / 23.6 us.  Neither endpoint "
                     "reproduces here: the fastest fair double-precision quadrature we "
                     "could build at matched accuracy is much faster than 1.5 ms, and "
                     "the series is slower than 23.6 us on this stack."),
        }
    comp["speedup_vs_meijerg"] = {
        "published": 1900.0,
        "measured_scalar_medians": sp["strong"]["speedup_vs_B0_scalar_medians"],
        "measured_per_candidate_medians":
            sp["strong"]["speedup_vs_B0_per_candidate_medians"]}
    rec["comparison_with_published"] = {
        "published_values": published,
        "published_platform": ("'Platform A (Intel i7)', not reproducible from the "
                               "release; this file reports one machine, an "
                               "i5-14600KF under Windows 11 and CPython 3.14"),
        "regime_used_for_comparison": "strong (alpha=1.2, beta=1.1), the campaign's "
                                      "operating point",
        "per_claim": comp,
    }

    # ---------------------------------------------------------------- 11. HEADLINE
    # The ambient median of a bimodal distribution is a knife-edge statistic: with the
    # contended fraction near one half it flips between the two modes from run to run.
    # The mode-resolved figures do not, so those are what the table should carry, with
    # the contended mode and the contended fraction printed beside them.
    hn = {"regime": "strong", "cpu": args.cpu,
          "state": "one P-core logical processor, SMT sibling idle (fast mode)"}
    for K in ORDERS:
        st = rec["kernel"]["kernel_strong_K%d_single" % K]
        sw = rec["kernel"]["kernel_strong_K%d_swarm%d" % (K, N_SWARM)]
        hn["K%d" % K] = {
            "single_candidate_us": {
                "uncontended_median": st["median_fast_mode"],
                "smt_contended_median": st["median_slow_mode"],
                "fraction_contended": st["fraction_above_1p5x_min"],
                "min": st["min"], "ambient_median": st["median"],
                "p95": st["p95"], "p99": st["p99"], "p999": st["p999"], "max": st["max"]},
            "per_candidate_swarm30_us": {
                "uncontended_median": sw["median_fast_mode"],
                "smt_contended_median": sw["median_slow_mode"],
                "min": sw["min"], "ambient_median": sw["median"],
                "p95": sw["p95"], "p99": sw["p99"], "p999": sw["p999"], "max": sw["max"]},
            "published_single_us": published["K%d_us" % K],
        }
    rec["headline_numbers"] = hn

    # How the cost scales in K.  The published column rises 14.2 -> 23.6 -> 45.9, i.e.
    # roughly in proportion to the (K+1) terms of the truncation (6 : 11 : 21).  Whether
    # the measured cost does the same is a separate question from whether its magnitude
    # matches, and it is checked here rather than assumed.
    def _ratio(vals):
        return [v / vals[0] for v in vals]

    sing = [rec["kernel"]["kernel_strong_K%d_single" % K]["median_fast_mode"]
            for K in ORDERS]
    swm = [rec["kernel"]["kernel_strong_K%d_swarm%d" % (K, N_SWARM)]["median_fast_mode"]
           for K in ORDERS]
    rec["K_scaling"] = {
        "orders": list(ORDERS),
        "terms_in_truncation_K_plus_1": [K + 1 for K in ORDERS],
        "expected_ratio_if_cost_is_proportional_to_K_plus_1":
            _ratio([K + 1 for K in ORDERS]),
        "published_us": [published["K%d_us" % K] for K in ORDERS],
        "published_ratio": _ratio([published["K%d_us" % K] for K in ORDERS]),
        "measured_single_candidate_us": sing,
        "measured_single_candidate_ratio": _ratio(sing),
        "measured_per_candidate_swarm_us": swm,
        "measured_per_candidate_swarm_ratio": _ratio(swm),
        "finding": ("Read the ratios, not only the magnitudes.  A single-candidate "
                    "evaluation on this stack costs about the same at K=5, 10 and 20 "
                    "because its cost is dominated by fixed CPython/NumPy dispatch, not "
                    "by the O(K) arithmetic; the truncation order only starts to show in "
                    "the swarm-vectorised figure, and even there it grows far more slowly "
                    "than (K+1).  The published column's near-proportionality to (K+1) "
                    "therefore does not reproduce here as a shape, independently of "
                    "whether any single entry reproduces as a number."),
    }

    rec["not_measured_here"] = {
        "platform_B_jetson_agx_xavier": ("not present; no measurement of it exists in "
                                         "this file and none should be inferred"),
        "platform_C_cortex_a72": "not present; same",
        "linux_scheduling_arms": ("SCHED_OTHER, chrt and isolcpus are Linux mechanisms "
                                  "and cannot run on Windows.  The priority-class arm in "
                                  "this file is an analogue and is labelled as one."),
        "end_to_end_cycle_latency": ("this file measures the ABER kernel and its "
                                     "baselines only.  It says nothing about the five "
                                     "pipeline stages, tau_O = 600 us, T_u = 1 ms, "
                                     "tau_A, the 800 us computation budget, the "
                                     "0.77-0.79 ms median cycle or the 78.0% joint "
                                     "real-time success rate.  Those are separate "
                                     "claims and are not evidenced here."),
        "guaranteed_quiet_machine": ("this is a general-purpose Windows desktop under "
                                     "ambient background load.  About half of all "
                                     "samples land in an SMT-contended mode.  The "
                                     "uncontended figure is reported as a mode of the "
                                     "measured distribution, not as a claim that the "
                                     "machine is ever guaranteed to be in it."),
        "a_faster_quadrature_may_exist": ("B2 is the cheapest rule in the family "
                                          "searched here.  A better rule outside that "
                                          "family would lower the reported speedup.  The "
                                          "family and the search are both recorded so "
                                          "the claim can be attacked."),
    }

    rule_tag = ("Gauss--Jacobi $(0,\\xi^2{-}1)\\times$Gauss--Legendre, "
                "$n_t{=}%d$, $n_x{=}%d$" % (chosen[tgt][1], chosen[tgt][2])
                if chosen[tgt] else "n/a")
    bb = b2["%.0e|strong|single" % tgt]["stats"] if chosen[tgt] else None
    bbs = b2["%.0e|strong|swarm" % tgt]["stats_per_candidate"] if chosen[tgt] else None
    bq = b1["%.0e|strong" % tgt]["stats"]
    bm = b0["strong"]["stats"]
    spS = sp["strong"]

    def _f(x, nd=1):
        return ("%%.%df" % nd) % x

    tex = []
    tex.append("% ---------------------------------------------------------------")
    tex.append("% Generated by bench_kernel.py.  Every number is a measurement made")
    tex.append("% by time.perf_counter_ns on ONE machine: Intel Core i5-14600KF,")
    tex.append("%% Windows 11 Pro build 26200, CPython %s, numpy %s,"
               % (sys.version.split()[0], np.__version__))
    tex.append("%% pinned and verified to logical processor %d (a P-core), BLAS capped"
               % args.cpu)
    tex.append("% to one thread.  Strong turbulence (alpha=1.2, beta=1.1), gbar=38 dB,")
    tex.append("%% xi=1.967, sigma_s=0.05 m, A_0=%.5f.  n=%d samples per series row."
               % (pts[XI_REF]["A0"], args.reps))
    tex.append("% Raw per-call sample arrays: kernel_timing_raw.npz")
    tex.append("% ---------------------------------------------------------------")
    tex.append("\\begin{tabular}{lrrrrrrr}")
    tex.append("\\hline")
    tex.append("& \\multicolumn{5}{c}{Single candidate ($\\mu$s/eval)}"
               " & \\multicolumn{2}{c}{Swarm $N_p{=}30$ ($\\mu$s/particle)} \\\\")
    tex.append("\\cline{2-6}\\cline{7-8}")
    tex.append("Evaluator & floor$^{\\dagger}$ & med.$^{\\ast}$ & P95 & P99.9 & max"
               " & floor$^{\\dagger}$ & P99.9 \\\\")
    tex.append("\\hline")
    for K in ORDERS:
        st = rec["kernel"]["kernel_strong_K%d_single" % K]
        sw = rec["kernel"]["kernel_strong_K%d_swarm%d" % (K, N_SWARM)]
        tex.append("RT-ODT series, $K=%d$ & %s & %s & %s & %s & %s & %s & %s \\\\"
                   % (K, _f(st["min"]), _f(st["median_fast_mode"]), _f(st["p95"]),
                      _f(st["p999"]), _f(st["max"]),
                      _f(sw["min"], 3), _f(sw["p999"], 2)))
    if bb:
        tex.append("Optimized f64 quadrature$^{\\ddagger}$ "
                   "& %s & %s & %s & %s & %s & %s & %s \\\\"
                   % (_f(bb["min"]), _f(bb["median_fast_mode"] or bb["median"]),
                      _f(bb["p95"]), _f(bb["p999"]), _f(bb["max"]),
                      _f(bbs["min"], 1), _f(bbs["p999"], 1)))
    tex.append("Adaptive Gauss--Kronrod$^{\\S}$ & %s & %s & %s & %s & %s & --- & --- \\\\"
               % (_f(bq["min"], 0), _f(bq["median"], 0), _f(bq["p95"], 0),
                  _f(bq["p999"], 0), _f(bq["max"], 0)))
    tex.append("Exact Meijer-G, mpmath$^{\\P}$ & %s & %s & %s & %s & %s & --- & --- \\\\"
               % (_f(bm["min"], 0), _f(bm["median"], 0), _f(bm["p95"], 0),
                  _f(bm["p999"], 0), _f(bm["max"], 0)))
    tex.append("\\hline")
    tex.append("\\end{tabular}")
    tex.append("")
    tex.append("% $^{\\dagger}$ ``floor'' is the observed minimum: the one estimator here "
               "that interference")
    tex.append("% cannot move, since contention only ever inflates a measured cost.")
    tex.append("% $^{\\ast}$ The single-candidate distribution is bimodal -- the cost "
               "roughly doubles")
    tex.append("%% whenever Windows schedules other work on the SMT sibling of the pinned "
               "core (measured %.2fx)."
               % rec.get("smt_sibling_arm", {})
               .get("loaded_over_ambient_ratio_min", float("nan")))
    _slow = rec["kernel"]["kernel_strong_K10_single"]["median_slow_mode"]
    tex.append("%% ``med.'' is the median of the uncontended mode; the contended mode "
               "sits at %.1f us and" % (_slow if _slow else float("nan")))
    tex.append("%% carried %.1f%% of samples in this run.  A swarm call is 30x longer, "
               "straddles both modes,"
               % (100 * rec["kernel"]["kernel_strong_K10_single"]
                  ["fraction_above_1p5x_min"]))
    tex.append("% and so has no mode-resolved median; its floor is quoted instead.  "
               "Ambient medians and")
    tex.append("% every other percentile are in the JSON and recomputable from the .npz.")
    tex.append("%% $^{\\ddagger}$ %s;" % rule_tag)
    tex.append("%% cheapest rule in a searched family reaching abs.\\ error $\\le$ %.0e "
               "(reached %.1e)" % (tgt, err_grid[chosen[tgt]] if chosen[tgt] else 0))
    tex.append("% against a 90-digit reference in all three regimes.")
    tex.append("%% $^{\\S}$ nested \\texttt{scipy.integrate.quad} with "
               "\\texttt{scipy.special.kv}, epsabs=epsrel=%.0e." % tgt)
    tex.append("% $^{\\P}$ \\texttt{mpmath} integration of the Meijer-G composite "
               "density at 15 significant digits.")
    tex.append("")
    tex.append("% Speedups: each is a ratio of two quantities measured on this machine,")
    tex.append("% round-robin against each other so both saw the same ambient load.")
    if bb:
        tex.append("%%   single candidate      : %.2fx on floors, %.2fx on uncontended "
                   "medians"
                   % (spS.get("speedup_vs_B2_scalar_minima", float("nan")),
                      spS.get("speedup_vs_B2_scalar_fast_mode", float("nan"))))
        tex.append("%%   per particle, N_p=30  : %.1fx on floors, %.1fx on ambient "
                   "medians"
                   % (spS.get("speedup_vs_B2_per_candidate_minima", float("nan")),
                      spS.get("speedup_vs_B2_per_candidate_medians", float("nan"))))
    tex.append("%%   vs adaptive Gauss-Kronrod : %.0f$\\times$"
               % spS["speedup_vs_B1_scalar_medians"])
    tex.append("%%   vs exact Meijer-G         : %.0f$\\times$"
               % spS["speedup_vs_B0_scalar_medians"])
    out_tex = os.path.splitext(out_json)[0] + "_table.tex"
    with open(out_tex, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    rec["latex_table_file"] = os.path.basename(out_tex)

    # ---------------------------------------------------------------- write
    rec["elapsed_s"] = round(time.time() - t_start, 2)
    rec["raw_sample_arrays"] = {
        "file": os.path.basename(out_npz),
        "unit": "nanoseconds per timed call (NOT divided by swarm size)",
        "keys": sorted(raw.keys()),
        "note": ("one element per call of the timed function; no averaging, no inner "
                 "repetition loop, no outlier removal, no smoothing.  Every percentile "
                 "reported in this JSON can be recomputed from these arrays, and so can "
                 "the serial correlation."),
    }
    np.savez_compressed(out_npz, **{k: v.astype(np.float64) for k, v in raw.items()})
    with open(out_json, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)

    set_affinity(original_affinity or list(range(os.cpu_count())))
    print("\n[+] wrote %s" % out_json)
    print("[+] wrote %s (%d arrays, %.1f KB)"
          % (out_npz, len(raw), os.path.getsize(out_npz) / 1024.0))
    print("[i] elapsed %.1f s" % rec["elapsed_s"])


if __name__ == "__main__":
    main()
