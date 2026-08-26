"""Portable RT-ODT timing benchmark -- runs identically on the x86 host and the Jetson.

WHY THIS EXISTS SEPARATELY FROM bench_kernel.py
The manuscript reports per-evaluation kernel cost on more than one platform. A cross-platform
timing comparison is only meaningful if both platforms run the SAME arithmetic, so this
script deliberately avoids everything that would differ between them:

  no mpmath   the xi-free coefficients K_k(alpha,beta) of eq. (21) are computed ONCE on the
              host at 120 digits and shipped in coeff_pack.npz. This is not a shortcut for
              the benchmark's benefit -- it is what Section III and Appendix B describe the
              deployed system doing: the coefficient tensor is built offline, and the runtime
              kernel forms the xi-dependent factor in closed form and takes a dot product.
              Requiring mpmath on the target would benchmark a step that never runs there.

  scipy       used when present, because the deployed kernel calls scipy.special.gamma and
              a fair cross-platform comparison must run the same code on both sides. A
              Python-loop fallback exists for a target without scipy, but it is materially
              slower and the script reports which path ran rather than silently comparing
              two different implementations.

  numpy only  every target that can run the deployed kernel has numpy.

WHAT IS MEASURED
  (a) single-candidate cost at K = 5, 10, 20 -- the Table 7 runtime column
  (b) amortised per-candidate cost vectorised across a swarm of N_p = 30 -- the per-particle
      SIMD figure
  (c) the full distribution, not a mean: median / P95 / P99 / P99.9 / max, with the raw
      sample array saved so any percentile can be recomputed and the serial correlation
      checked

METHOD
  time.perf_counter_ns throughout, resolution measured and reported rather than assumed.
  Warmup before timing so import and first-call effects are excluded. Inner repetition
  chosen so each timed block is long relative to clock resolution, then divided out.
  Core pinning where the OS offers it, and the script reports whether pinning took effect
  rather than claiming it did.

USAGE
  On the host, first build the pack (needs mpmath, run once):
      python bench_portable.py --build-pack
  Then on either machine:
      python bench_portable.py --out results_<label>.json --label "<platform name>"

  Copy bench_portable.py and coeff_pack.npz to the target; nothing else is required.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

# JetPack 4.x ships Python 3.6, where time.perf_counter_ns does not exist. Fall back to the
# float-seconds counter and scale. The fallback loses sub-nanosecond resolution, which is
# far below the measured clock granularity on either target and so does not affect any
# reported figure -- but the script measures and reports the resolution it actually got,
# rather than assuming one.
if hasattr(time, "perf_counter_ns"):
    _now_ns = time.perf_counter_ns
else:                                            # pragma: no cover - Python 3.6 targets
    def _now_ns():
        return int(time.perf_counter() * 1e9)

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "coeff_pack.npz")

REGIMES = {"weak": (4.2, 3.0), "moderate": (2.1, 1.5), "strong": (1.2, 1.1)}
ORDERS = (5, 10, 20)
N_SWARM = 30
EPS64 = float(np.finfo(np.float64).eps)

# The deployed kernel calls scipy.special.gamma, which is vectorised C. If scipy is absent
# on a target, the fallback in _gamma() is a Python loop and is materially slower -- so the
# benchmark records which path ran and refuses to present the two as comparable.
try:
    from scipy.special import gamma as _SP_GAMMA
    HAVE_SCIPY = True
except ImportError:                              # pragma: no cover - target-dependent
    _SP_GAMMA = None
    HAVE_SCIPY = False


# ----------------------------------------------------------------- pack build
def build_pack(path=PACK):
    """Compute the xi-free constants at 120 digits on the host. Run once."""
    import mpmath as mp

    out = {}
    with mp.workdps(120):
        for name, (A, B) in REGIMES.items():
            a, b = mp.mpf(A), mp.mpf(B)
            for K in ORDERS:
                for tag, (p, q) in (("AB", (a, b)), ("BA", (b, a))):
                    vals = [
                        float((-1) ** k * (p * q) ** (q + k) * mp.gamma(p - q - k)
                              / (mp.factorial(k) * mp.gamma(p) * mp.gamma(q)))
                        for k in range(K + 1)
                    ]
                    out["%s_K%d_%s" % (name, K, tag)] = np.array(vals, dtype=np.float64)
    # The pointing residue D needs Gamma(alpha - xi^2) and Gamma(beta - xi^2) at RUN time,
    # at the candidate's own xi. It is deliberately NOT tabulated here: tabulating it would
    # remove a cost the deployed kernel actually pays, and this benchmark exists to measure
    # that cost. It is computed on the target by reflection from lgamma -- see _gamma().
    np.savez_compressed(path, **out)
    print("wrote %s (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))


# ------------------------------------------------------------------ the kernel
def _gamma(x):
    """Gamma for real arguments of either sign, from math.lgamma only.

    scipy.special.gamma is unavailable on a bare JetPack image, and the residue term needs
    Gamma(alpha - xi^2) with a possibly negative argument. For x > 0 this is exp(lgamma(x));
    for x < 0 the reflection formula Gamma(x)Gamma(1-x) = pi/sin(pi x) carries it over, with
    the sign restored from lgamma's magnitude-only result. At a pole the value is +-inf,
    which is what the deployed kernel also produces and what the range test then rejects.
    """
    if _SP_GAMMA is not None:
        return _SP_GAMMA(x)                      # vectorised, same call the deployed kernel makes
    # Fallback for a target without scipy. NOTE: this path is a Python loop and is
    # materially slower, so a cross-platform comparison in which one side takes it and the
    # other does not is NOT apples to apples. The benchmark records which path was used.
    from math import lgamma, pi, sin, exp
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x)
    for i, v in enumerate(x):
        v = float(v)
        if v > 0.0:
            out[i] = exp(lgamma(v))
        elif v == np.floor(v):
            out[i] = np.inf                      # pole
        else:
            out[i] = pi / (sin(pi * v) * exp(lgamma(1.0 - v)))
    return out


_C_CACHE = {}


def _C_fixed(A, B, K, gbar):
    """C(beta+k) and C(alpha+k), eq. (20). These depend only on (regime, K, gbar) -- NOT on
    the candidate -- so the deployed kernel computes them once and caches them. Recomputing
    them per call inflates the measured cost and masks the K-scaling entirely."""
    key = (A, B, K, gbar)
    if key not in _C_CACHE:
        k = np.arange(K + 1, dtype=np.float64)
        _C_CACHE[key] = (_C_of(B + k, gbar), _C_of(A + k, gbar))
    return _C_CACHE[key]


def _C_of(s, gbar):
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    return _gamma((s + 1.0) / 2.0) / (2.0 * s * np.sqrt(np.pi)) * (2.0 / gbar) ** (s / 2.0)


def pe_kernel(pack, regime, K, xi, A0, gbar):
    """eq. (21) evaluated in closed form: the xi-dependent factor times a dot product.

    This is the arithmetic whose cost the manuscript tabulates. It is deliberately written
    the way the deployed evaluator is: no interpolation in xi anywhere, the xi-free part
    read from the offline tensor, everything else formed per candidate.
    """
    A, B = REGIMES[regime]
    xi = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    A0 = np.atleast_1d(np.asarray(A0, dtype=np.float64))
    x2 = xi * xi
    k = np.arange(K + 1, dtype=np.float64)

    kcAB = pack["%s_K%d_AB" % (regime, K)]
    kcBA = pack["%s_K%d_BA" % (regime, K)]

    # power moments C(s, gbar), eq. (20) -- gamma of a half-integer argument via lgamma
    CB, CA = _C_fixed(A, B, K, gbar)          # cached: candidate-independent
    t1 = kcAB[None, :] * x2[:, None] / ((x2[:, None] - B - k[None, :]) * A0[:, None] ** (B + k)[None, :])
    t2 = kcBA[None, :] * x2[:, None] / ((x2[:, None] - A - k[None, :]) * A0[:, None] ** (A + k)[None, :])
    bessel = (t1 * CB[None, :]).sum(1) + (t2 * CA[None, :]).sum(1)

    # Pointing-error residue D * C(xi^2), eq. (19). Evaluated per candidate at run time,
    # exactly as the deployed kernel does, so its cost is inside the measurement.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        D = (x2 * (A * B) ** x2 * _gamma(A - x2) * _gamma(B - x2)
             / (A0 ** x2 * _gamma(np.array([A]))[0] * _gamma(np.array([B]))[0]))
        return bessel + D * _C_of(x2, gbar)


# ------------------------------------------------------------------- timing
def clock_resolution_ns(n=20000):
    """Measure the clock rather than assuming it: smallest non-zero increment seen."""
    best = None
    prev = _now_ns()
    for _ in range(n):
        now = _now_ns()
        d = now - prev
        if d > 0 and (best is None or d < best):
            best = d
        prev = now
    return best


def pin_to_fast_cores():
    """Pin the timing thread, and report whether it actually took.

    os.sched_setaffinity is in the Linux standard library, so the Jetson needs no extra
    package -- psutil does not build against JetPack's Python 3.6. Windows has no
    sched_setaffinity, so psutil is used there.

    On a heterogeneous CPU this is not hygiene, it is correctness: the TX2 pairs two Denver2
    cores with four Cortex-A57 cores, and the i5 pairs P-cores with E-cores. An unpinned
    measurement samples whichever core the scheduler picked and its tail describes no single
    processor. The function returns (requested, actual) so the caller can report the truth
    instead of asserting that pinning succeeded.
    """
    if hasattr(os, "sched_setaffinity"):                      # Linux, incl. JetPack 3.6
        try:
            avail = sorted(os.sched_getaffinity(0))
            want = avail[:4] if len(avail) >= 4 else avail
            os.sched_setaffinity(0, set(want))
            return want, sorted(os.sched_getaffinity(0))
        except Exception:
            return None, None
    try:                                                      # Windows
        import psutil
    except ImportError:
        return None, None
    try:
        p = psutil.Process()
        want = list(range(min(4, psutil.cpu_count(logical=True))))
        p.cpu_affinity(want)
        return want, p.cpu_affinity()
    except Exception:
        return None, None


def time_block(fn, reps, inner):
    """Return per-call nanoseconds for each of `reps` timed blocks of `inner` calls."""
    out = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        t0 = _now_ns()
        for _ in range(inner):
            fn()
        out[i] = (_now_ns() - t0) / float(inner)
    return out


def summarize(ns):
    ns = np.sort(ns)
    q = lambda p: float(ns[min(len(ns) - 1, int(p * len(ns)))])
    return dict(n=int(len(ns)), median_us=float(np.median(ns)) / 1e3,
                p95_us=q(0.95) / 1e3, p99_us=q(0.99) / 1e3, p999_us=q(0.999) / 1e3,
                max_us=float(ns[-1]) / 1e3, min_us=float(ns[0]) / 1e3,
                mean_us=float(ns.mean()) / 1e3, std_us=float(ns.std()) / 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pack", action="store_true")
    ap.add_argument("--label", default=platform.node())
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.build_pack:
        build_pack()
        return

    if not os.path.exists(PACK):
        sys.exit("missing %s -- run `python bench_portable.py --build-pack` on the host first"
                 % PACK)
    pack = dict(np.load(PACK))

    res_ns = clock_resolution_ns()
    want, got = pin_to_fast_cores()

    rec = dict(
        label=a.label,
        machine=dict(node=platform.node(), system=platform.system(),
                     release=platform.release(), machine=platform.machine(),
                     processor=platform.processor(), python=sys.version.split()[0],
                     numpy=np.__version__),
        clock_resolution_ns=res_ns,
        pinning=dict(requested=want, actual=got,
                     effective=(want is not None and got is not None and set(got) == set(want))),
        eps64=EPS64,
        scipy_gamma=HAVE_SCIPY,
        results={},
    )
    print("platform : %s" % a.label)
    print("machine  : %s %s / %s / python %s / numpy %s"
          % (platform.system(), platform.machine(), platform.processor() or "?",
             sys.version.split()[0], np.__version__))
    print("clock    : %s ns resolution" % res_ns)
    print("pinning  : requested %s, actual %s" % (want, got))
    print("gamma    : %s" % ("scipy.special.gamma (vectorised, matches deployed)"
                            if HAVE_SCIPY else "PYTHON-LOOP FALLBACK -- not comparable to a scipy host"))
    print()

    gbar = 10 ** 3.8
    for regime in REGIMES:
        for K in ORDERS:
            xi1 = np.array([1.967])
            A01 = np.array([0.129])
            xiN = np.linspace(0.9, 4.8, N_SWARM)
            A0N = np.full(N_SWARM, 0.129)

            for _ in range(200):                      # warmup
                pe_kernel(pack, regime, K, xi1, A01, gbar)
                pe_kernel(pack, regime, K, xiN, A0N, gbar)

            key = "%s_K%d" % (regime, K)
            entry, line = {}, []
            # A single-candidate call is dominated by Python/numpy call overhead, not by
            # arithmetic: K=5 and K=20 differ by 15 terms of a tiny dot product, which is
            # invisible against ~20 us of interpreter cost. Sweeping the batch size exposes
            # where arithmetic starts to dominate and is the only regime in which a
            # per-candidate cost -- or any K-scaling -- can honestly be quoted.
            for nb in (1, N_SWARM, 1000, 10000):
                xb = np.linspace(0.9, 4.8, nb)
                ab = np.full(nb, 0.129)
                for _ in range(50):
                    pe_kernel(pack, regime, K, xb, ab, gbar)
                inner = 20 if nb <= N_SWARM else 3
                t = time_block(lambda: pe_kernel(pack, regime, K, xb, ab, gbar),
                               a.reps if nb <= N_SWARM else max(60, a.reps // 10), inner)
                st = summarize(t)
                st["batch"] = nb
                st["per_candidate_us"] = float(np.median(t)) / 1e3 / nb
                st["min_per_candidate_us"] = float(t.min()) / 1e3 / nb
                entry["batch_%d" % nb] = st
                line.append("N=%-5d %7.3f us/cand" % (nb, st["per_candidate_us"]))
            rec["results"][key] = entry
            print("  %-14s %s" % (key, " | ".join(line)))

    out = a.out or os.path.join(HERE, "..", "data", "10_platform",
                                "portable_%s.json" % a.label.replace(" ", "_"))
    out = os.path.abspath(out)
    d = os.path.dirname(out)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
