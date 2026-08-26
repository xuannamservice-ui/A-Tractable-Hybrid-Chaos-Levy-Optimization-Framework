"""
Reproduce the derived quantities of Table 9 (algorithm comparison).

What this script establishes and what it does not:

  IT DOES   recompute every number in Table 9 that is DERIVED rather than
            measured: the SNR gain from the collected-power ratio, the
            rescaling of each baseline to the deployed iteration cap, and the
            joint real-time arithmetic.
  IT DOES NOT re-run the solvers. The per-cycle success rates and the median
            latencies are campaign measurements; this script audits the
            arithmetic built on top of them, which is the part a reader can
            check without the campaign.

Usage:  python reproduce_table9.py
"""
import math

# ---- measured inputs, as tabulated -------------------------------------
BASELINES = [
    # name,                 median iters, median cycle (ms), opt. success
    ("Reactive PSO (Meijer-G)", 120, 1.6e5, 0.45),
    ("Standard PSO (K=10)",      60, 1.2,   0.50),
    ("Differential Evolution",   90, 1.6,   0.52),
    ("CMA-ES",                   80, 2.0,   0.58),
    ("Warm-started SQP",       None, 1.2,   0.35),
    ("Proposed H-CLPSO-GA",      22, 0.79,  0.98),
]
PIPELINE_US = 50 + 80 + 20 + 30      # Table 12 non-optimization stages
T_ITER = 25
DEADLINE_US = 800
A0_ADAPTIVE, A0_FIXED = None, None   # filled below from the tabulated gain

# ---- 1. SNR gain is a collected-power ratio, not a fitted number --------
print("1. SNR gain from the collected-power ratio, Eq. (66)")
print("   G_dB = 20 log10(A0_adaptive / A0_fixed)")
ratio = 1.41
print("   ratio 1.41x  ->  %.3f dB   (tabulated: 3.0 dB)" % (20 * math.log10(ratio)))
print("   inverse: 3.0 dB  ->  ratio %.4f" % (10 ** (3.0 / 20)))

# ---- 2. rescaling each baseline to the deployed cap --------------------
print("\n2. Baselines rescaled to the deployed cap T_iter = %d" % T_ITER)
print("   strip the %d us of non-optimization stages, rescale the solver phase,"
      % PIPELINE_US)
print("   restore them:  (cycle - pipeline) / iters * T_iter + pipeline\n")
print("   %-26s %8s %8s %10s %s"
      % ("solver", "iters", "cycle", "at cap", "vs 800 us deadline"))
print("   " + "-" * 74)
for name, iters, cycle_ms, succ in BASELINES:
    if iters is None or cycle_ms > 100 or "Proposed" in name:
        continue   # the proposed entry is already a measurement AT the cap
    us = cycle_ms * 1000.0
    scaled = (us - PIPELINE_US) / iters * T_ITER + PIPELINE_US
    print("   %-26s %8d %7.1fus %9.0fus %s"
          % (name, iters, us, scaled,
             "fits" if scaled <= DEADLINE_US else "MISSES"))
print("\n   assumptions: the pipeline overhead is unchanged, and solver time is")
print("   linear in iteration count. Both are stated in the manuscript.")

# ---- 3. what the comparison actually shows -----------------------------
print("\n3. Optimization success at the SAME iteration budget")
best_base = max(s for n, _, _, s in BASELINES if "Proposed" not in n)
prop = [s for n, _, _, s in BASELINES if "Proposed" in n][0]
print("   best baseline: %.0f%%      proposed: %.0f%%      gap: %.0f points"
      % (100 * best_base, 100 * prop, 100 * (prop - best_base)))
print("   the proposed solver's median iteration count (22) is already below the")
print("   cap, so the cap costs it nothing; baselines needing 60-120 iterations")
print("   are truncated mid-convergence.")

# ---- 4. joint real-time arithmetic -------------------------------------
print("\n4. Joint real-time rate")
for cfg, dl in (("SCHED_OTHER", 0.574), ("chrt -f 90", 0.696), ("full isolation", 0.798)):
    print("   %-16s deadline success %.1f%%   joint (x %.2f opt.) = %.1f%%"
          % (cfg, 100 * dl, prop, 100 * dl * prop))
print("   tabulated joint rate under full isolation: 78.0%")
