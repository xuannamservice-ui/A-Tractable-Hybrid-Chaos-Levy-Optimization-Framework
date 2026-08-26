"""
Check the ARITHMETIC of Table 9 (algorithm comparison).  Not a reproduction.

WHY THE NAME CHANGED
    This file was called `reproduce_table9.py` and was listed under "Scripts
    that reproduce published tables".  Only section 1 below regenerates anything
    from the model.  Sections 2-4 take Table 9's own measured columns -- median
    iteration counts, median cycle times, per-solver success rates -- and
    Table 12's own deadline-success rates, and verify that arithmetic performed
    on those published numbers closes.  Published numbers in, arithmetic on
    them out: that is a derived-column audit, and calling it a reproduction
    overstates it by the whole distance between "the sums add up" and "the
    measurements are right".

WHAT THIS SCRIPT ESTABLISHES, SECTION BY SECTION

  1. REGENERATED FROM THE MODEL.  The collected-power ratio behind the SNR-gain
     column, recomputed from the beam geometry of eq. (3) with no tabulated
     input.  It does not agree with a single 3.0 dB figure and says why.
  2. DERIVED-COLUMN AUDIT.  Rescaling each baseline to the deployed iteration
     cap.  Inputs: the tabulated median iterations and median cycle times.
  3. DERIVED-COLUMN AUDIT.  The equal-budget success comparison.  Inputs: the
     tabulated per-solver success rates.
  4. DERIVED-COLUMN AUDIT.  The joint real-time rate.  Inputs: the tabulated
     optimization success rate and the Table 12 deadline-success rates.

  IT DOES NOT re-run the solvers, and it cannot check any measured column.  In
     particular it takes the 98% optimization-success rate as an input in
     sections 3 and 4; nothing here is evidence for that number.  The
     system-level feasibility of the 1e-6 target across the swept decision box
     is measured instead by `system_metric.py` (section [5] of its self-check)
     and by `run_campaign.py`'s feasibility ceiling.

Usage:  python check_table9_arithmetic.py
"""
import math

import numpy as np

from campaign import (geom, pe_exact, NODES, SIGMAS, ALPHA, BETA, GAMMA_OP,
                      LADDER_K, wzeq_min)

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
# Table 12 deadline-success rates by scheduling configuration. These are
# measurements, reproduced here only so the joint arithmetic in section 4 has
# its inputs on the page.
DEADLINE_SUCCESS = (("SCHED_OTHER", 0.574), ("chrt -f 90", 0.696),
                    ("full isolation", 0.798))
TABULATED_JOINT_ISOLATION = 0.780
TABULATED_GAIN_DB = 3.0

# ---- 1. the SNR gain column --------------------------------------------
print("=" * 78)
print("TABLE 9 ARITHMETIC CHECK -- NOT A REPRODUCTION OF TABLE 9")
print("=" * 78)
print("Section 1 is regenerated from the model. Sections 2-4 take Table 9's")
print("and Table 12's own measured columns as INPUTS and verify that the")
print("arithmetic built on them closes. They cannot check a measured column,")
print("and the 98% optimization-success rate is one of those inputs, not a")
print("result. See the module docstring.")
print("=" * 78)
print()
print("1. [REGENERATED FROM THE MODEL] SNR gain from the collected-power")
print("   ratio, Eq. (66)")
print("   G_dB = 20 log10(A0_adaptive / A0_fixed)")
print()
print("   CIRCULARITY WARNING. An earlier version of this script asserted a")
print("   collected-power ratio of 1.41 and reported that it yields 3.0 dB.")
print("   But 10^(3.0/20) = %.4f, so that 1.41 IS the tabulated 3.0 dB read"
      % (10 ** (TABULATED_GAIN_DB / 20)))
print("   backwards. Checking it against 3.0 dB verifies nothing except that")
print("   the round trip through a logarithm works. The ratio is recomputed")
print("   from the geometry below instead.")
print()

# A_0 is fixed by the beam waist through eq. (3), so the ratio the gain column
# rests on is computable. What it is a ratio OF is not stated in the paper, so
# the definition used here is spelled out: A0_adaptive is the collected power
# at the divergence the controller would choose for the jitter it actually
# faces; A0_fixed is the collected power of a beam frozen at the divergence
# that is optimal for the mid-range jitter sigma_s = 0.1 m. Any other choice of
# the frozen setting gives a different number, which is exactly why this column
# cannot be pinned down from the manuscript alone.
_wz = np.linspace(0.055, 3.0, 20000)
_A0, _wq = geom(_wz)
_WEQ_MIN = wzeq_min()
_z = np.sqrt(2) * ALPHA * BETA / (_A0 * np.sqrt(GAMMA_OP))

best = {}
for s in SIGMAS:
    xi = _wq / (2 * s)
    inbox = (xi >= max(0.5, _WEQ_MIN / (2 * s))) & (xi <= 4.888)
    pe = pe_exact(np.clip(xi, NODES[0], NODES[-1]), _A0, LADDER_K(_z))
    v = np.where(inbox & np.isfinite(pe) & (pe >= 0.0) & (pe <= 0.5), pe, np.inf)
    i = int(np.argmin(v))
    best[s] = (_wz[i], _A0[i], v[i])

A0_FIXED = best[0.1][1]
print("   optimal beam per jitter level (38 dB, strong turbulence, ladder K):")
print("   %-10s %10s %10s %12s %12s %10s"
      % ("sigma_s", "w_z* [m]", "A_0*", "Pe*", "A_0*/A_0_fix", "G_dB"))
for s in SIGMAS:
    w, a0, pe = best[s]
    r = a0 / A0_FIXED
    print("   %-10.2f %10.4f %10.5f %12.4e %12.4f %+10.2f"
          % (s, w, a0, pe, r, 20 * math.log10(r)))
print()
print("   The ratio the geometry supplies spans %+.1f to %+.1f dB across the"
      % (20 * math.log10(best[SIGMAS[-1]][1] / A0_FIXED),
         20 * math.log10(best[SIGMAS[0]][1] / A0_FIXED)))
print("   swept jitter range, so a single 3.0 dB figure is a point on that")
print("   curve, not a property of the model. Which point is not recoverable")
print("   from the manuscript; that is reported, not resolved.")

# ---- 2. rescaling each baseline to the deployed cap --------------------
print("\n2. [DERIVED-COLUMN AUDIT -- inputs are Table 9's measured iteration")
print("   counts and cycle times] Baselines rescaled to the deployed cap")
print("   T_iter = %d" % T_ITER)
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
print("\n3. [DERIVED-COLUMN AUDIT -- inputs are Table 9's measured success")
print("   rates, the 98% among them] Optimization success at the SAME")
print("   iteration budget")
best_base = max(s for n, _, _, s in BASELINES if "Proposed" not in n)
prop = [s for n, _, _, s in BASELINES if "Proposed" in n][0]
print("   best baseline: %.0f%%      proposed: %.0f%%      gap: %.0f points"
      % (100 * best_base, 100 * prop, 100 * (prop - best_base)))
print("   the proposed solver's median iteration count (22) is already below the")
print("   cap, so the cap costs it nothing; baselines needing 60-120 iterations")
print("   are truncated mid-convergence.")

# ---- 4. joint real-time arithmetic -------------------------------------
print("\n4. [DERIVED-COLUMN AUDIT -- inputs are Table 12's measured deadline-")
print("   success rates and Table 9's 98%] Joint real-time arithmetic")
for cfg, dl in DEADLINE_SUCCESS:
    print("   %-16s deadline success %.1f%%   joint (x %.2f opt.) = %.1f%%"
          % (cfg, 100 * dl, prop, 100 * dl * prop))
joint = DEADLINE_SUCCESS[-1][1] * prop
print("   tabulated joint rate under full isolation: %.1f%%   recomputed: %.1f%%   %s"
      % (100 * TABULATED_JOINT_ISOLATION, 100 * joint,
         "OK" if abs(joint - TABULATED_JOINT_ISOLATION) <= 0.005 else "MISMATCH"))
print("   (the product assumes deadline misses and optimization failures are")
print("   independent, which the manuscript states and this script cannot test.)")
