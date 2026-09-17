"""Check the ARITHMETIC of the optimizer-comparison table.  Not a reproduction.

WHY THE NAME CHANGED
    This file was called `reproduce_table9.py` and was listed under "Scripts
    that reproduce published tables".  It regenerates nothing from the model:
    it takes the table's own measured columns -- median iteration counts,
    median cycle times -- and verifies that arithmetic performed on those
    published numbers closes.  Published numbers in, arithmetic on them out:
    that is a derived-column audit, and calling it a reproduction overstates
    it by the whole distance between "the sums add up" and "the measurements
    are right".

WHAT CHANGED SINCE, AND WHAT WAS REMOVED
    Earlier versions also audited two columns the manuscript no longer
    contains: a per-solver optimization-success comparison at a common budget,
    and a joint real-time rate formed as (deadline success) x (optimization
    success) across three Linux scheduling configurations.  Those columns were
    withdrawn, so auditing them here would be auditing a table that does not
    exist.  They are gone rather than updated.

    A third section derived the reported SNR gain from a collected-power ratio,
    G_dB = 20 log10(A0_adaptive / A0_fixed).  That is not how the manuscript
    defines the gain: it is a REQUIRED-SNR reduction, the difference between
    the SNRs at which two beams first reach the 1e-6 post-EGC target.  That
    quantity needs the system evaluator, not a closed-form ratio, so it is
    computed by `gamma_min_probe.py` instead and is not reproduced here.

WHAT THIS SCRIPT STILL ESTABLISHES
    Only the rescaling audit: what each RT-ODT baseline's median cycle time
    would be if it were held to the deployed iteration cap T_iter = 25 rather
    than run to its own natural budget.  Inputs are the tabulated median
    iteration counts and median cycle times, plus the non-optimization stage
    total from the latency table.  It does not re-run any solver and cannot
    check any measured column.

Usage:  python check_table9_arithmetic.py
"""

# ---- measured inputs, as tabulated -------------------------------------
BASELINES = [
    # name,                     median iters, median cycle (ms)
    ("Standard PSO (K=10)",      60, 1.2),
    ("Differential Evolution",   90, 1.6),
    ("CMA-ES",                   80, 2.0),
]
PIPELINE_US = 50 + 80 + 20 + 30      # latency table's non-optimization stages
T_ITER = 25
DEADLINE_US = 800

print("=" * 78)
print("OPTIMIZER-TABLE ARITHMETIC CHECK -- NOT A REPRODUCTION")
print("=" * 78)
print("Inputs are the table's own measured median iteration counts and median")
print("cycle times. This audit rescales them; it cannot check them. See the")
print("module docstring for the two sections that were removed and why.")
print("=" * 78)

print("\n[DERIVED-COLUMN AUDIT] Baselines rescaled to the deployed cap")
print("   T_iter = %d" % T_ITER)
print("   strip the %d us of non-optimization stages, rescale the solver phase,"
      % PIPELINE_US)
print("   restore them:  (cycle - pipeline) / iters * T_iter + pipeline\n")
print("   %-26s %8s %9s %10s %s"
      % ("solver", "iters", "cycle", "at cap", "vs 800 us deadline"))
print("   " + "-" * 74)
for name, iters, cycle_ms in BASELINES:
    us = cycle_ms * 1000.0
    scaled = (us - PIPELINE_US) / iters * T_ITER + PIPELINE_US
    print("   %-26s %8d %8.1fus %9.0fus %s"
          % (name, iters, us, scaled,
             "fits" if scaled <= DEADLINE_US else "MISSES"))
print("\n   assumptions: the pipeline overhead is unchanged, and solver time is")
print("   linear in iteration count. Both are stated in the manuscript, and")
print("   neither is measured -- the rescaled column is a hypothetical, not a")
print("   configuration that was run.")
