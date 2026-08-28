# Verifying this paper

```
python verify.py
```

Roughly 40 seconds, no arguments, no network, no GPU. Exit status is 0 only if every
check passes. Requires Python 3.8+, NumPy, SciPy and mpmath; a Python 3.6 fallback path
exists for the Jetson (see *Platforms* below).

The script prints one line per check and ends with a count. A run that says `7/7` and
exits 0 means every claim in the table below was recomputed on your machine and agreed
with what the manuscript prints.

---

## What is checked, and how

The checks are grouped by what kind of evidence they are, because they are not equally
strong and a reader should not have to work out which is which.

### Tier 1 — deterministic

Each is either a regeneration from the equations as printed in the paper, or two
independent computations of the same quantity that must agree. These do not depend on
timing, scheduling, or a random seed, so they pass or fail identically on any machine.

| Check | What it establishes |
|---|---|
| Model rebuilt from the printed equations | 19 sub-checks. The system model was re-implemented from the equations in the manuscript alone and reproduces the paper's own four published ABER values. |
| float64 kernel vs 200-digit reference | Worst relative departure 3.75e-13, against a float64 floor of ~1e-16 amplified by series cancellation. |
| Off-grid draws | 34,864 draws **off** the design grid: worst departure 2.231e-9, i.e. 448x inside the 1e-6 requirement, and every returned value is a probability (34,864/34,864 in range). |
| Eq. (22) vs an independent reference | Median 0.0043%, max 0.0825% over 39 in-band points. |
| Surrogate ranks like the system metric | Spearman 0.9997 over 34 admissible beams. |
| Manuscript cites the dataset it ships | Parses the numbers out of `access.tex` and compares them against the shipped `.npz`. |
| Table 7 bounds regenerated | 17 lines of comparison, regenerated from the equations. |

The off-grid check is the one that matters most and is worth stating precisely: it is a
**worst observed departure over 34,864 draws**, not a pointwise bound. It is evidence
about the sampled region, not a proof about every point in it.

### Tier 2 — measured, machine-dependent

Reported by the script but not counted toward the 7/7, because your numbers will differ
from ours and *should*:

- **Feasibility ceiling.** A solver-free brute-force scan. This is a property of the link
  budget, not of any optimizer, and it is the reason the reported optimization success
  rate of 24.9% sits against a ceiling of 25.0%.
- **Platform timings.** Wall-clock, therefore dependent on your CPU, its scheduler, and
  its thermal state.

### Tier 3 — what this package cannot reproduce

Printed by the script itself, at the end of every run, so that no reader has to discover
it by trying:

- The closed-loop campaign driver **as originally deployed** is not in the release. The
  published optimization-success and latency figures therefore cannot be re-executed from
  this package. What the release supports is the *bound*, not the reproduction.
- The 200 us TCN INT8 inference figure is a **design target** on an AGX-class device. No
  engine was built for the Jetson TX2 available here, and nothing measured it.
- **No physical testbed or steering mirror was involved anywhere in this work.** The
  optical link is simulated throughout. The computation is measured on real hardware; the
  propagation is not.

---

## The component ablation, and why the paper reports a null

`code/ablation_continuous.py` and `code/levy_mechanism_probe.py` re-score the ablation on a
continuous metric and probe the Lévy operator directly. They take a few minutes each and
are not part of the 7-check run, because their outputs are measurements rather than
pass/fail assertions. Both write to `data/12_continuous/`.

The short version of what they establish, which the manuscript states in
Section VII-D6:

- At the deployed budget the ablation arms are not merely indistinguishable but
  **bit-identical**. One iteration completes in 600 µs, so the Lévy jump perturbs a swarm
  that has not moved, GA refinement recombines an elite that does not exist, and the
  fidelity ladder has nothing to adapt across. The components are inert at that budget,
  not weak.
- On the continuous metric with all 25 iterations running, **chaotic initialisation** is
  the component that pays: p < 10⁻⁴, and its benefit is 15× larger over the coupled
  20-stage trajectory (2.4% ABER) than over one ranked stage (0.16%) — which is the
  paper's own multimodality argument coming out the way it predicted.
- **The Lévy operator does not separate from a Gaussian step of the same scale** under any
  configuration tested: p = 0.58 (one stage), 0.084 (twenty stages), 0.72 (stagnation-gated
  and run to 200 iterations, where the gate is open on 193.8 of them), 0.997 (ungated).

One discrepancy was found and corrected in the manuscript rather than left for a reader to
discover: earlier text described the Lévy jump as gated on a stagnation threshold
`Var(J_gbest) < ε_s`. The released solver has never contained that trigger; it fires
unconditionally at 25% per particle per iteration. The gated form is now implemented in
`hclpso_ga.py` behind `stagnation_gated`, **off by default** — the default path was verified
bit-identical to the pre-patch solver, so every previously published number is unaffected.

## Platforms

The kernel benchmark `code/bench_portable.py` runs on both machines used in the paper and
pins itself to a core on each:

| | Platform A' | Platform B |
|---|---|---|
| Device | x86-64 desktop, Windows | Jetson TX2, Linux |
| Pinning | `psutil` affinity | `os.sched_setaffinity` |
| Python | 3.11 | 3.6 |

`bench_portable.py` is deliberately written to the Python 3.6 subset (no
`from __future__ import annotations`, an explicit `_now_ns` shim) so that one file runs on
both. It ships `coeff_pack.npz`, 5.4 KB of xi-free constants, so the two machines execute
byte-identical arithmetic and any difference in result is a difference in the hardware
rather than in the inputs.

---

## If a check fails

Report the failing line. A Tier 1 failure is a real disagreement and we want to know: the
whole point of the tier split is that those checks are supposed to be
machine-independent, so a Tier 1 failure on your machine is a defect in the paper, not in
your setup.

Two known-benign causes of a *near* miss, neither of which should trip the tolerances:
the checks compare against tolerances that already allow for float64 cancellation in the
weak-turbulence regime, and mpmath's precision is set explicitly by the script rather
than inherited from your environment.
