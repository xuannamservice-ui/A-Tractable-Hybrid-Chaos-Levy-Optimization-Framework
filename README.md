# Data and Code Availability

Reproduction package for *Real-Time Robust Beam Steering for 6G MIMO-FSO Networks:
A Tractable Hybrid Chaos–Lévy Optimization Framework*.

Everything here is regenerated from the equations printed in the manuscript. No
result depends on a pre-computed artefact that cannot be rebuilt from `code/`.

## Start here

```
python code/validate_model.py
```

This is the trust anchor. It rebuilds the channel geometry and the series
coefficients from Eqs. (2), (16)–(17) and (20) alone, and checks them against
values printed in the paper:

| Quantity | Paper | Rebuilt |
|---|---|---|
| `w_zeq` minimum | 0.0877 m at `w_z`=0.0549 | ✓ 0.08772 at 0.05487 |
| `A_0` at ξ=0.992, σ_s = 0.05/0.1/0.2/0.3 | 0.533 / 0.127 / 0.0318 / 0.0141 | ✓ |
| `max_k｜a_k(α,β)｜` at ξ=0.500 and 4.888 (weak, K=10, σ_s=0.1) | 5.9e5 and 8.2e31 | ✓ 5.903e5, 8.239e31 |
| float64 round-off floor `η_f64` | 1.3e-10 and 1.8e16 | ✓ 1.31e-10, 1.83e16 |
| `z` at the campaign operating point | 4.52, break-even 33 dB | ✓ 4.49, 32.99 |

Every row is printed with `got`, `want` and either `OK` or `MISMATCH (x%)` at a
2% tolerance. The table above records what the script prints; it is not a
promise the script makes.

Two notes on the coefficient row, since it is the one that carries the
argument. It is the family `a_k(α, β, ξ)` of Eq. (16), evaluated at σ_s = 0.1 m;
the argument-swapped family `a_k(β, α, ξ)` used by the same emulator is a decade
larger at ξ=4.888, and the script prints it alongside rather than maximising
over the two. The σ_s and the family are *choices* — the manuscript line that
quotes these figures states neither — and they are the ones that make the
surrounding text self-consistent. Both are recorded in a comment in the script.

The `η_f64` row multiplies the manuscript's own printed coefficient maxima by
`eps_mach`. That checks Eq. (27) was applied consistently to the numbers in the
text; it does not exercise this implementation. The script also prints the same
quantity built from the coefficient it just computed, which does.

If those reproduce, the rest of the package is built on a faithful model.

## Layout

```
code/     the scripts that produce every number below
data/     generated datasets (CSV + NPZ), see per-block notes
logs/     run log with timestamps, plus one provenance sidecar per block
MANIFEST.json     what produced each block, when, how long, complete or partial
generate.py       the driver
requirements.txt  third-party dependencies, with the reason for each floor
```

`MANIFEST.json` covers **every** directory under `data/` — it enumerates the
tree rather than a fixed list — with the producing script and command, the
wall-clock seconds, the finish time, a `complete`/`partial`/`unknown` status,
and per-file sizes, record counts and SHA-256. It is derived by
`code/build_manifest.py`, not hand-maintained; run it with `--check` to fail a
release in which any block lacks provenance **or is only partly accounted
for**.

The released `MANIFEST.json` listed **4** of the 10 blocks — `01_admissibility`,
`02_z_map`, `03_coefficient_tensors` and `05_eq22_validation` — and said
nothing about the other six. Two independent causes:

* `generate.py` rewrote the whole `blocks` dict from scratch on every run and
  wrote it only as each block *finished*, so any block still running when the
  driver was interrupted got no entry. Block 04 is last in the block order and
  open-ended, so it never finished inside a deadline; block 06 is the
  longest-running fixed-size block and was interrupted too. Both shipped files
  with no record of what made them.
* Blocks 07, 08 and 09 are not produced by `generate.py` at all — they come
  from `run_campaign.py` and `landscape_probe.py`, which write their own
  self-describing JSON, and nothing ever read it into the manifest.

Both are fixed: each block now drops a sidecar in
`logs/provenance_<block>.json` the moment it finishes, and
`build_manifest.py` assembles the manifest from those sidecars plus the
campaign artefacts' embedded metadata. Provenance is sourced in three ways, in
this order, and each entry records which one it used:

1. `logs/provenance_<block>.json`, written by `generate.py` as the block
   finishes — the only source that observed the run, so the only one that can
   assert `complete` or `partial`.
2. `build_manifest.EXTERNAL`, for blocks 07–09, whose own JSON carries the
   run's parameters and its `seconds`.
3. Failing both, whatever the block's own artefacts declare about themselves
   (`generated_by` / `generator`), accepted **only** if the named script
   actually exists in `code/`. Such an entry is `status: "unknown"` — nothing
   here watched the run, so completeness is not asserted — and it carries
   `provenance_source` saying the attribution is self-declared.

A block that none of the three can account for is written with
`produced_by: null` and an explicit `provenance_gap` rather than omitted, so a
gap is visible instead of silent. A block only *partly* accounted for — some
files attributed, others not — also carries `provenance_gap`, naming the
unattributed files, and also fails `--check`. `10_platform` is in that state
at the time of writing: 3 of its 11 files name their producer and 8 do not.

`MANIFEST.json` records a SHA-256 per file, so it is only as current as the
last `python code/build_manifest.py`. Re-run it after anything writes under
`data/` — including the open-ended block 04, which `code/offgrid_parallel.py`
can keep extending after `generate.py` has stopped.

Install with `pip install -r requirements.txt`. The floors that matter are
`numpy >= 2.0` (`np.trapezoid`, used by `egc_system.py` and `generate.py`) and
`mpmath >= 1.2` (`workdps`, used by `rtodt_fast.py`).

## Datasets

**`data/01_admissibility/admissibility_grid.csv`**
Truncation and round-off behaviour over the full grid: 3 turbulence regimes ×
4 jitter levels × the pole-free ξ nodes attainable at each × 16 SNR points
(20–50 dB) × K ∈ {5, 10, 20}. Columns include the conditioning parameter `z`,
the ladder-selected order, the float64 floor `η_f64 = max_k|a_k C| · ε_mach`,
the first omitted term, and the evaluated series. Rows with `admissible = 0`
show the series returning values outside [0, ½] — the behaviour the runtime
guard exists to reject.

**`data/02_z_map/z_map.{npz,csv}`**
The conditioning map: `A_0`, `z`, the ladder order and the admissible flag on
the same grid. This is the artefact the manuscript refers to as `z_map.npz`.

**`data/03_coefficient_tensors/lookup_tensor_<regime>.npz`**
`a_k(α,β,ξ)`, `a_k(β,α,ξ)`, `D(ξ)` and `A_0(ξ; σ_s)` on the pole-free node grid
up to `K_max = 20`, one file per regime, with the σ_s axis explicit because the
same ξ gives a different `A_0` at each jitter level.

**`data/04_offgrid_error/offgrid_error.csv`**
Randomly sampled off-grid ξ, comparing the **deployed** float64 kernel
`rtodt_fast.pe_series_f64` (eq. 21, interpolation-free, the one the manuscript
reports results on) against `rtodt.Pe_series` carried in mpmath at 200 digits.
34 864 rows over 3 regimes × 4 σ_s × 5 SNR points, ξ sampled uniformly in
[0.500, 4.888]. Open-ended: it samples until the run's deadline, so its
`MANIFEST.json` status is **`partial`** on every full run by construction, not
because a run failed.

⚠ **`abs_err_interp_free` in the previous release was exactly `0.000e+00` on
every row, and that was not a result.** Both sides of the subtraction were the
same mpmath function — `rtodt.Pe_series` at dps 200 and at dps 90 — and both
were cast to float64 before differencing. float64 carries ~16 digits, so
rounding a 200-digit and a 90-digit value of the same quantity gives the
identical double. The column measured mpmath's round-off against itself. It
exercised **no** float64 kernel and **no** interpolation, in a block whose
column name and README entry both advertised exactly that.

`code/verify_block04.py` replays *both* comparisons on the same sampled rows.
The old one reproduces its defect exactly — **200 / 200** rows at `0.000e+00`,
maximum `0.000e+00`. The same rows through the deployed kernel give a spread of
real errors (4 / 200 at exactly zero, median 2.0e-15, max 1.2e-10). On the full
regenerated file:

| | value |
|---|---|
| rows | 34 864 |
| `abs_err` exactly `0.000e+00` | 855 (2.45%) — the double genuinely round-trips |
| `abs_err` median / p99 / max | 1.53e-15 / 7.61e-12 / **2.23e-09** |
| `rel_err` median / p99 / max | 9.37e-15 / 4.62e-11 / **2.54e-08** |
| rows failing guard test (ii), `0 ≤ Pe ≤ ½` | 0 reference, 0 float64 |

Two things this measures that nothing in the package measured before.

* **The worst case is two decades worse than the self-check reports.** The
  `rtodt_fast.py` self-check quotes 2.7e-10 worst relative disagreement over
  three regimes × three ladder rungs; random off-grid sampling finds 2.5e-8.
  The mechanism is the one the self-check already names — proximity to the
  `a_k` poles at ξ² = β+k. Over all 34 864 rows the median distance to the
  nearest pole is 0.249; over the 188 rows (0.54%) with `rel_err > 1e-10` it
  is 0.039, and every one of the worst eight sits within 0.03 of a pole.
* **Eq. (27)'s round-off floor under-predicts the error the kernel commits.**
  `eta_f64 = max_k|a_k C|·ε_mach` is a per-term estimate; the evaluation sums
  2(K+1) terms. `abs_err` exceeds `eta_f64` on **68.7%** of rows, with a
  median ratio of 1.75 — consistent with that term count — and a worst ratio
  of 7.2e4 beside a pole. Eq. (27) is a useful scale, not a bound, and the
  file lets a reader see the difference row by row.

`code/offgrid_parallel.py` extends the same file in parallel without
rewriting it; it refuses to append if the header does not match.

**`data/05_eq22_validation/eq22_vs_reference.csv`**
The λ_j / C_j convolved series of Eq. (22) evaluated against an independent
16-fold convolution reference, across regimes, jitter levels, beam
configurations and SNR. 132 rows. This is the dataset that extends the
single-point validation of Section III-D across the parameter box.
`summary.json` beside it carries the statistics below in machine-readable
form, rebuilt from the CSV by `code/eq22_summary.py`.

⚠ **Read the `admissible` column before reading `rel_diff_percent`.** The
previous release of this file shipped no `z` column, no admissibility column
and no caveat, and on it the median |relative difference| across all 132 rows
was **100.0%** (worst 2.8×10¹⁶⁵), with 3 rows carrying a *negative*
`exact_reference` — a number that cannot be a probability — and nothing
marking them. A reader had no way to tell which rows the manuscript's claims
were ever meant to cover.

On the regenerated file the same unscoped median is **86.1%**. That number is
real and is printed here rather than suppressed, but on its own it is
misleading. Splitting the same 132 rows on the band the manuscript actually
claims Eq. (22) on:

| scoping | rows scored | median &#124;rel diff&#124; | worst |
|---|---|---|---|
| all rows, no scoping | 130 | 86.1% | 2.8×10¹⁶⁵ |
| out-of-band (`admissible = 0`) | 88 | 61 758% | 2.8×10¹⁶⁵ |
| in-band (`admissible = 1`) | 42 | 0.0048% | 99.998% |
| in-band **and** reference resolved | 39 | **0.0043%** | **0.0825%** |

("rows scored" is 130 of 132 because two rows have `ref_quad = 0` — the
reference underflowed to zero — and no relative difference is defined for
them. They are in the file, flagged `ref_resolved = 0`; 44 rows are in-band
in total, 42 of them scoreable.)

The scoping is not chosen for the number it produces; it is the manuscript's
own. Fig. `odt_validation` plots the surrogate "only where the conditioning
parameter *z* … lies inside the admissible band for the plotted order (shaded
region: *z*>2, where the truncation error becomes 𝒪(1))", and Section III-C
states that "beyond *z*=8 the surrogate is declared *inadmissible* rather than
merely inaccurate, and the candidate is rejected by the safety guard … instead
of scored on an untrustworthy value". The `admissible` column applies the same
predicate `admissible = (ladder admits z) and (K ≥ ladder K)` that block 01
already used, evaluated at each row's own SNR.

**What the out-of-band rows are actually measuring.** They are not evidence
that Eq. (22) fails at low SNR. The sweep picks **one** K per configuration,
from *z* at the *highest* SNR, then reuses it at every lower SNR; since
*z* ∝ 1/√γ̄, that K is up to ten times too small in *z* at 20 dB. The deployed
ladder never does this — it keys K to each candidate's own *z*.
`code/eq22_ladder_check.py` re-evaluates the same configurations at the
ladder-selected order and prints both columns side by side. At the
manuscript's own validation point (strong, ξ=1.967, σ_s=0.05):

| γ̄ | z | sweep K | rel (sweep K) | ladder K | rel (ladder K) |
|---|---|---|---|---|---|
| 20 dB | 1.44 | 5 | +131.06% | 10 | **+0.0019%** |
| 24 dB | 0.91 | 5 | +4.0111% | 10 | **+0.0006%** |
| 28 dB | 0.57 | 5 | +0.1442% | 10 | **+0.0004%** |
| 32 dB | 0.36 | 5 | +0.0058% | 5 | +0.0058% |

So the out-of-band population has three distinct causes, none of which is an
error in Eq. (22): the sweep's own K-selection (recoverable, as above);
genuine inadmissibility at *z* > 8, where the guard refuses the candidate and
no K helps; and, at 36–40 dB, the *reference* falling below the round-off
floor of the 16-fold FFT — `ref_resolved = 0` marks those, and there the
reference is the unreliable side, not the series.

**Two references, and why.** The previous release compared against
`egc_system.aber_system` alone, which convolves in the linear domain and
returns negative values below its floor; rows at 36 and 40 dB carried a
negative `exact_reference`, unflagged, and a "−100%" difference against a
negative reference means nothing. The file now carries `ref_quad` (the
manuscript's prescribed quadrature over the pointing law) and `ref_logdomain`
(a Mellin/log-domain construction of the same density), both on the corrected
cell-mass discretisation. `ref_spread_percent` is their disagreement — about
0.002–0.05% where the answer is resolved, jumping to 𝒪(100%) where it is not.
Both paths share `convolve_MN`, so the spread bounds the branch-density
construction rather than the transform; it detects the floor because below it
the two densities produce different noise.

**The bottom line, stated as the data supports it.** On the 39 rows where the
series is inside its admissible band and the reference is resolved, Eq. (22)
reproduces an independent 16-fold convolution to within **0.083%** worst case
and 0.0043% median — consistent with, and tighter than, the 0.17% the
manuscript reports at its single configuration. On the remaining 93 rows this
dataset does **not** establish agreement, and for the 88 out-of-band ones it
was never entitled to: those are candidates the guard is specified to reject.
Extending the validation *into* the out-of-band region would require the
ladder-correct order at each SNR, which `eq22_ladder_check.py` demonstrates
for four configurations but which this sweep does not do across the box.

**`data/06_system_aber/system_aber_curves.csv`**
Post-EGC system ABER on the corrected `system_metric.py` cell-mass machinery:
**1734 rows**, 3 regimes × 4 σ_s × 9 ξ nodes × 17 SNR points (16–48 dB). Each
row carries **two independent constructions** of the branch density —
`aber_system` (Mellin/log-domain) and `aber_system_quad` (the manuscript's own
prescribed quadrature over the pointing law) — their disagreement
`ref_spread_percent`, a **measured** round-off floor for each
(`floor_fast`, `floor_quad`), the derived `resolved` flag, and the
mass/mean self-checks `f_H_mass`, `E_H_numeric`, `E_H_analytic`.

**What the previous release shipped, measured against this one.** The old file
held 20 rows — one regime, one σ_s, two ξ — from an `egc_system.aber_system`
that sampled the branch density at h = 0 and patched the first node to
`h[0] = h[1]·1e-6`, mis-weighting the integrable divergence `f_h ~ h^(ξ²−1)`.
The README's caveat on it understated the damage on both counts it gave:

| | caveat said | measured |
|---|---|---|
| how far the shipped values move | 2–13% | **2.6–66.1%** on the 14 rows where both old and new are positive |
| rows carrying a **negative** "ABER" | two | **six** (38–48 dB), all in a column named `aber_system_exact` |
| `f_H` mass recovery, corrected code | 0.996–1.0002 | **0.999991–1.000012** across all 1734 rows |

It was also internally inconsistent on its own terms: at ξ = 0.992 the file's
own `f_H_mass` self-check reads **0.988319**, and a branch mass of 0.988319
raised to the 16th power by the convolution is 0.829 — the reported density is
missing **17%** of its probability — printed beside a column called
`aber_system_exact`. On the regenerated file `E_H_numeric` matches
`E_H_analytic` to 2.5e-5 relative, and there are **no** negative values.

⚠ **The remaining defect, and its measured size.** It is not in `aber_system`.
It is in `aber_system_quad`, the construction the *manuscript* prescribes, and
it is confined to **ξ < 1**.

`code/verify_block06.py` arbitrates both against direct Monte Carlo over the
16-branch sum — a third method that builds no density and convolves nothing —
at 8×10⁶ samples per row:

| selection | `aber_system` vs MC | `aber_system_quad` vs MC |
|---|---|---|
| 12 worst-spread rows above 1e-5 (MC rse ≈ 3.5%, so σ only) | worst **0.3 σ** | worst **12.5 σ**, low by up to **50.2%** |
| 12 rows above 2e-3, where MC *is* precise to <2% | within **0.038%** | low by up to **39.1%**, worst 79.4 σ |

Every arbitrated row is at ξ = 0.500. The cause is inherited and documented in
`egc_system.f_h_exact`: its fixed-order Gauss–Legendre rule under-resolves
`y^(ξ²−2)` near y = 0 when ξ < 1. At branch level that is 0.26%; through the
16-fold convolution it reaches 39–50%. Clear of the round-off floor (above
1e-14) the two constructions agree to **0.068%** for ξ ≥ 0.992 and 0.59% at
ξ = 0.789, and part company only at ξ = 0.500, where the exponent is
`y^(−1.75)` and the spread reaches 63.0%. **Use `aber_system`; read
`aber_system_quad` as the manuscript's construction shown failing, not as an
error bar.**

⚠ **The floor is now a column, and it is necessary but not sufficient.**
`system_aber(..., return_floor=True)` integrates the negative excursions of the
reconstructed `f_H` — which can only be FFT round-off — against the same Q
weight, giving noise in the same units as the answer, per row and per SNR
rather than as one module-wide constant. `aber_system` is clamped at 0, so an
exact `0.00000000e+00` means *under the floor*, never *zero*; 18 of 1734 rows
read that way and each states the floor it fell under. `resolved` is
`aber_system > 10·floor_fast`, a stated margin rather than a tuned one — both
raw columns ship so a reader can impose their own.

It does not catch everything, and the file shows where it fails: 1710 of 1734
rows are `resolved = 1`, yet a handful at ≥ 44 dB and ξ ≥ 1.548, where both
paths return 1e-20…1e-18, have the two constructions disagreeing by 60–377%.
`floor_*` counts only the *negative* half of the round-off; the positive half
enters the answer unseen, so the column is a lower bound on the noise, not a
bound on the error. **A row is trustworthy when `resolved = 1` *and*
`ref_spread_percent` is small.** The two-construction spread is the more
sensitive floor detector of the two, and it is in the file for that reason.

## Scripts that regenerate published quantities

Exactly one script in this package regenerates a published table from the
printed equations.

| Script | Regenerates | Needs |
|---|---|---|
| `code/admissibility_bounds.py` | **Table 7** in full, from Eqs. (16), (20), (26), (27) — no pre-computed tensor | nothing |

It prints each regenerated entry beside the tabulated one. Seven of the nine
Table 7 entries reproduce to within 7%; the two that differ more (K=10 weak,
K=20 moderate) are printed with their ratio rather than tuned to agree.

## Scripts that check published arithmetic — not reproductions

These two consume published numbers as inputs and verify that arithmetic
performed on them closes. They were previously called `reproduce_table9.py` and
`reproduce_table11.py` and were listed above under "reproduce published
tables", which they never did: a script that is handed Table 9's measured
columns and checks that the sums add up cannot tell you whether the
measurements are right. They have been renamed to say what they do.

| Script | Checks | Needs |
|---|---|---|
| `code/check_table9_arithmetic.py` | **Table 9's derived columns**: the rescaling of each baseline to `T_iter = 25`, the equal-budget comparison, the joint-rate arithmetic. Section 1 additionally *regenerates* the collected-power ratio behind the SNR-gain column from the beam geometry | nothing |
| `code/check_table11_statistics.py` | **Table 11's statistics**: exact two-sided McNemar p-values and Clopper–Pearson intervals, computed from paired success indicators | an `ablation_success.npz` of paired indicators — **not shipped**; the script now fails loudly without one |

Neither re-runs the optimizer. In particular `check_table9_arithmetic.py` takes
the 98% optimization-success rate as an *input* to sections 3 and 4; nothing in
it is evidence for that number. What is evidence about it — whether the 1e-6
post-EGC target is reachable at all across the swept box — is measured by
`system_metric.py` and by `run_campaign.py`'s feasibility ceiling, and is
summarised under "The optimization-success rate" below.

**No measured `ablation_success.npz` ships with this package.** The campaign
that produced Table 11 is not part of the release, so there is no measured
indicator file to publish.

`check_table11_statistics.py` used to fall back, silently and by default, to the
published (b, c) whenever no indicator file was present — and then print a
complete, internally consistent table, which in a long log reads exactly like a
reproduction while having consumed the published answer as its input. **A
missing file is now a hard error** (exit code 1) naming what to run instead. The
transcription check is still available, but only by asking for it in as many
words, and it prints a banner saying what it is:

```
python code/check_table11_statistics.py --published-arithmetic-only
```

To exercise the full path on real paired data, generate indicators from the
re-implemented solver:

```
python code/ablation_bc.py --realizations 1000 --out ablation_success.npz
python code/check_table11_statistics.py ablation_success.npz
```

Those indicators are the re-implementation's own and do **not** reproduce the
published counts. `ablation_bc.py` exists to answer a different question:
whether the c = 0 assumption behind the tabulated p-values survives on an
equivalent solver. On the runs done here it does not — `no_chaos` and `no_levy`
both produce c > 0, which makes `p = 2·2^(−b)` anti-conservative.

`check_table11_statistics.py` also refuses to run on an indicator file that is
missing an arm, rather than printing a partially populated table, and it
performs the `(b−c)/n` consistency check it previously only described.

`check_table9_arithmetic.py` recomputes the collected-power ratio behind the
SNR-gain column from the beam geometry. It does not check the tabulated 3.0 dB
against a ratio of 1.41, because 10^(3.0/20) = 1.4125 — that ratio *is* the
tabulated figure read backwards, and the round trip verifies nothing. What the
geometry supplies instead spans −13.6 to +10.4 dB across the swept jitter range,
so a single 3.0 dB figure is a point on that curve; which point is not
recoverable from the manuscript, and the script says so rather than picking one.

## Reference implementation of the closed loop

`code/` also contains a runnable assembly of the architecture as specified in
the manuscript. It is a **reference implementation**, not the campaign driver
that produced Tables 9–12, and the numbers it reports are its own.

| Module | Implements |
|---|---|
| `channel.py` | MNLT correlated gamma–gamma scintillation (Appendix A), Beckmann pointing loss, sway process, beam geometry — with a domain guard on eq. (3), see below |
| `rtodt_fast.py` | vectorised float64 interpolation-free kernel, eq. (21). `python code/rtodt_fast.py` runs the self-check against the arbitrary-precision `rtodt.py`: it measures rather than asserts, and reports a worst relative disagreement of **2.7e-10** over three regimes × three ladder rungs. That worst case sits at ξ ≈ 2.0007, beside the `a_k` pole at ξ² = β+1 = 4; away from the poles the rungs agree to 1e-15–1e-13. (An earlier version of this table claimed 7e-14 with nothing measuring it.) |
| `hclpso_ga.py` | the solver of Section V: logistic-map initialisation, Mantegna Lévy jumps, PSO core, GA elite crossover, monotone anytime incumbent, per-candidate fidelity ladder |
| `mpc_loop.py` | steady-state Kalman predictor, receding-horizon trajectory cost with slew coupling, envelope guard |
| `system_metric.py` | post-EGC **system** ABER, eq. `mimo_egc_aber`, as the 16-fold convolution of the branch density, plus `success()` — the manuscript's `P_e ≤ 1e-6 at γ̄_op = 38 dB` criterion |
| `run_campaign.py` | A/B driver: ablations × the two guard forms of Sec. VI-C, everything held fixed but one component. Reports the per-branch surrogate the solver ranks by **and** the system-level success rate of `system_metric.py`, a brute-force feasibility ceiling so a 0% rate can be read as "no such beam exists" or "the solver missed it", and the success rate under every scoping of the sweep |
| `eq22_ladder_check.py` | separates "Eq. (22) is inaccurate" from "block 05's sweep chose K at the wrong SNR", by re-evaluating the same configurations at the ladder-selected order against the same reference. Prints both columns; asserts nothing |
| `eq22_summary.py` | rebuilds `data/05_eq22_validation/summary.json`: the in-band, out-of-band, reference-resolved and unscoped statistics side by side, with the admissibility predicate and the caveat carried in the file |
| `verify_block04.py` | replays block 04's *old* error column (`Pe_series` at dps 200 vs dps 90, both cast to float64 — identically zero by construction) beside the real one (`pe_series_f64` vs the 200-digit reference), on the same shipped rows. Also checks `abs_err` against the eq. (27) floor recorded on each row, and against distance to the `a_k` poles. Asserts nothing |
| `verify_block06.py` | arbitrates block 06's two branch-density constructions against direct Monte Carlo over the 16-branch sum, which builds no density and convolves nothing. Reports deviations in units of the MC standard error, and refuses rows the arbiter cannot resolve |
| `build_manifest.py` | derives `MANIFEST.json` by enumerating `data/`, taking provenance from the per-block sidecars, then the campaign artefacts' own metadata, then whatever a block's artefacts declare about themselves. `--check` exits non-zero if any shipped block has no provenance, or is only partly accounted for |

**What it does not do.** It does not reproduce the ablation ordering of
Table 11. Several components make little difference in this implementation and
removing chaotic initialisation can even help. The likely reason is that the
manuscript does not specify the slew limit, the penalty weight `lambda_u`, the
decision box or the stage weighting, so this implementation had to choose them;
the ordering is sensitive to those choices. It is stated here rather than
tuned away.

**The guard A/B.** Section VI-C's envelope guard is three tests:

```
(i)   z(u) <= z_max          admissibility of the per-branch surrogate
(ii)  0 <= Pe(u) <= 1/2      the evaluation must be a probability
(iii) Pe(u) < eps_safe       connectivity threshold, eps_safe = 1e-3
```

The manuscript applies them "to the command about to be published, within the
safety-check stage", with test (i) additionally applied per candidate inside the
fitness evaluation, "where it is the same quantity that selects the order `K` on
the adaptive-fidelity ladder". Test (iii) is evaluated at system level — "the
success test, the guard threshold `eps_safe` and the fallback figure of
Numerical Result 1 are therefore all evaluated at system level". So the two
forms the manuscript contrasts are the **full** guard, (i)+(ii)+(iii), and the
**threshold-only** guard, (iii) alone, which is "the form the campaign ran
under". `run_campaign.py` now sweeps those two, at the command level, with the
`xi_safe` override on failure and both the published and the actuated beam
scored.

This replaces an earlier A/B that swept `(i)+(ii)` against `(ii)`, per candidate
inside the swarm loop. That comparison had two problems and the second is the
serious one.

* Neither arm was a configuration the manuscript describes — test (iii) was in
  neither.
* The two arms were **the same experiment**. `run_campaign.guard_test_overlap()`
  scans the manuscript's decision box (3 regimes × 4 σ_s × 4000 waists) and
  counts candidates rejected by test (i) that test (ii) would have admitted. The
  count is **0**. The reason is structural: the fidelity ladder returns order −1
  for `z > z_max` and the kernel returns NaN for order −1, so "z exceeds z_max"
  and "the fitness is not a number" are the same event. The identical numbers the
  old A/B produced in every cell were not a null result about guard design; they
  were the guard being compared against itself. The overlap count is printed on
  every run so that can never be mistaken again.

`run_campaign.py` also prints the manuscript's own `z_worst` per sway level (its
eq. for the widest admissible beam) beside the figures the manuscript prints —
1.12 / 4.52 / 17.97 / 40.43 at σ_s = 0.05 / 0.1 / 0.2 / 0.3 m, against 1.12 /
4.49 / 17.97 / 40.43 here — so its claim that "beyond σ_s ≈ 0.1 m the widest
beams are genuinely inadmissible and the two guard forms do differ" is testable
from the run log.

**What still cannot be reproduced.** The manuscript's headline guard experiment
is *two kernels × two guard forms*: the tabulate-and-interpolate kernel against
the interpolation-free one. Only the interpolation-free kernel is plumbed into
this MPC loop, so `run_campaign.py` has no kernel axis. That axis is exercised
separately, on a 1-D solver, by `campaign2.py`. Nothing in `run_campaign.py`
measures the interpolated kernel, and the JSON it writes says so.

**The domain guard on eq. (3).** `w_zeq(w_z)` is non-monotonic, with an interior
minimum at `w_z = 0.054869 m`, `w_zeq = 0.087719 m` — the same minimum the
manuscript computes in Sec. VII-A to set the box floor `ξ_min(σ_s) =
0.0877/(2σ_s)`. Only `w_z ≥ 0.054869 m` is a beam model. Below it the map
inverts: narrowing the beam *raises* `w_zeq`, hence raises ξ, while `A_0` rises
towards 1 — full collected power *and* unlimited immunity to pointing jitter, the
trade-off the whole optimization exists to resolve running backwards.

This was reaching `success()`. At σ_s = 0.10 m the narrow-branch waist
`w_z = 0.0286173 m` reported ξ = 1.0000 with `A_0 = 0.9961` and
`P_e,sys = 1.22e-14`, so `success()` returned **True**. The real beam at that
same ξ is `w_z = 0.1930499 m`, with `A_0 = 0.1252` and `P_e,sys = 1.42e-06`,
which **fails**. A success rate computed without the guard can be inflated by
configurations that do not exist.

`system_metric.beam_geometry` now raises `BeamGeometryDomainError`;
`channel.beam_geometry`, which the solver calls per candidate, returns NaN
instead, which propagates to a non-finite fitness and is rejected by the
envelope guard on the same path as any other unscoreable candidate. The boundary
is not a tuning knob: it is the manuscript's own minimum, and it coincides
exactly with the lower edge of the manuscript's box at σ_s = 0.05 m.
`code/test_beam_geometry.py` checks that every beam in the box at every swept
σ_s passes, so the guard excludes nothing the manuscript admits:

```
python code/test_beam_geometry.py     # 8 of 8 passed
```

## The optimization-success rate

The manuscript reports 98.0% optimization success. That figure is a campaign
measurement and the campaign driver is not in this release, so nothing here
reproduces it. What *is* here is the decision box, the channel model and the
success criterion, all of which are the manuscript's own — and from those a
**ceiling** can be computed without any solver at all: for each (regime, σ_s)
cell, brute-force scan the manuscript's ξ box and ask whether *any* beam in it
reaches `P_e,sys ≤ 1e-6` at `γ̄_op = 38 dB`. Where no beam does, no algorithm can
score a success in that cell, and the cell contributes 0 to any rate that
includes it.

`system_metric.py` computes that scan at perfect pointing (`r_d = 0`);
`run_campaign.py` recomputes it at the median residual offset each sway level
actually produces. Perfect pointing is the more generous of the two and no
controller can beat it.

**Reachability of the 1e-6 target, per cell, at perfect pointing** (best
`P_e,sys` attainable by any beam in the box):

| | σ_s = 0.05 m | 0.10 m | 0.20 m | 0.30 m |
|---|---|---|---|---|
| weak | ✓ ≤1e-16 | ✓ 8.0e-11 | ✗ 2.3e-04 | ✗ 1.3e-02 |
| moderate | ✓ 9.3e-16 | ✓ 3.8e-08 | ✗ 1.0e-03 | ✗ 2.2e-02 |
| strong | ✓ 6.3e-12 | ✗ 1.4e-06 | ✗ 2.8e-03 | ✗ 3.2e-02 |

At σ_s = 0.20 and 0.30 m no beam in any regime reaches the target. Under strong
turbulence the target is reachable only at σ_s = 0.05 m; the σ_s = 0.10 m
optimum misses by a factor of 1.42, and that margin is not a numerical artefact
— `system_metric.py` section [8b] arbitrates that single decision-critical point
against a 4·10⁷-sample Monte Carlo and finds `1.437e-06 ± 1.9e-07`, putting the
target 2.31σ below the estimate.

**Success rate under every scoping evaluated.** The rates below are the
reference implementation's, on its own protocol (one cold-started MPC cycle per
realization, 300 realizations per cell, σ_s drawn uniformly from the four swept
levels). The ceilings are not implementation-dependent: they are properties of
the manuscript's box and equations.

| Scoping | Ceiling | Measured (95% CI) | Does the manuscript scope this metric that way? |
|---|---|---|---|
| all regimes, all four σ_s | 33.3% | **32.9%** [29.8, 36.1] | **Yes** — this is the manuscript's own scoping |
| strong turbulence, all four σ_s | 25.0% | **25.3%** [20.5, 30.7] | **Yes** — Table 9's caption reads "Strong Turbulence" |
| all regimes, σ_s ≤ 0.1 m | 66.7% | 66.2% [61.6, 70.6] | **No**, not for this metric |
| strong turbulence, σ_s ≤ 0.1 m | 50.0% | 51.0% [42.7, 59.3] | **No**, not for this metric |
| strong turbulence, σ_s = 0.05 m only | 100.0% | 88.4% [79.7, 94.3] | **No** |

**On the σ_s ≲ 0.1 m envelope.** The obvious hypothesis is that 98% was measured
inside the jitter envelope to which the paper scopes many of its other claims,
in which case the printed *definition* would be wrong but the *result* sound.
That hypothesis has to be tested against the manuscript rather than against the
number it would produce, and the manuscript settles it explicitly. Section III-B
draws exactly this distinction and puts the success rate on the other side of
it:

> "The link-continuity observation of Section VII-B, stated within σ_s ≲ 0.1 m,
> is unaffected; **the optimization success rate, however, is swept across all
> four jitter levels**, so on these measurements up to ≈3.5% of cycles could
> have been scored on a corrupted evaluation."

The envelope is real and it does scope other claims — link continuity (Sec.
VII-B), the fallback bound of Numerical Result 1 — but the manuscript states in
as many words that it does not scope this one. Section VII-A independently says
"Building sway is swept over σ_s ∈ [0.05, 0.1, 0.2, 0.3] m".

The envelope scoping is therefore not available. It is tabulated above anyway,
because a scoping that is rejected should be rejected in public with its number
next to it — and because it does not rescue the figure in any case: the ceiling
under it is 66.7% across all regimes and 50.0% under the strong turbulence
Table 9 specifies. The only scoping that admits a ceiling near 98% is strong
turbulence at σ_s = 0.05 m alone, which no sentence in the manuscript licenses
and where the measured rate is 88.4% regardless.

**What this does and does not establish.** The measured rates are this
implementation's and could move with a better solver, a different protocol or a
different set of unspecified choices. The ceilings cannot: they are brute-force
scans of the manuscript's own decision box under its own channel model and its
own criterion, and they bound every rate in their row. Under the scoping the
manuscript applies to this metric, the ceiling is 33.3% at the campaign's
pointing residual and 41.7% at perfect pointing. 98.0% is above every ceiling in
the table except the one no sentence supports.

Reproduce with:

```
python code/system_metric.py                          # section [5]: the per-cell scan
python code/run_campaign.py --realizations 300        # ceilings, rates, scoping table
```

## Regenerating

```
python generate.py --smoke                       # ~4 minute sanity run -> data_smoke/
python generate.py --deadline "YYYY-MM-DD HH:MM" # full run -> data/
python generate.py --smoke --out /tmp/scratch    # anywhere else
python generate.py --only 06_system_aber         # rebuild one block
python code/build_manifest.py                    # re-derive MANIFEST.json
python code/build_manifest.py --check            # release gate: fails on a gap
```

`--only` rebuilds named blocks and leaves the rest alone, including their
`MANIFEST.json` entries — the driver now merges into the existing manifest
instead of rewriting it from scratch. Each block also drops a sidecar in
`logs/provenance_<block>.json` as it finishes, so two `--only` runs in
parallel cannot race each other's provenance, and `build_manifest.py`
assembles the manifest from those sidecars plus the campaign artefacts' own
embedded metadata.

A smoke run writes smoke-*scoped* blocks (one regime, one σ_s, one SNR point).
It used to write them into `data/`, so running the recommended sanity check
replaced the full multi-hour dataset with a handful of stubs — and rewrote
`MANIFEST.json` too, erasing the record that the blocks had ever been complete.
There is no undo. `--smoke` now defaults to `data_smoke/`, and overwriting the
release takes an explicit `--out .`.

The driver checkpoints after every item and records progress in
`MANIFEST.json`, so an interrupted run still leaves a usable dataset; blocks are
marked `complete` or `partial` accordingly.

## Notes on precision

The series is sign-alternating with roughly 26 decades of dynamic range between
its largest and smallest coefficients, so intermediate arithmetic is carried at
60–260 significant digits depending on the block. `validate_model.py` documents
where float64 is sufficient and where it is not.
