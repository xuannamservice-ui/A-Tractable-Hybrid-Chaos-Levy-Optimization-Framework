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
| `w_zeq` minimum | 0.0877 m at `w_z`=0.0549 | ✓ |
| `A_0` at ξ=0.992, σ_s = 0.05/0.1/0.2/0.3 | 0.533 / 0.127 / 0.0318 / 0.0141 | ✓ |
| `max_k|a_k|` at ξ=0.500 and 4.888 (weak, K=10) | 5.9e5 and 8.2e31 | ✓ |
| float64 round-off floor `η_f64` | 1.3e-10 and 1.8e16 | ✓ |

If those reproduce, the rest of the package is built on a faithful model.

## Layout

```
code/     the 13 scripts that produce every number below
data/     generated datasets (CSV + NPZ), see per-block notes
logs/     run log with timestamps
MANIFEST.json   what ran, how long, complete or partial
generate.py     the driver
```

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
Randomly sampled off-grid ξ inside the admissible band, comparing the deployed
float64 evaluation against a 200-digit reference. Supports the accuracy claims
of Section III-B. Open-ended: the longer the run, the more samples.

**`data/05_eq22_validation/eq22_vs_reference.csv`**
The λ_j / C_j convolved series of Eq. (22) evaluated against an independent
16-fold convolution reference, across regimes, jitter levels, beam
configurations and SNR. This is the dataset that extends the single-point
validation of Section III-D across the parameter box.

**`data/06_system_aber/system_aber_curves.csv`**
Exact post-EGC ABER curves obtained by quadrature over the pointing law
followed by a 16-fold FFT convolution, with the recovered mass of `f_H`
reported alongside each point as a self-check.

## Scripts that reproduce published tables

| Script | Reproduces | Needs |
|---|---|---|
| `code/admissibility_bounds.py` | **Table 7** in full, from Eqs. (16), (20), (26), (27) — no pre-computed tensor | nothing |
| `code/reproduce_table11.py` | **Table 11**: the discordant counts (b, c), exact two-sided McNemar p-values, Clopper–Pearson intervals | `ablation_success.npz` (falls back to the published counts) |
| `code/reproduce_table9.py` | **Table 9**: the *derived* columns — SNR gain from the collected-power ratio, the rescaling of each baseline to `T_iter = 25`, the joint-rate arithmetic | nothing |

Each script prints its regenerated value beside the published one so the two can
be compared directly. `admissibility_bounds.py` reproduces seven of the nine
Table 7 entries to within 7%; the two that differ more (K=10 weak, K=20
moderate) are printed with their ratio rather than tuned to agree.

Neither `reproduce_table9.py` nor `reproduce_table11.py` re-runs the optimizer.
The per-cycle success indicators and median latencies are campaign outputs; what
these scripts audit is the arithmetic built on top of them, which is the part a
reader can check independently.

## Regenerating

```
python generate.py --smoke                       # ~4 minute sanity run
python generate.py --deadline "YYYY-MM-DD HH:MM" # full run to a wall-clock deadline
```

The driver checkpoints after every item and records progress in
`MANIFEST.json`, so an interrupted run still leaves a usable dataset; blocks are
marked `complete` or `partial` accordingly.

## Notes on precision

The series is sign-alternating with roughly 26 decades of dynamic range between
its largest and smallest coefficients, so intermediate arithmetic is carried at
60–260 significant digits depending on the block. `validate_model.py` documents
where float64 is sufficient and where it is not.
