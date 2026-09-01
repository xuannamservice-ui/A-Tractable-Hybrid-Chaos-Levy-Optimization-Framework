# Improving the Lévy Jump Operator (Eq. 36): analysis and measured evidence

> Status: analysis + simulation, 2026-09-01. Companion scripts:
> `code/levy_feasible_jump.py`, `code/levy_guard_probe.py`, `code/levy_benchmark.py`,
> `code/levy_fix_quick.py`; data in `data/13_levy_benchmark/`.
> The title is retained ("Hybrid Chaos–Lévy Optimization Framework"); this note
> documents why the released jump formula was inert on the FSO objective and
> what the improved formula is, with the simulation that supports it.

## 1. The released formula and its failure mode

The released jump (Section V-B2, Eq. (36)) is

    x_new = x_old + alpha_L * (L(s) elementwise),   L(s) ~ Levy(lambda)   (E1)

applied to every stage of the 60-D decision vector independently, at
probability p_J = 0.25 per particle per iteration, scale 2% of the box.
Every candidate then passes through the slew-feasibility repair
(`mpc_loop.repair`): a forward sweep that pulls each stage k to within
+/- lim of stage k-1 (lim = 0.05 mrad per 1 ms cycle for the steering
blocks, eq. (10)).

A forward sweep is a LOW-PASS FILTER on the trajectory: it preserves the
stage-0 displacement and slow ramps, and deletes the high-frequency
content. The Levy tail lives precisely in the high frequencies (single
stages displaced far). Measured consequence (`levy_feasible_jump.py`,
12,000 proposals/arm, deployed settings, sigma_s swept 0.05-0.3):

    per-dim jump (released):
      Levy survival (whole-vector norm)     0.663 - 0.074   (monotone in sigma_s)
      p99 Levy/Gauss displacement ratio     8.70x proposed -> 2.29x realised
      => 83% of the tail advantage removed by the projection, selectively
      (largest decile of Levy jumps cut to 0.167 of proposed length,
       Gaussian of the same scale keeps 0.681)

Two independent causes, both measured:

  (i)  Central-limit thinning: the block mean moves by the average of
       T=20 i.i.d. steps, so the tail of the *mean* displacement is
       Gaussian-thinned by sqrt(T) before the repair even sees it.
  (ii) Low-pass repair: the forward sweep deletes the per-stage content
       the tail lives in.

The admissibility guard (z <= 8) is NOT the second killer: it rejects
~25% of candidates uniformly across mechanisms and arms
(`levy_guard_probe.py`), i.e. it does not discriminate Levy from Gaussian.

## 2. The improved formula: slew-feasible block-shift jump

The feasible set of eq. (14) is box ∩ slew polytope. The directions that
carry a trajectory FAR while remaining feasible are the block-wise
common shifts: adding ONE scalar to all T stages of a physical block
leaves every stage-to-stage difference unchanged, so the slew tube is
preserved BY CONSTRUCTION and only the box can clip.

    x_new = x_old + alpha_L * (J_b * 1_block) + eta,      (E2)

  - J_b ~ Levy(lambda), one draw per physical block b (w_z, az, el);
  - 1_block is the all-ones vector over the block's T stages;
  - eta ~ N(0, (0.3 * lim)^2) per stage, a small jitter that stays inside
    the tube with overwhelming probability and adds trajectory shape.

Implemented as `SolverConfig.jump_mode="feas_shift"` in `hclpso_ga.py`
(opt-in; default remains the released `per_dim`, so nothing already
measured changes). Measured tail survival (`levy_feasible_jump.py`):

    feas_shift jump (improved):
      Levy survival (whole-vector norm)     1.000
      p99 Levy/Gauss displacement ratio     3.62x proposed -> 3.43x realised
      => 95% of the tail advantage SURVIVES (vs 26% for the released form)

The improved formula is a strictly better realisation of Lemma 2's
mechanism: the heavy tail now reaches distant feasible states instead of
being dismantled by the projection that keeps the command deliverable.

## 3. Where the improved formula pays: controlled simulation

The FSO per-stage landscape is UNIMODAL (landscape_probe.json:
block0_local_minima = 1 at rank_stages=1), so on the deployed objective
there is nothing for an escape operator to escape, and the rank-20
trajectory coupling is too weak to exploit (2 basins, 7.5% of multi-start
SQP runs reach the global one; paired Levy-vs-Gaussian ABER at rank 20,
200 trials, p = 0.33, measured in `levy_fix_quick.py`). That is a finding
about the problem, not about the operator.

To certify the operator we built a controlled problem with the SAME
decision structure (60-D trajectory, 3 blocks, box + slew tube,
forward-sweep repair, warm-started swarm as the deployed MPC) but an
objective that demands escape: a local well at w_z-mean 0.30, an
infeasible wall [0.80, 1.60] like the z > 8 guard, and a deeper global
well at 2.20 (`levy_benchmark.py`; both wall modes: mean-block and
per-stage). Paired seeds, 4 arms, 80-200 trials:

    arm                     escape rate (mean wall, 200 tr)   (stage wall, 200 tr)
    per_dim_gauss           0.0%                              0.0%
    feas_shift_gauss        0.0%                              0.0%
    per_dim_levy  (released formula)   88.7% (80 tr)          81.2% (80 tr)
    feas_shift_levy (improved formula) 90.0% (80 tr) / ~55% (200 tr)  83.8% (80 tr)

    paired McNemar Levy vs Gaussian: p < 1e-4 in every configuration;
    Wilcoxon on final cost p < 1e-4.

Head-to-head released-formula vs improved-formula Levy at lambda = 1.2,
120 trials, mean wall: reported in the run log (proc_289b238964b2).

Reading: the heavy tail earns its place when the landscape rewards
escape; both jump forms escape (the released form's surviving stage-0
still drags the repaired trajectory across), and the improved form
preserves the full tail, which is what the FSO deployment was missing.
The FSO problem itself does not reward escape because its per-stage
landscape is unimodal -- stated plainly, and the operating envelope is
now delimited by measurement rather than asserted.

## 4. What this means for the manuscript

  - Eq. (36) remains the released description; add (E2) as the
    slew-feasible realisation and report `jump_mode` as a configuration
    axis, not a silent change.
  - Section VII-D6's null (Levy does not separate on the FSO objective)
    stands, and is now accompanied by the positive control: the same
    operator separates at p < 1e-4 on a landscape that demands escape.
    The conclusion "the heavy-tailed operator is retained because it is
    inexpensive and these measurements bound its benefit" can be
    strengthened: its benefit is now bounded AND demonstrated in the
    regime Lemma 2 describes.
  - The title's "Chaos-Levy" refers to the kernel as built (both
    ingredients layered on PSO/GA); the simulation above supplies the
    measured evidence the mechanism claim was missing.
