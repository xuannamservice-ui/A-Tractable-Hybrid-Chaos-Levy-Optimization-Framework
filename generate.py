"""
Artefact generator for the Data and Code Availability package.

Runs unattended until a wall-clock deadline, working through priority-ordered
blocks and CHECKPOINTING after every item, so an interrupted run still leaves a
complete, self-describing dataset.

Blocks, in the order they are attempted:

  01 admissibility     truncation + round-off behaviour over the full
                       (regime, sigma_s, xi, SNR, K) grid            -- Table 7
  02 z_map             conditioning parameter, A_0, ladder order,
                       admissible flag on the same grid              -- z_map.npz
  03 coefficient_tensors  a_k and D over the pole-free node grid     -- lookup tensors
  04 offgrid_error     deployed interpolation vs interpolation-free
                       vs high-precision reference, random off-grid  -- Sec. III-B
  05 eq22_validation   the lambda_j/C_j convolved series against an
                       independent 16-fold reference, across the
                       parameter box                    -- closes future work (ii)
  06 system_aber       exact post-EGC ABER curves for reference use

Blocks 04-06 are open-ended: they keep deepening (more samples, more
configurations) until the deadline, so a longer run simply yields more.

Usage:  python generate.py [--deadline "YYYY-MM-DD HH:MM"] [--smoke]
"""
import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
sys.path.insert(0, CODE)

from rtodt import (NODES, REGIMES, SIGMAS, A0_for, a_k, D_coef, C_moment,
                   Pe_series, z_param, db, APERTURE)                 # noqa: E402
from rtodt_fast import pe_series_f64                                 # noqa: E402
# MN only: blocks 05 and 06 no longer call egc_system.aber_system directly.
# They go through system_metric, which uses the corrected cell-mass
# discretisation and can report two independent branch densities.
from egc_system import MN                                            # noqa: E402
import system_metric as sm                                           # noqa: E402

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")
SNR_GRID = list(range(20, 51, 2))          # the 16-point grid of Table 7
LADDER = ((0.5, 5), (2.0, 10), (8.0, 20))  # z threshold -> K
EPS64 = float(mp.mpf(2) ** -52)

_log_fh = None


def log(msg):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (stamp, msg)
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def ladder_K(z):
    for zt, K in LADDER:
        if z <= zt:
            return K
    return None                              # inadmissible


def ensure(*parts):
    p = os.path.join(*parts)
    os.makedirs(p, exist_ok=True)
    return p


def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


# ----------------------------------------------------------------- block 01
def block_admissibility(out, deadline, smoke):
    mp.mp.dps = 90
    rows = []
    path = os.path.join(out, "admissibility_grid.csv")
    hdr = ["regime", "sigma_s", "xi", "snr_db", "K", "A0", "z", "admissible",
           "max_abs_ak", "eta_f64", "first_omitted", "Pe_series"]
    regs = list(REGIMES.items())[:1] if smoke else list(REGIMES.items())
    sig = SIGMAS[1:2] if smoke else SIGMAS
    nodes = NODES[3:5] if smoke else NODES
    snrs = SNR_GRID[:2] if smoke else SNR_GRID
    for rname, (A, B) in regs:
        for s in sig:
            for xi in nodes:
                A0 = A0_for(xi, s)
                if A0 is None:
                    continue
                for gdb in snrs:
                    if time.time() > deadline:
                        save_csv(path, hdr, rows)
                        return len(rows), False
                    g = db(gdb)
                    z = float(z_param(A, B, A0, g))
                    Kl = ladder_K(z)
                    for K in (5, 10, 20):
                        try:
                            terms = []
                            for k in range(K + 2):
                                terms.append(abs(a_k(A, B, xi, A0, k) * C_moment(B + k, g)))
                                terms.append(abs(a_k(B, A, xi, A0, k) * C_moment(A + k, g)))
                            mx = float(max(terms))
                            first_om = float(max(terms[-2], terms[-1]))
                            pe = float(Pe_series(A, B, xi, A0, g, K))
                            rows.append([rname, float(s), float(xi), gdb, K,
                                         "%.6e" % float(A0), "%.4f" % z,
                                         int(Kl is not None and K >= Kl),
                                         "%.6e" % mx, "%.6e" % (mx * EPS64),
                                         "%.6e" % first_om, "%.6e" % pe])
                        except Exception as e:
                            log("  admissibility item failed: %s" % e)
    save_csv(path, hdr, rows)
    return len(rows), True


# ----------------------------------------------------------------- block 02
def block_zmap(out, deadline, smoke):
    mp.mp.dps = 60
    recs = []
    regs = list(REGIMES.items())[:1] if smoke else list(REGIMES.items())
    for rname, (A, B) in regs:
        for s in SIGMAS:
            for xi in NODES:
                A0 = A0_for(xi, s)
                if A0 is None:
                    continue
                for gdb in SNR_GRID:
                    z = float(z_param(A, B, A0, db(gdb)))
                    K = ladder_K(z)
                    recs.append((rname, float(s), float(xi), gdb,
                                 float(A0), z, K if K else -1, int(K is not None)))
    arr = np.array([(r[3], r[4], r[5], r[6], r[7]) for r in recs], dtype=float)
    np.savez_compressed(os.path.join(out, "z_map.npz"),
                        snr_db=arr[:, 0], A0=arr[:, 1], z=arr[:, 2],
                        ladder_K=arr[:, 3], admissible=arr[:, 4],
                        regime=np.array([r[0] for r in recs]),
                        sigma_s=np.array([r[1] for r in recs]),
                        xi=np.array([r[2] for r in recs]))
    save_csv(os.path.join(out, "z_map.csv"),
             ["regime", "sigma_s", "xi", "snr_db", "A0", "z", "ladder_K", "admissible"],
             [[r[0], r[1], r[2], r[3], "%.6e" % r[4], "%.4f" % r[5], r[6], r[7]] for r in recs])
    return len(recs), True


# ----------------------------------------------------------------- block 03
def block_tensors(out, deadline, smoke):
    mp.mp.dps = 120
    Kmax = 6 if smoke else 20
    regs = list(REGIMES.items())[:1] if smoke else list(REGIMES.items())
    n = 0
    for rname, (A, B) in regs:
        ak1 = np.zeros((len(SIGMAS), len(NODES), Kmax + 1))
        ak2 = np.zeros_like(ak1)
        Dv = np.zeros((len(SIGMAS), len(NODES)))
        A0v = np.zeros_like(Dv)
        for si, s in enumerate(SIGMAS):
            for xj, xi in enumerate(NODES):
                if time.time() > deadline:
                    return n, False
                A0 = A0_for(xi, s)
                if A0 is None:
                    continue
                A0v[si, xj] = float(A0)
                Dv[si, xj] = float(D_coef(A, B, xi, A0))
                for k in range(Kmax + 1):
                    ak1[si, xj, k] = float(a_k(A, B, xi, A0, k))
                    ak2[si, xj, k] = float(a_k(B, A, xi, A0, k))
                    n += 2
        np.savez_compressed(
            os.path.join(out, "lookup_tensor_%s.npz" % rname),
            a_k_alpha_beta=ak1, a_k_beta_alpha=ak2, D=Dv, A0=A0v,
            sigma_s=np.array([float(x) for x in SIGMAS]),
            xi_nodes=np.array([float(x) for x in NODES]),
            alpha=float(A), beta=float(B), K_max=Kmax)
    return n, True


# ----------------------------------------------------------------- block 04
def block_offgrid(out, deadline, smoke):
    """Deployed float64 kernel vs arbitrary-precision reference, off-grid.

    WHAT THIS COMPARES, AND WHY IT IS TWO DIFFERENT IMPLEMENTATIONS.
    The point of this block is to measure the error the DEPLOYED evaluator
    actually commits at off-grid xi.  The deployed evaluator is
    `rtodt_fast.pe_series_f64` -- the vectorised, interpolation-free float64
    kernel of eq. (21), the one the manuscript reports results on.  The
    reference is `rtodt.Pe_series`, the same expressions carried in mpmath at
    200 digits.  Those are two independent arithmetic paths, so the difference
    between them is a measurement.

    An earlier version of this block called `rtodt.Pe_series` for BOTH sides --
    once at dps 200 and once at dps 90 -- and then cast both to float64 before
    subtracting.  Since float64 carries ~16 digits, rounding a 200-digit and a
    90-digit value of the same quantity to double gives the identical double,
    so `abs_err_interp_free` was exactly 0.000e+00 on every one of the 560250
    rows it produced.  It exercised no float64 kernel and no interpolation
    despite being advertised as doing exactly that.  The column measured the
    round-off of mpmath against itself, which is not a property of anything
    that ships.

    Columns.  `eta_f64` is the round-off floor of eq. (27), max_k|a_k C| times
    eps_mach, computed at the sampled xi: it is the error the float64 kernel is
    PREDICTED to commit, so `abs_err_interp_free` can be read against its own
    bound rather than against nothing.  `ref_in_range` / `f64_in_range` record
    test (ii) of the guard, 0 <= Pe <= 1/2, separately for the two paths --
    the reference can be a probability while the deployed value is not, and
    that gap is the thing the range test exists to catch.
    """
    rng = random.Random(20260826)
    path = os.path.join(out, "offgrid_error.csv")
    hdr = ["regime", "sigma_s", "snr_db", "xi", "z", "K", "A0",
           "Pe_reference", "Pe_interp_free", "abs_err_interp_free",
           "rel_err_interp_free", "eta_f64", "ref_in_range", "f64_in_range"]
    rows = []
    target = 40 if smoke else 10 ** 9
    combos = [(rn, AB, s, g) for rn, AB in REGIMES.items()
              for s in SIGMAS for g in (30, 34, 38, 40, 44)]
    i = 0
    # This block is open-ended BY DESIGN: it samples until the deadline, so on
    # a full run it always stops early and its provenance status is `partial`,
    # never `complete`.  It used to return `True` unconditionally, which put
    # `"status": "complete"` in MANIFEST.json for a file that had simply run
    # out of clock -- the one status a reader would use to decide the sample
    # count was the intended one.  `hit_target` is now what is reported.
    hit_target = False
    while True:
        if len(rows) >= target:
            hit_target = True
            break
        if time.time() > deadline:
            break
        rn, (A, B), s, gdb = combos[i % len(combos)]
        i += 1
        mp.mp.dps = 90
        xi = NODES[0] + (NODES[-1] - NODES[0]) * mp.mpf(rng.random())
        A0 = A0_for(xi, s)
        if A0 is None:
            continue
        g = db(gdb)
        z = float(z_param(A, B, A0, g))
        K = ladder_K(z)
        if K is None:
            continue
        try:
            # --- reference: arbitrary precision, 200 digits
            mp.mp.dps = 200
            ref = float(Pe_series(A, B, xi, A0, g, K))
            # --- deployed: vectorised float64 kernel, eq. (21)
            fast = float(pe_series_f64(float(A), float(B), float(xi),
                                       float(A0), float(g), K)[0])
            # --- predicted float64 floor, eq. (27), at this xi
            mp.mp.dps = 90
            terms = []
            for k in range(K + 1):
                terms.append(abs(a_k(A, B, xi, A0, k) * C_moment(B + k, g)))
                terms.append(abs(a_k(B, A, xi, A0, k) * C_moment(A + k, g)))
            eta = float(max(terms)) * EPS64
        except Exception:
            continue
        if not np.isfinite(fast):
            # the float64 kernel overflowed or hit a Gamma pole; that IS the
            # deployed behaviour at this xi, so it is recorded, not skipped
            err = float("inf")
            rel = float("inf")
        else:
            err = abs(fast - ref)
            rel = err / abs(ref) if ref else float("nan")
        # xi and A_0 are written at full float64 precision (%.17g), not
        # rounded for display: a reader must be able to recompute
        # abs_err_interp_free from the row itself.  At 6 significant figures
        # the stored xi differs from the sampled one by ~1e-7, which near an
        # a_k pole moves the error by tens of percent -- the row would then
        # not reproduce from its own columns.
        rows.append([rn, float(s), gdb, "%.17g" % float(xi), "%.6f" % z, K,
                     "%.17g" % float(A0), "%.17e" % ref, "%.17e" % fast,
                     "%.6e" % err, "%.6e" % rel, "%.6e" % eta,
                     int(0.0 <= ref <= 0.5), int(0.0 <= fast <= 0.5)])
        if len(rows) % 250 == 0:
            save_csv(path, hdr, rows)
            log("  offgrid: %d samples" % len(rows))
    save_csv(path, hdr, rows)
    return len(rows), hit_target


# ----------------------------------------------------------------- block 05
def eq22_series(A, B, xi, A0, K, snrs):
    """The lambda_j/C_j convolved series, lattice-collapsed."""
    x2 = xi ** 2
    uD = D_coef(A, B, xi, A0) * mp.gamma(x2)
    Bp = [a_k(A, B, xi, A0, k) * mp.gamma(B + k) for k in range(K + 1)]
    Ap = [a_k(B, A, xi, A0, k) * mp.gamma(A + k) for k in range(K + 1)]
    fact = [mp.factorial(i) for i in range(MN + 1)]
    cB, cA = {0: [mp.mpf(1)]}, {0: [mp.mpf(1)]}

    def ppow(coeffs, n, cache):
        if n in cache:
            return cache[n]
        prev = ppow(coeffs, n - 1, cache)
        r = [mp.mpf(0)] * (len(prev) + len(coeffs) - 1)
        for i, a in enumerate(prev):
            if a:
                for j, b in enumerate(coeffs):
                    if b:
                        r[i + j] += a * b
        cache[n] = r
        return r

    acc = {}
    for nD in range(MN + 1):
        for nB in range(MN + 1 - nD):
            nA = MN - nD - nB
            base = fact[MN] / (fact[nD] * fact[nB] * fact[nA]) * (uD ** nD if nD else 1)
            pB, pA = ppow(Bp, nB, cB), ppow(Ap, nA, cA)
            lam0 = nD * x2 + nB * B + nA * A
            for i, bi in enumerate(pB):
                if not bi:
                    continue
                bb = base * bi
                for j, aj in enumerate(pA):
                    if not aj:
                        continue
                    lam = lam0 + i + j
                    key = mp.nstr(lam, 22)
                    if key in acc:
                        acc[key] = (acc[key][0], acc[key][1] + bb * aj)
                    else:
                        acc[key] = (lam, bb * aj)
    out = {}
    for gdb in snrs:
        gb = db(gdb) / MN
        out[gdb] = sum((Bj / mp.gamma(lam)) * C_moment(lam, gb) for lam, Bj in acc.values())
    return out, len(acc)


ZMAX_FOR_K = {K: zt for zt, K in LADDER}      # 5 -> 0.5, 10 -> 2.0, 20 -> 8.0


def block_eq22(out, deadline, smoke):
    """Eq. (22) across the parameter box, WITH the band it is claimed on.

    The shipped version of this sweep had no `z` column, no admissibility
    column and no caveat, so a reader met a table whose median relative
    difference was of order 100% with nothing to tell them that most of those
    rows are outside the band in which the manuscript claims Eq. (22) holds at
    all.  Two separate things were being conflated:

    (1) THE SERIES SIDE.  One truncation order K is chosen per configuration,
        from the conditioning parameter z at the HIGHEST SNR of the sweep.
        Since z = sqrt(2) alpha beta / (A_0 sqrt(gbar)) scales as 1/sqrt(gbar),
        that same K is applied at SNRs where z is up to ten times larger and
        the truncation is no longer admissible.  Fig. odt_validation plots the
        surrogate ONLY inside the band ("shaded region: z>2, where the
        truncation error becomes O(1)"), and Sec. III-C declares candidates
        beyond z=8 "inadmissible rather than merely inaccurate", rejected by
        the guard "instead of scored on an untrustworthy value".  The
        `admissible` column applies exactly that test, row by row, using the
        same predicate as block 01: the ladder must admit z at THIS row's SNR,
        and the K actually used must be at least the order the ladder asks for.

    (2) THE REFERENCE SIDE.  The old reference was `egc_system.aber_system`
        verbatim, a linear-domain FFT convolution that carries an absolute
        round-off floor and returns NEGATIVE values below it.  In the shipped
        file the reference itself goes negative at 36 and 40 dB, so a
        "rel_diff_percent" of -100% there is the reference failing, not
        Eq. (22).  Two reference constructions are now carried, differing in
        how the branch density is built: `ref_quad`, the manuscript's
        prescribed quadrature over the pointing law (egc_system.f_h_exact),
        and `ref_logdomain`, a Mellin/log-domain construction that never
        evaluates the density at h = 0.  Both are placed on the corrected
        cell-mass discretisation, and both then use system_metric's shared
        MN-fold `convolve_MN`, so they differ ONLY in f_h.

        `ref_spread_percent` is their disagreement.  Note what it does and
        does not measure: because the two paths share the MN-fold FFT they
        also share its absolute round-off floor, so the spread is NOT an
        independent check on that floor.  What makes it a usable floor
        detector anyway is that at and below the floor the returned value is
        FFT noise, and two slightly different branch densities produce
        different noise, so the two paths diverge from each other -- which is
        exactly what is observed at 36-40 dB.  Above the floor they agree to
        ~1e-4.  `ref_resolved` therefore reads: the two constructions agree to
        better than 1%, so the reference is above the noise of the transform
        both of them use.  Where they do not agree, neither reference value is
        meaningful and no statement about Eq. (22) is supported at that row;
        `comparison_valid` is 0 there.

    Neither column is a filter applied to the data: every row is written, with
    its flags, so in-band and out-of-band statistics can both be computed.
    """
    mp.mp.dps = 260
    path = os.path.join(out, "eq22_vs_reference.csv")
    hdr = ["regime", "sigma_s", "xi", "K", "n_exponents", "snr_db",
           "z", "z_max_for_K", "ladder_K_at_snr", "admissible",
           "eq22", "ref_quad", "ref_logdomain", "ref_spread_percent",
           "ref_resolved", "comparison_valid", "rel_diff_percent"]
    rows = []
    xis = ["1.967"] if smoke else ["1.548", "1.967", "2.511", "3.104"]
    sigs = ["0.05"] if smoke else ["0.05", "0.1"]
    snrs = [30] if smoke else [20, 24, 28, 32, 36, 40]
    regs = list(REGIMES.items())[:1] if smoke else list(REGIMES.items())
    for rname, (A, B) in regs:
        for ss in sigs:
            for xs in xis:
                if time.time() > deadline:
                    save_csv(path, hdr, rows)
                    return len(rows), False
                xi, s = mp.mpf(xs), mp.mpf(ss)
                A0 = A0_for(xi, s)
                if A0 is None:
                    continue
                K = ladder_K(float(z_param(A, B, A0, db(max(snrs)))))
                if K is None:
                    continue
                try:
                    log("  eq22: %s sigma=%s xi=%s K=%d" % (rname, ss, xs, K))
                    ser, nexp = eq22_series(A, B, xi, A0, K, snrs)
                    for gdb in snrs:
                        if time.time() > deadline:
                            save_csv(path, hdr, rows)
                            return len(rows), False
                        gbar = float(db(gdb))
                        z = float(z_param(A, B, A0, db(gdb)))
                        Kl = ladder_K(z)
                        adm = int(Kl is not None and K >= Kl)

                        rq = sm.system_aber(float(A), float(B), float(xi),
                                            float(A0), gbar, method="quad")
                        rf = sm.system_aber(float(A), float(B), float(xi),
                                            float(A0), gbar, method="fast")
                        spread = (abs(rq - rf) / abs(rq) * 100.0
                                  if rq else float("nan"))
                        # the two independent reference constructions must
                        # agree to better than 1% for the row to certify
                        # anything about eq. (22)
                        resolved = int(rq > 0.0 and np.isfinite(spread)
                                       and spread < 1.0)
                        v = float(ser[gdb])
                        rel = (v - rq) / rq * 100 if rq else float("nan")
                        rows.append([rname, ss, xs, K, nexp, gdb,
                                     "%.4f" % z, ZMAX_FOR_K[K],
                                     Kl if Kl is not None else -1, adm,
                                     "%.8e" % v, "%.8e" % rq, "%.8e" % rf,
                                     "%.6f" % spread, resolved,
                                     int(adm and resolved), "%+.4f" % rel])
                        save_csv(path, hdr, rows)
                except Exception as e:
                    log("  eq22 config failed: %s" % e)
    save_csv(path, hdr, rows)
    return len(rows), True


# ----------------------------------------------------------------- block 06
def block_system_aber(out, deadline, smoke):
    """Post-EGC system ABER curves, on the corrected system_metric machinery.

    The previous release of this block called `egc_system.aber_system` as it
    then stood: the branch density was POINT-SAMPLED on a lattice that included
    h = 0 and the singular first sample was patched to h[0] = h[1]*1e-6.  For
    xi < 1 the branch density diverges integrably as h^(xi^2-1), so that patch
    assigns the first cell an essentially arbitrary mass; the recovered branch
    mass is then raised to the 16th power by the MN-fold convolution.  It also
    convolved in the linear domain, which puts an absolute round-off floor
    under the answer and lets it come out NEGATIVE in a column named
    `aber_system_exact`.

    This block now evaluates through `system_metric.system_aber`, which
    represents the branch density by exact CELL MASSES (no h = 0 sample, no
    arbitrary offset) and carries two independent constructions of it:

      `aber_system`      method='fast': Mellin/log-domain branch density.
                         ln h = ln h_a + ln h_p is an ordinary additive
                         convolution, so the density is never evaluated at
                         h = 0 and no arbitrary offset is needed.
      `aber_system_quad` method='quad': the manuscript's own prescribed
                         construction -- quadrature over the pointing law via
                         egc_system.f_h_exact -- on the same corrected
                         discretisation.

    `ref_spread_percent` is their relative disagreement.  READ IT AS A FLAG ON
    `aber_system_quad`, NOT AS AN ERROR BAR ON `aber_system`.  Where the two
    disagree, the Monte Carlo arbiter -- `system_metric.system_aber(...,
    method='mc')`, which builds no density at all and samples the 16-branch
    sum directly -- says which one is wrong, and on every configuration tested
    it is the quadrature path.  Measured on THIS file, 8e6 MC samples per row
    (`code/verify_block06.py --samples 8000000`):

      twelve worst-spread rows above 1e-5, where the arbiter's relative
      standard error is ~3.5% so only a sigma statement is meaningful:
        regime    sigma_s  xi     gbar    fast/MC   quad/MC
        weak      0.1      0.500  36 dB    1.0006    0.4981
        weak      0.2      0.500  48 dB    1.0002    0.5022
        strong    0.1      0.500  42 dB    1.0096    0.5160
        moderate  0.1      0.500  38 dB    0.9835    0.5048
      -> `aber_system` worst 0.3 sigma from the arbiter over all twelve;
         `aber_system_quad` worst 12.5 sigma, low by up to 50.2%.

      twelve rows above 2e-3, where the arbiter IS precise to better than 2%
      and a percentage is therefore meaningful (`--min-aber 2e-3`):
      -> `aber_system` within 0.038% of the arbiter, worst 0.1 sigma;
         `aber_system_quad` low by up to 39.1%, worst 79.4 sigma.

    The cause is inherited and already documented in egc_system.f_h_exact: its
    fixed-order Gauss-Legendre rule under-resolves y^(xi^2-2) near y = 0 when
    xi < 1.  At branch level that was measured at 0.26%; through the 16-fold
    convolution it reaches 39-50% at the system level, and the effect is
    confined to xi < 1: on this file the two constructions agree to 0.068%
    worst case for xi >= 0.992 above 1e-14, to 0.59% at xi = 0.789, and part
    company only at xi = 0.500, where the exponent y^(xi^2-2) = y^(-1.75) is
    most singular and the spread reaches 63.0%.  `code/verify_block06.py`
    reproduces this arbitration against the shipped file.

    Two further things the spread does NOT measure.  Both paths hand their
    cell masses to the same `convolve_MN`, so they share its absolute FFT
    round-off floor and the spread cannot detect it independently -- though in
    practice it does, because below the floor the value is noise and two
    slightly different densities give different noise, so the spread jumps
    from ~1e-3% to O(100%).  And below the floor `system_aber` clamps to zero
    rather than returning the negative noise the old routine reported, so a
    row reading exactly 0.00000000e+00 means "unresolvable here", not "zero".

    THE FLOOR IS NOW A COLUMN, NOT A FOOTNOTE.  The previous release put six
    NEGATIVE numbers in a column named `aber_system_exact` and gave a reader
    nothing to measure them against.  Clamping the negatives to zero fixes the
    sign but destroys the evidence -- after the clamp the value no longer shows
    that it failed -- so each row now carries `floor_fast` and `floor_quad`
    beside the two answers.  These are MEASURED per row, not the module-wide
    constant `system_metric.ROUNDOFF_FLOOR`: a density cannot be negative, so
    the negative excursions of the reconstructed f_H are pure FFT round-off,
    and integrating their absolute value against the same Q weight puts that
    noise in the same units as the answer.  `resolved` is the derived flag,
    `aber_system > 10 * floor_fast` -- an order of magnitude of headroom over
    the transform's own noise.  The 10 is a stated margin, not a tuned one:
    both raw columns ship, so a reader who wants a different margin can apply
    it without rerunning anything.

    `resolved` IS NECESSARY BUT NOT SUFFICIENT, and the measured file says so.
    `floor_*` counts only the NEGATIVE excursions of f_H; the positive half of
    the same round-off is invisible to it but does enter the answer, so the
    column is a lower bound on the noise, not a bound on the error.  On the
    regenerated file 1710 of 1734 rows are `resolved = 1`, yet a handful at
    >= 44 dB and xi >= 1.548 -- where both paths return 1e-20..1e-18 -- have
    the two constructions disagreeing by 60-377%.  That disagreement is the
    more sensitive floor detector of the two, exactly as described above.  Use
    both: a row is trustworthy when `resolved = 1` AND `ref_spread_percent` is
    small.  Above 1e-14, clear of the floor by orders of magnitude, the two
    constructions agree to 0.068% worst case for xi >= 0.992 -- the residual
    disagreement there is the xi < 1 quadrature defect above, not round-off.

    `f_H_mass`, `E_H_numeric` and `E_H_analytic` are the manuscript's own two
    stated checks (recovery of unit mass and of E[H] = MN A_0 xi^2/(xi^2+1)),
    evaluated on the FULL branch support once per configuration -- not on the
    deliberately truncated grid `system_aber` integrates on, where the mass is
    supposed to be less than 1.
    """
    path = os.path.join(out, "system_aber_curves.csv")
    hdr = ["regime", "sigma_s", "xi", "snr_db",
           "aber_system", "aber_system_quad", "ref_spread_percent",
           "floor_fast", "floor_quad", "resolved",
           "f_H_mass", "E_H_numeric", "E_H_analytic", "A0"]
    rows = []
    # xi grid widened at BOTH ends relative to the previous release: 0.500 and
    # 0.789 are inside the manuscript's decision box (xi is clipped to
    # [max(0.5, xi_min(sigma_s)), 4.888]) and are precisely where the old
    # endpoint patch failed worst, so omitting them hid the defect.
    xis = ["1.967"] if smoke else ["0.500", "0.789", "0.992", "1.548",
                                   "1.967", "2.511", "3.104", "3.912", "4.888"]
    sigs = ["0.05"] if smoke else ["0.05", "0.1", "0.2", "0.3"]
    snrs = [30] if smoke else list(range(16, 49, 2))
    regs = list(REGIMES.items())[:1] if smoke else list(REGIMES.items())
    for rname, (A, B) in regs:
        for ss in sigs:
            for xs in xis:
                xi, s = mp.mpf(xs), mp.mpf(ss)
                A0 = A0_for(xi, s)
                if A0 is None:
                    continue
                a, b = float(A), float(B)
                xf, a0 = float(xi), float(A0)
                try:
                    mass, mean, mean_an = sm.verify_density(a, b, xf, a0,
                                                            method="fast")
                except Exception as e:
                    log("  density check failed %s/%s/%s: %s" % (rname, ss, xs, e))
                    mass = mean = mean_an = float("nan")
                for gdb in snrs:
                    if time.time() > deadline:
                        save_csv(path, hdr, rows)
                        return len(rows), False
                    try:
                        gbar = float(db(gdb))
                        vf, flf = sm.system_aber(a, b, xf, a0, gbar,
                                                 method="fast",
                                                 return_floor=True)
                        vq, flq = sm.system_aber(a, b, xf, a0, gbar,
                                                 method="quad",
                                                 return_floor=True)
                        spread = (abs(vq - vf) / abs(vf) * 100.0
                                  if vf > 0 else float("nan"))
                        resolved = int(vf > 10.0 * flf)
                        rows.append([rname, ss, xs, gdb,
                                     "%.8e" % vf, "%.8e" % vq,
                                     "%.6f" % spread,
                                     "%.6e" % flf, "%.6e" % flq, resolved,
                                     "%.6f" % mass,
                                     "%.6f" % mean, "%.6f" % mean_an,
                                     "%.6e" % a0])
                    except Exception as e:
                        log("  sysaber failed: %s" % e)
                    if len(rows) % 50 == 0:
                        save_csv(path, hdr, rows)
                        log("  system_aber: %d points" % len(rows))
    save_csv(path, hdr, rows)
    return len(rows), True


BLOCKS = [
    ("01_admissibility", block_admissibility),
    ("02_z_map", block_zmap),
    ("03_coefficient_tensors", block_tensors),
    ("05_eq22_validation", block_eq22),
    ("06_system_aber", block_system_aber),
    ("04_offgrid_error", block_offgrid),      # last: open-ended, soaks up remaining time
]


def main():
    global _log_fh
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="2026-08-27 12:00")
    ap.add_argument("--smoke", action="store_true")
    # A smoke run writes smoke-SCOPED blocks (one regime, one sigma, one SNR)
    # over whatever is already in the target directory, and the driver has no
    # undo. Before this option existed the target was always the package's own
    # data/, so the "~4 minute sanity run" the README recommends silently
    # replaced the full multi-hour dataset with stubs -- and the MANIFEST with
    # it, so afterwards there was no record that it had ever been complete.
    # Smoke runs now default to a sibling directory and must be pointed at the
    # real one deliberately.
    ap.add_argument("--out", default=None,
                    help="output root (default: data/ for a full run, "
                         "data_smoke/ for --smoke)")
    ap.add_argument("--only", default=None,
                    help="comma-separated block names to run (default: all). "
                         "Blocks not named keep their existing MANIFEST entry "
                         "rather than being erased.")
    a = ap.parse_args()

    global DATA, LOGS
    if a.out:
        DATA = os.path.abspath(os.path.join(a.out, "data"))
        LOGS = os.path.abspath(os.path.join(a.out, "logs"))
        manifest_dir = os.path.abspath(a.out)
    elif a.smoke:
        DATA = os.path.join(HERE, "data_smoke", "data")
        LOGS = os.path.join(HERE, "data_smoke", "logs")
        manifest_dir = os.path.join(HERE, "data_smoke")
    else:
        manifest_dir = HERE
    ensure(manifest_dir)

    ensure(DATA)
    ensure(LOGS)
    _log_fh = open(os.path.join(LOGS, "generate.log"), "a", encoding="utf-8")

    if a.smoke:
        deadline = time.time() + 240
        log("SMOKE TEST: 240 s budget")
        log("writing to %s  (pass --out . to overwrite the released dataset)" % DATA)
    else:
        deadline = datetime.strptime(a.deadline, "%Y-%m-%d %H:%M").timestamp()
        log("deadline %s  (%.1f h from now)" % (a.deadline, (deadline - time.time()) / 3600))

    # Merge into whatever provenance already exists rather than starting from
    # an empty dict: a targeted re-run of one block must not erase the record
    # of the blocks it did not touch, and must not erase the entries that
    # describe blocks this driver does not produce at all (07-09).
    manifest_path = os.path.join(manifest_dir, "MANIFEST.json")
    manifest = {"generated_by": "generate.py", "deadline": a.deadline,
                "started": datetime.now().isoformat(timespec="seconds"),
                "blocks": {}}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev.get("blocks"), dict):
                manifest["blocks"] = prev["blocks"]
                manifest["previous_started"] = prev.get("started")
        except Exception as e:
            log("could not read existing MANIFEST (%s); starting fresh" % e)

    only = None
    if a.only:
        only = {s.strip() for s in a.only.split(",") if s.strip()}
        known = {n for n, _ in BLOCKS}
        bad = only - known
        if bad:
            ap.error("unknown block(s): %s; known: %s"
                     % (", ".join(sorted(bad)), ", ".join(sorted(known))))
        log("running only: %s" % ", ".join(sorted(only)))

    for name, fn in BLOCKS:
        if only is not None and name not in only:
            continue
        if time.time() > deadline:
            log("deadline reached; skipping %s" % name)
            manifest["blocks"][name] = {"status": "skipped"}
            continue
        out = ensure(DATA, name)
        log("=== %s ===" % name)
        t0 = time.time()
        try:
            n, complete = fn(out, deadline, a.smoke)
            manifest["blocks"][name] = {
                "status": "complete" if complete else "partial",
                "produced_by": "generate.py::%s" % fn.__name__,
                "records": n, "seconds": round(time.time() - t0, 1),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "files": sorted(os.listdir(out)),
                "scope": "smoke" if a.smoke else "full"}
            log("  %s: %d records in %.1f s (%s)"
                % (name, n, time.time() - t0, "complete" if complete else "partial"))
        except Exception:
            log("  %s FAILED:\n%s" % (name, traceback.format_exc()))
            manifest["blocks"][name] = {"status": "failed"}
        # Per-block sidecar.  Two --only runs in parallel would otherwise race
        # on MANIFEST.json -- each reads the file at startup and rewrites the
        # whole blocks dict at the end, so the slower one silently drops the
        # faster one's entry.  The sidecar is written by exactly one process,
        # and code/build_manifest.py assembles the manifest from the sidecars.
        try:
            with open(os.path.join(LOGS, "provenance_%s.json" % name), "w",
                      encoding="utf-8") as f:
                json.dump(manifest["blocks"][name], f, indent=2)
        except Exception as e:
            log("  could not write sidecar for %s: %s" % (name, e))

        manifest["finished"] = datetime.now().isoformat(timespec="seconds")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    log("ALL DONE")


if __name__ == "__main__":
    main()
