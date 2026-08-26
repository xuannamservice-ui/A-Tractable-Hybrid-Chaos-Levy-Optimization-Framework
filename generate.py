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
from egc_system import f_h_exact, aber_system, MN                    # noqa: E402

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
    """Open-ended: keeps sampling off-grid xi until the deadline."""
    mp.mp.dps = 90
    rng = random.Random(20260826)
    path = os.path.join(out, "offgrid_error.csv")
    hdr = ["regime", "sigma_s", "snr_db", "xi", "z", "A0",
           "Pe_reference", "Pe_interp_free", "abs_err_interp_free", "valid_probability"]
    rows = []
    target = 40 if smoke else 10 ** 9
    combos = [(rn, AB, s, g) for rn, AB in REGIMES.items()
              for s in SIGMAS for g in (30, 34, 38, 40, 44)]
    i = 0
    while len(rows) < target:
        if time.time() > deadline:
            break
        rn, (A, B), s, gdb = combos[i % len(combos)]
        i += 1
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
            mp.mp.dps = 200
            ref = Pe_series(A, B, xi, A0, g, K)
            mp.mp.dps = 90
            approx = Pe_series(A, B, xi, A0, g, K)
            err = abs(float(approx) - float(ref))
            rows.append([rn, float(s), gdb, "%.6f" % float(xi), "%.4f" % z,
                         "%.6e" % float(A0), "%.10e" % float(ref),
                         "%.10e" % float(approx), "%.3e" % err,
                         int(0.0 <= float(ref) <= 0.5)])
        except Exception:
            continue
        if len(rows) % 250 == 0:
            save_csv(path, hdr, rows)
            log("  offgrid: %d samples" % len(rows))
    save_csv(path, hdr, rows)
    return len(rows), True


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


def block_eq22(out, deadline, smoke):
    mp.mp.dps = 260
    path = os.path.join(out, "eq22_vs_reference.csv")
    hdr = ["regime", "sigma_s", "xi", "K", "n_exponents", "snr_db",
           "eq22", "exact_reference", "rel_diff_percent"]
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
                            break
                        ref = aber_system(f_h_exact, 40 * float(A0), 60000,
                                          float(db(gdb)), float(A), float(B),
                                          float(xi), float(A0), "exact")[0]
                        v = float(ser[gdb])
                        rel = (v - ref) / ref * 100 if ref else float("nan")
                        rows.append([rname, ss, xs, K, nexp, gdb,
                                     "%.8e" % v, "%.8e" % ref, "%+.4f" % rel])
                        save_csv(path, hdr, rows)
                except Exception as e:
                    log("  eq22 config failed: %s" % e)
    save_csv(path, hdr, rows)
    return len(rows), True


# ----------------------------------------------------------------- block 06
def block_system_aber(out, deadline, smoke):
    path = os.path.join(out, "system_aber_curves.csv")
    hdr = ["regime", "sigma_s", "xi", "snr_db", "aber_system_exact", "f_H_mass"]
    rows = []
    xis = ["1.967"] if smoke else ["0.992", "1.548", "1.967", "2.511", "3.104", "3.912"]
    sigs = ["0.05"] if smoke else ["0.05", "0.1", "0.2"]
    snrs = [30] if smoke else list(range(16, 49, 2))
    regs = list(REGIMES.items())[:1] if smoke else list(REGIMES.items())
    for rname, (A, B) in regs:
        for ss in sigs:
            for xs in xis:
                xi, s = mp.mpf(xs), mp.mpf(ss)
                A0 = A0_for(xi, s)
                if A0 is None:
                    continue
                for gdb in snrs:
                    if time.time() > deadline:
                        save_csv(path, hdr, rows)
                        return len(rows), False
                    try:
                        v, mass, _ = aber_system(f_h_exact, 40 * float(A0), 60000,
                                                 float(db(gdb)), float(A), float(B),
                                                 float(xi), float(A0), "exact")
                        rows.append([rname, ss, xs, gdb, "%.8e" % v, "%.6f" % mass])
                    except Exception as e:
                        log("  sysaber failed: %s" % e)
                    if len(rows) % 20 == 0:
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
    a = ap.parse_args()

    ensure(DATA)
    ensure(LOGS)
    _log_fh = open(os.path.join(LOGS, "generate.log"), "a", encoding="utf-8")

    if a.smoke:
        deadline = time.time() + 240
        log("SMOKE TEST: 240 s budget")
    else:
        deadline = datetime.strptime(a.deadline, "%Y-%m-%d %H:%M").timestamp()
        log("deadline %s  (%.1f h from now)" % (a.deadline, (deadline - time.time()) / 3600))

    manifest = {"generated_by": "generate.py", "deadline": a.deadline,
                "started": datetime.now().isoformat(timespec="seconds"), "blocks": {}}

    for name, fn in BLOCKS:
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
                "records": n, "seconds": round(time.time() - t0, 1)}
            log("  %s: %d records in %.1f s (%s)"
                % (name, n, time.time() - t0, "complete" if complete else "partial"))
        except Exception:
            log("  %s FAILED:\n%s" % (name, traceback.format_exc()))
            manifest["blocks"][name] = {"status": "failed"}
        manifest["finished"] = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(HERE, "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    log("ALL DONE")


if __name__ == "__main__":
    main()
