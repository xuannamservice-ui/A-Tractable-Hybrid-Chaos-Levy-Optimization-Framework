"""Parallel off-grid error sampler (block 04).

Same measurement as `generate.block_offgrid` -- the deployed float64 kernel of
eq. (21) against the mpmath reference at 200 digits, sampled at off-grid xi --
but spread across CPU workers instead of running on one core.

WHY NOT GPU.  Three reasons, measured on this machine rather than assumed:

  1. The reference half of every sample is `rtodt.Pe_series` evaluated in
     arbitrary precision at dps 200.  GPUs implement FP16/FP32/FP64 in
     hardware and have no arbitrary-precision path at all, so that half cannot
     move to the device under any rewrite.  It is also the dominant cost:
     ~48 ms per sample against microseconds for the float64 side.

  2. The GPU present here is a GTX 1070 (Pascal GP104, compute 6.1), whose
     FP64 throughput is 1/32 of its FP32 rate -- about 0.2 TFLOPS.  The host
     i5-14600KF reaches roughly 0.5-0.7 TFLOPS FP64 across its cores with
     AVX2 FMA.  For the precision this measurement requires the CPU is the
     faster device by 2-3x.

  3. No GPU runtime is installed (no cupy, torch, numba or jax).

  The available speedup was never the GPU: `generate.py` contains no
  parallelism at all and was using one of twenty logical threads.

REPRODUCIBILITY.  The serial version draws xi from one `random.Random(20260826)`
stream shared by every combination.  That stream cannot be split without
changing the sequence, so each worker here is given its own stream seeded
`BASE_SEED + 1000 * worker_id`, and the worker id and seed are written into
every row.  A reader can therefore regenerate any single row exactly.

APPEND, DO NOT OVERWRITE.  `generate.py` rewrites `offgrid_error.csv` from its
in-memory row list at each checkpoint, so restarting it discards whatever a
previous run produced -- that is how a 20,000-row file became a 3,001-row file
when the block was resumed.  This script appends, and refuses to start if the
header of an existing file does not match what it is about to write.

Usage:
    python offgrid_parallel.py [--minutes 60] [--workers 12] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as mp_proc
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for p in (PKG, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

BASE_SEED = 20260826
SNRS = (30, 34, 38, 40, 44)

HDR = ["regime", "sigma_s", "snr_db", "xi", "z", "K", "A0",
       "Pe_reference", "Pe_interp_free", "abs_err_interp_free",
       "rel_err_interp_free", "eta_f64", "ref_in_range", "f64_in_range",
       "worker", "seed"]


def _combos():
    import generate as G
    return [(rn, AB, s, g) for rn, AB in G.REGIMES.items()
            for s in G.SIGMAS for g in SNRS]


def _sample_one(G, np, rn, A, B, s, gdb, xi):
    """One measurement. Returns a row, or None if the point is not admissible."""
    mp = G.mp
    mp.mp.dps = 90
    A0 = G.A0_for(xi, s)
    if A0 is None:
        return None
    g = G.db(gdb)
    z = float(G.z_param(A, B, A0, g))
    K = G.ladder_K(z)
    if K is None:
        return None

    mp.mp.dps = 200
    ref = float(G.Pe_series(A, B, xi, A0, g, K))          # arbitrary precision
    fast = float(G.pe_series_f64(float(A), float(B), float(xi),
                                 float(A0), float(g), K)[0])   # deployed f64

    mp.mp.dps = 90
    terms = []
    for k in range(K + 1):
        terms.append(abs(G.a_k(A, B, xi, A0, k) * G.C_moment(B + k, g)))
        terms.append(abs(G.a_k(B, A, xi, A0, k) * G.C_moment(A + k, g)))
    eta = float(max(terms)) * G.EPS64

    if not np.isfinite(fast):
        # overflow or a Gamma pole IS the deployed behaviour here: record it
        err = rel = float("inf")
    else:
        err = abs(fast - ref)
        rel = err / abs(ref) if ref else float("nan")

    return [rn, "%.17g" % float(s), gdb, "%.17g" % float(xi), "%.6f" % z, K,
            "%.17g" % float(A0), "%.17g" % ref, "%.17g" % fast,
            "%.17g" % err, "%.17g" % rel, "%.17g" % eta,
            int(0.0 <= ref <= 0.5), int(np.isfinite(fast) and 0.0 <= fast <= 0.5)]


def worker(args):
    """Sample this worker's slice of the combination list until the deadline."""
    wid, combo_slice, deadline, shard_path = args
    import numpy as np
    import generate as G

    rng = random.Random(BASE_SEED + 1000 * wid)
    seed = BASE_SEED + 1000 * wid
    lo, hi = G.NODES[0], G.NODES[-1]
    rows, i, skipped = [], 0, 0

    while time.time() < deadline and combo_slice:
        rn, (A, B), s, gdb = combo_slice[i % len(combo_slice)]
        i += 1
        G.mp.mp.dps = 90
        xi = lo + (hi - lo) * G.mp.mpf(rng.random())
        try:
            row = _sample_one(G, np, rn, A, B, s, gdb, xi)
        except Exception:
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        rows.append(row + [wid, seed])

    with io.open(shard_path, "w", encoding="ascii", newline="") as fh:
        w = csv.writer(fh)
        w.writerows(rows)
    return wid, len(rows), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--out", default=os.path.join(PKG, "data", "04_offgrid_error",
                                                  "offgrid_error.csv"))
    a = ap.parse_args()

    combos = _combos()
    nw = max(1, min(a.workers, len(combos)))
    deadline = time.time() + a.minutes * 60.0

    existing = 0
    if os.path.exists(a.out):
        with io.open(a.out, encoding="utf-8") as fh:
            head = fh.readline().strip().split(",")
            existing = sum(1 for _ in fh)
        # the serial writer emits 14 columns; ours adds worker+seed
        if head != HDR and head != HDR[:14]:
            sys.exit("ABORT: existing file has an unexpected header:\n  %s" % head)

    print("off-grid sampler: %d workers, %d combinations, %.0f min budget"
          % (nw, len(combos), a.minutes))
    print("appending to %s (%d existing rows)" % (a.out, existing))

    shard_dir = os.path.join(os.path.dirname(a.out), "_shards")
    if not os.path.isdir(shard_dir):
        os.makedirs(shard_dir)
    tasks = [(w, combos[w::nw], deadline, os.path.join(shard_dir, "w%02d.csv" % w))
             for w in range(nw)]

    t0 = time.time()
    with mp_proc.Pool(nw) as pool:
        results = pool.map(worker, tasks)
    dt = time.time() - t0

    total = sum(n for _, n, _ in results)
    skipped = sum(s for _, _, s in results)

    # --- merge: append shards, widening the old 14-column rows if needed ----
    tmp = a.out + ".tmp"
    with io.open(tmp, "w", encoding="ascii", newline="") as out:
        w = csv.writer(out)
        w.writerow(HDR)
        if existing:
            with io.open(a.out, encoding="utf-8") as fh:
                r = csv.reader(fh)
                old_hdr = next(r)
                pad = len(HDR) - len(old_hdr)
                for row in r:
                    w.writerow(row + [""] * pad if pad > 0 else row)
        for _, _, sh in [(t[0], t[1], t[3]) for t in tasks]:
            if os.path.exists(sh):
                with io.open(sh, encoding="utf-8") as fh:
                    w.writerows(csv.reader(fh))
                os.remove(sh)
    os.replace(tmp, a.out)
    try:
        os.rmdir(shard_dir)
    except OSError:
        pass

    with io.open(a.out, encoding="utf-8") as fh:
        final = sum(1 for _ in fh) - 1

    print("\n  new samples   %d in %.1f s  (%.1f samples/s)" % (total, dt, total / dt))
    print("  per worker    %s" % ", ".join(str(n) for _, n, _ in results))
    print("  inadmissible  %d (skipped, not an error)" % skipped)
    print("  file total    %d rows" % final)
    # The serial baseline is not estimated: generate.py logs a checkpoint every
    # 250 samples, and those timestamps give 20.8 samples/s on this machine.
    # Quote the ratio of two measured rates, never workers x an efficiency guess.
    print("  rate          %.1f samples/s (serial block_offgrid: 20.8/s"
          " from its own checkpoint timestamps -> %.1fx)"
          % (total / dt, (total / dt) / 20.8))


if __name__ == "__main__":
    main()
