"""
Unified real-time + certified-accuracy campaign.

Re-scores every cycle of the compiled (zig cc) closed-loop timing campaign
(compiled_poc_kernel.c, rank_stages=1, strong regime, sigma_s=0.10 m,
gbar_op=38 dB) against the CERTIFIED post-EGC system ABER criterion of
system_metric.py (Sec. VI-C's success test), using the beam configuration
ACTUALLY PUBLISHED that cycle (post safety-fallback substitution) and that
cycle's own residual pointing offset r_d.

This is the joint statistic the manuscript's withdrawn "78% joint real-time
rate" number was supposed to be and never was: real-time compliance and
optimization success, measured on the SAME cycles, from the SAME campaign,
with the certified (not per-branch-surrogate) accuracy test.

Run: python joint_evaluator.py [csv_path] [regime] [sigma_s]
Defaults reproduce the strong/0.10 m headline campaign.
"""
import csv
import sys

from system_metric import BeamConfig, success, GBAR_OP_DB, ABER_TARGET


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "compiled_poc_per_cycle.csv"
    REGIME = sys.argv[2] if len(sys.argv) > 2 else "strong"
    SIGMA_S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    n_deadline = 0
    n_success = 0
    n_joint = 0
    n_domain_error = 0
    for row in rows:
        within_deadline = row["within_deadline"] == "1"
        if within_deadline:
            n_deadline += 1
        w_cmd = float(row["w_cmd"])
        r_d = float(row["r_d"])
        try:
            cfg = BeamConfig(REGIME, w_cmd, SIGMA_S, r_d)
            ok = success(cfg, gbar_db=GBAR_OP_DB, target=ABER_TARGET)
        except Exception:
            n_domain_error += 1
            ok = False
        if ok:
            n_success += 1
        if ok and within_deadline:
            n_joint += 1

    print("=" * 78)
    print("Unified real-time + certified-accuracy campaign")
    print("  source: %s (n=%d cycles, rank_stages=1," % (csv_path, n))
    print("  %s regime, sigma_s=%.2f m, gbar_op=%.1f dB, target<=%.0e)"
          % (REGIME, SIGMA_S, GBAR_OP_DB, ABER_TARGET))
    print("=" * 78)
    print("  within 800us deadline      : %5d/%d  (%.2f%%)"
          % (n_deadline, n, 100.0 * n_deadline / n))
    print("  certified success (post-EGC): %5d/%d  (%.2f%%)"
          % (n_success, n, 100.0 * n_success / n))
    print("  JOINT (both)                : %5d/%d  (%.2f%%)"
          % (n_joint, n, 100.0 * n_joint / n))
    if n_domain_error:
        print("  (%d cycles raised a domain error, scored as failure)"
              % n_domain_error)
    print()
    print("  for reference, deadline-miss cycles: %d" % (n - n_deadline))
    print("  for reference, success-miss cycles : %d" % (n - n_success))


if __name__ == "__main__":
    sys.exit(main())
