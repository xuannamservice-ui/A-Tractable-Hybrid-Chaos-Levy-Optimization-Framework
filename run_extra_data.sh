#!/usr/bin/env bash
# run_extra_data.sh — Chạy toàn bộ các script sinh/tái sinh dữ liệu phụ trợ
# để tăng độ dày dataset, KHÔNG đụng vào cycle 4h (generate.py) đang chạy.
#
# Nhóm A: script nhẹ (giây - phút), an toàn chạy song song với offgrid.
# Nhóm B: script vừa, ít tranh CPU (nice 10).
# Nhóm C: script nặng, chỉ chạy khi có flag --heavy (thường sau 16:00).
#
# Mọi output append vào logs/extra_data.log; mỗi script chạy xong đều in rc.
set -u
REPO="/home/sim/A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework"
PY="$REPO/.venv/bin/python"
LOG="$REPO/logs/extra_data.log"
cd "$REPO"

run() {
  local name="$1"; shift
  local nice_level="${1:-0}"; shift
  echo "[$(date '+%H:%M:%S')] === $name (nice $nice_level) ===" >> "$LOG"
  if [ "$nice_level" != "0" ]; then
    nice -n "$nice_level" "$PY" "$@" >> "$LOG" 2>&1
  else
    "$PY" "$@" >> "$LOG" 2>&1
  fi
  echo "    rc=$?  [$(date '+%H:%M:%S')]" >> "$LOG"
}

# ---------------- Nhóm A: nhẹ ----------------
run "eq22_summary (summary.json mới)" 0 code/eq22_summary.py
run "admissibility_bounds (Table 7)" 0 code/admissibility_bounds.py
run "validate_model" 0 code/validate_model.py
run "test_beam_geometry" 0 code/test_beam_geometry.py
run "check_table9_arithmetic" 0 code/check_table9_arithmetic.py
run "check_table11_statistics" 0 code/check_table11_statistics.py
run "exp_diagnose" 0 code/exp_diagnose.py
run "exp_exact_runtime" 0 code/exp_exact_runtime.py
run "exp_offgrid" 0 code/exp_offgrid.py
run "exp_timing" 0 code/exp_timing.py
run "make_fig2" 0 code/make_fig2.py

# ---------------- Nhóm B: vừa ----------------
run "levy_mechanism_probe" 10 code/levy_mechanism_probe.py
for sc in 0.005 0.05 0.15 0.3; do
  run "levy_truncation s=$sc" 10 code/levy_truncation.py --jump-scale "$sc"
done
run "levy_envelope quick" 10 code/levy_envelope.py --quick --trials 200 --iters 60

echo "[$(date '+%H:%M:%S')] NHOM A+B XONG — xem $LOG" >> "$LOG"
