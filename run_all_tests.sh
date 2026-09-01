#!/usr/bin/env bash
# run_all_tests.sh — Chạy TOÀN BỘ các bộ test/self-check của repo, ghi PASS/FAIL.
# Không chạy generator nặng (đã có cycle + Nhóm C). Log: logs/test_report.log
set -u
REPO="/home/sim/A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework"
PY="$REPO/.venv/bin/python"
LOG="$REPO/logs/test_report.log"
cd "$REPO"
: > "$LOG"

PASS=0; FAIL=0; SKIP=0
report() {  # name rc [note]
  if [ "$2" = "0" ]; then PASS=$((PASS+1)); st="PASS";
  elif [ "$2" = "77" ]; then SKIP=$((SKIP+1)); st="SKIP";
  else FAIL=$((FAIL+1)); st="FAIL"; fi
  echo "[$st] $1 (rc=$2) ${3:-}" | tee -a "$LOG"
}

run() {  # name timeout cmd...
  local name="$1"; shift; local to="$1"; shift
  echo "=== $name ===" >> "$LOG"
  timeout "$to" "$@" >> "$LOG" 2>&1
  report "$name" "$?"
}

echo "=== TOÀN BỘ TEST BẮT ĐẦU $(date '+%H:%M:%S') ===" | tee -a "$LOG"

# --- verify.py tổng (tier 1-3) ---
run "verify.py (tier 1+3, 7 checks)" 300 "$PY" verify.py

# --- kernel self-check (đối chiếu mpmath 90-digit) ---
run "rtodt_fast self-check (vs 90-digit)" 300 "$PY" code/rtodt_fast.py
run "rtodt_turbo agreement+speedup" 300 "$PY" code/rtodt_turbo.py
run "rtodt_batch self-check (bit-identical)" 600 "$PY" code/rtodt_batch.py

# --- model/geometry ---
run "test_beam_geometry (8-case domain)" 120 "$PY" code/test_beam_geometry.py
run "validate_model (rebuild từ eq in bài)" 300 "$PY" code/validate_model.py
run "exact_reference sanity+convergence" 300 "$PY" code/exact_reference.py

# --- bit-identity fast objective ---
run "compare_objective (466560 bit-identical)" 900 "$PY" code/compare_objective.py

# --- bảng số ---
run "admissibility_bounds (Table 7)" 300 "$PY" code/admissibility_bounds.py
run "check_table9_arithmetic" 120 "$PY" code/check_table9_arithmetic.py
run "check_table11 --published-arithmetic-only" 120 "$PY" code/check_table11_statistics.py --published-arithmetic-only
run "eq22_summary" 120 "$PY" code/eq22_summary.py

# --- Eq22 system-level ---
run "eq22_recursion (4 SNR vs FFT ref)" 1500 "$PY" code/eq22_recursion.py
run "eq22_ladder_check (sweep vs ladder)" 1200 "$PY" code/eq22_ladder_check.py

# --- block verifier ---
run "verify_block04 (offgrid)" 600 "$PY" code/verify_block04.py
run "verify_block06 (system_aber)" 600 "$PY" code/verify_block06.py

# --- bench_cycle equivalence (chỉ check, không full arms) ---
run "bench_cycle check_equivalence" 300 "$PY" -c "import sys; sys.path.insert(0,'code'); import bench_cycle as bc; r=bc.check_equivalence(); print(r); sys.exit(0 if r['bit_identical'] else 1)"

echo "=== TỔNG KẾT: PASS=$PASS FAIL=$FAIL SKIP=$SKIP $(date '+%H:%M:%S') ===" | tee -a "$LOG"
echo "chi tiết: $LOG"
