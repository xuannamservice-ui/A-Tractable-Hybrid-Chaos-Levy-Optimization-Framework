#!/usr/bin/env bash
# run_heavy_data.sh — Nhóm C: script nặng, chạy SAU khi cycle offgrid xong
# (dự kiến 16:00). Không chạy song song với offgrid để không làm chậm nó.
set -u
REPO="/home/sim/A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework"
PY="$REPO/.venv/bin/python"
LOG="$REPO/logs/extra_data.log"
cd "$REPO"

# chờ cycle hiện tại nhả khóa (offgrid xong, deadline 16:00)
for i in $(seq 1 120); do
  if flock -n "$REPO/logs/cycle.lock" -c true 2>/dev/null; then
    break
  fi
  sleep 30
done

echo "[$(date '+%H:%M:%S')] ===== NHOM C (heavy) bat dau =====" >> "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date '+%H:%M:%S')] === $name ===" >> "$LOG"
  nice -n 5 "$PY" "$@" >> "$LOG" 2>&1
  echo "    rc=$?  [$(date '+%H:%M:%S')]" >> "$LOG"
}

# bench_kernel full trên máy này (đã patch no-E-core)
run "bench_kernel full" code/bench_kernel.py
# bench_cycle full (đã fix check_equivalence — giờ chạy được trên VM)
run "bench_cycle" code/bench_cycle.py
# measure_all: baselines + ablation với trials lớn hơn
run "measure_all baselines 500" code/measure_all.py --part baselines --trials 500
run "measure_all ablation 500" code/measure_all.py --part ablation --trials 500
# levy_envelope FULL grid (96 cells) — envelope dày đặc cho paper
run "levy_envelope full grid" code/levy_envelope.py --trials 200 --iters 60
# campaign 300 faithful (chạy lại để có bản mới nhất)
run "run_campaign 300 faithful" code/run_campaign.py --realizations 300 --objective faithful
# rebuild MANIFEST sau tất cả
"$PY" code/build_manifest.py >> "$LOG" 2>&1
echo "[$(date '+%H:%M:%S')] ===== NHOM C XONG =====" >> "$LOG"
