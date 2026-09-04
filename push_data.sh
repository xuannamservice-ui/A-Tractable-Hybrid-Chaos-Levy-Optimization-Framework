#!/usr/bin/env bash
# push_data.sh — Commit + push dữ liệu lên GitHub origin/main.
#
# AN TOÀN: chỉ chạy khi KHÔNG có cycle nào đang ghi dữ liệu (generate.py).
# Nếu gọi nhầm lúc cycle đang chạy → thoát ngay, không add gì cả (bài học
# 711555a: git add -A lúc cycle ghi sẽ chụp file partial).
#
# Được gọi từ cycle_launcher.sh NGAY SAU "CYCLE END" (dữ liệu hoàn chỉnh,
# MANIFEST vừa build xong) → mỗi ~4h tự push 1 lần. Retry tối đa 3 lần.
set -u
REPO="/home/sim/A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework"
cd "$REPO"
LOG="$REPO/logs/push.log"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

# --- 1. Từ chối nếu cycle đang chạy ---
if pgrep -f 'generate.py --deadline' >/dev/null 2>&1; then
  echo "[$TS] push skipped: cycle đang chạy (generate.py) — không đụng working tree" >> "$LOG"
  exit 0
fi

# --- 2. Commit nếu có thay đổi ---
git add -A
if git diff --cached --quiet; then
  echo "[$TS] push: không có gì mới để commit" >> "$LOG"
  exit 0
fi
git commit -m "data: push $(date '+%Y-%m-%d %H:%M') cycle output" >> "$LOG" 2>&1 \
  || { echo "[$TS] commit failed" >> "$LOG"; exit 1; }

# --- 3. Push (retry 3 lần, đợi 15s giữa các lần) ---
for attempt in 1 2 3; do
  if timeout 120 git push origin main >> "$LOG" 2>&1; then
    echo "[$TS] push OK (attempt $attempt)" >> "$LOG"
    exit 0
  fi
  echo "[$TS] push attempt $attempt failed — retry..." >> "$LOG"
  sleep 15
done
echo "[$TS] push FAILED after 3 attempts — sẽ thử lại ở cycle kế tiếp" >> "$LOG"
exit 1
