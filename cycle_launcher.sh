#!/usr/bin/env bash
# cycle_launcher.sh — Khởi động một chu kỳ sinh dữ liệu 4h NẾU chưa có chu kỳ nào đang chạy.
#
# Được gọi bởi CẢ HAI bộ lập lịch (không bao giờ chồng lấn nhờ flock):
#   1. System cron  (crontab của user sim): 0 */4 * * *
#   2. Hermes cron  (tab Cron trong Hermes): every 4h
# Bộ nào chạy trước sẽ giành khóa; bộ kia in "cycle skipped" và thoát ngay.
# Chu kỳ thật (generate.py + build_manifest) được tách nền (setsid) nên sống
# sót sau khi script này thoát, và giữ khóa cho tới khi hoàn tất (~4h).
set -u

REPO="/home/sim/A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework"
PY="$REPO/.venv/bin/python"
LOCK="$REPO/logs/cycle.lock"
LOGDIR="$REPO/logs"
CYCLOG="$LOGDIR/cycle.log"

mkdir -p "$LOGDIR"
exec 9>"$LOCK"

if ! flock -n 9; then
  MSG="[$(date '+%Y-%m-%d %H:%M:%S')] cycle skipped: another cycle is already running"
  echo "$MSG" >> "$CYCLOG"
  echo "$MSG"
  exit 0
fi

DEADLINE=$(date -d '+4 hours' '+%Y-%m-%d %H:%M')
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==================== CYCLE LAUNCH (deadline $DEADLINE) ====================" >> "$CYCLOG"

# Tách nền: generate.py chạy tới deadline, xong tự dựng lại MANIFEST.
# Con cháu kế thừa fd 9 (khóa) → giữ khóa tới khi chu kỳ kết thúc.
setsid nohup bash -c '
  set -u
  DEADLINE="$1"; PY="$2"; REPO="$3"
  cd "$REPO"
  "$PY" generate.py --deadline "$DEADLINE" >> "$REPO/logs/cycle.log" 2>&1
  GRC=$?
  "$PY" code/build_manifest.py >> "$REPO/logs/cycle.log" 2>&1
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] ==================== CYCLE END (generate rc=$GRC) ====================" >> "$REPO/logs/cycle.log"
' _ "$DEADLINE" "$PY" "$REPO" >/dev/null 2>&1 < /dev/null &

MSG="[$(date '+%Y-%m-%d %H:%M:%S')] cycle launched in background (pid $!) — deadline $DEADLINE"
echo "$MSG" >> "$CYCLOG"
echo "$MSG"
exit 0
