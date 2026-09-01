#!/usr/bin/env bash
# run_cycle.sh — Một chu kỳ sinh dữ liệu của A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework.
#
# Chạy `generate.py` với wall-clock deadline = now + 4 giờ. Các block 01/02/03/05
# hoàn tất nhanh (deterministic); block 06 (system_aber) và 04 (offgrid_error,
# block cuối) là open-ended — ngấm hết thời gian còn lại, nên thời lượng chu kỳ
# quyết định độ sâu của dataset. Sau khi generate.py kết thúc, manifest được
# dựng lại từ sidecar provenance (bắt buộc theo README sau mọi ghi vào data/).
#
# flock (blocking) đảm bảo hai chu kỳ không chạy chồng lấn: chu kỳ trễ chờ chu kỳ
# trước nhả khóa rồi mới tính deadline — luôn được đủ 4 giờ.
set -u

REPO="/home/sim/A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework"
PY="$REPO/.venv/bin/python"
LOCK="$REPO/logs/cycle.lock"
LOGDIR="$REPO/logs"
CYCLOG="$LOGDIR/cycle.log"

mkdir -p "$LOGDIR"
exec 9>"$LOCK"

# blocking flock: chờ chu kỳ trước kết thúc nếu cần
flock 9 || { echo "[$(date '+%Y-%m-%d %H:%M:%S')] flock failed" >> "$CYCLOG"; exit 1; }

DEADLINE=$(date -d '+4 hours' '+%Y-%m-%d %H:%M')
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==================== CYCLE START (deadline $DEADLINE) ===================="
} >> "$CYCLOG"

cd "$REPO"
"$PY" generate.py --deadline "$DEADLINE" >> "$CYCLOG" 2>&1
RC=$?

# Dựng lại MANIFEST.json từ các sidecar provenance (README: bắt buộc sau khi data/ thay đổi)
"$PY" code/build_manifest.py >> "$CYCLOG" 2>&1 || true
MANIFEST_RC=$?

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==================== CYCLE END (generate rc=$RC, manifest rc=$MANIFEST_RC) ===================="
} >> "$CYCLOG"

exit "$RC"
