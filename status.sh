#!/usr/bin/env bash
# status.sh — Xem trạng thái chu kỳ sinh dữ liệu 4h chỉ bằng một lệnh.
REPO="/home/sim/A-Tractable-Hybrid-Chaos-Levy-Optimization-Framework"

echo "=== 1. Tiến trình generate.py đang chạy? ==="
P=$(pgrep -af "generate.py --deadline" | grep -v grep)
if [ -n "$P" ]; then
  echo "  ✅ ĐANG CHẠY: $P"
else
  echo "  ⛔ KHÔNG có tiến trình generate.py nào"
fi

echo
echo "=== 2. Khóa chu kỳ (cycle.lock) ==="
if flock -n "$REPO/logs/cycle.lock" -c true 2>/dev/null; then
  echo "  Khóa ĐANG MỞ → hiện không có chu kỳ nào chạy"
else
  echo "  Khóa ĐANG BỊ GIỮ → một chu kỳ đang chạy (chu kỳ cron tiếp theo sẽ tự chờ)"
fi

echo
echo "=== 3. Lịch cron 4h ==="
crontab -l | grep run_cycle | sed 's/^/  /'

echo
echo "=== 4. Tiến độ gần nhất (cycle.log, 6 dòng cuối) ==="
tail -6 "$REPO/logs/cycle.log" | sed 's/^/  /'

echo
echo "=== 5. Dữ liệu các block open-ended (04, 06) ==="
for f in "$REPO"/data/04_offgrid_error/offgrid_error.csv "$REPO"/data/06_system_aber/system_aber_curves.csv; do
  if [ -f "$f" ]; then
    n=$(($(wc -l < "$f") - 1))
    echo "  $(basename "$(dirname "$f")"): $n records — $(du -h "$f" | cut -f1)"
  else
    echo "  $(basename "$(dirname "$f")"): chưa có file (block chưa chạy tới)"
  fi
done

echo
echo "=== 6. Trạng thái từng block trong MANIFEST.json ==="
"$REPO/.venv/bin/python" - "$REPO/MANIFEST.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
for k, v in m.get("blocks", {}).items():
    st = v.get("status", "?")
    rec = v.get("records", "-")
    fin = v.get("finished", "")[11:16] if v.get("finished") else ""
    print(f"  {k}: {st:<9} records={rec:<8} finished={fin}")
EOF
