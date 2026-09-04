#!/usr/bin/env python3
"""Live dashboard cho pipeline Chaos-Lévy (stdlib only, không cần cài thêm).

Server HTTP nhỏ: mở http://localhost:8899 trên browser, tự refresh mỗi 2s.
  * Bảng trạng thái 14 block data (file count + mtime mới nhất)
  * Số liệu chính nhảy liên tục: offgrid samples, eq22 rows, system_aber points
  * Biểu đồ sparkline (canvas, không thư viện ngoài): offgrid count theo thời gian
  * 8 dòng cycle.log gần nhất + trạng thái khóa cycle + disk
  * Endpoint /api trả JSON snapshot (để tool khác dùng được)

Chạy:  python3 code/live_dashboard.py [--port 8899]
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
LOGS = os.path.join(REPO, "logs")

HISTORY = {"offgrid": [], "system_aber": []}   # (epoch, count) for sparklines
HIST_MAX = 600                                 # 20 phút @2s


def _mtime(path):
    try:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    except OSError:
        return "-"


def _last_lines(path, n=8):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            return f.read().decode("utf-8", "replace").rstrip().splitlines()[-n:]
    except OSError:
        return []


def _count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return 0


def _block_status():
    blocks = []
    if not os.path.isdir(DATA):
        return blocks
    for d in sorted(os.listdir(DATA)):
        full = os.path.join(DATA, d)
        if not os.path.isdir(full):
            continue
        files = [f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))]
        newest = max((os.path.getmtime(os.path.join(full, f)) for f in files), default=0)
        blocks.append({"name": d, "n": len(files),
                       "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(newest)) if newest else "-",
                       "age_s": int(time.time() - newest) if newest else -1})
    return blocks


def snapshot():
    now = time.time()
    og = _count_lines(os.path.join(DATA, "04_offgrid_error", "offgrid_error.csv"))
    sa = _count_lines(os.path.join(DATA, "06_system_aber", "system_aber_curves.csv"))
    eq = _count_lines(os.path.join(DATA, "05_eq22_validation", "eq22_vs_reference.csv"))
    eqx = _count_lines(os.path.join(DATA, "05_eq22_validation", "eq22_vs_reference_sigma0p2_0p3.csv"))

    HISTORY["offgrid"].append((now, og))
    HISTORY["system_aber"].append((now, sa))
    for k in HISTORY:
        HISTORY[k] = [p for p in HISTORY[k] if now - p[0] < 1800][-HIST_MAX:]

    # extra-data batch (run_extra_data.sh / run_heavy_data.sh)
    extra = []
    try:
        with open(os.path.join(LOGS, "extra_data.log"), "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            lines = f.read().decode("utf-8", "replace").splitlines()
        for ln in lines:
            if "===" in ln:
                extra.append(ln.strip())
    except OSError:
        pass

    # test report (run_all_tests.sh)
    tests = []
    try:
        with open(os.path.join(LOGS, "test_report.log"), "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            lines = f.read().decode("utf-8", "replace").splitlines()
        for ln in lines:
            if ln.startswith("[PASS]") or ln.startswith("[FAIL]") or ln.startswith("[SKIP]") or "===" in ln:
                tests.append(ln.strip())
    except OSError:
        pass

    # cycle process?
    proc = ""
    try:
        out = os.popen("pgrep -af 'generate.py --deadline' | grep -v grep | head -1").read().strip()
        proc = out.split()[0] if out else ""
    except Exception:
        pass

    locked = True
    try:
        lock = os.path.join(LOGS, "cycle.lock")
        if os.path.exists(lock):
            with open(lock, "w") as fh:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
                locked = False
        else:
            locked = False
    except Exception:
        locked = True

    import shutil
    disk = shutil.disk_usage(REPO)

    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle": {"running": bool(proc), "pid": proc, "locked": locked,
                  "log": _last_lines(os.path.join(LOGS, "cycle.log"), 8)},
        "extra_batch": {"log": extra[-10:]},
        "tests": {"log": tests[-12:]},
        "blocks": _block_status(),
        "counts": {"offgrid": og, "system_aber": sa, "eq22": eq, "eq22_ext": eqx,
                   "paper_offgrid": 34864},
        "spark": {"offgrid": HISTORY["offgrid"], "system_aber": HISTORY["system_aber"]},
        "disk": {"free_gb": round(disk.free / 1e9, 1), "used_pct": round(100 * disk.used / disk.total, 1)},
    }


HTML = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Chaos-Lévy Live Dashboard</title>
<style>
 body{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#e6edf3;margin:16px}
 h1{font-size:18px;color:#58a6ff}.green{color:#3fb950}.red{color:#f85149}.dim{color:#8b949e}
 table{border-collapse:collapse;width:100%;margin:8px 0}
 th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #21262d;font-size:13px}
 th{color:#8b949e;font-weight:normal;text-transform:uppercase;font-size:11px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin:10px 0}
 .big{font-size:26px;font-weight:bold}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
 .bar{height:8px;background:#21262d;border-radius:4px;overflow:hidden}
 .bar>div{height:100%;background:#3fb950}
 canvas{width:100%;height:70px}
 .badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px}
 .ok{background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb66}
 .run{background:#3fb95022;color:#3fb950;border:1px solid #3fb95066}
 .warn{background:#d2992222;color:#d29922;border:1px solid #d2992266}
</style></head><body>
<h1>🛰 Chaos-Lévy — Live Data Dashboard <span class="dim" id="clock"></span></h1>
<div class="grid">
 <div class="card"><div class="dim">CYCLE 4h</div><div class="big" id="cyc">…</div><div id="cycpid" class="dim"></div></div>
 <div class="card"><div class="dim">OFFGRID samples</div><div class="big green" id="og">…</div><div class="dim" id="ogpct"></div></div>
 <div class="card"><div class="dim">SYSTEM_ABER points</div><div class="big" id="sa">…</div></div>
 <div class="card"><div class="dim">EQ22 rows (paper 132)</div><div class="big" id="eq">…</div></div>
 <div class="card"><div class="dim">EQ22 ext σ0.2/0.3</div><div class="big" id="eqx">…</div></div>
 <div class="card"><div class="dim">DISK free</div><div class="big" id="disk">…</div></div>
</div>
<div class="grid">
 <div class="card"><div class="dim">OFFGRID samples — 30 phút gần nhất (sparkline)</div><canvas id="sp_og"></canvas></div>
 <div class="card"><div class="dim">SYSTEM_ABER points — 30 phút gần nhất</div><canvas id="sp_sa"></canvas></div>
</div>
<div class="card"><div class="dim">DATA BLOCKS (file count • mtime mới nhất)</div>
 <table id="blocks"><tr><th>block</th><th>files</th><th>mtime mới nhất</th><th>age</th></tr></table></div>
<div class="card"><div class="dim">CYCLE LOG (8 dòng gần nhất)</div><pre id="log" style="font-size:12px;white-space:pre-wrap;margin:4px 0"></pre></div>
<div class="card"><div class="dim">EXTRA DATA BATCH — các script tăng độ dày (10 dòng gần nhất)</div><pre id="extra" style="font-size:12px;white-space:pre-wrap;margin:4px 0"></pre></div>
<div class="card"><div class="dim">TEST REPORT — run_all_tests.sh (PASS/FAIL từng bộ)</div><pre id="tests" style="font-size:12px;white-space:pre-wrap;margin:4px 0"></pre></div>
<script>
const $=id=>document.getElementById(id);
function spark(cv,pts){const c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);
 if(pts.length<2){c.fillStyle='#8b949e';c.fillText('chờ dữ liệu…',8,35);return}
 const vs=pts.map(p=>p[1]);let mn=Math.min(...vs),mx=Math.max(...vs);
 if(mx===mn){mx+=1}const w=cv.width,h=cv.height;
 c.strokeStyle='#3fb950';c.lineWidth=2;c.beginPath();
 pts.forEach((p,i)=>{const x=i/(pts.length-1)*w;const y=h-6-(p[1]-mn)/(mx-mn)*(h-14);
  i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();
 c.fillStyle='#8b949e';c.fillText(mn.toLocaleString(),4,h-4);c.fillText(mx.toLocaleString(),w-70,12);}
function age(s){if(s<0)return '—';if(s<60)return s+'s';if(s<3600)return Math.floor(s/60)+'m';return Math.floor(s/3600)+'h'+Math.floor(s%3600/60)+'m'}
async function tick(){try{
 const r=await fetch('/api');const d=await r.json();
 $('clock').textContent=d.now;
 $('cyc').innerHTML=d.cycle.running?'<span class="badge run">ĐANG CHẠY</span>':'<span class="badge warn">KHÔNG chạy</span>';
 $('cycpid').textContent=(d.cycle.running?('pid '+d.cycle.pid+' • '):'')+(d.cycle.locked?'khóa đang giữ':'khóa free');
 $('og').textContent=d.counts.offgrid.toLocaleString();
 $('ogpct').textContent='paper cũ: 34,864 • '+(100*d.counts.offgrid/d.counts.paper_offgrid).toFixed(0)+'%';
 $('sa').textContent=d.counts.system_aber.toLocaleString();
 $('eq').textContent=d.counts.eq22.toLocaleString();
 $('eqx').textContent=d.counts.eq22_ext.toLocaleString();
 $('disk').textContent=d.disk.free_gb+' GB ('+d.disk.used_pct+'% dùng)';
 spark($('sp_og'),d.spark.offgrid);spark($('sp_sa'),d.spark.system_aber);
 const tb=$('blocks');tb.innerHTML='<tr><th>block</th><th>files</th><th>mtime mới nhất</th><th>age</th></tr>';
 d.blocks.forEach(b=>{const tr=document.createElement('tr');
  tr.innerHTML='<td>'+b.name+'</td><td>'+b.n+'</td><td>'+b.mtime+'</td><td class="dim">'+age(b.age_s)+'</td>';
  tb.appendChild(tr)});
 $('log').textContent=d.cycle.log.join('\\n');
 $('extra').textContent=(d.extra_batch?.log||[]).join('\\n')||'chưa có — batch sẽ chạy';
 $('tests').textContent=(d.tests?.log||[]).join('\\n')||'chưa có — test đang chạy...';
}catch(e){$('log').textContent='Lỗi fetch: '+e}
setTimeout(tick,2000)}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api":
            body = json.dumps(snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address; 0.0.0.0 = reachable over Tailscale/LAN, "
                         "127.0.0.1 = localhost only")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("Live dashboard:  http://localhost:%d  (Ctrl-C để dừng)" % a.port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
