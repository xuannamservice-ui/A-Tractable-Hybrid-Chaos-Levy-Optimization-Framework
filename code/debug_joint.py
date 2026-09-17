import csv
import numpy as np
from system_metric import BeamConfig, aber_of, success, GBAR_OP_DB, ABER_TARGET

rows = []
with open("compiled_poc_per_cycle.csv", newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print("total rows:", len(rows))

# take first 5 rows and print full detail
for row in rows[:5]:
    w = float(row["w_cmd"]); rd = float(row["r_d"])
    cfg = BeamConfig("strong", w, 0.10, rd)
    cfg0 = BeamConfig("strong", w, 0.10, 0.0)
    v = aber_of(cfg, gbar_db=GBAR_OP_DB)
    v0 = aber_of(cfg0, gbar_db=GBAR_OP_DB)
    print("cycle=%s w=%.6f r_d=%.6f xi=%.4f xi_eff=%.4f A0=%.4e  "
          "P_e,sys(r_d)=%.4e  P_e,sys(r_d=0)=%.4e  success=%s"
          % (row["cycle"], w, rd, cfg.xi, cfg.xi_eff, cfg.A0, v, v0, v <= ABER_TARGET))

# distribution of r_d and w_cmd across the whole campaign
rds = np.array([float(r["r_d"]) for r in rows])
ws = np.array([float(r["w_cmd"]) for r in rows])
print("\nr_d stats: min=%.4f  median=%.4f  mean=%.4f  max=%.4f" % (rds.min(), np.median(rds), rds.mean(), rds.max()))
print("w_cmd stats: min=%.4f  median=%.4f  mean=%.4f  max=%.4f" % (ws.min(), np.median(ws), ws.mean(), ws.max()))

# best-over-box at r_d=0 for reference (system_metric's own ceiling)
from scipy.optimize import minimize_scalar
lo = 0.5
for cand in np.linspace(0.5, 6.0, 400):
    try:
        BeamConfig.from_xi("strong", cand, 0.10); lo = cand; break
    except ValueError:
        continue
fun = lambda x: np.log10(max(aber_of(BeamConfig.from_xi("strong", x, 0.10), gbar_db=GBAR_OP_DB), 1e-16))
r = minimize_scalar(fun, bounds=(lo, 4.888), method="bounded", options=dict(xatol=1e-6))
print("\nbox ceiling (minimize_scalar over xi, r_d=0): xi*=%.6f  P_e,sys*=%.6e" % (r.x, 10.0**r.fun))

# brute force fine grid to check for missed global minimum
xs = np.linspace(lo, 4.888, 2000)
vals = [aber_of(BeamConfig.from_xi("strong", x, 0.10), gbar_db=GBAR_OP_DB) for x in xs]
i = int(np.argmin(vals))
print("brute force fine grid (r_d=0): xi*=%.6f  P_e,sys*=%.6e  (w_z*=%.6f)"
      % (xs[i], vals[i], BeamConfig.from_xi("strong", xs[i], 0.10).w_z))
