"""Re-run the paper's own brute-force feasibility scan (system_metric.py's
certified aber_of / BeamConfig, exactly as used for Table success_feasibility)
at alternative reference SNRs gbar_op, to see what the feasibility ceiling
would be if the operating point were relocated away from 38 dB.

Not a new methodology: same brute-force min-over-xi-box search system_metric.py
performs at 38 dB, just swept over a small set of gbar_db values so the
12-cell (regime, sigma_s) ceiling at each candidate operating point can be
compared directly against the published 25.0%/41.7% figures at 38 dB.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar
from system_metric import REGIMES, SIGMAS, XI_MAX, BeamConfig, aber_of, ROUNDOFF_FLOOR, ABER_TARGET

CANDIDATES_DB = [38.0, 38.5, 39.0, 40.0]


def xi_min_for(regime, s):
    for cand in np.linspace(0.5, 6.0, 400):
        try:
            BeamConfig.from_xi(regime, cand, s)
            return cand
        except ValueError:
            continue
    return 0.5


def best_at(regime, s, gbar_db):
    lo = xi_min_for(regime, s)
    fun = lambda x: np.log10(max(aber_of(BeamConfig.from_xi(regime, x, s), gbar_db=gbar_db),
                                  ROUNDOFF_FLOOR))
    r = minimize_scalar(fun, bounds=(lo, XI_MAX), method="bounded", options=dict(xatol=1e-6))
    return 10.0 ** r.fun


def main():
    out = {}
    for gdb in CANDIDATES_DB:
        cells = {}
        n_feasible_strong = 0
        n_feasible_all = 0
        n_cells_strong = 0
        n_cells_all = 0
        for regime in REGIMES:
            for s in SIGMAS:
                v = best_at(regime, s, gdb)
                feasible = bool(v <= ABER_TARGET)
                cells["%s_%.2f" % (regime, s)] = dict(min_aber=v, feasible=feasible)
                n_cells_all += 1
                n_feasible_all += int(feasible)
                if regime == "strong":
                    n_cells_strong += 1
                    n_feasible_strong += int(feasible)
        out[str(gdb)] = dict(
            cells=cells,
            strong_feasible=n_feasible_strong, strong_total=n_cells_strong,
            strong_pct=100.0 * n_feasible_strong / n_cells_strong,
            pooled_feasible=n_feasible_all, pooled_total=n_cells_all,
            pooled_pct=100.0 * n_feasible_all / n_cells_all,
        )
        print("gbar_op = %.1f dB: strong %d/%d (%.1f%%), pooled %d/%d (%.1f%%)"
              % (gdb, n_feasible_strong, n_cells_strong, out[str(gdb)]["strong_pct"],
                 n_feasible_all, n_cells_all, out[str(gdb)]["pooled_pct"]))
        for k, v in cells.items():
            print("    %-16s min P_e,sys = %.4e  feasible=%s" % (k, v["min_aber"], v["feasible"]))
    with open("feasibility_at_gbar.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote feasibility_at_gbar.json")


if __name__ == "__main__":
    main()
