"""Re-run the paper's own brute-force feasibility scan (system_metric.py's
certified aber_of / BeamConfig, exactly as used for Table success_feasibility)
at alternative reference SNRs gbar_op, to see what the feasibility ceiling
would be if the operating point were relocated away from 38 dB.

Not a new methodology: same brute-force min-over-xi-box search system_metric.py
performs at 38 dB, just swept over a small set of gbar_db values so the
12-cell (regime, sigma_s) ceiling at each candidate operating point can be
compared directly against the published 25.0%/41.7% figures at 38 dB.

xi_min_for below used to be a fragile 400-point linear scan over [0.5, 6.0]
that silently fell through to a bogus xi_min=0.5 whenever the true floor
lay outside that range -- which happens at small enough sigma_s (e.g. the
tracked residual, ~7-45x smaller than raw sway): xi_min(sigma_s) =
0.087719/(2*sigma_s) then exceeds 6.0, so nothing in the scan ever
succeeded. Replaced with the closed form, which cannot fall through.

That floor can also exceed XI_MAX (system_metric.py), which per access.tex
Sec. VII-A is the RT-ODT series emulator's widest admissibility-ladder node,
sized so it is "never binding" for the raw-sway box (every optimum sits at
w_z <~ 0.2 m there) -- not a physical or geometric ceiling on this "lowest
SNR where ANY beam meets the target" search. When it does, best_at no longer
runs a bounded search between the floor and XI_MAX (which would be an empty,
invalid bracket): ABER is monotonically non-decreasing away from the floor
at every (regime, sigma_s) this module has been checked against -- matching
Sec. VII-D3's own dominance finding ("narrower dominates outright among the
configurations checked here") -- so the floor itself, verified directly at
sigma_s=0.0067 m (ABER=1.0016e-6 at gbar=6.98 dB, the published
Table success_feasibility weak/6.7mm crossing, to within 0.003 dB), is the
minimum, and is evaluated directly rather than searched for.

An earlier version of this fix instead widened the search's upper bound
past XI_MAX for every call. That silently regressed the nine cells whose
floor already sits below XI_MAX (bounded Brent optimisation does not
converge precisely at a boundary optimum, and a wider bracket changed where
it settled): gamma_min for weak/13.3mm moved from 7.6698 dB (matching the
published 7.71 dB to 0.04 dB) to 7.5884 dB (0.12 dB off). Reverted: XI_MAX
still bounds the search whenever the floor is beneath it, unchanged from
the original method; only the floor-exceeds-XI_MAX case is handled
specially, and only by direct evaluation, never by search.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar
from system_metric import REGIMES, SIGMAS, XI_MAX, BeamConfig, aber_of, ROUNDOFF_FLOOR, ABER_TARGET, branch_min_wz

CANDIDATES_DB = [38.0, 38.5, 39.0, 40.0]


def xi_min_for(regime, s):
    """Exact geometric floor xi_min(s) = weq_floor / (2*s) (see module docstring)."""
    _, weq_floor = branch_min_wz()
    return weq_floor / (2.0 * s)


def best_at(regime, s, gbar_db):
    lo = xi_min_for(regime, s)
    fun = lambda x: np.log10(max(aber_of(BeamConfig.from_xi(regime, x, s), gbar_db=gbar_db),
                                  ROUNDOFF_FLOOR))
    if lo >= XI_MAX:
        return 10.0 ** fun(lo)          # see module docstring: floor beyond XI_MAX, evaluated directly
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
