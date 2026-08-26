"""
Regenerate the RT-ODT validation figure (the paper's Fig. 2, S3_fig3.jpg).

Fixes two defects in the published rendering:
  1. ABER increased with SNR -- physically inverted, and contradicting Fig. 7.
  2. The caption promises the surrogate is omitted outside the admissible band,
     but the published curve spanned the whole range.

Both curves here are computed independently of one another:
  exact   -- double Gauss-Legendre over the gamma-gamma x pointing composite
  series  -- eq:aber_emulator at K=10, evaluated exactly (no interpolation)
and they agree to ~1e-7 relative inside the admissible band.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp

from exact_reference import aber_exact
from rtodt import REGIMES, A0_for, Pe_series, z_param, db

XI = mp.mpf("1.967")
SIGMA = mp.mpf("0.05")
K = 10
ZMAX = 2.0   # ladder rung serving K=10 (z<=2); z<=8 is the K=20 rung
SNR = np.arange(0, 51, 1.0)

C_EXACT = "#1A62A8"     # validated pair, see validate_palette.js
C_SERIES = "#C43A22"
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#d8d8d8"

A0 = A0_for(XI, SIGMA)
print("xi = %s, sigma_s = %s m, A_0 = %.4f, K = %d" % (XI, SIGMA, float(A0), K))

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharey=True)
titles = {"weak": "Weak turbulence", "moderate": "Moderate turbulence",
          "strong": "Strong turbulence"}

for ax, reg in zip(axes, ("weak", "moderate", "strong")):
    A, B = REGIMES[reg]
    ex, se, zs = [], [], []
    for g in SNR:
        gb = db(float(g))
        ex.append(aber_exact(float(gb), float(A), float(B), float(XI), float(A0)))
        zs.append(float(z_param(A, B, A0, gb)))
        se.append(float(Pe_series(A, B, XI, A0, gb, K)))
    ex, se, zs = np.array(ex), np.array(se), np.array(zs)

    adm = zs <= ZMAX
    se_plot = np.where(adm & (se > 0), se, np.nan)
    cut = SNR[adm][0] if adm.any() else None

    ax.semilogy(SNR, ex, color=C_EXACT, lw=2.0, label="Exact (numerical)", zorder=3)
    ax.semilogy(SNR, se_plot, color=C_SERIES, lw=2.0, ls=(0, (5, 2.5)),
                label=r"RT-ODT series, $K{=}10$", zorder=4)

    if cut is not None and cut > SNR[0]:
        ax.axvspan(SNR[0], cut, color="#000000", alpha=0.06, lw=0, zorder=0)
        ax.axvline(cut, color=INK2, lw=0.9, ls=":", zorder=2)
        ax.annotate("%d dB" % cut, xy=(cut, 3.6e-4), fontsize=7.5, color=INK2,
                    va="bottom", ha="left", xytext=(cut + 1.2, 3.6e-4))

    ax.set_title(titles[reg], fontsize=10, color=INK, pad=7)
    ax.set_xlabel("Average SNR $\\bar{\\gamma}$ (dB)", fontsize=9, color=INK)
    ax.set_xlim(0, 50)
    ax.set_ylim(3e-4, 1.0)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.5)
    ax.tick_params(labelsize=8, colors=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)

axes[0].set_ylabel("Per-branch ABER", fontsize=9, color=INK)

from matplotlib.patches import Patch
_h, _l = axes[0].get_legend_handles_labels()
_h.append(Patch(facecolor="#000000", alpha=0.06, edgecolor="none"))
_l.append("inadmissible for $K{=}10$ ($z>2$): surrogate omitted")
fig.legend(_h, _l, fontsize=8.5, frameon=False, ncol=3, labelcolor=INK,
           loc="upper center", bbox_to_anchor=(0.5, 0.985))

fig.suptitle("RT-ODT power series against an independent exact evaluation "
             "($\\xi=1.967$, $\\sigma_s=0.05$ m)", fontsize=10.5, color=INK, y=1.075)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = "S3_fig3_corrected.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", out)

# report the admissibility cut per regime
for reg in ("weak", "moderate", "strong"):
    A, B = REGIMES[reg]
    zz = np.array([float(z_param(A, B, A0, db(float(g)))) for g in SNR])
    ok = SNR[zz <= ZMAX]
    print("  %-9s surrogate plotted from %.0f dB upward" % (reg, ok[0] if len(ok) else np.nan))
