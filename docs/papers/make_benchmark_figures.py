"""Generate the manuscript figures from the benchmark analysis.

Run from the repository root:  python docs/papers/make_benchmark_figures.py
Writes fig1_parity.png, fig2_residual_by_group.png and fig3_effective_conductivity.png
next to this script.
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from benchmark_analysis import GROUPS, load  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 7.6, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "figure.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
})
INK, GREY, BAND = "#1a1a1a", "#7a7a7a", "#ececec"
STYLE = [("#4d4d4d", "s"), ("#9a9a9a", "D"), ("#1f77b4", "o"), ("#17a2b8", "^"), ("#d62728", "v"), ("#e08a00", "P")]


def panel(ax, letter, x=-0.15):
    ax.text(x, 1.03, f"({letter})", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK)


def scatter(ax, rows, xkey, ykey, legend=False):
    for i, (label, _) in enumerate(GROUPS):
        colour, marker = STYLE[i]
        for graded, kw in ((True, dict(facecolor=colour, edgecolor="white", linewidth=0.4)),
                           (False, dict(facecolor="white", edgecolor=colour, linewidth=0.9))):
            sub = [r for r in rows if r["group"] == i and (r["quality"] != "C") == graded]
            n_all = sum(r["group"] == i for r in rows)
            ax.scatter([r[xkey] for r in sub], [r[ykey] for r in sub], s=24, marker=marker, zorder=3,
                       label=f"{label} ({n_all})" if (legend and graded) else None, **kw)


def fig1(rows):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.30})
    lim = (0, 130)
    a.fill_between(lim, [lim[0] - 10, lim[1] - 10], [lim[0] + 10, lim[1] + 10], color=BAND, lw=0)
    a.plot(lim, lim, color=INK, lw=0.9)
    scatter(a, rows, "slab", "se", legend=True)
    a.set(xlim=lim, ylim=lim, xlabel="Slab prediction (dB)", ylabel="Measured shielding effectiveness (dB)")
    a.set_aspect("equal")
    a.set_xticks(range(0, 131, 25)); a.set_yticks(range(0, 131, 25))
    panel(a, "a")
    b.axhspan(-10, 10, color=BAND, lw=0)
    b.axhline(0, color=INK, lw=0.9)
    scatter(b, [dict(r, lg=r["sigma"]) for r in rows], "lg", "res")
    b.set_xscale("log")
    b.set(xlim=(1e-3, 3e7), ylim=(-60, 60), xlabel=r"Reported conductivity (S m$^{-1}$)",
          ylabel=r"Residual, measured $-$ predicted (dB)")
    panel(b, "b")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.17),
               columnspacing=1.6, handletextpad=0.3)
    fig.savefig(HERE / "fig1_parity.png"); plt.close(fig)


def fig2(rows):
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.axvspan(-10, 10, color=BAND, lw=0)
    ax.axvline(0, color=INK, lw=0.9)
    rng = np.random.default_rng(3)
    for i, (label, _) in enumerate(GROUPS):
        colour, marker = STYLE[i]
        y = len(GROUPS) - 1 - i
        sub = [r for r in rows if r["group"] == i]
        prim = [r["res"] for r in sub if r["quality"] != "C"]
        q1, med, q3 = np.percentile(prim, [25, 50, 75])
        ax.plot([q1, q3], [y - 0.33, y - 0.33], color=INK, lw=1.6, solid_capstyle="butt")
        ax.plot([med], [y - 0.33], marker="|", color=INK, ms=7, mew=1.4)
        for r in sub:
            c = r["quality"] == "C"
            ax.scatter(r["res"], y + rng.uniform(-0.14, 0.2), s=20, marker=marker, zorder=3,
                       facecolor="white" if c else colour, edgecolor=colour if c else "white", linewidth=0.9 if c else 0.4)
        ax.text(62, y, f"n = {len(prim)}, {len({r['paper'] for r in sub if r['quality'] != 'C'})} papers",
                va="center", ha="left", fontsize=7.5, color=GREY)
    ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels([g for g, _ in GROUPS][::-1], fontsize=8)
    ax.set(xlim=(-60, 60), ylim=(-0.7, len(GROUPS) - 0.5), xlabel=r"Residual, measured $-$ predicted (dB)")
    ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "fig2_residual_by_group.png"); plt.close(fig)


def fig3(rows):
    fig, ax = plt.subplots(figsize=(3.9, 3.7))
    lim = (1e-3, 1e8)
    ax.fill_between(lim, [lim[0] / 10, lim[1] / 10], [lim[0] * 10, lim[1] * 10], color=BAND, lw=0)
    ax.plot(lim, lim, color=INK, lw=0.9)
    scatter(ax, [r for r in rows if r["sigma_eff"] == r["sigma_eff"]], "sigma", "sigma_eff")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlim=lim, ylim=lim, xlabel=r"Reported conductivity (S m$^{-1}$)",
           ylabel=r"Effective conductivity from measured SE (S m$^{-1}$)")
    ax.set_aspect("equal")
    ticks = [10.0 ** k for k in range(-3, 9, 2)]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.minorticks_off()
    fig.savefig(HERE / "fig3_effective_conductivity.png"); plt.close(fig)


def main() -> None:
    rows = load()
    fig1(rows); fig2(rows); fig3(rows)
    print("Wrote fig1_parity.png, fig2_residual_by_group.png, fig3_effective_conductivity.png")


if __name__ == "__main__":
    main()
