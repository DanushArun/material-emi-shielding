"""Generate every manuscript figure from committed code and data.

Run from the repository root:  python docs/papers/make_figures.py
Writes fig1_parity.png ... fig5_sensitivity.png and group_statistics.txt
next to this script.
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from recompute_manuscript_numbers import se, sobol_total, tmm  # noqa: E402
from src.physics.composite_models import mclachlan_gem  # noqa: E402

HERE = Path(__file__).parent
DATA = ROOT / "data_collection" / "verified_benchmark.csv"
GAIN = HERE / "architecture_gain.csv"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "figure.facecolor": "white",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
})
INK, GREY = "#1a1a1a", "#7a7a7a"

# Analysis groups: (label, colour, marker, member classes)
GROUPS = [
    ("Thin metal films, transparent conductors", "#4d4d4d", "s",
     {"Metal film", "Transparent conductor", "Metal film (magnetic filler)"}),
    ("Metal meshes, coated fabric and paper", "#9a9a9a", "D",
     {"Metal mesh", "Metal-coated fabric", "Coated paper"}),
    ("Dense MXene and graphene films", "#1f77b4", "o",
     {"MXene film", "Graphene film", "MXene hybrid film"}),
    ("Porous MXene and graphene films", "#17a2b8", "^",
     {"MXene film (porous after annealing)", "Graphene film (porous)", "MXene hybrid film (porous)"}),
    ("Solid carbon-polymer composites", "#d62728", "v",
     {"Carbon polymer composite", "CNT polymer composite"}),
    ("Segregated composites, foams, aerogels", "#e08a00", "P",
     {"Segregated composite", "Foam", "Aerogel"}),
]


def load():
    bench = {r["id"]: r for r in csv.DictReader(DATA.open())}
    rows = []
    for g in csv.DictReader(GAIN.open()):
        b = bench[g["id"]]
        grp = next(i for i, (_, _, _, members) in enumerate(GROUPS) if b["material_class"] in members)
        rows.append({"id": g["id"], "group": grp, "meas": float(g["se_measured_db"]),
                     "slab": float(g["se_slab_db"]), "gain": float(g["gain_db"]),
                     "sigma": float(b["sigma_s_per_m"]), "td": float(g["t_over_delta"]),
                     "cite": b["citation"]})
    return rows


def panel(ax, letter):
    ax.text(-0.14, 1.03, f"({letter})", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK)


def fig1(rows):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"wspace": 0.32})
    lim = (0, 130)
    a.fill_between(lim, [lim[0] - 10, lim[1] - 10], [lim[0] + 10, lim[1] + 10], color="#ececec", lw=0)
    a.plot(lim, lim, color=INK, lw=0.9)
    for i, (label, colour, marker, _) in enumerate(GROUPS):
        sub = [r for r in rows if r["group"] == i]
        a.scatter([r["slab"] for r in sub], [r["meas"] for r in sub], s=24, marker=marker,
                  facecolor=colour, edgecolor="white", linewidth=0.4, label=f"{label} ({len(sub)})", zorder=3)
        b.scatter([r["sigma"] for r in sub], [r["gain"] for r in sub], s=24, marker=marker,
                  facecolor=colour, edgecolor="white", linewidth=0.4, zorder=3)
    a.set(xlim=lim, ylim=lim, xlabel="Homogeneous-slab prediction (dB)", ylabel="Measured shielding effectiveness (dB)")
    a.set_aspect("equal")
    a.text(126, 113, "±10 dB", color=GREY, fontsize=8, ha="right", va="top", rotation=45)
    b.axhspan(-10, 10, color="#ececec", lw=0)
    b.axhline(0, color=INK, lw=0.9)
    b.set(xscale="log", xlabel=r"Reported conductivity (S m$^{-1}$)", ylabel="Architecture gain (dB)", ylim=(-60, 60))
    panel(a, "a"); panel(b, "b")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.17), columnspacing=1.6, handletextpad=0.3)
    fig.savefig(HERE / "fig1_parity.png"); plt.close(fig)


def fig2(rows):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    rng = np.random.default_rng(3)
    ax.axvspan(-10, 10, color="#ececec", lw=0)
    ax.axvline(0, color=INK, lw=0.9)
    n = len(GROUPS)
    for i, (label, colour, marker, _) in enumerate(GROUPS):
        g = np.array([r["gain"] for r in rows if r["group"] == i])
        y = n - 1 - i
        ax.scatter(g, y + rng.uniform(-0.17, 0.17, len(g)), s=22, marker=marker, facecolor=colour,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([np.median(g)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.6, zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels([g[0] for g in GROUPS][::-1])
    ax.set(xlabel="Architecture gain: measured minus homogeneous-slab prediction (dB)", xlim=(-60, 60), ylim=(-0.6, n - 0.4))
    ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "fig2_gain_by_class.png"); plt.close(fig)


def fig3():
    sf, sm, fc, t = 1e5, 1e-10, 0.01, 2.0
    f = np.concatenate([np.linspace(1e-4, 0.02, 400), np.linspace(0.02, 0.10, 200)])
    gem = np.array([mclachlan_gem(x, sm, sf, fc, t) for x in f])
    lin = f * sf + (1 - f) * sm
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.34})
    a.plot(f * 100, lin, color="#1f77b4", lw=1.6, label="Linear mixing")
    a.plot(f * 100, gem, color="#d62728", lw=1.6, label="McLachlan GEM")
    a.axvline(fc * 100, color=GREY, lw=0.8, ls=(0, (4, 3)))
    a.text(fc * 100 + 0.15, 3e-9, r"$f_c$", color=GREY)
    a.set(yscale="log", xlabel="Filler volume fraction (%)", ylabel=r"Effective conductivity (S m$^{-1}$)", xlim=(0, 10))
    a.legend(frameon=False, loc="lower right")
    for sig_fn, colour in ((lin, "#1f77b4"), (gem, "#d62728")):
        b.plot(f * 100, [se(max(s, 1e-12), 1.0, 2e-3, 8.2e9)["total_se"] for s in sig_fn], color=colour, lw=1.6)
    b.axvline(fc * 100, color=GREY, lw=0.8, ls=(0, (4, 3)))
    b.set(xlabel="Filler volume fraction (%)", ylabel="Slab shielding effectiveness (dB)", xlim=(0, 10), ylim=(0, None))
    panel(a, "a"); panel(b, "b")
    fig.savefig(HERE / "fig3_percolation.png"); plt.close(fig)


def fig4():
    cu = 5.96e7
    d = np.logspace(-6, -2, 120)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for f, colour, lab in ((1e9, "#1f77b4", "1 GHz"), (1e10, "#d62728", "10 GHz")):
        ax.plot(d * 1e6, [tmm([(cu, 1.0, 0.5e-6), (0.0, 3.0, x), (cu, 1.0, 0.5e-6)], f)["total_se"] for x in d],
                color=colour, lw=1.6, label=f"Cu / PET / Cu, {lab}")
        ax.axhline(se(cu, 1.0, 1.0e-6, f)["total_se"], color=colour, lw=0.9, ls=(0, (4, 3)))
    ax.text(1.3, 77.5, "single 1.0 µm Cu layer", color=GREY, fontsize=8, va="top")
    ax.set(xscale="log", xlabel="PET spacer thickness (µm)", ylabel="Shielding effectiveness (dB)", ylim=(70, 160))
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(HERE / "fig4_multilayer.png"); plt.close(fig)


def fig5():
    cases = [
        ("Copper, 10 µm, 1 GHz", dict(sigma=5.96e7, mu=1.0, t=10e-6, f=1e9), dict(sigma=0.05, mu=1e-4, t=0.02, f=0.001)),
        ("Mild steel, 1 mm, 1 MHz", dict(sigma=6.99e6, mu=300.0, t=1e-3, f=1e6), dict(sigma=0.08, mu=0.20, t=0.02, f=0.001)),
        ("CNT composite, 2 mm, 8.2 GHz", dict(sigma=10.0, mu=1.0, t=2e-3, f=8.2e9), dict(sigma=0.30, mu=1e-4, t=0.05, f=0.001)),
    ]
    names = {"sigma": "Conductivity", "mu": "Permeability", "t": "Thickness", "f": "Frequency"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, letter, (title, nominal, cv) in zip(axes, "abc", cases):
        st = sobol_total(nominal, cv)
        keys = list(names)[::-1]
        vals = [max(st[k], 0.0) for k in keys]
        ax.barh(range(4), vals, color="#4d4d4d", height=0.62)
        for y, v in enumerate(vals):
            ax.text(v + 0.03, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(4)); ax.set_yticklabels([names[k] for k in keys])
        ax.set(xlim=(0, 1.15), xlabel=r"Total-order index $S_T$")
        ax.set_title(f"({letter}) {title}", fontsize=9, loc="left", pad=6)
        ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "fig5_sensitivity.png"); plt.close(fig)


def statistics(rows):
    out = [f"Verified benchmark: {len(rows)} entries from {len({r['cite'] for r in rows})} papers", ""]
    out.append(f"{'group':<44} {'n':>3} {'median gain':>12} {'min':>7} {'max':>7} {'within 10 dB':>13}")
    for i, (label, *_ ) in enumerate(GROUPS):
        g = np.array([r["gain"] for r in rows if r["group"] == i])
        out.append(f"{label:<44} {len(g):>3} {np.median(g):>+12.1f} {g.min():>+7.1f} {g.max():>+7.1f} "
                   f"{int((np.abs(g) <= 10).sum()):>6}/{len(g)}")
    g = np.array([r["gain"] for r in rows])
    out += ["", f"All: median |gain| {np.median(np.abs(g)):.1f} dB; within 10 dB: {int((np.abs(g) <= 10).sum())}/{len(g)}; "
                f"gain > +10 dB: {int((g > 10).sum())}; gain < -10 dB: {int((g < -10).sum())}"]
    low = np.array([r["gain"] for r in rows if r["sigma"] < 1.0])
    high = np.array([r["gain"] for r in rows if r["sigma"] >= 1e4])
    out.append(f"Conductivity below 1 S/m: n={len(low)}, median gain {np.median(low):+.1f} dB (range {low.min():+.1f} to {low.max():+.1f})")
    out.append(f"Conductivity at or above 1e4 S/m: n={len(high)}, median gain {np.median(high):+.1f} dB, "
               f"within 10 dB: {int((np.abs(high) <= 10).sum())}/{len(high)}")
    out += ["", "Entries with |gain| > 30 dB:"]
    for r in sorted(rows, key=lambda r: r["gain"]):
        if abs(r["gain"]) > 30:
            out.append(f"  {r['gain']:+6.1f} dB  {r['id']}  ({r['cite'].split(',')[0]})")
    text = "\n".join(out)
    (HERE / "group_statistics.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    rows = load()
    fig1(rows); fig2(rows); fig3(); fig4(); fig5()
    statistics(rows)
