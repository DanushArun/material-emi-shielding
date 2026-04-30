"""Generate publication-quality figures for Materials and Design journal paper.

All figures use the ACTUAL physics engine from the codebase, not mock data.
Benchmark data from data_collection/experimental_benchmark_data.py provides
experimental overlay points for model validation.

Note on model-experiment discrepancy:
    The Schelkunoff plane-wave analytical model computes exact electromagnetic
    SE.  For good conductors where thickness >> skin depth, absorption grows
    as 8.686 * alpha * d (hundreds to thousands of dB), whereas measurements
    are limited to ~120 dB instrument dynamic range.  The parity plot
    therefore separates the low-SE validated region from the high-SE
    extrapolation region.

Outputs five 300-DPI PNG files into docs/papers/:
    figure1_parity.png          Predicted vs experimental SE parity plot
    figure2_gem_comparison.png  GEM vs linear mixing across percolation
    figure3_sobol.png           Sobol sensitivity tornado (two cases)
    figure4_frequency_sweep.png Frequency sweep with decomposition + overlay
    figure5_mc_histogram.png    Monte Carlo SE distribution with CI
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.physics.emi_calculations import EMICalculator
from src.physics.composite_models import (
    mclachlan_gem,
    percolation_conductivity,
)
from src.physics.uncertainty import monte_carlo_se, sobol_sensitivity
from data_collection.experimental_benchmark_data import (
    PURE_METAL_DATA,
    COMPOSITE_DATA,
    MULTILAYER_DATA,
    TEMPERATURE_EFFECTS_DATA,
    MICROSTRUCTURE_EFFECTS_DATA,
    FREQUENCY_BAND_DATA,
)

OUTPUT_DIR = PROJECT_ROOT / "docs" / "papers"
DPI = 300

calculator = EMICalculator()


def _journal_style() -> None:
    """Apply journal-ready matplotlib rcParams."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#222222",
        "grid.color": "#DDDDDD",
        "grid.alpha": 0.7,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def _has_valid_sigma(entry: dict) -> bool:
    """True when entry has a numeric conductivity value."""
    sigma = entry.get("conductivity_sm")
    return sigma is not None and isinstance(sigma, (int, float))


def _predict_se(entry: dict) -> float:
    """Compute predicted SE (dB) for one benchmark entry."""
    sigma = entry["conductivity_sm"]
    mu_r = entry.get("permeability", 1.0)
    if mu_r is None:
        mu_r = 1.0
    mu_r = max(mu_r, 0.999)
    t_m = entry["thickness_mm"] * 1e-3
    f_hz = entry["frequency_hz"]

    result = calculator.calculate_shielding_effectiveness(
        conductivity=sigma,
        relative_permeability=mu_r,
        relative_permittivity=1.0,
        thickness=t_m,
        frequency=f_hz,
        include_confidence=False,
    )
    return float(result["total_se"])


# -----------------------------------------------------------------------
# Figure 1 -- Parity Plot (two panels)
# -----------------------------------------------------------------------

def figure1_parity() -> None:
    """Two-panel parity plot.

    Panel (a): full dataset on log-log axes with measurement-ceiling
    annotation.
    Panel (b): validated subset (experimental SE <= 100 dB) on linear
    axes with R^2, MAE, RMSE computed only on this subset.
    """
    categories = [
        ("Pure metals", PURE_METAL_DATA, "#1f77b4", "o"),
        ("Composites", COMPOSITE_DATA, "#d62728", "s"),
        ("Temperature", TEMPERATURE_EFFECTS_DATA, "#ff7f0e", "^"),
        ("Microstructure", MICROSTRUCTURE_EFFECTS_DATA, "#9467bd", "D"),
        ("Freq. bands", FREQUENCY_BAND_DATA, "#17becf", "v"),
    ]

    se_ceiling = 100.0

    all_exp: list[float] = []
    all_pred: list[float] = []
    low_exp: list[float] = []
    low_pred: list[float] = []
    cat_data: dict[str, dict] = {}

    for label, dataset, color, marker in categories:
        exps: list[float] = []
        preds: list[float] = []
        for entry in dataset:
            if not _has_valid_sigma(entry):
                continue
            exp = entry["se_db"]
            pred = _predict_se(entry)
            if exp <= 0 or pred <= 0:
                continue
            exps.append(exp)
            preds.append(pred)
            if exp <= se_ceiling:
                low_exp.append(exp)
                low_pred.append(pred)
        cat_data[label] = {
            "exps": exps, "preds": preds,
            "color": color, "marker": marker,
        }
        all_exp.extend(exps)
        all_pred.extend(preds)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.32},
    )

    # --- Panel (a): full log-log ---
    for label, d in cat_data.items():
        n = len(d["exps"])
        ax_a.scatter(
            d["exps"], d["preds"],
            label=f"{label} (n={n})", color=d["color"],
            marker=d["marker"], s=40, alpha=0.75,
            edgecolors="white", linewidth=0.4, zorder=3,
        )

    arr_all_e = np.array(all_exp)
    arr_all_p = np.array(all_pred)
    lo_log, hi_log = 1.0, max(arr_all_e.max(), arr_all_p.max()) * 1.5
    diag = np.array([lo_log, hi_log])

    ax_a.plot(
        diag, diag, "k-", linewidth=1.2,
        label="y = x", zorder=2,
    )
    ax_a.fill_between(
        diag, diag / 2, diag * 2,
        color="#CCCCCC", alpha=0.25, zorder=1,
    )

    ax_a.axhspan(
        se_ceiling, hi_log, color="#fff3e0", alpha=0.30, zorder=0,
    )
    ax_a.axvspan(
        se_ceiling, hi_log, color="#fff3e0", alpha=0.30, zorder=0,
    )
    ax_a.text(
        se_ceiling * 2.5, hi_log * 0.25,
        "Model extrapolation\n(beyond measurement\ndynamic range)",
        fontsize=8, color="#996600", style="italic", ha="center",
    )

    rect_x = [1, se_ceiling, se_ceiling, 1, 1]
    rect_y = [1, 1, se_ceiling, se_ceiling, 1]
    ax_a.plot(
        rect_x, rect_y, color="#2ca02c", linewidth=1.5,
        linestyle="--", zorder=4,
    )
    ax_a.text(
        4, se_ceiling * 0.65, "Validated\nregion (b)",
        fontsize=8, color="#2ca02c", weight="bold",
    )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Experimental SE (dB)")
    ax_a.set_ylabel("Predicted SE (dB)")
    ax_a.set_title("(a) Full dataset (log-log)")
    ax_a.set_xlim(lo_log, hi_log)
    ax_a.set_ylim(lo_log, hi_log)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.legend(loc="lower right", framealpha=0.9, fontsize=8)

    # --- Panel (b): validated subset, linear ---
    for label, d in cat_data.items():
        ex_lo = [e for e, p in zip(d["exps"], d["preds"]) if e <= se_ceiling]
        pr_lo = [p for e, p in zip(d["exps"], d["preds"]) if e <= se_ceiling]
        if not ex_lo:
            continue
        ax_b.scatter(
            ex_lo, pr_lo,
            label=label, color=d["color"],
            marker=d["marker"], s=50, alpha=0.8,
            edgecolors="white", linewidth=0.4, zorder=3,
        )

    arr_lo_e = np.array(low_exp)
    arr_lo_p = np.array(low_pred)

    hi_lin = max(arr_lo_e.max(), arr_lo_p.max()) * 1.1
    diag_lin = np.array([0, hi_lin])
    ax_b.plot(diag_lin, diag_lin, "k-", linewidth=1.2, zorder=2)
    ax_b.fill_between(
        diag_lin, diag_lin - 10, diag_lin + 10,
        color="#CCCCCC", alpha=0.25, label="+/- 10 dB", zorder=1,
    )

    ss_res = np.sum((arr_lo_p - arr_lo_e) ** 2)
    ss_tot = np.sum((arr_lo_e - arr_lo_e.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(arr_lo_p - arr_lo_e)))
    rmse = float(np.sqrt(np.mean((arr_lo_p - arr_lo_e) ** 2)))
    n_lo = len(low_exp)

    textstr = (
        f"$R^2$ = {r2:.3f}\n"
        f"MAE = {mae:.1f} dB\n"
        f"RMSE = {rmse:.1f} dB\n"
        f"n = {n_lo}"
    )
    ax_b.text(
        0.05, 0.95, textstr,
        transform=ax_b.transAxes, fontsize=10.5,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#888888",
            "alpha": 0.9,
        },
    )

    ax_b.set_xlabel("Experimental SE (dB)")
    ax_b.set_ylabel("Predicted SE (dB)")
    ax_b.set_title(
        f"(b) Validated region (SE$_{{exp}}$ < {se_ceiling:.0f} dB)"
    )
    ax_b.set_xlim(0, hi_lin)
    ax_b.set_ylim(0, hi_lin)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    fig.suptitle(
        "Model Validation: Predicted vs Experimental SE",
        fontsize=15, y=1.01,
    )

    out = OUTPUT_DIR / "figure1_parity.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


# -----------------------------------------------------------------------
# Figure 2 -- GEM vs Linear Mixing
# -----------------------------------------------------------------------

def figure2_gem_comparison() -> None:
    """GEM vs linear mixing across the percolation transition."""
    sigma_filler = 1e5
    sigma_matrix = 1e-10
    f_c = 0.01
    t_exp = 2.0
    thickness_m = 2e-3
    freq_hz = 8.2e9

    fracs = np.linspace(1e-4, 0.15, 300)

    sigma_linear = np.empty_like(fracs)
    sigma_gem = np.empty_like(fracs)
    sigma_perc = np.empty_like(fracs)
    se_gem = np.empty_like(fracs)

    for i, f in enumerate(fracs):
        sigma_linear[i] = f * sigma_filler + (1 - f) * sigma_matrix
        sigma_gem[i] = mclachlan_gem(
            f_filler=f, sigma_matrix=sigma_matrix,
            sigma_filler=sigma_filler, f_c=f_c, t=t_exp,
        )
        sigma_perc[i] = percolation_conductivity(
            filler_fraction=f, percolation_threshold=f_c,
            sigma_filler=sigma_filler, sigma_matrix=sigma_matrix,
            t=t_exp,
        )
        res = calculator.calculate_shielding_effectiveness(
            conductivity=sigma_gem[i],
            relative_permeability=1.0,
            relative_permittivity=1.0,
            thickness=thickness_m,
            frequency=freq_hz,
            include_confidence=False,
        )
        se_gem[i] = res["total_se"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    c_lin = "#1f77b4"
    c_gem = "#d62728"
    c_perc = "#2ca02c"
    c_se = "#ff7f0e"

    ax1.semilogy(fracs, sigma_linear, color=c_lin, linewidth=2,
                 label="Linear mixing rule")
    ax1.semilogy(fracs, sigma_gem, color=c_gem, linewidth=2.5,
                 label="GEM (McLachlan)")
    ax1.semilogy(fracs, sigma_perc, color=c_perc, linewidth=2,
                 linestyle="--", label="Percolation power law")
    ax1.axvline(f_c, color="#666666", linestyle=":", linewidth=1.3,
                label=f"Percolation threshold $f_c$ = {f_c}")

    ax1.set_xlabel("Filler volume fraction")
    ax1.set_ylabel("Effective conductivity (S/m)")
    ax1.set_ylim(1e-11, 1e6)

    ax2 = ax1.twinx()
    ax2.plot(fracs, se_gem, color=c_se, linewidth=2, linestyle="-.",
             label="Predicted SE (GEM-based)")
    ax2.set_ylabel("Shielding Effectiveness (dB)", color=c_se)
    ax2.tick_params(axis="y", labelcolor=c_se)

    exp_fracs: list[float] = []
    exp_ses: list[float] = []
    for entry in COMPOSITE_DATA:
        mat_lower = entry["material"].lower()
        if not any(k in mat_lower for k in ("cnt", "mwcnt", "swcnt")):
            continue
        for key in ("MWCNT", "SWCNT"):
            if key in entry["composition"]:
                frac = entry["composition"][key]
                if frac <= 0.15:
                    exp_fracs.append(frac)
                    exp_ses.append(entry["se_db"])

    if exp_fracs:
        ax2.scatter(
            exp_fracs, exp_ses, color=c_se, marker="D", s=65,
            edgecolors="black", linewidth=0.8, zorder=5,
            label="Experimental (CNT/polymer)",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper left", framealpha=0.9, fontsize=9,
    )

    ax1.set_title(
        "Composite Conductivity Models Across Percolation Transition\n"
        r"($\sigma_{\mathrm{filler}}$ = 10$^5$ S/m,  "
        r"$\sigma_{\mathrm{matrix}}$ = 10$^{-10}$ S/m,  "
        r"$f_c$ = 0.01,  $t$ = 2.0)"
    )
    fig.tight_layout()

    out = OUTPUT_DIR / "figure2_gem_comparison.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


# -----------------------------------------------------------------------
# Figure 3 -- Sobol Sensitivity Tornado
# -----------------------------------------------------------------------

def figure3_sobol() -> None:
    """Sobol sensitivity tornado for Cu metal vs CNT/polymer composite."""
    n_samples = 256

    result_cu = sobol_sensitivity(
        conductivity=5.96e7, permeability=1.0, permittivity=1.0,
        thickness=0.01e-3, frequency=1e9,
        n_samples=n_samples, seed=42,
    )
    result_comp = sobol_sensitivity(
        conductivity=10.0, permeability=1.0, permittivity=1.0,
        thickness=2e-3, frequency=8.2e9,
        n_samples=n_samples, seed=42,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = sns.color_palette("deep", n_colors=6)

    pretty = {
        "conductivity": r"$\sigma$ (conductivity)",
        "permeability": r"$\mu_r$ (permeability)",
        "thickness": r"$t$ (thickness)",
        "frequency": r"$f$ (frequency)",
        "grain_size": r"$d_g$ (grain size)",
    }

    cases = [
        (result_cu, "Case A: 10 um Cu at 1 GHz"),
        (result_comp,
         "Case B: CNT/polymer composite\n"
         r"($\sigma$ = 10 S/m, 2 mm, 8.2 GHz)"),
    ]

    for idx, (result, title) in enumerate(cases):
        ax = axes[idx]
        params = result["parameters"]
        st_vals = [result["ST"][p] for p in params]

        pairs = sorted(zip(params, st_vals), key=lambda x: x[1])
        names = [p for p, _ in pairs]
        vals = [v for _, v in pairs]
        labels = [pretty.get(n, n) for n in names]

        colors = [palette[i % len(palette)] for i in range(len(vals))]
        ax.barh(labels, vals, color=colors, edgecolor="white", height=0.55)
        for j, v in enumerate(vals):
            ax.text(v + max(vals) * 0.03, j, f"{v:.3f}",
                    va="center", fontsize=9.5)

        ax.set_xlabel(r"Total-order Sobol index ($S_T$)")
        ax.set_title(title, fontsize=11.5)
        ax.set_xlim(0, max(vals) * 1.35 if max(vals) > 0 else 1.0)
        ax.axvline(0, color="#333333", linewidth=0.5)

    fig.suptitle(
        "Sobol Sensitivity Analysis of SE Prediction",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()

    out = OUTPUT_DIR / "figure3_sobol.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


# -----------------------------------------------------------------------
# Figure 4 -- Frequency Sweep with decomposition
# -----------------------------------------------------------------------

def _collect_exp_points(
    material_key: str,
    thickness_mm: float,
    dataset: Optional[list] = None,
) -> tuple[list[float], list[float]]:
    """Collect experimental (freq_hz, se_db) matching material+thickness."""
    if dataset is None:
        dataset = PURE_METAL_DATA
    freqs: list[float] = []
    ses: list[float] = []
    for entry in dataset:
        mat = entry["material"].lower()
        if material_key not in mat:
            continue
        if abs(entry["thickness_mm"] - thickness_mm) > 0.011:
            continue
        freqs.append(entry["frequency_hz"])
        ses.append(entry["se_db"])
    return freqs, ses


def figure4_frequency_sweep() -> None:
    """Frequency sweep 1 MHz - 10 GHz for three materials.

    Cu and Al at 0.1 mm (non-magnetic, moderate SE) and SS304 at 0.5 mm
    (non-magnetic, lower conductivity). Uses log-scale y-axis to span
    the range from ~50 dB to ~1500 dB while keeping all three curves
    visible. Includes SE decomposition for copper.
    """
    n_pts = 200
    sweep_cu = calculator.frequency_sweep(
        conductivity=5.96e7, relative_permeability=1.0,
        relative_permittivity=1.0, thickness=0.1e-3,
        freq_start=1e6, freq_end=10e9, num_points=n_pts,
    )
    sweep_al = calculator.frequency_sweep(
        conductivity=3.77e7, relative_permeability=1.0,
        relative_permittivity=1.0, thickness=0.1e-3,
        freq_start=1e6, freq_end=10e9, num_points=n_pts,
    )
    sweep_ss = calculator.frequency_sweep(
        conductivity=1.45e6, relative_permeability=1.02,
        relative_permittivity=1.0, thickness=0.5e-3,
        freq_start=1e6, freq_end=10e9, num_points=n_pts,
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    f = sweep_cu["frequencies"]
    ax.loglog(f, sweep_cu["total_ses"], color="#1f77b4", linewidth=2.5,
              label="Cu 0.1 mm -- Total SE")
    ax.loglog(f, sweep_cu["reflection_losses"], color="#1f77b4",
              linewidth=1.2, linestyle="--", alpha=0.55,
              label="Cu -- Reflection")
    ax.loglog(f, sweep_cu["absorption_losses"], color="#1f77b4",
              linewidth=1.2, linestyle=":", alpha=0.55,
              label="Cu -- Absorption")
    ax.loglog(f, sweep_al["total_ses"], color="#ff7f0e", linewidth=2.5,
              label="Al 0.1 mm -- Total SE")
    ax.loglog(sweep_ss["frequencies"], sweep_ss["total_ses"],
              color="#2ca02c", linewidth=2.5,
              label="SS304 0.5 mm -- Total SE")

    exp_cu_f, exp_cu_se = _collect_exp_points("copper", 0.1)
    exp_al_f, exp_al_se = _collect_exp_points("aluminum", 0.1)
    exp_ss_f, exp_ss_se = _collect_exp_points("stainless", 0.5)

    mk = {"edgecolors": "black", "linewidth": 0.6, "zorder": 5, "s": 70}
    ax.scatter(exp_cu_f, exp_cu_se, color="#1f77b4", marker="o",
               label="Cu experimental", **mk)
    ax.scatter(exp_al_f, exp_al_se, color="#ff7f0e", marker="s",
               label="Al experimental", **mk)
    ax.scatter(exp_ss_f, exp_ss_se, color="#2ca02c", marker="^",
               label="SS304 experimental", **mk)

    ax.axhspan(
        120, 2000, color="#fff3e0", alpha=0.3, zorder=0,
    )
    ax.text(
        2e6, 300,
        "Typical measurement ceiling (~120 dB)",
        fontsize=8.5, color="#996600", style="italic",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Shielding Effectiveness (dB)")
    ax.set_title(
        "Frequency Dependence of Shielding Effectiveness\n"
        "(Schelkunoff plane-wave model)"
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5, ncol=2)
    ax.set_ylim(10, 2000)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}")
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    out = OUTPUT_DIR / "figure4_frequency_sweep.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


# -----------------------------------------------------------------------
# Figure 5 -- Monte Carlo Histogram
# -----------------------------------------------------------------------

def figure5_mc_histogram() -> None:
    """MC SE distribution for a CNT/polymer composite (moderate SE)."""
    mc = monte_carlo_se(
        conductivity=100.0,
        permeability=1.0,
        permittivity=1.0,
        thickness=2e-3,
        frequency=8.2e9,
        n_samples=5000,
        seed=42,
    )

    se_dist = mc["se_distribution"]
    mean_se = mc["se_mean"]
    ci_lo = mc["se_ci_lower"]
    ci_hi = mc["se_ci_upper"]
    std_se = mc["se_std"]
    det_se = mc["deterministic_se"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        se_dist, bins=55, color="#4c72b0", alpha=0.72,
        edgecolor="white", linewidth=0.5, zorder=2,
        label="MC distribution",
    )
    ax.axvline(mean_se, color="#222222", linewidth=2.2, linestyle="-",
               label=f"Mean = {mean_se:.2f} dB", zorder=4)
    ax.axvline(det_se, color="#2ca02c", linewidth=1.5, linestyle="-.",
               label=f"Deterministic = {det_se:.2f} dB", zorder=4)
    ax.axvline(ci_lo, color="#d62728", linewidth=1.5, linestyle="--",
               label=f"2.5th pct = {ci_lo:.2f} dB", zorder=4)
    ax.axvline(ci_hi, color="#d62728", linewidth=1.5, linestyle="--",
               label=f"97.5th pct = {ci_hi:.2f} dB", zorder=4)
    ax.axvspan(ci_lo, ci_hi, color="#d62728", alpha=0.07, zorder=1)

    textstr = (
        f"Mean: {mean_se:.2f} dB\n"
        f"Std:  {std_se:.2f} dB\n"
        f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}] dB\n"
        f"Deterministic: {det_se:.2f} dB\n"
        f"n = {mc['n_samples']}"
    )
    ax.text(
        0.03, 0.97, textstr,
        transform=ax.transAxes, fontsize=10.5,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#888888",
            "alpha": 0.9,
        },
    )

    ax.set_xlabel("Shielding Effectiveness (dB)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Monte Carlo Uncertainty Propagation\n"
        r"CNT/epoxy composite ($\sigma$=100 S/m, "
        "2 mm, 8.2 GHz, 5000 samples)"
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    out = OUTPUT_DIR / "figure5_mc_histogram.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {out.name}  ({out.stat().st_size / 1024:.0f} KB)")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    """Generate all five publication figures."""
    _journal_style()
    print("Generating publication figures...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    figure1_parity()
    figure2_gem_comparison()
    figure3_sobol()
    figure4_frequency_sweep()
    figure5_mc_histogram()

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
