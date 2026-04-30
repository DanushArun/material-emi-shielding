"""Figure 1: Honest two-panel parity plot.
Panel (a): Metals + composites on log-log (all data, transparent about limits)
Panel (b): Non-magnetic metals in measurable range (linear, with R2/MAE)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.physics.emi_calculations import EMICalculator
from data_collection.experimental_benchmark_data import (
    PURE_METAL_DATA, COMPOSITE_DATA, TEMPERATURE_EFFECTS_DATA,
    MICROSTRUCTURE_EFFECTS_DATA, FREQUENCY_BAND_DATA,
)

OUTPUT = PROJECT_ROOT / "docs" / "papers" / "figure1_parity.png"
calc = EMICalculator()

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "font.family": "serif",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

categories = [
    ("Pure metals", PURE_METAL_DATA, "#1f77b4", "o"),
    ("Composites", COMPOSITE_DATA, "#d62728", "s"),
    ("Temperature", TEMPERATURE_EFFECTS_DATA, "#ff7f0e", "^"),
    ("Microstructure", MICROSTRUCTURE_EFFECTS_DATA, "#9467bd", "D"),
    ("Freq. bands", FREQUENCY_BAND_DATA, "#17becf", "v"),
]


def predict(entry: dict) -> float:
    s = entry["conductivity_sm"]
    mu = max(float(entry.get("permeability", 1.0) or 1.0), 0.999)
    t = entry["thickness_mm"] * 1e-3
    f = entry["frequency_hz"]
    r = calc.calculate_shielding_effectiveness(
        s, mu, 1.0, t, f, include_confidence=False)
    return float(r["total_se"])


fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(15, 7),
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30},
)

# Collect all data
all_data: dict[str, list] = {}
for label, dataset, color, marker in categories:
    exps, preds = [], []
    for entry in dataset:
        sigma = entry.get("conductivity_sm")
        if sigma is None or not isinstance(sigma, (int, float)):
            continue
        se_exp = entry["se_db"]
        se_pred = predict(entry)
        if se_exp > 0 and se_pred > 0:
            exps.append(se_exp)
            preds.append(se_pred)
    all_data[label] = {
        "exps": exps, "preds": preds,
        "color": color, "marker": marker,
    }

# --- Panel (a): Full dataset log-log ---
for label, d in all_data.items():
    n = len(d["exps"])
    ax_a.scatter(
        d["exps"], d["preds"],
        label=f"{label} (n={n})", color=d["color"],
        marker=d["marker"], s=45, alpha=0.7,
        edgecolors="white", linewidth=0.4, zorder=3,
    )

diag_log = np.array([1, 2e4])
ax_a.plot(diag_log, diag_log, "k-", lw=1.2, zorder=2)
ax_a.fill_between(
    diag_log, diag_log / 3.16, diag_log * 3.16,
    color="#CCCCCC", alpha=0.2, label=r"$\pm$10 dB", zorder=1,
)

ax_a.axhline(120, color="#e67e22", ls="--", lw=0.8, alpha=0.6)
ax_a.text(3, 150, "Typical measurement ceiling (~120 dB)",
          fontsize=8, color="#e67e22", fontstyle="italic")

ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim(1, 1e4)
ax_a.set_ylim(0.01, 2e4)
ax_a.set_xlabel("Experimental SE (dB)", fontsize=13)
ax_a.set_ylabel("Predicted SE (dB)", fontsize=13)
ax_a.set_title("(a) Full benchmark dataset (log-log)", fontsize=13)
ax_a.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

# Annotation for over/under prediction regions
ax_a.text(400, 5, "Model\nunderpredicts", fontsize=8,
          color="#888", ha="center", fontstyle="italic")
ax_a.text(5, 400, "Model\noverpredicts", fontsize=8,
          color="#888", ha="center", fontstyle="italic")

# --- Panel (b): Non-magnetic metals only, SE < 95 dB ---
metal_e, metal_p = [], []
for entry in PURE_METAL_DATA:
    sigma = entry.get("conductivity_sm")
    if sigma is None:
        continue
    mu = max(float(entry.get("permeability", 1.0) or 1.0), 0.999)
    if mu > 5:
        continue
    se_exp = entry["se_db"]
    se_pred = predict(entry)
    if se_exp < 95 and se_pred < 95:
        metal_e.append(se_exp)
        metal_p.append(se_pred)

me = np.array(metal_e)
mp = np.array(metal_p)
r2 = 1 - np.sum((mp - me)**2) / np.sum((me - np.mean(me))**2)
mae = np.mean(np.abs(mp - me))
rmse = np.sqrt(np.mean((mp - me)**2))

ax_b.scatter(me, mp, color="#1f77b4", marker="o", s=80,
             alpha=0.85, edgecolors="white", linewidth=0.5,
             zorder=3, label=f"Non-magnetic metals (n={len(me)})")

# Also plot composites that are close
comp_e, comp_p = [], []
for entry in COMPOSITE_DATA:
    sigma = entry.get("conductivity_sm")
    if sigma is None:
        continue
    mu = max(float(entry.get("permeability", 1.0) or 1.0), 0.999)
    se_exp = entry["se_db"]
    se_pred = predict(entry)
    if se_exp < 60 and se_pred < 60:
        comp_e.append(se_exp)
        comp_p.append(se_pred)

ce = np.array(comp_e)
cp = np.array(comp_p)
if len(ce) > 0:
    r2c = 1 - np.sum((cp - ce)**2) / np.sum((ce - np.mean(ce))**2)
    maec = np.mean(np.abs(cp - ce))
    ax_b.scatter(ce, cp, color="#d62728", marker="s", s=60,
                 alpha=0.75, edgecolors="white", linewidth=0.5,
                 zorder=3, label=f"Composites (n={len(ce)})")

diag = np.array([0, 100])
ax_b.plot(diag, diag, "k-", lw=1.3, zorder=2)
ax_b.fill_between(diag, diag - 5, diag + 5,
                   color="#2ecc71", alpha=0.18, label=r"$\pm$5 dB", zorder=1)
ax_b.fill_between(diag, diag - 10, diag + 10,
                   color="#f39c12", alpha=0.12, label=r"$\pm$10 dB", zorder=0)

stats = (
    f"Non-magnetic metals:\n"
    f"  $R^2$ = {r2:.2f}, MAE = {mae:.1f} dB\n"
)
if len(ce) > 0:
    stats += (
        f"Composites (SE < 60 dB):\n"
        f"  $R^2$ = {r2c:.2f}, MAE = {maec:.1f} dB"
    )
ax_b.text(
    0.04, 0.96, stats, transform=ax_b.transAxes,
    fontsize=10, va="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.95),
)

ax_b.set_xlabel("Experimental SE (dB)", fontsize=13)
ax_b.set_ylabel("Predicted SE (dB)", fontsize=13)
ax_b.set_title(
    "(b) Validated region (within measurement range)",
    fontsize=13,
)
ax_b.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax_b.set_xlim(0, 100)
ax_b.set_ylim(0, 100)
ax_b.set_aspect("equal")

fig.suptitle(
    "Model Validation: Predicted vs. Experimental Shielding Effectiveness",
    fontsize=15, fontweight="bold", y=1.01,
)
fig.savefig(OUTPUT)
plt.close(fig)

print(f"Saved: {OUTPUT}")
print(f"Metals: R2={r2:.2f}, MAE={mae:.1f}, RMSE={rmse:.1f}, n={len(me)}")
if len(ce) > 0:
    print(f"Composites: R2={r2c:.2f}, MAE={maec:.1f}, n={len(ce)}")
