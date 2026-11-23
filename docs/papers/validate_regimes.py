"""Regime-stratified validation of the single-layer model against the benchmark set.

Reproduces every validation number quoted in the manuscript from the committed
code and data. Run from the repository root:

    python docs/papers/validate_regimes.py

Writes docs/papers/validation_regimes.csv (one row per benchmark entry) and
docs/papers/validation_regimes.txt (summary metrics).

Regimes (dimensionless thickness t/delta at the measurement frequency):
    A  directly measurable : t/delta <= 3  and SE_exp <= 80 dB
    B  near ceiling        : not A, t/delta <= 10 and SE_exp <= 120 dB
    C  extrapolation       : everything else
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_collection.experimental_benchmark_data import (  # noqa: E402
    COMPOSITE_DATA, FREQUENCY_BAND_DATA, MICROSTRUCTURE_EFFECTS_DATA,
    MULTILAYER_DATA, PURE_METAL_DATA, TEMPERATURE_EFFECTS_DATA,
)
from src.physics.emi_calculations import EMICalculator  # noqa: E402
from src.utils.constants import MU_0  # noqa: E402

OUT_CSV = Path(__file__).with_name("validation_regimes.csv")
OUT_TXT = Path(__file__).with_name("validation_regimes.txt")

CATEGORIES = [
    ("Pure metals", PURE_METAL_DATA),
    ("Composites", COMPOSITE_DATA),
    ("Multilayer", MULTILAYER_DATA),
    ("Temperature", TEMPERATURE_EFFECTS_DATA),
    ("Microstructure", MICROSTRUCTURE_EFFECTS_DATA),
    ("Frequency bands", FREQUENCY_BAND_DATA),
]
MAGNETIC_MU_THRESHOLD = 5.0


def regime(t_over_delta: float, se_exp: float) -> str:
    if t_over_delta <= 3 and se_exp <= 80:
        return "A"
    if t_over_delta <= 10 and se_exp <= 120:
        return "B"
    return "C"


def build_rows() -> tuple[list[dict], dict[str, int], dict[str, int]]:
    calc = EMICalculator()
    rows, totals, skipped = [], {}, {}
    for category, dataset in CATEGORIES:
        totals[category] = len(dataset)
        for entry in dataset:
            sigma = entry.get("conductivity_sm")
            usable = (isinstance(sigma, (int, float)) and sigma > 0
                      and "thickness_mm" in entry and "frequency_hz" in entry)
            if not usable:
                skipped[category] = skipped.get(category, 0) + 1
                continue
            mu_r = max(float(entry.get("permeability", 1.0) or 1.0), 0.999)
            t = entry["thickness_mm"] * 1e-3
            f = entry["frequency_hz"]
            res = calc.calculate_shielding_effectiveness(
                sigma, mu_r, 1.0, t, f, include_confidence=False)
            delta = math.sqrt(2 / (2 * math.pi * f * mu_r * MU_0 * sigma))
            se_exp = float(entry["se_db"])
            rows.append({
                "category": category,
                "material": entry["material"],
                "frequency_hz": f,
                "thickness_mm": entry["thickness_mm"],
                "conductivity_sm": sigma,
                "permeability": mu_r,
                "t_over_delta": t / delta,
                "regime": regime(t / delta, se_exp),
                "se_exp_db": se_exp,
                "se_pred_db": float(res["total_se"]),
                "se_r_db": float(res["reflection_loss"]),
                "se_a_db": float(res["absorption_loss"]),
                "se_m_db": float(res["multiple_reflection_loss"]),
                "error_db": float(res["total_se"]) - se_exp,
                "source": entry.get("source", ""),
            })
    return rows, totals, skipped


def metrics(rows: list[dict]) -> str:
    if not rows:
        return "n=0"
    e = np.array([r["se_exp_db"] for r in rows])
    d = np.array([r["error_db"] for r in rows])
    r2 = 1 - np.sum(d ** 2) / np.sum((e - e.mean()) ** 2) if len(rows) > 2 else float("nan")
    return (f"n={len(rows)}  MAE={np.abs(d).mean():.1f}  RMSE={math.sqrt((d ** 2).mean()):.1f}  "
            f"median|err|={np.median(np.abs(d)):.1f}  bias={d.mean():+.1f}  "
            f"max|err|={np.abs(d).max():.1f}  R2={r2:.2f}")


def main() -> None:
    rows, totals, skipped = build_rows()
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    out = []
    out.append("Benchmark entries per category (total / not usable for single-layer prediction):")
    for category, _ in CATEGORIES:
        out.append(f"  {category:<16} {totals[category]:>3} / {skipped.get(category, 0)}")
    out.append(f"  {'All':<16} {sum(totals.values()):>3} / {sum(skipped.values())}   usable = {len(rows)}")
    out.append("")
    for reg in "ABC":
        out.append(f"Regime {reg}: {metrics([r for r in rows if r['regime'] == reg])}")
    reg_a = [r for r in rows if r["regime"] == "A"]
    out.append("")
    out.append("Regime A by category:")
    for category, _ in CATEGORIES:
        sub = [r for r in reg_a if r["category"] == category]
        if sub:
            out.append(f"  {category:<16} {metrics(sub)}")
    nonmag = [r for r in reg_a if r["category"] == "Pure metals" and r["permeability"] < MAGNETIC_MU_THRESHOLD]
    out.append(f"  {'Non-mag. metals':<16} {metrics(nonmag)}")
    out.append("")
    out.append("Regime A entries:")
    out.append(f"  {'material':<44} {'f (Hz)':>8} {'t (mm)':>8} {'t/delta':>8} {'exp':>7} {'pred':>7} {'err':>7}")
    for r in sorted(reg_a, key=lambda r: (r["category"], r["material"], r["frequency_hz"])):
        out.append(f"  {r['material'][:44]:<44} {r['frequency_hz']:>8.1e} {r['thickness_mm']:>8.4f} "
                   f"{r['t_over_delta']:>8.2f} {r['se_exp_db']:>7.1f} {r['se_pred_db']:>7.1f} {r['error_db']:>+7.1f}")
    text = "\n".join(out)
    OUT_TXT.write_text(text + "\n")
    print(text)
    print(f"\nWrote {OUT_CSV.name} and {OUT_TXT.name}")


if __name__ == "__main__":
    main()
