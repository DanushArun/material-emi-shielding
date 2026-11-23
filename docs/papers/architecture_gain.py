"""Architecture gain: measured SE minus the exact homogeneous-slab prediction.

For every entry in data_collection/verified_benchmark.csv (values traceable to
a quoted sentence, table or caption in the cited source) this computes the
plane-wave shielding effectiveness of a homogeneous slab with the reported
conductivity and thickness, and reports

    gain = SE_measured - SE_slab

A gain near zero means the sample behaves as a plain conductor; a large
positive gain means its structure adds shielding beyond its bulk conductivity;
a large negative gain means the reported conductivity, thickness and SE are
mutually inconsistent with a homogeneous slab.

Run from the repository root:  python docs/papers/architecture_gain.py
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.physics.emi_calculations import EMICalculator  # noqa: E402
from src.utils.constants import MU_0  # noqa: E402

DATA = ROOT / "data_collection" / "verified_benchmark.csv"
OUT = Path(__file__).with_name("architecture_gain.csv")
BAND_POINTS = 41


def main() -> None:
    calc = EMICalculator()
    rows = list(csv.DictReader(DATA.open()))
    out = []
    for r in rows:
        sigma, t = float(r["sigma_s_per_m"]), float(r["thickness_m"])
        f_lo, f_hi = float(r["f_low_hz"]), float(r["f_high_hz"])
        freqs = [f_lo] if f_lo == f_hi else list(np.linspace(f_lo, f_hi, BAND_POINTS))
        preds = [calc.calculate_shielding_effectiveness(
            sigma, 1.0, 1.0, t, f, include_confidence=False)["total_se"] for f in freqs]
        f_mid = 0.5 * (f_lo + f_hi)
        delta = math.sqrt(2 / (2 * math.pi * f_mid * MU_0 * sigma))
        pred = float(np.mean(preds))
        out.append({
            "id": r["id"], "material": r["material"], "material_class": r["material_class"],
            "t_over_delta": t / delta, "se_measured_db": float(r["se_db"]),
            "se_slab_db": pred, "se_slab_min_db": float(min(preds)), "se_slab_max_db": float(max(preds)),
            "gain_db": float(r["se_db"]) - pred, "access": r["access"], "citation": r["citation"],
        })
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for o in out:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in o.items()})
    print(f"{'material':<38} {'t/delta':>7} {'measured':>9} {'slab':>7} {'gain':>7}  source")
    for o in out:
        print(f"{o['material'][:38]:<38} {o['t_over_delta']:>7.2f} {o['se_measured_db']:>9.1f} "
              f"{o['se_slab_db']:>7.1f} {o['gain_db']:>+7.1f}  {o['citation'].split(',')[0]}")
    g = np.array([o["gain_db"] for o in out])
    print(f"\nn={len(out)}  mean |gain|={np.abs(g).mean():.1f} dB  median |gain|={np.median(np.abs(g)):.1f} dB")
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()
