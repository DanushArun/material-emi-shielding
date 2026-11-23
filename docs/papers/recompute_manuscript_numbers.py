"""Recompute every model-derived number quoted in the manuscript.

Run from the repository root:  python docs/papers/recompute_manuscript_numbers.py
Writes docs/papers/manuscript_numbers.txt.
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.physics.emi_calculations import EMICalculator  # noqa: E402
from src.physics.material_models import (  # noqa: E402
    freq_dependent_permeability, snoek_resonance_frequency,
)
from src.physics.multilayer import MultilayerShield, ShieldLayer  # noqa: E402
from src.physics.uncertainty import UncertaintySpec, monte_carlo_se  # noqa: E402
from src.utils.constants import MU_0  # noqa: E402

OUT = Path(__file__).with_name("manuscript_numbers.txt")
calc = EMICalculator()
lines: list[str] = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text)


def se(sigma, mu_r, t, f, eps_r=1.0, grain=None):
    return calc.calculate_shielding_effectiveness(
        sigma, mu_r, eps_r, t, f, grain_size=grain, include_confidence=False)


def tmm(layers, f):
    shield = MultilayerShield()
    for sigma, eps_r, t in layers:
        shield.add_layer(ShieldLayer(conductivity=sigma, relative_permeability=1.0,
                                     relative_permittivity=eps_r, thickness=t))
    return shield.calculate_se(f)


def skin_depth(sigma, mu_r, f):
    return math.sqrt(2 / (2 * math.pi * f * mu_r * MU_0 * sigma))


def min_thickness(sigma, mu_r, f, target):
    lo, hi = 1e-10, 1e-1
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if se(sigma, mu_r, mid, f)["total_se"] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def sobol_total(nominal: dict, cv: dict, n: int = 4096, seed: int = 42) -> dict:
    """Saltelli total-order indices with normal inputs of given coefficient of variation."""
    names = list(nominal)
    d = len(names)
    u = qmc.Sobol(2 * d, scramble=True, seed=seed).random(n)
    z = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))

    def phys(zm):
        cols = {}
        for i, k in enumerate(names):
            v = nominal[k] * (1 + cv[k] * zm[:, i])
            floor = 0.999 if k == "mu" else nominal[k] * 1e-3
            cols[k] = np.maximum(v, floor)
        return cols

    def run(cols):
        return np.array([se(cols["sigma"][i], cols["mu"][i], cols["t"][i], cols["f"][i])["total_se"]
                         for i in range(n)])

    za, zb = z[:, :d], z[:, d:]
    fa, fb = run(phys(za)), run(phys(zb))
    var = np.var(np.concatenate([fa, fb]), ddof=1)
    out = {}
    for i, k in enumerate(names):
        zab = za.copy()
        zab[:, i] = zb[:, i]
        out[k] = float(np.mean((fa - run(phys(zab))) ** 2) / (2 * var))
    out["_mean"], out["_std"] = float(fa.mean()), float(fa.std(ddof=1))
    return out


CU = 5.96e7

say("== Section 2.2 / 4.5: single layer and Cu/PET/Cu sandwich, 1 GHz ==")
single = se(CU, 1.0, 0.5e-6, 1e9)
say(f"0.5 um Cu, Schelkunoff: total {single['total_se']:.2f} dB "
    f"(R {single['reflection_loss']:.2f}, A {single['absorption_loss']:.2f}, B {single['multiple_reflection_loss']:.2f})")
say(f"0.5 um Cu, TMM: {tmm([(CU, 1.0, 0.5e-6)], 1e9)['total_se']:.2f} dB")
sand = tmm([(CU, 1.0, 0.5e-6), (0.0, 3.0, 100e-6), (CU, 1.0, 0.5e-6)], 1e9)
say(f"Cu(0.5 um)/PET(100 um)/Cu(0.5 um), TMM: {sand['total_se']:.2f} dB")
say(f"Single 1.0 um Cu layer (same total metal): {se(CU, 1.0, 1.0e-6, 1e9)['total_se']:.2f} dB")
say(f"Sum of two isolated 0.5 um layers: {2 * single['total_se']:.2f} dB")
say("Sandwich vs PET spacer thickness at 1 GHz and 10 GHz:")
for f in (1e9, 1e10):
    vals = ", ".join(f"{d * 1e6:.0f} um: {tmm([(CU, 1.0, 0.5e-6), (0.0, 3.0, d), (CU, 1.0, 0.5e-6)], f)['total_se']:.1f}"
                     for d in (1e-6, 10e-6, 100e-6, 1e-3, 5e-3))
    say(f"  {f:.0e} Hz -> {vals}")

say("\n== Section 2.4: permeability and temperature ==")
fr = snoek_resonance_frequency(1e5, 8.6e5)
mu_1ghz = abs(freq_dependent_permeability(1e5, 1e9, M_s=8.6e5))
say(f"Mu-metal Snoek resonance: {fr / 1e6:.3f} MHz; |mu_r| at 1 GHz: {mu_1ghz:.1f}")
say(f"Mu-metal 1 mm, 1 GHz, static mu_r=1e5: {se(1.82e6, 1e5, 1e-3, 1e9)['total_se']:.0f} dB; "
    f"with |mu_r|={mu_1ghz:.1f}: {se(1.82e6, mu_1ghz, 1e-3, 1e9)['total_se']:.0f} dB")
s423 = CU / (1 + 0.00393 * (423.15 - 293.15))
say(f"Cu conductivity at 423 K: {s423:.3e} S/m ({100 * (1 - s423 / CU):.1f}% below 293 K)")
for t, f in ((1e-3, 1e9), (10e-6, 1e9), (0.1e-3, 1e6)):
    say(f"  Cu {t * 1e6:g} um at {f:.0e} Hz: 293 K {se(CU, 1, t, f)['total_se']:.1f} dB, "
        f"423 K {se(s423, 1, t, f)['total_se']:.1f} dB")

say("\n== Section 2.5: Mayadas-Shatzkes grain-boundary correction (Cu, l = 40 nm, R = 0.25) ==")
for dg in (30e-9, 100e-9, 1e-6, 50e-6):
    say(f"  d_g = {dg * 1e9:g} nm: sigma_eff/sigma_bulk = {calc.calculate_grain_size_effect(CU, dg) / CU:.3f}")
say(f"  Cu 0.1 mm, 1 GHz: d_g 30 nm {se(CU, 1, 1e-4, 1e9, grain=30e-9)['total_se']:.0f} dB, "
    f"d_g 50 um {se(CU, 1, 1e-4, 1e9, grain=50e-6)['total_se']:.0f} dB")

say("\n== Section 4.4: ferromagnets ==")
for name, sigma, mu_s, ms, t, f in (("Mild steel 1018", 6.99e6, 300, 1.7e6, 0.5e-3, 1e6),
                                    ("Nickel", 1.44e7, 50, 4.9e5, 1e-3, 1e9)):
    mu_f = abs(freq_dependent_permeability(mu_s, f, M_s=ms))
    say(f"{name}, {t * 1e3:g} mm, {f:.0e} Hz: static mu_r={mu_s}: {se(sigma, mu_s, t, f)['total_se']:.0f} dB; "
        f"Snoek-Debye |mu_r|={mu_f:.1f} (f_r {snoek_resonance_frequency(mu_s, ms) / 1e6:.0f} MHz): "
        f"{se(sigma, mu_f, t, f)['total_se']:.0f} dB; mu_r=1: {se(sigma, 1, t, f)['total_se']:.0f} dB")

say("\n== Section 4.6: total-order Sobol indices, class-specific CVs ==")
cases = [
    ("Cu 10 um, 1 GHz", dict(sigma=CU, mu=1.0, t=10e-6, f=1e9), dict(sigma=0.05, mu=1e-4, t=0.02, f=0.001)),
    ("Mild steel 1 mm, 1 MHz", dict(sigma=6.99e6, mu=300.0, t=1e-3, f=1e6), dict(sigma=0.08, mu=0.20, t=0.02, f=0.001)),
    ("CNT/polymer 10 S/m, 2 mm, 8.2 GHz", dict(sigma=10.0, mu=1.0, t=2e-3, f=8.2e9), dict(sigma=0.30, mu=1e-4, t=0.05, f=0.001)),
]
for name, nominal, cv in cases:
    st = sobol_total(nominal, cv)
    say(f"{name}: SE {st['_mean']:.1f} +/- {st['_std']:.1f} dB; "
        + ", ".join(f"ST[{k}]={st[k]:.3f}" for k in nominal))

say("\n== Section 6.2: minimum thickness for 80 dB at 10 GHz (plane wave, homogeneous sheet) ==")
for name, sigma, rho in (("Copper", 5.96e7, 8960), ("Aluminium 6061", 2.5e7, 2700),
                         ("Beryllium copper", 1.4e7, 8250), ("Stainless 304", 1.45e6, 8000)):
    t = min_thickness(sigma, 1.0, 1e10, 80.0)
    say(f"  {name}: {t * 1e6:.2f} um, areal density {rho * t * 1e3:.2f} g/m2 "
        f"(skin depth {skin_depth(sigma, 1, 1e10) * 1e6:.2f} um)")

say("\n== Section 7: design examples ==")
big = tmm([(CU, 1.0, 12e-6), (0.0, 3.0, 500e-6), (CU, 1.0, 12e-6)], 1e10)
say(f"Cu(12 um)/PET(500 um)/Cu(12 um), 10 GHz: {big['total_se']:.0f} dB; "
    f"areal density {(8960 * 24e-6 + 1380 * 500e-6):.2f} kg/m2")
for f in (1e6, 1e8, 1e9, 1e10):
    say(f"  Cu 0.1 mm at {f:.0e} Hz: {se(CU, 1, 1e-4, f)['total_se']:.0f} dB")
say("CNT/epoxy sheet, sigma = 100 S/m, 2 mm, 8.2 GHz:")
det = se(100.0, 1.0, 2e-3, 8.2e9)["total_se"]
mc = monte_carlo_se(100.0, 1.0, 1.0, 2e-3, 8.2e9,
                    uncertainty=UncertaintySpec(conductivity_cv=0.30, thickness_cv=0.05, permeability_cv=1e-4),
                    n_samples=5000, seed=42)
keys = {k: v for k, v in mc.items() if isinstance(v, (int, float))}
say(f"  deterministic {det:.1f} dB; Monte Carlo summary: "
    + ", ".join(f"{k}={v:.2f}" for k, v in keys.items()))

OUT.write_text("\n".join(lines) + "\n")
print(f"\nWrote {OUT.name}")
