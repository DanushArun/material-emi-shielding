"""Regression tests: single-layer Schelkunoff, TMM and the exact slab formula must agree.

For a homogeneous slab in free space at normal incidence the Schelkunoff
decomposition R + A + B is algebraically identical to the exact transmission
loss, and to a one-layer transfer-matrix calculation. These tests pin that
identity so the reflection term can never silently revert to a
single-interface form.
"""
import cmath
import math

import numpy as np
import pytest

from src.physics.emi_calculations import EMICalculator
from src.physics.multilayer import MultilayerShield, ShieldLayer
from src.utils.constants import EPSILON_0, MU_0, Z_0

CU = 5.96e7


def exact_slab_se(sigma: float, mu_r: float, eps_r: float, t: float, f: float) -> float:
    """Independent reference: exact plane-wave transmission loss of a slab (dB)."""
    w = 2 * math.pi * f
    mu = mu_r * MU_0
    eps_c = eps_r * EPSILON_0 - 1j * sigma / w
    gamma = 1j * w * cmath.sqrt(mu * eps_c)
    eta = cmath.sqrt(mu / eps_c)
    x = gamma * t
    back = cmath.exp(-2 * x) if x.real < 300 else 0.0
    denom = (Z_0 + eta) ** 2 - (Z_0 - eta) ** 2 * back
    return 20 * math.log10(abs(denom / (4 * Z_0 * eta))) + (20 / math.log(10)) * x.real


def schelkunoff(sigma, mu_r, eps_r, t, f):
    return EMICalculator().calculate_shielding_effectiveness(
        sigma, mu_r, eps_r, t, f, include_confidence=False)


def tmm_se(layers, f):
    shield = MultilayerShield()
    for sigma, mu_r, eps_r, t in layers:
        shield.add_layer(ShieldLayer(conductivity=sigma, relative_permeability=mu_r,
                                     relative_permittivity=eps_r, thickness=t))
    return shield.calculate_se(f)["total_se"]


CASES = [
    # sigma, mu_r, eps_r, thickness (m), frequency (Hz)
    (CU, 1.0, 1.0, 0.5e-6, 1e9),     # thin film, t << skin depth
    (CU, 1.0, 1.0, 10e-6, 1e9),
    (CU, 1.0, 1.0, 0.1e-3, 1e6),     # foil at low frequency
    (CU, 1.0, 1.0, 1e-3, 1e9),       # thick conductor
    (3.77e7, 1.0, 1.0, 0.1e-3, 1e7),
    (1.45e6, 1.02, 1.0, 0.5e-3, 1e6),
    (6.99e6, 300.0, 1.0, 0.5e-3, 1e6),   # ferromagnet
    (100.0, 1.0, 3.0, 2e-3, 8.2e9),      # lossy composite
    (1.0, 1.0, 3.0, 1e-3, 8.2e9),        # weakly conducting composite
    (1e-3, 1.0, 3.0, 2e-3, 1e9),         # near-insulator
]


@pytest.mark.parametrize("sigma,mu_r,eps_r,t,f", CASES)
def test_schelkunoff_matches_exact_slab(sigma, mu_r, eps_r, t, f):
    result = schelkunoff(sigma, mu_r, eps_r, t, f)
    assert result["total_se"] == pytest.approx(exact_slab_se(sigma, mu_r, eps_r, t, f), abs=1e-6)


@pytest.mark.parametrize("sigma,mu_r,eps_r,t,f", CASES)
def test_schelkunoff_matches_single_layer_tmm(sigma, mu_r, eps_r, t, f):
    result = schelkunoff(sigma, mu_r, eps_r, t, f)
    assert result["total_se"] == pytest.approx(tmm_se([(sigma, mu_r, eps_r, t)], f), abs=1e-6)


@pytest.mark.parametrize("sigma,mu_r,eps_r,t,f", CASES)
def test_components_sum_to_total_and_se_is_non_negative(sigma, mu_r, eps_r, t, f):
    r = schelkunoff(sigma, mu_r, eps_r, t, f)
    total = r["reflection_loss"] + r["absorption_loss"] + r["multiple_reflection_loss"]
    assert r["total_se"] == pytest.approx(total, abs=1e-9)
    assert r["total_se"] >= -1e-9  # a passive slab cannot amplify


def test_thin_film_limit_copper_half_micron_1ghz():
    """t << skin depth: SE -> 20 log10(1 + Z0*sigma*t/2), about 75 dB here."""
    expected = 20 * math.log10(1 + Z_0 * CU * 0.5e-6 / 2)
    assert schelkunoff(CU, 1.0, 1.0, 0.5e-6, 1e9)["total_se"] == pytest.approx(expected, abs=0.1)
    assert expected == pytest.approx(75.0, abs=0.1)


def test_good_conductor_reflection_loss_counts_both_interfaces():
    """Thick good conductor: R = 20 log10(Z0 / (4|eta|)); ~108 dB for Cu at 1 MHz."""
    r = schelkunoff(CU, 1.0, 1.0, 1e-3, 1e6)
    eta = abs(cmath.sqrt(1j * 2 * math.pi * 1e6 * MU_0 / CU))
    assert r["reflection_loss"] == pytest.approx(20 * math.log10(Z_0 / (4 * eta)), abs=0.01)
    assert r["reflection_loss"] == pytest.approx(108.3, abs=0.1)


def test_multiple_reflection_vanishes_for_thick_shield():
    assert abs(schelkunoff(CU, 1.0, 1.0, 1e-3, 1e9)["multiple_reflection_loss"]) < 1e-6


@pytest.mark.parametrize("t", [0.2e-3, 0.5e-3, 1.0e-3, 1.3e-3, 5e-3])
def test_tmm_overflow_branch_matches_exact(t):
    """Layers thick enough to trigger the scaled (overflow-safe) TMM branch."""
    assert tmm_se([(CU, 1.0, 1.0, t)], 1e9) == pytest.approx(
        exact_slab_se(CU, 1.0, 1.0, t, 1e9), rel=1e-9)


def test_tmm_is_continuous_across_overflow_threshold():
    """Re(gamma*d) = 200 is the branch switch; SE must not jump there."""
    delta = math.sqrt(2 / (2 * math.pi * 1e9 * MU_0 * CU))
    below = tmm_se([(CU, 1.0, 1.0, 199.999 * delta)], 1e9)
    above = tmm_se([(CU, 1.0, 1.0, 200.001 * delta)], 1e9)
    assert above - below == pytest.approx(0.002 * 20 / math.log(10), abs=1e-3)


def test_splitting_a_layer_does_not_change_tmm_result():
    whole = tmm_se([(CU, 1.0, 1.0, 1e-3)], 1e9)
    halves = tmm_se([(CU, 1.0, 1.0, 0.5e-3), (CU, 1.0, 1.0, 0.5e-3)], 1e9)
    assert halves == pytest.approx(whole, rel=1e-9)


def test_sandwich_is_below_sum_of_isolated_layers():
    """Cu/PET/Cu 0.5/100/0.5 um at 1 GHz: 102.5 dB, less than 2 x 75 dB."""
    sandwich = tmm_se([(CU, 1.0, 1.0, 0.5e-6), (0.0, 1.0, 3.0, 100e-6), (CU, 1.0, 1.0, 0.5e-6)], 1e9)
    single = schelkunoff(CU, 1.0, 1.0, 0.5e-6, 1e9)["total_se"]
    assert sandwich == pytest.approx(102.5, abs=0.1)
    assert sandwich < 2 * single
