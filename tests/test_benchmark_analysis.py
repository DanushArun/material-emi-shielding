"""The benchmark analysis must agree with the physics package and with known limits."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "papers"))

from benchmark_analysis import Z0, cutoff, effective_sigma, predict, slab_se  # noqa: E402
from src.physics.emi_calculations import EMICalculator  # noqa: E402

CASES = [(5.96e7, 0.5e-6, 1e9), (5.96e7, 1e-3, 1e9), (1.0e6, 5e-6, 10e9), (100.0, 2e-3, 8.2e9),
         (1.0, 1e-3, 8.2e9), (1e-3, 2e-3, 1e9)]


@pytest.mark.parametrize("sigma,t,f", CASES)
def test_free_space_matches_schelkunoff(sigma, t, f):
    ref = EMICalculator().calculate_shielding_effectiveness(sigma, 1.0, 1.0, t, f, include_confidence=False)["total_se"]
    assert slab_se(sigma, t, f) == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("sigma,t,f", CASES)
def test_te10_reduces_to_free_space_as_cutoff_vanishes(sigma, t, f):
    assert slab_se(sigma, t, f, fc=f * 1e-6) == pytest.approx(slab_se(sigma, t, f), abs=1e-6)


@pytest.mark.parametrize("f", [8.2e9, 10e9, 12.4e9, 15e9])
def test_te10_offset_for_a_reflecting_sheet(f):
    """Thin good conductor: waveguide SE exceeds free-space SE by 20 log10(Z_TE / Z0)."""
    fc = cutoff(f)
    z_te = Z0 / math.sqrt(1 - (fc / f) ** 2)
    offset = slab_se(1e6, 5e-6, f, fc) - slab_se(1e6, 5e-6, f)
    assert offset == pytest.approx(20 * math.log10(z_te / Z0), abs=0.05)


def test_standard_waveguide_cutoffs():
    assert cutoff(10e9) == pytest.approx(6.557e9)
    assert cutoff(15e9) == pytest.approx(9.488e9)
    assert cutoff(1e9) is None


@pytest.mark.parametrize("sigma,t", [(1e6, 5e-6), (50.0, 2e-3), (5.0, 5e-3)])
@pytest.mark.parametrize("waveguide", [False, True])
def test_effective_sigma_inverts_the_prediction(sigma, t, waveguide):
    se = predict(sigma, t, 8.2e9, 12.4e9, waveguide)
    assert effective_sigma(se, t, 8.2e9, 12.4e9, waveguide) == pytest.approx(sigma, rel=1e-6)
