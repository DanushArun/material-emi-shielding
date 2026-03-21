"""Tests for src/physics/uncertainty.py

Covers:
1. MC with zero uncertainty produces a deterministic (zero-std) result.
2. MC mean is close to the deterministic SE.
3. The 95% CI brackets the deterministic value.
4. Higher uncertainty coefficient produces a wider CI.
5. Sobol: thickness and conductivity dominate for a copper shield.
6. Latin Hypercube gives results consistent with pure MC using fewer samples.
"""

import numpy as np
import pytest

from src.physics.uncertainty import (
    UncertaintySpec,
    monte_carlo_se,
    latin_hypercube_se,
    sobol_sensitivity,
    reliability_se,
)
from src.physics.emi_calculations import EMICalculator

# ---------------------------------------------------------------------------
# Shared nominal parameters (copper shield, 1 GHz, 1 mm)
# ---------------------------------------------------------------------------
COPPER_PARAMS = dict(
    conductivity=5.96e7,   # S/m
    permeability=1.0,      # relative
    permittivity=1.0,      # relative
    thickness=1e-3,        # 1 mm
    frequency=1e9,         # 1 GHz
)

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_se(**params) -> float:
    calc = EMICalculator()
    result = calc.calculate_shielding_effectiveness(
        conductivity=params["conductivity"],
        relative_permeability=params["permeability"],
        relative_permittivity=params["permittivity"],
        thickness=params["thickness"],
        frequency=params["frequency"],
        include_confidence=False,
    )
    return float(result["total_se"])


# ---------------------------------------------------------------------------
# Test 1: Zero uncertainty -> deterministic (std ~ 0)
# ---------------------------------------------------------------------------

class TestZeroUncertainty:
    """With all CV = 0, every sample should produce exactly the nominal SE."""

    def test_std_is_zero(self):
        zero_unc = UncertaintySpec(
            conductivity_cv=0.0,
            thickness_cv=0.0,
            grain_size_cv=0.0,
            permeability_cv=0.0,
            frequency_cv=0.0,
        )
        result = monte_carlo_se(
            **COPPER_PARAMS,
            uncertainty=zero_unc,
            n_samples=200,
            seed=SEED,
        )
        assert result["se_std"] == pytest.approx(0.0, abs=1e-10), (
            f"Expected std ~ 0 with zero CV, got {result['se_std']}"
        )

    def test_mean_equals_deterministic(self):
        det = _deterministic_se(**COPPER_PARAMS)
        zero_unc = UncertaintySpec(
            conductivity_cv=0.0,
            thickness_cv=0.0,
            grain_size_cv=0.0,
            permeability_cv=0.0,
            frequency_cv=0.0,
        )
        result = monte_carlo_se(
            **COPPER_PARAMS,
            uncertainty=zero_unc,
            n_samples=200,
            seed=SEED,
        )
        assert result["se_mean"] == pytest.approx(det, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 2: MC mean close to deterministic calculation
# ---------------------------------------------------------------------------

class TestMCMeanCloseToDeterministic:
    """With realistic uncertainty the MC mean should remain near nominal SE."""

    def test_mean_within_10_percent(self):
        det = _deterministic_se(**COPPER_PARAMS)
        result = monte_carlo_se(
            **COPPER_PARAMS,
            n_samples=2000,
            seed=SEED,
        )
        # Relative deviation < 10 % of the deterministic value.
        rel_error = abs(result["se_mean"] - det) / abs(det)
        assert rel_error < 0.10, (
            f"MC mean {result['se_mean']:.2f} dB too far from deterministic "
            f"{det:.2f} dB (rel error {rel_error:.2%})"
        )

    def test_mean_close_with_grain_size(self):
        det_calc = EMICalculator()
        det = float(det_calc.calculate_shielding_effectiveness(
            **{
                "conductivity": COPPER_PARAMS["conductivity"],
                "relative_permeability": COPPER_PARAMS["permeability"],
                "relative_permittivity": COPPER_PARAMS["permittivity"],
                "thickness": COPPER_PARAMS["thickness"],
                "frequency": COPPER_PARAMS["frequency"],
                "grain_size": 50e-6,
                "include_confidence": False,
            }
        )["total_se"])
        result = monte_carlo_se(
            **COPPER_PARAMS,
            grain_size=50e-6,
            n_samples=2000,
            seed=SEED,
        )
        rel_error = abs(result["se_mean"] - det) / abs(det)
        assert rel_error < 0.15


# ---------------------------------------------------------------------------
# Test 3: 95% CI contains the deterministic value
# ---------------------------------------------------------------------------

class TestCIContainsDeterministic:
    """The 95 % credible interval [2.5th, 97.5th] should bracket nominal SE."""

    def test_ci_brackets_deterministic(self):
        det = _deterministic_se(**COPPER_PARAMS)
        result = monte_carlo_se(
            **COPPER_PARAMS,
            n_samples=2000,
            seed=SEED,
        )
        assert result["se_ci_lower"] <= det <= result["se_ci_upper"], (
            f"Deterministic SE {det:.2f} not in CI "
            f"[{result['se_ci_lower']:.2f}, {result['se_ci_upper']:.2f}]"
        )

    def test_ci_width_positive(self):
        result = monte_carlo_se(**COPPER_PARAMS, n_samples=500, seed=SEED)
        assert result["se_ci_upper"] > result["se_ci_lower"]


# ---------------------------------------------------------------------------
# Test 4: Higher uncertainty produces a wider CI
# ---------------------------------------------------------------------------

class TestHigherUncertaintyWiderCI:
    """Doubling the CVs should strictly widen the confidence interval."""

    def test_wider_ci_with_higher_cv(self):
        low_unc = UncertaintySpec(
            conductivity_cv=0.02,
            thickness_cv=0.01,
            permeability_cv=0.02,
            frequency_cv=0.0005,
            grain_size_cv=0.10,
        )
        high_unc = UncertaintySpec(
            conductivity_cv=0.10,
            thickness_cv=0.05,
            permeability_cv=0.10,
            frequency_cv=0.002,
            grain_size_cv=0.40,
        )
        low_result  = monte_carlo_se(**COPPER_PARAMS, uncertainty=low_unc,  n_samples=1000, seed=SEED)
        high_result = monte_carlo_se(**COPPER_PARAMS, uncertainty=high_unc, n_samples=1000, seed=SEED)

        ci_low  = low_result["se_ci_upper"]  - low_result["se_ci_lower"]
        ci_high = high_result["se_ci_upper"] - high_result["se_ci_lower"]

        assert ci_high > ci_low, (
            f"Expected wider CI for higher uncertainty: "
            f"high={ci_high:.2f} dB, low={ci_low:.2f} dB"
        )

    def test_std_increases_with_cv(self):
        low_unc  = UncertaintySpec(conductivity_cv=0.01, thickness_cv=0.005,
                                   permeability_cv=0.01, frequency_cv=0.0001,
                                   grain_size_cv=0.05)
        high_unc = UncertaintySpec(conductivity_cv=0.15, thickness_cv=0.08,
                                   permeability_cv=0.15, frequency_cv=0.005,
                                   grain_size_cv=0.50)
        low_r  = monte_carlo_se(**COPPER_PARAMS, uncertainty=low_unc,  n_samples=1000, seed=SEED)
        high_r = monte_carlo_se(**COPPER_PARAMS, uncertainty=high_unc, n_samples=1000, seed=SEED)
        assert high_r["se_std"] > low_r["se_std"]


# ---------------------------------------------------------------------------
# Test 5: Sobol — thickness and conductivity dominate for copper
# ---------------------------------------------------------------------------

class TestSobolSensitivity:
    """For a thick copper shield the dominant parameters should be
    thickness (absorption) and conductivity (both absorption and reflection).
    We accept either of them appearing in the top-2 total-order indices."""

    def test_returns_expected_keys(self):
        result = sobol_sensitivity(**COPPER_PARAMS, n_samples=512, seed=SEED)
        assert "parameters" in result
        assert "S1" in result
        assert "ST" in result
        for p in ["conductivity", "thickness"]:
            assert p in result["S1"]
            assert p in result["ST"]

    def test_thickness_conductivity_dominate(self):
        result = sobol_sensitivity(**COPPER_PARAMS, n_samples=1024, seed=SEED)
        ST = result["ST"]
        # Sort parameter names by total-order index (descending).
        ranked = sorted(ST.items(), key=lambda kv: kv[1], reverse=True)
        top_two = {name for name, _ in ranked[:2]}
        dominant = {"thickness", "conductivity"}
        overlap = dominant & top_two
        assert len(overlap) >= 1, (
            f"Expected thickness or conductivity in top-2, got: {ranked}"
        )

    def test_indices_in_zero_one(self):
        result = sobol_sensitivity(**COPPER_PARAMS, n_samples=512, seed=SEED)
        for name, val in result["S1"].items():
            assert 0.0 <= val <= 1.0, f"S1[{name}]={val} out of [0,1]"
        for name, val in result["ST"].items():
            assert 0.0 <= val <= 1.0, f"ST[{name}]={val} out of [0,1]"

    def test_st_ge_s1(self):
        """Total-order index must be >= first-order (it includes interactions)."""
        result = sobol_sensitivity(**COPPER_PARAMS, n_samples=512, seed=SEED)
        for name in result["parameters"]:
            assert result["ST"][name] >= result["S1"][name] - 0.05, (
                f"ST[{name}] < S1[{name}] (unexpected)"
            )

    def test_grain_size_included_when_provided(self):
        result = sobol_sensitivity(**COPPER_PARAMS, grain_size=50e-6,
                                   n_samples=512, seed=SEED)
        assert "grain_size" in result["parameters"]
        assert "grain_size" in result["S1"]


# ---------------------------------------------------------------------------
# Test 6: Latin Hypercube produces results consistent with pure MC
# ---------------------------------------------------------------------------

class TestLatinHypercube:
    """LHS with fewer samples should produce a mean and CI comparable to
    pure MC with many samples."""

    def test_lhs_mean_close_to_mc(self):
        mc = monte_carlo_se(**COPPER_PARAMS, n_samples=2000, seed=SEED)
        lhs = latin_hypercube_se(**COPPER_PARAMS, n_samples=200, seed=SEED)

        # Means should agree within 5 % of deterministic SE.
        det = mc["deterministic_se"]
        assert abs(lhs["se_mean"] - mc["se_mean"]) / abs(det) < 0.05, (
            f"LHS mean {lhs['se_mean']:.2f} differs too much from "
            f"MC mean {mc['se_mean']:.2f}"
        )

    def test_lhs_ci_overlaps_mc_ci(self):
        mc  = monte_carlo_se(**COPPER_PARAMS, n_samples=2000, seed=SEED)
        lhs = latin_hypercube_se(**COPPER_PARAMS, n_samples=200, seed=SEED)
        # The two CIs should have non-trivial overlap: upper of one exceeds
        # lower of the other in both directions.
        assert lhs["se_ci_upper"] > mc["se_ci_lower"]
        assert mc["se_ci_upper"] > lhs["se_ci_lower"]

    def test_lhs_std_close_to_mc_std(self):
        mc  = monte_carlo_se(**COPPER_PARAMS, n_samples=2000, seed=SEED)
        lhs = latin_hypercube_se(**COPPER_PARAMS, n_samples=300, seed=SEED)
        # Std should agree within 50 % (variance estimator noise is expected).
        rel = abs(lhs["se_std"] - mc["se_std"]) / max(mc["se_std"], 1e-3)
        assert rel < 0.5, (
            f"LHS std {lhs['se_std']:.2f} too far from MC std {mc['se_std']:.2f}"
        )

    def test_lhs_returns_correct_n_samples(self):
        n = 50
        result = latin_hypercube_se(**COPPER_PARAMS, n_samples=n, seed=SEED)
        assert result["n_samples"] == n
        assert len(result["se_distribution"]) == n

    def test_lhs_ci_contains_deterministic(self):
        det = _deterministic_se(**COPPER_PARAMS)
        result = latin_hypercube_se(**COPPER_PARAMS, n_samples=500, seed=SEED)
        assert result["se_ci_lower"] <= det <= result["se_ci_upper"]


# ---------------------------------------------------------------------------
# Additional: reliability_se sanity checks
# ---------------------------------------------------------------------------

class TestReliabilitySE:
    """Verify reliability_se works and returns meaningful probabilities."""

    def test_very_low_target_gives_high_probability(self):
        # Target = 10 dB, copper @ 1 GHz / 1 mm achieves >4000 dB -> P ~ 1.
        result = reliability_se(**COPPER_PARAMS, target_se=10.0, n_samples=500, seed=SEED)
        assert result["probability_exceeds_target"] > 0.99

    def test_impossible_target_gives_low_probability(self):
        # Target absurdly high -> almost no sample should exceed it.
        result = reliability_se(**COPPER_PARAMS, target_se=1e6, n_samples=500, seed=SEED)
        assert result["probability_exceeds_target"] < 0.01

    def test_returns_required_keys(self):
        result = reliability_se(**COPPER_PARAMS, target_se=40.0, n_samples=200, seed=SEED)
        for key in ("probability_exceeds_target", "se_95_reliable", "se_mean",
                    "se_std", "target_se", "n_samples"):
            assert key in result, f"Missing key: {key}"

    def test_se_95_reliable_lte_mean(self):
        result = reliability_se(**COPPER_PARAMS, target_se=40.0, n_samples=500, seed=SEED)
        assert result["se_95_reliable"] <= result["se_mean"]
