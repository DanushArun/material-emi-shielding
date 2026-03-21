"""
Tests for src/physics/composite_models.py

Covers:
    - percolation_conductivity: below/above threshold behaviour
    - percolation_threshold_rods: realistic CNT geometry
    - percolation_threshold_disks: realistic graphene geometry
    - maxwell_garnett: boundary conditions
    - bruggeman_emt: symmetric mid-point identity
    - mclachlan_gem: reduction to percolation-like behaviour
    - hashin_shtrikman_bounds: ordering and consistency with Bruggeman
    - composite_conductivity: dispatcher routing and error handling
"""

import pytest
import numpy as np

from src.physics.composite_models import (
    percolation_conductivity,
    percolation_threshold_rods,
    percolation_threshold_disks,
    maxwell_garnett,
    bruggeman_emt,
    mclachlan_gem,
    hashin_shtrikman_bounds,
    composite_conductivity,
)


# ---------------------------------------------------------------------------
# Percolation conductivity
# ---------------------------------------------------------------------------

class TestPercolationConductivity:
    """Tests for the power-law percolation model."""

    SIGMA_FILLER = 1e6   # S/m  (conductive CNT network)
    SIGMA_MATRIX = 1e-10  # S/m  (insulating polymer)
    F_C = 0.02           # 2 vol% percolation threshold

    def test_below_threshold_returns_matrix_conductivity(self):
        """Below f_c the composite is dominated by the insulating matrix."""
        sigma = percolation_conductivity(
            filler_fraction=0.01,
            percolation_threshold=self.F_C,
            sigma_filler=self.SIGMA_FILLER,
            sigma_matrix=self.SIGMA_MATRIX,
        )
        assert sigma == self.SIGMA_MATRIX

    def test_at_threshold_returns_matrix_conductivity(self):
        """Exactly at f_c the model still returns the matrix value (<=)."""
        sigma = percolation_conductivity(
            filler_fraction=self.F_C,
            percolation_threshold=self.F_C,
            sigma_filler=self.SIGMA_FILLER,
            sigma_matrix=self.SIGMA_MATRIX,
        )
        assert sigma == self.SIGMA_MATRIX

    def test_above_threshold_returns_higher_value(self):
        """Above f_c the conductivity must exceed the matrix conductivity."""
        sigma = percolation_conductivity(
            filler_fraction=0.05,
            percolation_threshold=self.F_C,
            sigma_filler=self.SIGMA_FILLER,
            sigma_matrix=self.SIGMA_MATRIX,
        )
        assert sigma > self.SIGMA_MATRIX

    def test_above_threshold_increases_with_filler_fraction(self):
        """Conductivity is monotonically increasing above f_c."""
        fractions = [0.03, 0.05, 0.10, 0.20, 0.40]
        values = [
            percolation_conductivity(f, self.F_C, self.SIGMA_FILLER, self.SIGMA_MATRIX)
            for f in fractions
        ]
        for i in range(len(values) - 1):
            assert values[i + 1] > values[i], (
                f"Conductivity did not increase from f={fractions[i]} to f={fractions[i+1]}"
            )

    def test_above_threshold_bounded_by_filler_conductivity(self):
        """Effective conductivity must not exceed the filler conductivity."""
        sigma = percolation_conductivity(
            filler_fraction=0.9,
            percolation_threshold=self.F_C,
            sigma_filler=self.SIGMA_FILLER,
            sigma_matrix=self.SIGMA_MATRIX,
        )
        assert sigma <= self.SIGMA_FILLER

    def test_invalid_filler_fraction_raises(self):
        with pytest.raises(ValueError, match="filler_fraction"):
            percolation_conductivity(1.5, self.F_C, self.SIGMA_FILLER, self.SIGMA_MATRIX)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="percolation_threshold"):
            percolation_conductivity(0.05, 0.0, self.SIGMA_FILLER, self.SIGMA_MATRIX)

    def test_negative_conductivity_raises(self):
        with pytest.raises(ValueError):
            percolation_conductivity(0.05, self.F_C, -1.0, self.SIGMA_MATRIX)


# ---------------------------------------------------------------------------
# Percolation threshold for rods (CNTs)
# ---------------------------------------------------------------------------

class TestPercolationThresholdRods:
    """Tests for CNT percolation threshold estimation."""

    def test_cnt_realistic_geometry(self):
        """
        Multi-walled CNTs: L = 10 um, D = 10 nm  =>  AR = 1000
        Expected f_c = 0.7 / 1000 = 0.0007  (0.07%)
        """
        f_c = percolation_threshold_rods(length=10e-6, diameter=10e-9)
        assert pytest.approx(f_c, rel=1e-6) == 0.0007

    def test_result_is_very_low_for_high_ar(self):
        """High-AR CNTs percolate at sub-percent loading."""
        f_c = percolation_threshold_rods(length=10e-6, diameter=10e-9)
        assert f_c < 0.01, f"Expected f_c < 1%, got {f_c:.4%}"

    def test_higher_ar_gives_lower_threshold(self):
        """Longer tubes percolate at lower volume fraction."""
        f_c_short = percolation_threshold_rods(length=1e-6, diameter=10e-9)
        f_c_long = percolation_threshold_rods(length=10e-6, diameter=10e-9)
        assert f_c_long < f_c_short

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            percolation_threshold_rods(length=0.0, diameter=10e-9)

    def test_invalid_diameter_raises(self):
        with pytest.raises(ValueError, match="diameter"):
            percolation_threshold_rods(length=10e-6, diameter=-1e-9)


# ---------------------------------------------------------------------------
# Percolation threshold for disks (graphene / MXene)
# ---------------------------------------------------------------------------

class TestPercolationThresholdDisks:
    """Tests for disk filler percolation threshold estimation."""

    def test_graphene_realistic_geometry(self):
        """
        Graphene flake: radius = 1 um, thickness = 1 nm  =>  AR = 1000
        Expected f_c = 0.5 / 1000 = 0.0005
        """
        f_c = percolation_threshold_disks(radius=1e-6, thickness=1e-9)
        assert pytest.approx(f_c, rel=1e-6) == 0.0005

    def test_result_is_very_low_for_high_ar(self):
        f_c = percolation_threshold_disks(radius=1e-6, thickness=1e-9)
        assert f_c < 0.01

    def test_thicker_disk_gives_higher_threshold(self):
        f_c_thin = percolation_threshold_disks(radius=1e-6, thickness=1e-9)
        f_c_thick = percolation_threshold_disks(radius=1e-6, thickness=10e-9)
        assert f_c_thick > f_c_thin

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError, match="radius"):
            percolation_threshold_disks(radius=0.0, thickness=1e-9)

    def test_invalid_thickness_raises(self):
        with pytest.raises(ValueError, match="thickness"):
            percolation_threshold_disks(radius=1e-6, thickness=0.0)


# ---------------------------------------------------------------------------
# Maxwell-Garnett
# ---------------------------------------------------------------------------

class TestMaxwellGarnett:
    """Tests for the Maxwell-Garnett effective medium model."""

    SIGMA_HOST = 1.0       # S/m (reference value for clean arithmetic)
    SIGMA_INCL = 100.0     # S/m

    def test_zero_inclusion_returns_host_conductivity(self):
        """At f=0 the effective medium is pure host."""
        sigma = maxwell_garnett(
            sigma_host=self.SIGMA_HOST,
            sigma_inclusion=self.SIGMA_INCL,
            volume_fraction=0.0,
        )
        assert pytest.approx(sigma, rel=1e-9) == self.SIGMA_HOST

    def test_equal_conductivities_returns_host(self):
        """When sigma_host == sigma_inclusion the result is trivially sigma_host."""
        sigma = maxwell_garnett(
            sigma_host=5.0,
            sigma_inclusion=5.0,
            volume_fraction=0.3,
        )
        assert pytest.approx(sigma, rel=1e-9) == 5.0

    def test_conductive_inclusions_increase_sigma(self):
        """Adding conductive inclusions to an insulating host raises conductivity."""
        sigma = maxwell_garnett(
            sigma_host=self.SIGMA_HOST,
            sigma_inclusion=self.SIGMA_INCL,
            volume_fraction=0.2,
        )
        assert sigma > self.SIGMA_HOST

    def test_insulating_inclusions_decrease_sigma(self):
        """Adding insulating inclusions to a conductive host lowers conductivity."""
        sigma = maxwell_garnett(
            sigma_host=self.SIGMA_INCL,
            sigma_inclusion=self.SIGMA_HOST,
            volume_fraction=0.2,
        )
        assert sigma < self.SIGMA_INCL

    def test_conductivity_increases_with_inclusion_fraction(self):
        """MG prediction should be monotone in volume fraction (conductive inclusions)."""
        fractions = [0.0, 0.05, 0.10, 0.20]
        values = [
            maxwell_garnett(self.SIGMA_HOST, self.SIGMA_INCL, f)
            for f in fractions
        ]
        for i in range(len(values) - 1):
            assert values[i + 1] >= values[i]

    def test_invalid_volume_fraction_raises(self):
        with pytest.raises(ValueError):
            maxwell_garnett(self.SIGMA_HOST, self.SIGMA_INCL, volume_fraction=1.0)

    def test_negative_conductivity_raises(self):
        with pytest.raises(ValueError):
            maxwell_garnett(-1.0, self.SIGMA_INCL, volume_fraction=0.1)


# ---------------------------------------------------------------------------
# Bruggeman EMT
# ---------------------------------------------------------------------------

class TestBruggemanEMT:
    """Tests for the Bruggeman symmetric effective medium theory."""

    def test_equal_conductivities_returns_that_value(self):
        """sigma_1 == sigma_2 => sigma_eff == sigma_1 == sigma_2."""
        sigma = bruggeman_emt(sigma_1=10.0, sigma_2=10.0, f_1=0.5)
        assert pytest.approx(sigma, rel=1e-6) == 10.0

    def test_symmetric_composition_equal_phases(self):
        """At f_1=0.5 with equal sigma the result equals either phase."""
        sigma = bruggeman_emt(sigma_1=3.0, sigma_2=3.0, f_1=0.5)
        assert pytest.approx(sigma, rel=1e-6) == 3.0

    def test_pure_phase_1_returns_sigma_1(self):
        """At f_1=1 the composite is pure phase 1."""
        sigma = bruggeman_emt(sigma_1=50.0, sigma_2=1.0, f_1=1.0)
        assert pytest.approx(sigma, rel=1e-6) == 50.0

    def test_pure_phase_2_returns_sigma_2(self):
        """At f_1=0 the composite is pure phase 2."""
        sigma = bruggeman_emt(sigma_1=50.0, sigma_2=1.0, f_1=0.0)
        assert pytest.approx(sigma, rel=1e-6) == 1.0

    def test_result_between_two_phases(self):
        """Bruggeman result must lie strictly between sigma_1 and sigma_2."""
        sigma_1, sigma_2 = 1.0, 100.0
        sigma = bruggeman_emt(sigma_1=sigma_1, sigma_2=sigma_2, f_1=0.4)
        assert sigma_1 < sigma < sigma_2

    def test_symmetry(self):
        """Swapping phases and fractions must give the same result."""
        s1 = bruggeman_emt(sigma_1=10.0, sigma_2=100.0, f_1=0.3)
        s2 = bruggeman_emt(sigma_1=100.0, sigma_2=10.0, f_1=0.7)
        assert pytest.approx(s1, rel=1e-8) == s2

    def test_invalid_f1_raises(self):
        with pytest.raises(ValueError, match="f_1"):
            bruggeman_emt(sigma_1=10.0, sigma_2=100.0, f_1=1.5)


# ---------------------------------------------------------------------------
# McLachlan GEM
# ---------------------------------------------------------------------------

class TestMclachlanGEM:
    """Tests for the McLachlan General Effective Media equation."""

    SIGMA_MATRIX = 1e-10  # S/m  insulating polymer
    SIGMA_FILLER = 1e6    # S/m  conductive filler
    F_C = 0.02            # 2 vol%

    def test_well_below_threshold_close_to_matrix(self):
        """Deep below f_c the GEM result should be close to sigma_matrix."""
        sigma = mclachlan_gem(
            f_filler=0.001,
            sigma_matrix=self.SIGMA_MATRIX,
            sigma_filler=self.SIGMA_FILLER,
            f_c=self.F_C,
        )
        assert sigma < 1e-5  # still insulating

    def test_well_above_threshold_close_to_percolation(self):
        """
        Well above f_c the GEM result should be substantially higher than the
        matrix and show percolation-like growth.
        """
        sigma_low = mclachlan_gem(
            f_filler=0.03,
            sigma_matrix=self.SIGMA_MATRIX,
            sigma_filler=self.SIGMA_FILLER,
            f_c=self.F_C,
        )
        sigma_high = mclachlan_gem(
            f_filler=0.10,
            sigma_matrix=self.SIGMA_MATRIX,
            sigma_filler=self.SIGMA_FILLER,
            f_c=self.F_C,
        )
        # Both should be above matrix and sigma_high > sigma_low
        assert sigma_low > self.SIGMA_MATRIX
        assert sigma_high > sigma_low

    def test_monotone_increasing_with_filler_fraction(self):
        """GEM conductivity must be non-decreasing in filler fraction."""
        fractions = np.linspace(0.001, 0.40, 20)
        values = [
            mclachlan_gem(f, self.SIGMA_MATRIX, self.SIGMA_FILLER, self.F_C)
            for f in fractions
        ]
        for i in range(len(values) - 1):
            assert values[i + 1] >= values[i] - 1e-30, (
                f"Non-monotone at index {i}: {values[i]:.3e} -> {values[i+1]:.3e}"
            )

    def test_equal_conductivities_returns_that_value(self):
        """If sigma_matrix == sigma_filler there is nothing to solve."""
        sigma = mclachlan_gem(
            f_filler=0.1,
            sigma_matrix=5.0,
            sigma_filler=5.0,
            f_c=0.02,
        )
        assert pytest.approx(sigma, rel=1e-6) == 5.0

    def test_invalid_f_filler_raises(self):
        with pytest.raises(ValueError):
            mclachlan_gem(1.5, self.SIGMA_MATRIX, self.SIGMA_FILLER, self.F_C)

    def test_invalid_f_c_raises(self):
        with pytest.raises(ValueError, match="f_c"):
            mclachlan_gem(0.05, self.SIGMA_MATRIX, self.SIGMA_FILLER, f_c=0.0)


# ---------------------------------------------------------------------------
# Hashin-Shtrikman bounds
# ---------------------------------------------------------------------------

class TestHashinShtrikmanBounds:
    """Tests for the Hashin-Shtrikman conductivity bounds."""

    def test_lower_le_upper(self):
        """HS lower bound must always be <= HS upper bound."""
        lower, upper = hashin_shtrikman_bounds(1.0, 100.0, 0.3)
        assert lower <= upper

    def test_equal_conductivities_equal_bounds(self):
        """When sigma_1 == sigma_2 both bounds collapse to that value."""
        lower, upper = hashin_shtrikman_bounds(10.0, 10.0, 0.5)
        assert pytest.approx(lower, rel=1e-9) == 10.0
        assert pytest.approx(upper, rel=1e-9) == 10.0

    def test_bounds_bracket_bruggeman(self):
        """
        Bruggeman EMT result must lie between the HS lower and upper bounds.
        This is the key physically-required inequality:
            HS_lower <= sigma_Bruggeman <= HS_upper
        """
        sigma_1, sigma_2, f_1 = 1.0, 100.0, 0.3
        lower, upper = hashin_shtrikman_bounds(sigma_1, sigma_2, f_1)
        bruggeman = bruggeman_emt(sigma_1, sigma_2, f_1)
        assert lower <= bruggeman + 1e-10, (
            f"Bruggeman ({bruggeman:.4f}) below HS lower ({lower:.4f})"
        )
        assert bruggeman <= upper + 1e-10, (
            f"Bruggeman ({bruggeman:.4f}) above HS upper ({upper:.4f})"
        )

    def test_bounds_bracket_bruggeman_various_fractions(self):
        """HS bracketing should hold across composition range."""
        sigma_1, sigma_2 = 0.5, 50.0
        for f_1 in np.linspace(0.05, 0.95, 10):
            lower, upper = hashin_shtrikman_bounds(sigma_1, sigma_2, f_1)
            bruggeman = bruggeman_emt(sigma_1, sigma_2, f_1)
            assert lower <= bruggeman + 1e-9, f"Failed at f_1={f_1:.2f}"
            assert bruggeman <= upper + 1e-9, f"Failed at f_1={f_1:.2f}"

    def test_bounds_between_component_conductivities(self):
        """Both bounds must lie between min(sigma_1, sigma_2) and max(sigma_1, sigma_2)."""
        sigma_1, sigma_2 = 2.0, 200.0
        lower, upper = hashin_shtrikman_bounds(sigma_1, sigma_2, 0.4)
        assert lower >= sigma_1
        assert upper <= sigma_2

    def test_symmetry_of_bounds(self):
        """Swapping phases must produce the same pair of bounds."""
        l1, u1 = hashin_shtrikman_bounds(1.0, 100.0, 0.3)
        l2, u2 = hashin_shtrikman_bounds(100.0, 1.0, 0.7)
        assert pytest.approx(l1, rel=1e-8) == l2
        assert pytest.approx(u1, rel=1e-8) == u2

    def test_invalid_f1_raises(self):
        with pytest.raises(ValueError):
            hashin_shtrikman_bounds(1.0, 100.0, f_1=-0.1)

    def test_negative_conductivity_raises(self):
        with pytest.raises(ValueError):
            hashin_shtrikman_bounds(-1.0, 100.0, f_1=0.3)


# ---------------------------------------------------------------------------
# High-level dispatcher: composite_conductivity
# ---------------------------------------------------------------------------

class TestCompositeConductivity:
    """Tests for the high-level composite_conductivity dispatcher."""

    def test_alloy_dispatches_to_bruggeman(self):
        """'alloy' mode should match bruggeman_emt directly."""
        sigma_1, sigma_2, f_1 = 10.0, 100.0, 0.4
        expected = bruggeman_emt(sigma_1, sigma_2, f_1)
        result = composite_conductivity("alloy", sigma_1=sigma_1, sigma_2=sigma_2, f_1=f_1)
        assert pytest.approx(result, rel=1e-9) == expected

    def test_dilute_composite_dispatches_to_maxwell_garnett(self):
        """'dilute_composite' mode should match maxwell_garnett directly."""
        expected = maxwell_garnett(1.0, 100.0, 0.1)
        result = composite_conductivity(
            "dilute_composite",
            sigma_host=1.0,
            sigma_inclusion=100.0,
            volume_fraction=0.1,
        )
        assert pytest.approx(result, rel=1e-9) == expected

    def test_concentrated_composite_dispatches_to_hs_upper(self):
        """'concentrated_composite' should return the HS upper bound."""
        _lower, upper = hashin_shtrikman_bounds(1.0, 100.0, 0.3)
        result = composite_conductivity(
            "concentrated_composite",
            sigma_1=1.0,
            sigma_2=100.0,
            f_1=0.3,
        )
        assert pytest.approx(result, rel=1e-9) == upper

    def test_nanocomposite_dispatches_to_gem(self):
        """'nanocomposite' mode should match mclachlan_gem directly."""
        expected = mclachlan_gem(0.05, 1e-10, 1e6, 0.02)
        result = composite_conductivity(
            "nanocomposite",
            f_filler=0.05,
            sigma_matrix=1e-10,
            sigma_filler=1e6,
            f_c=0.02,
        )
        assert pytest.approx(result, rel=1e-6) == expected

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown composition_type"):
            composite_conductivity("unknown_type", sigma_1=1.0, sigma_2=10.0, f_1=0.5)

    def test_missing_required_param_raises(self):
        with pytest.raises(ValueError, match="missing required parameters"):
            composite_conductivity("alloy", sigma_1=1.0, sigma_2=10.0)
            # f_1 is missing

    def test_optional_depolarization_passed_through(self):
        """Custom depolarization factor should be forwarded to Maxwell-Garnett."""
        result_sphere = composite_conductivity(
            "dilute_composite",
            sigma_host=1.0,
            sigma_inclusion=100.0,
            volume_fraction=0.1,
            depolarization=1.0 / 3.0,
        )
        result_needle = composite_conductivity(
            "dilute_composite",
            sigma_host=1.0,
            sigma_inclusion=100.0,
            volume_fraction=0.1,
            depolarization=0.0,
        )
        # Needles aligned with field give higher effective conductivity than spheres
        assert result_needle > result_sphere
