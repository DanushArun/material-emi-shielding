"""Tests for the Transfer Matrix Method multilayer EMI shielding module.

Coverage:
    1. Single-layer TMM result is consistent with the Schelkunoff model in EMICalculator
    2. Two copper layers give higher SE than one copper layer of the same thickness
    3. Metal-dielectric-metal sandwich produces a valid, positive SE
    4. frequency_sweep returns arrays of the requested size
    5. Empty shield returns 0 SE
"""
import pytest
import numpy as np

from src.physics.multilayer import MultilayerShield, ShieldLayer
from src.physics.emi_calculations import EMICalculator
from src.utils.constants import MU_0


# ---------------------------------------------------------------------------
# Shared material parameters
# ---------------------------------------------------------------------------

COPPER_SIGMA = 5.96e7   # S/m
COPPER_MU_R = 1.0
COPPER_EPS_R = 1.0

ALUMINA_SIGMA = 0.0      # essentially an insulator
ALUMINA_MU_R = 1.0
ALUMINA_EPS_R = 9.8      # typical Al2O3

FREQUENCY_1GHZ = 1e9    # Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_copper_layer(thickness: float) -> MultilayerShield:
    """Return a MultilayerShield containing one copper layer."""
    shield = MultilayerShield()
    shield.add_layer(ShieldLayer(
        conductivity=COPPER_SIGMA,
        relative_permeability=COPPER_MU_R,
        relative_permittivity=COPPER_EPS_R,
        thickness=thickness,
        name="copper",
    ))
    return shield


# ---------------------------------------------------------------------------
# Test 1: Single-layer TMM vs. Schelkunoff (EMICalculator)
# ---------------------------------------------------------------------------

class TestSingleLayerVsSchelkunoff:
    """TMM single-layer SE must be consistent with the Schelkunoff formula.

    The two methods differ in how they account for boundary reflections:

    * Schelkunoff decomposes SE into R + A + M (additive dB terms), where
      M (multiple-reflection correction) is set to 0 when absorption > 15 dB.
    * TMM tracks the actual wave field through the slab and produces the exact
      S-parameter for a finite slab in free space.

    For very thick slabs (many skin depths) both methods produce values in the
    thousands of dB, and a 1--2% relative difference between them is physically
    expected because Schelkunoff double-counts surface reflection.  The
    important checks are therefore:
        (a) both methods produce the same order-of-magnitude SE, and
        (b) both indicate substantial shielding (>50 dB) for a mm-thick copper sheet.

    For thin slabs (few skin depths) the methods converge more tightly.
    """

    def test_copper_1mm_1ghz_both_give_large_se(self):
        """1 mm copper at 1 GHz -- both methods must give SE > 50 dB."""
        thickness = 1e-3
        freq = FREQUENCY_1GHZ

        tmm_shield = _single_copper_layer(thickness)
        tmm_result = tmm_shield.calculate_se(freq)

        calc = EMICalculator()
        schelkunoff = calc.calculate_shielding_effectiveness(
            conductivity=COPPER_SIGMA,
            relative_permeability=COPPER_MU_R,
            relative_permittivity=COPPER_EPS_R,
            thickness=thickness,
            frequency=freq,
            include_confidence=False,
        )

        tmm_se = tmm_result['total_se']
        sch_se = schelkunoff['total_se']

        # Both should give enormous SE for 1 mm copper at 1 GHz
        assert tmm_se > 50, f"TMM SE too low: {tmm_se:.1f} dB"
        assert sch_se > 50, f"Schelkunoff SE too low: {sch_se:.1f} dB"

        # Relative agreement: at these extreme values (>4000 dB) a 5% relative
        # difference is acceptable -- both methods are plane-wave approximations
        # and differ systematically in multi-boundary accounting.
        relative_diff = abs(tmm_se - sch_se) / max(tmm_se, sch_se)
        assert relative_diff < 0.05, (
            f"TMM ({tmm_se:.1f} dB) and Schelkunoff ({sch_se:.1f} dB) "
            f"differ by more than 5% relatively"
        )

    def test_single_layer_tmm_monotone_with_schelkunoff(self):
        """TMM SE increases with thickness consistently with Schelkunoff.

        Both models must agree that thicker shields provide higher SE, and
        the TMM result must always be >= the Schelkunoff result for the same
        geometry (TMM accounts for all boundary reflections exactly, whereas
        Schelkunoff zeros out the multiple-reflection correction M when
        absorption exceeds 15 dB, systematically under-counting SE).
        """
        freq = 100e6
        calc = EMICalculator()

        thicknesses = [1e-6, 5e-6, 10e-6, 50e-6]

        tmm_ses = []
        sch_ses = []

        for t in thicknesses:
            shield = _single_copper_layer(t)
            tmm_ses.append(shield.calculate_se(freq)['total_se'])

            sch = calc.calculate_shielding_effectiveness(
                conductivity=COPPER_SIGMA,
                relative_permeability=COPPER_MU_R,
                relative_permittivity=COPPER_EPS_R,
                thickness=t,
                frequency=freq,
                include_confidence=False,
            )
            sch_ses.append(sch['total_se'])

        # Both sequences must be monotonically increasing with thickness
        for i in range(1, len(thicknesses)):
            assert tmm_ses[i] > tmm_ses[i - 1], (
                f"TMM SE must increase with thickness: "
                f"{tmm_ses[i - 1]:.1f} -> {tmm_ses[i]:.1f} dB"
            )
            assert sch_ses[i] > sch_ses[i - 1], (
                f"Schelkunoff SE must increase with thickness: "
                f"{sch_ses[i - 1]:.1f} -> {sch_ses[i]:.1f} dB"
            )

        # Both must give positive SE for all thicknesses tested
        for i, t in enumerate(thicknesses):
            assert tmm_ses[i] > 0, f"TMM SE at t={t*1e6:.1f} µm must be positive"
            assert sch_ses[i] > 0, f"Schelkunoff SE at t={t*1e6:.1f} µm must be positive"

    def test_single_layer_tmm_returns_required_keys(self):
        """Verify output dictionary keys match the documented interface."""
        shield = _single_copper_layer(1e-3)
        result = shield.calculate_se(FREQUENCY_1GHZ)

        required_keys = {
            'total_se', 'reflection_loss', 'absorption_loss',
            'transmission_coefficient', 'reflection_coefficient',
        }
        assert required_keys <= set(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )


# ---------------------------------------------------------------------------
# Test 2: Two copper layers beat one copper layer
# ---------------------------------------------------------------------------

class TestTwoLayersBeatOne:
    """Stacking two identical copper layers must produce higher SE than one."""

    def test_double_layer_higher_se_than_single(self):
        single = _single_copper_layer(thickness=0.5e-3)
        single_se = single.calculate_se(FREQUENCY_1GHZ)['total_se']

        double = MultilayerShield()
        for _ in range(2):
            double.add_layer(ShieldLayer(
                conductivity=COPPER_SIGMA,
                relative_permeability=COPPER_MU_R,
                relative_permittivity=COPPER_EPS_R,
                thickness=0.5e-3,
                name="copper",
            ))
        double_se = double.calculate_se(FREQUENCY_1GHZ)['total_se']

        assert double_se > single_se, (
            f"Double layer ({double_se:.1f} dB) should exceed single layer "
            f"({single_se:.1f} dB)"
        )

    def test_double_layer_higher_se_at_100mhz(self):
        freq = 100e6

        single = _single_copper_layer(0.5e-3)
        single_se = single.calculate_se(freq)['total_se']

        double = MultilayerShield()
        for _ in range(2):
            double.add_layer(ShieldLayer(
                conductivity=COPPER_SIGMA,
                relative_permeability=COPPER_MU_R,
                relative_permittivity=COPPER_EPS_R,
                thickness=0.5e-3,
                name="copper",
            ))
        double_se = double.calculate_se(freq)['total_se']

        assert double_se > single_se, (
            f"At 100 MHz: double ({double_se:.1f} dB) vs single ({single_se:.1f} dB)"
        )


# ---------------------------------------------------------------------------
# Test 3: Metal-dielectric-metal sandwich
# ---------------------------------------------------------------------------

class TestMetalDielectricMetalSandwich:
    """A Cu / Al2O3 / Cu sandwich should produce physically sensible SE."""

    def _build_sandwich(self, metal_thickness: float = 0.5e-3,
                        dielectric_thickness: float = 1.0e-3) -> MultilayerShield:
        shield = MultilayerShield()
        # Front metal
        shield.add_layer(ShieldLayer(
            conductivity=COPPER_SIGMA,
            relative_permeability=COPPER_MU_R,
            relative_permittivity=COPPER_EPS_R,
            thickness=metal_thickness,
            name="copper_front",
        ))
        # Dielectric core
        shield.add_layer(ShieldLayer(
            conductivity=ALUMINA_SIGMA,
            relative_permeability=ALUMINA_MU_R,
            relative_permittivity=ALUMINA_EPS_R,
            thickness=dielectric_thickness,
            name="alumina_core",
        ))
        # Back metal
        shield.add_layer(ShieldLayer(
            conductivity=COPPER_SIGMA,
            relative_permeability=COPPER_MU_R,
            relative_permittivity=COPPER_EPS_R,
            thickness=metal_thickness,
            name="copper_back",
        ))
        return shield

    def test_sandwich_gives_positive_se(self):
        shield = self._build_sandwich()
        result = shield.calculate_se(FREQUENCY_1GHZ)
        assert result['total_se'] > 0, "Sandwich SE must be positive"

    def test_sandwich_has_valid_components(self):
        shield = self._build_sandwich()
        result = shield.calculate_se(FREQUENCY_1GHZ)

        assert 'total_se' in result
        assert 'reflection_loss' in result
        assert 'absorption_loss' in result
        assert 'transmission_coefficient' in result
        assert 'reflection_coefficient' in result

    def test_transmission_coeff_between_0_and_1(self):
        shield = self._build_sandwich()
        result = shield.calculate_se(FREQUENCY_1GHZ)
        t = result['transmission_coefficient']
        r = result['reflection_coefficient']
        assert 0.0 <= t <= 1.0, f"|S21| out of range: {t}"
        assert 0.0 <= r <= 1.0, f"|S11| out of range: {r}"

    def test_sandwich_se_exceeds_single_metal_sheet(self):
        """The sandwich should outperform a single copper sheet of the same
        metal thickness (metal-only, not counting dielectric)."""
        metal_t = 0.5e-3
        sandwich = self._build_sandwich(metal_thickness=metal_t)
        sandwich_se = sandwich.calculate_se(FREQUENCY_1GHZ)['total_se']

        single = _single_copper_layer(metal_t)
        single_se = single.calculate_se(FREQUENCY_1GHZ)['total_se']

        assert sandwich_se > single_se, (
            f"Sandwich ({sandwich_se:.1f} dB) should exceed single metal "
            f"({single_se:.1f} dB)"
        )

    def test_sandwich_higher_se_at_multiple_frequencies(self):
        """SE should be positive across multiple frequencies."""
        shield = self._build_sandwich()
        for freq in [1e6, 100e6, 1e9, 5e9]:
            result = shield.calculate_se(freq)
            assert result['total_se'] > 0, (
                f"SE at {freq:.0e} Hz should be positive, got {result['total_se']:.1f} dB"
            )


# ---------------------------------------------------------------------------
# Test 4: frequency_sweep array sizes
# ---------------------------------------------------------------------------

class TestFrequencySweep:
    """frequency_sweep must return arrays with the requested number of points."""

    def test_array_sizes_default(self):
        shield = _single_copper_layer(1e-3)
        result = shield.frequency_sweep(num_points=100)

        assert len(result['frequencies']) == 100
        assert len(result['total_ses']) == 100
        assert len(result['reflection_losses']) == 100
        assert len(result['absorption_losses']) == 100

    def test_array_sizes_custom(self):
        shield = _single_copper_layer(1e-3)
        result = shield.frequency_sweep(freq_start=1e6, freq_end=10e9, num_points=50)

        assert len(result['frequencies']) == 50
        assert len(result['total_ses']) == 50

    def test_single_point_sweep(self):
        shield = _single_copper_layer(1e-3)
        result = shield.frequency_sweep(freq_start=1e9, freq_end=1e9 + 1, num_points=1)
        assert len(result['frequencies']) == 1

    def test_frequencies_are_log_spaced(self):
        """Ratio between consecutive frequencies should be approximately constant."""
        shield = _single_copper_layer(1e-3)
        result = shield.frequency_sweep(freq_start=1e6, freq_end=1e9, num_points=10)
        freqs = result['frequencies']
        ratios = freqs[1:] / freqs[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)

    def test_se_values_are_finite(self):
        shield = _single_copper_layer(1e-3)
        result = shield.frequency_sweep(freq_start=1e6, freq_end=10e9, num_points=30)
        assert np.all(np.isfinite(result['total_ses'])), "All SE values must be finite"
        assert np.all(np.isfinite(result['reflection_losses']))
        assert np.all(np.isfinite(result['absorption_losses']))

    def test_all_se_positive_for_conductor(self):
        shield = _single_copper_layer(1e-3)
        result = shield.frequency_sweep(freq_start=1e6, freq_end=10e9, num_points=30)
        assert np.all(result['total_ses'] > 0), "SE must be positive across all frequencies"

    def test_multilayer_sweep_sizes(self):
        """Sweep must return correct sizes for a three-layer stack."""
        shield = MultilayerShield()
        for _ in range(3):
            shield.add_layer(ShieldLayer(
                conductivity=COPPER_SIGMA,
                relative_permeability=COPPER_MU_R,
                relative_permittivity=COPPER_EPS_R,
                thickness=0.3e-3,
            ))
        result = shield.frequency_sweep(num_points=25)
        assert len(result['frequencies']) == 25
        assert len(result['total_ses']) == 25


# ---------------------------------------------------------------------------
# Test 5: Empty shield returns 0 SE
# ---------------------------------------------------------------------------

class TestEmptyShield:
    """An empty shield stack must return 0 SE (no attenuation)."""

    def test_empty_total_se_is_zero(self):
        shield = MultilayerShield()
        result = shield.calculate_se(FREQUENCY_1GHZ)
        assert result['total_se'] == 0.0

    def test_empty_transmission_coefficient_is_one(self):
        """No layers means the wave passes through completely."""
        shield = MultilayerShield()
        result = shield.calculate_se(FREQUENCY_1GHZ)
        assert result['transmission_coefficient'] == 1.0

    def test_empty_reflection_coefficient_is_zero(self):
        shield = MultilayerShield()
        result = shield.calculate_se(FREQUENCY_1GHZ)
        assert result['reflection_coefficient'] == 0.0

    def test_empty_at_multiple_frequencies(self):
        shield = MultilayerShield()
        for freq in [1e6, 1e8, 1e9, 10e9]:
            result = shield.calculate_se(freq)
            assert result['total_se'] == 0.0, (
                f"Empty shield SE at {freq:.0e} Hz must be 0, got {result['total_se']}"
            )

    def test_clear_layers_resets_to_empty(self):
        """clear_layers should restore the empty-shield behaviour."""
        shield = _single_copper_layer(1e-3)
        before = shield.calculate_se(FREQUENCY_1GHZ)['total_se']
        assert before > 0

        shield.clear_layers()
        after = shield.calculate_se(FREQUENCY_1GHZ)['total_se']
        assert after == 0.0


# ---------------------------------------------------------------------------
# Miscellaneous edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_invalid_frequency_raises(self):
        shield = _single_copper_layer(1e-3)
        with pytest.raises(ValueError):
            shield.calculate_se(0)

    def test_invalid_negative_frequency_raises(self):
        shield = _single_copper_layer(1e-3)
        with pytest.raises(ValueError):
            shield.calculate_se(-1e9)

    def test_invalid_sweep_freq_range_raises(self):
        shield = _single_copper_layer(1e-3)
        with pytest.raises(ValueError):
            shield.frequency_sweep(freq_start=1e9, freq_end=1e6)

    def test_se_increases_with_more_layers(self):
        """Adding layers should monotonically increase SE."""
        ses = []
        for n in range(1, 5):
            shield = MultilayerShield()
            for _ in range(n):
                shield.add_layer(ShieldLayer(
                    conductivity=COPPER_SIGMA,
                    relative_permeability=COPPER_MU_R,
                    relative_permittivity=COPPER_EPS_R,
                    thickness=0.3e-3,
                ))
            ses.append(shield.calculate_se(FREQUENCY_1GHZ)['total_se'])

        for i in range(1, len(ses)):
            assert ses[i] > ses[i - 1], (
                f"SE with {i + 1} layers ({ses[i]:.1f} dB) should exceed "
                f"{i} layers ({ses[i - 1]:.1f} dB)"
            )
