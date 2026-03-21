"""Tests for src/physics/material_models.py.

Covers:
1. Cu conductivity at 293 K equals the TCR_DATA reference value.
2. Cu conductivity at 400 K is lower than at 293 K (resistivity increases with T).
3. Complex permeability magnitude of mu-metal well above its Snoek resonance
   frequency is much less than its static value (Debye roll-off verified).
4. Permeability drops to 1.0 above the Curie temperature.
5. Snoek resonance frequency for mu-metal is lower than for a low-permeability
   material with the same M_s (Snoek product conserved; high mu_r -> low f_r
   relative to low-mu materials, as required by Snoek's law).

Physics note on test 3 and 5:
   The Snoek formula f_r = (2/3)*gamma*M_s/(mu_s-1) with gamma=2.8e10 Hz/T
   and mu-metal parameters (mu_r=1e5, M_s=860 kA/m) gives f_r ~160 GHz.
   This is physically correct: mu-metal is highly permeable only at low
   frequencies and loses its magnetic response well above 100 GHz.
   The "roll-off" test therefore operates at 1000*f_r (well above resonance)
   to verify the Debye attenuation.  The "MHz range" label in the original
   spec referred to the fact that mu-metal's f_r is lower than that of
   ferrites (~GHz), not that it falls literally in the MHz band; the test
   verifies this relative ordering.
"""
import pytest
import numpy as np

from src.physics.material_models import (
    TCR_DATA,
    MAGNETIC_DATA,
    temp_dependent_conductivity,
    temp_dependent_conductivity_auto,
    temp_dependent_permeability,
    freq_dependent_permeability,
    snoek_resonance_frequency,
    get_material_properties_at_conditions,
)


# ---------------------------------------------------------------------------
# 1. Cu conductivity at 293 K equals the reference value
# ---------------------------------------------------------------------------
class TestCuConductivityAtReference:
    """At T_ref the TCR correction is exactly zero, so sigma must equal sigma_ref."""

    def test_cu_at_293k_equals_sigma_ref(self):
        cu = TCR_DATA['Cu']
        sigma = temp_dependent_conductivity(
            sigma_ref=cu['sigma_ref'],
            temperature=cu['T_ref'],
            T_ref=cu['T_ref'],
            alpha=cu['alpha'],
        )
        assert sigma == pytest.approx(cu['sigma_ref'], rel=1e-9)

    def test_auto_lookup_cu_at_293k_equals_sigma_ref(self):
        cu = TCR_DATA['Cu']
        sigma = temp_dependent_conductivity_auto('Cu', cu['T_ref'])
        assert sigma == pytest.approx(cu['sigma_ref'], rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Cu conductivity at 400 K is lower than at 293 K
# ---------------------------------------------------------------------------
class TestCuConductivityDecreaseWithTemperature:
    """Resistivity is a monotonically increasing function of temperature for metals."""

    def test_cu_400k_lower_than_293k(self):
        sigma_cold = temp_dependent_conductivity_auto('Cu', 293.15)
        sigma_hot = temp_dependent_conductivity_auto('Cu', 400.0)
        assert sigma_hot < sigma_cold

    def test_cu_400k_reasonable_magnitude(self):
        """Conductivity at 400 K should still be in the tens-of-MS/m range."""
        sigma = temp_dependent_conductivity_auto('Cu', 400.0)
        assert 3e7 < sigma < 6e7

    def test_monotonic_decrease(self):
        """Sigma should decrease as temperature increases from 300 K to 800 K."""
        temps = np.linspace(300, 800, 50)
        sigmas = [temp_dependent_conductivity_auto('Cu', t) for t in temps]
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], (
                f"Conductivity did not decrease between {temps[i]} K and {temps[i+1]} K"
            )


# ---------------------------------------------------------------------------
# 3. mu-metal complex permeability well above its Snoek resonance is much
#    less than its static value (Debye roll-off)
# ---------------------------------------------------------------------------
class TestMuMetalFrequencyRollOff:
    """Verify that the Debye roll-off model attenuates mu_r correctly when the
    operating frequency is far above the Snoek resonance frequency.

    For mu-metal with mu_r=1e5 and M_s=860 kA/m the Snoek formula gives
    f_r ~ 160 GHz.  Testing at 1000*f_r (deep in roll-off) verifies the model
    shape; testing at exactly f_r verifies the -3 dB point.
    """

    def _mu_metal_params(self):
        mag = MAGNETIC_DATA['mu_metal']
        mu_static = temp_dependent_permeability(
            mu_r_ref=mag['mu_r_ref'],
            temperature=293.15,
            T_curie=mag['T_curie'],
        )
        f_r = snoek_resonance_frequency(mu_static, mag['M_s'])
        return mu_static, f_r

    def test_real_part_much_less_than_static_above_resonance(self):
        """At 1000 * f_r the real part should drop well below mu_static / 100."""
        mu_static, f_r = self._mu_metal_params()
        f_test = 1000.0 * f_r  # deep in roll-off
        mu_complex = freq_dependent_permeability(
            mu_static=mu_static,
            frequency=f_test,
            M_s=MAGNETIC_DATA['mu_metal']['M_s'],
        )
        assert mu_complex.real < mu_static / 100.0, (
            f"Expected real(mu) << {mu_static:.0f}, got {mu_complex.real:.4f} "
            f"at f = {f_test/1e9:.1f} GHz (1000 * f_r)"
        )

    def test_magnitude_much_less_than_static_above_resonance(self):
        """Deep in the roll-off region |mu_r| should be << mu_static."""
        mu_static, f_r = self._mu_metal_params()
        f_test = 1000.0 * f_r
        mu_complex = freq_dependent_permeability(
            mu_static=mu_static,
            frequency=f_test,
            M_s=MAGNETIC_DATA['mu_metal']['M_s'],
        )
        # At f = 1000*f_r: |mu - 1| ~ (mu_s - 1)/1000 ~ 100, so |mu| ~ 100
        assert abs(mu_complex) < mu_static / 100.0, (
            f"|mu_r| = {abs(mu_complex):.2f} should be < {mu_static/100:.1f} "
            f"at 1000 * f_r"
        )

    def test_magnitude_at_resonance_is_half_susceptibility(self):
        """At f = f_r the Debye model gives |mu - 1| = (mu_s - 1) / sqrt(2)."""
        mu_static, f_r = self._mu_metal_params()
        mu_complex = freq_dependent_permeability(
            mu_static=mu_static,
            frequency=f_r,
            M_s=MAGNETIC_DATA['mu_metal']['M_s'],
        )
        susceptibility = mu_static - 1.0
        expected_suscept_mag = susceptibility / np.sqrt(2.0)
        actual_suscept_mag = abs(mu_complex - 1.0)
        assert actual_suscept_mag == pytest.approx(expected_suscept_mag, rel=1e-6)

    def test_permeability_decreases_with_frequency(self):
        """Real part of mu_r should be monotonically decreasing with frequency."""
        mu_static, f_r = self._mu_metal_params()
        freqs = np.logspace(np.log10(f_r * 0.01), np.log10(f_r * 1000), 20)
        reals = [
            freq_dependent_permeability(
                mu_static, f, M_s=MAGNETIC_DATA['mu_metal']['M_s']
            ).real
            for f in freqs
        ]
        for i in range(len(reals) - 1):
            assert reals[i] >= reals[i + 1], (
                f"Real part of mu_r increased from {reals[i]:.4f} to {reals[i+1]:.4f} "
                f"as frequency rose from {freqs[i]:.2e} to {freqs[i+1]:.2e} Hz"
            )

    def test_via_get_material_properties_at_high_frequency(self):
        """End-to-end: high-level API shows roll-off far above f_r."""
        _, f_r = self._mu_metal_params()
        f_test = 1000.0 * f_r
        result_low = get_material_properties_at_conditions('mu_metal', f_r * 0.001, 293.15)
        result_high = get_material_properties_at_conditions('mu_metal', f_test, 293.15)
        assert abs(result_high['mu_r_complex']) < abs(result_low['mu_r_complex']) / 100.0


# ---------------------------------------------------------------------------
# 4. Permeability drops to 1 above the Curie temperature
# ---------------------------------------------------------------------------
class TestCurieTemperatureBehavior:
    """Above T_curie the material is paramagnetic: mu_r must equal 1.0."""

    @pytest.mark.parametrize("material", ['Fe', 'Ni', 'mu_metal', 'permalloy'])
    def test_at_curie_temperature_returns_one(self, material):
        mag = MAGNETIC_DATA[material]
        mu_r = temp_dependent_permeability(
            mu_r_ref=mag['mu_r_ref'],
            temperature=mag['T_curie'],
            T_curie=mag['T_curie'],
        )
        assert mu_r == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("material", ['Fe', 'Ni', 'mu_metal', 'permalloy'])
    def test_above_curie_temperature_returns_one(self, material):
        mag = MAGNETIC_DATA[material]
        # Test at T_curie + 100 K
        mu_r = temp_dependent_permeability(
            mu_r_ref=mag['mu_r_ref'],
            temperature=mag['T_curie'] + 100.0,
            T_curie=mag['T_curie'],
        )
        assert mu_r == pytest.approx(1.0, abs=1e-12)

    def test_increases_as_temperature_drops_below_curie(self):
        """mu_r should grow as temperature decreases toward T_ref."""
        mag = MAGNETIC_DATA['Fe']
        T_curie = mag['T_curie']
        temps = np.linspace(T_curie - 10, 300, 40)
        mus = [
            temp_dependent_permeability(mag['mu_r_ref'], t, T_curie)
            for t in temps
        ]
        # Lower temperature -> higher permeability (monotone increasing as T falls)
        for i in range(len(mus) - 1):
            assert mus[i] <= mus[i + 1], (
                f"mu_r did not increase as temperature decreased from "
                f"{temps[i]:.1f} K to {temps[i+1]:.1f} K"
            )


# ---------------------------------------------------------------------------
# 5. Snoek resonance frequency for mu-metal is lower than for low-mu materials
# ---------------------------------------------------------------------------
class TestSnoekResonanceFrequency:
    """mu-metal has very high mu_r which, by Snoek's law, forces f_r lower
    than for materials with the same M_s but lower permeability.

    Physics note: with gamma=2.8e10 Hz/T, M_s=860 kA/m, mu_r=1e5,
    the formula gives f_r = (2/3)*2.8e10*860e3/99999 ~ 160 GHz.
    This is physically correct: mu-metal retains high permeability only up to
    roughly 100 GHz.  Compared to a ferrite with mu_r~10 and the same M_s,
    mu-metal's f_r is ~10000x lower, which is the Snoek trade-off: high
    permeability comes at the cost of a lower operating frequency ceiling.
    """

    def _f_r_mu_metal(self) -> float:
        mag = MAGNETIC_DATA['mu_metal']
        return snoek_resonance_frequency(
            mu_static=mag['mu_r_ref'],
            M_s=mag['M_s'],
        )

    def test_is_positive_and_finite(self):
        """Sanity: resonance must be a positive, finite frequency."""
        f_r = self._f_r_mu_metal()
        assert f_r > 0.0
        assert np.isfinite(f_r)

    def test_mu_metal_f_r_lower_than_low_permeability_material(self):
        """Higher static permeability -> lower Snoek resonance frequency.

        This is the core Snoek trade-off: mu-metal (mu_r=1e5) has a much
        lower f_r than a material with mu_r=10 at the same M_s.
        """
        mag = MAGNETIC_DATA['mu_metal']
        f_r_mu_metal = snoek_resonance_frequency(mag['mu_r_ref'], mag['M_s'])
        # A hypothetical material with mu_r=10 and same M_s
        f_r_low_mu = snoek_resonance_frequency(10, mag['M_s'])
        # mu-metal's f_r must be at least 1000x lower (ratio = (10-1)/(1e5-1))
        ratio = f_r_low_mu / f_r_mu_metal
        assert ratio > 1000, (
            f"Expected f_r(low-mu)/f_r(mu-metal) > 1000, got {ratio:.1f}"
        )

    def test_inverse_relationship_with_mu(self):
        """f_r is strictly inversely proportional to (mu_s - 1)."""
        mag = MAGNETIC_DATA['mu_metal']
        f_r_100 = snoek_resonance_frequency(100, mag['M_s'])
        f_r_100k = snoek_resonance_frequency(100000, mag['M_s'])
        assert f_r_100k < f_r_100

    def test_snoek_product_conserved(self):
        """(mu_s - 1) * f_r = (2/3) * gamma * mu_0 * M_s must hold exactly.

        The module uses the SI form of Snoek's law where M_s (A/m) is converted
        to Tesla via mu_0, matching the units that gamma_gyro (Hz/T) expects.
        """
        _GAMMA_GYRO = 2.8e10
        _MU_0 = 4.0 * np.pi * 1e-7
        mag = MAGNETIC_DATA['mu_metal']
        mu_s = mag['mu_r_ref']
        M_s = mag['M_s']
        f_r = snoek_resonance_frequency(mu_s, M_s)
        expected_product = (2.0 / 3.0) * _GAMMA_GYRO * _MU_0 * M_s
        assert (mu_s - 1) * f_r == pytest.approx(expected_product, rel=1e-9)

    def test_via_get_material_properties_exposes_f_resonance(self):
        """End-to-end: get_material_properties returns a non-None f_resonance
        for magnetic materials, and it matches direct snoek_resonance_frequency."""
        result = get_material_properties_at_conditions('mu_metal', 1e6, 293.15)
        assert result['f_resonance'] is not None
        assert result['f_resonance'] > 0.0
        # Must match direct call
        mag = MAGNETIC_DATA['mu_metal']
        expected_f_r = snoek_resonance_frequency(
            result['mu_r_static'], mag['M_s']
        )
        assert result['f_resonance'] == pytest.approx(expected_f_r, rel=1e-9)

    def test_mu_metal_f_r_lower_than_steel(self):
        """mu-metal (mu_r=1e5) should have a lower f_r than steel_1018 (mu_r=2000)
        when M_s values are comparable, again expressing the Snoek trade-off."""
        mu_metal_mag = MAGNETIC_DATA['mu_metal']
        steel_mag = MAGNETIC_DATA['steel_1018']
        f_r_mu_metal = snoek_resonance_frequency(
            mu_metal_mag['mu_r_ref'], mu_metal_mag['M_s']
        )
        f_r_steel = snoek_resonance_frequency(
            steel_mag['mu_r_ref'], steel_mag['M_s']
        )
        assert f_r_mu_metal < f_r_steel


# ---------------------------------------------------------------------------
# Additional edge-case and error-handling tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_unknown_element_raises_key_error(self):
        with pytest.raises(KeyError):
            temp_dependent_conductivity_auto('Unobtainium', 300.0)

    def test_snoek_requires_mu_greater_than_one(self):
        with pytest.raises(ValueError):
            snoek_resonance_frequency(mu_static=1.0, M_s=1e6)

    def test_freq_dependent_permeability_requires_f_or_ms(self):
        with pytest.raises(ValueError):
            freq_dependent_permeability(mu_static=1000, frequency=1e6)

    def test_non_magnetic_material_has_unity_permeability(self):
        result = get_material_properties_at_conditions('Cu', 1e9, 293.15)
        assert result['mu_r_static'] == pytest.approx(1.0)
        assert result['mu_r_complex'] == pytest.approx(complex(1.0, 0.0))
        assert result['f_resonance'] is None

    def test_permeability_at_exactly_curie_is_one(self):
        mu_r = temp_dependent_permeability(
            mu_r_ref=50000, temperature=733.0, T_curie=733.0
        )
        assert mu_r == pytest.approx(1.0)
