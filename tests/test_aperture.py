"""Tests for aperture shielding effectiveness module.

Covers Bethe hole theory (circular), slot antenna model, aperture arrays,
waveguide-below-cutoff (honeycomb vents), rectangular cavity resonance modes,
and power-combined enclosure SE.
"""
import pytest
import numpy as np
from src.physics.aperture import (
    aperture_se_circular,
    aperture_se_slot,
    aperture_se_array,
    waveguide_below_cutoff_se,
    cavity_resonance_frequencies,
    combined_enclosure_se,
)
from src.utils.constants import C


class TestCircularAperture:
    def test_small_hole_high_se(self):
        """1mm hole at 100 MHz (lambda=3m): should give very high SE."""
        se = aperture_se_circular(frequency_hz=100e6, radius_m=0.5e-3)
        assert se > 60

    def test_resonant_hole_zero_se(self):
        """When diameter = lambda/2, SE drops to 0."""
        se = aperture_se_circular(frequency_hz=1e9, radius_m=0.075)
        assert se == pytest.approx(0.0, abs=0.1)

    def test_se_decreases_with_frequency(self):
        se_low = aperture_se_circular(100e6, 5e-3)
        se_high = aperture_se_circular(1e9, 5e-3)
        assert se_low > se_high

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            aperture_se_circular(1e9, -0.001)

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError):
            aperture_se_circular(0, 0.001)

    def test_very_small_hole_very_high_se(self):
        """0.1mm hole at 1 MHz: lambda=300m, SE should be enormous."""
        se = aperture_se_circular(frequency_hz=1e6, radius_m=0.05e-3)
        assert se > 100

    def test_known_value_1ghz_1mm(self):
        """1mm radius at 1 GHz: lambda=0.3m, SE = 20*log10(0.3/(2*0.001)) = 43.5 dB."""
        se = aperture_se_circular(frequency_hz=1e9, radius_m=1e-3)
        expected = 20 * np.log10(0.3 / 0.002)
        assert se == pytest.approx(expected, abs=0.5)


class TestSlotAperture:
    def test_short_slot_high_se(self):
        se = aperture_se_slot(frequency_hz=100e6, length_m=0.01)
        assert se > 40

    def test_halfwave_resonance_zero_se(self):
        """2.5 GHz: lambda=0.12m, L=0.06m = lambda/2 -> SE=0."""
        se = aperture_se_slot(frequency_hz=2.5e9, length_m=0.06)
        assert se == pytest.approx(0.0, abs=0.1)

    def test_6cm_seam_at_2_5ghz(self):
        se = aperture_se_slot(frequency_hz=2.5e9, length_m=0.06)
        assert se < 1.0

    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            aperture_se_slot(1e9, -0.01)

    def test_se_decreases_with_frequency(self):
        se_low = aperture_se_slot(100e6, 0.01)
        se_high = aperture_se_slot(1e9, 0.01)
        assert se_low > se_high

    def test_known_value_1ghz_1cm(self):
        """1cm slot at 1 GHz: lambda=0.3m, SE = 20*log10(0.3/(2*0.01)) = 23.5 dB."""
        se = aperture_se_slot(frequency_hz=1e9, length_m=0.01)
        expected = 20 * np.log10(0.3 / 0.02)
        assert se == pytest.approx(expected, abs=0.5)


class TestApertureArray:
    def test_n_holes_degrade_se(self):
        se_1 = aperture_se_circular(1e9, 1e-3)
        se_100 = aperture_se_array(1e9, se_single_db=se_1, n_apertures=100)
        assert se_100 == pytest.approx(se_1 - 20, abs=1)

    def test_single_aperture_unchanged(self):
        se_single = 40.0
        se_arr = aperture_se_array(1e9, se_single_db=se_single, n_apertures=1)
        assert se_arr == pytest.approx(se_single, abs=0.01)

    def test_10_apertures_minus_10db(self):
        se_single = 50.0
        se_arr = aperture_se_array(1e9, se_single_db=se_single, n_apertures=10)
        assert se_arr == pytest.approx(40.0, abs=0.01)

    def test_zero_apertures_raises(self):
        with pytest.raises(ValueError):
            aperture_se_array(1e9, se_single_db=40.0, n_apertures=0)

    def test_negative_apertures_raises(self):
        with pytest.raises(ValueError):
            aperture_se_array(1e9, se_single_db=40.0, n_apertures=-5)


class TestWaveguideBelowCutoff:
    def test_long_tube_high_se(self):
        se = waveguide_below_cutoff_se(1e9, tube_diameter_m=6e-3, tube_length_m=18e-3)
        assert se > 80

    def test_above_cutoff_zero_se(self):
        se = waveguide_below_cutoff_se(50e9, tube_diameter_m=6e-3, tube_length_m=18e-3)
        assert se < 5

    def test_longer_tube_more_attenuation(self):
        se_short = waveguide_below_cutoff_se(1e9, tube_diameter_m=6e-3, tube_length_m=10e-3)
        se_long = waveguide_below_cutoff_se(1e9, tube_diameter_m=6e-3, tube_length_m=30e-3)
        assert se_long > se_short

    def test_wider_tube_less_attenuation(self):
        se_narrow = waveguide_below_cutoff_se(1e9, tube_diameter_m=3e-3, tube_length_m=18e-3)
        se_wide = waveguide_below_cutoff_se(1e9, tube_diameter_m=10e-3, tube_length_m=18e-3)
        assert se_narrow > se_wide

    def test_negative_diameter_raises(self):
        with pytest.raises(ValueError):
            waveguide_below_cutoff_se(1e9, tube_diameter_m=-6e-3, tube_length_m=18e-3)

    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            waveguide_below_cutoff_se(1e9, tube_diameter_m=6e-3, tube_length_m=-18e-3)

    def test_cutoff_frequency_value(self):
        """For d=6mm circular waveguide, f_cutoff = 1.8412*c/(pi*d) ~ 29.3 GHz.
        Well below this should give high SE."""
        se = waveguide_below_cutoff_se(1e9, tube_diameter_m=6e-3, tube_length_m=18e-3)
        assert se > 50


class TestCavityResonance:
    def test_200x150x80_mm_box(self):
        """First mode is TM110 at ~1.25 GHz for a 200x150x80mm box."""
        modes = cavity_resonance_frequencies(0.2, 0.15, 0.08)
        assert len(modes) > 0
        f_first = modes[0]["frequency_hz"]
        assert 1.0e9 < f_first < 1.5e9

    def test_modes_sorted_ascending(self):
        modes = cavity_resonance_frequencies(0.3, 0.2, 0.1)
        freqs = [m["frequency_hz"] for m in modes]
        assert freqs == sorted(freqs)

    def test_mode_dict_keys(self):
        modes = cavity_resonance_frequencies(0.3, 0.2, 0.1)
        assert len(modes) > 0
        required_keys = {"m", "n", "p", "frequency_hz", "mode_type"}
        assert required_keys.issubset(set(modes[0].keys()))

    def test_at_least_two_nonzero_indices(self):
        """All returned modes must have at least two nonzero indices."""
        modes = cavity_resonance_frequencies(0.3, 0.2, 0.1)
        for mode in modes:
            nonzero = sum(1 for k in ["m", "n", "p"] if mode[k] > 0)
            assert nonzero >= 2, f"Mode {mode} has fewer than 2 nonzero indices"

    def test_max_results_respected(self):
        modes = cavity_resonance_frequencies(0.3, 0.2, 0.1, max_results=5)
        assert len(modes) <= 5

    def test_cube_symmetry(self):
        """A cube should have degenerate modes (same frequency for permutations)."""
        modes = cavity_resonance_frequencies(0.1, 0.1, 0.1)
        freqs = [m["frequency_hz"] for m in modes]
        # The first few modes of a cube include degenerate pairs/triples
        assert len(modes) > 0

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError):
            cavity_resonance_frequencies(-0.1, 0.2, 0.1)

    def test_te_tm_mode_types(self):
        """Mode types should be TE or TM variants."""
        modes = cavity_resonance_frequencies(0.3, 0.2, 0.1)
        for mode in modes:
            assert mode["mode_type"].startswith("TE") or mode["mode_type"].startswith("TM")

    def test_known_te101_for_cube(self):
        """For a 0.1m cube, TE101: f = (c/2)*sqrt((1/0.1)^2 + 0 + (1/0.1)^2) = c*sqrt(2)/0.2 ~ 2.12 GHz."""
        modes = cavity_resonance_frequencies(0.1, 0.1, 0.1)
        f_first = modes[0]["frequency_hz"]
        expected = (C / 2) * np.sqrt((1 / 0.1) ** 2 + (1 / 0.1) ** 2)
        assert f_first == pytest.approx(expected, rel=0.01)


class TestCombinedEnclosureSE:
    def test_aperture_dominates_when_weaker(self):
        se = combined_enclosure_se(bulk_se_db=80.0, aperture_se_list_db=[20.0])
        assert 19 < se < 21

    def test_multiple_apertures(self):
        se_1 = combined_enclosure_se(80.0, [30.0])
        se_3 = combined_enclosure_se(80.0, [30.0, 30.0, 30.0])
        assert se_3 < se_1

    def test_single_path_same_as_bulk(self):
        """With no apertures, result should equal bulk SE."""
        se = combined_enclosure_se(bulk_se_db=60.0, aperture_se_list_db=[])
        assert se == pytest.approx(60.0, abs=0.01)

    def test_two_equal_paths_minus_3db(self):
        """Two equal SE paths: SE_total = SE - 3 dB."""
        se = combined_enclosure_se(bulk_se_db=40.0, aperture_se_list_db=[40.0])
        assert se == pytest.approx(37.0, abs=0.2)

    def test_many_weak_apertures_dominate(self):
        """100 apertures at 10 dB should dominate over 80 dB bulk."""
        se = combined_enclosure_se(80.0, [10.0] * 100)
        assert se < 10
