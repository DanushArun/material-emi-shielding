"""Tests for the Multiconductor Transmission Line (MTL) cable crosstalk module.

Validates per-unit-length parameter extraction, distributed-parameter
NEXT/FEXT against known physical behaviour, transfer impedance model,
and cable SE computation.
"""
import pytest
import numpy as np
from src.physics.cables_mtl import (
    per_unit_length_params_two_wire,
    mtl_crosstalk_two_wire,
    transfer_impedance_kley,
    cable_se_from_zt,
)
from src.utils.constants import MU_0, EPSILON_0, C


class TestPerUnitLengthParams:
    """Validate PUL L and C matrix extraction for two wires above ground."""

    def test_two_wires_above_ground(self):
        """L_self should be ~0.5-1 uH/m for typical geometry; matrices symmetric."""
        L, C_mat = per_unit_length_params_two_wire(1e-3, 10e-3, 20e-3)
        # Self-inductance in expected range
        assert 5e-7 < L[0, 0] < 1e-6
        # Mutual < self (coupling always weaker)
        assert 0 < L[0, 1] < L[0, 0]
        # Capacitance positive
        assert C_mat[0, 0] > 0
        # Symmetry
        assert L[0, 1] == pytest.approx(L[1, 0], rel=1e-10)
        assert C_mat[0, 1] == pytest.approx(C_mat[1, 0], rel=1e-10)

    def test_wider_separation_reduces_mutual(self):
        """Moving wires further apart should decrease mutual inductance."""
        _, _ = per_unit_length_params_two_wire(1e-3, 10e-3, 20e-3)
        L_close, _ = per_unit_length_params_two_wire(1e-3, 5e-3, 20e-3)
        L_far, _ = per_unit_length_params_two_wire(1e-3, 50e-3, 20e-3)
        assert L_close[0, 1] > L_far[0, 1]

    def test_dielectric_increases_capacitance(self):
        """Higher epsilon_r should increase capacitance."""
        _, C_air = per_unit_length_params_two_wire(1e-3, 10e-3, 20e-3, epsilon_r=1.0)
        _, C_ptfe = per_unit_length_params_two_wire(1e-3, 10e-3, 20e-3, epsilon_r=2.1)
        assert C_ptfe[0, 0] > C_air[0, 0]

    def test_lc_product_gives_correct_velocity(self):
        """For lossless TEM lines, v = 1/sqrt(LC) = c/sqrt(epsilon_r)."""
        eps_r = 2.25
        L, C_mat = per_unit_length_params_two_wire(1e-3, 10e-3, 20e-3, epsilon_r=eps_r)
        # Product of self terms
        lc_product = L[0, 0] * C_mat[0, 0]
        v_expected = C / np.sqrt(eps_r)
        v_from_lc = 1.0 / np.sqrt(lc_product)
        assert v_from_lc == pytest.approx(v_expected, rel=0.15)


class TestMTLCrosstalk:
    """Validate distributed-parameter NEXT and FEXT computations."""

    def test_next_increases_with_frequency(self):
        """NEXT should generally increase with frequency (more coupling)."""
        freqs = [1e6, 10e6, 100e6]
        nexts = []
        for f in freqs:
            result = mtl_crosstalk_two_wire(1.0, 0.5e-3, 5e-3, 10e-3, f, 50.0, 50.0)
            nexts.append(result["next_db"])
        assert nexts[1] > nexts[0]
        assert nexts[2] > nexts[1]

    def test_fext_shows_propagation_effects(self):
        """FEXT must exhibit wave propagation nulls/dips -- not monotonic increase."""
        results = []
        for f in np.linspace(50e6, 500e6, 20):
            r = mtl_crosstalk_two_wire(1.0, 0.5e-3, 5e-3, 10e-3, f, 50.0, 50.0)
            results.append(r["fext_db"])
        diffs = np.diff(results)
        has_decrease = any(d < 0 for d in diffs)
        assert has_decrease, "FEXT must show wave propagation dips, not monotonic increase"

    def test_crosstalk_very_short_cable(self):
        """A 1 mm cable should have very low crosstalk at 100 MHz.

        FEXT is negligible because beta*L is tiny.  NEXT depends on
        termination mismatch: with z_s = z_l = z_c (matched), NEXT
        is also negligible; with z_s = z_l = 50 ohm into a ~200 ohm
        line, the mismatch reflection keeps NEXT higher but still low.
        """
        # Matched terminations: both NEXT and FEXT are very low
        result_matched = mtl_crosstalk_two_wire(
            0.001, 0.5e-3, 5e-3, 10e-3, 100e6, 204.0, 204.0
        )
        assert result_matched["next_db"] < -80
        assert result_matched["fext_db"] < -80

        # 50-ohm mismatched terminations: FEXT still low, NEXT bounded
        result_50 = mtl_crosstalk_two_wire(
            0.001, 0.5e-3, 5e-3, 10e-3, 100e6, 50.0, 50.0
        )
        assert result_50["fext_db"] < -80
        assert result_50["next_db"] < -40  # mismatch reflection raises NEXT

    def test_wider_separation_reduces_crosstalk(self):
        """Increasing wire separation should reduce both NEXT and FEXT."""
        close = mtl_crosstalk_two_wire(1.0, 0.5e-3, 3e-3, 10e-3, 100e6, 50.0, 50.0)
        far = mtl_crosstalk_two_wire(1.0, 0.5e-3, 30e-3, 10e-3, 100e6, 50.0, 50.0)
        assert far["next_db"] < close["next_db"]
        assert far["fext_db"] < close["fext_db"]

    def test_output_keys_present(self):
        """All expected output keys should be in the result dict."""
        result = mtl_crosstalk_two_wire(1.0, 0.5e-3, 5e-3, 10e-3, 100e6, 50.0, 50.0)
        expected_keys = {
            "next_db", "fext_db", "next_voltage", "fext_voltage",
            "phase_velocity_m_per_s", "beta_l_rad",
        }
        assert expected_keys == set(result.keys())

    def test_crosstalk_values_are_negative_db(self):
        """Crosstalk in dB must be <= 0 (coupling cannot exceed source)."""
        result = mtl_crosstalk_two_wire(1.0, 0.5e-3, 5e-3, 10e-3, 100e6, 50.0, 50.0)
        assert result["next_db"] <= 0.0
        assert result["fext_db"] <= 0.0

    def test_phase_velocity_near_speed_of_light(self):
        """For air dielectric, phase velocity should be close to c."""
        result = mtl_crosstalk_two_wire(1.0, 0.5e-3, 5e-3, 10e-3, 100e6, 50.0, 50.0)
        assert result["phase_velocity_m_per_s"] == pytest.approx(C, rel=0.1)

    def test_longer_cable_more_crosstalk(self):
        """Longer cables accumulate more coupling."""
        short = mtl_crosstalk_two_wire(0.1, 0.5e-3, 5e-3, 10e-3, 50e6, 50.0, 50.0)
        long = mtl_crosstalk_two_wire(2.0, 0.5e-3, 5e-3, 10e-3, 50e6, 50.0, 50.0)
        assert long["next_db"] > short["next_db"]


class TestTransferImpedance:
    """Validate Kley braid transfer impedance model."""

    def test_rg58_low_freq(self):
        """At low freq, Z_t should be close to DC resistance (~14 mohm/m for RG-58)."""
        zt = transfer_impedance_kley(1e3, 14e-3, 200e3, 1e-9)
        assert abs(zt) == pytest.approx(14e-3, rel=0.1)

    def test_zt_increases_with_frequency(self):
        """Z_t magnitude must increase with frequency (skin effect + leakage)."""
        zt_low = abs(transfer_impedance_kley(1e3, 14e-3, 200e3, 1e-9))
        zt_high = abs(transfer_impedance_kley(100e6, 14e-3, 200e3, 1e-9))
        assert zt_high > zt_low

    def test_dc_resistance_dominates_below_corner(self):
        """Well below f_corner, real part should be ~R_dc."""
        zt = transfer_impedance_kley(100, 14e-3, 200e3, 1e-9)
        assert zt.real == pytest.approx(14e-3, rel=0.01)

    def test_reactive_part_above_corner(self):
        """Above corner freq, imaginary part should be significant."""
        zt = transfer_impedance_kley(10e6, 14e-3, 200e3, 1e-9)
        assert abs(zt.imag) > 0

    def test_zt_returns_complex(self):
        """Result should be a complex number."""
        zt = transfer_impedance_kley(1e6, 14e-3, 200e3, 1e-9)
        assert isinstance(zt, complex)


class TestCableSE:
    """Validate cable SE from transfer impedance."""

    def test_good_braid_high_se(self):
        """A 14 mohm/m braid over 1 m in 50-ohm system should give > 40 dB SE."""
        se = cable_se_from_zt(1e6, 14e-3, 1.0, 50.0)
        assert se > 40

    def test_se_decreases_with_length(self):
        """Longer cable means more Z_t*L, so less SE."""
        se_short = cable_se_from_zt(1e6, 14e-3, 0.5, 50.0)
        se_long = cable_se_from_zt(1e6, 14e-3, 5.0, 50.0)
        assert se_short > se_long

    def test_zero_zt_gives_max_se(self):
        """Zero transfer impedance should give perfect (200 dB) shielding."""
        se = cable_se_from_zt(1e6, 0.0, 1.0, 50.0)
        assert se == 200.0

    def test_se_always_non_negative(self):
        """SE should be >= 0 even for very poor shields."""
        se = cable_se_from_zt(1e6, 100.0, 10.0, 50.0)
        assert se >= 0.0
