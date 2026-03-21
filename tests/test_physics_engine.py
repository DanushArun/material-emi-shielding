import pytest
import numpy as np
from src.physics.emi_calculations import EMICalculator, emi_calculator
from src.utils.constants import MU_0


class TestEMICalculatorImport:
    def test_can_instantiate(self):
        calc = EMICalculator()
        assert calc is not None

    def test_global_instance_exists(self):
        assert emi_calculator is not None


class TestSkinDepth:
    def test_copper_1ghz(self):
        calc = EMICalculator()
        delta = calc.calculate_skin_depth(5.96e7, MU_0, 1e9)
        assert 1e-6 < delta < 5e-6  # ~2.06 um

    def test_aluminum_1ghz(self):
        calc = EMICalculator()
        delta = calc.calculate_skin_depth(3.5e7, MU_0, 1e9)
        assert 1e-6 < delta < 5e-6

    def test_zero_conductivity_returns_inf(self):
        calc = EMICalculator()
        delta = calc.calculate_skin_depth(0, MU_0, 1e9)
        assert delta == float('inf')


class TestShieldingEffectiveness:
    def test_copper_1mm_1ghz(self):
        calc = EMICalculator()
        result = calc.calculate_shielding_effectiveness(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            thickness=0.001,
            frequency=1e9,
        )
        assert result['total_se'] > 50
        assert 'reflection_loss' in result
        assert 'absorption_loss' in result
        assert 'skin_depth' in result

    def test_with_grain_size(self):
        calc = EMICalculator()
        result = calc.calculate_shielding_effectiveness(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            thickness=0.001,
            frequency=1e9,
            grain_size=50e-6,
        )
        assert result['total_se'] > 0
        assert 'conductivity_reduction' in result

    def test_se_increases_with_thickness(self):
        calc = EMICalculator()
        thin = calc.calculate_shielding_effectiveness(5.96e7, 1.0, 1.0, 0.0001, 1e9)
        thick = calc.calculate_shielding_effectiveness(5.96e7, 1.0, 1.0, 0.001, 1e9)
        assert thick['total_se'] > thin['total_se']


class TestFrequencySweep:
    def test_basic_sweep(self):
        calc = EMICalculator()
        result = calc.frequency_sweep(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            thickness=0.001,
            freq_start=1e6,
            freq_end=1e9,
            num_points=10,
        )
        assert len(result['frequencies']) == 10
        assert len(result['total_ses']) == 10
        assert len(result['reflection_losses']) == 10
        assert len(result['absorption_losses']) == 10


class TestThicknessSweep:
    def test_basic_sweep(self):
        calc = EMICalculator()
        result = calc.thickness_sweep(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            frequency=1e9,
            num_points=10,
        )
        assert len(result['thicknesses']) == 10
        assert len(result['total_ses']) == 10


class TestGrainSizeSweep:
    def test_basic_sweep(self):
        calc = EMICalculator()
        result = calc.grain_size_sweep(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            thickness=0.001,
            frequency=1e9,
            num_points=10,
        )
        assert len(result['grain_sizes']) == 10
        assert len(result['total_ses']) == 10
        assert len(result['effective_conductivities']) == 10


class TestOptimizeThickness:
    def test_copper_60db(self):
        calc = EMICalculator()
        result = calc.optimize_thickness(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            frequency=1e9,
            target_se=60.0,
        )
        assert result['achieved_se'] >= 55  # close to target
        assert result['optimal_thickness'] > 0
