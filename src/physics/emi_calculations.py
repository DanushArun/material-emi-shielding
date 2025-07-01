"""
Core EMI shielding calculations based on electromagnetic theory.
"""

import numpy as np
from typing import Dict, Tuple, Optional
try:
    from ..utils.constants import (
        MU_0, EPSILON_0, Z_0, C,
        validate_conductivity, validate_permeability, 
        validate_permittivity, validate_frequency, validate_thickness
    )
except ImportError:
    # Fallback for when running from streamlit app
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.utils.constants import (
        MU_0, EPSILON_0, Z_0, C,
        validate_conductivity, validate_permeability, 
        validate_permittivity, validate_frequency, validate_thickness
    )

class EMICalculator:
    """Performs electromagnetic interference shielding calculations."""
    
    def __init__(self):
        """Initialize the EMI calculator."""
        pass
    
    def calculate_skin_depth(self, conductivity: float, permeability: float, 
                           frequency: float) -> float:
        """
        Calculate the skin depth of electromagnetic waves in a material.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            permeability: Absolute permeability (H/m)
            frequency: Frequency (Hz)
            
        Returns:
            Skin depth (m)
        """
        validate_conductivity(conductivity)
        validate_frequency(frequency)
        
        if conductivity == 0:
            return float('inf')
        
        omega = 2 * np.pi * frequency
        delta = np.sqrt(2 / (omega * permeability * conductivity))
        
        return delta
    
    def calculate_intrinsic_impedance(self, conductivity: float, 
                                    relative_permeability: float,
                                    relative_permittivity: float,
                                    frequency: float) -> complex:
        """
        Calculate the intrinsic impedance of a material.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            frequency: Frequency (Hz)
            
        Returns:
            Complex intrinsic impedance (Ω)
        """
        validate_conductivity(conductivity)
        validate_permeability(relative_permeability)
        validate_permittivity(relative_permittivity)
        validate_frequency(frequency)
        
        omega = 2 * np.pi * frequency
        
        # Absolute values
        mu = relative_permeability * MU_0
        epsilon = relative_permittivity * EPSILON_0
        
        # Complex permittivity
        epsilon_complex = epsilon - 1j * (conductivity / omega)
        
        # Intrinsic impedance
        eta = np.sqrt(mu / epsilon_complex)
        
        return eta
    
    def calculate_propagation_constant(self, conductivity: float,
                                     relative_permeability: float,
                                     relative_permittivity: float,
                                     frequency: float) -> complex:
        """
        Calculate the propagation constant.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            frequency: Frequency (Hz)
            
        Returns:
            Complex propagation constant (1/m)
        """
        omega = 2 * np.pi * frequency
        
        # Absolute values
        mu = relative_permeability * MU_0
        epsilon = relative_permittivity * EPSILON_0
        
        # Complex permittivity
        epsilon_complex = epsilon - 1j * (conductivity / omega)
        
        # Propagation constant
        gamma = 1j * omega * np.sqrt(mu * epsilon_complex)
        
        return gamma
    
    def calculate_reflection_loss(self, intrinsic_impedance: complex) -> float:
        """
        Calculate reflection loss at the air-material interface.
        
        Args:
            intrinsic_impedance: Complex intrinsic impedance of material (Ω)
            
        Returns:
            Reflection loss (dB)
        """
        # Reflection coefficient at air-material interface
        gamma_1 = (intrinsic_impedance - Z_0) / (intrinsic_impedance + Z_0)
        
        # Reflection coefficient at material-air interface
        gamma_2 = (Z_0 - intrinsic_impedance) / (Z_0 + intrinsic_impedance)
        
        # Power reflection coefficient
        R = abs(gamma_1) ** 2
        
        # Reflection loss in dB
        if R < 1:
            reflection_loss = -10 * np.log10(1 - R)
        else:
            reflection_loss = 0
        
        return reflection_loss
    
    def calculate_absorption_loss(self, thickness: float, 
                                propagation_constant: complex) -> float:
        """
        Calculate absorption loss through the material.
        
        Args:
            thickness: Material thickness (m)
            propagation_constant: Complex propagation constant (1/m)
            
        Returns:
            Absorption loss (dB)
        """
        validate_thickness(thickness)
        
        # Attenuation constant (real part of propagation constant)
        alpha = propagation_constant.real
        
        # Absorption loss in dB
        absorption_loss = 8.686 * alpha * thickness
        
        return absorption_loss
    
    def calculate_multiple_reflection_loss(self, intrinsic_impedance: complex,
                                         thickness: float,
                                         propagation_constant: complex) -> float:
        """
        Calculate multiple reflection loss.
        
        Args:
            intrinsic_impedance: Complex intrinsic impedance (Ω)
            thickness: Material thickness (m)
            propagation_constant: Complex propagation constant (1/m)
            
        Returns:
            Multiple reflection loss (dB)
        """
        # Reflection coefficients
        gamma_1 = (intrinsic_impedance - Z_0) / (intrinsic_impedance + Z_0)
        gamma_2 = (Z_0 - intrinsic_impedance) / (Z_0 + intrinsic_impedance)
        
        # Transmission through material
        exp_term = np.exp(-2 * propagation_constant * thickness)
        
        # Multiple reflection factor
        K = (1 - gamma_1 * gamma_2 * exp_term) / (1 + gamma_1 * gamma_2 * exp_term)
        
        # Multiple reflection loss in dB
        multiple_reflection_loss = -20 * np.log10(abs(K))
        
        # Only significant for thin materials or low absorption
        absorption_loss = self.calculate_absorption_loss(thickness, propagation_constant)
        if absorption_loss > 15:
            multiple_reflection_loss = 0
        
        return multiple_reflection_loss
    
    def calculate_shielding_effectiveness(self, conductivity: float,
                                        relative_permeability: float,
                                        relative_permittivity: float,
                                        thickness: float,
                                        frequency: float) -> Dict[str, float]:
        """
        Calculate total shielding effectiveness and its components.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            
        Returns:
            Dictionary with SE components and total SE (dB)
        """
        # Calculate electromagnetic properties
        eta = self.calculate_intrinsic_impedance(
            conductivity, relative_permeability, relative_permittivity, frequency
        )
        
        gamma = self.calculate_propagation_constant(
            conductivity, relative_permeability, relative_permittivity, frequency
        )
        
        # Calculate SE components
        reflection_loss = self.calculate_reflection_loss(eta)
        absorption_loss = self.calculate_absorption_loss(thickness, gamma)
        multiple_reflection_loss = self.calculate_multiple_reflection_loss(
            eta, thickness, gamma
        )
        
        # Total shielding effectiveness
        total_se = reflection_loss + absorption_loss + multiple_reflection_loss
        
        # Calculate skin depth for reference
        mu = relative_permeability * MU_0
        skin_depth = self.calculate_skin_depth(conductivity, mu, frequency)
        
        return {
            'reflection_loss': reflection_loss,
            'absorption_loss': absorption_loss,
            'multiple_reflection_loss': multiple_reflection_loss,
            'total_se': total_se,
            'skin_depth': skin_depth,
            'intrinsic_impedance_real': eta.real,
            'intrinsic_impedance_imag': eta.imag,
            'propagation_constant_real': gamma.real,
            'propagation_constant_imag': gamma.imag
        }
    
    def calculate_near_field_shielding(self, conductivity: float,
                                     relative_permeability: float,
                                     thickness: float,
                                     frequency: float,
                                     source_distance: float,
                                     source_type: str = 'electric') -> float:
        """
        Calculate shielding effectiveness in the near field.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            source_distance: Distance from source (m)
            source_type: 'electric' or 'magnetic'
            
        Returns:
            Near field shielding effectiveness (dB)
        """
        wavelength = C / frequency
        
        # Check if in near field (r < λ/2π)
        near_field_boundary = wavelength / (2 * np.pi)
        
        if source_distance > near_field_boundary:
            # Use far field calculations
            return self.calculate_shielding_effectiveness(
                conductivity, relative_permeability, 1, thickness, frequency
            )['total_se']
        
        # Near field corrections
        mu = relative_permeability * MU_0
        skin_depth = self.calculate_skin_depth(conductivity, mu, frequency)
        
        if source_type == 'electric':
            # Electric field dominant
            k = 3 / (2 * np.pi * frequency * source_distance)
            se_correction = 20 * np.log10(1 + k)
        else:
            # Magnetic field dominant
            k = 1 / (2 * np.pi * frequency * mu * source_distance)
            se_correction = -20 * np.log10(1 + k)
        
        # Base SE
        base_se = 8.686 * thickness / skin_depth
        
        return base_se + se_correction
    
    def optimize_thickness(self, conductivity: float,
                         relative_permeability: float,
                         relative_permittivity: float,
                         frequency: float,
                         target_se: float,
                         max_thickness: float = 0.01) -> Dict[str, float]:
        """
        Optimize material thickness for target shielding effectiveness.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            frequency: Frequency (Hz)
            target_se: Target shielding effectiveness (dB)
            max_thickness: Maximum allowed thickness (m)
            
        Returns:
            Dictionary with optimal thickness and achieved SE
        """
        # Binary search for optimal thickness
        min_t = 0
        max_t = max_thickness
        tolerance = 1e-6
        
        while (max_t - min_t) > tolerance:
            mid_t = (min_t + max_t) / 2
            
            result = self.calculate_shielding_effectiveness(
                conductivity, relative_permeability, relative_permittivity,
                mid_t, frequency
            )
            
            if result['total_se'] < target_se:
                min_t = mid_t
            else:
                max_t = mid_t
        
        optimal_thickness = max_t
        
        # Calculate final SE
        final_result = self.calculate_shielding_effectiveness(
            conductivity, relative_permeability, relative_permittivity,
            optimal_thickness, frequency
        )
        
        return {
            'optimal_thickness': optimal_thickness,
            'achieved_se': final_result['total_se'],
            'reflection_loss': final_result['reflection_loss'],
            'absorption_loss': final_result['absorption_loss'],
            'skin_depths': optimal_thickness / final_result['skin_depth']
        }
    
    def frequency_sweep(self, conductivity: float,
                       relative_permeability: float,
                       relative_permittivity: float,
                       thickness: float,
                       freq_start: float = 1e6,
                       freq_end: float = 10e9,
                       num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate SE across a frequency range.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            thickness: Material thickness (m)
            freq_start: Start frequency (Hz)
            freq_end: End frequency (Hz)
            num_points: Number of frequency points
            
        Returns:
            Dictionary with frequency array and SE components
        """
        frequencies = np.logspace(np.log10(freq_start), np.log10(freq_end), num_points)
        
        reflection_losses = np.zeros(num_points)
        absorption_losses = np.zeros(num_points)
        total_ses = np.zeros(num_points)
        skin_depths = np.zeros(num_points)
        
        for i, freq in enumerate(frequencies):
            result = self.calculate_shielding_effectiveness(
                conductivity, relative_permeability, relative_permittivity,
                thickness, freq
            )
            
            reflection_losses[i] = result['reflection_loss']
            absorption_losses[i] = result['absorption_loss']
            total_ses[i] = result['total_se']
            skin_depths[i] = result['skin_depth']
        
        return {
            'frequencies': frequencies,
            'reflection_losses': reflection_losses,
            'absorption_losses': absorption_losses,
            'total_ses': total_ses,
            'skin_depths': skin_depths
        }

# Create global instance
emi_calculator = EMICalculator()