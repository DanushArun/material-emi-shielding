"""
Core EMI shielding calculations based on electromagnetic theory.
Enhanced with advanced microstructure modeling and cooling rate dependencies.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
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
    """Performs electromagnetic interference shielding calculations based on electromagnetic theory."""

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
    
    def calculate_grain_size_effect(self, bulk_conductivity: float, 
                                  grain_size: float,
                                  electron_mean_free_path: float = 40e-9) -> float:
        """
        Calculate the effect of grain size on electrical conductivity.
        Uses the Mayadas-Shatzkes model for grain boundary scattering.
        
        Args:
            bulk_conductivity: Bulk electrical conductivity (S/m)
            grain_size: Average grain size (m)
            electron_mean_free_path: Electron mean free path (m), default 40nm
            
        Returns:
            Effective conductivity considering grain boundary scattering (S/m)
        """
        if grain_size <= 0:
            return bulk_conductivity
            
        # Reflection coefficient at grain boundaries (typically 0.1-0.5)
        R = 0.25  # Average value for most metals
        
        # Calculate the grain boundary scattering parameter
        alpha = electron_mean_free_path / grain_size * (R / (1 - R))
        
        # Mayadas-Shatzkes formula for conductivity reduction
        if alpha < 0.01:
            # For small alpha, use approximation
            conductivity_ratio = 1 - 1.5 * alpha
        else:
            # Full formula
            conductivity_ratio = 1 - (3/2) * alpha + 3 * alpha**2 - 3 * alpha**3 * np.log(1 + 1/alpha)
        
        # Ensure ratio is positive
        conductivity_ratio = max(conductivity_ratio, 0.1)
        
        return bulk_conductivity * conductivity_ratio

    def calculate_shielding_effectiveness(self, conductivity: float,
                                        relative_permeability: float,
                                        relative_permittivity: float,
                                        thickness: float,
                                        frequency: float,
                                        grain_size: Optional[float] = None,
                                        include_confidence: bool = True) -> Dict[str, float]:
        """
        Calculate total shielding effectiveness and its components.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            grain_size: Average grain size (m), optional
            include_confidence: Whether to include confidence estimation
            
        Returns:
            Dictionary with SE components, total SE (dB), and confidence metrics
        """
        # Store original conductivity
        bulk_conductivity = conductivity
        
        # Apply grain size effect if specified
        if grain_size is not None and grain_size > 0:
            conductivity = self.calculate_grain_size_effect(bulk_conductivity, grain_size)
        
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
        
        result = {
            'reflection_loss': reflection_loss,
            'absorption_loss': absorption_loss,
            'multiple_reflection_loss': multiple_reflection_loss,
            'total_se': total_se,
            'skin_depth': skin_depth,
            'intrinsic_impedance_real': eta.real,
            'intrinsic_impedance_imag': eta.imag,
            'propagation_constant_real': gamma.real,
            'propagation_constant_imag': gamma.imag,
            'bulk_conductivity': bulk_conductivity,
            'effective_conductivity': conductivity
        }
        
        if grain_size is not None:
            result['grain_size'] = grain_size
            result['conductivity_reduction'] = conductivity / bulk_conductivity
        
        if include_confidence:
            # Add confidence estimation
            confidence_data = self._estimate_confidence(
                conductivity, relative_permeability, relative_permittivity,
                thickness, frequency, result
            )
            result.update(confidence_data)
        
        return result
    
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
                       grain_size: Optional[float] = None,
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
            grain_size: Average grain size (m), optional
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
                thickness, freq, grain_size
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
    
    def thickness_sweep(self, conductivity: float,
                       relative_permeability: float,
                       relative_permittivity: float,
                       frequency: float,
                       grain_size: Optional[float] = None,
                       thickness_start: float = 0.1e-3,
                       thickness_end: float = 10e-3,
                       num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate SE across a thickness range.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            frequency: Frequency (Hz)
            grain_size: Average grain size (m), optional
            thickness_start: Start thickness (m)
            thickness_end: End thickness (m)
            num_points: Number of thickness points
            
        Returns:
            Dictionary with thickness array and SE components
        """
        thicknesses = np.linspace(thickness_start, thickness_end, num_points)
        
        reflection_losses = np.zeros(num_points)
        absorption_losses = np.zeros(num_points)
        total_ses = np.zeros(num_points)
        skin_depths = np.zeros(num_points)
        
        for i, thick in enumerate(thicknesses):
            result = self.calculate_shielding_effectiveness(
                conductivity, relative_permeability, relative_permittivity,
                thick, frequency, grain_size
            )
            
            reflection_losses[i] = result['reflection_loss']
            absorption_losses[i] = result['absorption_loss']
            total_ses[i] = result['total_se']
            skin_depths[i] = result['skin_depth']
        
        return {
            'thicknesses': thicknesses,
            'reflection_losses': reflection_losses,
            'absorption_losses': absorption_losses,
            'total_ses': total_ses,
            'skin_depths': skin_depths
        }
    
    def grain_size_sweep(self, conductivity: float,
                        relative_permeability: float,
                        relative_permittivity: float,
                        thickness: float,
                        frequency: float,
                        grain_start: float = 10e-9,
                        grain_end: float = 100e-6,
                        num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate SE across a grain size range.
        
        Args:
            conductivity: Electrical conductivity (S/m)
            relative_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            grain_start: Start grain size (m)
            grain_end: End grain size (m)
            num_points: Number of grain size points
            
        Returns:
            Dictionary with grain size array and SE components
        """
        grain_sizes = np.logspace(np.log10(grain_start), np.log10(grain_end), num_points)
        
        reflection_losses = np.zeros(num_points)
        absorption_losses = np.zeros(num_points)
        total_ses = np.zeros(num_points)
        effective_conductivities = np.zeros(num_points)
        
        for i, grain in enumerate(grain_sizes):
            result = self.calculate_shielding_effectiveness(
                conductivity, relative_permeability, relative_permittivity,
                thickness, frequency, grain
            )
            
            reflection_losses[i] = result['reflection_loss']
            absorption_losses[i] = result['absorption_loss']
            total_ses[i] = result['total_se']
            effective_conductivities[i] = result['effective_conductivity']
        
        return {
            'grain_sizes': grain_sizes,
            'reflection_losses': reflection_losses,
            'absorption_losses': absorption_losses,
            'total_ses': total_ses,
            'effective_conductivities': effective_conductivities
        }

    def _estimate_confidence(self, conductivity: float,
                           relative_permeability: float,
                           relative_permittivity: float,
                           thickness: float,
                           frequency: float,
                           result: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate confidence in the calculation results.
        
        Returns:
            Dictionary with confidence metrics
        """
        confidence = 1.0
        
        # Physics constraints confidence
        if result['total_se'] < 0:
            confidence *= 0.1
        
        if result['skin_depth'] <= 0 or result['skin_depth'] > 1:
            confidence *= 0.8
        
        # Check if parameters are in reasonable ranges
        if conductivity < 1e-10 or conductivity > 1e10:
            confidence *= 0.7
        
        if relative_permeability < 0.999 or relative_permeability > 1e6:
            confidence *= 0.8
        
        # Frequency range confidence
        if frequency < 1e3 or frequency > 1e12:  # Outside 1kHz - 1THz
            confidence *= 0.85
        
        # Thickness vs skin depth
        if thickness < result['skin_depth'] / 10:
            confidence *= 0.9  # Very thin shield
        
        # Convert to uncertainty in dB
        if confidence > 0.95:
            uncertainty_db = 2.0
        elif confidence > 0.85:
            uncertainty_db = 5.0
        elif confidence > 0.70:
            uncertainty_db = 10.0
        else:
            uncertainty_db = 15.0
        
        return {
            'confidence': confidence,
            'uncertainty_db': uncertainty_db,
            'confidence_level': self._get_confidence_level(confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence > 0.95:
            return 'high'
        elif confidence > 0.85:
            return 'medium'
        elif confidence > 0.70:
            return 'low'
        else:
            return 'very_low'
    
    def calculate_shielding_from_processing(self, 
                                          composition: Dict[str, float],
                                          processing: ProcessingParams,
                                          thickness: float,
                                          frequency: float,
                                          base_conductivity: float,
                                          base_permeability: float = 1.0,
                                          relative_permittivity: float = 1.0) -> Dict[str, float]:
        """
        Calculate EMI shielding effectiveness from processing conditions and composition.
        
        This is the revolutionary method that links cooling rate → microstructure → EMI performance.
        
        Args:
            composition: Elemental composition (weight %)
            processing: Processing parameters including cooling rate
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            base_conductivity: Single crystal conductivity (S/m)
            base_permeability: Relative permeability
            relative_permittivity: Relative permittivity
            
        Returns:
            Dictionary with EMI performance and microstructure details
        """
        # Step 1: Predict microstructure from cooling rate and composition
        microstructure = self.advanced_microstructure.predict_microstructure_from_cooling_rate(
            composition, processing
        )
        
        # Step 2: Calculate effective electromagnetic properties
        effective_conductivity = self.advanced_microstructure.calculate_effective_conductivity(
            base_conductivity, microstructure, frequency
        )
        
        effective_permeability = self.advanced_microstructure.calculate_effective_permeability(
            base_permeability, microstructure, frequency
        )
        
        # Step 3: Calculate EMI shielding effectiveness
        emi_result = self.calculate_shielding_effectiveness(
            effective_conductivity,
            effective_permeability.real,
            relative_permittivity,
            thickness,
            frequency,
            grain_size=microstructure.grain_size
        )
        
        # Step 4: Add microstructure information to results
        emi_result.update({
            'microstructure': {
                'grain_size': microstructure.grain_size,
                'dendrite_spacing': microstructure.dendrite_spacing,
                'secondary_phase_fraction': microstructure.secondary_phase_fraction,
                'magnetic_domain_size': microstructure.magnetic_domain_size,
                'texture_intensity': microstructure.texture_intensity,
                'dislocation_density': microstructure.dislocation_density
            },
            'processing': {
                'cooling_rate': processing.cooling_rate,
                'temperature_gradient': processing.temperature_gradient,
                'solidification_time': processing.solidification_time
            },
            'effective_properties': {
                'conductivity': effective_conductivity,
                'permeability_real': effective_permeability.real,
                'permeability_imag': effective_permeability.imag,
                'conductivity_enhancement_factor': effective_conductivity / base_conductivity
            }
        })
        
        return emi_result
    
    def optimize_cooling_rate_for_emi(self,
                                     composition: Dict[str, float],
                                     target_se: float,
                                     thickness: float,
                                     frequency: float,
                                     base_conductivity: float,
                                     cooling_rate_range: Tuple[float, float] = (0.1, 1000.0)) -> Dict[str, float]:
        """
        Optimize cooling rate to achieve target EMI shielding effectiveness.
        
        This revolutionary method finds the optimal processing conditions for desired EMI performance.
        
        Args:
            composition: Elemental composition (weight %)
            target_se: Target shielding effectiveness (dB)
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            base_conductivity: Single crystal conductivity (S/m)
            cooling_rate_range: Min and max cooling rates to explore (K/s)
            
        Returns:
            Dictionary with optimal cooling rate and achieved performance
        """
        min_rate, max_rate = cooling_rate_range
        tolerance = 0.1  # dB
        
        def se_objective(cooling_rate: float) -> float:
            """Objective function: difference between achieved and target SE."""
            processing = ProcessingParams(cooling_rate=cooling_rate)
            result = self.calculate_shielding_from_processing(
                composition, processing, thickness, frequency, base_conductivity
            )
            return abs(result['total_se'] - target_se)
        
        # Golden section search for optimal cooling rate
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        resphi = 2 - phi
        
        # Initial bracket
        a, b = min_rate, max_rate
        tol = (b - a) * 1e-5
        
        # Initial points
        x1 = a + resphi * (b - a)
        x2 = a + (1 - resphi) * (b - a)
        f1 = se_objective(x1)
        f2 = se_objective(x2)
        
        # Golden section iterations
        while abs(b - a) > tol:
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = se_objective(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (1 - resphi) * (b - a)
                f2 = se_objective(x2)
        
        optimal_cooling_rate = (a + b) / 2
        
        # Calculate final result with optimal cooling rate
        optimal_processing = ProcessingParams(cooling_rate=optimal_cooling_rate)
        optimal_result = self.calculate_shielding_from_processing(
            composition, optimal_processing, thickness, frequency, base_conductivity
        )
        
        return {
            'optimal_cooling_rate': optimal_cooling_rate,
            'achieved_se': optimal_result['total_se'],
            'target_se': target_se,
            'se_error': abs(optimal_result['total_se'] - target_se),
            'microstructure': optimal_result['microstructure'],
            'processing_window': {
                'min_cooling_rate': max(0.1, optimal_cooling_rate * 0.5),
                'max_cooling_rate': min(1000.0, optimal_cooling_rate * 2.0),
                'tolerance_range': f"±{tolerance} dB"
            }
        }
    
    def cooling_rate_sweep(self,
                          composition: Dict[str, float],
                          thickness: float,
                          frequency: float,
                          base_conductivity: float,
                          cooling_rates: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate EMI performance across a range of cooling rates.
        
        This method reveals how processing controls electromagnetic performance.
        
        Args:
            composition: Elemental composition (weight %)
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            base_conductivity: Single crystal conductivity (S/m)
            cooling_rates: Array of cooling rates (K/s), if None uses default range
            
        Returns:
            Dictionary with arrays of cooling rates and corresponding EMI performance
        """
        if cooling_rates is None:
            cooling_rates = np.logspace(-1, 3, 50)  # 0.1 to 1000 K/s
        
        total_ses = np.zeros_like(cooling_rates)
        grain_sizes = np.zeros_like(cooling_rates)
        dendrite_spacings = np.zeros_like(cooling_rates)
        effective_conductivities = np.zeros_like(cooling_rates)
        
        for i, cooling_rate in enumerate(cooling_rates):
            processing = ProcessingParams(cooling_rate=cooling_rate)
            result = self.calculate_shielding_from_processing(
                composition, processing, thickness, frequency, base_conductivity
            )
            
            total_ses[i] = result['total_se']
            grain_sizes[i] = result['microstructure']['grain_size']
            dendrite_spacings[i] = result['microstructure']['dendrite_spacing']
            effective_conductivities[i] = result['effective_properties']['conductivity']
        
        return {
            'cooling_rates': cooling_rates,
            'total_ses': total_ses,
            'grain_sizes': grain_sizes,
            'dendrite_spacings': dendrite_spacings,
            'effective_conductivities': effective_conductivities,
            'conductivity_enhancement_factors': effective_conductivities / base_conductivity
        }
    
    def predict_microstructure_evolution(self,
                                       composition: Dict[str, float],
                                       processing_history: List[ProcessingParams]) -> List[MicrostructureParams]:
        """
        Predict microstructure evolution through multi-step processing.
        
        This method handles complex processing routes (e.g., casting → rolling → annealing).
        
        Args:
            composition: Elemental composition (weight %)
            processing_history: List of sequential processing steps
            
        Returns:
            List of microstructure states after each processing step
        """
        microstructure_evolution = []
        current_microstructure = None
        
        for step, processing in enumerate(processing_history):
            if step == 0:
                # Initial microstructure from first processing step
                microstructure = self.advanced_microstructure.predict_microstructure_from_cooling_rate(
                    composition, processing
                )
            else:
                # Evolve from previous microstructure (simplified for now)
                # In reality, this would need complex recrystallization/phase transformation models
                microstructure = self.advanced_microstructure.predict_microstructure_from_cooling_rate(
                    composition, processing
                )
                
                # Apply inheritance from previous step (simplified)
                if current_microstructure is not None:
                    # Grain size evolution: faster cooling refines, slower cooling coarsens
                    if processing.cooling_rate > 10:
                        grain_refinement = 0.7
                    else:
                        grain_refinement = 1.3
                    
                    microstructure.grain_size = current_microstructure.grain_size * grain_refinement
                    microstructure.dislocation_density = max(
                        microstructure.dislocation_density,
                        current_microstructure.dislocation_density * 0.8
                    )
            
            microstructure_evolution.append(microstructure)
            current_microstructure = microstructure
        
        return microstructure_evolution
    
    def calculate_degraded_shielding_effectiveness(self,
                                                 composition: Dict[str, float],
                                                 processing: ProcessingParams,
                                                 mechanical_state: MechanicalState,
                                                 environmental_conditions: EnvironmentalConditions,
                                                 thickness: float,
                                                 frequency: float,
                                                 base_conductivity: float,
                                                 base_permeability: float = 1.0,
                                                 material_type: str = 'steel') -> Dict[str, float]:
        """
        Calculate EMI shielding effectiveness including mechanical degradation effects.
        
        This revolutionary method predicts EMI performance after mechanical loading,
        fatigue, thermal cycling, and environmental degradation.
        
        Args:
            composition: Elemental composition (weight %)
            processing: Processing parameters
            mechanical_state: Current mechanical state (stress, cycles, damage)
            environmental_conditions: Environmental conditions
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            base_conductivity: Virgin material conductivity (S/m)
            base_permeability: Virgin material permeability
            material_type: Material type for property lookup
            
        Returns:
            Dictionary with degraded EMI performance and reliability metrics
        """
        # Step 1: Calculate base microstructure-dependent properties
        base_result = self.calculate_shielding_from_processing(
            composition, processing, thickness, frequency, 
            base_conductivity, base_permeability
        )
        
        # Step 2: Apply mechanical degradation effects
        degraded_conductivity = self.mechanical_coupling.calculate_degraded_conductivity(
            base_result['effective_properties']['conductivity'],
            mechanical_state,
            self.mechanical_coupling.failure_models[FailureMode.FATIGUE_CRACKING],
            material_type
        )
        
        degraded_permeability = self.mechanical_coupling.calculate_degraded_permeability(
            base_result['effective_properties']['permeability_real'],
            mechanical_state,
            material_type
        )
        
        # Step 3: Recalculate EMI performance with degraded properties
        degraded_result = self.calculate_shielding_effectiveness(
            degraded_conductivity,
            degraded_permeability.real,
            1.0,  # relative_permittivity
            thickness,
            frequency
        )
        
        # Step 4: Calculate reliability metrics
        failure_probabilities = self.mechanical_coupling.predict_failure_mode(
            mechanical_state, environmental_conditions, material_type
        )
        
        remaining_life = self.mechanical_coupling.calculate_remaining_life(
            mechanical_state, environmental_conditions, material_type
        )
        
        # Step 5: Combine results
        degraded_result.update({
            'original_se': base_result['total_se'],
            'degraded_se': degraded_result['total_se'],
            'se_degradation': base_result['total_se'] - degraded_result['total_se'],
            'degradation_factor': degraded_result['total_se'] / base_result['total_se'],
            'mechanical_state': {
                'stress_max': np.max(np.linalg.eigvals(mechanical_state.stress)),
                'cycles': mechanical_state.cycles,
                'service_time_years': mechanical_state.time / (365 * 24 * 3600),
                'damage_parameter': mechanical_state.damage_parameter,
                'crack_density': mechanical_state.crack_density
            },
            'degraded_properties': {
                'conductivity': degraded_conductivity,
                'permeability_real': degraded_permeability.real,
                'permeability_imag': degraded_permeability.imag,
                'conductivity_degradation': 1 - degraded_conductivity / base_result['effective_properties']['conductivity']
            },
            'failure_analysis': {
                'failure_probabilities': {mode.value: prob for mode, prob in failure_probabilities.items()},
                'remaining_life_years': {mode: life / (365 * 24 * 3600) for mode, life in remaining_life.items()},
                'critical_failure_mode': max(failure_probabilities.items(), key=lambda x: x[1])[0].value if failure_probabilities else None
            }
        })
        
        return degraded_result
    
    def predict_emi_reliability_over_time(self,
                                        composition: Dict[str, float],
                                        processing: ProcessingParams,
                                        service_conditions: Dict[str, Any],
                                        thickness: float,
                                        frequency: float,
                                        base_conductivity: float,
                                        design_life_years: float = 20.0,
                                        time_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Predict EMI shielding effectiveness over service life.
        
        This method predicts how EMI performance degrades over 20+ years of service.
        
        Args:
            composition: Elemental composition (weight %)
            processing: Processing parameters
            service_conditions: Service loading and environment
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            base_conductivity: Virgin material conductivity (S/m)
            design_life_years: Design life (years)
            time_points: Number of time points to evaluate
            
        Returns:
            Dictionary with time-dependent EMI performance predictions
        """
        # Parse service conditions
        max_stress = service_conditions.get('max_stress', 100e6)  # Pa
        cycles_per_year = service_conditions.get('cycles_per_year', 1e6)
        temperature_range = service_conditions.get('temperature_range', (253, 373))  # K
        environment = service_conditions.get('environment', 'outdoor')
        
        # Create environmental conditions
        env_conditions = EnvironmentalConditions(
            temperature_range=temperature_range,
            humidity=0.7 if environment == 'humid' else 0.3,
            atmosphere='marine' if environment == 'marine' else 'air',
            thermal_cycles_per_year=365.0,
            corrosive_species=['Cl-'] if environment == 'marine' else None
        )
        
        # Time array
        total_time = design_life_years * 365 * 24 * 3600  # seconds
        times = np.linspace(0, total_time, time_points)
        time_years = times / (365 * 24 * 3600)
        
        # Initialize arrays
        se_values = np.zeros(time_points)
        degradation_factors = np.zeros(time_points)
        failure_probs = np.zeros(time_points)
        conductivity_values = np.zeros(time_points)
        
        # Create stress tensor (simplified uniaxial)
        stress_tensor = np.zeros((3, 3))
        stress_tensor[0, 0] = max_stress
        
        for i, time in enumerate(times):
            # Current mechanical state
            mechanical_state = MechanicalState(
                stress=stress_tensor,
                strain=np.zeros((3, 3)),  # Simplified
                temperature=np.mean(temperature_range),
                cycles=int(cycles_per_year * time_years[i]),
                time=time,
                crack_density=0.0,  # Will be calculated from damage
                damage_parameter=0.0
            )
            
            # Calculate degraded performance
            result = self.calculate_degraded_shielding_effectiveness(
                composition, processing, mechanical_state, env_conditions,
                thickness, frequency, base_conductivity
            )
            
            se_values[i] = result['degraded_se']
            degradation_factors[i] = result['degradation_factor']
            conductivity_values[i] = result['degraded_properties']['conductivity']
            
            # Calculate maximum failure probability
            failure_probs_dict = result['failure_analysis']['failure_probabilities']
            failure_probs[i] = max(failure_probs_dict.values()) if failure_probs_dict else 0.0
        
        return {
            'time_years': time_years,
            'se_values': se_values,
            'degradation_factors': degradation_factors,
            'failure_probabilities': failure_probs,
            'conductivity_values': conductivity_values,
            'initial_se': se_values[0] if len(se_values) > 0 else 0.0,
            'final_se': se_values[-1] if len(se_values) > 0 else 0.0,
            'total_degradation_percent': (1 - degradation_factors[-1]) * 100 if len(degradation_factors) > 0 else 0.0,
            'service_life_summary': {
                'design_life_years': design_life_years,
                'se_at_end_of_life': se_values[-1] if len(se_values) > 0 else 0.0,
                'reliability_at_end_of_life': 1 - failure_probs[-1] if len(failure_probs) > 0 else 1.0,
                'meets_design_requirements': se_values[-1] > se_values[0] * 0.8 if len(se_values) > 0 else False  # 80% retention
            }
        }
    
    def optimize_for_reliability(self,
                               composition: Dict[str, float],
                               service_conditions: Dict[str, Any],
                               thickness: float,
                               frequency: float,
                               base_conductivity: float,
                               min_se_end_of_life: float,
                               design_life_years: float = 20.0) -> Dict[str, Any]:
        """
        Optimize processing parameters for EMI reliability over service life.
        
        This method finds processing conditions that maintain EMI performance
        throughout the design life.
        
        Args:
            composition: Elemental composition (weight %)
            service_conditions: Service loading and environment
            thickness: Material thickness (m)
            frequency: Frequency (Hz)
            base_conductivity: Virgin material conductivity (S/m)
            min_se_end_of_life: Minimum SE required at end of life (dB)
            design_life_years: Design life (years)
            
        Returns:
            Dictionary with optimal processing and reliability analysis
        """
        # Define optimization objective
        def reliability_objective(cooling_rate: float) -> float:
            """Objective: maximize SE at end of life."""
            processing = ProcessingParams(cooling_rate=cooling_rate)
            
            reliability_prediction = self.predict_emi_reliability_over_time(
                composition, processing, service_conditions,
                thickness, frequency, base_conductivity, design_life_years, 50
            )
            
            final_se = reliability_prediction['final_se']
            
            # Penalty if below minimum requirement
            if final_se < min_se_end_of_life:
                penalty = (min_se_end_of_life - final_se) * 10
                return -(final_se - penalty)  # Minimize negative SE
            else:
                return -final_se  # Maximize SE (minimize negative)
        
        # Optimize cooling rate using golden section search
        from scipy.optimize import minimize_scalar
        
        result = minimize_scalar(
            reliability_objective,
            bounds=(0.1, 1000.0),
            method='bounded'
        )
        
        optimal_cooling_rate = result.x
        
        # Calculate final reliability prediction with optimal processing
        optimal_processing = ProcessingParams(cooling_rate=optimal_cooling_rate)
        final_prediction = self.predict_emi_reliability_over_time(
            composition, optimal_processing, service_conditions,
            thickness, frequency, base_conductivity, design_life_years, 100
        )
        
        return {
            'optimal_processing': {
                'cooling_rate': optimal_cooling_rate,
                'temperature_gradient': optimal_processing.temperature_gradient,
                'solidification_time': optimal_processing.solidification_time
            },
            'reliability_prediction': final_prediction,
            'optimization_summary': {
                'meets_requirements': final_prediction['final_se'] >= min_se_end_of_life,
                'se_margin': final_prediction['final_se'] - min_se_end_of_life,
                'reliability_margin': final_prediction['service_life_summary']['reliability_at_end_of_life'],
                'processing_sensitivity': {
                    'cooling_rate_tolerance': optimal_cooling_rate * 0.1,  # ±10%
                    'recommended_range': (optimal_cooling_rate * 0.9, optimal_cooling_rate * 1.1)
                }
            }
        }
    
    def design_for_application_reliability(self,
                                         application_requirements: Dict[str, Any],
                                         candidate_compositions: List[Dict[str, float]],
                                         design_life_years: float = 20.0) -> Dict[str, Any]:
        """
        Design complete EMI solution for specific application with reliability requirements.
        
        This method combines materials discovery, processing optimization, and 
        reliability analysis for complete application-specific design.
        
        Args:
            application_requirements: Application specifications and constraints
            candidate_compositions: List of candidate material compositions
            design_life_years: Design life requirement (years)
            
        Returns:
            Complete design solution with materials, processing, and reliability analysis
        """
        # Extract requirements
        target_se = application_requirements['target_se']  # dB
        frequency = application_requirements['frequency']  # Hz
        max_thickness = application_requirements.get('max_thickness', 5e-3)  # m
        service_conditions = application_requirements['service_conditions']
        reliability_requirement = application_requirements.get('reliability_requirement', 0.95)
        
        design_solutions = []
        
        for composition in candidate_compositions:
            # Estimate base conductivity from composition
            base_conductivity = self._estimate_conductivity_from_composition(composition)
            
            # Try different thicknesses
            for thickness in np.linspace(0.5e-3, max_thickness, 5):
                # Optimize processing for this composition and thickness
                optimization_result = self.optimize_for_reliability(
                    composition, service_conditions, thickness, frequency,
                    base_conductivity, target_se * 0.8, design_life_years  # 80% retention target
                )
                
                if optimization_result['optimization_summary']['meets_requirements']:
                    design_solutions.append({
                        'composition': composition,
                        'thickness': thickness,
                        'processing': optimization_result['optimal_processing'],
                        'performance': optimization_result['reliability_prediction'],
                        'scores': {
                            'initial_se': optimization_result['reliability_prediction']['initial_se'],
                            'final_se': optimization_result['reliability_prediction']['final_se'],
                            'reliability': optimization_result['reliability_prediction']['service_life_summary']['reliability_at_end_of_life'],
                            'weight_factor': 1 / thickness,  # Thinner is better
                            'cost_factor': self._estimate_material_cost(composition) / 100.0
                        }
                    })
        
        if not design_solutions:
            return {
                'success': False,
                'message': 'No solutions found that meet reliability requirements',
                'recommendations': 'Consider relaxing constraints or using thicker shields'
            }
        
        # Rank solutions by composite score
        for solution in design_solutions:
            scores = solution['scores']
            # Weighted composite score (higher is better)
            composite_score = (
                scores['final_se'] / target_se * 0.4 +  # Performance weight
                scores['reliability'] * 0.3 +  # Reliability weight
                scores['weight_factor'] * 0.2 +  # Weight factor
                (1 / scores['cost_factor']) * 0.1  # Cost factor (lower cost is better)
            )
            solution['composite_score'] = composite_score
        
        # Sort by composite score
        design_solutions.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return {
            'success': True,
            'recommended_solution': design_solutions[0],
            'alternative_solutions': design_solutions[1:3],  # Top 3 alternatives
            'design_summary': {
                'best_composition': design_solutions[0]['composition'],
                'optimal_thickness': design_solutions[0]['thickness'] * 1000,  # mm
                'processing_conditions': design_solutions[0]['processing'],
                'predicted_performance': {
                    'initial_se': design_solutions[0]['performance']['initial_se'],
                    'end_of_life_se': design_solutions[0]['performance']['final_se'],
                    'degradation_percent': design_solutions[0]['performance']['total_degradation_percent'],
                    'reliability': design_solutions[0]['performance']['service_life_summary']['reliability_at_end_of_life']
                }
            }
        }
    
    def _estimate_conductivity_from_composition(self, composition: Dict[str, float]) -> float:
        """Estimate electrical conductivity from composition (simplified)."""
        element_conductivities = {
            'Ag': 63e6, 'Cu': 59e6, 'Au': 45e6, 'Al': 37e6,
            'Fe': 10e6, 'Ni': 14e6, 'Co': 17e6, 'Zn': 17e6
        }
        
        total_conductivity = 0
        total_weight = 0
        
        for element, fraction in composition.items():
            if element in element_conductivities:
                total_conductivity += element_conductivities[element] * fraction / 100.0
                total_weight += fraction / 100.0
        
        return total_conductivity / total_weight if total_weight > 0 else 1e6
    
    def _estimate_material_cost(self, composition: Dict[str, float]) -> float:
        """Estimate material cost ($/kg) from composition."""
        element_costs = {
            'Fe': 0.5, 'Al': 1.8, 'Cu': 6.0, 'Ni': 13.0, 'Co': 32.0,
            'Ag': 800.0, 'Au': 65000.0, 'C': 0.1
        }
        
        total_cost = 0
        for element, fraction in composition.items():
            cost = element_costs.get(element, 10.0)  # Default cost
            total_cost += cost * fraction / 100.0
        
        return total_cost

# Create global instance
emi_calculator = EMICalculator()