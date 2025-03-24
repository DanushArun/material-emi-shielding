"""
Core physics calculations for EMI shielding prediction.
"""

import numpy as np
from scipy import constants
from typing import Dict, List, Tuple, Optional

class EMIShieldingCalculator:
    def __init__(self):
        self.epsilon_0 = constants.epsilon_0  # Vacuum permittivity
        self.mu_0 = constants.mu_0  # Vacuum permeability
        self.c = constants.c  # Speed of light

    def calculate_skin_depth(self, frequency: float, conductivity: float, 
                           relative_permeability: float) -> float:
        """
        Calculate the skin depth of a material.
        
        Args:
            frequency (float): Frequency in Hz
            conductivity (float): Electrical conductivity in S/m
            relative_permeability (float): Relative magnetic permeability
            
        Returns:
            float: Skin depth in meters
        """
        omega = 2 * np.pi * frequency
        mu = self.mu_0 * relative_permeability
        return np.sqrt(2 / (omega * mu * conductivity))

    def calculate_wave_impedance(self, frequency: float, relative_permittivity: float,
                               relative_permeability: float) -> float:
        """
        Calculate the wave impedance in a material.
        
        Args:
            frequency (float): Frequency in Hz
            relative_permittivity (float): Relative permittivity
            relative_permeability (float): Relative magnetic permeability
            
        Returns:
            float: Wave impedance in ohms
        """
        return np.sqrt((self.mu_0 * relative_permeability) / 
                      (self.epsilon_0 * relative_permittivity))

    def calculate_reflection_loss(self, frequency: float, conductivity: float,
                                relative_permeability: float) -> float:
        """
        Calculate reflection loss at normal incidence.
        
        Args:
            frequency (float): Frequency in Hz
            conductivity (float): Electrical conductivity in S/m
            relative_permeability (float): Relative magnetic permeability
            
        Returns:
            float: Reflection loss in dB
        """
        omega = 2 * np.pi * frequency
        mu = self.mu_0 * relative_permeability
        sigma = conductivity
        
        # Wave impedance of free space
        z0 = np.sqrt(self.mu_0 / self.epsilon_0)
        
        # Wave impedance in the material
        zm = np.sqrt(1j * omega * mu / sigma)
        
        # Reflection coefficient
        gamma = (zm - z0) / (zm + z0)
        
        # Reflection loss
        return -20 * np.log10(np.abs(gamma))

    def calculate_absorption_loss(self, frequency: float, conductivity: float,
                                relative_permeability: float, thickness: float) -> float:
        """
        Calculate absorption loss.
        
        Args:
            frequency (float): Frequency in Hz
            conductivity (float): Electrical conductivity in S/m
            relative_permeability (float): Relative magnetic permeability
            thickness (float): Material thickness in meters
            
        Returns:
            float: Absorption loss in dB
        """
        skin_depth = self.calculate_skin_depth(frequency, conductivity, relative_permeability)
        return 20 * (thickness / skin_depth) * np.log10(np.e)

    def calculate_multiple_reflection_loss(self, frequency: float, conductivity: float,
                                        relative_permeability: float, thickness: float) -> float:
        """
        Calculate multiple reflection loss.
        
        Args:
            frequency (float): Frequency in Hz
            conductivity (float): Electrical conductivity in S/m
            relative_permeability (float): Relative magnetic permeability
            thickness (float): Material thickness in meters
            
        Returns:
            float: Multiple reflection loss in dB
        """
        skin_depth = self.calculate_skin_depth(frequency, conductivity, relative_permeability)
        if thickness / skin_depth > 1.3:
            return 0  # Negligible for thick shields
        
        # Simplified model for multiple reflections
        return -20 * np.log10(1 - np.exp(-2 * thickness / skin_depth))

    def calculate_total_shielding_effectiveness(self, frequency: float, conductivity: float,
                                             relative_permeability: float, relative_permittivity: float,
                                             thickness: float) -> Dict[str, float]:
        """
        Calculate total shielding effectiveness and its components.
        
        Args:
            frequency (float): Frequency in Hz
            conductivity (float): Electrical conductivity in S/m
            relative_permeability (float): Relative magnetic permeability
            relative_permittivity (float): Relative permittivity
            thickness (float): Material thickness in meters
            
        Returns:
            Dict[str, float]: Dictionary containing reflection loss, absorption loss,
                            multiple reflection loss, and total shielding effectiveness
        """
        r_loss = self.calculate_reflection_loss(frequency, conductivity, relative_permeability)
        a_loss = self.calculate_absorption_loss(frequency, conductivity, relative_permeability, thickness)
        m_loss = self.calculate_multiple_reflection_loss(frequency, conductivity, relative_permeability, thickness)
        
        total_se = r_loss + a_loss + m_loss
        
        return {
            "reflection_loss": r_loss,
            "absorption_loss": a_loss,
            "multiple_reflection_loss": m_loss,
            "total_se": total_se
        }

    def calculate_frequency_response(self, freq_range: np.ndarray, material_params: Dict) -> Dict[str, np.ndarray]:
        """
        Calculate shielding effectiveness across a frequency range.
        
        Args:
            freq_range (np.ndarray): Array of frequencies in Hz
            material_params (Dict): Dictionary containing material parameters
                                 (conductivity, relative_permeability, relative_permittivity, thickness)
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing arrays of reflection loss, absorption loss,
                                 multiple reflection loss, and total SE for each frequency
        """
        results = {
            "frequency": freq_range,
            "reflection_loss": np.zeros_like(freq_range),
            "absorption_loss": np.zeros_like(freq_range),
            "multiple_reflection_loss": np.zeros_like(freq_range),
            "total_se": np.zeros_like(freq_range)
        }
        
        for i, freq in enumerate(freq_range):
            se_results = self.calculate_total_shielding_effectiveness(
                freq,
                material_params["conductivity"],
                material_params["relative_permeability"],
                material_params["relative_permittivity"],
                material_params["thickness"]
            )
            
            results["reflection_loss"][i] = se_results["reflection_loss"]
            results["absorption_loss"][i] = se_results["absorption_loss"]
            results["multiple_reflection_loss"][i] = se_results["multiple_reflection_loss"]
            results["total_se"][i] = se_results["total_se"]
            
        return results
