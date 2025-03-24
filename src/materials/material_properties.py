"""
Material properties and data handling for EMI shielding calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from periodictable import elements

class MaterialProperties:
    def __init__(self):
        self.elements_data = self._initialize_elements_data()
        
    def _initialize_elements_data(self) -> Dict:
        """
        Initialize basic element properties from the periodic table.
        """
        elements_data = {}
        for element in elements:
            if element.number > 0:  # Skip neutrons
                elements_data[element.symbol] = {
                    "atomic_number": element.number,
                    "atomic_mass": float(element.mass),
                    "density": float(element.density if element.density else 0),
                    "conductivity": self._get_element_conductivity(element.symbol),
                    "relative_permeability": self._get_element_permeability(element.symbol),
                    "relative_permittivity": self._get_element_permittivity(element.symbol)
                }
        return elements_data
    
    def _get_element_conductivity(self, symbol: str) -> float:
        """
        Get electrical conductivity for an element (S/m).
        These are approximate values at room temperature.
        """
        conductivity_data = {
            "Ag": 6.30e7,  # Silver
            "Cu": 5.96e7,  # Copper
            "Au": 4.10e7,  # Gold
            "Al": 3.50e7,  # Aluminum
            "Fe": 1.00e7,  # Iron
            "Ni": 1.43e7,  # Nickel
            "Pt": 9.43e6,  # Platinum
            "Pb": 4.55e6,  # Lead
            "Sn": 9.17e6,  # Tin
            "Ti": 2.38e6,  # Titanium
        }
        return conductivity_data.get(symbol, 1e3)  # Default value for unknown elements
    
    def _get_element_permeability(self, symbol: str) -> float:
        """
        Get relative magnetic permeability for an element.
        These are approximate values at room temperature.
        """
        permeability_data = {
            "Fe": 5000,    # Iron
            "Ni": 100,     # Nickel
            "Co": 250,     # Cobalt
            "Gd": 1.48,    # Gadolinium
            "Dy": 1.7,     # Dysprosium
        }
        return permeability_data.get(symbol, 1.0)  # Default to non-magnetic
    
    def _get_element_permittivity(self, symbol: str) -> float:
        """
        Get relative permittivity (dielectric constant) for an element.
        These are approximate values.
        """
        permittivity_data = {
            "Si": 11.7,    # Silicon
            "Ge": 16.0,    # Germanium
            "C": 5.7,      # Diamond form
            "GaAs": 12.9,  # Gallium Arsenide
        }
        return permittivity_data.get(symbol, 1.0)  # Default value
    
    def calculate_mixture_properties(self, composition: Dict[str, float]) -> Dict:
        """
        Calculate effective properties for a mixture of elements.
        Uses simple mixture rules for demonstration.
        
        Args:
            composition: Dictionary of element symbols and their weight fractions
            
        Returns:
            Dictionary of effective material properties
        """
        # Normalize weight fractions
        total_weight = sum(composition.values())
        normalized_composition = {k: v/total_weight for k, v in composition.items()}
        
        # Initialize properties
        effective_properties = {
            "conductivity": 0.0,
            "relative_permeability": 0.0,
            "relative_permittivity": 0.0,
            "density": 0.0
        }
        
        # Calculate weighted properties
        for element, fraction in normalized_composition.items():
            if element in self.elements_data:
                element_data = self.elements_data[element]
                # Simplified mixing rules - more sophisticated models needed for real applications
                effective_properties["conductivity"] += fraction * element_data["conductivity"]
                effective_properties["relative_permeability"] += fraction * element_data["relative_permeability"]
                effective_properties["relative_permittivity"] += fraction * element_data["relative_permittivity"]
                effective_properties["density"] += fraction * element_data["density"]
        
        return effective_properties
    
    def get_element_properties(self, symbol: str) -> Optional[Dict]:
        """
        Get properties for a single element.
        
        Args:
            symbol: Element symbol
            
        Returns:
            Dictionary of element properties or None if not found
        """
        return self.elements_data.get(symbol)
    
    def validate_composition(self, composition: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate a material composition.
        
        Args:
            composition: Dictionary of element symbols and their weight fractions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not composition:
            return False, "Composition is empty"
        
        # Check if all elements exist
        for element in composition:
            if element not in self.elements_data:
                return False, f"Unknown element: {element}"
        
        # Check if weight fractions sum to approximately 1
        total = sum(composition.values())
        if not 0.99 <= total <= 1.01:
            return False, f"Weight fractions sum to {total}, should be close to 1.0"
        
        return True, "Valid composition"

    def estimate_frequency_dependent_properties(self, composition: Dict[str, float], 
                                             frequency: float) -> Dict:
        """
        Estimate frequency-dependent material properties.
        This is a simplified model - real materials have more complex behavior.
        
        Args:
            composition: Dictionary of element symbols and their weight fractions
            frequency: Frequency in Hz
            
        Returns:
            Dictionary of frequency-dependent properties
        """
        base_properties = self.calculate_mixture_properties(composition)
        
        # Simplified frequency dependence - real materials are more complex
        freq_ghz = frequency / 1e9
        
        # Conductivity typically decreases slightly with frequency
        conductivity_factor = 1.0 / (1.0 + 0.1 * freq_ghz)
        
        # Permeability typically decreases with frequency
        permeability_factor = 1.0 / (1.0 + 0.2 * freq_ghz)
        
        # Permittivity can be complex and frequency dependent
        permittivity_factor = 1.0 / (1.0 + 0.05 * freq_ghz)
        
        return {
            "conductivity": base_properties["conductivity"] * conductivity_factor,
            "relative_permeability": base_properties["relative_permeability"] * permeability_factor,
            "relative_permittivity": base_properties["relative_permittivity"] * permittivity_factor,
            "density": base_properties["density"]  # Density is not frequency dependent
        }
