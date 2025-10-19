"""
Advanced microstructure modeling for EMI shielding materials.
Implements cooling rate dependent microstructure evolution and its effects on electromagnetic properties.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

@dataclass
class MicrostructureParams:
    """Parameters describing material microstructure."""
    grain_size: float = 50e-6  # Average grain size (m)
    dendrite_spacing: float = 20e-6  # Primary dendrite arm spacing (m)
    secondary_phase_fraction: float = 0.0  # Volume fraction of secondary phases
    magnetic_domain_size: float = 100e-6  # Magnetic domain size (m)
    texture_intensity: float = 1.0  # Texture parameter (1 = random, >1 = textured)
    porosity: float = 0.0  # Porosity fraction
    dislocation_density: float = 1e12  # Dislocation density (1/m²)

@dataclass
class ProcessingParams:
    """Parameters describing material processing conditions."""
    cooling_rate: float = 1.0  # Cooling rate (K/s)
    temperature_gradient: float = 1000.0  # Temperature gradient (K/m)
    melt_temperature: float = 1500.0  # Melting temperature (K)
    solidification_time: float = 100.0  # Solidification time (s)
    atmosphere: str = "air"  # Processing atmosphere
    pressure: float = 101325.0  # Processing pressure (Pa)

class AdvancedMicrostructure:
    """Advanced microstructure modeling for electromagnetic property prediction."""
    
    def __init__(self):
        """Initialize the advanced microstructure calculator."""
        # Material-specific constants for common EMI materials
        self.material_constants = {
            'Fe': {
                'dendrite_constant': 80e-6,  # K^0.5 * s^0.5 / m
                'grain_constant': 150e-6,
                'magnetic_domain_constant': 200e-6,
                'curie_temperature': 1043.0,  # K
                'saturation_magnetization': 2.16,  # T
            },
            'Cu': {
                'dendrite_constant': 45e-6,
                'grain_constant': 100e-6, 
                'magnetic_domain_constant': 0,  # Non-magnetic
                'curie_temperature': 0,
                'saturation_magnetization': 0,
            },
            'Al': {
                'dendrite_constant': 35e-6,
                'grain_constant': 80e-6,
                'magnetic_domain_constant': 0,
                'curie_temperature': 0,
                'saturation_magnetization': 0,
            },
            'Ni': {
                'dendrite_constant': 60e-6,
                'grain_constant': 120e-6,
                'magnetic_domain_constant': 150e-6,
                'curie_temperature': 631.0,
                'saturation_magnetization': 0.64,
            }
        }
    
    def predict_microstructure_from_cooling_rate(self, 
                                                composition: Dict[str, float],
                                                processing: ProcessingParams) -> MicrostructureParams:
        """
        Predict microstructure parameters from cooling rate and composition.
        
        Args:
            composition: Elemental composition (weight %)
            processing: Processing parameters
            
        Returns:
            Predicted microstructure parameters
        """
        # Primary dendrite arm spacing (Kurz-Fisher model)
        dendrite_spacing = self._calculate_dendrite_spacing(composition, processing)
        
        # Grain size from dendrite spacing and cooling rate
        grain_size = self._calculate_grain_size(composition, processing, dendrite_spacing)
        
        # Secondary phase precipitation
        secondary_phase_fraction = self._calculate_secondary_phases(composition, processing)
        
        # Magnetic domain structure (for ferromagnetic materials)
        magnetic_domain_size = self._calculate_magnetic_domains(composition, processing, grain_size)
        
        # Crystallographic texture development
        texture_intensity = self._calculate_texture(composition, processing)
        
        # Defect density
        dislocation_density = self._calculate_dislocation_density(processing)
        
        return MicrostructureParams(
            grain_size=grain_size,
            dendrite_spacing=dendrite_spacing,
            secondary_phase_fraction=secondary_phase_fraction,
            magnetic_domain_size=magnetic_domain_size,
            texture_intensity=texture_intensity,
            dislocation_density=dislocation_density
        )
    
    def _calculate_dendrite_spacing(self, composition: Dict[str, float], 
                                   processing: ProcessingParams) -> float:
        """
        Calculate primary dendrite arm spacing using Kurz-Fisher model.
        
        λ₁ = A * (G * R)^(-n)
        where G is temperature gradient, R is cooling rate, n ≈ 0.25
        """
        # Get material-specific constant
        primary_element = max(composition, key=composition.get)
        A = self.material_constants.get(primary_element, {}).get('dendrite_constant', 50e-6)
        
        # Kurz-Fisher model
        G = processing.temperature_gradient  # K/m
        R = processing.cooling_rate  # K/s
        n = 0.25  # Universal exponent
        
        # Account for compositional effects
        solute_effect = 1.0
        for element, fraction in composition.items():
            if element != primary_element and fraction > 1.0:
                # Solute elements refine dendrite spacing
                solute_effect *= (1 + 0.1 * fraction / 100.0)
        
        dendrite_spacing = A * (G * R)**(-n) / solute_effect
        
        # Physical limits: 1μm to 1mm
        return np.clip(dendrite_spacing, 1e-6, 1e-3)
    
    def _calculate_grain_size(self, composition: Dict[str, float],
                             processing: ProcessingParams, 
                             dendrite_spacing: float) -> float:
        """
        Calculate grain size from dendrite spacing and cooling rate.
        
        Grain size is typically 2-5 times dendrite spacing for cast materials.
        Fast cooling creates finer grains.
        """
        primary_element = max(composition, key=composition.get)
        base_ratio = 3.0  # Typical grain/dendrite ratio
        
        # Cooling rate effect on grain refinement
        if processing.cooling_rate > 10.0:  # Fast cooling
            grain_refinement = 0.5 + 0.5 * np.exp(-processing.cooling_rate / 100.0)
        else:  # Slow cooling
            grain_refinement = 1.0 + 0.5 * np.log10(processing.cooling_rate + 1)
        
        # Inoculant effects (simplified)
        inoculant_effect = 1.0
        if 'Ti' in composition or 'Zr' in composition:
            inoculant_effect = 0.3  # Strong grain refinement
        elif 'Al' in composition and composition['Al'] > 0.1:
            inoculant_effect = 0.7  # Moderate grain refinement
        
        grain_size = base_ratio * dendrite_spacing * grain_refinement * inoculant_effect
        
        # Physical limits: 100nm to 10mm
        return np.clip(grain_size, 100e-9, 10e-3)
    
    def _calculate_secondary_phases(self, composition: Dict[str, float],
                                   processing: ProcessingParams) -> float:
        """
        Calculate secondary phase fraction based on composition and cooling rate.
        
        Slower cooling allows more equilibrium precipitation.
        """
        secondary_fraction = 0.0
        
        # Carbon content effect (carbides)
        if 'C' in composition and composition['C'] > 0.01:
            carbon_fraction = composition['C'] / 100.0
            # Slower cooling allows more carbide precipitation
            cooling_factor = 1.0 / (1.0 + processing.cooling_rate / 10.0)
            secondary_fraction += carbon_fraction * 0.8 * cooling_factor
        
        # Intermetallic phases
        if 'Al' in composition and 'Fe' in composition:
            al_content = composition.get('Al', 0) / 100.0
            fe_content = composition.get('Fe', 0) / 100.0
            if al_content > 0.05 and fe_content > 0.1:
                # FeAl intermetallics
                secondary_fraction += min(al_content, fe_content) * 0.5
        
        # Oxide inclusions (from atmosphere)
        if processing.atmosphere == 'air' and processing.cooling_rate < 1.0:
            oxide_fraction = 0.001 * (2.0 - processing.cooling_rate)
            secondary_fraction += max(0, oxide_fraction)
        
        return min(secondary_fraction, 0.3)  # Cap at 30%
    
    def _calculate_magnetic_domains(self, composition: Dict[str, float],
                                   processing: ProcessingParams,
                                   grain_size: float) -> float:
        """
        Calculate magnetic domain size for ferromagnetic materials.
        
        Domain size depends on grain size, magnetocrystalline anisotropy,
        and exchange energy.
        """
        # Check if material is ferromagnetic
        ferromagnetic_elements = ['Fe', 'Ni', 'Co']
        ferromagnetic_content = sum(composition.get(elem, 0) 
                                  for elem in ferromagnetic_elements)
        
        if ferromagnetic_content < 50:  # Less than 50% ferromagnetic content
            return 0.0  # Non-magnetic
        
        # Domain size is typically 10-1000 times smaller than grain size
        domain_grain_ratio = 0.1
        
        # Cooling rate affects domain structure
        if processing.cooling_rate > 100:  # Fast cooling
            domain_grain_ratio *= 0.5  # Smaller domains
        elif processing.cooling_rate < 1:  # Slow cooling
            domain_grain_ratio *= 2.0  # Larger domains
        
        # Magnetocrystalline anisotropy effect
        primary_element = max((elem for elem in ferromagnetic_elements 
                             if elem in composition), 
                            key=lambda x: composition.get(x, 0), default='Fe')
        
        if primary_element == 'Fe':
            K1 = 48000  # J/m³ (magnetocrystalline anisotropy)
        elif primary_element == 'Ni':
            K1 = -5700  # J/m³
        else:  # Co
            K1 = 530000  # J/m³
        
        # Domain wall energy considerations
        domain_size = grain_size * domain_grain_ratio * np.sqrt(abs(K1) / 48000)
        
        # Physical limits: 1μm to 1mm
        return np.clip(domain_size, 1e-6, 1e-3)
    
    def _calculate_texture(self, composition: Dict[str, float],
                          processing: ProcessingParams) -> float:
        """
        Calculate crystallographic texture intensity.
        
        Texture affects electromagnetic anisotropy.
        """
        texture_intensity = 1.0  # Start with random orientation
        
        # Directional solidification creates texture
        if processing.temperature_gradient > 5000:  # High gradient
            texture_intensity = 2.0 + processing.temperature_gradient / 10000
        
        # Rolling or deformation (simplified - would need actual processing history)
        if processing.cooling_rate > 1000:  # Rapid quenching suggests deformation
            texture_intensity *= 1.5
        
        # Anisotropic crystal structure elements
        if 'Zn' in composition and composition['Zn'] > 10:
            texture_intensity *= 1.3  # HCP structure
        
        return min(texture_intensity, 5.0)  # Cap at moderate texture
    
    def _calculate_dislocation_density(self, processing: ProcessingParams) -> float:
        """
        Calculate dislocation density from processing conditions.
        """
        base_density = 1e12  # 1/m² (annealed condition)
        
        # Cooling rate increases dislocation density
        if processing.cooling_rate > 10:
            thermal_stress_factor = 1 + processing.cooling_rate / 100
            base_density *= thermal_stress_factor
        
        # Physical limits: 1e10 to 1e16 /m²
        return np.clip(base_density, 1e10, 1e16)
    
    def calculate_effective_conductivity(self, bulk_conductivity: float,
                                       microstructure: MicrostructureParams,
                                       frequency: float = 1e9) -> float:
        """
        Calculate effective conductivity including all microstructural effects.
        
        Args:
            bulk_conductivity: Single crystal conductivity (S/m)
            microstructure: Microstructure parameters
            frequency: Frequency (Hz)
            
        Returns:
            Effective conductivity (S/m)
        """
        conductivity = bulk_conductivity
        
        # Grain boundary scattering (Mayadas-Shatzkes model)
        conductivity = self._apply_grain_boundary_scattering(
            conductivity, microstructure.grain_size
        )
        
        # Dislocation scattering
        conductivity = self._apply_dislocation_scattering(
            conductivity, microstructure.dislocation_density
        )
        
        # Secondary phase effects
        conductivity = self._apply_secondary_phase_effects(
            conductivity, microstructure.secondary_phase_fraction
        )
        
        # Porosity effects
        if microstructure.porosity > 0:
            conductivity = self._apply_porosity_effects(
                conductivity, microstructure.porosity
            )
        
        # Frequency-dependent effects
        conductivity = self._apply_frequency_effects(
            conductivity, microstructure, frequency
        )
        
        return max(conductivity, 1e-10)  # Minimum conductivity
    
    def _apply_grain_boundary_scattering(self, conductivity: float, 
                                        grain_size: float) -> float:
        """Apply Mayadas-Shatzkes grain boundary scattering model."""
        if grain_size <= 0:
            return conductivity
            
        # Enhanced Mayadas-Shatzkes with temperature dependence
        electron_mean_free_path = 40e-9  # m
        R = 0.25  # Reflection coefficient
        
        alpha = electron_mean_free_path / grain_size * (R / (1 - R))
        
        if alpha < 0.01:
            conductivity_ratio = 1 - 1.5 * alpha
        else:
            conductivity_ratio = (1 - (3/2) * alpha + 3 * alpha**2 - 
                                3 * alpha**3 * np.log(1 + 1/alpha))
        
        return conductivity * max(conductivity_ratio, 0.1)
    
    def _apply_dislocation_scattering(self, conductivity: float,
                                     dislocation_density: float) -> float:
        """Apply dislocation scattering effects."""
        # Dislocation scattering factor (simplified model)
        if dislocation_density > 1e12:
            scattering_factor = 1 / (1 + (dislocation_density / 1e14) * 0.1)
            return conductivity * scattering_factor
        return conductivity
    
    def _apply_secondary_phase_effects(self, conductivity: float,
                                      phase_fraction: float) -> float:
        """Apply secondary phase effects on conductivity."""
        if phase_fraction <= 0:
            return conductivity
            
        # Assume secondary phases are less conductive
        # Use Maxwell-Garnett effective medium theory
        sigma_matrix = conductivity
        sigma_inclusion = conductivity * 0.1  # Secondary phases less conductive
        f = phase_fraction
        
        # Maxwell-Garnett formula for conductivity
        sigma_eff = sigma_matrix * (1 + 2*f*(sigma_inclusion - sigma_matrix)/(sigma_inclusion + 2*sigma_matrix - f*(sigma_inclusion - sigma_matrix)))
        
        return max(sigma_eff, conductivity * 0.1)
    
    def _apply_porosity_effects(self, conductivity: float, porosity: float) -> float:
        """Apply porosity effects using effective medium theory."""
        if porosity <= 0:
            return conductivity
            
        # Pores are insulators (sigma = 0)
        # Use Bruggeman effective medium theory for high porosity
        if porosity < 0.3:
            # Maxwell-Garnett for low porosity
            return conductivity * (1 - 1.5 * porosity)
        else:
            # Bruggeman for high porosity
            return conductivity * (1 - porosity)**(3/2)
    
    def _apply_frequency_effects(self, conductivity: float,
                                microstructure: MicrostructureParams,
                                frequency: float) -> float:
        """Apply frequency-dependent effects."""
        # Skin effect in grains
        grain_skin_depth = np.sqrt(2 / (2 * np.pi * frequency * 4*np.pi*1e-7 * conductivity))
        
        if grain_skin_depth < microstructure.grain_size:
            # High frequency regime - current flows mainly near grain boundaries
            frequency_factor = np.sqrt(grain_skin_depth / microstructure.grain_size)
            return conductivity * frequency_factor
        
        return conductivity
    
    def calculate_effective_permeability(self, bulk_permeability: float,
                                        microstructure: MicrostructureParams,
                                        frequency: float = 1e9) -> complex:
        """
        Calculate effective permeability including microstructural effects.
        
        Args:
            bulk_permeability: Bulk relative permeability
            microstructure: Microstructure parameters
            frequency: Frequency (Hz)
            
        Returns:
            Complex effective permeability
        """
        if microstructure.magnetic_domain_size == 0:
            return complex(bulk_permeability, 0)  # Non-magnetic
        
        mu_real = bulk_permeability
        mu_imag = 0.0
        
        # Domain wall motion losses
        if frequency < 1e6:  # Low frequency - domain wall motion dominates
            domain_wall_loss = self._calculate_domain_wall_loss(
                microstructure, frequency
            )
            mu_imag += domain_wall_loss
            
        # Eddy current losses in domains
        elif frequency > 1e6:  # High frequency - eddy currents dominate
            eddy_loss = self._calculate_eddy_current_loss(
                microstructure, frequency
            )
            mu_imag += eddy_loss
            mu_real *= (1 - eddy_loss / bulk_permeability)
        
        # Grain boundary pinning effects
        if microstructure.grain_size < microstructure.magnetic_domain_size:
            pinning_factor = microstructure.grain_size / microstructure.magnetic_domain_size
            mu_real *= (1 + pinning_factor)
        
        return complex(max(mu_real, 1.0), mu_imag)
    
    def _calculate_domain_wall_loss(self, microstructure: MicrostructureParams,
                                   frequency: float) -> float:
        """Calculate domain wall motion losses."""
        if microstructure.magnetic_domain_size == 0:
            return 0.0
            
        # Simplified domain wall loss model
        domain_mobility = 1e-4  # m²/(A·s) typical value
        loss_factor = frequency * domain_mobility / microstructure.magnetic_domain_size
        
        return min(loss_factor * 100, 10.0)  # Cap losses
    
    def _calculate_eddy_current_loss(self, microstructure: MicrostructureParams,
                                    frequency: float) -> float:
        """Calculate eddy current losses in magnetic domains."""
        if microstructure.magnetic_domain_size == 0:
            return 0.0
            
        # Eddy current loss in spherical domains
        domain_size = microstructure.magnetic_domain_size
        conductivity = 1e6  # Typical ferromagnetic conductivity
        
        # Skin depth in domain
        skin_depth = np.sqrt(2 / (2 * np.pi * frequency * 4*np.pi*1e-7 * conductivity))
        
        if skin_depth < domain_size:
            loss_factor = (domain_size / skin_depth)**2 * frequency / 1e9
            return min(loss_factor, 50.0)
        
        return 0.1 * frequency / 1e9  # Linear low-loss regime

# Create global instance
advanced_microstructure = AdvancedMicrostructure()