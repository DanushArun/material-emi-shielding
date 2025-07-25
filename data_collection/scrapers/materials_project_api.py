"""
Materials Project API integration for fetching material properties.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import asyncio
from mp_api.client import MPRester
from pymatgen.core import Composition
import pandas as pd
import numpy as np


load_dotenv()


class MaterialsProjectScraper:
    """Fetch material properties from Materials Project database."""
    
    # Material properties relevant for EMI shielding
    RELEVANT_PROPERTIES = [
        'material_id',
        'formula_pretty',
        'band_gap',
        'density',
        'volume',
        'formation_energy_per_atom',
        'energy_above_hull',
        'is_stable',
        'is_metal',
        'theoretical_density',
        'magnetic_ordering',
        'total_magnetization',
        'num_magnetic_sites',
        'num_unique_magnetic_sites'
    ]
    
    # Elements commonly used in EMI shielding materials
    EMI_ELEMENTS = [
        'Fe', 'Co', 'Ni', 'Cu', 'Al', 'Ag', 'Au', 'Zn',  # Metals
        'C', 'Si', 'Ge',  # Semiconductors
        'Ti', 'V', 'Cr', 'Mn',  # Transition metals
        'Ba', 'Sr',  # For ferrites
        'O', 'N', 'S'  # Non-metals
    ]
    
    def __init__(self, api_key: str = None):
        """
        Initialize Materials Project API client.
        
        Args:
            api_key: Materials Project API key
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('MP_API_KEY')
        if not self.api_key:
            raise ValueError("Materials Project API key required. Set MP_API_KEY environment variable.")
        
        # Initialize API client
        self.mpr = MPRester(self.api_key)
        
        # Cache for storing fetched data
        self.cache = {}
        
    def search_emi_materials(self, 
                           elements: List[str] = None,
                           max_elements: int = 4,
                           is_metal: bool = None,
                           magnetic_only: bool = False) -> List[Dict]:
        """
        Search for materials suitable for EMI shielding.
        
        Args:
            elements: List of elements to include
            max_elements: Maximum number of elements in composition
            is_metal: Filter for metallic materials
            magnetic_only: Only return magnetic materials
            
        Returns:
            List of material data dictionaries
        """
        if elements is None:
            elements = self.EMI_ELEMENTS
        
        # Build query
        query_params = {
            'elements': elements,
            'num_elements': {'$lte': max_elements},
            'is_stable': True
        }
        
        if is_metal is not None:
            query_params['is_metal'] = is_metal
            
        if magnetic_only:
            query_params['magnetic_ordering'] = {'$ne': None}
        
        try:
            # Query Materials Project
            results = self.mpr.summary.search(
                **query_params,
                fields=self.RELEVANT_PROPERTIES
            )
            
            self.logger.info(f"Found {len(results)} materials matching criteria")
            
            # Convert to dictionaries and enhance with calculated properties
            materials = []
            for result in results:
                material_data = self._process_material_data(result)
                if material_data:
                    materials.append(material_data)
            
            return materials
            
        except Exception as e:
            self.logger.error(f"Error searching materials: {e}")
            return []
    
    def get_material_properties(self, material_id: str) -> Optional[Dict]:
        """
        Get detailed properties for a specific material.
        
        Args:
            material_id: Materials Project ID (e.g., 'mp-1234')
            
        Returns:
            Dictionary of material properties
        """
        # Check cache first
        if material_id in self.cache:
            return self.cache[material_id]
        
        try:
            # Get material data
            material = self.mpr.summary.get_data_by_id(
                material_id,
                fields=self.RELEVANT_PROPERTIES + [
                    'structure',
                    'electronic_structure',
                    'dos',
                    'elasticity',
                    'dielectric'
                ]
            )
            
            if material:
                processed_data = self._process_material_data(material)
                
                # Extract additional properties
                if hasattr(material, 'structure'):
                    processed_data['lattice_parameters'] = {
                        'a': material.structure.lattice.a,
                        'b': material.structure.lattice.b,
                        'c': material.structure.lattice.c,
                        'alpha': material.structure.lattice.alpha,
                        'beta': material.structure.lattice.beta,
                        'gamma': material.structure.lattice.gamma,
                        'volume': material.structure.lattice.volume
                    }
                
                # Cache the result
                self.cache[material_id] = processed_data
                
                return processed_data
                
        except Exception as e:
            self.logger.error(f"Error fetching material {material_id}: {e}")
            
        return None
    
    def _process_material_data(self, material: Any) -> Optional[Dict]:
        """Process raw material data from Materials Project."""
        try:
            # Extract basic properties
            data = {
                'material_id': material.material_id,
                'formula': material.formula_pretty,
                'composition': self._parse_composition(material.formula_pretty),
                'density': material.density,  # g/cm³
                'is_metal': material.is_metal,
                'is_stable': material.is_stable,
                'band_gap': material.band_gap,  # eV
                'formation_energy': material.formation_energy_per_atom,  # eV/atom
                'energy_above_hull': material.energy_above_hull  # eV/atom
            }
            
            # Add magnetic properties if available
            if hasattr(material, 'magnetic_ordering') and material.magnetic_ordering:
                data['magnetic_properties'] = {
                    'ordering': material.magnetic_ordering,
                    'total_magnetization': material.total_magnetization,
                    'num_magnetic_sites': material.num_magnetic_sites
                }
            
            # Estimate electrical conductivity based on band gap
            data['estimated_conductivity'] = self._estimate_conductivity(material.band_gap)
            
            # Estimate relative permeability for magnetic materials
            if hasattr(material, 'total_magnetization') and material.total_magnetization:
                data['estimated_permeability'] = self._estimate_permeability(
                    material.total_magnetization,
                    material.magnetic_ordering
                )
            else:
                data['estimated_permeability'] = 1.0
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing material data: {e}")
            return None
    
    def _parse_composition(self, formula: str) -> Dict[str, float]:
        """Parse composition from formula string."""
        try:
            comp = Composition(formula)
            total = sum(comp.values())
            
            # Convert to percentage
            composition = {}
            for element, amount in comp.items():
                composition[str(element)] = (amount / total) * 100
                
            return composition
            
        except Exception as e:
            self.logger.error(f"Error parsing composition {formula}: {e}")
            return {}
    
    def _estimate_conductivity(self, band_gap: float) -> float:
        """
        Estimate electrical conductivity from band gap.
        
        This is a rough approximation:
        - Metals (band_gap = 0): High conductivity
        - Semiconductors (0 < band_gap < 3): Moderate conductivity
        - Insulators (band_gap > 3): Low conductivity
        """
        if band_gap == 0:
            # Metal - high conductivity (10^6 to 10^8 S/m)
            return 1e7
        elif band_gap < 0.5:
            # Small gap semiconductor
            return 1e4
        elif band_gap < 1.5:
            # Semiconductor
            return 1e2
        elif band_gap < 3.0:
            # Wide gap semiconductor
            return 1e0
        else:
            # Insulator
            return 1e-6
    
    def _estimate_permeability(self, magnetization: float, ordering: str) -> float:
        """
        Estimate relative permeability from magnetic properties.
        
        This is a very rough approximation based on magnetic ordering.
        """
        if not ordering or ordering == 'NM':  # Non-magnetic
            return 1.0
        elif ordering == 'FM':  # Ferromagnetic
            # Rough estimate based on magnetization
            if magnetization > 2:
                return 1000  # High permeability like iron
            elif magnetization > 1:
                return 100
            else:
                return 10
        elif ordering == 'AFM':  # Antiferromagnetic
            return 1.5
        elif ordering == 'FiM':  # Ferrimagnetic
            return 50
        else:
            return 1.0
    
    def fetch_conductive_materials(self, 
                                 min_conductivity: float = 1e6,
                                 limit: int = 100) -> pd.DataFrame:
        """
        Fetch materials with high electrical conductivity.
        
        Args:
            min_conductivity: Minimum conductivity threshold (S/m)
            limit: Maximum number of results
            
        Returns:
            DataFrame of conductive materials
        """
        # Search for metallic materials
        metals = self.search_emi_materials(is_metal=True)
        
        # Filter by estimated conductivity
        conductive_materials = []
        for material in metals[:limit]:
            if material.get('estimated_conductivity', 0) >= min_conductivity:
                conductive_materials.append({
                    'material_id': material['material_id'],
                    'formula': material['formula'],
                    'density': material['density'],
                    'conductivity': material['estimated_conductivity'],
                    'permeability': material.get('estimated_permeability', 1.0),
                    'is_magnetic': 'magnetic_properties' in material
                })
        
        return pd.DataFrame(conductive_materials)
    
    def fetch_magnetic_materials(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch materials with magnetic properties.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            DataFrame of magnetic materials
        """
        # Search for magnetic materials
        magnetic = self.search_emi_materials(magnetic_only=True)
        
        # Extract relevant data
        magnetic_materials = []
        for material in magnetic[:limit]:
            if 'magnetic_properties' in material:
                magnetic_materials.append({
                    'material_id': material['material_id'],
                    'formula': material['formula'],
                    'density': material['density'],
                    'magnetic_ordering': material['magnetic_properties']['ordering'],
                    'magnetization': material['magnetic_properties']['total_magnetization'],
                    'estimated_permeability': material.get('estimated_permeability', 1.0),
                    'estimated_conductivity': material['estimated_conductivity']
                })
        
        return pd.DataFrame(magnetic_materials)
    
    def estimate_emi_performance(self, material_data: Dict, 
                               thickness: float = 1.0,
                               frequency: float = 1e9) -> Dict:
        """
        Estimate EMI shielding performance based on material properties.
        
        Args:
            material_data: Material properties dictionary
            thickness: Material thickness in mm
            frequency: Frequency in Hz
            
        Returns:
            Dictionary with estimated SE components
        """
        # Get material properties
        conductivity = material_data.get('estimated_conductivity', 1e3)
        permeability = material_data.get('estimated_permeability', 1.0)
        
        # Convert thickness to meters
        thickness_m = thickness / 1000
        
        # Calculate skin depth
        mu = permeability * 4 * np.pi * 1e-7  # Absolute permeability
        skin_depth = np.sqrt(2 / (2 * np.pi * frequency * mu * conductivity))
        
        # Estimate shielding effectiveness components
        # These are simplified estimates
        if conductivity > 1e6:  # Good conductor
            reflection_loss = 20 * np.log10(1 + 0.5 * np.sqrt(conductivity * mu * frequency))
            absorption_loss = 8.68 * thickness_m / skin_depth
        else:
            reflection_loss = 10 * np.log10(1 + conductivity / 1000)
            absorption_loss = 20 * thickness_m * np.sqrt(frequency / 1e9)
        
        total_se = reflection_loss + absorption_loss
        
        return {
            'estimated_se': total_se,
            'reflection_loss': reflection_loss,
            'absorption_loss': absorption_loss,
            'skin_depth': skin_depth * 1000,  # Convert to mm
            'thickness_to_skin_depth_ratio': thickness_m / skin_depth
        }