"""
Material properties database management for EMI shielding calculations.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

class MaterialDatabase:
    """Manages the periodic table and material properties database."""
    
    def __init__(self):
        """Initialize the material database."""
        self.periodic_table = {}
        self.alloys = {}
        self.custom_materials = {}
        self._load_periodic_table()
        self._load_alloys()
        
    def _load_periodic_table(self):
        """Load the periodic table data from JSON file."""
        try:
            db_path = os.path.join(os.path.dirname(__file__), 'periodic_table.json')
            with open(db_path, 'r') as f:
                self.periodic_table = json.load(f)
        except FileNotFoundError:
            # Fallback: try relative to the project root
            try:
                project_root = os.path.join(os.path.dirname(__file__), '..', '..')
                db_path = os.path.join(project_root, 'src', 'materials', 'periodic_table.json')
                with open(db_path, 'r') as f:
                    self.periodic_table = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Could not find periodic_table.json. Using empty database.")
                self.periodic_table = {}
    
    def _load_alloys(self):
        """Load common alloys and their properties."""
        # Common alloys with their compositions and properties
        self.alloys = {
            "steel_1018": {
                "name": "Steel 1018 (Mild Steel)",
                "composition": {"Fe": 0.982, "C": 0.018},
                "density": 7850,
                "electrical_conductivity": 6.99e6,
                "relative_permeability": 2000,
                "relative_permittivity": 1,
                "note": "Common mild steel"
            },
            "stainless_steel_304": {
                "name": "Stainless Steel 304",
                "composition": {"Fe": 0.70, "Cr": 0.19, "Ni": 0.09, "Mn": 0.02},
                "density": 8000,
                "electrical_conductivity": 1.45e6,
                "relative_permeability": 1.02,
                "relative_permittivity": 1,
                "note": "Non-magnetic stainless steel"
            },
            "aluminum_6061": {
                "name": "Aluminum 6061",
                "composition": {"Al": 0.972, "Mg": 0.01, "Si": 0.006, "Cu": 0.003, "Cr": 0.002},
                "density": 2700,
                "electrical_conductivity": 2.5e7,
                "relative_permeability": 1.000022,
                "relative_permittivity": 1,
                "note": "Common structural aluminum alloy"
            },
            "brass_70_30": {
                "name": "Brass 70/30",
                "composition": {"Cu": 0.70, "Zn": 0.30},
                "density": 8530,
                "electrical_conductivity": 1.6e7,
                "relative_permeability": 0.999994,
                "relative_permittivity": 1,
                "note": "Cartridge brass"
            },
            "bronze": {
                "name": "Bronze (Phosphor)",
                "composition": {"Cu": 0.95, "Sn": 0.05},
                "density": 8900,
                "electrical_conductivity": 8.7e6,
                "relative_permeability": 0.999994,
                "relative_permittivity": 1,
                "note": "Common bronze alloy"
            },
            "mu_metal": {
                "name": "Mu-Metal",
                "composition": {"Ni": 0.77, "Fe": 0.16, "Cu": 0.05, "Mo": 0.02},
                "density": 8700,
                "electrical_conductivity": 1.82e6,
                "relative_permeability": 100000,
                "relative_permittivity": 1,
                "note": "High permeability magnetic shielding"
            },
            "permalloy": {
                "name": "Permalloy 80",
                "composition": {"Ni": 0.80, "Fe": 0.20},
                "density": 8600,
                "electrical_conductivity": 2e6,
                "relative_permeability": 50000,
                "relative_permittivity": 1,
                "note": "High permeability alloy"
            },
            "kovar": {
                "name": "Kovar",
                "composition": {"Fe": 0.54, "Ni": 0.29, "Co": 0.17},
                "density": 8360,
                "electrical_conductivity": 2.04e6,
                "relative_permeability": 450,
                "relative_permittivity": 1,
                "note": "Controlled expansion alloy"
            },
            "inconel_600": {
                "name": "Inconel 600",
                "composition": {"Ni": 0.76, "Cr": 0.155, "Fe": 0.08},
                "density": 8470,
                "electrical_conductivity": 1.03e6,
                "relative_permeability": 1.01,
                "relative_permittivity": 1,
                "note": "High temperature alloy"
            },
            "carbon_fiber": {
                "name": "Carbon Fiber Composite",
                "composition": {"C": 1.0},
                "density": 1750,
                "electrical_conductivity": 1e4,
                "relative_permeability": 0.999995,
                "relative_permittivity": 10,
                "note": "Anisotropic properties"
            }
        }
    
    def get_element(self, symbol: str) -> Optional[Dict]:
        """Get properties of an element by its symbol."""
        return self.periodic_table.get(symbol)
    
    def get_alloy(self, name: str) -> Optional[Dict]:
        """Get properties of an alloy by its name."""
        return self.alloys.get(name)
    
    def get_material(self, name: str) -> Optional[Dict]:
        """Get material properties by name (element, alloy, or custom)."""
        # Try element first
        if name in self.periodic_table:
            return self.periodic_table[name]
        # Try alloy
        elif name in self.alloys:
            return self.alloys[name]
        # Try custom material
        elif name in self.custom_materials:
            return self.custom_materials[name]
        return None
    
    def add_custom_material(self, name: str, properties: Dict):
        """Add a custom material to the database."""
        required_props = ['density', 'electrical_conductivity', 
                         'relative_permeability', 'relative_permittivity']
        
        for prop in required_props:
            if prop not in properties:
                raise ValueError(f"Missing required property: {prop}")
        
        self.custom_materials[name] = properties
    
    def calculate_alloy_properties(self, composition: Dict[str, float]) -> Dict:
        """Calculate approximate properties of an alloy from its composition."""
        # Normalize composition
        total = sum(composition.values())
        comp_norm = {k: v/total for k, v in composition.items()}
        
        # Initialize properties
        density = 0
        conductivity = 0
        permeability = 0
        permittivity = 0
        
        # Rule of mixtures for density
        for element, fraction in comp_norm.items():
            elem_props = self.get_element(element)
            if elem_props:
                density += fraction * elem_props['density']
        
        # Weighted harmonic mean for conductivity
        for element, fraction in comp_norm.items():
            elem_props = self.get_element(element)
            if elem_props and elem_props['electrical_conductivity'] > 0:
                conductivity += fraction / elem_props['electrical_conductivity']
        
        if conductivity > 0:
            conductivity = 1 / conductivity
        
        # Weighted average for permeability and permittivity
        for element, fraction in comp_norm.items():
            elem_props = self.get_element(element)
            if elem_props:
                permeability += fraction * elem_props['relative_permeability']
                permittivity += fraction * elem_props['relative_permittivity']
        
        return {
            'density': density,
            'electrical_conductivity': conductivity,
            'relative_permeability': permeability,
            'relative_permittivity': permittivity,
            'composition': comp_norm,
            'note': 'Calculated using rule of mixtures'
        }
    
    def search_by_property(self, property_name: str, min_value: float = None, 
                          max_value: float = None) -> List[Tuple[str, Dict]]:
        """Search materials by property range."""
        results = []
        
        # Search elements
        for symbol, props in self.periodic_table.items():
            if property_name in props:
                value = props[property_name]
                if isinstance(value, (int, float)):
                    if (min_value is None or value >= min_value) and \
                       (max_value is None or value <= max_value):
                        results.append((symbol, props))
        
        # Search alloys
        for name, props in self.alloys.items():
            if property_name in props:
                value = props[property_name]
                if isinstance(value, (int, float)):
                    if (min_value is None or value >= min_value) and \
                       (max_value is None or value <= max_value):
                        results.append((name, props))
        
        # Sort by property value
        results.sort(key=lambda x: x[1].get(property_name, 0), reverse=True)
        
        return results
    
    def get_best_conductors(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the top N best electrical conductors."""
        conductors = self.search_by_property('electrical_conductivity', min_value=1e6)
        return [(name, props['electrical_conductivity']) for name, props in conductors[:top_n]]
    
    def get_magnetic_materials(self, min_permeability: float = 100) -> List[Tuple[str, float]]:
        """Get materials with high magnetic permeability."""
        magnetic = self.search_by_property('relative_permeability', min_value=min_permeability)
        return [(name, props['relative_permeability']) for name, props in magnetic]
    
    def get_material_for_shielding(self, frequency: float, target_se: float = 60) -> List[Dict]:
        """Recommend materials for specific shielding requirements."""
        recommendations = []
        
        # Calculate skin depth for reference
        omega = 2 * np.pi * frequency
        
        # Check common shielding materials
        candidates = ['Cu', 'Al', 'Fe', 'steel_1018', 'mu_metal', 'brass_70_30']
        
        for material in candidates:
            props = self.get_material(material)
            if props:
                # Simple estimation based on conductivity and permeability
                sigma = props.get('electrical_conductivity', 0)
                mu_r = props.get('relative_permeability', 1)
                
                if sigma > 0:
                    # Skin depth
                    mu = mu_r * 4 * np.pi * 1e-7
                    delta = np.sqrt(2 / (omega * mu * sigma))
                    
                    # Rough SE estimate for 1mm thickness
                    thickness = 0.001  # 1mm
                    se_estimate = 8.686 * thickness / delta + 20 * np.log10(377 / (2 * np.sqrt(mu_r)))
                    
                    recommendations.append({
                        'material': material,
                        'name': props.get('name', material),
                        'conductivity': sigma,
                        'permeability': mu_r,
                        'skin_depth_mm': delta * 1000,
                        'se_estimate_1mm': se_estimate,
                        'thickness_needed_mm': (target_se / se_estimate) if se_estimate > 0 else float('inf')
                    })
        
        # Sort by thickness needed
        recommendations.sort(key=lambda x: x['thickness_needed_mm'])
        
        return recommendations

# Create global instance
material_db = MaterialDatabase()