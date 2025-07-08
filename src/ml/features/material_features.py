"""
Feature extraction for material compositions.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd


class MaterialFeatureExtractor:
    """Extract ML-ready features from material compositions."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.scaler = StandardScaler()
        self.element_properties = self._load_element_properties()
    
    def _load_element_properties(self) -> Dict[str, Dict[str, float]]:
        """Load atomic properties for feature calculation."""
        # Comprehensive element properties for feature engineering
        properties = {
            'H': {'Z': 1, 'r': 53, 'EN': 2.20, 'IE': 13.6, 'VE': 1, 'mass': 1.008},
            'C': {'Z': 6, 'r': 67, 'EN': 2.55, 'IE': 11.3, 'VE': 4, 'mass': 12.011},
            'N': {'Z': 7, 'r': 56, 'EN': 3.04, 'IE': 14.5, 'VE': 5, 'mass': 14.007},
            'O': {'Z': 8, 'r': 48, 'EN': 3.44, 'IE': 13.6, 'VE': 6, 'mass': 15.999},
            'F': {'Z': 9, 'r': 42, 'EN': 3.98, 'IE': 17.4, 'VE': 7, 'mass': 18.998},
            'Na': {'Z': 11, 'r': 190, 'EN': 0.93, 'IE': 5.1, 'VE': 1, 'mass': 22.990},
            'Mg': {'Z': 12, 'r': 145, 'EN': 1.31, 'IE': 7.6, 'VE': 2, 'mass': 24.305},
            'Al': {'Z': 13, 'r': 118, 'EN': 1.61, 'IE': 6.0, 'VE': 3, 'mass': 26.982},
            'Si': {'Z': 14, 'r': 111, 'EN': 1.90, 'IE': 8.2, 'VE': 4, 'mass': 28.085},
            'P': {'Z': 15, 'r': 98, 'EN': 2.19, 'IE': 10.5, 'VE': 5, 'mass': 30.974},
            'S': {'Z': 16, 'r': 88, 'EN': 2.58, 'IE': 10.4, 'VE': 6, 'mass': 32.06},
            'Cl': {'Z': 17, 'r': 79, 'EN': 3.16, 'IE': 13.0, 'VE': 7, 'mass': 35.45},
            'K': {'Z': 19, 'r': 243, 'EN': 0.82, 'IE': 4.3, 'VE': 1, 'mass': 39.098},
            'Ca': {'Z': 20, 'r': 194, 'EN': 1.00, 'IE': 6.1, 'VE': 2, 'mass': 40.078},
            'Fe': {'Z': 26, 'r': 156, 'EN': 1.83, 'IE': 7.9, 'VE': 8, 'mass': 55.845},
            'Co': {'Z': 27, 'r': 152, 'EN': 1.88, 'IE': 7.9, 'VE': 9, 'mass': 58.933},
            'Ni': {'Z': 28, 'r': 149, 'EN': 1.91, 'IE': 7.6, 'VE': 10, 'mass': 58.693},
            'Cu': {'Z': 29, 'r': 145, 'EN': 1.90, 'IE': 7.7, 'VE': 11, 'mass': 63.546},
            'Zn': {'Z': 30, 'r': 142, 'EN': 1.65, 'IE': 9.4, 'VE': 12, 'mass': 65.38},
            'Ag': {'Z': 47, 'r': 165, 'EN': 1.93, 'IE': 7.6, 'VE': 11, 'mass': 107.868},
            'Au': {'Z': 79, 'r': 174, 'EN': 2.54, 'IE': 9.2, 'VE': 11, 'mass': 196.967},
            'Pb': {'Z': 82, 'r': 154, 'EN': 2.33, 'IE': 7.4, 'VE': 14, 'mass': 207.2},
        }
        # Z: atomic number, r: atomic radius (pm), EN: electronegativity
        # IE: ionization energy (eV), VE: valence electrons, mass: atomic mass
        return properties
    
    def extract_compositional_features(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Extract features from elemental composition."""
        features = {}
        
        # Basic statistics
        features['n_elements'] = len(composition)
        
        # Weighted averages
        total_weight = sum(composition.values())
        if total_weight == 0:
            return features
        
        # Initialize weighted sums
        weighted_z = 0
        weighted_r = 0
        weighted_en = 0
        weighted_ie = 0
        weighted_ve = 0
        weighted_mass = 0
        
        # Lists for variance calculations
        z_list = []
        r_list = []
        en_list = []
        ie_list = []
        
        for element, percentage in composition.items():
            weight = percentage / total_weight
            
            if element in self.element_properties:
                props = self.element_properties[element]
                
                # Weighted averages
                weighted_z += props['Z'] * weight
                weighted_r += props['r'] * weight
                weighted_en += props['EN'] * weight
                weighted_ie += props['IE'] * weight
                weighted_ve += props['VE'] * weight
                weighted_mass += props['mass'] * weight
                
                # Store for variance
                z_list.append(props['Z'])
                r_list.append(props['r'])
                en_list.append(props['EN'])
                ie_list.append(props['IE'])
        
        # Average features
        features['mean_atomic_number'] = weighted_z
        features['mean_atomic_radius'] = weighted_r
        features['mean_electronegativity'] = weighted_en
        features['mean_ionization_energy'] = weighted_ie
        features['mean_valence_electrons'] = weighted_ve
        features['mean_atomic_mass'] = weighted_mass
        
        # Variance features (compositional complexity)
        if len(z_list) > 1:
            features['var_atomic_number'] = np.var(z_list)
            features['var_atomic_radius'] = np.var(r_list)
            features['var_electronegativity'] = np.var(en_list)
            features['var_ionization_energy'] = np.var(ie_list)
            
            # Range features
            features['range_electronegativity'] = max(en_list) - min(en_list)
            features['range_atomic_radius'] = max(r_list) - min(r_list)
        else:
            features['var_atomic_number'] = 0
            features['var_atomic_radius'] = 0
            features['var_electronegativity'] = 0
            features['var_ionization_energy'] = 0
            features['range_electronegativity'] = 0
            features['range_atomic_radius'] = 0
        
        # Mixing entropy (configurational entropy)
        mixing_entropy = 0
        for percentage in composition.values():
            if percentage > 0:
                x = percentage / 100.0
                mixing_entropy -= x * np.log(x)
        features['mixing_entropy'] = mixing_entropy
        
        # Valence electron concentration (VEC)
        features['valence_electron_concentration'] = weighted_ve
        
        # Atomic size difference parameter (δ)
        if len(r_list) > 1:
            r_avg = np.mean(r_list)
            delta = 0
            for element, percentage in composition.items():
                if element in self.element_properties:
                    r_i = self.element_properties[element]['r']
                    x_i = percentage / 100.0
                    delta += x_i * ((1 - r_i/r_avg) ** 2)
            features['atomic_size_parameter'] = 100 * np.sqrt(delta)
        else:
            features['atomic_size_parameter'] = 0
        
        # Element type features (counts)
        features['n_transition_metals'] = sum(1 for elem in composition 
                                             if elem in ['Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ag', 'Au'])
        features['n_alkali_metals'] = sum(1 for elem in composition 
                                         if elem in ['Li', 'Na', 'K', 'Rb', 'Cs'])
        features['n_nonmetals'] = sum(1 for elem in composition 
                                     if elem in ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'])
        
        # Specific element indicators (important for EMI)
        features['has_Fe'] = 1 if 'Fe' in composition else 0
        features['has_Cu'] = 1 if 'Cu' in composition else 0
        features['has_Al'] = 1 if 'Al' in composition else 0
        features['has_Ni'] = 1 if 'Ni' in composition else 0
        features['has_C'] = 1 if 'C' in composition else 0
        
        # Percentages of key elements
        features['Fe_percentage'] = composition.get('Fe', 0)
        features['Cu_percentage'] = composition.get('Cu', 0)
        features['Al_percentage'] = composition.get('Al', 0)
        features['Ni_percentage'] = composition.get('Ni', 0)
        features['C_percentage'] = composition.get('C', 0)
        
        return features
    
    def extract_processing_features(self, conditions: Dict) -> Dict[str, float]:
        """Extract features from processing conditions."""
        features = {}
        
        # Temperature features
        temp = conditions.get('temperature_C', 25)
        features['processing_temp'] = temp
        features['processing_temp_squared'] = temp ** 2
        features['processing_temp_log'] = np.log(temp + 1)
        
        # Categorize temperature ranges
        features['is_room_temp'] = 1 if 15 <= temp <= 30 else 0
        features['is_low_temp'] = 1 if temp < 100 else 0
        features['is_medium_temp'] = 1 if 100 <= temp < 500 else 0
        features['is_high_temp'] = 1 if temp >= 500 else 0
        
        # Pressure features
        pressure = conditions.get('pressure', 'ambient')
        features['is_ambient_pressure'] = 1 if pressure == 'ambient' else 0
        features['is_vacuum'] = 1 if pressure == 'vacuum' else 0
        features['is_high_pressure'] = 1 if pressure == 'high' else 0
        
        # Time features
        time_hours = conditions.get('time_hours', 1)
        features['processing_time'] = time_hours
        features['processing_time_log'] = np.log(time_hours + 1)
        
        # Atmosphere features
        atmosphere = conditions.get('atmosphere', 'air')
        features['is_air_atmosphere'] = 1 if atmosphere == 'air' else 0
        features['is_inert_atmosphere'] = 1 if atmosphere in ['nitrogen', 'argon'] else 0
        features['is_vacuum_atmosphere'] = 1 if atmosphere == 'vacuum' else 0
        
        return features
    
    def extract_morphology_features(self, morphology_data: Optional[Dict]) -> Dict[str, float]:
        """Extract features from morphology data (placeholder for CNN features)."""
        features = {}
        
        if morphology_data is None:
            # Default morphology features
            features['particle_size_mean'] = 10.0  # μm
            features['particle_size_std'] = 2.0
            features['porosity'] = 0.1
            features['connectivity'] = 0.5
            features['tortuosity'] = 1.5
            features['surface_area'] = 100.0  # m²/g
        else:
            features.update(morphology_data)
        
        return features
    
    def extract_all_features(self, 
                           composition: Dict[str, float],
                           frequency: float,
                           thickness: float,
                           conditions: Optional[Dict] = None,
                           morphology: Optional[Dict] = None) -> np.ndarray:
        """Extract all features and return as numpy array."""
        all_features = {}
        
        # Compositional features
        comp_features = self.extract_compositional_features(composition)
        all_features.update(comp_features)
        
        # Frequency features
        all_features['frequency_mhz'] = frequency
        all_features['frequency_log'] = np.log10(frequency + 1)
        all_features['is_low_freq'] = 1 if frequency < 100 else 0
        all_features['is_mid_freq'] = 1 if 100 <= frequency < 1000 else 0
        all_features['is_high_freq'] = 1 if frequency >= 1000 else 0
        
        # Thickness features
        all_features['thickness_mm'] = thickness
        all_features['thickness_log'] = np.log10(thickness + 0.1)
        all_features['thickness_squared'] = thickness ** 2
        all_features['is_thin'] = 1 if thickness < 1 else 0
        all_features['is_medium'] = 1 if 1 <= thickness < 5 else 0
        all_features['is_thick'] = 1 if thickness >= 5 else 0
        
        # Processing features
        if conditions:
            proc_features = self.extract_processing_features(conditions)
            all_features.update(proc_features)
        
        # Morphology features
        if morphology:
            morph_features = self.extract_morphology_features(morphology)
            all_features.update(morph_features)
        
        # Convert to numpy array in consistent order
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        return feature_vector, feature_names
    
    def create_feature_dataframe(self, samples: List[Dict]) -> pd.DataFrame:
        """Create a feature dataframe from multiple samples."""
        all_data = []
        feature_names = None
        
        for sample in samples:
            features, names = self.extract_all_features(
                composition=sample['composition'],
                frequency=sample['frequency'],
                thickness=sample['thickness'],
                conditions=sample.get('conditions'),
                morphology=sample.get('morphology')
            )
            all_data.append(features)
            
            if feature_names is None:
                feature_names = names
        
        return pd.DataFrame(all_data, columns=feature_names)