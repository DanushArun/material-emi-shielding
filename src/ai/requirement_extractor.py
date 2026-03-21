"""
Natural Language Requirement Extraction for EMI Shielding Applications

This module extracts EMI shielding requirements from natural language descriptions
of use cases and applications.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class ApplicationType(Enum):
    """Common EMI shielding application types"""
    CONSUMER_ELECTRONICS = "consumer_electronics"
    MEDICAL_DEVICE = "medical_device"
    AUTOMOTIVE = "automotive"
    AEROSPACE = "aerospace"
    INDUSTRIAL = "industrial"
    MILITARY = "military"
    TELECOM = "telecom"
    DATA_CENTER = "data_center"
    RF_ENCLOSURE = "rf_enclosure"
    PCB_SHIELDING = "pcb_shielding"
    CABLE_SHIELDING = "cable_shielding"
    ROOM_SHIELDING = "room_shielding"
    GENERAL = "general"


class Environment(Enum):
    """Operating environment conditions"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    HARSH = "harsh"
    CLEANROOM = "cleanroom"
    HIGH_TEMP = "high_temperature"
    CRYOGENIC = "cryogenic"
    CORROSIVE = "corrosive"
    VACUUM = "vacuum"
    UNDERWATER = "underwater"
    SPACE = "space"


@dataclass
class EMIRequirements:
    """Extracted EMI shielding requirements from user input"""

    # Frequency requirements
    frequency_min: float = 1e6  # Hz (default 1 MHz)
    frequency_max: float = 10e9  # Hz (default 10 GHz)
    primary_frequencies: List[float] = field(default_factory=list)

    # Shielding effectiveness requirements
    target_shielding_db: float = 60  # dB (default)
    min_shielding_db: float = 40  # dB

    # Physical constraints
    max_thickness: Optional[float] = None  # meters
    min_thickness: Optional[float] = None  # meters
    max_weight: Optional[float] = None  # kg/m²

    # Environmental requirements
    operating_temp_min: float = -40  # °C
    operating_temp_max: float = 85  # °C
    environment: Environment = Environment.INDOOR
    requires_corrosion_resistance: bool = False
    requires_flexibility: bool = False

    # Application details
    application_type: ApplicationType = ApplicationType.GENERAL
    near_field: bool = False
    electric_field_dominant: bool = False
    magnetic_field_dominant: bool = False

    # Cost and manufacturing
    cost_sensitive: bool = False
    mass_production: bool = False
    requires_transparency: bool = False
    requires_breathability: bool = False

    # Mechanical properties
    requires_structural_strength: bool = False
    requires_thermal_conductivity: bool = False
    requires_electrical_insulation: bool = False

    # Confidence scores
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class RequirementExtractor:
    """Extract EMI shielding requirements from natural language text"""

    def __init__(self):
        self._init_patterns()
        self._init_keyword_maps()

    def _init_patterns(self):
        """Initialize regex patterns for requirement extraction"""
        self.patterns = {
            # Frequency patterns
            'frequency_value': re.compile(
                r'(\d+(?:\.\d+)?)\s*(hz|khz|mhz|ghz)',
                re.IGNORECASE
            ),
            'frequency_range': re.compile(
                r'(\d+(?:\.\d+)?)\s*(hz|khz|mhz|ghz)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(hz|khz|mhz|ghz)',
                re.IGNORECASE
            ),
            'frequency_band': re.compile(
                r'(wifi|wi-fi|bluetooth|5g|4g|lte|gsm|gps|fm|am|vhf|uhf|microwave|x-band|s-band|c-band|l-band|k-band|ka-band|ku-band)',
                re.IGNORECASE
            ),

            # Shielding effectiveness patterns
            'shielding_db': re.compile(
                r'(\d+(?:\.\d+)?)\s*db',
                re.IGNORECASE
            ),
            'shielding_qualitative': re.compile(
                r'(high|medium|low|moderate|excellent|good|basic|minimal)\s*(?:shielding|attenuation|protection)',
                re.IGNORECASE
            ),

            # Thickness patterns
            'thickness_value': re.compile(
                r'(\d+(?:\.\d+)?)\s*(mm|cm|m|mil|inch|inches|µm|um|micrometer|micron)',
                re.IGNORECASE
            ),
            'thickness_constraint': re.compile(
                r'(thin|thick|ultra-thin|ultrathin|lightweight|heavy)',
                re.IGNORECASE
            ),

            # Temperature patterns
            'temperature': re.compile(
                r'(-?\d+(?:\.\d+)?)\s*(?:°|deg|degrees?)?\s*(c|celsius|f|fahrenheit|k|kelvin)',
                re.IGNORECASE
            ),

            # Weight patterns
            'weight': re.compile(
                r'(\d+(?:\.\d+)?)\s*(g|kg|lb|lbs|oz|gram|kilogram|pound)',
                re.IGNORECASE
            )
        }

    def _init_keyword_maps(self):
        """Initialize keyword mappings for different requirements"""
        self.keyword_maps = {
            'application_type': {
                ApplicationType.CONSUMER_ELECTRONICS: [
                    'phone', 'smartphone', 'laptop', 'computer', 'tablet', 'tv',
                    'television', 'monitor', 'display', 'consumer', 'device'
                ],
                ApplicationType.MEDICAL_DEVICE: [
                    'medical', 'mri', 'ct', 'x-ray', 'hospital', 'healthcare',
                    'pacemaker', 'implant', 'diagnostic', 'surgical'
                ],
                ApplicationType.AUTOMOTIVE: [
                    'car', 'automotive', 'vehicle', 'ev', 'electric vehicle',
                    'autonomous', 'adas', 'radar', 'lidar', 'infotainment'
                ],
                ApplicationType.AEROSPACE: [
                    'aircraft', 'airplane', 'aerospace', 'satellite', 'spacecraft',
                    'aviation', 'avionics', 'rocket', 'drone', 'uav'
                ],
                ApplicationType.MILITARY: [
                    'military', 'defense', 'radar', 'stealth', 'emp',
                    'electromagnetic pulse', 'warfare', 'tactical'
                ],
                ApplicationType.TELECOM: [
                    'telecom', 'telecommunication', 'base station', 'antenna',
                    '5g', '4g', 'cellular', 'tower', 'transmitter'
                ],
                ApplicationType.DATA_CENTER: [
                    'data center', 'datacenter', 'server', 'rack', 'networking',
                    'switch', 'router', 'cloud', 'colocation'
                ],
                ApplicationType.PCB_SHIELDING: [
                    'pcb', 'circuit board', 'board level', 'component',
                    'ic', 'chip', 'processor', 'cpu', 'gpu'
                ]
            },

            'environment': {
                Environment.OUTDOOR: ['outdoor', 'outside', 'external', 'weather'],
                Environment.HARSH: ['harsh', 'extreme', 'rugged', 'demanding'],
                Environment.HIGH_TEMP: ['high temperature', 'hot', 'heat', 'thermal'],
                Environment.CORROSIVE: ['corrosive', 'salt', 'marine', 'chemical'],
                Environment.SPACE: ['space', 'orbit', 'satellite', 'vacuum'],
                Environment.UNDERWATER: ['underwater', 'submarine', 'subsea', 'marine']
            },

            'frequency_bands': {
                'wifi': [(2.4e9, 2.5e9), (5e9, 6e9)],  # 2.4 GHz and 5 GHz bands
                'bluetooth': [(2.4e9, 2.485e9)],
                '5g': [(600e6, 6e9), (24e9, 100e9)],  # FR1 and FR2
                '4g': [(700e6, 2.6e9)],
                'gps': [(1.227e9, 1.575e9)],
                'fm': [(88e6, 108e6)],
                'am': [(530e3, 1700e3)],
                'gsm': [(850e6, 1900e6)],
                'microwave': [(300e6, 300e9)],
                'x-band': [(8e9, 12e9)],
                's-band': [(2e9, 4e9)],
                'c-band': [(4e9, 8e9)],
                'l-band': [(1e9, 2e9)],
                'k-band': [(18e9, 27e9)],
                'ka-band': [(27e9, 40e9)],
                'ku-band': [(12e9, 18e9)]
            },

            'shielding_levels': {
                'minimal': 20,
                'basic': 30,
                'low': 30,
                'moderate': 50,
                'medium': 50,
                'good': 60,
                'high': 80,
                'excellent': 100
            }
        }

    def extract_requirements(self, user_input: str) -> EMIRequirements:
        """
        Extract EMI shielding requirements from natural language input

        Args:
            user_input: Natural language description of the use case

        Returns:
            EMIRequirements object with extracted parameters
        """
        requirements = EMIRequirements()
        user_input_lower = user_input.lower()

        # Extract frequency requirements
        self._extract_frequencies(user_input, requirements)

        # Extract shielding effectiveness
        self._extract_shielding_effectiveness(user_input, requirements)

        # Extract physical constraints
        self._extract_physical_constraints(user_input, requirements)

        # Extract application type
        self._extract_application_type(user_input_lower, requirements)

        # Extract environment
        self._extract_environment(user_input_lower, requirements)

        # Extract special requirements
        self._extract_special_requirements(user_input_lower, requirements)

        # Apply application-specific defaults
        self._apply_application_defaults(requirements)

        # Calculate confidence scores
        self._calculate_confidence(requirements, user_input)

        return requirements

    def _extract_frequencies(self, text: str, req: EMIRequirements):
        """Extract frequency requirements from text"""
        # Check for frequency ranges
        range_matches = self.patterns['frequency_range'].findall(text)
        if range_matches:
            for match in range_matches:
                freq_min = self._convert_to_hz(float(match[0]), match[1])
                freq_max = self._convert_to_hz(float(match[2]), match[3])
                req.frequency_min = min(req.frequency_min, freq_min)
                req.frequency_max = max(req.frequency_max, freq_max)

        # Check for single frequency values
        freq_matches = self.patterns['frequency_value'].findall(text)
        for match in freq_matches:
            freq = self._convert_to_hz(float(match[0]), match[1])
            req.primary_frequencies.append(freq)

        # Check for frequency bands
        band_matches = self.patterns['frequency_band'].findall(text)
        for band in band_matches:
            band_lower = band.lower().replace('-', '')
            if band_lower in self.keyword_maps['frequency_bands']:
                for freq_range in self.keyword_maps['frequency_bands'][band_lower]:
                    req.frequency_min = min(req.frequency_min, freq_range[0])
                    req.frequency_max = max(req.frequency_max, freq_range[1])
                    # Add center frequency as primary
                    center_freq = (freq_range[0] + freq_range[1]) / 2
                    req.primary_frequencies.append(center_freq)

    def _extract_shielding_effectiveness(self, text: str, req: EMIRequirements):
        """Extract shielding effectiveness requirements"""
        # Check for specific dB values
        db_matches = self.patterns['shielding_db'].findall(text)
        if db_matches:
            db_values = [float(match) for match in db_matches]
            req.target_shielding_db = max(db_values)  # Use highest requirement
            req.min_shielding_db = min(db_values) * 0.8  # 80% of lowest as minimum

        # Check for qualitative descriptions
        qual_matches = self.patterns['shielding_qualitative'].findall(text)
        for match in qual_matches:
            level = match.lower().split()[0]
            if level in self.keyword_maps['shielding_levels']:
                suggested_db = self.keyword_maps['shielding_levels'][level]
                req.target_shielding_db = max(req.target_shielding_db, suggested_db)

    def _extract_physical_constraints(self, text: str, req: EMIRequirements):
        """Extract physical constraints like thickness and weight"""
        # Extract thickness
        thickness_matches = self.patterns['thickness_value'].findall(text)
        for match in thickness_matches:
            thickness_m = self._convert_to_meters(float(match[0]), match[1])
            # Context-based assignment (looking for max/min keywords nearby)
            if any(word in text.lower() for word in ['max', 'maximum', 'less than', 'under', 'below']):
                req.max_thickness = thickness_m
            elif any(word in text.lower() for word in ['min', 'minimum', 'at least', 'greater than']):
                req.min_thickness = thickness_m
            else:
                req.max_thickness = thickness_m  # Default to maximum constraint

        # Extract thickness constraints
        thickness_qual = self.patterns['thickness_constraint'].findall(text)
        for constraint in thickness_qual:
            if 'thin' in constraint.lower():
                req.max_thickness = 0.001  # 1mm for thin
            elif 'ultra' in constraint.lower():
                req.max_thickness = 0.0001  # 0.1mm for ultra-thin
            elif 'thick' in constraint.lower() and 'ultra' not in constraint.lower():
                req.min_thickness = 0.003  # 3mm for thick

        # Extract weight constraints
        weight_matches = self.patterns['weight'].findall(text)
        for match in weight_matches:
            weight_kg = self._convert_to_kg(float(match[0]), match[1])
            req.max_weight = weight_kg

    def _extract_application_type(self, text: str, req: EMIRequirements):
        """Identify application type from text"""
        best_match = ApplicationType.GENERAL
        best_score = 0

        for app_type, keywords in self.keyword_maps['application_type'].items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_match = app_type

        req.application_type = best_match

    def _extract_environment(self, text: str, req: EMIRequirements):
        """Extract environmental requirements"""
        for env_type, keywords in self.keyword_maps['environment'].items():
            if any(keyword in text for keyword in keywords):
                req.environment = env_type
                break

        # Temperature extraction
        temp_matches = self.patterns['temperature'].findall(text)
        for match in temp_matches:
            temp_c = self._convert_to_celsius(float(match[0]), match[1])
            # Context-based assignment
            if any(word in text.lower() for word in ['operating', 'range', 'between']):
                if temp_c < 0:
                    req.operating_temp_min = temp_c
                else:
                    req.operating_temp_max = temp_c

    def _extract_special_requirements(self, text: str, req: EMIRequirements):
        """Extract special requirements from text"""
        # Field type dominance
        if any(word in text for word in ['electric field', 'e-field', 'capacitive']):
            req.electric_field_dominant = True
        if any(word in text for word in ['magnetic field', 'h-field', 'inductive']):
            req.magnetic_field_dominant = True

        # Near field
        if any(word in text for word in ['near field', 'near-field', 'close proximity']):
            req.near_field = True

        # Material properties
        if any(word in text for word in ['corrosion', 'rust', 'oxidation', 'weathering']):
            req.requires_corrosion_resistance = True
        if any(word in text for word in ['flexible', 'bendable', 'conform', 'curved']):
            req.requires_flexibility = True
        if any(word in text for word in ['transparent', 'see through', 'optical', 'visibility']):
            req.requires_transparency = True
        if any(word in text for word in ['breathable', 'ventilation', 'airflow', 'perforated']):
            req.requires_breathability = True

        # Cost and production
        if any(word in text for word in ['cost', 'budget', 'economical', 'affordable', 'cheap']):
            req.cost_sensitive = True
        if any(word in text for word in ['mass production', 'volume', 'scale', 'manufacturing']):
            req.mass_production = True

        # Mechanical properties
        if any(word in text for word in ['structural', 'strength', 'rigid', 'support', 'mechanical']):
            req.requires_structural_strength = True
        if any(word in text for word in ['thermal', 'heat dissipation', 'cooling', 'temperature management']):
            req.requires_thermal_conductivity = True
        if any(word in text for word in ['insulation', 'isolate', 'non-conductive', 'dielectric']):
            req.requires_electrical_insulation = True

    def _apply_application_defaults(self, req: EMIRequirements):
        """Apply application-specific default values"""
        defaults = {
            ApplicationType.CONSUMER_ELECTRONICS: {
                'target_shielding_db': 40,
                'max_thickness': 0.001,  # 1mm
                'cost_sensitive': True
            },
            ApplicationType.MEDICAL_DEVICE: {
                'target_shielding_db': 60,
                'requires_corrosion_resistance': True,
                'environment': Environment.CLEANROOM
            },
            ApplicationType.AUTOMOTIVE: {
                'target_shielding_db': 50,
                'operating_temp_min': -40,
                'operating_temp_max': 125,
                'environment': Environment.HARSH
            },
            ApplicationType.AEROSPACE: {
                'target_shielding_db': 80,
                'requires_structural_strength': True,
                'environment': Environment.HARSH
            },
            ApplicationType.MILITARY: {
                'target_shielding_db': 100,
                'environment': Environment.HARSH,
                'requires_structural_strength': True
            },
            ApplicationType.TELECOM: {
                'frequency_min': 700e6,
                'frequency_max': 6e9,
                'target_shielding_db': 60
            },
            ApplicationType.DATA_CENTER: {
                'target_shielding_db': 50,
                'requires_thermal_conductivity': True,
                'environment': Environment.INDOOR
            }
        }

        if req.application_type in defaults:
            app_defaults = defaults[req.application_type]
            for key, value in app_defaults.items():
                if hasattr(req, key):
                    # Only apply default if not already set
                    current_value = getattr(req, key)
                    if current_value == EMIRequirements.__dataclass_fields__[key].default:
                        setattr(req, key, value)

    def _calculate_confidence(self, req: EMIRequirements, text: str):
        """Calculate confidence scores for extracted requirements"""
        scores = {}

        # Frequency confidence
        if req.primary_frequencies or 'hz' in text.lower() or any(
            band in text.lower() for band in ['wifi', 'bluetooth', '5g', '4g']):
            scores['frequency'] = 0.9
        else:
            scores['frequency'] = 0.5

        # Shielding effectiveness confidence
        if 'db' in text.lower() or any(
            word in text.lower() for word in ['shielding', 'attenuation', 'protection']):
            scores['shielding'] = 0.9
        else:
            scores['shielding'] = 0.6

        # Physical constraints confidence
        if any(word in text.lower() for word in ['mm', 'cm', 'thickness', 'weight']):
            scores['physical'] = 0.8
        else:
            scores['physical'] = 0.4

        # Application confidence
        if req.application_type != ApplicationType.GENERAL:
            scores['application'] = 0.8
        else:
            scores['application'] = 0.3

        req.confidence_scores = scores

    def _convert_to_hz(self, value: float, unit: str) -> float:
        """Convert frequency to Hz"""
        unit = unit.lower()
        conversions = {
            'hz': 1,
            'khz': 1e3,
            'mhz': 1e6,
            'ghz': 1e9
        }
        return value * conversions.get(unit, 1)

    def _convert_to_meters(self, value: float, unit: str) -> float:
        """Convert length to meters"""
        unit = unit.lower()
        conversions = {
            'm': 1,
            'cm': 0.01,
            'mm': 0.001,
            'um': 1e-6,
            'µm': 1e-6,
            'micrometer': 1e-6,
            'micron': 1e-6,
            'mil': 2.54e-5,
            'inch': 0.0254,
            'inches': 0.0254
        }
        return value * conversions.get(unit, 1)

    def _convert_to_kg(self, value: float, unit: str) -> float:
        """Convert weight to kg"""
        unit = unit.lower()
        conversions = {
            'kg': 1,
            'kilogram': 1,
            'g': 0.001,
            'gram': 0.001,
            'lb': 0.453592,
            'lbs': 0.453592,
            'pound': 0.453592,
            'oz': 0.0283495
        }
        return value * conversions.get(unit, 1)

    def _convert_to_celsius(self, value: float, unit: str) -> float:
        """Convert temperature to Celsius"""
        unit = unit.lower()
        if unit in ['f', 'fahrenheit']:
            return (value - 32) * 5/9
        elif unit in ['k', 'kelvin']:
            return value - 273.15
        return value

    def get_requirements_summary(self, req: EMIRequirements) -> str:
        """Generate a human-readable summary of extracted requirements"""
        summary = []
        summary.append(f"Application Type: {req.application_type.value.replace('_', ' ').title()}")
        summary.append(f"Frequency Range: {req.frequency_min/1e6:.1f} MHz - {req.frequency_max/1e9:.1f} GHz")
        summary.append(f"Target Shielding: {req.target_shielding_db} dB (min: {req.min_shielding_db} dB)")

        if req.max_thickness:
            summary.append(f"Max Thickness: {req.max_thickness*1000:.2f} mm")
        if req.max_weight:
            summary.append(f"Max Weight: {req.max_weight:.2f} kg/m²")

        summary.append(f"Environment: {req.environment.value.replace('_', ' ').title()}")
        summary.append(f"Temperature Range: {req.operating_temp_min}°C to {req.operating_temp_max}°C")

        # Special requirements
        special = []
        if req.requires_corrosion_resistance:
            special.append("Corrosion Resistant")
        if req.requires_flexibility:
            special.append("Flexible")
        if req.requires_transparency:
            special.append("Transparent")
        if req.requires_structural_strength:
            special.append("Structural Strength")
        if req.cost_sensitive:
            special.append("Cost Sensitive")

        if special:
            summary.append(f"Special Requirements: {', '.join(special)}")

        # Confidence
        avg_confidence = sum(req.confidence_scores.values()) / len(req.confidence_scores) if req.confidence_scores else 0
        summary.append(f"Extraction Confidence: {avg_confidence*100:.0f}%")

        return "\n".join(summary)