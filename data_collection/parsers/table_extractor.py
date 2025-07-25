"""
PDF table extractor specialized for EMI shielding data.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import tabula
import pdfplumber
import camelot
import logging
from dataclasses import dataclass


@dataclass
class EMIData:
    """Structure for EMI shielding measurement data."""
    material: str
    composition: Optional[Dict[str, float]] = None
    conductivity: Optional[float] = None  # S/m
    permeability: Optional[float] = None  # relative
    permittivity: Optional[float] = None  # relative
    thickness: Optional[float] = None  # mm
    frequency: Optional[float] = None  # Hz
    frequency_range: Optional[Tuple[float, float]] = None  # Hz
    shielding_effectiveness: Optional[float] = None  # dB
    reflection_loss: Optional[float] = None  # dB
    absorption_loss: Optional[float] = None  # dB
    synthesis_method: Optional[str] = None
    filler_loading: Optional[float] = None  # wt% or vol%
    particle_size: Optional[float] = None  # nm or μm
    measurement_standard: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class EMITableExtractor:
    """Extract EMI shielding data from PDF tables."""
    
    # Common EMI-related keywords to identify relevant tables
    EMI_KEYWORDS = [
        'shielding effectiveness', 'se ', 'emi', 'electromagnetic',
        'reflection loss', 'absorption loss', 'db', 'ghz', 'mhz',
        'conductivity', 's/m', 's/cm', 'permeability', 'permittivity'
    ]
    
    # Unit conversion factors
    UNIT_CONVERSIONS = {
        'conductivity': {
            's/cm': 100,          # S/cm to S/m
            'ms/cm': 0.1,         # mS/cm to S/m
            's/mm': 1000,         # S/mm to S/m
            'ω·cm': lambda x: 100/x,  # Ω·cm to S/m
            'ω·m': lambda x: 1/x,     # Ω·m to S/m
        },
        'frequency': {
            'hz': 1,
            'khz': 1e3,
            'mhz': 1e6,
            'ghz': 1e9,
        },
        'thickness': {
            'mm': 1,
            'cm': 10,
            'm': 1000,
            'μm': 0.001,
            'um': 0.001,
            'nm': 1e-6,
        },
        'particle_size': {
            'nm': 1,
            'μm': 1000,
            'um': 1000,
            'mm': 1e6,
        }
    }
    
    def __init__(self):
        """Initialize the table extractor."""
        self.logger = logging.getLogger(__name__)
        
    def extract_from_pdf(self, pdf_path: Path) -> List[EMIData]:
        """
        Extract EMI shielding data from all tables in a PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of EMIData objects extracted from tables
        """
        all_data = []
        
        # Try multiple extraction methods
        methods = [
            ('tabula', self._extract_with_tabula),
            ('camelot', self._extract_with_camelot),
            ('pdfplumber', self._extract_with_pdfplumber)
        ]
        
        for method_name, method_func in methods:
            try:
                self.logger.info(f"Trying {method_name} for {pdf_path.name}")
                tables = method_func(pdf_path)
                
                for i, table in enumerate(tables):
                    if self._is_emi_table(table):
                        data = self._parse_emi_table(table)
                        all_data.extend(data)
                        self.logger.info(f"Extracted {len(data)} records from table {i+1}")
                        
                if all_data:
                    break  # Stop if we successfully extracted data
                    
            except Exception as e:
                self.logger.warning(f"{method_name} failed: {e}")
                
        return all_data
    
    def _extract_with_tabula(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extract tables using tabula-py."""
        return tabula.read_pdf(
            str(pdf_path),
            pages='all',
            multiple_tables=True,
            lattice=True  # Try lattice-based extraction first
        )
    
    def _extract_with_camelot(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extract tables using camelot."""
        tables = camelot.read_pdf(
            str(pdf_path),
            pages='all',
            flavor='lattice'
        )
        return [table.df for table in tables]
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extract tables using pdfplumber."""
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        # Convert to DataFrame
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)
                        
        return tables
    
    def _is_emi_table(self, table: pd.DataFrame) -> bool:
        """Check if table contains EMI-related data."""
        # Convert table to string and check for keywords
        table_text = ' '.join(table.astype(str).values.flatten()).lower()
        
        keyword_count = sum(1 for keyword in self.EMI_KEYWORDS if keyword in table_text)
        
        return keyword_count >= 2  # At least 2 EMI keywords
    
    def _parse_emi_table(self, table: pd.DataFrame) -> List[EMIData]:
        """Parse EMI data from a table."""
        data_list = []
        
        # Clean column names
        table.columns = [self._clean_column_name(col) for col in table.columns]
        
        # Identify column mappings
        column_map = self._identify_columns(table.columns)
        
        # Parse each row
        for _, row in table.iterrows():
            try:
                emi_data = self._parse_row(row, column_map)
                if emi_data and emi_data.shielding_effectiveness is not None:
                    data_list.append(emi_data)
            except Exception as e:
                self.logger.debug(f"Failed to parse row: {e}")
                
        return data_list
    
    def _clean_column_name(self, col_name: str) -> str:
        """Clean and standardize column names."""
        if pd.isna(col_name):
            return 'unknown'
            
        # Convert to string and lowercase
        col_name = str(col_name).lower().strip()
        
        # Remove extra whitespace
        col_name = re.sub(r'\s+', ' ', col_name)
        
        return col_name
    
    def _identify_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map table columns to EMI data fields."""
        column_map = {}
        
        # Patterns for different fields
        patterns = {
            'material': r'material|sample|composition|composite',
            'conductivity': r'conductivity|σ|sigma',
            'permeability': r'permeability|μ|mu',
            'permittivity': r'permittivity|ε|epsilon',
            'thickness': r'thickness|t\s',
            'frequency': r'frequency|freq|f\s',
            'shielding_effectiveness': r'shielding|se\s|emi\s+se|effectiveness',
            'reflection_loss': r'reflection|rl\s',
            'absorption_loss': r'absorption|al\s',
            'filler_loading': r'loading|content|wt%|vol%|concentration',
            'particle_size': r'size|diameter|nm|μm',
            'synthesis_method': r'method|synthesis|preparation|process'
        }
        
        for col in columns:
            for field, pattern in patterns.items():
                if re.search(pattern, col):
                    column_map[field] = col
                    break
                    
        return column_map
    
    def _parse_row(self, row: pd.Series, column_map: Dict[str, str]) -> Optional[EMIData]:
        """Parse a single row into EMIData."""
        emi_data = EMIData(material='')
        
        # Extract material name
        if 'material' in column_map:
            emi_data.material = self._clean_material_name(row[column_map['material']])
        
        # Extract numerical values
        for field in ['conductivity', 'permeability', 'permittivity', 'thickness',
                     'frequency', 'shielding_effectiveness', 'reflection_loss',
                     'absorption_loss', 'filler_loading', 'particle_size']:
            if field in column_map:
                value = self._extract_numerical_value(row[column_map[field]], field)
                if value is not None:
                    setattr(emi_data, field, value)
        
        # Extract text fields
        if 'synthesis_method' in column_map:
            emi_data.synthesis_method = str(row[column_map['synthesis_method']]).strip()
        
        # Parse composition if in material name
        emi_data.composition = self._parse_composition(emi_data.material)
        
        return emi_data if emi_data.material else None
    
    def _clean_material_name(self, material: Any) -> str:
        """Clean and standardize material name."""
        if pd.isna(material):
            return ''
            
        material = str(material).strip()
        
        # Remove reference numbers in brackets
        material = re.sub(r'\[\d+\]', '', material)
        
        # Standardize separators
        material = material.replace('/', '-')
        
        return material
    
    def _extract_numerical_value(self, value: Any, field_type: str) -> Optional[float]:
        """Extract numerical value with unit conversion."""
        if pd.isna(value):
            return None
            
        value_str = str(value).lower().strip()
        
        # Handle ranges (take average)
        if '-' in value_str and not value_str.startswith('-'):
            parts = value_str.split('-')
            if len(parts) == 2:
                try:
                    val1 = self._parse_single_value(parts[0], field_type)
                    val2 = self._parse_single_value(parts[1], field_type)
                    if val1 is not None and val2 is not None:
                        return (val1 + val2) / 2
                except:
                    pass
        
        # Handle single values
        return self._parse_single_value(value_str, field_type)
    
    def _parse_single_value(self, value_str: str, field_type: str) -> Optional[float]:
        """Parse a single numerical value with units."""
        # Extract number and unit
        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Zμ·Ω/]+)?', value_str)
        
        if not match:
            return None
            
        try:
            number = float(match.group(1))
            unit = match.group(2) if match.group(2) else ''
            
            # Apply unit conversion if needed
            if field_type in self.UNIT_CONVERSIONS and unit:
                conversions = self.UNIT_CONVERSIONS[field_type]
                
                for unit_pattern, factor in conversions.items():
                    if unit_pattern in unit.lower():
                        if callable(factor):
                            return factor(number)
                        else:
                            return number * factor
                            
            return number
            
        except ValueError:
            return None
    
    def _parse_composition(self, material_name: str) -> Optional[Dict[str, float]]:
        """Parse composition from material name."""
        composition = {}
        
        # Pattern for element-percentage pairs (e.g., Fe70Co30)
        pattern = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, material_name)
        
        if matches:
            total = sum(float(pct) for _, pct in matches)
            
            # Normalize to 100% if close
            if 90 <= total <= 110:
                for element, pct in matches:
                    composition[element] = float(pct) * 100 / total
                    
        return composition if composition else None
    
    def extract_from_text(self, text: str) -> List[EMIData]:
        """Extract EMI data from plain text (for non-table data)."""
        data_list = []
        
        # Pattern for inline EMI values
        # Example: "The Fe-Ni composite showed SE of 45 dB at 1 GHz"
        pattern = r'([A-Za-z0-9\-/]+)\s+(?:showed|exhibited|has|achieved)\s+' \
                 r'(?:SE|shielding effectiveness)\s+of\s+' \
                 r'([\d.]+)\s*dB\s+at\s+([\d.]+)\s*(MHz|GHz)'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            emi_data = EMIData(
                material=match.group(1),
                shielding_effectiveness=float(match.group(2)),
                frequency=self._convert_frequency(
                    float(match.group(3)), 
                    match.group(4).lower()
                )
            )
            data_list.append(emi_data)
            
        return data_list
    
    def _convert_frequency(self, value: float, unit: str) -> float:
        """Convert frequency to Hz."""
        multipliers = {'hz': 1, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}
        return value * multipliers.get(unit, 1)