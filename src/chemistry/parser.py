"""
Chemical formula parsing and reaction composition calculations.
Extracted from the Streamlit app for reuse across backend API and frontend.
"""

import re
from typing import Dict, List, Optional

from src.materials.material_properties import material_db


class ChemicalParser:
    """Parse and validate chemical formulas."""

    @staticmethod
    def parse_formula(formula: str) -> Dict[str, int]:
        """Parse a chemical formula into element counts.

        Examples:
            "H2O" -> {"H": 2, "O": 1}
            "Al2O3" -> {"Al": 2, "O": 3}
            "Cu" -> {"Cu": 1}
        """
        formula = formula.replace(" ", "")
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        composition: Dict[str, int] = {}
        for element, count in matches:
            if not element:
                continue
            count = int(count) if count else 1
            composition[element] = composition.get(element, 0) + count
        return composition

    @staticmethod
    def format_formula(composition: Dict[str, int]) -> str:
        """Format element composition to a plain-text chemical formula string.

        Returns plain text (e.g. "Al2O3"), not HTML.
        """
        if not composition:
            return ""
        parts = []
        for element, count in sorted(composition.items()):
            if count == 1:
                parts.append(element)
            else:
                parts.append(f"{element}{count}")
        return "".join(parts)


class ReactionEngine:
    """Handle chemical reactions and composition calculations for EMI materials."""

    def __init__(self):
        self.molecules: List[dict] = []

    def add_molecule(self, formula: str, coefficient: int = 1):
        """Add a molecule to the reaction by chemical formula."""
        composition = ChemicalParser.parse_formula(formula)
        if composition:
            self.molecules.append({
                'formula': formula,
                'composition': composition,
                'coefficient': coefficient,
                'molecular_weight': self._calculate_molecular_weight(composition),
            })

    def add_direct_composition(self, composition: Dict[str, float], display_name: str):
        """Add a direct weight-percentage composition (e.g. {"Fe": 70.0, "Cr": 30.0})."""
        self.molecules.append({
            'type': 'direct',
            'composition': composition.copy(),
            'formula': 'Direct Composition',
            'coefficient': 1,
            'molecular_weight': sum(
                material_db.get_material(elem).get('atomic_weight', 50) * pct / 100
                for elem, pct in composition.items()
                if material_db.get_material(elem)
            ),
            'display_name': display_name,
        })

    def _calculate_molecular_weight(self, composition: Dict[str, int]) -> float:
        """Calculate molecular weight from element counts."""
        total = 0.0
        for element, count in composition.items():
            elem_data = material_db.get_material(element)
            weight = elem_data.get('atomic_weight', 50.0) if elem_data else 50.0
            total += weight * count
        return total

    def get_reaction_equation(self) -> str:
        """Get the formatted reaction equation string."""
        if not self.molecules:
            return "No reaction defined"
        parts = []
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                parts.append(mol.get('display_name', 'Direct Composition'))
            else:
                coeff = f"{mol['coefficient']}" if mol['coefficient'] > 1 else ""
                formula = ChemicalParser.format_formula(mol['composition'])
                parts.append(f"{coeff}{formula}")
        return " + ".join(parts)

    def get_total_composition(self) -> Dict[str, float]:
        """Calculate total elemental composition by mass percentage.

        For direct compositions, returns as-is.
        For molecular compositions, calculates mass fractions from molecular weights.
        """
        if not self.molecules:
            return {}
        if len(self.molecules) == 1 and self.molecules[0].get('type') == 'direct':
            return self.molecules[0]['composition']

        element_masses: Dict[str, float] = {}
        total_mass = 0.0
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                continue
            mol_mass = mol['molecular_weight'] * mol['coefficient']
            total_mass += mol_mass
            for element, count in mol['composition'].items():
                elem_data = material_db.get_material(element)
                atomic_weight = elem_data.get('atomic_weight', 50.0) if elem_data else 50.0
                element_mass = atomic_weight * count * mol['coefficient']
                element_masses[element] = element_masses.get(element, 0) + element_mass

        if total_mass > 0:
            return {el: (mass / total_mass) * 100 for el, mass in element_masses.items()}
        return {}

    def clear(self):
        """Clear all molecules."""
        self.molecules.clear()
