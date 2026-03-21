"""Shared helpers for API routes.

Centralizes composite material property calculation logic
so it's consistent across physics, materials, and analysis endpoints.
"""
from typing import Dict, Tuple
from src.materials.material_properties import material_db


def calculate_composite_properties(composition: Dict[str, float]) -> Dict[str, float]:
    """Calculate effective electromagnetic properties from elemental composition.

    Uses weighted-sum for conductivity/density, geometric mean for permeability/permittivity.
    Same logic that was in app.py lines 1109-1142.

    Args:
        composition: Element symbols to weight percentages (must sum to ~100).

    Returns:
        Dict with conductivity, permeability, permittivity, density.

    Raises:
        ValueError: If an unknown element is provided.
    """
    conductivity = 0.0
    permeability = 1.0
    permittivity = 1.0
    density = 0.0

    for element, percentage in composition.items():
        elem_data = material_db.get_material(element)
        if not elem_data:
            raise ValueError(f"Unknown element: {element}")

        weight = percentage / 100.0

        # Conductivity: weighted sum
        conductivity += elem_data.get('electrical_conductivity', 1e6) * weight

        # Permeability: geometric mean
        elem_perm = max(elem_data.get('relative_permeability', 1.0), 0.999)
        permeability *= elem_perm ** weight

        # Permittivity: geometric mean
        elem_eps = max(elem_data.get('relative_permittivity', 1.0), 1.0)
        permittivity *= elem_eps ** weight

        # Density: weighted sum
        density += elem_data.get('density', 1000) * weight

    return {
        'conductivity': max(conductivity, 1e-10),
        'permeability': max(permeability, 0.999),
        'permittivity': max(permittivity, 1.0),
        'density': density,
    }
