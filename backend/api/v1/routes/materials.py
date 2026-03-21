"""Materials database endpoints - periodic table, alloys, and composite property calculation."""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, List, Any

from src.materials.material_properties import material_db
from backend.api.v1.schemas.materials import (
    CompositePropertiesRequest, CompositePropertiesResponse,
)
from backend.api.v1.routes._helpers import calculate_composite_properties

router = APIRouter()


@router.get("/elements")
async def list_elements() -> List[Dict[str, Any]]:
    """Get all elements from the periodic table with their EM properties."""
    elements = []
    for symbol in sorted(material_db.periodic_table.keys()):
        data = material_db.periodic_table[symbol]
        elements.append({"symbol": symbol, **data})
    return elements


@router.get("/elements/{symbol}")
async def get_element(symbol: str) -> Dict[str, Any]:
    """Get properties of a single element by symbol (e.g. Cu, Fe, Al)."""
    data = material_db.get_material(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Element '{symbol}' not found")
    return {"symbol": symbol, **data}


@router.get("/alloys")
async def list_alloys() -> List[Dict[str, Any]]:
    """Get all pre-defined alloys and their properties."""
    alloys = []
    for key, data in material_db.alloys.items():
        alloys.append({"key": key, **data})
    return alloys


@router.get("/alloys/{key}")
async def get_alloy(key: str) -> Dict[str, Any]:
    """Get properties of a specific alloy by key (e.g. steel_1018, mu_metal)."""
    if key not in material_db.alloys:
        raise HTTPException(status_code=404, detail=f"Alloy '{key}' not found")
    return {"key": key, **material_db.alloys[key]}


@router.post("/composite-properties", response_model=CompositePropertiesResponse)
async def compute_composite_properties(request: CompositePropertiesRequest):
    """Calculate effective EM properties from elemental composition (weight %)."""
    try:
        props = calculate_composite_properties(request.elements)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CompositePropertiesResponse(
        conductivity=props['conductivity'],
        permeability=props['permeability'],
        permittivity=props['permittivity'],
        density=props['density'],
        composition=request.elements,
    )
