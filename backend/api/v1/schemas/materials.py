"""Pydantic schemas for materials API endpoints."""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class ElementResponse(BaseModel):
    symbol: str
    name: str = ""
    atomic_number: int = 0
    atomic_weight: float = 0.0
    electrical_conductivity: float = Field(0.0, description="S/m")
    relative_permeability: float = 1.0
    density: float = Field(0.0, description="kg/m3")


class AlloyResponse(BaseModel):
    key: str
    name: str
    composition: Dict[str, float]
    density: float
    electrical_conductivity: float
    relative_permeability: float
    relative_permittivity: float = 1.0
    note: Optional[str] = None


class CompositePropertiesRequest(BaseModel):
    elements: Dict[str, float] = Field(
        ..., description="Element symbols and weight percentages"
    )


class CompositePropertiesResponse(BaseModel):
    conductivity: float = Field(description="Effective conductivity (S/m)")
    permeability: float = Field(description="Effective relative permeability")
    permittivity: float = Field(description="Effective relative permittivity")
    density: float = Field(description="Effective density (kg/m3)")
    composition: Dict[str, float]
