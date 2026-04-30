"""Material recommendation endpoint.

Wraps ``src.physics.recommendation`` to expose the inverse SE solver
as a REST API.  Engineers specify requirements (target SE, frequency,
max thickness, optional weight limit) and receive ranked material
recommendations.

Endpoints
---------
POST /recommendation/search    Search for materials meeting SE requirements.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from dataclasses import asdict

from src.physics.recommendation import (
    MaterialConstraints,
    MaterialRecommendation,
    recommend_materials,
    pareto_filter,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RecommendationSearchRequest(BaseModel):
    """Request for material recommendation search."""

    target_se_db: float = Field(
        ..., gt=0.0,
        description="Minimum required shielding effectiveness in dB.",
    )
    frequency_mhz: float = Field(
        ..., gt=0.0,
        description="Primary frequency in MHz.",
    )
    max_thickness_mm: float = Field(
        ..., gt=0.0,
        description="Maximum allowable thickness in mm.",
    )
    max_density_kg_m3: Optional[float] = Field(
        None, gt=0.0,
        description="Maximum material density in kg/m^3 (optional weight filter).",
    )
    freq_range_start_mhz: Optional[float] = Field(
        None, gt=0.0,
        description="Start of frequency range in MHz (optional).",
    )
    freq_range_end_mhz: Optional[float] = Field(
        None, gt=0.0,
        description="End of frequency range in MHz (optional).",
    )
    n_results: int = Field(
        5, ge=1, le=50,
        description="Maximum number of results to return.",
    )
    pareto_only: bool = Field(
        False,
        description="If true, return only Pareto-optimal solutions.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "target_se_db": 40.0,
                "frequency_mhz": 1000.0,
                "max_thickness_mm": 2.0,
                "max_density_kg_m3": 9000.0,
                "n_results": 5,
                "pareto_only": False,
            }
        }
    }


class RecommendationItem(BaseModel):
    """A single material recommendation."""

    material_name: str
    achieved_se_db: float
    optimal_thickness_m: float
    density_kg_m3: float
    se_margin_db: float
    meets_target: bool
    conductivity_s_m: float
    relative_permeability: float
    reflection_loss_db: float
    absorption_loss_db: float
    skin_depth_m: float
    explanation: str


class RecommendationSearchResponse(BaseModel):
    """Response from material recommendation search."""

    recommendations: List[RecommendationItem]
    constraints: dict
    total_candidates_evaluated: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/search", response_model=RecommendationSearchResponse)
async def search_materials(req: RecommendationSearchRequest):
    """Search for materials meeting SE requirements.

    Uses the EMICalculator forward solver to evaluate every element and alloy
    in the material database against the specified constraints.  Returns ranked
    recommendations sorted by target-meeting first, then SE margin, thickness,
    and density.
    """
    try:
        # Unit conversion: MHz -> Hz, mm -> m
        frequency_hz = req.frequency_mhz * 1e6
        max_thickness_m = req.max_thickness_mm * 1e-3

        frequency_range_hz = None
        if req.freq_range_start_mhz is not None and req.freq_range_end_mhz is not None:
            frequency_range_hz = (
                req.freq_range_start_mhz * 1e6,
                req.freq_range_end_mhz * 1e6,
            )

        constraints = MaterialConstraints(
            target_se_db=req.target_se_db,
            frequency_hz=frequency_hz,
            max_thickness_m=max_thickness_m,
            max_density_kg_m3=req.max_density_kg_m3,
            frequency_range_hz=frequency_range_hz,
        )

        # Run the recommendation engine (returns all candidates up to n_results)
        # First get a large pool so we can report total_candidates_evaluated
        all_recs = recommend_materials(constraints, n_results=1000)
        total_evaluated = len(all_recs)

        if req.pareto_only:
            all_recs = pareto_filter(all_recs)

        # Trim to requested count
        result_recs = all_recs[: req.n_results]

        # Serialise dataclasses to dicts
        rec_dicts = [asdict(r) for r in result_recs]

        constraints_echo = {
            "target_se_db": req.target_se_db,
            "frequency_mhz": req.frequency_mhz,
            "max_thickness_mm": req.max_thickness_mm,
            "max_density_kg_m3": req.max_density_kg_m3,
            "n_results": req.n_results,
            "pareto_only": req.pareto_only,
        }

        return RecommendationSearchResponse(
            recommendations=rec_dicts,
            constraints=constraints_echo,
            total_candidates_evaluated=total_evaluated,
        )

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
