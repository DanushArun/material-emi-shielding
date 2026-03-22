from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np

from src.physics.emi_calculations import EMICalculator
from backend.api.v1.routes._helpers import calculate_composite_properties

router = APIRouter()
calculator = EMICalculator()


class HeatmapRequest(BaseModel):
    composition: Dict[str, float]
    freq_start_mhz: float
    freq_end_mhz: float
    thickness_start_mm: float
    thickness_end_mm: float
    num_points: int = 20


class HeatmapResponse(BaseModel):
    frequencies_mhz: List[float]
    thicknesses_mm: List[float]
    se_matrix_db: List[List[float]]


@router.post("/generate", response_model=HeatmapResponse)
async def generate_heatmap(req: HeatmapRequest):
    """Generate a 2D SE heatmap (frequency x thickness) using the real physics engine."""
    try:
        props = calculate_composite_properties(req.composition)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)
    thicks_mm = np.linspace(req.thickness_start_mm, req.thickness_end_mm, req.num_points)

    se_matrix = []
    for t_mm in thicks_mm:
        row = []
        for f_mhz in freqs_mhz:
            result = calculator.calculate_shielding_effectiveness(
                conductivity=props["conductivity"],
                relative_permeability=props["permeability"],
                relative_permittivity=props["permittivity"],
                thickness=t_mm * 1e-3,       # mm -> m
                frequency=f_mhz * 1e6,       # MHz -> Hz
            )
            row.append(float(result["total_se"]))
        se_matrix.append(row)

    return HeatmapResponse(
        frequencies_mhz=freqs_mhz.tolist(),
        thicknesses_mm=thicks_mm.tolist(),
        se_matrix_db=se_matrix,
    )
