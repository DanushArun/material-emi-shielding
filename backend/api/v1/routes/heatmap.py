from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
import numpy as np

router = APIRouter()

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
    # Generating a mock 2D grid for the heatmap visualization.
    # In a real scenario, this would call the core TMM physics engine.
    
    freqs = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)
    thicks = np.linspace(req.thickness_start_mm, req.thickness_end_mm, req.num_points)
    
    se_matrix = []
    for thick in thicks:
        row = []
        for freq in freqs:
            # Simulate physics: SE increases with thickness and frequency, with some "resonance" dips
            base_se = 30 + 10 * np.log10(freq / 100) + 15 * thick
            resonance = 10 * np.sin(freq / 1000 * np.pi) * np.cos(thick * np.pi)
            se = base_se + resonance
            row.append(max(0, float(se)))
        se_matrix.append(row)
        
    return HeatmapResponse(
        frequencies_mhz=freqs.tolist(),
        thicknesses_mm=thicks.tolist(),
        se_matrix_db=se_matrix
    )
