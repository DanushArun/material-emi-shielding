"""Physics calculation endpoints - single-point EMI shielding calculations.

For sweep/optimization endpoints, see analysis.py.
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import time

from src.physics.emi_calculations import EMICalculator
from backend.api.v1.schemas.physics import SingleCalculationRequest, CalculationResponse
from backend.api.v1.routes._helpers import calculate_composite_properties

router = APIRouter()
calculator = EMICalculator()


@router.post("/calculate", response_model=CalculationResponse)
async def calculate_shielding(request: SingleCalculationRequest) -> Dict[str, Any]:
    """
    Calculate EMI shielding effectiveness for a single configuration.

    Performs full electromagnetic shielding calculations including:
    - Reflection loss (impedance mismatch at air-material interface)
    - Absorption loss (exponential decay through material)
    - Multiple reflection corrections (thin shield bouncing)
    - Grain boundary scattering effects (Mayadas-Shatzkes model)
    """
    start_time = time.time()

    try:
        props = calculate_composite_properties(request.composition)
        conductivity = props['conductivity']

        # Apply grain size effect if provided
        if request.grain_size_um is not None:
            conductivity = calculator.calculate_grain_size_effect(
                bulk_conductivity=conductivity,
                grain_size=request.grain_size_um * 1e-6,
            )

        # Calculate shielding effectiveness
        result = calculator.calculate_shielding_effectiveness(
            conductivity=conductivity,
            relative_permeability=props['permeability'],
            relative_permittivity=props['permittivity'],
            thickness=request.thickness_mm * 1e-3,
            frequency=request.frequency_mhz * 1e6,
            grain_size=request.grain_size_um * 1e-6 if request.grain_size_um else None,
        )

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "shielding_effectiveness_db": result['total_se'],
            "reflection_loss_db": result['reflection_loss'],
            "absorption_loss_db": result['absorption_loss'],
            "multiple_reflection_db": result.get('multiple_reflection_loss', 0.0),
            "skin_depth_um": result['skin_depth'] * 1e6,
            "effective_conductivity": result.get('effective_conductivity', conductivity),
            "execution_time_ms": execution_time_ms,
            "confidence": result.get('confidence'),
            "confidence_level": result.get('confidence_level'),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Calculation failed: {str(e)}",
        )


@router.post("/material-properties")
async def get_material_properties(request: Dict[str, float]) -> Dict[str, Any]:
    """Get calculated material properties for a composition.

    Accepts a JSON body like {"Cu": 70, "Zn": 30}.
    Returns conductivity, permeability, permittivity, and density.
    """
    try:
        props = calculate_composite_properties(request)
        return {
            "composition": request,
            "conductivity_s_per_m": props['conductivity'],
            "relative_permeability": props['permeability'],
            "relative_permittivity": props['permittivity'],
            "density_kg_per_m3": props['density'],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
