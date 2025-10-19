"""
Physics calculation endpoints
Wraps the existing EMI calculations from src/physics/emi_calculations.py
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.physics.emi_calculations import EMICalculator
from core.security import get_current_user_id, require_subscription

router = APIRouter()

# Initialize calculator
calculator = EMICalculator()


# Request/Response Models
class MaterialComposition(BaseModel):
    """Material composition as element percentages"""
    elements: Dict[str, float] = Field(..., description="Element symbols and percentages (e.g., {'Cu': 100})")

    @validator('elements')
    def validate_composition(cls, v):
        """Ensure percentages sum to approximately 100"""
        total = sum(v.values())
        if not (99.0 <= total <= 101.0):
            raise ValueError(f"Element percentages must sum to ~100%, got {total}%")
        return v


class SingleCalculationRequest(BaseModel):
    """Request for single-point EMI calculation"""
    composition: MaterialComposition
    frequency_mhz: float = Field(..., gt=0.001, lt=100000, description="Frequency in MHz")
    thickness_mm: float = Field(..., gt=0.001, lt=100, description="Thickness in mm")
    grain_size_um: Optional[float] = Field(None, gt=0.001, description="Grain size in micrometers")

    class Config:
        schema_extra = {
            "example": {
                "composition": {"elements": {"Cu": 100}},
                "frequency_mhz": 1000,
                "thickness_mm": 1.0,
                "grain_size_um": 50.0
            }
        }


class FrequencySweepRequest(BaseModel):
    """Request for frequency sweep calculation"""
    composition: MaterialComposition
    frequency_start_mhz: float = Field(..., gt=0.001, description="Start frequency in MHz")
    frequency_end_mhz: float = Field(..., gt=0.001, description="End frequency in MHz")
    num_points: int = Field(100, gt=1, le=1000, description="Number of frequency points")
    thickness_mm: float = Field(..., gt=0.001, lt=100, description="Thickness in mm")
    grain_size_um: Optional[float] = Field(None, gt=0.001, description="Grain size in micrometers")


class ThicknessOptimizationRequest(BaseModel):
    """Request for thickness optimization"""
    composition: MaterialComposition
    frequency_mhz: float = Field(..., gt=0.001, lt=100000, description="Frequency in MHz")
    target_se_db: float = Field(..., gt=0, description="Target shielding effectiveness in dB")
    thickness_min_mm: float = Field(0.1, gt=0.001, description="Minimum thickness in mm")
    thickness_max_mm: float = Field(10.0, gt=0.001, description="Maximum thickness in mm")
    grain_size_um: Optional[float] = Field(None, gt=0.001, description="Grain size in micrometers")


class CalculationResponse(BaseModel):
    """Response for single calculation"""
    shielding_effectiveness_db: float
    reflection_loss_db: float
    absorption_loss_db: float
    multiple_reflection_db: float
    skin_depth_um: float
    intrinsic_impedance: complex
    effective_conductivity: float
    execution_time_ms: float


# Endpoints
@router.post("/calculate", response_model=CalculationResponse)
async def calculate_shielding(
    request: SingleCalculationRequest,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Calculate EMI shielding effectiveness for a single configuration

    This endpoint performs full electromagnetic shielding calculations including:
    - Reflection loss (impedance mismatch)
    - Absorption loss (exponential decay)
    - Multiple reflection corrections
    - Grain boundary scattering effects (if grain size provided)
    """
    start_time = time.time()

    try:
        # Calculate material properties
        properties = calculator.calculate_material_properties(
            request.composition.elements
        )

        # Apply grain size effect if provided
        effective_conductivity = properties['conductivity']
        if request.grain_size_um is not None:
            effective_conductivity = calculator.apply_grain_size_effect(
                base_conductivity=properties['conductivity'],
                grain_size_um=request.grain_size_um,
                composition=request.composition.elements
            )

        # Calculate shielding effectiveness
        results = calculator.calculate_shielding_effectiveness(
            frequency_hz=request.frequency_mhz * 1e6,
            thickness_m=request.thickness_mm * 1e-3,
            conductivity=effective_conductivity,
            permeability=properties['permeability'],
            permittivity=properties['permittivity']
        )

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "shielding_effectiveness_db": results['total_se'],
            "reflection_loss_db": results['reflection_loss'],
            "absorption_loss_db": results['absorption_loss'],
            "multiple_reflection_db": results['multiple_reflection'],
            "skin_depth_um": results['skin_depth'] * 1e6,  # Convert to micrometers
            "intrinsic_impedance": results['intrinsic_impedance'],
            "effective_conductivity": effective_conductivity,
            "execution_time_ms": execution_time_ms
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Calculation failed: {str(e)}"
        )


@router.post("/frequency-sweep")
async def frequency_sweep(
    request: FrequencySweepRequest,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Perform frequency sweep analysis

    Calculates shielding effectiveness across a range of frequencies
    to visualize frequency-dependent performance.
    """
    start_time = time.time()

    try:
        # Calculate material properties
        properties = calculator.calculate_material_properties(
            request.composition.elements
        )

        # Apply grain size effect if provided
        effective_conductivity = properties['conductivity']
        if request.grain_size_um is not None:
            effective_conductivity = calculator.apply_grain_size_effect(
                base_conductivity=properties['conductivity'],
                grain_size_um=request.grain_size_um,
                composition=request.composition.elements
            )

        # Perform frequency sweep
        results = calculator.frequency_sweep(
            frequency_start_hz=request.frequency_start_mhz * 1e6,
            frequency_end_hz=request.frequency_end_mhz * 1e6,
            num_points=request.num_points,
            thickness_m=request.thickness_mm * 1e-3,
            conductivity=effective_conductivity,
            permeability=properties['permeability'],
            permittivity=properties['permittivity']
        )

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "frequencies_mhz": [f / 1e6 for f in results['frequencies']],
            "total_se_db": results['total_se'],
            "reflection_loss_db": results['reflection_loss'],
            "absorption_loss_db": results['absorption_loss'],
            "skin_depth_um": [sd * 1e6 for sd in results['skin_depth']],
            "execution_time_ms": execution_time_ms
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Frequency sweep failed: {str(e)}"
        )


@router.post("/optimize-thickness")
async def optimize_thickness(
    request: ThicknessOptimizationRequest,
    user_id: str = Depends(require_subscription("pro"))
) -> Dict[str, Any]:
    """
    Find optimal thickness to achieve target shielding effectiveness

    **Requires Pro or Enterprise subscription**

    Uses binary search to efficiently find the minimum thickness
    required to achieve the target SE.
    """
    start_time = time.time()

    try:
        # Calculate material properties
        properties = calculator.calculate_material_properties(
            request.composition.elements
        )

        # Apply grain size effect if provided
        effective_conductivity = properties['conductivity']
        if request.grain_size_um is not None:
            effective_conductivity = calculator.apply_grain_size_effect(
                base_conductivity=properties['conductivity'],
                grain_size_um=request.grain_size_um,
                composition=request.composition.elements
            )

        # Optimize thickness
        results = calculator.optimize_thickness(
            target_se_db=request.target_se_db,
            frequency_hz=request.frequency_mhz * 1e6,
            conductivity=effective_conductivity,
            permeability=properties['permeability'],
            permittivity=properties['permittivity'],
            thickness_min_m=request.thickness_min_mm * 1e-3,
            thickness_max_m=request.thickness_max_mm * 1e-3
        )

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "optimal_thickness_mm": results['optimal_thickness'] * 1000,
            "achieved_se_db": results['achieved_se'],
            "iterations": results.get('iterations', 0),
            "execution_time_ms": execution_time_ms
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Thickness optimization failed: {str(e)}"
        )


@router.get("/material-properties")
async def get_material_properties(
    composition: str,  # JSON string like '{"Cu":100}'
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Get calculated material properties for a composition

    Returns conductivity, permeability, permittivity, and density
    calculated from elemental composition.
    """
    try:
        import json
        composition_dict = json.loads(composition)

        properties = calculator.calculate_material_properties(composition_dict)

        return {
            "composition": composition_dict,
            "conductivity_s_per_m": properties['conductivity'],
            "relative_permeability": properties['permeability'],
            "relative_permittivity": properties['permittivity'],
            "density_kg_per_m3": properties.get('density', None)
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid composition format. Expected JSON object like {\"Cu\":100}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Property calculation failed: {str(e)}"
        )
