"""Composite material conductivity model endpoints.

Wraps src.physics.composite_models for heterogeneous (filler-in-matrix)
composites.  Supports percolation theory, the McLachlan GEM equation,
Maxwell-Garnett EMT, Bruggeman symmetric EMT, and Hashin-Shtrikman bounds.

All conductivity values are in S/m; volume fractions are dimensionless [0, 1].

Endpoints
---------
POST /composites/percolation        Power-law percolation conductivity.
POST /composites/gem                McLachlan General Effective Media equation.
POST /composites/threshold-estimate Estimate percolation threshold from filler geometry.
POST /composites/all-models         Compare results from all available models.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional
import time

from src.physics.composite_models import (
    percolation_conductivity,
    mclachlan_gem,
    maxwell_garnett,
    bruggeman_emt,
    hashin_shtrikman_bounds,
    percolation_threshold_rods,
    percolation_threshold_disks,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared sub-schemas
# ---------------------------------------------------------------------------

class FillerMatrixParams(BaseModel):
    """Base conductivity parameters shared by percolation and GEM requests."""

    filler_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Volume fraction of conductive filler (0 to 1).",
    )
    percolation_threshold: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Critical volume fraction f_c at which a percolating network forms.",
    )
    sigma_filler: float = Field(
        ...,
        ge=0.0,
        description="Electrical conductivity of the filler phase (S/m).",
    )
    sigma_matrix: float = Field(
        ...,
        ge=0.0,
        description="Electrical conductivity of the insulating matrix (S/m).",
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PercolationRequest(FillerMatrixParams):
    """Request body for the power-law percolation conductivity model."""

    t: float = Field(
        2.0,
        gt=0.0,
        description="Critical exponent controlling the sharpness of the percolation transition. "
                    "Universal 3-D value is 2.0.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filler_fraction": 0.05,
                "percolation_threshold": 0.02,
                "sigma_filler": 1e6,
                "sigma_matrix": 1e-3,
                "t": 2.0,
            }
        }


class GEMRequest(FillerMatrixParams):
    """Request body for the McLachlan GEM equation."""

    t: float = Field(
        2.0,
        gt=0.0,
        description="Critical exponent for the percolation transition (~2.0 in 3-D).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filler_fraction": 0.05,
                "percolation_threshold": 0.02,
                "sigma_filler": 1e6,
                "sigma_matrix": 1e-3,
                "t": 2.0,
            }
        }


class ThresholdEstimateRequest(BaseModel):
    """Request body to estimate the percolation threshold from filler geometry."""

    filler_type: Literal["rod", "disk"] = Field(
        ...,
        description='Geometry of the filler particle. "rod" for cylinders (e.g. CNTs); '
                    '"disk" for platelets (e.g. graphene, MXene).',
    )

    # Rod-specific dimensions (required when filler_type == "rod")
    length_um: Optional[float] = Field(
        None,
        gt=0.0,
        description="Length of the cylindrical filler particle (micrometres). Required for rods.",
    )
    diameter_um: Optional[float] = Field(
        None,
        gt=0.0,
        description="Diameter of the cylindrical filler particle (micrometres). Required for rods.",
    )

    # Disk-specific dimensions (required when filler_type == "disk")
    radius_um: Optional[float] = Field(
        None,
        gt=0.0,
        description="Radius of the disk-shaped filler (micrometres). Required for disks.",
    )
    thickness_um: Optional[float] = Field(
        None,
        gt=0.0,
        description="Thickness of the disk-shaped filler (micrometres). Required for disks.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filler_type": "rod",
                "length_um": 10.0,
                "diameter_um": 0.02,
            }
        }


class AllModelsRequest(BaseModel):
    """Request body to evaluate all conductivity models and compare results."""

    filler_fraction: float = Field(..., ge=0.0, le=1.0)
    percolation_threshold: float = Field(..., gt=0.0, lt=1.0)
    sigma_filler: float = Field(..., ge=0.0)
    sigma_matrix: float = Field(..., ge=0.0)
    t: float = Field(2.0, gt=0.0, description="Percolation critical exponent.")
    depolarization: float = Field(
        1.0 / 3.0,
        ge=0.0,
        le=1.0,
        description="Maxwell-Garnett depolarization factor L. 1/3 for spheres.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filler_fraction": 0.05,
                "percolation_threshold": 0.02,
                "sigma_filler": 1e6,
                "sigma_matrix": 1e-3,
                "t": 2.0,
                "depolarization": 0.333,
            }
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/percolation")
async def compute_percolation(request: PercolationRequest) -> Dict[str, Any]:
    """Compute composite conductivity using the power-law percolation model.

    Above the percolation threshold the conductivity follows:

        sigma_eff = sigma_filler * ((f - f_c) / (1 - f_c))^t

    Below the threshold the composite conductivity equals sigma_matrix because
    no continuous conductive path exists through the filler network.
    """
    start_time = time.time()
    try:
        sigma_eff = percolation_conductivity(
            filler_fraction=request.filler_fraction,
            percolation_threshold=request.percolation_threshold,
            sigma_filler=request.sigma_filler,
            sigma_matrix=request.sigma_matrix,
            t=request.t,
        )
        above_threshold = request.filler_fraction > request.percolation_threshold
        return {
            "effective_conductivity_s_per_m": sigma_eff,
            "model": "percolation_power_law",
            "above_threshold": above_threshold,
            "filler_fraction": request.filler_fraction,
            "percolation_threshold": request.percolation_threshold,
            "critical_exponent_t": request.t,
            "execution_time_ms": (time.time() - start_time) * 1000,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Percolation calculation failed: {exc}",
        )


@router.post("/gem")
async def compute_gem(request: GEMRequest) -> Dict[str, Any]:
    """Compute composite conductivity using the McLachlan GEM equation.

    The General Effective Media (GEM) equation unifies percolation scaling and
    effective medium theory into a single implicit equation solved numerically.
    It is the most accurate model for nanofillers (CNTs, graphene, MXene)
    where percolation governs the electrical transport.
    """
    start_time = time.time()
    try:
        sigma_eff = mclachlan_gem(
            f_filler=request.filler_fraction,
            sigma_matrix=request.sigma_matrix,
            sigma_filler=request.sigma_filler,
            f_c=request.percolation_threshold,
            t=request.t,
        )
        return {
            "effective_conductivity_s_per_m": sigma_eff,
            "model": "mclachlan_gem",
            "filler_fraction": request.filler_fraction,
            "percolation_threshold": request.percolation_threshold,
            "critical_exponent_t": request.t,
            "execution_time_ms": (time.time() - start_time) * 1000,
        }
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"GEM calculation failed: {exc}",
        )


@router.post("/threshold-estimate")
async def estimate_threshold(request: ThresholdEstimateRequest) -> Dict[str, Any]:
    """Estimate the percolation threshold from filler particle geometry.

    Uses excluded-volume arguments:
    - Rods (e.g. CNTs):        f_c ~ 0.7 / AR   where AR = length / diameter
    - Disks (e.g. graphene):   f_c ~ 0.5 / AR   where AR = radius / thickness

    Dimensions are accepted in micrometres and converted to metres internally.
    """
    start_time = time.time()
    try:
        if request.filler_type == "rod":
            if request.length_um is None or request.diameter_um is None:
                raise ValueError(
                    'Both "length_um" and "diameter_um" are required for filler_type "rod".'
                )
            length_m = request.length_um * 1e-6
            diameter_m = request.diameter_um * 1e-6
            threshold = percolation_threshold_rods(length_m, diameter_m)
            aspect_ratio = length_m / diameter_m
            geometry_info = {
                "length_um": request.length_um,
                "diameter_um": request.diameter_um,
                "aspect_ratio": aspect_ratio,
            }
        else:  # "disk"
            if request.radius_um is None or request.thickness_um is None:
                raise ValueError(
                    'Both "radius_um" and "thickness_um" are required for filler_type "disk".'
                )
            radius_m = request.radius_um * 1e-6
            thickness_m = request.thickness_um * 1e-6
            threshold = percolation_threshold_disks(radius_m, thickness_m)
            aspect_ratio = radius_m / thickness_m
            geometry_info = {
                "radius_um": request.radius_um,
                "thickness_um": request.thickness_um,
                "aspect_ratio": aspect_ratio,
            }

        return {
            "estimated_percolation_threshold": threshold,
            "filler_type": request.filler_type,
            "geometry": geometry_info,
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Threshold estimation failed: {exc}",
        )


@router.post("/all-models")
async def compare_all_models(request: AllModelsRequest) -> Dict[str, Any]:
    """Evaluate all conductivity models and return their predictions for comparison.

    Runs the following models in parallel and returns all results:
    - Percolation power-law (above/below threshold behaviour)
    - McLachlan GEM (unified percolation + EMT)
    - Maxwell-Garnett EMT (dilute inclusions regime)
    - Bruggeman symmetric EMT (valid across full composition range)
    - Hashin-Shtrikman bounds (rigorous lower and upper bounds)

    Use this endpoint to compare model predictions and assess the sensitivity
    of the conductivity estimate to the choice of mixing rule.
    """
    start_time = time.time()
    errors: Dict[str, str] = {}
    results: Dict[str, Any] = {}

    # Percolation power-law
    try:
        results["percolation"] = percolation_conductivity(
            filler_fraction=request.filler_fraction,
            percolation_threshold=request.percolation_threshold,
            sigma_filler=request.sigma_filler,
            sigma_matrix=request.sigma_matrix,
            t=request.t,
        )
    except Exception as exc:
        errors["percolation"] = str(exc)

    # McLachlan GEM
    try:
        results["gem"] = mclachlan_gem(
            f_filler=request.filler_fraction,
            sigma_matrix=request.sigma_matrix,
            sigma_filler=request.sigma_filler,
            f_c=request.percolation_threshold,
            t=request.t,
        )
    except Exception as exc:
        errors["gem"] = str(exc)

    # Maxwell-Garnett (filler = inclusion, matrix = host)
    try:
        results["maxwell_garnett"] = maxwell_garnett(
            sigma_host=request.sigma_matrix,
            sigma_inclusion=request.sigma_filler,
            volume_fraction=request.filler_fraction,
            depolarization=request.depolarization,
        )
    except Exception as exc:
        errors["maxwell_garnett"] = str(exc)

    # Bruggeman symmetric EMT (phase 1 = filler, phase 2 = matrix)
    try:
        results["bruggeman"] = bruggeman_emt(
            sigma_1=request.sigma_filler,
            sigma_2=request.sigma_matrix,
            f_1=request.filler_fraction,
        )
    except Exception as exc:
        errors["bruggeman"] = str(exc)

    # Hashin-Shtrikman bounds
    try:
        hs_lower, hs_upper = hashin_shtrikman_bounds(
            sigma_1=request.sigma_filler,
            sigma_2=request.sigma_matrix,
            f_1=request.filler_fraction,
        )
        results["hashin_shtrikman_lower"] = hs_lower
        results["hashin_shtrikman_upper"] = hs_upper
    except Exception as exc:
        errors["hashin_shtrikman"] = str(exc)

    if not results and errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"All models failed. Errors: {errors}",
        )

    return {
        "effective_conductivities_s_per_m": results,
        "model_errors": errors,
        "filler_fraction": request.filler_fraction,
        "percolation_threshold": request.percolation_threshold,
        "sigma_filler_s_per_m": request.sigma_filler,
        "sigma_matrix_s_per_m": request.sigma_matrix,
        "above_percolation_threshold": request.filler_fraction > request.percolation_threshold,
        "execution_time_ms": (time.time() - start_time) * 1000,
    }
