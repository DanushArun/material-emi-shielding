"""Analysis endpoints - sweeps, optimizations, and advanced calculations.

All endpoints accept elemental composition and calculate composite properties
internally before running the physics engine.
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import time
import numpy as np

from src.physics.emi_calculations import EMICalculator
from src.physics.advanced_microstructure import ProcessingParams
from backend.api.v1.schemas.physics import (
    FrequencySweepRequest,
    ThicknessSweepRequest,
    GrainSizeSweepRequest,
    CoolingRateSweepRequest,
    ThicknessOptimizationRequest,
)
from backend.api.v1.routes._helpers import calculate_composite_properties

router = APIRouter()
calculator = EMICalculator()


@router.post("/frequency-sweep")
async def frequency_sweep(request: FrequencySweepRequest) -> Dict[str, Any]:
    """Calculate shielding effectiveness across a frequency range.

    Returns arrays of frequency vs SE components for Plotly charting.
    """
    start = time.time()
    try:
        props = calculate_composite_properties(request.composition)
        conductivity = props['conductivity']

        if request.grain_size_um is not None:
            conductivity = calculator.calculate_grain_size_effect(
                conductivity, request.grain_size_um * 1e-6
            )

        result = calculator.frequency_sweep(
            conductivity=conductivity,
            relative_permeability=props['permeability'],
            relative_permittivity=props['permittivity'],
            thickness=request.thickness_mm * 1e-3,
            freq_start=request.freq_start_mhz * 1e6,
            freq_end=request.freq_end_mhz * 1e6,
            num_points=request.num_points,
        )

        return {
            "frequencies_mhz": (result['frequencies'] / 1e6).tolist(),
            "total_se_db": result['total_ses'].tolist(),
            "reflection_loss_db": result['reflection_losses'].tolist(),
            "absorption_loss_db": result['absorption_losses'].tolist(),
            "skin_depth_um": (result['skin_depths'] * 1e6).tolist(),
            "execution_time_ms": (time.time() - start) * 1000,
        }
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/thickness-sweep")
async def thickness_sweep(request: ThicknessSweepRequest) -> Dict[str, Any]:
    """Calculate shielding effectiveness across a thickness range."""
    start = time.time()
    try:
        props = calculate_composite_properties(request.composition)
        conductivity = props['conductivity']

        if request.grain_size_um is not None:
            conductivity = calculator.calculate_grain_size_effect(
                conductivity, request.grain_size_um * 1e-6
            )

        result = calculator.thickness_sweep(
            conductivity=conductivity,
            relative_permeability=props['permeability'],
            relative_permittivity=props['permittivity'],
            frequency=request.frequency_mhz * 1e6,
            thickness_start=request.thickness_start_mm * 1e-3,
            thickness_end=request.thickness_end_mm * 1e-3,
            num_points=request.num_points,
        )

        return {
            "thicknesses_mm": (result['thicknesses'] * 1000).tolist(),
            "total_se_db": result['total_ses'].tolist(),
            "reflection_loss_db": result['reflection_losses'].tolist(),
            "absorption_loss_db": result['absorption_losses'].tolist(),
            "skin_depth_um": (result['skin_depths'] * 1e6).tolist(),
            "execution_time_ms": (time.time() - start) * 1000,
        }
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/grain-size-sweep")
async def grain_size_sweep(request: GrainSizeSweepRequest) -> Dict[str, Any]:
    """Calculate shielding effectiveness across a grain size range."""
    start = time.time()
    try:
        props = calculate_composite_properties(request.composition)

        result = calculator.grain_size_sweep(
            conductivity=props['conductivity'],
            relative_permeability=props['permeability'],
            relative_permittivity=props['permittivity'],
            thickness=request.thickness_mm * 1e-3,
            frequency=request.frequency_mhz * 1e6,
            grain_start=request.grain_start_um * 1e-6,
            grain_end=request.grain_end_um * 1e-6,
            num_points=request.num_points,
        )

        return {
            "grain_sizes_um": (result['grain_sizes'] * 1e6).tolist(),
            "total_se_db": result['total_ses'].tolist(),
            "reflection_loss_db": result['reflection_losses'].tolist(),
            "absorption_loss_db": result['absorption_losses'].tolist(),
            "effective_conductivities": result['effective_conductivities'].tolist(),
            "execution_time_ms": (time.time() - start) * 1000,
        }
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cooling-rate-sweep")
async def cooling_rate_sweep(request: CoolingRateSweepRequest) -> Dict[str, Any]:
    """Calculate EMI performance across a range of cooling rates.

    Links processing conditions -> microstructure -> EMI performance.
    """
    start = time.time()
    try:
        props = calculate_composite_properties(request.composition)
        cooling_rates = np.logspace(
            np.log10(request.cooling_rate_min),
            np.log10(request.cooling_rate_max),
            request.num_points,
        )

        result = calculator.cooling_rate_sweep(
            composition=request.composition,
            thickness=request.thickness_mm * 1e-3,
            frequency=request.frequency_mhz * 1e6,
            base_conductivity=props['conductivity'],
            cooling_rates=cooling_rates,
        )

        return {
            "cooling_rates": result['cooling_rates'].tolist(),
            "total_se_db": result['total_ses'].tolist(),
            "grain_sizes_um": (result['grain_sizes'] * 1e6).tolist() if hasattr(result['grain_sizes'], 'tolist') else result['grain_sizes'],
            "effective_conductivities": result['effective_conductivities'].tolist(),
            "execution_time_ms": (time.time() - start) * 1000,
        }
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/optimize-thickness")
async def optimize_thickness(request: ThicknessOptimizationRequest) -> Dict[str, Any]:
    """Find the minimum thickness to achieve a target shielding effectiveness."""
    start = time.time()
    try:
        props = calculate_composite_properties(request.composition)
        conductivity = props['conductivity']

        if request.grain_size_um is not None:
            conductivity = calculator.calculate_grain_size_effect(
                conductivity, request.grain_size_um * 1e-6
            )

        result = calculator.optimize_thickness(
            conductivity=conductivity,
            relative_permeability=props['permeability'],
            relative_permittivity=props['permittivity'],
            frequency=request.frequency_mhz * 1e6,
            target_se=request.target_se_db,
            max_thickness=request.thickness_max_mm * 1e-3,
        )

        return {
            "optimal_thickness_mm": result['optimal_thickness'] * 1000,
            "achieved_se_db": result['achieved_se'],
            "reflection_loss_db": result['reflection_loss'],
            "absorption_loss_db": result['absorption_loss'],
            "skin_depths_ratio": result['skin_depths'],
            "execution_time_ms": (time.time() - start) * 1000,
        }
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
