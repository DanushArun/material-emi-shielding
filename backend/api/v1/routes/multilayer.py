"""Multilayer Transfer Matrix Method (TMM) endpoints.

Wraps src.physics.multilayer.MultilayerShield for N-layer EMI shield stacks.
Each layer's composite EM properties are derived from an elemental composition
dict using the same weighted-average logic as the single-layer physics routes.

Endpoints
---------
POST /multilayer/calculate        Single-frequency SE for a layer stack.
POST /multilayer/frequency-sweep  SE vs frequency sweep for a layer stack.
POST /multilayer/optimize         Scale all layer thicknesses to hit a target SE.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import time

from src.physics.multilayer import MultilayerShield, ShieldLayer
from backend.api.v1.routes._helpers import calculate_composite_properties

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class LayerSpec(BaseModel):
    """Specification for a single layer in the multilayer stack."""

    composition: Dict[str, float] = Field(
        ...,
        description="Element symbols to weight percentages (must sum to ~100).",
    )
    thickness_mm: float = Field(
        ...,
        gt=0.0,
        description="Layer thickness in millimetres (converted to metres internally).",
    )
    name: str = Field(
        "",
        description="Optional human-readable label for this layer.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "composition": {"Cu": 100},
                "thickness_mm": 0.5,
                "name": "copper",
            }
        }


class MultilayerCalculateRequest(BaseModel):
    """Request body for the single-frequency multilayer calculation."""

    layers: List[LayerSpec] = Field(
        ...,
        min_length=1,
        description="Ordered list of shield layers (first layer faces the incident wave).",
    )
    frequency_mhz: float = Field(
        ...,
        gt=0.001,
        lt=100_000,
        description="Frequency at which SE is evaluated (MHz).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "layers": [
                    {"composition": {"Cu": 100}, "thickness_mm": 0.5, "name": "copper"},
                    {"composition": {"Fe": 100}, "thickness_mm": 1.0, "name": "iron"},
                ],
                "frequency_mhz": 1000,
            }
        }


class MultilayerSweepRequest(BaseModel):
    """Request body for the multilayer frequency sweep."""

    layers: List[LayerSpec] = Field(..., min_length=1)
    freq_start_mhz: float = Field(1.0, gt=0.001, description="Start frequency (MHz).")
    freq_end_mhz: float = Field(10_000.0, gt=0.001, description="End frequency (MHz).")
    num_points: int = Field(100, gt=1, le=1000, description="Number of log-spaced frequency points.")

    class Config:
        json_schema_extra = {
            "example": {
                "layers": [
                    {"composition": {"Cu": 100}, "thickness_mm": 0.5, "name": "copper"},
                    {"composition": {"Ni": 100}, "thickness_mm": 0.1, "name": "nickel"},
                ],
                "freq_start_mhz": 1,
                "freq_end_mhz": 10000,
                "num_points": 200,
            }
        }


class MultilayerOptimizeRequest(BaseModel):
    """Request body for multilayer thickness optimisation."""

    layers: List[LayerSpec] = Field(..., min_length=1)
    frequency_mhz: float = Field(..., gt=0.001, lt=100_000)
    target_se_db: float = Field(..., gt=0, description="Target shielding effectiveness (dB).")
    max_total_thickness_mm: float = Field(
        10.0,
        gt=0,
        description="Maximum allowed total stack thickness (mm). Reserved for future use.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "layers": [
                    {"composition": {"Cu": 100}, "thickness_mm": 0.5, "name": "copper"},
                    {"composition": {"Fe": 100}, "thickness_mm": 1.0, "name": "iron"},
                ],
                "frequency_mhz": 1000,
                "target_se_db": 80,
                "max_total_thickness_mm": 10,
            }
        }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_shield(layers: List[LayerSpec]) -> MultilayerShield:
    """Convert a list of LayerSpec objects into a configured MultilayerShield.

    Raises:
        ValueError: If any layer contains an unknown element.
    """
    shield = MultilayerShield()
    for spec in layers:
        props = calculate_composite_properties(spec.composition)
        shield.add_layer(ShieldLayer(
            conductivity=props['conductivity'],
            relative_permeability=props['permeability'],
            relative_permittivity=props['permittivity'],
            thickness=spec.thickness_mm * 1e-3,
            name=spec.name,
        ))
    return shield


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/calculate")
async def calculate_multilayer(request: MultilayerCalculateRequest) -> Dict[str, Any]:
    """Calculate shielding effectiveness for an N-layer stack at a single frequency.

    Builds a MultilayerShield from the provided layers, derives composite EM
    properties for each layer from its elemental composition, then runs the
    Transfer Matrix Method (TMM) cascade to extract total SE, reflection loss,
    and absorption loss.

    Returns total SE in dB plus transmission and reflection coefficients.
    """
    start_time = time.time()
    try:
        shield = _build_shield(request.layers)
        frequency_hz = request.frequency_mhz * 1e6
        result = shield.calculate_se(frequency_hz)

        layer_summary = []
        for i, spec in enumerate(request.layers):
            props = calculate_composite_properties(spec.composition)
            layer_summary.append({
                "index": i,
                "name": spec.name or f"layer_{i}",
                "thickness_mm": spec.thickness_mm,
                "conductivity_s_per_m": props['conductivity'],
                "relative_permeability": props['permeability'],
                "relative_permittivity": props['permittivity'],
            })

        return {
            "total_se_db": result['total_se'],
            "reflection_loss_db": result['reflection_loss'],
            "absorption_loss_db": result['absorption_loss'],
            "transmission_coefficient": result['transmission_coefficient'],
            "reflection_coefficient": result['reflection_coefficient'],
            "frequency_mhz": request.frequency_mhz,
            "num_layers": len(request.layers),
            "total_thickness_mm": sum(s.thickness_mm for s in request.layers),
            "layers": layer_summary,
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Multilayer calculation failed: {exc}",
        )


@router.post("/frequency-sweep")
async def multilayer_frequency_sweep(request: MultilayerSweepRequest) -> Dict[str, Any]:
    """Sweep shielding effectiveness across a frequency range for an N-layer stack.

    Runs the TMM at each log-spaced frequency point and returns arrays suitable
    for client-side plotting.  Frequency is returned in MHz; SE components in dB.
    """
    start_time = time.time()
    try:
        if request.freq_end_mhz <= request.freq_start_mhz:
            raise ValueError("freq_end_mhz must be greater than freq_start_mhz.")

        shield = _build_shield(request.layers)
        result = shield.frequency_sweep(
            freq_start=request.freq_start_mhz * 1e6,
            freq_end=request.freq_end_mhz * 1e6,
            num_points=request.num_points,
        )

        return {
            "frequencies_mhz": (result['frequencies'] / 1e6).tolist(),
            "total_se_db": result['total_ses'].tolist(),
            "reflection_loss_db": result['reflection_losses'].tolist(),
            "absorption_loss_db": result['absorption_losses'].tolist(),
            "num_layers": len(request.layers),
            "total_thickness_mm": sum(s.thickness_mm for s in request.layers),
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Frequency sweep failed: {exc}",
        )


@router.post("/optimize")
async def optimize_multilayer_thicknesses(request: MultilayerOptimizeRequest) -> Dict[str, Any]:
    """Optimise layer thicknesses to reach a target shielding effectiveness.

    Uses a bounded scalar minimiser (scipy minimize_scalar) to find the uniform
    scale factor applied to all layer thicknesses that brings the total SE
    closest to the requested target.  The optimised per-layer thicknesses and
    the achieved SE are returned.

    The layers in the response reflect the optimised (scaled) thicknesses.
    Original thicknesses are echoed back for comparison.
    """
    start_time = time.time()
    try:
        shield = _build_shield(request.layers)
        frequency_hz = request.frequency_mhz * 1e6
        max_total_m = request.max_total_thickness_mm * 1e-3

        result = shield.optimize_layer_thicknesses(
            frequency=frequency_hz,
            target_se=request.target_se_db,
            max_total_thickness=max_total_m,
        )

        # Map optimised thicknesses (metres) back to mm, paired with layer names.
        optimised_layers = []
        for i, (spec, opt_t_m) in enumerate(
            zip(request.layers, result['optimal_thicknesses'])
        ):
            optimised_layers.append({
                "index": i,
                "name": spec.name or f"layer_{i}",
                "original_thickness_mm": spec.thickness_mm,
                "optimised_thickness_mm": opt_t_m * 1e3,
            })

        return {
            "achieved_se_db": result['achieved_se'],
            "target_se_db": result['target_se'],
            "scale_factor": result['scale_factor'],
            "total_thickness_mm": result['total_thickness'] * 1e3,
            "frequency_mhz": request.frequency_mhz,
            "optimised_layers": optimised_layers,
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Thickness optimisation failed: {exc}",
        )
