"""Advanced physics endpoints: temperature/frequency-dependent material properties
and Monte Carlo uncertainty quantification (UQ).

Wraps:
    src.physics.material_models  -- temperature and frequency dependent properties
    src.physics.uncertainty      -- Monte Carlo SE, Sobol sensitivity indices

All internal SI units are used by the physics modules; this layer performs
unit conversion at the boundary (MHz -> Hz, mm -> m, um -> m, K is native).

Endpoints
---------
POST /advanced/material-at-conditions  Conductivity and complex mu_r at T and f.
POST /advanced/monte-carlo             Monte Carlo SE uncertainty propagation.
POST /advanced/sobol-sensitivity       Sobol first-order and total-order indices.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import time

from src.physics.material_models import (
    get_material_properties_at_conditions,
    TCR_DATA,
    MAGNETIC_DATA,
)
from src.physics.uncertainty import (
    monte_carlo_se,
    sobol_sensitivity,
    UncertaintySpec,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MaterialAtConditionsRequest(BaseModel):
    """Request body for temperature- and frequency-dependent material properties."""

    element: str = Field(
        ...,
        description=(
            "Material key from the TCR database. "
            "Supported: Cu, Al, Ni, Fe, Ag, Au, steel_1018, ss_304, mu_metal, permalloy."
        ),
    )
    frequency_hz: float = Field(
        ...,
        gt=0.0,
        description="Operating frequency in Hz.",
    )
    temperature_k: float = Field(
        293.15,
        gt=0.0,
        description="Operating temperature in Kelvin. Default 293.15 K (20 C).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "element": "Fe",
                "frequency_hz": 1e9,
                "temperature_k": 500.0,
            }
        }


class UncertaintySpecRequest(BaseModel):
    """Optional per-parameter coefficient-of-variation overrides."""

    conductivity_cv: float = Field(0.05, ge=0.0, le=1.0, description="CV for conductivity (default 5%).")
    thickness_cv: float = Field(0.02, ge=0.0, le=1.0, description="CV for thickness (default 2%).")
    grain_size_cv: float = Field(0.30, ge=0.0, le=1.0, description="CV for grain size, log-normal (default 30%).")
    permeability_cv: float = Field(0.10, ge=0.0, le=1.0, description="CV for permeability (default 10%).")
    frequency_cv: float = Field(0.001, ge=0.0, le=1.0, description="CV for frequency measurement precision (default 0.1%).")


class MonteCarloRequest(BaseModel):
    """Request body for Monte Carlo SE uncertainty propagation."""

    conductivity_s_per_m: float = Field(
        ..., gt=0.0, description="Nominal electrical conductivity (S/m)."
    )
    relative_permeability: float = Field(
        1.0, ge=0.999, description="Nominal relative permeability."
    )
    relative_permittivity: float = Field(
        1.0, ge=1.0, description="Nominal relative permittivity."
    )
    thickness_mm: float = Field(
        ..., gt=0.0, description="Nominal shield thickness (mm)."
    )
    frequency_mhz: float = Field(
        ..., gt=0.001, lt=100_000, description="Operating frequency (MHz)."
    )
    grain_size_um: Optional[float] = Field(
        None, gt=0.0, description="Nominal grain size (micrometres). Omit to ignore grain boundary effects."
    )
    n_samples: int = Field(
        1000, ge=10, le=50_000, description="Number of Monte Carlo draws."
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility."
    )
    uncertainty: Optional[UncertaintySpecRequest] = Field(
        None,
        description="Per-parameter CV overrides. Uses physics-module defaults when omitted.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "conductivity_s_per_m": 5.96e7,
                "relative_permeability": 1.0,
                "relative_permittivity": 1.0,
                "thickness_mm": 1.0,
                "frequency_mhz": 1000,
                "grain_size_um": 50.0,
                "n_samples": 1000,
                "seed": 42,
                "uncertainty": {
                    "conductivity_cv": 0.05,
                    "thickness_cv": 0.02,
                    "grain_size_cv": 0.30,
                    "permeability_cv": 0.10,
                    "frequency_cv": 0.001,
                },
            }
        }


class SobolRequest(BaseModel):
    """Request body for Sobol sensitivity index calculation."""

    conductivity_s_per_m: float = Field(..., gt=0.0)
    relative_permeability: float = Field(1.0, ge=0.999)
    relative_permittivity: float = Field(1.0, ge=1.0)
    thickness_mm: float = Field(..., gt=0.0)
    frequency_mhz: float = Field(..., gt=0.001, lt=100_000)
    grain_size_um: Optional[float] = Field(
        None, gt=0.0, description="Include grain size as a sensitivity dimension when provided."
    )
    n_samples: int = Field(
        1024,
        ge=64,
        le=16_384,
        description="Base sample count N; total evaluations = N * (2D + 2). Must be a power of 2.",
    )
    seed: Optional[int] = Field(None, description="Random seed for the Sobol sampler.")

    class Config:
        json_schema_extra = {
            "example": {
                "conductivity_s_per_m": 5.96e7,
                "relative_permeability": 1.0,
                "relative_permittivity": 1.0,
                "thickness_mm": 1.0,
                "frequency_mhz": 1000,
                "grain_size_um": 50.0,
                "n_samples": 1024,
                "seed": 0,
            }
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_uncertainty_spec(req: Optional[UncertaintySpecRequest]) -> UncertaintySpec:
    """Convert the optional request CV overrides to a UncertaintySpec dataclass."""
    if req is None:
        return UncertaintySpec()
    return UncertaintySpec(
        conductivity_cv=req.conductivity_cv,
        thickness_cv=req.thickness_cv,
        grain_size_cv=req.grain_size_cv,
        permeability_cv=req.permeability_cv,
        frequency_cv=req.frequency_cv,
    )


def _serialise_complex(z: complex) -> Dict[str, float]:
    """Convert a complex number to a JSON-serialisable dict with real/imag keys."""
    return {"real": float(z.real), "imaginary": float(z.imag)}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/material-at-conditions")
async def material_at_conditions(request: MaterialAtConditionsRequest) -> Dict[str, Any]:
    """Return conductivity and complex permeability at specified temperature and frequency.

    Combines three physics sub-models:
    1. Temperature-dependent conductivity (linear TCR model).
    2. Temperature-dependent static permeability (power-law falloff to Curie point).
    3. Frequency-dependent complex permeability (Debye/Snoek relaxation model).

    For non-magnetic materials (not in the magnetic database) the complex
    permeability is returned as 1.0 + 0j and f_resonance_hz is null.

    Supported material keys: Cu, Al, Ni, Fe, Ag, Au, steel_1018, ss_304,
    mu_metal, permalloy.
    """
    start_time = time.time()
    try:
        props = get_material_properties_at_conditions(
            element=request.element,
            frequency=request.frequency_hz,
            temperature=request.temperature_k,
        )

        is_magnetic = request.element in MAGNETIC_DATA
        mu_r_complex: complex = props['mu_r_complex']

        return {
            "element": request.element,
            "temperature_k": request.temperature_k,
            "frequency_hz": request.frequency_hz,
            "conductivity_s_per_m": props['conductivity'],
            "mu_r_static": props['mu_r_static'],
            "mu_r_complex": _serialise_complex(mu_r_complex),
            "f_resonance_hz": props['f_resonance'],
            "is_magnetic": is_magnetic,
            "reference_conductivity_s_per_m": TCR_DATA[request.element]['sigma_ref'],
            "reference_temperature_k": TCR_DATA[request.element]['T_ref'],
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    except KeyError as exc:
        available = list(TCR_DATA.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown element '{request.element}'. "
                f"Available materials: {available}. "
                f"Original error: {exc}"
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Material property calculation failed: {exc}",
        )


@router.post("/monte-carlo")
async def run_monte_carlo(request: MonteCarloRequest) -> Dict[str, Any]:
    """Run Monte Carlo uncertainty propagation for the SE prediction.

    Each input parameter is sampled independently around its nominal value:
    - conductivity, thickness, permeability, frequency: normal distribution
    - grain_size: log-normal distribution (right-skewed manufacturing spread)

    The coefficient of variation (CV) for each parameter can be controlled via
    the optional uncertainty field; otherwise physics-module defaults are used
    (conductivity 5%, thickness 2%, grain size 30%, permeability 10%, frequency 0.1%).

    Returns the mean, standard deviation, 95% confidence interval (2.5th and
    97.5th percentiles), and the deterministic SE at nominal parameter values.
    The full SE distribution is not returned; use n_samples for resolution
    control.
    """
    start_time = time.time()
    try:
        unc_spec = _build_uncertainty_spec(request.uncertainty)
        grain_size_m = (
            request.grain_size_um * 1e-6 if request.grain_size_um is not None else None
        )

        mc_result = monte_carlo_se(
            conductivity=request.conductivity_s_per_m,
            permeability=request.relative_permeability,
            permittivity=request.relative_permittivity,
            thickness=request.thickness_mm * 1e-3,
            frequency=request.frequency_mhz * 1e6,
            grain_size=grain_size_m,
            uncertainty=unc_spec,
            n_samples=request.n_samples,
            seed=request.seed,
        )

        # se_distribution is a numpy array; convert for JSON serialisation.
        se_distribution: List[float] = mc_result["se_distribution"].tolist()

        return {
            "se_mean_db": mc_result["se_mean"],
            "se_std_db": mc_result["se_std"],
            "se_ci_lower_db": mc_result["se_ci_lower"],
            "se_ci_upper_db": mc_result["se_ci_upper"],
            "deterministic_se_db": mc_result["deterministic_se"],
            "n_samples": mc_result["n_samples"],
            "se_distribution_db": se_distribution,
            "uncertainty_spec": {
                "conductivity_cv": unc_spec.conductivity_cv,
                "thickness_cv": unc_spec.thickness_cv,
                "grain_size_cv": unc_spec.grain_size_cv,
                "permeability_cv": unc_spec.permeability_cv,
                "frequency_cv": unc_spec.frequency_cv,
            },
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Monte Carlo simulation failed: {exc}",
        )


@router.post("/sobol-sensitivity")
async def run_sobol_sensitivity(request: SobolRequest) -> Dict[str, Any]:
    """Compute Sobol first-order and total-order sensitivity indices.

    Uses the Saltelli (2002) estimator with quasi-random Sobol sequences.
    Each sensitivity index quantifies what fraction of the total SE output
    variance is attributable to uncertainty in that parameter:

    - S1 (first-order): variance contribution from that parameter alone.
    - ST (total-order): variance contribution including all interactions with
      other parameters.

    Parameters with large ST - S1 gaps have strong interactions with others.

    n_samples must be a power of two; total function evaluations = n * (2D + 2)
    where D is the number of input dimensions (4 without grain size, 5 with).
    """
    start_time = time.time()
    try:
        grain_size_m = (
            request.grain_size_um * 1e-6 if request.grain_size_um is not None else None
        )

        sobol_result = sobol_sensitivity(
            conductivity=request.conductivity_s_per_m,
            permeability=request.relative_permeability,
            permittivity=request.relative_permittivity,
            thickness=request.thickness_mm * 1e-3,
            frequency=request.frequency_mhz * 1e6,
            grain_size=grain_size_m,
            n_samples=request.n_samples,
            seed=request.seed,
        )

        # Identify the dominant parameter by total-order index.
        st_dict: Dict[str, float] = sobol_result["ST"]
        dominant_param = max(st_dict, key=lambda k: st_dict[k]) if st_dict else None

        return {
            "parameters": sobol_result["parameters"],
            "S1": sobol_result["S1"],
            "ST": sobol_result["ST"],
            "dominant_parameter": dominant_param,
            "n_samples": request.n_samples,
            "n_evaluations": request.n_samples * (2 * len(sobol_result["parameters"]) + 2),
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sobol sensitivity analysis failed: {exc}",
        )
