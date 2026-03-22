"""Enclosure shielding effectiveness endpoints.

Wraps src.physics.aperture for combined enclosure SE analysis, including
bulk material shielding, aperture leakage (circular holes, slots, waveguide
vents), cavity resonance computation, and frequency sweeps.

Endpoints
---------
POST /enclosure/analyze              Combined enclosure SE at a single frequency.
POST /enclosure/cavity-resonances    Rectangular cavity resonant modes.
POST /enclosure/frequency-sweep      Bulk + aperture + combined SE vs frequency.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np

from src.physics.emi_calculations import EMICalculator
from src.physics.aperture import (
    aperture_se_circular,
    aperture_se_slot,
    aperture_se_array,
    waveguide_below_cutoff_se,
    cavity_resonance_frequencies,
    combined_enclosure_se,
)
from backend.api.v1.routes._helpers import calculate_composite_properties

router = APIRouter()
calculator = EMICalculator()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CircularAperture(BaseModel):
    radius_mm: float = Field(..., gt=0, description="Aperture radius in mm.")
    count: int = Field(1, ge=1, description="Number of identical apertures.")


class SlotAperture(BaseModel):
    length_mm: float = Field(..., gt=0, description="Slot length in mm.")
    count: int = Field(1, ge=1, description="Number of identical slots.")


class WaveguideVent(BaseModel):
    diameter_mm: float = Field(..., gt=0, description="Tube inner diameter in mm.")
    depth_mm: float = Field(..., gt=0, description="Tube depth (length) in mm.")
    count: int = Field(1, ge=1, description="Number of identical vents.")


class EnclosureAnalyzeRequest(BaseModel):
    composition: Dict[str, float]
    wall_thickness_mm: float = Field(..., gt=0, description="Wall thickness in mm.")
    frequency_mhz: float = Field(..., gt=0, description="Frequency in MHz.")
    circular_apertures: List[CircularAperture] = Field(
        default_factory=list,
        description="List of circular aperture groups.",
    )
    slot_apertures: List[SlotAperture] = Field(
        default_factory=list,
        description="List of slot aperture groups.",
    )
    waveguide_vents: List[WaveguideVent] = Field(
        default_factory=list,
        description="List of waveguide vent groups.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "composition": {"Cu": 100.0},
                "wall_thickness_mm": 1.0,
                "frequency_mhz": 1000.0,
                "circular_apertures": [{"radius_mm": 5.0, "count": 4}],
                "slot_apertures": [{"length_mm": 50.0, "count": 2}],
                "waveguide_vents": [
                    {"diameter_mm": 3.0, "depth_mm": 10.0, "count": 20}
                ],
            }
        }


class ApertureDetail(BaseModel):
    type: str
    se_single_db: float
    count: int
    se_array_db: float


class EnclosureAnalyzeResponse(BaseModel):
    bulk_se_db: float
    aperture_se_details: List[ApertureDetail]
    combined_se_db: float
    dominant_leakage_path: str
    frequency_mhz: float


class CavityResonanceRequest(BaseModel):
    length_mm: float = Field(..., gt=0, description="Cavity length in mm.")
    width_mm: float = Field(..., gt=0, description="Cavity width in mm.")
    height_mm: float = Field(..., gt=0, description="Cavity height in mm.")
    max_modes: int = Field(5, ge=1, le=20, description="Max mode index to search.")

    class Config:
        json_schema_extra = {
            "example": {
                "length_mm": 300.0,
                "width_mm": 200.0,
                "height_mm": 100.0,
                "max_modes": 5,
            }
        }


class CavityResonanceResponse(BaseModel):
    modes: List[Dict]
    dimensions_mm: Dict[str, float]


class EnclosureSweepRequest(BaseModel):
    composition: Dict[str, float]
    wall_thickness_mm: float = Field(..., gt=0, description="Wall thickness in mm.")
    freq_start_mhz: float = Field(..., gt=0, description="Start frequency in MHz.")
    freq_end_mhz: float = Field(..., gt=0, description="End frequency in MHz.")
    num_points: int = Field(100, ge=2, le=10000, description="Number of frequency points.")
    circular_apertures: List[CircularAperture] = Field(default_factory=list)
    slot_apertures: List[SlotAperture] = Field(default_factory=list)
    waveguide_vents: List[WaveguideVent] = Field(default_factory=list)
    enclosure_length_mm: Optional[float] = Field(
        None, gt=0, description="Enclosure length in mm (for cavity resonance overlay)."
    )
    enclosure_width_mm: Optional[float] = Field(
        None, gt=0, description="Enclosure width in mm."
    )
    enclosure_height_mm: Optional[float] = Field(
        None, gt=0, description="Enclosure height in mm."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "composition": {"Al": 100.0},
                "wall_thickness_mm": 2.0,
                "freq_start_mhz": 100.0,
                "freq_end_mhz": 5000.0,
                "num_points": 200,
                "circular_apertures": [{"radius_mm": 5.0, "count": 10}],
                "slot_apertures": [],
                "waveguide_vents": [],
                "enclosure_length_mm": 300.0,
                "enclosure_width_mm": 200.0,
                "enclosure_height_mm": 100.0,
            }
        }


class EnclosureSweepResponse(BaseModel):
    frequencies_mhz: List[float]
    bulk_se_db: List[float]
    combined_se_db: List[float]
    worst_aperture_se_db: List[float]
    cavity_resonances: Optional[List[Dict]] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_apertures(
    frequency_hz: float,
    circular_apertures: List[CircularAperture],
    slot_apertures: List[SlotAperture],
    waveguide_vents: List[WaveguideVent],
) -> List[ApertureDetail]:
    """Evaluate SE for each aperture group and return detail records."""
    details: List[ApertureDetail] = []

    for circ in circular_apertures:
        radius_m = circ.radius_mm * 1e-3
        se_single = aperture_se_circular(frequency_hz, radius_m)
        se_arr = aperture_se_array(frequency_hz, se_single, circ.count)
        details.append(ApertureDetail(
            type="circular",
            se_single_db=se_single,
            count=circ.count,
            se_array_db=se_arr,
        ))

    for slot in slot_apertures:
        length_m = slot.length_mm * 1e-3
        se_single = aperture_se_slot(frequency_hz, length_m)
        se_arr = aperture_se_array(frequency_hz, se_single, slot.count)
        details.append(ApertureDetail(
            type="slot",
            se_single_db=se_single,
            count=slot.count,
            se_array_db=se_arr,
        ))

    for vent in waveguide_vents:
        diameter_m = vent.diameter_mm * 1e-3
        depth_m = vent.depth_mm * 1e-3
        se_single = waveguide_below_cutoff_se(frequency_hz, diameter_m, depth_m)
        se_arr = aperture_se_array(frequency_hz, se_single, vent.count)
        details.append(ApertureDetail(
            type="waveguide_vent",
            se_single_db=se_single,
            count=vent.count,
            se_array_db=se_arr,
        ))

    return details


def _dominant_leakage(bulk_se_db: float, details: List[ApertureDetail]) -> str:
    """Identify the dominant leakage path (lowest SE contribution)."""
    worst_label = "bulk_material"
    worst_se = bulk_se_db

    for i, d in enumerate(details):
        if d.se_array_db < worst_se:
            worst_se = d.se_array_db
            worst_label = f"{d.type}_{i}"

    return worst_label


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=EnclosureAnalyzeResponse)
async def analyze_enclosure(req: EnclosureAnalyzeRequest):
    """Compute combined enclosure SE at a single frequency.

    Calculates the bulk material SE, evaluates each aperture group,
    then combines all leakage paths using power-summation to produce
    the overall enclosure SE.
    """
    try:
        # Material properties
        props = calculate_composite_properties(req.composition)
        frequency_hz = req.frequency_mhz * 1e6
        thickness_m = req.wall_thickness_mm * 1e-3

        # Bulk material SE
        bulk_result = calculator.calculate_shielding_effectiveness(
            conductivity=props["conductivity"],
            relative_permeability=props["permeability"],
            relative_permittivity=props["permittivity"],
            thickness=thickness_m,
            frequency=frequency_hz,
        )
        bulk_se = float(bulk_result["total_se"])

        # Aperture SE details
        details = _evaluate_apertures(
            frequency_hz,
            req.circular_apertures,
            req.slot_apertures,
            req.waveguide_vents,
        )

        # Combine all leakage paths
        aperture_se_values = [d.se_array_db for d in details]
        combined_se = combined_enclosure_se(bulk_se, aperture_se_values)

        # Dominant leakage path
        dominant = _dominant_leakage(bulk_se, details)

        return EnclosureAnalyzeResponse(
            bulk_se_db=bulk_se,
            aperture_se_details=details,
            combined_se_db=combined_se,
            dominant_leakage_path=dominant,
            frequency_mhz=req.frequency_mhz,
        )

    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cavity-resonances", response_model=CavityResonanceResponse)
async def compute_cavity_resonances(req: CavityResonanceRequest):
    """Compute resonant frequencies for a rectangular enclosure.

    Returns mode indices, frequencies, and mode types (TE/TM) sorted
    by ascending frequency.
    """
    try:
        length_m = req.length_mm * 1e-3
        width_m = req.width_mm * 1e-3
        height_m = req.height_mm * 1e-3

        modes = cavity_resonance_frequencies(
            length_m=length_m,
            width_m=width_m,
            height_m=height_m,
            max_modes=req.max_modes,
        )

        return CavityResonanceResponse(
            modes=modes,
            dimensions_mm={
                "length": req.length_mm,
                "width": req.width_mm,
                "height": req.height_mm,
            },
        )

    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/frequency-sweep", response_model=EnclosureSweepResponse)
async def enclosure_frequency_sweep(req: EnclosureSweepRequest):
    """Sweep frequency showing bulk, aperture, and combined SE curves.

    At each frequency point, computes the bulk material SE, evaluates
    all aperture groups, and combines via power-summation.  Optionally
    overlays cavity resonance frequencies if enclosure dimensions are
    provided.
    """
    try:
        props = calculate_composite_properties(req.composition)
        thickness_m = req.wall_thickness_mm * 1e-3

        freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)

        bulk_se_list: List[float] = []
        combined_se_list: List[float] = []
        worst_aperture_se_list: List[float] = []

        for f_mhz in freqs_mhz:
            f_hz = float(f_mhz) * 1e6

            # Bulk SE
            bulk_result = calculator.calculate_shielding_effectiveness(
                conductivity=props["conductivity"],
                relative_permeability=props["permeability"],
                relative_permittivity=props["permittivity"],
                thickness=thickness_m,
                frequency=f_hz,
            )
            bulk_se = float(bulk_result["total_se"])
            bulk_se_list.append(bulk_se)

            # Aperture SE
            details = _evaluate_apertures(
                f_hz,
                req.circular_apertures,
                req.slot_apertures,
                req.waveguide_vents,
            )

            aperture_se_values = [d.se_array_db for d in details]

            # Worst aperture SE at this frequency
            if aperture_se_values:
                worst_aperture_se_list.append(min(aperture_se_values))
            else:
                worst_aperture_se_list.append(bulk_se)

            # Combined SE
            combined = combined_enclosure_se(bulk_se, aperture_se_values)
            combined_se_list.append(combined)

        # Optional cavity resonances
        cavity_res = None
        if (
            req.enclosure_length_mm is not None
            and req.enclosure_width_mm is not None
            and req.enclosure_height_mm is not None
        ):
            cavity_res = cavity_resonance_frequencies(
                length_m=req.enclosure_length_mm * 1e-3,
                width_m=req.enclosure_width_mm * 1e-3,
                height_m=req.enclosure_height_mm * 1e-3,
            )

        return EnclosureSweepResponse(
            frequencies_mhz=freqs_mhz.tolist(),
            bulk_se_db=bulk_se_list,
            combined_se_db=combined_se_list,
            worst_aperture_se_db=worst_aperture_se_list,
            cavity_resonances=cavity_res,
        )

    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
