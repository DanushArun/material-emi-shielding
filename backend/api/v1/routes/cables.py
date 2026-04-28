"""Cable crosstalk and shielding endpoints.

Wraps src.physics.cables_mtl for distributed-parameter MTL crosstalk
analysis (Clayton Paul's theory) and braided-shield transfer impedance
modelling (Kley model).

Endpoints
---------
POST /cables/crosstalk             NEXT/FEXT frequency sweep via MTL theory.
POST /cables/transfer-impedance    Braided shield Z_t and cable SE.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from src.physics.cables_mtl import (
    mtl_crosstalk_two_wire,
    transfer_impedance_kley,
    cable_se_from_zt,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CrosstalkRequest(BaseModel):
    """Request for MTL crosstalk frequency sweep."""

    cable_length_m: float = Field(
        ..., gt=0.0,
        description="Cable run length in metres.",
    )
    wire_separation_m: float = Field(
        ..., gt=0.0,
        description="Centre-to-centre wire separation in metres.",
    )
    freq_start_mhz: float = Field(
        ..., gt=0.0,
        description="Start frequency in MHz.",
    )
    freq_end_mhz: float = Field(
        ..., gt=0.0,
        description="End frequency in MHz.",
    )
    num_points: int = Field(
        100, ge=2, le=10000,
        description="Number of frequency points.",
    )
    # New optional fields with backward-compatible defaults
    wire_radius_m: float = Field(
        0.001, gt=0.0,
        description="Conductor radius in metres (default 1 mm).",
    )
    height_above_ground_m: float = Field(
        0.02, gt=0.0,
        description="Height above ground plane in metres (default 20 mm).",
    )
    z_source_ohm: float = Field(
        50.0, gt=0.0,
        description="Source impedance in ohms.",
    )
    z_load_ohm: float = Field(
        50.0, gt=0.0,
        description="Load impedance in ohms.",
    )
    epsilon_r: float = Field(
        1.0, ge=1.0,
        description="Relative permittivity of surrounding dielectric.",
    )
    r_per_m: float = Field(
        0.0, ge=0.0,
        description="Per-unit-length conductor resistance (ohm/m).",
    )
    g_per_m: float = Field(
        0.0, ge=0.0,
        description="Per-unit-length dielectric conductance (S/m).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "cable_length_m": 2.0,
                "wire_separation_m": 0.01,
                "freq_start_mhz": 1.0,
                "freq_end_mhz": 500.0,
                "num_points": 200,
                "wire_radius_m": 0.0005,
                "height_above_ground_m": 0.015,
                "z_source_ohm": 50.0,
                "z_load_ohm": 50.0,
            }
        }


class CrosstalkResponse(BaseModel):
    """Response from MTL crosstalk frequency sweep."""

    frequencies_mhz: List[float]
    next_db: List[float]
    fext_db: List[float]
    next_voltages: List[float]
    fext_voltages: List[float]
    phase_velocity_m_per_s: float
    beta_l_at_max_freq: float


class TransferImpedanceRequest(BaseModel):
    """Request for braided shield transfer impedance and cable SE."""

    freq_start_mhz: float = Field(
        ..., gt=0.0,
        description="Start frequency in MHz.",
    )
    freq_end_mhz: float = Field(
        ..., gt=0.0,
        description="End frequency in MHz.",
    )
    num_points: int = Field(
        100, ge=2, le=10000,
        description="Number of frequency points.",
    )
    r_dc_mohm_per_m: float = Field(
        ..., gt=0.0,
        description="DC braid resistance in milliohm/m.",
    )
    f_corner_khz: float = Field(
        ..., gt=0.0,
        description="Skin-effect corner frequency in kHz.",
    )
    mutual_inductance_nh_per_m: float = Field(
        1.0, ge=0.0,
        description="Braid porpoising mutual inductance in nH/m.",
    )
    cable_length_m: float = Field(
        1.0, gt=0.0,
        description="Cable length in metres.",
    )
    z0_ohm: float = Field(
        50.0, gt=0.0,
        description="System characteristic impedance in ohms.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "freq_start_mhz": 0.001,
                "freq_end_mhz": 100.0,
                "num_points": 200,
                "r_dc_mohm_per_m": 14.0,
                "f_corner_khz": 200.0,
                "mutual_inductance_nh_per_m": 1.0,
                "cable_length_m": 1.0,
                "z0_ohm": 50.0,
            }
        }


class TransferImpedanceResponse(BaseModel):
    """Response from transfer impedance sweep."""

    frequencies_mhz: List[float]
    zt_magnitude_ohm_per_m: List[float]
    zt_real_ohm_per_m: List[float]
    zt_imag_ohm_per_m: List[float]
    cable_se_db: List[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/crosstalk", response_model=CrosstalkResponse)
async def calculate_crosstalk(req: CrosstalkRequest):
    """Compute NEXT and FEXT crosstalk vs frequency using distributed MTL theory.

    Replaces the previous lumped-element approximation with Clayton Paul's
    multiconductor transmission line model, which is valid for electrically
    long cables (length > lambda/10).
    """
    try:
        freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)
        next_db_list = []
        fext_db_list = []
        next_v_list = []
        fext_v_list = []
        last_v_p = 0.0
        last_beta_l = 0.0

        for f_mhz in freqs_mhz:
            f_hz = float(f_mhz) * 1e6
            result = mtl_crosstalk_two_wire(
                cable_length_m=req.cable_length_m,
                wire_radius_m=req.wire_radius_m,
                separation_m=req.wire_separation_m,
                height_above_ground_m=req.height_above_ground_m,
                frequency_hz=f_hz,
                z_source=req.z_source_ohm,
                z_load=req.z_load_ohm,
                epsilon_r=req.epsilon_r,
                r_per_m=req.r_per_m,
                g_per_m=req.g_per_m,
            )
            next_db_list.append(result["next_db"])
            fext_db_list.append(result["fext_db"])
            next_v_list.append(result["next_voltage"])
            fext_v_list.append(result["fext_voltage"])
            last_v_p = result["phase_velocity_m_per_s"]
            last_beta_l = result["beta_l_rad"]

        return CrosstalkResponse(
            frequencies_mhz=freqs_mhz.tolist(),
            next_db=next_db_list,
            fext_db=fext_db_list,
            next_voltages=next_v_list,
            fext_voltages=fext_v_list,
            phase_velocity_m_per_s=last_v_p,
            beta_l_at_max_freq=last_beta_l,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.post("/transfer-impedance", response_model=TransferImpedanceResponse)
async def calculate_transfer_impedance(req: TransferImpedanceRequest):
    """Compute braided shield transfer impedance and cable SE vs frequency.

    Uses the Kley (1993) model: DC resistance + skin-effect AC rise +
    mutual-inductance leakage through braid apertures.
    """
    try:
        freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)

        # Convert user-friendly units to SI
        r_dc = req.r_dc_mohm_per_m * 1e-3           # mohm/m -> ohm/m
        f_corner = req.f_corner_khz * 1e3            # kHz -> Hz
        m_inductance = req.mutual_inductance_nh_per_m * 1e-9  # nH/m -> H/m

        zt_mag_list = []
        zt_real_list = []
        zt_imag_list = []
        se_list = []

        for f_mhz in freqs_mhz:
            f_hz = float(f_mhz) * 1e6

            zt = transfer_impedance_kley(
                frequency_hz=f_hz,
                r_dc_ohm_per_m=r_dc,
                f_corner_hz=f_corner,
                mutual_inductance_h_per_m=m_inductance,
            )
            se = cable_se_from_zt(
                frequency_hz=f_hz,
                zt_ohm_per_m=abs(zt),
                cable_length_m=req.cable_length_m,
                z0_ohm=req.z0_ohm,
            )

            zt_mag_list.append(float(abs(zt)))
            zt_real_list.append(float(zt.real))
            zt_imag_list.append(float(zt.imag))
            se_list.append(se)

        return TransferImpedanceResponse(
            frequencies_mhz=freqs_mhz.tolist(),
            zt_magnitude_ohm_per_m=zt_mag_list,
            zt_real_ohm_per_m=zt_real_list,
            zt_imag_ohm_per_m=zt_imag_list,
            cable_se_db=se_list,
        )
    except (ValueError, ZeroDivisionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
