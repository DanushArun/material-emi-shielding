from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np

router = APIRouter()

class CrosstalkRequest(BaseModel):
    cable_length_m: float
    wire_separation_m: float
    freq_start_mhz: float
    freq_end_mhz: float
    num_points: int = 100

class CrosstalkResponse(BaseModel):
    frequencies_mhz: list[float]
    next_db: list[float] # Near-End Crosstalk
    fext_db: list[float] # Far-End Crosstalk

@router.post("/crosstalk", response_model=CrosstalkResponse)
async def calculate_crosstalk(req: CrosstalkRequest):
    """
    Calculate Near-End (NEXT) and Far-End (FEXT) crosstalk between two parallel wires.
    This uses a simplified lumped element (L, C) transmission line model approximation.
    """
    freqs = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)
    
    # Constants
    mu_0 = 4 * np.pi * 1e-7 # Permeability of free space
    eps_0 = 8.854e-12       # Permittivity of free space
    
    # Wire assumptions (simplified)
    r_wire = 0.001 # 1mm radius
    d = req.wire_separation_m
    length = req.cable_length_m
    
    # Mutual Inductance (L_m) and Mutual Capacitance (C_m) approximations
    # For two parallel wires separated by distance d
    L_m = (mu_0 / (2 * np.pi)) * np.log(d / r_wire)
    C_m = (np.pi * eps_0) / np.log(d / r_wire)
    
    # Source characteristics
    Z_0 = 50.0 # 50 ohm system
    V_in = 1.0 # 1V input
    
    next_db_list = []
    fext_db_list = []
    
    for f_mhz in freqs:
        f = f_mhz * 1e6 # Convert to Hz
        omega = 2 * np.pi * f
        
        # Crosstalk coupling coefficients (Simplified Paul's MTL equations)
        # K_ne = Near-end coupling coefficient
        # K_fe = Far-end coupling coefficient
        
        # Inductive and capacitive coupling terms
        term_L = (omega * L_m * length) / Z_0
        term_C = (omega * C_m * length) * Z_0
        
        # Near-end voltage
        V_ne = 0.25 * V_in * (term_C + term_L)
        
        # Far-end voltage
        # In a perfectly homogeneous medium (like air), FEXT is theoretically zero for bare wires,
        # but we add a slight asymmetry/dielectric mismatch factor to make it realistic.
        asymmetry_factor = 0.1 
        V_fe = -0.25 * V_in * (term_C - term_L) * asymmetry_factor * (length / (3e8 / f))
        
        # Convert to dB (ensure we don't take log of 0)
        ne_db = 20 * np.log10(max(abs(V_ne), 1e-12))
        fe_db = 20 * np.log10(max(abs(V_fe), 1e-12))
        
        # Cap the crosstalk to physically realistic maximums (e.g. 0 dB means 100% coupling)
        next_db_list.append(min(0.0, float(ne_db)))
        fext_db_list.append(min(0.0, float(fe_db)))
        
    return CrosstalkResponse(
        frequencies_mhz=freqs.tolist(),
        next_db=next_db_list,
        fext_db=fext_db_list
    )
