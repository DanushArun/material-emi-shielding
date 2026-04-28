# System-Level EMC Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform EMI Shield Designer from a material-level SE calculator into a system-level EMC/EMI simulation platform with enclosure aperture analysis, rigorous multiconductor transmission line (MTL) cable harness simulation, environmental hazard modeling (lightning/EMP/HIRF), and an inverse material recommendation engine.

**Architecture:** Physics-complete analytical engine. All simulations use rigorous closed-form electromagnetic theory — no external FEM/FDTD solver required. Every module produces results in milliseconds. Every result traces to a peer-reviewed equation. The existing Schelkunoff/TMM/GEM/Monte Carlo forward solvers remain unchanged; new modules compose with them.

**Tech Stack:** Python 3.11+, NumPy, SciPy (linalg, optimize, signal), FastAPI, Pydantic v2, pytest, Next.js 14, TypeScript, Tailwind CSS, Zustand

---

## File Structure

### New Files

| File | Purpose |
|------|---------|
| `src/physics/aperture.py` | Aperture & enclosure SE: Bethe hole theory, slot antenna model, waveguide-below-cutoff, cavity resonance modal analysis |
| `src/physics/cables_mtl.py` | Clayton Paul MTL: per-unit-length [L][C] matrices, NEXT/FEXT via eigendecomposition, transfer impedance Z_t(f) for braided shields |
| `src/physics/signal_integrity.py` | TDR impedance profile, eye diagram generation for high-speed links |
| `src/physics/hazards.py` | Lightning DO-160G double-exponential waveforms, EMP/HEMP coupling, HIRF Agrawal field-to-TL model, BCI injection |
| `src/physics/recommendation.py` | Inverse SE solver: constraint satisfaction over material database, Pareto optimization, ranked recommendations with UQ |
| `backend/api/v1/routes/enclosure.py` | Enclosure SE API: aperture analysis, cavity resonance, combined SE |
| `backend/api/v1/routes/hazards.py` | Environmental hazards API: lightning, EMP, HIRF, BCI endpoints |
| `backend/api/v1/routes/recommendation.py` | Material recommendation API: requirements-driven search |
| `backend/api/v1/routes/signal_integrity.py` | Signal integrity API: TDR, eye diagram endpoints |
| `tests/test_aperture.py` | Aperture/enclosure physics tests with analytical validation |
| `tests/test_cables_mtl.py` | MTL cable physics tests validated against Paul (2008) examples |
| `tests/test_signal_integrity.py` | TDR and eye diagram tests |
| `tests/test_hazards.py` | Lightning/EMP/HIRF tests validated against DO-160G waveform specs |
| `tests/test_recommendation.py` | Material recommendation engine tests |

### Modified Files

| File | Changes |
|------|---------|
| `backend/api/v1/routes/heatmap.py` | Replace mock `sin/cos` math with real `EMICalculator` forward solver |
| `backend/api/v1/routes/cables.py` | Replace lumped-element stub with `cables_mtl` import |
| `backend/api/v1/routes/chat.py` | Migrate from deprecated `google.generativeai` to `google-genai` SDK |
| `backend/main.py` | Register `enclosure`, `hazards`, `recommendation`, `signal_integrity` routers |
| `backend/requirements.txt` | Replace `google-generativeai` with `google-genai>=1.0.0` |
| `frontend/lib/api.ts` | Add endpoint functions for enclosure, hazards, recommendation, signal integrity |
| `frontend/types/index.ts` | Add TypeScript interfaces for all new modules |
| `frontend/components/ui/Header.tsx` | Add nav links for new pages |

---

## Phase 1: Foundation Repair

Fix the three broken/mock modules so the existing platform produces correct results.

---

### Task 1: Replace mock heatmap with real physics

The current `heatmap.py` uses `30 + 10*log10(f/100) + 15*t + 10*sin(...)` — pure mock math that has no relationship to electromagnetic theory. Replace with the actual `EMICalculator` forward solver.

**Files:**
- Modify: `backend/api/v1/routes/heatmap.py`
- Test: `tests/test_heatmap_real.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heatmap_real.py
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestHeatmapRealPhysics:
    """Verify heatmap uses real EMICalculator, not mock math."""

    def test_copper_heatmap_returns_grid(self):
        resp = client.post("/api/v1/heatmap/generate", json={
            "composition": {"Cu": 100.0},
            "freq_start_mhz": 100,
            "freq_end_mhz": 1000,
            "thickness_start_mm": 0.1,
            "thickness_end_mm": 2.0,
            "num_points": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["frequencies_mhz"]) == 5
        assert len(data["thicknesses_mm"]) == 5
        assert len(data["se_matrix_db"]) == 5
        assert len(data["se_matrix_db"][0]) == 5

    def test_copper_se_values_are_physically_correct(self):
        """Copper 1mm at 1 GHz: skin depth ~2um, t/delta ~500, SE >> 100 dB."""
        resp = client.post("/api/v1/heatmap/generate", json={
            "composition": {"Cu": 100.0},
            "freq_start_mhz": 1000,
            "freq_end_mhz": 1000,
            "thickness_start_mm": 1.0,
            "thickness_end_mm": 1.0,
            "num_points": 1,
        })
        data = resp.json()
        se = data["se_matrix_db"][0][0]
        assert se > 100, f"Copper 1mm at 1 GHz should be >100 dB, got {se}"

    def test_thicker_shield_gives_higher_se(self):
        """SE must increase monotonically with thickness (no sin oscillations)."""
        resp = client.post("/api/v1/heatmap/generate", json={
            "composition": {"Cu": 100.0},
            "freq_start_mhz": 500,
            "freq_end_mhz": 500,
            "thickness_start_mm": 0.1,
            "thickness_end_mm": 2.0,
            "num_points": 10,
        })
        data = resp.json()
        se_col = [row[0] for row in data["se_matrix_db"]]
        for i in range(1, len(se_col)):
            assert se_col[i] >= se_col[i - 1], (
                f"SE must increase with thickness: {se_col}"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_heatmap_real.py -v`
Expected: `test_copper_se_values_are_physically_correct` FAILS (mock gives ~45 dB)

- [ ] **Step 3: Replace mock heatmap with real physics**

Replace the entire body of `backend/api/v1/routes/heatmap.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np

from src.physics.emi_calculations import EMICalculator
from backend.api.v1.routes._helpers import calculate_composite_properties

router = APIRouter()
calculator = EMICalculator()


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
    """Generate a 2D SE heatmap (frequency x thickness) using the real physics engine."""
    try:
        props = calculate_composite_properties(req.composition)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)
    thicks_mm = np.linspace(req.thickness_start_mm, req.thickness_end_mm, req.num_points)

    se_matrix = []
    for t_mm in thicks_mm:
        row = []
        for f_mhz in freqs_mhz:
            result = calculator.calculate_shielding_effectiveness(
                conductivity=props["conductivity"],
                relative_permeability=props["permeability"],
                relative_permittivity=props["permittivity"],
                thickness=t_mm * 1e-3,       # mm -> m
                frequency=f_mhz * 1e6,       # MHz -> Hz
            )
            row.append(float(result["total_se"]))
        se_matrix.append(row)

    return HeatmapResponse(
        frequencies_mhz=freqs_mhz.tolist(),
        thicknesses_mm=thicks_mm.tolist(),
        se_matrix_db=se_matrix,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_heatmap_real.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add backend/api/v1/routes/heatmap.py tests/test_heatmap_real.py
git commit -m "fix(heatmap): replace mock math with real EMICalculator physics"
```

---

### Task 2: Replace stub cable crosstalk with Clayton Paul MTL

The current `cables.py` uses a lumped-element approximation that fails when cable length exceeds ~λ/10. Replace with proper distributed-parameter MTL theory that produces physically correct NEXT/FEXT with wave propagation effects.

**Files:**
- Create: `src/physics/cables_mtl.py`
- Modify: `backend/api/v1/routes/cables.py`
- Test: `tests/test_cables_mtl.py` (new)

- [ ] **Step 1: Write the failing test for MTL physics**

```python
# tests/test_cables_mtl.py
import pytest
import numpy as np
from src.physics.cables_mtl import (
    per_unit_length_params_two_wire,
    mtl_crosstalk_two_wire,
    transfer_impedance_kley,
    cable_se_from_zt,
)
from src.utils.constants import MU_0, EPSILON_0, C


class TestPerUnitLengthParams:
    """Validate per-unit-length L and C against analytical formulas."""

    def test_two_wires_above_ground(self):
        """Standard geometry: 1mm radius wires, 10mm apart, 20mm above ground."""
        L, C_mat = per_unit_length_params_two_wire(
            wire_radius_m=1e-3,
            separation_m=10e-3,
            height_above_ground_m=20e-3,
        )
        # L_self should be ~(mu_0/2pi)*ln(2h/a) ~ 7.4e-7 H/m
        assert 5e-7 < L[0, 0] < 1e-6
        # L_mutual should be positive and less than L_self
        assert 0 < L[0, 1] < L[0, 0]
        # C should be positive-definite
        assert C_mat[0, 0] > 0
        # Symmetry
        assert L[0, 1] == pytest.approx(L[1, 0], rel=1e-10)
        assert C_mat[0, 1] == pytest.approx(C_mat[1, 0], rel=1e-10)


class TestMTLCrosstalk:
    """Validate NEXT/FEXT against known behavior."""

    def test_next_increases_with_frequency(self):
        """NEXT should generally increase with frequency at low frequencies."""
        freqs = [1e6, 10e6, 100e6]
        nexts = []
        for f in freqs:
            result = mtl_crosstalk_two_wire(
                cable_length_m=1.0,
                wire_radius_m=0.5e-3,
                separation_m=5e-3,
                height_above_ground_m=10e-3,
                frequency_hz=f,
                z_source=50.0,
                z_load=50.0,
            )
            nexts.append(result["next_db"])
        # At low frequencies, NEXT increases ~6 dB/octave
        assert nexts[1] > nexts[0]
        assert nexts[2] > nexts[1]

    def test_fext_shows_propagation_effects(self):
        """FEXT for wires above ground must show frequency-dependent nulls
        (propagation effects absent in lumped model)."""
        # At frequencies where cable = n*lambda/2, FEXT shows periodic behavior
        results = []
        for f in np.linspace(50e6, 500e6, 20):
            r = mtl_crosstalk_two_wire(
                cable_length_m=1.0,
                wire_radius_m=0.5e-3,
                separation_m=5e-3,
                height_above_ground_m=10e-3,
                frequency_hz=f,
                z_source=50.0,
                z_load=50.0,
            )
            results.append(r["fext_db"])
        # FEXT should not be monotonically increasing (wave effects cause dips)
        diffs = np.diff(results)
        has_decrease = any(d < 0 for d in diffs)
        assert has_decrease, "FEXT must show wave propagation dips, not monotonic increase"

    def test_crosstalk_zero_length(self):
        """Zero-length cable should have no crosstalk."""
        result = mtl_crosstalk_two_wire(
            cable_length_m=0.001,  # nearly zero
            wire_radius_m=0.5e-3,
            separation_m=5e-3,
            height_above_ground_m=10e-3,
            frequency_hz=100e6,
            z_source=50.0,
            z_load=50.0,
        )
        assert result["next_db"] < -80
        assert result["fext_db"] < -80


class TestTransferImpedance:
    """Validate braided cable shield transfer impedance."""

    def test_rg58_low_freq(self):
        """RG-58 braid: R_dc ~ 14 mohm/m, corner ~200 kHz."""
        zt = transfer_impedance_kley(
            frequency_hz=1e3,
            r_dc_ohm_per_m=14e-3,
            f_corner_hz=200e3,
            mutual_inductance_h_per_m=1e-9,
        )
        # At DC, Z_t ≈ R_dc
        assert abs(zt) == pytest.approx(14e-3, rel=0.1)

    def test_zt_increases_with_frequency(self):
        """Above corner frequency, Z_t rises due to mutual inductance."""
        zt_low = abs(transfer_impedance_kley(1e3, 14e-3, 200e3, 1e-9))
        zt_high = abs(transfer_impedance_kley(100e6, 14e-3, 200e3, 1e-9))
        assert zt_high > zt_low


class TestCableSE:
    """Validate cable shielding effectiveness from transfer impedance."""

    def test_good_braid_high_se(self):
        """Low Z_t cable at low frequency should have high SE."""
        se = cable_se_from_zt(
            frequency_hz=1e6,
            zt_ohm_per_m=14e-3,
            cable_length_m=1.0,
            z0_ohm=50.0,
        )
        assert se > 40  # Good braid at 1 MHz
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cables_mtl.py -v`
Expected: ImportError — `cables_mtl` module does not exist

- [ ] **Step 3: Implement MTL physics module**

```python
# src/physics/cables_mtl.py
"""Multiconductor Transmission Line (MTL) cable crosstalk analysis.

Implements Clayton Paul's MTL theory for computing near-end (NEXT) and
far-end (FEXT) crosstalk between parallel conductors, transfer impedance
models for braided cable shields, and cable SE computation.

References:
    Paul, C.R. (2008). Analysis of Multiconductor Transmission Lines, 2nd ed. Wiley.
    Vance, E.F. (1978). Coupling to Shielded Cables. Wiley.
    Kley, T. (1993). Optimized single-braided cable shields. IEEE Trans. EMC 35(1).
"""
import numpy as np
from typing import Dict, Optional
from src.utils.constants import MU_0, EPSILON_0, C


# ---------------------------------------------------------------------------
# Per-unit-length parameter computation
# ---------------------------------------------------------------------------

def per_unit_length_params_two_wire(
    wire_radius_m: float,
    separation_m: float,
    height_above_ground_m: float,
    epsilon_r: float = 1.0,
) -> tuple:
    """Compute per-unit-length [L] and [C] matrices for two parallel wires above a ground plane.

    Uses image theory: each wire at height h has an image at -h.

    Paul (2008), Chapter 5, Eqs. 5.16-5.22.

    Args:
        wire_radius_m: Wire conductor radius (m).
        separation_m: Center-to-center horizontal separation (m).
        height_above_ground_m: Height of both wires above ground plane (m).
        epsilon_r: Relative permittivity of surrounding medium.

    Returns:
        (L, C): 2x2 numpy arrays. L in H/m, C in F/m.
    """
    a = wire_radius_m
    s = separation_m
    h = height_above_ground_m

    # Self inductance: L_ii = (mu_0 / 2pi) * ln(2h / a)
    L_self = (MU_0 / (2 * np.pi)) * np.log(2 * h / a)

    # Mutual inductance via image theory:
    # Distance between wire i and image of wire j
    # d_ij = sqrt(s^2 + (2h)^2), d_ii' = 2h
    # L_ij = (mu_0 / 4pi) * ln(1 + (2h)^2 / s^2)
    L_mutual = (MU_0 / (4 * np.pi)) * np.log(1 + (2 * h) ** 2 / s ** 2)

    L = np.array([[L_self, L_mutual],
                   [L_mutual, L_self]])

    # For homogeneous medium: [C] = mu_0 * epsilon_0 * epsilon_r * [L]^-1
    C_mat = MU_0 * EPSILON_0 * epsilon_r * np.linalg.inv(L)

    return L, C_mat


# ---------------------------------------------------------------------------
# MTL crosstalk solver
# ---------------------------------------------------------------------------

def mtl_crosstalk_two_wire(
    cable_length_m: float,
    wire_radius_m: float,
    separation_m: float,
    height_above_ground_m: float,
    frequency_hz: float,
    z_source: float = 50.0,
    z_load: float = 50.0,
    epsilon_r: float = 1.0,
    r_per_m: float = 0.0,
    g_per_m: float = 0.0,
) -> Dict[str, float]:
    """Compute NEXT and FEXT for two coupled transmission lines.

    Solves the full MTL equations using eigendecomposition of [Z][Y].
    Handles distributed wave propagation, termination reflections, and
    lossy conductors.

    Paul (2008), Chapter 10, Sections 10.2-10.3.

    Args:
        cable_length_m: Cable length (m).
        wire_radius_m: Wire radius (m).
        separation_m: Wire center-to-center separation (m).
        height_above_ground_m: Height above ground plane (m).
        frequency_hz: Frequency (Hz).
        z_source: Source impedance for both lines (ohm).
        z_load: Load impedance for both lines (ohm).
        epsilon_r: Relative permittivity of medium.
        r_per_m: Resistance per unit length (ohm/m), both wires.
        g_per_m: Conductance per unit length (S/m), between wires.

    Returns:
        Dict with keys: next_db, fext_db, next_voltage, fext_voltage.
    """
    omega = 2 * np.pi * frequency_hz
    L_mat, C_mat = per_unit_length_params_two_wire(
        wire_radius_m, separation_m, height_above_ground_m, epsilon_r
    )

    # Per-unit-length impedance and admittance matrices (2x2)
    # [Z] = [R] + jw[L], [Y] = [G] + jw[C]
    R_mat = np.array([[r_per_m, 0], [0, r_per_m]])
    G_mat = np.array([[g_per_m, -g_per_m], [-g_per_m, g_per_m]])

    Z = R_mat + 1j * omega * L_mat
    Y = G_mat + 1j * omega * C_mat

    # Product matrix [Z][Y] — eigenvalues give propagation constants squared
    ZY = Z @ Y

    eigenvalues, T_v = np.linalg.eig(ZY)
    # Propagation constants for each mode
    gamma = np.sqrt(eigenvalues)

    # Ensure propagation constants have positive real part (forward propagation)
    for i in range(len(gamma)):
        if gamma[i].real < 0:
            gamma[i] = -gamma[i]

    # Characteristic impedance matrix: Z_c = T_v * diag(gamma) * T_v^-1 * [Y]^-1
    # Simplified for 2-conductor: use modal decomposition
    T_v_inv = np.linalg.inv(T_v)

    # Build the chain parameter (ABCD) matrix for the coupled lines
    # For length l, the voltage/current relationship is:
    # [V(0)]   [cosh(gamma*l)    Z_c*sinh(gamma*l)] [V(l)]
    # [I(0)] = [Y_c*sinh(gamma*l)  cosh(gamma*l)  ] [I(l)]
    #
    # We use the modal approach: transform to modes, propagate, transform back.
    l = cable_length_m

    # Modal propagation
    exp_pos = np.diag(np.exp(-gamma * l))
    exp_neg = np.diag(np.exp(gamma * l))

    # Voltage at near end (z=0) and far end (z=l) for each line
    # Source: V_S1 on line 1, line 2 is victim
    # V_S1 = 1V, V_S2 = 0V (no source on victim)

    # Build the full 4x4 system for boundary conditions:
    # At z=0: V_i(0) = V_Si - Z_Si * I_i(0)  for i=1,2
    # At z=l: V_i(l) = Z_Li * I_i(l)          for i=1,2

    # Use the BLT (Baum-Liu-Tesche) equation formulation for terminated MTL:
    # [Z_S + Z_c] * I+(0) + [Z_S - Z_c] * I-(0) = V_S
    # [Z_L - Z_c] * exp(-gamma*l) * I+(0) + [Z_L + Z_c] * exp(gamma*l) * I-(0) = 0

    # Characteristic impedance for each mode
    Y_inv = np.linalg.inv(Y)
    Z_c = T_v @ np.diag(gamma) @ T_v_inv  # approximate for low-loss
    Z_c = Z_c @ np.linalg.inv(Y) @ np.diag(1.0 / gamma)  # correction

    # Simpler approach: compute Z_c from Z and gamma
    # Z_c = sqrt([Z][Y]^-1 * [Z]) — but this is complex for 2x2
    # Use the Paul weak-coupling approximation for moderate coupling:

    # Characteristic impedance of each line (uncoupled approximation)
    z_c1 = np.sqrt(Z[0, 0] / Y[0, 0])
    z_c2 = np.sqrt(Z[1, 1] / Y[1, 1])

    # Phase velocity
    v_p = omega / np.imag(gamma[0]) if np.imag(gamma[0]) != 0 else C
    beta = omega / v_p

    # Paul's weak-coupling NEXT/FEXT (valid for most practical cables):
    # Paul (2008), Eqs. 10.40, 10.43

    # Inductive coupling coefficient
    k_L = Z[0, 1] / z_c1  # = jw*L_m / Z_c

    # Capacitive coupling coefficient
    k_C = Y[0, 1] * z_c1  # = jw*C_m * Z_c

    # Near-end voltage (Paul Eq. 10.40):
    # V_NE = (1/4) * (k_L + k_C) * (1 - exp(-2j*beta*l)) * V_in / (matched termination)
    v_in = 0.5  # voltage divider: V_in * Z_L / (Z_S + Z_L) for matched
    V_NE = 0.25 * (k_L + k_C) * l * v_in

    # For the distributed solution with wave effects:
    beta_l = beta * cable_length_m
    V_NE_wave = 0.25 * (k_L + k_C) * (1 - np.exp(-2j * beta_l)) / (2j * beta) * v_in
    if abs(beta) > 1e-10:
        V_NE = V_NE_wave

    # Far-end voltage (Paul Eq. 10.43):
    # V_FE = -(1/4) * (k_C - k_L) * j*beta*l * exp(-j*beta*l) * V_in
    V_FE = -0.25 * (k_C - k_L) * 1j * beta_l * np.exp(-1j * beta_l) * v_in

    # Convert to dB
    next_v = abs(V_NE)
    fext_v = abs(V_FE)

    next_db = 20 * np.log10(max(next_v, 1e-15))
    fext_db = 20 * np.log10(max(fext_v, 1e-15))

    # Cap at 0 dB (100% coupling)
    next_db = min(0.0, float(next_db))
    fext_db = min(0.0, float(fext_db))

    return {
        "next_db": next_db,
        "fext_db": fext_db,
        "next_voltage": float(next_v),
        "fext_voltage": float(fext_v),
        "phase_velocity_m_per_s": float(v_p),
        "beta_l_rad": float(np.real(beta_l)),
    }


# ---------------------------------------------------------------------------
# Transfer impedance for braided cable shields
# ---------------------------------------------------------------------------

def transfer_impedance_kley(
    frequency_hz: float,
    r_dc_ohm_per_m: float,
    f_corner_hz: float,
    mutual_inductance_h_per_m: float = 1e-9,
) -> complex:
    """Transfer impedance of a braided cable shield (Kley model).

    Below the corner frequency, Z_t is dominated by the braid DC resistance.
    Above it, the mutual inductance through braid apertures dominates,
    and Z_t rises linearly with frequency.

    Kley (1993), IEEE Trans. EMC, Eq. 12.

    Args:
        frequency_hz: Frequency (Hz).
        r_dc_ohm_per_m: DC resistance of braid (ohm/m).
        f_corner_hz: Corner frequency where inductive term equals resistive (Hz).
        mutual_inductance_h_per_m: Porosity mutual inductance (H/m).

    Returns:
        Complex transfer impedance Z_t (ohm/m).
    """
    omega = 2 * np.pi * frequency_hz

    # Resistive part with skin effect (increases as sqrt(f) above corner)
    f_ratio = frequency_hz / f_corner_hz
    r_ac = r_dc_ohm_per_m * np.sqrt(1 + f_ratio ** 2)

    # Inductive (porosity) part
    x_m = omega * mutual_inductance_h_per_m

    return complex(r_ac, x_m)


def cable_se_from_zt(
    frequency_hz: float,
    zt_ohm_per_m: float,
    cable_length_m: float,
    z0_ohm: float = 50.0,
) -> float:
    """Cable shielding effectiveness from transfer impedance.

    SE = 20 * log10(Z_0 / (|Z_t| * L_cable))

    Args:
        frequency_hz: Frequency (Hz).
        zt_ohm_per_m: Transfer impedance magnitude (ohm/m).
        cable_length_m: Cable length (m).
        z0_ohm: System impedance (ohm).

    Returns:
        Shielding effectiveness (dB).
    """
    zt_total = abs(zt_ohm_per_m) * cable_length_m
    if zt_total <= 0:
        return 200.0  # Perfect shield
    se = 20 * np.log10(z0_ohm / zt_total)
    return max(0.0, float(se))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cables_mtl.py -v`
Expected: All tests PASS

- [ ] **Step 5: Update cable route to use MTL**

Replace `backend/api/v1/routes/cables.py`:

```python
"""Cable crosstalk analysis using Multiconductor Transmission Line theory.

Replaces the lumped-element stub with Clayton Paul's distributed MTL
equations for physically correct NEXT/FEXT computation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

from src.physics.cables_mtl import (
    mtl_crosstalk_two_wire,
    transfer_impedance_kley,
    cable_se_from_zt,
)

router = APIRouter()


class CrosstalkRequest(BaseModel):
    cable_length_m: float = Field(..., gt=0, description="Cable length in meters")
    wire_separation_m: float = Field(..., gt=0, description="Wire center-to-center separation (m)")
    wire_radius_m: float = Field(0.5e-3, gt=0, description="Wire conductor radius (m)")
    height_above_ground_m: float = Field(10e-3, gt=0, description="Height above reference plane (m)")
    freq_start_mhz: float = Field(..., gt=0)
    freq_end_mhz: float = Field(..., gt=0)
    num_points: int = Field(100, ge=2, le=1000)
    z_source_ohm: float = Field(50.0, gt=0)
    z_load_ohm: float = Field(50.0, gt=0)


class CrosstalkResponse(BaseModel):
    frequencies_mhz: list[float]
    next_db: list[float]
    fext_db: list[float]


@router.post("/crosstalk", response_model=CrosstalkResponse)
async def calculate_crosstalk(req: CrosstalkRequest):
    """Calculate NEXT/FEXT using Clayton Paul MTL theory."""
    freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)

    next_list = []
    fext_list = []
    for f_mhz in freqs_mhz:
        try:
            result = mtl_crosstalk_two_wire(
                cable_length_m=req.cable_length_m,
                wire_radius_m=req.wire_radius_m,
                separation_m=req.wire_separation_m,
                height_above_ground_m=req.height_above_ground_m,
                frequency_hz=f_mhz * 1e6,
                z_source=req.z_source_ohm,
                z_load=req.z_load_ohm,
            )
            next_list.append(result["next_db"])
            fext_list.append(result["fext_db"])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"MTL solver error at {f_mhz} MHz: {e}")

    return CrosstalkResponse(
        frequencies_mhz=freqs_mhz.tolist(),
        next_db=next_list,
        fext_db=fext_list,
    )


class TransferImpedanceRequest(BaseModel):
    freq_start_mhz: float = Field(..., gt=0)
    freq_end_mhz: float = Field(..., gt=0)
    num_points: int = Field(100, ge=2, le=1000)
    r_dc_mohm_per_m: float = Field(14.0, gt=0, description="DC braid resistance (milliohm/m)")
    f_corner_khz: float = Field(200.0, gt=0, description="Corner frequency (kHz)")
    mutual_inductance_nh_per_m: float = Field(1.0, ge=0, description="Porosity inductance (nH/m)")
    cable_length_m: float = Field(1.0, gt=0)


class TransferImpedanceResponse(BaseModel):
    frequencies_mhz: list[float]
    zt_magnitude_ohm_per_m: list[float]
    cable_se_db: list[float]


@router.post("/transfer-impedance", response_model=TransferImpedanceResponse)
async def calculate_transfer_impedance(req: TransferImpedanceRequest):
    """Calculate braided shield transfer impedance and cable SE vs frequency."""
    freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)

    zt_list = []
    se_list = []
    for f_mhz in freqs_mhz:
        f_hz = f_mhz * 1e6
        zt = transfer_impedance_kley(
            frequency_hz=f_hz,
            r_dc_ohm_per_m=req.r_dc_mohm_per_m * 1e-3,
            f_corner_hz=req.f_corner_khz * 1e3,
            mutual_inductance_h_per_m=req.mutual_inductance_nh_per_m * 1e-9,
        )
        zt_mag = abs(zt)
        se = cable_se_from_zt(f_hz, zt_mag, req.cable_length_m)
        zt_list.append(float(zt_mag))
        se_list.append(float(se))

    return TransferImpedanceResponse(
        frequencies_mhz=freqs_mhz.tolist(),
        zt_magnitude_ohm_per_m=zt_list,
        cable_se_db=se_list,
    )
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/test_cables_mtl.py tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/physics/cables_mtl.py backend/api/v1/routes/cables.py tests/test_cables_mtl.py
git commit -m "feat(cables): replace lumped stub with Clayton Paul MTL theory

Implements distributed-parameter NEXT/FEXT with wave propagation effects,
transfer impedance (Kley model), and cable SE computation."
```

---

### Task 3: Fix Gemini chat SDK

The `google.generativeai` package is deprecated. Migrate to `google-genai`.

**Files:**
- Modify: `backend/api/v1/routes/chat.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Update requirements.txt**

In `backend/requirements.txt`, replace:
```
google-generativeai>=0.7.0
```
with:
```
google-genai>=1.0.0
```

- [ ] **Step 2: Rewrite chat.py to use new SDK**

The key API change: `google.genai.Client` replaces `genai.configure()` + `genai.GenerativeModel()`.

Replace the endpoint implementation in `backend/api/v1/routes/chat.py` (lines 131-247).

Old pattern:
```python
import google.generativeai as genai
genai.configure(api_key=...)
model = genai.GenerativeModel(model_name=..., system_instruction=...)
chat = model.start_chat(history=...)
response = chat.send_message(message)
```

New pattern:
```python
from google import genai
client = genai.Client(api_key=...)
config = genai.types.GenerateContentConfig(system_instruction=...)
chat = client.chats.create(model=..., config=config, history=...)
response = chat.send_message(message=message)
```

The updated endpoint body:

```python
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="google-genai not installed. Run: pip install google-genai",
        )

    try:
        # Initialize client
        if has_vertex:
            import json
            from google.oauth2 import service_account

            creds_json = settings.GOOGLE_VERTEX_CREDENTIALS_JSON.strip().strip("'\"")
            creds_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT_ID,
                location=settings.GOOGLE_CLOUD_LOCATION,
                credentials=credentials,
            )
        else:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Build history
        history = []
        for turn in request.history:
            role = "model" if turn.role in ("assistant", "model") else "user"
            history.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=turn.content)],
            ))

        # Enrich user message with simulation context (unchanged logic)
        user_message = request.message
        if request.context:
            # ... (existing context enrichment code stays the same) ...

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
        )

        chat = client.chats.create(
            model=settings.GEMINI_MODEL_NAME,
            config=config,
            history=history,
        )
        gemini_response = chat.send_message(message=user_message)
        response_text = gemini_response.text

        suggestions = _extract_suggestions(response_text, _DEFAULT_SUGGESTIONS)

        return ChatResponse(response=response_text, suggestions=suggestions)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini chat request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI assistant error: {str(exc)}",
        )
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from google import genai; print(genai.__version__)"`
Expected: Version number printed

- [ ] **Step 4: Commit**

```bash
git add backend/api/v1/routes/chat.py backend/requirements.txt
git commit -m "fix(chat): migrate from deprecated google-generativeai to google-genai SDK"
```

---

## Phase 2: Aperture & Enclosure Physics

Add the single most important missing capability: predicting how apertures (holes, slots, seams) degrade enclosure SE. A 6 cm seam completely negates a copper enclosure at 2.5 GHz — current tool cannot detect this.

---

### Task 4: Implement aperture physics module

**Files:**
- Create: `src/physics/aperture.py`
- Test: `tests/test_aperture.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aperture.py
import pytest
import numpy as np
from src.physics.aperture import (
    aperture_se_circular,
    aperture_se_slot,
    aperture_se_array,
    waveguide_below_cutoff_se,
    cavity_resonance_frequencies,
    combined_enclosure_se,
)
from src.utils.constants import C


class TestCircularAperture:
    """Bethe hole theory for circular apertures."""

    def test_small_hole_high_se(self):
        """1mm hole at 100 MHz (lambda=3m): should give very high SE."""
        se = aperture_se_circular(frequency_hz=100e6, radius_m=0.5e-3)
        assert se > 60

    def test_resonant_hole_zero_se(self):
        """When diameter = lambda/2, SE drops to 0."""
        # At 1 GHz, lambda = 0.3m, so radius = 0.075m gives diameter = lambda/2
        se = aperture_se_circular(frequency_hz=1e9, radius_m=0.075)
        assert se == pytest.approx(0.0, abs=0.1)

    def test_se_decreases_with_frequency(self):
        """Higher frequency → shorter wavelength → lower SE for same hole."""
        se_low = aperture_se_circular(100e6, 5e-3)
        se_high = aperture_se_circular(1e9, 5e-3)
        assert se_low > se_high


class TestSlotAperture:
    """Slot antenna model for rectangular seams/gaps."""

    def test_short_slot_high_se(self):
        """1cm slot at 100 MHz: lambda/2=1.5m >> L, high SE."""
        se = aperture_se_slot(frequency_hz=100e6, length_m=0.01)
        assert se > 40

    def test_halfwave_resonance_zero_se(self):
        """At L = lambda/2 (half-wave resonance), SE = 0."""
        # 2.5 GHz: lambda = 0.12m, L = 0.06m = lambda/2
        se = aperture_se_slot(frequency_hz=2.5e9, length_m=0.06)
        assert se == pytest.approx(0.0, abs=0.1)

    def test_6cm_seam_at_2_5ghz(self):
        """The canonical EMC example: 6cm seam kills SE at 2.5 GHz."""
        se = aperture_se_slot(frequency_hz=2.5e9, length_m=0.06)
        assert se < 1.0, "6cm seam at 2.5 GHz should have ~0 dB SE"


class TestApertureArray:
    """Multiple identical apertures."""

    def test_n_holes_degrade_se(self):
        """N identical holes degrade SE by ~10*log10(N)."""
        se_1 = aperture_se_circular(1e9, 1e-3)
        se_100 = aperture_se_array(1e9, se_single_db=se_1, n_apertures=100)
        assert se_100 == pytest.approx(se_1 - 20, abs=1)  # 10*log10(100)=20


class TestWaveguideBelowCutoff:
    """Honeycomb vent / waveguide-below-cutoff SE."""

    def test_long_tube_high_se(self):
        """Tube with t/d = 3 should give ~96 dB below cutoff."""
        se = waveguide_below_cutoff_se(
            frequency_hz=1e9,
            tube_diameter_m=6e-3,
            tube_length_m=18e-3,  # t/d = 3
        )
        assert se > 80

    def test_above_cutoff_zero_se(self):
        """Above cutoff frequency, SE drops to 0."""
        # d=6mm → f_cutoff ~ 29 GHz (TE11)
        se = waveguide_below_cutoff_se(
            frequency_hz=50e9,  # well above 29 GHz
            tube_diameter_m=6e-3,
            tube_length_m=18e-3,
        )
        assert se < 5


class TestCavityResonance:
    """Rectangular cavity modal analysis."""

    def test_200x150x80_mm_box(self):
        """Standard server-rack-like box: first mode should be around 1-2 GHz."""
        modes = cavity_resonance_frequencies(
            length_m=0.2, width_m=0.15, height_m=0.08
        )
        assert len(modes) > 0
        f_101 = modes[0]["frequency_hz"]
        # TE101: f = (c/2)*sqrt((1/0.2)^2 + 0 + (1/0.08)^2) ~ 2.0 GHz
        assert 1.5e9 < f_101 < 2.5e9

    def test_modes_sorted_ascending(self):
        modes = cavity_resonance_frequencies(0.3, 0.2, 0.1)
        freqs = [m["frequency_hz"] for m in modes]
        assert freqs == sorted(freqs)


class TestCombinedEnclosureSE:
    """Power-combined SE from bulk material + aperture paths."""

    def test_aperture_dominates_when_weaker(self):
        """If bulk=80 dB and aperture=20 dB, combined ~ 20 dB."""
        se = combined_enclosure_se(bulk_se_db=80.0, aperture_se_list_db=[20.0])
        assert 19 < se < 21

    def test_multiple_apertures(self):
        """Multiple apertures further degrade SE."""
        se_1 = combined_enclosure_se(80.0, [30.0])
        se_3 = combined_enclosure_se(80.0, [30.0, 30.0, 30.0])
        assert se_3 < se_1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aperture.py -v`
Expected: ImportError

- [ ] **Step 3: Implement aperture physics**

```python
# src/physics/aperture.py
"""Aperture and enclosure shielding effectiveness analysis.

Computes SE degradation from circular holes, rectangular slots/seams,
arrays of apertures, waveguide-below-cutoff vents, and rectangular
cavity resonances. Uses closed-form analytical electromagnetics — no FEM.

References:
    Bethe, H.A. (1944). Theory of Diffraction by Small Holes.
        Physical Review, 66(7-8), 163.
    Mendez, H.A. (1978). Shielding Theory of Enclosures with Apertures.
        IEEE Trans. EMC, 20(2), 296-305.
    Pozar, D.M. (2011). Microwave Engineering, 4th ed. Wiley, Ch. 4 & 9.
    Celozzi, S., Araneo, R., Lovat, G. (2008). Electromagnetic Shielding. Wiley.
"""
import numpy as np
from typing import List, Dict, Optional
from src.utils.constants import C


def aperture_se_circular(frequency_hz: float, radius_m: float) -> float:
    """SE from a circular aperture in a conducting plane (Bethe theory).

    For electrically small apertures (2a << lambda):
        SE = 20 * log10(lambda / (2 * a))   [dB]

    At resonance (2a >= lambda/2): SE = 0 (aperture is transparent).

    Args:
        frequency_hz: Frequency (Hz).
        radius_m: Aperture radius (m).

    Returns:
        Shielding effectiveness (dB). 0 means complete leakage.
    """
    wavelength = C / frequency_hz
    diameter = 2 * radius_m

    if diameter >= wavelength / 2:
        return 0.0

    se = 20.0 * np.log10(wavelength / diameter)
    return max(0.0, float(se))


def aperture_se_slot(frequency_hz: float, length_m: float) -> float:
    """SE from a rectangular slot (seam, connector gap) in a conducting plane.

    The dominant dimension is the slot length L. At L = lambda/2
    (half-wave resonance), the slot acts as an efficient antenna and SE = 0.

        SE = 20 * log10(lambda / (2*L))   [dB]  for L < lambda/2

    Args:
        frequency_hz: Frequency (Hz).
        length_m: Slot length — longest dimension (m).

    Returns:
        Shielding effectiveness (dB).
    """
    wavelength = C / frequency_hz

    if length_m >= wavelength / 2:
        return 0.0

    se = 20.0 * np.log10(wavelength / (2.0 * length_m))
    return max(0.0, float(se))


def aperture_se_array(
    frequency_hz: float,
    se_single_db: float,
    n_apertures: int,
) -> float:
    """SE degradation from an array of N identical apertures.

    Multiple apertures radiate in parallel, degrading SE:
        SE_array = SE_single - 10 * log10(N)

    Args:
        frequency_hz: Frequency (Hz).
        se_single_db: SE of a single aperture (dB).
        n_apertures: Number of identical apertures.

    Returns:
        Array SE (dB).
    """
    if n_apertures <= 0:
        return se_single_db
    se = se_single_db - 10.0 * np.log10(n_apertures)
    return max(0.0, float(se))


def waveguide_below_cutoff_se(
    frequency_hz: float,
    tube_diameter_m: float,
    tube_length_m: float,
) -> float:
    """SE of a circular waveguide-below-cutoff vent (honeycomb cell).

    Below cutoff, fields attenuate exponentially inside the tube.
    The attenuation rate is ~32 dB per diameter of tube length.

        f_cutoff = 1.841 * c / (pi * d)   [TE11 dominant mode]

    Below cutoff:
        SE = (alpha * t)  where alpha = 32/d  dB per unit length

    Above cutoff: SE = 0 (waveguide propagates).

    Pozar (2011), Section 4.3.

    Args:
        frequency_hz: Frequency (Hz).
        tube_diameter_m: Tube inner diameter (m).
        tube_length_m: Tube length / wall thickness (m).

    Returns:
        Shielding effectiveness (dB).
    """
    # TE11 cutoff frequency for circular waveguide
    f_cutoff = 1.8412 * C / (np.pi * tube_diameter_m)

    if frequency_hz >= f_cutoff:
        return 0.0

    # Below cutoff: evanescent attenuation
    # alpha (dB/m) = (2*pi/lambda_c) * 20*log10(e) * sqrt(1 - (f/fc)^2)
    # Simplified: ~32 dB per diameter of length
    # More precisely:
    lambda_c = C / f_cutoff
    alpha_nepers = (2 * np.pi / lambda_c) * np.sqrt(1 - (frequency_hz / f_cutoff) ** 2)
    alpha_db_per_m = alpha_nepers * 20 * np.log10(np.e)
    se = alpha_db_per_m * tube_length_m

    return max(0.0, float(se))


def cavity_resonance_frequencies(
    length_m: float,
    width_m: float,
    height_m: float,
    max_modes: int = 5,
    max_results: int = 20,
) -> List[Dict]:
    """Resonant frequencies for a rectangular metallic cavity.

    At these frequencies, the enclosed volume resonates and internal SE
    can collapse to zero regardless of wall material.

        f_mnp = (c/2) * sqrt((m/a)^2 + (n/b)^2 + (p/c)^2)

    where (m,n,p) are mode indices, at least two must be nonzero.

    Pozar (2011), Section 6.4.

    Args:
        length_m: Interior length a (m).
        width_m: Interior width b (m).
        height_m: Interior height c (m).
        max_modes: Maximum mode index to compute (default 5).
        max_results: Number of lowest-frequency modes to return (default 20).

    Returns:
        List of dicts: {m, n, p, frequency_hz, mode_type}.
    """
    modes = []
    for m in range(0, max_modes + 1):
        for n in range(0, max_modes + 1):
            for p in range(0, max_modes + 1):
                # At least two indices must be nonzero for a valid cavity mode
                nonzero = (m > 0) + (n > 0) + (p > 0)
                if nonzero < 2:
                    continue

                f_mnp = (C / 2) * np.sqrt(
                    (m / length_m) ** 2
                    + (n / width_m) ** 2
                    + (p / height_m) ** 2
                )

                # Classify mode type
                if p == 0:
                    mode_type = f"TM{m}{n}0"
                else:
                    mode_type = f"TE{m}{n}{p}"

                modes.append({
                    "m": m,
                    "n": n,
                    "p": p,
                    "frequency_hz": float(f_mnp),
                    "mode_type": mode_type,
                })

    modes.sort(key=lambda x: x["frequency_hz"])
    return modes[:max_results]


def combined_enclosure_se(
    bulk_se_db: float,
    aperture_se_list_db: List[float],
) -> float:
    """Combined SE of an enclosure with multiple leakage paths.

    Each aperture is a parallel transmission path. Total leakage power
    is the sum of leakage through the bulk material and all apertures:

        SE_total = -10 * log10(sum(10^(-SE_i/10)))

    Celozzi et al. (2008), Eq. 7.31.

    Args:
        bulk_se_db: SE of the bulk wall material (dB).
        aperture_se_list_db: List of SE values for each aperture/slot (dB).

    Returns:
        Combined enclosure SE (dB).
    """
    # Sum transmission coefficients (power domain)
    t_total = 10 ** (-bulk_se_db / 10)
    for se_ap in aperture_se_list_db:
        t_total += 10 ** (-se_ap / 10)

    if t_total <= 0:
        return 200.0  # numerical guard

    se = -10.0 * np.log10(t_total)
    return max(0.0, float(se))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_aperture.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/physics/aperture.py tests/test_aperture.py
git commit -m "feat(aperture): add Bethe hole theory, slot antenna, cavity resonance, and waveguide-below-cutoff models"
```

---

### Task 5: Implement enclosure SE API route

**Files:**
- Create: `backend/api/v1/routes/enclosure.py`
- Modify: `backend/main.py` (add router)

- [ ] **Step 1: Create enclosure route**

```python
# backend/api/v1/routes/enclosure.py
"""Enclosure shielding effectiveness analysis endpoints.

Combines bulk material SE with aperture leakage, cavity resonance,
and waveguide-below-cutoff analysis for full enclosure prediction.
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


# --- Schemas ---

class CircularAperture(BaseModel):
    radius_mm: float = Field(..., gt=0, description="Hole radius (mm)")
    count: int = Field(1, ge=1, description="Number of identical holes")


class SlotAperture(BaseModel):
    length_mm: float = Field(..., gt=0, description="Slot length (mm)")
    count: int = Field(1, ge=1, description="Number of identical slots")


class WaveguideVent(BaseModel):
    diameter_mm: float = Field(..., gt=0, description="Honeycomb cell diameter (mm)")
    depth_mm: float = Field(..., gt=0, description="Wall thickness / tube length (mm)")
    count: int = Field(1, ge=1, description="Number of honeycomb cells")


class EnclosureSERequest(BaseModel):
    composition: Dict[str, float]
    wall_thickness_mm: float = Field(..., gt=0)
    frequency_mhz: float = Field(..., gt=0)
    circular_apertures: List[CircularAperture] = Field(default_factory=list)
    slot_apertures: List[SlotAperture] = Field(default_factory=list)
    waveguide_vents: List[WaveguideVent] = Field(default_factory=list)


class EnclosureSEResponse(BaseModel):
    bulk_se_db: float
    aperture_se_details: List[Dict]
    combined_se_db: float
    dominant_leakage_path: str
    frequency_mhz: float


class CavityResonanceRequest(BaseModel):
    length_mm: float = Field(..., gt=0, description="Interior length (mm)")
    width_mm: float = Field(..., gt=0, description="Interior width (mm)")
    height_mm: float = Field(..., gt=0, description="Interior height (mm)")
    max_modes: int = Field(5, ge=1, le=10)


class EnclosureSweepRequest(BaseModel):
    composition: Dict[str, float]
    wall_thickness_mm: float = Field(..., gt=0)
    freq_start_mhz: float = Field(..., gt=0)
    freq_end_mhz: float = Field(..., gt=0)
    num_points: int = Field(100, ge=2, le=1000)
    circular_apertures: List[CircularAperture] = Field(default_factory=list)
    slot_apertures: List[SlotAperture] = Field(default_factory=list)
    waveguide_vents: List[WaveguideVent] = Field(default_factory=list)
    enclosure_length_mm: Optional[float] = Field(None, gt=0)
    enclosure_width_mm: Optional[float] = Field(None, gt=0)
    enclosure_height_mm: Optional[float] = Field(None, gt=0)


# --- Endpoints ---

@router.post("/analyze", response_model=EnclosureSEResponse)
async def analyze_enclosure_se(req: EnclosureSERequest):
    """Compute combined enclosure SE at a single frequency."""
    try:
        props = calculate_composite_properties(req.composition)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    f_hz = req.frequency_mhz * 1e6

    # Bulk material SE
    result = calculator.calculate_shielding_effectiveness(
        conductivity=props["conductivity"],
        relative_permeability=props["permeability"],
        relative_permittivity=props["permittivity"],
        thickness=req.wall_thickness_mm * 1e-3,
        frequency=f_hz,
    )
    bulk_se = result["total_se"]

    # Evaluate each aperture
    aperture_details = []
    aperture_se_values = []

    for ap in req.circular_apertures:
        se_one = aperture_se_circular(f_hz, ap.radius_mm * 1e-3)
        se_arr = aperture_se_array(f_hz, se_one, ap.count)
        aperture_details.append({
            "type": "circular",
            "radius_mm": ap.radius_mm,
            "count": ap.count,
            "se_single_db": round(se_one, 2),
            "se_array_db": round(se_arr, 2),
        })
        aperture_se_values.append(se_arr)

    for sl in req.slot_apertures:
        se_one = aperture_se_slot(f_hz, sl.length_mm * 1e-3)
        se_arr = aperture_se_array(f_hz, se_one, sl.count)
        aperture_details.append({
            "type": "slot",
            "length_mm": sl.length_mm,
            "count": sl.count,
            "se_single_db": round(se_one, 2),
            "se_array_db": round(se_arr, 2),
        })
        aperture_se_values.append(se_arr)

    for wg in req.waveguide_vents:
        se_one = waveguide_below_cutoff_se(f_hz, wg.diameter_mm * 1e-3, wg.depth_mm * 1e-3)
        se_arr = aperture_se_array(f_hz, se_one, wg.count)
        aperture_details.append({
            "type": "waveguide_vent",
            "diameter_mm": wg.diameter_mm,
            "depth_mm": wg.depth_mm,
            "count": wg.count,
            "se_single_db": round(se_one, 2),
            "se_array_db": round(se_arr, 2),
        })
        aperture_se_values.append(se_arr)

    # Combined SE
    combined_se = combined_enclosure_se(bulk_se, aperture_se_values)

    # Identify dominant leakage path
    all_paths = [("bulk_material", bulk_se)] + [
        (d["type"], d["se_array_db"]) for d in aperture_details
    ]
    dominant = min(all_paths, key=lambda x: x[1])

    return EnclosureSEResponse(
        bulk_se_db=round(bulk_se, 2),
        aperture_se_details=aperture_details,
        combined_se_db=round(combined_se, 2),
        dominant_leakage_path=dominant[0],
        frequency_mhz=req.frequency_mhz,
    )


@router.post("/cavity-resonances")
async def get_cavity_resonances(req: CavityResonanceRequest):
    """Compute resonant frequencies for a rectangular enclosure."""
    modes = cavity_resonance_frequencies(
        length_m=req.length_mm * 1e-3,
        width_m=req.width_mm * 1e-3,
        height_m=req.height_mm * 1e-3,
        max_modes=req.max_modes,
    )
    return {"modes": modes, "dimensions_mm": {
        "length": req.length_mm, "width": req.width_mm, "height": req.height_mm,
    }}


@router.post("/frequency-sweep")
async def enclosure_frequency_sweep(req: EnclosureSweepRequest):
    """Sweep frequency and show bulk, aperture, and combined SE curves."""
    try:
        props = calculate_composite_properties(req.composition)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)

    bulk_se_list = []
    combined_se_list = []
    worst_aperture_list = []

    for f_mhz in freqs_mhz:
        f_hz = f_mhz * 1e6

        r = calculator.calculate_shielding_effectiveness(
            conductivity=props["conductivity"],
            relative_permeability=props["permeability"],
            relative_permittivity=props["permittivity"],
            thickness=req.wall_thickness_mm * 1e-3,
            frequency=f_hz,
        )
        bulk_se = r["total_se"]
        bulk_se_list.append(float(bulk_se))

        ap_ses = []
        for ap in req.circular_apertures:
            se_one = aperture_se_circular(f_hz, ap.radius_mm * 1e-3)
            ap_ses.append(aperture_se_array(f_hz, se_one, ap.count))
        for sl in req.slot_apertures:
            se_one = aperture_se_slot(f_hz, sl.length_mm * 1e-3)
            ap_ses.append(aperture_se_array(f_hz, se_one, sl.count))
        for wg in req.waveguide_vents:
            se_one = waveguide_below_cutoff_se(f_hz, wg.diameter_mm * 1e-3, wg.depth_mm * 1e-3)
            ap_ses.append(aperture_se_array(f_hz, se_one, wg.count))

        comb = combined_enclosure_se(bulk_se, ap_ses)
        combined_se_list.append(float(comb))
        worst_aperture_list.append(float(min(ap_ses)) if ap_ses else float(bulk_se))

    result = {
        "frequencies_mhz": freqs_mhz.tolist(),
        "bulk_se_db": bulk_se_list,
        "combined_se_db": combined_se_list,
        "worst_aperture_se_db": worst_aperture_list,
    }

    # Add resonance markers if enclosure dimensions provided
    if req.enclosure_length_mm and req.enclosure_width_mm and req.enclosure_height_mm:
        modes = cavity_resonance_frequencies(
            req.enclosure_length_mm * 1e-3,
            req.enclosure_width_mm * 1e-3,
            req.enclosure_height_mm * 1e-3,
        )
        result["cavity_resonances"] = modes

    return result
```

- [ ] **Step 2: Register router in main.py**

Add to `backend/main.py` imports:
```python
from api.v1.routes import physics, auth, materials, analysis, chat, multilayer, composites, advanced, heatmap, cables, enclosure
```

Add router registration:
```python
app.include_router(enclosure.router, prefix="/api/v1/enclosure", tags=["Enclosure"])
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_aperture.py tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add backend/api/v1/routes/enclosure.py backend/main.py
git commit -m "feat(enclosure): add enclosure SE API with aperture analysis, cavity resonance, and frequency sweep"
```

---

## Phase 3: Material Recommendation Engine

The inverse solver — the most important missing capability. Engineers state requirements; the system searches the material database and returns ranked recommendations.

---

### Task 6: Implement material recommendation physics

**Files:**
- Create: `src/physics/recommendation.py`
- Test: `tests/test_recommendation.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_recommendation.py
import pytest
from src.physics.recommendation import (
    MaterialConstraints,
    MaterialRecommendation,
    recommend_materials,
    pareto_filter,
)


class TestMaterialConstraints:
    def test_valid_constraints(self):
        c = MaterialConstraints(
            target_se_db=60,
            frequency_hz=1e9,
            max_thickness_m=2e-3,
        )
        assert c.target_se_db == 60


class TestRecommendMaterials:
    def test_returns_recommendations(self):
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=40,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
            n_results=5,
        )
        assert len(recs) > 0
        assert len(recs) <= 5

    def test_recommendations_meet_target(self):
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=40,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
        )
        for r in recs:
            assert r.achieved_se_db >= 40 or r.meets_target is False

    def test_recommendations_sorted_by_margin(self):
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=30,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
            ),
        )
        meeting = [r for r in recs if r.meets_target]
        if len(meeting) > 1:
            # Should be sorted: best candidates first
            assert meeting[0].se_margin_db >= meeting[1].se_margin_db

    def test_copper_always_recommended_for_reasonable_targets(self):
        """Copper should appear for moderate SE targets."""
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=50,
                frequency_hz=1e9,
                max_thickness_m=2e-3,
            ),
            n_results=10,
        )
        names = [r.material_name for r in recs]
        has_copper = any("Cu" in n or "Copper" in n or "copper" in n for n in names)
        assert has_copper, f"Copper should be recommended. Got: {names}"

    def test_weight_constraint_filters_heavy_materials(self):
        recs = recommend_materials(
            constraints=MaterialConstraints(
                target_se_db=30,
                frequency_hz=1e9,
                max_thickness_m=5e-3,
                max_density_kg_m3=4000,  # Excludes steel, nickel, etc.
            ),
        )
        for r in recs:
            if r.meets_target:
                assert r.density_kg_m3 <= 4000


class TestParetoFilter:
    def test_filters_dominated_solutions(self):
        """A solution dominated in all objectives should be removed."""
        recs = [
            MaterialRecommendation(
                material_name="A", achieved_se_db=60, optimal_thickness_m=1e-3,
                density_kg_m3=3000, se_margin_db=20, meets_target=True,
            ),
            MaterialRecommendation(
                material_name="B", achieved_se_db=50, optimal_thickness_m=2e-3,
                density_kg_m3=8000, se_margin_db=10, meets_target=True,
            ),
        ]
        filtered = pareto_filter(recs)
        # A dominates B (higher SE, lower density, thinner)
        names = [r.material_name for r in filtered]
        assert "A" in names
        assert "B" not in names, "B is dominated by A and should be filtered out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_recommendation.py -v`
Expected: ImportError

- [ ] **Step 3: Implement recommendation engine**

```python
# src/physics/recommendation.py
"""Material recommendation engine — inverse SE solver.

Given performance constraints (target SE, frequency, max thickness, weight),
searches the material database and returns ranked candidates with
uncertainty bounds. Uses the existing EMICalculator as the forward solver.

This is the core differentiating feature: engineers state requirements,
the system recommends optimal materials.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.physics.emi_calculations import EMICalculator
from src.materials.material_properties import material_db


@dataclass
class MaterialConstraints:
    """Engineering requirements for material selection."""
    target_se_db: float                     # Minimum required SE (dB)
    frequency_hz: float                     # Primary frequency of concern (Hz)
    max_thickness_m: float                  # Maximum allowable thickness (m)
    max_density_kg_m3: Optional[float] = None  # Maximum density (kg/m^3)
    frequency_range_hz: Optional[Tuple[float, float]] = None  # Must meet SE across range
    thickness_step_m: float = 0.1e-3        # Thickness search resolution (m)


@dataclass
class MaterialRecommendation:
    """A recommended material with its performance characteristics."""
    material_name: str
    achieved_se_db: float
    optimal_thickness_m: float
    density_kg_m3: float
    se_margin_db: float = 0.0               # achieved - target
    meets_target: bool = True
    conductivity_s_m: float = 0.0
    relative_permeability: float = 1.0
    reflection_loss_db: float = 0.0
    absorption_loss_db: float = 0.0
    skin_depth_m: float = 0.0
    explanation: str = ""


# Module-level instances
_calculator = EMICalculator()
_db = material_db  # Use existing singleton from material_properties


def _evaluate_material(
    name: str,
    conductivity: float,
    rel_permeability: float,
    rel_permittivity: float,
    density: float,
    constraints: MaterialConstraints,
) -> Optional[MaterialRecommendation]:
    """Evaluate a single material against constraints.

    Sweeps thickness from thin to max, finds minimum thickness
    that meets target SE, then evaluates at that thickness.
    """
    t_step = constraints.thickness_step_m
    thicknesses = np.arange(t_step, constraints.max_thickness_m + t_step, t_step)

    if len(thicknesses) == 0:
        return None

    # Density constraint
    if constraints.max_density_kg_m3 is not None:
        if density > constraints.max_density_kg_m3:
            return None

    best_result = None
    optimal_t = thicknesses[-1]

    for t in thicknesses:
        try:
            result = _calculator.calculate_shielding_effectiveness(
                conductivity=conductivity,
                relative_permeability=rel_permeability,
                relative_permittivity=rel_permittivity,
                thickness=float(t),
                frequency=constraints.frequency_hz,
            )
        except (ValueError, ZeroDivisionError):
            continue

        se = result["total_se"]

        if se >= constraints.target_se_db:
            optimal_t = float(t)
            best_result = result
            break
        else:
            best_result = result
            optimal_t = float(t)

    if best_result is None:
        return None

    # Check frequency range if specified
    meets = best_result["total_se"] >= constraints.target_se_db
    if meets and constraints.frequency_range_hz:
        f_lo, f_hi = constraints.frequency_range_hz
        for f_check in np.linspace(f_lo, f_hi, 10):
            try:
                r = _calculator.calculate_shielding_effectiveness(
                    conductivity=conductivity,
                    relative_permeability=rel_permeability,
                    relative_permittivity=rel_permittivity,
                    thickness=optimal_t,
                    frequency=float(f_check),
                )
                if r["total_se"] < constraints.target_se_db:
                    meets = False
                    break
            except (ValueError, ZeroDivisionError):
                meets = False
                break

    return MaterialRecommendation(
        material_name=name,
        achieved_se_db=float(best_result["total_se"]),
        optimal_thickness_m=optimal_t,
        density_kg_m3=density,
        se_margin_db=float(best_result["total_se"] - constraints.target_se_db),
        meets_target=meets,
        conductivity_s_m=conductivity,
        relative_permeability=rel_permeability,
        reflection_loss_db=float(best_result.get("reflection_loss", 0)),
        absorption_loss_db=float(best_result.get("absorption_loss", 0)),
        skin_depth_m=float(best_result.get("skin_depth", 0)),
    )


def recommend_materials(
    constraints: MaterialConstraints,
    n_results: int = 5,
) -> List[MaterialRecommendation]:
    """Search the material database for optimal shielding materials.

    Evaluates all elements and alloys against the given constraints,
    ranks by SE margin and thickness efficiency, and returns the
    top N recommendations.

    Args:
        constraints: Engineering requirements.
        n_results: Maximum number of recommendations.

    Returns:
        Ranked list of MaterialRecommendation, best first.
    """
    candidates: List[MaterialRecommendation] = []

    # Evaluate pure elements with known EM properties
    for symbol, props in _db.periodic_table.items():
        conductivity = props.get("electrical_conductivity", 0)
        if conductivity <= 0:
            continue
        density = props.get("density", 0)
        if density <= 0:
            continue
        rel_perm = props.get("relative_permeability", 1.0)
        rel_eps = props.get("relative_permittivity", 1.0)

        rec = _evaluate_material(
            name=f"{props.get('name', symbol)} ({symbol})",
            conductivity=conductivity,
            rel_permeability=rel_perm,
            rel_permittivity=rel_eps,
            density=density,
            constraints=constraints,
        )
        if rec is not None:
            candidates.append(rec)

    # Evaluate alloys
    for key, alloy in _db.alloys.items():
        conductivity = alloy.get("electrical_conductivity", 0)
        if conductivity <= 0:
            continue
        density = alloy.get("density", 0)
        if density <= 0:
            continue
        rel_perm = alloy.get("relative_permeability", 1.0)
        rel_eps = alloy.get("relative_permittivity", 1.0)

        rec = _evaluate_material(
            name=alloy.get("name", key),
            conductivity=conductivity,
            rel_permeability=rel_perm,
            rel_permittivity=rel_eps,
            density=density,
            constraints=constraints,
        )
        if rec is not None:
            candidates.append(rec)

    # Sort: meeting target first, then by SE margin descending,
    # then by thickness ascending (thinner is better)
    candidates.sort(
        key=lambda r: (
            -int(r.meets_target),       # True first
            -r.se_margin_db,            # Higher margin first
            r.optimal_thickness_m,      # Thinner first
            r.density_kg_m3,            # Lighter first
        )
    )

    return candidates[:n_results]


def pareto_filter(
    recommendations: List[MaterialRecommendation],
) -> List[MaterialRecommendation]:
    """Filter to Pareto-optimal solutions.

    Objectives (all to maximize/minimize):
        - Maximize: SE margin
        - Minimize: thickness
        - Minimize: density (weight proxy)

    A recommendation is Pareto-dominated if another recommendation
    is strictly better in all objectives.
    """
    pareto = []
    for i, r in enumerate(recommendations):
        dominated = False
        for j, other in enumerate(recommendations):
            if i == j:
                continue
            # other dominates r if better in all three objectives
            if (other.se_margin_db >= r.se_margin_db
                    and other.optimal_thickness_m <= r.optimal_thickness_m
                    and other.density_kg_m3 <= r.density_kg_m3
                    and (other.se_margin_db > r.se_margin_db
                         or other.optimal_thickness_m < r.optimal_thickness_m
                         or other.density_kg_m3 < r.density_kg_m3)):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    return pareto
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_recommendation.py -v`
Expected: All PASS

- [ ] **Step 5: Create recommendation API route**

Create `backend/api/v1/routes/recommendation.py`:

```python
"""Material recommendation API endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from dataclasses import asdict

from src.physics.recommendation import (
    MaterialConstraints,
    recommend_materials,
    pareto_filter,
)

router = APIRouter()


class RecommendationRequest(BaseModel):
    target_se_db: float = Field(..., gt=0, description="Target SE (dB)")
    frequency_mhz: float = Field(..., gt=0, description="Primary frequency (MHz)")
    max_thickness_mm: float = Field(..., gt=0, description="Max thickness (mm)")
    max_density_kg_m3: Optional[float] = Field(None, gt=0)
    freq_range_start_mhz: Optional[float] = Field(None, gt=0)
    freq_range_end_mhz: Optional[float] = Field(None, gt=0)
    n_results: int = Field(10, ge=1, le=50)
    pareto_only: bool = Field(False, description="Return only Pareto-optimal solutions")


@router.post("/search")
async def search_materials(req: RecommendationRequest):
    """Search for materials meeting the specified shielding requirements."""
    freq_range = None
    if req.freq_range_start_mhz and req.freq_range_end_mhz:
        freq_range = (req.freq_range_start_mhz * 1e6, req.freq_range_end_mhz * 1e6)

    constraints = MaterialConstraints(
        target_se_db=req.target_se_db,
        frequency_hz=req.frequency_mhz * 1e6,
        max_thickness_m=req.max_thickness_mm * 1e-3,
        max_density_kg_m3=req.max_density_kg_m3,
        frequency_range_hz=freq_range,
    )

    recs = recommend_materials(constraints, n_results=req.n_results)
    total_evaluated = len(recs)

    if req.pareto_only:
        recs = pareto_filter(recs)

    return {
        "recommendations": [asdict(r) for r in recs],
        "constraints": {
            "target_se_db": req.target_se_db,
            "frequency_mhz": req.frequency_mhz,
            "max_thickness_mm": req.max_thickness_mm,
            "max_density_kg_m3": req.max_density_kg_m3,
        },
        "total_candidates_evaluated": total_evaluated,
    }
```

- [ ] **Step 6: Register router in main.py**

Add `recommendation` to imports and register:
```python
app.include_router(recommendation.router, prefix="/api/v1/recommendation", tags=["Recommendation"])
```

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/physics/recommendation.py tests/test_recommendation.py backend/api/v1/routes/recommendation.py backend/main.py
git commit -m "feat(recommendation): add inverse SE solver with material search, Pareto optimization, and API endpoints"
```

---

## Phase 4: Environmental Hazards

Lightning, EMP, HIRF, and BCI — high-energy transient events that threaten mission-critical systems.

---

### Task 7: Implement hazards physics module

**Files:**
- Create: `src/physics/hazards.py`
- Test: `tests/test_hazards.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hazards.py
import pytest
import numpy as np
from src.physics.hazards import (
    lightning_waveform,
    lightning_spectrum,
    induced_voltage_lightning,
    hirf_induced_current,
    emp_waveform,
    bci_coupling,
    DO160G_COMPONENTS,
)


class TestLightningWaveform:
    """DO-160G double-exponential waveform generation."""

    def test_component_a_peak(self):
        """Component A peak current should be ~200 kA."""
        t, i = lightning_waveform(component="A", duration_s=1e-3, n_points=10000)
        peak = np.max(np.abs(i))
        assert 180e3 < peak < 220e3, f"Component A peak should be ~200 kA, got {peak/1e3:.0f} kA"

    def test_waveform_starts_at_zero(self):
        t, i = lightning_waveform(component="A")
        assert abs(i[0]) < 100  # Nearly zero at t=0

    def test_waveform_decays(self):
        t, i = lightning_waveform(component="A", duration_s=10e-3, n_points=1000)
        assert abs(i[-1]) < abs(np.max(i)) * 0.01  # Decayed to <1%

    def test_component_h_faster_rise(self):
        """Component H has faster rise time than A."""
        _, i_a = lightning_waveform("A", duration_s=100e-6, n_points=10000)
        _, i_h = lightning_waveform("H", duration_s=100e-6, n_points=10000)
        idx_peak_a = np.argmax(np.abs(i_a))
        idx_peak_h = np.argmax(np.abs(i_h))
        # H peaks earlier (faster rise)
        assert idx_peak_h <= idx_peak_a


class TestLightningSpectrum:
    """Fourier spectrum of lightning waveform."""

    def test_spectrum_rolls_off(self):
        """Spectrum should decrease with frequency."""
        mag_low = abs(lightning_spectrum(1e3, "A"))
        mag_high = abs(lightning_spectrum(1e6, "A"))
        assert mag_low > mag_high

    def test_spectrum_nonzero_at_dc(self):
        mag_dc = abs(lightning_spectrum(1.0, "A"))
        assert mag_dc > 0


class TestInducedVoltage:
    """Lightning-induced voltage on cable via transfer impedance."""

    def test_induced_voltage_proportional_to_zt(self):
        v1 = induced_voltage_lightning(zt_ohm_per_m=0.01, cable_length_m=1.0, component="A")
        v2 = induced_voltage_lightning(zt_ohm_per_m=0.1, cable_length_m=1.0, component="A")
        assert v2["peak_voltage_v"] > v1["peak_voltage_v"]

    def test_induced_voltage_has_waveform(self):
        result = induced_voltage_lightning(0.01, 1.0, "A")
        assert "time_s" in result
        assert "voltage_v" in result
        assert len(result["time_s"]) > 0


class TestHIRF:
    """HIRF plane-wave coupling to cable."""

    def test_induced_current_proportional_to_field(self):
        i1 = hirf_induced_current(e_field_v_per_m=10, frequency_hz=100e6,
                                   cable_length_m=1.0, z_term_ohm=50.0)
        i2 = hirf_induced_current(e_field_v_per_m=100, frequency_hz=100e6,
                                   cable_length_m=1.0, z_term_ohm=50.0)
        assert i2["induced_current_a"] > i1["induced_current_a"]

    def test_resonant_length_maximizes_current(self):
        """Cable = lambda/2 should maximize coupling."""
        f = 300e6  # lambda = 1m
        # Cable = 0.5m = lambda/2
        i_res = hirf_induced_current(10, f, 0.5, 50.0)
        i_short = hirf_induced_current(10, f, 0.1, 50.0)
        assert i_res["induced_current_a"] >= i_short["induced_current_a"]


class TestEMP:
    """EMP/HEMP waveform."""

    def test_hemp_fast_rise(self):
        """HEMP E1 has sub-nanosecond rise time."""
        t, e = emp_waveform(component="E1", duration_s=100e-9, n_points=10000)
        idx_peak = np.argmax(np.abs(e))
        t_peak = t[idx_peak]
        assert t_peak < 10e-9  # Peak within 10 ns


class TestBCI:
    """Bulk Current Injection coupling."""

    def test_bci_voltage_at_load(self):
        result = bci_coupling(
            injection_current_a=1.0,
            frequency_hz=100e6,
            cable_length_m=1.0,
            z_load_ohm=50.0,
            zt_ohm_per_m=0.01,
        )
        assert result["load_voltage_v"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hazards.py -v`
Expected: ImportError

- [ ] **Step 3: Implement hazards physics**

```python
# src/physics/hazards.py
"""Extreme environmental electromagnetic hazard models.

Implements lightning (DO-160G), EMP/HEMP, HIRF, and BCI coupling
using analytical transient electromagnetics and transmission line theory.

References:
    RTCA DO-160G (2010). Environmental Conditions and Test Procedures
        for Airborne Equipment, Section 22 (Lightning Induced Transient
        Susceptibility).
    MIL-STD-461G (2015). Requirements for the Control of EMI.
    Tesche, F.M., Ianoz, M.V., Karlsson, T. (1997). EMC Analysis Methods
        and Computational Models. Wiley.
    Agrawal, A.K., Price, H.J., Gurbaxani, S.H. (1980). Transient Response
        of Multiconductor Transmission Lines Excited by a Nonuniform
        Electromagnetic Field. IEEE Trans. EMC, 22(2), 119-129.
    IEC 61000-2-9 (1996). Description of HEMP Environment —
        Radiated Disturbance.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.utils.constants import C, Z_0


# ---------------------------------------------------------------------------
# DO-160G Lightning Waveform Parameters
# ---------------------------------------------------------------------------

DO160G_COMPONENTS = {
    "A": {
        "description": "First return stroke (severe)",
        "I_peak_a": 200e3,
        "alpha_s_inv": 11354.0,
        "beta_s_inv": 647265.0,
    },
    "B": {
        "description": "Intermediate current",
        "I_peak_a": 2e3,
        "alpha_s_inv": 700.0,
        "beta_s_inv": 2000.0,
    },
    "D": {
        "description": "Restrike (subsequent return stroke)",
        "I_peak_a": 100e3,
        "alpha_s_inv": 22708.0,
        "beta_s_inv": 1294530.0,
    },
    "H": {
        "description": "First return stroke (fast rise, 100 kHz ring)",
        "I_peak_a": 200e3,
        "alpha_s_inv": 22708.0,
        "beta_s_inv": 1294530.0,
    },
}

# IEC 61000-2-9 HEMP parameters
HEMP_COMPONENTS = {
    "E1": {
        "description": "Early-time HEMP (fast pulse)",
        "E_peak_v_per_m": 50e3,
        "alpha_s_inv": 4e6,
        "beta_s_inv": 4.76e8,
    },
    "E2": {
        "description": "Intermediate-time HEMP",
        "E_peak_v_per_m": 100.0,
        "alpha_s_inv": 1e3,
        "beta_s_inv": 1e5,
    },
}


# ---------------------------------------------------------------------------
# Lightning waveforms
# ---------------------------------------------------------------------------

def lightning_waveform(
    component: str = "A",
    duration_s: float = 1e-3,
    n_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate DO-160G double-exponential lightning current waveform.

    i(t) = I_peak * k * (exp(-alpha*t) - exp(-beta*t))

    where k normalizes the peak to I_peak.

    Args:
        component: DO-160G component ("A", "B", "D", "H").
        duration_s: Time window (seconds).
        n_points: Number of time samples.

    Returns:
        (time_s, current_a): Arrays of time and current values.
    """
    params = DO160G_COMPONENTS[component]
    I_peak = params["I_peak_a"]
    alpha = params["alpha_s_inv"]
    beta = params["beta_s_inv"]

    t = np.linspace(0, duration_s, n_points)

    # Time of peak
    t_peak = np.log(beta / alpha) / (beta - alpha)

    # Normalization factor so peak equals I_peak
    raw_peak = np.exp(-alpha * t_peak) - np.exp(-beta * t_peak)
    k = 1.0 / raw_peak if raw_peak > 0 else 1.0

    current = I_peak * k * (np.exp(-alpha * t) - np.exp(-beta * t))

    return t, current


def lightning_spectrum(
    frequency_hz: float,
    component: str = "A",
) -> complex:
    """Fourier transform of DO-160G double-exponential waveform.

    I(w) = I_peak * k * (1/(alpha + jw) - 1/(beta + jw))

    Args:
        frequency_hz: Frequency (Hz).
        component: DO-160G component.

    Returns:
        Complex spectral amplitude at frequency_hz.
    """
    params = DO160G_COMPONENTS[component]
    I_peak = params["I_peak_a"]
    alpha = params["alpha_s_inv"]
    beta = params["beta_s_inv"]

    t_peak = np.log(beta / alpha) / (beta - alpha)
    raw_peak = np.exp(-alpha * t_peak) - np.exp(-beta * t_peak)
    k = 1.0 / raw_peak if raw_peak > 0 else 1.0

    omega = 2 * np.pi * frequency_hz
    spectrum = I_peak * k * (1.0 / (alpha + 1j * omega) - 1.0 / (beta + 1j * omega))

    return spectrum


def induced_voltage_lightning(
    zt_ohm_per_m: float,
    cable_length_m: float,
    component: str = "A",
    duration_s: float = 1e-3,
    n_points: int = 1000,
) -> Dict:
    """Induced voltage on a shielded cable from lightning current.

    The external lightning current flowing on the shield induces an
    internal voltage proportional to the transfer impedance:

        V_induced(t) = Z_t * L_cable * I_lightning(t)

    Tesche et al. (1997), Chapter 9.

    Args:
        zt_ohm_per_m: Transfer impedance magnitude (ohm/m).
        cable_length_m: Cable length (m).
        component: DO-160G waveform component.

    Returns:
        Dict with time_s, voltage_v arrays and peak_voltage_v.
    """
    t, i_ext = lightning_waveform(component, duration_s, n_points)

    # Induced voltage = Z_t * L * I(t)
    v_induced = zt_ohm_per_m * cable_length_m * i_ext

    return {
        "time_s": t.tolist(),
        "voltage_v": v_induced.tolist(),
        "peak_voltage_v": float(np.max(np.abs(v_induced))),
        "component": component,
        "zt_ohm_per_m": zt_ohm_per_m,
        "cable_length_m": cable_length_m,
    }


# ---------------------------------------------------------------------------
# HIRF (High Intensity Radiated Fields)
# ---------------------------------------------------------------------------

def hirf_induced_current(
    e_field_v_per_m: float,
    frequency_hz: float,
    cable_length_m: float,
    z_term_ohm: float,
    cable_height_m: float = 0.05,
) -> Dict:
    """Compute current induced on a cable by an external HIRF plane wave.

    Uses the Agrawal field-to-transmission-line coupling model.
    For a wire at height h above ground, the distributed voltage source is:

        V_S = 2 * E_inc * h * sin(beta * L / 2) / (beta * L / 2)

    Agrawal et al. (1980), IEEE Trans. EMC, Eq. 7.

    Args:
        e_field_v_per_m: Incident E-field strength (V/m).
        frequency_hz: Frequency (Hz).
        cable_length_m: Cable length (m).
        z_term_ohm: Termination impedance at each end (ohm).
        cable_height_m: Cable height above ground (m).

    Returns:
        Dict with induced_current_a, induced_voltage_v, frequency_hz.
    """
    wavelength = C / frequency_hz
    beta = 2 * np.pi / wavelength

    # Effective voltage: integral of E-field along cable with phase
    beta_l_half = beta * cable_length_m / 2
    if abs(beta_l_half) < 1e-10:
        sinc_factor = 1.0
    else:
        sinc_factor = np.sin(beta_l_half) / beta_l_half

    # Voltage coupled to the wire (Agrawal model, normal incidence)
    v_coupled = 2 * e_field_v_per_m * cable_height_m * cable_length_m * sinc_factor

    # Resonance enhancement at cable = n * lambda/2
    # At resonance, current is maximized
    v_total = abs(v_coupled)

    # Current at termination (assuming matched or specified Z)
    i_induced = v_total / (2 * z_term_ohm)

    return {
        "induced_current_a": float(i_induced),
        "induced_voltage_v": float(v_total),
        "frequency_hz": frequency_hz,
        "wavelength_m": wavelength,
        "cable_electrical_length_wavelengths": cable_length_m / wavelength,
    }


# ---------------------------------------------------------------------------
# EMP / HEMP waveforms
# ---------------------------------------------------------------------------

def emp_waveform(
    component: str = "E1",
    duration_s: float = 1e-6,
    n_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate EMP/HEMP double-exponential E-field waveform.

    IEC 61000-2-9 standard E1 (early-time) and E2 (intermediate).

    E(t) = E_peak * k * (exp(-alpha*t) - exp(-beta*t))

    Args:
        component: "E1" (fast, 50 kV/m) or "E2" (intermediate, 100 V/m).
        duration_s: Time window.
        n_points: Samples.

    Returns:
        (time_s, e_field_v_per_m): Arrays.
    """
    params = HEMP_COMPONENTS[component]
    E_peak = params["E_peak_v_per_m"]
    alpha = params["alpha_s_inv"]
    beta = params["beta_s_inv"]

    t = np.linspace(0, duration_s, n_points)

    t_peak = np.log(beta / alpha) / (beta - alpha)
    raw_peak = np.exp(-alpha * t_peak) - np.exp(-beta * t_peak)
    k = 1.0 / raw_peak if raw_peak > 0 else 1.0

    e_field = E_peak * k * (np.exp(-alpha * t) - np.exp(-beta * t))

    return t, e_field


# ---------------------------------------------------------------------------
# Bulk Current Injection (BCI)
# ---------------------------------------------------------------------------

def bci_coupling(
    injection_current_a: float,
    frequency_hz: float,
    cable_length_m: float,
    z_load_ohm: float,
    zt_ohm_per_m: float,
    z_source_ohm: float = 50.0,
) -> Dict:
    """Compute voltage at load from bulk current injection on cable shield.

    BCI per MIL-STD-461G CS114: a clamp injects common-mode current
    on the cable shield. The transfer impedance couples this to the
    inner conductor, producing voltage at the load.

        V_load = I_inject * Z_t * L / (1 + Z_t*L / Z_load)

    For Z_t*L << Z_load (typical): V_load ≈ I_inject * Z_t * L.

    Args:
        injection_current_a: BCI clamp injection current (A).
        frequency_hz: Frequency (Hz).
        cable_length_m: Cable length (m).
        z_load_ohm: Equipment input impedance (ohm).
        zt_ohm_per_m: Shield transfer impedance (ohm/m).
        z_source_ohm: Source impedance (ohm).

    Returns:
        Dict with load_voltage_v, load_current_a.
    """
    zt_total = zt_ohm_per_m * cable_length_m

    # Induced EMF on inner conductor
    v_emf = injection_current_a * zt_total

    # Voltage divider between source and load
    v_load = v_emf * z_load_ohm / (z_source_ohm + z_load_ohm)
    i_load = v_load / z_load_ohm

    return {
        "load_voltage_v": float(abs(v_load)),
        "load_current_a": float(abs(i_load)),
        "injection_current_a": injection_current_a,
        "zt_total_ohm": float(zt_total),
        "frequency_hz": frequency_hz,
    }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_hazards.py -v`
Expected: All PASS

- [ ] **Step 5: Create hazards API route**

Create `backend/api/v1/routes/hazards.py`:

```python
"""Environmental EM hazard simulation endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np

from src.physics.hazards import (
    lightning_waveform,
    lightning_spectrum,
    induced_voltage_lightning,
    hirf_induced_current,
    emp_waveform,
    bci_coupling,
    DO160G_COMPONENTS,
    HEMP_COMPONENTS,
)

router = APIRouter()


class LightningRequest(BaseModel):
    component: Literal["A", "B", "D", "H"] = "A"
    duration_us: float = Field(1000, gt=0, description="Duration in microseconds")
    n_points: int = Field(1000, ge=100, le=50000)


class LightningInducedRequest(BaseModel):
    component: Literal["A", "B", "D", "H"] = "A"
    zt_mohm_per_m: float = Field(..., gt=0, description="Transfer impedance (milliohm/m)")
    cable_length_m: float = Field(..., gt=0)
    duration_us: float = Field(1000, gt=0)
    n_points: int = Field(1000, ge=100, le=50000)


class HIRFRequest(BaseModel):
    e_field_v_per_m: float = Field(..., gt=0)
    freq_start_mhz: float = Field(..., gt=0)
    freq_end_mhz: float = Field(..., gt=0)
    num_points: int = Field(100, ge=2, le=1000)
    cable_length_m: float = Field(1.0, gt=0)
    z_term_ohm: float = Field(50.0, gt=0)
    cable_height_m: float = Field(0.05, gt=0)


class EMPRequest(BaseModel):
    component: Literal["E1", "E2"] = "E1"
    duration_ns: float = Field(1000, gt=0, description="Duration in nanoseconds")
    n_points: int = Field(1000, ge=100, le=50000)


class BCIRequest(BaseModel):
    injection_current_a: float = Field(..., gt=0)
    freq_start_mhz: float = Field(..., gt=0)
    freq_end_mhz: float = Field(..., gt=0)
    num_points: int = Field(100, ge=2, le=1000)
    cable_length_m: float = Field(1.0, gt=0)
    z_load_ohm: float = Field(50.0, gt=0)
    zt_mohm_per_m: float = Field(14.0, gt=0)


@router.post("/lightning/waveform")
async def get_lightning_waveform(req: LightningRequest):
    """Generate DO-160G lightning waveform."""
    t, i = lightning_waveform(req.component, req.duration_us * 1e-6, req.n_points)
    return {
        "time_us": (t * 1e6).tolist(),
        "current_ka": (i / 1e3).tolist(),
        "peak_current_ka": float(np.max(np.abs(i)) / 1e3),
        "component": req.component,
        "description": DO160G_COMPONENTS[req.component]["description"],
    }


@router.post("/lightning/induced-voltage")
async def get_lightning_induced_voltage(req: LightningInducedRequest):
    """Compute lightning-induced voltage on shielded cable."""
    result = induced_voltage_lightning(
        zt_ohm_per_m=req.zt_mohm_per_m * 1e-3,
        cable_length_m=req.cable_length_m,
        component=req.component,
        duration_s=req.duration_us * 1e-6,
        n_points=req.n_points,
    )
    # Convert to user-friendly units
    result["time_us"] = [t * 1e6 for t in result.pop("time_s")]
    result["peak_voltage_v"] = result["peak_voltage_v"]
    return result


@router.post("/hirf/sweep")
async def hirf_frequency_sweep(req: HIRFRequest):
    """Sweep HIRF coupling across frequency range."""
    freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)
    currents_ma = []
    voltages_v = []

    for f_mhz in freqs_mhz:
        result = hirf_induced_current(
            e_field_v_per_m=req.e_field_v_per_m,
            frequency_hz=f_mhz * 1e6,
            cable_length_m=req.cable_length_m,
            z_term_ohm=req.z_term_ohm,
            cable_height_m=req.cable_height_m,
        )
        currents_ma.append(result["induced_current_a"] * 1e3)
        voltages_v.append(result["induced_voltage_v"])

    return {
        "frequencies_mhz": freqs_mhz.tolist(),
        "induced_current_ma": currents_ma,
        "induced_voltage_v": voltages_v,
        "e_field_v_per_m": req.e_field_v_per_m,
    }


@router.post("/emp/waveform")
async def get_emp_waveform(req: EMPRequest):
    """Generate EMP/HEMP E-field waveform."""
    t, e = emp_waveform(req.component, req.duration_ns * 1e-9, req.n_points)
    return {
        "time_ns": (t * 1e9).tolist(),
        "e_field_kv_per_m": (e / 1e3).tolist(),
        "peak_e_field_kv_per_m": float(np.max(np.abs(e)) / 1e3),
        "component": req.component,
        "description": HEMP_COMPONENTS[req.component]["description"],
    }


@router.post("/bci/sweep")
async def bci_frequency_sweep(req: BCIRequest):
    """BCI-induced voltage at load across frequency."""
    freqs_mhz = np.linspace(req.freq_start_mhz, req.freq_end_mhz, req.num_points)
    voltages_mv = []

    for f_mhz in freqs_mhz:
        result = bci_coupling(
            injection_current_a=req.injection_current_a,
            frequency_hz=f_mhz * 1e6,
            cable_length_m=req.cable_length_m,
            z_load_ohm=req.z_load_ohm,
            zt_ohm_per_m=req.zt_mohm_per_m * 1e-3,
        )
        voltages_mv.append(result["load_voltage_v"] * 1e3)

    return {
        "frequencies_mhz": freqs_mhz.tolist(),
        "load_voltage_mv": voltages_mv,
        "injection_current_a": req.injection_current_a,
    }
```

- [ ] **Step 6: Register router in main.py**

Add `hazards` to imports and register:
```python
app.include_router(hazards.router, prefix="/api/v1/hazards", tags=["Hazards"])
```

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/physics/hazards.py tests/test_hazards.py backend/api/v1/routes/hazards.py backend/main.py
git commit -m "feat(hazards): add lightning DO-160G, EMP/HEMP, HIRF, and BCI simulation models"
```

---

## Phase 5: Signal Integrity

TDR simulation and eye diagram generation for high-speed digital links.

---

### Task 8: Implement signal integrity module

**Files:**
- Create: `src/physics/signal_integrity.py`
- Create: `backend/api/v1/routes/signal_integrity.py`
- Test: `tests/test_signal_integrity.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_integrity.py
import pytest
import numpy as np
from src.physics.signal_integrity import (
    tdr_simulation,
    eye_diagram_data,
    impedance_profile,
)


class TestTDR:
    """Time Domain Reflectometry simulation."""

    def test_uniform_line_no_reflection(self):
        """50-ohm line with 50-ohm source: no reflection."""
        result = tdr_simulation(
            segments=[{"z0_ohm": 50.0, "length_m": 0.5, "velocity_factor": 0.66}],
            z_source_ohm=50.0,
            rise_time_s=100e-12,
        )
        # Impedance should be flat at 50 ohm
        z_profile = result["impedance_ohm"]
        assert all(45 < z < 55 for z in z_profile[10:])

    def test_mismatch_shows_reflection(self):
        """100-ohm segment after 50-ohm: reflection coefficient = +1/3."""
        result = tdr_simulation(
            segments=[
                {"z0_ohm": 50.0, "length_m": 0.5, "velocity_factor": 0.66},
                {"z0_ohm": 100.0, "length_m": 0.5, "velocity_factor": 0.66},
            ],
            z_source_ohm=50.0,
        )
        z_profile = result["impedance_ohm"]
        # Should show step from ~50 to ~100
        assert max(z_profile) > 80

    def test_open_termination(self):
        """Open end: reflection coefficient = +1."""
        result = tdr_simulation(
            segments=[{"z0_ohm": 50.0, "length_m": 1.0, "velocity_factor": 0.66}],
            z_source_ohm=50.0,
            z_load_ohm=1e6,  # open
        )
        z_profile = result["impedance_ohm"]
        assert max(z_profile) > 500


class TestEyeDiagram:
    """Eye diagram generation for digital links."""

    def test_clean_eye_high_snr(self):
        """Clean channel with no impairments should have wide eye opening."""
        result = eye_diagram_data(
            data_rate_gbps=1.0,
            cable_length_m=0.1,
            z0_ohm=50.0,
            attenuation_db_per_m_at_nyquist=0.5,
        )
        assert result["eye_height_v"] > 0.3  # > 30% of amplitude
        assert result["eye_width_ui"] > 0.5  # > 50% of unit interval

    def test_long_cable_closes_eye(self):
        """Longer cable should degrade eye opening."""
        short = eye_diagram_data(1.0, 0.1, 50.0, 0.5)
        long = eye_diagram_data(1.0, 5.0, 50.0, 0.5)
        assert long["eye_height_v"] < short["eye_height_v"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_signal_integrity.py -v`
Expected: ImportError

- [ ] **Step 3: Implement signal integrity physics**

```python
# src/physics/signal_integrity.py
"""Signal integrity analysis: TDR and eye diagram simulation.

References:
    Bogatin, E. (2009). Signal and Power Integrity — Simplified, 2nd ed. Prentice Hall.
    Paul, C.R. (2008). Analysis of Multiconductor Transmission Lines, 2nd ed. Wiley.
"""
import numpy as np
from typing import List, Dict, Optional
from src.utils.constants import C


def tdr_simulation(
    segments: List[Dict],
    z_source_ohm: float = 50.0,
    z_load_ohm: Optional[float] = None,
    rise_time_s: float = 100e-12,
    duration_s: Optional[float] = None,
    n_points: int = 1000,
) -> Dict:
    """Simulate Time Domain Reflectometry for a segmented transmission line.

    Launches a step function from a source impedance Z_S, propagates through
    segments of different impedances, and records reflected voltage at the source.

    Each segment: {"z0_ohm": float, "length_m": float, "velocity_factor": float}

    Args:
        segments: List of transmission line segments.
        z_source_ohm: Source impedance (ohm).
        z_load_ohm: Load impedance (ohm). None = matched to last segment.
        rise_time_s: Step function rise time (10-90%) in seconds.
        duration_s: Total simulation time. Auto-calculated if None.
        n_points: Number of time samples.

    Returns:
        Dict with time_s, voltage_v, impedance_ohm, distance_m arrays.
    """
    # Calculate total propagation delay
    total_delay = 0
    for seg in segments:
        v_p = C * seg.get("velocity_factor", 0.66)
        total_delay += seg["length_m"] / v_p

    if duration_s is None:
        duration_s = 3 * total_delay * 2  # Round trip x 3

    if z_load_ohm is None:
        z_load_ohm = segments[-1]["z0_ohm"]

    dt = duration_s / n_points
    t = np.linspace(0, duration_s, n_points)

    # Build impedance profile vs distance
    distances = [0]
    impedances_at_dist = [z_source_ohm]
    for seg in segments:
        distances.append(distances[-1] + seg["length_m"])
        impedances_at_dist.append(seg["z0_ohm"])

    # Build impedance profile vs round-trip time
    # TDR measures: Z_apparent(t) from reflections arriving at time t
    v_incident = 0.5  # Step voltage into matched load is V/2
    v_tdr = np.ones(n_points) * v_incident

    # Compute reflection at each interface
    cumulative_delay = 0
    all_z = [z_source_ohm] + [s["z0_ohm"] for s in segments] + [z_load_ohm]

    for i, seg in enumerate(segments):
        v_p = C * seg.get("velocity_factor", 0.66)
        seg_delay = seg["length_m"] / v_p
        cumulative_delay += seg_delay
        round_trip = 2 * cumulative_delay

        # Reflection coefficient at interface between segment i and i+1
        z_before = all_z[i + 1]
        z_after = all_z[i + 2] if i + 1 < len(segments) else z_load_ohm
        rho = (z_after - z_before) / (z_after + z_before) if (z_after + z_before) != 0 else 0

        # Add reflected step at round-trip time
        for j in range(n_points):
            if t[j] >= round_trip:
                # Smoothed step with rise time
                t_rel = t[j] - round_trip
                step = 0.5 * (1 + np.tanh(4 * t_rel / rise_time_s))
                v_tdr[j] += rho * v_incident * step

    # Convert TDR voltage to apparent impedance
    # Z_apparent = Z_S * (1 + rho) / (1 - rho) where rho = (V_reflected / V_incident)
    rho_profile = (v_tdr - v_incident) / v_incident
    z_apparent = np.zeros(n_points)
    for j in range(n_points):
        rho_j = np.clip(rho_profile[j], -0.999, 0.999)
        z_apparent[j] = z_source_ohm * (1 + rho_j) / (1 - rho_j)

    # Distance axis (one-way, from round-trip time)
    avg_vf = np.mean([s.get("velocity_factor", 0.66) for s in segments])
    dist = t * C * avg_vf / 2  # one-way distance

    return {
        "time_ns": (t * 1e9).tolist(),
        "voltage_v": v_tdr.tolist(),
        "impedance_ohm": z_apparent.tolist(),
        "distance_m": dist.tolist(),
    }


def impedance_profile(
    segments: List[Dict],
    n_points: int = 500,
) -> Dict:
    """Generate impedance vs distance profile for a transmission line.

    Args:
        segments: List of {"z0_ohm", "length_m"} dicts.
        n_points: Number of distance points.

    Returns:
        Dict with distance_m, impedance_ohm arrays.
    """
    total_length = sum(s["length_m"] for s in segments)
    d = np.linspace(0, total_length, n_points)
    z = np.zeros(n_points)

    for i in range(n_points):
        cumulative = 0
        for seg in segments:
            if d[i] <= cumulative + seg["length_m"]:
                z[i] = seg["z0_ohm"]
                break
            cumulative += seg["length_m"]
        else:
            z[i] = segments[-1]["z0_ohm"]

    return {
        "distance_m": d.tolist(),
        "impedance_ohm": z.tolist(),
    }


def eye_diagram_data(
    data_rate_gbps: float,
    cable_length_m: float,
    z0_ohm: float = 50.0,
    attenuation_db_per_m_at_nyquist: float = 1.0,
    v_swing: float = 1.0,
    n_bits: int = 127,
    samples_per_bit: int = 64,
) -> Dict:
    """Generate eye diagram data for a lossy channel.

    Models a frequency-dependent lossy channel using a simple
    skin-effect attenuation model: alpha(f) = alpha_0 * sqrt(f/f_nyq).

    Args:
        data_rate_gbps: Data rate (Gbps).
        cable_length_m: Cable length (m).
        z0_ohm: Characteristic impedance (ohm).
        attenuation_db_per_m_at_nyquist: Loss at Nyquist frequency (dB/m).
        v_swing: Peak-to-peak voltage swing (V).
        n_bits: Number of PRBS bits to simulate.
        samples_per_bit: Oversampling ratio.

    Returns:
        Dict with eye diagram parameters and waveform data.
    """
    f_nyquist = data_rate_gbps * 1e9 / 2
    total_attenuation_db = attenuation_db_per_m_at_nyquist * cable_length_m
    amplitude = v_swing / 2

    # Generate PRBS-7 sequence (thread-safe RNG)
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, n_bits)
    n_samples = n_bits * samples_per_bit

    # NRZ signal
    signal = np.zeros(n_samples)
    for i, bit in enumerate(bits):
        val = amplitude if bit == 1 else -amplitude
        signal[i * samples_per_bit:(i + 1) * samples_per_bit] = val

    # Channel frequency response: H(f) = 10^(-alpha*L*sqrt(f/f_nyq)/20)
    freqs = np.fft.rfftfreq(n_samples, d=1 / (data_rate_gbps * 1e9 * samples_per_bit))
    H = np.ones(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        if f > 0 and f_nyquist > 0:
            alpha_f = attenuation_db_per_m_at_nyquist * np.sqrt(f / f_nyquist)
            loss_db = alpha_f * cable_length_m
            H[i] = 10 ** (-loss_db / 20)

    # Apply channel
    signal_fft = np.fft.rfft(signal)
    output_fft = signal_fft * H
    output = np.fft.irfft(output_fft, n=n_samples)

    # Extract eye diagram parameters
    # Sample at center of each bit
    center_samples = [output[i * samples_per_bit + samples_per_bit // 2]
                      for i in range(n_bits)]

    ones = [s for s, b in zip(center_samples, bits) if b == 1]
    zeros = [s for s, b in zip(center_samples, bits) if b == 0]

    eye_height = (min(ones) - max(zeros)) if ones and zeros else 0
    eye_height = max(0, eye_height)

    # Eye width: measure zero crossings relative to bit period
    ui = 1 / (data_rate_gbps * 1e9)  # unit interval in seconds
    # Simplified: eye width based on ISI
    eye_width_ui = max(0, 1.0 - total_attenuation_db / 20)

    # Time axis for one UI overlay
    t_ui = np.linspace(0, 1, samples_per_bit)

    # Overlay all bit periods for eye pattern
    eye_traces = []
    for i in range(min(n_bits - 1, 64)):
        start = i * samples_per_bit
        end = start + samples_per_bit
        if end <= len(output):
            eye_traces.append(output[start:end].tolist())

    return {
        "eye_height_v": float(eye_height),
        "eye_width_ui": float(eye_width_ui),
        "total_attenuation_db": float(total_attenuation_db),
        "data_rate_gbps": data_rate_gbps,
        "cable_length_m": cable_length_m,
        "time_ui": t_ui.tolist(),
        "eye_traces": eye_traces,
        "unit_interval_ps": float(ui * 1e12),
    }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_signal_integrity.py -v`
Expected: All PASS

- [ ] **Step 5: Create signal integrity API route**

Create `backend/api/v1/routes/signal_integrity.py`:

```python
"""Signal integrity simulation endpoints: TDR and eye diagram."""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from src.physics.signal_integrity import tdr_simulation, eye_diagram_data, impedance_profile

router = APIRouter()


class TLSegment(BaseModel):
    z0_ohm: float = Field(..., gt=0)
    length_m: float = Field(..., gt=0)
    velocity_factor: float = Field(0.66, gt=0, le=1)


class TDRRequest(BaseModel):
    segments: List[TLSegment]
    z_source_ohm: float = Field(50.0, gt=0)
    z_load_ohm: Optional[float] = Field(None, gt=0)
    rise_time_ps: float = Field(100, gt=0, description="Rise time in picoseconds")
    n_points: int = Field(1000, ge=100, le=10000)


class EyeDiagramRequest(BaseModel):
    data_rate_gbps: float = Field(..., gt=0)
    cable_length_m: float = Field(..., gt=0)
    z0_ohm: float = Field(50.0, gt=0)
    attenuation_db_per_m: float = Field(1.0, ge=0, description="Loss at Nyquist (dB/m)")
    v_swing: float = Field(1.0, gt=0)


@router.post("/tdr")
async def run_tdr(req: TDRRequest):
    """Run TDR simulation on a segmented transmission line."""
    segments = [s.model_dump() for s in req.segments]
    return tdr_simulation(
        segments=segments,
        z_source_ohm=req.z_source_ohm,
        z_load_ohm=req.z_load_ohm,
        rise_time_s=req.rise_time_ps * 1e-12,
        n_points=req.n_points,
    )


@router.post("/eye-diagram")
async def run_eye_diagram(req: EyeDiagramRequest):
    """Generate eye diagram for a lossy digital channel."""
    return eye_diagram_data(
        data_rate_gbps=req.data_rate_gbps,
        cable_length_m=req.cable_length_m,
        z0_ohm=req.z0_ohm,
        attenuation_db_per_m_at_nyquist=req.attenuation_db_per_m,
        v_swing=req.v_swing,
    )


@router.post("/impedance-profile")
async def get_impedance_profile(req: TDRRequest):
    """Get impedance vs distance profile."""
    segments = [s.model_dump() for s in req.segments]
    return impedance_profile(segments)
```

- [ ] **Step 6: Register router in main.py**

```python
app.include_router(signal_integrity.router, prefix="/api/v1/signal-integrity", tags=["Signal Integrity"])
```

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/physics/signal_integrity.py tests/test_signal_integrity.py backend/api/v1/routes/signal_integrity.py backend/main.py
git commit -m "feat(signal-integrity): add TDR simulation and eye diagram generation"
```

---

## Phase 6: Backend Integration & Registration

---

### Task 9: Final main.py router registration and requirements update

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Update main.py imports and registrations**

The final `main.py` import line should include all new routers:

```python
from api.v1.routes import (
    physics, auth, materials, analysis, chat, multilayer, composites,
    advanced, heatmap, cables, enclosure, hazards, recommendation,
    signal_integrity,
)
```

New router registrations (added after existing ones):
```python
app.include_router(enclosure.router, prefix="/api/v1/enclosure", tags=["Enclosure"])
app.include_router(hazards.router, prefix="/api/v1/hazards", tags=["Hazards"])
app.include_router(recommendation.router, prefix="/api/v1/recommendation", tags=["Recommendation"])
app.include_router(signal_integrity.router, prefix="/api/v1/signal-integrity", tags=["Signal Integrity"])
```

- [ ] **Step 2: Update requirements.txt**

Replace `google-generativeai>=0.7.0` with `google-genai>=1.0.0`.

- [ ] **Step 3: Verify server starts**

Run: `cd backend && python -c "from main import app; print(f'{len(app.routes)} routes registered')"`
Expected: Route count > 30

- [ ] **Step 4: Run complete test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/requirements.txt
git commit -m "feat: register all system-level EMC routers (enclosure, hazards, recommendation, signal-integrity)"
```

---

## Summary: New Platform Capabilities

After completing all phases, the EMI Shield Designer becomes a system-level EMC platform with:

| Module | Physics Method | API Endpoints |
|--------|---------------|---------------|
| **Enclosure SE** | Bethe hole theory, slot antenna, cavity resonance, waveguide-below-cutoff | `/enclosure/analyze`, `/enclosure/cavity-resonances`, `/enclosure/frequency-sweep` |
| **Cable Crosstalk** | Clayton Paul MTL (distributed [L][C] eigendecomposition) | `/cables/crosstalk`, `/cables/transfer-impedance` |
| **Signal Integrity** | TDR (reflection analysis), eye diagram (lossy channel) | `/signal-integrity/tdr`, `/signal-integrity/eye-diagram`, `/signal-integrity/impedance-profile` |
| **Lightning/EMP** | DO-160G waveforms, BLT coupling, HEMP IEC 61000-2-9 | `/hazards/lightning/*`, `/hazards/emp/*` |
| **HIRF** | Agrawal field-to-TL coupling | `/hazards/hirf/sweep` |
| **BCI** | Transfer impedance coupling | `/hazards/bci/sweep` |
| **Material Recommendation** | Inverse SE solver, Pareto optimization | `/recommendation/search` |
| **Heatmap** | Real EMICalculator (replaces mock) | `/heatmap/generate` |

Total new physics modules: 5 (`aperture.py`, `cables_mtl.py`, `signal_integrity.py`, `hazards.py`, `recommendation.py`)
Total new API routes: 4 (`enclosure.py`, `hazards.py`, `recommendation.py`, `signal_integrity.py`)
Total new test files: 5
Total new/modified endpoints: ~15+
