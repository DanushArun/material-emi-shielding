# Phase 1: Cleanup and Refactor - Remove Streamlit, Strengthen Backend

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Streamlit dependency entirely, extract reusable business logic into clean Python modules, and expand the FastAPI backend to expose all physics capabilities as API endpoints that the Next.js frontend will consume.

**Architecture:** Three-layer separation: `src/` contains pure Python physics/materials/chemistry logic with zero UI dependencies. `backend/` is a FastAPI app that imports from `src/` and exposes REST endpoints. `frontend/` is a Next.js app that calls the backend API. No module should import from both layers.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, NumPy/SciPy, Next.js 14, TypeScript, TailwindCSS

---

## File Structure

### Files to DELETE
- `app.py` (2,129 lines - Streamlit UI, replaced by Next.js)
- `auth.py` (159 lines - Streamlit-specific auth, replaced by backend JWT auth)
- `calculation_history.json` (if exists - replaced by backend persistence)
- `.streamlit/` directory (Streamlit config)

### Files to CREATE
- `src/chemistry/parser.py` - ChemicalParser + ReactionEngine extracted from app.py
- `src/chemistry/__init__.py`
- `backend/api/v1/routes/materials.py` - Materials/periodic table endpoints
- `backend/api/v1/routes/analysis.py` - Frequency sweep, thickness sweep, grain size sweep, cooling rate endpoints
- `backend/api/v1/schemas/physics.py` - All Pydantic request/response models (extracted from routes)
- `backend/api/v1/schemas/materials.py` - Material-related schemas
- `backend/api/v1/schemas/__init__.py`
- `tests/test_chemistry_parser.py` - Tests for extracted chemistry logic
- `tests/test_physics_api.py` - Tests for backend API endpoints
- `tests/test_materials_api.py` - Tests for materials endpoints

### Files to MODIFY
- `src/physics/emi_calculations.py` - Remove Streamlit fallback imports, use clean relative imports
- `src/physics/advanced_microstructure.py` - No changes needed (already clean)
- `src/materials/material_properties.py` - Add methods needed by new API endpoints
- `src/utils/constants.py` - No changes needed (already clean)
- `backend/main.py` - Register new routers, remove emojis from startup logs
- `backend/core/config.py` - Remove ML_MODEL_PATH (not used), add GEMINI_API_KEY setting
- `backend/core/security.py` - Make auth optional for dev mode (remove hard dependency)
- `backend/api/v1/routes/physics.py` - Use shared schemas, fix sys.path hack, make auth optional
- `requirements.txt` - Remove streamlit, keep core scientific deps
- `backend/requirements.txt` - Add missing deps if any
- `frontend/package.json` - Will be updated in Phase 2 (no changes here)

---

### Task 1: Extract Chemistry Logic from app.py

**Files:**
- Create: `src/chemistry/__init__.py`
- Create: `src/chemistry/parser.py`
- Test: `tests/test_chemistry_parser.py`

- [ ] **Step 1: Write failing tests for ChemicalParser**

```python
# tests/test_chemistry_parser.py
import pytest
from src.chemistry.parser import ChemicalParser, ReactionEngine


class TestChemicalParser:
    def test_parse_simple_formula(self):
        result = ChemicalParser.parse_formula("Cu")
        assert result == {"Cu": 1}

    def test_parse_formula_with_count(self):
        result = ChemicalParser.parse_formula("H2O")
        assert result == {"H": 2, "O": 1}

    def test_parse_complex_formula(self):
        result = ChemicalParser.parse_formula("Al2O3")
        assert result == {"Al": 2, "O": 3}

    def test_parse_formula_strips_spaces(self):
        result = ChemicalParser.parse_formula("Cu Zn")
        assert result == {"Cu": 1, "Zn": 1}

    def test_format_formula(self):
        result = ChemicalParser.format_formula({"H": 2, "O": 1})
        assert "H" in result and "O" in result

    def test_format_empty(self):
        assert ChemicalParser.format_formula({}) == ""


class TestReactionEngine:
    def test_add_molecule(self):
        engine = ReactionEngine()
        engine.add_molecule("Cu", 1)
        assert len(engine.molecules) == 1

    def test_add_direct_composition(self):
        engine = ReactionEngine()
        engine.add_direct_composition({"Fe": 70.0, "Cr": 18.0, "Ni": 12.0}, "Stainless")
        assert len(engine.molecules) == 1

    def test_total_composition_single_element(self):
        engine = ReactionEngine()
        engine.add_direct_composition({"Cu": 100.0}, "Pure Copper")
        comp = engine.get_total_composition()
        assert comp == {"Cu": 100.0}

    def test_total_composition_direct_returns_as_is(self):
        engine = ReactionEngine()
        engine.add_direct_composition({"Fe": 70.0, "C": 30.0}, "Steel")
        comp = engine.get_total_composition()
        assert abs(comp["Fe"] - 70.0) < 0.01
        assert abs(comp["C"] - 30.0) < 0.01

    def test_reaction_equation_display(self):
        engine = ReactionEngine()
        engine.add_molecule("Cu", 2)
        eq = engine.get_reaction_equation()
        assert "Cu" in eq
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_chemistry_parser.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.chemistry'"

- [ ] **Step 3: Create chemistry package with extracted logic**

Create `src/chemistry/__init__.py`:
```python
from .parser import ChemicalParser, ReactionEngine

__all__ = ["ChemicalParser", "ReactionEngine"]
```

Create `src/chemistry/parser.py` - extract ChemicalParser and ReactionEngine from app.py lines 241-366. Remove all Streamlit imports. The ReactionEngine needs access to material_db for molecular weight calculation, so import it:

```python
"""
Chemical formula parsing and reaction composition calculations.
Extracted from the Streamlit app for reuse across frontend and API.
"""

import re
from typing import Dict, List, Optional

from src.materials.material_properties import material_db


class ChemicalParser:
    """Parse and validate chemical formulas."""

    @staticmethod
    def parse_formula(formula: str) -> Dict[str, int]:
        """Parse a chemical formula into element counts."""
        formula = formula.replace(" ", "")
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        composition = {}
        for element, count in matches:
            if not element:
                continue
            count = int(count) if count else 1
            composition[element] = composition.get(element, 0) + count
        return composition

    @staticmethod
    def format_formula(composition: Dict[str, int]) -> str:
        """Format element composition back to chemical formula string."""
        if not composition:
            return ""
        parts = []
        for element, count in sorted(composition.items()):
            if count == 1:
                parts.append(element)
            else:
                parts.append(f"{element}{count}")
        return "".join(parts)


class ReactionEngine:
    """Handle chemical reactions and composition calculations for EMI materials."""

    def __init__(self):
        self.molecules: List[dict] = []

    def add_molecule(self, formula: str, coefficient: int = 1):
        """Add a molecule to the reaction."""
        composition = ChemicalParser.parse_formula(formula)
        if composition:
            self.molecules.append({
                'formula': formula,
                'composition': composition,
                'coefficient': coefficient,
                'molecular_weight': self._calculate_molecular_weight(composition),
            })

    def add_direct_composition(self, composition: Dict[str, float], display_name: str):
        """Add a direct weight-percentage composition."""
        self.molecules.append({
            'type': 'direct',
            'composition': composition.copy(),
            'formula': 'Direct Composition',
            'coefficient': 1,
            'molecular_weight': sum(
                material_db.get_material(elem).get('atomic_weight', 50) * pct / 100
                for elem, pct in composition.items()
                if material_db.get_material(elem)
            ),
            'display_name': display_name,
        })

    def _calculate_molecular_weight(self, composition: Dict[str, int]) -> float:
        """Calculate molecular weight from composition."""
        total = 0.0
        for element, count in composition.items():
            elem_data = material_db.get_material(element)
            weight = elem_data.get('atomic_weight', 50.0) if elem_data else 50.0
            total += weight * count
        return total

    def get_reaction_equation(self) -> str:
        """Get the formatted reaction equation string."""
        if not self.molecules:
            return "No reaction defined"
        parts = []
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                parts.append(mol.get('display_name', 'Direct Composition'))
            else:
                coeff = f"{mol['coefficient']}" if mol['coefficient'] > 1 else ""
                formula = ChemicalParser.format_formula(mol['composition'])
                parts.append(f"{coeff}{formula}")
        return " + ".join(parts)

    def get_total_composition(self) -> Dict[str, float]:
        """Calculate total elemental composition by mass percentage."""
        if not self.molecules:
            return {}
        if len(self.molecules) == 1 and self.molecules[0].get('type') == 'direct':
            return self.molecules[0]['composition']

        element_masses: Dict[str, float] = {}
        total_mass = 0.0
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                continue
            mol_mass = mol['molecular_weight'] * mol['coefficient']
            total_mass += mol_mass
            for element, count in mol['composition'].items():
                elem_data = material_db.get_material(element)
                atomic_weight = elem_data.get('atomic_weight', 50.0) if elem_data else 50.0
                element_mass = atomic_weight * count * mol['coefficient']
                element_masses[element] = element_masses.get(element, 0) + element_mass

        if total_mass > 0:
            return {el: (mass / total_mass) * 100 for el, mass in element_masses.items()}
        return {}

    def clear(self):
        """Clear all molecules."""
        self.molecules.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_chemistry_parser.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/chemistry/__init__.py src/chemistry/parser.py tests/test_chemistry_parser.py
git commit -m "feat: extract ChemicalParser and ReactionEngine from Streamlit app"
```

---

### Task 2: Fix Physics Engine Imports

**Files:**
- Modify: `src/physics/emi_calculations.py:1-30`
- Modify: `src/__init__.py` (create if missing)
- Modify: `src/physics/__init__.py` (create if missing)
- Test: `tests/test_physics_engine.py`

- [ ] **Step 1: Write failing test for clean import**

```python
# tests/test_physics_engine.py
import pytest
from src.physics.emi_calculations import EMICalculator


class TestEMICalculatorImport:
    def test_can_instantiate(self):
        calc = EMICalculator()
        assert calc is not None

    def test_skin_depth_copper_1ghz(self):
        calc = EMICalculator()
        import numpy as np
        mu_0 = 4 * np.pi * 1e-7
        # Copper at 1 GHz
        delta = calc.calculate_skin_depth(5.96e7, mu_0, 1e9)
        # Should be ~2.06 micrometers
        assert 1e-6 < delta < 5e-6

    def test_shielding_effectiveness_copper(self):
        calc = EMICalculator()
        result = calc.calculate_shielding_effectiveness(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            thickness=0.001,  # 1mm
            frequency=1e9,  # 1 GHz
        )
        assert result['total_se'] > 50  # Copper 1mm at 1GHz should be >50 dB
        assert 'reflection_loss' in result
        assert 'absorption_loss' in result

    def test_frequency_sweep(self):
        calc = EMICalculator()
        result = calc.frequency_sweep(
            conductivity=5.96e7,
            relative_permeability=1.0,
            relative_permittivity=1.0,
            thickness=0.001,
            freq_start=1e6,
            freq_end=1e9,
            num_points=10,
        )
        assert len(result['frequencies']) == 10
        assert len(result['total_ses']) == 10
```

- [ ] **Step 2: Run tests to check current state**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_physics_engine.py -v`
Expected: May fail due to import issues with the try/except fallback pattern

- [ ] **Step 3: Fix imports in emi_calculations.py**

Replace lines 1-30 of `src/physics/emi_calculations.py`:

```python
"""
Core EMI shielding calculations based on electromagnetic theory.
Enhanced with advanced microstructure modeling and cooling rate dependencies.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any

from src.utils.constants import (
    MU_0, EPSILON_0, Z_0, C,
    validate_conductivity, validate_permeability,
    validate_permittivity, validate_frequency, validate_thickness
)
from src.physics.advanced_microstructure import (
    AdvancedMicrostructure, MicrostructureParams, ProcessingParams
)
```

Remove the `try/except ImportError` fallback block and the `sys.path.append` hack entirely.

Ensure `src/__init__.py`, `src/physics/__init__.py`, `src/utils/__init__.py`, and `src/materials/__init__.py` all exist (create empty ones if missing).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_physics_engine.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/physics/__init__.py src/utils/__init__.py src/materials/__init__.py src/physics/emi_calculations.py tests/test_physics_engine.py
git commit -m "fix: clean up physics engine imports, remove Streamlit fallback paths"
```

---

### Task 3: Expand Backend API - Materials Endpoints

**Files:**
- Create: `backend/api/v1/schemas/__init__.py`
- Create: `backend/api/v1/schemas/materials.py`
- Create: `backend/api/v1/routes/materials.py`
- Modify: `backend/main.py:94-98` (uncomment materials router)
- Test: `tests/test_materials_api.py`

- [ ] **Step 1: Write failing tests for materials API**

```python
# tests/test_materials_api.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

client = TestClient(app)


class TestPeriodicTable:
    def test_get_all_elements(self):
        response = client.get("/api/v1/materials/elements")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 80  # Should have most periodic table

    def test_get_single_element(self):
        response = client.get("/api/v1/materials/elements/Cu")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "Cu"
        assert data["electrical_conductivity"] > 1e7

    def test_get_invalid_element(self):
        response = client.get("/api/v1/materials/elements/Xx")
        assert response.status_code == 404


class TestAlloys:
    def test_list_alloys(self):
        response = client.get("/api/v1/materials/alloys")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 5

    def test_get_alloy_by_key(self):
        response = client.get("/api/v1/materials/alloys/steel_1018")
        assert response.status_code == 200
        data = response.json()
        assert "composition" in data


class TestCompositeProperties:
    def test_calculate_composite(self):
        response = client.post("/api/v1/materials/composite-properties", json={
            "elements": {"Cu": 70.0, "Zn": 30.0}
        })
        assert response.status_code == 200
        data = response.json()
        assert "conductivity" in data
        assert "permeability" in data
        assert "density" in data
        assert data["conductivity"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_materials_api.py -v`
Expected: FAIL - routes don't exist yet

- [ ] **Step 3: Create materials schemas**

Create `backend/api/v1/schemas/__init__.py` (empty).

Create `backend/api/v1/schemas/materials.py`:
```python
"""Pydantic schemas for materials API endpoints."""
from pydantic import BaseModel, Field
from typing import Dict, Optional


class ElementResponse(BaseModel):
    symbol: str
    name: str
    atomic_number: int
    atomic_weight: float
    electrical_conductivity: float = Field(description="S/m")
    relative_permeability: float
    density: float = Field(description="kg/m3")
    melting_point: Optional[float] = None
    boiling_point: Optional[float] = None


class AlloyResponse(BaseModel):
    key: str
    name: str
    composition: Dict[str, float]
    density: float
    electrical_conductivity: float
    relative_permeability: float
    relative_permittivity: float = 1.0
    note: Optional[str] = None


class CompositePropertiesRequest(BaseModel):
    elements: Dict[str, float] = Field(
        ..., description="Element symbols and weight percentages, must sum to ~100"
    )


class CompositePropertiesResponse(BaseModel):
    conductivity: float = Field(description="Effective conductivity (S/m)")
    permeability: float = Field(description="Effective relative permeability")
    permittivity: float = Field(description="Effective relative permittivity")
    density: float = Field(description="Effective density (kg/m3)")
    composition: Dict[str, float]
```

- [ ] **Step 4: Create materials route**

Create `backend/api/v1/routes/materials.py`:
```python
"""Materials database endpoints - periodic table, alloys, and composite property calculation."""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.materials.material_properties import material_db
from backend.api.v1.schemas.materials import (
    ElementResponse, AlloyResponse,
    CompositePropertiesRequest, CompositePropertiesResponse,
)

router = APIRouter()


@router.get("/elements", response_model=List[Dict[str, Any]])
async def list_elements():
    """Get all elements from the periodic table."""
    elements = []
    for symbol, data in sorted(material_db.periodic_table.items()):
        elements.append({"symbol": symbol, **data})
    return elements


@router.get("/elements/{symbol}")
async def get_element(symbol: str):
    """Get properties of a single element."""
    data = material_db.get_material(symbol)
    if not data:
        raise HTTPException(status_code=404, detail=f"Element '{symbol}' not found")
    return {"symbol": symbol, **data}


@router.get("/alloys", response_model=List[Dict[str, Any]])
async def list_alloys():
    """Get all pre-defined alloys."""
    alloys = []
    for key, data in material_db.alloys.items():
        alloys.append({"key": key, **data})
    return alloys


@router.get("/alloys/{key}")
async def get_alloy(key: str):
    """Get properties of a specific alloy."""
    if key not in material_db.alloys:
        raise HTTPException(status_code=404, detail=f"Alloy '{key}' not found")
    return {"key": key, **material_db.alloys[key]}


@router.post("/composite-properties", response_model=CompositePropertiesResponse)
async def calculate_composite_properties(request: CompositePropertiesRequest):
    """Calculate effective material properties from elemental composition."""
    conductivity = 0.0
    permeability = 1.0
    permittivity = 1.0
    density = 0.0

    for element, percentage in request.elements.items():
        elem_data = material_db.get_material(element)
        if not elem_data:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown element: {element}"
            )
        weight = percentage / 100.0
        conductivity += elem_data.get('electrical_conductivity', 1e6) * weight
        elem_perm = max(elem_data.get('relative_permeability', 1.0), 0.999)
        permeability *= elem_perm ** weight
        elem_eps = max(elem_data.get('relative_permittivity', 1.0), 1.0)
        permittivity *= elem_eps ** weight
        density += elem_data.get('density', 1000) * weight

    return CompositePropertiesResponse(
        conductivity=max(conductivity, 1e-10),
        permeability=max(permeability, 0.999),
        permittivity=max(permittivity, 1.0),
        density=density,
        composition=request.elements,
    )
```

- [ ] **Step 5: Register materials router in backend/main.py**

Uncomment/add the materials router line in `backend/main.py`:
```python
from api.v1.routes import physics, auth, materials
# ...
app.include_router(materials.router, prefix="/api/v1/materials", tags=["Materials"])
```

- [ ] **Step 6: Make auth optional for development**

In `backend/api/v1/routes/physics.py`, make the `user_id: str = Depends(get_current_user_id)` optional so the API works without a database. Replace the Depends with an optional dependency or remove auth requirement for now. This is needed so the API is testable without a full PostgreSQL setup.

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_materials_api.py -v`
Expected: All 6 tests PASS

- [ ] **Step 8: Commit**

```bash
git add backend/api/v1/schemas/ backend/api/v1/routes/materials.py backend/main.py backend/api/v1/routes/physics.py tests/test_materials_api.py
git commit -m "feat: add materials API endpoints for periodic table, alloys, and composites"
```

---

### Task 4: Expand Backend API - Analysis Endpoints

**Files:**
- Create: `backend/api/v1/schemas/physics.py`
- Create: `backend/api/v1/routes/analysis.py`
- Modify: `backend/main.py` (register analysis router)
- Test: `tests/test_analysis_api.py`

- [ ] **Step 1: Write failing tests for analysis endpoints**

```python
# tests/test_analysis_api.py
import pytest
from fastapi.testclient import TestClient
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

client = TestClient(app)


class TestFrequencySweep:
    def test_frequency_sweep(self):
        response = client.post("/api/v1/analysis/frequency-sweep", json={
            "composition": {"Cu": 100.0},
            "thickness_mm": 1.0,
            "freq_start_mhz": 1.0,
            "freq_end_mhz": 1000.0,
            "num_points": 20,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["frequencies_mhz"]) == 20
        assert len(data["total_se_db"]) == 20


class TestThicknessSweep:
    def test_thickness_sweep(self):
        response = client.post("/api/v1/analysis/thickness-sweep", json={
            "composition": {"Cu": 100.0},
            "frequency_mhz": 1000.0,
            "thickness_start_mm": 0.1,
            "thickness_end_mm": 5.0,
            "num_points": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["thicknesses_mm"]) == 10


class TestGrainSizeSweep:
    def test_grain_size_sweep(self):
        response = client.post("/api/v1/analysis/grain-size-sweep", json={
            "composition": {"Cu": 100.0},
            "frequency_mhz": 1000.0,
            "thickness_mm": 1.0,
            "grain_start_um": 0.01,
            "grain_end_um": 100.0,
            "num_points": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["grain_sizes_um"]) == 10


class TestCoolingRateSweep:
    def test_cooling_rate_sweep(self):
        response = client.post("/api/v1/analysis/cooling-rate-sweep", json={
            "composition": {"Fe": 98.0, "C": 2.0},
            "frequency_mhz": 1000.0,
            "thickness_mm": 1.0,
            "num_points": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["cooling_rates"]) == 10


class TestThicknessOptimization:
    def test_optimize_thickness(self):
        response = client.post("/api/v1/analysis/optimize-thickness", json={
            "composition": {"Cu": 100.0},
            "frequency_mhz": 1000.0,
            "target_se_db": 60.0,
            "thickness_max_mm": 10.0,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["achieved_se_db"] >= 55  # Should be close to target
        assert data["optimal_thickness_mm"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_analysis_api.py -v`
Expected: FAIL - routes don't exist

- [ ] **Step 3: Create physics schemas (shared across physics + analysis routes)**

Create `backend/api/v1/schemas/physics.py`:
```python
"""Pydantic schemas for physics calculation endpoints."""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class CompositionInput(BaseModel):
    """Material composition as element weight percentages."""
    composition: Dict[str, float] = Field(..., description="Element symbols to weight %")


class FrequencySweepRequest(BaseModel):
    composition: Dict[str, float]
    thickness_mm: float = Field(..., gt=0.001, lt=100)
    freq_start_mhz: float = Field(..., gt=0.001)
    freq_end_mhz: float = Field(..., gt=0.001)
    num_points: int = Field(100, gt=1, le=1000)
    grain_size_um: Optional[float] = Field(None, gt=0)


class ThicknessSweepRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    thickness_start_mm: float = Field(0.1, gt=0.001)
    thickness_end_mm: float = Field(10.0, gt=0.001)
    num_points: int = Field(100, gt=1, le=1000)
    grain_size_um: Optional[float] = Field(None, gt=0)


class GrainSizeSweepRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    thickness_mm: float = Field(..., gt=0.001)
    grain_start_um: float = Field(0.01, gt=0)
    grain_end_um: float = Field(100.0, gt=0)
    num_points: int = Field(100, gt=1, le=1000)


class CoolingRateSweepRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    thickness_mm: float = Field(..., gt=0.001)
    cooling_rate_min: float = Field(0.1, gt=0)
    cooling_rate_max: float = Field(1000.0, gt=0)
    num_points: int = Field(50, gt=1, le=500)


class ThicknessOptimizationRequest(BaseModel):
    composition: Dict[str, float]
    frequency_mhz: float = Field(..., gt=0.001)
    target_se_db: float = Field(..., gt=0)
    thickness_max_mm: float = Field(10.0, gt=0.001)
    grain_size_um: Optional[float] = Field(None, gt=0)
```

- [ ] **Step 4: Create analysis route**

Create `backend/api/v1/routes/analysis.py`:

This file implements all sweep and optimization endpoints. Each endpoint:
1. Accepts composition as Dict[str, float]
2. Calculates composite material properties using the same logic from app.py lines 1109-1142
3. Calls the appropriate EMICalculator method
4. Returns arrays suitable for Plotly charts

Key: Extract the composite property calculation into a shared helper function `_calculate_composite_properties(composition)` that both `physics.py` and `analysis.py` can use. Place this in `backend/api/v1/routes/_helpers.py`.

- [ ] **Step 5: Register analysis router in backend/main.py**

```python
from api.v1.routes import physics, auth, materials, analysis
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_analysis_api.py -v`
Expected: All 5 tests PASS

- [ ] **Step 7: Commit**

```bash
git add backend/api/v1/schemas/physics.py backend/api/v1/routes/analysis.py backend/api/v1/routes/_helpers.py backend/main.py tests/test_analysis_api.py
git commit -m "feat: add analysis endpoints for frequency/thickness/grain/cooling sweeps"
```

---

### Task 5: Remove Streamlit and Clean Up Dependencies

**Files:**
- Delete: `app.py`
- Delete: `auth.py`
- Delete: `.streamlit/` directory
- Modify: `requirements.txt` - remove streamlit
- Modify: `backend/core/config.py` - add GEMINI_API_KEY, remove ML_MODEL_PATH
- Modify: `.env.example` - add GEMINI_API_KEY placeholder
- Modify: `backend/main.py` - remove emojis from startup logs

- [ ] **Step 1: Verify all extracted logic is tested**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/ -v`
Expected: All tests pass (chemistry, physics, materials API, analysis API)

- [ ] **Step 2: Delete Streamlit files**

```bash
cd /Users/danusharun/Documents/EMI-shielding
rm app.py
rm auth.py
rm -rf .streamlit/
```

- [ ] **Step 3: Update requirements.txt**

Remove `streamlit>=1.28.2` from `requirements.txt`. Keep:
```
numpy>=1.26.0
pandas>=2.0.3
scipy>=1.11.0
plotly>=5.18.0
scikit-learn>=1.3.0
```

- [ ] **Step 4: Update backend config**

In `backend/core/config.py`:
- Remove `ML_MODEL_PATH` and `ML_CACHE_ENABLED` (not used)
- Add `GEMINI_API_KEY: str = ""` for AI chat integration (Phase 2)

In `backend/main.py`:
- Replace emoji-laden startup/shutdown log messages with plain text

In `.env.example`:
- Add `GEMINI_API_KEY=your-gemini-api-key-here`

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Verify no remaining Streamlit imports**

Run: `cd /Users/danusharun/Documents/EMI-shielding && grep -r "import streamlit" src/ backend/ --include="*.py"`
Expected: No results

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove Streamlit, clean up deps, prepare for Next.js frontend"
```

---

### Task 6: Verify Backend Starts and All Endpoints Work

**Files:**
- No new files
- Test: Manual verification + `tests/test_backend_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/test_backend_smoke.py
"""Smoke tests to verify the backend starts and all endpoints respond."""
import pytest
from fastapi.testclient import TestClient
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_docs_accessible():
    r = client.get("/docs")
    assert r.status_code == 200


def test_full_calculation_flow():
    """End-to-end: get element -> calculate composite -> calculate SE -> frequency sweep."""
    # 1. Get copper properties
    r = client.get("/api/v1/materials/elements/Cu")
    assert r.status_code == 200

    # 2. Calculate composite properties for brass
    r = client.post("/api/v1/materials/composite-properties", json={
        "elements": {"Cu": 70.0, "Zn": 30.0}
    })
    assert r.status_code == 200
    props = r.json()
    assert props["conductivity"] > 1e6

    # 3. Run frequency sweep
    r = client.post("/api/v1/analysis/frequency-sweep", json={
        "composition": {"Cu": 70.0, "Zn": 30.0},
        "thickness_mm": 1.0,
        "freq_start_mhz": 1.0,
        "freq_end_mhz": 10000.0,
        "num_points": 50,
    })
    assert r.status_code == 200
    sweep = r.json()
    assert len(sweep["frequencies_mhz"]) == 50
    # SE should increase with frequency for metals
    assert sweep["total_se_db"][-1] > sweep["total_se_db"][0]
```

- [ ] **Step 2: Run smoke tests**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/test_backend_smoke.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite one final time**

Run: `cd /Users/danusharun/Documents/EMI-shielding && python -m pytest tests/ -v --tb=short`
Expected: All tests pass, zero Streamlit dependencies remain

- [ ] **Step 4: Commit**

```bash
git add tests/test_backend_smoke.py
git commit -m "test: add smoke tests for full backend verification"
```

---

## Summary of What This Plan Produces

After completing all 6 tasks:

1. **Clean `src/` layer** - Pure Python physics, chemistry, and materials logic with no UI dependencies
2. **Complete FastAPI backend** with endpoints for:
   - Single-point EMI calculation
   - Material properties lookup (periodic table + alloys)
   - Composite property calculation
   - Frequency sweep, thickness sweep, grain size sweep, cooling rate sweep
   - Thickness optimization
3. **Test suite** covering all extracted logic and API endpoints
4. **Zero Streamlit dependencies** - the project is ready for Next.js frontend development (Phase 2)
5. **Gemini API key** config ready for AI chat integration (Phase 2)
