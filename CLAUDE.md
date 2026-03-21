# EMI Shield Designer - CLAUDE.md

<!-- AUTO-MANAGED: project-description -->
## Project Description

EMI Shield Designer: a physics-based tool for calculating electromagnetic interference shielding effectiveness (SE) of metallic and composite materials. Computes reflection loss, absorption loss, multiple reflection correction, and skin depth using analytical EM theory. Supports grain-size microstructure effects via the Mayadas-Shatzkes model.

- **Version:** 4.0 (active branch: `version-4.0`)
- **Status:** Active refactor — migrating from Streamlit monolith to three-layer architecture
- **Plan doc:** `docs/superpowers/plans/2026-03-21-cleanup-and-refactor.md`
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Architecture

Three strict layers — no cross-layer imports allowed:

```
src/          Pure Python physics/materials/chemistry logic. Zero UI dependencies.
backend/      FastAPI app. Imports only from src/. Exposes REST endpoints.
frontend/     Next.js 14 app. Calls backend API only.
```

### src/ modules

| Module | Purpose |
|--------|---------|
| `src/physics/emi_calculations.py` | Core EMI calculations — `EMICalculator` class + global `emi_calculator` instance |
| `src/physics/advanced_microstructure.py` | Grain size / cooling rate microstructure modeling (`AdvancedMicrostructure`, `ProcessingParams`) |
| `src/physics/composite_models.py` | Composite conductivity models — percolation theory (rods/disks), Maxwell-Garnett, Bruggeman EMT |
| `src/physics/material_models.py` | Temperature/frequency-dependent material properties — TCR model, Curie falloff, Snoek's law, Debye permeability |
| `src/physics/multilayer.py` | Transfer Matrix Method for N-layer shields — `ShieldLayer` dataclass, `MultilayerShield` class, `calculate_se()` |
| `src/physics/uncertainty.py` | Monte Carlo uncertainty quantification — `UncertaintySpec`, `monte_carlo_se()` |
| `src/chemistry/parser.py` | `ChemicalParser` + `ReactionEngine` — extracted from app.py, no Streamlit deps |
| `src/chemistry/__init__.py` | Exports `ChemicalParser`, `ReactionEngine` |
| `src/materials/material_properties.py` | `material_db` — periodic table + alloy database |
| `src/utils/constants.py` | Physical constants: `MU_0`, `EPSILON_0`, `Z_0`, `C`; validators |

### backend/ structure

```
backend/
  main.py                        FastAPI app entry point
  core/config.py                 Pydantic Settings (env vars, APP_VERSION=4.0.0)
  api/v1/routes/
    _helpers.py                  Shared calculate_composite_properties() used by all routes
    physics.py                   Single-point SE calculation + material-properties
    analysis.py                  Sweeps and optimization endpoints
    materials.py                 Elements, alloys, composite-properties endpoints
  api/v1/schemas/
    physics.py                   Request/response models for physics + analysis routes
    materials.py                 Request/response models for materials routes
```

### data_collection/

- `experimental_benchmark_data.py` — curated SE measurements from peer-reviewed literature (pure metals, composites, multilayer, microstructure effects, 5G/Wi-Fi/radar bands)
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: build-commands -->
## Build Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_chemistry_parser.py -v
python -m pytest tests/test_physics_engine.py -v

# Start backend API
cd backend && uvicorn main:app --reload

# Start frontend (Next.js)
cd frontend && npm run dev
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: conventions -->
## Conventions

### Imports
- `src/` modules use absolute imports starting with `src.` (e.g. `from src.materials.material_properties import material_db`)
- Backend routes import from `src.*` and `backend.api.v1.*` — never from `app.py` or Streamlit
- No Streamlit imports anywhere in `src/` or `backend/`
- `tests/conftest.py` inserts project root and `backend/` onto `sys.path` — all test files rely on this; do not add path hacks in individual test files

### Composite property calculation
- All routes call `calculate_composite_properties(composition)` from `backend/api/v1/routes/_helpers.py` — never inline
- conductivity: weighted sum; permeability: geometric mean (`elem_perm ** weight`); permittivity: geometric mean; density: weighted sum
- Raises `ValueError` for unknown elements

### Pydantic schemas
- All request/response models live in `backend/api/v1/schemas/` — not inline in route files
- Use Pydantic v2 (`model_config` / `json_schema_extra`, not `class Config` where possible)
- Physics input units: frequency in MHz, thickness in mm, grain size in um — converted to SI inside route handlers

### Chemical formulas
- `ChemicalParser.format_formula()` returns plain text (e.g. `"Al2O3"`), never HTML with `<sub>` tags
- `ReactionEngine.get_total_composition()` returns weight percentages as `Dict[str, float]`

### EMICalculator return shapes
- `calculate_shielding_effectiveness()` → `dict` with keys: `total_se`, `reflection_loss`, `absorption_loss`, `skin_depth`; optional `confidence`, `confidence_level`, `conductivity_reduction`
- `frequency_sweep()` → `dict` with numpy arrays: `frequencies`, `total_ses`, `reflection_losses`, `absorption_losses`, `skin_depths`
- `thickness_sweep()` → same shape but `thicknesses` instead of `frequencies`
- `grain_size_sweep()` → `grain_sizes`, `total_ses`, `effective_conductivities`, `reflection_losses`, `absorption_losses`
- `optimize_thickness()` → `optimal_thickness`, `achieved_se`, `reflection_loss`, `absorption_loss`, `skin_depths`

### New physics module APIs
- `MultilayerShield.calculate_se(frequency)` → `dict`: `total_se`, `reflection_loss`, `absorption_loss`, `transmission_coefficient` (|S21|), `reflection_coefficient` (|S11|)
- `monte_carlo_se()` → `dict`: `se_mean`, `se_std`, `se_ci_lower` (2.5th pct), `se_ci_upper` (97.5th pct), `se_distribution`, `n_samples`, `deterministic_se`
- `multilayer.py` imports `MU_0`, `EPSILON_0`, `Z_0` from `src.utils.constants`
- `uncertainty.py` uses a module-level `_calculator = EMICalculator()` instance — do not instantiate per-call
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: patterns -->
## Patterns

- **Route pattern:** Each route calls `calculate_composite_properties(request.composition)`, optionally applies `calculator.calculate_grain_size_effect()`, then calls the relevant `EMICalculator` sweep/calculate method. Errors caught as `ValueError`/`KeyError` → `HTTPException(400)`.
- **Unit conversion at route boundary:** Inputs arrive in human-friendly units (MHz, mm, um) and are converted to SI (Hz, m) before passing to `EMICalculator`.
- **Global calculator instance:** `calculator = EMICalculator()` is module-level in each route file; `emi_calculator` global also exported from `emi_calculations.py` for tests.
- **Sweep → list conversion:** All numpy arrays in sweep results are serialized via `.tolist()` before returning from endpoints.
- **MultilayerShield overflow guard:** `gd` is capped at 500.0 (preserving phase) when `real(gd) > 500` to avoid float64 overflow in cosh/sinh for thick conductors. This produces a lower-bound SE estimate rather than NaN/inf.
- **MC sampling strategy:** `UncertaintySpec.grain_size_cv` drives log-normal sampling (right-skewed manufacturing spread); all other CVs (`conductivity_cv`, `thickness_cv`, `permeability_cv`, `frequency_cv`) use normal sampling clipped to > 0.
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: git-insights -->
## Git Insights

- `c9b49ac` — Major refactor: extracted `ChemicalParser`/`ReactionEngine` from app.py into `src/chemistry/parser.py`; built full backend API with schemas, routes, helpers; cleaned Streamlit imports from physics engine
- `4f7b9f2` — Scientific research phase: 6 parallel investigations for physics model validation
- `d97f320` — Added foundational citations to `docs/references.bib` and physics model analysis
- Active plan to delete `app.py` (2129-line Streamlit monolith) and `auth.py` — see refactor plan doc
<!-- END AUTO-MANAGED -->

<!-- MANUAL -->
## Notes

Add any project-specific notes here that should not be auto-updated.
<!-- END MANUAL -->
