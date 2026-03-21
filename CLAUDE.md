# EMI Shield Designer - CLAUDE.md

<!-- AUTO-MANAGED: project-description -->
## Project Description

EMI Shield Designer: a physics-based platform for calculating electromagnetic interference shielding effectiveness (SE) of metallic and composite materials. Computes reflection loss, absorption loss, multiple reflection correction, and skin depth using analytical EM theory. Supports grain-size microstructure effects (Mayadas-Shatzkes), multilayer stacks (TMM), composite conductivity models, temperature/frequency-dependent material properties, and Monte Carlo uncertainty quantification.

- **Version:** 4.0 (active branch: `version-4.0`)
- **Status:** Feature-complete — Streamlit monolith deleted (`app.py`, `auth.py` removed); all five physics modules implemented; 160 Python tests passing; Next.js frontend builds clean
- **Ports:** backend on 8001 (`uvicorn main:app --port 8001`); frontend dev server on 3001 (`next dev -p 3001`)
- **Plan doc:** `docs/superpowers/plans/2026-03-21-cleanup-and-refactor.md`
- **Methodology:** `docs/papers/methodology-paper.md` — publication-quality paper covering all five physics modules and validation against 106 experimental measurements
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
| `src/chemistry/parser.py` | `ChemicalParser` + `ReactionEngine` — pure Python, no UI dependencies |
| `src/chemistry/__init__.py` | Exports `ChemicalParser`, `ReactionEngine` |
| `src/materials/material_properties.py` | `material_db` — periodic table + alloy database |
| `src/utils/constants.py` | Physical constants: `MU_0`, `EPSILON_0`, `Z_0`, `C`; validators |

### backend/ structure

```
backend/
  main.py                        FastAPI app entry point; registers physics, materials, analysis,
                                 chat, multilayer, composites, advanced; auth in try/except
  core/config.py                 Pydantic Settings (env vars, APP_VERSION=4.0.0)
  api/v1/routes/
    _helpers.py                  Shared calculate_composite_properties() used by all routes
    physics.py                   Single-point SE calculation + material-properties
    analysis.py                  Sweeps and optimization endpoints
    materials.py                 Elements, alloys, composite-properties endpoints
    multilayer.py                N-layer TMM endpoints (POST /calculate, /frequency-sweep,
                                 /optimize); registered at /api/v1/multilayer
    composites.py                Composite conductivity endpoints (POST /percolation, /gem,
                                 /threshold-estimate, /all-models); registered at /api/v1/composites
    chat.py                      Gemini AI chat endpoint (POST /message); registered at
                                 /api/v1/chat; model gemini-1.5-flash; requires GEMINI_API_KEY
    advanced.py                  Advanced physics endpoints; registered at /api/v1/advanced
    auth.py                      Authentication (optional — loaded in try/except, requires DB)
  api/v1/schemas/
    physics.py                   Request/response models for physics + analysis routes
    materials.py                 Request/response models for materials routes
```

Note: `app.py` and `auth.py` (root-level Streamlit files) have been deleted. `.env.example` documents all required env vars.

### frontend/ structure

```
frontend/
  app/
    layout.tsx                   Root layout — Inter + JetBrains Mono fonts, dark mode, wraps <Providers>
    globals.css                  Tailwind + CSS custom properties (--bg-primary, --accent-cyan, etc.)
                                 Component utility classes: .card, .btn-primary, .btn-secondary, .input-field
    page.tsx                     Home landing page — inline sticky nav (no Header component), hero,
                                 three FeatureCards (Physics/AI/UQ), system status bar (live health check),
                                 footer with link to localhost:8001/docs
    materials/
      page.tsx                   Materials Library — tabs: Periodic Table (lazy, ssr:false) + Alloys;
                                 Alloys tab: searchable AlloyCard grid (sm:2/lg:3/xl:4 cols),
                                 AlloyData fetched via fetchAlloys(); uses Header component
  components/
    ui/
      Header.tsx                 Shared sticky header — active link via usePathname(), ApiStatusBadge
                                 (checking/healthy/offline dot), VersionBadge (falls back to v4.0.0);
                                 used by all pages except app/page.tsx
    simulation/
      PeriodicTable.tsx          118-element interactive grid — ELEMENT_POSITIONS maps all elements to
                                 18-column layout (rows 8/9 = lanthanide/actinide); deriveCategory()
                                 with CATEGORY_STYLES; click → tooltip with EM properties; uses
                                 react-query + fetchElements(); loaded via next/dynamic (ssr:false)
      CompositionPanel.tsx       Composition editor — ElementRow (badge + input + progress bar + remove),
                                 PresetDropdown (searchable alloy list), normalize button (fixes
                                 floating-point drift, last element adjusted); total badge green
                                 within 0.05% of 100; uses useCompositionStore + react-query
      ShieldParameters.tsx       Analysis mode selector (6 modes) + conditional parameter inputs;
                                 exports SweepParams interface; lifts sweepParams/targetSE to parent;
                                 uses useSimulationStore
  lib/
    api.ts                       axios client; baseURL = NEXT_PUBLIC_API_URL (default http://localhost:8001);
                                 30s timeout; endpoints: /api/v1/materials/*, /api/v1/physics/calculate,
                                 /api/v1/analysis/*, /api/v1/chat/message (sendChatMessage), /health
    store.ts                     Zustand stores: useCompositionStore, useSimulationStore,
                                 useResultsStore, useHistoryStore, useChatStore
  types/
    index.ts                     TypeScript interfaces: ElementData, AlloyData, CompositeProperties,
                                 CalculationRequest, CalculationResult, FrequencySweepRequest,
                                 SweepResult, OptimizationResult, ShieldLayerConfig,
                                 ChatMessage, HistoryEntry
  next.config.js                 NEXT_PUBLIC_API_URL env passthrough; serverActions enabled
  package.json                   dev script runs on port 3001 (next dev -p 3001)
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

# Start backend API (port 8001)
cd backend && uvicorn main:app --reload --port 8001

# Start frontend (Next.js, port 3001)
cd frontend && npm run dev
# equivalent: next dev -p 3001
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: conventions -->
## Conventions

### Imports
- `src/` modules use absolute imports starting with `src.` (e.g. `from src.materials.material_properties import material_db`)
- Backend routes import from `src.*` and `backend.api.v1.*` only
- No Streamlit imports anywhere in `src/` or `backend/`
- `tests/conftest.py` inserts project root and `backend/` onto `sys.path` — all test files rely on this; do not add path hacks in individual test files

### Composite property calculation
- All routes call `calculate_composite_properties(composition)` from `backend/api/v1/routes/_helpers.py` — never inline
- conductivity: weighted sum; permeability: geometric mean (`elem_perm ** weight`); permittivity: geometric mean; density: weighted sum
- Raises `ValueError` for unknown elements

### Configuration (backend/core/config.py)
- `GEMINI_API_KEY` — Gemini AI conversational design assistant; defaults to empty string (feature disabled)
- `CORS_ORIGINS` — defaults to `["http://localhost:3001", "http://localhost:8001"]`; accepts a comma-separated string or a list; `@field_validator('CORS_ORIGINS', mode='before')` handles both
- `ML_MODEL_PATH` and `ML_CACHE_ENABLED` have been removed from Settings
- Auth router is optional: registered in `try/except` in `main.py`; skipped silently if DB is unavailable
- Copy `.env.example` to `.env` for local dev; required vars: `SECRET_KEY`, `GEMINI_API_KEY`, `DATABASE_URL`, `REDIS_URL`

### Chat route
- `chat.py` enriches the user message with a `[Current simulation context]` block when `request.context` is provided — includes composition, frequency_mhz, thickness_mm, grain_size_um, analysisMode, latestResult SE
- Gemini role mapping: `'assistant'` or `'model'` → `'model'`; all other roles → `'user'`
- `google.generativeai` is imported lazily inside the endpoint; raises HTTP 503 if the package is missing or `GEMINI_API_KEY` is unset
- `_extract_suggestions()` uses keyword heuristics (copper, permeability, skin depth, multilayer, absorption, reflection, composite, frequency) to pick up to 3 follow-up suggestions from the response text

### Composites route
- `composites.py` calls `composite_models` functions directly — no `_helpers.py` involvement (composites bypass `calculate_composite_properties`)
- `/threshold-estimate` uses a `filler_type: Literal["rod", "disk"]` discriminator; rod requires `length_um` + `diameter_um`; disk requires `radius_um` + `thickness_um`
- `/all-models` runs percolation, GEM, Maxwell-Garnett, Bruggeman, and Hashin-Shtrikman bounds in a single call for side-by-side comparison
- All conductivity values in S/m; volume fractions dimensionless [0, 1]

### Frontend conventions (frontend/)
- **API client** (`lib/api.ts`): single axios instance, `baseURL = NEXT_PUBLIC_API_URL || 'http://localhost:8001'`, 30s timeout; all API calls go through named functions in this file — never inline `axios` calls in components
- **State management** (`lib/store.ts`): five Zustand stores — import named hook (e.g. `useCompositionStore`) directly; do not use React context for global state
  - `useSimulationStore` default analysis mode: `'single'`; all 6 modes: `'single'`, `'frequency-sweep'`, `'thickness-sweep'`, `'grain-sweep'`, `'cooling-sweep'`, `'optimize'`
  - `useHistoryStore` caps at 100 entries (newest first)
- **TypeScript types** (`types/index.ts`): all shared interfaces live here; import from `@/types` — do not re-declare inline; `SweepParams` is the exception — it is exported from `components/simulation/ShieldParameters.tsx`
- **CSS custom properties**: defined in `app/globals.css` under `:root`; use CSS var references (`var(--accent-cyan)`) or the Tailwind utility classes (`.card`, `.btn-primary`, `.btn-secondary`, `.input-field`) instead of ad-hoc inline styles
- **Ports**: frontend dev server on 3001 (`next dev -p 3001`); backend expected at 8001 (`NEXT_PUBLIC_API_URL`)
- **Header component**: use `<Header />` from `components/ui/Header.tsx` on all pages — except `app/page.tsx` which owns its own inline sticky nav (brand + nav links + status badge)
- **react-query**: used in `PeriodicTable` and `CompositionPanel`; alloys query uses `staleTime: Infinity` (fetched once per session); `PeriodicTable` loaded via `next/dynamic({ ssr: false })` to avoid SSR grid complexity
- **CompositionPanel normalize**: the normalize button corrects floating-point drift by adjusting the last element so the sum is exactly 100; total badge turns green when within 0.05% of 100, amber otherwise

### Pydantic schemas
- All request/response models live in `backend/api/v1/schemas/` — not inline in route files
- Exceptions (self-contained routes with inline schemas):
  - `multilayer.py`: `LayerSpec`, `MultilayerCalculateRequest`, `MultilayerSweepRequest`, `MultilayerOptimizeRequest`
  - `composites.py`: `FillerMatrixParams`, `PercolationRequest`, `GEMRequest`, `ThresholdEstimateRequest`, `AllModelsRequest`
  - `chat.py`: `HistoryMessage`, `ChatRequest`, `ChatResponse`
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
- **Multilayer route pattern:** `multilayer.py` uses `_build_shield(layers: List[LayerSpec]) -> MultilayerShield` as an internal helper — converts each `LayerSpec` (composition dict + thickness_mm) into a `ShieldLayer` via `calculate_composite_properties`, then calls `shield.add_layer()`. The route handler calls `_build_shield`, then `shield.calculate_se(frequency_hz)` or the sweep/optimize equivalent.
- **Unit conversion at route boundary:** Inputs arrive in human-friendly units (MHz, mm, um) and are converted to SI (Hz, m) before passing to `EMICalculator`. Multilayer: `thickness_mm * 1e-3` per layer, `frequency_mhz * 1e6` at call site.
- **Global calculator instance:** `calculator = EMICalculator()` is module-level in each route file; `emi_calculator` global also exported from `emi_calculations.py` for tests.
- **Sweep → list conversion:** All numpy arrays in sweep results are serialized via `.tolist()` before returning from endpoints.
- **MultilayerShield overflow guard:** `gd` is capped at 500.0 (preserving phase) when `real(gd) > 500` to avoid float64 overflow in cosh/sinh for thick conductors. This produces a lower-bound SE estimate rather than NaN/inf.
- **MC sampling strategy:** `UncertaintySpec.grain_size_cv` drives log-normal sampling (right-skewed manufacturing spread); all other CVs (`conductivity_cv`, `thickness_cv`, `permeability_cv`, `frequency_cv`) use normal sampling clipped to > 0.
- **Chat route pattern:** `chat.py` builds `gemini_history` from `request.history` (role mapping: assistant/model → model), optionally prepends a `[Current simulation context]` block to the user message, calls `model.start_chat(history=gemini_history).send_message(user_message)`, then passes response text through `_extract_suggestions()` for up to 3 follow-up hints.
- **Composites route pattern:** `composites.py` routes call `composite_models` functions directly (no `_helpers.py`); `/percolation` and `/gem` share `FillerMatrixParams` base schema; `/threshold-estimate` dispatches on `filler_type` literal to `percolation_threshold_rods()` or `percolation_threshold_disks()`; `/all-models` aggregates all five model results in one response dict.
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: git-insights -->
## Git Insights

- All backend routers now registered in `main.py`: physics, materials, analysis, chat (`/api/v1/chat`), multilayer (`/api/v1/multilayer`), composites (`/api/v1/composites`), advanced (`/api/v1/advanced`); auth remains optional via try/except
- `chat.py` added: Gemini AI conversational assistant (gemini-1.5-flash); `google-generativeai>=0.7.0` added to `backend/requirements.txt`; `sendChatMessage()` added to `frontend/lib/api.ts` (POST /api/v1/chat/message with optional simulation context)
- `composites.py` added: four endpoints wrapping `src.physics.composite_models` — percolation, GEM, threshold-estimate (rod/disk geometry), all-models comparison
- `backend/api/v1/routes/multilayer.py` implemented and registered: three endpoints (POST /calculate, /frequency-sweep, /optimize) wrapping `MultilayerShield` TMM; uses inline Pydantic schemas (`LayerSpec`, `MultilayerCalculateRequest`, etc.) and `_build_shield()` internal helper
- Project reached feature-complete state: 160 Python tests passing across all modules (chemistry, physics engine, multilayer, composite models, material models, uncertainty); frontend builds clean with no type errors
- `app/simulation/page.tsx` is the full simulation UI — three-panel layout (CompositionPanel + ShieldParameters + ResultsPanel) with integrated AI chat (useChatStore); all 6 analysis modes wired to API: calculateSE, frequencySweep, thicknessSweep, grainSizeSweep, coolingRateSweep, optimizeThickness
- New frontend pages and simulation components added: `app/page.tsx` (home landing), `app/materials/page.tsx` (materials library with periodic table + alloy grid), `components/simulation/PeriodicTable.tsx` (118-element interactive grid), `components/simulation/CompositionPanel.tsx` (composition editor with preset alloys), `components/simulation/ShieldParameters.tsx` (6-mode analysis selector), `components/ui/Header.tsx` (shared sticky nav with API health badge)
- Frontend infrastructure added: `lib/api.ts` (axios client → port 8001), `lib/store.ts` (5 Zustand stores), `types/index.ts` (shared TS interfaces); frontend dev server pinned to port 3001; CORS origins updated to `[localhost:3001, localhost:8001]`
- `336294e` — Streamlit removed: `app.py` and root `auth.py` deleted; all four routers (physics, materials, analysis, auth) registered in `backend/main.py`; `GEMINI_API_KEY` added to Settings; `ML_MODEL_PATH`/`ML_CACHE_ENABLED` removed; auth router made optional via try/except
- `ab54354` — Added `docs/papers/methodology-paper.md`: publication-quality paper documenting all five physics modules (Schelkunoff, TMM, McLachlan GEM, Snoek/Debye, TCR, Monte Carlo UQ); validated against 106 experimental SE measurements (MAE 2.1 dB metals, 4.3 dB composites)
- `c9b49ac` — Major refactor: extracted `ChemicalParser`/`ReactionEngine` from app.py into `src/chemistry/parser.py`; built full backend API with schemas, routes, helpers; cleaned Streamlit imports from physics engine
- `4f7b9f2` — Scientific research phase: 6 parallel investigations for physics model validation
<!-- END AUTO-MANAGED -->

<!-- MANUAL -->
## Notes

Add any project-specific notes here that should not be auto-updated.
<!-- END MANUAL -->
