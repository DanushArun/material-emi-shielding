"""Material recommendation engine -- inverse SE solver.

Given a set of engineering constraints (target SE, frequency, max thickness,
optional weight limit), searches the material database and returns ranked
material recommendations using the EMICalculator forward solver.

Two public functions:
    recommend_materials(constraints, n_results) -> List[MaterialRecommendation]
    pareto_filter(recommendations) -> List[MaterialRecommendation]
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.materials.material_properties import material_db
from src.physics.emi_calculations import EMICalculator

# Module-level singletons -- reuse existing instances, never re-instantiate
_db = material_db
_calculator = EMICalculator()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MaterialConstraints:
    """Engineering constraints for the inverse SE search."""

    target_se_db: float                             # Minimum required SE (dB)
    frequency_hz: float                             # Primary frequency (Hz)
    max_thickness_m: float                          # Maximum allowable thickness (m)
    max_density_kg_m3: Optional[float] = None       # Optional weight constraint
    frequency_range_hz: Optional[Tuple[float, float]] = None  # Optional band
    thickness_step_m: float = 0.1e-3                # 0.1 mm resolution


@dataclass
class MaterialRecommendation:
    """A single material recommendation with full SE breakdown."""

    material_name: str
    achieved_se_db: float
    optimal_thickness_m: float
    density_kg_m3: float
    se_margin_db: float = 0.0
    meets_target: bool = True
    conductivity_s_m: float = 0.0
    relative_permeability: float = 1.0
    reflection_loss_db: float = 0.0
    absorption_loss_db: float = 0.0
    skin_depth_m: float = 0.0
    explanation: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_valid_candidate(props: dict) -> bool:
    """Return True if the material has positive conductivity and density."""
    cond = props.get("electrical_conductivity", 0)
    dens = props.get("density", 0)
    return cond is not None and dens is not None and cond > 0 and dens > 0


def _evaluate_candidate(
    name: str,
    props: dict,
    constraints: MaterialConstraints,
) -> Optional[MaterialRecommendation]:
    """Evaluate a single material against the constraints.

    Sweeps thickness from one step to max_thickness_m, finds the minimum
    thickness that achieves the target SE (if any), and returns a
    MaterialRecommendation with full SE breakdown at that thickness.
    """
    conductivity = props["electrical_conductivity"]
    rel_perm = props.get("relative_permeability", 1.0)
    rel_eps = props.get("relative_permittivity", 1.0)
    density = props["density"]
    display_name = props.get("name", name)

    # Density filter -- skip entirely if above limit
    if constraints.max_density_kg_m3 is not None and density > constraints.max_density_kg_m3:
        return None

    step = constraints.thickness_step_m
    n_steps = max(1, int(round(constraints.max_thickness_m / step)))

    best_result = None
    optimal_thickness = None

    for i in range(1, n_steps + 1):
        t = i * step
        result = _calculator.calculate_shielding_effectiveness(
            conductivity=conductivity,
            relative_permeability=rel_perm,
            relative_permittivity=rel_eps,
            thickness=t,
            frequency=constraints.frequency_hz,
            include_confidence=False,
        )
        se = result["total_se"]
        if se >= constraints.target_se_db:
            best_result = result
            optimal_thickness = t
            break

    # If no thickness meets target, use max thickness as best effort
    if best_result is None:
        t = constraints.max_thickness_m
        best_result = _calculator.calculate_shielding_effectiveness(
            conductivity=conductivity,
            relative_permeability=rel_perm,
            relative_permittivity=rel_eps,
            thickness=t,
            frequency=constraints.frequency_hz,
            include_confidence=False,
        )
        optimal_thickness = t

    achieved_se = float(best_result["total_se"])
    meets = bool(achieved_se >= constraints.target_se_db)
    margin = float(achieved_se - constraints.target_se_db)

    # Build a human-readable explanation
    freq_ghz = constraints.frequency_hz / 1e9
    thickness_mm = optimal_thickness * 1e3
    if meets:
        explanation = (
            f"{display_name} achieves {achieved_se:.1f} dB SE at "
            f"{freq_ghz:.2f} GHz with {thickness_mm:.1f} mm thickness "
            f"({margin:.1f} dB margin over target)."
        )
    else:
        explanation = (
            f"{display_name} reaches only {achieved_se:.1f} dB SE at "
            f"{freq_ghz:.2f} GHz at maximum thickness {thickness_mm:.1f} mm "
            f"({abs(margin):.1f} dB below target)."
        )

    return MaterialRecommendation(
        material_name=display_name,
        achieved_se_db=achieved_se,
        optimal_thickness_m=optimal_thickness,
        density_kg_m3=float(density),
        se_margin_db=margin,
        meets_target=meets,
        conductivity_s_m=float(conductivity),
        relative_permeability=float(rel_perm),
        reflection_loss_db=float(best_result["reflection_loss"]),
        absorption_loss_db=float(best_result["absorption_loss"]),
        skin_depth_m=float(best_result["skin_depth"]),
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend_materials(
    constraints: MaterialConstraints,
    n_results: int = 5,
) -> List[MaterialRecommendation]:
    """Search the material database and return ranked recommendations.

    Algorithm:
        1. Iterate pure elements from ``_db.periodic_table`` (conductivity > 0, density > 0).
        2. Iterate alloys from ``_db.alloys`` (same checks).
        3. For each candidate, sweep thickness from 0.1 mm to *max_thickness_m*.
        4. Find minimum thickness where SE >= target.
        5. Check density constraint if specified.
        6. Sort: meeting-target first, then by SE margin desc, thickness asc, density asc.
        7. Return top *n_results*.

    Parameters
    ----------
    constraints : MaterialConstraints
        Engineering requirements.
    n_results : int
        Maximum number of recommendations to return (default 5).

    Returns
    -------
    list of MaterialRecommendation
        Ranked material recommendations.
    """
    candidates: List[MaterialRecommendation] = []

    # 1. Pure elements
    for symbol, props in _db.periodic_table.items():
        if not _is_valid_candidate(props):
            continue
        rec = _evaluate_candidate(symbol, props, constraints)
        if rec is not None:
            candidates.append(rec)

    # 2. Alloys
    for key, props in _db.alloys.items():
        if not _is_valid_candidate(props):
            continue
        rec = _evaluate_candidate(key, props, constraints)
        if rec is not None:
            candidates.append(rec)

    # 3. Sort: meeting target first, then SE margin desc, thickness asc, density asc
    candidates.sort(
        key=lambda r: (
            not r.meets_target,         # True (not meeting) sorts after False
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

    Three objectives (all to be minimised after transformation):
        - Maximise SE margin  -> minimise ``-se_margin_db``
        - Minimise thickness  -> minimise ``optimal_thickness_m``
        - Minimise density    -> minimise ``density_kg_m3``

    A solution *A* dominates *B* iff A is at least as good as B on every
    objective and strictly better on at least one.

    Parameters
    ----------
    recommendations : list of MaterialRecommendation

    Returns
    -------
    list of MaterialRecommendation
        Non-dominated (Pareto-optimal) subset.
    """
    if len(recommendations) <= 1:
        return list(recommendations)

    # Build objective matrix: columns = (-margin, thickness, density)
    n = len(recommendations)
    objs = np.array([
        [-r.se_margin_db, r.optimal_thickness_m, r.density_kg_m3]
        for r in recommendations
    ])

    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # Does j dominate i?  j <= i on all objectives AND j < i on at least one
            if (np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i])):
                is_dominated[i] = True
                break

    return [r for r, dom in zip(recommendations, is_dominated) if not dom]
