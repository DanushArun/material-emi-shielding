"""
Composite material conductivity models for EMI shielding applications.

Implements percolation theory, effective medium theories (EMT), and
bounding models used to predict the electrical conductivity of
heterogeneous composite materials containing conductive fillers such as
carbon nanotubes (CNTs), graphene, MXene flakes, and metallic particles.

References:
    - McLachlan, D.S. et al., J. Am. Ceram. Soc. 73 (1990) 2187-2203
    - Hashin, Z. & Shtrikman, S., Phys. Rev. 130 (1963) 129-133
    - Maxwell Garnett, J.C., Philos. Trans. R. Soc. 203 (1904) 385-420
    - Bruggeman, D.A.G., Ann. Phys. 416 (1935) 636-664
"""

import numpy as np
from scipy.optimize import brentq
from typing import Tuple


# ---------------------------------------------------------------------------
# Percolation models
# ---------------------------------------------------------------------------

def percolation_conductivity(
    filler_fraction: float,
    percolation_threshold: float,
    sigma_filler: float,
    sigma_matrix: float,
    t: float = 2.0,
) -> float:
    """
    Power-law percolation model for composite conductivity.

    Above the percolation threshold f_c the conductivity follows a power law
    in the reduced volume fraction.  Below f_c the composite is dominated by
    the insulating matrix.

    Equation (above threshold):
        sigma_eff = sigma_filler * ((f - f_c) / (1 - f_c))^t

    Equation (below threshold):
        sigma_eff = sigma_matrix

    Args:
        filler_fraction: Volume fraction of conductive filler (0 <= f <= 1).
        percolation_threshold: Critical volume fraction f_c at which a
            continuous conductive network first forms (0 < f_c < 1).
        sigma_filler: Conductivity of the filler phase (S/m).
        sigma_matrix: Conductivity of the insulating matrix (S/m).
        t: Critical exponent controlling the sharpness of the percolation
            transition.  Universal value for 3-D systems is ~2.0.

    Returns:
        Effective electrical conductivity of the composite (S/m).

    Raises:
        ValueError: If filler_fraction or percolation_threshold are outside
            [0, 1], or if sigma values are negative.
    """
    if not (0.0 <= filler_fraction <= 1.0):
        raise ValueError(
            f"filler_fraction must be in [0, 1], got {filler_fraction}"
        )
    if not (0.0 < percolation_threshold < 1.0):
        raise ValueError(
            f"percolation_threshold must be in (0, 1), got {percolation_threshold}"
        )
    if sigma_filler < 0 or sigma_matrix < 0:
        raise ValueError("Conductivity values must be non-negative.")

    if filler_fraction <= percolation_threshold:
        return sigma_matrix

    reduced = (filler_fraction - percolation_threshold) / (1.0 - percolation_threshold)
    return sigma_filler * (reduced ** t)


def percolation_threshold_rods(length: float, diameter: float) -> float:
    """
    Estimate the percolation threshold for cylindrical fillers (e.g. CNTs).

    High-aspect-ratio rods form a connected network at very low volume
    fractions.  The excluded-volume argument gives:

        f_c ~ 0.7 / AR

    where AR = length / diameter is the aspect ratio.

    Args:
        length: Length of the cylindrical filler particle (m).
        diameter: Diameter of the cylindrical filler particle (m).

    Returns:
        Estimated percolation threshold volume fraction (dimensionless).

    Raises:
        ValueError: If length or diameter are non-positive.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if diameter <= 0:
        raise ValueError(f"diameter must be positive, got {diameter}")

    aspect_ratio = length / diameter
    return 0.7 / aspect_ratio


def percolation_threshold_disks(radius: float, thickness: float) -> float:
    """
    Estimate the percolation threshold for disk-shaped fillers
    (e.g. graphene flakes, MXene sheets).

    Thin disk fillers cover a large projected area relative to their volume,
    so they percolate at very low loading.  The excluded-volume estimate is:

        f_c ~ 0.5 / AR

    where AR = radius / thickness is the aspect ratio.

    Args:
        radius: Radius of the disk filler (m).
        thickness: Thickness of the disk filler (m).

    Returns:
        Estimated percolation threshold volume fraction (dimensionless).

    Raises:
        ValueError: If radius or thickness are non-positive.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    aspect_ratio = radius / thickness
    return 0.5 / aspect_ratio


# ---------------------------------------------------------------------------
# Effective medium theories
# ---------------------------------------------------------------------------

def maxwell_garnett(
    sigma_host: float,
    sigma_inclusion: float,
    volume_fraction: float,
    depolarization: float = 1.0 / 3.0,
) -> float:
    """
    Maxwell-Garnett effective medium theory for dilute spherical inclusions.

    Valid when the inclusion volume fraction is small (f << 1) and the
    inclusions are well separated so that inter-particle interactions can be
    neglected.

    Equation:
        sigma_eff = sigma_host * [
            1 + f * (sigma_i - sigma_h) /
                (sigma_h + L * (1 - f) * (sigma_i - sigma_h))
        ]

    where L is the depolarization factor (1/3 for spheres).

    Args:
        sigma_host: Conductivity of the host / matrix medium (S/m).
        sigma_inclusion: Conductivity of the inclusion phase (S/m).
        volume_fraction: Volume fraction of inclusions (0 <= f < 1).
        depolarization: Depolarization factor L of the inclusion geometry.
            L = 1/3 for spheres, L = 0 for needles along field axis,
            L = 1 for disk perpendicular to field axis.

    Returns:
        Effective conductivity predicted by Maxwell-Garnett theory (S/m).

    Raises:
        ValueError: If volume_fraction is outside [0, 1) or conductivities
            are negative.
    """
    if not (0.0 <= volume_fraction < 1.0):
        raise ValueError(
            f"volume_fraction must be in [0, 1), got {volume_fraction}"
        )
    if sigma_host < 0 or sigma_inclusion < 0:
        raise ValueError("Conductivity values must be non-negative.")

    delta_sigma = sigma_inclusion - sigma_host
    numerator = volume_fraction * delta_sigma
    denominator = sigma_host + depolarization * (1.0 - volume_fraction) * delta_sigma

    if denominator == 0.0:
        raise ValueError(
            "Maxwell-Garnett denominator is zero; check input parameters."
        )

    return sigma_host * (1.0 + numerator / denominator)


def bruggeman_emt(
    sigma_1: float,
    sigma_2: float,
    f_1: float,
) -> float:
    """
    Bruggeman symmetric effective medium theory (self-consistent EMT).

    Treats both phases on equal footing; valid across the full composition
    range.  The implicit equation:

        f_1 * (sigma_1 - sigma_eff) / (sigma_1 + 2*sigma_eff)
      + f_2 * (sigma_2 - sigma_eff) / (sigma_2 + 2*sigma_eff) = 0

    is solved analytically via the quadratic formula.  With f_2 = 1 - f_1:

        A * sigma_eff^2 + B * sigma_eff + C = 0

    where:
        A = 3
        B = -(f_1*(3*sigma_1 - sigma_2) + sigma_2 * (3 - 1) ... )

    The positive physical root is returned.

    Closed-form coefficients (after expanding and collecting terms):
        A = 3
        B = -(f_1 * (3*sigma_1 - sigma_2) + (1 - f_1) * (3*sigma_2 - sigma_1)
               - sigma_1 - sigma_2)
           simplified to:
        B = -((3*f_1 - 1)*sigma_1 + (2 - 3*f_1)*sigma_2)
        C = sigma_1 * sigma_2 / 3  ... see derivation below.

    The standard closed-form result is:
        sigma_eff = (1/4) * {
            (3*f_1 - 1)*sigma_1 + (3*f_2 - 1)*sigma_2
            + sqrt([(3*f_1-1)*sigma_1 + (3*f_2-1)*sigma_2]^2
                   + 8*sigma_1*sigma_2)
        }

    Args:
        sigma_1: Conductivity of phase 1 (S/m).
        sigma_2: Conductivity of phase 2 (S/m).
        f_1: Volume fraction of phase 1 (0 <= f_1 <= 1).

    Returns:
        Effective conductivity predicted by Bruggeman EMT (S/m).

    Raises:
        ValueError: If f_1 is outside [0, 1] or conductivities are negative.
    """
    if not (0.0 <= f_1 <= 1.0):
        raise ValueError(f"f_1 must be in [0, 1], got {f_1}")
    if sigma_1 < 0 or sigma_2 < 0:
        raise ValueError("Conductivity values must be non-negative.")

    f_2 = 1.0 - f_1
    term1 = (3.0 * f_1 - 1.0) * sigma_1
    term2 = (3.0 * f_2 - 1.0) * sigma_2
    discriminant = (term1 + term2) ** 2 + 8.0 * sigma_1 * sigma_2

    sigma_eff = 0.25 * ((term1 + term2) + np.sqrt(discriminant))
    return float(sigma_eff)


def mclachlan_gem(
    f_filler: float,
    sigma_matrix: float,
    sigma_filler: float,
    f_c: float,
    t: float = 2.0,
) -> float:
    """
    McLachlan General Effective Media (GEM) equation.

    The GEM equation unifies percolation scaling and effective medium theory
    into a single implicit equation:

        f_filler * (sigma_filler^(1/t) - sigma_eff^(1/t)) /
                   (sigma_filler^(1/t) + A * sigma_eff^(1/t))
      + (1 - f_filler) * (sigma_matrix^(1/t) - sigma_eff^(1/t)) /
                         (sigma_matrix^(1/t) + A * sigma_eff^(1/t)) = 0

    where A = (1 - f_c) / f_c.

    The equation is solved numerically using scipy.optimize.brentq on the
    interval [sigma_matrix, sigma_filler] (or the reverse if sigma_matrix >
    sigma_filler).

    Args:
        f_filler: Volume fraction of conductive filler (0 <= f <= 1).
        sigma_matrix: Conductivity of the insulating matrix (S/m).
        sigma_filler: Conductivity of the filler phase (S/m).
        f_c: Percolation threshold volume fraction (0 < f_c < 1).
        t: Critical exponent for the percolation transition (~2.0 in 3-D).

    Returns:
        Effective conductivity predicted by the GEM equation (S/m).

    Raises:
        ValueError: If parameters are out of valid range.
        RuntimeError: If the root-finding algorithm fails to converge.
    """
    if not (0.0 <= f_filler <= 1.0):
        raise ValueError(f"f_filler must be in [0, 1], got {f_filler}")
    if not (0.0 < f_c < 1.0):
        raise ValueError(f"f_c must be in (0, 1), got {f_c}")
    if sigma_matrix < 0 or sigma_filler < 0:
        raise ValueError("Conductivity values must be non-negative.")

    # Avoid degenerate cases
    if sigma_filler == sigma_matrix:
        return sigma_matrix

    inv_t = 1.0 / t
    A = (1.0 - f_c) / f_c

    sigma_low = min(sigma_matrix, sigma_filler)
    sigma_high = max(sigma_matrix, sigma_filler)

    # Add small epsilon to avoid division by zero at the exact boundaries
    eps = 1e-30

    def gem_residual(sigma_eff: float) -> float:
        se = max(sigma_eff, eps)
        sf_t = sigma_filler ** inv_t
        sm_t = sigma_matrix ** inv_t
        se_t = se ** inv_t

        filler_term = f_filler * (sf_t - se_t) / (sf_t + A * se_t)
        matrix_term = (1.0 - f_filler) * (sm_t - se_t) / (sm_t + A * se_t)
        return filler_term + matrix_term

    # Bracket: residual must change sign across [sigma_low, sigma_high]
    lo = sigma_low + eps
    hi = sigma_high - eps

    try:
        sigma_eff = brentq(gem_residual, lo, hi, xtol=1e-12, rtol=1e-10)
    except ValueError as exc:
        raise RuntimeError(
            f"GEM root-finding failed to bracket a root for the given "
            f"parameters. Details: {exc}"
        ) from exc

    return float(sigma_eff)


def hashin_shtrikman_bounds(
    sigma_1: float,
    sigma_2: float,
    f_1: float,
) -> Tuple[float, float]:
    """
    Hashin-Shtrikman (HS) bounds on the effective conductivity of a
    two-phase isotropic composite.

    The HS bounds are the tightest possible bounds that can be derived from
    volume fraction information alone (without microstructural details).
    They are given by:

        sigma_HS- = sigma_1 + f_2 / (
            1/(sigma_2 - sigma_1) + f_1 / (3 * sigma_1)
        )

        sigma_HS+ = sigma_2 + f_1 / (
            1/(sigma_1 - sigma_2) + f_2 / (3 * sigma_2)
        )

    where sigma_1 <= sigma_2 and f_2 = 1 - f_1.

    Args:
        sigma_1: Conductivity of phase 1 (S/m).  Need not be the lower phase.
        sigma_2: Conductivity of phase 2 (S/m).
        f_1: Volume fraction of phase 1 (0 <= f_1 <= 1).

    Returns:
        Tuple (lower_bound, upper_bound) giving the HS lower and upper
        bounds for the effective conductivity (S/m).

    Raises:
        ValueError: If f_1 is outside [0, 1] or conductivities are negative.
    """
    if not (0.0 <= f_1 <= 1.0):
        raise ValueError(f"f_1 must be in [0, 1], got {f_1}")
    if sigma_1 < 0 or sigma_2 < 0:
        raise ValueError("Conductivity values must be non-negative.")

    f_2 = 1.0 - f_1

    # Ensure sigma_a is the smaller conductivity for the lower bound formula
    sigma_a = min(sigma_1, sigma_2)
    sigma_b = max(sigma_1, sigma_2)
    # Adjust volume fractions to match re-labelled phases
    if sigma_1 <= sigma_2:
        f_a, f_b = f_1, f_2
    else:
        f_a, f_b = f_2, f_1

    # Handle edge cases where phases have equal conductivity
    if sigma_a == sigma_b:
        return sigma_a, sigma_b

    # Lower bound: matrix is the less conductive phase
    lower = sigma_a + f_b / (
        1.0 / (sigma_b - sigma_a) + f_a / (3.0 * sigma_a)
    )

    # Upper bound: matrix is the more conductive phase
    upper = sigma_b + f_a / (
        1.0 / (sigma_a - sigma_b) + f_b / (3.0 * sigma_b)
    )

    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def composite_conductivity(composition_type: str, **params) -> float:
    """
    High-level function that dispatches to the appropriate conductivity model
    based on the composite type.

    Supported composition types and their required parameters:

    "alloy"
        Uses Bruggeman symmetric EMT - appropriate when both phases are
        conductive and are intermixed on a comparable length scale.
        Required params: sigma_1, sigma_2, f_1

    "dilute_composite"
        Uses Maxwell-Garnett EMT - appropriate for well-separated inclusions
        at low volume fraction (f < 0.3).
        Required params: sigma_host, sigma_inclusion, volume_fraction
        Optional params: depolarization (default 1/3)

    "concentrated_composite"
        Uses Hashin-Shtrikman upper bound as a practical estimate when
        inclusions are conductive and the composite is above the dilute
        regime.  The upper bound corresponds to inclusions forming the matrix
        topology (co-continuous structure).
        Required params: sigma_1, sigma_2, f_1

    "nanocomposite"
        Uses the McLachlan GEM equation - most accurate for nanofillers
        (CNTs, graphene, MXene) where percolation governs conductivity.
        Required params: f_filler, sigma_matrix, sigma_filler, f_c
        Optional params: t (default 2.0)

    Args:
        composition_type: One of "alloy", "dilute_composite",
            "concentrated_composite", or "nanocomposite".
        **params: Keyword arguments forwarded to the underlying model
            function.  See individual model docstrings for details.

    Returns:
        Effective electrical conductivity of the composite (S/m).

    Raises:
        ValueError: If composition_type is not recognised or required
            parameters are missing.

    Examples:
        >>> composite_conductivity("alloy", sigma_1=1e7, sigma_2=5e7, f_1=0.4)
        >>> composite_conductivity(
        ...     "nanocomposite",
        ...     f_filler=0.05,
        ...     sigma_matrix=1e-3,
        ...     sigma_filler=1e6,
        ...     f_c=0.02,
        ... )
    """
    if composition_type == "alloy":
        required = ("sigma_1", "sigma_2", "f_1")
        _check_required(required, params, composition_type)
        return bruggeman_emt(
            sigma_1=params["sigma_1"],
            sigma_2=params["sigma_2"],
            f_1=params["f_1"],
        )

    elif composition_type == "dilute_composite":
        required = ("sigma_host", "sigma_inclusion", "volume_fraction")
        _check_required(required, params, composition_type)
        return maxwell_garnett(
            sigma_host=params["sigma_host"],
            sigma_inclusion=params["sigma_inclusion"],
            volume_fraction=params["volume_fraction"],
            depolarization=params.get("depolarization", 1.0 / 3.0),
        )

    elif composition_type == "concentrated_composite":
        required = ("sigma_1", "sigma_2", "f_1")
        _check_required(required, params, composition_type)
        _lower, upper = hashin_shtrikman_bounds(
            sigma_1=params["sigma_1"],
            sigma_2=params["sigma_2"],
            f_1=params["f_1"],
        )
        return upper

    elif composition_type == "nanocomposite":
        required = ("f_filler", "sigma_matrix", "sigma_filler", "f_c")
        _check_required(required, params, composition_type)
        return mclachlan_gem(
            f_filler=params["f_filler"],
            sigma_matrix=params["sigma_matrix"],
            sigma_filler=params["sigma_filler"],
            f_c=params["f_c"],
            t=params.get("t", 2.0),
        )

    else:
        raise ValueError(
            f"Unknown composition_type '{composition_type}'. "
            "Choose from: 'alloy', 'dilute_composite', "
            "'concentrated_composite', 'nanocomposite'."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_required(required: tuple, params: dict, model_name: str) -> None:
    """Raise ValueError if any required key is missing from params."""
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(
            f"composite_conductivity('{model_name}') is missing required "
            f"parameters: {missing}"
        )
