"""Temperature and frequency dependent electromagnetic material property models.

References:
    Snoek (1948). Physica, 14(4), 207-217. DOI: 10.1016/0031-8914(48)90038-X
    Matula (1979). J. Phys. Chem. Ref. Data, 8(4), 1147-1298.
"""
import numpy as np
from typing import Dict, Optional, Tuple

# Permeability of free space (H/m) — needed for SI Snoek product
_MU_0 = 4.0 * np.pi * 1e-7

# Temperature coefficient of resistivity (TCR) data at 293K
TCR_DATA = {
    'Cu':  {'sigma_ref': 5.96e7, 'alpha': 0.00393, 'T_ref': 293.15},
    'Al':  {'sigma_ref': 3.77e7, 'alpha': 0.00390, 'T_ref': 293.15},
    'Ni':  {'sigma_ref': 1.44e7, 'alpha': 0.00690, 'T_ref': 293.15},
    'Fe':  {'sigma_ref': 1.04e7, 'alpha': 0.00651, 'T_ref': 293.15},
    'Ag':  {'sigma_ref': 6.30e7, 'alpha': 0.00380, 'T_ref': 293.15},
    'Au':  {'sigma_ref': 4.10e7, 'alpha': 0.00340, 'T_ref': 293.15},
    'steel_1018': {'sigma_ref': 6.99e6, 'alpha': 0.00600, 'T_ref': 293.15},
    'ss_304':     {'sigma_ref': 1.45e6, 'alpha': 0.00094, 'T_ref': 293.15},
    'mu_metal':   {'sigma_ref': 1.82e6, 'alpha': 0.00200, 'T_ref': 293.15},
    'permalloy':  {'sigma_ref': 2.00e6, 'alpha': 0.00200, 'T_ref': 293.15},
}

# Curie temperature and Snoek's limit data
MAGNETIC_DATA = {
    'Fe':       {'T_curie': 1043, 'mu_r_ref': 5000,   'M_s': 1714e3},  # A/m
    'Ni':       {'T_curie': 627,  'mu_r_ref': 300,     'M_s': 510e3},
    'Co':       {'T_curie': 1388, 'mu_r_ref': 250,     'M_s': 1422e3},
    'mu_metal': {'T_curie': 673,  'mu_r_ref': 100000,  'M_s': 860e3},
    'permalloy':{'T_curie': 733,  'mu_r_ref': 50000,   'M_s': 860e3},
    'steel_1018':{'T_curie': 1043,'mu_r_ref': 2000,    'M_s': 1714e3},
}

# Gyromagnetic ratio for free electron (Hz/T) used in Snoek's law
_GAMMA_GYRO = 2.8e10  # Hz/T


def temp_dependent_conductivity(
    sigma_ref: float,
    temperature: float,
    T_ref: float = 293.15,
    alpha: float = 0.00393,
) -> float:
    """Compute electrical conductivity using a linear TCR model.

    The resistivity increases linearly with temperature, so conductivity
    decreases as:

        sigma(T) = sigma_ref / (1 + alpha * (T - T_ref))

    Args:
        sigma_ref: Reference conductivity at T_ref (S/m).
        temperature: Target temperature (K).
        T_ref: Reference temperature (K). Default 293.15 K (20 C).
        alpha: Temperature coefficient of resistivity (1/K). Default is Cu.

    Returns:
        Conductivity at the requested temperature (S/m).

    Raises:
        ValueError: If the denominator would be zero or negative, indicating
            the temperature is below the material's valid range.
    """
    denominator = 1.0 + alpha * (temperature - T_ref)
    if denominator <= 0.0:
        raise ValueError(
            f"Temperature {temperature} K produces a non-physical denominator "
            f"({denominator:.4f}) for TCR model with alpha={alpha}. "
            "Check that the temperature is within the valid range for this material."
        )
    return sigma_ref / denominator


def temp_dependent_conductivity_auto(element_symbol: str, temperature: float) -> float:
    """Compute conductivity at a given temperature using TCR_DATA lookup.

    Wraps :func:`temp_dependent_conductivity` with automatic parameter
    retrieval from the :data:`TCR_DATA` dictionary.

    Args:
        element_symbol: Key into TCR_DATA (e.g. ``'Cu'``, ``'mu_metal'``).
        temperature: Target temperature (K).

    Returns:
        Conductivity at the requested temperature (S/m).

    Raises:
        KeyError: If element_symbol is not present in TCR_DATA.
    """
    if element_symbol not in TCR_DATA:
        raise KeyError(
            f"'{element_symbol}' not found in TCR_DATA. "
            f"Available materials: {list(TCR_DATA.keys())}"
        )
    entry = TCR_DATA[element_symbol]
    return temp_dependent_conductivity(
        sigma_ref=entry['sigma_ref'],
        temperature=temperature,
        T_ref=entry['T_ref'],
        alpha=entry['alpha'],
    )


def temp_dependent_permeability(
    mu_r_ref: float,
    temperature: float,
    T_curie: float,
    T_ref: float = 293.15,
    n: float = 1.5,
) -> float:
    """Compute relative permeability as a function of temperature.

    Uses a power-law scaling relative to the Curie temperature:

        mu_r(T) = 1 + (mu_r_ref - 1) * [ (1 - (T/T_c)^2) /
                                           (1 - (T_ref/T_c)^2) ]^n   for T < T_c

        mu_r(T) = 1.0                                                  for T >= T_c

    The material becomes paramagnetic (mu_r -> 1) at and above the Curie point.

    Args:
        mu_r_ref: Static relative permeability at T_ref (dimensionless).
        temperature: Target temperature (K).
        T_curie: Curie temperature of the material (K).
        T_ref: Reference temperature (K). Default 293.15 K.
        n: Exponent controlling the shape of the falloff curve. Default 1.5.

    Returns:
        Relative permeability at the requested temperature (dimensionless, >= 1.0).

    Raises:
        ValueError: If T_ref >= T_curie, which makes the normalisation factor
            undefined.
    """
    if T_ref >= T_curie:
        raise ValueError(
            f"T_ref ({T_ref} K) must be less than T_curie ({T_curie} K)."
        )

    if temperature >= T_curie:
        return 1.0

    numerator = 1.0 - (temperature / T_curie) ** 2
    denominator = 1.0 - (T_ref / T_curie) ** 2

    # Clamp to zero to avoid tiny negative float values from rounding
    scaled = max(numerator / denominator, 0.0)
    return 1.0 + (mu_r_ref - 1.0) * (scaled ** n)


def snoek_resonance_frequency(mu_static: float, M_s: float) -> float:
    """Compute the Snoek resonance frequency from Snoek's law.

    Snoek's law relates the static permeability and the saturation
    magnetisation to the upper frequency limit of magnetic response:

        (mu_s - 1) * f_r = (2/3) * gamma_gyro * M_s

    Rearranged for f_r:

        f_r = (2/3) * gamma_gyro * M_s / (mu_s - 1)

    where gamma_gyro = 2.8e10 Hz/T (gyromagnetic ratio for free electrons,
    expressed in Hz per Tesla) and M_s is saturation magnetisation in A/m.

    Note: for high-permeability materials such as mu-metal (mu_r ~ 1e5,
    M_s ~ 860 kA/m) this formula yields f_r in the tens-to-hundreds-of-GHz
    range.  For typical ferrites (mu_r ~ 10, M_s ~ 300 kA/m) f_r falls in
    the GHz range.  The high f_r for mu-metal means the Debye roll-off only
    becomes significant well above 100 GHz for that material.

    Args:
        mu_static: Static (low-frequency) relative permeability.
        M_s: Saturation magnetisation (A/m).

    Returns:
        Snoek resonance frequency (Hz).

    Raises:
        ValueError: If mu_static <= 1 (no magnetic response to model).
    """
    if mu_static <= 1.0:
        raise ValueError(
            f"mu_static must be > 1 for a magnetic material; got {mu_static}."
        )
    # M_s is in A/m; gamma_gyro (Hz/T) expects Tesla, so multiply by _MU_0
    return (2.0 / 3.0) * _GAMMA_GYRO * _MU_0 * M_s / (mu_static - 1.0)


def freq_dependent_permeability(
    mu_static: float,
    frequency: float,
    f_resonance: Optional[float] = None,
    M_s: Optional[float] = None,
) -> complex:
    """Compute complex relative permeability using a Debye relaxation model.

    The single-pole Debye model gives:

        mu(f) = 1 + (mu_s - 1) / (1 + j * f / f_r)

    where f_r is determined either from the caller-supplied ``f_resonance``
    or calculated via Snoek's law when ``M_s`` is provided instead.

    At least one of ``f_resonance`` or ``M_s`` must be supplied.

    Args:
        mu_static: Static (DC) relative permeability.
        frequency: Operating frequency (Hz).
        f_resonance: Resonance (relaxation) frequency (Hz). If ``None``,
            it is calculated from Snoek's law using ``M_s``.
        M_s: Saturation magnetisation (A/m). Used to derive f_resonance
            via Snoek's law when ``f_resonance`` is ``None``.

    Returns:
        Complex relative permeability mu_r(f). The real part represents
        the dispersive response; the imaginary part represents loss.

    Raises:
        ValueError: If neither ``f_resonance`` nor ``M_s`` is provided,
            or if mu_static <= 1 when Snoek's law is needed.
    """
    if f_resonance is None:
        if M_s is None:
            raise ValueError(
                "Either f_resonance or M_s must be supplied to determine "
                "the relaxation frequency."
            )
        f_resonance = snoek_resonance_frequency(mu_static, M_s)

    susceptibility = mu_static - 1.0
    return 1.0 + susceptibility / (1.0 + 1j * frequency / f_resonance)


def get_material_properties_at_conditions(
    element: str,
    frequency: float,
    temperature: float = 293.15,
) -> Dict[str, object]:
    """Return conductivity and complex permeability at specified conditions.

    Combines the temperature-dependent conductivity, temperature-dependent
    static permeability, and frequency-dependent permeability models into a
    single convenience function.

    For materials that appear in both TCR_DATA and MAGNETIC_DATA, both
    conductivity and complex permeability are returned.  For purely
    conductive materials (not in MAGNETIC_DATA) the relative permeability
    is 1.0 (non-magnetic).

    Args:
        element: Material key (e.g. ``'Cu'``, ``'mu_metal'``, ``'Fe'``).
            Must exist in at least TCR_DATA.
        frequency: Operating frequency (Hz).
        temperature: Operating temperature (K). Default 293.15 K (20 C).

    Returns:
        Dictionary with keys:

        - ``'conductivity'`` (float): Conductivity at temperature (S/m).
        - ``'mu_r_static'`` (float): Temperature-corrected static mu_r.
        - ``'mu_r_complex'`` (complex): Frequency- and temperature-corrected
          mu_r from the Debye/Snoek model. Equal to the static value cast to
          complex for non-magnetic materials.
        - ``'f_resonance'`` (float or None): Snoek resonance frequency (Hz),
          or ``None`` for non-magnetic materials.
        - ``'temperature'`` (float): Temperature used (K).
        - ``'frequency'`` (float): Frequency used (Hz).

    Raises:
        KeyError: If ``element`` is not found in TCR_DATA.
    """
    # --- Conductivity ---
    conductivity = temp_dependent_conductivity_auto(element, temperature)

    # --- Permeability ---
    if element in MAGNETIC_DATA:
        mag = MAGNETIC_DATA[element]

        mu_r_static = temp_dependent_permeability(
            mu_r_ref=mag['mu_r_ref'],
            temperature=temperature,
            T_curie=mag['T_curie'],
        )

        if mu_r_static <= 1.0:
            # Above or very close to Curie temperature: no magnetic response
            mu_r_complex: complex = complex(1.0, 0.0)
            f_res: Optional[float] = None
        else:
            f_res = snoek_resonance_frequency(mu_r_static, mag['M_s'])
            mu_r_complex = freq_dependent_permeability(
                mu_static=mu_r_static,
                frequency=frequency,
                f_resonance=f_res,
            )
    else:
        mu_r_static = 1.0
        mu_r_complex = complex(1.0, 0.0)
        f_res = None

    return {
        'conductivity': conductivity,
        'mu_r_static': mu_r_static,
        'mu_r_complex': mu_r_complex,
        'f_resonance': f_res,
        'temperature': temperature,
        'frequency': frequency,
    }
