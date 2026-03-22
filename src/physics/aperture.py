"""Aperture and enclosure shielding effectiveness calculations.

Models the degradation of shielding effectiveness caused by apertures
(holes, slots, seams, ventilation openings) in metallic enclosures.
Includes waveguide-below-cutoff attenuation for honeycomb vents and
rectangular cavity resonance mode computation.

References:
    Bethe, H.A. (1944). Theory of Diffraction by Small Holes.
        Physical Review, 66(7-8), 163-182.
    Pozar, D.M. (2011). Microwave Engineering, 4th ed., Section 4.3.
        Wiley.
    Celozzi, S., Araneo, R., Lovat, G. (2008). Electromagnetic Shielding.
        Wiley, Eq. 7.31.
    Paul, C.R. (2006). Introduction to Electromagnetic Compatibility,
        2nd ed. Wiley.
"""
import math
from typing import Dict, List

from src.utils.constants import C


# TE11 first root of the derivative of J1 (Bessel function)
_TE11_ROOT = 1.8412


# ---------------------------------------------------------------------------
# Circular aperture — Bethe hole theory
# ---------------------------------------------------------------------------

def aperture_se_circular(frequency_hz: float, radius_m: float) -> float:
    """Shielding effectiveness of a single circular aperture (Bethe theory).

    For electrically small holes (2a < lambda/2), the SE is:

        SE = 20 * log10(lambda / (2 * a))

    At resonance (2a >= lambda/2) the aperture becomes transparent and
    SE drops to 0 dB.

    Args:
        frequency_hz: Frequency in Hz (must be > 0).
        radius_m: Aperture radius in metres (must be > 0).

    Returns:
        Shielding effectiveness in dB.

    Raises:
        ValueError: If frequency_hz or radius_m is non-positive.

    Reference:
        Bethe (1944), Physical Review 66(7-8), 163.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    wavelength = C / frequency_hz
    diameter = 2.0 * radius_m

    # At or above resonance the hole is transparent
    if diameter >= wavelength / 2.0:
        return 0.0

    return 20.0 * math.log10(wavelength / diameter)


# ---------------------------------------------------------------------------
# Slot aperture — slot antenna model
# ---------------------------------------------------------------------------

def aperture_se_slot(frequency_hz: float, length_m: float) -> float:
    """Shielding effectiveness of a slot (seam) aperture.

    Models a rectangular slot/seam as a slot antenna.  For L < lambda/2:

        SE = 20 * log10(lambda / (2 * L))

    At half-wave resonance (L >= lambda/2) the slot radiates efficiently
    and SE = 0 dB.

    Args:
        frequency_hz: Frequency in Hz (must be > 0).
        length_m: Slot length in metres (must be > 0).

    Returns:
        Shielding effectiveness in dB.

    Raises:
        ValueError: If frequency_hz or length_m is non-positive.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if length_m <= 0:
        raise ValueError("length_m must be positive")

    wavelength = C / frequency_hz

    # Half-wave resonance
    if length_m >= wavelength / 2.0:
        return 0.0

    return 20.0 * math.log10(wavelength / (2.0 * length_m))


# ---------------------------------------------------------------------------
# Aperture array — N identical openings
# ---------------------------------------------------------------------------

def aperture_se_array(
    frequency_hz: float,
    se_single_db: float,
    n_apertures: int,
) -> float:
    """Shielding effectiveness of an array of N identical apertures.

    The total leakage power scales linearly with the number of apertures,
    reducing the overall SE by 10 * log10(N):

        SE_array = SE_single - 10 * log10(N)

    Args:
        frequency_hz: Frequency in Hz (must be > 0).
        se_single_db: SE of a single aperture in dB.
        n_apertures: Number of identical apertures (must be >= 1).

    Returns:
        Combined array SE in dB.

    Raises:
        ValueError: If n_apertures < 1 or frequency_hz <= 0.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if n_apertures < 1:
        raise ValueError("n_apertures must be >= 1")

    return se_single_db - 10.0 * math.log10(n_apertures)


# ---------------------------------------------------------------------------
# Waveguide below cutoff — honeycomb vent SE
# ---------------------------------------------------------------------------

def waveguide_below_cutoff_se(
    frequency_hz: float,
    tube_diameter_m: float,
    tube_length_m: float,
) -> float:
    """Shielding effectiveness of a circular waveguide below cutoff.

    A cylindrical tube (e.g. honeycomb vent cell) acts as a waveguide.
    Below the TE11 cutoff frequency the fields decay exponentially,
    providing high attenuation for long tubes.

    Cutoff frequency for the dominant TE11 mode in a circular waveguide:

        f_c = 1.8412 * c / (pi * d)

    Below cutoff, the evanescent attenuation constant is:

        alpha = sqrt(k_c^2 - k^2)   [Np/m]

    where k_c = 2 * 1.8412 / d is the cutoff wavenumber for TE11 and
    k = 2 * pi * f / c is the free-space wavenumber.  The SE is:

        SE = 20 * log10(e) * alpha * L  =  8.686 * alpha * L   [dB]

    Above cutoff the tube is a propagating waveguide and SE ~ 0 dB.

    Args:
        frequency_hz: Frequency in Hz (must be > 0).
        tube_diameter_m: Inner diameter of the tube in metres (must be > 0).
        tube_length_m: Length of the tube in metres (must be > 0).

    Returns:
        Shielding effectiveness in dB.

    Raises:
        ValueError: If any parameter is non-positive.

    Reference:
        Pozar (2011), Microwave Engineering, Section 4.3.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if tube_diameter_m <= 0:
        raise ValueError("tube_diameter_m must be positive")
    if tube_length_m <= 0:
        raise ValueError("tube_length_m must be positive")

    # Cutoff wavenumber for TE11 mode: k_c = p'_11 / a = 1.8412 / radius
    k_c = 2.0 * _TE11_ROOT / tube_diameter_m

    # Free-space wavenumber
    k = 2.0 * math.pi * frequency_hz / C

    # Above cutoff: propagating mode, no attenuation
    if k >= k_c:
        return 0.0

    # Below cutoff: evanescent attenuation constant (Np/m)
    alpha = math.sqrt(k_c ** 2 - k ** 2)

    # Convert Np to dB: 1 Np = 20*log10(e) ~ 8.6859 dB
    se_db = 8.685889638 * alpha * tube_length_m
    return se_db


# ---------------------------------------------------------------------------
# Rectangular cavity resonance frequencies
# ---------------------------------------------------------------------------

def cavity_resonance_frequencies(
    length_m: float,
    width_m: float,
    height_m: float,
    max_modes: int = 5,
    max_results: int = 20,
) -> List[Dict]:
    """Compute resonance frequencies of a rectangular cavity.

    For a lossless rectangular cavity of dimensions a x b x c, the
    resonant frequencies of the TE_mnp and TM_mnp modes are:

        f_mnp = (c / 2) * sqrt((m/a)^2 + (n/b)^2 + (p/c_dim)^2)

    Constraints:
        - At least two of (m, n, p) must be nonzero.
        - For TE modes: one index may be zero (the one aligned with the
          E-field direction).
        - For TM modes: m and n must both be nonzero; p may be zero.

    Args:
        length_m: Cavity length *a* in metres (must be > 0).
        width_m: Cavity width *b* in metres (must be > 0).
        height_m: Cavity height *c* in metres (must be > 0).
        max_modes: Maximum mode index to search (default 5).
        max_results: Maximum number of modes to return (default 20).

    Returns:
        Sorted list of dicts, each with keys:
            m, n, p (int), frequency_hz (float), mode_type (str).

    Raises:
        ValueError: If any dimension is non-positive.
    """
    if length_m <= 0 or width_m <= 0 or height_m <= 0:
        raise ValueError("All cavity dimensions must be positive")

    a, b, c_dim = length_m, width_m, height_m
    modes: List[Dict] = []

    for m in range(0, max_modes + 1):
        for n in range(0, max_modes + 1):
            for p in range(0, max_modes + 1):
                # At least two indices must be nonzero
                nonzero_count = sum(1 for idx in (m, n, p) if idx > 0)
                if nonzero_count < 2:
                    continue

                freq = (C / 2.0) * math.sqrt(
                    (m / a) ** 2 + (n / b) ** 2 + (p / c_dim) ** 2
                )

                # Determine mode type:
                # TM modes: m != 0 and n != 0 (p may be 0)
                # TE modes: at least one of m or n may be 0, but p must be nonzero
                #           (or one transverse index is 0)
                # Both TE and TM can exist for the same (m,n,p) when all are nonzero
                if m > 0 and n > 0:
                    # TM mode exists (p can be 0 or nonzero)
                    modes.append({
                        "m": m,
                        "n": n,
                        "p": p,
                        "frequency_hz": freq,
                        "mode_type": f"TM{m}{n}{p}",
                    })
                    # If p > 0, TE mode also exists for same indices
                    if p > 0:
                        modes.append({
                            "m": m,
                            "n": n,
                            "p": p,
                            "frequency_hz": freq,
                            "mode_type": f"TE{m}{n}{p}",
                        })
                else:
                    # One of m or n is zero -> only TE mode possible
                    # (requires p > 0, which is guaranteed since we need
                    #  at least 2 nonzero indices and one of m,n is 0)
                    modes.append({
                        "m": m,
                        "n": n,
                        "p": p,
                        "frequency_hz": freq,
                        "mode_type": f"TE{m}{n}{p}",
                    })

    # Sort by frequency ascending, then by mode name for tie-breaking
    modes.sort(key=lambda mode: (mode["frequency_hz"], mode["mode_type"]))

    return modes[:max_results]


# ---------------------------------------------------------------------------
# Combined enclosure SE — power-based combination
# ---------------------------------------------------------------------------

def combined_enclosure_se(
    bulk_se_db: float,
    aperture_se_list_db: List[float],
) -> float:
    """Power-combined shielding effectiveness of an enclosure.

    Treats each leakage path (bulk shield + individual apertures) as
    independent power transmission channels and combines them:

        SE_total = -10 * log10( sum_i( 10^(-SE_i / 10) ) )

    The bulk SE and each aperture SE are separate paths in the sum.

    Args:
        bulk_se_db: Shielding effectiveness of the bulk material in dB.
        aperture_se_list_db: List of individual aperture SE values in dB.
            May be empty (only bulk path).

    Returns:
        Combined enclosure SE in dB.

    Reference:
        Celozzi et al. (2008), Electromagnetic Shielding, Eq. 7.31.
    """
    # Collect all leakage paths
    all_se_values = [bulk_se_db] + list(aperture_se_list_db)

    # Sum transmitted power fractions
    total_power = sum(10.0 ** (-se / 10.0) for se in all_se_values)

    return -10.0 * math.log10(total_power)
