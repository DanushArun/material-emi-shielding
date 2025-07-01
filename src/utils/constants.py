"""
Physical constants and unit conversions for EMI shielding calculations.
"""

import numpy as np

# Fundamental physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
Z_0 = 376.730313668  # Impedance of free space (Ω)
C = 299792458  # Speed of light (m/s)
K_B = 1.380649e-23  # Boltzmann constant (J/K)
E = 1.602176634e-19  # Elementary charge (C)

# Frequency ranges
FREQ_MIN = 1e6  # 1 MHz
FREQ_MAX = 10e9  # 10 GHz

# Material categories for correction factors
GOOD_CONDUCTOR_THRESHOLD = 1e6  # S/m
MODERATE_CONDUCTOR_MIN = 1e2  # S/m
MODERATE_CONDUCTOR_MAX = 1e6  # S/m
POOR_CONDUCTOR_THRESHOLD = 1e2  # S/m

# Correction factors for hybrid model
GOOD_CONDUCTOR_CORRECTION = 0.05  # 5%
MODERATE_CONDUCTOR_CORRECTION = 0.15  # 15%
POOR_CONDUCTOR_CORRECTION = 0.25  # 25%

# Unit conversion factors
def hz_to_mhz(freq_hz):
    """Convert frequency from Hz to MHz."""
    return freq_hz / 1e6

def hz_to_ghz(freq_hz):
    """Convert frequency from Hz to GHz."""
    return freq_hz / 1e9

def mhz_to_hz(freq_mhz):
    """Convert frequency from MHz to Hz."""
    return freq_mhz * 1e6

def ghz_to_hz(freq_ghz):
    """Convert frequency from GHz to Hz."""
    return freq_ghz * 1e9

def db_to_linear(db_value):
    """Convert dB to linear scale."""
    return 10 ** (db_value / 20)

def linear_to_db(linear_value):
    """Convert linear scale to dB."""
    return 20 * np.log10(linear_value)

def mm_to_m(mm_value):
    """Convert millimeters to meters."""
    return mm_value / 1000

def m_to_mm(m_value):
    """Convert meters to millimeters."""
    return m_value * 1000

def celsius_to_kelvin(celsius):
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

# Complex number utilities
def complex_sqrt(z):
    """Calculate square root of complex number."""
    return np.sqrt(z)

def complex_abs(z):
    """Calculate absolute value of complex number."""
    return np.abs(z)

def complex_phase(z):
    """Calculate phase angle of complex number in radians."""
    return np.angle(z)

# Material property validators
def validate_conductivity(sigma):
    """Validate electrical conductivity value."""
    if sigma < 0:
        raise ValueError("Conductivity must be non-negative")
    return sigma

def validate_permeability(mu_r):
    """Validate relative permeability value."""
    if mu_r < 0.999:  # Allow slight diamagnetic materials
        raise ValueError("Relative permeability must be >= 0.999")
    return mu_r

def validate_permittivity(epsilon_r):
    """Validate relative permittivity value."""
    if epsilon_r < 1:
        raise ValueError("Relative permittivity must be >= 1")
    return epsilon_r

def validate_frequency(freq):
    """Validate frequency value."""
    if freq <= 0:
        raise ValueError("Frequency must be positive")
    return freq

def validate_thickness(thickness):
    """Validate material thickness value."""
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    return thickness

# Material classification
def classify_conductor(sigma):
    """Classify material based on conductivity."""
    if sigma >= GOOD_CONDUCTOR_THRESHOLD:
        return "good"
    elif MODERATE_CONDUCTOR_MIN <= sigma < MODERATE_CONDUCTOR_MAX:
        return "moderate"
    else:
        return "poor"

def get_correction_factor(sigma):
    """Get ML correction factor based on conductivity."""
    classification = classify_conductor(sigma)
    if classification == "good":
        return GOOD_CONDUCTOR_CORRECTION
    elif classification == "moderate":
        return MODERATE_CONDUCTOR_CORRECTION
    else:
        return POOR_CONDUCTOR_CORRECTION