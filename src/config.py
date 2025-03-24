"""
Configuration settings for the EMI Shielding Prediction System.
"""

import os
from pathlib import Path

# Project structure
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Frequency range settings (Hz)
DEFAULT_FREQ_MIN = 1e6  # 1 MHz
DEFAULT_FREQ_MAX = 1e9  # 1 GHz
DEFAULT_FREQ_POINTS = 100

# Physical constants
EPSILON_0 = 8.854e-12  # Vacuum permittivity (F/m)
MU_0 = 4 * 3.14159e-7  # Vacuum permeability (H/m)
C = 299792458  # Speed of light (m/s)

# Model settings
ML_MODEL_VERSION = "0.1.0"
PHYSICS_MODEL_VERSION = "0.1.0"

# API settings
MATERIALS_PROJECT_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY", "")
