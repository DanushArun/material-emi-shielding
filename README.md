# EMI Shield Designer

A **physics-based** electromagnetic interference (EMI) shielding calculator with an interactive web interface.

## What It Does

This application calculates EMI shielding effectiveness for various material compositions using **electromagnetic theory** (Maxwell's equations). It helps engineers:

- Calculate shielding effectiveness (SE) in dB
- Analyze frequency-dependent performance (1 MHz - 10 GHz)
- Optimize material thickness
- Study grain size effects on conductivity
- Visualize reflection, absorption, and multiple reflection losses

## What It Does NOT Do

**There is NO machine learning in this application** despite the ML folder structure. The predictions are purely physics-based calculations.

## Features

### ✅ Working Features

1. **EMI Calculations** - Core electromagnetic shielding physics
   - Skin depth calculation
   - Intrinsic impedance
   - Reflection loss
   - Absorption loss
   - Multiple reflection corrections
   - Grain boundary scattering effects

2. **Material Composition** - Two input modes:
   - **Molecular Builder**: Build materials from molecules
   - **Direct Composition**: Enter elemental percentages directly

3. **Analysis Tools**:
   - Frequency sweep (1 MHz to 10 GHz)
   - Thickness optimization
   - Grain size analysis
   - Multi-parameter visualization

4. **Material Database** - Common EMI shielding materials:
   - Copper, Aluminum, Steel
   - Nickel, Silver, Mu-Metal
   - Various alloys

5. **Interactive UI** - Built with Streamlit:
   - Real-time calculations
   - Interactive charts (Plotly)
   - Calculation history
   - Export to CSV

### ❌ Not Implemented (Despite File Structure)

- Machine learning predictions
- AI-based material recommendations
- Quantum materials discovery
- FEM multiphysics simulations
- Mechanical-EMI coupling
- Reliability analysis over time

## Technology Stack

- **Physics Engine**: NumPy/SciPy
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Language**: Python 3.9+

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd EMI-shielding

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Requirements

```
numpy>=1.26.0
pandas>=2.0.3
scipy>=1.11.0
streamlit>=1.28.2
plotly>=5.18.0
scikit-learn>=1.3.0
```

## Project Structure

```
EMI-shielding/
├── app.py                      # Main Streamlit application (2,129 lines)
├── requirements.txt            # Python dependencies
├── auth.py                    # Authentication module
│
├── src/
│   ├── physics/
│   │   ├── emi_calculations.py          # Core EMI physics (1,308 lines)
│   │   └── advanced_microstructure.py   # Microstructure modeling
│   │
│   ├── materials/
│   │   └── material_properties.py       # Material database
│   │
│   ├── utils/
│   │   └── constants.py                  # Physical constants
│   │
│   └── ml/                              # Structure exists, not used
│       ├── features/                    # Feature engineering (ready but unused)
│       └── accuracy/                    # Validation (ready but unused)
│
└── data_collection/                     # Separate data scraping module
```

## Usage

### Basic Workflow

1. **Choose Input Mode**:
   - Molecular Builder: Select molecules and quantities
   - Direct Composition: Enter element percentages

2. **Set Parameters**:
   - Frequency (MHz)
   - Thickness (mm)
   - Grain size (μm) - optional

3. **Calculate**:
   - View shielding effectiveness
   - Analyze frequency response
   - Optimize thickness
   - Export results

### Example Calculation

For a **1mm copper shield** at **1 GHz**:
- Total SE: ~100 dB
- Reflection Loss: ~90 dB
- Absorption Loss: ~10 dB
- Skin Depth: ~2 μm

## Physics Background

The calculator implements standard EMI shielding theory:

**Total SE = Reflection Loss + Absorption Loss + Multiple Reflection Correction**

- **Reflection Loss**: Impedance mismatch at air-material interface
- **Absorption Loss**: Exponential decay through material thickness
- **Skin Depth**: δ = √(2 / ωμσ)

Includes advanced features:
- Grain boundary scattering (Mayadas-Shatzkes model)
- Frequency-dependent properties
- Near-field corrections

## Limitations

1. **Assumes plane wave incidence** (far-field)
2. **Homogeneous materials** (no layered structures)
3. **No geometric effects** (assumes infinite sheet)
4. **Linear material properties**
5. **No temperature effects**

## Development Status

**Current Phase**: Cleanup and stabilization

### Recently Completed
- ✅ Removed unused AI/ML code (2,428 lines)
- ✅ Removed experimental physics modules (2,052 lines)
- ✅ Fixed dependency management
- ✅ Cleaned up imports

### Next Steps
- Extract CSS from monolithic app.py
- Modularize UI components
- Add comprehensive tests
- Improve documentation
- Actually implement ML (if needed)

## Known Issues

1. **app.py is too large** (2,129 lines) - needs refactoring
2. **No test coverage**
3. **ML folder structure misleading** - no ML actually runs
4. **Import patterns inconsistent** - mix of relative and absolute

## Contributing

Before adding features:
1. **Write tests first**
2. **Keep modules focused** (Single Responsibility)
3. **Don't claim features that don't exist**
4. **Document what actually works**

## License

[Specify your license]

## Citation

If you use this calculator in research:

```bibtex
@software{emi_shield_designer,
  title = {EMI Shield Designer},
  author = {[Your Name]},
  year = {2024},
  note = {Physics-based EMI shielding calculator}
}
```

## Contact

[Your contact information]

---

**Note**: This README reflects the **actual current state** of the codebase after cleanup. Previous documentation was deleted as it described features that were not implemented.
