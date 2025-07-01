# Chemical EMI Designer

A Chemical Reaction EMI Shield Designer that allows users to build molecular compounds and analyze their electromagnetic interference (EMI) shielding effectiveness.

## Features

- **Molecular Builder**: Select elements with adjustable quantities to form molecules (e.g., Fe₂O₃, C₁₄H₇Mo₄)
- **Chemical Reactions**: Combine multiple molecules to create chemical reactions
- **EMI Shielding Analysis**: Click "⚛️ REACT" to perform step-by-step EMI shielding calculations
- **Real-time Results**: View calculations and final results with performance ratings
- **Molecular Presets**: 25 pre-configured molecules across 6 categories
- **Reaction Presets**: 5 common reaction combinations for quick testing

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd EMI-shielding
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run streamlit_app/app.py
```

2. Open your browser to the displayed URL (typically http://localhost:8501)

3. Build molecules:
   - Select elements from the periodic table
   - Adjust quantities to form compounds
   - Add molecules to create reactions

4. Analyze shielding:
   - Click "⚛️ REACT" to calculate EMI effectiveness
   - View step-by-step calculations
   - Check final performance rating

## Project Structure

```
EMI-shielding/
├── streamlit_app/
│   ├── app.py                    # Main application
│   └── molecular_presets.py      # Pre-configured molecules and reactions
├── src/
│   ├── materials/
│   │   ├── material_properties.py   # Material property calculations
│   │   └── periodic_table.json      # Element data
│   ├── physics/
│   │   ├── emi_calculations.py      # EMI shielding physics
│   │   └── shielding_theory.py      # Theoretical calculations
│   └── utils/
│       └── constants.py             # Physical constants
└── requirements.txt
```

## Technical Details

### Chemical Formula Parsing
- Supports complex chemical formulas with subscripts
- Validates molecular compositions
- Calculates molecular weights and properties

### EMI Shielding Physics
- Reflection loss calculations
- Absorption loss analysis
- Multiple reflection effects
- Frequency-dependent material properties

### Material Properties
- Weighted average calculations for mixtures
- Conductivity, permeability, and permittivity
- Safety validation for diamagnetic materials

## Dependencies

- **streamlit**: Web interface framework
- **numpy**: Numerical calculations
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations

## License

This project is for educational and research purposes.