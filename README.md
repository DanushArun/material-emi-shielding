<<<<<<< HEAD
# EMI Shielding Prediction System

A sophisticated web application for predicting electromagnetic interference (EMI) shielding effectiveness of material combinations. This system combines physics-based calculations with machine learning to provide accurate predictions for any combination of elements from the periodic table.

## Features

- Dynamic element selection from the periodic table with adjustable percentages
- Physics-based EMI shielding calculations with real-time visualization
- Application-specific material recommendations for different interference types
- Material comparison tools for optimizing shielding designs
- Advanced SUPER MODE with:
  - Direct material property manipulation
  - Environmental effects simulation (temperature, humidity, aging)
  - Multi-layer shielding configuration
  - Advanced visualization options
  - Machine learning model customization
- Export results to CSV format for further analysis
- Comprehensive technical reference and documentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emi-shielding.git
cd emi-shielding
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to:
   - Select materials from the periodic table
   - Set composition ratios
   - Adjust material thickness
   - Configure frequency range

4. View results:
   - Material properties table
   - Interactive shielding effectiveness plots
   - Detailed numerical results at specific frequencies
   - Export data to CSV

## Project Structure

```
emi-shielding/
├── src/
│   ├── physics/
│   │   └── emi_calculations.py    # EMI shielding physics calculations
│   ├── materials/
│   │   └── material_properties.py # Material property handling
│   ├── ml/
│   │   └── emi_model.py          # Machine learning models
│   └── config.py                 # Configuration settings
├── data/                         # Data storage
├── models/                       # Trained model storage
├── results/                      # Results output
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Technical Details

### Physics-Based Calculations

The system implements fundamental electromagnetic theory including:
- Reflection loss calculation
- Absorption loss calculation
- Multiple reflection loss calculation
- Skin depth effects
- Frequency-dependent material properties

### Machine Learning Model

Uses a physics-informed neural network (PINN) that:
- Incorporates physical constraints in the architecture
- Ensures predictions follow electromagnetic theory
- Provides uncertainty quantification
- Combines data-driven learning with physical principles

### Material Properties

Includes comprehensive material data:
- Electrical conductivity
- Magnetic permeability
- Dielectric properties
- Density and other physical properties
- Frequency-dependent behavior

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# material-emi-shielding
>>>>>>> 96b15ed7abb465c3b9f5f5265f7905cbfb30a0ac
