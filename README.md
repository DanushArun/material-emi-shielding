# Chemical EMI Designer

A Streamlit exploration tool for constructing chemical formulas and estimating electromagnetic
shielding from material-property assumptions, frequency and thickness.

## Workflow

1. Select elements and quantities or choose a molecular preset.
2. Combine formulas into a proposed material mixture.
3. Calculate material properties and shielding components.
4. Inspect reflection, absorption, multiple-reflection and total shielding outputs.

The calculations are a model of supplied properties. Building a formula in the UI does not
establish that a chemical reaction occurs or that a synthesized material has those properties.

## Run locally

```bash
git clone https://github.com/DanushArun/material-emi-shielding.git
cd material-emi-shielding
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## Computational structure

| Source | Responsibility |
| --- | --- |
| [app.py](streamlit_app/app.py) | Formula builder and results interface |
| [molecular_presets.py](streamlit_app/molecular_presets.py) | Molecule/reaction presets |
| [material_properties.py](src/materials/material_properties.py) | Formula/property handling |
| [emi_calculations.py](src/physics/emi_calculations.py) | Shielding and frequency calculations |
| [constants.py](src/utils/constants.py) | Constants and input validation |

The physics module computes skin depth and complex impedance/propagation, combines shielding
terms, and includes frequency sweeps and a thickness search. The multiple-reflection correction
is suppressed when modeled absorption exceeds 15 dB.

## Evidence and limits

Tracked source and dependency/setup paths were reviewed. No laboratory shielding measurement,
material synthesis, numerical benchmark or UI acceptance run was performed for this README.
There is no committed automated test suite.

Weighted material-property estimates and simplified electromagnetic assumptions need independent
validation for the material, frequency range and geometry of interest. Displayed ratings are
software outputs, not a material certification or experimental shielding result.
