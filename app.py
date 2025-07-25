"""
🔬 EMI Shield Designer - Unified App with Mode Selection
Allows users to choose between Molecular Builder or Direct Composition modes
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import re
from typing import Dict
import time
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Also add parent directory in case we're in a subdirectory
parent_dir = os.path.dirname(current_dir)
if os.path.exists(os.path.join(parent_dir, 'src')):
    sys.path.insert(0, parent_dir)

# Import modules
from src.physics.emi_calculations import emi_calculator
from src.materials.material_properties import material_db

# Import auth with proper error handling
try:
    from auth import check_password
except ImportError:
    # Try importing from current directory
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("auth", os.path.join(current_dir, "auth.py"))
        auth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auth)
        check_password = auth.check_password
    except Exception as e:
        import streamlit as st
        st.error(f"Could not import auth module: {e}")
        st.error(f"Current directory: {current_dir}")
        st.error(f"Files in current directory: {os.listdir(current_dir)}")
        st.stop()


# Page configuration
st.set_page_config(
    page_title="EMI Shield Designer",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function for formatting numbers
def format_number(num):
    """Format number in scientific notation for display."""
    if num >= 1e6:
        return f"{num:.2e}"
    elif num >= 1000:
        return f"{num:.0f}"
    elif num >= 1:
        return f"{num:.2f}"
    else:
        return f"{num:.2e}"

# History file path
HISTORY_FILE = "calculation_history.json"

# Function to load history from file
def load_history():
    """Load calculation history from file."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load history: {e}")
    return []

# Function to save history to file
def save_history(history):
    """Save calculation history to file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Could not save history: {e}")

# Helper function to save calculation to history
def save_to_history(mode, composition, thickness, frequency, result, molecules=None, grain_size=None):
    """Save a calculation to the history."""
    # Create composition summary
    if isinstance(composition, dict):
        sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        comp_summary = ", ".join([f"{elem}: {pct:.1f}%" for elem, pct in sorted_comp[:3]])
        if len(sorted_comp) > 3:
            comp_summary += "..."
    else:
        comp_summary = "Unknown composition"
    
    # Prepare history entry
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'mode': mode,
        'composition': composition.copy() if isinstance(composition, dict) else {},
        'composition_summary': comp_summary,
        'thickness': thickness,
        'frequency': frequency,
        'grain_size': grain_size * 1e6 if grain_size else 10.0,  # Convert to micrometers for display
        'total_se': result['total_se'],
        'reflection_loss': result['reflection_loss'],
        'absorption_loss': result['absorption_loss'],
        'molecules': molecules if molecules else []
    }
    
    # Load existing history
    history = load_history()
    
    # Add new entry to beginning
    history.insert(0, history_entry)
    
    # Limit to 100 entries
    if len(history) > 100:
        history = history[:100]
    
    # Save to file
    save_history(history)

# Apply the same dark theme CSS from main app
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Dark theme root variables - 8px grid system */
    :root {
        /* Spacing scale (8px grid) */
        --space-1: 8px;
        --space-2: 16px;
        --space-3: 24px;
        --space-4: 32px;
        --space-5: 40px;
        --space-6: 48px;
        --space-8: 64px;
        --space-10: 80px;
        --space-12: 96px;
        
        /* Colors */
        --bg-primary: #0f0f0f;
        --bg-secondary: #1a1a1a;
        --bg-tertiary: #2d2d2d;
        --bg-card: #1e1e1e;
        --border-color: #404040;
        --border-light: #555555;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --text-muted: #707070;
        --accent-blue: #00d4ff;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-red: #f87171;
        --accent-yellow: #fbbf24;
        --shadow-glow: rgba(0, 212, 255, 0.15);
        --gradient-primary: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%);
        --gradient-secondary: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        
        /* Typography scale */
        --font-xs: 0.75rem;
        --font-sm: 0.875rem;
        --font-base: 1rem;
        --font-lg: 1.125rem;
        --font-xl: 1.25rem;
        --font-2xl: 1.5rem;
        --font-3xl: 1.875rem;
        --font-4xl: 2.25rem;
        
        /* Border radius */
        --radius-sm: 6px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --radius-full: 50px;
    }
    
    /* Main app styling */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden;
    }
    
    /* Style the history button */
    [data-testid="column"]:last-child .stButton > button {
        background: var(--bg-tertiary) !important;
        border: 2px solid var(--accent-blue) !important;
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="column"]:last-child .stButton > button:hover {
        background: rgba(0, 212, 255, 0.1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-secondary);
    }
    
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 2px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--accent-blue) !important;
        background: rgba(0, 212, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Chemical Parser and Reaction Engine classes
class ChemicalParser:
    """Parse and validate chemical formulas and reactions."""
    
    @staticmethod
    def parse_formula(formula: str) -> Dict[str, int]:
        """Parse a chemical formula into element counts."""
        formula = formula.replace(" ", "")
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        composition = {}
        for element, count in matches:
            count = int(count) if count else 1
            composition[element] = composition.get(element, 0) + count
        
        return composition
    
    @staticmethod
    def format_formula(composition: Dict[str, int]) -> str:
        """Format element composition back to chemical formula."""
        if not composition:
            return ""
        
        formula_parts = []
        for element, count in sorted(composition.items()):
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}<sub>{count}</sub>")
        
        return "".join(formula_parts)

class ReactionEngine:
    """Handle chemical reactions and composition calculations."""
    
    def __init__(self):
        self.molecules = []
        self.coefficients = []
    
    def add_molecule(self, formula: str, coefficient: int = 1):
        """Add a molecule to the reaction."""
        composition = ChemicalParser.parse_formula(formula)
        if composition:
            self.molecules.append({
                'formula': formula,
                'composition': composition,
                'coefficient': coefficient,
                'molecular_weight': self._calculate_molecular_weight(composition)
            })
    
    def add_direct_composition(self, composition: Dict[str, float], display_name: str):
        """Add a direct composition to the reaction."""
        self.molecules.append({
            'type': 'direct',
            'composition': composition.copy(),
            'formula': 'Direct Composition',
            'coefficient': 1,
            'molecular_weight': sum(
                material_db.get_material(elem).get('atomic_weight', 50) * pct / 100
                for elem, pct in composition.items()
            ),
            'display_name': display_name
        })
    
    def _calculate_molecular_weight(self, composition: Dict[str, int]) -> float:
        """Calculate molecular weight from composition."""
        total_weight = 0
        for element, count in composition.items():
            elem_data = material_db.get_material(element)
            if elem_data and 'atomic_weight' in elem_data:
                total_weight += elem_data['atomic_weight'] * count
            else:
                total_weight += 50.0 * count  # Fallback
        return total_weight
    
    def get_reaction_equation(self) -> str:
        """Get the formatted reaction equation."""
        if not self.molecules:
            return "No reaction defined"
        
        equation_parts = []
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                equation_parts.append(mol.get('display_name', 'Direct Composition'))
            else:
                coeff = f"{mol['coefficient']}" if mol['coefficient'] > 1 else ""
                formula = ChemicalParser.format_formula(mol['composition'])
                equation_parts.append(f"{coeff}{formula}")
        
        return " + ".join(equation_parts)
    
    def get_total_composition(self) -> Dict[str, float]:
        """Calculate total elemental composition by mass percentage."""
        if not self.molecules:
            return {}
        
        # Check if we have a direct composition
        if len(self.molecules) == 1 and self.molecules[0].get('type') == 'direct':
            return self.molecules[0]['composition']
        
        # Calculate for molecular mode
        element_masses = {}
        total_mass = 0
        
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                continue
                
            mol_mass = mol['molecular_weight'] * mol['coefficient']
            total_mass += mol_mass
            
            for element, count in mol['composition'].items():
                elem_data = material_db.get_material(element)
                if elem_data and 'atomic_weight' in elem_data:
                    atomic_weight = elem_data['atomic_weight']
                else:
                    atomic_weight = 50.0
                
                element_mass = atomic_weight * count * mol['coefficient']
                element_masses[element] = element_masses.get(element, 0) + element_mass
        
        # Convert to percentages
        if total_mass > 0:
            return {element: (mass / total_mass) * 100 
                   for element, mass in element_masses.items()}
        return {}

# Initialize session state
if 'reaction_engine' not in st.session_state:
    st.session_state.reaction_engine = ReactionEngine()

if 'current_molecule' not in st.session_state:
    st.session_state.current_molecule = {}

if 'direct_composition' not in st.session_state:
    st.session_state.direct_composition = {}

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Molecular Builder"

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

if 'calculation_saved' not in st.session_state:
    st.session_state.calculation_saved = False

# Authentication
if not check_password():
    st.stop()

# Main header with history button
header_col1, header_col2, header_col3 = st.columns([1, 5, 1])

with header_col1:
    st.empty()  # Left spacer

with header_col2:
    st.markdown("""
    <div style="text-align: center; width: 100%;">
        <h1 style="
            font-size: var(--font-4xl);
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
            letter-spacing: 2px;
            text-align: center;
            display: block;
        ">EMI SHIELDER</h1>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    # History button in top-right (smaller)
    history_count = len(load_history())
    if st.button(f"History ({history_count})", key="history_button", help="View calculation history"):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()

# Create a container for right sidebar
if st.session_state.show_history:
    # Create columns for main content and right sidebar
    main_col, sidebar_col = st.columns([8, 3])
    
    with sidebar_col:
        # History sidebar content
        st.markdown("""
        <div style="
            background: var(--bg-card);
            border: 2px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--space-3);
            margin-bottom: var(--space-3);
        ">
            <h3 style="
                font-size: var(--font-xl);
                color: var(--text-primary);
                margin: 0;
                text-align: center;
            ">History</h3>
        </div>
        """, unsafe_allow_html=True)
        
        history = load_history()
        
        if not history:
            st.info("No calculations yet. Complete a calculation to see it here.")
        else:
            # Add export button
            if st.button("Export CSV", use_container_width=True):
                # Prepare data for export
                export_data = []
                for calc in history:
                    export_data.append({
                        'Timestamp': calc['timestamp'],
                        'Mode': calc['mode'],
                        'Composition': calc['composition_summary'],
                        'Thickness (mm)': calc['thickness'],
                        'Frequency (MHz)': calc['frequency'],
                        'Total SE (dB)': calc['total_se'],
                        'Reflection Loss (dB)': calc['reflection_loss'],
                        'Absorption Loss (dB)': calc['absorption_loss']
                    })
                
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f"emi_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
            
            # Display history entries (show only recent 10 in sidebar)
            for idx, calc in enumerate(history[:10]):
                with st.expander(f"{calc['composition_summary']}", expanded=False):
                    st.markdown(f"""
                    **{calc['total_se']:.1f} dB**  
                    {calc['thickness']}mm @ {calc['frequency']}MHz  
                    {calc['timestamp']}
                    """)
                    
                    if st.button(f"Load", key=f"load_{idx}", use_container_width=True):
                        # Load this calculation
                        load_calc = calc
                        
                        # Set input mode
                        st.session_state.input_mode = load_calc['mode']
                        
                        # Clear current state
                        st.session_state.reaction_engine = ReactionEngine()
                        st.session_state.current_molecule = {}
                        st.session_state.direct_composition = {}
                        
                        # Restore composition
                        if load_calc['mode'] == "Molecular Builder":
                            # Restore molecules
                            for mol_data in load_calc['molecules']:
                                st.session_state.reaction_engine.add_molecule(
                                    mol_data['formula'],
                                    mol_data['coefficient']
                                )
                        else:
                            # Restore direct composition
                            st.session_state.reaction_engine.add_direct_composition(
                                load_calc['composition'],
                                load_calc['composition_summary']
                            )
                        
                        # Store parameters for loading
                        st.session_state.loaded_thickness = load_calc['thickness']
                        st.session_state.loaded_frequency = load_calc['frequency']
                        
                        # Show results
                        st.session_state.show_results = True
                        st.session_state.show_history = False
                        st.rerun()
            
            if len(history) > 10:
                st.info(f"Showing 10 of {len(history)} entries")
        
        # Close button at bottom
        st.markdown("---")
        if st.button("Close", use_container_width=True):
            st.session_state.show_history = False
            st.rerun()
else:
    # No sidebar, use full width
    main_col = st.container()

# Wrap all main content in the main_col container
with main_col:
    # Mode selection header without box
    st.markdown("""
    <div style="margin: var(--space-4) 0; width: 100%;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            text-align: center;
            width: 100%;
            display: block;
        ">Select Input Method</h3>
    </div>
    """, unsafe_allow_html=True)

    # Mode selection dropdown (left-aligned)
    input_mode = st.selectbox(
        "Choose how to define your materials:",
        ["Molecular Builder", "Direct Composition (Weight %)"],
        index=0 if st.session_state.input_mode == "Molecular Builder" else 1,
        help="Molecular Builder: Build molecules from elements | Direct Composition: Enter percentages directly",
        key="mode_selector"
    )
    st.session_state.input_mode = input_mode
    
    # Show instructions based on mode
    with st.expander("How To Use", expanded=True):
        if "Molecular Builder" in input_mode:
            st.write("""
            ### Molecular Builder Mode
            1. Select elements from the periodic table below
            2. Adjust quantities to build your molecule
            3. Add molecules to create your material reaction
            4. Set shield thickness and frequency parameters
            5. Click REACT to analyze EMI shielding
            """)
        else:
            st.write("""
            ### Direct Composition Mode
            1. Enter material composition by weight percentage (e.g., 70% Fe, 30% C)
            2. Total must equal 100%
            3. Add composition to reaction
            4. Set shield thickness and frequency parameters
            5. Click REACT to analyze EMI shielding
            
            ### Example Compositions:
            - **Steel**: Fe: 98%, C: 2%
            - **Brass**: Cu: 70%, Zn: 30%
            - **Stainless Steel**: Fe: 70%, Cr: 18%, Ni: 10%, Mo: 2%
            """)
    
    st.markdown("---")
    
    # Main content based on mode
    if "Molecular Builder" in input_mode:
        # MOLECULAR BUILDER MODE
        st.markdown("""
        <div style="margin: var(--space-4) 0; width: 100%;">
            <h3 style="
                font-size: var(--font-2xl);
                font-weight: 600;
                color: var(--text-primary);
                text-align: center;
                width: 100%;
                display: block;
            ">Build Your Molecule</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display current molecule
        current_formula = ChemicalParser.format_formula(st.session_state.current_molecule)
        if current_formula:
            st.markdown(f"""
            <div style="
                background: rgba(0, 212, 255, 0.1);
                border: 2px solid var(--accent-blue);
                border-radius: var(--radius-lg);
                padding: var(--space-4);
                text-align: center;
                margin: var(--space-4) 0;
                font-size: var(--font-2xl);
                font-family: 'JetBrains Mono', monospace;
            ">
                {current_formula}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Select elements from the periodic table below to build your molecule")
        
        # Element selection - simplified periodic table
        st.markdown("### Select Elements")
        
        # Common elements in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Metals", "Non-metals", "Transition Metals", "All Elements"])
        
        with tab1:
            metals = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Zn', 'Sn', 'Pb']
            cols = st.columns(6)
            for i, elem in enumerate(metals):
                with cols[i % 6]:
                    if st.button(elem, key=f"metal_{elem}", use_container_width=True):
                        if elem in st.session_state.current_molecule:
                            st.session_state.current_molecule[elem] += 1
                        else:
                            st.session_state.current_molecule[elem] = 1
                            st.rerun()
            
        with tab2:
            nonmetals = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
            cols = st.columns(6)
            for i, elem in enumerate(nonmetals):
                with cols[i % 6]:
                    if st.button(elem, key=f"nonmetal_{elem}", use_container_width=True):
                        if elem in st.session_state.current_molecule:
                            st.session_state.current_molecule[elem] += 1
                        else:
                            st.session_state.current_molecule[elem] = 1
                        st.rerun()
            
        with tab3:
            transition = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Mo', 'Ag', 'Au', 'W']
            cols = st.columns(6)
            for i, elem in enumerate(transition):
                with cols[i % 6]:
                    if st.button(elem, key=f"trans_{elem}", use_container_width=True):
                        if elem in st.session_state.current_molecule:
                            st.session_state.current_molecule[elem] += 1
                        else:
                            st.session_state.current_molecule[elem] = 1
                        st.rerun()
            
        with tab4:
            st.info("For more elements, use the element selector below")
            
            # Create element options with both symbol and name
            element_options = []
            elem_map = {}  # To map display names back to symbols
            
            for elem in sorted(material_db.periodic_table.keys()):
                elem_data = material_db.get_material(elem)
                elem_name = elem_data.get('name', elem) if elem_data else elem
                display_name = f"{elem} ({elem_name})"
                element_options.append(display_name)
                elem_map[display_name] = elem
            
            selected_display = st.selectbox("Select element:", [""] + element_options)
            selected_elem = elem_map.get(selected_display, "")
            
            if selected_elem and st.button("Add Element"):
                if selected_elem in st.session_state.current_molecule:
                    st.session_state.current_molecule[selected_elem] += 1
                else:
                    st.session_state.current_molecule[selected_elem] = 1
                st.rerun()
            
        # Quantity adjustment
        if st.session_state.current_molecule:
            st.markdown("### Adjust Quantities")
            cols = st.columns(4)
            for i, (element, quantity) in enumerate(st.session_state.current_molecule.items()):
                with cols[i % 4]:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{element}**")
                    with col2:
                        new_qty = st.number_input(
                            f"Qty",
                            min_value=0,
                            max_value=99,
                            value=quantity,
                            key=f"qty_{element}",
                            label_visibility="collapsed"
                        )
                        if new_qty != quantity:
                            if new_qty == 0:
                                del st.session_state.current_molecule[element]
                            else:
                                st.session_state.current_molecule[element] = new_qty
                            st.rerun()
                    with col3:
                        if st.button("X", key=f"del_{element}"):
                            del st.session_state.current_molecule[element]
                            st.rerun()
                
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col2:
                if st.button("Add to Reaction", type="primary", use_container_width=True):
                    formula = ChemicalParser.format_formula(st.session_state.current_molecule)
                    st.session_state.reaction_engine.add_molecule(
                        formula.replace('<sub>', '').replace('</sub>', ''),
                        1
                    )
                    st.session_state.current_molecule = {}
                    st.success("Molecule added to reaction!")
                    st.rerun()
            with col3:
                if st.button("Clear Molecule", use_container_width=True):
                    st.session_state.current_molecule = {}
                    st.rerun()

    else:
        # DIRECT COMPOSITION MODE
        st.markdown("""
        <div style="margin: var(--space-4) 0; width: 100%;">
            <h3 style="
                font-size: var(--font-2xl);
                font-weight: 600;
                color: var(--text-primary);
                text-align: center;
                width: 100%;
                display: block;
            ">Direct Composition Input</h3>
            <p style="text-align: center; color: var(--text-secondary); width: 100%;">
                Enter your material composition by weight percentage
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add element interface
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            # Create element options with both symbol and name
            available_elements = []
            element_map = {}  # To map display names back to symbols
            
            for elem in sorted(material_db.periodic_table.keys()):
                if elem not in st.session_state.direct_composition:
                    elem_data = material_db.get_material(elem)
                    elem_name = elem_data.get('name', elem) if elem_data else elem
                    display_name = f"{elem} - {elem_name}"
                    available_elements.append(display_name)
                    element_map[display_name] = elem
            
            selected_display = st.selectbox(
                "Select Element",
                [""] + available_elements,
                key="new_element_select"
            )
            
            # Get the actual element symbol from the display name
            new_element = element_map.get(selected_display, "")
        
        with col2:
            new_percentage = st.number_input(
                "Percentage (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key="new_percentage_input"
            )
        
        with col3:
            if st.button("Add", key="add_element_btn", use_container_width=True):
                if new_element and new_percentage > 0:
                    st.session_state.direct_composition[new_element] = new_percentage
                    st.rerun()
            
        # Display current composition
        if st.session_state.direct_composition:
            total = sum(st.session_state.direct_composition.values())
            
            # Show total
            if abs(total - 100.0) < 0.01:
                st.success(f"Total: {total:.1f}%")
            else:
                st.error(f"Total: {total:.1f}% (must equal 100%)")
            
            # Composition table
            st.markdown("### Current Composition")
            for element, percentage in list(st.session_state.direct_composition.items()):
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    elem_data = material_db.get_material(element)
                    elem_name = elem_data.get('name', element) if elem_data else element
                    st.write(f"**{element}** - {elem_name}")
                
                with col2:
                    new_pct = st.number_input(
                        f"{element} %",
                        min_value=0.0,
                        max_value=100.0,
                        value=percentage,
                        step=0.1,
                        key=f"pct_{element}",
                        label_visibility="collapsed"
                    )
                    if new_pct != percentage:
                        if new_pct == 0:
                            del st.session_state.direct_composition[element]
                        else:
                            st.session_state.direct_composition[element] = new_pct
                        st.rerun()
                
                with col3:
                    if st.button("Delete", key=f"del_direct_{element}"):
                        del st.session_state.direct_composition[element]
                        st.rerun()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if abs(total - 100.0) > 0.01 and total > 0:
                    if st.button("Normalize to 100%", use_container_width=True):
                        for elem in st.session_state.direct_composition:
                            st.session_state.direct_composition[elem] *= (100.0 / total)
                        st.rerun()
            
            with col2:
                if abs(total - 100.0) < 0.01:
                    if st.button("Add to Reaction", type="primary", use_container_width=True):
                        # Create display name
                        sorted_comp = sorted(st.session_state.direct_composition.items(), 
                                           key=lambda x: x[1], reverse=True)
                        display_parts = [f"{elem}({pct:.1f}%)" for elem, pct in sorted_comp[:3]]
                        if len(sorted_comp) > 3:
                            display_parts.append("...")
                        display_name = " ".join(display_parts)
                        
                        # Add to reaction
                        st.session_state.reaction_engine.add_direct_composition(
                            st.session_state.direct_composition,
                            display_name
                        )
                        st.session_state.direct_composition = {}
                        st.success("Composition added to reaction!")
                        st.rerun()
            
            with col3:
                if st.button("Clear", use_container_width=True):
                    st.session_state.direct_composition = {}
                    st.rerun()

    # COMPLETE REACTION section
    st.markdown("---")
    st.markdown("""
    <div style="margin: var(--space-6) 0 var(--space-4) 0; width: 100%;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            text-align: center;
            width: 100%;
            display: block;
        ">Complete Reaction</h3>
    </div>
    """, unsafe_allow_html=True)

    reaction_eq = st.session_state.reaction_engine.get_reaction_equation()
    if len(st.session_state.reaction_engine.molecules) > 0:
        st.markdown(f"""
        <div style="
            background: rgba(139, 92, 246, 0.1);
            border: 2px solid var(--accent-purple);
            border-radius: var(--radius-lg);
            padding: var(--space-4);
            text-align: center;
            font-size: var(--font-xl);
            font-family: 'JetBrains Mono', monospace;
        ">
            {reaction_eq}
        </div>
        """, unsafe_allow_html=True)
        
        # Clear reaction button
        if st.button("Clear Reaction", use_container_width=True):
            st.session_state.reaction_engine = ReactionEngine()
            st.session_state.show_results = False
            st.session_state.calculation_saved = False  # Reset for new calculation
            st.rerun()
        
        # Shield parameters
        st.markdown("### Shield Parameters")
        
        # Use loaded values if available
        default_thickness = st.session_state.get('loaded_thickness', 1.0)
        default_frequency = st.session_state.get('loaded_frequency', 100.0)
        default_grain_size = st.session_state.get('loaded_grain_size', 10.0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            thickness = st.number_input(
                "Thickness (mm)",
                min_value=0.01,
                max_value=100.0,
                value=float(default_thickness),
                step=0.1
            )
        
        with col2:
            frequency = st.number_input(
                "Frequency (MHz)",
                min_value=0.1,
                max_value=30000.0,
                value=float(default_frequency),
                step=10.0,
                help="Range: 0.1 MHz to 30 GHz (30,000 MHz)"
            )
        
        with col3:
            # Grain size input
            grain_size_unit = st.selectbox(
                "Grain Size Unit",
                ["μm (micrometers)", "nm (nanometers)"],
                key="grain_unit"
            )
            
            if "nm" in grain_size_unit:
                grain_size_value = st.number_input(
                    "Grain Size",
                    min_value=10.0,
                    max_value=1000.0,
                    value=float(default_grain_size * 1000) if default_grain_size < 1e-6 else 100.0,
                    step=10.0,
                    help="Typical: 10-1000 nm for nanocrystalline materials"
                )
                grain_size_m = grain_size_value * 1e-9
            else:
                grain_size_value = st.number_input(
                    "Grain Size",
                    min_value=0.1,
                    max_value=1000.0,
                    value=float(default_grain_size),
                    step=1.0,
                    help="Typical: 10-100 μm for conventional materials"
                )
                grain_size_m = grain_size_value * 1e-6
        
        # Clear loaded values after use
        if 'loaded_thickness' in st.session_state:
            del st.session_state.loaded_thickness
        if 'loaded_frequency' in st.session_state:
            del st.session_state.loaded_frequency
        
        # REACT button
        st.markdown("""
        <div style="margin: var(--space-6) 0; text-align: center;">
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("REACT", type="primary", use_container_width=True):
                st.session_state.show_results = True
                st.session_state.calculation_saved = False  # Reset flag for new calculation
                st.rerun()

    else:
        st.info("Add materials to build your reaction using the input method above")

    # Results section
    if st.session_state.show_results and len(st.session_state.reaction_engine.molecules) > 0:
        st.markdown("---")
        st.markdown("""
        <div style="margin: var(--space-6) 0; width: 100%;">
            <h2 style="
                font-size: var(--font-3xl);
                font-weight: 700;
                color: var(--text-primary);
                text-align: center;
                width: 100%;
                display: block;
            ">RESULTS</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate results
        total_comp = st.session_state.reaction_engine.get_total_composition()
        
        if total_comp:
            # Material properties calculation - matching main app logic
            conductivity = 0
            permeability = 1.0
            permittivity = 1.0
            density = 0
            
            for element, percentage in total_comp.items():
                elem_data = material_db.get_material(element)
                if elem_data:
                    weight = percentage / 100.0
                    
                    # Conductivity calculation (weighted sum)
                    elem_conductivity = elem_data.get('electrical_conductivity', 1e6)
                    conductivity += elem_conductivity * weight
                    
                    # Permeability calculation (geometric mean)
                    elem_permeability = elem_data.get('relative_permeability', 1.0)
                    # Ensure permeability is at least 0.999 for diamagnetic materials
                    elem_permeability = max(elem_permeability, 0.999)
                    permeability *= elem_permeability ** weight
                    
                    # Permittivity calculation (geometric mean)
                    elem_permittivity = elem_data.get('relative_permittivity', 1.0)
                    # Ensure permittivity is at least 1.0
                    elem_permittivity = max(elem_permittivity, 1.0)
                    permittivity *= elem_permittivity ** weight
                    
                    # Density calculation (weighted sum)
                    elem_density = elem_data.get('density', 1000)
                    density += elem_density * weight
            
            # Ensure values are within valid ranges
            conductivity = max(conductivity, 1e-10)  # Minimum conductivity
            permeability = max(permeability, 0.999)  # Minimum permeability
            permittivity = max(permittivity, 1.0)    # Minimum permittivity
            
            # EMI calculation with grain size
            try:
                result = emi_calculator.calculate_shielding_effectiveness(
                    conductivity,
                    permeability,
                    permittivity,  # Use calculated permittivity
                    thickness / 1000,
                    frequency * 1e6,
                    grain_size=grain_size_m,
                    include_confidence=True
                )
            except:
                result = emi_calculator.calculate_shielding_effectiveness(
                    conductivity,
                    permeability,
                    permittivity,
                    thickness / 1000,
                    frequency * 1e6,
                    grain_size=grain_size_m
                )
            
            # Save to history only once per calculation
            if not st.session_state.calculation_saved:
                molecules_data = []
                for mol in st.session_state.reaction_engine.molecules:
                    if mol.get('type') != 'direct':
                        molecules_data.append({
                            'formula': mol['formula'],
                            'coefficient': mol['coefficient']
                        })
                
                save_to_history(
                    mode=st.session_state.input_mode,
                    composition=total_comp,
                    thickness=thickness,
                    frequency=frequency,
                    result=result,
                    molecules=molecules_data,
                    grain_size=grain_size_m
                )
                st.session_state.calculation_saved = True  # Mark as saved
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total SE", f"{result['total_se']:.1f} dB")
            with col2:
                st.metric("Reflection Loss", f"{result['reflection_loss']:.1f} dB")
            with col3:
                st.metric("Absorption Loss", f"{result['absorption_loss']:.1f} dB")
            
            # Detailed breakdown
            with st.expander("Detailed Analysis", expanded=True):
                # Step 1: Material Composition
                st.markdown("### Step 1: Material Composition")
                st.write("**Total Elemental Composition (by mass %):**")
                comp_df = pd.DataFrame([
                    {'Element': elem, 'Percentage': f"{perc:.2f}%"}
                    for elem, perc in sorted(total_comp.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Step 2: Grain Size Effects (if applicable)
                if grain_size_m is not None and result.get('conductivity_reduction'):
                    st.markdown("### Step 2: Grain Size Effects on Conductivity")
                    st.markdown("""
                    **Mayadas-Shatzkes Model for Grain Boundary Scattering:**
                    
                    The grain boundaries scatter electrons, reducing conductivity:
                    """)
                    
                    # Calculate grain size parameters
                    lambda_mfp = 40e-9  # electron mean free path
                    R = 0.25  # reflection coefficient
                    alpha = lambda_mfp / grain_size_m * (R / (1 - R))
                    
                    st.latex(r"\alpha = \frac{\lambda}{d} \cdot \frac{R}{1-R}")
                    
                    st.markdown("""
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                    • α (alpha) = Grain boundary scattering parameter (dimensionless)<br>
                    • λ (lambda) = Electron mean free path in the material (m)<br>
                    • d = Average grain size of the material (m)<br>
                    • R = Reflection coefficient at grain boundaries (typically 0.1-0.5)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.code(f"""
α = λ/d × R/(1-R)
  = ({lambda_mfp:.2e} m)/({grain_size_m:.2e} m) × {R}/(1-{R})
  = {lambda_mfp/grain_size_m:.3f} × {R/(1-R):.3f}
  = {alpha:.3f}
""", language="text")
                    
                    st.markdown("**Conductivity Reduction Factor:**")
                    if alpha < 0.01:
                        st.latex(r"\frac{\sigma_{eff}}{\sigma_{bulk}} \approx 1 - 1.5\alpha")
                    else:
                        st.latex(r"\frac{\sigma_{eff}}{\sigma_{bulk}} = 1 - \frac{3}{2}\alpha + 3\alpha^2 - 3\alpha^3\ln(1 + \frac{1}{\alpha})")
                    
                    st.markdown("""
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                    • σ<sub>eff</sub> = Effective conductivity accounting for grain boundaries (S/m)<br>
                    • σ<sub>bulk</sub> = Bulk conductivity without grain boundary effects (S/m)<br>
                    • α = Grain boundary scattering parameter calculated above<br>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.code(f"""
Bulk conductivity: σ_bulk = {result['bulk_conductivity']:.2e} S/m
Reduction factor: {result['conductivity_reduction']:.3f}
Effective conductivity: σ_eff = {result['effective_conductivity']:.2e} S/m

Grain size effect: {(1 - result['conductivity_reduction']) * 100:.1f}% reduction in conductivity
""", language="text")
                
                # Step 3: Material Properties (renumber based on grain size)
                step_num = 3 if (grain_size_m is not None and result.get('conductivity_reduction')) else 2
                st.markdown(f"### Step {step_num}: Material Properties Calculation")
                st.markdown("**Individual Element Properties:**")
                
                # Verify total percentage
                total_percentage = sum(total_comp.values())
                if abs(total_percentage - 100.0) > 0.1:
                    st.error(f"Total composition is {total_percentage:.1f}% (should be 100%)")
                
                # Show properties for each element with calculation details
                st.markdown("**Weighted Property Calculations:**")
                cond_contributions = []
                perm_contributions = []
                dens_contributions = []
                
                for element, percentage in sorted(total_comp.items(), key=lambda x: x[1], reverse=True):
                    elem_data = material_db.get_material(element)
                    if elem_data:
                        elem_cond = elem_data.get('electrical_conductivity', 1e6)
                        elem_perm = elem_data.get('relative_permeability', 1.0)
                        elem_perm_safe = max(elem_perm, 0.999)
                        elem_dens = elem_data.get('density', 1000)
                        
                        # Calculate contributions
                        weight = percentage / 100.0
                        cond_contrib = elem_cond * weight
                        perm_contrib = elem_perm_safe ** weight
                        dens_contrib = elem_dens * weight
                        
                        cond_contributions.append(cond_contrib)
                        perm_contributions.append((elem_perm_safe, weight))
                        dens_contributions.append(dens_contrib)
                        
                        if percentage > 5:  # Show details for significant components
                            st.code(f"""
{element} ({percentage:.1f}%):
Base properties:
  σ = {elem_cond:.2e} S/m
  μᵣ = {elem_perm_safe:.3f}
  ρ = {elem_dens:.0f} kg/m³

Contributions:
  σ_contrib = {elem_cond:.2e} × {weight:.3f} = {cond_contrib:.2e} S/m
  μᵣ_contrib = {elem_perm_safe:.3f}^{weight:.3f} = {perm_contrib:.6f}
  ρ_contrib = {elem_dens:.0f} × {weight:.3f} = {dens_contrib:.1f} kg/m³
""", language="text")
                
                st.markdown("**Composite Material Properties:**")
                
                # Show calculation verification
                cond_sum = sum(cond_contributions)
                perm_product = 1.0
                for perm, weight in perm_contributions:
                    perm_product *= perm ** weight
                dens_sum = sum(dens_contributions)
                
                st.code(f"""
Conductivity (weighted sum):
  σ_eff = Σ(σᵢ × wᵢ) = {cond_sum:.2e} S/m
  Verification: {conductivity:.2e} S/m

Permeability (geometric mean):
  μᵣ,eff = Π(μᵣ,ᵢ^wᵢ) = {perm_product:.6f}
  Verification: {permeability:.6f}

Permittivity: εᵣ,eff = {permittivity:.3f}

Density (weighted sum):
  ρ_eff = Σ(ρᵢ × wᵢ) = {dens_sum:.0f} kg/m³
  Verification: {density:.0f} kg/m³
""", language="text")
                
                # Step 4 or 3: EMI Shielding Physics
                physics_step = step_num + 1
                st.markdown(f"### Step {physics_step}: EMI Shielding Physics")
                
                # Show detailed calculations
                st.markdown("**Input Parameters:**")
                st.code(f"""
Frequency: f = {frequency} MHz = {frequency * 1e6:.2e} Hz
Thickness: t = {thickness} mm = {thickness / 1000:.6f} m
Angular frequency: ω = 2πf = {2 * np.pi * frequency * 1e6:.2e} rad/s
""", language="text")
                
                # Skin Depth
                st.markdown("**Skin Depth Calculation:**")
                omega = 2 * np.pi * frequency * 1e6
                mu_abs = permeability * 4 * np.pi * 1e-7
                skin_depth_calc = np.sqrt(2 / (omega * mu_abs * conductivity))
                
                st.latex(r"\delta = \sqrt{\frac{2}{\omega\mu\sigma}}")
                
                st.markdown("""
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                • δ (delta) = Skin depth - the distance EM waves penetrate into the material (m)<br>
                • ω (omega) = Angular frequency = 2πf (rad/s)<br>
                • μ (mu) = Absolute permeability = μ<sub>r</sub> × μ<sub>0</sub> (H/m)<br>
                • σ (sigma) = Electrical conductivity of the material (S/m)
                </div>
                """, unsafe_allow_html=True)
                
                st.code(f"""
δ = √(2 / (ωμσ))
δ = √(2 / ({omega:.2e} × {mu_abs:.2e} × {conductivity:.2e}))
δ = √({2 / (omega * mu_abs * conductivity):.2e})
δ = {skin_depth_calc:.6f} m = {skin_depth_calc * 1000:.3f} mm
""", language="text")
                st.markdown("*Penetration depth of electromagnetic waves*")
                
                # Intrinsic Impedance
                st.markdown("**Intrinsic Impedance Calculation:**")
                
                st.latex(r"\eta = \sqrt{\frac{\mu}{\varepsilon^*}}")
                st.latex(r"\varepsilon^* = \varepsilon - j\frac{\sigma}{\omega}")
                
                st.markdown("""
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                • η (eta) = Intrinsic impedance of the material (Ω)<br>
                • μ = Absolute permeability (H/m)<br>
                • ε* = Complex permittivity (F/m)<br>
                • ε = Real part of permittivity = ε<sub>r</sub> × ε<sub>0</sub> (F/m)<br>
                • j = Imaginary unit (√-1)<br>
                • σ = Electrical conductivity (S/m)<br>
                • ω = Angular frequency (rad/s)
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate complex permittivity
                epsilon_abs = permittivity * 8.854e-12
                epsilon_complex_real = epsilon_abs
                epsilon_complex_imag = -conductivity / omega
                
                # Calculate intrinsic impedance components
                eta_complex = np.sqrt(complex(mu_abs, 0) / complex(epsilon_complex_real, epsilon_complex_imag))
                eta_mag = abs(eta_complex)
                eta_phase = np.angle(eta_complex) * 180 / np.pi
                
                st.code(f"""
ε* = ε - j(σ/ω) = {epsilon_abs:.2e} - j({conductivity:.2e}/{omega:.2e})
ε* = {epsilon_complex_real:.2e} - j{abs(epsilon_complex_imag):.2e}

η = √(μ / ε*)
η = √({mu_abs:.2e} / ({epsilon_complex_real:.2e} - j{abs(epsilon_complex_imag):.2e}))
|η| = {eta_mag:.2f} Ω
∠η = {eta_phase:.1f}°
""", language="text")
                st.markdown("*Material's resistance to electromagnetic wave propagation*")
                
                # Shielding Components
                st.markdown("**Shielding Components Calculation:**")
                
                st.latex(r"\gamma = j\omega\sqrt{\mu\varepsilon^*} = \alpha + j\beta")
                
                st.markdown("""
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                • γ (gamma) = Propagation constant (1/m)<br>
                • α (alpha) = Attenuation constant - real part of γ (Np/m)<br>
                • β (beta) = Phase constant - imaginary part of γ (rad/m)<br>
                • j = Imaginary unit<br>
                • ω, μ, ε* = As defined above
                </div>
                """, unsafe_allow_html=True)
                
                # Propagation constant
                gamma_complex = 1j * omega * np.sqrt(mu_abs * complex(epsilon_complex_real, epsilon_complex_imag))
                alpha = gamma_complex.real  # Attenuation constant
                beta = gamma_complex.imag   # Phase constant
                
                st.latex(r"\Gamma = \frac{\eta - Z_0}{\eta + Z_0}")
                st.latex(r"R = |\Gamma|^2")
                
                st.markdown("""
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                • Γ (Gamma) = Reflection coefficient at the material interface<br>
                • R = Power reflection coefficient<br>
                • Z<sub>0</sub> = Free space impedance ≈ 377 Ω<br>
                • η = Intrinsic impedance of the material (Ω)
                </div>
                """, unsafe_allow_html=True)
                
                # Reflection coefficient
                Z0 = 377  # Free space impedance
                gamma_r = (eta_complex - Z0) / (eta_complex + Z0)
                R = abs(gamma_r) ** 2
                
                st.latex(r"SE_R = -10\log_{10}(1 - R)")
                st.latex(r"SE_A = 8.686 \alpha t")
                
                st.markdown("""
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                • SE<sub>R</sub> = Reflection loss component (dB)<br>
                • SE<sub>A</sub> = Absorption loss component (dB)<br>
                • R = Power reflection coefficient<br>
                • α = Attenuation constant (Np/m)<br>
                • t = Material thickness (m)<br>
                • 8.686 = Conversion factor from Nepers to decibels
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate components
                reflection_loss_calc = -10 * np.log10(1 - R) if R < 1 else 0
                absorption_loss_calc = 8.686 * alpha * (thickness / 1000)
                
                st.code(f"""
Propagation constant: γ = jω√(με*) = {alpha:.2e} + j{beta:.2e}
Attenuation constant: α = {alpha:.2e} Np/m
Phase constant: β = {beta:.2e} rad/m

Reflection coefficient: Γ = (η - Z₀)/(η + Z₀) = {abs(gamma_r):.3f}∠{np.angle(gamma_r)*180/np.pi:.1f}°
Power reflection coefficient: R = |Γ|² = {R:.3f}

Reflection Loss: R_dB = -10log₁₀(1 - R) = {reflection_loss_calc:.1f} dB
Absorption Loss: A_dB = 8.686αt = 8.686 × {alpha:.2e} × {thickness/1000:.6f} = {absorption_loss_calc:.1f} dB

Multiple Reflection: M_dB = {result['multiple_reflection_loss']:.1f} dB
(Negligible when A_dB > 15 dB)

Note: M_dB refers to Multiple Reflection Loss in decibels

Total SE = R_dB + A_dB + M_dB = {result['total_se']:.1f} dB
""", language="text")
                
                st.latex(r"SE_{total} = SE_R + SE_A + SE_M")
                
                st.markdown("""
                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 20px; line-height: 1.4;">
                • SE<sub>total</sub> = Total shielding effectiveness (dB)<br>
                • SE<sub>R</sub> = Reflection loss - power lost due to impedance mismatch (dB)<br>
                • SE<sub>A</sub> = Absorption loss - power absorbed as waves propagate through material (dB)<br>
                • SE<sub>M</sub> = Multiple reflection loss - additional loss from internal reflections (dB)
                </div>
                """, unsafe_allow_html=True)
                
                # Verification section
                st.markdown("**Calculation Verification:**")
                
                # Compare skin depths
                skin_depth_diff = abs(skin_depth_calc - result['skin_depth']) / result['skin_depth'] * 100
                
                # Compare reflection loss
                refl_diff = abs(reflection_loss_calc - result['reflection_loss']) / max(result['reflection_loss'], 0.1) * 100
                
                # Compare absorption loss  
                abs_diff = abs(absorption_loss_calc - result['absorption_loss']) / max(result['absorption_loss'], 0.1) * 100
                
                verification_data = [
                    ["Skin Depth", f"{skin_depth_calc*1000:.3f} mm", f"{result['skin_depth']*1000:.3f} mm", f"{skin_depth_diff:.1f}%"],
                    ["Reflection Loss", f"{reflection_loss_calc:.1f} dB", f"{result['reflection_loss']:.1f} dB", f"{refl_diff:.1f}%"],
                    ["Absorption Loss", f"{absorption_loss_calc:.1f} dB", f"{result['absorption_loss']:.1f} dB", f"{abs_diff:.1f}%"],
                ]
                
                import pandas as pd
                verify_df = pd.DataFrame(verification_data, columns=["Parameter", "Calculated", "EMI Calculator", "Difference"])
                st.dataframe(verify_df, hide_index=True, use_container_width=True)
                
                if max(skin_depth_diff, refl_diff, abs_diff) < 1:
                    st.success("Calculations verified")
                elif max(skin_depth_diff, refl_diff, abs_diff) < 5:
                    st.info("Calculations verified - Good accuracy")
                else:
                    st.warning("Minor discrepancies detected - Check input values")
            
            # Visualization Section - At the bottom
            st.markdown("---")
            st.markdown("""
            <div style="margin: var(--space-6) 0; width: 100%;">
                <h2 style="
                    font-size: var(--font-3xl);
                    font-weight: 700;
                    color: var(--text-primary);
                    text-align: center;
                    width: 100%;
                    display: block;
                ">EMI SHIELDING ANALYSIS GRAPHS</h2>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Visualization & Analysis", expanded=True):
                # Generate data for plots
                freq_sweep = emi_calculator.frequency_sweep(
                    conductivity, permeability, permittivity,
                    thickness / 1000, grain_size_m,
                    freq_start=1e6, freq_end=30e9, num_points=50
                )
                
                thick_sweep = emi_calculator.thickness_sweep(
                    conductivity, permeability, permittivity,
                    frequency * 1e6, grain_size_m,
                    thickness_start=0.1e-3, thickness_end=10e-3, num_points=50
                )
                
                # Graph 1: EMI Shielding vs Frequency
                st.markdown("### Graph 1: EMI Shielding vs Frequency")
                st.markdown(f"*Fixed thickness: {thickness:.1f} mm, Grain size: {grain_size_value:.1f} {grain_size_unit.split()[0]}*")
                
                fig_freq = go.Figure()
                
                # Add total SE
                fig_freq.add_trace(go.Scatter(
                    x=freq_sweep['frequencies'] / 1e6,  # Convert to MHz
                    y=freq_sweep['total_ses'],
                    mode='lines',
                    name='Total SE',
                    line=dict(color='#00d4ff', width=3),
                    hovertemplate='%{x:.1f} MHz<br>%{y:.1f} dB<extra></extra>'
                ))
                
                # Add reflection loss
                fig_freq.add_trace(go.Scatter(
                    x=freq_sweep['frequencies'] / 1e6,
                    y=freq_sweep['reflection_losses'],
                    mode='lines',
                    name='Reflection Loss',
                    line=dict(color='#8b5cf6', width=2, dash='dash'),
                    hovertemplate='%{x:.1f} MHz<br>%{y:.1f} dB<extra></extra>'
                ))
                
                # Add absorption loss
                fig_freq.add_trace(go.Scatter(
                    x=freq_sweep['frequencies'] / 1e6,
                    y=freq_sweep['absorption_losses'],
                    mode='lines',
                    name='Absorption Loss',
                    line=dict(color='#10b981', width=2, dash='dot'),
                    hovertemplate='%{x:.1f} MHz<br>%{y:.1f} dB<extra></extra>'
                ))
                
                fig_freq.update_xaxes(
                    title_text="Frequency (MHz)",
                    type="log",
                    gridcolor='#404040',
                    showgrid=True
                )
                fig_freq.update_yaxes(
                    title_text="Shielding Effectiveness (dB)",
                    gridcolor='#404040',
                    showgrid=True
                )
                fig_freq.update_layout(
                    template="plotly_dark",
                    height=500,
                    hovermode='x unified',
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
                )
                
                st.plotly_chart(fig_freq, use_container_width=True)
                
                # Graph 2: EMI Shielding vs Thickness
                st.markdown("### Graph 2: EMI Shielding vs Thickness")
                st.markdown(f"*Fixed frequency: {frequency:.1f} MHz, Grain size: {grain_size_value:.1f} {grain_size_unit.split()[0]}*")
                
                fig_thick = go.Figure()
                
                # Multiple frequencies
                frequencies_to_plot = [10, 100, 1000, 10000, 30000]  # MHz
                colors = ['#00d4ff', '#8b5cf6', '#10b981', '#f87171', '#fbbf24']
                
                for freq_mhz, color in zip(frequencies_to_plot, colors):
                    if freq_mhz <= 30000:  # Only plot if within our range
                        thick_sweep_freq = emi_calculator.thickness_sweep(
                            conductivity, permeability, permittivity,
                            freq_mhz * 1e6, grain_size_m,
                            thickness_start=0.1e-3, thickness_end=10e-3, num_points=50
                        )
                        
                        fig_thick.add_trace(go.Scatter(
                            x=thick_sweep_freq['thicknesses'] * 1000,  # Convert to mm
                            y=thick_sweep_freq['total_ses'],
                            mode='lines',
                            name=f'{freq_mhz} MHz',
                            line=dict(color=color, width=2),
                            hovertemplate='%{x:.1f} mm<br>%{y:.1f} dB<extra></extra>'
                        ))
                
                fig_thick.update_xaxes(
                    title_text="Thickness (mm)",
                    gridcolor='#404040',
                    showgrid=True
                )
                fig_thick.update_yaxes(
                    title_text="Total Shielding Effectiveness (dB)",
                    gridcolor='#404040',
                    showgrid=True
                )
                fig_thick.update_layout(
                    template="plotly_dark",
                    height=500,
                    hovermode='x unified',
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
                )
                
                st.plotly_chart(fig_thick, use_container_width=True)
                
                # Graph 3: 3D Surface Plot
                st.markdown("### Graph 3: Interactive 3D Surface Plot")
                st.markdown(f"*Grain size: {grain_size_value:.1f} {grain_size_unit.split()[0]}*")
                st.markdown("*Note: M = Mega (million), so 1M = 1 MHz, 10M = 10 MHz, etc.*")
                
                # Generate mesh data
                freq_points = 30
                thick_points = 30
                
                freq_range = np.logspace(6, np.log10(30e9), freq_points)  # 1 MHz to 30 GHz
                thick_range = np.linspace(0.1e-3, 10e-3, thick_points)  # 0.1 to 10 mm
                
                SE_mesh = np.zeros((thick_points, freq_points))
                
                for i, thick in enumerate(thick_range):
                    for j, freq in enumerate(freq_range):
                        se_result = emi_calculator.calculate_shielding_effectiveness(
                            conductivity, permeability, permittivity,
                            thick, freq, grain_size_m
                        )
                        SE_mesh[i, j] = se_result['total_se']
                
                fig_3d = go.Figure(data=[go.Surface(
                    x=freq_range / 1e6,  # MHz
                    y=thick_range * 1000,  # mm
                    z=SE_mesh,
                    colorscale='Viridis',
                    colorbar=dict(title="SE (dB)"),
                    hovertemplate='Freq: %{x:.1f} MHz<br>Thickness: %{y:.2f} mm<br>SE: %{z:.1f} dB<extra></extra>'
                )])
                
                fig_3d.update_layout(
                    template="plotly_dark",
                    height=800,
                    scene=dict(
                        xaxis=dict(
                            title="Frequency (MHz)",
                            type="log",
                            gridcolor='#404040'
                        ),
                        yaxis=dict(
                            title="Thickness (mm)",
                            gridcolor='#404040'
                        ),
                        zaxis=dict(
                            title="SE (dB)",
                            gridcolor='#404040'
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    )
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Graph 4: Grain Size Effects
                if grain_size_m is not None:
                    st.markdown("### Graph 4: Grain Size Effects on EMI Shielding")
                    st.markdown(f"*Fixed thickness: {thickness:.1f} mm, Frequency: {frequency:.1f} MHz*")
                    
                    grain_sweep = emi_calculator.grain_size_sweep(
                        conductivity, permeability, permittivity,
                        thickness / 1000, frequency * 1e6,
                        grain_start=10e-9, grain_end=100e-6, num_points=50
                    )
                    
                    fig_grain = go.Figure()
                    
                    # Create secondary y-axis for conductivity
                    fig_grain = go.Figure().set_subplots(
                        specs=[[{"secondary_y": True}]]
                    )
                    
                    # Add total SE
                    fig_grain.add_trace(go.Scatter(
                        x=grain_sweep['grain_sizes'] * 1e6,  # Convert to μm
                        y=grain_sweep['total_ses'],
                        mode='lines',
                        name='Total SE',
                        line=dict(color='#00d4ff', width=3),
                        hovertemplate='%{x:.2f} μm<br>%{y:.1f} dB<extra></extra>'
                    ), secondary_y=False)
                    
                    # Add effective conductivity
                    fig_grain.add_trace(go.Scatter(
                        x=grain_sweep['grain_sizes'] * 1e6,
                        y=grain_sweep['effective_conductivities'] / 1e6,  # MS/m
                        mode='lines',
                        name='Effective Conductivity',
                        line=dict(color='#f87171', width=2, dash='dash'),
                        hovertemplate='%{x:.2f} μm<br>%{y:.2f} MS/m<extra></extra>'
                    ), secondary_y=True)
                    
                    fig_grain.update_xaxes(
                        title_text="Grain Size (μm)",
                        type="log",
                        gridcolor='#404040',
                        showgrid=True
                    )
                    fig_grain.update_yaxes(
                        title_text="Shielding Effectiveness (dB)",
                        gridcolor='#404040',
                        showgrid=True,
                        secondary_y=False
                    )
                    fig_grain.update_yaxes(
                        title_text="Effective Conductivity (MS/m)",
                        gridcolor='#404040',
                        showgrid=False,
                        secondary_y=True
                    )
                    fig_grain.update_layout(
                        template="plotly_dark",
                        height=500,
                        hovermode='x unified',
                        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
                    )
                    
                    st.plotly_chart(fig_grain, use_container_width=True)
                
                # Export buttons
                st.markdown("---")
                st.markdown("### Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Prepare data for CSV export
                    export_data = {
                        'Parameter': ['Frequency (MHz)', 'Thickness (mm)', 'Grain Size (μm)', 
                                     'Total SE (dB)', 'Reflection Loss (dB)', 'Absorption Loss (dB)',
                                     'Skin Depth (mm)', 'Effective Conductivity (S/m)'],
                        'Value': [frequency, thickness, grain_size_value,
                                 result['total_se'], result['reflection_loss'], result['absorption_loss'],
                                 result['skin_depth'] * 1000, result.get('effective_conductivity', conductivity)]
                    }
                    
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv,
                        file_name=f"emi_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Frequency sweep data
                    freq_df = pd.DataFrame({
                        'Frequency (MHz)': freq_sweep['frequencies'] / 1e6,
                        'Total SE (dB)': freq_sweep['total_ses'],
                        'Reflection Loss (dB)': freq_sweep['reflection_losses'],
                        'Absorption Loss (dB)': freq_sweep['absorption_losses']
                    })
                    
                    freq_csv = freq_df.to_csv(index=False)
                    st.download_button(
                        label="Download Frequency Data (CSV)",
                        data=freq_csv,
                        file_name=f"emi_freq_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Thickness sweep data
                    thick_df = pd.DataFrame({
                        'Thickness (mm)': thick_sweep['thicknesses'] * 1000,
                        'Total SE (dB)': thick_sweep['total_ses'],
                        'Reflection Loss (dB)': thick_sweep['reflection_losses'],
                        'Absorption Loss (dB)': thick_sweep['absorption_losses']
                    })
                    
                    thick_csv = thick_df.to_csv(index=False)
                    st.download_button(
                        label="Download Thickness Data (CSV)",
                        data=thick_csv,
                        file_name=f"emi_thick_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
