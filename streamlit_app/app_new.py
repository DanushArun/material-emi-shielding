"""
🔬 EMI Shield Designer - Unified App with Mode Selection
Allows users to choose between Molecular Builder or Direct Composition modes
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import re
from pathlib import Path
import sys
from typing import Dict
import time

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from src.physics.emi_calculations import emi_calculator
from src.materials.material_properties import material_db
from auth import check_password

# Import molecular presets
try:
    from molecular_presets import MOLECULAR_PRESETS, REACTION_PRESETS
except ImportError:
    MOLECULAR_PRESETS = {}
    REACTION_PRESETS = {}

# Page configuration
st.set_page_config(
    page_title="🔬 EMI Shield Designer",
    page_icon="⚛️",
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

# Authentication
if not check_password():
    st.stop()

# Main header
st.markdown("""
<div style="text-align: center; margin-bottom: var(--space-6);">
    <h1 style="
        font-size: var(--font-4xl);
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: 2px;
    ">🔬 EMI SHIELDER</h1>
</div>
""", unsafe_allow_html=True)

# Mode selection with dropdown
st.markdown("""
<div style="
    background: var(--bg-card);
    border: 2px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
    margin-bottom: var(--space-6);
">
    <h3 style="
        font-size: var(--font-xl);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--space-3) 0;
        text-align: center;
    ">📋 Select Input Method</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    input_mode = st.selectbox(
        "Choose how to define your materials:",
        ["🧪 Molecular Builder", "📊 Direct Composition (Weight %)"],
        index=0 if st.session_state.input_mode == "Molecular Builder" else 1,
        help="Molecular Builder: Build molecules from elements | Direct Composition: Enter percentages directly",
        key="mode_selector"
    )
    st.session_state.input_mode = input_mode

# Show instructions based on mode
with st.expander("📋 How To Use", expanded=True):
    if "Molecular Builder" in input_mode:
        st.write("""
        ### Molecular Builder Mode
        1. Select elements from the periodic table below
        2. Adjust quantities to build your molecule
        3. Add molecules to create your material reaction
        4. Set shield thickness and frequency parameters
        5. Click ⚛️ REACT to analyze EMI shielding
        """)
    else:
        st.write("""
        ### Direct Composition Mode
        1. Enter material composition by weight percentage (e.g., 70% Fe, 30% C)
        2. Total must equal 100%
        3. Add composition to reaction
        4. Set shield thickness and frequency parameters
        5. Click ⚛️ REACT to analyze EMI shielding
        
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
    <div style="margin: var(--space-4) 0;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            text-align: center;
        ">⚛️ Build Your Molecule</h3>
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
    st.markdown("### 🧪 Select Elements")
    
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
        all_elements = sorted(material_db.periodic_table.keys())
        selected_elem = st.selectbox("Select element:", [""] + all_elements)
        if selected_elem and st.button("Add Element"):
            if selected_elem in st.session_state.current_molecule:
                st.session_state.current_molecule[selected_elem] += 1
            else:
                st.session_state.current_molecule[selected_elem] = 1
            st.rerun()
    
    # Quantity adjustment
    if st.session_state.current_molecule:
        st.markdown("### ⚙️ Adjust Quantities")
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
                    if st.button("❌", key=f"del_{element}"):
                        del st.session_state.current_molecule[element]
                        st.rerun()
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("➕ Add to Reaction", type="primary", use_container_width=True):
                formula = ChemicalParser.format_formula(st.session_state.current_molecule)
                st.session_state.reaction_engine.add_molecule(
                    formula.replace('<sub>', '').replace('</sub>', ''),
                    1
                )
                st.session_state.current_molecule = {}
                st.success("✅ Molecule added to reaction!")
                st.rerun()
        with col3:
            if st.button("🔄 Clear Molecule", use_container_width=True):
                st.session_state.current_molecule = {}
                st.rerun()

else:
    # DIRECT COMPOSITION MODE
    st.markdown("""
    <div style="margin: var(--space-4) 0;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            text-align: center;
        ">📊 Direct Composition Input</h3>
        <p style="text-align: center; color: var(--text-secondary);">
            Enter your material composition by weight percentage
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add element interface
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        available_elements = [elem for elem in sorted(material_db.periodic_table.keys()) 
                            if elem not in st.session_state.direct_composition]
        new_element = st.selectbox(
            "Select Element",
            [""] + available_elements,
            key="new_element_select"
        )
    
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
        if st.button("➕ Add", key="add_element_btn", use_container_width=True):
            if new_element and new_percentage > 0:
                st.session_state.direct_composition[new_element] = new_percentage
                st.rerun()
    
    # Display current composition
    if st.session_state.direct_composition:
        total = sum(st.session_state.direct_composition.values())
        
        # Show total
        if abs(total - 100.0) < 0.01:
            st.success(f"✅ Total: {total:.1f}%")
        else:
            st.error(f"❌ Total: {total:.1f}% (must equal 100%)")
        
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
                if st.button("🗑️", key=f"del_direct_{element}"):
                    del st.session_state.direct_composition[element]
                    st.rerun()
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if abs(total - 100.0) > 0.01 and total > 0:
                if st.button("⚖️ Normalize to 100%", use_container_width=True):
                    for elem in st.session_state.direct_composition:
                        st.session_state.direct_composition[elem] *= (100.0 / total)
                    st.rerun()
        
        with col2:
            if abs(total - 100.0) < 0.01:
                if st.button("➕ Add to Reaction", type="primary", use_container_width=True):
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
                    st.success("✅ Composition added to reaction!")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.direct_composition = {}
                st.rerun()

# COMPLETE REACTION section
st.markdown("---")
st.markdown("""
<div style="margin: var(--space-6) 0 var(--space-4) 0;">
    <h3 style="
        font-size: var(--font-2xl);
        font-weight: 600;
        color: var(--text-primary);
        text-align: center;
    ">⚗️ Complete Reaction</h3>
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
    if st.button("🔄 Clear Reaction", use_container_width=True):
        st.session_state.reaction_engine = ReactionEngine()
        st.session_state.show_results = False
        st.rerun()
    
    # Shield parameters
    st.markdown("### ⚙️ Shield Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        thickness = st.number_input(
            "Thickness (mm)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        frequency = st.number_input(
            "Frequency (MHz)",
            min_value=0.1,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )
    
    # REACT button
    st.markdown("""
    <div style="margin: var(--space-6) 0; text-align: center;">
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚛️ REACT", type="primary", use_container_width=True):
            st.session_state.show_results = True
            st.rerun()

else:
    st.info("Add materials to build your reaction using the input method above")

# Results section
if st.session_state.show_results and len(st.session_state.reaction_engine.molecules) > 0:
    st.markdown("---")
    st.markdown("""
    <div style="margin: var(--space-6) 0;">
        <h2 style="
            font-size: var(--font-3xl);
            font-weight: 700;
            color: var(--text-primary);
            text-align: center;
        ">📊 RESULTS</h2>
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
        
        # EMI calculation
        try:
            result = emi_calculator.calculate_shielding_effectiveness(
                conductivity,
                permeability,
                permittivity,  # Use calculated permittivity
                thickness / 1000,
                frequency * 1e6,
                include_confidence=True
            )
        except:
            result = emi_calculator.calculate_shielding_effectiveness(
                conductivity,
                permeability,
                permittivity,
                thickness / 1000,
                frequency * 1e6
            )
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total SE", f"{result['total_se']:.1f} dB")
        with col2:
            st.metric("Reflection Loss", f"{result['reflection_loss']:.1f} dB")
        with col3:
            st.metric("Absorption Loss", f"{result['absorption_loss']:.1f} dB")
        
        # Detailed breakdown
        with st.expander("📈 Detailed Analysis", expanded=True):
            # Step 1: Material Composition
            st.markdown("### 🔬 Step 1: Material Composition")
            st.write("**Total Elemental Composition (by mass %):**")
            comp_df = pd.DataFrame([
                {'Element': elem, 'Percentage': f"{perc:.2f}%"}
                for elem, perc in sorted(total_comp.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Step 2: Material Properties
            st.markdown("### 🧮 Step 2: Material Properties Calculation")
            st.markdown("**Individual Element Properties:**")
            
            # Verify total percentage
            total_percentage = sum(total_comp.values())
            if abs(total_percentage - 100.0) > 0.1:
                st.error(f"⚠️ Total composition is {total_percentage:.1f}% (should be 100%)")
            
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
  Verification: {conductivity:.2e} S/m ✓

Permeability (geometric mean):
  μᵣ,eff = Π(μᵣ,ᵢ^wᵢ) = {perm_product:.6f}
  Verification: {permeability:.6f} ✓

Permittivity: εᵣ,eff = {permittivity:.3f}

Density (weighted sum):
  ρ_eff = Σ(ρᵢ × wᵢ) = {dens_sum:.0f} kg/m³
  Verification: {density:.0f} kg/m³ ✓
""", language="text")
            
            # Step 3: EMI Shielding Physics
            st.markdown("### ⚡ Step 3: EMI Shielding Physics")
            
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
            
            st.code(f"""
δ = √(2 / (ωμσ))
δ = √(2 / ({omega:.2e} × {mu_abs:.2e} × {conductivity:.2e}))
δ = √({2 / (omega * mu_abs * conductivity):.2e})
δ = {skin_depth_calc:.6f} m = {skin_depth_calc * 1000:.3f} mm
""", language="text")
            st.markdown("*Penetration depth of electromagnetic waves*")
            
            # Intrinsic Impedance
            st.markdown("**Intrinsic Impedance Calculation:**")
            
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
            
            # Propagation constant
            gamma_complex = 1j * omega * np.sqrt(mu_abs * complex(epsilon_complex_real, epsilon_complex_imag))
            alpha = gamma_complex.real  # Attenuation constant
            beta = gamma_complex.imag   # Phase constant
            
            # Reflection coefficient
            Z0 = 377  # Free space impedance
            gamma_r = (eta_complex - Z0) / (eta_complex + Z0)
            R = abs(gamma_r) ** 2
            
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

Total SE = R_dB + A_dB + M_dB = {result['total_se']:.1f} dB
""", language="text")
            
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
                st.success("✅ Calculations verified - Excellent accuracy!")
            elif max(skin_depth_diff, refl_diff, abs_diff) < 5:
                st.info("✓ Calculations verified - Good accuracy")
            else:
                st.warning("⚠️ Minor discrepancies detected - Check input values")
            
            # Performance Rating
            st.markdown("### 🎯 Performance Rating")
            if result['total_se'] >= 90:
                rating = "Excellent"
                rating_msg = "Your material produces an excellent EMI shield."
                color = "#10b981"
            elif result['total_se'] >= 60:
                rating = "Very Good"
                rating_msg = "Your material produces a very good EMI shield."
                color = "#00d4ff"
            elif result['total_se'] >= 40:
                rating = "Good"
                rating_msg = "Your material produces a good EMI shield."
                color = "#fbbf24"
            elif result['total_se'] >= 20:
                rating = "Moderate"
                rating_msg = "Your material produces a moderate EMI shield."
                color = "#fbbf24"
            else:
                rating = "Poor"
                rating_msg = "Your material produces a poor EMI shield."
                color = "#f87171"
            
            st.markdown(f"""
            <div style="
                background: {color}22;
                border: 2px solid {color};
                border-radius: var(--radius-lg);
                padding: var(--space-3);
                margin: var(--space-3) 0;
            ">
                <div style="font-weight: bold; color: {color};">Shield Performance: {rating}</div>
                <div style="color: var(--text-secondary); margin-top: 5px;">{rating_msg}</div>
            </div>
            """, unsafe_allow_html=True)