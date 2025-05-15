"""
Streamlit application for EMI shielding prediction.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.physics.emi_calculations import EMIShieldingCalculator
from src.materials.material_properties import MaterialProperties
from src.ml.emi_model import EMIPredictor
from src.config import DEFAULT_FREQ_MIN, DEFAULT_FREQ_MAX, DEFAULT_FREQ_POINTS

# Initialize components
materials = MaterialProperties()
calculator = EMIShieldingCalculator()
predictor = EMIPredictor(input_dim=6)  # 6 features: conductivity, permeability, permittivity, density, frequency, thickness

# Define material presets
MATERIAL_PRESETS = {
    "Magnetic Shielding (Low Frequency)": {
        "name": "Mu-Metal",
        "composition": {"Ni": 0.80, "Fe": 0.15, "Cu": 0.05},
        "thickness": 0.5,
        "description": "Best for shielding low-frequency magnetic fields (< 100 kHz)"
    },
    "Electric Shielding (High Frequency)": {
        "name": "Copper Shield",
        "composition": {"Cu": 0.95, "Sn": 0.05},
        "thickness": 1.0,
        "description": "Excellent for high-frequency electric fields (> 1 MHz)"
    },
    "General Purpose Shielding": {
        "name": "Aluminum Alloy",
        "composition": {"Al": 0.95, "Cu": 0.05},
        "thickness": 2.0,
        "description": "Good balance of shielding and weight"
    },
    "Cost-Effective Shielding": {
        "name": "Steel Alloy",
        "composition": {"Fe": 0.98, "Ni": 0.02},
        "thickness": 1.5,
        "description": "Economical option with good performance"
    }
}

def create_frequency_plot(results: dict) -> go.Figure:
    """Create an interactive plot of shielding effectiveness vs frequency."""
    fig = go.Figure()
    
    # Add traces for each component
    fig.add_trace(go.Scatter(
        x=results["frequency"],
        y=results["reflection_loss"],
        name="Reflection Loss",
        line=dict(color="blue")
    ))
    
    fig.add_trace(go.Scatter(
        x=results["frequency"],
        y=results["absorption_loss"],
        name="Absorption Loss",
        line=dict(color="green")
    ))
    
    fig.add_trace(go.Scatter(
        x=results["frequency"],
        y=results["multiple_reflection_loss"],
        name="Multiple Reflection Loss",
        line=dict(color="red")
    ))
    
    fig.add_trace(go.Scatter(
        x=results["frequency"],
        y=results["total_se"],
        name="Total Shielding Effectiveness",
        line=dict(color="purple", width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title="EMI Shielding Effectiveness vs Frequency",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Shielding Effectiveness (dB)",
        xaxis_type="log",
        hovermode="x unified"
    )
    
    return fig

def add_technical_details():
    """Add technical details section to the dashboard."""
    with st.expander("Technical Details"):
        st.markdown("""
        ### EMI Shielding Theory
        
        #### Shielding Mechanisms
        1. **Reflection Loss (R)**
        - Occurs at shield boundaries
        - Depends on impedance mismatch
        - Proportional to material conductivity
        - Formula: R = 20 log₁₀|Z₁ - Z₂|/|Z₁ + Z₂|
        
        2. **Absorption Loss (A)**
        - Occurs within shield material
        - Related to skin depth
        - Increases with thickness
        - Formula: A = 20(t/δ)log₁₀(e), where:
          - t = material thickness
          - δ = skin depth = √(2/ωμσ)
        
        3. **Multiple Reflection Loss (M)**
        - Internal reflections
        - Significant in thin shields
        - Decreases with thickness
        - Formula: M = 20log₁₀(1 - e⁻²ᵗ/δ)
        
        #### Material Properties Impact
        - **Conductivity (σ)**: Higher conductivity increases reflection loss
        - **Permeability (μ)**: Higher permeability improves absorption
        - **Thickness (t)**: Affects both absorption and multiple reflections
        - **Frequency (f)**: Influences skin depth and overall effectiveness
        
        #### Advanced Concepts
        - **Skin Effect**: Current flow concentration at material surface
        - **Near-field vs Far-field**: Different shielding behaviors
        - **Frequency Dependence**: Material properties vary with frequency
        - **Interface Effects**: Important in multi-layer shields
        """)

def add_example_combinations(prefix=""):
    """Add example material combinations section.
    
    Args:
        prefix: A string prefix to add to button keys to make them unique
    """
    with st.expander("Example Combinations"):
        st.markdown("""
        ### Common EMI Shielding Materials
        
        Click on any combination to load it:
        """)
        
        examples = {
            "Mu-Metal (High Permeability)": {
                "description": "Excellent for low-frequency magnetic fields",
                "composition": {"Ni": 0.80, "Fe": 0.15, "Cu": 0.05},
                "thickness": 0.5
            },
            "Copper-Based Shield": {
                "description": "High conductivity for electric field shielding",
                "composition": {"Cu": 0.95, "Sn": 0.05},
                "thickness": 1.0
            },
            "Aluminum Alloy": {
                "description": "Lightweight general-purpose shielding",
                "composition": {"Al": 0.95, "Cu": 0.05},
                "thickness": 2.0
            },
            "Steel Alloy": {
                "description": "Cost-effective broad spectrum shielding",
                "composition": {"Fe": 0.98, "Ni": 0.02},
                "thickness": 1.5
            }
        }
        
        for name, data in examples.items():
            if st.button(name, key=f"{prefix}_button_{name}"):
                st.session_state['example_composition'] = data['composition']
                st.session_state['example_thickness'] = data['thickness']
                st.rerun()  # Force a rerun to update the interface immediately
            st.markdown(f"**{name}**")
            st.markdown(f"- {data['description']}")
            st.markdown("- Composition: " + ", ".join([f"{k}: {v*100:.1f}%" for k, v in data['composition'].items()]))
            st.markdown(f"- Recommended thickness: {data['thickness']} mm")
            st.markdown("---")

def create_comparison_plot(comparison_data: list) -> go.Figure:
    """Create a comparison plot for multiple material combinations."""
    fig = go.Figure()
    
    for data in comparison_data:
        fig.add_trace(go.Scatter(
            x=data["frequency"],
            y=data["total_se"],
            name=data["name"],
            line=dict(dash='solid' if data["active"] else 'dot')
        ))
    
    fig.update_layout(
        title="Comparison of Shielding Effectiveness",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Total Shielding Effectiveness (dB)",
        xaxis_type="log",
        hovermode="x unified"
    )
    
    return fig

def main():
    # Set page configuration for better space utilization
    st.set_page_config(
        page_title="EMI Shielding Prediction System",
        page_icon="🛡️",
        layout="wide",  # Use wide layout for better space utilization
        initial_sidebar_state="auto"  # Let user decide sidebar state
    )
    
    # Custom CSS for better space utilization
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E1E1E;
        border-bottom: 2px solid #4CAF50;
    }
    div.stButton > button:first-child {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("EMI Shielding Prediction System")
    
    # Initialize session state
    if 'example_composition' not in st.session_state:
        st.session_state['example_composition'] = None
    if 'example_thickness' not in st.session_state:
        st.session_state['example_thickness'] = None
    if 'comparison_data' not in st.session_state:
        st.session_state['comparison_data'] = []
    if 'super_mode' not in st.session_state:
        st.session_state['super_mode'] = False
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state['sidebar_collapsed'] = False
    
    # SUPER MODE toggle
    super_mode = st.sidebar.toggle("🚀 SUPER MODE", help="Enable advanced controls and detailed analysis")
    st.session_state['super_mode'] = super_mode
    
    if super_mode:
        st.sidebar.warning("⚡ SUPER MODE enabled - Advanced controls available")
    
    # Add tabs for different sections with a more logical flow and better space utilization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Material Design", "📈 Analysis Results", "🔄 Comparison", "📚 Technical Reference"])
    
    with tab1:
        # Material Design Tab
        st.write("""
    Design custom materials for electromagnetic interference (EMI) shielding by selecting elements and adjusting their proportions.
    The system will calculate the shielding effectiveness based on material properties and physics principles.
    """)
    
    # Add example combinations to the Material Design tab
    add_example_combinations("tab1")
    
    # How to Use section
    with st.expander("Design Guide"):
        st.markdown("""
        ### Step 1: Define Your Shielding Need
        - Select the type of interference you want to shield against
        - Choose from common sources like transformers, RF equipment, etc.
        - Or select "Custom Requirements" for specialized needs
        
        ### Step 2: Choose Your Material
        **For Common Applications:**
        - Select from recommended materials optimized for your needs
        - Optionally fine-tune the composition
        
        **For Custom Design:**
        - Select elements from the periodic table
        - Adjust their proportions as needed
        
        ### Step 3: Set Parameters
        - Adjust material thickness
        - Select frequency range based on your application
        - Use preset ranges or specify custom frequencies
        
        ### Step 4: Analyze Performance
        - View material properties and shielding effectiveness
        - Check performance at specific frequencies
        - Compare different materials
        - Export results for documentation
        
        ### Advanced Features (SUPER MODE)
        Enable SUPER MODE in the sidebar for:
        - Direct property manipulation
        - Environmental effects simulation
        - Multi-layer configurations
        - Advanced visualizations
        - ML-based predictions
        
        ### Understanding Results
        **Shielding Components:**
        - **Reflection Loss**: Electromagnetic waves bouncing off the surface
        - **Absorption Loss**: Energy dissipated within the material
        - **Multiple Reflection Loss**: Internal reflections and interactions
        - **Total Effectiveness**: Combined shielding performance
        
        **Performance Ratings:**
        - < 30 dB: Basic shielding
        - 30-60 dB: Professional grade
        - 60-90 dB: Military grade
        - > 90 dB: Ultra-high performance
        
        ### Tips for Best Results
        - Start with recommended materials for your application
        - Consider cost vs. performance tradeoffs
        - Use the comparison feature to optimize your design
        - Export results for documentation and analysis
        """)
    
    # Material Selection Section
    st.sidebar.header("Material Selection")
    
    # Application Guide
    st.sidebar.markdown("""
    ### What are you shielding against?
    Choose the type of interference you want to protect against:
    """)
    
    interference_type = st.sidebar.selectbox(
        "Primary Interference Type",
        [
            "Low-frequency Magnetic Fields (Transformers, Motors)",
            "High-frequency Electric Fields (RF/Wireless)",
            "Broad Spectrum EMI (Mixed Sources)",
            "Custom Requirements"
        ],
        help="Select the main type of interference you need to shield against"
    )
    
    # Material Selection Method based on interference type
    if interference_type != "Custom Requirements":
        st.sidebar.markdown("### Recommended Materials")
        st.sidebar.markdown("Based on your interference type, here are optimized materials:")
        
        interference_to_materials = {
            "Low-frequency Magnetic Fields (Transformers, Motors)": [
                "Magnetic Shielding (Low Frequency)",
                "Cost-Effective Shielding"
            ],
            "High-frequency Electric Fields (RF/Wireless)": [
                "Electric Shielding (High Frequency)",
                "General Purpose Shielding"
            ],
            "Broad Spectrum EMI (Mixed Sources)": [
                "General Purpose Shielding",
                "Electric Shielding (High Frequency)"
            ]
        }
        
        selection_method = "Common Materials (Recommended)"
        material_type = st.sidebar.selectbox(
            "Select Material",
            interference_to_materials[interference_type] if interference_type is not None else [],
            help="Choose from materials optimized for your application"
        )
    else:
        st.sidebar.markdown("### Custom Material Design")
        st.sidebar.markdown("""
        Design your own material combination:
        - Choose up to 5 elements
        - Adjust their proportions
        - Fine-tune properties
        """)
        selection_method = "Custom Composition"
    
    # Get list of available elements
    available_elements = list(materials.elements_data.keys())
    
    if selection_method == "Common Materials (Recommended)":
        material_type = st.sidebar.selectbox(
            "Select Material Type",
            [
                "Magnetic Shielding (Low Frequency)",
                "Electric Shielding (High Frequency)",
                "General Purpose Shielding",
                "Cost-Effective Shielding"
            ]
        )
        
        # Predefined compositions based on material type
        material_presets = {
            "Magnetic Shielding (Low Frequency)": {
                "name": "Mu-Metal",
                "composition": {"Ni": 0.80, "Fe": 0.15, "Cu": 0.05},
                "thickness": 0.5,
                "description": "Best for shielding low-frequency magnetic fields (< 100 kHz)"
            },
            "Electric Shielding (High Frequency)": {
                "name": "Copper Shield",
                "composition": {"Cu": 0.95, "Sn": 0.05},
                "thickness": 1.0,
                "description": "Excellent for high-frequency electric fields (> 1 MHz)"
            },
            "General Purpose Shielding": {
                "name": "Aluminum Alloy",
                "composition": {"Al": 0.95, "Cu": 0.05},
                "thickness": 2.0,
                "description": "Good balance of shielding and weight"
            },
            "Cost-Effective Shielding": {
                "name": "Steel Alloy",
                "composition": {"Fe": 0.98, "Ni": 0.02},
                "thickness": 1.5,
                "description": "Economical option with good performance"
            }
        }
        
        selected_preset = material_presets[material_type] if material_type is not None else material_presets["General Purpose Shielding"]
        st.sidebar.info(selected_preset["description"])
        composition = selected_preset["composition"]
        
        # Allow fine-tuning of preset composition
        st.sidebar.markdown("### Fine-tune Composition (Optional)")
        fine_tune = st.sidebar.checkbox("Adjust composition")
        
        if fine_tune:
            composition = {}
            total = 0
            for element, default_weight in selected_preset["composition"].items():
                weight = st.sidebar.slider(
                    f"{element} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_weight * 100,
                    help=f"Adjust the percentage of {element} in the composition"
                )
                composition[element] = weight / 100
                total += weight
            
            # Normalize weights
            if total > 0:
                composition = {k: v/total for k, v in composition.items()}
            
    else:  # Custom Composition
        st.sidebar.markdown("""
        ### Custom Material Design
        Select elements and their proportions for your unique material.
        """)
        
        # Initialize session state for dynamic element selection
        if 'custom_elements' not in st.session_state:
            st.session_state['custom_elements'] = [{'element': 'Cu', 'percentage': 50.0}]
        
        # Display current composition
        st.sidebar.subheader("Current Composition")
        composition_text = ", ".join([f"{elem['element']}: {elem['percentage']:.1f}%" 
                                    for elem in st.session_state['custom_elements']])
        st.sidebar.text(composition_text)
        
        # Add new element
        st.sidebar.subheader("Add Element")
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            new_element = st.selectbox(
                "Select Element",
                [e for e in available_elements if e not in [elem['element'] for elem in st.session_state['custom_elements']]],
                help="Choose an element to add to your material"
            )
        with col2:
            if st.button("Add"):
                if len(st.session_state['custom_elements']) < 10:  # Limit to 10 elements
                    st.session_state['custom_elements'].append({'element': new_element, 'percentage': 0.0})
                    st.rerun()
        
        # Element adjustment
        st.sidebar.subheader("Adjust Elements")
        
        # Calculate total percentage
        total_percentage = sum(elem['percentage'] for elem in st.session_state['custom_elements'])
        
        # Display and adjust each element
        elements_to_remove = []
        composition = {}
        
        for i, elem_data in enumerate(st.session_state['custom_elements']):
            col1, col2, col3 = st.sidebar.columns([3, 6, 1])
            with col1:
                st.text(elem_data['element'])
            with col2:
                st.session_state['custom_elements'][i]['percentage'] = st.slider(
                    f"###{elem_data['element']}",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(elem_data['percentage']),
                    label_visibility="collapsed"
                )
            with col3:
                if st.button("X", key=f"remove_{i}"):
                    elements_to_remove.append(i)
            
            # Add to composition
            composition[elem_data['element']] = elem_data['percentage'] / 100
        
        # Remove elements marked for deletion
        for i in sorted(elements_to_remove, reverse=True):
            st.session_state['custom_elements'].pop(i)
            st.rerun()
        
        # Show element properties
        if st.sidebar.checkbox("Show Element Properties"):
            for elem in st.session_state['custom_elements']:
                if elem['percentage'] > 0:
                    element_props = materials.get_element_properties(elem['element'])
                    if element_props:
                        st.sidebar.markdown(f"**{elem['element']} Properties:**")
                        st.sidebar.text(f"Conductivity: {element_props['conductivity']:.2e} S/m")
                        st.sidebar.text(f"Permeability: {element_props['relative_permeability']:.2f}")
                        st.sidebar.text(f"Permittivity: {element_props['relative_permittivity']:.2f}")
    
    # Normalize weights
    total_weight = sum(composition.values())
    composition = {k: v/total_weight for k, v in composition.items()}
    
    # Material parameters
    st.sidebar.header("Material Parameters")
    
    # Check if we should load example thickness
    if st.session_state['example_thickness'] is not None:
        thickness = st.session_state['example_thickness']
        st.session_state['example_thickness'] = None  # Clear after loading
    else:
        thickness = st.sidebar.number_input(
            "Thickness (mm)",
            min_value=0.1,
            max_value=10.0,
            value=1.0
        ) / 1000  # Convert to meters
    
    # Frequency settings
    st.sidebar.header("Frequency Range")
    
    # Frequency range selection method
    freq_selection = st.sidebar.radio(
        "Select frequency range",
        ["Common Applications", "Custom Range"]
    )
    
    if freq_selection == "Common Applications":
        application = st.sidebar.selectbox(
            "Choose Application",
            [
                "Power Line Interference (50/60 Hz)",
                "Audio Equipment (20 Hz - 20 kHz)",
                "Radio Frequency (1 MHz - 1 GHz)",
                "Microwave Equipment (1 GHz - 10 GHz)",
                "Full Spectrum Analysis"
            ]
        )
        
        # Preset frequency ranges
        freq_ranges = {
            "Power Line Interference (50/60 Hz)": (30, 100),
            "Audio Equipment (20 Hz - 20 kHz)": (20, 20e3),
            "Radio Frequency (1 MHz - 1 GHz)": (1e6, 1e9),
            "Microwave Equipment (1 GHz - 10 GHz)": (1e9, 10e9),
            "Full Spectrum Analysis": (1e4, 1e12)
        }
        
        freq_min, freq_max = freq_ranges[application] if application is not None else (DEFAULT_FREQ_MIN, DEFAULT_FREQ_MAX)
        
        # Handle potential None value in application
        app_name = application.split('(')[0].strip() if application is not None else "selected"
        
        st.sidebar.info(f"""
        Selected Range: {freq_min:.0f} Hz - {freq_max:.0e} Hz
        
        This range is optimized for {app_name} applications.
        """)
        
    else:
        st.sidebar.markdown("""
        ### Custom Frequency Range
        Set your specific frequency range for analysis.
        
        Common frequency units:
        - 1 kHz = 1,000 Hz
        - 1 MHz = 1,000,000 Hz
        - 1 GHz = 1,000,000,000 Hz
        """)
        
        freq_min = st.sidebar.number_input(
            "Minimum Frequency (Hz)",
            min_value=float(1e4),
            max_value=float(1e12),
            value=float(DEFAULT_FREQ_MIN),
            format="%e",
            help="Lower bound of frequency range"
        )
        freq_max = st.sidebar.number_input(
            "Maximum Frequency (Hz)",
            min_value=float(1e4),
            max_value=float(1e12),
            value=float(DEFAULT_FREQ_MAX),
            format="%e",
            help="Upper bound of frequency range"
        )
    
    # Calculate material properties
    material_properties = materials.calculate_mixture_properties(composition)
    
    # Display material properties in a more space-efficient layout
    st.header("Material Properties")
    
    # Create a more compact layout for material properties
    prop_col1, prop_col2 = st.columns(2)
    
    with prop_col1:
        st.metric("Conductivity (S/m)", f"{material_properties['conductivity']:.2e}")
        st.metric("Relative Permeability", f"{material_properties['relative_permeability']:.2f}")
    
    with prop_col2:
        st.metric("Relative Permittivity", f"{material_properties['relative_permittivity']:.2f}")
        st.metric("Density (kg/m³)", f"{material_properties['density']:.2f}")
    
    # Fix for Number vs SupportsIndex type error
    def fix_freq_range(min_freq, max_freq, points):
        return np.logspace(np.log10(min_freq), np.log10(max_freq), int(points))
    
    # Calculate frequency response using the fixed function
    freq_range = fix_freq_range(freq_min, freq_max, DEFAULT_FREQ_POINTS)
    
    # Initialize SUPER MODE variables
    if super_mode:
        # Store SUPER MODE values in session state
        if 'super_mode_values' not in st.session_state:
            st.session_state['super_mode_values'] = {
                'custom_conductivity': float(material_properties["conductivity"]),
                'custom_permeability': float(material_properties["relative_permeability"]),
                'custom_permittivity': float(material_properties["relative_permittivity"]),
                'loss_tangent': 0.01,
                'temperature': 25,
                'humidity': 50,
                'aging_years': 0,
                'mechanical_stress': 0,
                'incident_angle': 0,
                'field_type': 'Far-field',
                'polarization': 'TE Mode',
                'field_strength': 100.0
            }
    
    # Prepare material parameters for physics calculations
    if super_mode and 'super_mode_values' in st.session_state:
        # Get values from session state
        super_values = st.session_state['super_mode_values']
        
        # Use custom properties from SUPER MODE
        material_params = {
            "conductivity": super_values['custom_conductivity'],
            "relative_permeability": super_values['custom_permeability'],
            "relative_permittivity": super_values['custom_permittivity'],
            "thickness": thickness
        }
        
        # Apply environmental effects
        if super_values['temperature'] != 25:  # Not room temperature
            # Temperature effect on conductivity (simplified model)
            temp_factor = 1.0 - 0.004 * (super_values['temperature'] - 25)  # ~0.4% change per degree C
            material_params["conductivity"] *= temp_factor
        
        if super_values['humidity'] > 0:
            # Humidity effect on permittivity (simplified model)
            humidity_factor = 1.0 + (super_values['humidity'] / 100) * 0.2  # Up to 20% increase at 100% humidity
            material_params["relative_permittivity"] *= humidity_factor
        
        if super_values['aging_years'] > 0:
            # Aging effect on material properties (simplified model)
            aging_factor = 1.0 - (super_values['aging_years'] / 50) * 0.15  # Up to 15% degradation after 50 years
            material_params["conductivity"] *= aging_factor
            material_params["relative_permeability"] *= aging_factor
        
        if super_values['mechanical_stress'] > 0:
            # Mechanical stress effect (simplified model)
            stress_factor = 1.0 - (super_values['mechanical_stress'] / 100) * 0.1  # Up to 10% degradation at 100% stress
            material_params["conductivity"] *= stress_factor
    else:
        # Use calculated material properties
        material_params = {
            "conductivity": material_properties["conductivity"],
            "relative_permeability": material_properties["relative_permeability"],
            "relative_permittivity": material_properties["relative_permittivity"],
            "thickness": thickness
        }
    
    # Physics-based calculations
    physics_results = calculator.calculate_frequency_response(
        freq_range,
        material_params
    )
    
    # Create and display the plot
    st.header("Shielding Effectiveness vs Frequency")
    fig = create_frequency_plot(physics_results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Results Analysis Section with improved space utilization
    st.header("Detailed Analysis")
    
    # Create a more compact layout for material composition and effectiveness rating
    comp_col1, comp_col2 = st.columns([1, 1])
    
    with comp_col1:
        # Material Composition Details
        st.subheader("Material Composition")
        composition_details = []
        for element, percentage in composition.items():
            element_props = materials.get_element_properties(element)
            if element_props:
                composition_details.append(f"{element}: {percentage*100:.1f}%")
        st.markdown("**Chemical Composition:**\n" + "\n".join([f"- {detail}" for detail in composition_details]))
    
    with comp_col2:
        # Calculate initial frequency index for middle frequency
        initial_freq = (freq_min + freq_max) / 2
        freq_idx = np.abs(freq_range - initial_freq).argmin()
        
        # Overall effectiveness rating
        total_se = physics_results['total_se'][freq_idx]
        effectiveness_levels = {
            (0, 30): ("Poor", "🔴", "Minimal shielding, not suitable for sensitive applications"),
            (30, 60): ("Good", "🟡", "Moderate shielding, suitable for general applications"),
            (60, 90): ("Excellent", "🟢", "High-performance shielding, suitable for sensitive equipment"),
            (90, float('inf')): ("Superior", "⭐", "Exceptional shielding, suitable for critical applications")
        }
        
        for (min_se, max_se), (rating, icon, description) in effectiveness_levels.items():
            if min_se <= total_se < max_se:
                st.markdown(f"### Overall Shielding Rating: {rating} {icon}")
                st.info(description)
                break
    
    # Create a container with columns for the analysis sections
    analysis_container = st.container()
    
    # Detailed results at specific frequency
    col1, col2 = analysis_container.columns([1, 1])
    
    with col1:
        st.subheader("Analysis at Specific Frequency")
        specific_freq = st.number_input(
            "Select frequency (Hz)",
            min_value=float(freq_min),
            max_value=float(freq_max),
            value=float((freq_min + freq_max)/2),
            format="%e",
            help="Choose a specific frequency to analyze shielding performance"
        )
        
        # Find closest frequency in our results
        freq_idx = np.abs(freq_range - specific_freq).argmin()
        
        # Use a more compact display for results
        results_df = pd.DataFrame({
            "Component": ["Reflection Loss", "Absorption Loss", "Multiple Reflection Loss", "Total Shielding Effectiveness"],
            "Value (dB)": [
                f"{physics_results['reflection_loss'][freq_idx]:.2f}",
                f"{physics_results['absorption_loss'][freq_idx]:.2f}",
                f"{physics_results['multiple_reflection_loss'][freq_idx]:.2f}",
                f"{physics_results['total_se'][freq_idx]:.2f}"
            ]
        })
        st.table(results_df)
    
    with col2:
        st.subheader("Performance Breakdown")
        st.markdown("""
        **Interpretation Guide:**
        
        - **< 30 dB**: Basic shielding
        - **30-60 dB**: Professional grade
        - **60-90 dB**: Military grade
        - **> 90 dB**: Ultra-high performance
        
        **Primary Shielding Mechanism:**
        """)
        
        # Determine primary shielding mechanism
        mechanisms = {
            "Reflection": physics_results['reflection_loss'][freq_idx],
            "Absorption": physics_results['absorption_loss'][freq_idx],
            "Multiple Reflection": physics_results['multiple_reflection_loss'][freq_idx]
        }
        primary_mechanism = max(mechanisms.items(), key=lambda x: abs(x[1]))
        
        st.success(f"This material primarily shields through **{primary_mechanism[0]}** " +
                  f"({primary_mechanism[1]:.1f} dB)")
    
    # Export and comparison buttons in a more compact layout
    button_container = st.container()
    btn_col1, btn_col2 = button_container.columns([1, 1])
    
    with btn_col1:
        if st.button("Export Results", use_container_width=True):
            # Create DataFrame with all results
            export_df = pd.DataFrame({
                "Frequency (Hz)": freq_range,
                "Reflection Loss (dB)": physics_results["reflection_loss"],
                "Absorption Loss (dB)": physics_results["absorption_loss"],
                "Multiple Reflection Loss (dB)": physics_results["multiple_reflection_loss"],
                "Total SE (dB)": physics_results["total_se"]
            })
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="emi_shielding_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with btn_col2:
        if st.button("Add to Comparison", use_container_width=True):
            material_name = "+".join([f"{k}{v*100:.0f}%" for k, v in composition.items()])
            st.session_state['comparison_data'].append({
                "name": material_name,
                "frequency": freq_range,
                "total_se": physics_results["total_se"],
                "composition": composition.copy(),
                "thickness": thickness,
                "active": True
            })
            st.success(f"Added {material_name} to comparison")

    with tab2:
        # Analysis Results Tab
        st.header("Shielding Performance Analysis")
        st.write("""
        View detailed analysis of your material's EMI shielding performance across different frequencies.
        The results are calculated using physics-based models and can be exported for further analysis.
        """)
        
        if not st.session_state['comparison_data']:
            st.info("Add materials from the Main Analysis tab to compare them here.")
        else:
            # Display comparison controls
            st.subheader("Saved Materials")
            for i, data in enumerate(st.session_state['comparison_data']):
                if i is None or not isinstance(i, int):
                    continue
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    checkbox_key = f"compare_{str(i)}"
                    st.session_state['comparison_data'][i]['active'] = st.checkbox(
                        str(data.get('name', '')) if data.get('name', '') is not None else "",
                        value=data['active'],
                        key=checkbox_key
                    )
                with col2:
                    if st.button("Load", key=f"load_{i}"):
                        st.session_state['example_composition'] = data['composition']
                        st.session_state['example_thickness'] = data['thickness']
                with col3:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state['comparison_data'].pop(i)
                        st.rerun()
            
            # Display comparison plot
            active_data = [d for d in st.session_state['comparison_data'] if d['active']]
            if active_data:
                fig = create_comparison_plot(active_data)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Comparison Tab with improved space utilization
        st.header("Material Comparison")
        st.write("""
        Compare different material combinations to find the optimal solution for your EMI shielding needs.
        Add materials from the Analysis Results tab to see how they perform against each other.
        """)
        
        if not st.session_state['comparison_data']:
            st.info("No materials added for comparison yet. Use the 'Add to Comparison' button in the Analysis Results tab to add materials.")
        else:
            # Create a more space-efficient layout for comparison controls
            st.subheader("Saved Materials")
            
            # Use a grid layout for material comparison controls
            num_materials = len(st.session_state['comparison_data'])
            cols_per_row = 2  # Display 2 materials per row for better space utilization
            
            # Create rows of materials
            for i in range(0, num_materials, cols_per_row):
                cols = st.columns(cols_per_row)
                
                # Add materials to each column in the row
                for j in range(cols_per_row):
                    idx = i + j
                    if idx is None or not isinstance(idx, int):
                        continue
                    if idx < num_materials:
                        data = st.session_state['comparison_data'][idx]
                        with cols[j]:
                            st.markdown(f"**{data['name']}**")
                            col1, col2, col3 = st.columns([3, 2, 2])
                            with col1:
                                checkbox_key = f"compare_tab3_{str(idx)}"
                                st.session_state['comparison_data'][idx]['active'] = st.checkbox(
                                    "Active",
                                    value=data['active'],
                                    key=checkbox_key
                                )
                            with col2:
                                if st.button("Load", key=f"load_{idx}", use_container_width=True):
                                    st.session_state['example_composition'] = data['composition']
                                    st.session_state['example_thickness'] = data['thickness']
                                    st.rerun()
                            with col3:
                                if st.button("Remove", key=f"remove_{idx}", use_container_width=True):
                                    st.session_state['comparison_data'].pop(idx)
                                    st.rerun()
            
            # Display comparison plot
            active_data = [d for d in st.session_state['comparison_data'] if d['active']]
            if active_data:
                st.subheader("Comparison Chart")
                fig = create_comparison_plot(active_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add comparison analysis in a more compact layout
                if len(active_data) > 1:
                    st.subheader("Comparative Analysis")
                    
                    # Define frequency ranges for analysis
                    ranges = [
                        ("Low Frequency (< 100 kHz)", 0, 1e5),
                        ("Medium Frequency (100 kHz - 10 MHz)", 1e5, 1e7),
                        ("High Frequency (10 MHz - 1 GHz)", 1e7, 1e9),
                        ("Very High Frequency (> 1 GHz)", 1e9, float('inf'))
                    ]
                    
                    # Create a grid layout for frequency range analysis
                    range_cols = st.columns(2)
                    for i, (range_name, range_min, range_max) in enumerate(ranges):
                        # Find indices within this range
                        indices = [i for i, f in enumerate(active_data[0]["frequency"]) 
                                if range_min <= f <= range_max]
                        
                        if indices:
                            # Calculate average SE in this range for each material
                            avg_se_by_material = []
                            for data in active_data:
                                avg_se = sum(data["total_se"][i] for i in indices) / len(indices)
                                avg_se_by_material.append((data["name"], avg_se))
                            
                            # Find best material
                            best_material = max(avg_se_by_material, key=lambda x: x[1])
                            
                            # Display in alternating columns for better space utilization
                            with range_cols[i % 2]:
                                st.markdown(f"**{range_name}**: {best_material[0]} ({best_material[1]:.1f} dB)")
        
    with tab4:
        # Technical Reference Tab
        st.header("Technical Reference")
        st.write("""
        Learn about the science behind EMI shielding, material properties, and best practices for effective shielding design.
        This section provides technical information to help you understand and optimize your shielding solutions.
        """)
        
        # Add technical details
        add_technical_details()
        
        # Add EMI shielding applications
        with st.expander("EMI Shielding Applications"):
            st.markdown("""
            ### Common Applications
            
            #### Electronics & Communications
            - **Consumer Electronics**: Smartphones, laptops, tablets, TVs
            - **Medical Devices**: MRI machines, patient monitors, implantable devices
            - **Telecommunications**: Base stations, routers, network equipment
            - **Automotive Electronics**: Engine control units, infotainment systems
            
            #### Industrial & Military
            - **Industrial Equipment**: Motors, generators, power supplies
            - **Military Systems**: Radar equipment, communication systems
            - **Aerospace**: Aircraft avionics, satellite components
            - **Scientific Instruments**: Sensitive measurement equipment
            
            ### Application-Specific Requirements
            
            | Application | Frequency Range | Key Requirements | Recommended Materials |
            |-------------|-----------------|------------------|----------------------|
            | Medical Devices | 10 kHz - 3 GHz | High reliability, biocompatibility | Copper, aluminum, specialized alloys |
            | Automotive | 100 kHz - 2.4 GHz | Temperature resistance, durability | Aluminum alloys, steel alloys |
            | Consumer Electronics | 800 MHz - 5 GHz | Lightweight, cost-effective | Conductive plastics, thin metal films |
            | Military/Aerospace | 100 MHz - 40 GHz | High performance, reliability | Specialized composites, mu-metal |
            """)
        
        # Add material selection guide
        with st.expander("Material Selection Guide"):
            st.markdown("""
            ### Choosing the Right Material
            
            #### Key Properties to Consider
            
            1. **Electrical Conductivity**
               - Higher conductivity → Better reflection of electric fields
               - Copper and silver have excellent conductivity
               - Aluminum offers good conductivity at lower cost
            
            2. **Magnetic Permeability**
               - Higher permeability → Better absorption of magnetic fields
               - Nickel-iron alloys (mu-metal) have high permeability
               - Important for low-frequency magnetic shielding
            
            3. **Thickness**
               - Thicker shields → Better absorption
               - Diminishing returns beyond skin depth
               - Consider weight and space constraints
            
            4. **Frequency Range**
               - Low frequencies (< 100 kHz): Magnetic permeability is critical
               - High frequencies (> 1 MHz): Conductivity becomes dominant
               - Very high frequencies: Multiple thin layers may outperform single thick layer
            
            #### Material Comparison
            
            | Material | Conductivity (S/m) | Rel. Permeability | Best For | Limitations |
            |----------|-------------------|-------------------|----------|-------------|
            | Copper | 5.8 × 10⁷ | 1.0 | High-frequency electric fields | Heavy, expensive, corrosion |
            | Aluminum | 3.5 × 10⁷ | 1.0 | General purpose, lightweight | Poor magnetic shielding |
            | Steel | 1.0 × 10⁷ | 100-1000 | Cost-effective, structural | Weight, corrosion |
            | Mu-Metal | 1.6 × 10⁶ | 2,000-100,000 | Low-frequency magnetic fields | Expensive, mechanical sensitivity |
            | Conductive Polymers | 10² - 10⁵ | 1.0 | Lightweight, flexible | Limited effectiveness |
            """)
        
        # Add references section
        with st.expander("References & Further Reading"):
            st.markdown("""
            ### Scientific References
            
            1. **EMI Shielding Theory**
               - Paul, C. R. (2006). Introduction to Electromagnetic Compatibility
               - Schulz, R. B. et al. (1988). Shielding Theory and Practice
            
            2. **Material Properties**
               - ASM Handbook Volume 2: Properties and Selection of Nonferrous Alloys
               - IEEE Standard 299-2006 for Measuring Shielding Effectiveness
            
            3. **Advanced Topics**
               - Near-field & Far-field Effects in EMI Shielding
               - Frequency-dependent Material Properties
               - Multi-layer Shielding Design
            
            ### Online Resources
            
            - [IEEE EMC Society](https://www.emcs.org/)
            - [Materials Project Database](https://materialsproject.org/)
            - [NIST Materials Data Repository](https://materials.nist.gov/)
            
            ### Standards & Guidelines
            
            - MIL-STD-461G: EMI/EMC Testing
            - IEC 61000: Electromagnetic Compatibility
            - ASTM D4935: Shielding Effectiveness Testing
            """)
        
        # Advanced Analysis Tab (SUPER MODE)
        if super_mode:
            st.header("🚀 Advanced Analysis")
        
        if not super_mode:
            st.info("Enable SUPER MODE in the sidebar to access advanced controls and analysis features.")
            return
        
        # Advanced Material Controls
        st.subheader("Advanced Material Controls")
        
        with st.expander("Direct Property Manipulation"):
            st.markdown("""
            Override calculated material properties with custom values.
            Use with caution - these values will bypass the standard mixing rules.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state['super_mode_values']['custom_conductivity'] = st.number_input(
                    "Electrical Conductivity (S/m)",
                    min_value=1e2,
                    max_value=1e8,
                    value=st.session_state['super_mode_values']['custom_conductivity'],
                    format="%e",
                    help="Range: 1e2 - 1e8 S/m"
                )
                st.session_state['super_mode_values']['custom_permittivity'] = st.number_input(
                    "Relative Permittivity",
                    min_value=1.0,
                    max_value=100.0,
                    value=st.session_state['super_mode_values']['custom_permittivity'],
                    help="Range: 1 - 100"
                )
            
            with col2:
                st.session_state['super_mode_values']['custom_permeability'] = st.number_input(
                    "Relative Permeability",
                    min_value=0.1,
                    max_value=10000.0,
                    value=st.session_state['super_mode_values']['custom_permeability'],
                    help="Range: 0.1 - 10,000"
                )
                st.session_state['super_mode_values']['loss_tangent'] = st.number_input(
                    "Loss Tangent",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state['super_mode_values']['loss_tangent'],
                    help="Range: 0 - 1"
                )
                
            # Apply button to update material properties
            if st.button("Apply Custom Properties"):
                st.success("Custom material properties applied!")
                st.rerun()
        
        with st.expander("Environmental Effects"):
            st.markdown("Simulate environmental conditions and their effects on shielding.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state['super_mode_values']['temperature'] = st.slider(
                    "Temperature (°C)",
                    min_value=-50,
                    max_value=250,
                    value=st.session_state['super_mode_values']['temperature'],
                    help="Operating temperature affects material properties"
                )
                st.session_state['super_mode_values']['aging_years'] = st.slider(
                    "Material Age (years)",
                    min_value=0,
                    max_value=50,
                    value=st.session_state['super_mode_values']['aging_years'],
                    help="Simulate material degradation over time"
                )
            
            with col2:
                st.session_state['super_mode_values']['humidity'] = st.slider(
                    "Relative Humidity (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state['super_mode_values']['humidity'],
                    help="Humidity can affect material performance"
                )
                st.session_state['super_mode_values']['mechanical_stress'] = st.slider(
                    "Mechanical Stress (%)",
                    min_value=0,
                    max_value=100,
                    value=st.session_state['super_mode_values']['mechanical_stress'],
                    help="Simulate mechanical stress effects"
                )
                
            # Apply button to update environmental effects
            if st.button("Apply Environmental Effects"):
                st.success("Environmental effects applied!")
                st.rerun()
        
        # Advanced Physics Parameters
        st.subheader("Advanced Physics Parameters")
        
        with st.expander("Field Configuration"):
            st.markdown("Configure advanced electromagnetic field parameters.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state['super_mode_values']['incident_angle'] = st.slider(
                    "Incident Angle (degrees)",
                    min_value=0,
                    max_value=89,
                    value=st.session_state['super_mode_values']['incident_angle'],
                    help="Angle of incident electromagnetic wave"
                )
                st.session_state['super_mode_values']['field_type'] = st.radio(
                    "Field Type",
                    ["Near-field", "Far-field"],
                    index=0 if st.session_state['super_mode_values']['field_type'] == "Near-field" else 1,
                    help="Select field region for analysis"
                )
            
            with col2:
                st.session_state['super_mode_values']['polarization'] = st.radio(
                    "Wave Polarization",
                    ["TE Mode", "TM Mode"],
                    index=0 if st.session_state['super_mode_values']['polarization'] == "TE Mode" else 1,
                    help="Transverse Electric or Transverse Magnetic"
                )
                st.session_state['super_mode_values']['field_strength'] = st.number_input(
                    "Field Strength (V/m)",
                    min_value=1.0,
                    max_value=1000.0,
                    value=st.session_state['super_mode_values']['field_strength'],
                    help="Incident field strength"
                )
                
            # Apply button to update field configuration
            if st.button("Apply Field Configuration"):
                st.success("Field configuration applied!")
                st.rerun()
        
        with st.expander("Multi-layer Configuration"):
            st.markdown("Configure multi-layer shielding (up to 5 layers)")
            
            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=5,
                value=1,
                help="Configure multiple material layers"
            )
            
            for i in range(int(num_layers)):
                st.markdown(f"#### Layer {i+1}")
                col1, col2 = st.columns(2)
                with col1:
                    st.selectbox(
                        f"Material {i+1}",
                        ["Custom"] + list(MATERIAL_PRESETS.keys()),
                        key=f"layer_material_{i}"
                    )
                    st.number_input(
                        f"Thickness {i+1} (mm)",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        key=f"layer_thickness_{i}"
                    )
                with col2:
                    st.checkbox(
                        "Enable Interface Effects",
                        key=f"layer_interface_{i}",
                        help="Consider interface phenomena between layers"
                    )
        
        # Advanced Visualization Options
        st.subheader("Advanced Visualization Options")
        
        with st.expander("3D Visualization"):
            st.markdown("Configure 3D visualization of shielding effectiveness.")
            
            plot_type = st.selectbox(
                "Plot Type",
                ["Field Penetration", "Skin Depth vs Frequency", "Material Property Space"],
                help="Select type of 3D visualization"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                colormap = st.selectbox(
                    "Color Scheme",
                    ["Viridis", "Plasma", "Inferno", "Magma"],
                    help="Select color scheme for 3D plot"
                )
                show_contours = st.checkbox(
                    "Show Contours",
                    value=True,
                    help="Display contour lines on plot"
                )
            
            with col2:
                animation = st.checkbox(
                    "Enable Animation",
                    value=False,
                    help="Animate field penetration over time"
                )
                interactive = st.checkbox(
                    "Interactive Mode",
                    value=True,
                    help="Enable interactive plot controls"
                )
        
        with st.expander("Advanced Plots"):
            st.markdown("Additional visualization options for detailed analysis.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox(
                    "Show Smith Chart",
                    value=False,
                    help="Display impedance matching on Smith chart"
                )
                st.checkbox(
                    "Show Polar Plot",
                    value=False,
                    help="Display angular dependency"
                )
            
            with col2:
                st.checkbox(
                    "Show Phase Plot",
                    value=False,
                    help="Display phase relationships"
                )
                st.checkbox(
                    "Show Error Bounds",
                    value=False,
                    help="Display uncertainty ranges"
                )
        
        # ML Model Controls
        st.subheader("Machine Learning Configuration")
        
        with st.expander("Model Architecture"):
            st.markdown("Configure neural network architecture and training parameters.")
            
            col1, col2 = st.columns(2)
            with col1:
                num_layers = st.slider(
                    "Hidden Layers",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Number of hidden layers in the neural network"
                )
                activation = st.selectbox(
                    "Activation Function",
                    ["ReLU", "Tanh", "Sigmoid"],
                    help="Activation function for hidden layers"
                )
            
            with col2:
                layer_width = st.select_slider(
                    "Layer Width",
                    options=[32, 64, 128, 256, 512],
                    value=128,
                    help="Number of neurons per hidden layer"
                )
                
                dropout = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    help="Dropout rate for regularization"
                )
        
        with st.expander("📊 Uncertainty Quantification"):
            st.markdown("Configure uncertainty estimation and sensitivity analysis.")
            
            col1, col2 = st.columns(2)
            with col1:
                uncertainty_method = st.selectbox(
                    "Uncertainty Method",
                    ["Monte Carlo", "Bayesian", "Ensemble"],
                    help="Method for uncertainty estimation"
                )
                num_samples = st.number_input(
                    "Number of Samples",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    help="Number of samples for uncertainty estimation"
                )
            
            with col2:
                confidence_level = st.slider(
                    "Confidence Level (%)",
                    min_value=80,
                    max_value=99,
                    value=95,
                    help="Confidence level for uncertainty bounds"
                )
                show_uncertainty = st.checkbox(
                    "Show Uncertainty Bands",
                    value=True,
                    help="Display uncertainty ranges in plots"
                )
        
        # Add example combinations
        add_example_combinations("tab4")
        
        # Add technical details
        add_technical_details()
        
        # Add references section
        with st.expander("📚 References & Further Reading"):
            st.markdown("""
            ### Scientific References
            
            1. **EMI Shielding Theory**
               - Paul, C. R. (2006). Introduction to Electromagnetic Compatibility
               - Schulz, R. B. et al. (1988). Shielding Theory and Practice
            
            2. **Material Properties**
               - ASM Handbook Volume 2: Properties and Selection of Nonferrous Alloys
               - IEEE Standard 299-2006 for Measuring Shielding Effectiveness
            
            3. **Advanced Topics**
               - Near-field & Far-field Effects in EMI Shielding
               - Frequency-dependent Material Properties
               - Multi-layer Shielding Design
            
            ### Online Resources
            
            - [IEEE EMC Society](https://www.emcs.org/)
            - [Materials Project Database](https://materialsproject.org/)
            - [NIST Materials Data Repository](https://materials.nist.gov/)
            
            ### Standards & Guidelines
            
            - MIL-STD-461G: EMI/EMC Testing
            - IEC 61000: Electromagnetic Compatibility
            - ASTM D4935: Shielding Effectiveness Testing
            """)

if __name__ == "__main__":
    main()
