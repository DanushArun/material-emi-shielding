"""
Enhanced EMI Shield Designer with Direct Composition mode.
This is a demonstration of the Direct Composition feature integration.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="🔬 EMI Shield Designer - Enhanced",
    page_icon="⚛️",
    layout="wide"
)

st.title("🔬 EMI Shield Designer - Direct Composition Demo")

# Import the direct composition manager
from direct_composition import DirectCompositionManager, render_direct_composition_ui
from src.materials.material_properties import material_db

# Initialize session state
if 'direct_composition' not in st.session_state:
    st.session_state.direct_composition = {}

# Mode selector
input_mode = st.radio(
    "Select Input Method:",
    [" Molecular Builder", "Direct Composition"],
    horizontal=True,
    help="Molecular: Build molecules from elements | Direct: Enter percentages directly"
)

if input_mode == "📊 Direct Composition":
    # Render the direct composition interface
    dc_manager = render_direct_composition_ui(material_db)
    
    # Show current composition
    if st.session_state.direct_composition:
        st.success(f"Current composition: {st.session_state.direct_composition}")
        
        # Example calculation button
        if dc_manager.is_valid():
            if st.button("Calculate EMI Shielding", type="primary"):
                composition = dc_manager.get_composition()
                st.write("### Composition for EMI Calculation:")
                for element, percentage in composition.items():
                    st.write(f"- {element}: {percentage:.1f}%")
                
                # This is where EMI calculations would be performed
                st.info("EMI calculation would be performed here with the direct composition")
else:
    st.info("Molecular Builder mode - Please use the main app.py for full functionality")

# Display advantages of Direct Composition mode
with st.expander("📖 Advantages of Direct Composition Mode"):
    st.write("""
    ### Why Use Direct Composition?
    
    1. **Industry Standard**: Enter compositions exactly as specified in material datasheets
    2. **No Formula Needed**: No need to figure out molecular formulas
    3. **Precise Control**: Specify exact percentages for each element
    4. **Complex Alloys**: Easily handle multi-element compositions
    5. **Real-world Applications**: Match actual material specifications
    
    ### Example Use Cases:
    - **Steel 1018**: Fe: 98.2%, C: 1.8%
    - **Brass 70/30**: Cu: 70%, Zn: 30%
    - **Stainless 316**: Fe: 68%, Cr: 17%, Ni: 12%, Mo: 2.5%, Mn: 0.5%
    """)