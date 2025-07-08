"""
EMI Shield Designer with Direct Composition Mode
This is a working demonstration of the percentage-based input feature.
"""

import streamlit as st
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.materials.material_properties import material_db
from src.physics.emi_calculations import emi_calculator

# Page config
st.set_page_config(
    page_title="🔬 EMI Shield Designer - Direct Composition",
    page_icon="⚛️",
    layout="wide"
)

# Initialize session state
if 'direct_composition' not in st.session_state:
    st.session_state.direct_composition = {}

# Header
st.markdown("""
<div style="text-align: center;">
    <h1>🔬 EMI Shield Designer</h1>
    <p style="font-size: 1.2em; color: #888;">with Direct Composition Mode</p>
</div>
""", unsafe_allow_html=True)

# How to use
with st.expander("📋 How To Use", expanded=True):
    st.write("""
    ### Direct Composition Mode (New Feature!)
    1. Enter material composition by weight percentage (e.g., 70% Fe, 30% C)
    2. No need to figure out molecular formulas
    3. Set shield parameters (thickness, frequency)
    4. Click Calculate to get EMI shielding effectiveness
    
    ### Example Compositions:
    - **Steel**: Fe: 98%, C: 2%
    - **Brass**: Cu: 70%, Zn: 30%
    - **Stainless Steel**: Fe: 70%, Cr: 18%, Ni: 10%, Mo: 2%
    """)

st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header(" Percentage Composition Input")
    st.markdown("*Enter your material composition by weight percentage*")
    
    # Add element interface
    c1, c2, c3 = st.columns([3, 2, 1])
    
    with c1:
        available_elements = [elem for elem in sorted(material_db.periodic_table.keys()) 
                            if elem not in st.session_state.direct_composition]
        new_element = st.selectbox(
            "Select Element",
            [""] + available_elements,
            key="new_element_select"
        )
    
    with c2:
        new_percentage = st.number_input(
            "Percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            key="new_percentage_input"
        )
    
    with c3:
        st.write("")  # Spacer
        if st.button("➕ Add", key="add_element_btn", use_container_width=True):
            if new_element and new_percentage > 0:
                st.session_state.direct_composition[new_element] = new_percentage
                st.rerun()
    
    # Display current composition
    if st.session_state.direct_composition:
        st.markdown("### Current Composition")
        
        # Calculate total
        total = sum(st.session_state.direct_composition.values())
        
        # Show total with color
        if abs(total - 100.0) < 0.01:
            st.success(f"✅ Total: {total:.1f}%")
        else:
            st.error(f"❌ Total: {total:.1f}% (must equal 100%)")
        
        # Composition table
        for element, percentage in list(st.session_state.direct_composition.items()):
            c1, c2, c3 = st.columns([2, 3, 1])
            
            with c1:
                elem_data = material_db.get_material(element)
                elem_name = elem_data.get('name', element) if elem_data else element
                st.write(f"**{element}** - {elem_name}")
            
            with c2:
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
                    st.session_state.direct_composition[element] = new_pct
            
            with c3:
                if st.button("🗑️", key=f"del_{element}"):
                    del st.session_state.direct_composition[element]
                    st.rerun()
        
        # Action buttons
        c1, c2 = st.columns(2)
        
        with c1:
            if abs(total - 100.0) > 0.01 and total > 0:
                if st.button("⚖️ Normalize to 100%", use_container_width=True):
                    for elem in st.session_state.direct_composition:
                        st.session_state.direct_composition[elem] *= (100.0 / total)
                    st.rerun()
        
        with c2:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.direct_composition = {}
                st.rerun()

with col2:
    st.header("⚙️ Shield Parameters")
    
    frequency = st.number_input(
        "Frequency (MHz)",
        min_value=0.1,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )
    
    thickness = st.number_input(
        "Thickness (mm)",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.1
    )
    
    st.markdown("### 🧮 Calculate")
    
    # Check if ready to calculate
    if st.session_state.direct_composition and abs(sum(st.session_state.direct_composition.values()) - 100.0) < 0.01:
        if st.button("Calculate EMI Shielding", type="primary", use_container_width=True):
            # Calculate effective properties (simplified for demo)
            composition = st.session_state.direct_composition
            
            # Initialize properties
            conductivity = 1e4  # Base conductivity
            permeability = 1.0
            
            # Adjust based on composition
            if 'Fe' in composition:
                conductivity += composition['Fe'] / 100 * 1e6
                permeability += composition['Fe'] / 100 * 200
            if 'Cu' in composition:
                conductivity += composition['Cu'] / 100 * 5e7
            if 'Al' in composition:
                conductivity += composition['Al'] / 100 * 3e7
            if 'Ni' in composition:
                conductivity += composition['Ni'] / 100 * 1e6
                permeability += composition['Ni'] / 100 * 50
            
            # Calculate EMI shielding
            result = emi_calculator.calculate_shielding_effectiveness(
                conductivity=conductivity,
                relative_permeability=permeability,
                relative_permittivity=1.0,
                thickness=thickness / 1000,  # Convert to meters
                frequency=frequency * 1e6,   # Convert to Hz
                include_confidence=True
            )
            
            # Display results
            st.markdown("### 📊 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total SE", f"{result['total_se']:.1f} dB")
            with col2:
                st.metric("Uncertainty", f"±{result['uncertainty_db']:.1f} dB")
            with col3:
                confidence_emoji = {
                    "high": "🟢 High",
                    "medium": "🟡 Medium", 
                    "low": "🟠 Low",
                    "very_low": "🔴 Very Low"
                }
                st.metric("Confidence", confidence_emoji.get(result['confidence_level'], ""))
            
            # Step-by-step calculations
            st.markdown("---")
            
            # Step 1: Material Composition
            st.markdown("### 🔬 Step 1: Material Composition")
            st.markdown("<p style='margin-bottom: 20px;'>Analyzing the composition of your custom material blend...</p>", unsafe_allow_html=True)
            
            # Create composition table
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Total Elemental Composition (by mass %):**")
                
                # Create a nice table
                table_data = []
                for elem, pct in sorted(composition.items(), key=lambda x: x[1], reverse=True):
                    elem_data = material_db.get_material(elem)
                    elem_name = elem_data.get('name', elem) if elem_data else elem
                    table_data.append([elem, f"{pct:.2f}%"])
                
                # Display as DataFrame for better formatting
                import pandas as pd
                df = pd.DataFrame(table_data, columns=['Element', 'Percentage'])
                st.dataframe(df, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            # Step 2: Material Properties
            st.markdown("### 🧮 Step 2: Material Properties Calculation")
            st.markdown("<p style='margin-bottom: 20px;'>Calculating effective properties based on composition...</p>", unsafe_allow_html=True)
            
            # Property Calculations box
            st.markdown("""
            <div style="background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <h4 style="margin-top: 0;">Property Calculations:</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Show individual element properties
            for elem, pct in sorted(composition.items(), key=lambda x: x[1], reverse=True):
                if pct > 5:  # Only show significant components
                    elem_data = material_db.get_material(elem)
                    if elem_data:
                        st.markdown(f"**{elem} ({pct:.1f}%):**")
                        elem_cond = elem_data.get('electrical_conductivity', 1e6)
                        elem_perm = elem_data.get('relative_permeability', 1.0)
                        elem_dens = elem_data.get('density', 5000)
                        st.code(f"""
{elem} ({pct:.1f}%):
σ = {elem_cond:.2e} S/m
μᵣ = {elem_perm:.1f}
εᵣ = 1.000
ρ = {elem_dens:.0f} kg/m³
""", language="text")
            
            # Composite Material Properties
            st.markdown("**Composite Material Properties:**")
            avg_density = sum(
                material_db.get_material(elem).get('density', 5000) * pct / 100
                for elem, pct in composition.items()
            )
            
            st.markdown("""
            <div style="background: #0066cc22; border: 1px solid #0066cc; border-radius: 8px; padding: 15px; margin: 20px 0;">
            </div>
            """, unsafe_allow_html=True)
            
            st.code(f"""
Effective Conductivity: σ_eff = {conductivity:.2e} S/m
Effective Permeability: μᵣ,eff = {permeability:.3f}
Effective Permittivity: εᵣ,eff = 1.000
Effective Density: ρ_eff = {avg_density:.0f} kg/m³
""", language="text")
            
            st.markdown("---")
            
            # Step 3: EMI Shielding Physics
            st.markdown("### ⚡ Step 3: EMI Shielding Physics")
            st.markdown("<p style='margin-bottom: 20px;'>Electromagnetic Analysis:</p>", unsafe_allow_html=True)
            
            # Skin Depth Calculation
            st.markdown("#### Skin Depth Calculation")
            st.markdown("""
            <div style="background: #1a1a1a; border-left: 3px solid #0066cc; padding: 15px; margin: 20px 0;">
            </div>
            """, unsafe_allow_html=True)
            
            st.code(f"""
δ = √(2 / (ωμσ))
δ = {result['skin_depth']*1000:.3f} mm
""", language="text")
            st.markdown(f"*Penetration depth of electromagnetic waves*")
            
            # Intrinsic Impedance
            st.markdown("#### Intrinsic Impedance")
            st.markdown("""
            <div style="background: #1a1a1a; border-left: 3px solid #0066cc; padding: 15px; margin: 20px 0;">
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate intrinsic impedance
            omega = 2 * np.pi * frequency * 1e6
            eta = np.sqrt((omega * 4e-7 * np.pi * permeability) / (conductivity + 1j * omega * 8.854e-12))
            
            st.code(f"""
η = √(μ / ε*)
|η| = {abs(eta):.2f} Ω
""", language="text")
            st.markdown(f"*Material's resistance to electromagnetic wave propagation*")
            
            # Shielding Components
            st.markdown("#### Shielding Components")
            st.markdown("""
            <div style="background: #1a1a1a; border-left: 3px solid #0066cc; padding: 15px; margin: 20px 0;">
            </div>
            """, unsafe_allow_html=True)
            
            st.code(f"""
Reflection Loss: {result['reflection_loss']:.1f} dB
Absorption Loss: {result['absorption_loss']:.1f} dB  
Multiple Reflection: {result['multiple_reflection_loss']:.1f} dB
""", language="text")
            
            st.markdown("---")
            
            # Final Results section
            st.markdown("### 🎯 Final Results")
            
            # Large result box
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #0066cc, #004499);
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
            ">
                <div style="color: #88ccff; font-size: 14px; margin-bottom: 10px;">Total Shielding Effectiveness</div>
                <div style="color: white; font-size: 48px; font-weight: bold;">{result['total_se']:.1f} dB</div>
                <div style="color: #88ccff; font-size: 14px; margin-top: 10px;">@ {frequency:.1f} MHz • {thickness:.1f} mm thickness</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Component breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px;">
                    <div style="color: #888; font-size: 12px;">REFLECTION LOSS</div>
                    <div style="color: white; font-size: 24px; font-weight: bold; margin-top: 10px;">{result['reflection_loss']:.1f} dB</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px;">
                    <div style="color: #888; font-size: 12px;">ABSORPTION LOSS</div>
                    <div style="color: white; font-size: 24px; font-weight: bold; margin-top: 10px;">{result['absorption_loss']:.1f} dB</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px;">
                    <div style="color: #888; font-size: 12px;">SKIN DEPTH</div>
                    <div style="color: white; font-size: 24px; font-weight: bold; margin-top: 10px;">{result['skin_depth']*1000:.3f} mm</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance rating box
            if result['total_se'] >= 90:
                rating = "Excellent"
                rating_msg = "Your reaction produces an excellent EMI shield."
                color = "#00ff00"
            elif result['total_se'] >= 60:
                rating = "Very Good"
                rating_msg = "Your reaction produces a very good EMI shield."
                color = "#66ff66"
            elif result['total_se'] >= 40:
                rating = "Good"
                rating_msg = "Your reaction produces a good EMI shield."
                color = "#ffff00"
            elif result['total_se'] >= 20:
                rating = "Fair"
                rating_msg = "Your reaction produces a fair EMI shield."
                color = "#ff9900"
            else:
                rating = "Poor"
                rating_msg = "Your reaction produces a poor EMI shield."
                color = "#ff0000"
            
            st.markdown(f"""
            <div style="
                background: {color}22;
                border: 1px solid {color};
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            ">
                <div style="font-weight: bold; color: {color};">Shield Performance Rating: {rating}</div>
                <div style="color: #ccc; margin-top: 5px;">{rating_msg}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Analysis expander
            with st.expander("📊 Detailed Analysis", expanded=False):
                # Create mechanism breakdown chart
                import plotly.graph_objects as go
                
                # Donut chart for shielding mechanisms
                labels = ['Absorption', 'Reflection', 'Multiple Reflection']
                values = [result['absorption_loss'], result['reflection_loss'], abs(result['multiple_reflection_loss'])]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.7,
                    marker_colors=['#8b5cf6', '#06b6d4', '#10b981'],
                    textposition='outside',
                    textinfo='percent'
                )])
                
                fig.update_layout(
                    title_text="Shielding Mechanism Breakdown",
                    title_x=0.5,
                    showlegend=True,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Frequency response
                st.markdown("**Frequency Response**")
                freq_range = np.logspace(-1, 4, 100)  # 0.1 MHz to 10 GHz
                se_values = []
                
                for f in freq_range:
                    temp_result = emi_calculator.calculate_shielding_effectiveness(
                        conductivity=conductivity,
                        relative_permeability=permeability,
                        relative_permittivity=1.0,
                        thickness=thickness / 1000,
                        frequency=f * 1e6,
                        include_confidence=False
                    )
                    se_values.append(temp_result['total_se'])
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=freq_range,
                    y=se_values,
                    mode='lines',
                    line=dict(color='#06b6d4', width=3),
                    name='Shielding Effectiveness'
                ))
                
                # Add current point
                fig2.add_trace(go.Scatter(
                    x=[frequency],
                    y=[result['total_se']],
                    mode='markers',
                    marker=dict(color='#ff0066', size=10),
                    name='Current Setting'
                ))
                
                fig2.update_xaxes(
                    type="log",
                    title_text="Frequency (MHz)",
                    gridcolor='#333',
                    zeroline=False
                )
                
                fig2.update_yaxes(
                    title_text="Shielding Effectiveness (dB)",
                    gridcolor='#333',
                    zeroline=False
                )
                
                fig2.update_layout(
                    title_text="Frequency Response",
                    title_x=0.5,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=True
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👆 Enter a composition that totals 100% to calculate")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p><strong>Direct Composition Mode</strong> - Enter materials by weight percentage</p>
    <p>Example: 70% Fe, 30% C for carbon steel</p>
</div>
""", unsafe_allow_html=True)