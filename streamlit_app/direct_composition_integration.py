"""
Integration code for Direct Composition mode.
This file shows how to integrate Direct Composition into the main app.
"""

def render_direct_composition_section(st, material_db):
    """Render the Direct Composition section in the main app."""
    
    # Import EMI calculator at the top of function
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.physics.emi_calculations import emi_calculator
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    
    # DIRECT COMPOSITION section
    st.markdown("""
    <div style="margin: var(--space-6) 0 var(--space-4) 0;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 var(--space-4) 0;
            text-align: center;
        ">📊 Direct Composition Input</h3>
        <p style="text-align: center; color: var(--text-secondary); margin-top: var(--space-2);">
            Enter your material composition by weight percentage
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add element interface
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        available_elements = [elem for elem in material_db.periodic_table.keys() 
                            if elem not in st.session_state.direct_composition]
        new_element = st.selectbox(
            "Select Element",
            [""] + sorted(available_elements),
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
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)  # Spacer
        if st.button("➕ Add", key="add_element_btn", use_container_width=True):
            if new_element and new_percentage > 0:
                st.session_state.direct_composition[new_element] = new_percentage
                st.rerun()
    
    # Display current composition
    if st.session_state.direct_composition:
        # Calculate total
        total = sum(st.session_state.direct_composition.values())
        
        # Show total with color coding
        color = "green" if abs(total - 100.0) < 0.01 else "red"
        st.markdown(f"""
        <div style="
            background: var(--bg-card);
            border: 2px solid {'var(--accent-green)' if color == 'green' else 'var(--accent-red)'};
            border-radius: var(--radius-md);
            padding: var(--space-3);
            margin: var(--space-4) 0;
            text-align: center;
        ">
            <h4 style="margin: 0; color: var(--text-primary);">Total: 
                <span style="color: {'var(--accent-green)' if color == 'green' else 'var(--accent-red)'}; font-weight: 700;">
                    {total:.1f}%
                </span>
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Composition table
        st.markdown("""
        <div class="category-header">
            <div class="category-indicator" style="background: var(--accent-blue);"></div>
            <span>Current Composition</span>
        </div>
        """, unsafe_allow_html=True)
        
        for element, percentage in list(st.session_state.direct_composition.items()):
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                elem_data = material_db.get_material(element)
                elem_name = elem_data.get('name', element) if elem_data else element
                st.markdown(f"**{element}** - {elem_name}")
            
            with col2:
                new_pct = st.number_input(
                    f"{element} percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=percentage,
                    step=0.1,
                    key=f"pct_{element}",
                    label_visibility="collapsed"
                )
                if new_pct != percentage:
                    st.session_state.direct_composition[element] = new_pct
            
            with col3:
                if st.button("🗑️", key=f"del_direct_{element}"):
                    del st.session_state.direct_composition[element]
                    st.rerun()
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Normalize button
            if abs(total - 100.0) > 0.01 and total > 0:
                if st.button("⚖️ Normalize to 100%", use_container_width=True):
                    # Normalize composition
                    for elem in st.session_state.direct_composition:
                        st.session_state.direct_composition[elem] *= (100.0 / total)
                    st.rerun()
        
        with col2:
            # Clear button in middle
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.direct_composition = {}
                st.rerun()
        
        # Instead of "Add to Reaction", add shield parameters and calculate
        if abs(total - 100.0) < 0.01:
            # Shield parameters section
            st.markdown("""
            <div style="margin: var(--space-6) 0 var(--space-4) 0;">
                <h3 style="
                    font-size: var(--font-xl);
                    font-weight: 600;
                    color: var(--text-primary);
                    margin: 0 0 var(--space-3) 0;
                    text-align: center;
                ">⚙️ Shield Parameters</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                frequency = st.number_input(
                    "Frequency (MHz)",
                    min_value=0.1,
                    max_value=10000.0,
                    value=1000.0,
                    step=10.0,
                    key="direct_freq"
                )
            
            with col2:
                thickness = st.number_input(
                    "Thickness (mm)",
                    min_value=0.01,
                    max_value=100.0,
                    value=1.0,
                    step=0.1,
                    key="direct_thick"
                )
            
            st.markdown("### 🧮 Calculate")
            
            if st.button("Calculate EMI Shielding", type="primary", use_container_width=True, key="calc_direct"):
                # Store current composition and parameters
                st.session_state.last_direct_composition = st.session_state.direct_composition.copy()
                st.session_state.last_direct_params = {'frequency': frequency, 'thickness': thickness}
                st.session_state.show_direct_results = True
                st.rerun()
    else:
        st.info("👆 Add elements and their percentages to create your material composition")
        
        # Show example
        with st.expander("📖 Example: 70% Iron, 30% Carbon"):
            st.write("""
            1. Select **Fe** from the dropdown
            2. Enter **70** as the percentage
            3. Click **Add**
            4. Select **C** from the dropdown
            5. Enter **30** as the percentage
            6. Click **Add**
            7. Set shield parameters and click **Calculate EMI Shielding**
            """)
    
    # Show results if calculation was performed
    if hasattr(st.session_state, 'show_direct_results') and st.session_state.show_direct_results:
        if hasattr(st.session_state, 'last_direct_composition') and st.session_state.last_direct_composition:
            show_direct_composition_results(st, material_db, emi_calculator, np, pd, go)


def show_direct_composition_results(st, material_db, emi_calculator, np, pd, go):
    """Show results for direct composition calculation."""
    composition = st.session_state.last_direct_composition
    params = st.session_state.last_direct_params
    frequency = params['frequency']
    thickness = params['thickness']
    
    # Calculate effective properties
    conductivity = 1e4  # Base conductivity
    permeability = 1.0
    density = 0
    
    # Adjust based on composition with more realistic values
    for elem, pct in composition.items():
        elem_data = material_db.get_material(elem)
        weight = pct / 100.0
        
        if elem_data:
            # Conductivity (weighted average)
            elem_cond = elem_data.get('electrical_conductivity', 1e6)
            conductivity += elem_cond * weight
            
            # Permeability (geometric mean)
            elem_perm = elem_data.get('relative_permeability', 1.0)
            elem_perm = max(elem_perm, 0.999)
            permeability *= elem_perm ** weight
            
            # Density (weighted average)
            elem_dens = elem_data.get('density', 5000)
            density += elem_dens * weight
    
    # Calculate EMI shielding
    try:
        result = emi_calculator.calculate_shielding_effectiveness(
            conductivity=conductivity,
            relative_permeability=permeability,
            relative_permittivity=1.0,
            thickness=thickness / 1000,  # Convert to meters
            frequency=frequency * 1e6,   # Convert to Hz
            include_confidence=True
        )
    except TypeError:
        # Fallback if include_confidence parameter doesn't exist
        result = emi_calculator.calculate_shielding_effectiveness(
            conductivity=conductivity,
            relative_permeability=permeability,
            relative_permittivity=1.0,
            thickness=thickness / 1000,  # Convert to meters
            frequency=frequency * 1e6    # Convert to Hz
        )
    
    # Display results section
    st.markdown("---")
    st.markdown("### 📊 Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total SE", f"{result['total_se']:.1f} dB")
    with col2:
        st.metric("Uncertainty", f"±{result.get('uncertainty_db', 0):.1f} dB")
    with col3:
        confidence_level = result.get('confidence_level', 'medium')
        confidence_emoji = {
            "high": "🟢 High",
            "medium": "🟡 Medium", 
            "low": "🟠 Low",
            "very_low": "🔴 Very Low"
        }
        st.metric("Confidence", confidence_emoji.get(confidence_level, "🟡 Medium"))
    
    # Step-by-step calculations
    st.markdown("---")
    
    # Step 1: Material Composition
    st.markdown("### 🔬 Step 1: Material Composition")
    st.markdown("<p style='margin-bottom: 20px;'>Analyzing the composition of your custom material blend...</p>", unsafe_allow_html=True)
    
    st.markdown("**Total Elemental Composition (by mass %):**")
    table_data = []
    for elem, pct in sorted(composition.items(), key=lambda x: x[1], reverse=True):
        table_data.append([elem, f"{pct:.2f}%"])
    
    df = pd.DataFrame(table_data, columns=['Element', 'Percentage'])
    st.dataframe(df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Step 2: Material Properties
    st.markdown("### 🧪 Step 2: Material Properties Calculation")
    st.markdown("<p style='margin-bottom: 20px;'>Calculating effective properties based on composition...</p>", unsafe_allow_html=True)
    
    st.markdown("**Property Calculations:**")
    
    # Show individual element properties
    for elem, pct in sorted(composition.items(), key=lambda x: x[1], reverse=True):
        if pct > 5:  # Only show significant components
            elem_data = material_db.get_material(elem)
            if elem_data:
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
    st.code(f"""
Effective Conductivity: σ_eff = {conductivity:.2e} S/m
Effective Permeability: μᵣ,eff = {permeability:.3f}
Effective Permittivity: εᵣ,eff = 1.000
Effective Density: ρ_eff = {density:.0f} kg/m³
""", language="text")
    
    st.markdown("---")
    
    # Step 3: EMI Shielding Physics
    st.markdown("### ⚡ Step 3: EMI Shielding Physics")
    st.markdown("<p style='margin-bottom: 20px;'>Electromagnetic Analysis:</p>", unsafe_allow_html=True)
    
    # Skin Depth Calculation
    st.markdown("#### Skin Depth Calculation")
    st.code(f"""
δ = √(2 / (ωμσ))
δ = {result['skin_depth']*1000:.3f} mm
""", language="text")
    st.markdown(f"*Penetration depth of electromagnetic waves*")
    
    # Intrinsic Impedance
    st.markdown("#### Intrinsic Impedance")
    omega = 2 * np.pi * frequency * 1e6
    eta = np.sqrt((omega * 4e-7 * np.pi * permeability) / (conductivity + 1j * omega * 8.854e-12))
    st.code(f"""
η = √(μ / ε*)
|η| = {abs(eta):.2f} Ω
""", language="text")
    st.markdown(f"*Material's resistance to electromagnetic wave propagation*")
    
    # Shielding Components
    st.markdown("#### Shielding Components")
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
        rating_msg = "Your material produces an excellent EMI shield."
        color = "#00ff00"
    elif result['total_se'] >= 60:
        rating = "Very Good"
        rating_msg = "Your material produces a very good EMI shield."
        color = "#66ff66"
    elif result['total_se'] >= 40:
        rating = "Good"
        rating_msg = "Your material produces a good EMI shield."
        color = "#ffff00"
    elif result['total_se'] >= 20:
        rating = "Fair"
        rating_msg = "Your material produces a fair EMI shield."
        color = "#ff9900"
    else:
        rating = "Poor"
        rating_msg = "Your material produces a poor EMI shield."
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
    with st.expander("📈 Detailed Analysis", expanded=False):
        # Create mechanism breakdown chart
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
            try:
                temp_result = emi_calculator.calculate_shielding_effectiveness(
                    conductivity=conductivity,
                    relative_permeability=permeability,
                    relative_permittivity=1.0,
                    thickness=thickness / 1000,
                    frequency=f * 1e6,
                    include_confidence=False
                )
            except TypeError:
                temp_result = emi_calculator.calculate_shielding_effectiveness(
                    conductivity=conductivity,
                    relative_permeability=permeability,
                    relative_permittivity=1.0,
                    thickness=thickness / 1000,
                    frequency=f * 1e6
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


# Integration instructions
INTEGRATION_GUIDE = """
To integrate Direct Composition mode into app.py:

1. After line 1140 (with main_container:), replace:
   # BUILD YOUR MOLECULE section
   
   With:
   if input_mode == "🧪 Molecular Builder":
       # BUILD YOUR MOLECULE section
       ... (existing molecular builder code)
   else:  # Direct Composition mode
       render_direct_composition_section(st, material_db)

2. Make sure all molecular builder code is indented inside the if block

3. Import this function at the top:
   from direct_composition_integration import render_direct_composition_section
"""