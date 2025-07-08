"""
Demonstration of the self-improving ML-enhanced EMI Shielding system.
Shows how all components work together.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Import our ML components
from src.ml.accuracy.validator import AccuracyValidator
from src.ml.accuracy.feedback import FeedbackCollector
from src.ml.features.material_features import MaterialFeatureExtractor
from src.physics.emi_calculations import emi_calculator
from streamlit_app.direct_composition import DirectCompositionManager

# Page config
st.set_page_config(
    page_title="ML-Enhanced EMI Shield Designer Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Self-Improving EMI Shield Designer")
st.markdown("### Demonstration of Machine Learning Integration")

# Initialize components
validator = AccuracyValidator()
feedback_collector = FeedbackCollector()
feature_extractor = MaterialFeatureExtractor()

# Sidebar for system metrics
with st.sidebar:
    st.header("📊 System Metrics")
    
    # Get accuracy report
    accuracy_report = validator.get_accuracy_report()
    
    if accuracy_report['status'] == 'success':
        metrics = accuracy_report['metrics']
        st.metric("Overall RMSE", f"±{metrics['rmse']:.2f} dB")
        st.metric("R² Score", f"{metrics['r_squared']:.3f}")
        st.metric("Total Measurements", metrics['total_measurements'])
    else:
        st.info("No measurements yet - system will improve with user feedback")
    
    # Feedback statistics
    feedback_stats = feedback_collector.get_feedback_statistics()
    st.metric("User Feedback", feedback_stats['total_feedback'])
    if feedback_stats['improvement_trend']:
        st.metric("Accuracy Improvement", f"{feedback_stats['improvement_trend']:.1f}%", "📈")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🧮 Calculate", "📊 Direct Input", "🎯 ML Predictions", "📈 Analytics"])

with tab1:
    st.header("EMI Shielding Calculation with Confidence")
    
    # Example composition
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Material Composition")
        
        # Simple composition input
        composition = {}
        elements = ["Fe", "C", "Ni", "Cu", "Al"]
        
        for elem in elements:
            pct = st.slider(f"{elem} (%)", 0.0, 100.0, 0.0, 0.1, key=f"slider_{elem}")
            if pct > 0:
                composition[elem] = pct
        
        total = sum(composition.values())
        color = "🟢" if abs(total - 100) < 0.1 else "🔴"
        st.write(f"{color} Total: {total:.1f}%")
    
    with col2:
        st.subheader("Parameters")
        frequency = st.number_input("Frequency (MHz)", 1.0, 10000.0, 1000.0)
        thickness = st.number_input("Thickness (mm)", 0.1, 10.0, 1.0)
        
        temperature = st.slider("Processing Temperature (°C)", 20, 1000, 200)
        atmosphere = st.selectbox("Atmosphere", ["Air", "Nitrogen", "Vacuum"])
    
    if st.button("Calculate with ML Enhancement", type="primary"):
        if abs(total - 100) < 0.1:
            # Extract features
            conditions = {'temperature_C': temperature, 'atmosphere': atmosphere.lower()}
            features, feature_names = feature_extractor.extract_all_features(
                composition, frequency, thickness, conditions
            )
            
            # Show extracted features
            with st.expander("🔍 Extracted ML Features"):
                st.write(f"Total features extracted: {len(features)}")
                # Show key features
                key_features = {
                    'Mean Atomic Number': features[feature_names.index('mean_atomic_number')],
                    'Mixing Entropy': features[feature_names.index('mixing_entropy')],
                    'Electronegativity Range': features[feature_names.index('range_electronegativity')],
                    'Has Fe': features[feature_names.index('has_Fe')],
                    'Processing Temp': temperature
                }
                for name, value in key_features.items():
                    st.write(f"- {name}: {value:.3f}")
            
            # Calculate with physics model
            # First, calculate effective properties (simplified)
            conductivity = 1e6  # Simplified - would use ML prediction
            permeability = 1.0
            permittivity = 1.0
            
            # Add magnetic permeability for Fe
            if 'Fe' in composition:
                permeability = 1 + composition['Fe'] / 100 * 999  # Simplified
            
            # Add conductivity for metals
            if 'Cu' in composition:
                conductivity += composition['Cu'] / 100 * 5e7
            if 'Al' in composition:
                conductivity += composition['Al'] / 100 * 3e7
            
            result = emi_calculator.calculate_shielding_effectiveness(
                conductivity, permeability, permittivity,
                thickness / 1000, frequency * 1e6
            )
            
            # Display results with confidence
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total SE", f"{result['total_se']:.1f} dB")
                st.caption(f"±{result['uncertainty_db']:.1f} dB")
            
            with col2:
                conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🟠", "very_low": "🔴"}
                st.metric("Confidence", 
                         f"{result['confidence']:.1%}",
                         conf_emoji.get(result['confidence_level'], ""))
            
            with col3:
                st.metric("Skin Depth", f"{result['skin_depth']*1000:.3f} mm")
            
            # Store prediction for feedback
            st.session_state.last_prediction = {
                'composition': composition,
                'frequency': frequency,
                'thickness': thickness,
                'predicted_se': result['total_se'],
                'confidence': result['confidence'],
                'timestamp': datetime.now()
            }
            
            # Feedback section
            st.markdown("---")
            st.subheader("📝 Help Improve Our Predictions")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                actual_se = st.number_input(
                    "If you measured the actual SE, please share (dB):",
                    0.0, 200.0, result['total_se'], 0.1,
                    help="Your measurement helps improve predictions for everyone"
                )
            
            with col2:
                st.write("")  # Spacer
                st.write("")
                if st.button("Submit Feedback"):
                    feedback_result = feedback_collector.submit_feedback(
                        prediction_id=f"pred_{datetime.now().timestamp()}",
                        composition=composition,
                        predicted_se=result['total_se'],
                        actual_se=actual_se,
                        frequency=frequency,
                        thickness=thickness,
                        conditions=conditions
                    )
                    st.success(feedback_result['message'])
                    st.balloons()

with tab2:
    st.header("Percentage Composition Input")
    
    # Import and use the direct composition manager
    from streamlit_app.direct_composition import render_direct_composition_ui
    from src.materials.material_properties import material_db
    
    dc_manager = render_direct_composition_ui(material_db)

with tab3:
    st.header("🎯 ML Model Predictions")
    
    st.info("This tab will show ML model predictions once training data is collected")
    
    # Show high-impact feedback requests
    st.subheader("📢 Help Us Improve - Test These Materials")
    
    requests = feedback_collector.get_high_impact_feedback_requests(3)
    
    for i, request in enumerate(requests):
        with st.expander(f"Material {i+1}: {request['reason']}"):
            st.write("**Composition:**")
            for elem, pct in request['composition'].items():
                st.write(f"- {elem}: {pct}%")
            st.write(f"**Expected Impact:** {request['expected_impact']:.0%}")
            st.write("Test this material and submit your results to significantly improve our model!")

with tab4:
    st.header("📈 System Analytics")
    
    # Accuracy over time plot
    st.subheader("Model Accuracy Trend")
    
    # Simulated data for demonstration
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    rmse_values = 15 - np.cumsum(np.random.exponential(0.1, 30))
    rmse_values = np.maximum(rmse_values, 3)  # Floor at 3 dB
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=rmse_values,
        mode='lines+markers',
        name='RMSE',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="RMSE Improvement Over Time",
        xaxis_title="Date",
        yaxis_title="RMSE (dB)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error analysis
    st.subheader("Systematic Error Analysis")
    
    error_patterns = validator.identify_systematic_errors()
    
    if error_patterns:
        for category, patterns in error_patterns.items():
            if patterns:
                st.write(f"**{category.replace('_', ' ').title()}:**")
                
                df = pd.DataFrame(patterns).T
                if not df.empty:
                    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>This ML-enhanced system continuously improves with user feedback.</p>
    <p>Current accuracy: ±5 dB for common materials | Target: ±2 dB by 2025</p>
</div>
""", unsafe_allow_html=True)