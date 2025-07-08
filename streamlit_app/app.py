"""
🔬 Chemical Reaction EMI Shield Designer
Dark Mode Chemistry Interface for EMI Shielding Analysis
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

# Import molecular presets
try:
    from molecular_presets import MOLECULAR_PRESETS, REACTION_PRESETS
except ImportError:
    MOLECULAR_PRESETS = {}
    REACTION_PRESETS = {}

# Import direct composition manager
try:
    from direct_composition_integration import render_direct_composition_section
except ImportError:
    render_direct_composition_section = None

# Page configuration
st.set_page_config(
    page_title="🔬 Chemical EMI Designer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# DARK MODE THEME & STYLING
# ================================

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
    
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: var(--space-4);
    }
    
    /* Section headers */
    .section-header {
        margin: var(--space-6) 0 var(--space-4) 0;
        padding-bottom: var(--space-2);
        border-bottom: 2px solid var(--border-color);
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        line-height: 1.2;
        margin: 0;
    }
    
    h1 { font-size: var(--font-4xl); }
    h2 { font-size: var(--font-3xl); }
    h3 { font-size: var(--font-2xl); }
    h4 { font-size: var(--font-xl); }
    h5 { font-size: var(--font-lg); }
    h6 { font-size: var(--font-base); }
    
    /* Main header */
    .main-header {
        background: var(--gradient-primary);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px var(--shadow-glow);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Card components */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        margin-bottom: var(--space-4);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: var(--border-light);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin-bottom: var(--space-3);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-color);
    }
    
    .card-title {
        font-size: var(--font-lg);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .card-subtitle {
        font-size: var(--font-sm);
        color: var(--text-secondary);
        margin: 0;
    }
    
    /* Glassmorphism effect */
    .glass-card {
        background: rgba(30, 30, 30, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--radius-xl);
        padding: var(--space-6);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Element grid system */
    .element-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: var(--space-2);
        margin: var(--space-4) 0;
    }
    
    .element-btn {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: var(--space-2);
        color: var(--text-primary);
        font-weight: 500;
        font-size: var(--font-xs);
        transition: all 0.2s ease;
        cursor: pointer;
        min-height: 60px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .element-btn:hover {
        border-color: var(--accent-blue);
        background: rgba(0, 212, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
    }
    
    .element-btn:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 212, 255, 0.3);
    }
    
    .element-btn.selected {
        background: var(--gradient-primary);
        border-color: var(--accent-blue);
        color: white;
        box-shadow: 0 0 16px var(--shadow-glow);
        transform: translateY(-2px);
    }
    
    .element-symbol {
        font-size: var(--font-base);
        font-weight: 700;
        margin-bottom: 2px;
    }
    
    .element-name {
        font-size: 10px;
        opacity: 0.7;
        text-align: center;
        line-height: 1.1;
        font-weight: 400;
    }
    
    /* Periodic table categories */
    .category-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin: var(--space-4) 0 var(--space-2) 0;
        font-size: var(--font-sm);
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .category-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    /* Element category colors */
    .alkali { border-left: 3px solid #ff6b6b; }
    .alkali .category-indicator { background: #ff6b6b; }
    
    .alkaline { border-left: 3px solid #feca57; }
    .alkaline .category-indicator { background: #feca57; }
    
    .transition { border-left: 3px solid #48dbfb; }
    .transition .category-indicator { background: #48dbfb; }
    
    .metalloid { border-left: 3px solid #ff9ff3; }
    .metalloid .category-indicator { background: #ff9ff3; }
    
    .nonmetal { border-left: 3px solid #54a0ff; }
    .nonmetal .category-indicator { background: #54a0ff; }
    
    .halogen { border-left: 3px solid #5f27cd; }
    .halogen .category-indicator { background: #5f27cd; }
    
    .noble { border-left: 3px solid #00d2d3; }
    .noble .category-indicator { background: #00d2d3; }
    
    /* Molecule display */
    .molecule-display {
        background: var(--bg-secondary);
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--space-6);
        text-align: center;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: var(--font-2xl);
        font-weight: 600;
        color: var(--text-secondary);
        transition: all 0.3s ease;
        margin: var(--space-4) 0;
    }
    
    .molecule-display.has-content {
        border-color: var(--accent-blue);
        background: rgba(0, 212, 255, 0.05);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
        color: var(--accent-blue);
        border-style: solid;
    }
    
    .molecule-display .placeholder {
        color: var(--text-muted);
        font-size: var(--font-base);
        font-weight: 400;
    }
    
    /* Reaction equation */
    .reaction-equation {
        background: var(--bg-card);
        border: 2px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--space-6);
        font-family: 'JetBrains Mono', monospace;
        font-size: var(--font-xl);
        font-weight: 500;
        color: var(--text-primary);
        text-align: center;
        min-height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: var(--space-3);
        margin: var(--space-4) 0;
        transition: all 0.3s ease;
    }
    
    .reaction-equation.active {
        border-color: var(--accent-purple);
        background: rgba(139, 92, 246, 0.05);
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.2);
    }
    
    .reaction-equation .molecule {
        background: var(--bg-tertiary);
        padding: var(--space-2) var(--space-3);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        color: var(--accent-blue);
    }
    
    /* Buttons */
    .btn {
        padding: var(--space-3) var(--space-4);
        border-radius: var(--radius-md);
        font-weight: 500;
        font-size: var(--font-sm);
        transition: all 0.2s ease;
        border: 1px solid var(--border-color);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: var(--space-2);
        text-decoration: none;
    }
    
    .btn-primary {
        background: var(--gradient-primary);
        border: none;
        color: white;
        box-shadow: 0 2px 8px rgba(0, 212, 255, 0.3);
    }
    
    .btn-primary:hover {
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.4);
        transform: translateY(-2px);
    }
    
    .btn-secondary {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }
    
    .btn-secondary:hover {
        background: var(--bg-card);
        border-color: var(--border-light);
    }
    
    /* React button */
    .react-button {
        background: var(--gradient-primary);
        border: none;
        border-radius: var(--radius-full);
        padding: var(--space-4) var(--space-8);
        font-size: var(--font-2xl);
        font-weight: 700;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin: var(--space-6) auto;
        width: fit-content;
        box-shadow: 0 8px 32px var(--shadow-glow);
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
    }
    
    .react-button:hover {
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.5);
        transform: translateY(-4px) scale(1.02);
    }
    
    .react-button:active {
        transform: translateY(-2px) scale(1.0);
    }
    
    /* Improved spacing */
    .stColumn > div {
        padding: var(--space-2);
    }
    
    /* Preset card hover effects */
    .preset-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border-color: var(--border-light);
    }
    
    /* Loading states */
    .loading {
        opacity: 0.6;
        pointer-events: none;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .react-button {
            font-size: var(--font-lg);
            padding: var(--space-3) var(--space-6);
        }
        
        h1 { font-size: var(--font-3xl); }
        h2 { font-size: var(--font-2xl); }
        h3 { font-size: var(--font-xl); }
    }
    
    /* Element buttons are now individually styled */
    
    /* Style the main REACT button */
    button[key="react_button"] {
        background: var(--gradient-primary) !important;
        border: none !important;
        border-radius: var(--radius-full) !important;
        color: white !important;
        font-size: var(--font-2xl) !important;
        font-weight: 700 !important;
        padding: var(--space-4) var(--space-8) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        box-shadow: 0 8px 32px var(--shadow-glow) !important;
        transition: all 0.3s ease !important;
        min-height: 80px !important;
    }
    
    button[key="react_button"]:hover {
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.5) !important;
        transform: translateY(-4px) scale(1.02) !important;
    }
    
    
    /* Selected element styling */
    .stButton > button:focus {
        border-color: var(--accent-blue) !important;
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: 0 0 16px var(--shadow-glow) !important;
        transform: translateY(-2px) !important;
        overflow: hidden;
    }
    
    .react-button:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 48px var(--shadow-glow);
    }
    
    .react-button:active {
        transform: scale(0.98);
    }
    
    .react-button.loading {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 8px 32px var(--shadow-glow); }
        50% { box-shadow: 0 12px 48px rgba(0, 212, 255, 0.4); }
        100% { box-shadow: 0 8px 32px var(--shadow-glow); }
    }
    
    /* Calculation steps */
    .calc-step {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .calc-step.active {
        border-color: var(--accent-green);
        background: rgba(16, 185, 129, 0.05);
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.1);
    }
    
    .calc-step h4 {
        color: var(--accent-blue);
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .calc-step .formula {
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-secondary);
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-blue);
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    /* Results hero section */
    .results-hero {
        background: var(--gradient-primary);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 12px 48px var(--shadow-glow);
        position: relative;
        overflow: hidden;
    }
    
    .results-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at center, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .results-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: var(--text-primary);
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Streamlit component overrides */
    .stButton > button {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        border-color: var(--accent-blue) !important;
        background: rgba(0, 212, 255, 0.1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.2) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: var(--gradient-primary) !important;
        border-color: var(--accent-blue) !important;
        color: white !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    .stNumberInput > div > div > input {
        background: var(--bg-tertiary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 10px !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    .slide-up {
        animation: slideUp 0.8s ease-out;
    }
    
    .scale-in {
        animation: scaleIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes scaleIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    /* Chemical formula styling */
    .chemical-formula {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .chemical-formula sub {
        font-size: 0.7em;
        vertical-align: sub;
        color: var(--accent-blue);
    }
    
    .chemical-formula sup {
        font-size: 0.7em;
        vertical-align: super;
        color: var(--accent-red);
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots:after {
        content: '';
        animation: dots 1.5s steps(5, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    /* Molecular orbital animation */
    .orbital-animation {
        animation: orbit 3s linear infinite;
    }
    
    @keyframes orbit {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Glow effects */
    .glow-blue {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        animation: glow-pulse 2s ease-in-out infinite alternate;
    }
    
    .glow-purple {
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        animation: glow-pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow-pulse {
        from { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
        to { box-shadow: 0 0 30px rgba(0, 212, 255, 0.6); }
    }
    
    /* Floating particles effect */
    .particles {
        position: relative;
        overflow: hidden;
    }
    
    .particles::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(2px 2px at 20px 30px, rgba(0, 212, 255, 0.3), transparent),
                    radial-gradient(2px 2px at 40px 70px, rgba(139, 92, 246, 0.3), transparent),
                    radial-gradient(1px 1px at 90px 40px, rgba(16, 185, 129, 0.3), transparent),
                    radial-gradient(1px 1px at 130px 80px, rgba(251, 191, 36, 0.3), transparent);
        animation: particle-float 20s linear infinite;
    }
    
    @keyframes particle-float {
        0% { transform: translateY(100%); }
        100% { transform: translateY(-100%); }
    }
    
    /* Notification styles */
    .notification {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-green);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    .notification.warning {
        border-left-color: var(--accent-yellow);
    }
    
    .notification.error {
        border-left-color: var(--accent-red);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# CHEMICAL PARSER & ENGINE
# ================================

class ChemicalParser:
    """Parse and validate chemical formulas and reactions."""
    
    @staticmethod
    def parse_formula(formula: str) -> Dict[str, int]:
        """Parse a chemical formula into element counts."""
        # Remove spaces and handle subscripts
        formula = formula.replace(" ", "")
        
        # Pattern to match element and count
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
    
    @staticmethod
    def calculate_molecular_weight(composition: Dict[str, int]) -> float:
        """Calculate molecular weight from composition."""
        total_weight = 0
        for element, count in composition.items():
            elem_data = material_db.get_material(element)
            if elem_data and 'atomic_weight' in elem_data:
                total_weight += elem_data['atomic_weight'] * count
            else:
                # Fallback weights for common elements
                weights = {
                    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                    'F': 18.998, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982,
                    'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45,
                    'K': 39.098, 'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546,
                    'Zn': 65.38, 'Ag': 107.868, 'Au': 196.967, 'Pb': 207.2
                }
                total_weight += weights.get(element, 50.0) * count
        
        return total_weight

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
                'molecular_weight': ChemicalParser.calculate_molecular_weight(composition)
            })
    
    def remove_molecule(self, index: int):
        """Remove a molecule from the reaction."""
        if 0 <= index < len(self.molecules):
            self.molecules.pop(index)
    
    def get_total_composition(self) -> Dict[str, float]:
        """Calculate total elemental composition by mass percentage."""
        if not self.molecules:
            return {}
        
        # Check if we have a direct composition (single molecule with type='direct')
        if len(self.molecules) == 1 and self.molecules[0].get('type') == 'direct':
            # Direct composition - percentages are already provided
            return self.molecules[0]['composition']
        
        # Calculate total mass for each element (molecular mode)
        element_masses = {}
        total_mass = 0
        
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                # Skip direct compositions in mixed mode
                continue
                
            mol_mass = mol['molecular_weight'] * mol['coefficient']
            total_mass += mol_mass
            
            for element, count in mol['composition'].items():
                elem_data = material_db.get_material(element)
                if elem_data and 'atomic_weight' in elem_data:
                    atomic_weight = elem_data['atomic_weight']
                else:
                    # Fallback weights
                    weights = {
                        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                        'F': 18.998, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982,
                        'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45,
                        'K': 39.098, 'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546,
                        'Zn': 65.38, 'Ag': 107.868, 'Au': 196.967, 'Pb': 207.2
                    }
                    atomic_weight = weights.get(element, 50.0)
                
                element_mass = atomic_weight * count * mol['coefficient']
                element_masses[element] = element_masses.get(element, 0) + element_mass
        
        # Convert to percentages
        if total_mass > 0:
            return {element: (mass / total_mass) * 100 
                   for element, mass in element_masses.items()}
        return {}
    
    def get_reaction_equation(self) -> str:
        """Get the formatted reaction equation."""
        if not self.molecules:
            return "No reaction defined"
        
        equation_parts = []
        for mol in self.molecules:
            if mol.get('type') == 'direct':
                # Format direct composition differently
                display_name = mol.get('display_name', 'Direct Composition')
                equation_parts.append(display_name)
            else:
                coeff = f"{mol['coefficient']}" if mol['coefficient'] > 1 else ""
                formula = ChemicalParser.format_formula(mol['composition'])
                equation_parts.append(f"{coeff}{formula}")
        
        return " + ".join(equation_parts)

# ================================
# SESSION STATE INITIALIZATION
# ================================

if 'reaction_engine' not in st.session_state:
    st.session_state.reaction_engine = ReactionEngine()

if 'current_molecule' not in st.session_state:
    st.session_state.current_molecule = {}

if 'calculation_steps' not in st.session_state:
    st.session_state.calculation_steps = []

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

if 'shield_thickness' not in st.session_state:
    st.session_state.shield_thickness = 1.0

if 'shield_frequency' not in st.session_state:
    st.session_state.shield_frequency = 100.0

if 'direct_composition' not in st.session_state:
    st.session_state.direct_composition = {}

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'last_conditions' not in st.session_state:
    st.session_state.last_conditions = None

# ================================
# HELPER FUNCTIONS
# ================================

def get_element_category(symbol: str) -> str:
    """Get the periodic table category for styling."""
    categories = {
        'H': 'nonmetal', 'He': 'noble',
        'Li': 'alkali', 'Be': 'alkaline', 'B': 'metalloid', 'C': 'nonmetal', 'N': 'nonmetal', 'O': 'nonmetal', 'F': 'halogen', 'Ne': 'noble',
        'Na': 'alkali', 'Mg': 'alkaline', 'Al': 'metalloid', 'Si': 'metalloid', 'P': 'nonmetal', 'S': 'nonmetal', 'Cl': 'halogen', 'Ar': 'noble',
        'K': 'alkali', 'Ca': 'alkaline', 'Sc': 'transition', 'Ti': 'transition', 'V': 'transition', 'Cr': 'transition', 'Mn': 'transition', 'Fe': 'transition', 'Co': 'transition', 'Ni': 'transition', 'Cu': 'transition', 'Zn': 'transition', 'Ga': 'metalloid', 'Ge': 'metalloid', 'As': 'metalloid', 'Se': 'nonmetal', 'Br': 'halogen', 'Kr': 'noble',
        'Rb': 'alkali', 'Sr': 'alkaline', 'Y': 'transition', 'Zr': 'transition', 'Nb': 'transition', 'Mo': 'transition', 'Tc': 'transition', 'Ru': 'transition', 'Rh': 'transition', 'Pd': 'transition', 'Ag': 'transition', 'Cd': 'transition', 'In': 'metalloid', 'Sn': 'metalloid', 'Sb': 'metalloid', 'Te': 'metalloid', 'I': 'halogen', 'Xe': 'noble',
        'Cs': 'alkali', 'Ba': 'alkaline', 'Au': 'transition', 'Hg': 'transition', 'Tl': 'metalloid', 'Pb': 'metalloid', 'Bi': 'metalloid'
    }
    return categories.get(symbol, 'transition')

def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate precision."""
    if num >= 1000:
        return f"{num:.{precision}e}"
    elif num >= 1:
        return f"{num:.{precision}f}"
    else:
        return f"{num:.{precision}e}"

# ================================
# MAIN APP LAYOUT
# ================================


# Main header with title
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

# HOW TO USE section at the top
st.markdown("""
<div class="card" style="margin-bottom: var(--space-6);">
    <div class="card-header">
        <h3 class="card-title">📋 How To Use</h3>
    </div>
    <div style="padding: var(--space-3);">
        <p style="margin: 0 0 var(--space-2) 0;">1a. <strong>Molecular Builder</strong>: Select elements from the list below to build molecules</p>
        <p style="margin: 0 0 var(--space-2) 0;">1b. <strong>Direct Composition</strong>: OR enter percentages directly (e.g., 70% Fe, 30% C)</p>
        <p style="margin: 0 0 var(--space-2) 0;">2. Add molecules or compositions to create your material</p>
        <p style="margin: 0 0 var(--space-2) 0;">3. Set shield thickness and frequency parameters</p>
        <p style="margin: 0 0 var(--space-2) 0;">4. Click ⚛️ REACT to analyze EMI shielding</p>
        <p style="margin: 0;">5. View step-by-step calculations and results</p>
    </div>
</div>
""", unsafe_allow_html=True)

# INPUT MODE SELECTOR
st.markdown("""
<div style="margin: var(--space-4) 0 var(--space-6) 0;">
    <h3 style="
        font-size: var(--font-xl);
        font-weight: 600;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: var(--space-3);
    ">Select Input Method</h3>
</div>
""", unsafe_allow_html=True)

input_mode = st.radio(
    "Choose how to define your material:",
    ["Molecular Builder", "Direct Composition"],
    horizontal=True,
    help="Molecular: Build molecules from elements | Direct: Enter percentages directly",
    key="input_mode"
)

# Single column layout for better flow
main_container = st.container()

with main_container:
    if input_mode == "Molecular Builder":
        # BUILD YOUR MOLECULE section
        st.markdown("""
    <div style="margin: var(--space-6) 0 var(--space-4) 0;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 var(--space-4) 0;
            text-align: center;
        ">⚛️ Molecule</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display current molecule
    current_formula = ChemicalParser.format_formula(st.session_state.current_molecule)
    if current_formula:
        st.markdown(f"""
        <div class="molecule-display has-content">
            <span class="chemical-formula">{current_formula}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="molecule-display">
            <span class="placeholder">Build your molecule by selecting elements below</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Quantity adjustments for selected elements (moved to be under molecule display)
    if st.session_state.current_molecule:
        st.markdown("""
        <div class="category-header">
            <div class="category-indicator" style="background: var(--accent-green);"></div>
            <span>Adjust Quantities</span>
        </div>
        """, unsafe_allow_html=True)
        
        for element, quantity in list(st.session_state.current_molecule.items()):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"**{element}**")
            
            with col2:
                new_quantity = st.number_input(
                    f"Quantity for {element}",
                    min_value=1,
                    max_value=999,
                    value=quantity,
                    key=f"qty_{element}",
                    label_visibility="collapsed"
                )
                st.session_state.current_molecule[element] = new_quantity
            
            with col3:
                if st.button("🗑️", key=f"del_{element}", help="Remove element"):
                    del st.session_state.current_molecule[element]
                    st.rerun()
        
        # Add molecule to reaction
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add to Reaction", type="primary", use_container_width=True):
                if st.session_state.current_molecule:
                    formula = ChemicalParser.format_formula(st.session_state.current_molecule)
                    st.session_state.reaction_engine.add_molecule(
                        formula.replace('<sub>', '').replace('</sub>', ''),
                        1
                    )
                    st.session_state.current_molecule = {}
                    st.rerun()
        
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.current_molecule = {}
                st.rerun()
    
    # ELEMENT LIST section
    st.markdown("""
    <div style="margin: var(--space-6) 0 var(--space-4) 0;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 var(--space-4) 0;
            text-align: center;
        ">🧪 Element List</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Complete periodic table layout with all 118 elements
    periodic_table = [
        # Period 1
        ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
        # Period 2  
        ["Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
        # Period 3
        ["Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
        # Period 4
        ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
        # Period 5
        ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
        # Period 6
        ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
        # Period 7
        ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
        # Spacer
        ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        # Lanthanides
        ["", "", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "", ""],
        # Actinides  
        ["", "", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "", ""]
    ]
    
    # Element categories for color coding (darker shades for better text readability)
    element_colors = {
        # Alkali metals (darker coral/salmon)
        "Li": "#dc2626", "Na": "#dc2626", "K": "#dc2626", "Rb": "#dc2626", "Cs": "#dc2626", "Fr": "#dc2626",
        # Alkaline earth metals (darker golden yellow)
        "Be": "#ca8a04", "Mg": "#ca8a04", "Ca": "#ca8a04", "Sr": "#ca8a04", "Ba": "#ca8a04", "Ra": "#ca8a04",
        # Transition metals (darker green)
        "Sc": "#16a34a", "Ti": "#16a34a", "V": "#16a34a", "Cr": "#16a34a", "Mn": "#16a34a", "Fe": "#16a34a", 
        "Co": "#16a34a", "Ni": "#16a34a", "Cu": "#16a34a", "Zn": "#16a34a", "Y": "#16a34a", "Zr": "#16a34a", 
        "Nb": "#16a34a", "Mo": "#16a34a", "Tc": "#16a34a", "Ru": "#16a34a", "Rh": "#16a34a", "Pd": "#16a34a", 
        "Ag": "#16a34a", "Cd": "#16a34a", "Hf": "#16a34a", "Ta": "#16a34a", "W": "#16a34a", "Re": "#16a34a", 
        "Os": "#16a34a", "Ir": "#16a34a", "Pt": "#16a34a", "Au": "#16a34a", "Hg": "#16a34a", "Rf": "#16a34a", 
        "Db": "#16a34a", "Sg": "#16a34a", "Bh": "#16a34a", "Hs": "#16a34a", "Mt": "#16a34a", "Ds": "#16a34a", 
        "Rg": "#16a34a", "Cn": "#16a34a",
        # Post-transition metals (darker teal)
        "Al": "#0891b2", "Ga": "#0891b2", "In": "#0891b2", "Sn": "#0891b2", "Tl": "#0891b2", "Pb": "#0891b2", 
        "Bi": "#0891b2", "Nh": "#0891b2", "Fl": "#0891b2", "Mc": "#0891b2", "Lv": "#0891b2",
        # Metalloids (darker pink/magenta)
        "B": "#c026d3", "Si": "#c026d3", "Ge": "#c026d3", "As": "#c026d3", "Sb": "#c026d3", "Te": "#c026d3", "Po": "#c026d3",
        # Nonmetals (darker blue)
        "H": "#2563eb", "C": "#2563eb", "N": "#2563eb", "O": "#2563eb", "P": "#2563eb", "S": "#2563eb", "Se": "#2563eb",
        # Halogens (darker purple)
        "F": "#7c3aed", "Cl": "#7c3aed", "Br": "#7c3aed", "I": "#7c3aed", "At": "#7c3aed", "Ts": "#7c3aed",
        # Noble gases (darker pink)
        "He": "#db2777", "Ne": "#db2777", "Ar": "#db2777", "Kr": "#db2777", "Xe": "#db2777", "Rn": "#db2777", "Og": "#db2777",
        # Lanthanides (darker lavender)
        "La": "#6d28d9", "Ce": "#6d28d9", "Pr": "#6d28d9", "Nd": "#6d28d9", "Pm": "#6d28d9", "Sm": "#6d28d9", 
        "Eu": "#6d28d9", "Gd": "#6d28d9", "Tb": "#6d28d9", "Dy": "#6d28d9", "Ho": "#6d28d9", "Er": "#6d28d9", 
        "Tm": "#6d28d9", "Yb": "#6d28d9", "Lu": "#6d28d9",
        # Actinides (darker blue)
        "Ac": "#1e40af", "Th": "#1e40af", "Pa": "#1e40af", "U": "#1e40af", "Np": "#1e40af", "Pu": "#1e40af", 
        "Am": "#1e40af", "Cm": "#1e40af", "Bk": "#1e40af", "Cf": "#1e40af", "Es": "#1e40af", "Fm": "#1e40af", 
        "Md": "#1e40af", "No": "#1e40af", "Lr": "#1e40af"
    }
    
    # Create container for periodic table
    periodic_container = st.container()
    
    with periodic_container:
        # Apply button styles using JavaScript after DOM loads
        st.markdown("""
    <script>
    // Wait for Streamlit to render all elements
    setTimeout(() => {
        // Find all element button markers
        const markers = document.querySelectorAll('.element-button-marker');
        markers.forEach(marker => {
            const element = marker.getAttribute('data-element');
            const isSelected = marker.getAttribute('data-selected') === 'true';
            
            // Find the next sibling that contains the button
            let nextSibling = marker.nextElementSibling;
            while (nextSibling && !nextSibling.querySelector('.stButton')) {
                nextSibling = nextSibling.nextElementSibling;
            }
            
            if (nextSibling) {
                const button = nextSibling.querySelector('.stButton button');
                if (button) {
                    // Apply button styles
                    button.style.cssText = `
                        background: ${isSelected ? '#dc3545' : '#28a745'} !important;
                        color: white !important;
                        border: ${isSelected ? '3px solid #00d4ff' : '1px solid #444444'} !important;
                        border-top: none !important;
                        border-radius: 0 0 8px 8px !important;
                        height: 20px !important;
                        min-height: 20px !important;
                        width: 75px !important;
                        padding: 0 !important;
                        margin: 0 !important;
                        font-size: 10px !important;
                        line-height: 20px !important;
                        cursor: pointer !important;
                    `;
                    
                    // Style the button container
                    const buttonContainer = button.closest('.stButton');
                    if (buttonContainer) {
                        buttonContainer.style.cssText = `
                            width: 75px !important;
                            margin: -1px auto 0 auto !important;
                            padding: 0 !important;
                        `;
                    }
                }
            }
        });
        }, 500);
        </script>
        """, unsafe_allow_html=True)
        
        # Display periodic table with better styling
        legend_added = False
        for row_idx, row in enumerate(periodic_table):
            cols = st.columns(18)  # 18 columns for period table width
            
            
            for col_idx, element in enumerate(row):
                # Add legend in the empty space between H and He (columns 1-16 of row 0)
                if row_idx == 0 and col_idx == 1 and not legend_added:
                    # Use columns 1 through 16 for the legend
                    with cols[1]:
                        st.markdown(f"""
                        <div style="
                            background: var(--bg-secondary);
                            border: 2px solid var(--border-color);
                            border-radius: var(--radius-lg);
                            padding: 16px 20px;
                            width: calc(75px * 16 + 8px * 15);
                            box-sizing: border-box;
                            height: 105px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        ">
                            <h3 style="text-align: center; margin-bottom: 10px; color: var(--accent-red); font-size: 14px; font-weight: 600;">Element Categories</h3>
                            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; font-size: 10px;">
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #dc2626; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Alkali Metals</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #ca8a04; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Alkaline Earth</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #16a34a; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Transition</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #0891b2; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Post-Transition</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #c026d3; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Metalloids</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #2563eb; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Nonmetals</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #7c3aed; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Halogens</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #db2777; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Noble Gases</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #6d28d9; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Lanthanides</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <div style="width: 10px; height: 10px; background: #1e40af; border-radius: 2px; flex-shrink: 0;"></div>
                                    <span style="color: var(--text-primary);">Actinides</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    legend_added = True
                    continue  # Skip to next iteration since we used this column
                
                # Skip columns that would be covered by the legend
                if row_idx == 0 and col_idx >= 1 and col_idx < 17 and legend_added:
                    continue
                
                with cols[col_idx]:
                    if element and element != "":
                        elem_data = material_db.get_material(element)
                        if elem_data:
                            is_selected = element in st.session_state.current_molecule
                            element_name = elem_data.get('name', element)
                            atomic_number = elem_data.get('atomic_number', '')
                            atomic_weight = elem_data.get('atomic_weight', 0)
                            weight_display = f"{atomic_weight:.3f}" if atomic_weight > 0 else ""
                            color = element_colors.get(element, "#666666")
                            
                            # Fixed size element display cell
                            element_width = "75px"
                            element_height = "85px"
                            button_height = "20px"
                            
                            # Container for element + button
                            st.markdown(f"""
                            <div style="width: {element_width}; margin: 0 auto;">
                                <!-- Element cell -->
                                <div style="
                                    background: {color}; 
                                    color: white; 
                                    border: {'3px solid #00d4ff' if is_selected else '1px solid #444444'}; 
                                    border-bottom: none;
                                    border-radius: 8px 8px 0 0; 
                                    height: {element_height}; 
                                    width: {element_width}; 
                                    padding: 6px 4px; 
                                    font-family: 'Inter', sans-serif; 
                                    display: flex; 
                                    flex-direction: column; 
                                    justify-content: space-between; 
                                    align-items: center; 
                                    text-align: center; 
                                    box-sizing: border-box;
                                    margin-bottom: 0;
                                ">
                                    <div style="font-size: 9px; font-weight: 400; opacity: 0.9;">{atomic_number}</div>
                                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; flex-grow: 1;">
                                        <div style="font-size: 18px; font-weight: 700; line-height: 1; margin-bottom: 2px;">{element}</div>
                                        <div style="font-size: 9px; font-weight: 400; opacity: 0.9;">{element_name[:8]}</div>
                                    </div>
                                    <div style="font-size: 8px; font-weight: 300; opacity: 0.8;">{weight_display}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Slim select button with exact same width
                            button_key = f"select_{element}"
                            button_text = ""  # No text in the button
                            
                            # Add a marker div before the button to help with styling
                            st.markdown(f'<div class="element-button-marker" data-element="{element}" data-selected="{str(is_selected).lower()}"></div>', unsafe_allow_html=True)
                            
                            if st.button(button_text, key=button_key, use_container_width=True):
                                if is_selected:
                                    del st.session_state.current_molecule[element]
                                else:
                                    st.session_state.current_molecule[element] = 1
                                st.rerun()
                        else:
                            # Element not in database, show as disabled
                            st.markdown(f"""
                            <div style="
                                background: #333333;
                                border: 1px solid #555555;
                                border-radius: 8px;
                                padding: 4px;
                                text-align: center;
                                min-height: 60px;
                                margin: 1px;
                                color: #888888;
                                font-size: 11px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            ">{element}</div>
                            """, unsafe_allow_html=True)
    
    # Molecular presets section
    if MOLECULAR_PRESETS:
        st.markdown("""
        <div style="margin: var(--space-6) 0 var(--space-4) 0;">
            <h3 style="
                font-size: var(--font-2xl);
                font-weight: 600;
                color: var(--text-primary);
                margin: 0 0 var(--space-4) 0;
                text-align: center;
            ">🧪 Quick Molecules</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Group presets by category
        categories = {}
        for name, preset in MOLECULAR_PRESETS.items():
            category = preset.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append((name, preset))
        
        # Display preset cards by category
        for category, presets in categories.items():
            with st.expander(f"🧪 {category}", expanded=False):
                # Create grid layout for preset cards
                for i in range(0, len(presets), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(presets):
                            name, preset = presets[i + j]
                            with col:
                                # Create preset card
                                st.markdown(f"""
                                <div class="preset-card" style="
                                    background: var(--bg-tertiary);
                                    border: 1px solid var(--border-color);
                                    border-radius: var(--radius-md);
                                    padding: var(--space-3);
                                    margin-bottom: var(--space-2);
                                    cursor: pointer;
                                    transition: all 0.2s ease;
                                ">
                                    <div style="
                                        font-family: 'JetBrains Mono', monospace;
                                        font-size: var(--font-lg);
                                        font-weight: 600;
                                        color: var(--accent-purple);
                                        margin-bottom: var(--space-1);
                                    ">{preset['formula']}</div>
                                    <div style="
                                        font-size: var(--font-sm);
                                        color: var(--text-primary);
                                        font-weight: 500;
                                        margin-bottom: var(--space-1);
                                    ">{name}</div>
                                    <div style="
                                        font-size: var(--font-xs);
                                        color: var(--text-secondary);
                                        line-height: 1.3;
                                    ">{preset.get('description', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if st.button(
                                    f"Add {name}",
                                    key=f"preset_{name}",
                                    help=f"Add {preset['formula']} directly to reaction - {preset.get('description', '')}",
                                    use_container_width=True
                                ):
                                    # Add directly to reaction instead of current molecule
                                    formula = preset['formula']
                                    st.session_state.reaction_engine.add_molecule(formula, 1)
                                    st.rerun()
    
    # COMPLETE REACTION section
    st.markdown("""
    <div style="margin: var(--space-8) 0 var(--space-4) 0;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 var(--space-4) 0;
            text-align: center;
        ">⚗️ Complete Reaction</h3>
    </div>
    """, unsafe_allow_html=True)
    
    reaction_eq = st.session_state.reaction_engine.get_reaction_equation()
    if len(st.session_state.reaction_engine.molecules) > 0:
        # Parse reaction to highlight individual molecules
        molecules = reaction_eq.split(' + ')
        molecule_html = []
        for mol in molecules:
            molecule_html.append(f'<span class="molecule">{mol.strip()}</span>')
        
        st.markdown(f"""
        <div class="reaction-equation active">
            {' + '.join(molecule_html)}
        </div>
        """, unsafe_allow_html=True)
        
        # REACTION COMPONENTS section
        st.markdown("""
        <div style="margin: var(--space-6) 0 var(--space-4) 0;">
            <h4 style="
                font-size: var(--font-xl);
                font-weight: 600;
                color: var(--text-primary);
                margin: 0 0 var(--space-3) 0;
                text-align: center;
            ">→ Reaction Components</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, mol in enumerate(st.session_state.reaction_engine.molecules):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                formula = ChemicalParser.format_formula(mol['composition'])
                st.markdown(f"<span class='chemical-formula'>{formula}</span>", unsafe_allow_html=True)
            
            with col2:
                new_coeff = st.number_input(
                    f"Coefficient {i}",
                    min_value=1,
                    value=mol['coefficient'],
                    key=f"coeff_{i}",
                    label_visibility="collapsed"
                )
                st.session_state.reaction_engine.molecules[i]['coefficient'] = new_coeff
            
            with col3:
                if st.button("🗑️", key=f"del_mol_{i}"):
                    st.session_state.reaction_engine.remove_molecule(i)
                    st.rerun()
        
        # Clear reaction
        if st.button("🔄 Clear Reaction", use_container_width=True):
            st.session_state.reaction_engine = ReactionEngine()
            st.session_state.calculation_steps = []
            st.session_state.show_results = False
            st.rerun()
        
        # Reaction presets
        if REACTION_PRESETS:
            st.markdown("**Preset Reactions:**")
            
            preset_reaction = st.selectbox(
                "Choose a preset reaction",
                options=["Custom"] + list(REACTION_PRESETS.keys()),
                help="Load a predefined reaction"
            )
            
            if preset_reaction != "Custom" and preset_reaction in REACTION_PRESETS:
                if st.button(f"Load {preset_reaction}", use_container_width=True):
                    # Clear current reaction
                    st.session_state.reaction_engine = ReactionEngine()
                    
                    # Load preset
                    preset = REACTION_PRESETS[preset_reaction]
                    for mol in preset['molecules']:
                        st.session_state.reaction_engine.add_molecule(
                            mol['formula'],
                            mol['coefficient']
                        )
                    
                    st.session_state.calculation_steps = []
                    st.session_state.show_results = False
                    st.rerun()
                
                # Show description
                if preset_reaction in REACTION_PRESETS:
                    description = REACTION_PRESETS[preset_reaction].get('description', '')
                    if description:
                        st.markdown(f"*{description}*")
    else:
        st.markdown("""
        <div class="reaction-equation">
            Add molecules to build your reaction
        </div>
        """, unsafe_allow_html=True)
    
    # Shield parameters
    st.markdown("### ⚙️ Shield Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.shield_thickness = st.number_input(
            "Thickness (mm)",
            min_value=0.01,
            max_value=100.0,
            value=st.session_state.shield_thickness,
            step=0.1
        )
    
    with col2:
        st.session_state.shield_frequency = st.number_input(
            "Frequency (MHz)",
            min_value=0.1,
            max_value=10000.0,
            value=st.session_state.shield_frequency,
            step=10.0
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Big REACT button section
    if len(st.session_state.reaction_engine.molecules) > 0:
        st.markdown("""
        <div style="margin: var(--space-8) 0; text-align: center;">
            <div style="
                background: var(--bg-card);
                border: 2px solid var(--border-color);
                border-radius: var(--radius-lg);
                padding: var(--space-6);
                margin: var(--space-4) auto;
                max-width: 400px;
            ">
                <h4 style="
                    font-size: var(--font-xl);
                    color: var(--text-primary);
                    margin: 0 0 var(--space-4) 0;
                ">Ready to Analyze!</h4>
                <p style="
                    color: var(--text-secondary);
                    margin: 0 0 var(--space-4) 0;
                    font-size: var(--font-sm);
                ">Click below to start EMI shielding analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚛️ REACT", key="react_button", use_container_width=True):
                # Show loading animation
                with st.spinner("🔬 Analyzing molecular structure..."):
                    time.sleep(0.5)
                
                with st.spinner("⚗️ Calculating material properties..."):
                    time.sleep(0.8)
                
                with st.spinner("🛡️ Computing EMI shielding..."):
                    time.sleep(0.7)
                
                st.session_state.show_results = True
                st.session_state.calculation_steps = []
                
                # Show success message
                st.success("✨ Reaction complete! View results below.")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
    
    # RESULTS section
    if st.session_state.show_results and len(st.session_state.reaction_engine.molecules) > 0:
        st.markdown("""
        <div style="margin: var(--space-8) 0 var(--space-6) 0;">
            <h2 style="
                font-size: var(--font-3xl);
                font-weight: 700;
                color: var(--text-primary);
                margin: 0 0 var(--space-2) 0;
                text-align: center;
                border-bottom: 3px solid var(--accent-green);
                padding-bottom: var(--space-3);
            ">📊 RESULTS</h2>
            <p style="
                text-align: center;
                color: var(--text-secondary);
                font-size: var(--font-lg);
                margin: var(--space-3) 0 0 0;
            ">Step-by-step EMI shielding analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Step 1: Molecular Analysis
        with st.expander("🔬 Step 1: Molecular Analysis", expanded=True):
            st.markdown("**Reaction Components:**")
            
            for i, mol in enumerate(st.session_state.reaction_engine.molecules):
                formula = ChemicalParser.format_formula(mol['composition'])
                st.markdown(f"""
                <div class="calc-step">
                    <strong>Molecule {i+1}:</strong> {formula}<br>
                    <strong>Coefficient:</strong> {mol['coefficient']}<br>
                    <strong>Molecular Weight:</strong> {mol['molecular_weight']:.2f} g/mol
                </div>
                """, unsafe_allow_html=True)
            
            # Total composition
            total_comp = st.session_state.reaction_engine.get_total_composition()
            
            st.markdown("**Total Elemental Composition (by mass %):**")
            comp_df = pd.DataFrame([
                {'Element': elem, 'Percentage': f"{perc:.2f}%"}
                for elem, perc in total_comp.items()
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Step 2: Material Properties
        with st.expander("⚗️ Step 2: Material Properties Calculation", expanded=True):
            if total_comp:
                conductivity = 0
                permeability = 1.0
                permittivity = 1.0
                density = 0
                
                st.markdown("**Property Calculations:**")
                
                for element, percentage in total_comp.items():
                    elem_data = material_db.get_material(element)
                    if elem_data:
                        weight = percentage / 100.0
                        
                        # Conductivity calculation
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
                        
                        # Density calculation
                        elem_density = elem_data.get('density', 1000)
                        density += elem_density * weight
                        
                        st.markdown(f"""
                        <div class="calc-step">
                            <strong>{element} ({percentage:.1f}%):</strong><br>
                            σ = {format_number(elem_conductivity)} S/m<br>
                            μᵣ = {elem_permeability:.3f}<br>
                            εᵣ = {elem_permittivity:.3f}<br>
                            ρ = {elem_density:.0f} kg/m³
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display composite properties
                st.markdown("**Composite Material Properties:**")
                st.markdown(f"""
                <div class="calc-step active">
                    <div class="formula">
                        Effective Conductivity: σₑff = {format_number(conductivity)} S/m<br>
                        Effective Permeability: μᵣ,ₑff = {permeability:.3f}<br>
                        Effective Permittivity: εᵣ,ₑff = {permittivity:.3f}<br>
                        Effective Density: ρₑff = {density:.0f} kg/m³
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Step 3: EMI Calculations
        with st.expander("🛡️ Step 3: EMI Shielding Physics", expanded=True):
            if total_comp:
                try:
                    # Ensure values are within valid ranges
                    conductivity = max(conductivity, 1e-10)  # Minimum conductivity
                    permeability = max(permeability, 0.999)  # Minimum permeability
                    permittivity = max(permittivity, 1.0)    # Minimum permittivity
                    
                    # Perform EMI calculation
                    result = emi_calculator.calculate_shielding_effectiveness(
                        conductivity,
                        permeability,
                        permittivity,
                        st.session_state.shield_thickness / 1000,  # Convert mm to m
                        st.session_state.shield_frequency * 1e6    # Convert MHz to Hz
                    )
                    
                    # Display calculation steps
                    st.markdown("**Electromagnetic Analysis:**")
                    
                    # Skin depth
                    st.markdown(f"""
                    <div class="calc-step">
                        <h4>Skin Depth Calculation</h4>
                        <div class="formula">
                            δ = √(2 / (ωμσ))<br>
                            δ = {result['skin_depth']*1000:.3f} mm
                        </div>
                        <em>Penetration depth of electromagnetic waves</em>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Intrinsic impedance
                    impedance_mag = abs(complex(result['intrinsic_impedance_real'], result['intrinsic_impedance_imag']))
                    st.markdown(f"""
                    <div class="calc-step">
                        <h4>Intrinsic Impedance</h4>
                        <div class="formula">
                            η = √(μ / ε*)<br>
                            |η| = {impedance_mag:.2f} Ω
                        </div>
                        <em>Material's resistance to electromagnetic wave propagation</em>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Shielding components
                    st.markdown(f"""
                    <div class="calc-step">
                        <h4>Shielding Components</h4>
                        <div class="formula">
                            Reflection Loss: {result['reflection_loss']:.1f} dB<br>
                            Absorption Loss: {result['absorption_loss']:.1f} dB<br>
                            Multiple Reflection: {result['multiple_reflection_loss']:.1f} dB
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store result for final display
                    st.session_state.final_result = result
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"⚠️ **Calculation Error**: {error_msg}")
                    
                    # Provide helpful suggestions
                    if "permeability" in error_msg.lower():
                        st.info("""
                        💡 **Tip**: Some elements like Fluorine have diamagnetic properties. 
                        Try using more common EMI shielding materials like:
                        - **Metals**: Fe, Cu, Al, Ni, Ag
                        - **Metal Oxides**: Fe₂O₃, Al₂O₃
                        - **Conductors**: Carbon structures
                        """)
                    elif "conductivity" in error_msg.lower():
                        st.info("💡 **Tip**: Add more conductive elements like Cu, Ag, Al, or Fe to your reaction.")
                    else:
                        st.info("💡 **Tip**: Try using preset reactions or common EMI shielding materials.")
                    
                    st.session_state.final_result = None
        
        # Step 4: Final Results
        if hasattr(st.session_state, 'final_result') and st.session_state.final_result:
            result = st.session_state.final_result
            
            st.markdown("---")
            st.markdown("### 🎯 Final Results")
            
            # Hero result
            st.markdown(f"""
            <div class="results-hero">
                <div style="font-size: 1.3rem; opacity: 0.9; margin-bottom: 0.5rem;">
                    Total Shielding Effectiveness
                </div>
                <div class="results-value">{result['total_se']:.1f} dB</div>
                <div style="opacity: 0.8; font-size: 1.1rem;">
                    @ {st.session_state.shield_frequency} MHz • {st.session_state.shield_thickness} mm thickness
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Reflection Loss</div>
                    <div class="metric-value">{result['reflection_loss']:.1f} dB</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Absorption Loss</div>
                    <div class="metric-value">{result['absorption_loss']:.1f} dB</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Skin Depth</div>
                    <div class="metric-value">{result['skin_depth']*1000:.3f} mm</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance rating
            effectiveness = (
                "Excellent" if result['total_se'] > 80 else
                "Very Good" if result['total_se'] > 60 else
                "Good" if result['total_se'] > 40 else
                "Moderate" if result['total_se'] > 20 else
                "Poor"
            )
            
            rating_colors = {
                "Excellent": "var(--accent-green)",
                "Very Good": "var(--accent-blue)",
                "Good": "var(--accent-yellow)",
                "Moderate": "var(--accent-yellow)",
                "Poor": "var(--accent-red)"
            }
            
            st.markdown(f"""
            <div class="notification" style="border-left-color: {rating_colors[effectiveness]};">
                <strong>Shield Performance Rating: {effectiveness}</strong><br>
                Your reaction produces a {effectiveness.lower()} EMI shield.
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizations
            with st.expander("📈 Detailed Analysis", expanded=False):
                # Shielding breakdown pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Reflection', 'Absorption', 'Multiple Reflection'],
                    values=[
                        max(0, result['reflection_loss']),
                        max(0, result['absorption_loss']),
                        max(0, result['multiple_reflection_loss'])
                    ],
                    hole=0.4,
                    marker_colors=['#00d4ff', '#8b5cf6', '#10b981']
                )])
                
                fig_pie.update_layout(
                    title="Shielding Mechanism Breakdown",
                    font=dict(color='#f0f0f0'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Frequency response
                frequencies = np.logspace(5, 10, 50)
                se_values = []
                
                for freq in frequencies:
                    res = emi_calculator.calculate_shielding_effectiveness(
                        conductivity, permeability, permittivity,
                        st.session_state.shield_thickness / 1000, freq
                    )
                    se_values.append(res['total_se'])
                
                fig_freq = go.Figure()
                
                fig_freq.add_trace(go.Scatter(
                    x=frequencies / 1e6,
                    y=se_values,
                    mode='lines',
                    name='SE',
                    line=dict(color='#00d4ff', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 212, 255, 0.1)'
                ))
                
                # Current point
                fig_freq.add_trace(go.Scatter(
                    x=[st.session_state.shield_frequency],
                    y=[result['total_se']],
                    mode='markers',
                    name='Current',
                    marker=dict(size=12, color='#f87171')
                ))
                
                fig_freq.update_layout(
                    title="Frequency Response",
                    xaxis_title="Frequency (MHz)",
                    yaxis_title="Shielding Effectiveness (dB)",
                    xaxis_type="log",
                    font=dict(color='#f0f0f0'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_freq, use_container_width=True)
    
    else:  # Direct Composition mode
        if render_direct_composition_section:
            render_direct_composition_section(st, material_db)
        else:
            st.error("Direct Composition module not found. Please ensure direct_composition_integration.py is in the streamlit_app directory.")

st.markdown('</div>', unsafe_allow_html=True)
