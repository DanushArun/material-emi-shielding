#!/usr/bin/env python3
"""
Test script to verify Direct Composition integration
"""

import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streamlit_app'))

print("Testing Direct Composition Integration...")

# Test imports
try:
    from streamlit_app.direct_composition_integration import render_direct_composition_section
    print("✓ Direct composition integration module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import direct composition integration: {e}")
    sys.exit(1)

# Test material database import
try:
    from src.materials.material_properties import material_db
    print("✓ Material database imported successfully")
except ImportError as e:
    print(f"✗ Failed to import material database: {e}")
    sys.exit(1)

# Test EMI calculator import
try:
    from src.physics.emi_calculations import emi_calculator
    print("✓ EMI calculator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import EMI calculator: {e}")
    sys.exit(1)

# Test app.py syntax
try:
    import ast
    with open('streamlit_app/app.py', 'r') as f:
        ast.parse(f.read())
    print("✓ app.py syntax is valid")
except SyntaxError as e:
    print(f"✗ Syntax error in app.py: {e}")
    sys.exit(1)

print("\n✅ All integration tests passed!")
print("\nTo run the app with Direct Composition mode:")
print("  streamlit run streamlit_app/app.py")
print("\nThe Direct Composition mode can be selected using the radio button at the top of the app.")