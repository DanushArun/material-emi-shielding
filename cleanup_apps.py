#!/usr/bin/env python3
"""Clean up old app files"""

import os
import shutil

app_dir = "streamlit_app"

# Files to remove
files_to_remove = [
    "app.py",
    "app_enhanced.py", 
    "app_unified.py",
    "app_with_direct_composition.py",
    "direct_composition.py",
    "direct_composition_integration.py"
]

# Remove old files
for file in files_to_remove:
    filepath = os.path.join(app_dir, file)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Removed: {filepath}")

# Rename app_new.py to app.py
old_path = os.path.join(app_dir, "app_new.py")
new_path = os.path.join(app_dir, "app.py")
if os.path.exists(old_path):
    shutil.move(old_path, new_path)
    print(f"Renamed: {old_path} -> {new_path}")

print("\nCleanup complete!")
print("\nRemaining files in streamlit_app:")
for file in os.listdir(app_dir):
    print(f"  - {file}")