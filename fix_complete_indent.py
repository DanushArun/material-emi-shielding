#!/usr/bin/env python3
"""Fix indentation after COMPLETE REACTION section"""

# Read the file
with open('streamlit_app/app.py', 'r') as f:
    lines = f.readlines()

# Find the line with "# COMPLETE REACTION section"
complete_reaction_line = None
for i, line in enumerate(lines):
    if "# COMPLETE REACTION section - Outside of columns" in line:
        complete_reaction_line = i
        break

if complete_reaction_line is None:
    print("Could not find COMPLETE REACTION section")
    exit(1)

# Fix indentation from that line onwards
fixed_lines = []
for i, line in enumerate(lines):
    if i < complete_reaction_line:
        # Keep lines before COMPLETE REACTION as they are
        fixed_lines.append(line)
    else:
        # Remove 4 spaces of indentation from lines after COMPLETE REACTION
        if line.startswith('    ') and not line.strip() == '':
            fixed_lines.append(line[4:])
        else:
            fixed_lines.append(line)

# Write back
with open('streamlit_app/app.py', 'w') as f:
    f.writelines(fixed_lines)

print(f"Fixed indentation from line {complete_reaction_line + 1} onwards")