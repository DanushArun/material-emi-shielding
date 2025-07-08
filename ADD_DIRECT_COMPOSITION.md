# How to Add Direct Composition Mode to app.py

## Quick Fix

The Direct Composition mode is already partially integrated but not showing. Here's how to fix it:

### 1. Find this section (around line 1140):
```python
with main_container:
    # BUILD YOUR MOLECULE section
    st.markdown("""
    <div style="margin: var(--space-6) 0 var(--space-4) 0;">
```

### 2. Add the conditional check:
Change:
```python
with main_container:
    # BUILD YOUR MOLECULE section
```

To:
```python
with main_container:
    if input_mode == " Molecular Builder":
        # BUILD YOUR MOLECULE section
```

### 3. Add Direct Composition after molecular section

After the molecular presets section (around line 1580), before `# COMPLETE REACTION section`, add:

```python
    else:  # Direct Composition mode
        # DIRECT COMPOSITION section
        from direct_composition_integration import render_direct_composition_section
        render_direct_composition_section(st, material_db)
```

### 4. Update the "How To Use" section (line 1107):

Add this line after line 1107:
```python
        <p style="margin: 0 0 var(--space-2) 0;">1b. OR use Direct Composition mode to enter percentages directly (e.g., 70% Fe, 30% C)</p>
```

## Alternative: Use the Demo App

If you want to see it working immediately:
```bash
streamlit run demo_ml_system.py
```

This includes the Direct Composition mode in the second tab.

## Full Integration

For a complete integration, the Direct Composition UI code is in:
- `streamlit_app/direct_composition.py` - Complete UI component
- `streamlit_app/direct_composition_integration.py` - Integration helper

The mode selector is already in app.py (line 1129-1135), it just needs the conditional rendering to be properly implemented.