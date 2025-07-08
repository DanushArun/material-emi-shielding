"""
Direct Composition mode implementation for EMI Shield Designer.
"""

import streamlit as st
from typing import Dict


class DirectCompositionManager:
    """Manages direct percentage-based composition input."""
    
    def __init__(self):
        """Initialize the direct composition manager."""
        if 'direct_composition' not in st.session_state:
            st.session_state.direct_composition = {}
    
    def add_element(self, element: str, percentage: float):
        """Add or update an element in the composition."""
        st.session_state.direct_composition[element] = percentage
    
    def remove_element(self, element: str):
        """Remove an element from the composition."""
        if element in st.session_state.direct_composition:
            del st.session_state.direct_composition[element]
    
    def get_total_percentage(self) -> float:
        """Get the total percentage of all elements."""
        return sum(st.session_state.direct_composition.values())
    
    def is_valid(self) -> bool:
        """Check if the composition totals to 100%."""
        total = self.get_total_percentage()
        return abs(total - 100.0) < 0.01
    
    def normalize(self):
        """Normalize the composition to total 100%."""
        total = self.get_total_percentage()
        if total > 0:
            for element in st.session_state.direct_composition:
                st.session_state.direct_composition[element] *= (100.0 / total)
    
    def get_composition(self) -> Dict[str, float]:
        """Get the current composition."""
        return st.session_state.direct_composition.copy()
    
    def clear(self):
        """Clear the composition."""
        st.session_state.direct_composition = {}
    
    def create_reaction_molecule(self) -> Dict:
        """Create a molecule object for the reaction engine."""
        composition = self.get_composition()
        
        # Import here to avoid circular imports
        from src.materials.material_properties import material_db
        
        # Calculate weighted molecular weight
        molecular_weight = 0
        for element, percentage in composition.items():
            elem_data = material_db.get_material(element)
            if elem_data and 'atomic_weight' in elem_data:
                molecular_weight += elem_data['atomic_weight'] * percentage / 100
            else:
                molecular_weight += 50 * percentage / 100  # Default weight
        
        return {
            'type': 'direct',
            'composition': composition,
            'formula': 'Direct Composition',
            'coefficient': 1,
            'molecular_weight': molecular_weight,
            'display_name': self._create_display_name(composition)
        }
    
    def _create_display_name(self, composition: Dict[str, float]) -> str:
        """Create a display name for the composition."""
        # Sort by percentage (descending)
        sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 3 elements
        display_parts = []
        for elem, pct in sorted_comp[:3]:
            display_parts.append(f"{elem}({pct:.1f}%)")
        
        if len(sorted_comp) > 3:
            display_parts.append("...")
        
        return " ".join(display_parts)


def render_direct_composition_ui(material_db):
    """Render the Direct Composition input interface."""
    dc_manager = DirectCompositionManager()
    
    # Header
    st.markdown("""
    <div style="margin: var(--space-6) 0 var(--space-4) 0;">
        <h3 style="
            font-size: var(--font-2xl);
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 var(--space-4) 0;
            text-align: center;
        ">Percentage Composition Input</h3>
        <p style="text-align: center; color: var(--text-secondary); margin-top: var(--space-2);">
            Add elements and specify their weight percentages
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
                dc_manager.add_element(new_element, new_percentage)
                st.rerun()
    
    # Display current composition
    if st.session_state.direct_composition:
        # Calculate total
        total = dc_manager.get_total_percentage()
        
        # Show total with color coding
        color = "green" if dc_manager.is_valid() else "red"
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
                    dc_manager.add_element(element, new_pct)
            
            with col3:
                if st.button("🗑️", key=f"del_direct_{element}"):
                    dc_manager.remove_element(element)
                    st.rerun()
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Normalize button
            if not dc_manager.is_valid() and total > 0:
                if st.button("⚖️ Normalize to 100%", use_container_width=True):
                    dc_manager.normalize()
                    st.rerun()
        
        with col2:
            # Add to reaction button
            if dc_manager.is_valid():
                if st.button("➕ Add to Reaction", type="primary", use_container_width=True):
                    # Add to reaction engine
                    molecule = dc_manager.create_reaction_molecule()
                    st.session_state.reaction_engine.molecules.append(molecule)
                    dc_manager.clear()
                    st.rerun()
            else:
                st.button("➕ Add to Reaction", disabled=True, use_container_width=True,
                         help="Total must equal 100%")
        
        with col3:
            if st.button("🔄 Clear", use_container_width=True):
                dc_manager.clear()
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
            7. Click **Add to Reaction** when total equals 100%
            """)
    
    return dc_manager