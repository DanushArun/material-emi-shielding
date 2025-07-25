#!/usr/bin/env python3
"""Test database connection and basic operations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.database.connection import DatabaseManager

def test_connection():
    """Test database connection and basic operations."""
    print("Testing database connection...")
    
    try:
        # Initialize database manager
        db = DatabaseManager()
        print("✓ Database connection successful")
        
        # Get statistics
        stats = db.get_statistics()
        print("\nDatabase Statistics:")
        print(f"- Total materials: {stats.get('total_materials', {}).get('count', 0)}")
        print(f"- Total measurements: {stats.get('total_measurements', {}).get('count', 0)}")
        print(f"- Total sources: {stats.get('total_sources', {}).get('count', 0)}")
        
        # Test inserting a sample material
        print("\nTesting data insertion...")
        material_data = {
            'name': 'Test_Fe70Co30',
            'material_class': 'alloy',
            'composition': {'Fe': 70, 'Co': 30},
            'synthesis_method': 'Test method',
            'processing_temperature': 500,
            'processing_time': 2.0,
            'processing_atmosphere': 'air',
            'particle_size': 100,
            'morphology': 'nanoparticles'
        }
        
        material_id = db.insert_material(material_data)
        if material_id:
            print(f"✓ Successfully inserted material with ID: {material_id}")
            
            # Test inserting measurement
            measurement_data = {
                'material_id': material_id,
                'conductivity': 1e6,
                'relative_permeability': 100,
                'relative_permittivity': 1,
                'thickness': 2.0,
                'frequency': 1e9,
                'total_se': 75.5,
                'reflection_loss': 40.2,
                'absorption_loss': 35.3,
                'filler_loading': None,
                'filler_type': None,
                'measurement_standard': 'ASTM D4935',
                'source_doi': '10.1234/test.2024',
                'source_title': 'Test Paper',
                'source_year': 2024,
                'confidence_score': 0.95
            }
            
            measurement_id = db.insert_measurement(measurement_data)
            if measurement_id:
                print(f"✓ Successfully inserted measurement with ID: {measurement_id}")
        
        # Close connection
        db.close()
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)