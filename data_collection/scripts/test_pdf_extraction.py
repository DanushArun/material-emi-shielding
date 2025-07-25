#!/usr/bin/env python3
"""
Test PDF extraction capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.parsers.table_extractor import EMITableExtractor
from data_collection.database.connection import DatabaseManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_extraction():
    """Test extracting EMI data from text."""
    logger.info("Testing text extraction...")
    
    extractor = EMITableExtractor()
    
    # Sample text with EMI data
    sample_texts = [
        "The Fe70Ni30 composite showed excellent shielding effectiveness of 85 dB at 1 GHz with a thickness of 2 mm.",
        "Cu/PVDF composite achieved SE of 72 dB at 2.4 GHz.",
        "The MXene film exhibited shielding effectiveness of 92 dB at 10 GHz for 25 μm thickness.",
        "Carbon nanotube buckypaper demonstrated SE of 60 dB at 8.2 GHz with only 0.5 mm thickness.",
        "Graphene/polymer composite showed 45 dB shielding at 12 GHz."
    ]
    
    db = DatabaseManager()
    
    all_data = []
    for text in sample_texts:
        logger.info(f"\nProcessing: {text}")
        extracted = extractor.extract_from_text(text)
        
        for data in extracted:
            logger.info(f"  Extracted: {data.material} - {data.shielding_effectiveness} dB at {data.frequency/1e9:.1f} GHz")
            all_data.append(data)
            
            # Insert into database
            try:
                # First insert material
                material_data = {
                    'name': data.material,
                    'material_class': 'composite',
                    'composition': data.composition or {},
                    'synthesis_method': 'Unknown',
                    'processing_temperature': None,
                    'processing_time': None,
                    'processing_atmosphere': None,
                    'particle_size': None,
                    'morphology': 'Unknown'
                }
                
                material_id = db.insert_material(material_data)
                
                if material_id:
                    # Insert measurement
                    measurement_data = {
                        'material_id': material_id,
                        'conductivity': data.conductivity,
                        'relative_permeability': data.permeability or 1.0,
                        'relative_permittivity': data.permittivity or 1.0,
                        'thickness': data.thickness or 1.0,
                        'frequency': data.frequency,
                        'total_se': data.shielding_effectiveness,
                        'reflection_loss': data.reflection_loss,
                        'absorption_loss': data.absorption_loss,
                        'filler_loading': data.filler_loading,
                        'filler_type': None,
                        'measurement_standard': data.measurement_standard,
                        'source_doi': 'text_extraction_test',
                        'source_title': 'Text Extraction Test',
                        'source_year': 2024,
                        'confidence_score': 0.8
                    }
                    
                    measurement_id = db.insert_measurement(measurement_data)
                    if measurement_id:
                        logger.info(f"  ✓ Saved to database")
                        
            except Exception as e:
                logger.error(f"  Failed to save: {e}")
    
    logger.info(f"\n✓ Extracted {len(all_data)} measurements from text")
    
    # Get statistics
    stats = db.get_statistics()
    logger.info(f"\nDatabase now contains:")
    logger.info(f"- Total materials: {stats.get('total_materials', {}).get('count', 0)}")
    logger.info(f"- Total measurements: {stats.get('total_measurements', {}).get('count', 0)}")
    
    db.close()


def test_table_parsing():
    """Test parsing EMI data from tables."""
    logger.info("\n\nTesting table parsing...")
    
    import pandas as pd
    
    # Create a sample table
    table_data = {
        'Material': ['Fe3O4/PVDF', 'Ni/Epoxy', 'Cu-coated fabric', 'Graphene foam'],
        'Thickness (mm)': [2.0, 1.5, 0.5, 3.0],
        'Frequency (GHz)': [1.0, 2.4, 8.2, 10.0],
        'SE (dB)': [65.5, 72.3, 85.0, 92.5],
        'Conductivity (S/m)': ['1e5', '5e4', '1e6', '2e3']
    }
    
    df = pd.DataFrame(table_data)
    logger.info("\nSample table:")
    print(df.to_string(index=False))
    
    # Parse the table
    extractor = EMITableExtractor()
    emi_data = extractor._parse_emi_table(df)
    
    logger.info(f"\n✓ Extracted {len(emi_data)} measurements from table")
    
    for data in emi_data:
        logger.info(f"  {data.material}: {data.shielding_effectiveness} dB at {data.frequency/1e9} GHz, {data.thickness} mm thick")


if __name__ == "__main__":
    test_text_extraction()
    test_table_parsing()