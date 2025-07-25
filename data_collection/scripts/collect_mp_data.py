#!/usr/bin/env python3
"""
Collect EMI-relevant materials from Materials Project database.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.scrapers.materials_project_api import MaterialsProjectScraper
from data_collection.database.connection import DatabaseManager
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Collect materials data from Materials Project."""
    logger.info("Starting Materials Project data collection...")
    
    # Initialize components
    try:
        mp = MaterialsProjectScraper()
        logger.info("✓ Materials Project API initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Materials Project API: {e}")
        logger.info("\n⚠️  You need to set up your Materials Project API key:")
        logger.info("1. Go to https://materialsproject.org")
        logger.info("2. Create a free account")
        logger.info("3. Go to Dashboard → API → Generate API Key")
        logger.info("4. Add to .env file: MP_API_KEY=your_key_here")
        return
    
    db = DatabaseManager()
    logger.info("✓ Database connection established")
    
    # Search for different types of materials
    material_searches = [
        {
            'name': 'Conductive Metals',
            'params': {
                'elements': ['Fe', 'Co', 'Ni', 'Cu', 'Al', 'Ag'],
                'is_metal': True,
                'max_elements': 3
            }
        },
        {
            'name': 'Magnetic Materials',
            'params': {
                'elements': ['Fe', 'Co', 'Ni', 'Mn', 'Cr'],
                'magnetic_only': True,
                'max_elements': 4
            }
        },
        {
            'name': 'Carbon-based Materials',
            'params': {
                'elements': ['C', 'Fe', 'Co', 'Ni'],
                'max_elements': 3
            }
        }
    ]
    
    total_materials = 0
    total_measurements = 0
    
    for search in material_searches:
        logger.info(f"\nSearching for {search['name']}...")
        
        try:
            # Search materials
            materials = mp.search_emi_materials(**search['params'])
            logger.info(f"Found {len(materials)} materials")
            
            # Process each material
            for material in tqdm(materials[:100], desc=f"Processing {search['name']}"):
                try:
                    # Insert material into database
                    material_data = {
                        'name': material['formula'],
                        'material_class': 'metal' if material.get('is_metal') else 'ceramic',
                        'composition': material.get('composition', {}),
                        'synthesis_method': 'Materials Project prediction',
                        'processing_temperature': None,
                        'processing_time': None,
                        'processing_atmosphere': None,
                        'particle_size': None,
                        'morphology': 'bulk'
                    }
                    
                    material_id = db.insert_material(material_data)
                    
                    if material_id:
                        total_materials += 1
                        
                        # Estimate EMI performance at different frequencies
                        frequencies = [1e6, 10e6, 100e6, 1e9, 10e9]  # 1 MHz to 10 GHz
                        thicknesses = [0.1, 0.5, 1.0, 2.0, 5.0]  # mm
                        
                        for freq in frequencies:
                            for thickness in thicknesses:
                                performance = mp.estimate_emi_performance(
                                    material,
                                    thickness=thickness,
                                    frequency=freq
                                )
                                
                                # Insert measurement
                                measurement_data = {
                                    'material_id': material_id,
                                    'conductivity': material.get('estimated_conductivity', 1e3),
                                    'relative_permeability': material.get('estimated_permeability', 1.0),
                                    'relative_permittivity': 1.0,  # Default
                                    'thickness': thickness,
                                    'frequency': freq,
                                    'total_se': performance['estimated_se'],
                                    'reflection_loss': performance['reflection_loss'],
                                    'absorption_loss': performance['absorption_loss'],
                                    'filler_loading': None,
                                    'filler_type': None,
                                    'measurement_standard': 'Theoretical calculation',
                                    'source_doi': f"mp-{material['material_id']}",
                                    'source_title': 'Materials Project Database',
                                    'source_year': 2024,
                                    'confidence_score': 0.7  # Lower confidence for estimates
                                }
                                
                                measurement_id = db.insert_measurement(measurement_data)
                                if measurement_id:
                                    total_measurements += 1
                                    
                except Exception as e:
                    logger.warning(f"Failed to process material {material.get('formula', 'Unknown')}: {e}")
                    
        except Exception as e:
            logger.error(f"Search failed for {search['name']}: {e}")
    
    # Get final statistics
    stats = db.get_statistics()
    
    logger.info("\n" + "="*50)
    logger.info("Collection Summary:")
    logger.info(f"✓ Added {total_materials} new materials")
    logger.info(f"✓ Added {total_measurements} EMI measurements")
    logger.info(f"\nDatabase now contains:")
    logger.info(f"- Total materials: {stats.get('total_materials', {}).get('count', 0)}")
    logger.info(f"- Total measurements: {stats.get('total_measurements', {}).get('count', 0)}")
    
    db.close()
    logger.info("\n✓ Data collection completed!")


if __name__ == "__main__":
    main()