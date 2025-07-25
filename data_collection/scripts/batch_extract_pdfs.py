#!/usr/bin/env python3
"""
Batch extract EMI shielding data from PDF files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.parsers.table_extractor import EMITableExtractor
from data_collection.database.connection import DatabaseManager
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf(pdf_path: Path, db: DatabaseManager, extractor: EMITableExtractor) -> dict:
    """
    Process a single PDF file and extract EMI data.
    
    Returns:
        Dictionary with extraction results
    """
    results = {
        'pdf_name': pdf_path.name,
        'pdf_path': str(pdf_path),
        'tables_found': 0,
        'measurements_extracted': 0,
        'materials_added': 0,
        'errors': []
    }
    
    try:
        logger.info(f"\nProcessing: {pdf_path.name}")
        
        # Extract EMI data from tables
        emi_data_list = extractor.extract_from_pdf(pdf_path)
        results['tables_found'] = len(emi_data_list)
        
        if not emi_data_list:
            logger.warning(f"No EMI data found in {pdf_path.name}")
            return results
        
        logger.info(f"Found {len(emi_data_list)} EMI measurements")
        
        # Process each extracted measurement
        for emi_data in emi_data_list:
            try:
                # Skip if missing critical data
                if not emi_data.material or emi_data.shielding_effectiveness is None:
                    continue
                
                # Prepare material data
                material_data = {
                    'name': emi_data.material,
                    'material_class': 'unknown',  # Could be enhanced with classification
                    'composition': emi_data.composition or {},
                    'synthesis_method': emi_data.synthesis_method or 'Unknown',
                    'processing_temperature': None,
                    'processing_time': None,
                    'processing_atmosphere': None,
                    'particle_size': emi_data.particle_size,
                    'morphology': 'Unknown'
                }
                
                # Check if material already exists
                existing_material = db.get_material_by_name(emi_data.material)
                if existing_material:
                    material_id = existing_material['id']
                else:
                    material_id = db.insert_material(material_data)
                    if material_id:
                        results['materials_added'] += 1
                
                if material_id:
                    # Prepare measurement data
                    measurement_data = {
                        'material_id': material_id,
                        'conductivity': emi_data.conductivity,
                        'relative_permeability': emi_data.permeability or 1.0,
                        'relative_permittivity': emi_data.permittivity or 1.0,
                        'thickness': emi_data.thickness or 1.0,
                        'frequency': emi_data.frequency or 1e9,  # Default 1 GHz
                        'total_se': emi_data.shielding_effectiveness,
                        'reflection_loss': emi_data.reflection_loss,
                        'absorption_loss': emi_data.absorption_loss,
                        'filler_loading': emi_data.filler_loading,
                        'filler_type': 'wt%' if emi_data.filler_loading else None,
                        'measurement_standard': emi_data.measurement_standard,
                        'source_doi': f'pdf_{pdf_path.stem}',
                        'source_title': pdf_path.stem.replace('_', ' ').title(),
                        'source_year': 2024,  # Could extract from PDF metadata
                        'confidence_score': 0.85
                    }
                    
                    measurement_id = db.insert_measurement(measurement_data)
                    if measurement_id:
                        results['measurements_extracted'] += 1
                        
            except Exception as e:
                error_msg = f"Error processing measurement: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        logger.info(f"✓ Extracted {results['measurements_extracted']} measurements from {pdf_path.name}")
        
    except Exception as e:
        error_msg = f"Error processing PDF {pdf_path.name}: {str(e)}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    return results


def main():
    """Main batch extraction function."""
    logger.info("EMI Shielding PDF Batch Extraction")
    logger.info("=" * 50)
    
    # Set up paths
    pdf_dir = Path("/Users/danusharun/Documents/EMI-shielding/data_collection/papers/pdf")
    output_dir = Path("/Users/danusharun/Documents/EMI-shielding/data_collection/output")
    output_dir.mkdir(exist_ok=True)
    
    # Check for PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"\nNo PDF files found in {pdf_dir}")
        logger.info("\nPlease add PDF files to:")
        logger.info(f"  {pdf_dir}")
        logger.info("\nSupported formats: .pdf")
        return
    
    logger.info(f"\nFound {len(pdf_files)} PDF files to process")
    
    # Initialize components
    db = DatabaseManager()
    extractor = EMITableExtractor()
    
    # Get initial statistics
    initial_stats = db.get_statistics()
    initial_materials = initial_stats.get('total_materials', {}).get('count', 0)
    initial_measurements = initial_stats.get('total_measurements', {}).get('count', 0)
    
    # Process each PDF
    all_results = []
    total_measurements = 0
    total_materials = 0
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        results = process_pdf(pdf_path, db, extractor)
        all_results.append(results)
        total_measurements += results['measurements_extracted']
        total_materials += results['materials_added']
    
    # Save extraction report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"extraction_report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'total_pdfs_processed': len(pdf_files),
        'total_measurements_extracted': total_measurements,
        'total_materials_added': total_materials,
        'pdf_results': all_results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Get final statistics
    final_stats = db.get_statistics()
    final_materials = final_stats.get('total_materials', {}).get('count', 0)
    final_measurements = final_stats.get('total_measurements', {}).get('count', 0)
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Extraction Summary:")
    logger.info(f"✓ Processed {len(pdf_files)} PDF files")
    logger.info(f"✓ Extracted {total_measurements} new measurements")
    logger.info(f"✓ Added {total_materials} new materials")
    
    logger.info(f"\nDatabase Growth:")
    logger.info(f"- Materials: {initial_materials} → {final_materials} (+{final_materials - initial_materials})")
    logger.info(f"- Measurements: {initial_measurements} → {final_measurements} (+{final_measurements - initial_measurements})")
    
    # Show PDFs with most data
    logger.info(f"\nTop PDFs by measurements extracted:")
    sorted_results = sorted(all_results, key=lambda x: x['measurements_extracted'], reverse=True)
    for result in sorted_results[:5]:
        if result['measurements_extracted'] > 0:
            logger.info(f"- {result['pdf_name']}: {result['measurements_extracted']} measurements")
    
    # Show any errors
    pdfs_with_errors = [r for r in all_results if r['errors']]
    if pdfs_with_errors:
        logger.warning(f"\nPDFs with errors ({len(pdfs_with_errors)}):")
        for result in pdfs_with_errors:
            logger.warning(f"- {result['pdf_name']}: {len(result['errors'])} errors")
    
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    db.close()
    logger.info("\n✓ Batch extraction completed!")
    
    # Next steps
    if total_measurements == 0:
        logger.info("\n💡 No measurements were extracted. Possible reasons:")
        logger.info("- PDFs might not contain tables with EMI data")
        logger.info("- Tables might be in image format (use graph digitizer)")
        logger.info("- Different table format than expected")
        logger.info("\nTry adding EMI shielding papers with data tables.")


if __name__ == "__main__":
    main()