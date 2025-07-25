#!/usr/bin/env python3
"""
Import EMI shielding data from CSV/Excel files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.database.connection import DatabaseManager
import pandas as pd
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_composition(comp_str: str) -> dict:
    """Parse composition string like 'Fe:70,Co:30' into dict."""
    if not comp_str or pd.isna(comp_str):
        return {}
    
    composition = {}
    try:
        # Handle different formats
        # Format 1: "Fe:70,Co:30"
        # Format 2: "Fe70Co30"
        # Format 3: "70% Fe, 30% Co"
        
        # Try colon-separated format first
        if ':' in comp_str:
            parts = comp_str.split(',')
            for part in parts:
                if ':' in part:
                    element, percentage = part.strip().split(':')
                    composition[element.strip()] = float(percentage)
        
        # Try percentage format
        elif '%' in comp_str:
            pattern = r'(\d+(?:\.\d+)?)\s*%\s*([A-Za-z]+)'
            matches = re.findall(pattern, comp_str)
            for percentage, element in matches:
                composition[element] = float(percentage)
        
        # Try element-number format (Fe70Co30)
        else:
            pattern = r'([A-Z][a-z]?)(\d+(?:\.\d+)?)'
            matches = re.findall(pattern, comp_str)
            for element, percentage in matches:
                composition[element] = float(percentage)
    
    except Exception as e:
        logger.warning(f"Could not parse composition '{comp_str}': {e}")
    
    return composition


def import_csv_data(csv_path: Path, db: DatabaseManager):
    """Import data from CSV file."""
    logger.info(f"Importing data from: {csv_path}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except:
        # Try Excel if CSV fails
        df = pd.read_excel(csv_path)
    
    logger.info(f"Found {len(df)} rows in file")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Standardize column names (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()
    
    # Map common column variations
    column_mappings = {
        'material': ['material', 'materials', 'sample', 'name', 'composite'],
        'composition': ['composition', 'comp', 'formula'],
        'thickness': ['thickness', 'thickness(mm)', 'thickness (mm)', 't (mm)', 't'],
        'frequency': ['frequency', 'frequency(ghz)', 'frequency (ghz)', 'freq', 'f (ghz)'],
        'se': ['se', 'se(db)', 'se (db)', 'shielding', 'shielding effectiveness', 'emi se'],
        'conductivity': ['conductivity', 'conductivity(s/m)', 'sigma', 'σ'],
        'permeability': ['permeability', 'mu', 'μ'],
        'source': ['source', 'reference', 'ref', 'doi', 'paper']
    }
    
    # Find actual column names
    found_columns = {}
    for target, variations in column_mappings.items():
        for col in df.columns:
            if any(var in col for var in variations):
                found_columns[target] = col
                break
    
    logger.info(f"Mapped columns: {found_columns}")
    
    # Check required columns
    required = ['material', 'se']
    missing = [col for col in required if col not in found_columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info("Required columns: material/name, se/shielding")
        return 0
    
    # Process each row
    imported_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Get material name
            material_name = str(row[found_columns['material']]).strip()
            if not material_name or pd.isna(material_name):
                continue
            
            # Get SE value
            se_value = float(row[found_columns['se']])
            
            # Get optional values with defaults
            thickness = float(row[found_columns.get('thickness', 1.0)]) if 'thickness' in found_columns else 1.0
            
            # Handle frequency (convert GHz to Hz if needed)
            freq_value = float(row[found_columns.get('frequency', 1.0)]) if 'frequency' in found_columns else 1.0
            # Assume GHz if value is small
            frequency = freq_value * 1e9 if freq_value < 1000 else freq_value
            
            # Parse composition if available
            composition = {}
            if 'composition' in found_columns:
                comp_str = row[found_columns['composition']]
                composition = parse_composition(str(comp_str))
            
            # Get other optional values
            conductivity = float(row[found_columns['conductivity']]) if 'conductivity' in found_columns and not pd.isna(row[found_columns['conductivity']]) else None
            permeability = float(row[found_columns['permeability']]) if 'permeability' in found_columns and not pd.isna(row[found_columns['permeability']]) else 1.0
            source = str(row[found_columns['source']]) if 'source' in found_columns else csv_path.stem
            
            # Insert material
            material_data = {
                'name': material_name,
                'material_class': 'imported',
                'composition': composition,
                'synthesis_method': 'Imported from CSV',
                'processing_temperature': None,
                'processing_time': None,
                'processing_atmosphere': None,
                'particle_size': None,
                'morphology': 'Unknown'
            }
            
            # Check if material exists
            existing = db.get_material_by_name(material_name)
            if existing:
                material_id = existing['id']
            else:
                material_id = db.insert_material(material_data)
            
            if material_id:
                # Insert measurement
                measurement_data = {
                    'material_id': material_id,
                    'conductivity': conductivity,
                    'relative_permeability': permeability,
                    'relative_permittivity': 1.0,
                    'thickness': thickness,
                    'frequency': frequency,
                    'total_se': se_value,
                    'reflection_loss': None,
                    'absorption_loss': None,
                    'filler_loading': None,
                    'filler_type': None,
                    'measurement_standard': None,
                    'source_doi': source,
                    'source_title': f'Imported from {csv_path.name}',
                    'source_year': 2024,
                    'confidence_score': 0.9
                }
                
                measurement_id = db.insert_measurement(measurement_data)
                if measurement_id:
                    imported_count += 1
                    logger.info(f"  ✓ {material_name}: {se_value} dB at {frequency/1e9:.1f} GHz")
        
        except Exception as e:
            logger.error(f"Error on row {idx}: {e}")
            continue
    
    return imported_count


def create_template():
    """Create a template CSV file."""
    template_data = {
        'Material': ['Fe70Co30/PVDF', 'MXene/PVA', 'Graphene/Epoxy', 'CNT/PMMA'],
        'Composition': ['Fe:70,Co:30', 'MXene:10', 'Graphene:5', 'CNT:2'],
        'Thickness(mm)': [2.0, 0.5, 1.0, 3.0],
        'Frequency(GHz)': [8.2, 10.0, 12.0, 8.2],
        'SE(dB)': [65.5, 92.0, 45.0, 55.0],
        'Conductivity(S/m)': [1e5, 1e4, 1e3, 5e3],
        'Source': ['Zhang2023', 'Li2024', 'Wang2023', 'Chen2024']
    }
    
    df = pd.DataFrame(template_data)
    template_path = Path('data_collection/templates/emi_data_template.csv')
    template_path.parent.mkdir(exist_ok=True)
    
    df.to_csv(template_path, index=False)
    logger.info(f"Created template at: {template_path}")
    
    # Also create Excel version
    excel_path = template_path.with_suffix('.xlsx')
    df.to_excel(excel_path, index=False)
    logger.info(f"Created Excel template at: {excel_path}")


def main():
    """Main import function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import EMI data from CSV/Excel files')
    parser.add_argument('files', nargs='*', help='CSV or Excel files to import')
    parser.add_argument('--template', action='store_true', help='Create template file')
    parser.add_argument('--dir', help='Import all CSV/Excel files from directory')
    
    args = parser.parse_args()
    
    if args.template:
        create_template()
        return
    
    db = DatabaseManager()
    
    # Get initial stats
    initial_stats = db.get_statistics()
    initial_count = initial_stats.get('total_measurements', {}).get('count', 0)
    
    # Collect files to import
    files_to_import = []
    
    if args.files:
        files_to_import.extend([Path(f) for f in args.files])
    
    if args.dir:
        dir_path = Path(args.dir)
        files_to_import.extend(dir_path.glob('*.csv'))
        files_to_import.extend(dir_path.glob('*.xlsx'))
    
    if not files_to_import and not args.template:
        logger.info("No files specified. Looking in default locations...")
        # Check common locations
        locations = [
            Path('data_collection/data'),
            Path('data_collection/import'),
            Path('.')
        ]
        
        for loc in locations:
            if loc.exists():
                files_to_import.extend(loc.glob('*.csv'))
                files_to_import.extend(loc.glob('*.xlsx'))
    
    if not files_to_import:
        logger.warning("No CSV or Excel files found to import")
        logger.info("\nUsage:")
        logger.info("  python import_csv.py data.csv")
        logger.info("  python import_csv.py --dir ./data_folder")
        logger.info("  python import_csv.py --template")
        return
    
    # Import files
    total_imported = 0
    
    for file_path in files_to_import:
        if file_path.exists():
            count = import_csv_data(file_path, db)
            total_imported += count
            logger.info(f"Imported {count} measurements from {file_path.name}\n")
    
    # Final stats
    final_stats = db.get_statistics()
    final_count = final_stats.get('total_measurements', {}).get('count', 0)
    
    logger.info("="*50)
    logger.info(f"Import Summary:")
    logger.info(f"✓ Processed {len(files_to_import)} files")
    logger.info(f"✓ Imported {total_imported} measurements")
    logger.info(f"✓ Database: {initial_count} → {final_count} measurements")
    
    db.close()


if __name__ == "__main__":
    main()