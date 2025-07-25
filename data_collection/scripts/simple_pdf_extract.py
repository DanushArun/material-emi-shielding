#!/usr/bin/env python3
"""
Simple PDF text extraction for EMI shielding data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.database.connection import DatabaseManager
from pathlib import Path
import logging
import PyPDF2
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyPDF2."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(min(num_pages, 50)):  # Limit to first 50 pages
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""


def extract_emi_values_from_text(text: str) -> list:
    """Extract EMI shielding values from text using patterns."""
    measurements = []
    
    # Common patterns for EMI shielding data
    patterns = [
        # Pattern: "X showed/exhibited SE of Y dB at Z GHz"
        r'([A-Za-z0-9\-/\s]+?)\s+(?:showed|exhibited|demonstrated|achieved|has|had)\s+(?:SE|shielding effectiveness|EMI SE)\s+of\s+([\d.]+)\s*dB\s+at\s+([\d.]+)\s*(MHz|GHz)',
        
        # Pattern: "SE = Y dB at Z GHz for X"
        r'SE\s*[=:]\s*([\d.]+)\s*dB\s+at\s+([\d.]+)\s*(MHz|GHz)\s+for\s+([A-Za-z0-9\-/\s]+)',
        
        # Pattern: "X: Y dB (Z GHz)"
        r'([A-Za-z0-9\-/\s]+?):\s*([\d.]+)\s*dB\s*\(([\d.]+)\s*(MHz|GHz)\)',
        
        # Pattern: "shielding effectiveness of Y dB"
        r'([A-Za-z0-9\-/\s]+?)\s+(?:with|having)\s+(?:a\s+)?shielding effectiveness\s+of\s+([\d.]+)\s*dB',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if len(match.groups()) >= 4:  # Full pattern with frequency
                    material = match.group(1).strip()
                    se_value = float(match.group(2))
                    freq_value = float(match.group(3))
                    freq_unit = match.group(4).lower()
                    
                    # Convert frequency to Hz
                    freq_hz = freq_value * (1e9 if freq_unit == 'ghz' else 1e6)
                    
                    measurements.append({
                        'material': material,
                        'se': se_value,
                        'frequency': freq_hz,
                        'frequency_display': f"{freq_value} {freq_unit.upper()}"
                    })
                elif len(match.groups()) >= 2:  # Pattern without frequency
                    material = match.group(1).strip()
                    se_value = float(match.group(2))
                    
                    measurements.append({
                        'material': material,
                        'se': se_value,
                        'frequency': 1e9,  # Default 1 GHz
                        'frequency_display': "1 GHz (default)"
                    })
                    
            except (ValueError, IndexError):
                continue
    
    # Also look for thickness information
    thickness_pattern = r'thickness\s+(?:of\s+)?([\d.]+)\s*(mm|μm|um|cm)'
    thickness_matches = re.finditer(thickness_pattern, text, re.IGNORECASE)
    
    # Remove duplicates
    seen = set()
    unique_measurements = []
    for m in measurements:
        key = (m['material'][:20], m['se'], m['frequency'])  # Use first 20 chars of material
        if key not in seen:
            seen.add(key)
            unique_measurements.append(m)
    
    return unique_measurements


def main():
    """Process PDFs and extract EMI data."""
    pdf_dir = Path("/Users/danusharun/Documents/EMI-shielding/data_collection/papers/pdf")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found!")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    db = DatabaseManager()
    total_extracted = 0
    
    for pdf_path in pdf_files:
        logger.info(f"\nProcessing: {pdf_path.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            continue
        
        logger.info(f"Extracted {len(text)} characters of text")
        
        # Find EMI values
        measurements = extract_emi_values_from_text(text)
        logger.info(f"Found {len(measurements)} EMI measurements")
        
        # Save to database
        for m in measurements[:20]:  # Limit to 20 per PDF for now
            try:
                # Clean material name
                material_name = re.sub(r'\s+', ' ', m['material'])
                material_name = material_name[:100]  # Limit length
                
                # Insert material
                material_data = {
                    'name': material_name,
                    'material_class': 'unknown',
                    'composition': {},
                    'synthesis_method': 'Extracted from PDF',
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
                        'conductivity': None,
                        'relative_permeability': 1.0,
                        'relative_permittivity': 1.0,
                        'thickness': 1.0,  # Default 1mm
                        'frequency': m['frequency'],
                        'total_se': m['se'],
                        'reflection_loss': None,
                        'absorption_loss': None,
                        'filler_loading': None,
                        'filler_type': None,
                        'measurement_standard': None,
                        'source_doi': f'pdf_{pdf_path.stem}',
                        'source_title': pdf_path.stem.replace('_', ' '),
                        'source_year': 2024,
                        'confidence_score': 0.7
                    }
                    
                    measurement_id = db.insert_measurement(measurement_data)
                    if measurement_id:
                        total_extracted += 1
                        logger.info(f"  ✓ {material_name}: {m['se']} dB at {m['frequency_display']}")
                        
            except Exception as e:
                logger.error(f"Error saving measurement: {e}")
    
    # Summary
    stats = db.get_statistics()
    logger.info(f"\n{'='*50}")
    logger.info(f"Extraction complete!")
    logger.info(f"✓ Extracted {total_extracted} measurements")
    logger.info(f"✓ Database now has {stats.get('total_measurements', {}).get('count', 0)} total measurements")
    
    db.close()


if __name__ == "__main__":
    main()