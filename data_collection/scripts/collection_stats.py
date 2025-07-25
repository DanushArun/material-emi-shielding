#!/usr/bin/env python3
"""
Display current data collection statistics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collection.database.connection import DatabaseManager
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Display collection statistics."""
    logger.info("EMI Shielding Data Collection Statistics")
    logger.info("=" * 50)
    
    db = DatabaseManager()
    
    # Get basic statistics
    stats = db.get_statistics()
    
    logger.info(f"\nDatabase Overview:")
    logger.info(f"- Total materials: {stats.get('total_materials', {}).get('count', 0)}")
    logger.info(f"- Total measurements: {stats.get('total_measurements', {}).get('count', 0)}")
    logger.info(f"- Total sources: {stats.get('total_sources', {}).get('count', 0)}")
    logger.info(f"- Validated measurements: {stats.get('validated_measurements', {}).get('count', 0)}")
    
    # Frequency range
    freq_range = stats.get('frequency_range', {})
    if freq_range.get('min_freq') and freq_range.get('max_freq'):
        logger.info(f"\nFrequency Range:")
        logger.info(f"- Min: {freq_range['min_freq']/1e6:.1f} MHz")
        logger.info(f"- Max: {freq_range['max_freq']/1e9:.1f} GHz")
    
    # SE range
    se_range = stats.get('se_range', {})
    if se_range.get('min_se') is not None:
        logger.info(f"\nShielding Effectiveness Range:")
        logger.info(f"- Min: {se_range['min_se']:.1f} dB")
        logger.info(f"- Max: {se_range['max_se']:.1f} dB")
        logger.info(f"- Average: {se_range['avg_se']:.1f} dB")
    
    # Get top performing materials
    logger.info(f"\nTop Performing Materials (60+ dB):")
    top_materials = db.search_materials(min_se=60, frequency_range=(1e9, 10e9))
    
    if not top_materials.empty:
        for _, material in top_materials.head(10).iterrows():
            comp_str = str(material['composition']) if material['composition'] else ""
            logger.info(f"- {material['name']} {comp_str}: {material['avg_se']:.1f} dB (n={material['measurement_count']})")
    else:
        logger.info("- No high-performance materials found yet")
    
    # Material classes distribution
    query = """
        SELECT material_class, COUNT(DISTINCT id) as count
        FROM materials
        WHERE material_class IS NOT NULL
        GROUP BY material_class
        ORDER BY count DESC
    """
    
    class_dist = db.execute_query(query)
    if class_dist:
        logger.info(f"\nMaterial Classes:")
        for row in class_dist:
            logger.info(f"- {row['material_class']}: {row['count']} materials")
    
    # Recent additions
    query = """
        SELECT name, created_at
        FROM materials
        ORDER BY created_at DESC
        LIMIT 5
    """
    
    recent = db.execute_query(query)
    if recent:
        logger.info(f"\nRecent Additions:")
        for row in recent:
            logger.info(f"- {row['name']} ({row['created_at'].strftime('%Y-%m-%d %H:%M')})")
    
    db.close()
    
    logger.info("\n" + "=" * 50)
    logger.info("Run data collection scripts to add more data!")


if __name__ == "__main__":
    main()