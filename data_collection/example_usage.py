"""
Example usage of the EMI shielding data collection system.
"""

import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime

# Import our modules
from scrapers.base_scraper import BaseScraper
from scrapers.materials_project_api import MaterialsProjectScraper
from parsers.table_extractor import EMITableExtractor
from parsers.graph_digitizer import EMIGraphDigitizer
from database.connection import DatabaseManager


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExampleEMIScraper(BaseScraper):
    """Example implementation of an EMI scraper."""
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Example search implementation.
        In real implementation, this would search scientific databases.
        """
        # For demonstration, return mock results
        return [
            {
                'title': f'EMI Shielding Study {i+1}',
                'url': f'https://example.com/paper{i+1}.pdf',
                'doi': f'10.1234/example.{i+1}',
                'year': 2023
            }
            for i in range(max_results)
        ]
    
    async def extract_data(self, source: Dict) -> Optional[Dict]:
        """
        Example data extraction.
        In real implementation, this would download and parse the paper.
        """
        # For demonstration, return mock EMI data
        return {
            'material': 'Fe70Co30/PVDF composite',
            'conductivity': 1e5,  # S/m
            'thickness': 2.0,  # mm
            'frequency': 1e9,  # Hz (1 GHz)
            'total_se': 65.5,  # dB
            'reflection_loss': 35.2,  # dB
            'absorption_loss': 30.3,  # dB
            'source_doi': source['doi']
        }


def demo_pdf_extraction():
    """Demonstrate PDF table extraction."""
    logger.info("=== PDF Table Extraction Demo ===")
    
    # Initialize extractor
    extractor = EMITableExtractor()
    
    # Example: Extract from a PDF (you would provide actual PDF path)
    pdf_path = Path("example_papers/emi_study.pdf")
    
    if pdf_path.exists():
        # Extract EMI data from tables
        emi_data = extractor.extract_from_pdf(pdf_path)
        
        logger.info(f"Extracted {len(emi_data)} EMI measurements from PDF")
        
        # Display first few results
        for i, data in enumerate(emi_data[:3]):
            logger.info(f"\nMeasurement {i+1}:")
            logger.info(f"  Material: {data.material}")
            logger.info(f"  SE: {data.shielding_effectiveness} dB at {data.frequency} Hz")
            logger.info(f"  Thickness: {data.thickness} mm")
    else:
        logger.warning(f"PDF file not found: {pdf_path}")
        
    # Demonstrate text extraction
    sample_text = """
    The Fe-Ni composite showed excellent shielding effectiveness of 85 dB at 1 GHz
    with a thickness of 2 mm. The Cu/PVDF composite achieved SE of 72 dB at 2.4 GHz.
    """
    
    text_data = extractor.extract_from_text(sample_text)
    logger.info(f"\nExtracted {len(text_data)} measurements from text")
    for data in text_data:
        logger.info(f"  {data.material}: {data.shielding_effectiveness} dB at {data.frequency/1e9} GHz")


def demo_materials_project():
    """Demonstrate Materials Project API usage."""
    logger.info("\n=== Materials Project API Demo ===")
    
    try:
        # Initialize API client (requires MP_API_KEY environment variable)
        mp_scraper = MaterialsProjectScraper()
        
        # Search for conductive materials
        logger.info("Searching for conductive materials...")
        conductive_df = mp_scraper.fetch_conductive_materials(min_conductivity=1e6, limit=5)
        
        if not conductive_df.empty:
            logger.info(f"\nFound {len(conductive_df)} conductive materials:")
            print(conductive_df[['formula', 'conductivity', 'density']].to_string(index=False))
        
        # Search for magnetic materials
        logger.info("\nSearching for magnetic materials...")
        magnetic_df = mp_scraper.fetch_magnetic_materials(limit=5)
        
        if not magnetic_df.empty:
            logger.info(f"\nFound {len(magnetic_df)} magnetic materials:")
            print(magnetic_df[['formula', 'magnetic_ordering', 'estimated_permeability']].to_string(index=False))
        
        # Estimate EMI performance for a material
        if not conductive_df.empty:
            material = conductive_df.iloc[0].to_dict()
            performance = mp_scraper.estimate_emi_performance(
                material,
                thickness=1.0,  # mm
                frequency=1e9   # 1 GHz
            )
            
            logger.info(f"\nEstimated EMI performance for {material['formula']}:")
            logger.info(f"  Total SE: {performance['estimated_se']:.1f} dB")
            logger.info(f"  Reflection Loss: {performance['reflection_loss']:.1f} dB")
            logger.info(f"  Absorption Loss: {performance['absorption_loss']:.1f} dB")
            
    except Exception as e:
        logger.error(f"Materials Project demo failed: {e}")
        logger.info("Make sure to set MP_API_KEY environment variable")


async def demo_web_scraping():
    """Demonstrate web scraping with base scraper."""
    logger.info("\n=== Web Scraping Demo ===")
    
    # Initialize example scraper
    scraper = ExampleEMIScraper()
    
    # Define search queries
    queries = [
        "EMI shielding effectiveness composite materials",
        "electromagnetic interference carbon nanotubes",
        "MXene EMI shielding"
    ]
    
    # Run scraper
    all_data = await scraper.run(queries, max_results_per_query=5)
    
    logger.info(f"\nScraped {len(all_data)} EMI measurements")
    
    # Display statistics
    stats = scraper.get_statistics()
    logger.info(f"\nScraping statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def demo_database_operations():
    """Demonstrate database operations."""
    logger.info("\n=== Database Operations Demo ===")
    
    try:
        # Initialize database manager
        db = DatabaseManager()
        
        # Insert a material
        material_data = {
            'name': 'Fe70Co30/PVDF',
            'material_class': 'composite',
            'composition': {'Fe': 70, 'Co': 30},
            'synthesis_method': 'Ball milling followed by hot pressing',
            'processing_temperature': 180,
            'processing_time': 2.0,
            'particle_size': 50,
            'morphology': 'nanoparticles'
        }
        
        material_id = db.insert_material(material_data)
        logger.info(f"Inserted material with ID: {material_id}")
        
        # Insert measurement
        measurement_data = {
            'material_id': material_id,
            'conductivity': 1e5,
            'relative_permeability': 100,
            'thickness': 2.0,
            'frequency': 1e9,
            'total_se': 65.5,
            'reflection_loss': 35.2,
            'absorption_loss': 30.3,
            'measurement_standard': 'ASTM D4935',
            'source_doi': '10.1234/example.2023',
            'source_title': 'EMI Shielding Study Example',
            'source_year': 2023,
            'confidence_score': 0.95
        }
        
        measurement_id = db.insert_measurement(measurement_data)
        logger.info(f"Inserted measurement with ID: {measurement_id}")
        
        # Query database
        stats = db.get_statistics()
        logger.info("\nDatabase statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Search materials
        results_df = db.search_materials(min_se=60, frequency_range=(1e9, 10e9))
        if not results_df.empty:
            logger.info(f"\nFound {len(results_df)} high-performance materials")
        
        db.close()
        
    except Exception as e:
        logger.error(f"Database demo failed: {e}")
        logger.info("Make sure PostgreSQL is running and configured")


def create_example_files():
    """Create example files for demonstration."""
    # Create directories
    Path("example_papers").mkdir(exist_ok=True)
    Path("data_collection/output").mkdir(exist_ok=True)
    
    # Create example environment file
    env_content = """# Example environment configuration
# Copy this to .env and fill in your actual values

# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=emi_shielding
DB_USER=postgres
DB_PASSWORD=your_password

# API Keys
MP_API_KEY=your_materials_project_api_key

# Scraping configuration
RATE_LIMIT_CALLS=10
RATE_LIMIT_PERIOD=60
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    logger.info("Created example configuration files")


async def main():
    """Run all demonstrations."""
    logger.info("EMI Shielding Data Collection System - Demo")
    logger.info("=" * 50)
    
    # Create example files
    create_example_files()
    
    # Run demonstrations
    demo_pdf_extraction()
    demo_materials_project()
    await demo_web_scraping()
    demo_database_operations()
    
    logger.info("\n" + "=" * 50)
    logger.info("Demo completed!")
    logger.info("\nNext steps:")
    logger.info("1. Set up PostgreSQL database and run schema.sql")
    logger.info("2. Get Materials Project API key from https://materialsproject.org")
    logger.info("3. Configure .env file with your settings")
    logger.info("4. Implement specific scrapers for IEEE, ScienceDirect, etc.")
    logger.info("5. Start collecting EMI shielding data!")


if __name__ == "__main__":
    asyncio.run(main())