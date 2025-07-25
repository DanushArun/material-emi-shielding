#!/usr/bin/env python3
"""
Download example EMI shielding papers from open access sources.
"""

import requests
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, filename: str, output_dir: Path) -> bool:
    """Download a file from URL."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        filepath = output_dir / filename
        
        with open(filepath, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"✓ Downloaded: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download {filename}: {e}")
        return False


def main():
    """Download example papers."""
    output_dir = Path("/Users/danusharun/Documents/EMI-shielding/data_collection/papers/pdf")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading example EMI shielding papers...")
    logger.info("=" * 50)
    
    # Example papers (you would need to replace with actual URLs)
    # These are placeholder examples - real papers would need proper URLs
    example_sources = [
        {
            'name': 'EMI_shielding_review_2023.pdf',
            'url': 'https://example.com/paper1.pdf',  # Replace with actual URL
            'description': 'Review of EMI shielding materials'
        },
        # Add more papers here
    ]
    
    logger.info("\n⚠️  Note: This script needs real paper URLs to work.")
    logger.info("\nTo get EMI shielding papers:")
    logger.info("1. Search Google Scholar: https://scholar.google.com")
    logger.info("   Query: 'EMI shielding effectiveness table filetype:pdf'")
    logger.info("2. Look for open access papers (PDF available)")
    logger.info("3. Download manually to the papers/pdf directory")
    logger.info("\nAlternatively, use your institutional access to download from:")
    logger.info("- IEEE Xplore")
    logger.info("- ScienceDirect")
    logger.info("- ACS Publications")
    logger.info("- Nature")
    
    logger.info(f"\nPDF directory: {output_dir}")
    logger.info("\nOnce you have PDFs, run:")
    logger.info("  python3 data_collection/scripts/batch_extract_pdfs.py")


if __name__ == "__main__":
    main()