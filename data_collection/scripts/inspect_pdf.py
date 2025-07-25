#!/usr/bin/env python3
"""
Inspect PDF content to understand format.
"""

import PyPDF2
from pathlib import Path

def inspect_pdf(pdf_path: Path, pages_to_check: int = 3):
    """Inspect first few pages of PDF."""
    print(f"\nInspecting: {pdf_path.name}")
    print("=" * 50)
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"Total pages: {num_pages}")
            
            for i in range(min(pages_to_check, num_pages)):
                print(f"\n--- Page {i+1} ---")
                text = pdf_reader.pages[i].extract_text()
                
                # Show first 1000 characters
                print(text[:1000])
                
                # Look for keywords
                keywords = ['shielding', 'dB', 'GHz', 'MHz', 'effectiveness', 'SE', 'EMI']
                found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                if found_keywords:
                    print(f"\nFound keywords: {found_keywords}")
                
    except Exception as e:
        print(f"Error: {e}")


# Check first PDF
pdf_dir = Path("/Users/danusharun/Documents/EMI-shielding/data_collection/papers/pdf")
pdf_files = list(pdf_dir.glob("*.pdf"))

if pdf_files:
    # Check the EMI shielding PDF specifically
    emi_pdf = pdf_dir / "EMI shielding effectiveness of carbon.pdf"
    if emi_pdf.exists():
        inspect_pdf(emi_pdf, pages_to_check=5)
    else:
        inspect_pdf(pdf_files[0], pages_to_check=3)