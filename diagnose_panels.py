#!/usr/bin/env python3
"""
Quick diagnostic: Why are 0 panels detected?
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import fitz  # PyMuPDF
    from PIL import Image
    import numpy as np
    print("✓ Dependencies loaded")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

def diagnose_pdf(pdf_path: str):
    """Diagnose why panels aren't detected"""
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"✗ PDF not found: {pdf_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Diagnosing: {pdf_path.name}")
    print(f"{'='*70}\n")
    
    # Open PDF
    try:
        doc = fitz.open(str(pdf_path))
        print(f"✓ PDF opened: {len(doc)} pages\n")
    except Exception as e:
        print(f"✗ Cannot open PDF: {e}")
        return
    
    # Check first 3 pages
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        print(f"📄 Page {page_num + 1}")
        print(f"   Size: {page.rect.width:.0f} x {page.rect.height:.0f} points")
        
        # Render at 150 DPI
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img_array = np.array(img)
        
        print(f"   Rendered: {img.width} x {img.height} pixels")
        print(f"   Total area: {img.width * img.height:,} pixels")
        
        # Check against thresholds
        area = img.width * img.height
        if area < 10000:
            print(f"   ⚠️  Too small for MIN_PANEL_AREA=10000")
        elif area < 80000:
            print(f"   ⚠️  Too small for default MIN_PANEL_AREA=80000")
            print(f"   ✓  Would be detected with MIN_PANEL_AREA=10000")
        else:
            print(f"   ✓  Large enough for detection")
        
        # Check image stats
        mean_val = img_array.mean()
        std_val = img_array.std()
        print(f"   Brightness: {mean_val:.1f} (std: {std_val:.1f})")
        
        if std_val < 10:
            print(f"   ⚠️  Very low variance (blank/uniform image?)")
        
        print()
    
    doc.close()
    
    print(f"{'='*70}")
    print("Recommendations:")
    print(f"{'='*70}")
    print("1. If areas < 80,000: Lower MIN_PANEL_AREA to 10000")
    print("2. If low variance: Check if PDF has actual images")
    print("3. If very small: Increase DPI from 150 to 200")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_panels.py <pdf_path>")
        print("\nExample:")
        print('  python3 diagnose_panels.py "/Users/zijiefeng/Desktop/Guo\'s lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf"')
        sys.exit(1)
    
    diagnose_pdf(sys.argv[1])

