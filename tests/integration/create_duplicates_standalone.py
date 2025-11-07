#!/usr/bin/env python3
"""
Standalone script to extract pages from PDF and create duplicates.
Uses PyMuPDF if available, otherwise provides instructions.
"""

import sys
from pathlib import Path
from PIL import Image
import shutil

def extract_pages_simple(pdf_path: Path, output_dir: Path, max_pages: int = 5):
    """Extract pages from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF (fitz) not installed.")
        print("Please install: pip install pymupdf")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    
    doc = fitz.open(pdf_path)
    num_pages = min(len(doc), max_pages)
    
    print(f"Extracting {num_pages} pages from PDF...")
    for page_num in range(num_pages):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))  # 150 DPI
        page_path = output_dir / f"page_{page_num+1:03d}.png"
        pix.save(str(page_path))
        pages.append(page_path)
        print(f"  Extracted page {page_num+1}")
    
    doc.close()
    return pages


def create_duplicates_from_pages(pages: list, output_dir: Path):
    """Create duplicates from page images."""
    if len(pages) < 3:
        print(f"ERROR: Need at least 3 pages, got {len(pages)}")
        return {}
    
    duplicates_dir = output_dir / "intentional_duplicates"
    duplicates_dir.mkdir(parents=True, exist_ok=True)
    
    duplicates_info = {}
    
    # Use first 3 pages for WB, confocal, IHC
    print("\nCreating duplicate variants...")
    
    # Page 1: WB
    print(f"\n1. Creating WB duplicates from: {pages[0].name}")
    wb_variants = create_variants(pages[0], duplicates_dir / "WB", "WB")
    duplicates_info['WB'] = wb_variants
    
    # Page 2: Confocal
    print(f"\n2. Creating confocal duplicates from: {pages[1].name}")
    confocal_variants = create_variants(pages[1], duplicates_dir / "confocal", "confocal")
    duplicates_info['confocal'] = confocal_variants
    
    # Page 3: IHC
    print(f"\n3. Creating IHC duplicates from: {pages[2].name}")
    ihc_variants = create_variants(pages[2], duplicates_dir / "IHC", "IHC")
    duplicates_info['IHC'] = ihc_variants
    
    return duplicates_info


def create_variants(original: Path, output_dir: Path, prefix: str):
    """Create duplicate variants."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        img = Image.open(original).convert('RGB')
    except Exception as e:
        print(f"  ERROR loading {original}: {e}")
        return {}
    
    variants = {}
    base_name = original.stem
    
    # Exact duplicate
    exact = output_dir / f"{prefix}_{base_name}_exact.png"
    shutil.copy(original, exact)
    variants['exact'] = exact
    print(f"    ✓ Exact duplicate: {exact.name}")
    
    # Rotated 90
    try:
        rotated_90 = img.rotate(90, expand=True)
        rotated_90_path = output_dir / f"{prefix}_{base_name}_rotated_90.png"
        rotated_90.save(rotated_90_path)
        variants['rotated_90'] = rotated_90_path
        print(f"    ✓ Rotated 90°: {rotated_90_path.name}")
    except Exception as e:
        print(f"    ✗ Rotated 90° failed: {e}")
    
    # Rotated 180
    try:
        rotated_180 = img.rotate(180, expand=True)
        rotated_180_path = output_dir / f"{prefix}_{base_name}_rotated_180.png"
        rotated_180.save(rotated_180_path)
        variants['rotated_180'] = rotated_180_path
        print(f"    ✓ Rotated 180°: {rotated_180_path.name}")
    except Exception as e:
        print(f"    ✗ Rotated 180° failed: {e}")
    
    # Mirrored
    try:
        mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirrored_path = output_dir / f"{prefix}_{base_name}_mirrored.png"
        mirrored.save(mirrored_path)
        variants['mirrored'] = mirrored_path
        print(f"    ✓ Mirrored: {mirrored_path.name}")
    except Exception as e:
        print(f"    ✗ Mirrored failed: {e}")
    
    # Partial 70%
    try:
        w, h = img.size
        crop_w = int(w * 0.7)
        crop_h = int(h * 0.7)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        partial = img.crop((x, y, x + crop_w, y + crop_h))
        partial_path = output_dir / f"{prefix}_{base_name}_partial_70pct.png"
        partial.save(partial_path)
        variants['partial_70'] = partial_path
        print(f"    ✓ Partial 70%: {partial_path.name}")
    except Exception as e:
        print(f"    ✗ Partial 70% failed: {e}")
    
    # Partial 50%
    try:
        w, h = img.size
        crop_w2 = int(w * 0.5)
        crop_h2 = int(h * 0.5)
        x2 = (w - crop_w2) // 2
        y2 = (h - crop_h2) // 2
        partial2 = img.crop((x2, y2, x2 + crop_w2, y2 + crop_h2))
        partial2_path = output_dir / f"{prefix}_{base_name}_partial_50pct.png"
        partial2.save(partial2_path)
        variants['partial_50'] = partial2_path
        print(f"    ✓ Partial 50%: {partial2_path.name}")
    except Exception as e:
        print(f"    ✗ Partial 50% failed: {e}")
    
    return variants


def main():
    """Main function."""
    print(f"{'='*70}")
    print("Duplicate Test Creation - Standalone Version")
    print(f"{'='*70}\n")
    
    pdf_path = Path("/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf")
    
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        return False
    
    output_dir = Path("test_duplicate_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Extract pages
    print("Step 1: Extracting pages from PDF...")
    pages_dir = output_dir / "pages"
    pages = extract_pages_simple(pdf_path, pages_dir, max_pages=5)
    
    if not pages:
        print("\n⚠ Could not extract pages. Please install PyMuPDF:")
        print("  pip install pymupdf")
        return False
    
    print(f"✓ Extracted {len(pages)} pages\n")
    
    # Step 2: Create duplicates
    print("Step 2: Creating duplicate variants...")
    duplicates_info = create_duplicates_from_pages(pages, output_dir)
    
    if not duplicates_info:
        print("ERROR: Failed to create duplicates")
        return False
    
    # Summary
    total_variants = sum(len(v) for v in duplicates_info.values())
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"✓ Created {total_variants} duplicate variants")
    print(f"✓ Location: {output_dir / 'intentional_duplicates'}")
    
    print(f"\nDuplicate types:")
    for category, variants in duplicates_info.items():
        print(f"  - {category}: {len(variants)} variants")
    
    # Create test panels directory
    test_panels_dir = output_dir / "test_panels"
    test_panels_dir.mkdir(exist_ok=True)
    
    # Copy pages
    for page in pages:
        shutil.copy(page, test_panels_dir / page.name)
    
    # Copy duplicates
    for category, variants in duplicates_info.items():
        for variant_path in variants.values():
            shutil.copy(variant_path, test_panels_dir / variant_path.name)
    
    total_test = len(list(test_panels_dir.glob("*.png")))
    print(f"\n✓ Test panels directory: {total_test} images")
    print(f"  Location: {test_panels_dir}")
    
    print(f"\n{'='*70}")
    print("✓ Duplicate creation complete!")
    print(f"{'='*70}")
    print("\nNext: Run detection pipeline on these test panels")
    print(f"Test panels are in: {test_panels_dir}")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

