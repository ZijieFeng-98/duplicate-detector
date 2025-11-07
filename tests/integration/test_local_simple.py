#!/usr/bin/env python3
"""
Test duplicate detection locally using available libraries.
Tests pHash bundles and basic detection on created duplicates.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

def test_phash_bundles():
    """Test pHash bundle detection (rotation-robust)."""
    try:
        import imagehash
    except ImportError:
        print("ERROR: imagehash not installed")
        print("Install: pip3 install imagehash")
        return False
    
    print(f"{'='*70}")
    print("Testing Duplicate Detection Locally")
    print(f"{'='*70}\n")
    
    duplicates_dir = Path("test_duplicate_detection/intentional_duplicates")
    if not duplicates_dir.exists():
        print(f"ERROR: Duplicates not found: {duplicates_dir}")
        return False
    
    results = {
        'exact_detected': False,
        'rotated_detected': False,
        'partial_detected': False,
        'wb_detected': False,
        'confocal_detected': False,
        'ihc_detected': False
    }
    
    # Test WB duplicates
    print("Testing WB duplicates...")
    wb_dir = duplicates_dir / "WB"
    if wb_dir.exists():
        wb_files = sorted(list(wb_dir.glob("*.png")))
        if len(wb_files) >= 2:
            # Find original (not exact duplicate)
            original = None
            exact = None
            rotated = None
            partial = None
            
            for f in wb_files:
                if 'exact' in f.name:
                    exact = f
                elif 'rotated' in f.name:
                    rotated = f
                elif 'partial' in f.name:
                    partial = f
                elif 'exact' not in f.name and 'rotated' not in f.name and 'partial' not in f.name:
                    original = f
            
            if not original:
                original = wb_files[0]
            
            # Test exact duplicate
            if exact:
                hash1 = imagehash.phash(Image.open(original))
                hash2 = imagehash.phash(Image.open(exact))
                distance = hash1 - hash2
                if distance <= 3:
                    results['exact_detected'] = True
                    results['wb_detected'] = True
                    print(f"  ✓ Exact duplicate detected (distance: {distance})")
                else:
                    print(f"  ✗ Exact duplicate NOT detected (distance: {distance}, threshold: 3)")
            
            # Test rotated duplicate with bundle approach
            if rotated:
                # Create rotation bundle
                img1 = Image.open(original)
                img2 = Image.open(rotated)
                
                # Test all rotations
                min_dist = float('inf')
                for angle in [0, 90, 180, 270]:
                    rotated_img = img1.rotate(angle, expand=True)
                    hash1 = imagehash.phash(rotated_img)
                    hash2 = imagehash.phash(img2)
                    dist = hash1 - hash2
                    min_dist = min(min_dist, dist)
                
                if min_dist <= 5:
                    results['rotated_detected'] = True
                    results['wb_detected'] = True
                    print(f"  ✓ Rotated duplicate detected (min distance: {min_dist})")
                else:
                    print(f"  ✗ Rotated duplicate NOT detected (min distance: {min_dist}, threshold: 5)")
            
            # Test partial duplicate
            if partial:
                hash1 = imagehash.phash(Image.open(original))
                hash2 = imagehash.phash(Image.open(partial))
                distance = hash1 - hash2
                if distance <= 10:  # Partial matches have higher distance
                    results['partial_detected'] = True
                    results['wb_detected'] = True
                    print(f"  ✓ Partial duplicate detected (distance: {distance})")
                else:
                    print(f"  ⚠ Partial duplicate (distance: {distance}, may need ORB-RANSAC)")
    
    # Test Confocal duplicates
    print("\nTesting Confocal duplicates...")
    confocal_dir = duplicates_dir / "confocal"
    if confocal_dir.exists():
        confocal_files = sorted(list(confocal_dir.glob("*.png")))
        if len(confocal_files) >= 2:
            exact = [f for f in confocal_files if 'exact' in f.name]
            if exact:
                hash1 = imagehash.phash(Image.open(confocal_files[0]))
                hash2 = imagehash.phash(Image.open(exact[0]))
                distance = hash1 - hash2
                if distance <= 3:
                    results['confocal_detected'] = True
                    print(f"  ✓ Confocal exact duplicate detected (distance: {distance})")
    
    # Test IHC duplicates
    print("\nTesting IHC duplicates...")
    ihc_dir = duplicates_dir / "IHC"
    if ihc_dir.exists():
        ihc_files = sorted(list(ihc_dir.glob("*.png")))
        if len(ihc_files) >= 2:
            exact = [f for f in ihc_files if 'exact' in f.name]
            if exact:
                hash1 = imagehash.phash(Image.open(ihc_files[0]))
                hash2 = imagehash.phash(Image.open(exact[0]))
                distance = hash1 - hash2
                if distance <= 3:
                    results['ihc_detected'] = True
                    print(f"  ✓ IHC exact duplicate detected (distance: {distance})")
    
    # Summary
    print(f"\n{'='*70}")
    print("Detection Summary")
    print(f"{'='*70}")
    print(f"Exact duplicates: {'✓ DETECTED' if results['exact_detected'] else '✗ NOT DETECTED'}")
    print(f"Rotated duplicates: {'✓ DETECTED' if results['rotated_detected'] else '✗ NOT DETECTED'}")
    print(f"Partial duplicates: {'✓ DETECTED' if results['partial_detected'] else '⚠ MAY NEED ORB-RANSAC'}")
    print(f"WB panels: {'✓ DETECTED' if results['wb_detected'] else '✗ NOT DETECTED'}")
    print(f"Confocal panels: {'✓ DETECTED' if results['confocal_detected'] else '✗ NOT DETECTED'}")
    print(f"IHC panels: {'✓ DETECTED' if results['ihc_detected'] else '✗ NOT DETECTED'}")
    
    print(f"\n{'='*70}")
    print("Note: Full detection requires:")
    print("  - CLIP embeddings (for semantic similarity)")
    print("  - ORB-RANSAC (for partial duplicates)")
    print("  - Full pipeline (install all dependencies)")
    print(f"{'='*70}\n")
    
    return True

if __name__ == "__main__":
    try:
        success = test_phash_bundles()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

