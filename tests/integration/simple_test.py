#!/usr/bin/env python3
"""
Simple test script to verify duplicate creation and detection works.
This can run without all dependencies installed.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import imagehash

def test_exact_duplicate():
    """Test that exact duplicates are detected."""
    print("Testing exact duplicate detection...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a test image
        test_img = Image.new('RGB', (200, 200), color='blue')
        original_path = tmp_path / "original.png"
        test_img.save(original_path)
        print(f"  ✓ Created original image: {original_path}")
        
        # Create exact duplicate
        duplicate_path = tmp_path / "duplicate.png"
        shutil.copy(original_path, duplicate_path)
        print(f"  ✓ Created exact duplicate: {duplicate_path}")
        
        # Test with pHash
        hash1 = str(imagehash.phash(Image.open(original_path)))
        hash2 = str(imagehash.phash(Image.open(duplicate_path)))
        
        # Exact duplicates should have distance 0
        distance = imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)
        
        if distance == 0:
            print(f"  ✓ PASS: Exact duplicate detected correctly (distance={distance})")
            return True
        else:
            print(f"  ✗ FAIL: Expected distance 0, got {distance}")
            return False


def test_rotated_duplicate():
    """Test that rotated duplicates are detected."""
    print("\nTesting rotated duplicate detection...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a test image with distinctive pattern
        test_img = Image.new('RGB', (200, 200), color='red')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([50, 50, 150, 150], fill='blue')
        
        original_path = tmp_path / "original_rot.png"
        test_img.save(original_path)
        print(f"  ✓ Created original image: {original_path}")
        
        # Create rotated duplicate
        rotated = test_img.rotate(90)
        rotated_path = tmp_path / "rotated_90.png"
        rotated.save(rotated_path)
        print(f"  ✓ Created rotated duplicate (90°): {rotated_path}")
        
        # Test with pHash bundle (simplified - just test rotations)
        hashes_original = {}
        hashes_rotated = {}
        
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated_img = test_img
            else:
                rotated_img = test_img.rotate(angle, expand=True)
            hashes_original[f'rot_{angle}'] = str(imagehash.phash(rotated_img))
            
            if angle == 0:
                rotated_test = rotated
            else:
                rotated_test = rotated.rotate(angle, expand=True)
            hashes_rotated[f'rot_{angle}'] = str(imagehash.phash(rotated_test))
        
        # Find minimum distance
        min_dist = 999
        for key1 in hashes_original:
            for key2 in hashes_rotated:
                dist = imagehash.hex_to_hash(hashes_original[key1]) - imagehash.hex_to_hash(hashes_rotated[key2])
                if dist < min_dist:
                    min_dist = dist
        
        if min_dist <= 5:
            print(f"  ✓ PASS: Rotated duplicate detected correctly (min_distance={min_dist})")
            return True
        else:
            print(f"  ✗ FAIL: Expected distance <= 5, got {min_dist}")
            return False


def test_pdf_exists():
    """Test that the PDF file exists."""
    print("\nTesting PDF file existence...")
    
    pdf_path = Path("/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf")
    
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ PASS: PDF file exists: {pdf_path}")
        print(f"  ✓ File size: {size_mb:.2f} MB")
        return True
    else:
        print(f"  ✗ FAIL: PDF file not found: {pdf_path}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Duplicate Detection Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: PDF exists
    results.append(("PDF Exists", test_pdf_exists()))
    
    # Test 2: Exact duplicate
    results.append(("Exact Duplicate", test_exact_duplicate()))
    
    # Test 3: Rotated duplicate
    results.append(("Rotated Duplicate", test_rotated_duplicate()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

