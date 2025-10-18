#!/usr/bin/env python3
"""
Example test script for CLAHE A/B testing

Usage:
    python tools/test_clahe_ab.py --test-dir /path/to/test/images
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clahe_ab_test import ABTestHarness
import argparse


def create_test_pairs_from_directory(test_dir: Path):
    """
    Create test pairs from a directory structure:
    
    test_dir/
        duplicates/
            pair1_a.png
            pair1_b.png
            pair2_a.png
            pair2_b.png
        non_duplicates/
            diff1_a.png
            diff1_b.png
    
    Returns:
        List of (path_a, path_b, ground_truth) tuples
    """
    test_pairs = []
    
    # Find duplicate pairs
    dup_dir = test_dir / "duplicates"
    if dup_dir.exists():
        images = sorted(dup_dir.glob("*.png")) + sorted(dup_dir.glob("*.jpg"))
        
        # Pair up images (assuming _a and _b naming convention)
        for i in range(0, len(images) - 1, 2):
            if i + 1 < len(images):
                test_pairs.append((
                    str(images[i]),
                    str(images[i + 1]),
                    'duplicate'
                ))
    
    # Find non-duplicate pairs
    non_dup_dir = test_dir / "non_duplicates"
    if non_dup_dir.exists():
        images = sorted(non_dup_dir.glob("*.png")) + sorted(non_dup_dir.glob("*.jpg"))
        
        for i in range(0, len(images) - 1, 2):
            if i + 1 < len(images):
                test_pairs.append((
                    str(images[i]),
                    str(images[i + 1]),
                    'not_duplicate'
                ))
    
    return test_pairs


def run_simple_test():
    """Run a simple synthetic test with generated images"""
    import cv2
    import numpy as np
    
    print("\n" + "="*70)
    print("🧪 RUNNING SYNTHETIC CLAHE A/B TEST")
    print("="*70)
    print("\nGenerating synthetic test images...")
    
    # Create output directory
    output_dir = Path("./clahe_test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create test directory
    test_dir = output_dir / "synthetic_images"
    test_dir.mkdir(exist_ok=True)
    
    # Generate synthetic duplicate pairs with different brightness
    test_pairs = []
    
    for i in range(5):
        # Create a base image with some structure
        base = np.random.randint(100, 150, (384, 384), dtype=np.uint8)
        
        # Add some structure (vertical and horizontal lines)
        base[100:150, :] = 200
        base[:, 150:200] = 80
        base[200:250, 100:300] = 180
        
        # Create brightened version (simulate lighting variation)
        bright = np.clip(base.astype(float) * 1.5, 0, 255).astype(np.uint8)
        
        # Save images
        path_a = test_dir / f"pair{i}_normal.png"
        path_b = test_dir / f"pair{i}_bright.png"
        
        cv2.imwrite(str(path_a), base)
        cv2.imwrite(str(path_b), bright)
        
        test_pairs.append((str(path_a), str(path_b), 'duplicate'))
    
    print(f"✓ Generated {len(test_pairs)} test pairs")
    
    # Run A/B test
    harness = ABTestHarness(output_dir)
    stats = harness.run_ab_test(test_pairs, output_name='synthetic_ab_test')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Run CLAHE A/B test')
    parser.add_argument('--test-dir', type=str, help='Directory containing test images')
    parser.add_argument('--synthetic', action='store_true', help='Run synthetic test')
    parser.add_argument('--output', type=str, default='./clahe_test_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.synthetic:
        # Run synthetic test
        stats = run_simple_test()
    elif args.test_dir:
        # Run test on provided directory
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            print(f"Error: Test directory not found: {test_dir}")
            return 1
        
        print(f"\n🔍 Scanning test directory: {test_dir}")
        test_pairs = create_test_pairs_from_directory(test_dir)
        
        if not test_pairs:
            print("❌ No test pairs found!")
            print("\nExpected directory structure:")
            print("  test_dir/")
            print("      duplicates/")
            print("          pair1_a.png")
            print("          pair1_b.png")
            print("      non_duplicates/")
            print("          diff1_a.png")
            print("          diff1_b.png")
            return 1
        
        print(f"✓ Found {len(test_pairs)} test pairs")
        
        # Run A/B test
        output_dir = Path(args.output)
        harness = ABTestHarness(output_dir)
        stats = harness.run_ab_test(test_pairs, output_name='clahe_ab_test')
    else:
        # No arguments - show help and run synthetic test
        parser.print_help()
        print("\n" + "="*70)
        print("No arguments provided - running synthetic test as demo...")
        print("="*70)
        stats = run_simple_test()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

