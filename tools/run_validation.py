#!/usr/bin/env python3
"""
Run validation experiment on detection pipeline

Usage:
    # Build dataset
    python tools/run_validation.py build --panels-dir ./panels --output ./validation_dataset
    
    # Run validation
    python tools/run_validation.py test --dataset ./validation_dataset --output ./validation_results
"""

import sys
from pathlib import Path
import argparse
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_experiment import ValidationDatasetBuilder, ValidationRunner
import cv2
import numpy as np
from PIL import Image
import imagehash


def build_validation_dataset(panels_dir: Path, output_dir: Path, num_negatives: int = 20):
    """
    Build validation dataset from panels directory
    
    Args:
        panels_dir: Directory containing extracted panels
        output_dir: Where to save validation dataset
        num_negatives: Number of hard negative pairs to generate
    """
    print("\n" + "="*70)
    print("📦 BUILDING VALIDATION DATASET")
    print("="*70)
    
    panels_dir = Path(panels_dir)
    if not panels_dir.exists():
        print(f"❌ Panels directory not found: {panels_dir}")
        return None
    
    # Find all panel images
    panels = sorted(list(panels_dir.glob("*.png")) + list(panels_dir.glob("*.jpg")))
    
    if len(panels) < 10:
        print(f"❌ Not enough panels found: {len(panels)} (need at least 10)")
        return None
    
    print(f"✓ Found {len(panels)} panels")
    
    # Initialize builder
    builder = ValidationDatasetBuilder(output_dir)
    
    # 1. Add synthetic transformed duplicates (guaranteed true positives)
    print("\n📐 Creating transformed duplicates...")
    sample_panels = random.sample(panels, min(5, len(panels)))
    
    for panel in sample_panels:
        transforms = ['rotate_90', 'mirror_h', 'brightness_+20', 'crop_15pct', 'rotate_180']
        pair_ids = builder.add_transformed_duplicate(str(panel), transforms=transforms)
        print(f"  ✓ Created {len(pair_ids)} transform pairs from {panel.name}")
    
    # 2. Add hard negatives (same modality, different content)
    print(f"\n🚫 Creating {num_negatives} hard negative pairs...")
    for i in range(num_negatives):
        # Randomly select two different panels
        pair = random.sample(panels, 2)
        builder.add_hard_negative_pair(
            str(pair[0]),
            str(pair[1]),
            modality='unknown'
        )
    
    print(f"  ✓ Created {num_negatives} hard negative pairs")
    
    # 3. Save ground truth
    builder.save_ground_truth()
    
    return builder


def simple_detection_function(path_a: str, path_b: str) -> dict:
    """
    Simple detection function using basic image comparison
    This is a lightweight version for quick testing
    """
    try:
        # Load images
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        
        if img_a is None or img_b is None:
            return {
                'detected': False,
                'clip_score': None,
                'ssim_score': None,
                'phash_distance': None,
                'method': 'error'
            }
        
        # Compute pHash
        pil_a = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
        pil_b = Image.fromarray(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
        
        hash_a = imagehash.phash(pil_a)
        hash_b = imagehash.phash(pil_b)
        phash_dist = hash_a - hash_b
        
        # Compute basic similarity
        # Resize to same size for comparison
        h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
        img_a_resized = cv2.resize(img_a, (w, h))
        img_b_resized = cv2.resize(img_b, (w, h))
        
        # Convert to grayscale
        gray_a = cv2.cvtColor(img_a_resized, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        from skimage.metrics import structural_similarity as ssim_func
        ssim_score = ssim_func(gray_a, gray_b)
        
        # Simple decision rule
        detected = (phash_dist <= 4) or (ssim_score >= 0.95)
        
        return {
            'detected': detected,
            'clip_score': None,  # Not computed in simple version
            'ssim_score': float(ssim_score),
            'phash_distance': int(phash_dist),
            'method': 'phash' if (phash_dist <= 4) else ('ssim' if (ssim_score >= 0.95) else 'none')
        }
    
    except Exception as e:
        print(f"  ⚠️  Error processing {Path(path_a).name} vs {Path(path_b).name}: {e}")
        return {
            'detected': False,
            'clip_score': None,
            'ssim_score': None,
            'phash_distance': None,
            'method': 'error'
        }


def run_validation(dataset_dir: Path, output_dir: Path):
    """
    Run validation experiment
    
    Args:
        dataset_dir: Directory containing validation dataset
        output_dir: Where to save results
    """
    manifest_path = dataset_dir / 'ground_truth_manifest.json'
    
    if not manifest_path.exists():
        print(f"❌ Ground truth manifest not found: {manifest_path}")
        print("Run with 'build' command first to create validation dataset")
        return None
    
    # Initialize runner
    runner = ValidationRunner(manifest_path)
    
    # Run validation with simple detection function
    print("\n🔬 Using simple detection function (pHash + SSIM)...")
    metrics = runner.run_validation(
        detection_function=simple_detection_function,
        output_path=output_dir
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run validation experiment')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build validation dataset')
    build_parser.add_argument('--panels-dir', type=str, required=True,
                            help='Directory containing extracted panels')
    build_parser.add_argument('--output', type=str, default='./validation_dataset',
                            help='Output directory for validation dataset')
    build_parser.add_argument('--num-negatives', type=int, default=20,
                            help='Number of hard negative pairs to generate')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run validation test')
    test_parser.add_argument('--dataset', type=str, required=True,
                            help='Validation dataset directory')
    test_parser.add_argument('--output', type=str, default='./validation_results',
                            help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        builder = build_validation_dataset(
            panels_dir=Path(args.panels_dir),
            output_dir=Path(args.output),
            num_negatives=args.num_negatives
        )
        
        if builder:
            print(f"\n✅ Validation dataset created: {args.output}")
            print(f"   Run validation with:")
            print(f"   python tools/run_validation.py test --dataset {args.output}")
    
    elif args.command == 'test':
        metrics = run_validation(
            dataset_dir=Path(args.dataset),
            output_dir=Path(args.output)
        )
        
        if metrics:
            print(f"\n✅ Validation complete!")
            print(f"   Results saved to: {args.output}")
            print(f"\n   Key metrics:")
            print(f"     F1 Score: {metrics['overall']['f1_score']:.4f}")
            print(f"     Recall:   {metrics['overall']['recall']:.4f}")
            print(f"     FPR:      {metrics['overall']['false_positive_rate']:.4f}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

