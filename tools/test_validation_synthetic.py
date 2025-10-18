#!/usr/bin/env python3
"""
Quick synthetic test of validation framework
Generates test images and runs validation
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_experiment import ValidationDatasetBuilder, ValidationRunner
from PIL import Image
import imagehash


def generate_synthetic_panels(output_dir: Path, num_panels: int = 10):
    """Generate synthetic panel images for testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Generating {num_panels} synthetic panels...")
    
    panels = []
    
    for i in range(num_panels):
        # Create a synthetic image with distinct patterns
        img = np.random.randint(80, 120, (512, 512), dtype=np.uint8)
        
        # Add unique patterns for each panel
        # Vertical lines
        x_offset = (i * 50) % 400
        img[:, x_offset:x_offset+30] = 200
        
        # Horizontal lines
        y_offset = (i * 40) % 400
        img[y_offset:y_offset+25, :] = 70
        
        # Squares
        sq_x = (i * 60) % 400
        sq_y = (i * 70) % 400
        img[sq_y:sq_y+50, sq_x:sq_x+50] = 180
        
        # Circles (simulated)
        center_x, center_y = 256 + (i * 30) % 100, 256 + (i * 40) % 100
        for y in range(512):
            for x in range(512):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 40:
                    img[y, x] = min(255, int(img[y, x] * 1.5))
        
        # Save panel
        panel_path = output_dir / f'synthetic_panel_{i:02d}.png'
        cv2.imwrite(str(panel_path), img)
        panels.append(panel_path)
    
    print(f"  ✓ Generated {len(panels)} panels in {output_dir}")
    return panels


def simple_detection_function(path_a: str, path_b: str) -> dict:
    """Simple detection using pHash + SSIM"""
    try:
        # Load images
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        
        if img_a is None or img_b is None:
            return {'detected': False, 'method': 'error'}
        
        # pHash
        pil_a = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
        pil_b = Image.fromarray(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
        
        hash_a = imagehash.phash(pil_a)
        hash_b = imagehash.phash(pil_b)
        phash_dist = hash_a - hash_b
        
        # SSIM
        from skimage.metrics import structural_similarity as ssim_func
        
        # Resize to same size
        h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
        img_a_resized = cv2.resize(img_a, (w, h))
        img_b_resized = cv2.resize(img_b, (w, h))
        
        gray_a = cv2.cvtColor(img_a_resized, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY)
        
        ssim_score = ssim_func(gray_a, gray_b)
        
        # Decision: pHash ≤ 4 OR SSIM ≥ 0.95
        detected = (phash_dist <= 4) or (ssim_score >= 0.95)
        
        return {
            'detected': detected,
            'clip_score': None,
            'ssim_score': float(ssim_score),
            'phash_distance': int(phash_dist),
            'method': 'phash' if (phash_dist <= 4) else ('ssim' if detected else 'none')
        }
    
    except Exception as e:
        return {'detected': False, 'method': 'error'}


def run_synthetic_validation():
    """Run complete synthetic validation test"""
    
    print("\n" + "="*70)
    print("🧪 SYNTHETIC VALIDATION TEST")
    print("="*70)
    
    # Setup directories
    work_dir = Path('./validation_synthetic_test')
    panels_dir = work_dir / 'synthetic_panels'
    dataset_dir = work_dir / 'validation_dataset'
    results_dir = work_dir / 'validation_results'
    
    # Clean up previous runs
    import shutil
    if work_dir.exists():
        shutil.rmtree(work_dir)
    
    # Step 1: Generate synthetic panels
    panels = generate_synthetic_panels(panels_dir, num_panels=10)
    
    # Step 2: Build validation dataset
    print("\n📦 Building validation dataset...")
    builder = ValidationDatasetBuilder(dataset_dir)
    
    # Add known duplicates (transformed versions)
    print("  Adding transformed duplicates...")
    for i, panel in enumerate(panels[:3]):  # Use first 3 panels
        transforms = ['rotate_90', 'mirror_h', 'brightness_+20', 'rotate_180']
        pair_ids = builder.add_transformed_duplicate(str(panel), transforms=transforms)
        print(f"    ✓ Panel {i}: {len(pair_ids)} transform pairs")
    
    # Add hard negatives (different panels)
    print("  Adding hard negative pairs...")
    import random
    for i in range(10):
        pair = random.sample(panels, 2)
        builder.add_hard_negative_pair(str(pair[0]), str(pair[1]), modality='synthetic')
    print(f"    ✓ Added 10 hard negative pairs")
    
    # Save ground truth
    builder.save_ground_truth()
    
    # Step 3: Run validation
    print("\n🔬 Running validation...")
    runner = ValidationRunner(dataset_dir / 'ground_truth_manifest.json')
    
    metrics = runner.run_validation(
        detection_function=simple_detection_function,
        output_path=results_dir
    )
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("✅ SYNTHETIC VALIDATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"\n📊 Summary:")
    print(f"   F1 Score:   {metrics['overall']['f1_score']:.4f}")
    print(f"   Precision:  {metrics['overall']['precision']:.4f}")
    print(f"   Recall:     {metrics['overall']['recall']:.4f}")
    print(f"   FPR:        {metrics['overall']['false_positive_rate']:.4f}")
    
    # Check if it meets expectations
    print(f"\n💡 Expectations:")
    if metrics['overall']['recall'] >= 0.90:
        print(f"   ✅ Recall ≥ 90%: Detecting most transformed duplicates")
    else:
        print(f"   ⚠️  Recall < 90%: Missing some transforms")
    
    if metrics['overall']['false_positive_rate'] <= 0.20:
        print(f"   ✅ FPR ≤ 20%: Low false alarm rate on hard negatives")
    else:
        print(f"   ⚠️  FPR > 20%: Too many false positives")
    
    print(f"\n📈 This demonstrates the validation framework!")
    print(f"   Next: Run on real panels from your PDFs")
    
    return metrics


if __name__ == '__main__':
    try:
        metrics = run_synthetic_validation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

