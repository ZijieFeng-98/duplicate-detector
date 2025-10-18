#!/usr/bin/env python3
"""
Validation Experiment Design
Test detection pipeline with known duplicates and known non-duplicates
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import pandas as pd
import json
from datetime import datetime
import shutil


class ValidationDatasetBuilder:
    """Build validation dataset from your known duplicates file"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'true_positives').mkdir(exist_ok=True)
        (self.output_dir / 'hard_negatives').mkdir(exist_ok=True)
        (self.output_dir / 'transformed_duplicates').mkdir(exist_ok=True)
        
        self.ground_truth = []
    
    def add_known_duplicate_pair(self, panel_a_path: str, panel_b_path: str, 
                                 label: str = 'known_duplicate'):
        """Add a pair from your known duplicates file"""
        pair_id = len(self.ground_truth)
        
        # Copy to validation set
        dest_a = self.output_dir / 'true_positives' / f'pair_{pair_id:03d}_a.png'
        dest_b = self.output_dir / 'true_positives' / f'pair_{pair_id:03d}_b.png'
        
        shutil.copy2(panel_a_path, dest_a)
        shutil.copy2(panel_b_path, dest_b)
        
        self.ground_truth.append({
            'pair_id': pair_id,
            'category': 'true_positive',
            'path_a': str(dest_a),
            'path_b': str(dest_b),
            'label': label,
            'should_detect': True,
            'original_a': panel_a_path,
            'original_b': panel_b_path
        })
        
        return pair_id
    
    def add_hard_negative_pair(self, panel_a_path: str, panel_b_path: str,
                               modality: str = 'confocal'):
        """
        Add hard negative: same modality, different content
        These are the critical cases that should NOT be flagged
        """
        pair_id = len(self.ground_truth)
        
        dest_a = self.output_dir / 'hard_negatives' / f'pair_{pair_id:03d}_a.png'
        dest_b = self.output_dir / 'hard_negatives' / f'pair_{pair_id:03d}_b.png'
        
        shutil.copy2(panel_a_path, dest_a)
        shutil.copy2(panel_b_path, dest_b)
        
        self.ground_truth.append({
            'pair_id': pair_id,
            'category': 'hard_negative',
            'path_a': str(dest_a),
            'path_b': str(dest_b),
            'label': f'different_{modality}',
            'should_detect': False,
            'modality': modality
        })
        
        return pair_id
    
    def add_transformed_duplicate(self, source_path: str, 
                                  transforms: List[str] = None):
        """
        Create synthetic duplicates with known transforms
        Tests rotation/flip/crop robustness
        """
        if transforms is None:
            transforms = ['rotate_90', 'mirror_h', 'brightness_+20', 'crop_15pct']
        
        img = cv2.imread(source_path)
        if img is None:
            return []
        
        pair_ids = []
        base_id = len(self.ground_truth)
        
        for i, transform in enumerate(transforms):
            pair_id = base_id + i
            
            # Apply transform
            transformed = self._apply_transform(img.copy(), transform)
            
            # Save pair
            dest_orig = self.output_dir / 'transformed_duplicates' / f'pair_{pair_id:03d}_orig.png'
            dest_trans = self.output_dir / 'transformed_duplicates' / f'pair_{pair_id:03d}_{transform}.png'
            
            cv2.imwrite(str(dest_orig), img)
            cv2.imwrite(str(dest_trans), transformed)
            
            self.ground_truth.append({
                'pair_id': pair_id,
                'category': 'transformed_duplicate',
                'path_a': str(dest_orig),
                'path_b': str(dest_trans),
                'label': f'transform_{transform}',
                'should_detect': True,
                'transform_type': transform
            })
            
            pair_ids.append(pair_id)
        
        return pair_ids
    
    def _apply_transform(self, img: np.ndarray, transform: str) -> np.ndarray:
        """Apply a specific transform to an image"""
        h, w = img.shape[:2]
        
        if transform == 'rotate_90':
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        elif transform == 'rotate_180':
            return cv2.rotate(img, cv2.ROTATE_180)
        
        elif transform == 'rotate_270':
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        elif transform == 'mirror_h':
            return cv2.flip(img, 1)
        
        elif transform == 'mirror_v':
            return cv2.flip(img, 0)
        
        elif transform.startswith('brightness'):
            # e.g., 'brightness_+20' or 'brightness_-15'
            delta = int(transform.split('_')[1])
            return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        
        elif transform.startswith('crop'):
            # e.g., 'crop_15pct' removes 15% from edges
            pct = int(transform.split('_')[1].replace('pct', ''))
            margin = int(min(h, w) * pct / 100 / 2)
            return img[margin:h-margin, margin:w-margin]
        
        elif transform == 'blur_slight':
            return cv2.GaussianBlur(img, (5, 5), 1.0)
        
        elif transform == 'jpeg_compress':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, encoded = cv2.imencode('.jpg', img, encode_param)
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return img
    
    def save_ground_truth(self):
        """Save ground truth manifest"""
        manifest_path = self.output_dir / 'ground_truth_manifest.json'
        
        with open(manifest_path, 'w') as f:
            json.dump({
                'created': datetime.now().isoformat(),
                'total_pairs': len(self.ground_truth),
                'true_positives': sum(1 for x in self.ground_truth if x['should_detect']),
                'hard_negatives': sum(1 for x in self.ground_truth if not x['should_detect']),
                'pairs': self.ground_truth
            }, f, indent=2)
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame(self.ground_truth)
        df.to_csv(self.output_dir / 'ground_truth_manifest.csv', index=False)
        
        print(f"✓ Ground truth saved: {manifest_path}")
        print(f"  Total pairs: {len(self.ground_truth)}")
        print(f"  Should detect: {sum(1 for x in self.ground_truth if x['should_detect'])}")
        print(f"  Should NOT detect: {sum(1 for x in self.ground_truth if not x['should_detect'])}")


class ValidationRunner:
    """Run pipeline on validation dataset and compute metrics"""
    
    def __init__(self, ground_truth_manifest: Path):
        with open(ground_truth_manifest) as f:
            data = json.load(f)
        
        self.pairs = data['pairs']
        self.results = []
    
    def run_validation(self, detection_function: Callable, output_path: Path):
        """
        Run detection pipeline on all validation pairs
        
        Args:
            detection_function: Function that takes (path_a, path_b) and returns detection result
            output_path: Where to save results
        
        Returns:
            Performance metrics dict
        """
        print("\n" + "="*70)
        print("🧪 RUNNING VALIDATION EXPERIMENT")
        print("="*70)
        print(f"Testing {len(self.pairs)} pairs...")
        
        self.results = []  # Reset results
        
        for i, pair in enumerate(self.pairs):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(self.pairs)} pairs...")
            
            result = detection_function(pair['path_a'], pair['path_b'])
            
            self.results.append({
                'pair_id': int(pair['pair_id']),
                'category': str(pair['category']),
                'label': str(pair['label']),
                'should_detect': bool(pair['should_detect']),
                'was_detected': bool(result['detected']),
                'clip_score': float(result['clip_score']) if result.get('clip_score') is not None else None,
                'ssim_score': float(result['ssim_score']) if result.get('ssim_score') is not None else None,
                'phash_distance': int(result['phash_distance']) if result.get('phash_distance') is not None else None,
                'method_triggered': str(result.get('method', 'none')) if result.get('method') else None
            })
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Save results
        self._save_results(output_path, metrics)
        
        # Print report
        self._print_report(metrics)
        
        return metrics
    
    def _compute_metrics(self) -> Dict:
        """Compute precision, recall, F1, FPR"""
        tp = sum(1 for r in self.results if r['should_detect'] and r['was_detected'])
        fp = sum(1 for r in self.results if not r['should_detect'] and r['was_detected'])
        fn = sum(1 for r in self.results if r['should_detect'] and not r['was_detected'])
        tn = sum(1 for r in self.results if not r['should_detect'] and not r['was_detected'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Category breakdown
        category_metrics = {}
        for category in set(r['category'] for r in self.results):
            cat_results = [r for r in self.results if r['category'] == category]
            cat_tp = sum(1 for r in cat_results if r['should_detect'] and r['was_detected'])
            cat_fn = sum(1 for r in cat_results if r['should_detect'] and not r['was_detected'])
            cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
            
            category_metrics[category] = {
                'count': len(cat_results),
                'detected': sum(1 for r in cat_results if r['was_detected']),
                'recall': cat_recall
            }
        
        return {
            'overall': {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fpr,
                'accuracy': (tp + tn) / len(self.results) if len(self.results) > 0 else 0.0
            },
            'by_category': category_metrics,
            'total_pairs': len(self.results)
        }
    
    def _save_results(self, output_path: Path, metrics: Dict):
        """Save results and metrics"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        results_file = output_path / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save results CSV
        df = pd.DataFrame(self.results)
        df.to_csv(output_path / 'validation_results.csv', index=False)
        
        # Save metrics summary
        with open(output_path / 'metrics_summary.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def _print_report(self, metrics: Dict):
        """Print validation report"""
        m = metrics['overall']
        
        print("\n" + "="*70)
        print("📊 VALIDATION RESULTS")
        print("="*70)
        
        print("\n🎯 Overall Performance:")
        print(f"  Precision: {m['precision']:.4f} ({m['true_positives']}/{m['true_positives']+m['false_positives']})")
        print(f"  Recall:    {m['recall']:.4f} ({m['true_positives']}/{m['true_positives']+m['false_negatives']})")
        print(f"  F1 Score:  {m['f1_score']:.4f}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        
        print(f"\n🚨 False Positive Rate: {m['false_positive_rate']:.4f} ({m['false_positives']} false alarms)")
        
        print("\n📂 Performance by Category:")
        for category, cat_metrics in metrics['by_category'].items():
            print(f"  {category}:")
            print(f"    Detected: {cat_metrics['detected']}/{cat_metrics['count']}")
            print(f"    Recall: {cat_metrics['recall']:.4f}")
        
        # Assessment
        print("\n💡 Assessment:")
        if m['false_positive_rate'] <= 0.005:
            print("  ✅ FPR ≤ 0.5% - Excellent! Meets target threshold")
        elif m['false_positive_rate'] <= 0.01:
            print("  ⚠️  FPR ≤ 1% - Good, but slightly above target")
        else:
            print("  ❌ FPR > 1% - Consider stricter thresholds")
        
        if m['recall'] >= 0.95:
            print("  ✅ Recall ≥ 95% - Catching most duplicates")
        elif m['recall'] >= 0.85:
            print("  ⚠️  Recall ≥ 85% - Missing some duplicates")
        else:
            print("  ❌ Recall < 85% - Consider looser thresholds")


# ═══════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════

def example_usage():
    """Example: How to use validation framework"""
    
    example = '''
# ═══════════════════════════════════════════════════════════════
# STEP 1: BUILD VALIDATION DATASET
# ═══════════════════════════════════════════════════════════════

from validation_experiment import ValidationDatasetBuilder

builder = ValidationDatasetBuilder(output_dir=Path('validation_dataset'))

# Add your known duplicates (from the file you mentioned)
builder.add_known_duplicate_pair(
    panel_a_path='panels/page_01_panel_003.png',
    panel_b_path='panels/page_05_panel_012.png',
    label='known_duplicate_confocal'
)

# Add hard negatives (same modality, different content)
builder.add_hard_negative_pair(
    panel_a_path='panels/page_02_panel_004.png',
    panel_b_path='panels/page_03_panel_008.png',
    modality='confocal'
)

# Add synthetic transforms
builder.add_transformed_duplicate(
    source_path='panels/page_01_panel_003.png',
    transforms=['rotate_90', 'mirror_h', 'brightness_+20', 'crop_15pct']
)

# Save ground truth
builder.save_ground_truth()

# ═══════════════════════════════════════════════════════════════
# STEP 2: RUN VALIDATION WITH YOUR PIPELINE
# ═══════════════════════════════════════════════════════════════

from validation_experiment import ValidationRunner

# Define your detection function
def my_detection_function(path_a, path_b):
    """Wrapper around your pipeline"""
    # Run your tile detection or panel detection
    # Return dict with:
    return {
        'detected': True/False,
        'clip_score': 0.98,
        'ssim_score': 0.94,
        'phash_distance': 3,
        'method': 'tile_ssim' or 'panel_clip' etc.
    }

runner = ValidationRunner(
    ground_truth_manifest=Path('validation_dataset/ground_truth_manifest.json')
)

metrics = runner.run_validation(
    detection_function=my_detection_function,
    output_path=Path('validation_results')
)

# ═══════════════════════════════════════════════════════════════
# STEP 3: COMPARE CLAHE VS NO CLAHE
# ═══════════════════════════════════════════════════════════════

# Run twice with different normalization settings
metrics_zscore = runner.run_validation(
    detection_function=lambda a, b: detect_with_zscore(a, b),
    output_path=Path('validation_results/zscore_only')
)

metrics_clahe = runner.run_validation(
    detection_function=lambda a, b: detect_with_clahe(a, b),
    output_path=Path('validation_results/clahe_zscore')
)

# Compare FPR
print(f"Z-score FPR: {metrics_zscore['overall']['false_positive_rate']:.4f}")
print(f"CLAHE FPR:   {metrics_clahe['overall']['false_positive_rate']:.4f}")
'''
    
    return example


if __name__ == '__main__':
    print("Validation Experiment Design")
    print("\nUsage:")
    print(example_usage())

