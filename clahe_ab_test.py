#!/usr/bin/env python3
"""
CLAHE A/B Testing Module
Add to your tile_detection.py to test photometric normalization impact
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class CLAHEConfig:
    """CLAHE configuration parameters"""
    clip_limit: float = 2.5
    tile_size: int = 8
    apply_zscore_after: bool = True
    
    
class PhotometricNormalizer:
    """Handles both z-score only and CLAHE+z-score normalization"""
    
    def __init__(self, config: CLAHEConfig = None):
        self.config = config or CLAHEConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=(self.config.tile_size, self.config.tile_size)
        )
    
    def normalize_zscore_only(self, img: np.ndarray) -> np.ndarray:
        """Current method: z-score normalization only"""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mu = img.mean()
        sd = img.std()
        
        if sd < 1e-8:
            return img
        
        return (img - mu) / sd
    
    def normalize_clahe_zscore(self, img: np.ndarray) -> np.ndarray:
        """New method: CLAHE + z-score normalization"""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ensure uint8 for CLAHE
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE
        clahe_img = self.clahe.apply(img)
        
        # Then z-score (if configured)
        if self.config.apply_zscore_after:
            mu = clahe_img.mean()
            sd = clahe_img.std()
            if sd > 1e-8:
                clahe_img = (clahe_img - mu) / sd
        
        return clahe_img
    
    def get_normalization_params(self, img: np.ndarray, method: str) -> Dict:
        """Get normalization parameters for audit trail"""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        params = {
            'method': method,
            'original_mean': float(img.mean()),
            'original_std': float(img.std()),
            'original_min': float(img.min()),
            'original_max': float(img.max())
        }
        
        if method == 'clahe_zscore':
            params.update({
                'clahe_clip_limit': self.config.clip_limit,
                'clahe_tile_size': self.config.tile_size,
                'apply_zscore': self.config.apply_zscore_after
            })
        
        return params


# ═══════════════════════════════════════════════════════════════
# A/B TESTING HARNESS
# ═══════════════════════════════════════════════════════════════

class ABTestHarness:
    """Compare z-score vs CLAHE+z-score performance"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.normalizer = PhotometricNormalizer()
        self.results = {
            'zscore_only': [],
            'clahe_zscore': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'config': {
                    'clahe_clip_limit': 2.5,
                    'clahe_tile_size': 8
                }
            }
        }
    
    def process_tile_pair(self, tile_a_path: str, tile_b_path: str, 
                         ground_truth: str) -> Dict:
        """
        Process one tile pair with both methods
        
        Args:
            tile_a_path: Path to first tile
            tile_b_path: Path to second tile
            ground_truth: 'duplicate' or 'not_duplicate'
        
        Returns:
            Comparison metrics for both methods
        """
        # Load images
        img_a = cv2.imread(tile_a_path)
        img_b = cv2.imread(tile_b_path)
        
        if img_a is None or img_b is None:
            return None
        
        # Method A: Z-score only
        norm_a_zscore = self.normalizer.normalize_zscore_only(img_a)
        norm_b_zscore = self.normalizer.normalize_zscore_only(img_b)
        
        # Method B: CLAHE + Z-score
        norm_a_clahe = self.normalizer.normalize_clahe_zscore(img_a)
        norm_b_clahe = self.normalizer.normalize_clahe_zscore(img_b)
        
        # Compute SSIM for both methods
        from skimage.metrics import structural_similarity as ssim_func
        
        # Determine data range for each method
        # Z-score images are roughly in [-3, 3] range but unbounded
        # Use max - min as data_range
        zscore_range = max(norm_a_zscore.max(), norm_b_zscore.max()) - min(norm_a_zscore.min(), norm_b_zscore.min())
        clahe_range = max(norm_a_clahe.max(), norm_b_clahe.max()) - min(norm_a_clahe.min(), norm_b_clahe.min())
        
        ssim_zscore = ssim_func(norm_a_zscore, norm_b_zscore, data_range=zscore_range if zscore_range > 0 else 1.0)
        ssim_clahe = ssim_func(norm_a_clahe, norm_b_clahe, data_range=clahe_range if clahe_range > 0 else 1.0)
        
        # Compute NCC for both methods
        ncc_zscore = self._compute_ncc(norm_a_zscore, norm_b_zscore)
        ncc_clahe = self._compute_ncc(norm_a_clahe, norm_b_clahe)
        
        result = {
            'tile_a': tile_a_path,
            'tile_b': tile_b_path,
            'ground_truth': ground_truth,
            'zscore_only': {
                'ssim': float(ssim_zscore),
                'ncc': float(ncc_zscore)
            },
            'clahe_zscore': {
                'ssim': float(ssim_clahe),
                'ncc': float(ncc_clahe)
            },
            'delta_ssim': float(ssim_clahe - ssim_zscore),
            'delta_ncc': float(ncc_clahe - ncc_zscore)
        }
        
        return result
    
    def _compute_ncc(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Compute normalized cross-correlation"""
        # Ensure same size
        if img_a.shape != img_b.shape:
            return 0.0
        
        # Flatten and compute NCC
        a_flat = img_a.flatten()
        b_flat = img_b.flatten()
        
        ncc = np.corrcoef(a_flat, b_flat)[0, 1]
        return float(ncc) if not np.isnan(ncc) else 0.0
    
    def run_ab_test(self, test_pairs: list, output_name: str = 'ab_test_results'):
        """
        Run A/B test on a list of tile pairs
        
        Args:
            test_pairs: List of (path_a, path_b, ground_truth) tuples
            output_name: Base name for output files
        """
        print("\n" + "="*70)
        print("🧪 CLAHE A/B TEST")
        print("="*70)
        print(f"Testing {len(test_pairs)} pairs...")
        
        for tile_a, tile_b, ground_truth in test_pairs:
            result = self.process_tile_pair(tile_a, tile_b, ground_truth)
            
            if result is None:
                continue
            
            # Store results
            if ground_truth == 'duplicate':
                self.results['zscore_only'].append(result['zscore_only'])
                self.results['clahe_zscore'].append(result['clahe_zscore'])
        
        # Compute statistics
        stats = self._compute_statistics()
        
        # Save results
        self._save_results(output_name, stats)
        
        # Print summary
        self._print_summary(stats)
        
        return stats
    
    def _compute_statistics(self) -> Dict:
        """Compute comparative statistics"""
        zscore_ssims = [r['ssim'] for r in self.results['zscore_only']]
        clahe_ssims = [r['ssim'] for r in self.results['clahe_zscore']]
        
        zscore_nccs = [r['ncc'] for r in self.results['zscore_only']]
        clahe_nccs = [r['ncc'] for r in self.results['clahe_zscore']]
        
        stats = {
            'zscore_only': {
                'ssim_mean': np.mean(zscore_ssims) if zscore_ssims else 0.0,
                'ssim_std': np.std(zscore_ssims) if zscore_ssims else 0.0,
                'ssim_min': np.min(zscore_ssims) if zscore_ssims else 0.0,
                'ncc_mean': np.mean(zscore_nccs) if zscore_nccs else 0.0,
                'ncc_std': np.std(zscore_nccs) if zscore_nccs else 0.0
            },
            'clahe_zscore': {
                'ssim_mean': np.mean(clahe_ssims) if clahe_ssims else 0.0,
                'ssim_std': np.std(clahe_ssims) if clahe_ssims else 0.0,
                'ssim_min': np.min(clahe_ssims) if clahe_ssims else 0.0,
                'ncc_mean': np.mean(clahe_nccs) if clahe_nccs else 0.0,
                'ncc_std': np.std(clahe_nccs) if clahe_nccs else 0.0
            },
            'improvement': {
                'ssim_delta': np.mean(clahe_ssims) - np.mean(zscore_ssims) if clahe_ssims and zscore_ssims else 0.0,
                'ncc_delta': np.mean(clahe_nccs) - np.mean(zscore_nccs) if clahe_nccs and zscore_nccs else 0.0,
                'ssim_pct_change': ((np.mean(clahe_ssims) - np.mean(zscore_ssims)) / np.mean(zscore_ssims) * 100) if zscore_ssims and np.mean(zscore_ssims) > 0 else 0.0
            }
        }
        
        return stats
    
    def _save_results(self, output_name: str, stats: Dict):
        """Save results to JSON and CSV"""
        # Save full results
        results_file = self.output_dir / f'{output_name}_full.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save statistics
        stats_file = self.output_dir / f'{output_name}_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Results saved to: {self.output_dir}")
        print(f"  • Full results: {results_file.name}")
        print(f"  • Statistics: {stats_file.name}")
    
    def _print_summary(self, stats: Dict):
        """Print comparison summary"""
        print("\n" + "="*70)
        print("📊 A/B TEST RESULTS SUMMARY")
        print("="*70)
        
        print("\n🔵 Method A: Z-Score Only")
        print(f"  SSIM: {stats['zscore_only']['ssim_mean']:.4f} ± {stats['zscore_only']['ssim_std']:.4f}")
        print(f"  NCC:  {stats['zscore_only']['ncc_mean']:.4f} ± {stats['zscore_only']['ncc_std']:.4f}")
        
        print("\n🟢 Method B: CLAHE + Z-Score")
        print(f"  SSIM: {stats['clahe_zscore']['ssim_mean']:.4f} ± {stats['clahe_zscore']['ssim_std']:.4f}")
        print(f"  NCC:  {stats['clahe_zscore']['ncc_mean']:.4f} ± {stats['clahe_zscore']['ncc_std']:.4f}")
        
        print("\n📈 Improvement")
        print(f"  SSIM Δ: {stats['improvement']['ssim_delta']:+.4f} ({stats['improvement']['ssim_pct_change']:+.2f}%)")
        print(f"  NCC Δ:  {stats['improvement']['ncc_delta']:+.4f}")
        
        # Recommendation
        print("\n💡 Recommendation:")
        if stats['improvement']['ssim_delta'] > 0.01:
            print("  ✅ CLAHE shows meaningful improvement - consider adopting")
        elif stats['improvement']['ssim_delta'] > 0:
            print("  ⚠️  CLAHE shows slight improvement - test on more pairs")
        else:
            print("  ❌ CLAHE does not improve performance - stick with z-score only")


# ═══════════════════════════════════════════════════════════════
# INTEGRATION EXAMPLE
# ═══════════════════════════════════════════════════════════════

def integrate_clahe_into_tile_detection():
    """
    Example: How to integrate CLAHE into your tile_detection.py
    
    Replace your current _zscore() function with:
    """
    
    example_code = '''
# Add at top of tile_detection.py:
from clahe_ab_test import PhotometricNormalizer, CLAHEConfig

# Initialize normalizer (do this once, at module level)
NORMALIZER = PhotometricNormalizer(CLAHEConfig(
    clip_limit=2.5,
    tile_size=8,
    apply_zscore_after=True
))

# Replace your _zscore() function with:
def _normalize_photometric(img: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """
    Apply photometric normalization
    
    Args:
        img: Input image
        use_clahe: If True, apply CLAHE+zscore; if False, zscore only
    
    Returns:
        Normalized image
    """
    if use_clahe:
        return NORMALIZER.normalize_clahe_zscore(img)
    else:
        return NORMALIZER.normalize_zscore_only(img)

# Then in verify_tile_pair_confocal(), replace:
#   gray_a = _zscore(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY))
# with:
#   gray_a = _normalize_photometric(img_a, use_clahe=True)
'''
    
    return example_code


if __name__ == '__main__':
    print("CLAHE A/B Test Module")
    print("Import this module and use ABTestHarness class")
    print("\nExample usage:")
    print(integrate_clahe_into_tile_detection())

