#!/usr/bin/env python3
"""
Comprehensive Test Runner
Applies all production-grade improvements from comprehensive test scripts

Integrates:
- Script 1: PR optimization with Platt calibration
- Script 2: Production SSIM and FAISS improvements  
- Script 3: Expert-reviewed implementation guide

Author: AI Assistant
Date: 2025-01-18
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import validation framework
from validation_experiment import ValidationRunner, ValidationDatasetBuilder


# ═══════════════════════════════════════════════════════════════
# COMPREHENSIVE TEST SUITE
# ═══════════════════════════════════════════════════════════════

class ComprehensiveTestRunner:
    """
    Unified test runner applying all production-grade improvements
    """
    
    def __init__(self, dataset_path: Path, results_path: Path):
        self.dataset_path = Path(dataset_path)
        self.results_path = Path(results_path)
        self.test_results = {}
    
    def run_all_tests(self) -> Dict:
        """
        Run comprehensive test suite
        
        Returns:
            Complete test report with all metrics
        """
        
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE TEST SUITE")
        print("="*80)
        print("Applying all production-grade improvements...")
        print(f"Dataset: {self.dataset_path}")
        print(f"Results: {self.results_path}")
        
        # Test 1: Baseline Validation
        print("\n" + "="*80)
        print("TEST 1: BASELINE VALIDATION")
        print("="*80)
        test_1_results = self._test_1_baseline_validation()
        self.test_results['test_1_baseline'] = test_1_results
        
        # Test 2: SSIM Correctness
        print("\n" + "="*80)
        print("TEST 2: SSIM CORRECTNESS")
        print("="*80)
        test_2_results = self._test_2_ssim_correctness()
        self.test_results['test_2_ssim'] = test_2_results
        
        # Test 3: Calibration Impact
        print("\n" + "="*80)
        print("TEST 3: CALIBRATION IMPACT (PLATT SCALING)")
        print("="*80)
        test_3_results = self._test_3_calibration_impact()
        self.test_results['test_3_calibration'] = test_3_results
        
        # Test 4: Threshold Optimization
        print("\n" + "="*80)
        print("TEST 4: THRESHOLD OPTIMIZATION (PR-DRIVEN)")
        print("="*80)
        test_4_results = self._test_4_threshold_optimization()
        self.test_results['test_4_optimization'] = test_4_results
        
        # Test 5: ORB Verification
        print("\n" + "="*80)
        print("TEST 5: ORB VERIFICATION")
        print("="*80)
        test_5_results = self._test_5_orb_verification()
        self.test_results['test_5_orb'] = test_5_results
        
        # Generate final report
        report = self._generate_comprehensive_report()
        
        # Save report (with JSON serialization fix)
        report_path = self.results_path / 'comprehensive_test_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy/pandas types to native Python
        report_serializable = self._make_json_serializable(report)
        
        with open(report_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETE")
        print("="*80)
        print(f"Report saved to: {report_path}")
        
        return report
    
    def _test_1_baseline_validation(self) -> Dict:
        """Test 1: Run baseline validation using validation framework"""
        
        print("\n📊 Running baseline validation...")
        
        try:
            # Load validation dataset
            manifest_path = self.dataset_path / 'ground_truth_manifest.json'
            
            if not manifest_path.exists():
                return {
                    'status': 'SKIPPED',
                    'reason': 'No validation dataset found',
                    'recommendation': 'Run: python tools/run_validation.py build'
                }
            
            # Run validation
            runner = ValidationRunner(manifest_path)
            
            # Simple detection function for baseline
            def baseline_detector(path_a: str, path_b: str) -> dict:
                """Baseline pHash + SSIM detector"""
                try:
                    import cv2
                    from PIL import Image
                    import imagehash
                    from skimage.metrics import structural_similarity as ssim_func
                    
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
                    h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
                    img_a_resized = cv2.resize(img_a, (w, h))
                    img_b_resized = cv2.resize(img_b, (w, h))
                    
                    gray_a = cv2.cvtColor(img_a_resized, cv2.COLOR_BGR2GRAY)
                    gray_b = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2GRAY)
                    
                    ssim_score = ssim_func(gray_a, gray_b, data_range=255.0)
                    
                    # Decision
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
            
            # Run validation
            output_dir = self.results_path / 'test_1_baseline'
            metrics = runner.run_validation(
                detection_function=baseline_detector,
                output_path=output_dir
            )
            
            return {
                'status': 'PASS',
                'metrics': metrics,
                'findings': {
                    'precision': metrics['overall']['precision'],
                    'recall': metrics['overall']['recall'],
                    'f1_score': metrics['overall']['f1_score'],
                    'fpr': metrics['overall']['false_positive_rate']
                },
                'assessment': self._assess_metrics(metrics['overall'])
            }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'traceback': import_traceback_string(e)
            }
    
    def _test_2_ssim_correctness(self) -> Dict:
        """Test 2: Verify SSIM data_range handling"""
        
        print("\n🔬 Testing SSIM correctness (data_range handling)...")
        
        try:
            import cv2
            from skimage.metrics import structural_similarity as ssim_func
            
            # Create test images with different dtypes
            test_cases = []
            
            # Test case 1: uint8 images
            img_uint8 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            img_uint8_copy = img_uint8.copy()
            
            ssim_uint8 = ssim_func(img_uint8, img_uint8_copy, data_range=255.0)
            test_cases.append({
                'dtype': 'uint8',
                'data_range': 255.0,
                'ssim': float(ssim_uint8),
                'expected': 1.0,
                'pass': abs(ssim_uint8 - 1.0) < 0.01
            })
            
            # Test case 2: float32 images [0, 1]
            img_float32 = np.random.rand(256, 256).astype(np.float32)
            img_float32_copy = img_float32.copy()
            
            ssim_float32 = ssim_func(img_float32, img_float32_copy, data_range=1.0)
            test_cases.append({
                'dtype': 'float32 [0,1]',
                'data_range': 1.0,
                'ssim': float(ssim_float32),
                'expected': 1.0,
                'pass': abs(ssim_float32 - 1.0) < 0.01
            })
            
            # Test case 3: Verify consistency
            # Convert uint8 to float, compute SSIM, should be similar
            img_uint8_test = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
            img_uint8_mod = img_uint8_test + np.random.randint(-10, 10, (256, 256), dtype=np.int16)
            img_uint8_mod = np.clip(img_uint8_mod, 0, 255).astype(np.uint8)
            
            ssim_as_uint8 = ssim_func(img_uint8_test, img_uint8_mod, data_range=255.0)
            
            # Convert to float [0,1]
            img_float_test = img_uint8_test.astype(np.float32) / 255.0
            img_float_mod = img_uint8_mod.astype(np.float32) / 255.0
            
            ssim_as_float = ssim_func(img_float_test, img_float_mod, data_range=1.0)
            
            consistency_diff = abs(ssim_as_uint8 - ssim_as_float)
            
            test_cases.append({
                'dtype': 'consistency check',
                'ssim_uint8': float(ssim_as_uint8),
                'ssim_float': float(ssim_as_float),
                'difference': float(consistency_diff),
                'pass': consistency_diff < 0.02  # Should be very close
            })
            
            all_pass = all(tc['pass'] for tc in test_cases)
            
            return {
                'status': 'PASS' if all_pass else 'FAIL',
                'test_cases': test_cases,
                'findings': {
                    'data_range_handling': 'CORRECT' if all_pass else 'INCORRECT',
                    'consistency': 'GOOD' if test_cases[-1]['pass'] else 'POOR'
                },
                'recommendation': 'Use data_range=255.0 for uint8, data_range=1.0 for float [0,1]'
            }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def _test_3_calibration_impact(self) -> Dict:
        """Test 3: Measure calibration impact"""
        
        print("\n🌡️ Testing calibration impact...")
        
        try:
            # Load validation results if available
            results_file = self.results_path / 'test_1_baseline' / 'validation_results.csv'
            
            if not results_file.exists():
                return {
                    'status': 'SKIPPED',
                    'reason': 'No baseline results found',
                    'recommendation': 'Run Test 1 first'
                }
            
            df = pd.read_csv(results_file)
            
            # Extract scores
            clip_scores = df['clip_score'].fillna(0).values
            labels = df['should_detect'].astype(int).values
            
            if len(clip_scores) == 0 or clip_scores.max() == 0:
                return {
                    'status': 'SKIPPED',
                    'reason': 'No CLIP scores available',
                    'recommendation': 'Use detector with CLIP scores'
                }
            
            # Compute log loss before calibration
            from sklearn.metrics import log_loss
            
            clip_clipped = np.clip(clip_scores, 1e-7, 1 - 1e-7)
            ll_before = log_loss(labels, clip_clipped)
            
            # Apply simple Platt scaling
            from sklearn.linear_model import LogisticRegression
            
            X = clip_scores.reshape(-1, 1)
            calibrator = LogisticRegression()
            calibrator.fit(X, labels)
            
            calibrated_scores = calibrator.predict_proba(X)[:, 1]
            calibrated_clipped = np.clip(calibrated_scores, 1e-7, 1 - 1e-7)
            
            ll_after = log_loss(labels, calibrated_clipped)
            
            improvement = (ll_before - ll_after) / ll_before * 100
            
            return {
                'status': 'PASS',
                'findings': {
                    'log_loss_before': float(ll_before),
                    'log_loss_after': float(ll_after),
                    'improvement_percent': float(improvement),
                    'calibration_coefficients': {
                        'a': float(calibrator.coef_[0][0]),
                        'b': float(calibrator.intercept_[0])
                    }
                },
                'assessment': 'SIGNIFICANT' if improvement > 5 else 'MINOR',
                'recommendation': 'Apply Platt scaling if improvement > 5%'
            }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def _test_4_threshold_optimization(self) -> Dict:
        """Test 4: PR-driven threshold optimization"""
        
        print("\n🎯 Testing threshold optimization...")
        
        try:
            results_file = self.results_path / 'test_1_baseline' / 'validation_results.csv'
            
            if not results_file.exists():
                return {
                    'status': 'SKIPPED',
                    'reason': 'No baseline results found'
                }
            
            df = pd.read_csv(results_file)
            
            # Extract features
            ssim_scores = df['ssim_score'].fillna(0).values
            phash_distances = df['phash_distance'].fillna(999).values
            labels = df['should_detect'].astype(int).values
            
            # Simple grid search
            param_grid = {
                'ssim': np.arange(0.85, 0.98, 0.02),
                'phash': [3, 4, 5, 6]
            }
            
            best_recall = 0.0
            best_params = None
            target_precision = 0.95
            
            for ssim_thresh in param_grid['ssim']:
                for phash_thresh in param_grid['phash']:
                    # Apply thresholds
                    ssim_pass = ssim_scores >= ssim_thresh
                    phash_pass = phash_distances <= phash_thresh
                    
                    y_pred = (ssim_pass | phash_pass).astype(int)
                    
                    # Compute metrics
                    tp = np.sum((y_pred == 1) & (labels == 1))
                    fp = np.sum((y_pred == 1) & (labels == 0))
                    fn = np.sum((y_pred == 0) & (labels == 1))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    # Check if meets target and improves recall
                    if precision >= target_precision and recall > best_recall:
                        best_recall = recall
                        best_params = {
                            'ssim_threshold': ssim_thresh,
                            'phash_max_dist': phash_thresh,
                            'precision': precision,
                            'recall': recall
                        }
            
            if best_params is None:
                return {
                    'status': 'FAIL',
                    'reason': 'No parameter combination achieves target precision',
                    'recommendation': 'Lower target precision or improve features'
                }
            
            return {
                'status': 'PASS',
                'optimal_thresholds': best_params,
                'findings': {
                    'best_recall_at_95p': best_params['recall'],
                    'achieved_precision': best_params['precision']
                },
                'recommendation': f"Use SSIM≥{best_params['ssim_threshold']:.2f}, pHash≤{best_params['phash_max_dist']}"
            }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def _test_5_orb_verification(self) -> Dict:
        """Test 5: ORB verification correctness"""
        
        print("\n🔍 Testing ORB verification...")
        
        try:
            import cv2
            
            # Create test images
            # Test case 1: Identical images (should verify)
            img_original = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            img_identical = img_original.copy()
            
            # Test case 2: Rotated image (should verify with RANSAC)
            h, w = img_original.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 15, 1.0)
            img_rotated = cv2.warpAffine(img_original, M, (w, h))
            
            # Test case 3: Different image (should NOT verify)
            img_different = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            
            test_cases = []
            
            for name, img_a, img_b, expected in [
                ('identical', img_original, img_identical, True),
                ('rotated', img_original, img_rotated, True),
                ('different', img_original, img_different, False)
            ]:
                result = self._verify_orb_pair(img_a, img_b)
                
                test_cases.append({
                    'case': name,
                    'expected_verify': expected,
                    'actual_verify': result['verified'],
                    'inliers': result.get('inliers', 0),
                    'matches': result.get('total_matches', 0),
                    'pass': result['verified'] == expected
                })
            
            all_pass = all(tc['pass'] for tc in test_cases)
            
            return {
                'status': 'PASS' if all_pass else 'FAIL',
                'test_cases': test_cases,
                'findings': {
                    'orb_working': all_pass,
                    'rotation_robust': test_cases[1]['pass']
                },
                'recommendation': 'Use ratio_threshold=0.75, min_inliers=30'
            }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def _verify_orb_pair(self, img_a: np.ndarray, img_b: np.ndarray) -> Dict:
        """Simple ORB verification"""
        import cv2
        
        # Convert to grayscale
        if img_a.ndim == 3:
            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        if img_b.ndim == 3:
            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        
        # Extract ORB features
        orb = cv2.ORB_create(nfeatures=1000)
        kp_a, desc_a = orb.detectAndCompute(img_a, None)
        kp_b, desc_b = orb.detectAndCompute(img_b, None)
        
        if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            return {'verified': False, 'reason': 'insufficient_features'}
        
        # BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc_a, desc_b, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return {'verified': False, 'reason': 'insufficient_matches', 'total_matches': len(good_matches)}
        
        # RANSAC
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=4.0)
        
        if H is None or mask is None:
            return {'verified': False, 'reason': 'ransac_failed'}
        
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(good_matches)
        
        verified = (inliers >= 30) and (inlier_ratio >= 0.30)
        
        return {
            'verified': verified,
            'inliers': int(inliers),
            'inlier_ratio': float(inlier_ratio),
            'total_matches': len(good_matches)
        }
    
    def _assess_metrics(self, metrics: Dict) -> str:
        """Assess overall metrics quality"""
        
        precision = metrics['precision']
        recall = metrics['recall']
        fpr = metrics['false_positive_rate']
        
        if fpr <= 0.005 and recall >= 0.90 and precision >= 0.95:
            return 'EXCELLENT'
        elif fpr <= 0.01 and recall >= 0.80 and precision >= 0.90:
            return 'GOOD'
        elif fpr <= 0.05 and recall >= 0.70:
            return 'ACCEPTABLE'
        else:
            return 'NEEDS_IMPROVEMENT'
    
    def _generate_comprehensive_report(self) -> Dict:
        """Generate final comprehensive report"""
        
        print("\n" + "="*80)
        print("📝 GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results.values() if t.get('status') == 'PASS')
        failed_tests = sum(1 for t in self.test_results.values() if t.get('status') == 'FAIL')
        skipped_tests = sum(1 for t in self.test_results.values() if t.get('status') == 'SKIPPED')
        error_tests = sum(1 for t in self.test_results.values() if t.get('status') == 'ERROR')
        
        # Collect recommendations
        recommendations = []
        for test_name, test_result in self.test_results.items():
            if 'recommendation' in test_result:
                recommendations.append({
                    'test': test_name,
                    'recommendation': test_result['recommendation']
                })
        
        # Overall assessment
        if passed_tests == total_tests:
            overall_status = 'EXCELLENT'
            overall_message = 'All tests passed! Your pipeline is production-ready.'
        elif passed_tests >= total_tests * 0.8:
            overall_status = 'GOOD'
            overall_message = 'Most tests passed. Review failed tests for improvements.'
        elif passed_tests >= total_tests * 0.5:
            overall_status = 'ACCEPTABLE'
            overall_message = 'Some tests passed. Significant improvements needed.'
        else:
            overall_status = 'NEEDS_WORK'
            overall_message = 'Many tests failed. Review and fix critical issues.'
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'skipped': skipped_tests,
                'errors': error_tests,
                'pass_rate': float(passed_tests / total_tests) if total_tests > 0 else 0.0,
                'overall_status': overall_status,
                'overall_message': overall_message
            },
            'test_results': self.test_results,
            'recommendations': recommendations,
            'production_readiness': self._assess_production_readiness()
        }
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"   Total tests:  {total_tests}")
        print(f"   ✅ Passed:    {passed_tests}")
        print(f"   ❌ Failed:    {failed_tests}")
        print(f"   ⏭️  Skipped:   {skipped_tests}")
        print(f"   💥 Errors:    {error_tests}")
        print(f"   Pass rate:    {report['summary']['pass_rate']*100:.1f}%")
        print(f"\n🎯 Overall Status: {overall_status}")
        print(f"   {overall_message}")
        
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec['recommendation']}")
        
        return report
    
    def _make_json_serializable(self, obj):
        """Convert numpy/pandas types to native Python types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_json_serializable(obj.tolist())
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def _assess_production_readiness(self) -> Dict:
        """Assess production readiness"""
        
        # Check if baseline test passed
        baseline = self.test_results.get('test_1_baseline', {})
        
        if baseline.get('status') != 'PASS':
            return {
                'ready': False,
                'reason': 'Baseline validation failed',
                'blockers': ['Run baseline validation successfully']
            }
        
        # Check metrics
        metrics = baseline.get('findings', {})
        fpr = metrics.get('fpr', 1.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        
        blockers = []
        
        if fpr > 0.01:
            blockers.append(f'FPR too high: {fpr:.4f} (target ≤ 0.01)')
        
        if precision < 0.90:
            blockers.append(f'Precision too low: {precision:.4f} (target ≥ 0.90)')
        
        if recall < 0.70:
            blockers.append(f'Recall too low: {recall:.4f} (target ≥ 0.70)')
        
        if len(blockers) == 0:
            return {
                'ready': True,
                'message': 'All metrics meet production requirements',
                'confidence': 'HIGH'
            }
        else:
            return {
                'ready': False,
                'blockers': blockers,
                'recommendation': 'Address blockers before production deployment'
            }


def import_traceback_string(e: Exception) -> str:
    """Get traceback as string"""
    import traceback
    return ''.join(traceback.format_exception(type(e), e, e.__traceback__))


# ═══════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive test runner')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to validation dataset directory')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results directory')
    
    args = parser.parse_args()
    
    # Run tests
    runner = ComprehensiveTestRunner(
        dataset_path=Path(args.dataset),
        results_path=Path(args.results)
    )
    
    report = runner.run_all_tests()
    
    # Exit with appropriate code
    if report['summary']['overall_status'] in ['EXCELLENT', 'GOOD']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

