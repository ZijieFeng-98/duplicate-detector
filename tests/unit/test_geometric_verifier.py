"""
Unit tests for geometric verifier module.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from duplicate_detector.core.geometric_verifier import (
    extract_orb_features,
    match_orb_features,
    estimate_homography_ransac,
    compute_crop_coverage
)


class TestORBExtraction:
    """Test ORB feature extraction."""
    
    def test_extract_orb_features(self, sample_image):
        """Test ORB feature extraction."""
        kp, desc = extract_orb_features(
            str(sample_image),
            max_keypoints=1000,
            retry_scales=[1.0]
        )
        
        # Should return keypoints and descriptors (or None)
        if kp is not None:
            assert isinstance(kp, list)
            assert len(kp) > 0
        
        if desc is not None:
            assert isinstance(desc, np.ndarray)
            assert desc.ndim == 2
    
    def test_extract_orb_features_nonexistent(self, temp_dir):
        """Test ORB extraction on nonexistent file."""
        nonexistent = temp_dir / "nonexistent.png"
        kp, desc = extract_orb_features(str(nonexistent))
        
        assert kp is None
        assert desc is None


class TestORBMatching:
    """Test ORB feature matching."""
    
    def test_match_orb_features(self):
        """Test ORB feature matching."""
        # Create dummy descriptors
        desc_a = np.random.randint(0, 255, (100, 32), dtype=np.uint8)
        desc_b = np.random.randint(0, 255, (100, 32), dtype=np.uint8)
        
        matches = match_orb_features(desc_a, desc_b, ratio_threshold=0.75)
        
        assert isinstance(matches, list)
        # All matches should be cv2.DMatch objects
        for match in matches:
            assert hasattr(match, 'queryIdx')
            assert hasattr(match, 'trainIdx')
            assert hasattr(match, 'distance')
    
    def test_match_orb_features_none(self):
        """Test matching with None descriptors."""
        desc_a = np.random.randint(0, 255, (100, 32), dtype=np.uint8)
        
        matches = match_orb_features(None, desc_a)
        assert matches == []
        
        matches = match_orb_features(desc_a, None)
        assert matches == []


class TestHomographyEstimation:
    """Test homography estimation."""
    
    def test_estimate_homography_ransac_insufficient_matches(self):
        """Test homography with insufficient matches."""
        kp_a = [cv2.KeyPoint(0, 0, 1) for _ in range(10)]
        kp_b = [cv2.KeyPoint(0, 0, 1) for _ in range(10)]
        matches = [cv2.DMatch(i, i, 0) for i in range(10)]
        
        result = estimate_homography_ransac(
            kp_a, kp_b, matches,
            min_inliers=30  # More than available matches
        )
        
        assert result['H'] is None
        assert result['inliers'] == 0
        assert not result['is_partial_dupe']
    
    def test_estimate_homography_ransac_valid(self):
        """Test homography with valid matches."""
        # Create keypoints in a pattern that should match
        kp_a = [
            cv2.KeyPoint(0, 0, 1),
            cv2.KeyPoint(100, 0, 1),
            cv2.KeyPoint(100, 100, 1),
            cv2.KeyPoint(0, 100, 1),
        ] + [cv2.KeyPoint(i*10, i*10, 1) for i in range(30)]
        
        # Create corresponding keypoints with slight offset
        kp_b = [
            cv2.KeyPoint(10, 10, 1),
            cv2.KeyPoint(110, 10, 1),
            cv2.KeyPoint(110, 110, 1),
            cv2.KeyPoint(10, 110, 1),
        ] + [cv2.KeyPoint(i*10+10, i*10+10, 1) for i in range(30)]
        
        matches = [cv2.DMatch(i, i, 0) for i in range(len(kp_a))]
        
        result = estimate_homography_ransac(
            kp_a, kp_b, matches,
            min_inliers=10,
            max_reproj_error=10.0
        )
        
        assert isinstance(result, dict)
        assert 'H' in result
        assert 'homography_type' in result
        assert 'inliers' in result
        assert 'inlier_ratio' in result
        assert 'reproj_error' in result
        assert 'is_partial_dupe' in result
        assert 'is_degenerate' in result


class TestCropCoverage:
    """Test crop coverage computation."""
    
    def test_compute_crop_coverage_none(self):
        """Test coverage with None homography."""
        coverage = compute_crop_coverage(None, (100, 100), (100, 100))
        assert coverage == 0.0
    
    def test_compute_crop_coverage_identity(self):
        """Test coverage with identity homography."""
        H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        coverage = compute_crop_coverage(H, (100, 100), (100, 100))
        
        assert 0.0 <= coverage <= 1.0
    
    def test_compute_crop_coverage_translation(self):
        """Test coverage with translation homography."""
        H = np.array([
            [1, 0, 50],
            [0, 1, 50],
            [0, 0, 1]
        ], dtype=np.float32)
        
        coverage = compute_crop_coverage(H, (100, 100), (200, 200))
        
        assert 0.0 <= coverage <= 1.0

