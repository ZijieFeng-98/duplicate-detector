"""
Unit tests for similarity engine module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from duplicate_detector.core.similarity_engine import (
    compute_file_hash,
    get_cache_path,
    get_cache_meta_path,
    phash_hex,
    compute_phash_bundle,
    hamming_min_transform,
    normalize_photometric,
    apply_clahe
)


class TestCacheHelpers:
    """Test cache helper functions."""
    
    def test_compute_file_hash(self, sample_panel_paths):
        """Test file hash computation."""
        hash1 = compute_file_hash(sample_panel_paths)
        hash2 = compute_file_hash(sample_panel_paths)
        
        # Same files should produce same hash
        assert hash1 == hash2
        
        # Different files should produce different hash
        different_paths = sample_panel_paths[:2]
        hash3 = compute_file_hash(different_paths)
        assert hash1 != hash3
    
    def test_get_cache_path(self, temp_dir):
        """Test cache path generation."""
        cache_path = get_cache_path("test_cache", temp_dir, "v1")
        assert cache_path.parent == temp_dir / "cache"
        assert cache_path.name == "test_cache_v1.npy"
        
        # Should create parent directory
        assert cache_path.parent.exists()
    
    def test_get_cache_meta_path(self, temp_dir):
        """Test metadata cache path generation."""
        meta_path = get_cache_meta_path("test_cache", temp_dir, "v1")
        assert meta_path.parent == temp_dir / "cache"
        assert meta_path.name == "test_cache_v1_meta.json"


class TestPHash:
    """Test perceptual hashing functions."""
    
    def test_phash_hex(self, sample_image):
        """Test pHash computation."""
        hash_str = phash_hex(sample_image)
        
        assert isinstance(hash_str, str)
        assert len(hash_str) > 0
    
    def test_compute_phash_bundle(self, sample_image):
        """Test pHash bundle computation."""
        img = Image.open(sample_image)
        bundle = compute_phash_bundle(img)
        
        assert isinstance(bundle, dict)
        # Should have 8 transforms
        expected_keys = [
            'rot_0', 'rot_90', 'rot_180', 'rot_270',
            'mirror_h_rot_0', 'mirror_h_rot_90', 'mirror_h_rot_180', 'mirror_h_rot_270'
        ]
        for key in expected_keys:
            assert key in bundle
            assert isinstance(bundle[key], str)
    
    def test_hamming_min_transform(self):
        """Test Hamming distance computation across transforms."""
        bundle_a = {
            'rot_0': 'a' * 16,
            'rot_90': 'b' * 16,
        }
        bundle_b = {
            'rot_0': 'a' * 16,  # Should match bundle_a['rot_0']
            'rot_90': 'c' * 16,
        }
        
        min_dist, transform = hamming_min_transform(bundle_a, bundle_b)
        
        assert isinstance(min_dist, int)
        assert min_dist >= 0
        assert isinstance(transform, str)
        
        # Identical bundles should have distance 0
        bundle_c = bundle_a.copy()
        min_dist_identical, _ = hamming_min_transform(bundle_a, bundle_c)
        assert min_dist_identical == 0


class TestImageProcessing:
    """Test image processing functions."""
    
    def test_normalize_photometric(self):
        """Test photometric normalization."""
        import cv2
        import numpy as np
        
        # Create a test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        normalized, params = normalize_photometric(img, apply_clahe_flag=False)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape[:2] == img.shape[:2]
        assert normalized.dtype == np.uint8
        assert isinstance(params, dict)
        assert 'mean' in params
        assert 'std' in params
    
    def test_apply_clahe(self):
        """Test CLAHE application."""
        import cv2
        import numpy as np
        
        # Create grayscale test image
        img_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        enhanced = apply_clahe(img_gray)
        
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == img_gray.shape
        assert enhanced.dtype == np.uint8


class TestCLIPFunctions:
    """Test CLIP-related functions."""
    
    @pytest.mark.skip(reason="Requires CLIP model loading - integration test")
    def test_load_clip(self):
        """Test CLIP model loading."""
        # This would require actual model download
        pass
    
    @pytest.mark.skip(reason="Requires CLIP model - integration test")
    def test_embed_images(self):
        """Test image embedding."""
        pass


class TestSSIMFunctions:
    """Test SSIM-related functions."""
    
    @pytest.mark.skip(reason="Requires actual image files - integration test")
    def test_compute_ssim_normalized(self):
        """Test SSIM computation."""
        # This requires actual image files
        pass

