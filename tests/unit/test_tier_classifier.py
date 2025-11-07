"""
Unit tests for tier classifier module.
"""

import pytest
import pandas as pd
import numpy as np

from duplicate_detector.core.tier_classifier import apply_tier_gating


class TestTierGating:
    """Test tier gating functions."""
    
    def test_apply_tier_gating_empty(self):
        """Test tier gating with empty DataFrame."""
        df = pd.DataFrame()
        result = apply_tier_gating(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_apply_tier_gating_tier_a_exact(self):
        """Test Tier A classification via exact pHash match."""
        df = pd.DataFrame([
            {
                'Image_A': 'panel_00.png',
                'Image_B': 'panel_01.png',
                'Path_A': '/path/to/panel_00.png',
                'Path_B': '/path/to/panel_01.png',
                'Cosine_Similarity': 0.50,  # Low CLIP
                'SSIM': 0.50,  # Low SSIM
                'Hamming_Distance': 2,  # Low pHash (exact match)
            }
        ])
        
        result = apply_tier_gating(
            df,
            tier_a_phash=3,
            tier_a_clip=0.99,
            tier_a_ssim=0.95
        )
        
        assert 'Tier' in result.columns
        assert 'Tier_Path' in result.columns
        assert result.iloc[0]['Tier'] == 'A'
        assert 'Exact' in result.iloc[0]['Tier_Path'] or 'pHash' in result.iloc[0]['Tier_Path']
    
    def test_apply_tier_gating_tier_a_strict(self):
        """Test Tier A classification via strict CLIP+SSIM."""
        df = pd.DataFrame([
            {
                'Image_A': 'panel_00.png',
                'Image_B': 'panel_01.png',
                'Path_A': '/path/to/panel_00.png',
                'Path_B': '/path/to/panel_01.png',
                'Cosine_Similarity': 0.995,  # High CLIP
                'SSIM': 0.96,  # High SSIM
                'Hamming_Distance': 10,  # High pHash
            }
        ])
        
        result = apply_tier_gating(
            df,
            tier_a_phash=3,
            tier_a_clip=0.99,
            tier_a_ssim=0.95
        )
        
        assert result.iloc[0]['Tier'] == 'A'
        assert 'Strict' in result.iloc[0]['Tier_Path'] or 'CLIP+SSIM' in result.iloc[0]['Tier_Path']
    
    def test_apply_tier_gating_tier_a_orb(self):
        """Test Tier A classification via ORB-RANSAC."""
        df = pd.DataFrame([
            {
                'Image_A': 'panel_00.png',
                'Image_B': 'panel_01.png',
                'Path_A': '/path/to/panel_00.png',
                'Path_B': '/path/to/panel_01.png',
                'Cosine_Similarity': 0.50,
                'SSIM': 0.50,
                'Hamming_Distance': 10,
                'ORB_Inliers': 35,  # High inliers
                'Inlier_Ratio': 0.35,  # Good ratio
                'Reproj_Error': 3.0,  # Low error
                'Crop_Coverage': 0.60,  # Good coverage
            }
        ])
        
        result = apply_tier_gating(
            df,
            tier_a_orb_inliers=30,
            tier_a_orb_ratio=0.30,
            tier_a_orb_error=4.0,
            tier_a_orb_coverage=0.50
        )
        
        assert result.iloc[0]['Tier'] == 'A'
        assert 'ORB' in result.iloc[0]['Tier_Path']
    
    def test_apply_tier_gating_tier_b(self):
        """Test Tier B classification."""
        df = pd.DataFrame([
            {
                'Image_A': 'panel_00.png',
                'Image_B': 'panel_01.png',
                'Path_A': '/path/to/panel_00.png',
                'Path_B': '/path/to/panel_01.png',
                'Cosine_Similarity': 0.987,  # Borderline CLIP
                'SSIM': 0.93,  # Borderline SSIM
                'Hamming_Distance': 4,  # Borderline pHash
            }
        ])
        
        result = apply_tier_gating(
            df,
            tier_a_clip=0.99,
            tier_a_ssim=0.95,
            tier_b_clip_min=0.985,
            tier_b_clip_max=0.99,
            tier_b_ssim_min=0.92,
            tier_b_ssim_max=0.95,
            tier_b_phash_min=4,
            tier_b_phash_max=5
        )
        
        assert result.iloc[0]['Tier'] == 'B'
    
    def test_apply_tier_gating_no_tier(self):
        """Test that low-confidence pairs get no tier."""
        df = pd.DataFrame([
            {
                'Image_A': 'panel_00.png',
                'Image_B': 'panel_01.png',
                'Path_A': '/path/to/panel_00.png',
                'Path_B': '/path/to/panel_01.png',
                'Cosine_Similarity': 0.50,  # Very low
                'SSIM': 0.50,  # Very low
                'Hamming_Distance': 20,  # Very high
            }
        ])
        
        result = apply_tier_gating(df)
        
        # Should not be Tier A or B
        assert pd.isna(result.iloc[0]['Tier']) or result.iloc[0]['Tier'] is None
    
    def test_apply_tier_gating_confocal_fp_filter(self):
        """Test confocal false positive filter."""
        df = pd.DataFrame([
            {
                'Image_A': 'panel_00.png',
                'Image_B': 'panel_01.png',
                'Path_A': '/path/to/panel_00.png',
                'Path_B': '/path/to/panel_01.png',
                'Cosine_Similarity': 0.97,  # High CLIP
                'SSIM': 0.55,  # Low SSIM (typical FP pattern)
                'Hamming_Distance': 8,  # Not exact match
                'Patch_SSIM_TopK': 0.50,  # Low patch SSIM
            }
        ])
        
        result = apply_tier_gating(
            df,
            enable_confocal_fp_filter=True,
            confocal_fp_clip_min=0.96,
            confocal_fp_ssim_max=0.60,
            confocal_fp_phash_min=3
        )
        
        assert 'Confocal_FP' in result.columns
        # Should be marked as FP (not Tier A)
        if result.iloc[0]['Confocal_FP']:
            assert result.iloc[0]['Tier'] != 'A'

