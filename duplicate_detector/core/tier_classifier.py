"""
Tier Classifier Module

Classifies duplicate pairs into Tier A (high confidence) and Tier B (manual review).
Implements multiple discrimination paths and modality-specific thresholds.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np


def apply_tier_gating(
    df: pd.DataFrame,
    tier_a_phash: int = 3,
    tier_a_clip: float = 0.99,
    tier_a_ssim: float = 0.95,
    tier_b_phash_min: int = 4,
    tier_b_phash_max: int = 5,
    tier_b_clip_min: float = 0.985,
    tier_b_clip_max: float = 0.99,
    tier_b_ssim_min: float = 0.92,
    tier_b_ssim_max: float = 0.95,
    tier_a_orb_inliers: int = 30,
    tier_a_orb_ratio: float = 0.30,
    tier_a_orb_error: float = 4.0,
    tier_a_orb_coverage: float = 0.50,
    tier_a_relaxed_clip: float = 0.94,
    tier_a_relaxed_ssim: float = 0.70,
    tier_a_relaxed_combined: float = 1.64,
    tier_a_western_clip: float = 0.95,
    tier_a_western_ssim: float = 0.60,
    tier_a_western_combined: float = 1.55,
    require_clip_z: bool = False,
    clip_zscore_min: float = 2.0,
    require_patch_min: bool = False,
    patch_min_gate: float = 0.85,
    enable_orb_relax: bool = False,
    orb_relax_clip_min: float = 0.92,
    orb_relax_patch_topk_min: float = 0.75,
    orb_relax_inliers_min: int = 20,
    orb_relax_coverage_min: float = 0.60,
    orb_relax_ratio_min: float = 0.25,
    orb_relax_reproj_max: float = 5.0,
    enable_confocal_fp_filter: bool = True,
    confocal_fp_clip_min: float = 0.96,
    confocal_fp_ssim_max: float = 0.60,
    confocal_fp_phash_min: int = 3,
    modality_cache: Optional[Dict] = None,
    use_modality_specific: bool = False
) -> pd.DataFrame:
    """
    Apply tier gating to classify duplicate pairs
    
    Tier A Paths (high confidence):
    1. Exact: pHash ≤ threshold (rotation-robust)
    2. Strict: CLIP ≥ threshold AND SSIM ≥ threshold
    3. ORB: Partial duplicate with geometric verification
    4. Relaxed: CLIP ≥ relaxed_threshold AND SSIM ≥ relaxed_threshold
    5. Western: CLIP ≥ western_threshold AND SSIM ≥ western_threshold
    
    Tier B Paths (manual review):
    - Borderline cases that don't meet Tier A criteria
    
    Args:
        df: DataFrame with duplicate pairs
        tier_a_phash: pHash threshold for Tier A
        tier_a_clip: CLIP threshold for Tier A strict
        tier_a_ssim: SSIM threshold for Tier A strict
        tier_b_phash_min: Minimum pHash for Tier B
        tier_b_phash_max: Maximum pHash for Tier B
        tier_b_clip_min: Minimum CLIP for Tier B
        tier_b_clip_max: Maximum CLIP for Tier B
        tier_b_ssim_min: Minimum SSIM for Tier B
        tier_b_ssim_max: Maximum SSIM for Tier B
        tier_a_orb_inliers: Minimum ORB inliers
        tier_a_orb_ratio: Minimum ORB inlier ratio
        tier_a_orb_error: Maximum ORB reprojection error
        tier_a_orb_coverage: Minimum crop coverage
        tier_a_relaxed_clip: Relaxed CLIP threshold
        tier_a_relaxed_ssim: Relaxed SSIM threshold
        tier_a_relaxed_combined: Combined relaxed threshold
        tier_a_western_clip: Western blot CLIP threshold
        tier_a_western_ssim: Western blot SSIM threshold
        tier_a_western_combined: Western blot combined threshold
        require_clip_z: Require CLIP z-score
        clip_zscore_min: Minimum CLIP z-score
        require_patch_min: Require patch SSIM minimum
        patch_min_gate: Patch SSIM minimum gate
        enable_orb_relax: Enable relaxed ORB path
        orb_relax_*: Relaxed ORB parameters
        enable_confocal_fp_filter: Enable confocal false positive filter
        confocal_fp_*: Confocal FP filter parameters
        modality_cache: Modality cache dictionary
        use_modality_specific: Use modality-specific thresholds
    
    Returns:
        DataFrame with Tier column added
    """
    if df.empty:
        return df
    
    df = df.copy()
    df['Tier'] = None
    df['Tier_Path'] = None
    df['Confocal_FP'] = False
    
    # Extract signals
    clip_score = pd.to_numeric(df.get('Cosine_Similarity', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    ssim_score = pd.to_numeric(df.get('SSIM', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    phash_dist = pd.to_numeric(df.get('Hamming_Distance', pd.Series([999] * len(df))), errors='coerce').fillna(999)
    orb_inliers = pd.to_numeric(df.get('ORB_Inliers', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    orb_ratio = pd.to_numeric(df.get('Inlier_Ratio', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    orb_error = pd.to_numeric(df.get('Reproj_Error', pd.Series([999] * len(df))), errors='coerce').fillna(999)
    orb_coverage = pd.to_numeric(df.get('Crop_Coverage', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    
    # Modality-specific mode
    if use_modality_specific and modality_cache:
        return _apply_modality_specific_tier_gating(
            df, clip_score, ssim_score, phash_dist, orb_inliers,
            orb_ratio, orb_error, orb_coverage, modality_cache
        )
    
    # Universal mode
    # Path 1: Exact match (pHash with rotation bundles)
    path1_exact = (phash_dist <= tier_a_phash)
    
    # Path 2: Strict (high confidence)
    path2_strict = (clip_score >= tier_a_clip) & (ssim_score >= tier_a_ssim)
    
    # Optional: Require CLIP z-score
    if require_clip_z and 'CLIP_Z' in df.columns:
        clip_z = pd.to_numeric(df.get('CLIP_Z'), errors='coerce').fillna(0)
        path2_strict = path2_strict & (clip_z >= clip_zscore_min)
    
    # Optional: Require minimum patch SSIM
    if require_patch_min and 'Patch_SSIM_Min' in df.columns:
        patch_min = pd.to_numeric(df.get('Patch_SSIM_Min'), errors='coerce').fillna(0)
        path2_strict = path2_strict & (patch_min >= patch_min_gate)
    
    # Path 3: ORB-RANSAC (geometric verification)
    path3_orb = (
        (orb_inliers >= tier_a_orb_inliers) & 
        (orb_ratio >= tier_a_orb_ratio) & 
        (orb_error <= tier_a_orb_error) & 
        (orb_coverage >= tier_a_orb_coverage)
    )
    
    # Path 4: Relaxed (catches compressed/rotated)
    path4_relaxed = (
        (clip_score >= tier_a_relaxed_clip) &
        (ssim_score >= tier_a_relaxed_ssim) &
        ((clip_score + ssim_score) >= tier_a_relaxed_combined)
    )
    
    # Path 5: Western Blot specific (even more tolerant)
    path5_western = (
        (clip_score >= tier_a_western_clip) &
        (ssim_score >= tier_a_western_ssim) &
        ((clip_score + ssim_score) >= tier_a_western_combined)
    )
    
    # Get patch SSIM top-K
    patch_ssim_topk = pd.to_numeric(
        df.get('Patch_SSIM_TopK', df.get('Patch_SSIM_Topk', df.get('Patch_SSIM_TopK_Mean'))),
        errors='coerce'
    )
    if not isinstance(patch_ssim_topk, pd.Series):
        patch_ssim_topk = pd.Series([patch_ssim_topk] * len(df))
    if patch_ssim_topk.isna().all():
        patch_ssim_topk = pd.to_numeric(df.get('Patch_SSIM_Min', ssim_score), errors='coerce').fillna(ssim_score)
    
    # Path 6: Relaxed ORB (optional)
    path6_orb_relax = pd.Series([False] * len(df))
    if enable_orb_relax:
        path6_orb_relax = (
            (clip_score >= orb_relax_clip_min) &
            (patch_ssim_topk >= orb_relax_patch_topk_min) &
            (orb_inliers >= orb_relax_inliers_min) &
            (orb_coverage >= orb_relax_coverage_min) &
            (orb_ratio >= orb_relax_ratio_min) &
            (orb_error <= orb_relax_reproj_max)
        )
    
    # Confocal false positive filter
    if enable_confocal_fp_filter:
        local_or_geom_ok = (
            (patch_ssim_topk >= 0.72) |
            ((orb_inliers >= 20) & (orb_coverage >= 0.65) & (orb_error <= 5.0))
        )
        
        confocal_false_positive = (
            (clip_score >= confocal_fp_clip_min) &
            (ssim_score < confocal_fp_ssim_max) &
            (phash_dist > confocal_fp_phash_min) &
            (~local_or_geom_ok)
        )
        
        df['Confocal_FP'] = confocal_false_positive
    
    # Combine all Tier A paths
    tier_a_mask = (
        path1_exact | path2_strict | path3_orb | path4_relaxed |
        path5_western | path6_orb_relax
    )
    
    # Apply Tier A
    df.loc[tier_a_mask, 'Tier'] = 'A'
    df.loc[path1_exact & tier_a_mask, 'Tier_Path'] = 'Exact (pHash)'
    df.loc[path2_strict & tier_a_mask, 'Tier_Path'] = 'Strict (CLIP+SSIM)'
    df.loc[path3_orb & tier_a_mask, 'Tier_Path'] = 'ORB-RANSAC'
    df.loc[path4_relaxed & tier_a_mask, 'Tier_Path'] = 'Relaxed'
    df.loc[path5_western & tier_a_mask, 'Tier_Path'] = 'Western Blot'
    df.loc[path6_orb_relax & tier_a_mask, 'Tier_Path'] = 'ORB-Relaxed'
    
    # Tier B: Borderline cases
    tier_b_mask = (
        (~tier_a_mask) &
        (
            ((tier_b_phash_min <= phash_dist) & (phash_dist <= tier_b_phash_max)) |
            ((tier_b_clip_min <= clip_score) & (clip_score <= tier_b_clip_max) &
             (tier_b_ssim_min <= ssim_score) & (ssim_score <= tier_b_ssim_max))
        )
    )
    
    df.loc[tier_b_mask, 'Tier'] = 'B'
    df.loc[tier_b_mask & (tier_b_phash_min <= phash_dist) & (phash_dist <= tier_b_phash_max), 'Tier_Path'] = 'Borderline (pHash)'
    df.loc[tier_b_mask & (tier_b_clip_min <= clip_score) & (clip_score <= tier_b_clip_max), 'Tier_Path'] = 'Borderline (CLIP+SSIM)'
    
    return df


def _apply_modality_specific_tier_gating(
    df: pd.DataFrame,
    clip_score: pd.Series,
    ssim_score: pd.Series,
    phash_dist: pd.Series,
    orb_inliers: pd.Series,
    orb_ratio: pd.Series,
    orb_error: pd.Series,
    orb_coverage: pd.Series,
    modality_cache: Dict
) -> pd.DataFrame:
    """
    Apply modality-specific tier gating
    
    Args:
        df: DataFrame with pairs
        clip_score: CLIP scores
        ssim_score: SSIM scores
        phash_dist: pHash distances
        orb_*: ORB metrics
        modality_cache: Modality cache
    
    Returns:
        DataFrame with Tier column
    """
    # Modality-specific parameters
    MODALITY_PARAMS = {
        'western_blot': {
            'tier_a': {'clip': 0.94, 'ssim': 0.60, 'combined': 1.55, 'phash': 3},
            'tier_b': {'clip': 0.92, 'ssim': 0.50, 'combined': 1.45, 'phash': 4},
        },
        'confocal': {
            'tier_a': {'clip': 0.95, 'ssim': 0.75, 'combined': 1.70, 'phash': 3, 'patch_min': 0.70},
            'tier_b': {'clip': 0.92, 'ssim': 0.65, 'combined': 1.60, 'phash': 4},
            'fp_gate': {'clip_high': 0.96, 'ssim_low': 0.50}
        },
        'tem': {
            'tier_a': {'clip': 0.95, 'ssim': 0.85, 'combined': 1.80, 'phash': 2},
            'tier_b': {'clip': 0.93, 'ssim': 0.75, 'combined': 1.70, 'phash': 3}
        },
        'bright_field': {
            'tier_a': {'clip': 0.93, 'ssim': 0.75, 'combined': 1.68, 'phash': 4},
            'tier_b': {'clip': 0.90, 'ssim': 0.65, 'combined': 1.58, 'phash': 5}
        },
        'gel': {
            'tier_a': {'clip': 0.94, 'ssim': 0.70, 'combined': 1.64, 'phash': 3},
            'tier_b': {'clip': 0.91, 'ssim': 0.60, 'combined': 1.54, 'phash': 4}
        },
        'unknown': {
            'tier_a': {'clip': 0.95, 'ssim': 0.65, 'combined': 1.65, 'phash': 3},
            'tier_b': {'clip': 0.92, 'ssim': 0.40, 'combined': 1.50, 'phash': 4}
        }
    }
    
    # Attach modality columns
    df['Modality_A'] = df['Path_A'].map(lambda p: modality_cache.get(str(p), {}).get('modality', 'unknown'))
    df['Modality_B'] = df['Path_B'].map(lambda p: modality_cache.get(str(p), {}).get('modality', 'unknown'))
    df['Same_Modality'] = df['Modality_A'] == df['Modality_B']
    
    # Exact matches first (all modalities)
    exact_mask = (phash_dist <= 2)
    df.loc[exact_mask, 'Tier'] = 'A'
    df.loc[exact_mask, 'Tier_Path'] = 'Exact (pHash)'
    
    # Apply modality-specific rules
    for modality, params in MODALITY_PARAMS.items():
        mask = (df['Modality_A'] == modality) & (df['Modality_B'] == modality) & (~exact_mask)
        
        if mask.sum() == 0:
            continue
        
        # Tier A criteria
        if 'tier_a' in params:
            ta = params['tier_a']
            
            tier_a_combined = mask & (
                (clip_score >= ta['clip']) & 
                (ssim_score >= ta['ssim']) & 
                ((clip_score + ssim_score) >= ta['combined'])
            )
            
            tier_a_phash = mask & (phash_dist <= ta['phash'])
            
            tier_a_final = tier_a_combined | tier_a_phash
            df.loc[tier_a_final, 'Tier'] = 'A'
            df.loc[tier_a_final, 'Tier_Path'] = f'{modality.replace("_", " ").title()}-Specific'
        
        # Tier B criteria
        if 'tier_b' in params:
            tb = params['tier_b']
            tier_b = mask & (df['Tier'].isna()) & (
                (clip_score >= tb['clip']) & 
                (ssim_score >= tb['ssim'])
            )
            df.loc[tier_b, 'Tier'] = 'B'
            df.loc[tier_b, 'Tier_Path'] = f'{modality.replace("_", " ").title()}-Borderline'
        
        # Confocal false positive filter
        if modality == 'confocal' and 'fp_gate' in params:
            fp = params['fp_gate']
            fp_mask = mask & (df['Tier'].isna()) & (
                (clip_score >= fp['clip_high']) & 
                (ssim_score < fp['ssim_low'])
            )
            df.loc[fp_mask, 'Tier_Path'] = 'Confocal-FP-Filtered'
    
    # Unknown or cross-modality: Use universal rule fallback
    unknown_mask = (
        ((df['Modality_A'] == 'unknown') | (df['Modality_B'] == 'unknown') | (~df['Same_Modality'])) &
        (df['Tier'].isna())
    )
    tier_a_universal = unknown_mask & (clip_score >= 0.95) & (ssim_score >= 0.65) & ((clip_score + ssim_score) >= 1.65)
    df.loc[tier_a_universal, 'Tier'] = 'A'
    df.loc[tier_a_universal, 'Tier_Path'] = 'Universal-Fallback'
    
    # Remaining borderline → Tier B
    tier_b_universal = (df['Tier'].isna()) & (clip_score >= 0.92) & (ssim_score >= 0.40)
    df.loc[tier_b_universal, 'Tier'] = 'B'
    df.loc[tier_b_universal, 'Tier_Path'] = 'Universal-Borderline'
    
    return df

