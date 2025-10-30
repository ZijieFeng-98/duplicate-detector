"""
Document 54 Improvements for Duplicate Detection Pipeline
==========================================================

Key fixes:
1. Conditional SSIM Gate (instead of flat 0.75)
2. Enhanced Confocal FP filtering with rescue logic
3. Same-page context downgrading
4. Improved tier classification logic

These functions can be integrated into ai_pdf_panel_duplicate_check_AUTO.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def apply_conditional_ssim_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    DOCUMENT 54 FIX #1: Conditional SSIM Gate
    
    Keep pairs if: global SSIM ≥ 0.75 OR any structural/geometric evidence
    
    This replaces the flat 0.75 SSIM threshold to preserve recall on:
    - Partial crops/transforms (ORB evidence)
    - Near-exact matches (pHash evidence)
    - Strong local agreement (Patch SSIM evidence)
    """
    print("\n🔒 Applying CONDITIONAL SSIM Gate (Doc 54)...")
    
    # Extract signals
    ssim_global = pd.to_numeric(df.get('SSIM', 0), errors='coerce').fillna(0.0)
    patch_topk = pd.to_numeric(df.get('Patch_SSIM_TopK', 0), errors='coerce').fillna(0.0)
    orb_inliers = pd.to_numeric(df.get('ORB_Inliers', 0), errors='coerce').fillna(0)
    orb_ratio = pd.to_numeric(df.get('Inlier_Ratio', 0), errors='coerce').fillna(0.0)
    orb_coverage = pd.to_numeric(df.get('Crop_Coverage', 0), errors='coerce').fillna(0.0)
    orb_error = pd.to_numeric(df.get('Reproj_Error', 999), errors='coerce').fillna(999.0)
    phash_dist = pd.to_numeric(df.get('Hamming_Distance', 999), errors='coerce').fillna(999)
    
    # Define rescue conditions (alternatives to high global SSIM)
    has_patch_evidence = (patch_topk >= 0.72)
    has_orb_evidence = (
        (orb_inliers >= 30) & 
        (orb_ratio >= 0.30) & 
        (orb_coverage >= 0.85) & 
        (orb_error <= 4.0)
    )
    has_phash_evidence = (phash_dist <= 5)
    
    # CONDITIONAL GATE: Keep if high global SSIM OR any structural evidence
    conditional_pass = (
        (ssim_global >= 0.75) |           # Standard high-quality match
        has_patch_evidence |               # Strong local agreement
        has_orb_evidence |                 # Geometric match (partial/crop)
        has_phash_evidence                 # Perceptual match (near-exact)
    )
    
    df_filtered = df[conditional_pass].copy()
    
    blocked_count = len(df) - len(df_filtered)
    print(f"  ✅ Conditional Gate: Kept {len(df_filtered)}/{len(df)} pairs")
    print(f"     • Blocked {blocked_count} with low SSIM and no rescue evidence")
    
    # Add diagnostic columns
    df_filtered['Has_Patch_Evidence'] = has_patch_evidence[conditional_pass]
    df_filtered['Has_ORB_Evidence'] = has_orb_evidence[conditional_pass]
    df_filtered['Has_pHash_Evidence'] = has_phash_evidence[conditional_pass]
    
    return df_filtered


def annotate_same_page_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    DOCUMENT 54 FIX #2: Same-Page Context Downgrading
    
    Downgrade adjacent same-page pairs UNLESS they have hard evidence
    """
    print("\n📄 Annotating same-page context...")
    
    # Extract page numbers from panel names
    def get_page_num(panel_name: str) -> int:
        import re
        match = re.search(r'page_(\d+)', str(panel_name))
        return int(match.group(1)) if match else -1
    
    df['Page_A'] = df['Path_A'].apply(lambda x: get_page_num(Path(x).name))
    df['Page_B'] = df['Path_B'].apply(lambda x: get_page_num(Path(x).name))
    df['Same_Page'] = (df['Page_A'] == df['Page_B'])
    
    # For same-page pairs, check if adjacent panels
    def extract_panel_num(panel_name: str) -> int:
        import re
        match = re.search(r'panel(\d+)', str(panel_name))
        return int(match.group(1)) if match else -1
    
    df['Panel_Num_A'] = df['Path_A'].apply(lambda x: extract_panel_num(Path(x).name))
    df['Panel_Num_B'] = df['Path_B'].apply(lambda x: extract_panel_num(Path(x).name))
    df['Is_Adjacent'] = (
        df['Same_Page'] & 
        (abs(df['Panel_Num_A'] - df['Panel_Num_B']) == 1)
    )
    
    # Check for hard evidence
    phash_dist = pd.to_numeric(df.get('Hamming_Distance', 999), errors='coerce').fillna(999)
    orb_inliers = pd.to_numeric(df.get('ORB_Inliers', 0), errors='coerce').fillna(0)
    tile_evidence = pd.to_numeric(df.get('Tile_Evidence_Count', 0), errors='coerce').fillna(0)
    
    has_hard_evidence = (
        (phash_dist <= 3) |
        (orb_inliers >= 30) |
        (tile_evidence >= 2)
    )
    
    # Downgrade adjacent same-page pairs WITHOUT hard evidence
    downgrade_mask = (
        df['Is_Adjacent'] & 
        (df.get('Tier') == 'A') & 
        (~has_hard_evidence)
    )
    
    if downgrade_mask.sum() > 0:
        df.loc[downgrade_mask, 'Tier'] = 'B'
        df.loc[downgrade_mask, 'Downgrade_Reason'] = 'Adjacent same-page without hard evidence'
        print(f"  📉 Downgraded {downgrade_mask.sum()} adjacent same-page pairs → Tier B")
    
    return df


def enhanced_confocal_fp_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    DOCUMENT 54 FIX #3: Enhanced Confocal FP Filtering
    
    Only mark as FP if high CLIP + low SSIM + NO patch/ORB/pHash support
    """
    print("\n🔬 Applying enhanced confocal FP filter...")
    
    # Extract signals
    clip_score = pd.to_numeric(df.get('Cosine_Similarity', 0), errors='coerce').fillna(0)
    ssim_score = pd.to_numeric(df.get('SSIM', 0), errors='coerce').fillna(0)
    patch_topk = pd.to_numeric(df.get('Patch_SSIM_TopK', 0), errors='coerce').fillna(0.0)
    orb_inliers = pd.to_numeric(df.get('ORB_Inliers', 0), errors='coerce').fillna(0)
    phash_dist = pd.to_numeric(df.get('Hamming_Distance', 999), errors='coerce').fillna(999)
    
    # Define rescue conditions
    has_patch_evidence = (patch_topk >= 0.72)
    has_orb_evidence = (orb_inliers >= 30)
    has_phash_evidence = (phash_dist <= 5)
    
    # Confocal FP: high CLIP + low SSIM + NO support
    confocal_fp_mask = (
        (clip_score >= 0.96) &
        (ssim_score < 0.60) &
        (phash_dist > 5) &
        (~has_patch_evidence) &  # ← KEY: Don't downgrade if strong local agreement
        (~has_orb_evidence)      # ← KEY: Don't downgrade if geometric match
    )
    
    df['Confocal_FP_Enhanced'] = confocal_fp_mask
    
    # Downgrade confocal FPs: Tier A → B, Tier B → None
    tier_a_downgrade = confocal_fp_mask & (df.get('Tier') == 'A')
    tier_b_filter = confocal_fp_mask & (df.get('Tier') == 'B')
    
    if tier_a_downgrade.sum() > 0:
        df.loc[tier_a_downgrade, 'Tier'] = 'B'
        df.loc[tier_a_downgrade, 'Downgrade_Reason'] = 'Confocal FP (high CLIP, low SSIM, no evidence)'
    
    if tier_b_filter.sum() > 0:
        df.loc[tier_b_filter, 'Tier'] = None
    
    filtered_count = confocal_fp_mask.sum()
    print(f"  ✅ Enhanced confocal FP filter: {filtered_count} pairs marked")
    print(f"     • {tier_a_downgrade.sum()} Tier A → Tier B")
    print(f"     • {tier_b_filter.sum()} Tier B → Filtered")
    
    return df


def apply_doc54_tier_improvements(df: pd.DataFrame) -> pd.DataFrame:
    """
    DOCUMENT 54 COMPREHENSIVE IMPROVEMENTS
    
    Apply all Document 54 improvements in the correct order:
    1. Conditional SSIM Gate (preserves recall)
    2. Enhanced Confocal FP filtering (precision)
    3. Same-page context downgrading (reduces reviewer noise)
    
    Returns: Improved DataFrame with better precision/recall balance
    """
    print("\n" + "="*70)
    print("📋 APPLYING DOCUMENT 54 IMPROVEMENTS")
    print("="*70)
    
    initial_count = len(df)
    initial_tier_a = len(df[df.get('Tier') == 'A'])
    initial_tier_b = len(df[df.get('Tier') == 'B'])
    
    # Step 1: Conditional SSIM Gate (applied to ALL pairs before tier assignment)
    df = apply_conditional_ssim_gate(df)
    
    # Step 2: Enhanced Confocal FP filtering (applied to tiered pairs)
    df = enhanced_confocal_fp_filter(df)
    
    # Step 3: Same-page context downgrading (final polish)
    df = annotate_same_page_context(df)
    
    # Summary
    final_count = len(df)
    final_tier_a = len(df[df.get('Tier') == 'A'])
    final_tier_b = len(df[df.get('Tier') == 'B'])
    
    print("\n" + "="*70)
    print("📊 DOCUMENT 54 IMPROVEMENTS SUMMARY")
    print("="*70)
    print(f"  Total pairs:  {initial_count} → {final_count} ({100*(final_count-initial_count)/max(initial_count,1):+.1f}%)")
    print(f"  Tier A:       {initial_tier_a} → {final_tier_a} ({100*(final_tier_a-initial_tier_a)/max(initial_tier_a,1):+.1f}%)")
    print(f"  Tier B:       {initial_tier_b} → {final_tier_b} ({100*(final_tier_b-initial_tier_b)/max(initial_tier_b,1):+.1f}%)")
    print(f"  Avg SSIM:     {df['SSIM'].mean():.3f}")
    print("="*70 + "\n")
    
    return df


# Integration helper
def integrate_doc54_into_pipeline():
    """
    INTEGRATION INSTRUCTIONS
    
    To integrate Document 54 improvements into your pipeline:
    
    1. In ai_pdf_panel_duplicate_check_AUTO.py, find the line (around line 4873):
       
       if USE_TIER_GATING and len(df_merged) > 0:
           df_merged = apply_tier_gating(df_merged, modality_cache=modality_cache)
    
    2. AFTER the apply_tier_gating call, add:
       
       # Apply Document 54 improvements
       from doc54_improvements import apply_doc54_tier_improvements
       df_merged = apply_doc54_tier_improvements(df_merged)
    
    3. Save and run your pipeline!
    
    This will add:
    - Conditional SSIM gating (preserves ORB/pHash matches with low SSIM)
    - Enhanced confocal FP filtering (with rescue logic)
    - Same-page downgrading (reduces false positives)
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("📋 DOCUMENT 54 IMPROVEMENTS MODULE")
    print("="*70)
    print("\nKey Functions:")
    print("  1. apply_conditional_ssim_gate() - Preserves ORB/pHash matches")
    print("  2. enhanced_confocal_fp_filter() - Better FP discrimination")
    print("  3. annotate_same_page_context() - Reduces reviewer noise")
    print("  4. apply_doc54_tier_improvements() - Apply all improvements")
    print("\nSee integrate_doc54_into_pipeline() for integration instructions")
    print("="*70)

