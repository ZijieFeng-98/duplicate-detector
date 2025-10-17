#!/usr/bin/env python3
"""
Modality-aware KPI summary (post-run analysis, no app changes)

Usage:
  python3 tools/modality_kpi.py output_dir/final_merged_report.tsv

Output: 10-line per-modality KPI summary
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def get_modality_from_path(p: str, modality_col: str = None, df_row=None) -> str:
    """
    Infer modality from filename patterns if Modality columns don't exist.
    This is a best-effort heuristic for post-run analysis.
    """
    if modality_col and df_row is not None and modality_col in df_row:
        return str(df_row[modality_col])
    
    # Fallback: filename heuristics (rough approximation)
    p_lower = str(p).lower()
    if 'confocal' in p_lower or 'fluor' in p_lower:
        return 'confocal'
    elif 'ihc' in p_lower or 'dab' in p_lower or 'hema' in p_lower:
        return 'bright_field'
    elif 'gel' in p_lower or 'blot' in p_lower:
        return 'gel'
    elif 'tem' in p_lower or 'electron' in p_lower:
        return 'tem'
    else:
        return 'unknown'

def analyze_modality_kpis(tsv_path: str):
    """Generate per-modality KPI summary"""
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    
    print(f"\n{'='*70}")
    print(f"  🔬 MODALITY-AWARE KPI SUMMARY")
    print(f"{'='*70}")
    print(f"  File: {Path(tsv_path).name}")
    print(f"  Total pairs: {len(df)}")
    
    # Check if Modality columns exist (exposed mode)
    has_modality = 'Modality_A' in df.columns and 'Modality_B' in df.columns
    
    if not has_modality:
        print(f"  Note: No Modality_A/B columns (internal routing mode)")
        print(f"        Using heuristics for post-analysis\n")
    
    # Coerce signals
    clip = pd.to_numeric(df.get("Cosine_Similarity", 0), errors="coerce").fillna(0.0)
    ssim = pd.to_numeric(df.get("SSIM", 0), errors="coerce").fillna(0.0)
    tier = df.get("Tier", pd.Series([""] * len(df))).astype(str)
    confocal_fp = df.get("Confocal_FP", pd.Series([False] * len(df))).astype(bool)
    
    # Deep-verify columns (ensure Series, not scalar)
    deep_ssim = pd.to_numeric(df.get("Deep_SSIM", pd.Series([np.nan]*len(df))), errors="coerce")
    deep_ncc = pd.to_numeric(df.get("Deep_NCC", pd.Series([np.nan]*len(df))), errors="coerce")
    ihc_ssim = pd.to_numeric(df.get("IHC_SSIM", pd.Series([np.nan]*len(df))), errors="coerce")
    ihc_ncc = pd.to_numeric(df.get("IHC_NCC", pd.Series([np.nan]*len(df))), errors="coerce")
    
    # Infer modality per pair
    if has_modality:
        # Use actual columns
        modality_pairs = list(zip(df['Modality_A'], df['Modality_B']))
        pair_modality = [ma if ma == mb else 'mixed' for ma, mb in modality_pairs]
    else:
        # Use heuristics
        pair_modality = [
            get_modality_from_path(df.iloc[i]['Path_A'])
            for i in range(len(df))
        ]
    
    df['_modality'] = pair_modality
    
    # Overall stats
    print(f"{'─'*70}")
    print(f"  OVERALL")
    print(f"{'─'*70}")
    tier_a = (tier == 'A').sum()
    tier_b = (tier == 'B').sum()
    fp_proxy = confocal_fp.sum()
    
    print(f"  Tier A: {tier_a} ({100*tier_a/len(df):.1f}%)")
    print(f"  Tier B: {tier_b} ({100*tier_b/len(df):.1f}%)")
    print(f"  Confocal FP: {fp_proxy} ({100*fp_proxy/len(df):.1f}%)")
    
    # Deep-verify stats
    confocal_dv_ran = deep_ssim.notna().sum()
    confocal_dv_promoted = ((deep_ssim >= 0.90) & (deep_ncc >= 0.985)).sum()
    ihc_dv_ran = ihc_ssim.notna().sum()
    ihc_dv_promoted = ((ihc_ssim >= 0.88) & (ihc_ncc >= 0.980)).sum()
    
    print(f"\n  Deep Verify:")
    print(f"    Confocal: {confocal_dv_ran} ran, {confocal_dv_promoted} promoted")
    print(f"    IHC: {ihc_dv_ran} ran, {ihc_dv_promoted} promoted")
    
    # Per-modality breakdown
    modalities = df['_modality'].value_counts().index.tolist()
    
    for mod in sorted(modalities):
        if mod == 'mixed':
            continue  # Skip mixed pairs
        
        mod_df = df[df['_modality'] == mod]
        if len(mod_df) == 0:
            continue
        
        n = len(mod_df)
        mod_tier_a = (mod_df['Tier'] == 'A').sum()
        mod_tier_b = (mod_df['Tier'] == 'B').sum()
        mod_fp = mod_df['Confocal_FP'].sum() if 'Confocal_FP' in mod_df else 0
        
        # FP proxy (high CLIP, low SSIM)
        mod_clip = pd.to_numeric(mod_df['Cosine_Similarity'], errors='coerce').fillna(0)
        mod_ssim = pd.to_numeric(mod_df['SSIM'], errors='coerce').fillna(0)
        fp_prox = ((mod_clip >= 0.96) & (mod_ssim < 0.60)).sum()
        
        # Deep-verify for this modality
        mod_deep_ssim = pd.to_numeric(mod_df.get('Deep_SSIM', np.nan), errors='coerce')
        mod_deep_ncc = pd.to_numeric(mod_df.get('Deep_NCC', np.nan), errors='coerce')
        dv_ran = mod_deep_ssim.notna().sum()
        dv_pass = ((mod_deep_ssim >= 0.90) & (mod_deep_ncc >= 0.985)).sum() if dv_ran > 0 else 0
        
        print(f"\n{'─'*70}")
        print(f"  {mod.upper()} ({n} pairs)")
        print(f"{'─'*70}")
        print(f"  Tier A: {mod_tier_a} ({100*mod_tier_a/n:.1f}%)")
        print(f"  Tier B: {mod_tier_b} ({100*mod_tier_b/n:.1f}%)")
        print(f"  FP proxy (CLIP≥0.96 & SSIM<0.60): {fp_prox} ({100*fp_prox/n:.1f}%)")
        print(f"  Confocal FP flagged: {mod_fp}")
        
        if dv_ran > 0:
            print(f"  Deep Verify: {dv_ran} ran, {dv_pass} promoted ({100*dv_pass/dv_ran:.1f}%)")
            if dv_pass > 0:
                avg_ssim = mod_deep_ssim[mod_deep_ssim >= 0.90].mean()
                avg_ncc = mod_deep_ncc[mod_deep_ncc >= 0.985].mean()
                print(f"    Promoted avg: Deep_SSIM={avg_ssim:.3f}, Deep_NCC={avg_ncc:.3f}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/modality_kpi.py <final_merged_report.tsv>")
        sys.exit(1)
    
    tsv = sys.argv[1]
    if not Path(tsv).exists():
        print(f"Error: File not found: {tsv}")
        sys.exit(1)
    
    analyze_modality_kpis(tsv)

