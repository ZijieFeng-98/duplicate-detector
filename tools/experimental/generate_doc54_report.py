#!/usr/bin/env python3
"""Generate comprehensive Document 54 improvements report"""
import pandas as pd
import json
from pathlib import Path

print("="*70)
print("📊 DOCUMENT 54 IMPROVEMENTS - COMPREHENSIVE REPORT")
print("="*70)

# Load new results (with Document 54 improvements)
df_new = pd.read_csv('doc54_results/final_merged_report.tsv', sep='\t')

print(f"\n{'='*70}")
print("📈 RESULTS WITH DOCUMENT 54 IMPROVEMENTS")
print(f"{'='*70}")

# Basic stats
print(f"\n✅ Total duplicate pairs found: {len(df_new)}")

# Tier breakdown
if 'Tier' in df_new.columns:
    print(f'\n📊 Breakdown by Tier:')
    tier_a = len(df_new[df_new['Tier'] == 'A'])
    tier_b = len(df_new[df_new['Tier'] == 'B'])
    print(f'   • Tier A (High Confidence): {tier_a} pairs ({tier_a/len(df_new)*100:.1f}%)')
    print(f'   • Tier B (Review Needed): {tier_b} pairs ({tier_b/len(df_new)*100:.1f}%)')

# Document 54 specific features
if 'Has_Patch_Evidence' in df_new.columns:
    patch_rescued = df_new['Has_Patch_Evidence'].sum()
    print(f'\n🔍 Document 54 Rescue Statistics:')
    print(f'   • Pairs rescued by Patch SSIM: {patch_rescued}')

if 'Has_ORB_Evidence' in df_new.columns:
    orb_rescued = df_new['Has_ORB_Evidence'].sum()
    print(f'   • Pairs rescued by ORB: {orb_rescued}')

if 'Has_pHash_Evidence' in df_new.columns:
    phash_rescued = df_new['Has_pHash_Evidence'].sum()
    print(f'   • Pairs rescued by pHash: {phash_rescued}')

# Same-page analysis
if 'Same_Page' in df_new.columns:
    same_page = df_new['Same_Page'].sum()
    cross_page = len(df_new) - same_page
    print(f'\n📄 Page Distribution:')
    print(f'   • Same-page pairs: {same_page} ({same_page/len(df_new)*100:.1f}%)')
    print(f'   • Cross-page pairs: {cross_page} ({cross_page/len(df_new)*100:.1f}%)')
    
    if 'Is_Adjacent' in df_new.columns:
        adjacent = df_new['Is_Adjacent'].sum()
        print(f'   • Adjacent panels: {adjacent}')

# Downgrade analysis
if 'Downgrade_Reason' in df_new.columns:
    downgrades = df_new[df_new['Downgrade_Reason'].notna()]
    if len(downgrades) > 0:
        print(f'\n📉 Downgraded Pairs:')
        print(f'   • Total downgrades: {len(downgrades)}')
        downgrade_reasons = downgrades['Downgrade_Reason'].value_counts()
        for reason, count in downgrade_reasons.items():
            print(f'   • {reason}: {count}')

# Enhanced Confocal FP
if 'Confocal_FP_Enhanced' in df_new.columns:
    confocal_fp = df_new['Confocal_FP_Enhanced'].sum()
    print(f'\n🔬 Enhanced Confocal FP Filter:')
    print(f'   • Pairs marked as Confocal FP: {confocal_fp}')

# Similarity metrics
print(f'\n📊 Similarity Scores (averages):')
print(f'   • CLIP: {df_new["Cosine_Similarity"].mean():.3f} (range: {df_new["Cosine_Similarity"].min():.3f}-{df_new["Cosine_Similarity"].max():.3f})')
print(f'   • SSIM: {df_new["SSIM"].mean():.3f} (range: {df_new["SSIM"].min():.3f}-{df_new["SSIM"].max():.3f})')
print(f'   • pHash: {df_new["Hamming_Distance"].mean():.1f} (range: {df_new["Hamming_Distance"].min():.0f}-{df_new["Hamming_Distance"].max():.0f})')

# Detection methods
if 'Tier_Path' in df_new.columns:
    print(f'\n🎯 Detection Methods (Top 5):')
    top_methods = df_new['Tier_Path'].value_counts().head(5)
    for method, count in top_methods.items():
        if pd.notna(method):
            print(f'   • {method}: {count} pairs')

# Performance metrics
metadata_path = Path('doc54_results/RUN_METADATA.json')
if metadata_path.exists():
    with open(metadata_path) as f:
        meta = json.load(f)
    
    print(f'\n⏱️ PERFORMANCE METRICS:')
    print(f'   • Runtime: {meta["runtime_seconds"]:.1f} seconds ({meta["runtime_seconds"]/60:.1f} minutes)')
    print(f'   • Panels detected: {meta["results"]["panels"]}')
    print(f'   • Pages processed: {meta["results"]["pages"]}')

# Top high-confidence pairs
print(f'\n🔝 TOP 10 HIGH-CONFIDENCE PAIRS:')
print('-'*70)
top_pairs = df_new.nlargest(10, 'Cosine_Similarity')
for idx, row in top_pairs.iterrows():
    img_a = Path(row['Path_A']).name
    img_b = Path(row['Path_B']).name
    tier = row.get('Tier', 'N/A')
    clip = row['Cosine_Similarity']
    ssim = row['SSIM']
    
    # Check for evidence
    evidence_flags = []
    if row.get('Has_Patch_Evidence', False):
        evidence_flags.append('Patch')
    if row.get('Has_ORB_Evidence', False):
        evidence_flags.append('ORB')
    if row.get('Has_pHash_Evidence', False):
        evidence_flags.append('pHash')
    
    evidence_str = f" [{', '.join(evidence_flags)}]" if evidence_flags else ""
    
    print(f'{img_a} ↔ {img_b}')
    print(f'  Tier: {tier} | CLIP: {clip:.3f} | SSIM: {ssim:.3f}{evidence_str}')

# Tier A quality assessment
tier_a_pairs = df_new[df_new.get('Tier') == 'A']
if len(tier_a_pairs) > 0:
    print(f'\n✅ TIER A QUALITY ASSESSMENT:')
    print(f'   • Count: {len(tier_a_pairs)} pairs')
    print(f'   • Avg CLIP: {tier_a_pairs["Cosine_Similarity"].mean():.3f}')
    print(f'   • Avg SSIM: {tier_a_pairs["SSIM"].mean():.3f}')
    print(f'   • Min SSIM: {tier_a_pairs["SSIM"].min():.3f} (should be ≥0.60 typically)')
    
    # Check evidence distribution
    if 'Has_Patch_Evidence' in tier_a_pairs.columns:
        patch_count = tier_a_pairs['Has_Patch_Evidence'].sum()
        orb_count = tier_a_pairs['Has_ORB_Evidence'].sum()
        phash_count = tier_a_pairs['Has_pHash_Evidence'].sum()
        
        print(f'\n   📋 Evidence Distribution in Tier A:')
        print(f'   • High global SSIM (≥0.75): {len(tier_a_pairs[tier_a_pairs["SSIM"] >= 0.75])} pairs')
        print(f'   • Rescued by Patch SSIM: {patch_count} pairs')
        print(f'   • Rescued by ORB: {orb_count} pairs')
        print(f'   • Rescued by pHash: {phash_count} pairs')

print('\n' + '='*70)
print('✅ DOCUMENT 54 IMPROVEMENTS APPLIED SUCCESSFULLY!')
print('='*70)
print(f'\n📁 Full report: doc54_results/final_merged_report.tsv')
print(f'📁 Visualizations: doc54_results/duplicate_comparisons/')
print('='*70)

