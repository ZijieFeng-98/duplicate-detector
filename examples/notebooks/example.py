"""
Jupyter Notebook Example: Using Duplicate Detector

This script demonstrates how to use the duplicate detector programmatically.
Convert to .ipynb format for use in Jupyter.
"""

# Cell 1: Setup
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from duplicate_detector import DuplicateDetector, DetectorConfig

print("Duplicate Detector - Jupyter Notebook Example")
print("=" * 60)

# Cell 2: Initialize Detector
config = DetectorConfig.from_preset("balanced")
config.pdf_path = Path("your_paper.pdf")  # Update with your PDF path
config.output_dir = Path("notebook_results")

detector = DuplicateDetector(config=config)
print(f"Initialized detector with preset: balanced")
print(f"PDF: {config.pdf_path}")
print(f"Output: {config.output_dir}")

# Cell 3: Run Analysis
print("\nRunning analysis...")
results = detector.analyze_pdf()

print(f"\nResults:")
print(f"  Total duplicate pairs: {results.total_pairs}")
print(f"  Tier A (high confidence): {len(results.tier_a_pairs)}")
print(f"  Tier B (manual review): {len(results.tier_b_pairs)}")

# Cell 4: Explore Results
df_all = results.all_pairs

print("\nSample duplicate pairs:")
print(df_all.head(10))

# Cell 5: Visualize Statistics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Tier distribution
tier_counts = df_all['Tier'].value_counts()
axes[0].bar(tier_counts.index, tier_counts.values)
axes[0].set_title('Duplicate Pairs by Tier')
axes[0].set_xlabel('Tier')
axes[0].set_ylabel('Count')

# CLIP score distribution
axes[1].hist(df_all['CLIP_Score'].dropna(), bins=30, edgecolor='black')
axes[1].set_title('CLIP Score Distribution')
axes[1].set_xlabel('CLIP Score')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Cell 6: Filter High-Confidence Duplicates
tier_a_df = df_all[df_all['Tier'] == 'A']
print(f"\nTier A duplicates (high confidence): {len(tier_a_df)}")

if len(tier_a_df) > 0:
    print("\nTop Tier A duplicates:")
    print(tier_a_df[['Image_A', 'Image_B', 'CLIP_Score', 'SSIM_Score']].head())

# Cell 7: Custom Configuration
print("\n" + "=" * 60)
print("Custom Configuration Example")
print("=" * 60)

custom_config = DetectorConfig(
    pdf_path=Path("your_paper.pdf"),
    output_dir=Path("custom_results"),
    dpi=150,
    duplicate_detection=DetectorConfig.DuplicateDetectionConfig(
        sim_threshold=0.95,
        phash_max_dist=4,
        ssim_threshold=0.85
    ),
    feature_flags=DetectorConfig.FeatureFlags(
        use_phash_bundles=True,
        use_orb_ransac=True,
        use_tier_gating=True
    )
)

detector_custom = DuplicateDetector(config=custom_config)
results_custom = detector_custom.analyze_pdf()

print(f"Custom config results: {results_custom.total_pairs} pairs")

# Cell 8: Export Results
output_file = Path("duplicate_results.tsv")
df_all.to_csv(output_file, sep='\t', index=False)
print(f"\nResults exported to: {output_file}")

# Cell 9: Performance Comparison
from duplicate_detector.utils.performance import PerformanceBenchmark

benchmark = PerformanceBenchmark(
    pdf_path=Path("your_paper.pdf"),
    output_dir=Path("benchmark_results")
)

comparison = benchmark.compare_presets(["fast", "balanced", "thorough"])

print("\nPerformance comparison complete!")

