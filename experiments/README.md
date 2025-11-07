# Experiments and Ablation Studies

## Overview

This document describes experimental setups for ablation studies and parameter sensitivity analysis.

## Ablation Studies

### Study 1: Component Contribution

**Question:** How much does each component contribute to accuracy?

**Setup:**
- Baseline: CLIP only
- + pHash bundles
- + SSIM validation
- + ORB-RANSAC
- + Tier gating

**Metrics:**
- Precision
- Recall
- F1 Score
- False Positive Rate

### Study 2: Threshold Sensitivity

**Question:** How sensitive is performance to threshold changes?

**Parameters:**
- `sim_threshold`: [0.90, 0.92, 0.94, 0.96, 0.98]
- `phash_max_dist`: [1, 2, 3, 4, 5]
- `ssim_threshold`: [0.80, 0.85, 0.90, 0.95]

### Study 3: Modality-Specific Performance

**Question:** Does performance vary by image modality?

**Modalities:**
- Western blot
- Confocal microscopy
- TEM
- Bright-field
- Gel electrophoresis

## Parameter Sensitivity Analysis

See `scripts/ablation_study.py` for example implementation.

## Cross-Domain Validation

- **Training:** 60% of dataset
- **Validation:** 20% of dataset
- **Test:** 20% of dataset

## Publication Figures

### Figure 1: Pipeline Architecture
- Flowchart of detection pipeline

### Figure 2: Ablation Study Results
- Bar chart comparing components

### Figure 3: Precision-Recall Curves
- PR curves for different configurations

### Figure 4: Threshold Sensitivity
- F1 vs threshold plots

### Figure 5: Example Detections
- True positives, false positives, false negatives

