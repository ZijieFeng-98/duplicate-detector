# Benchmark Dataset Creation Guide

## Overview

This guide describes how to create and annotate a benchmark dataset for evaluating duplicate detection performance.

## Dataset Structure

```
benchmarks/
├── dataset/
│   ├── images/              # Source images
│   ├── duplicates/          # Known duplicate pairs
│   ├── negatives/           # Hard negatives (similar but not duplicates)
│   └── annotations.json     # Ground truth annotations
├── scripts/
│   ├── create_duplicates.py # Script to create duplicates
│   └── evaluate.py          # Evaluation script
└── README.md                # Dataset documentation
```

## Creating Duplicate Pairs

### Types of Duplicates

1. **Exact Duplicates**
   - Same image, no modifications

2. **Rotated Duplicates**
   - Rotated 90°, 180°, 270°

3. **Mirrored Duplicates**
   - Horizontal/vertical flips

4. **Cropped Duplicates**
   - Partial crops (50-90% overlap)

5. **Compressed Duplicates**
   - JPEG compression artifacts

6. **Resized Duplicates**
   - Different resolutions

7. **Brightness/Contrast Adjusted**
   - Photometric variations

## Annotation Format

```json
{
  "image_a": "path/to/image_a.png",
  "image_b": "path/to/image_b.png",
  "type": "rotated|exact|cropped|mirrored|compressed|negative",
  "is_duplicate": true,
  "metadata": {
    "angle": 90,
    "overlap_ratio": 0.8
  }
}
```

## Evaluation Metrics

- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **False Positive Rate:** FP / (FP + TN)
- **AUC-ROC:** Area under ROC curve

## Dataset Requirements

### Minimum Dataset Size

- **50-100 duplicate pairs** (various types)
- **20-30 negative pairs** (hard negatives)
- **Multiple modalities:** Western blot, confocal, TEM, bright-field

## Publishing Dataset

### Zenodo/Figshare

1. Create dataset archive
2. Upload to Zenodo/Figshare
3. Get DOI
4. Cite in paper

See `scripts/create_duplicates.py` for example script to create duplicates.

