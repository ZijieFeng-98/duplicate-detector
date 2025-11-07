# Method Description

## Overview

The duplicate detection pipeline uses a multi-stage filtering approach combining semantic, perceptual, structural, and geometric similarity measures to identify duplicate panels in scientific PDFs.

## Pipeline Architecture

```
PDF Input
    ↓
[Stage 0] PDF → Pages (PyMuPDF)
    ↓
[Stage 1] Panel Detection (OpenCV + NMS)
    ↓
[Stage 2] CLIP Semantic Filtering
    ↓
[Stage 3] SSIM Structural Validation
    ↓
[Stage 4] pHash-RT Exact Detection
    ↓
[Stage 5] ORB-RANSAC Partial Duplicates
    ↓
[Stage 6] Merge & Tier Classification
    ↓
Results (Tier A/B)
```

## Algorithms

### 1. Panel Detection

**Method:** Computer vision-based panel extraction

**Algorithm:**
1. Convert PDF pages to PNG images at specified DPI
2. Apply Canny edge detection
3. Morphological operations (closing)
4. Contour detection
5. Filter by area and aspect ratio
6. Non-Maximum Suppression (NMS) with IoU threshold 0.5

**Parameters:**
- `MIN_PANEL_AREA`: Minimum panel area (default: 80,000 pixels)
- `MAX_PANEL_AREA`: Maximum panel area (default: 10,000,000 pixels)
- `MIN_ASPECT_RATIO`: Minimum width/height ratio (default: 0.2)
- `MAX_ASPECT_RATIO`: Maximum width/height ratio (default: 5.0)

### 2. CLIP Semantic Similarity

**Method:** Vision Transformer (ViT-B/32) embeddings

**Algorithm:**
1. Load CLIP model (OpenAI ViT-B/32)
2. Generate normalized embeddings for all panels
3. Compute cosine similarity matrix
4. Filter pairs above threshold

**Similarity Metric:**
```
similarity(A, B) = cosine(embed(A), embed(B))
                 = (embed(A) · embed(B)) / (||embed(A)|| × ||embed(B)||)
```

**Threshold:** Default 0.96 (configurable)

### 3. pHash-RT (Rotation-robust Perceptual Hash)

**Method:** Perceptual hashing with rotation/mirror bundles

**Algorithm:**
1. Compute pHash for 8 transform variants:
   - Original (0°)
   - Rotations: 90°, 180°, 270°
   - Mirror + rotations: mirror_h_0°, mirror_h_90°, mirror_h_180°, mirror_h_270°
2. For each pair, find minimum Hamming distance across all transform combinations
3. Filter pairs with distance ≤ threshold

**Hamming Distance:**
```
distance(A, B) = min_{transforms} Hamming(hash_A_transform, hash_B_transform)
```

**Threshold:** Default 3 (configurable, range 0-64)

### 4. SSIM (Structural Similarity Index)

**Method:** Multi-scale SSIM with patch-wise refinement

**Algorithm:**
1. Photometric normalization (CLAHE + z-score)
2. Resize to common height (512px)
3. Compute global SSIM
4. Compute patch-wise SSIM (grid-based)
5. Mix global and top-K patch SSIMs

**SSIM Formula:**
```
SSIM(x, y) = [l(x, y)]^α × [c(x, y)]^β × [s(x, y)]^γ

where:
l(x, y) = (2μ_x μ_y + C1) / (μ_x² + μ_y² + C1)  # luminance
c(x, y) = (2σ_x σ_y + C2) / (σ_x² + σ_y² + C2)  # contrast
s(x, y) = (σ_xy + C3) / (σ_x σ_y + C3)          # structure
```

**Patch-wise Refinement:**
- Grid: 3×3 patches (configurable)
- Top-K: Average of top 4 patches (configurable)
- Mix: 60% patch + 40% global (configurable)

**Threshold:** Default 0.90 (configurable)

### 5. ORB-RANSAC Geometric Verification

**Method:** Feature-based geometric matching

**Algorithm:**
1. Extract ORB keypoints and descriptors
2. Match features using BFMatcher + Lowe's ratio test
3. Estimate homography using RANSAC
4. Verify geometric consistency:
   - Minimum inliers (default: 30)
   - Inlier ratio (default: 0.30)
   - Reprojection error (default: ≤4.0 pixels)
   - Crop coverage (default: ≥0.50)

**Homography Estimation:**
```
H = argmin_H Σ_i ρ(||x'_i - H x_i||²)

where:
- x_i, x'_i are matched keypoints
- ρ is robust loss (RANSAC)
- H is 3×3 homography matrix
```

**Degeneracy Detection:**
- Check determinant of 2×2 submatrix
- Reject if |det(H[:2, :2])| < 0.1 or > 10.0

### 6. Tier Classification

**Method:** Multi-path discrimination with modality awareness

**Tier A Paths (High Confidence):**

1. **Exact Match:** pHash distance ≤ 3
2. **Strict:** CLIP ≥ 0.99 AND SSIM ≥ 0.95
3. **ORB-RANSAC:** Geometric verification passed
4. **Relaxed:** CLIP ≥ 0.94 AND SSIM ≥ 0.70 AND (CLIP + SSIM) ≥ 1.64
5. **Western Blot:** CLIP ≥ 0.95 AND SSIM ≥ 0.60 AND (CLIP + SSIM) ≥ 1.55

**Tier B Paths (Manual Review):**

- Borderline pHash: 4 ≤ distance ≤ 5
- Borderline CLIP+SSIM: 0.985 ≤ CLIP ≤ 0.99 AND 0.92 ≤ SSIM ≤ 0.95

**False Positive Filters:**

- **Confocal FP Filter:** High CLIP (≥0.96) + Low SSIM (<0.60) + No geometric evidence
- **CLIP Z-Score:** Self-normalized outlier detection (optional)
- **Patch SSIM Gate:** Minimum patch SSIM requirement (optional)

## Complexity Analysis

### Time Complexity

- **Panel Detection:** O(P × W × H) where P = pages, W×H = image dimensions
- **CLIP Embeddings:** O(N × B) where N = panels, B = batch size
- **CLIP Similarity:** O(N²) for full matrix (optimized with batching for N > 1000)
- **pHash:** O(N × T) where T = transforms (8), with bucketing O(B²) where B = bucket size
- **SSIM:** O(N² × W × H) for all pairs
- **ORB-RANSAC:** O(K × M) where K = triggered pairs, M = keypoints per image

**Overall:** O(N²) for similarity computation, O(N) for feature extraction

### Space Complexity

- **Embeddings:** O(N × D) where D = embedding dimension (512)
- **Similarity Matrix:** O(N²) (can be optimized with batching)
- **Cache:** O(N × D) for embeddings, O(N × T) for pHash bundles

**Overall:** O(N²) worst case, O(N) with optimizations

## Performance Optimizations

1. **Caching:** CLIP embeddings and pHash bundles cached to disk
2. **Memory-Mapped Loading:** 300x faster cache loading (0.1s vs 30s)
3. **Bucketing:** pHash uses prefix bucketing to reduce comparisons
4. **Batching:** CLIP processes in batches (default: 32)
5. **Subset Processing:** ORB-RANSAC only runs on triggered pairs

## Accuracy Metrics

### Validation Approach

- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **False Positive Rate:** FP / (FP + TN)

### Target Performance

- **F1 Score:** > 0.90 on validation set
- **False Positive Rate:** < 0.5% on hard negatives
- **Tier A Precision:** > 0.95 (high confidence)

## References

1. Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." CLIP paper.
2. Wang et al. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity." SSIM paper.
3. Rublee et al. (2011). "ORB: An efficient alternative to SIFT or SURF." ORB paper.
4. Zauner (2010). "Implementation and Benchmarking of Perceptual Image Hash Functions." pHash paper.

