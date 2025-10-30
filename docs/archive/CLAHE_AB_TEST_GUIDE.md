# 🧪 CLAHE A/B Testing Guide

**Date:** 2025-01-18  
**Purpose:** Test if CLAHE (Contrast Limited Adaptive Histogram Equalization) improves tile detection performance  
**Status:** ✅ **MODULE READY FOR TESTING**

---

## 🎯 **What is CLAHE?**

CLAHE (Contrast Limited Adaptive Histogram Equalization) is an image processing technique that:
- **Adaptive:** Works on small regions (tiles) of the image independently
- **Contrast-Limited:** Prevents over-amplification of noise
- **Robust:** Handles brightness/contrast variations better than simple z-score normalization

### **Why Test This?**

Microscopy images (especially confocal) often have:
- ✅ Varying illumination across the image
- ✅ Different exposure settings between duplicates
- ✅ Local brightness variations
- ❌ Z-score normalization alone may not handle these well

**CLAHE can potentially improve:**
- Detection of brightness-shifted duplicates
- Robustness to lighting variations
- SSIM/NCC scores for true positives

---

## 📊 **Synthetic Test Results**

**Test Setup:** 5 synthetic duplicate pairs with 50% brightness increase

| Metric | Z-Score Only | CLAHE + Z-Score | Improvement |
|--------|--------------|-----------------|-------------|
| **SSIM** | 0.8020 ± 0.003 | 0.8326 ± 0.002 | **+3.82%** ✅ |
| **NCC** | 0.9781 ± 0.000 | 0.9956 ± 0.000 | **+1.74%** ✅ |

**Recommendation:** ✅ CLAHE shows meaningful improvement on brightness variations

---

## 🚀 **Quick Start**

### **1. Run Synthetic Test**

```bash
# Activate virtual environment
source venv/bin/activate

# Run synthetic test (generates test images automatically)
python tools/test_clahe_ab.py --synthetic

# Results saved to: ./clahe_test_results/
```

### **2. Test on Real Images**

Create a test directory structure:

```
my_test_images/
    duplicates/
        pair1_a.png
        pair1_b.png
        pair2_a.png
        pair2_b.png
    non_duplicates/
        diff1_a.png
        diff1_b.png
```

Then run:

```bash
python tools/test_clahe_ab.py --test-dir ./my_test_images --output ./my_results
```

---

## 📝 **Understanding the Results**

### **Output Files**

1. **`*_stats.json`** - Summary statistics
   ```json
   {
     "zscore_only": {
       "ssim_mean": 0.8020,
       "ssim_std": 0.0029,
       "ncc_mean": 0.9781
     },
     "clahe_zscore": {
       "ssim_mean": 0.8326,
       "ssim_std": 0.0015,
       "ncc_mean": 0.9956
     },
     "improvement": {
       "ssim_delta": 0.0306,
       "ncc_delta": 0.0174,
       "ssim_pct_change": 3.82
     }
   }
   ```

2. **`*_full.json`** - Full results for each test pair
   - Individual SSIM/NCC scores
   - Per-pair deltas
   - Ground truth labels

### **Interpreting Results**

| SSIM Δ | Recommendation |
|--------|----------------|
| **> 0.01** | ✅ Adopt CLAHE - meaningful improvement |
| **0.001 - 0.01** | ⚠️ Test on more pairs - marginal improvement |
| **< 0** | ❌ Stick with z-score only - no improvement |

### **What to Look For**

✅ **Good Signs:**
- Positive SSIM Δ and NCC Δ
- Lower std deviation (more consistent)
- Improvement on duplicate pairs without hurting non-duplicates

❌ **Bad Signs:**
- Negative deltas (performance degradation)
- Higher std deviation (less stable)
- False positives increase

---

## 🔧 **Integration into tile_detection.py**

### **Option 1: Feature Flag (Recommended)**

Add CLAHE as an optional feature that can be toggled:

```python
# At top of tile_detection.py
from clahe_ab_test import PhotometricNormalizer, CLAHEConfig

# Configuration
ENABLE_CLAHE = False  # Set to True after testing

# Initialize normalizer (once, at module level)
NORMALIZER = PhotometricNormalizer(CLAHEConfig(
    clip_limit=2.5,
    tile_size=8,
    apply_zscore_after=True
))

# Replace _zscore() function
def _normalize_photometric(img: np.ndarray, use_clahe: bool = ENABLE_CLAHE) -> np.ndarray:
    """
    Apply photometric normalization
    
    Args:
        img: Input image
        use_clahe: If True, apply CLAHE+zscore; if False, zscore only
    
    Returns:
        Normalized image
    """
    if use_clahe:
        return NORMALIZER.normalize_clahe_zscore(img)
    else:
        return NORMALIZER.normalize_zscore_only(img)

# In verify_tile_pair_confocal(), replace:
#   gray_a = _zscore(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY))
# with:
#   gray_a = _normalize_photometric(img_a)
```

### **Option 2: Direct Replacement**

If testing shows consistent improvement, directly replace `_zscore()`:

```python
def _zscore(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE + z-score normalization"""
    from clahe_ab_test import PhotometricNormalizer
    normalizer = PhotometricNormalizer()
    return normalizer.normalize_clahe_zscore(img)
```

---

## ⚙️ **Configuration Parameters**

### **CLAHEConfig Options**

```python
@dataclass
class CLAHEConfig:
    clip_limit: float = 2.5          # Contrast limit (1.0-4.0 typical)
    tile_size: int = 8               # CLAHE tile size (8×8 pixels)
    apply_zscore_after: bool = True  # Apply z-score after CLAHE
```

### **Tuning Guidelines**

| Parameter | Low Value | High Value | Effect |
|-----------|-----------|------------|--------|
| **clip_limit** | 1.0 | 4.0 | Higher = more contrast enhancement (but more noise) |
| **tile_size** | 4 | 16 | Smaller = more local adaptation (but more noise) |

**Recommended starting point:** `clip_limit=2.5, tile_size=8`

---

## 🎓 **Advanced Usage**

### **Custom Test Harness**

```python
from pathlib import Path
from clahe_ab_test import ABTestHarness

# Create harness
harness = ABTestHarness(output_dir=Path("./my_results"))

# Define test pairs (path_a, path_b, ground_truth)
test_pairs = [
    ("tile_a1.png", "tile_b1.png", "duplicate"),
    ("tile_a2.png", "tile_b2.png", "duplicate"),
    ("tile_a3.png", "tile_b3.png", "not_duplicate"),
]

# Run A/B test
stats = harness.run_ab_test(test_pairs, output_name="my_ab_test")

# Check recommendation
if stats['improvement']['ssim_delta'] > 0.01:
    print("✅ Adopt CLAHE!")
else:
    print("❌ Stick with z-score only")
```

### **Process Single Pair**

```python
from clahe_ab_test import ABTestHarness

harness = ABTestHarness(Path("./results"))
result = harness.process_tile_pair(
    "image_a.png",
    "image_b.png",
    "duplicate"
)

print(f"Z-score SSIM: {result['zscore_only']['ssim']:.4f}")
print(f"CLAHE SSIM:   {result['clahe_zscore']['ssim']:.4f}")
print(f"Delta:        {result['delta_ssim']:+.4f}")
```

---

## 📊 **Performance Impact**

### **Computational Cost**

| Method | Relative Speed | Memory |
|--------|----------------|--------|
| **Z-score only** | 1.0x (baseline) | Low |
| **CLAHE + Z-score** | ~1.1-1.2x slower | Low |

**Overhead:** CLAHE adds ~10-20% computational cost, which is acceptable for the potential accuracy gains.

### **When to Use CLAHE**

✅ **Use CLAHE if:**
- Brightness/contrast varies between duplicates
- Working with microscopy images (confocal, IHC)
- Z-score alone gives inconsistent results
- A/B test shows >1% SSIM improvement

❌ **Skip CLAHE if:**
- Images have consistent illumination
- A/B test shows no improvement
- Speed is critical and images are already well-normalized

---

## 🔍 **Troubleshooting**

### **Issue: No improvement shown**

**Possible causes:**
- Test images already well-normalized
- Brightness variations are not the main issue
- Too few test pairs

**Solution:**
- Test on more pairs with known brightness variations
- Try different `clip_limit` values (1.0-4.0)
- Check if issue is rotation/cropping instead of brightness

### **Issue: Worse performance with CLAHE**

**Possible causes:**
- Clip limit too high (over-amplifying noise)
- Test images have minimal brightness variation
- CLAHE tile size mismatch with image structure

**Solution:**
- Reduce `clip_limit` (try 1.5-2.0)
- Increase `tile_size` (try 16)
- Stick with z-score only for these images

### **Issue: ModuleNotFoundError**

```bash
# Make sure you're in the venv
source venv/bin/activate

# Check if opencv-python is installed
pip list | grep opencv

# If missing, install it
pip install opencv-python scikit-image
```

---

## 📚 **References**

- **CLAHE Algorithm:** Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
- **OpenCV Documentation:** https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
- **scikit-image SSIM:** https://scikit-image.org/docs/stable/api/skimage.metrics.html

---

## 🎯 **Recommended Workflow**

1. **Baseline Test** ✅ DONE
   - Ran synthetic test: +3.82% SSIM improvement

2. **Real Image Test** ⏳ TODO
   - Collect 10-20 known duplicate pairs from your PDFs
   - Include pairs with brightness variations
   - Run: `python tools/test_clahe_ab.py --test-dir ./real_test_images`

3. **Decision Point**
   - If SSIM Δ > 1%: Integrate CLAHE with feature flag
   - If SSIM Δ < 0%: Skip CLAHE, stick with z-score

4. **Production Integration**
   - Add `ENABLE_CLAHE` flag to `tile_detection.py`
   - Test on full pipeline
   - Monitor false positive rate
   - If stable, make it default

5. **Validation**
   - Run comprehensive test on known dataset
   - Compare F1 score, precision, recall
   - Ensure no regression on non-brightness cases

---

## ✅ **Quick Commands Reference**

```bash
# Run synthetic test
python tools/test_clahe_ab.py --synthetic

# Run on real images
python tools/test_clahe_ab.py --test-dir ./my_images --output ./results

# View results
cat ./results/synthetic_ab_test_stats.json | python -m json.tool

# Test specific Python version
python3.12 tools/test_clahe_ab.py --synthetic
```

---

**Status:** ✅ Module implemented and tested  
**Next Step:** Test on real microscopy images from your PDFs  
**Decision:** Integrate if real-world test shows >1% SSIM improvement

