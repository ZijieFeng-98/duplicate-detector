# 🧪 Validation Experiment Guide

**Date:** 2025-01-18  
**Purpose:** Systematically test detection pipeline with known duplicates and non-duplicates  
**Status:** ✅ **FRAMEWORK COMPLETE & TESTED**

---

## 🎯 **What is Validation Testing?**

Validation testing measures your detection pipeline's performance using a **ground truth dataset** where you know exactly which pairs should be detected (true positives) and which shouldn't (true negatives).

### **Key Metrics**

| Metric | Formula | Target | Description |
|--------|---------|--------|-------------|
| **Precision** | TP / (TP + FP) | ≥ 0.95 | Of detected pairs, how many are truly duplicates? |
| **Recall** | TP / (TP + FN) | ≥ 0.90 | Of true duplicates, how many did we detect? |
| **F1 Score** | 2 × (P × R) / (P + R) | ≥ 0.92 | Harmonic mean of precision and recall |
| **FPR** | FP / (FP + TN) | ≤ 0.005 | **Critical**: False alarm rate (should be ≤ 0.5%) |

---

## 📊 **Synthetic Test Results**

**Test Setup:** 22 validation pairs (12 true positives, 10 hard negatives)

```
🎯 Overall Performance:
  Precision: 1.0000 (3/3)
  Recall:    0.2500 (3/12)
  F1 Score:  0.4000
  Accuracy:  0.5909

🚨 False Positive Rate: 0.0000 (0 false alarms)

💡 Assessment:
  ✅ FPR ≤ 0.5% - Excellent! Meets target threshold
  ❌ Recall < 85% - Consider looser thresholds
```

**Interpretation:**
- ✅ **Zero false positives** - Won't flag different images as duplicates
- ⚠️ **Low recall (25%)** - Simple pHash+SSIM detector is too strict
- 💡 **Your full pipeline** (CLIP + pHash + ORB + SSIM + Deep Verify) will have much higher recall

---

## 🚀 **Quick Start**

### **Option 1: Synthetic Test (Demo)**

```bash
# Activate environment
source venv/bin/activate

# Run synthetic test
python tools/test_validation_synthetic.py

# View results
cat validation_synthetic_test/validation_results/metrics_summary.json | python -m json.tool
```

### **Option 2: Real Panels Test**

```bash
# Step 1: Extract panels from a PDF
python ai_pdf_panel_duplicate_check_AUTO.py \
  --pdf your_test_file.pdf \
  --output ./test_output \
  --dpi 150

# Step 2: Build validation dataset
python tools/run_validation.py build \
  --panels-dir ./test_output/panels \
  --output ./validation_dataset \
  --num-negatives 20

# Step 3: Run validation
python tools/run_validation.py test \
  --dataset ./validation_dataset \
  --output ./validation_results
```

---

## 📂 **Dataset Structure**

A validation dataset contains three types of pairs:

### **1. True Positives (Transformed Duplicates)**

These are **guaranteed duplicates** created by applying known transformations:

| Transform | Description | Tests |
|-----------|-------------|-------|
| `rotate_90` | 90° clockwise rotation | Rotation robustness |
| `rotate_180` | 180° rotation | Rotation robustness |
| `rotate_270` | 270° rotation | Rotation robustness |
| `mirror_h` | Horizontal flip | Mirror robustness |
| `mirror_v` | Vertical flip | Mirror robustness |
| `brightness_+20` | Increase brightness by 20 | Brightness robustness |
| `brightness_-15` | Decrease brightness by 15 | Brightness robustness |
| `crop_15pct` | Crop 15% from edges | Cropping robustness |
| `blur_slight` | Slight Gaussian blur | Noise robustness |
| `jpeg_compress` | JPEG compression (Q=80) | Compression robustness |

**Expected:** Pipeline should detect all transformed duplicates (Recall ≥ 90%)

### **2. Hard Negatives**

These are **different images** that are intentionally challenging:
- Same modality (e.g., both confocal microscopy)
- Similar visual appearance (e.g., similar staining patterns)
- Same size and structure

**Expected:** Pipeline should NOT flag these as duplicates (FPR ≤ 0.5%)

### **3. Known Duplicates (Optional)**

Real duplicate pairs from your PDFs where you've manually verified they are duplicates.

**Expected:** Pipeline should detect these (adds to Recall score)

---

## 🔧 **Building Your Validation Dataset**

### **Method 1: From Existing PDF**

```python
from pathlib import Path
from validation_experiment import ValidationDatasetBuilder

# Initialize builder
builder = ValidationDatasetBuilder(output_dir=Path('my_validation_dataset'))

# Add transformed duplicates from 5 sample panels
sample_panels = [
    'panels/page_01_panel_003.png',
    'panels/page_02_panel_005.png',
    'panels/page_03_panel_008.png',
    'panels/page_04_panel_012.png',
    'panels/page_05_panel_015.png',
]

for panel in sample_panels:
    builder.add_transformed_duplicate(
        source_path=panel,
        transforms=['rotate_90', 'mirror_h', 'brightness_+20', 'crop_15pct', 'rotate_180']
    )

# Add hard negatives (different panels, same modality)
hard_negative_pairs = [
    ('panels/page_01_panel_003.png', 'panels/page_02_panel_004.png'),
    ('panels/page_03_panel_007.png', 'panels/page_04_panel_011.png'),
    # ... add more pairs ...
]

for panel_a, panel_b in hard_negative_pairs:
    builder.add_hard_negative_pair(panel_a, panel_b, modality='confocal')

# Save ground truth
builder.save_ground_truth()
```

### **Method 2: From Known Duplicates File**

If you have a file listing known duplicates:

```python
import pandas as pd

# Load your known duplicates
df = pd.read_csv('known_duplicates.csv')

# Add to validation dataset
for _, row in df.iterrows():
    builder.add_known_duplicate_pair(
        panel_a_path=row['panel_a'],
        panel_b_path=row['panel_b'],
        label='known_duplicate'
    )
```

---

## 🔬 **Running Validation**

### **With Simple Detector (Built-in)**

The framework includes a simple pHash + SSIM detector for quick testing:

```bash
python tools/run_validation.py test \
  --dataset ./validation_dataset \
  --output ./validation_results
```

### **With Your Full Pipeline**

Create a custom detection function:

```python
from validation_experiment import ValidationRunner
from pathlib import Path

def my_full_pipeline_detector(path_a: str, path_b: str) -> dict:
    """
    Wrapper around your full detection pipeline
    
    Returns:
        dict with keys: detected, clip_score, ssim_score, phash_distance, method
    """
    # TODO: Call your actual pipeline here
    # Example using subprocess:
    import subprocess
    import tempfile
    import json
    
    # Run your pipeline on this pair
    result = subprocess.run([
        'python', 'ai_pdf_panel_duplicate_check_AUTO.py',
        '--compare-pair', path_a, path_b,
        '--output-json'
    ], capture_output=True)
    
    # Parse result
    output = json.loads(result.stdout)
    
    return {
        'detected': output['is_duplicate'],
        'clip_score': output.get('clip_score'),
        'ssim_score': output.get('ssim_score'),
        'phash_distance': output.get('phash_distance'),
        'method': output.get('method_triggered')
    }

# Run validation with your detector
runner = ValidationRunner(
    ground_truth_manifest=Path('validation_dataset/ground_truth_manifest.json')
)

metrics = runner.run_validation(
    detection_function=my_full_pipeline_detector,
    output_path=Path('validation_results_full_pipeline')
)

print(f"F1 Score: {metrics['overall']['f1_score']:.4f}")
print(f"FPR:      {metrics['overall']['false_positive_rate']:.4f}")
```

---

## 📈 **Interpreting Results**

### **Output Files**

After running validation, you'll get:

```
validation_results/
├── validation_results.json     # Full results with all pairs
├── validation_results.csv      # CSV format for spreadsheet analysis
└── metrics_summary.json        # Summary metrics only
```

### **Example metrics_summary.json**

```json
{
  "overall": {
    "true_positives": 48,
    "false_positives": 1,
    "false_negatives": 2,
    "true_negatives": 19,
    "precision": 0.9796,
    "recall": 0.9600,
    "f1_score": 0.9697,
    "false_positive_rate": 0.0500,
    "accuracy": 0.9571
  },
  "by_category": {
    "transformed_duplicate": {
      "count": 50,
      "detected": 48,
      "recall": 0.9600
    },
    "hard_negative": {
      "count": 20,
      "detected": 1,
      "recall": 0.0000
    }
  }
}
```

### **What Each Metric Tells You**

✅ **Precision = 0.98** (98%)
- Of 49 pairs flagged as duplicates, 48 actually were duplicates
- Only 1 false alarm
- **Good**: Low false positive rate

✅ **Recall = 0.96** (96%)
- Of 50 true duplicates, detected 48
- Missed 2 duplicates
- **Good**: Catching most duplicates

✅ **F1 Score = 0.97**
- Excellent balance between precision and recall
- **Target**: ≥ 0.92

⚠️ **FPR = 0.05** (5%)
- 1 false alarm out of 20 hard negatives
- **Target**: ≤ 0.5% (0.005)
- **Action**: This is too high! Tighten thresholds to reduce false positives

---

## 🎯 **Tuning Based on Results**

### **Scenario 1: High FPR (Too Many False Positives)**

**Problem:** FPR > 1% (e.g., FPR = 0.05 in example above)

**Solution:** Tighten thresholds

```bash
# Increase CLIP threshold
--sim-threshold 0.98  # Instead of 0.96

# Decrease pHash max distance
--phash-max-dist 3  # Instead of 4

# Increase SSIM threshold
--ssim-threshold 0.92  # Instead of 0.90

# Enable stricter tier gating
--use-tier-gating
```

### **Scenario 2: Low Recall (Missing Duplicates)**

**Problem:** Recall < 85% (e.g., Recall = 0.70)

**Solution:** Loosen thresholds or enable more detection paths

```bash
# Decrease CLIP threshold
--sim-threshold 0.94  # Instead of 0.96

# Enable ORB for partial matches
--use-orb

# Enable pHash bundles for rotations
--use-phash-bundles

# Widen pHash distance
--phash-max-dist 5  # Instead of 4

# Enable tile-first mode for confocal
--tile-first-auto
```

### **Scenario 3: Missing Specific Transforms**

Check the validation_results.csv to see which transforms failed:

```bash
# View results by transform type
cat validation_results.csv | grep "transform_" | awk -F',' '{print $3, $5}'
```

Example output:
```
transform_rotate_90 True   ← Detected ✅
transform_mirror_h True    ← Detected ✅
transform_brightness_+20 False  ← Missed ❌
transform_crop_15pct False ← Missed ❌
```

**Action:**
- If missing brightness transforms: Enable CLAHE normalization
- If missing crop transforms: Check ORB-RANSAC is enabled
- If missing rotation transforms: Check pHash bundles are enabled

---

## 🔄 **Iterative Tuning Workflow**

1. **Baseline Run**
   ```bash
   python tools/run_validation.py test --dataset ./dataset --output ./run1_baseline
   ```
   Result: F1=0.85, FPR=0.10 (too high)

2. **Tighten Thresholds**
   Edit detection function to use stricter thresholds
   ```bash
   python tools/run_validation.py test --dataset ./dataset --output ./run2_strict
   ```
   Result: F1=0.88, FPR=0.02 (better, but FPR still high)

3. **Enable Confocal FP Filter**
   Add confocal false-positive filtering
   ```bash
   python tools/run_validation.py test --dataset ./dataset --output ./run3_confocal_fp
   ```
   Result: F1=0.92, FPR=0.004 ✅ (meets target!)

4. **Validate on Real Data**
   Test on actual PDF with known duplicates

---

## 🔬 **Comparing Detection Methods**

### **A/B Test: pHash vs pHash+CLIP**

```python
# Method A: pHash only
def detector_phash_only(path_a, path_b):
    phash_dist = compute_phash(path_a, path_b)
    return {'detected': phash_dist <= 4}

# Method B: pHash + CLIP
def detector_phash_clip(path_a, path_b):
    phash_dist = compute_phash(path_a, path_b)
    clip_score = compute_clip(path_a, path_b)
    return {'detected': (phash_dist <= 4) or (clip_score >= 0.96)}

# Run both
runner = ValidationRunner(manifest)
metrics_a = runner.run_validation(detector_phash_only, Path('results_phash'))
metrics_b = runner.run_validation(detector_phash_clip, Path('results_phash_clip'))

# Compare
print(f"pHash only:      F1={metrics_a['overall']['f1_score']:.4f}")
print(f"pHash + CLIP:    F1={metrics_b['overall']['f1_score']:.4f}")
```

---

## 📝 **Best Practices**

### ✅ **DO**

- Include at least 50 validation pairs (30 true positives, 20 hard negatives)
- Use diverse transforms (rotation, brightness, crop, etc.)
- Include hard negatives from the same modality
- Run validation after every threshold change
- Track FPR as your primary metric (aim for ≤ 0.5%)
- Save all validation runs for comparison

### ❌ **DON'T**

- Use the same images for training/tuning and validation
- Cherry-pick "easy" pairs for validation
- Ignore false positives (they're critical!)
- Only test one modality
- Skip hard negatives (they expose false positives)

---

## 🎓 **Advanced: Multi-Configuration Sweep**

Test multiple configurations automatically:

```python
configs = [
    {'sim_threshold': 0.94, 'phash_max': 4, 'name': 'loose'},
    {'sim_threshold': 0.96, 'phash_max': 4, 'name': 'balanced'},
    {'sim_threshold': 0.98, 'phash_max': 3, 'name': 'strict'},
]

results = []
for config in configs:
    # Update detector with config
    def detector(path_a, path_b):
        return detect_with_config(path_a, path_b, config)
    
    # Run validation
    metrics = runner.run_validation(detector, Path(f"results_{config['name']}"))
    
    results.append({
        'name': config['name'],
        'f1': metrics['overall']['f1_score'],
        'fpr': metrics['overall']['false_positive_rate'],
        'recall': metrics['overall']['recall']
    })

# Find best config (highest F1 with FPR ≤ 0.005)
best = max([r for r in results if r['fpr'] <= 0.005], key=lambda x: x['f1'])
print(f"Best config: {best['name']}")
print(f"  F1: {best['f1']:.4f}, FPR: {best['fpr']:.4f}, Recall: {best['recall']:.4f}")
```

---

## 📊 **Example: Full Workflow**

```bash
# 1. Generate validation dataset
python tools/test_validation_synthetic.py

# 2. Run baseline validation
python tools/run_validation.py test \
  --dataset validation_synthetic_test/validation_dataset \
  --output validation_results/baseline

# 3. Check results
cat validation_results/baseline/metrics_summary.json | python -m json.tool

# 4. If FPR too high, tighten thresholds and re-run
# (Edit detection function thresholds)

# 5. Compare runs
python -c "
import json
baseline = json.load(open('validation_results/baseline/metrics_summary.json'))
strict = json.load(open('validation_results/strict/metrics_summary.json'))
print(f'Baseline FPR: {baseline[\"overall\"][\"false_positive_rate\"]:.4f}')
print(f'Strict FPR:   {strict[\"overall\"][\"false_positive_rate\"]:.4f}')
"
```

---

## ✅ **Success Criteria**

Your detection pipeline is ready for production when:

✅ **F1 Score** ≥ 0.92  
✅ **Recall** ≥ 0.90 (catching 90%+ of duplicates)  
✅ **False Positive Rate** ≤ 0.005 (≤0.5% false alarms)  
✅ **Precision** ≥ 0.95 (95%+ of flagged pairs are truly duplicates)

---

**Status:** ✅ Framework implemented and tested  
**Next Step:** Run validation on your real PDF panels  
**Target:** F1 ≥ 0.92, FPR ≤ 0.5%

