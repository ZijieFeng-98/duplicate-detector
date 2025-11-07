# Local Testing - Complete Guide

## ✅ What's Ready

**Duplicates Created Successfully!**
- ✓ 18 duplicate variants (WB, confocal, IHC)
- ✓ 23 total test images
- ✓ Located in: `test_duplicate_detection/`

## 🚀 How to Run Detection Locally

### Method 1: Streamlit Web Interface (Recommended - Easiest)

```bash
# 1. Install streamlit if needed
pip3 install streamlit

# 2. Run the web app
streamlit run streamlit_app.py

# 3. Open browser to http://localhost:8501
# 4. Upload your PDF and run detection
```

**Advantages:**
- No command-line needed
- Visual interface
- Easy to use
- See results immediately

### Method 2: Command Line (Full Pipeline)

**Step 1: Install Dependencies**

```bash
# Install all required packages
pip3 install -r requirements.txt

# Or install individually:
pip3 install opencv-python-headless pillow pandas numpy imagehash scikit-image scikit-learn tqdm scipy pymupdf torch torchvision open-clip-torch streamlit
```

**Step 2: Run Detection**

```bash
python3 ai_pdf_panel_duplicate_check_AUTO.py \
  --pdf "/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf" \
  --output test_duplicate_detection/detection_results \
  --preset balanced \
  --sim-threshold 0.94 \
  --phash-max-dist 5 \
  --use-phash-bundles \
  --use-orb \
  --use-tier-gating
```

**Step 3: Check Results**

```bash
# View results
cat test_duplicate_detection/detection_results/final_merged_report.tsv

# Or open in Excel/Numbers
open test_duplicate_detection/detection_results/final_merged_report.tsv
```

### Method 3: Automated Script

```bash
# Run the automated test script
python3 tests/integration/run_detection_local.py
```

## 📊 What to Expect

After running detection, you should see:

1. **Exact Duplicates** ✓
   - pHash distance: 0
   - CLIP similarity: >0.99
   - Files with `exact` in name

2. **Rotated Duplicates** ✓
   - Detected via pHash bundles
   - Files with `rotated` in name

3. **Partial Duplicates** ✓
   - Detected via ORB-RANSAC
   - Files with `partial` in name

4. **Panel Types** ✓
   - WB panels (files with `WB` prefix)
   - Confocal panels (files with `confocal` prefix)
   - IHC panels (files with `IHC` prefix)

## 📁 File Locations

```
test_duplicate_detection/
├── pages/                          # Extracted PDF pages
├── intentional_duplicates/         # Created duplicates
│   ├── WB/                         # 6 variants
│   ├── confocal/                   # 6 variants
│   └── IHC/                        # 6 variants
├── test_panels/                    # Combined test set (23 images)
└── detection_results/              # Detection output (after running)
    ├── final_merged_report.tsv     # Main results
    ├── panel_manifest.tsv          # All panels
    └── duplicate_comparisons/      # Visual comparisons
```

## 🔧 Troubleshooting

### Issue: Missing Dependencies

**Solution:**
```bash
pip3 install <missing_package>
```

Common packages:
- `cv2` → `opencv-python-headless`
- `fitz` → `pymupdf`
- `PIL` → `pillow`
- `torch` → `torch torchvision`

### Issue: Detection Not Finding Duplicates

**Solution:** Lower thresholds
```bash
--sim-threshold 0.90    # Lower CLIP threshold
--phash-max-dist 6      # Higher pHash distance
```

### Issue: Too Slow

**Solution:** Use fast preset
```bash
--preset fast
```

### Issue: Too Many False Positives

**Solution:** Use thorough preset
```bash
--preset thorough
```

## 🎯 Quick Test Commands

```bash
# Test basic pHash (no dependencies needed)
python3 tests/integration/test_simple.py

# Run full detection
python3 tests/integration/run_detection_local.py

# Use Streamlit (easiest)
streamlit run streamlit_app.py
```

## 📝 Next Steps

1. **Install dependencies** (if not already installed)
2. **Choose a method** (Streamlit recommended)
3. **Run detection**
4. **Check results** in TSV file
5. **Verify** all duplicate types detected

## 💡 Tips

- **Streamlit is easiest** - No command-line needed
- **Use balanced preset** - Good balance of speed/accuracy
- **Check TSV file** - All results are there
- **Visual comparisons** - See side-by-side in `duplicate_comparisons/`

---

**Ready to test! Choose your preferred method above.** 🚀

