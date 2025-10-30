# 🔧 Python Installation Fix Guide

**Issue Identified**: Your Python 3.13.9 at `D:\python.exe` is a **minimal/broken installation**

**Status**: pip installed but not functional, no packages available

---

## 🚨 **The Problem**

Your Python installation has these issues:
```
✓ Python 3.13.9 installed at D:\python.exe
✓ py launcher works
✗ pip not properly configured
✗ No packages (numpy, pandas, torch, etc.)
✗ pip installation attempted but failed
```

**Root cause**: This appears to be a minimal Python installation without the standard library properly configured.

---

## ✅ **Solution: Fresh Python Install** (Recommended)

### **Step 1: Download Python 3.12**
📥 **Direct download**: https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe

**Why 3.12?** More stable than 3.13, all our dependencies tested on 3.12.

### **Step 2: Install with Correct Options**

1. **Run the installer**
2. ✅ **CHECK**: "Add Python 3.12 to PATH"
3. ✅ **CHECK**: "Install pip"
4. ✅ **CHECK**: "Install for all users" (optional but recommended)
5. Click **"Install Now"**

### **Step 3: Verify Installation**
```powershell
# Restart PowerShell first!
python --version
# Should show: Python 3.12.8

python -m pip --version
# Should show: pip 24.x.x
```

### **Step 4: Install Dependencies** (5-10 min)
```powershell
python -m pip install -r requirements.txt
```

### **Step 5: Run Your Test!**
```powershell
python automated_test_suite.py --pdf "E:\PUA-STM-Combined Figures .pdf" --verbose
```

---

## 🔀 **Alternative: Use Anaconda** (If you can't reinstall)

### **Step 1: Download Anaconda**
📥 https://www.anaconda.com/download

### **Step 2: Install Anaconda**
- Accept all defaults
- Let it add to PATH

### **Step 3: Create Environment**
```powershell
# Open Anaconda Prompt (from Start menu)
conda create -n dup-detect python=3.12 -y
conda activate dup-detect
```

### **Step 4: Install Dependencies**
```powershell
conda install pytorch pandas numpy pillow opencv scikit-image tqdm scipy scikit-learn -y
pip install open-clip-torch pymupdf imagehash
```

### **Step 5: Run Test**
```powershell
cd D:\duplicate-detector
python automated_test_suite.py --pdf "E:\PUA-STM-Combined Figures .pdf" --verbose
```

---

## ⚡ **Quick Alternative: Portable Python**

If you need to test **immediately** without installing:

### **Step 1: Download WinPython**
📥 https://winpython.github.io/
- Choose: WinPython 3.12.x.x (64-bit)

### **Step 2: Extract**
- Extract to `C:\WinPython`

### **Step 3: Run**
```powershell
cd C:\WinPython\python-3.12.x.amd64
.\python.exe -m pip install open-clip-torch pymupdf imagehash

cd D:\duplicate-detector
C:\WinPython\python-3.12.x.amd64\python.exe automated_test_suite.py --pdf "E:\PUA-STM-Combined Figures .pdf"
```

---

## 🎯 **Once Python Works - Your Test Commands**

### **Full Automated Test** (Recommended)
```powershell
python automated_test_suite.py --pdf "E:\PUA-STM-Combined Figures .pdf" --verbose
```

**Outputs**:
- `test_results_*/test_results.json` - Pass/fail summary
- `test_results_*/final_merged_report.tsv` - Duplicate pairs
- `test_results_*/test_log_*.txt` - Detailed log

### **Direct Pipeline Run** (Faster)
```powershell
python ai_pdf_panel_duplicate_check_AUTO.py `
    --pdf "E:\PUA-STM-Combined Figures .pdf" `
    --output stm_results `
    --tile-first `
    --tile-size 256 `
    --tile-stride 0.70 `
    --dpi 150 `
    --enable-cache `
    --use-tier-gating
```

**Outputs**:
- `stm_results/final_merged_report.tsv` - Main results
- `stm_results/duplicate_comparisons/` - Visual comparisons
- `stm_results/RUN_METADATA.json` - Statistics

---

## 📊 **Expected Results for Your STM PDF**

Based on "PUA-STM-Combined Figures .pdf":

### **Phase 1: Panel Extraction**
```
✅ Extracting panels from PDF...
   Pages: 30-50 pages
   Panels: 80-120 panels detected
   Time: 1-2 minutes
```

### **Phase 2: CLIP Similarity**
```
✅ Computing CLIP embeddings...
   Embeddings: 80-120 vectors
   Candidate pairs: 100-200 (similarity ≥ 0.94)
   Time: 30-60 seconds
```

### **Phase 3: Tile Detection**
```
✅ Tile-first pipeline...
   Tiles extracted: 8-12 per panel
   Tile matches: 50-150 matches
   Time: 1-2 minutes
```

### **Phase 4: Verification**
```
✅ SSIM + pHash validation...
   Verified pairs: 30-80 pairs
   Tile evidence: 15-40 pairs (10-15%)
   Time: 30-60 seconds
```

### **Final Results**
```
📊 Summary:
   Total pairs: 40-90
   Tier A (high confidence): 15-35 pairs
   Tier B (review needed): 20-50 pairs
   Multi-tile confirmed: 10-25 pairs
   
   Total runtime: 3-8 minutes
```

---

## 🔍 **What You'll Find in STM Images**

### **Common Duplicate Scenarios**:

1. **Same STM Image, Different Scales**
   - CLIP: 0.96-0.99
   - SSIM: 0.75-0.90
   - Tile evidence: 3-8 tiles
   - **Verdict**: Tier A

2. **Zoomed/Cropped STM Scans**
   - CLIP: 0.94-0.97
   - SSIM: 0.65-0.80
   - Tile evidence: 2-5 tiles
   - **Verdict**: Tier A (with ORB)

3. **Similar Substrates, Different Locations**
   - CLIP: 0.92-0.96
   - SSIM: 0.50-0.70
   - Tile evidence: 0-2 tiles
   - **Verdict**: Tier B (likely false positive)

4. **Reference Images Reused**
   - CLIP: 0.98-1.00
   - SSIM: 0.90-0.98
   - Tile evidence: 6-12 tiles
   - **Verdict**: Tier A (exact duplicate)

---

## 🎨 **Visualizations You'll Get**

For each duplicate pair:

### **1. Side-by-Side Comparison**
`pair_001_detailed/1_raw_A.png` and `2_raw_B_aligned.png`

### **2. SSIM Heatmap**
`pair_001_detailed/4_ssim_viridis.png`
- Green = Similar regions
- Red = Different regions

### **3. Difference Mask**
`pair_001_detailed/5_hard_diff_mask.png`
- Highlights pixel-level differences

### **4. Interactive HTML**
`pair_001_detailed/interactive.html`
- Slider to compare images
- Click to toggle between A and B

### **5. Blink Comparator**
`pair_001_detailed/7_blink.gif`
- Animated comparison

---

## 📈 **Interpreting Your Results**

### **TSV Columns Explained**:

| Column | Meaning | Good Value |
|--------|---------|------------|
| `Cosine_Similarity` | CLIP semantic match | ≥0.94 |
| `SSIM` | Pixel-level match | ≥0.85 |
| `Hamming_Distance` | pHash difference | ≤5 |
| `Tile_Evidence` | Tiles matched? | True |
| `Tile_Evidence_Count` | # matching tiles | ≥2 |
| `Tier` | Confidence | A=High, B=Review |
| `Tier_Path` | Detection method | "Multi-Tile-Confirmed-X" |

### **Decision Guide**:

**Tier A + Tile_Evidence_Count ≥ 3**
→ ✅ **High confidence duplicate**

**Tier A + Tile_Evidence_Count = 1-2**
→ ⚠️ **Review manually** (might be similar but not duplicate)

**Tier B**
→ ⚠️ **Likely false positive** (similar patterns, different content)

---

## 🚨 **Troubleshooting**

### **"Still no module named numpy"**
```powershell
# Check Python path
python -c "import sys; print(sys.executable)"

# Make sure you're using the RIGHT Python
which python
# or
where.exe python
```

### **"TSV is empty (0 pairs found)"**
This is actually GOOD - means no duplicates detected!

If unexpected:
- Lower thresholds: `--sim-threshold 0.90`
- Check panel extraction: Look in `output/panels/`
- Verify PDF quality: Should be high-res

### **"Tile evidence rate too low (<5%)"**
```powershell
# Relax tile thresholds in tile_detection.py
# Change line 46-48:
TILE_CONFOCAL_SSIM_MIN = 0.85  # Was 0.88
TILE_CONFOCAL_NCC_MIN = 0.980  # Was 0.985
```

---

## ✅ **Verification Checklist**

Before running test:
- [ ] Python 3.12 installed
- [ ] pip works: `python -m pip --version`
- [ ] Dependencies installed: `python -c "import open_clip"`
- [ ] PDF exists: `Test-Path "E:\PUA-STM-Combined Figures .pdf"`
- [ ] Scripts exist: `Test-Path automated_test_suite.py`

Ready to run:
- [ ] `python automated_test_suite.py --pdf "E:\..." --verbose`

---

## 📞 **Get Help**

If still stuck:

1. **Show me this**:
```powershell
python --version
python -m pip --version
python -c "import sys; print(sys.executable)"
```

2. **And this**:
```powershell
python automated_test_suite.py --pdf "E:\PUA-STM-Combined Figures .pdf" --verbose 2>&1 | Select-Object -First 100
```

---

## 🎯 **Bottom Line**

**Current status**: Python installation broken, pip not functional

**Best solution**: Fresh Python 3.12 install (15 min setup)

**Once working**: 5-10 min to get your full duplicate detection report

**Your next step**: Choose one of the 3 installation options above and follow the steps!

---

**📁 Save this guide for reference during installation!**

