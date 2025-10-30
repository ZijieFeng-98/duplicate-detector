
# 🚀 Cursor AI: One-Command Test Execution

**For Cursor AI**: Copy and paste the commands below into your terminal. No manual interaction required.

---

## ⚡ Quick Test (Single Command)

```bash
# Make script executable and run
chmod +x quick_test.sh && ./quick_test.sh /path/to/your/test.pdf
```

**Replace** `/path/to/your/test.pdf` with your actual PDF path.

---

## 🐍 Python Test Suite (Alternative)

```bash
# Install dependencies (if needed)
pip install -q open-clip-torch pandas opencv-python-headless pillow imagehash tqdm scikit-image

# Run full test suite
python automated_test_suite.py --pdf /path/to/your/test.pdf --verbose

# Or with custom tile size
python automated_test_suite.py --pdf /path/to/your/test.pdf --tile-size 256 --verbose
```

---

## 📊 What Gets Tested

| Test | Duration | What It Checks |
|------|----------|----------------|
| **Smoke Test** | 2-5 min | Pipeline runs without crashing, TSV output generated, tile evidence present |
| **Synthetic Test** | 1-3 min | Precision ≥90%, Recall ≥80% on synthetic duplicates |
| **Real Data Test** | 5-10 min | Validation on real labeled data (optional, if available) |

---

## ✅ Success Indicators

**Look for these in the output**:
```
✅ TSV generated: ./test_results/.../final_merged_report.tsv
✅ Pairs with tile evidence: 47 (9.1%)
✅ Tile evidence rate is reasonable
✅ SMOKE TEST PASSED
🎉 ALL TESTS PASSED!
```

**Files created**:
- `./test_results/test_log_*.txt` - Detailed execution log
- `./test_results/test_results.json` - Test summary (pass/fail)
- `./test_results/smoke_test/final_merged_report.tsv` - Duplicate pairs TSV
- `./test_results/smoke_test/run.log` - Pipeline execution log

---

## ❌ Common Failures & Auto-Fixes

### **Failure 1: `ModuleNotFoundError: No module named 'open_clip_wrapper'`**

**Auto-fix applied by test script**:
```bash
# Script will detect this and prompt you to:
# Edit ai_pdf_panel_duplicate_check_AUTO.py
# Find: from open_clip_wrapper import load_clip_model
# The load_clip() function should already be correct (direct open_clip import)
```

**Manual verification**:
```bash
# Check if load_clip() function uses direct imports:
grep -A 10 "def load_clip" ai_pdf_panel_duplicate_check_AUTO.py
```

Should show:
```python
def load_clip(model_name="ViT-B-32", pretrained="openai") -> CLIPModel:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
```

---

### **Failure 2: Low Tile Evidence Rate (<5%)**

**Diagnosis**: Tile extraction or verification too strict

**Auto-fix**: The TileConfig has already been updated with relaxed thresholds. If still too strict:

```bash
# Further relax thresholds in tile_detection.py
python << 'EOF'
import fileinput
import sys

changes = {
    'TILE_CONFOCAL_SSIM_MIN = 0.88': 'TILE_CONFOCAL_SSIM_MIN = 0.85',
    'TILE_CONFOCAL_NCC_MIN = 0.985': 'TILE_CONFOCAL_NCC_MIN = 0.980',
}

for line in fileinput.input('tile_detection.py', inplace=True):
    for old, new in changes.items():
        if old in line:
            line = line.replace(old, new)
    print(line, end='')

print("✅ Thresholds relaxed", file=sys.stderr)
EOF
```

**Then rerun**:
```bash
./quick_test.sh /path/to/test.pdf
```

---

### **Failure 3: Tile Size Too Large**

**Error message**: `Tile size 384 > panel height 271`

**Fix**: Already applied! The scripts use 256px tiles by default.

**Manual override if needed**:
```bash
./quick_test.sh /path/to/test.pdf  # Uses 256px by default

# Or with Python test suite:
python automated_test_suite.py --pdf test.pdf --tile-size 256
```

---

## 🤖 For Cursor AI: Execute This Sequence

```bash
# Step 1: Verify environment
python -c "import sys; print(f'Python {sys.version}')"

# Step 2: Check dependencies (auto-install if missing)
python -c "import open_clip, pandas, cv2, imagehash, PIL, tqdm" 2>/dev/null || \
  pip install -q open-clip-torch pandas opencv-python-headless pillow imagehash tqdm scikit-image

# Step 3: Run quick test (replace PATH_TO_PDF with actual path)
chmod +x quick_test.sh
./quick_test.sh PATH_TO_PDF

# Step 4: Check results
cat test_results_*/pipeline.log | tail -n 50

# Step 5: View tile evidence
python << 'EOF'
import pandas as pd
import glob

tsv_files = glob.glob('test_results_*/final_merged_report.tsv', recursive=True)
if tsv_files:
    df = pd.read_csv(tsv_files[0], sep='\t')
    print(f'Total pairs: {len(df)}')
    if 'Tile_Evidence' in df.columns:
        print(f'Tile evidence: {df["Tile_Evidence"].sum()} ({df["Tile_Evidence"].mean():.1%})')
    if 'Tier' in df.columns:
        print(f'Tier-A: {len(df[df["Tier"] == "A"])}')
else:
    print('No TSV files found')
EOF
```

---

## 📝 Test Results Interpretation

### **Good Result** ✅
```
Total pairs: 518
Pairs with tile evidence: 47 (9.1%)
Tier-A pairs: 23
Average tiles per match: 3.2
✅ Tile evidence rate is reasonable
✅ TEST PASSED
```
→ **Pipeline is working correctly**

### **Warning Result** ⚠️
```
Total pairs: 518
Pairs with tile evidence: 12 (2.3%)
⚠️  WARNING: Low tile evidence rate (<5%)
   Consider: Relax SSIM/NCC thresholds or reduce tile size
⚠️  TEST PASSED WITH WARNINGS
```
→ **Pipeline works but needs tuning** (apply Failure #2 fix)

### **Failure Result** ❌
```
❌ Pipeline failed with exit code 1
❌ Error analyzing TSV: [Errno 2] No such file or directory
```
→ **Check pipeline.log for errors**

---

## 🎯 Next Steps After Testing

### **If All Tests Pass** ✅
1. Review `final_merged_report.tsv` to validate duplicate pairs
2. Check `Tile_Evidence_Count` column (should be ≥2 for high-confidence matches)
3. Use pipeline in production:
   ```bash
   python ai_pdf_panel_duplicate_check_AUTO.py \
     --pdf production_paper.pdf \
     --output results \
     --tile-first \
     --tile-size 256
   ```

### **If Tests Fail** ❌
1. Check `pipeline.log` for error details
2. Apply relevant auto-fix from above
3. Rerun: `./quick_test.sh /path/to/test.pdf`
4. If still failing, check TILE_DETECTION_FIXES_COMPLETE.md

---

## 🔍 Debugging Commands

```bash
# View detailed log
less test_results_*/pipeline.log

# Check TSV structure
head -n 1 test_results_*/final_merged_report.tsv | tr '\t' '\n' | nl

# Count tile matches
python << 'EOF'
import pandas as pd, glob
tsv_files = glob.glob('test_results_*/final_merged_report.tsv', recursive=True)
if tsv_files:
    df = pd.read_csv(tsv_files[0], sep='\t')
    if 'Tile_Evidence_Count' in df.columns:
        print(df['Tile_Evidence_Count'].value_counts().sort_index())
EOF

# View Tier-A pairs with tile confirmation
python << 'EOF'
import pandas as pd, glob
tsv_files = glob.glob('test_results_*/final_merged_report.tsv', recursive=True)
if tsv_files:
    df = pd.read_csv(tsv_files[0], sep='\t')
    if 'Tier' in df.columns and 'Tile_Evidence' in df.columns:
        tier_a = df[(df['Tier'] == 'A') & (df['Tile_Evidence'] == True)]
        if 'Image_A' in tier_a.columns:
            print(tier_a[['Image_A', 'Image_B', 'Tile_Evidence_Count', 'Tier_Path']].head(10))
EOF
```

---

## 🚨 Emergency: Skip Tests and Run Directly

```bash
# Run pipeline directly (skip validation)
python ai_pdf_panel_duplicate_check_AUTO.py \
  --pdf test.pdf \
  --output direct_output \
  --tile-first \
  --tile-size 256 \
  --enable-cache

# Check if it worked
ls -lh direct_output/final_merged_report.tsv
```

---

**🎯 Cursor AI: Start with the quick test command at the top!**

**Expected runtime**: 2-5 minutes for quick test

**On success**: You'll see `🎉 TEST PASSED!` and results in `./test_results_*/`

**On failure**: Error logs will be in `./test_results_*/pipeline.log` with specific details

