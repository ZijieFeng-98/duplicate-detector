# 🧪 Testing Guide

## Overview

This project includes a comprehensive automated testing suite to validate the duplicate detection pipeline. The tests ensure that all components work correctly and catch regressions early.

---

## Quick Start

### **1. Run Quick Smoke Test** (~2-3 minutes)
```bash
./quick_test.sh
```

This runs a single test with default settings and displays basic results.

### **2. Run Full Test Suite** (~10-15 minutes)
```bash
python test_pipeline_auto.py
```

This runs multiple test configurations and validates all pipeline components.

---

## Test Files

### **`test_pipeline_auto.py`** - Main Test Suite
**What it tests:**
- ✅ Prerequisites (Python packages, test data)
- ✅ Pipeline execution (multiple configurations)
- ✅ Output structure (directories and files)
- ✅ Page extraction (PDF → PNG conversion)
- ✅ Panel detection (boundary detection)
- ✅ Duplicate detection (CLIP, pHash, ORB)
- ✅ Tier classification (A/B gating)
- ✅ Metadata integrity (JSON validation)

**Configuration:**
Edit these variables at the top of `test_pipeline_auto.py`:
```python
TEST_PDF_PATH = Path("your/test/pdf.pdf")
TEST_OUTPUT_DIR = Path("./test_output")
MIN_PAGES_EXPECTED = 1
MIN_PANELS_EXPECTED = 5
```

**Add custom test configs:**
```python
TEST_CONFIGS = [
    {
        "name": "Custom",
        "args": ["--sim-threshold", "0.98", "--phash-max-dist", "3"],
        "expect_results": True
    }
]
```

### **`quick_test.sh`** - Quick Smoke Test
**What it does:**
- Runs a single detection with default settings
- Validates basic output (TSV, metadata)
- Shows first 5 duplicate pairs
- Completes in ~2-3 minutes

**Usage:**
```bash
# Make executable (first time only)
chmod +x quick_test.sh

# Run test
./quick_test.sh
```

### **`.cursorrules`** - Cursor AI Integration
**What it provides:**
- Code quality guidelines
- Common patterns and best practices
- Debug workflow for troubleshooting
- Feature development workflow
- Deployment checklist

**How to use with Cursor:**
1. The file is automatically detected by Cursor
2. Ask questions like:
   - "Run tests" → Cursor knows to use `python test_pipeline_auto.py`
   - "Why is TSV empty?" → Cursor follows debug workflow
   - "Add feature X" → Cursor follows feature workflow with tests first

---

## Test Output

### **Success Example:**
```
======================================================================
🧪 TEST: Pipeline Run: Balanced_Default - PUA-STM-Combined Figures .pdf
======================================================================
  ✅ Pipeline completed successfully
  ✅ Found final_merged_report.tsv (45,231 bytes)

======================================================================
🧪 TEST: Duplicate Detection: Balanced_Default
======================================================================
  ℹ️  Found 12 duplicate pair(s)
  ✅ All expected columns present
  ℹ️  Sample results:
  ℹ️    1. page_19_panel02.png vs page_30_panel01.png
  ℹ️       CLIP=0.973, pHash=2

======================================================================
📊 TEST SUMMARY
======================================================================
  Tests run: 16
  Passed: 16 ✅
  Failed: 0 ❌
  Time: 142.3s
======================================================================
🎉 ALL TESTS PASSED!
```

### **Failure Example:**
```
======================================================================
🧪 TEST: Panel Detection
======================================================================
  ❌ Expected ≥5 panels, got 2
  ❌ File missing: panel_manifest.tsv

======================================================================
📊 TEST SUMMARY
======================================================================
  Tests run: 8
  Passed: 6 ✅
  Failed: 2 ❌
  Time: 67.4s
======================================================================
💥 2 TEST(S) FAILED
```

---

## Debugging Failed Tests

### **1. Check Test Logs**
```bash
cat test_output/*/test_run.log | grep "ERROR\|FAILED"
```

### **2. Inspect Intermediate Files**
```bash
ls -lh test_output/*/
# Check for:
#   - pages/ (PNG files)
#   - panels/ (extracted panels)
#   - panel_manifest.tsv (metadata)
#   - final_merged_report.tsv (results)
```

### **3. Review Metadata**
```bash
cat test_output/*/RUN_METADATA.json | jq .
```

### **4. Run Single Test**
Modify `test_pipeline_auto.py`:
```python
# Only run one config
TEST_CONFIGS = [TEST_CONFIGS[0]]  # Just first config
```

### **5. Lower Thresholds**
For debugging, use more permissive values:
```python
TEST_CONFIGS = [
    {
        "name": "Debug",
        "args": ["--sim-threshold", "0.85", "--phash-max-dist", "6"],
        "expect_results": True
    }
]
```

---

## Common Issues

### **Issue: No panels detected**
**Cause:** `MIN_PANEL_AREA` threshold too high  
**Fix:** Edit `ai_pdf_panel_duplicate_check_AUTO.py`:
```python
MIN_PANEL_AREA = 10000  # Lower from 80000
```

### **Issue: Empty TSV file**
**Causes:**
1. Import errors (check logs for `ModuleNotFoundError`)
2. No duplicates found (thresholds too strict)
3. Pipeline crash (check STDERR in test logs)

**Debug:**
```bash
# Run with debug output
python ai_pdf_panel_duplicate_check_AUTO.py \
  --pdf "test.pdf" \
  --output "./debug_out" \
  --sim-threshold 0.85 \
  --debug
```

### **Issue: Tests timeout**
**Cause:** Large PDF or slow hardware  
**Fix:** Increase timeout in `test_pipeline_auto.py`:
```python
result = subprocess.run(
    cmd,
    timeout=1200  # 20 minutes instead of 10
)
```

---

## Integration with CI/CD

### **GitHub Actions Example:**
```yaml
name: Test Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python test_pipeline_auto.py
```

---

## Adding New Tests

### **Step 1: Add test function**
```python
def test_new_feature(output_dir, logger):
    """Test 9: Verify new feature works"""
    logger.test_start("New Feature Test")
    
    # Your test logic here
    if feature_works:
        logger.success("Feature works correctly")
        return True
    else:
        logger.failure("Feature failed")
        return False
```

### **Step 2: Call in main runner**
```python
def run_all_tests():
    # ... existing tests ...
    test_new_feature(output_dir, logger)
```

### **Step 3: Run and verify**
```bash
python test_pipeline_auto.py
```

---

## Performance Benchmarks

**Expected timings on M1/M2 Mac:**
- Quick smoke test: ~2-3 minutes
- Full test suite (2 configs): ~10-15 minutes
- Per-config breakdown:
  - PDF → Pages: ~10s
  - Panel detection: ~20s
  - CLIP embeddings: ~60s
  - Pairwise comparison: ~30s
  - Report generation: ~5s

**Memory usage:**
- Peak: ~2-3 GB (CLIP model + images)
- Cached runs: ~1 GB

---

## Best Practices

1. **Run tests before committing:**
   ```bash
   ./quick_test.sh  # Quick validation
   ```

2. **Run full suite before major releases:**
   ```bash
   python test_pipeline_auto.py
   ```

3. **Keep test data small:**
   - Use a representative 10-20 page PDF
   - Avoid huge files (>100 pages) in automated tests

4. **Update test expectations:**
   - Adjust `MIN_PANELS_EXPECTED` based on your test PDF
   - Update `EXPECTED_COLUMNS` if adding new metrics

5. **Document test changes:**
   - Update this guide when adding new tests
   - Add comments explaining non-obvious assertions

---

## Troubleshooting Help

**Ask Cursor (with `.cursorrules` loaded):**
```
"Run tests and explain failures"
"Why did test_panel_detection fail?"
"Add a test for tile detection"
```

**Check test logs:**
```bash
# Find error messages
grep -r "ERROR\|FAIL" test_output/

# View full logs
less test_output/*/test_run.log
```

**Manual validation:**
```bash
# Check if files exist
ls test_output/*/final_merged_report.tsv

# Count results
wc -l test_output/*/final_merged_report.tsv
```

---

## Resources

- **Main Pipeline:** `ai_pdf_panel_duplicate_check_AUTO.py`
- **Streamlit UI:** `streamlit_app.py`
- **Test Suite:** `test_pipeline_auto.py`
- **Quick Test:** `quick_test.sh`
- **Cursor Integration:** `.cursorrules`
- **Deployment Status:** `DEPLOYMENT_STATUS.md`

---

**Last Updated:** October 18, 2025  
**Status:** ✅ All tests passing

