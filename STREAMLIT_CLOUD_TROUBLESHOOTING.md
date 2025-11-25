# Streamlit Cloud Troubleshooting Guide

## Issue: Pipeline Returns 0 Results (Runtime: 0.00s, Panels: 0, Pages: 0)

### What Was Fixed

1. **Preset Flag Support**: The app now uses `--preset` flag when available (simpler and more reliable than individual flags)
2. **Better Error Detection**: Added validation to detect when pipeline completes but doesn't create output files
3. **Improved Logging**: Command being executed is now shown in logs for debugging
4. **Pre-flight Checks**: Added checks to catch import errors early

### Common Causes on Streamlit Cloud

1. **Missing Dependencies**: Check `requirements.txt` includes all packages
2. **Import Errors**: The pipeline script may fail to import modules
3. **File Path Issues**: Absolute paths may not work on Cloud
4. **Memory Limits**: Large PDFs may exceed Cloud memory limits
5. **Timeout**: Pipeline may be killed before completion

### How to Debug

1. **Check Streamlit Cloud Logs**:
   - Go to your app → "Manage app" → "Logs"
   - Look for Python errors or import failures

2. **Check the Command Log**:
   - On the Run page, expand "Detailed Logs"
   - Copy the command shown and run it locally to test

3. **Verify Dependencies**:
   ```bash
   # Check if all packages are in requirements.txt
   pip freeze | grep -E "(torch|open-clip|opencv|pymupdf|pandas|numpy)"
   ```

4. **Test Locally First**:
   ```bash
   python ai_pdf_panel_duplicate_check_AUTO.py --pdf test.pdf --preset balanced --output test_output
   ```

### Quick Fixes

1. **If import errors**: Add missing packages to `requirements.txt`
2. **If timeout**: Use `--preset fast` instead of `thorough`
3. **If memory issues**: Reduce PDF size or DPI (use `--dpi 100`)
4. **If path issues**: Ensure all paths use `Path()` objects, not strings

### Next Steps

1. Deploy the updated `streamlit_app.py`
2. Try uploading a small test PDF
3. Check the logs for the actual error message
4. Report the specific error if it persists

