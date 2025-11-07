# Troubleshooting: 0 Panels Detected ❌

## Problem

Your detection shows:
- **Total Pairs**: 0
- **Panels**: 0  
- **Pages**: 0
- **Runtime**: 0.00s

## Root Cause

**"Pages: 0"** means the PDF wasn't processed at all. This is different from panel detection failing.

## Why This Happens

### 1. **PDF Upload Issue** (Most Common)
- Streamlit didn't save the PDF to temp storage
- File permissions issue
- Cloud environment temp storage full

### 2. **Silent Error During PDF Conversion**
- PyMuPDF failed to open PDF
- PDF is password-protected or corrupted
- Import error that stopped the pipeline early

### 3. **Backend Not Receiving PDF Path**
- Session state not preserved
- Streamlit rerun cleared the upload

## Solutions

### ✅ Solution 1: Re-upload the PDF

1. Go back to **📤 Upload** tab
2. Re-upload your PDF
3. Wait for the green success message
4. Click **Next →**
5. Click **▶️ Run Analysis** again

### ✅ Solution 2: Check the Logs

1. While running, look at the **📋 Logs** section
2. Find these lines:
   ```
   [1/7] Converting PDF (150 DPI)...
   Extracted X pages
   ```
3. If you see "Extracted 0 pages" → PDF read failed
4. If you don't see this at all → Backend didn't start

### ✅ Solution 3: Download and Run Locally

If Streamlit Cloud is having issues:

```bash
# In terminal
cd /Users/zijiefeng/.cursor/worktrees/Streamlit_Duplicate_Detector/BRDzr

python3 ai_pdf_panel_duplicate_check_AUTO.py \
  "/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf" \
  --output-dir ./test_local \
  --dpi 150 \
  --sim-threshold 0.96 \
  --tile-first-auto
```

This runs locally and will show clearer error messages.

### ✅ Solution 4: Check Streamlit Cloud Logs

If running on Streamlit Cloud:

1. Go to your app's management page
2. Click **Manage app** (hamburger menu)
3. View **Logs** tab
4. Look for Python errors or import failures

## Expected Behavior

A successful run should show:
- **Pages**: 34 (or your PDF's page count)
- **Panels**: 50-200 (typical range)
- **Runtime**: 60-300s (1-5 minutes)
- **Total Pairs**: Variable (0 is OK if no duplicates found, but Pages/Panels should be > 0)

## Quick Test

Run the diagnostic script to verify your PDF is readable:

```bash
python3 diagnose_panels.py "/path/to/your.pdf"
```

This will show if pages can be extracted at all.

## Still Not Working?

If you:
- ✓ Re-uploaded the PDF
- ✓ See "Extracted 34 pages" in logs
- ✓ But still get "Panels: 0"

Then the issue is **panel detection**, not PDF upload. Try:

1. Lower the panel detection threshold:
   - In **⚙️ Configure** → **🔧 Advanced Options**
   - Set **Min Panel Area**: `10000` (default: 80000)

2. Use **Tile-First Mode**:
   - Should already be on "Auto (Recommended)"
   - This subdivides large images automatically

---

## Summary

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pages: 0 | PDF upload failed | Re-upload PDF |
| Pages: 34, Panels: 0 | Panel detection failed | Lower MIN_PANEL_AREA |
| Runtime: 0.00s | Backend didn't run | Check logs for errors |

**Most Common Fix**: Just re-upload your PDF and try again! 🔄

