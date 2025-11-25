# 🚀 Streamlit Cloud Deployment Guide

## Quick Start (5 Minutes)

### Step 1: Push to GitHub
```bash
# Make sure all changes are committed
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Connect to Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click**: "New app"
4. **Select**:
   - Repository: `your-username/duplicate-detector` (or your repo name)
   - Branch: `main`
   - Main file: `streamlit_app.py`
   - Python version: `3.12`

### Step 3: Deploy
- Click **"Deploy"**
- Wait 3-5 minutes for build to complete
- Your app will be live at: `https://your-app-name.streamlit.app`

---

## ✅ Pre-Deployment Checklist

### Required Files (All Present ✅)

- [x] `streamlit_app.py` - Main app file
- [x] `requirements.txt` - All dependencies listed
- [x] `.streamlit/config.toml` - Streamlit configuration
- [x] `ai_pdf_panel_duplicate_check_AUTO.py` - Pipeline backend
- [x] `duplicate_detector/` - Core modules

### Verify Requirements.txt

Your `requirements.txt` should include:
```
streamlit>=1.31.0
torch>=2.2.0
open-clip-torch>=2.24.0
pymupdf>=1.23.0
opencv-python-headless>=4.9.0
pandas>=2.2.0
numpy>=1.26.0,<2.0.0
scikit-image>=0.22.0
scikit-learn>=1.3.0
imagehash>=4.3.0
Pillow>=10.0.0
tqdm>=4.66.0
scipy>=1.11.0
pydantic>=2.0.0
pyyaml>=6.0.0
python-dotenv>=1.0.0
```

---

## 🔧 Configuration Files

### `.streamlit/config.toml` (Already Created ✅)
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### `.python-version` (Optional but Recommended)
```
3.12
```

---

## 🐛 Common Issues & Fixes

### Issue 1: "ModuleNotFoundError"
**Fix**: Add missing package to `requirements.txt` and push again

### Issue 2: "Runtime: 0.00s, No Results"
**Fix**: Check Streamlit Cloud logs → "Manage app" → "Logs"
- Look for import errors
- Verify all dependencies installed
- Check if pipeline script exists

### Issue 3: "Timeout Error"
**Fix**: 
- Use `--preset fast` instead of `thorough`
- Reduce PDF size
- Lower DPI (use `--dpi 100`)

### Issue 4: "Memory Error"
**Fix**:
- Streamlit Cloud has 1GB RAM limit
- Use smaller PDFs (<50MB)
- Enable `--no-cache` flag

---

## 📊 Monitoring Deployment

### Check Build Status
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. View "Deployment logs"

### Test After Deployment
1. Upload a small test PDF (<5MB)
2. Use "Balanced" preset
3. Monitor logs for errors
4. Verify results appear

---

## 🎯 Post-Deployment

### First Test
```bash
# Test locally first to catch errors early
streamlit run streamlit_app.py
```

### Verify Online
1. Upload a small PDF
2. Check that panels are detected
3. Verify results appear
4. Test download functionality

---

## 🔄 Updating Your App

After making changes:
```bash
git add .
git commit -m "Your update message"
git push origin main
```

Streamlit Cloud will **auto-redeploy** in ~2-3 minutes.

---

## 📞 Need Help?

1. **Check Logs**: Streamlit Cloud → Manage app → Logs
2. **Test Locally**: `streamlit run streamlit_app.py`
3. **Verify Dependencies**: `pip install -r requirements.txt`
4. **Check File Paths**: Ensure all paths are relative, not absolute

---

## ✅ Success Indicators

Your deployment is successful when:
- ✅ App loads without errors
- ✅ PDF upload works
- ✅ Analysis completes (shows runtime > 0)
- ✅ Results appear (Total Pairs > 0)
- ✅ Download buttons work

---

**Last Updated**: November 25, 2025
**Status**: Ready for deployment 🚀

