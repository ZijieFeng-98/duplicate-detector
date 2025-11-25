# ✅ Quick Deployment Checklist

## Before You Deploy

- [ ] All code changes committed to git
- [ ] `requirements.txt` is up to date
- [ ] `streamlit_app.py` exists and is the main file
- [ ] `.streamlit/config.toml` exists
- [ ] GitHub repository is set up
- [ ] You have a Streamlit Cloud account (free)

## Deployment Steps

### Option 1: Use the Script (Easiest)
```bash
./deploy.sh
```

### Option 2: Manual Steps
```bash
# 1. Commit changes
git add .
git commit -m "Ready for deployment"

# 2. Push to GitHub
git push origin main

# 3. Go to Streamlit Cloud
# Visit: https://share.streamlit.io/
# Click "New app" → Select repo → Deploy
```

## After Deployment

- [ ] App loads without errors
- [ ] Can upload a PDF
- [ ] Analysis runs (check logs)
- [ ] Results appear (not 0/0/0)
- [ ] Download buttons work

## If Something Goes Wrong

1. **Check Streamlit Cloud Logs**
   - Go to your app → "Manage app" → "Logs"
   - Look for Python errors

2. **Common Fixes**
   - Missing dependency → Add to `requirements.txt`
   - Import error → Check file paths
   - Timeout → Use "Fast" preset
   - Memory error → Use smaller PDF

3. **Test Locally First**
   ```bash
   streamlit run streamlit_app.py
   ```

## Your App URL

After deployment, your app will be at:
```
https://your-app-name.streamlit.app
```

---

**Ready?** Run `./deploy.sh` to start! 🚀
