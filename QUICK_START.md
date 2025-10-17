# 🚀 Quick Start - Deploy in 10 Minutes!

**This folder contains everything you need to deploy your Duplicate Detection Studio to Streamlit Cloud.**

---

## ✅ What's in This Folder

```
APP/
├── streamlit_app.py                  ← Main web UI (READY!)
├── ai_pdf_panel_duplicate_check_AUTO.py  ← Backend (NEEDS CLI - see Step 1)
├── requirements.txt                  ← Python dependencies (READY!)
├── README.md                         ← GitHub docs (READY!)
├── STREAMLIT_DEPLOY.md               ← Detailed guide (READY!)
├── DEPLOY_NOW.md                     ← Quick guide (READY!)
├── .streamlit/config.toml            ← Streamlit config (READY!)
└── .gitignore                        ← Git ignore file (READY!)
```

---

## 🎯 3 Steps to Deploy

### Step 1: Add CLI Support (3 min) ⚠️ REQUIRED

**Open `ai_pdf_panel_duplicate_check_AUTO.py` and make 3 edits:**

**A) Add imports** (after line 16):
```python
import argparse
import sys
```

**B) Add parse function** (after imports, before CONFIG):
```python
def parse_cli_args():
    """Parse command-line arguments for Streamlit integration"""
    parser = argparse.ArgumentParser(
        description="AI-powered duplicate detection for scientific PDFs"
    )
    parser.add_argument("--pdf", type=str, help="Path to input PDF")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--sim-threshold", type=float, default=0.96)
    parser.add_argument("--phash-max-dist", type=int, default=4)
    parser.add_argument("--ssim-threshold", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-phash-bundles", action="store_true")
    parser.add_argument("--use-orb", action="store_true")
    parser.add_argument("--use-tier-gating", action="store_true")
    parser.add_argument("--highlight-diffs", action="store_true")
    parser.add_argument("--enable-cache", action="store_true")
    parser.add_argument("--suppress-same-page", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-auto-open", action="store_true")
    return parser.parse_args()
```

**C) Update main block** (at bottom, line ~2900):

Find:
```python
if __name__ == "__main__":
    main()
```

Replace with:
```python
if __name__ == "__main__":
    # CLI support for Streamlit
    if len(sys.argv) > 1:
        args = parse_cli_args()
        if args.pdf: PDF_PATH = Path(args.pdf)
        if args.output: OUT_DIR = Path(args.output)
        DPI = args.dpi
        SIM_THRESHOLD = args.sim_threshold
        PHASH_MAX_DIST = args.phash_max_dist
        SSIM_THRESHOLD = args.ssim_threshold
        BATCH_SIZE = args.batch_size
        USE_PHASH_BUNDLES = args.use_phash_bundles
        USE_ORB_RANSAC = args.use_orb
        USE_TIER_GATING = args.use_tier_gating
        HIGHLIGHT_DIFFERENCES = args.highlight_diffs
        ENABLE_CACHE = args.enable_cache
        SUPPRESS_SAME_PAGE_DUPES = args.suppress_same_page
        DEBUG_MODE = args.debug
        AUTO_OPEN_RESULTS = not args.no_auto_open
    main()
```

**Test it works:**
```bash
cd APP
python ai_pdf_panel_duplicate_check_AUTO.py --help
```

✅ Should show CLI options!

---

### Step 2: Push to GitHub (4 min)

```bash
# From this APP folder:
cd "/Users/zijiefeng/Desktop/Guo's lab/APP"

# Initialize git
git init
git add .
git commit -m "Initial deployment: Duplicate Detection Studio"

# Push to GitHub (choose one):

# Method A: GitHub CLI (if installed)
gh repo create duplicate-detector --public --source=. --push

# Method B: Manual
# 1. Go to github.com/new
# 2. Create repo "duplicate-detector" (public)
# 3. Then run:
git remote add origin https://github.com/YOUR_USERNAME/duplicate-detector.git
git branch -M main
git push -u origin main
```

✅ Verify at: `https://github.com/YOUR_USERNAME/duplicate-detector`

---

### Step 3: Deploy on Streamlit (3 min)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. **Configure:**
   - Repository: `YOUR_USERNAME/duplicate-detector`
   - Branch: `main`
   - Main file: `streamlit_app.py`
4. Click **"Deploy!"**

Wait 5-8 minutes...

🎉 **Your app is LIVE!**

---

## 🧪 Test Your App

Visit: `https://yourapp.streamlit.app`

1. Upload a small PDF (2-5 pages)
2. Select "Fast" preset
3. Click "Start Analysis"
4. Wait ~2 minutes
5. Download results

✅ Success!

---

## 📖 More Help

- **Quick guide:** `DEPLOY_NOW.md`
- **Detailed guide:** `STREAMLIT_DEPLOY.md`
- **Documentation:** `README.md`
- **Streamlit docs:** [docs.streamlit.io](https://docs.streamlit.io)

---

## ✅ Checklist

**Before deploying:**
- [ ] Added CLI support (Step 1)
- [ ] Tested `--help` command
- [ ] Pushed to GitHub
- [ ] Created Streamlit app

**After deploying:**
- [ ] App loads
- [ ] Upload works
- [ ] Analysis completes
- [ ] Downloads work
- [ ] Shared URL

---

## 🆘 Troubleshooting

**"Module not found"**  
→ Check `requirements.txt` is in repo root

**"Backend not found"**  
→ Both `.py` files must be in same folder

**"Timeout"**  
→ Use "Fast" preset or smaller PDF

**Need more help?**  
→ See `STREAMLIT_DEPLOY.md` for detailed troubleshooting

---

## 🎊 You're Ready!

This folder has everything. Just:
1. Add CLI (3 min)
2. Push to GitHub (4 min)
3. Deploy on Streamlit (3 min)

**Total: 10 minutes to production! 🚀**

Good luck!


