# Local Testing - Quick Reference

## Current Status

✅ **Duplicates Created Successfully!**
- 18 duplicate variants created
- 23 total test images
- Located in: `test_duplicate_detection/`

## To Run Detection Locally

### Option 1: Install Dependencies First

```bash
# Install required packages
pip3 install opencv-python-headless pillow pandas numpy imagehash scikit-image scikit-learn tqdm scipy pymupdf torch torchvision open-clip-torch

# Then run detection
python3 tests/integration/run_detection_local.py
```

### Option 2: Use Streamlit (Easier)

```bash
# Install streamlit if not already installed
pip3 install streamlit

# Run the web interface
streamlit run streamlit_app.py
```

Then upload your PDF through the web interface.

### Option 3: Manual Command

```bash
python3 ai_pdf_panel_duplicate_check_AUTO.py \
  --pdf "/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf" \
  --output test_duplicate_detection/detection_results \
  --preset balanced
```

## Files Ready for Testing

- **Test panels**: `test_duplicate_detection/test_panels/` (23 images)
- **Duplicates**: `test_duplicate_detection/intentional_duplicates/`
  - WB: 6 variants
  - Confocal: 6 variants  
  - IHC: 6 variants

## Quick Test (Basic)

```bash
python3 tests/integration/test_simple.py
```

This tests basic pHash detection without full dependencies.

## Troubleshooting

If you get dependency errors, install them one by one:

```bash
pip3 install opencv-python-headless
pip3 install pymupdf
pip3 install pillow
# etc.
```

Or use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
```

