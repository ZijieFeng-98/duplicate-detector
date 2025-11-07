# Local Testing Guide

## Quick Start

### Step 1: Install Dependencies

```bash
# Install basic dependencies
pip3 install opencv-python-headless pillow pandas numpy imagehash scikit-image scikit-learn tqdm scipy pymupdf

# Or install from requirements.txt
pip3 install -r requirements.txt
```

### Step 2: Run Detection Locally

```bash
# Option 1: Use the automated script
python3 tests/integration/run_detection_local.py

# Option 2: Run manually
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

### Step 3: Check Results

Results will be in: `test_duplicate_detection/detection_results/`

- `final_merged_report.tsv` - All duplicate pairs
- `panel_manifest.tsv` - All detected panels
- `duplicate_comparisons/` - Visual comparisons

## What to Look For

In the results TSV file, look for duplicate pairs containing:

- **Exact duplicates**: Files with `exact` in the name
- **Rotated duplicates**: Files with `rotated` in the name
- **Partial duplicates**: Files with `partial` in the name
- **WB panels**: Files with `WB` prefix
- **Confocal panels**: Files with `confocal` prefix
- **IHC panels**: Files with `IHC` prefix

## Troubleshooting

### Missing Dependencies

If you get `ModuleNotFoundError`:

```bash
pip3 install <missing_module>
```

Common ones:
- `cv2` → `opencv-python-headless`
- `fitz` → `pymupdf`
- `PIL` → `pillow`

### Detection Not Finding Duplicates

Try lowering thresholds:

```bash
--sim-threshold 0.90  # Lower CLIP threshold
--phash-max-dist 6    # Higher pHash distance
```

### Slow Performance

Use fast preset:

```bash
--preset fast
```

## Files Created

After running detection:

```
test_duplicate_detection/
├── pages/                    # Extracted PDF pages
├── intentional_duplicates/   # Created duplicates
│   ├── WB/
│   ├── confocal/
│   └── IHC/
├── test_panels/              # Combined test set
└── detection_results/         # Detection output
    ├── final_merged_report.tsv
    ├── panel_manifest.tsv
    └── duplicate_comparisons/
```

## Next Steps

1. Review `final_merged_report.tsv`
2. Check visual comparisons in `duplicate_comparisons/`
3. Verify all duplicate types were detected
4. Adjust thresholds if needed

