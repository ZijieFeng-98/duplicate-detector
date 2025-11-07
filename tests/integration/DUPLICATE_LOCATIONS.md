# Duplicate Files Location Guide

## 📍 Exact File Locations

### Base Directory
```
/Users/zijiefeng/.cursor/worktrees/Streamlit_Duplicate_Detector/BRDzr/test_duplicate_detection/
```

### Main Duplicate Files Location

**Primary Location (Organized by Type):**
```
test_duplicate_detection/intentional_duplicates/
├── WB/                              ← Western Blot duplicates (6 files)
│   ├── WB_page_001_exact.png       (2.7 MB)
│   ├── WB_page_001_rotated_90.png  (2.8 MB)
│   ├── WB_page_001_rotated_180.png (2.6 MB)
│   ├── WB_page_001_mirrored.png    (2.6 MB)
│   ├── WB_page_001_partial_70pct.png (1.5 MB)
│   └── WB_page_001_partial_50pct.png (566 KB)
│
├── confocal/                        ← Confocal duplicates (6 files)
│   ├── confocal_page_002_exact.png (1.4 MB)
│   ├── confocal_page_002_rotated_90.png (1.6 MB)
│   ├── confocal_page_002_rotated_180.png (1.4 MB)
│   ├── confocal_page_002_mirrored.png (1.5 MB)
│   ├── confocal_page_002_partial_70pct.png (757 KB)
│   └── confocal_page_002_partial_50pct.png (400 KB)
│
└── IHC/                             ← IHC duplicates (6 files)
    ├── IHC_page_003_exact.png      (2.9 MB)
    ├── IHC_page_003_rotated_90.png (2.9 MB)
    ├── IHC_page_003_rotated_180.png (2.6 MB)
    ├── IHC_page_003_mirrored.png   (2.6 MB)
    ├── IHC_page_003_partial_70pct.png (1.4 MB)
    └── IHC_page_003_partial_50pct.png (812 KB)
```

**Full Paths:**
- WB: `/Users/zijiefeng/.cursor/worktrees/Streamlit_Duplicate_Detector/BRDzr/test_duplicate_detection/intentional_duplicates/WB/`
- Confocal: `/Users/zijiefeng/.cursor/worktrees/Streamlit_Duplicate_Detector/BRDzr/test_duplicate_detection/intentional_duplicates/confocal/`
- IHC: `/Users/zijiefeng/.cursor/worktrees/Streamlit_Duplicate_Detector/BRDzr/test_duplicate_detection/intentional_duplicates/IHC/`

### Combined Test Set (All Files Together)

**Test Panels Directory:**
```
test_duplicate_detection/test_panels/
```

**Contains:**
- 5 original pages (page_001.png through page_005.png)
- 18 duplicate variants (all WB, confocal, IHC duplicates)
- **Total: 23 PNG files**

**Full Path:**
```
/Users/zijiefeng/.cursor/worktrees/Streamlit_Duplicate_Detector/BRDzr/test_duplicate_detection/test_panels/
```

### Original Pages

**Pages Directory:**
```
test_duplicate_detection/pages/
├── page_001.png
├── page_002.png
├── page_003.png
├── page_004.png
└── page_005.png
```

## 📊 Summary

### Duplicate Files Created

| Type | Count | Location |
|------|-------|----------|
| **WB** | 6 files | `intentional_duplicates/WB/` |
| **Confocal** | 6 files | `intentional_duplicates/confocal/` |
| **IHC** | 6 files | `intentional_duplicates/IHC/` |
| **Total Duplicates** | **18 files** | `intentional_duplicates/` |
| **Test Set** | **23 files** | `test_panels/` (includes originals + duplicates) |

### Duplicate Types

Each category has 6 variants:
1. **exact** - Exact duplicate (same image)
2. **rotated_90** - Rotated 90 degrees
3. **rotated_180** - Rotated 180 degrees
4. **mirrored** - Horizontal flip
5. **partial_70pct** - Cropped to 70% (partial overlap)
6. **partial_50pct** - Cropped to 50% (partial overlap)

## 🔍 Quick Access Commands

```bash
# View WB duplicates
ls -lh test_duplicate_detection/intentional_duplicates/WB/

# View confocal duplicates
ls -lh test_duplicate_detection/intentional_duplicates/confocal/

# View IHC duplicates
ls -lh test_duplicate_detection/intentional_duplicates/IHC/

# View all test panels (originals + duplicates)
ls -lh test_duplicate_detection/test_panels/

# Open in Finder (Mac)
open test_duplicate_detection/intentional_duplicates/
```

## 📝 Notes

- **Original source**: Pages extracted from `STM-Combined Figures.pdf`
- **WB duplicates**: Created from page_001.png
- **Confocal duplicates**: Created from page_002.png
- **IHC duplicates**: Created from page_003.png
- **All duplicates**: Also copied to `test_panels/` for easy testing

## ✅ Verification

All duplicate files are confirmed to exist at:
- `/Users/zijiefeng/.cursor/worktrees/Streamlit_Duplicate_Detector/BRDzr/test_duplicate_detection/intentional_duplicates/`

You can use these files to test the duplicate detection pipeline!

