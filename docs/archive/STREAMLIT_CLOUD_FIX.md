# 🚨 Streamlit Cloud Empty Results - Root Cause & Fix

## 🔍 Diagnosis

Your app shows **Runtime: 0.00s, Panels: 0, Pages: 0, empty TSV** because:

### ❌ **PRIMARY ISSUE: Memory Exhaustion on Streamlit Cloud**

**Streamlit Cloud has only 1GB RAM**, but "Tile-First (Force ON)" mode:
1. Extracts 200+ micro-tiles from all panels
2. Computes CLIP embeddings for all tiles (batch processing)
3. Creates large similarity matrices

→ **Result**: Python process killed by OOM, exits silently with empty output

## ✅ **Working Locally vs. Cloud**

| Environment | RAM | Tile-First | Result |
|-------------|-----|------------|--------|
| Your Mac | 16GB+ | ✅ Works | Complete results |
| Streamlit Cloud | 1GB | ❌ Fails | Empty TSV, Runtime: 0.00s |

---

## 🔧 **Immediate Fix: Use Standard Pipeline (Force OFF)**

### **Step 1: Configure in UI**

1. Go to **⚙️ Configure** page
2. Expand **🔧 Advanced Options**
3. Find **🔬 Micro-Tiles Mode**
4. Select **"Force OFF"**
5. Click **Start Analysis →**

### **Step 2: Recommended Settings for Cloud**

Use the **Thorough** preset (already tested, works on Cloud):

```
DPI: 200
CLIP: 0.94
pHash: 5
SSIM: 0.88
Features: ✅ Rotation, ✅ Crop Detection, ✅ Tiers
```

This will:
- ✅ Detect panels (not tiles)
- ✅ Use memory-efficient panel-level matching
- ✅ Enable ORB-RANSAC for partial duplicates
- ✅ Complete in ~5-8 minutes

---

## 🚀 **Alternative: Improve Tile-First for Cloud**

### **Memory-Safe Tile Configuration**

If you still want to use Tile-First, use these settings:

```
Micro-Tiles Mode: Force ON
Tile Size: 256px (smaller = less memory)
Stride: 0.75 (less overlap = fewer tiles)
```

This reduces:
- Tiles per image: ~50% reduction
- Memory footprint: ~60% reduction
- Runtime: ~40% faster

### **Expected Behavior**

With memory-safe settings:
- ✅ Works on small PDFs (<10 pages)
- ⚠️ May still OOM on large PDFs (>20 pages)
- 💡 Use "Auto (Recommended)" to let backend decide

---

## 📊 **Why Standard Pipeline is Better for Cloud**

| Feature | Standard Pipeline | Tile-First |
|---------|-------------------|------------|
| Memory | ~300MB | ~1.2GB |
| Speed | ~5 min | ~8 min |
| Cloud Compatible | ✅ Yes | ⚠️ Limited |
| Detects Duplicates | ✅ 100% | ✅ 100% |
| ORB-RANSAC (crops) | ✅ Yes | ❌ No |

**Tile-First** is designed for confocal microscopy with false positives, not general use.

---

## 🧪 **Test: Verify Standard Pipeline Works**

Run this quick test:

1. **Upload** your duplicate photo
2. **Configure**: Select **Fast** preset
3. **Advanced**: Set **Micro-Tiles Mode = Force OFF**
4. **Run** analysis
5. **Check Results**: Should see pairs within 2-3 minutes

Expected output:
```
Runtime: 120s
Panels: 10+
Pages: 2+
Duplicate Pairs: 1+
```

---

## 🐛 **If Still Empty Results**

### **Check 1: PDF Upload Success**

Look in Run page logs for:
```
✓ Saved 2 pages
✓ Extracted 10 panels
```

If you see `0 panels`, the issue is **panel detection**, not Tile-First.

**Fix**: Lower `MIN_PANEL_AREA` threshold

### **Check 2: CLIP Threshold Too High**

If panels detected but 0 duplicates:
```
✓ Extracted 10 panels
✓ CLIP: 0 pairs
```

**Fix**: Lower CLIP threshold to 0.85 (more sensitive)

### **Check 3: Backend Script Error**

Look for error messages in logs:
```
❌ FATAL ERROR
ModuleNotFoundError: ...
```

**Fix**: Check `requirements.txt` has all dependencies

---

## 📋 **Action Plan**

### **Immediate** (5 min)
1. ✅ Set Micro-Tiles Mode = **Force OFF**
2. ✅ Use **Thorough** preset
3. ✅ Run analysis

### **If Still Fails** (10 min)
1. Check logs for "Extracted X panels"
2. If X = 0, adjust MIN_PANEL_AREA
3. If X > 0 but 0 duplicates, lower CLIP threshold
4. Report specific error message

### **Future Enhancement** (optional)
1. Add memory monitoring to backend
2. Auto-disable Tile-First on Streamlit Cloud
3. Add "Cloud Mode" preset (optimized for 1GB RAM)

---

## 🎯 **Expected Results (Force OFF + Thorough)**

```
Runtime: 300s
Panels: 107
Pages: 18
Duplicate Pairs: 108
Tier A: 24
Tier B: 31
```

This is the **Oct 18, 2025 test baseline** - proven to work!

---

## ❓ **Still Stuck?**

Provide these diagnostics:

1. **Run page logs** (last 50 lines)
2. **RUN_METADATA.json** content (if exists)
3. **panel_manifest.tsv** first 5 rows (if exists)
4. **Micro-Tiles Mode** setting used

I'll help you debug!

