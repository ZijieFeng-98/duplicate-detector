# Reproducibility Guide

This document provides exact specifications for reproducing results.

## Environment

### Python Version

- **Required:** Python 3.12+
- **Tested:** Python 3.12.0

### Operating System

- **Tested on:** macOS, Linux (Ubuntu 20.04+)
- **Streamlit Cloud:** Linux-based

## Dependencies

### Exact Versions (from requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
open-clip-torch>=2.20.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-image>=0.21.0
scipy>=1.11.0
imagehash>=4.3.1
pymupdf>=1.23.0
pydantic>=2.0.0
pyyaml>=6.0.0
python-dotenv>=1.0.0
tqdm>=4.66.0
streamlit>=1.31.0
plotly>=5.18.0
```

### Installation

```bash
pip install -r requirements.txt
```

Or install package:

```bash
pip install -e .
```

## Hardware Specifications

### Minimum Requirements

- **CPU:** 2+ cores
- **RAM:** 4 GB
- **Storage:** 1 GB free space

### Recommended

- **CPU:** 4+ cores
- **RAM:** 8+ GB
- **GPU:** CUDA-compatible (optional, for faster CLIP)
- **Storage:** 5+ GB free space

### Tested Configurations

1. **macOS (M1/M2):**
   - CPU: Apple Silicon
   - RAM: 16 GB
   - Device: CPU (MPS not yet supported)

2. **Linux (Ubuntu):**
   - CPU: Intel/AMD x86_64
   - RAM: 8+ GB
   - Device: CPU or CUDA

3. **Streamlit Cloud:**
   - CPU: 2 cores
   - RAM: 1 GB
   - Device: CPU only

## Random Seeds

### Default Seed

- **Random Seed:** 123
- Set via `DetectorConfig.random_seed`

### Reproducibility

For reproducible results:

```python
from duplicate_detector import DetectorConfig

config = DetectorConfig(random_seed=123)
```

## Configuration Files

### Example Config

See `config.example.yaml` for a complete example.

### Preset Configurations

Three presets are available with fixed parameters:

1. **Fast:** Lower thresholds, faster runtime
2. **Balanced:** Optimal balance (recommended)
3. **Thorough:** Higher thresholds, slower but more accurate

## Reproducing Specific Results

### Step-by-Step

1. **Install exact dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set random seed:**
   ```python
   config = DetectorConfig(random_seed=123)
   ```

3. **Use same preset:**
   ```python
   config = DetectorConfig.from_preset("balanced")
   ```

4. **Run analysis:**
   ```python
   detector = DuplicateDetector(config=config)
   results = detector.analyze_pdf("paper.pdf")
   ```

5. **Verify output:**
   - Check `RUN_METADATA.json` for configuration
   - Compare `final_merged_report.tsv` with expected results

### Known Variations

Some non-deterministic aspects:

1. **CLIP Model:** Slight variations in embeddings (normal)
2. **ORB Features:** May vary slightly between runs
3. **Panel Detection:** Edge cases may differ
4. **Cache:** First run vs cached runs may differ

## Benchmark Dataset

### Test Data

- **Location:** `/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong`
- **File:** `STM-Combined Figures.pdf`
- **Size:** 50.42 MB

### Expected Results (Balanced Preset)

- **Pages:** ~30-40 (varies by PDF)
- **Panels:** ~100-150 (varies by PDF)
- **Runtime:** ~5-10 minutes (CPU)

## Version Information

### Package Version

- **Current:** 1.0.0
- **Location:** `duplicate_detector/__init__.py`

### Module Versions

Check `RUN_METADATA.json` after a run for exact versions of:
- Python
- NumPy
- Pandas
- PyTorch
- OpenCV
- etc.

## Troubleshooting Reproducibility

### Different Results Between Runs

1. **Check random seed:** Ensure same seed used
2. **Check cache:** Clear cache if needed
3. **Check versions:** Verify dependency versions match
4. **Check hardware:** GPU vs CPU may differ slightly

### Clearing Cache

```python
import shutil
from pathlib import Path

cache_dir = Path("output/cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
```

### Verifying Versions

```python
import sys
import numpy as np
import pandas as pd
import torch

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"PyTorch: {torch.__version__}")
```

## Citation

If using this software in research, please cite:

```bibtex
@software{duplicate_detector,
  title = {Duplicate Detector: Scientific Figure Duplicate Detection},
  version = {1.0.0},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/duplicate-detector}
}
```

