# User Guide

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/duplicate-detector.git
cd duplicate-detector

# Install in development mode
pip install -e ".[dev]"

# Or install production version
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Python API

```python
from duplicate_detector import DuplicateDetector, DetectorConfig
from pathlib import Path

# Simple usage with preset
detector = DuplicateDetector(config=DetectorConfig.from_preset("balanced"))
results = detector.analyze_pdf("paper.pdf")

# View results
print(f"Found {results.total_pairs} duplicate pairs")
print(f"Tier A (high confidence): {results.get_tier_a_count()}")
print(f"Tier B (manual review): {results.get_tier_b_count()}")

# Save results
results.save(Path("results/duplicates.csv"))
```

### Using the Command Line

```bash
# Use preset configuration
python ai_pdf_panel_duplicate_check_AUTO.py --preset balanced --pdf paper.pdf --output results/

# Use custom config file
python ai_pdf_panel_duplicate_check_AUTO.py --config config.yaml --pdf paper.pdf

# Customize thresholds
python ai_pdf_panel_duplicate_check_AUTO.py \
    --pdf paper.pdf \
    --sim-threshold 0.96 \
    --phash-max-dist 3 \
    --ssim-threshold 0.90
```

### Using Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## Configuration

### Preset Configurations

Three presets are available:

1. **Fast** - Quick analysis, lower thresholds
   - Runtime: ~2 minutes
   - May have more false positives

2. **Balanced** (Recommended) - Good balance of speed and accuracy
   - Runtime: ~5 minutes
   - Optimal for most use cases

3. **Thorough** - Maximum accuracy, slower
   - Runtime: ~10+ minutes
   - Best for publication-quality analysis

### Configuration Files

Create a `config.yaml` file:

```yaml
dpi: 150
random_seed: 123

duplicate_detection:
  sim_threshold: 0.96
  phash_max_dist: 3
  ssim_threshold: 0.90

feature_flags:
  use_phash_bundles: true
  use_orb_ransac: true
  use_tier_gating: true
  enable_cache: true

performance:
  batch_size: 32
  num_workers: 4
  device: "cpu"  # or "cuda" for GPU
```

Load it:

```python
config = DetectorConfig.from_yaml(Path("config.yaml"))
detector = DuplicateDetector(config=config)
```

### Environment Variables

Set environment variables:

```bash
export DUPLICATE_DETECTOR_PDF_PATH="/path/to/paper.pdf"
export DUPLICATE_DETECTOR_OUTPUT_DIR="/path/to/output"
export DUPLICATE_DETECTOR_SIM_THRESHOLD=0.96
```

Or use a `.env` file:

```env
DUPLICATE_DETECTOR_PDF_PATH=/path/to/paper.pdf
DUPLICATE_DETECTOR_OUTPUT_DIR=/path/to/output
DUPLICATE_DETECTOR_SIM_THRESHOLD=0.96
```

## Understanding Results

### Tier Classification

Results are classified into tiers:

- **Tier A** (High Confidence): Very likely duplicates
  - Exact matches (pHash distance ≤ 3)
  - High CLIP + High SSIM
  - ORB-RANSAC verified partial duplicates
  
- **Tier B** (Manual Review): Possible duplicates
  - Borderline cases
  - Require manual verification

### Output Files

After analysis, you'll find:

- `final_merged_report.tsv` - Complete duplicate pairs with all metrics
- `ai_duplicate_report.tsv` - CLIP-based duplicates
- `phash_duplicate_report.tsv` - pHash-based duplicates
- `panel_manifest.tsv` - All detected panels
- `duplicate_comparisons/` - Visual comparison images
- `RUN_METADATA.json` - Run configuration and statistics

### Reading Results

```python
import pandas as pd

# Load results
df = pd.read_csv("results/final_merged_report.tsv", sep="\t")

# Filter Tier A pairs
tier_a = df[df['Tier'] == 'A']

# Sort by confidence
tier_a_sorted = tier_a.sort_values('Cosine_Similarity', ascending=False)

# View top pairs
print(tier_a_sorted[['Image_A', 'Image_B', 'Cosine_Similarity', 'SSIM']].head(10))
```

## Advanced Usage

### Custom Thresholds

```python
from duplicate_detector import DetectorConfig, DuplicateDetectionConfig

config = DetectorConfig(
    pdf_path=Path("paper.pdf"),
    output_dir=Path("results"),
    duplicate_detection=DuplicateDetectionConfig(
        sim_threshold=0.98,  # Very strict CLIP threshold
        phash_max_dist=2,    # Very strict pHash
        ssim_threshold=0.95  # High SSIM requirement
    )
)

detector = DuplicateDetector(config=config)
results = detector.analyze_pdf()
```

### Modality-Specific Detection

```python
config = DetectorConfig.from_preset("balanced")
config.feature_flags.use_modality_specific_gating = True
config.feature_flags.enable_modality_detection = True

detector = DuplicateDetector(config=config)
results = detector.analyze_pdf()
```

### Batch Processing

```python
from pathlib import Path
from duplicate_detector import DuplicateDetector, DetectorConfig

pdf_files = list(Path("papers/").glob("*.pdf"))

for pdf in pdf_files:
    config = DetectorConfig.from_preset("balanced")
    config.pdf_path = pdf
    config.output_dir = Path(f"results/{pdf.stem}")
    
    detector = DuplicateDetector(config=config)
    results = detector.analyze_pdf()
    
    print(f"{pdf.name}: {results.total_pairs} duplicates found")
```

## Troubleshooting

### Common Issues

**Problem:** No panels detected
- **Solution:** Lower `min_panel_area` threshold in config
- Check if PDF has actual figure panels (not just text)

**Problem:** Too many false positives
- **Solution:** Increase `sim_threshold` (e.g., 0.98)
- Enable `use_tier_gating` for better filtering
- Enable `use_modality_specific_gating` for modality-aware filtering

**Problem:** Too slow
- **Solution:** Use "fast" preset
- Reduce `batch_size` if memory constrained
- Disable `use_orb_ransac` if not needed
- Enable caching (`enable_cache: true`)

**Problem:** Out of memory
- **Solution:** Reduce `batch_size` (e.g., 16 or 8)
- Process fewer pages at a time
- Use CPU instead of GPU if GPU memory limited

### Getting Help

- Check the [Developer Guide](DEVELOPER.md) for technical details
- Review [Reproducibility Guide](REPRODUCIBILITY.md) for version information
- Open an issue on GitHub for bugs or feature requests

## Examples

See the `examples/` directory for:
- Basic usage examples
- Advanced configuration examples
- Batch processing scripts
- Custom pipeline examples

