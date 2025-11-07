# 🔬 Duplicate Detector

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/yourusername/duplicate-detector/workflows/Tests/badge.svg)](https://github.com/yourusername/duplicate-detector/actions)
[![Coverage](https://codecov.io/gh/yourusername/duplicate-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/duplicate-detector)

**Professional scientific figure duplicate detection using multi-stage filtering with CLIP embeddings, pHash bundles, SSIM, and ORB-RANSAC.**

## ✨ Features

- **Multi-Stage Detection Pipeline**
  - CLIP semantic similarity for content matching
  - Rotation/mirror-robust pHash bundles
  - SSIM structural similarity with patch-wise refinement
  - ORB-RANSAC geometric verification for partial duplicates

- **Tier Classification**
  - Tier A: High-confidence duplicates (review required)
  - Tier B: Possible duplicates (manual verification)

- **Modality-Aware Detection**
  - Western blot, confocal microscopy, TEM, bright-field, gel electrophoresis
  - Modality-specific thresholds for optimal accuracy

- **Professional API**
  - Clean Python API (`DuplicateDetector` class)
  - Command-line interface
  - Streamlit web interface
  - Comprehensive configuration system

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/yourusername/duplicate-detector.git
cd duplicate-detector
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from duplicate_detector import DuplicateDetector, DetectorConfig
from pathlib import Path

# Simple usage with preset
detector = DuplicateDetector(config=DetectorConfig.from_preset("balanced"))
results = detector.analyze_pdf("paper.pdf")

print(f"Found {results.total_pairs} duplicate pairs")
print(f"Tier A: {results.get_tier_a_count()}, Tier B: {results.get_tier_b_count()}")
```

### Command Line

```bash
# Use preset configuration
python ai_pdf_panel_duplicate_check_AUTO.py --preset balanced --pdf paper.pdf --output results/

# Use custom config file
python ai_pdf_panel_duplicate_check_AUTO.py --config config.yaml --pdf paper.pdf
```

### Web Interface

```bash
streamlit run streamlit_app.py
```

## 📖 Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Installation, configuration, usage examples
- **[Developer Guide](docs/DEVELOPER.md)** - Architecture, development workflow, contributing
- **[Reproducibility Guide](docs/REPRODUCIBILITY.md)** - Environment specs, dependencies, reproduction steps
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation (coming soon)

## 🏗️ Architecture

The codebase is organized into focused modules:

```
duplicate_detector/
├── core/              # Core detection algorithms
│   ├── panel_detector.py      # Panel extraction
│   ├── similarity_engine.py   # CLIP, pHash, SSIM
│   ├── geometric_verifier.py  # ORB-RANSAC
│   └── tier_classifier.py     # Tier A/B classification
├── models/            # Configuration and data models
├── utils/             # Utilities (logging, etc.)
└── api/               # Public API (DuplicateDetector class)
```

## ⚙️ Configuration

### Presets

Three presets are available:

- **Fast** - Quick analysis (~2 min), may have more false positives
- **Balanced** (Recommended) - Optimal balance (~5 min)
- **Thorough** - Maximum accuracy (~10+ min)

### Configuration File

Create `config.yaml`:

```yaml
dpi: 150
duplicate_detection:
  sim_threshold: 0.96
  phash_max_dist: 3
  ssim_threshold: 0.90
feature_flags:
  use_phash_bundles: true
  use_orb_ransac: true
  use_tier_gating: true
```

See [config.example.yaml](config.example.yaml) for complete example.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=duplicate_detector --cov-report=html

# Run integration tests
pytest tests/integration/ -v -m integration
```

## 📊 Results

After analysis, you'll find:

- `final_merged_report.tsv` - Complete duplicate pairs with metrics
- `panel_manifest.tsv` - All detected panels
- `duplicate_comparisons/` - Visual comparison images
- `RUN_METADATA.json` - Run configuration and statistics

## 🔬 Research Use

This tool is designed for:
- Pre-publication duplicate detection
- Journal figure verification
- Scientific integrity checks
- Research reproducibility

## 🤝 Contributing

Contributions are welcome! Please see [DEVELOPER.md](docs/DEVELOPER.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## 📝 Citation

If you use this software in your research, please cite:

```bibtex
@software{duplicate_detector,
  title = {Duplicate Detector: Scientific Figure Duplicate Detection},
  version = {1.0.0},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/duplicate-detector}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CLIP model by OpenAI
- OpenCV for computer vision
- PyMuPDF for PDF processing
- Streamlit for web interface

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/duplicate-detector/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/duplicate-detector/discussions)

---

**Made with ❤️ for the scientific community**
