# Developer Guide

## Architecture Overview

The duplicate detector is organized into modular components:

```
duplicate_detector/
├── models/          # Configuration and data models
├── core/            # Core detection algorithms
│   ├── panel_detector.py      # Panel extraction
│   ├── similarity_engine.py   # CLIP, pHash, SSIM
│   ├── geometric_verifier.py  # ORB-RANSAC
│   └── tier_classifier.py     # Tier A/B classification
├── utils/           # Utilities
│   ├── logger.py    # Logging system
│   └── module_wrapper.py  # Module wrappers
└── api/             # Public API
    └── detector.py  # DuplicateDetector class
```

## Development Setup

### Clone and Install

```bash
git clone https://github.com/yourusername/duplicate-detector.git
cd duplicate-detector
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=duplicate_detector --cov-report=html

# Run specific test file
pytest tests/unit/test_panel_detector.py -v
```

### Code Quality

```bash
# Format code
black duplicate_detector tests

# Lint code
ruff check duplicate_detector tests

# Type checking
mypy duplicate_detector --ignore-missing-imports
```

## Core Modules

### Panel Detector

Extracts panels from PDF pages using computer vision.

**Key Functions:**
- `pdf_to_pages()` - Convert PDF to PNG pages
- `detect_panels_cv()` - Detect panels with NMS
- `pages_to_panels_auto()` - Batch panel extraction

**Usage:**
```python
from duplicate_detector.core.panel_detector import pdf_to_pages, pages_to_panels_auto

pages = pdf_to_pages(pdf_path, out_dir, dpi=150)
panels, meta_df = pages_to_panels_auto(pages, out_dir)
```

### Similarity Engine

Computes similarity using CLIP, pHash, and SSIM.

**Key Functions:**
- `load_clip()` - Load CLIP model
- `embed_images()` - Generate CLIP embeddings
- `phash_find_duplicates_with_bundles()` - pHash detection
- `compute_ssim_normalized()` - SSIM computation

**Usage:**
```python
from duplicate_detector.core.similarity_engine import (
    load_clip, load_or_compute_embeddings,
    clip_find_duplicates_threshold
)

clip = load_clip(device="cpu")
vecs = load_or_compute_embeddings(panels, clip, out_dir, "v1")
duplicates = clip_find_duplicates_threshold(panels, vecs, threshold=0.96, meta_df)
```

### Geometric Verifier

Detects partial duplicates using ORB-RANSAC.

**Key Functions:**
- `extract_orb_features()` - Extract ORB keypoints
- `estimate_homography_ransac()` - RANSAC homography
- `orb_find_partial_duplicates()` - Main ORB pipeline

**Usage:**
```python
from duplicate_detector.core.geometric_verifier import orb_find_partial_duplicates

orb_duplicates = orb_find_partial_duplicates(
    panels, clip_df, phash_df, meta_df, out_dir, "v1"
)
```

### Tier Classifier

Classifies duplicates into Tier A/B.

**Key Functions:**
- `apply_tier_gating()` - Universal tier classification
- `_apply_modality_specific_tier_gating()` - Modality-specific

**Usage:**
```python
from duplicate_detector.core.tier_classifier import apply_tier_gating

df_tiered = apply_tier_gating(df_merged)
```

## Adding New Features

### Adding a New Detection Method

1. Create function in appropriate module:
```python
# duplicate_detector/core/similarity_engine.py

def new_detection_method(panels, threshold):
    """New detection method."""
    # Implementation
    return results_df
```

2. Add configuration:
```python
# duplicate_detector/models/config.py

class DuplicateDetectionConfig(BaseModel):
    # ... existing fields ...
    new_method_threshold: float = Field(0.95, ge=0.0, le=1.0)
```

3. Integrate into pipeline:
```python
# duplicate_detector/api/detector.py

# In analyze_pdf() method
if config.feature_flags.use_new_method:
    df_new = new_detection_method(panels, config.duplicate_detection.new_method_threshold)
```

4. Add tests:
```python
# tests/unit/test_similarity_engine.py

def test_new_detection_method():
    # Test implementation
    pass
```

### Adding a New Configuration Option

1. Add to appropriate config class:
```python
# duplicate_detector/models/config.py

class FeatureFlags(BaseModel):
    # ... existing fields ...
    use_new_feature: bool = Field(False, description="Enable new feature")
```

2. Add to preset configurations:
```python
# In DetectorConfig.from_preset()

if preset == "balanced":
    config.feature_flags.use_new_feature = True
```

3. Update CLI arguments (if needed):
```python
# ai_pdf_panel_duplicate_check_AUTO.py

parser.add_argument("--use-new-feature", action="store_true")
```

## Testing Guidelines

### Unit Tests

- Test each function independently
- Use fixtures for common test data
- Mock external dependencies
- Aim for >80% code coverage

### Integration Tests

- Test full pipeline with real data
- Use actual PDF files when possible
- Verify output formats
- Test error handling

### Writing Tests

```python
import pytest
from duplicate_detector.core.panel_detector import detect_panels_cv

def test_detect_panels_cv(sample_image, temp_dir):
    """Test panel detection."""
    results = detect_panels_cv(sample_image, temp_dir)
    assert len(results) > 0
    assert isinstance(results[0][0], Path)
```

## Code Style

- Follow PEP 8
- Use type hints for all functions
- Add docstrings (Google style)
- Keep functions under 50 lines when possible
- Use meaningful variable names

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and linting
6. Submit a pull request

## Release Process

1. Update version in `duplicate_detector/__init__.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build package: `python -m build`
5. Tag release: `git tag v1.0.0`
6. Push tags: `git push --tags`
7. Publish to PyPI (if applicable)

