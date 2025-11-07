# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added
- **Modular Architecture**: Refactored 5,430-line monolithic file into focused modules
  - `panel_detector`: PDF to pages conversion and panel detection
  - `similarity_engine`: CLIP embeddings, pHash bundles, SSIM computation
  - `geometric_verifier`: ORB-RANSAC for partial duplicate detection
  - `tier_classifier`: Tier A/B classification with multiple discrimination paths

- **Configuration Management**: Comprehensive Pydantic-based config system
  - YAML/JSON config file support
  - Environment variable support
  - Preset configurations (fast/balanced/thorough)
  - No hardcoded paths

- **Clean Python API**: High-level `DuplicateDetector` class
  - Simple usage: `DuplicateDetector(config=DetectorConfig.from_preset("balanced"))`
  - Advanced usage with full configuration control
  - `DetectionResults` class for structured results

- **Structured Logging**: Professional logging system
  - File rotation, colored output
  - Stage timing context manager
  - Print-to-logger redirect

- **Streamlit Integration**: Updated UI to use new config system
  - Preset loading from config
  - Backward compatible fallback

### Changed
- Removed all hardcoded paths from main pipeline
- Updated CLI to support config files and presets
- Improved code organization and maintainability

### Technical Details
- **Type Hints**: 100% coverage across all modules
- **Docstrings**: Comprehensive documentation (Google style)
- **Code Quality**: Modular, testable, reusable components
- **Package Structure**: Proper Python package with `pyproject.toml`

## [Unreleased]

### Planned
- Comprehensive test suite (unit, integration, validation, performance)
- CI/CD pipeline (GitHub Actions)
- API documentation (Sphinx/MkDocs)
- Docker support
- REST API (FastAPI)
- Benchmark dataset (50-100 annotated pairs)
- Research paper materials

[1.0.0]: https://github.com/yourusername/duplicate-detector/releases/tag/v1.0.0

