"""
Unit tests for __init__.py - package imports.
"""

import pytest


def test_package_imports():
    """Test that main package components can be imported."""
    from duplicate_detector import DuplicateDetector, DetectionResults, DetectorConfig, __version__
    
    assert DuplicateDetector is not None
    assert DetectionResults is not None
    assert DetectorConfig is not None
    assert __version__ == "1.0.0"


def test_core_modules_import():
    """Test that core modules can be imported."""
    from duplicate_detector.core import panel_detector
    from duplicate_detector.core import similarity_engine
    from duplicate_detector.core import geometric_verifier
    from duplicate_detector.core import tier_classifier
    
    assert panel_detector is not None
    assert similarity_engine is not None
    assert geometric_verifier is not None
    assert tier_classifier is not None


def test_utils_import():
    """Test that utils can be imported."""
    from duplicate_detector.utils import logger
    
    assert logger is not None


def test_models_import():
    """Test that models can be imported."""
    from duplicate_detector.models import config
    
    assert config is not None

