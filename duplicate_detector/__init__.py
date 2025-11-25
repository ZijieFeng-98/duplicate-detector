"""
Duplicate Detector - Professional Scientific Figure Duplicate Detection

A comprehensive tool for detecting duplicate panels in scientific PDFs using
multi-stage filtering with CLIP embeddings, pHash, SSIM, and ORB-RANSAC.
"""

__version__ = "1.0.0"

# Main API
from duplicate_detector.api.detector import DuplicateDetector, DetectionResults

# Configuration
from duplicate_detector.models.config import DetectorConfig

__all__ = [
    'DuplicateDetector',
    'DetectionResults',
    'DetectorConfig',
    '__version__'
]
