"""
Configuration Migration Helper

This module provides backward compatibility between the old hardcoded configuration
and the new Pydantic-based configuration system.

Usage:
    from duplicate_detector.models.config import DetectorConfig, apply_config_to_module
    
    config = DetectorConfig.from_preset("balanced")
    apply_config_to_module(config, ai_pdf_panel_duplicate_check_AUTO)
"""

from pathlib import Path
from typing import Any, Optional
import sys
from duplicate_detector.models.config import DetectorConfig


def apply_config_to_module(config: DetectorConfig, target_module: Any) -> None:
    """
    Apply configuration to a module's global variables
    
    This function takes a DetectorConfig instance and applies all its values
    to the target module's global namespace, converting from the new config
    structure to the old flat variable names.
    
    Args:
        config: DetectorConfig instance
        target_module: Module object (e.g., sys.modules['ai_pdf_panel_duplicate_check_AUTO'])
    """
    config_dict = config.model_dump_for_pipeline()
    
    for key, value in config_dict.items():
        setattr(target_module, key, value)


def migrate_config_from_old_module(old_module: Any) -> DetectorConfig:
    """
    Migrate configuration from old module format to new DetectorConfig
    
    This function reads configuration values from the old module format
    and creates a new DetectorConfig instance.
    
    Args:
        old_module: Module object with old-style configuration variables
    
    Returns:
        DetectorConfig instance
    """
    config = DetectorConfig()
    
    # PDF settings
    if hasattr(old_module, 'PDF_PATH'):
        config.pdf_path = Path(old_module.PDF_PATH) if old_module.PDF_PATH else None
    if hasattr(old_module, 'OUT_DIR'):
        config.output_dir = Path(old_module.OUT_DIR) if old_module.OUT_DIR else None
    if hasattr(old_module, 'CAPTION_PAGES'):
        config.panel_detection.caption_pages = old_module.CAPTION_PAGES
    
    # Panel detection
    if hasattr(old_module, 'MIN_PANEL_AREA'):
        config.panel_detection.min_panel_area = old_module.MIN_PANEL_AREA
    if hasattr(old_module, 'MAX_PANEL_AREA'):
        config.panel_detection.max_panel_area = old_module.MAX_PANEL_AREA
    if hasattr(old_module, 'MIN_ASPECT_RATIO'):
        config.panel_detection.min_aspect_ratio = old_module.MIN_ASPECT_RATIO
    if hasattr(old_module, 'MAX_ASPECT_RATIO'):
        config.panel_detection.max_aspect_ratio = old_module.MAX_ASPECT_RATIO
    if hasattr(old_module, 'EDGE_THRESHOLD1'):
        config.panel_detection.edge_threshold1 = old_module.EDGE_THRESHOLD1
    if hasattr(old_module, 'EDGE_THRESHOLD2'):
        config.panel_detection.edge_threshold2 = old_module.EDGE_THRESHOLD2
    if hasattr(old_module, 'CONTOUR_APPROX_EPSILON'):
        config.panel_detection.contour_approx_epsilon = old_module.CONTOUR_APPROX_EPSILON
    
    # Duplicate detection
    if hasattr(old_module, 'SIM_THRESHOLD'):
        config.duplicate_detection.sim_threshold = old_module.SIM_THRESHOLD
    if hasattr(old_module, 'PHASH_MAX_DIST'):
        config.duplicate_detection.phash_max_dist = old_module.PHASH_MAX_DIST
    if hasattr(old_module, 'SSIM_THRESHOLD'):
        config.duplicate_detection.ssim_threshold = old_module.SSIM_THRESHOLD
    if hasattr(old_module, 'TOP_K_NEIGHBORS'):
        config.duplicate_detection.top_k_neighbors = old_module.TOP_K_NEIGHBORS
    if hasattr(old_module, 'CLIP_PAIRING_MODE'):
        config.duplicate_detection.clip_pairing_mode = old_module.CLIP_PAIRING_MODE
    if hasattr(old_module, 'CLIP_MAX_OUTPUT_PAIRS'):
        config.duplicate_detection.clip_max_output_pairs = old_module.CLIP_MAX_OUTPUT_PAIRS
    
    # Feature flags
    if hasattr(old_module, 'USE_PHASH_BUNDLES'):
        config.feature_flags.use_phash_bundles = old_module.USE_PHASH_BUNDLES
    if hasattr(old_module, 'USE_ORB_RANSAC'):
        config.feature_flags.use_orb_ransac = old_module.USE_ORB_RANSAC
    if hasattr(old_module, 'USE_TIER_GATING'):
        config.feature_flags.use_tier_gating = old_module.USE_TIER_GATING
    if hasattr(old_module, 'USE_SSIM_VALIDATION'):
        config.feature_flags.use_ssim_validation = old_module.USE_SSIM_VALIDATION
    if hasattr(old_module, 'ENABLE_CACHE'):
        config.feature_flags.enable_cache = old_module.ENABLE_CACHE
    if hasattr(old_module, 'DEBUG_MODE'):
        config.feature_flags.debug_mode = old_module.DEBUG_MODE
    
    # Performance
    if hasattr(old_module, 'BATCH_SIZE'):
        config.performance.batch_size = old_module.BATCH_SIZE
    if hasattr(old_module, 'NUM_WORKERS'):
        config.performance.num_workers = old_module.NUM_WORKERS
    if hasattr(old_module, 'RANDOM_SEED'):
        config.performance.random_seed = old_module.RANDOM_SEED
    
    return config

