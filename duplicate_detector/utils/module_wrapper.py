"""
Wrapper to use extracted modules in the main pipeline.

This file provides backward compatibility by allowing the main pipeline
to optionally use the extracted modules while maintaining the existing API.
"""

import warnings
from pathlib import Path
from typing import Optional

# Try to import extracted modules
try:
    from duplicate_detector.core.panel_detector import (
        pdf_to_pages as pdf_to_pages_module,
        pages_to_panels_auto as pages_to_panels_auto_module
    )
    from duplicate_detector.core.similarity_engine import (
        load_clip as load_clip_module,
        load_or_compute_embeddings as load_or_compute_embeddings_module,
        clip_find_duplicates_threshold as clip_find_duplicates_threshold_module,
        phash_find_duplicates_with_bundles as phash_find_duplicates_with_bundles_module,
        add_ssim_validation as add_ssim_validation_module
    )
    from duplicate_detector.core.geometric_verifier import (
        orb_find_partial_duplicates as orb_find_partial_duplicates_module
    )
    from duplicate_detector.core.tier_classifier import (
        apply_tier_gating as apply_tier_gating_module
    )
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    warnings.warn(
        "Extracted modules not available. Using legacy functions. "
        "Install duplicate_detector package for improved performance.",
        ImportWarning
    )


def use_extracted_modules(enable: bool = True):
    """
    Enable or disable use of extracted modules.
    
    Args:
        enable: If True, use extracted modules when available.
                If False, always use legacy functions.
    """
    global USE_EXTRACTED_MODULES
    USE_EXTRACTED_MODULES = enable and MODULES_AVAILABLE


# Default: Use modules if available
USE_EXTRACTED_MODULES = MODULES_AVAILABLE


# Wrapper functions that delegate to modules or legacy code
def pdf_to_pages_wrapper(pdf_path: Path, out_dir: Path, dpi: int, caption_pages: set, debug_mode: bool = False):
    """Wrapper for pdf_to_pages that uses modules if available."""
    if USE_EXTRACTED_MODULES:
        return pdf_to_pages_module(
            pdf_path=pdf_path,
            out_dir=out_dir,
            dpi=dpi,
            caption_pages=caption_pages,
            debug_mode=debug_mode
        )
    else:
        # Fall back to legacy function (imported from main file)
        from ai_pdf_panel_duplicate_check_AUTO import pdf_to_pages as pdf_to_pages_legacy
        return pdf_to_pages_legacy(pdf_path, out_dir, dpi)


def pages_to_panels_auto_wrapper(pages, out_dir: Path, **kwargs):
    """Wrapper for pages_to_panels_auto that uses modules if available."""
    if USE_EXTRACTED_MODULES:
        return pages_to_panels_auto_module(
            pages=pages,
            out_dir=out_dir,
            min_panel_area=kwargs.get('min_panel_area', 80000),
            max_panel_area=kwargs.get('max_panel_area', 10000000),
            min_aspect_ratio=kwargs.get('min_aspect_ratio', 0.2),
            max_aspect_ratio=kwargs.get('max_aspect_ratio', 5.0),
            edge_threshold1=kwargs.get('edge_threshold1', 40),
            edge_threshold2=kwargs.get('edge_threshold2', 120),
            contour_approx_epsilon=kwargs.get('contour_approx_epsilon', 0.02),
            debug_mode=kwargs.get('debug_mode', False)
        )
    else:
        from ai_pdf_panel_duplicate_check_AUTO import pages_to_panels_auto as pages_to_panels_auto_legacy
        return pages_to_panels_auto_legacy(pages, out_dir)

