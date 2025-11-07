"""
Example: Using Extracted Modules

This demonstrates how to use the newly extracted modules
instead of the monolithic file.
"""

from pathlib import Path
from duplicate_detector.core.panel_detector import pdf_to_pages, pages_to_panels_auto
from duplicate_detector.core.similarity_engine import (
    load_clip,
    load_or_compute_embeddings,
    clip_find_duplicates_threshold,
    phash_find_duplicates_with_bundles,
    add_ssim_validation
)
from duplicate_detector.models.config import DetectorConfig
from duplicate_detector.utils.logger import initialize_logging, StageLogger


def example_using_modules():
    """Example of using extracted modules"""
    
    # Initialize logging
    logger = initialize_logging(log_dir=Path("logs"), log_level="INFO")
    
    # Load configuration
    config = DetectorConfig.from_preset("balanced")
    config.pdf_path = Path("document.pdf")
    config.output_dir = Path("output")
    
    # Stage 1: PDF to Pages
    with StageLogger(logger, "PDF Conversion"):
        pages = pdf_to_pages(
            pdf_path=config.pdf_path,
            out_dir=config.output_dir,
            dpi=config.dpi,
            caption_pages=config.panel_detection.caption_pages
        )
    
    # Stage 2: Panel Detection
    with StageLogger(logger, "Panel Detection"):
        panels, meta_df = pages_to_panels_auto(
            pages=pages,
            out_dir=config.output_dir,
            min_panel_area=config.panel_detection.min_panel_area,
            max_panel_area=config.panel_detection.max_panel_area,
            min_aspect_ratio=config.panel_detection.min_aspect_ratio,
            max_aspect_ratio=config.panel_detection.max_aspect_ratio,
            edge_threshold1=config.panel_detection.edge_threshold1,
            edge_threshold2=config.panel_detection.edge_threshold2,
            contour_approx_epsilon=config.panel_detection.contour_approx_epsilon,
            debug_mode=config.feature_flags.debug_mode
        )
    
    # Stage 3: CLIP Embeddings
    with StageLogger(logger, "CLIP Embeddings"):
        clip = load_clip(device="cpu")
        vecs = load_or_compute_embeddings(
            panel_paths=panels,
            clip=clip,
            output_dir=config.output_dir,
            cache_version=config.performance.cache_version,
            enable_cache=config.feature_flags.enable_cache,
            batch_size=config.performance.batch_size
        )
    
    # Stage 4: CLIP Duplicate Detection
    with StageLogger(logger, "CLIP Detection"):
        df_clip = clip_find_duplicates_threshold(
            panel_paths=panels,
            vecs=vecs,
            threshold=config.duplicate_detection.sim_threshold,
            meta_df=meta_df,
            suppress_same_page=False,
            suppress_adjacent_page=False,
            max_output_pairs=config.duplicate_detection.clip_max_output_pairs
        )
    
    # Stage 5: pHash Detection
    if config.feature_flags.use_phash_bundles:
        with StageLogger(logger, "pHash Detection"):
            df_phash = phash_find_duplicates_with_bundles(
                panel_paths=panels,
                max_dist=config.duplicate_detection.phash_max_dist,
                meta_df=meta_df,
                output_dir=config.output_dir,
                cache_version=config.performance.cache_version,
                enable_cache=config.feature_flags.enable_cache
            )
    
    # Stage 6: SSIM Validation
    if config.feature_flags.use_ssim_validation:
        with StageLogger(logger, "SSIM Validation"):
            df_ssim = add_ssim_validation(
                df=df_clip,
                use_patchwise=config.advanced_discrimination.use_patchwise_ssim,
                ssim_threshold=config.duplicate_detection.ssim_threshold,
                patch_min_gate=config.advanced_discrimination.ssim_patch_min_gate
            )
    
    logger.info("Pipeline complete!")
    return df_clip, df_phash if config.feature_flags.use_phash_bundles else None, df_ssim if config.feature_flags.use_ssim_validation else None


if __name__ == "__main__":
    print("Example: Using Extracted Modules")
    print("=" * 50)
    print("This demonstrates the new modular structure.")
    print("Run with actual PDF file to test.")

