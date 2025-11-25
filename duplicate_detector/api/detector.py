"""
Clean Python API for Duplicate Detector

Provides a simple, high-level interface for duplicate detection.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from duplicate_detector.models.config import DetectorConfig
from duplicate_detector.core.panel_detector import pdf_to_pages, pages_to_panels_auto
from duplicate_detector.core.similarity_engine import (
    load_clip,
    load_or_compute_embeddings,
    clip_find_duplicates_threshold,
    phash_find_duplicates_with_bundles,
    add_ssim_validation
)
from duplicate_detector.core.geometric_verifier import orb_find_partial_duplicates
from duplicate_detector.core.tier_classifier import apply_tier_gating
from duplicate_detector.core.feature_scanner import whole_page_feature_scan
from duplicate_detector.utils.logger import initialize_logging, StageLogger
from duplicate_detector.utils.classifier import load_pair_classifier


class DuplicateDetector:
    """
    High-level API for duplicate detection pipeline.
    
    Example:
        ```python
        from duplicate_detector import DuplicateDetector, DetectorConfig
        
        # Simple usage
        detector = DuplicateDetector(config=DetectorConfig.from_preset("balanced"))
        results = detector.analyze_pdf("paper.pdf")
        
        # Advanced usage
        config = DetectorConfig(
            pdf_path=Path("paper.pdf"),
            output_dir=Path("results"),
            dpi=150,
            duplicate_detection=DuplicateDetectionConfig(sim_threshold=0.96)
        )
        detector = DuplicateDetector(config=config)
        results = detector.analyze_pdf()
        
        print(f"Found {results.total_pairs} duplicate pairs")
        for pair in results.tier_a_pairs:
            print(f"{pair.image_a} vs {pair.image_b}: {pair.clip_score:.3f}")
        ```
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None, **kwargs):
        """
        Initialize DuplicateDetector.
        
        Args:
            config: DetectorConfig instance. If None, creates default config.
            **kwargs: Additional config overrides (e.g., pdf_path, output_dir, sim_threshold)
        """
        if config is None:
            config = DetectorConfig()
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        self.logger = initialize_logging(
            log_dir=config.output_dir / "logs" if config.output_dir else Path("logs"),
            log_level="INFO"
        )
        self.clip_model = None
        self.panels = []
        self.meta_df = pd.DataFrame()
        self.clip_pairs = pd.DataFrame()
        self.phash_pairs = pd.DataFrame()
        self.final_pairs = pd.DataFrame()
        self.classifier = None
        if config.feature_flags.use_classifier_gating and config.classifier.enabled:
            self.classifier = load_pair_classifier(
                config.classifier.model_path,
                config.classifier.scaler_path,
                config.classifier.threshold,
            )
    
    def analyze_pdf(
        self,
        pdf_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> 'DetectionResults':
        """
        Analyze PDF for duplicate panels.
        
        Args:
            pdf_path: Path to PDF file. If None, uses config.pdf_path.
            output_dir: Output directory. If None, uses config.output_dir.
        
        Returns:
            DetectionResults object with duplicate pairs and metadata.
        """
        # Update paths
        if pdf_path:
            self.config.pdf_path = Path(pdf_path)
        if output_dir:
            self.config.output_dir = Path(output_dir)
        
        if self.config.pdf_path is None:
            raise ValueError("PDF path must be provided via config or argument")
        
        if self.config.output_dir is None:
            self.config.output_dir = Path.cwd() / "duplicate_detector_output"
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: PDF to Pages
        with StageLogger(self.logger, "PDF Conversion"):
            pages = pdf_to_pages(
                pdf_path=self.config.pdf_path,
                out_dir=self.config.output_dir,
                dpi=self.config.dpi,
                caption_pages=self.config.panel_detection.caption_pages or set(),
                debug_mode=self.config.feature_flags.debug_mode
            )
        
        # Stage 2: Panel Detection
        with StageLogger(self.logger, "Panel Detection"):
            self.panels, self.meta_df = pages_to_panels_auto(
                pages=pages,
                out_dir=self.config.output_dir,
                min_panel_area=self.config.panel_detection.min_panel_area,
                max_panel_area=self.config.panel_detection.max_panel_area,
                min_aspect_ratio=self.config.panel_detection.min_aspect_ratio,
                max_aspect_ratio=self.config.panel_detection.max_aspect_ratio,
                edge_threshold1=self.config.panel_detection.edge_threshold1,
                edge_threshold2=self.config.panel_detection.edge_threshold2,
                contour_approx_epsilon=self.config.panel_detection.contour_approx_epsilon,
                debug_mode=self.config.feature_flags.debug_mode
            )
        
        if len(self.panels) == 0:
            self.logger.warning("No panels detected")
            return DetectionResults(
                total_pairs=0,
                tier_a_pairs=[],
                tier_b_pairs=[],
                all_pairs=pd.DataFrame(),
                metadata={}
            )
        
        # Stage 3: CLIP Embeddings
        with StageLogger(self.logger, "CLIP Embeddings"):
            device = self.config.performance.device or "cpu"
            self.clip_model = load_clip(device=device)
            vecs = load_or_compute_embeddings(
                panel_paths=self.panels,
                clip=self.clip_model,
                output_dir=self.config.output_dir,
                cache_version=self.config.performance.cache_version,
                enable_cache=self.config.feature_flags.enable_cache,
                batch_size=self.config.performance.batch_size
            )
        
        # Stage 4: CLIP Duplicate Detection
        with StageLogger(self.logger, "CLIP Detection"):
            df_clip = clip_find_duplicates_threshold(
                panel_paths=self.panels,
                vecs=vecs,
                threshold=self.config.duplicate_detection.sim_threshold,
                meta_df=self.meta_df,
                suppress_same_page=False,
                suppress_adjacent_page=False,
                max_output_pairs=self.config.duplicate_detection.clip_max_output_pairs
            )
        self.clip_pairs = df_clip
        
        # Stage 5: pHash Detection
        df_phash = pd.DataFrame()
        if self.config.feature_flags.use_phash_bundles:
            with StageLogger(self.logger, "pHash Detection"):
                df_phash = phash_find_duplicates_with_bundles(
                    panel_paths=self.panels,
                    max_dist=self.config.duplicate_detection.phash_max_dist,
                    meta_df=self.meta_df,
                    output_dir=self.config.output_dir,
                    cache_version=self.config.performance.cache_version,
                    enable_cache=self.config.feature_flags.enable_cache,
                    num_workers=self.config.performance.num_workers,
                    suppress_same_page=False,
                    suppress_adjacent_page=False,
                    bundle_short_circuit=self.config.duplicate_detection.phash_bundle_short_circuit
                )
        self.phash_pairs = df_phash
        
        # Stage 6: SSIM Validation
        df_ssim = df_clip.copy()
        if self.config.feature_flags.use_ssim_validation:
            with StageLogger(self.logger, "SSIM Validation"):
                df_ssim = add_ssim_validation(
                    df=df_clip,
                    use_patchwise=self.config.advanced_discrimination.use_patchwise_ssim,
                    ssim_threshold=self.config.duplicate_detection.ssim_threshold,
                    patch_min_gate=self.config.advanced_discrimination.ssim_patch_min_gate
                )
        
        # Stage 7: ORB-RANSAC (if enabled)
        df_orb = pd.DataFrame()
        if self.config.feature_flags.use_orb_ransac:
            with StageLogger(self.logger, "ORB-RANSAC"):
                df_orb = orb_find_partial_duplicates(
                    panel_paths=self.panels,
                    clip_df=df_clip,
                    phash_df=df_phash,
                    meta_df=self.meta_df,
                    output_dir=self.config.output_dir,
                    cache_version=self.config.performance.cache_version,
                    orb_trigger_clip_threshold=self.config.advanced_discrimination.orb_trigger_clip_threshold,
                    orb_trigger_phash_threshold=self.config.advanced_discrimination.orb_trigger_phash_threshold,
                    min_inliers=self.config.advanced_discrimination.tier_a_orb_inliers,
                    min_inlier_ratio=self.config.advanced_discrimination.tier_a_orb_ratio,
                    max_reproj_error=self.config.advanced_discrimination.tier_a_orb_error,
                    min_coverage=self.config.advanced_discrimination.tier_a_orb_coverage,
                    enable_cache=self.config.feature_flags.enable_cache
                )
        
        # Stage 7b: Whole-Page Feature Scan (Hybrid Mode)
        df_whole_page = pd.DataFrame()
        if self.config.feature_flags.enable_figcheck_heuristics: # Piggyback on 'thorough' flag or add new one
             with StageLogger(self.logger, "Whole-Page Scan"):
                 df_whole_page = whole_page_feature_scan(
                     pages=pages,
                     out_dir=self.config.output_dir,
                     min_inliers=20 # Hardcoded safe threshold
                 )

        # Merge results
        df_merged = self._merge_results(df_clip, df_phash, df_orb, df_ssim)
        
        # Merge whole-page results if any
        if not df_whole_page.empty:
            # Only add pairs not already found
            if not df_merged.empty:
                existing_keys = set(zip(df_merged['Image_A'], df_merged['Image_B']))
                new_rows = []
                for _, row in df_whole_page.iterrows():
                    if (row['Image_A'], row['Image_B']) not in existing_keys:
                        new_rows.append(row)
                if new_rows:
                    df_merged = pd.concat([df_merged, pd.DataFrame(new_rows)], ignore_index=True)
            else:
                df_merged = df_whole_page

        classifier_scores = None
        if self.classifier is not None and not df_merged.empty:
            classifier_scores = self.classifier.score(df_merged)
            df_merged = df_merged.assign(Classifier_Score=classifier_scores)
        
        # Stage 8: Tier Classification
        with StageLogger(self.logger, "Tier Classification"):
            tier_cfg = self.config.tier_classification
            df_final = apply_tier_gating(
                df=df_merged,
                tier_a_phash=tier_cfg.tier_a_phash_rt,
                tier_a_clip=tier_cfg.tier_a_clip,
                tier_a_ssim=tier_cfg.tier_a_ssim,
                tier_b_phash_min=tier_cfg.tier_b_phash_rt_min,
                tier_b_phash_max=tier_cfg.tier_b_phash_rt_max,
                tier_b_clip_min=tier_cfg.tier_b_clip_min,
                tier_b_clip_max=tier_cfg.tier_b_clip_max,
                tier_b_ssim_min=tier_cfg.tier_b_ssim_min,
                tier_b_ssim_max=tier_cfg.tier_b_ssim_max,
                tier_a_orb_inliers=tier_cfg.tier_a_orb_inliers,
                tier_a_orb_ratio=tier_cfg.tier_a_orb_ratio,
                tier_a_orb_error=tier_cfg.tier_a_orb_error,
                tier_a_orb_coverage=tier_cfg.tier_a_orb_coverage,
                use_modality_specific=self.config.feature_flags.use_modality_specific_gating
            )
        self.final_pairs = df_final

        if self.classifier is not None and classifier_scores is not None:
            positive_idx = classifier_scores[self.classifier.apply_threshold(classifier_scores)].index
            if len(positive_idx) > 0:
                classifier_df = df_merged.loc[positive_idx].copy()
                classifier_df["Tier"] = classifier_df.get("Tier", "").replace("", "Classifier")
                classifier_df["Tier_Path"] = classifier_df.get("Tier_Path", "").replace("", "Classifier")
                if "Confocal_FP" not in classifier_df.columns:
                    classifier_df["Confocal_FP"] = False
                
                # Align columns with final pairs DataFrame
                if not self.final_pairs.empty:
                    for col in self.final_pairs.columns:
                        if col not in classifier_df.columns:
                            classifier_df[col] = None
                    classifier_df = classifier_df[self.final_pairs.columns]
                    combined = pd.concat([self.final_pairs, classifier_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["Path_A", "Path_B"], keep="first")
                    self.final_pairs = combined
                else:
                    classifier_df["Tier"] = classifier_df["Tier"].replace("", "Classifier")
                    self.final_pairs = classifier_df[self.final_pairs.columns] if not self.final_pairs.empty else classifier_df
        
        # Create results object
        tier_a_pairs = self.final_pairs[self.final_pairs['Tier'] == 'A'].to_dict('records')
        tier_b_pairs = self.final_pairs[self.final_pairs['Tier'] == 'B'].to_dict('records')
        
        return DetectionResults(
            total_pairs=len(self.final_pairs),
            tier_a_pairs=tier_a_pairs,
            tier_b_pairs=tier_b_pairs,
            all_pairs=self.final_pairs,
            metadata={
                'num_panels': len(self.panels),
                'num_pages': len(pages),
                'config': self.config.model_dump()
            }
        )
    
    def _merge_results(
        self,
        df_clip: pd.DataFrame,
        df_phash: pd.DataFrame,
        df_orb: pd.DataFrame,
        df_ssim: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge CLIP, pHash, ORB, and SSIM results"""
        def pair_key(a, b):
            return tuple(sorted([a, b]))
        
        all_pairs = {}
        
        # Add CLIP pairs
        for _, row in df_clip.iterrows():
            key = pair_key(row['Path_A'], row['Path_B'])
            all_pairs[key] = {
                'Image_A': row['Image_A'],
                'Image_B': row['Image_B'],
                'Path_A': row['Path_A'],
                'Path_B': row['Path_B'],
                'Cosine_Similarity': row.get('Cosine_Similarity', ''),
                'SSIM': row.get('SSIM', ''),
                'Hamming_Distance': '',
                'ORB_Inliers': '',
                'Source': 'CLIP'
            }
        
        # Add pHash pairs
        for _, row in df_phash.iterrows():
            key = pair_key(row['Path_A'], row['Path_B'])
            if key in all_pairs:
                all_pairs[key]['Hamming_Distance'] = row['Hamming_Distance']
                all_pairs[key]['Source'] = 'CLIP+pHash'
            else:
                all_pairs[key] = {
                    'Image_A': row['Image_A'],
                    'Image_B': row['Image_B'],
                    'Path_A': row['Path_A'],
                    'Path_B': row['Path_B'],
                    'Cosine_Similarity': '',
                    'SSIM': '',
                    'Hamming_Distance': row['Hamming_Distance'],
                    'ORB_Inliers': '',
                    'Source': 'pHash'
                }
        
        # Add ORB pairs
        for _, row in df_orb.iterrows():
            key = pair_key(row['Path_A'], row['Path_B'])
            if key in all_pairs:
                all_pairs[key]['ORB_Inliers'] = row.get('ORB_Inliers', '')
            else:
                all_pairs[key] = {
                    'Image_A': row['Image_A'],
                    'Image_B': row['Image_B'],
                    'Path_A': row['Path_A'],
                    'Path_B': row['Path_B'],
                    'Cosine_Similarity': '',
                    'SSIM': '',
                    'Hamming_Distance': '',
                    'ORB_Inliers': row.get('ORB_Inliers', ''),
                    'Source': 'ORB'
                }
        
        # Merge SSIM data
        for _, row in df_ssim.iterrows():
            key = pair_key(row['Path_A'], row['Path_B'])
            if key in all_pairs:
                all_pairs[key]['SSIM'] = row.get('SSIM', '')
                all_pairs[key]['Patch_SSIM_Min'] = row.get('Patch_SSIM_Min', '')
                all_pairs[key]['Patch_SSIM_TopK'] = row.get('Patch_SSIM_TopK', '')
        
        return pd.DataFrame(list(all_pairs.values()))


class DetectionResults:
    """
    Results from duplicate detection analysis.
    
    Attributes:
        total_pairs: Total number of duplicate pairs found
        tier_a_pairs: List of Tier A (high confidence) pairs
        tier_b_pairs: List of Tier B (manual review) pairs
        all_pairs: DataFrame with all pairs and metadata
        metadata: Additional metadata about the analysis
    """
    
    def __init__(
        self,
        total_pairs: int,
        tier_a_pairs: List[Dict],
        tier_b_pairs: List[Dict],
        all_pairs: pd.DataFrame,
        metadata: Dict[str, Any]
    ):
        self.total_pairs = total_pairs
        self.tier_a_pairs = tier_a_pairs
        self.tier_b_pairs = tier_b_pairs
        self.all_pairs = all_pairs
        self.metadata = metadata
    
    def __repr__(self):
        return (
            f"DetectionResults(total_pairs={self.total_pairs}, "
            f"tier_a={len(self.tier_a_pairs)}, tier_b={len(self.tier_b_pairs)})"
        )
    
    def save(self, output_path: Path):
        """Save results to CSV file"""
        self.all_pairs.to_csv(output_path, index=False)
    
    def get_tier_a_count(self) -> int:
        """Get count of Tier A pairs"""
        return len(self.tier_a_pairs)
    
    def get_tier_b_count(self) -> int:
        """Get count of Tier B pairs"""
        return len(self.tier_b_pairs)

