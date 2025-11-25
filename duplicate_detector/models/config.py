"""
Configuration Management for Duplicate Detector

This module provides centralized configuration management with:
- Pydantic models for validation
- Environment variable support
- YAML/JSON config file support
- Preset profiles (fast/balanced/thorough)
- No hardcoded paths
"""

from pathlib import Path
from typing import Optional, Set, Dict, List, Literal
from dataclasses import dataclass
import os
import json
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class PanelDetectionConfig(BaseModel):
    """Configuration for panel detection from PDF pages"""
    
    min_panel_area: int = Field(80000, ge=1000, le=100000000, description="Minimum panel area in pixels")
    max_panel_area: int = Field(10000000, ge=10000, le=100000000, description="Maximum panel area in pixels")
    min_aspect_ratio: float = Field(0.2, ge=0.1, le=10.0, description="Minimum aspect ratio (width/height)")
    max_aspect_ratio: float = Field(5.0, ge=0.1, le=10.0, description="Maximum aspect ratio (width/height)")
    edge_threshold1: int = Field(40, ge=10, le=200, description="Canny edge detection threshold 1")
    edge_threshold2: int = Field(120, ge=50, le=300, description="Canny edge detection threshold 2")
    contour_approx_epsilon: float = Field(0.02, ge=0.001, le=0.1, description="Contour approximation epsilon")
    caption_pages: Set[int] = Field(default_factory=set, description="Page numbers (1-indexed) to exclude")


class DuplicateDetectionConfig(BaseModel):
    """Configuration for duplicate detection thresholds"""
    
    sim_threshold: float = Field(0.85, ge=0.0, le=1.0, description="CLIP similarity threshold")
    phash_max_dist: int = Field(3, ge=0, le=64, description="pHash maximum Hamming distance")
    ssim_threshold: float = Field(0.37, ge=0.0, le=1.0, description="SSIM structural similarity threshold")
    top_k_neighbors: int = Field(50, ge=1, le=1000, description="Top-K neighbors for CLIP search")
    clip_pairing_mode: Literal["topk", "thresh"] = Field("thresh", description="CLIP pairing mode")
    clip_max_output_pairs: int = Field(120000, ge=100, le=1000000, description="Maximum CLIP output pairs")
    phash_bundle_short_circuit: int = Field(3, ge=1, le=64, description="Stop pHash bundle search after this distance")


class AdvancedDiscriminationConfig(BaseModel):
    """Advanced discrimination parameters"""
    
    # CLIP z-score filtering
    use_clip_zscore: bool = Field(True, description="Enable CLIP z-score filtering")
    clip_zscore_min: float = Field(3.0, ge=0.0, le=10.0, description="Minimum CLIP z-score")
    require_clip_z_for_clip_ssim: bool = Field(True, description="Require z-score for CLIP-SSIM path")
    
    # Patch-wise SSIM
    use_patchwise_ssim: bool = Field(True, description="Enable patch-wise SSIM")
    ssim_grid_h: int = Field(3, ge=1, le=10, description="SSIM grid rows")
    ssim_grid_w: int = Field(3, ge=1, le=10, description="SSIM grid columns")
    ssim_topk_patches: int = Field(4, ge=1, le=9, description="Top-K patches for SSIM")
    ssim_mix_weight: float = Field(0.6, ge=0.0, le=1.0, description="SSIM mix weight (patch vs global)")
    ssim_patch_min_gate: float = Field(0.85, ge=0.0, le=1.0, description="Minimum patch SSIM threshold")
    require_patch_min_for_clip_ssim: bool = Field(True, description="Require patch min for CLIP-SSIM")
    
    # Copy-paste detection
    tier_a_copypaste_clip_min: float = Field(0.95, ge=0.0, le=1.0)
    tier_a_copypaste_ssim_min: float = Field(0.65, ge=0.0, le=1.0)
    tier_a_copypaste_combined: float = Field(1.65, ge=0.0, le=2.0)

    # ORB trigger thresholds
    orb_trigger_clip_threshold: float = Field(0.985, ge=0.0, le=1.0, description="Minimum CLIP score to trigger ORB")
    orb_trigger_phash_threshold: int = Field(4, ge=0, le=64, description="Maximum pHash distance to trigger ORB")
    tier_a_orb_inliers: int = Field(30, ge=0, le=1000, description="Minimum ORB inliers for Tier A gating")
    tier_a_orb_ratio: float = Field(0.30, ge=0.0, le=1.0, description="Minimum ORB match ratio")
    tier_a_orb_error: float = Field(4.0, ge=0.0, le=20.0, description="Maximum ORB reprojection error")
    tier_a_orb_coverage: float = Field(0.85, ge=0.0, le=1.0, description="Minimum ORB coverage for Tier A")


class TierClassificationConfig(BaseModel):
    """Tier A/B classification thresholds"""
    
    # Tier A thresholds
    tier_a_phash_rt: int = Field(3, ge=0, le=64)
    tier_a_clip: float = Field(0.99, ge=0.0, le=1.0)
    tier_a_ssim: float = Field(0.95, ge=0.0, le=1.0)
    tier_a_orb_inliers: int = Field(30, ge=0, le=1000)
    tier_a_orb_ratio: float = Field(0.30, ge=0.0, le=1.0)
    tier_a_orb_error: float = Field(4.0, ge=0.0, le=20.0)
    tier_a_orb_coverage: float = Field(0.85, ge=0.0, le=1.0)
    tier_a_clone_area_pct: float = Field(1.0, ge=0.0, le=1.0)
    
    # Tier B thresholds
    tier_b_phash_rt_min: int = Field(4, ge=0, le=64)
    tier_b_phash_rt_max: int = Field(5, ge=0, le=64)
    tier_b_clip_min: float = Field(0.985, ge=0.0, le=1.0)
    tier_b_clip_max: float = Field(0.99, ge=0.0, le=1.0)
    tier_b_ssim_min: float = Field(0.92, ge=0.0, le=1.0)
    tier_b_ssim_max: float = Field(0.95, ge=0.0, le=1.0)
    
    # Universal discrimination
    tier_a_relaxed_clip: float = Field(0.94, ge=0.0, le=1.0)
    tier_a_relaxed_ssim: float = Field(0.65, ge=0.0, le=1.0)
    tier_a_relaxed_combined: float = Field(1.62, ge=0.0, le=2.0)
    
    # Western blot specific
    tier_a_western_clip: float = Field(0.95, ge=0.0, le=1.0)
    tier_a_western_ssim: float = Field(0.65, ge=0.0, le=1.0)
    tier_a_western_combined: float = Field(1.6, ge=0.0, le=2.0)
    
    # Confocal false positive filter
    confocal_fp_clip_min: float = Field(0.96, ge=0.0, le=1.0)
    confocal_fp_ssim_max: float = Field(0.6, ge=0.0, le=1.0)
    confocal_fp_phash_min: int = Field(10, ge=0, le=64)


class FeatureFlagsConfig(BaseModel):
    """Feature flags for enabling/disabling features"""
    
    use_phash_bundles: bool = Field(True, description="Enable rotation/mirror-robust pHash")
    use_orb_ransac: bool = Field(True, description="Enable ORB-RANSAC partial duplicate detection")
    use_tier_gating: bool = Field(True, description="Enable Tier A/B classification")
    use_ssim_validation: bool = Field(True, description="Enable SSIM validation")
    highlight_differences: bool = Field(True, description="Highlight visual differences")
    use_modality_specific_gating: bool = Field(False, description="Use modality-specific tier gating")
    enable_modality_detection: bool = Field(False, description="Pre-classify image types")
    enable_modality_routing: bool = Field(True, description="Internal modality routing")
    expose_modality_columns: bool = Field(False, description="Include Modality_A/B in TSV")
    enable_figcheck_heuristics: bool = Field(False, description="Enable FigCheck-inspired heuristics")
    enable_orb_relax: bool = Field(False, description="Enable relaxed ORB detection")
    enable_confocal_deep_verify: bool = Field(True, description="Enable confocal deep verification")
    enable_ihc_deep_verify: bool = Field(True, description="Enable IHC deep verification")
    enable_text_masking: bool = Field(True, description="Enable text masking")
    enable_cache: bool = Field(True, description="Enable embedding cache")
    debug_mode: bool = Field(False, description="Enable debug mode")
    use_classifier_gating: bool = Field(False, description="Leverage logistic classifier probabilities for gating")


class ClassifierConfig(BaseModel):
    """Configuration for logistic classifier gating."""

    enabled: bool = Field(False, description="Enable classifier-based gating")
    model_path: Optional[Path] = Field(None, description="Path to joblib model file")
    scaler_path: Optional[Path] = Field(None, description="Path to joblib scaler file")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Probability threshold for positive classification")


class PerformanceConfig(BaseModel):
    """Performance tuning parameters"""
    
    batch_size: int = Field(64, ge=1, le=512, description="CLIP batch size")
    num_workers: Optional[int] = Field(None, ge=0, le=32, description="Number of worker processes (None = auto)")
    enable_mps: bool = Field(True, description="Enable Metal Performance Shaders (M1/M2 Mac)")
    cache_version: str = Field("v7", description="Cache version (increment when params change)")
    random_seed: int = Field(123, ge=0, description="Random seed for reproducibility")
    device: Optional[str] = Field("cpu", description="Preferred device identifier (cpu, cuda, mps)")


class DetectorConfig(BaseModel):
    """Main configuration class for duplicate detector"""
    
    # Paths (from environment variables or defaults)
    pdf_path: Optional[Path] = Field(None, description="Path to input PDF (from env or CLI)")
    output_dir: Optional[Path] = Field(None, description="Output directory (from env or CLI)")
    
    # PDF conversion
    dpi: int = Field(150, ge=100, le=300, description="DPI for PDF rendering")
    
    # Sub-configurations
    panel_detection: PanelDetectionConfig = Field(default_factory=PanelDetectionConfig)
    duplicate_detection: DuplicateDetectionConfig = Field(default_factory=DuplicateDetectionConfig)
    advanced_discrimination: AdvancedDiscriminationConfig = Field(default_factory=AdvancedDiscriminationConfig)
    tier_classification: TierClassificationConfig = Field(default_factory=TierClassificationConfig)
    feature_flags: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    
    # UX settings
    auto_open_results: bool = Field(True, description="Auto-open results HTML")
    auto_open_per_pair: bool = Field(False, description="Auto-open per-pair artifacts")
    
    @field_validator('pdf_path', 'output_dir', mode='before')
    @classmethod
    def validate_paths(cls, v):
        """Convert string paths to Path objects"""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator('classifier', mode='after')
    @classmethod
    def validate_classifier(cls, value: ClassifierConfig):
        """Ensure classifier paths are Path objects."""
        if value.model_path and isinstance(value.model_path, str):
            value.model_path = Path(value.model_path)
        if value.scaler_path and isinstance(value.scaler_path, str):
            value.scaler_path = Path(value.scaler_path)
        return value
    
    @model_validator(mode='after')
    def load_from_environment(self):
        """Load paths from environment variables if not set"""
        if self.pdf_path is None:
            pdf_env = os.getenv('DUPLICATE_DETECTOR_PDF_PATH')
            if pdf_env:
                self.pdf_path = Path(pdf_env)
        
        if self.output_dir is None:
            output_env = os.getenv('DUPLICATE_DETECTOR_OUTPUT_DIR')
            if output_env:
                self.output_dir = Path(output_env)
            else:
                # Default to current directory
                self.output_dir = Path.cwd() / "duplicate_detector_output"
        
        return self
    
    @classmethod
    def from_preset(cls, preset: Literal["fast", "balanced", "thorough", "dr_zhong"] = "balanced") -> "DetectorConfig":
        """Create configuration from preset profile"""
        config = cls()
        
        if preset == "fast":
            config.dpi = 100
            config.duplicate_detection.sim_threshold = 0.97
            config.duplicate_detection.phash_max_dist = 3
            config.duplicate_detection.ssim_threshold = 0.90
            config.performance.batch_size = 32
            config.feature_flags.use_orb_ransac = False  # Skip ORB for speed
            
        elif preset == "balanced":
            config.panel_detection.min_panel_area = 25000
            config.duplicate_detection.sim_threshold = 0.85
            config.duplicate_detection.phash_max_dist = 4
            config.duplicate_detection.ssim_threshold = 0.40
            # Revert Tier B defaults to standard range
            config.tier_classification.tier_b_clip_min = 0.90
            config.tier_classification.tier_b_ssim_min = 0.30
            
        elif preset == "dr_zhong":
            # Hyper-sensitive preset for difficult datasets like Dr. Zhong
            config.panel_detection.min_panel_area = 15000
            config.duplicate_detection.sim_threshold = 0.75
            config.duplicate_detection.ssim_threshold = 0.28
            config.tier_classification.tier_b_clip_min = 0.60
            config.tier_classification.tier_b_clip_max = 0.95
            config.tier_classification.tier_b_ssim_min = 0.15
            config.tier_classification.tier_b_ssim_max = 0.95
            config.tier_classification.tier_b_phash_rt_min = 0
            config.tier_classification.tier_b_phash_rt_max = 40
            config.feature_flags.use_classifier_gating = True
            
        elif preset == "thorough":
            config.dpi = 200
            config.duplicate_detection.sim_threshold = 0.94
            config.duplicate_detection.phash_max_dist = 5
            config.duplicate_detection.ssim_threshold = 0.85
            config.performance.batch_size = 64
            config.feature_flags.use_orb_ransac = True
            config.feature_flags.enable_figcheck_heuristics = True
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DetectorConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_path: Path) -> "DetectorConfig":
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, json_path: Path) -> None:
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.model_dump(mode='json'), f, indent=2)
    
    def model_dump_for_pipeline(self) -> dict:
        """Convert to dictionary compatible with existing pipeline code"""
        """This method converts the Pydantic model to a flat dict matching old variable names"""
        result = {}
        
        # PDF settings
        result['DPI'] = self.dpi
        result['PDF_PATH'] = self.pdf_path
        result['OUT_DIR'] = self.output_dir
        result['CAPTION_PAGES'] = self.panel_detection.caption_pages
        
        # Panel detection
        result['MIN_PANEL_AREA'] = self.panel_detection.min_panel_area
        result['MAX_PANEL_AREA'] = self.panel_detection.max_panel_area
        result['MIN_ASPECT_RATIO'] = self.panel_detection.min_aspect_ratio
        result['MAX_ASPECT_RATIO'] = self.panel_detection.max_aspect_ratio
        result['EDGE_THRESHOLD1'] = self.panel_detection.edge_threshold1
        result['EDGE_THRESHOLD2'] = self.panel_detection.edge_threshold2
        result['CONTOUR_APPROX_EPSILON'] = self.panel_detection.contour_approx_epsilon
        
        # Duplicate detection
        result['SIM_THRESHOLD'] = self.duplicate_detection.sim_threshold
        result['PHASH_MAX_DIST'] = self.duplicate_detection.phash_max_dist
        result['SSIM_THRESHOLD'] = self.duplicate_detection.ssim_threshold
        result['TOP_K_NEIGHBORS'] = self.duplicate_detection.top_k_neighbors
        result['CLIP_PAIRING_MODE'] = self.duplicate_detection.clip_pairing_mode
        result['CLIP_MAX_OUTPUT_PAIRS'] = self.duplicate_detection.clip_max_output_pairs
        
        # Advanced discrimination
        result['USE_CLIP_ZSCORE'] = self.advanced_discrimination.use_clip_zscore
        result['CLIP_ZSCORE_MIN'] = self.advanced_discrimination.clip_zscore_min
        result['REQUIRE_CLIP_Z_FOR_CLIP_SSIM'] = self.advanced_discrimination.require_clip_z_for_clip_ssim
        result['USE_PATCHWISE_SSIM'] = self.advanced_discrimination.use_patchwise_ssim
        result['SSIM_GRID_H'] = self.advanced_discrimination.ssim_grid_h
        result['SSIM_GRID_W'] = self.advanced_discrimination.ssim_grid_w
        result['SSIM_TOPK_PATCHES'] = self.advanced_discrimination.ssim_topk_patches
        result['SSIM_MIX_WEIGHT'] = self.advanced_discrimination.ssim_mix_weight
        result['SSIM_PATCH_MIN_GATE'] = self.advanced_discrimination.ssim_patch_min_gate
        result['REQUIRE_PATCH_MIN_FOR_CLIP_SSIM'] = self.advanced_discrimination.require_patch_min_for_clip_ssim
        result['TIER_A_COPYPASTE_CLIP_MIN'] = self.advanced_discrimination.tier_a_copypaste_clip_min
        result['TIER_A_COPYPASTE_SSIM_MIN'] = self.advanced_discrimination.tier_a_copypaste_ssim_min
        result['TIER_A_COPYPASTE_COMBINED'] = self.advanced_discrimination.tier_a_copypaste_combined
        
        # Tier classification
        result['TIER_A_PHASH_RT'] = self.tier_classification.tier_a_phash_rt
        result['TIER_A_CLIP'] = self.tier_classification.tier_a_clip
        result['TIER_A_SSIM'] = self.tier_classification.tier_a_ssim
        result['TIER_A_ORB_INLIERS'] = self.tier_classification.tier_a_orb_inliers
        result['TIER_A_ORB_RATIO'] = self.tier_classification.tier_a_orb_ratio
        result['TIER_A_ORB_ERROR'] = self.tier_classification.tier_a_orb_error
        result['TIER_A_ORB_COVERAGE'] = self.tier_classification.tier_a_orb_coverage
        result['TIER_A_CLONE_AREA_PCT'] = self.tier_classification.tier_a_clone_area_pct
        result['TIER_B_PHASH_RT_MIN'] = self.tier_classification.tier_b_phash_rt_min
        result['TIER_B_PHASH_RT_MAX'] = self.tier_classification.tier_b_phash_rt_max
        result['TIER_B_CLIP_MIN'] = self.tier_classification.tier_b_clip_min
        result['TIER_B_CLIP_MAX'] = self.tier_classification.tier_b_clip_max
        result['TIER_B_SSIM_MIN'] = self.tier_classification.tier_b_ssim_min
        result['TIER_B_SSIM_MAX'] = self.tier_classification.tier_b_ssim_max
        result['TIER_A_RELAXED_CLIP'] = self.tier_classification.tier_a_relaxed_clip
        result['TIER_A_RELAXED_SSIM'] = self.tier_classification.tier_a_relaxed_ssim
        result['TIER_A_RELAXED_COMBINED'] = self.tier_classification.tier_a_relaxed_combined
        result['TIER_A_WESTERN_CLIP'] = self.tier_classification.tier_a_western_clip
        result['TIER_A_WESTERN_SSIM'] = self.tier_classification.tier_a_western_ssim
        result['TIER_A_WESTERN_COMBINED'] = self.tier_classification.tier_a_western_combined
        result['CONFOCAL_FP_CLIP_MIN'] = self.tier_classification.confocal_fp_clip_min
        result['CONFOCAL_FP_SSIM_MAX'] = self.tier_classification.confocal_fp_ssim_max
        result['CONFOCAL_FP_PHASH_MIN'] = self.tier_classification.confocal_fp_phash_min
        
        # Feature flags
        result['USE_PHASH_BUNDLES'] = self.feature_flags.use_phash_bundles
        result['USE_ORB_RANSAC'] = self.feature_flags.use_orb_ransac
        result['USE_TIER_GATING'] = self.feature_flags.use_tier_gating
        result['USE_SSIM_VALIDATION'] = self.feature_flags.use_ssim_validation
        result['HIGHLIGHT_DIFFERENCES'] = self.feature_flags.highlight_differences
        result['USE_MODALITY_SPECIFIC_GATING'] = self.feature_flags.use_modality_specific_gating
        result['ENABLE_MODALITY_DETECTION'] = self.feature_flags.enable_modality_detection
        result['ENABLE_MODALITY_ROUTING'] = self.feature_flags.enable_modality_routing
        result['EXPOSE_MODALITY_COLUMNS'] = self.feature_flags.expose_modality_columns
        result['ENABLE_FIGCHECK_HEURISTICS'] = self.feature_flags.enable_figcheck_heuristics
        result['ENABLE_ORB_RELAX'] = self.feature_flags.enable_orb_relax
        result['ENABLE_CONFOCAL_DEEP_VERIFY'] = self.feature_flags.enable_confocal_deep_verify
        result['ENABLE_IHC_DEEP_VERIFY'] = self.feature_flags.enable_ihc_deep_verify
        result['ENABLE_TEXT_MASKING'] = self.feature_flags.enable_text_masking
        result['ENABLE_CACHE'] = self.feature_flags.enable_cache
        result['DEBUG_MODE'] = self.feature_flags.debug_mode
        
        # Performance
        result['BATCH_SIZE'] = self.performance.batch_size
        result['NUM_WORKERS'] = self.performance.num_workers
        result['ENABLE_MPS'] = self.performance.enable_mps
        result['CACHE_VERSION'] = self.performance.cache_version
        result['RANDOM_SEED'] = self.performance.random_seed
        
        # UX
        result['AUTO_OPEN_RESULTS'] = self.auto_open_results
        result['AUTO_OPEN_PER_PAIR'] = self.auto_open_per_pair
        
        return result


# Convenience function for backward compatibility
def get_config(preset: Optional[Literal["fast", "balanced", "thorough"]] = None,
               config_file: Optional[Path] = None) -> DetectorConfig:
    """
    Get configuration from preset, file, or defaults
    
    Args:
        preset: Preset name ("fast", "balanced", "thorough")
        config_file: Path to YAML or JSON config file
    
    Returns:
        DetectorConfig instance
    """
    if config_file:
        if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
            return DetectorConfig.from_yaml(config_file)
        elif config_file.suffix.lower() == '.json':
            return DetectorConfig.from_json(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    
    if preset:
        return DetectorConfig.from_preset(preset)
    
    return DetectorConfig()

