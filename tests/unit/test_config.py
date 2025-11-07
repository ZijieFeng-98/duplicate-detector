"""
Unit tests for configuration management.
"""

import pytest
import tempfile
from pathlib import Path
import json
import yaml

from duplicate_detector.models.config import (
    DetectorConfig,
    PanelDetectionConfig,
    DuplicateDetectionConfig,
    FeatureFlags,
    PerformanceConfig
)


class TestDetectorConfig:
    """Test DetectorConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = DetectorConfig()
        
        assert config.dpi == 150
        assert config.random_seed == 123
        assert isinstance(config.panel_detection, PanelDetectionConfig)
        assert isinstance(config.duplicate_detection, DuplicateDetectionConfig)
        assert isinstance(config.feature_flags, FeatureFlags)
        assert isinstance(config.performance, PerformanceConfig)
    
    def test_config_from_preset_fast(self):
        """Test fast preset configuration."""
        config = DetectorConfig.from_preset("fast")
        
        assert config.dpi == 150
        assert config.duplicate_detection.sim_threshold < 0.98  # Lower threshold for speed
        assert config.performance.batch_size >= 32
    
    def test_config_from_preset_balanced(self):
        """Test balanced preset configuration."""
        config = DetectorConfig.from_preset("balanced")
        
        assert config.duplicate_detection.sim_threshold >= 0.95
        assert config.feature_flags.use_phash_bundles == True
    
    def test_config_from_preset_thorough(self):
        """Test thorough preset configuration."""
        config = DetectorConfig.from_preset("thorough")
        
        assert config.duplicate_detection.sim_threshold >= 0.96
        assert config.feature_flags.use_orb_ransac == True
    
    def test_config_from_yaml(self, temp_dir):
        """Test loading configuration from YAML file."""
        yaml_file = temp_dir / "config.yaml"
        yaml_content = """
dpi: 200
random_seed: 456
duplicate_detection:
  sim_threshold: 0.97
  phash_max_dist: 4
"""
        yaml_file.write_text(yaml_content)
        
        config = DetectorConfig.from_yaml(yaml_file)
        
        assert config.dpi == 200
        assert config.random_seed == 456
        assert config.duplicate_detection.sim_threshold == 0.97
        assert config.duplicate_detection.phash_max_dist == 4
    
    def test_config_from_json(self, temp_dir):
        """Test loading configuration from JSON file."""
        json_file = temp_dir / "config.json"
        json_content = {
            "dpi": 200,
            "random_seed": 456,
            "duplicate_detection": {
                "sim_threshold": 0.97,
                "phash_max_dist": 4
            }
        }
        json_file.write_text(json.dumps(json_content))
        
        config = DetectorConfig.from_json(json_file)
        
        assert config.dpi == 200
        assert config.random_seed == 456
        assert config.duplicate_detection.sim_threshold == 0.97
        assert config.duplicate_detection.phash_max_dist == 4
    
    def test_config_validation(self):
        """Test configuration validation."""
        # DPI out of range should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            DetectorConfig(dpi=50)  # Below minimum
        
        with pytest.raises(Exception):
            DetectorConfig(dpi=400)  # Above maximum
        
        # Valid DPI should work
        config = DetectorConfig(dpi=200)
        assert config.dpi == 200
    
    def test_config_model_dump(self):
        """Test configuration serialization."""
        config = DetectorConfig.from_preset("balanced")
        dumped = config.model_dump()
        
        assert isinstance(dumped, dict)
        assert 'dpi' in dumped
        assert 'duplicate_detection' in dumped
        assert isinstance(dumped['duplicate_detection'], dict)


class TestNestedConfigs:
    """Test nested configuration classes."""
    
    def test_panel_detection_config(self):
        """Test PanelDetectionConfig."""
        config = PanelDetectionConfig(
            min_panel_area=50000,
            max_panel_area=5000000
        )
        
        assert config.min_panel_area == 50000
        assert config.max_panel_area == 5000000
    
    def test_duplicate_detection_config(self):
        """Test DuplicateDetectionConfig."""
        config = DuplicateDetectionConfig(
            sim_threshold=0.96,
            phash_max_dist=3
        )
        
        assert config.sim_threshold == 0.96
        assert config.phash_max_dist == 3
    
    def test_feature_flags(self):
        """Test FeatureFlags."""
        flags = FeatureFlags(
            use_phash_bundles=True,
            use_orb_ransac=True,
            use_tier_gating=True
        )
        
        assert flags.use_phash_bundles == True
        assert flags.use_orb_ransac == True
        assert flags.use_tier_gating == True
    
    def test_performance_config(self):
        """Test PerformanceConfig."""
        perf = PerformanceConfig(
            batch_size=64,
            num_workers=4,
            device="cuda"
        )
        
        assert perf.batch_size == 64
        assert perf.num_workers == 4
        assert perf.device == "cuda"

