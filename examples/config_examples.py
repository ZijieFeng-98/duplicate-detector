"""
Configuration Usage Examples

This module demonstrates how to use the new configuration system.
"""

from pathlib import Path
from duplicate_detector.models.config import DetectorConfig, get_config


def example_basic_usage():
    """Example 1: Basic usage with preset"""
    # Use a preset (fast, balanced, thorough)
    config = DetectorConfig.from_preset("balanced")
    
    # Set paths
    config.pdf_path = Path("document.pdf")
    config.output_dir = Path("output")
    
    print(f"DPI: {config.dpi}")
    print(f"CLIP threshold: {config.duplicate_detection.sim_threshold}")
    print(f"Batch size: {config.performance.batch_size}")


def example_custom_config():
    """Example 2: Custom configuration"""
    config = DetectorConfig()
    
    # Customize settings
    config.dpi = 200
    config.duplicate_detection.sim_threshold = 0.98
    config.duplicate_detection.phash_max_dist = 2
    config.performance.batch_size = 128
    
    # Enable/disable features
    config.feature_flags.use_orb_ransac = False
    config.feature_flags.enable_cache = True
    
    return config


def example_from_file():
    """Example 3: Load from YAML/JSON file"""
    # Load from YAML
    config = DetectorConfig.from_yaml(Path("config.yaml"))
    
    # Or load from JSON
    # config = DetectorConfig.from_json(Path("config.json"))
    
    return config


def example_environment_variables():
    """Example 4: Using environment variables"""
    import os
    
    # Set environment variables (or use .env file)
    os.environ['DUPLICATE_DETECTOR_PDF_PATH'] = '/path/to/document.pdf'
    os.environ['DUPLICATE_DETECTOR_OUTPUT_DIR'] = '/path/to/output'
    
    # Config will automatically load from environment
    config = DetectorConfig()
    
    print(f"PDF path: {config.pdf_path}")
    print(f"Output dir: {config.output_dir}")


def example_save_config():
    """Example 5: Save configuration to file"""
    config = DetectorConfig.from_preset("thorough")
    
    # Save to YAML
    config.to_yaml(Path("my_config.yaml"))
    
    # Save to JSON
    config.to_json(Path("my_config.json"))


def example_backward_compatibility():
    """Example 6: Use with existing pipeline code"""
    from duplicate_detector.models.migration import apply_config_to_module
    import ai_pdf_panel_duplicate_check_AUTO as pipeline_module
    
    # Create config
    config = DetectorConfig.from_preset("balanced")
    config.pdf_path = Path("document.pdf")
    config.output_dir = Path("output")
    
    # Apply to existing module
    apply_config_to_module(config, pipeline_module)
    
    # Now the pipeline module has all the config values set
    print(f"Module DPI: {pipeline_module.DPI}")
    print(f"Module SIM_THRESHOLD: {pipeline_module.SIM_THRESHOLD}")


if __name__ == "__main__":
    print("Configuration Examples")
    print("=" * 50)
    
    example_basic_usage()
    print()
    
    config = example_custom_config()
    print(f"Custom config created: DPI={config.dpi}")
    print()
    
    print("See config.example.yaml for a full configuration template")

