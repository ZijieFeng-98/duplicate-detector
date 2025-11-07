"""
Unit tests for API module.
"""

import pytest
from pathlib import Path
import pandas as pd

from duplicate_detector.api.detector import DuplicateDetector, DetectionResults
from duplicate_detector.models.config import DetectorConfig


class TestDuplicateDetector:
    """Test DuplicateDetector class."""
    
    def test_init_default(self):
        """Test initialization with default config."""
        detector = DuplicateDetector()
        
        assert isinstance(detector.config, DetectorConfig)
        assert detector.clip_model is None
        assert detector.panels == []
        assert isinstance(detector.meta_df, pd.DataFrame)
        assert detector.meta_df.empty
    
    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = DetectorConfig.from_preset("fast")
        detector = DuplicateDetector(config=config)
        
        assert detector.config == config
    
    def test_init_with_kwargs(self):
        """Test initialization with kwargs override."""
        detector = DuplicateDetector(dpi=200, random_seed=999)
        
        assert detector.config.dpi == 200
        assert detector.config.random_seed == 999
    
    @pytest.mark.skip(reason="Requires actual PDF file - integration test")
    def test_analyze_pdf(self):
        """Test PDF analysis."""
        # This requires an actual PDF file
        pass


class TestDetectionResults:
    """Test DetectionResults class."""
    
    def test_init(self):
        """Test DetectionResults initialization."""
        results = DetectionResults(
            total_pairs=10,
            tier_a_pairs=[{'Image_A': 'a.png', 'Image_B': 'b.png'}],
            tier_b_pairs=[{'Image_A': 'c.png', 'Image_B': 'd.png'}],
            all_pairs=pd.DataFrame([{'Image_A': 'a.png', 'Image_B': 'b.png'}]),
            metadata={'num_panels': 50}
        )
        
        assert results.total_pairs == 10
        assert len(results.tier_a_pairs) == 1
        assert len(results.tier_b_pairs) == 1
        assert isinstance(results.all_pairs, pd.DataFrame)
        assert results.metadata['num_panels'] == 50
    
    def test_repr(self):
        """Test string representation."""
        results = DetectionResults(
            total_pairs=5,
            tier_a_pairs=[{}] * 2,
            tier_b_pairs=[{}] * 3,
            all_pairs=pd.DataFrame(),
            metadata={}
        )
        
        repr_str = repr(results)
        assert 'DetectionResults' in repr_str
        assert 'total_pairs=5' in repr_str
        assert 'tier_a=2' in repr_str
        assert 'tier_b=3' in repr_str
    
    def test_get_tier_counts(self):
        """Test tier count methods."""
        results = DetectionResults(
            total_pairs=5,
            tier_a_pairs=[{}] * 2,
            tier_b_pairs=[{}] * 3,
            all_pairs=pd.DataFrame(),
            metadata={}
        )
        
        assert results.get_tier_a_count() == 2
        assert results.get_tier_b_count() == 3
    
    def test_save(self, temp_dir):
        """Test saving results to CSV."""
        df = pd.DataFrame([
            {'Image_A': 'a.png', 'Image_B': 'b.png', 'Tier': 'A'},
            {'Image_A': 'c.png', 'Image_B': 'd.png', 'Tier': 'B'}
        ])
        
        results = DetectionResults(
            total_pairs=2,
            tier_a_pairs=[df.iloc[0].to_dict()],
            tier_b_pairs=[df.iloc[1].to_dict()],
            all_pairs=df,
            metadata={}
        )
        
        output_path = temp_dir / "results.csv"
        results.save(output_path)
        
        assert output_path.exists()
        # Verify CSV can be read back
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 2

