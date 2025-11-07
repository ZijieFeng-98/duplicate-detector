"""
Test fixtures and utilities for duplicate detector tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

from duplicate_detector.models.config import DetectorConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Create a sample DetectorConfig for testing."""
    return DetectorConfig.from_preset("fast")


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image."""
    img_path = tmp_path / "test_image.png"
    # Create a simple test image (100x100 RGB)
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_panel_paths(tmp_path):
    """Create multiple sample panel images."""
    panel_paths = []
    for i in range(5):
        img_path = tmp_path / f"panel_{i:02d}.png"
        # Create different colored images
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        img = Image.new('RGB', (200, 200), color=colors[i])
        img.save(img_path)
        panel_paths.append(img_path)
    return panel_paths


@pytest.fixture
def sample_meta_df(sample_panel_paths):
    """Create a sample metadata DataFrame."""
    data = []
    for i, path in enumerate(sample_panel_paths):
        data.append({
            "Panel_Path": str(path),
            "Panel_Name": path.name,
            "Page": f"page_{i//2 + 1}",
            "Panel_Num": (i % 2) + 1,
            "X": 0,
            "Y": 0,
            "Width": 200,
            "Height": 200,
            "Area": 40000
        })
    return pd.DataFrame(data)


@pytest.fixture
def mock_clip_embeddings(sample_panel_paths):
    """Create mock CLIP embeddings."""
    n = len(sample_panel_paths)
    # Create random normalized embeddings
    embeddings = np.random.randn(n, 512).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def sample_duplicate_pairs_df(sample_panel_paths):
    """Create a sample duplicate pairs DataFrame."""
    return pd.DataFrame([
        {
            "Image_A": "panel_00.png",
            "Image_B": "panel_01.png",
            "Path_A": str(sample_panel_paths[0]),
            "Path_B": str(sample_panel_paths[1]),
            "Cosine_Similarity": 0.98,
            "SSIM": 0.95,
            "Hamming_Distance": 2,
            "Tier": "A",
            "Tier_Path": "Strict (CLIP+SSIM)"
        },
        {
            "Image_A": "panel_02.png",
            "Image_B": "panel_03.png",
            "Path_A": str(sample_panel_paths[2]),
            "Path_B": str(sample_panel_paths[3]),
            "Cosine_Similarity": 0.93,
            "SSIM": 0.88,
            "Hamming_Distance": 5,
            "Tier": "B",
            "Tier_Path": "Borderline (CLIP+SSIM)"
        }
    ])

