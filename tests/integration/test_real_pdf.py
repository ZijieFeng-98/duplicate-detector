"""
Integration test using real PDF file from Dr_Zhong directory.

This test uses actual PDF files and creates intentional duplicates
to verify the duplicate detection pipeline works correctly.
"""

import pytest
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

from duplicate_detector import DuplicateDetector, DetectorConfig
from duplicate_detector.core.panel_detector import pages_to_panels_auto
from duplicate_detector.core.similarity_engine import (
    load_clip,
    load_or_compute_embeddings,
    clip_find_duplicates_threshold,
    phash_find_duplicates_with_bundles
)


# Path to test directory
TEST_DATA_DIR = Path("/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong")
TEST_PDF = TEST_DATA_DIR / "STM-Combined Figures.pdf"


@pytest.fixture(scope="module")
def test_pdf_path():
    """Get the test PDF path."""
    if not TEST_PDF.exists():
        pytest.skip(f"Test PDF not found: {TEST_PDF}")
    return TEST_PDF


@pytest.fixture(scope="module")
def test_output_dir(tmp_path_factory):
    """Create a temporary output directory for tests."""
    return tmp_path_factory.mktemp("integration_test_output")


@pytest.fixture(scope="module")
def test_config(test_output_dir):
    """Create test configuration."""
    config = DetectorConfig.from_preset("fast")
    config.output_dir = test_output_dir
    config.dpi = 150
    config.duplicate_detection.sim_threshold = 0.94  # Lower for testing
    config.duplicate_detection.phash_max_dist = 5
    config.feature_flags.enable_cache = True
    config.feature_flags.debug_mode = False
    return config


@pytest.mark.integration
class TestRealPDFIntegration:
    """Integration tests with real PDF file."""
    
    def test_pdf_exists(self, test_pdf_path):
        """Verify test PDF exists."""
        assert test_pdf_path.exists(), f"Test PDF not found: {test_pdf_path}"
        assert test_pdf_path.suffix == ".pdf"
    
    def test_panel_detection_on_real_pdf(self, test_pdf_path, test_config, test_output_dir):
        """Test panel detection on real PDF."""
        from duplicate_detector.core.panel_detector import pdf_to_pages
        
        # Convert PDF to pages
        pages = pdf_to_pages(
            pdf_path=test_pdf_path,
            out_dir=test_output_dir,
            dpi=test_config.dpi,
            caption_pages=set(),
            debug_mode=False
        )
        
        assert len(pages) > 0, "No pages extracted from PDF"
        
        # Extract panels
        panels, meta_df = pages_to_panels_auto(
            pages=pages[:3],  # Test with first 3 pages only
            out_dir=test_output_dir,
            min_panel_area=test_config.panel_detection.min_panel_area,
            max_panel_area=test_config.panel_detection.max_panel_area,
            debug_mode=False
        )
        
        assert len(panels) > 0, "No panels detected"
        assert isinstance(meta_df, pd.DataFrame)
        assert len(meta_df) == len(panels)
        assert 'Panel_Path' in meta_df.columns
    
    def test_create_intentional_duplicates(self, test_output_dir, test_config):
        """Create intentional duplicates for testing."""
        # First, get some panels
        panels_dir = test_output_dir / "panels"
        if not panels_dir.exists():
            pytest.skip("Panels not yet extracted. Run panel detection first.")
        
        # Find some panel images
        panel_files = list(panels_dir.rglob("*.png"))
        if len(panel_files) < 2:
            pytest.skip("Not enough panels found")
        
        # Create duplicates directory
        duplicates_dir = test_output_dir / "intentional_duplicates"
        duplicates_dir.mkdir(exist_ok=True)
        
        # Copy a panel as an exact duplicate
        original = panel_files[0]
        exact_duplicate = duplicates_dir / f"exact_dup_{original.name}"
        shutil.copy(original, exact_duplicate)
        
        # Create a rotated duplicate
        img = Image.open(original)
        rotated = img.rotate(90)
        rotated_duplicate = duplicates_dir / f"rotated_dup_{original.name}"
        rotated.save(rotated_duplicate)
        
        # Create a slightly modified duplicate (add noise)
        img_array = np.array(img)
        noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_array)
        noisy_duplicate = duplicates_dir / f"noisy_dup_{original.name}"
        noisy_img.save(noisy_duplicate)
        
        # Create a cropped duplicate
        w, h = img.size
        cropped = img.crop((w//4, h//4, 3*w//4, 3*h//4))
        cropped_duplicate = duplicates_dir / f"cropped_dup_{original.name}"
        cropped.save(cropped_duplicate)
        
        duplicates = {
            'exact': exact_duplicate,
            'rotated': rotated_duplicate,
            'noisy': noisy_duplicate,
            'cropped': cropped_duplicate,
            'original': original
        }
        
        return duplicates
    
    def test_clip_detection_on_duplicates(self, test_output_dir, test_config):
        """Test CLIP detection on intentional duplicates."""
        duplicates_dir = test_output_dir / "intentional_duplicates"
        if not duplicates_dir.exists():
            pytest.skip("Duplicates not created. Run create_intentional_duplicates first.")
        
        duplicate_files = list(duplicates_dir.glob("*.png"))
        if len(duplicate_files) < 2:
            pytest.skip("Not enough duplicate files")
        
        # Load CLIP model
        device = "cpu"  # Use CPU for testing
        clip_model = load_clip(device=device)
        
        # Generate embeddings
        vecs = load_or_compute_embeddings(
            panel_paths=duplicate_files,
            clip=clip_model,
            output_dir=test_output_dir,
            cache_version="test_v1",
            enable_cache=True,
            batch_size=4
        )
        
        assert vecs.shape[0] == len(duplicate_files)
        assert vecs.shape[1] == 512  # CLIP ViT-B-32 embedding size
        
        # Create metadata DataFrame
        meta_df = pd.DataFrame([
            {
                "Panel_Path": str(f),
                "Panel_Name": f.name,
                "Page": "test_page",
                "Panel_Num": i + 1,
                "X": 0, "Y": 0, "Width": 200, "Height": 200, "Area": 40000
            }
            for i, f in enumerate(duplicate_files)
        ])
        
        # Find duplicates
        df_duplicates = clip_find_duplicates_threshold(
            panel_paths=duplicate_files,
            vecs=vecs,
            threshold=0.90,  # Lower threshold for testing
            meta_df=meta_df,
            suppress_same_page=False,
            suppress_adjacent_page=False
        )
        
        # Should find at least the exact duplicate
        assert len(df_duplicates) > 0, "No duplicates detected"
        
        # Check that exact duplicate is found
        exact_found = False
        for _, row in df_duplicates.iterrows():
            if 'exact_dup' in row['Image_A'] or 'exact_dup' in row['Image_B']:
                exact_found = True
                assert row['Cosine_Similarity'] > 0.99  # Exact duplicate should be very similar
                break
        
        assert exact_found, "Exact duplicate should be detected"
    
    def test_phash_detection_on_duplicates(self, test_output_dir, test_config):
        """Test pHash detection on intentional duplicates."""
        duplicates_dir = test_output_dir / "intentional_duplicates"
        if not duplicates_dir.exists():
            pytest.skip("Duplicates not created")
        
        duplicate_files = list(duplicates_dir.glob("*.png"))
        if len(duplicate_files) < 2:
            pytest.skip("Not enough duplicate files")
        
        # Create metadata
        meta_df = pd.DataFrame([
            {
                "Panel_Path": str(f),
                "Panel_Name": f.name,
                "Page": "test_page",
                "Panel_Num": i + 1,
                "X": 0, "Y": 0, "Width": 200, "Height": 200, "Area": 40000
            }
            for i, f in enumerate(duplicate_files)
        ])
        
        # Find duplicates with pHash
        df_phash = phash_find_duplicates_with_bundles(
            panel_paths=duplicate_files,
            max_dist=5,  # Allow up to 5 Hamming distance
            meta_df=meta_df,
            output_dir=test_output_dir,
            cache_version="test_v1",
            enable_cache=True,
            num_workers=2,
            suppress_same_page=False,
            suppress_adjacent_page=False
        )
        
        # Should find at least the exact duplicate
        assert len(df_phash) > 0, "No pHash duplicates detected"
        
        # Check that exact duplicate is found
        exact_found = False
        for _, row in df_phash.iterrows():
            if 'exact_dup' in row['Image_A'] or 'exact_dup' in row['Image_B']:
                exact_found = True
                assert row['Hamming_Distance'] <= 2  # Exact duplicate should have low distance
                break
        
        assert exact_found, "Exact duplicate should be detected by pHash"
    
    def test_full_pipeline_on_real_pdf(self, test_pdf_path, test_config, test_output_dir):
        """Test full pipeline on real PDF."""
        # Use a smaller subset for faster testing
        config = test_config
        config.pdf_path = test_pdf_path
        
        detector = DuplicateDetector(config=config)
        
        # Run full pipeline
        results = detector.analyze_pdf()
        
        # Verify results
        assert isinstance(results.total_pairs, int)
        assert results.total_pairs >= 0
        assert isinstance(results.tier_a_pairs, list)
        assert isinstance(results.tier_b_pairs, list)
        assert isinstance(results.all_pairs, pd.DataFrame)
        
        # If duplicates found, verify structure
        if results.total_pairs > 0:
            assert len(results.all_pairs) == results.total_pairs
            assert 'Image_A' in results.all_pairs.columns
            assert 'Image_B' in results.all_pairs.columns
            
            # Check tier distribution
            tier_a_count = results.get_tier_a_count()
            tier_b_count = results.get_tier_b_count()
            assert tier_a_count + tier_b_count <= results.total_pairs


@pytest.mark.integration
class TestDuplicateCreation:
    """Test creating and detecting intentional duplicates."""
    
    def test_create_and_detect_exact_duplicate(self, test_output_dir):
        """Create an exact duplicate and verify it's detected."""
        # Create a test image
        test_img = Image.new('RGB', (200, 200), color='blue')
        original_path = test_output_dir / "original.png"
        test_img.save(original_path)
        
        # Create exact duplicate
        duplicate_path = test_output_dir / "duplicate.png"
        shutil.copy(original_path, duplicate_path)
        
        # Test with pHash (should detect exact match)
        from duplicate_detector.core.similarity_engine import phash_hex, hamming_min_transform
        
        hash1 = phash_hex(original_path)
        hash2 = phash_hex(duplicate_path)
        
        # Exact duplicates should have distance 0
        from imagehash import hex_to_hash
        distance = hex_to_hash(hash1) - hex_to_hash(hash2)
        assert distance == 0, f"Exact duplicate should have distance 0, got {distance}"
    
    def test_create_and_detect_rotated_duplicate(self, test_output_dir):
        """Create a rotated duplicate and verify pHash bundle detects it."""
        # Create a test image with distinctive pattern
        test_img = Image.new('RGB', (200, 200), color='red')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([50, 50, 150, 150], fill='blue')
        
        original_path = test_output_dir / "original_rot.png"
        test_img.save(original_path)
        
        # Create rotated duplicate
        rotated = test_img.rotate(90)
        rotated_path = test_output_dir / "rotated_90.png"
        rotated.save(rotated_path)
        
        # Test with pHash bundle (should detect rotated match)
        from duplicate_detector.core.similarity_engine import compute_phash_bundle, hamming_min_transform
        
        bundle1 = compute_phash_bundle(test_img)
        bundle2 = compute_phash_bundle(rotated)
        
        min_dist, transform = hamming_min_transform(bundle1, bundle2)
        
        # Rotated duplicate should be detected (low distance)
        assert min_dist <= 5, f"Rotated duplicate should have low distance, got {min_dist}"
        assert 'rot_90' in transform or 'rot_270' in transform, "Should detect rotation"

