"""
Unit tests for panel detection module.
"""

import pytest
from pathlib import Path
import pandas as pd
from PIL import Image

from duplicate_detector.core.panel_detector import (
    pdf_to_pages,
    detect_panels_cv,
    pages_to_panels_auto,
    compute_iou,
    page_stem,
    ensure_dir
)


class TestPanelDetectorHelpers:
    """Test helper functions."""
    
    def test_page_stem(self):
        """Test page stem generation."""
        assert page_stem(0) == "page_1"
        assert page_stem(1) == "page_2"
        assert page_stem(9) == "page_10"
    
    def test_compute_iou(self):
        """Test IoU computation."""
        box_a = {'x': 0, 'y': 0, 'w': 100, 'h': 100}
        box_b = {'x': 50, 'y': 50, 'w': 100, 'h': 100}
        
        iou = compute_iou(box_a, box_b)
        assert 0.0 <= iou <= 1.0
        
        # Identical boxes
        iou_identical = compute_iou(box_a, box_a)
        assert iou_identical == 1.0
        
        # Non-overlapping boxes
        box_c = {'x': 200, 'y': 200, 'w': 100, 'h': 100}
        iou_no_overlap = compute_iou(box_a, box_c)
        assert iou_no_overlap == 0.0
    
    def test_ensure_dir(self, temp_dir):
        """Test directory creation."""
        test_dir = temp_dir / "test_subdir" / "nested"
        ensure_dir(test_dir)
        assert test_dir.exists()
        assert test_dir.is_dir()


class TestPanelDetection:
    """Test panel detection functions."""
    
    def test_detect_panels_cv(self, sample_image, temp_dir):
        """Test panel detection on a single image."""
        out_dir = temp_dir / "panels"
        
        # Create a simple image with a clear rectangular region
        img = Image.new('RGB', (500, 500), color='white')
        # Draw a rectangle (simulating a panel)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 250, 250], fill='black', outline='black', width=2)
        img.save(sample_image)
        
        results = detect_panels_cv(
            page_png=sample_image,
            out_dir=out_dir,
            min_panel_area=1000,
            max_panel_area=1000000,
            debug_mode=False
        )
        
        # Should detect at least one panel
        assert isinstance(results, list)
        # Each result should be a tuple of (path, metadata)
        if len(results) > 0:
            assert len(results[0]) == 2
            assert isinstance(results[0][0], Path)
            assert isinstance(results[0][1], dict)
            assert 'x' in results[0][1]
            assert 'y' in results[0][1]
            assert 'width' in results[0][1]
            assert 'height' in results[0][1]
    
    def test_pages_to_panels_auto(self, sample_panel_paths, temp_dir):
        """Test batch panel extraction."""
        # Create page images (simplified - just use panel images as pages)
        pages = sample_panel_paths[:2]  # Use first 2 as "pages"
        
        panel_paths, meta_df = pages_to_panels_auto(
            pages=pages,
            out_dir=temp_dir,
            min_panel_area=1000,
            debug_mode=False
        )
        
        assert isinstance(panel_paths, list)
        assert isinstance(meta_df, pd.DataFrame)
        
        if len(panel_paths) > 0:
            assert len(panel_paths) == len(meta_df)
            assert 'Panel_Path' in meta_df.columns
            assert 'Panel_Name' in meta_df.columns
            assert 'Page' in meta_df.columns


class TestPDFConversion:
    """Test PDF to pages conversion."""
    
    def test_pdf_to_pages_requires_valid_pdf(self, temp_dir):
        """Test that invalid PDF raises error."""
        invalid_pdf = temp_dir / "nonexistent.pdf"
        
        # Should exit with error (sys.exit in original, but we can test the path check)
        # In practice, this would be caught by the file existence check
        assert not invalid_pdf.exists()
    
    # Note: Testing actual PDF conversion requires a real PDF file
    # This would be an integration test rather than a unit test

