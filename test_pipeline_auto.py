#!/usr/bin/env python3
"""
Automated Test Suite for Duplicate Detection Pipeline
Run with: python test_pipeline_auto.py
"""

import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import traceback

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Your test data path
TEST_PDF_PATH = Path("/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/PUA-STM-Combined Figures .pdf")
TEST_OUTPUT_DIR = Path("./test_output")
MAIN_SCRIPT = Path("./ai_pdf_panel_duplicate_check_AUTO.py")

# Expected thresholds (adjust based on your requirements)
MIN_PAGES_EXPECTED = 1
MIN_PANELS_EXPECTED = 5
EXPECTED_COLUMNS = [
    "Image_A", "Image_B", "Path_A", "Path_B", 
    "Cosine_Similarity", "Hamming_Distance", "Source"
]

# Test configurations to run
TEST_CONFIGS = [
    {
        "name": "Balanced_Default",
        "args": ["--sim-threshold", "0.96", "--ssim-threshold", "0.9", "--phash-max-dist", "4"],
        "expect_results": True,
        "expected_runtime": 90.0,  # Baseline from October 18, 2025
        "expected_panels": 107,
        "expected_pairs_min": 100
    },
    {
        "name": "Permissive",
        "args": ["--sim-threshold", "0.85", "--ssim-threshold", "0.70", "--phash-max-dist", "6"],
        "expect_results": True,
        "expected_runtime": 300.0,  # Baseline from October 18, 2025
        "expected_panels": 107,
        "expected_pairs_min": 600
    }
]


# ============================================================================
# UTILITIES
# ============================================================================

class TestLogger:
    """Simple test logger with emoji indicators"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_time = datetime.now()
    
    def test_start(self, name):
        print(f"\n{'='*70}")
        print(f"🧪 TEST: {name}")
        print(f"{'='*70}")
        self.tests_run += 1
    
    def success(self, msg):
        print(f"  ✅ {msg}")
        self.tests_passed += 1
    
    def failure(self, msg):
        print(f"  ❌ {msg}")
        self.tests_failed += 1
    
    def info(self, msg):
        print(f"  ℹ️  {msg}")
    
    def warning(self, msg):
        print(f"  ⚠️  {msg}")
    
    def summary(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'='*70}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*70}")
        print(f"  Tests run: {self.tests_run}")
        print(f"  Passed: {self.tests_passed} ✅")
        print(f"  Failed: {self.tests_failed} ❌")
        print(f"  Time: {elapsed:.1f}s")
        print(f"{'='*70}")
        
        if self.tests_failed == 0:
            print("🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"💥 {self.tests_failed} TEST(S) FAILED")
            return 1


def cleanup_test_output():
    """Remove previous test outputs"""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_tsv(path):
    """Read TSV file as list of dicts"""
    import csv
    if not path.exists():
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)


def check_file_exists(path, logger):
    """Check if file exists and is non-empty"""
    if not path.exists():
        logger.failure(f"File missing: {path.name}")
        return False
    
    size = path.stat().st_size
    if size == 0:
        logger.failure(f"File is empty: {path.name}")
        return False
    
    logger.success(f"Found {path.name} ({size:,} bytes)")
    return True


# ============================================================================
# TEST CASES
# ============================================================================

def test_prerequisites(logger):
    """Test 1: Check prerequisites"""
    logger.test_start("Prerequisites Check")
    
    # Check main script exists
    if not MAIN_SCRIPT.exists():
        logger.failure(f"Main script not found: {MAIN_SCRIPT}")
        return False
    logger.success(f"Found main script: {MAIN_SCRIPT}")
    
    # Check test PDF exists
    if not TEST_PDF_PATH.exists():
        logger.failure(f"Test PDF not found: {TEST_PDF_PATH}")
        return False
    logger.success(f"Found test PDF: {TEST_PDF_PATH.name}")
    
    # Check Python packages
    try:
        import torch
        import open_clip
        import imagehash
        import cv2
        import sklearn
        logger.success("All required packages installed")
    except ImportError as e:
        logger.failure(f"Missing package: {e}")
        return False
    
    return True


def test_pipeline_run(pdf_path, config, logger):
    """Test 2: Run pipeline with specific config"""
    logger.test_start(f"Pipeline Run: {config['name']} - {pdf_path.name}")
    
    output_dir = TEST_OUTPUT_DIR / f"{pdf_path.stem}_{config['name']}"
    
    # Build command
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--pdf", str(pdf_path),
        "--output", str(output_dir),
        "--no-auto-open",
        "--use-phash-bundles",
        "--use-orb",
        "--use-tier-gating",
        "--enable-cache",
        *config['args']
    ]
    
    logger.info(f"Running: {' '.join([str(c) for c in cmd])}")
    
    try:
        # Run pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Check exit code
        if result.returncode != 0:
            logger.failure(f"Pipeline exited with code {result.returncode}")
            logger.info(f"STDERR (last 1000 chars): {result.stderr[-1000:]}")
            return False, None
        
        logger.success("Pipeline completed successfully")
        
        # Save logs
        log_file = output_dir / "test_run.log"
        log_file.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}", encoding='utf-8')
        logger.info(f"Logs saved to: {log_file}")
        
        return True, output_dir
        
    except subprocess.TimeoutExpired:
        logger.failure("Pipeline timeout (>10 minutes)")
        return False, None
    except Exception as e:
        logger.failure(f"Pipeline error: {e}")
        traceback.print_exc()
        return False, None


def test_output_structure(output_dir, logger):
    """Test 3: Verify output directory structure"""
    logger.test_start(f"Output Structure: {output_dir.name}")
    
    required_items = [
        ("pages", "dir"),
        ("panels", "dir"),
        ("panel_manifest.tsv", "file"),
        ("ai_duplicate_report.tsv", "file"),
        ("final_merged_report.tsv", "file"),
        ("RUN_METADATA.json", "file")
    ]
    
    all_good = True
    for item, item_type in required_items:
        path = output_dir / item
        if not path.exists():
            logger.failure(f"Missing: {item}")
            all_good = False
        else:
            if item_type == "dir" and not path.is_dir():
                logger.failure(f"{item} should be a directory")
                all_good = False
            elif item_type == "file" and not path.is_file():
                logger.failure(f"{item} should be a file")
                all_good = False
            else:
                logger.success(f"Found: {item}")
    
    return all_good


def test_pages_extraction(output_dir, logger):
    """Test 4: Verify PDF → Pages conversion"""
    logger.test_start("Pages Extraction")
    
    pages_dir = output_dir / "pages"
    if not pages_dir.exists():
        logger.failure("pages/ directory missing")
        return False
    
    pages = list(pages_dir.glob("*.png"))
    if len(pages) < MIN_PAGES_EXPECTED:
        logger.failure(f"Expected ≥{MIN_PAGES_EXPECTED} pages, got {len(pages)}")
        return False
    
    logger.success(f"Extracted {len(pages)} pages")
    
    # Check image validity
    try:
        from PIL import Image
        for page in pages[:3]:  # Sample first 3
            img = Image.open(page)
            logger.info(f"  {page.name}: {img.size}")
    except Exception as e:
        logger.warning(f"Could not validate images: {e}")
    
    return True


def test_panel_detection(output_dir, logger):
    """Test 5: Verify panel detection"""
    logger.test_start("Panel Detection")
    
    panels_dir = output_dir / "panels"
    if not panels_dir.exists():
        logger.failure("panels/ directory missing")
        return False
    
    # Count all panels across subdirectories
    all_panels = list(panels_dir.rglob("*.png"))
    
    if len(all_panels) < MIN_PANELS_EXPECTED:
        logger.failure(f"Expected ≥{MIN_PANELS_EXPECTED} panels, got {len(all_panels)}")
        return False
    
    logger.success(f"Detected {len(all_panels)} panels")
    
    # Check manifest
    manifest = output_dir / "panel_manifest.tsv"
    if check_file_exists(manifest, logger):
        try:
            import pandas as pd
            df = pd.read_csv(manifest, sep='\t')
            logger.info(f"  Manifest has {len(df)} entries")
            logger.info(f"  Columns: {list(df.columns)[:10]}")
        except Exception as e:
            logger.warning(f"Could not parse manifest: {e}")
    
    return True


def test_duplicate_detection(output_dir, config, logger):
    """Test 6: Verify duplicate detection results"""
    logger.test_start(f"Duplicate Detection: {config['name']}")
    
    final_report = output_dir / "final_merged_report.tsv"
    if not check_file_exists(final_report, logger):
        return False
    
    # Read results
    data = read_tsv(final_report)
    if data is None:
        logger.failure("Could not read TSV")
        return False
    
    num_pairs = len(data)
    logger.info(f"Found {num_pairs} duplicate pair(s)")
    
    # Validate columns
    if data:
        actual_cols = set(data[0].keys())
        expected_cols = set(EXPECTED_COLUMNS)
        missing = expected_cols - actual_cols
        
        if missing:
            logger.warning(f"Missing columns: {missing}")
        else:
            logger.success("All expected columns present")
    
    # Check for expected results
    if config['expect_results'] and num_pairs == 0:
        logger.warning("Expected duplicates but found none (may need tuning)")
    elif not config['expect_results'] and num_pairs > 0:
        logger.info(f"Found {num_pairs} pair(s) with strict settings")
    else:
        logger.success(f"Results match expectations")
    
    # Sample results
    if data:
        logger.info("Sample results:")
        for i, row in enumerate(data[:3], 1):
            clip = row.get('Cosine_Similarity', 'N/A')
            phash = row.get('Hamming_Distance', 'N/A')
            img_a = Path(row.get('Image_A', 'N/A')).name
            img_b = Path(row.get('Image_B', 'N/A')).name
            logger.info(f"  {i}. {img_a} vs {img_b}")
            logger.info(f"     CLIP={clip}, pHash={phash}")
    
    return True


def test_tier_classification(output_dir, logger):
    """Test 7: Verify tier gating (if enabled)"""
    logger.test_start("Tier Classification")
    
    final_report = output_dir / "final_merged_report.tsv"
    data = read_tsv(final_report)
    
    if not data:
        logger.info("No results to classify")
        return True
    
    # Check for Tier column
    if 'Tier' not in data[0]:
        logger.info("Tier gating not enabled (okay)")
        return True
    
    tier_counts = {'A': 0, 'B': 0, 'Other': 0}
    for row in data:
        tier = row.get('Tier', 'Other')
        if tier in tier_counts:
            tier_counts[tier] += 1
        else:
            tier_counts['Other'] += 1
    
    logger.info(f"Tier A: {tier_counts['A']}")
    logger.info(f"Tier B: {tier_counts['B']}")
    logger.info(f"Other: {tier_counts['Other']}")
    
    logger.success("Tier classification present")
    return True


def test_metadata_integrity(output_dir, logger):
    """Test 8: Verify run metadata"""
    logger.test_start("Metadata Integrity")
    
    metadata_file = output_dir / "RUN_METADATA.json"
    if not check_file_exists(metadata_file, logger):
        return False
    
    try:
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        
        # Check required fields
        required_fields = ['timestamp', 'runtime_seconds', 'config']
        for field in required_fields:
            if field in metadata:
                logger.success(f"Field present: {field}")
            else:
                logger.warning(f"Field missing: {field}")
        
        # Show some stats
        if 'config' in metadata:
            logger.info(f"Config keys: {list(metadata['config'].keys())[:5]}")
        
        if 'runtime_seconds' in metadata:
            logger.info(f"Runtime: {metadata['runtime_seconds']:.1f}s")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.failure(f"Invalid JSON: {e}")
        return False


def test_sklearn_import(logger):
    """Test 9: Verify sklearn deployment fix"""
    logger.test_start("sklearn Import (Deployment Fix)")
    
    try:
        # Test both import styles that caused issues
        import sklearn
        from sklearn.metrics.pairwise import cosine_similarity
        logger.success("sklearn imported successfully")
        
        # Verify it's the right version
        version = sklearn.__version__
        logger.info(f"sklearn version: {version}")
        
        major, minor = map(int, version.split('.')[:2])
        if (major, minor) >= (1, 3):
            logger.success("sklearn version ≥ 1.3.0")
            return True
        else:
            logger.warning(f"sklearn version {version} < 1.3.0")
            return False
            
    except ImportError as e:
        logger.failure(f"sklearn import failed: {e}")
        return False


def test_empty_tsv_handling(output_dir, logger):
    """Test 10: Verify empty TSV never generated"""
    logger.test_start("Empty TSV Prevention")
    
    final_report = output_dir / "final_merged_report.tsv"
    if not final_report.exists():
        logger.warning("No TSV file to check")
        return True
    
    size = final_report.stat().st_size
    
    # Header line alone is ~100 bytes, so <50 = truly empty
    if size < 50:
        logger.failure(f"Empty TSV detected: {final_report.name}")
        return False
    
    # Verify has at least header
    try:
        import pandas as pd
        df = pd.read_csv(final_report, sep='\t')
        if 'Image_A' not in df.columns:
            logger.failure(f"Invalid TSV structure: missing Image_A column")
            return False
        
        logger.success(f"{final_report.name}: {size} bytes, {len(df)} rows")
        return True
    except Exception as e:
        logger.failure(f"TSV parse error: {e}")
        return False


def test_performance_benchmarks(output_dir, config, logger):
    """Test 11: Performance regression detection"""
    logger.test_start(f"Performance Benchmarks: {config['name']}")
    
    metadata_file = output_dir / "RUN_METADATA.json"
    if not metadata_file.exists():
        logger.warning("No metadata to check performance")
        return True
    
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    runtime = metadata.get('runtime_seconds', 0)
    expected_runtime = config.get('expected_runtime', None)
    
    if expected_runtime is None:
        logger.info(f"Runtime: {runtime:.1f}s (no baseline)")
        return True
    
    # Allow 20% variance
    tolerance = expected_runtime * 0.20
    min_acceptable = expected_runtime - tolerance
    max_acceptable = expected_runtime + tolerance
    
    if runtime < min_acceptable:
        logger.success(f"🚀 Performance improvement! {runtime:.1f}s (baseline: {expected_runtime:.1f}s)")
    elif runtime > max_acceptable:
        logger.warning(f"⚠️  Performance regression: {runtime:.1f}s (expected: {expected_runtime:.1f}s, +{((runtime/expected_runtime - 1) * 100):.1f}%)")
    else:
        logger.success(f"Performance within tolerance: {runtime:.1f}s (baseline: {expected_runtime:.1f}s)")
    
    return True


def test_visual_comparison_quality(output_dir, logger):
    """Test 12: Verify visual comparisons are valid images"""
    logger.test_start("Visual Comparison Quality")
    
    comp_dir = output_dir / "duplicate_comparisons"
    if not comp_dir.exists():
        logger.info("No visual comparisons (no duplicates or disabled)")
        return True
    
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.warning("PIL/numpy not available for validation")
        return True
    
    # Check comparison images
    images = list(comp_dir.glob("pair_*.png"))
    if not images:
        logger.info("No comparison images found")
        return True
    
    valid_count = 0
    
    for img_path in images[:5]:  # Sample first 5
        try:
            img = Image.open(img_path)
            width, height = img.size
            
            # Sanity checks
            if width < 100 or height < 100:
                logger.failure(f"{img_path.name}: Too small ({width}×{height})")
                continue
            
            if img.mode not in ['RGB', 'RGBA', 'L']:
                logger.failure(f"{img_path.name}: Invalid mode ({img.mode})")
                continue
            
            # Check it's not blank
            arr = np.array(img)
            if arr.std() < 1:  # Nearly uniform
                logger.failure(f"{img_path.name}: Appears blank (std={arr.std():.2f})")
                continue
            
            valid_count += 1
            logger.success(f"{img_path.name}: Valid ({width}×{height}, {img.mode})")
            
        except Exception as e:
            logger.failure(f"{img_path.name}: {e}")
    
    if valid_count == 0 and len(images) > 0:
        logger.failure("No valid comparison images")
        return False
    
    logger.success(f"{valid_count}/{min(len(images), 5)} sampled images valid")
    return True


# ============================================================================
# TEST HISTORY & REPORTING
# ============================================================================

def get_git_commit():
    """Get current git commit hash"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"


def track_test_run(logger):
    """Append test results to history"""
    history_file = Path("test_history.json")
    
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding='utf-8'))
        except:
            history = []
    
    elapsed = (datetime.now() - logger.start_time).total_seconds()
    
    history.append({
        "timestamp": datetime.now().isoformat(),
        "tests_run": logger.tests_run,
        "tests_passed": logger.tests_passed,
        "tests_failed": logger.tests_failed,
        "runtime_seconds": elapsed,
        "git_commit": get_git_commit(),
    })
    
    # Keep last 50 runs
    history = history[-50:]
    
    try:
        history_file.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f"📈 Test history updated: {len(history)} runs tracked")
    except Exception as e:
        print(f"⚠️  Could not update test history: {e}")


def generate_test_summary(logger):
    """Generate markdown summary of test results"""
    output_file = TEST_OUTPUT_DIR / "test_summary.txt"
    
    elapsed = (datetime.now() - logger.start_time).total_seconds()
    success_rate = (100 * logger.tests_passed / max(logger.tests_run, 1))
    
    summary = f"""
# 🧪 Test Summary

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** {'✅ PASSED' if logger.tests_failed == 0 else '❌ FAILED'}

## Results
- Total Tests: {logger.tests_run}
- Passed: {logger.tests_passed} ✅
- Failed: {logger.tests_failed} ❌
- Success Rate: {success_rate:.1f}%

## Performance
- Runtime: {elapsed:.1f}s

## Artifacts
Test outputs saved to: `{TEST_OUTPUT_DIR}/`

---
*Generated by test_pipeline_auto.py*
"""
    
    try:
        output_file.write_text(summary, encoding='utf-8')
        print(f"\n📄 Summary saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save summary: {e}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Execute all test cases"""
    logger = TestLogger()
    
    print("\n" + "="*70)
    print("🚀 AUTOMATED PIPELINE TEST SUITE")
    print("="*70)
    print(f"Test PDF: {TEST_PDF_PATH}")
    print(f"Output Dir: {TEST_OUTPUT_DIR}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    # Cleanup
    logger.info("Cleaning up previous test outputs...")
    cleanup_test_output()
    
    # Test 1: Prerequisites
    if not test_prerequisites(logger):
        logger.failure("Prerequisites check failed - aborting")
        return logger.summary()
    
    # Check test PDF exists
    if not TEST_PDF_PATH.exists():
        logger.failure(f"Test PDF not found: {TEST_PDF_PATH}")
        return logger.summary()
    
    # Test 9: sklearn import (deployment fix)
    test_sklearn_import(logger)
    
    # Run tests for each config
    for config in TEST_CONFIGS:
        
        # Test 2: Run pipeline
        success, output_dir = test_pipeline_run(TEST_PDF_PATH, config, logger)
        if not success:
            continue
        
        # Test 3-8: Validate outputs
        test_output_structure(output_dir, logger)
        test_pages_extraction(output_dir, logger)
        test_panel_detection(output_dir, logger)
        test_duplicate_detection(output_dir, config, logger)
        test_tier_classification(output_dir, logger)
        test_metadata_integrity(output_dir, logger)
        
        # Test 10-12: Advanced validation
        test_empty_tsv_handling(output_dir, logger)
        test_performance_benchmarks(output_dir, config, logger)
        test_visual_comparison_quality(output_dir, logger)
    
    # Generate reports
    generate_test_summary(logger)
    track_test_run(logger)
    
    # Final summary
    return logger.summary()


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

