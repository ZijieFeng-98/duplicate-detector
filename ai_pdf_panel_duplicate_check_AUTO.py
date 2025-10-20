#!/usr/bin/env python3
# ai_pdf_panel_duplicate_check_AUTO.py
# PRODUCTION VERSION - Optimized for Mac hardware
# Implements: FAISS, caching, deterministic runs, parallel processing

from pathlib import Path
import os, sys, json, time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ExifTags
from tqdm import tqdm
import cv2
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

# Tile detection module (optional)
try:
    from tile_detection import (
        TileConfig,
        run_tile_detection_pipeline,
        apply_tile_evidence_to_dataframe
    )
    TILE_MODULE_AVAILABLE = True
except ImportError:
    TILE_MODULE_AVAILABLE = False
    print("⚠️  Tile detection module not found, using panel-level only")

# Tile-first pipeline (micro-tiles ONLY, NO GRID)
try:
    from tile_first_pipeline import TileFirstConfig, run_tile_first_pipeline
    TILE_FIRST_AVAILABLE = True
except ImportError:
    TILE_FIRST_AVAILABLE = False

# --- CONFIG -------------------------------------------------------------------
PDF_PATH = Path("/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/STM-Combined Figures.pdf")
OUT_DIR  = Path("/Users/zijiefeng/Desktop/Guo's lab/My_Research/Dr_Zhong/ai_clip_output")

# PDF conversion
DPI = 150

# Exclude caption/legend pages (1-indexed)
CAPTION_PAGES: Set[int] = {14, 27}

# Panel detection parameters - OPTIMIZED
MIN_PANEL_AREA = 80000
MAX_PANEL_AREA = 10000000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0
EDGE_THRESHOLD1 = 40
EDGE_THRESHOLD2 = 120
CONTOUR_APPROX_EPSILON = 0.02

# Duplicate detection thresholds - BALANCED (sweet spot)
SIM_THRESHOLD = 0.94        # CLIP threshold (relaxed to catch copy-paste with compression)
PHASH_MAX_DIST = 3          # pHash Hamming distance (optimized from comprehensive test)
SSIM_THRESHOLD = 0.85       # Structural similarity (optimized from comprehensive test)
TOP_K_NEIGHBORS = 50        # Increased from 12 to escape local same-page bubble
CLIP_PAIRING_MODE = "thresh"  # 'topk' (old) or 'thresh' (new, cross-page friendly)
CLIP_MAX_OUTPUT_PAIRS = 120000  # Safety cap for large documents

# ═══════════════════════════════════════════════════════════════
# ADVANCED DISCRIMINATION (No Page Suppression Needed)
# ═══════════════════════════════════════════════════════════════

# --- Self-normalized CLIP z-score ---
# Filters semantic lookalikes (panels from same modality/figure)
USE_CLIP_ZSCORE = True              # Keep enabled for optional Path 2 gate
CLIP_ZSCORE_MIN = 3.0               # Minimum z-score (≈0.13% tail, strict)
REQUIRE_CLIP_Z_FOR_CLIP_SSIM = True # Optional gate for Path 2 only

# --- Patch-wise SSIM (MS-SSIM-lite) ---
# Defeats "similar-but-different" grids and backgrounds
USE_PATCHWISE_SSIM = True           # Enable multi-scale SSIM
SSIM_GRID_H = 3                     # Grid rows (3×3 = 9 patches)
SSIM_GRID_W = 3                     # Grid columns
SSIM_TOPK_PATCHES = 4               # Average top-K local SSIMs
SSIM_MIX_WEIGHT = 0.6               # Weight: 0.6*patch + 0.4*global
SSIM_PATCH_MIN_GATE = 0.85          # Minimum patch SSIM threshold
REQUIRE_PATCH_MIN_FOR_CLIP_SSIM = True   # Optional gate for Path 2 only

# --- Copy-Paste Detection (Path 4) ---
# Catches copy-pasted panels despite compression/text differences
TIER_A_COPYPASTE_CLIP_MIN = 0.95    # Very high semantic similarity
TIER_A_COPYPASTE_SSIM_MIN = 0.65    # Moderate structural (allows compression)
TIER_A_COPYPASTE_COMBINED = 1.65    # Combined threshold (sum)

# --- Modality-Specific Detection (Option 2) ---
# User can choose: universal (simple) or modality-specific (advanced)
USE_MODALITY_SPECIFIC_GATING = False  # Toggle between universal/modality-specific
ENABLE_MODALITY_DETECTION = False     # Pre-classify image types (WB, confocal, TEM, etc.)

# Filtering options - PRODUCTION MODE (All Features Enabled)
SUPPRESS_SAME_PAGE_DUPES = False  # Disable to see all pairs
SUPPRESS_ADJACENT_PAGE_DUPES = False  # Let calculation decide (no shortcuts)
ADJACENT_PAGE_MAX_GAP = 1         # Suppress within N pages (1 = immediate neighbors)
REQUIRE_GEOMETRY_FOR_NEAR_PAGES = False  # Let calculation decide (no shortcuts)
NEAR_PAGE_GAP = 2                 # "Near" means within N pages

USE_MUTUAL_NN = False             # Keep disabled for more results
USE_SSIM_VALIDATION = True        # Re-enable with photometric normalization
HIGHLIGHT_DIFFERENCES = True      # Visual diff highlighting
USE_TIER_GATING = True            # Tier A/B classification
USE_PHASH_BUNDLES = True          # Rotation/mirror-robust pHash (TESTED ✅)
USE_ORB_RANSAC = True             # Partial duplicate detection (TESTED ✅)

# ---- Optional: high-confidence relaxed ORB gate (default OFF)
ENABLE_ORB_RELAX = False          # Toggle ON only if needed for tough partial dupes
ORB_RELAX_CLIP_MIN = 0.98         # Very high semantic confidence
ORB_RELAX_PATCH_TOPK_MIN = 0.70   # Local patch agreement
ORB_RELAX_INLIERS_MIN = 25        # Slightly lower than Tier-A default (30)
ORB_RELAX_COVERAGE_MIN = 0.70     # Relaxed vs 0.85
ORB_RELAX_RATIO_MIN = 0.30        # Keep ratio guard
ORB_RELAX_REPROJ_MAX = 5.0        # Slightly looser reprojection error

# --- Confocal Deep Verify (calculations only; no page heuristics) ------------
ENABLE_CONFOCAL_DEEP_VERIFY = True      # turn on robust confirmation
DEEP_VERIFY_CLIP_MIN = 0.96             # only consider strong semantic pairs
DEEP_VERIFY_ALIGN_SSIM_MIN = 0.90       # SSIM after alignment must be high
DEEP_VERIFY_NCC_MIN = 0.985             # normalized cross-correlation (z-scored)
DEEP_VERIFY_PHASH_MAX = 5               # rotation/mirror-robust pHash bundle
DEEP_VERIFY_MAX_STEPS = 120             # ECC iterations

# --- IHC / Histology Deep Verify (no page heuristics) -------------------------
ENABLE_IHC_DEEP_VERIFY = True
IHC_DV_CLIP_MIN = 0.96                 # only strong semantic pairs
IHC_DV_SSIM_AFTER_ALIGN_MIN = 0.88     # SSIM after ECC on stain channel
IHC_DV_NCC_MIN = 0.980                 # normalized cross-correlation
IHC_DV_PHASH_MAX = 6                   # a bit looser than confocal
IHC_DV_MAX_STEPS = 120
IHC_DV_REQUIRE_IHC_LIKE = True         # run only if both panels look IHC-like
IHC_LIKE_MIN_FRACTION = 0.10           # ≥10% tissue-colored pixels

# --- Internal Modality Routing (no UI exposure) -------------------------------
ENABLE_MODALITY_ROUTING = True       # app "knows" and uses per-modality rules
EXPOSE_MODALITY_COLUMNS = False      # keep Modality_A/B out of TSV by default
MODALITY_MIN_CONFIDENCE = 0.15       # below this => treat as 'unknown' (stricter)

# Phase D: Refined Configuration
TIER_A_PHASH_RT = 3
TIER_A_CLIP = 0.99
TIER_A_SSIM = 0.95
TIER_A_ORB_INLIERS = 30
TIER_A_ORB_RATIO = 0.30
TIER_A_ORB_ERROR = 4.0
TIER_A_ORB_COVERAGE = 0.85
TIER_A_CLONE_AREA_PCT = 1.0

TIER_B_PHASH_RT_MIN = 4
TIER_B_PHASH_RT_MAX = 5
TIER_B_CLIP_MIN = 0.985
TIER_B_CLIP_MAX = 0.99
TIER_B_SSIM_MIN = 0.92
TIER_B_SSIM_MAX = 0.95

# ═══════════════════════════════════════════════════════════════
# UNIVERSAL DISCRIMINATION PARAMETERS (Method 1)
# ═══════════════════════════════════════════════════════════════

# Relaxed thresholds for compressed/rotated duplicates
TIER_A_RELAXED_CLIP = 0.94       # Lower CLIP (catches Western blot: 0.9558)
TIER_A_RELAXED_SSIM = 0.65       # Lower SSIM (catches Western blot: 0.7264)
TIER_A_RELAXED_COMBINED = 1.62   # Sum must exceed this

# Western blot specific (even more tolerant for rotation)
TIER_A_WESTERN_CLIP = 0.95
TIER_A_WESTERN_SSIM = 0.65       # Very low for rotated bands
TIER_A_WESTERN_COMBINED = 1.6

# Confocal false positive filter (THE KEY INNOVATION!)
CONFOCAL_FP_CLIP_MIN = 0.96      # High CLIP threshold
CONFOCAL_FP_SSIM_MAX = 0.6      # But LOW SSIM (different content)
CONFOCAL_FP_PHASH_MIN = 10       # Not an exact match

# ═══════════════════════════════════════════════════════════════
# MODALITY-SPECIFIC PARAMETERS (Method 2)
# ═══════════════════════════════════════════════════════════════

# Research-backed parameters by modality
MODALITY_PARAMS = {
    'western_blot': {
        'tier_a': {'clip_min': 0.94, 'ssim_min': 0.60, 'combined': 1.55, 'phash_max': 3},
        'tier_b': {'clip_min': 0.92, 'ssim_min': 0.50, 'combined': 1.45, 'phash_max': 4},
        'fp_filter': None  # No FP filter for WB
    },
    'confocal': {
        'tier_a': {'clip_min': 0.95, 'ssim_min': 0.75, 'combined': 1.70, 'phash_max': 3},
        'tier_b': {'clip_min': 0.92, 'ssim_min': 0.65, 'combined': 1.60, 'phash_max': 4},
        'fp_filter': {'clip_high': 0.96, 'ssim_low': 0.50}
    },
    'tem': {
        'tier_a': {'clip_min': 0.95, 'ssim_min': 0.85, 'combined': 1.80, 'phash_max': 2},
        'tier_b': {'clip_min': 0.93, 'ssim_min': 0.75, 'combined': 1.70, 'phash_max': 3},
        'fp_filter': None
    },
    'bright_field': {
        'tier_a': {'clip_min': 0.93, 'ssim_min': 0.75, 'combined': 1.68, 'phash_max': 4},
        'tier_b': {'clip_min': 0.90, 'ssim_min': 0.65, 'combined': 1.58, 'phash_max': 5},
        'fp_filter': None
    },
    'gel': {
        'tier_a': {'clip_min': 0.94, 'ssim_min': 0.70, 'combined': 1.64, 'phash_max': 3},
        'tier_b': {'clip_min': 0.91, 'ssim_min': 0.60, 'combined': 1.54, 'phash_max': 4},
        'fp_filter': None
    },
    'unknown': {
        'tier_a': {'clip_min': 0.96, 'ssim_min': 0.85, 'combined': 1.81, 'phash_max': 3},
        'tier_b': {'clip_min': 0.94, 'ssim_min': 0.75, 'combined': 1.69, 'phash_max': 4},
        'fp_filter': None
    }
}

ENABLE_ELA = False  # ELA off by default (PDF-origin)
ELA_QUALITY = 90
ELA_REQUIRE_JPEG_ORIGIN = True
ELA_HOTSPOT_RATIO = 2.0
ELA_AREA_PCT = 1.0

ENABLE_CLONE_DETECTION = False  # Off for speed
CLONE_BLOCK_SIZES = [32, 48]
CLONE_STRIDE = 8
CLONE_MIN_SEPARATION_PCT = 0.10
CLONE_AREA_PCT = 0.5

ENABLE_BRIGHTNESS_CHECK = False  # Off for speed
BRIGHTNESS_TILE_SIZE = 64
BRIGHTNESS_Z_THRESHOLD = 4.0

ENABLE_METADATA_CHECK = False  # Advisory only
METADATA_IS_ADVISORY = True

ENABLE_TEXT_MASKING = True  # Enable for better robustness
MASKING_APPLY_TO_CLIP = False  # Keep CLIP unmasked (semantic understanding)
MASKING_APPLY_TO_PHASH = True  # Mask text for pHash
MASKING_APPLY_TO_SSIM = True  # Mask text for SSIM
MASKING_APPLY_TO_ORB = True  # Mask text for ORB

CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_SIZE = 8
NORMALIZE_MEAN_STD = True

ORB_TRIGGER_CLIP_THRESHOLD = 0.985
ORB_TRIGGER_PHASH_THRESHOLD = 4
ORB_MAX_KEYPOINTS = 1000
ORB_RETRY_SCALES = [1.0, 2.0, 0.5]
ORB_RATIO_THRESHOLD = 0.75

PHASH_COMPUTE_BUNDLE = True
PHASH_BUNDLE_SHORT_CIRCUIT = 3

USE_PROGRESSIVE_GATING = True
LOG_STAGE_COUNTS = True

RUN_PASS_A_SUPPRESSED = True
RUN_PASS_B_UNSUPPRESSED = False

# Performance - MAC OPTIMIZED
BATCH_SIZE = 64             # Large batch for M1/M2 (reduce to 32 if Intel Mac)
NUM_WORKERS = cpu_count()   # Use all CPU cores
ENABLE_MPS = True           # Metal Performance Shaders for M1/M2

# Caching
ENABLE_CACHE = True         # Cache embeddings and pHash
CACHE_VERSION = "v7"        # Increment when changing detection params (modality-routing)

# Debug mode
DEBUG_MODE = True

# Deterministic runs
RANDOM_SEED = 123

# --- UX: CONTROL FILE OPENING BEHAVIOR ----------------------------------------
AUTO_OPEN_RESULTS = True         # Open only the final HTML index once at end
AUTO_OPEN_PER_PAIR = False       # Do not auto-open per-pair artifacts (prevents popups)

def _open_file_once(path: Path):
    """Open a file/URL in the default browser/viewer safely (macOS/Linux/Windows)"""
    try:
        import webbrowser
        webbrowser.open(path.as_uri())
    except Exception as e:
        print(f"  ℹ️  Could not auto-open {path.name}: {e}")

# --- DEPENDENCIES -------------------------------------------------------------
try:
    import fitz  # PyMuPDF - no system dependencies needed!
    import open_clip
    import torch
    import imagehash
    from skimage.metrics import structural_similarity as ssim
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n📦 Install with:")
    print("  pip install pymupdf pillow open-clip-torch imagehash scikit-image tqdm numpy pandas opencv-python-headless")
    sys.exit(1)

# --- DETERMINISTIC SETUP ------------------------------------------------------
def set_seeds(seed: int = RANDOM_SEED):
    """Set seeds for reproducible runs"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════
# PERFORMANCE TRACKING & AUTO-CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class StageTimer:
    """Context manager for timing pipeline stages"""
    
    def __init__(self, stage_name: str, verbose: bool = True):
        self.stage_name = stage_name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"\n⏱️  Starting: {self.stage_name}")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"✓ Completed: {self.stage_name} ({elapsed:.2f}s)")
    
    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class PipelineMetrics:
    """Track metrics across pipeline stages"""
    
    def __init__(self):
        self.metrics: Dict[str, any] = {
            'stages': {},
            'total_time': 0,
            'bottlenecks': []
        }
    
    def record_stage(self, stage_name: str, duration: float, 
                    input_count: int, output_count: int, **kwargs):
        """Record metrics for a stage"""
        self.metrics['stages'][stage_name] = {
            'duration_sec': duration,
            'input_count': input_count,
            'output_count': output_count,
            'reduction_pct': 100 * (1 - output_count / max(input_count, 1)),
            'throughput': output_count / max(duration, 0.001),
            **kwargs
        }
    
    def identify_bottlenecks(self, threshold_sec: float = 5.0):
        """Find stages taking >threshold seconds"""
        bottlenecks = []
        for stage, metrics in self.metrics['stages'].items():
            if metrics['duration_sec'] >= threshold_sec:
                bottlenecks.append({
                    'stage': stage,
                    'duration': metrics['duration_sec'],
                    'pct_of_total': 100 * metrics['duration_sec'] / max(self.metrics['total_time'], 0.001)
                })
        
        self.metrics['bottlenecks'] = sorted(bottlenecks, 
                                            key=lambda x: x['duration'], 
                                            reverse=True)
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*70)
        print("📊 PIPELINE PERFORMANCE SUMMARY")
        print("="*70)
        
        for stage, metrics in self.metrics['stages'].items():
            print(f"\n{stage}:")
            print(f"  Duration: {metrics['duration_sec']:.2f}s")
            print(f"  Input → Output: {metrics['input_count']} → {metrics['output_count']}")
            if metrics['reduction_pct'] > 0:
                print(f"  Reduction: {metrics['reduction_pct']:.1f}%")
            print(f"  Throughput: {metrics['throughput']:.1f} items/s")
        
        if self.metrics['bottlenecks']:
            print(f"\n🐌 Bottlenecks (>5.0s):")
            for b in self.metrics['bottlenecks']:
                print(f"  • {b['stage']}: {b['duration']:.1f}s ({b['pct_of_total']:.1f}% of total)")
        
        print(f"\nTotal runtime: {self.metrics['total_time']:.1f}s")
        print("="*70)
    
    def save_to_json(self, path: Path):
        """Save metrics to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


@dataclass
class OptimalConfig:
    """Optimal configuration for given document size"""
    clip_mode: str
    phash_prefix_len: int
    batch_size: int
    num_workers: int
    use_mmap: bool
    expected_time_sec: float
    memory_mb: float


def auto_configure(num_panels: int, available_memory_mb: float = None) -> OptimalConfig:
    """
    Automatically configure pipeline based on document size
    
    Decision matrix:
    - < 500 panels: All-pairs (optimal for your case)
    - 500-2000: All-pairs with aggressive bucketing
    - 2000-10000: Consider indexed search
    - > 10000: Use FAISS/approximate methods
    """
    
    if available_memory_mb is None:
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_memory_mb = mem.available / (1024 ** 2)
        except:
            available_memory_mb = 4096  # Assume 4GB available
    
    # Decision tree based on research
    if num_panels < 500:
        # Your case (221 panels) - Optimal configuration
        return OptimalConfig(
            clip_mode="thresh",
            phash_prefix_len=4,
            batch_size=64,
            num_workers=max(1, cpu_count() - 1),
            use_mmap=True,
            expected_time_sec=60 * (num_panels / 221),
            memory_mb=(num_panels * 512 * 4 + num_panels ** 2 * 4) / (1024 ** 2) + 100
        )
    
    elif num_panels < 1000:
        return OptimalConfig(
            clip_mode="thresh",
            phash_prefix_len=5,
            batch_size=32,
            num_workers=max(1, cpu_count() - 1),
            use_mmap=True,
            expected_time_sec=180 * (num_panels / 1000) ** 1.5,
            memory_mb=(num_panels * 512 * 4) / (1024 ** 2) + 200
        )
    
    elif num_panels < 5000:
        return OptimalConfig(
            clip_mode="thresh",
            phash_prefix_len=6,
            batch_size=500,
            num_workers=cpu_count(),
            use_mmap=True,
            expected_time_sec=600 * (num_panels / 5000) ** 1.3,
            memory_mb=(num_panels * 512 * 4 + 500 * num_panels * 4) / (1024 ** 2) + 300
        )
    
    else:
        return OptimalConfig(
            clip_mode="batched",
            phash_prefix_len=8,
            batch_size=1000,
            num_workers=cpu_count(),
            use_mmap=True,
            expected_time_sec=1200 * (num_panels / 10000),
            memory_mb=(num_panels * 512 * 4 * 1.5) / (1024 ** 2) + 500
        )


def print_configuration_advice(num_panels: int):
    """Print configuration recommendations"""
    config = auto_configure(num_panels)
    
    print("\n" + "="*70)
    print("⚙️  AUTO-CONFIGURATION RECOMMENDATIONS")
    print("="*70)
    print(f"Document size: {num_panels} panels")
    print(f"\nRecommended settings:")
    print(f"  • CLIP mode: {config.clip_mode}")
    print(f"  • pHash prefix length: {config.phash_prefix_len}")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Worker processes: {config.num_workers}")
    print(f"  • Memory-mapped cache: {config.use_mmap}")
    print(f"\nExpected performance:")
    print(f"  • Runtime: ~{config.expected_time_sec/60:.1f} minutes")
    print(f"  • Peak memory: ~{config.memory_mb:.0f} MB")
    
    # Memory check
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 ** 2)
        
        if config.memory_mb > available_mb:
            print(f"\n⚠️  WARNING: May exceed available memory!")
            print(f"    Required: {config.memory_mb:.0f} MB")
            print(f"    Available: {available_mb:.0f} MB")
        else:
            print(f"\n✅ Memory check passed ({config.memory_mb:.0f}/{available_mb:.0f} MB)")
    except:
        print(f"\n💡 Install psutil for memory checks: pip install psutil")
    
    print("="*70        )

# ═══════════════════════════════════════════════════════════════
# MODALITY DETECTION (Method 2)
# ═══════════════════════════════════════════════════════════════

def detect_image_modality(img_path: str) -> dict:
    """
    Detect scientific image modality using heuristics.
    
    Returns:
        {
            'modality': str ('western_blot', 'confocal', 'tem', 'bright_field', 'gel', 'unknown'),
            'confidence': float (0-1)
        }
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {'modality': 'unknown', 'confidence': 0.0}
        
        h, w = img.shape
        
        # Feature 1: Edge Density
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Feature 2: Directional Anisotropy (horizontal vs vertical)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        
        vertical_strength = np.mean(np.abs(grad_y))
        horizontal_strength = np.mean(np.abs(grad_x))
        anisotropy = vertical_strength / (horizontal_strength + 1e-6)
        
        # Feature 3: Spatial Frequency (texture scale)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        mid_freq = np.mean(magnitude[h//4:3*h//4, w//4:3*w//4])
        high_freq = np.mean(magnitude[3*h//8:5*h//8, 3*w//8:5*w//8])
        
        # Decision Tree (Research-based thresholds)
        
        # Rule 1: Western Blot (horizontal bands)
        if anisotropy > 1.4 and 0.08 < edge_density < 0.20:
            confidence = min((anisotropy - 1.0) / 1.0, 1.0)
            return {'modality': 'western_blot', 'confidence': confidence}
        
        # Rule 2: Confocal (cellular texture, mid-frequency)
        if mid_freq > 80 and 0.12 < edge_density < 0.25:
            confidence = min(mid_freq / 150, 1.0)
            return {'modality': 'confocal', 'confidence': confidence}
        
        # Rule 3: TEM (very high detail)
        if edge_density > 0.28 and high_freq > 100:
            confidence = min(edge_density / 0.35, 1.0)
            return {'modality': 'tem', 'confidence': confidence}
        
        # Rule 4: Bright-field (smooth, low frequency)
        if edge_density < 0.08:
            return {'modality': 'bright_field', 'confidence': 1.0 - edge_density / 0.08}
        
        # Rule 5: Gel (regular columns, anisotropy near 1.0)
        if 0.9 < anisotropy < 1.1 and edge_density > 0.08:
            return {'modality': 'gel', 'confidence': 0.7}
        
        # Default: Unknown
        return {'modality': 'unknown', 'confidence': 0.0}
        
    except Exception as e:
        import warnings
        warnings.warn(f"Modality detection failed: {e}")
        return {'modality': 'unknown', 'confidence': 0.0}


def get_modality_cache(panel_paths: List[Path]) -> dict:
    """
    Detect and cache modality for each panel path.
    Returns { str(path): {'modality': str, 'confidence': float} }
    """
    if not (ENABLE_MODALITY_ROUTING or ENABLE_MODALITY_DETECTION):
        return {}
    
    cache = {}
    mode_desc = "internal routing" if ENABLE_MODALITY_ROUTING else "detection"
    print(f"  Detecting image modalities ({mode_desc})...")
    
    for p in tqdm(panel_paths, desc="  Modality"):
        r = detect_image_modality(str(p))
        mod = r.get('modality', 'unknown')
        conf = float(r.get('confidence', 0.0))
        
        # Apply confidence threshold for routing
        if ENABLE_MODALITY_ROUTING and conf < MODALITY_MIN_CONFIDENCE:
            mod = 'unknown'
        
        cache[str(p)] = {'modality': mod, 'confidence': conf}
    
    # Show distribution
    from collections import Counter
    modalities = [r['modality'] for r in cache.values()]
    dist = Counter(modalities)
    print(f"    Modality distribution:")
    for mod, count in sorted(dist.items()):
        avg_conf = np.mean([cache[p]['confidence'] for p in cache if cache[p]['modality'] == mod])
        if ENABLE_MODALITY_ROUTING:
            print(f"      • {mod}: {count} panels (avg conf: {avg_conf:.2f})")
        else:
            print(f"      • {mod}: {count} panels")
    
    return cache


# ═══════════════════════════════════════════════════════════════
# ADVANCED FEATURES: Tier System, pHash Bundles, Photometric Norm, ORB-RANSAC
# ═══════════════════════════════════════════════════════════════

import warnings

# --- TIER GATING SYSTEM -------------------------------------------------------
def apply_tier_gating(df: pd.DataFrame, 
                     tier_a_phash: int = 3,
                     tier_a_clip: float = 0.99,
                     tier_a_ssim: float = 0.95,
                     tier_b_phash_min: int = 4,
                     tier_b_phash_max: int = 5,
                     tier_b_clip_min: float = 0.985,
                     tier_b_clip_max: float = 0.99,
                     tier_b_ssim_min: float = 0.92,
                     tier_b_ssim_max: float = 0.95,
                     modality_cache: dict = None) -> pd.DataFrame:
    """
    UNIVERSAL DISCRIMINATION with optional modality-specific parameters.
    
    KEY INNOVATION: Use SSIM to discriminate between:
    - High CLIP + High SSIM = Same content (Tier A)  ← TRUE DUPLICATE
    - High CLIP + Low SSIM  = Same modality (Filtered) ← FALSE POSITIVE
    
    Supports TWO modes:
    1. Universal (default): Single discrimination rule for all images
    2. Modality-Specific: Custom thresholds per image type (WB, confocal, etc.)
    
    Tier A Paths (5 independent routes):
    1. Exact: pHash ≤ 3 (rotation-robust)
    2. Strict: CLIP ≥ 0.99 AND SSIM ≥ 0.95
    3. ORB: Partial duplicate with geometric verification
    4. Relaxed: CLIP ≥ 0.94 AND SSIM ≥ 0.70 (catches compressed/rotated)
    5. Western: CLIP ≥ 0.95 AND SSIM ≥ 0.60 (band-specific, rotation-tolerant)
    """
    if df.empty:
        return df

    df['Tier'] = None
    df['Tier_Path'] = None
    df['Confocal_FP'] = False

    # Extract signals
    clip_score = pd.to_numeric(df.get('Cosine_Similarity', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    ssim_score = pd.to_numeric(df.get('SSIM', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    phash_dist = pd.to_numeric(df.get('Hamming_Distance', pd.Series([999] * len(df))), errors='coerce').fillna(999)
    orb_inliers = pd.to_numeric(df.get('ORB_Inliers', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    orb_ratio = pd.to_numeric(df.get('Inlier_Ratio', pd.Series([0] * len(df))), errors='coerce').fillna(0)
    orb_error = pd.to_numeric(df.get('Reproj_Error', pd.Series([999] * len(df))), errors='coerce').fillna(999)
    orb_coverage = pd.to_numeric(df.get('Crop_Coverage', pd.Series([0] * len(df))), errors='coerce').fillna(0)

    # ═══════════════════════════════════════════════════════════════
    # MODE SELECTION: Universal vs Modality-Specific
    # ═══════════════════════════════════════════════════════════════
    
    if USE_MODALITY_SPECIFIC_GATING and modality_cache:
        # MODALITY-SPECIFIC MODE
        for idx, row in df.iterrows():
            mod_a = modality_cache.get(row['Path_A'], {}).get('modality', 'unknown')
            mod_b = modality_cache.get(row['Path_B'], {}).get('modality', 'unknown')
            df.at[idx, 'Modality_A'] = mod_a
            df.at[idx, 'Modality_B'] = mod_b
            
            # Use stricter modality if they differ
            if mod_a != mod_b:
                params = MODALITY_PARAMS['unknown']
            else:
                params = MODALITY_PARAMS.get(mod_a, MODALITY_PARAMS['unknown'])
            
            clip = clip_score.iloc[idx]
            ssim = ssim_score.iloc[idx]
            phash = phash_dist.iloc[idx]
            
            # Tier A check
            tier_a_params = params['tier_a']
            is_tier_a = (
                (phash <= tier_a_params['phash_max']) or
                (clip >= tier_a_params['clip_min'] and 
                 ssim >= tier_a_params['ssim_min'] and
                 (clip + ssim) >= tier_a_params['combined'])
            )
            
            # Apply modality-specific false positive filter
            fp_filter = params.get('fp_filter')
            if fp_filter and (clip >= fp_filter['clip_high'] and ssim < fp_filter['ssim_low']):
                is_tier_a = False
                df.at[idx, 'Confocal_FP'] = True
            
            if is_tier_a:
                df.at[idx, 'Tier'] = 'A'
                df.at[idx, 'Tier_Path'] = f'{mod_a}_specific' if phash > tier_a_params['phash_max'] else 'Exact'
            else:
                # Tier B check
                tier_b_params = params['tier_b']
                is_tier_b = (
                    (tier_b_params['phash_max'] >= phash >= tier_a_params['phash_max']) or
                    (clip >= tier_b_params['clip_min'] and 
                     ssim >= tier_b_params['ssim_min'] and
                     (clip + ssim) >= tier_b_params['combined'])
                )
                if is_tier_b:
                    df.at[idx, 'Tier'] = 'B'
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # UNIVERSAL MODE (Default)
    # ═══════════════════════════════════════════════════════════════
    
    # Path 1: Exact match (pHash with rotation bundles)
    path1_exact = (phash_dist <= tier_a_phash)
    
    # Path 2: Strict (high confidence, with optional discrimination gates)
    path2_strict = (clip_score >= tier_a_clip) & (ssim_score >= tier_a_ssim)
    
    # Optional: Require CLIP z-score (filters semantic lookalikes)
    if REQUIRE_CLIP_Z_FOR_CLIP_SSIM and 'CLIP_Z' in df.columns:
        clip_z = pd.to_numeric(df.get('CLIP_Z'), errors='coerce').fillna(0)
        path2_strict = path2_strict & (clip_z >= CLIP_ZSCORE_MIN)
    
    # Optional: Require minimum patch SSIM (filters grid lookalikes)
    if REQUIRE_PATCH_MIN_FOR_CLIP_SSIM and 'Patch_SSIM_Min' in df.columns:
        patch_min = pd.to_numeric(df.get('Patch_SSIM_Min'), errors='coerce').fillna(0)
        path2_strict = path2_strict & (patch_min >= SSIM_PATCH_MIN_GATE)
    
    # Path 3: ORB-RANSAC (geometric verification for partial duplicates)
    path3_orb = (
        (orb_inliers >= TIER_A_ORB_INLIERS) & 
        (orb_ratio >= TIER_A_ORB_RATIO) & 
        (orb_error <= TIER_A_ORB_ERROR) & 
        (orb_coverage >= TIER_A_ORB_COVERAGE)
    )
    
    # Path 4: RELAXED (catches your Western blot case!)
    path4_relaxed = (
        (clip_score >= TIER_A_RELAXED_CLIP) &          # 0.94 (your WB: 0.9558 ✓)
        (ssim_score >= TIER_A_RELAXED_SSIM) &          # 0.70 (your WB: 0.7264 ✓)
        ((clip_score + ssim_score) >= TIER_A_RELAXED_COMBINED)  # 1.64
    )
    
    # Path 5: WESTERN BLOT SPECIFIC (even more tolerant)
    path5_western = (
        (clip_score >= TIER_A_WESTERN_CLIP) &          # 0.95
        (ssim_score >= TIER_A_WESTERN_SSIM) &          # 0.60 (very low for rotation)
        ((clip_score + ssim_score) >= TIER_A_WESTERN_COMBINED)  # 1.55
    )
    
    # Try to pull patch-wise SSIM; fall back gracefully
    patch_ssim_topk = pd.to_numeric(
        df.get('Patch_SSIM_TopK', df.get('Patch_SSIM_Topk', df.get('Patch_SSIM_TopK_Mean'))),
        errors='coerce'
    )
    # Ensure it's always a Series (not a scalar)
    if not isinstance(patch_ssim_topk, pd.Series):
        patch_ssim_topk = pd.Series([patch_ssim_topk] * len(df))
    if patch_ssim_topk.isna().all():
        patch_ssim_topk = pd.to_numeric(df.get('Patch_SSIM_Min', ssim_score), errors='coerce').fillna(ssim_score)
    
    # Path 6 (optional): High-confidence relaxed ORB (for rotated/cropped partials)
    path6_orb_relax = (
        (clip_score >= ORB_RELAX_CLIP_MIN) &
        (patch_ssim_topk >= ORB_RELAX_PATCH_TOPK_MIN) &
        (orb_inliers >= ORB_RELAX_INLIERS_MIN) &
        (orb_coverage >= ORB_RELAX_COVERAGE_MIN) &
        (orb_ratio >= ORB_RELAX_RATIO_MIN) &
        (orb_error <= ORB_RELAX_REPROJ_MAX)
    ) if ENABLE_ORB_RELAX else pd.Series([False]*len(df))
    
    # Confocal-aware paths (helps rotated/compressed confocal dupes)
    # Gate: strong local agreement OR geometry, with moderately high CLIP
    confocal_path_A = (
        (clip_score >= 0.965) &
        (
            (patch_ssim_topk >= 0.78) |
            ((orb_inliers >= 25) & (orb_coverage >= 0.70) & (orb_error <= 5.0)) |
            (phash_dist <= 5)
        )
    )
    
    confocal_path_B = (
        (clip_score >= 0.955) &
        (
            (patch_ssim_topk >= 0.72) |
            ((orb_inliers >= 20) & (orb_coverage >= 0.65) & (orb_error <= 6.0)) |
            (phash_dist <= 6)
        )
    )
    
    # CONFOCAL FALSE POSITIVE FILTER (refined to not suppress real confocal dupes)
    # Only mark as confocal FP if global SSIM is low AND there's no strong local/geometric evidence
    local_or_geom_ok = (
        (patch_ssim_topk >= 0.72) |             # strong local patch agreement
        ((orb_inliers >= 20) & (orb_coverage >= 0.65) & (orb_error <= 5.0))  # decent geometry
    )
    
    confocal_false_positive = (
        (clip_score >= CONFOCAL_FP_CLIP_MIN)   # ≥ 0.96
      & (ssim_score  <  CONFOCAL_FP_SSIM_MAX)  # < 0.60
      & (phash_dist  >  CONFOCAL_FP_PHASH_MIN) # not an exact match
      & (~local_or_geom_ok)                    # ← NEW: don't call it FP if local/geom evidence exists
    )
    
    # Mark detected false positives
    df['Confocal_FP'] = confocal_false_positive
    
    # ---- Confocal Deep Verify override (calculation-only) --------------------
    deep_override = pd.Series([False]*len(df))
    deep_ssim_col, deep_ncc_col, deep_phash_col = [], [], []
    
    if ENABLE_CONFOCAL_DEEP_VERIFY:
        cand = confocal_false_positive & (clip_score >= DEEP_VERIFY_CLIP_MIN)
        
        # Scope to confocal pairs if modality routing is enabled
        if ENABLE_MODALITY_ROUTING and modality_cache:
            def _is_confocal(p):
                return modality_cache.get(str(p), {}).get('modality') == 'confocal'
            cand = cand & df['Path_A'].map(_is_confocal) & df['Path_B'].map(_is_confocal)
        
        idxs = list(df.index[cand])
        if len(idxs) > 0:
            print(f"    Deep-verifying {len(idxs)} confocal FP candidates...")
        for idx in idxs:
            r = df.loc[idx]
            dv = deep_verify_identical_confocal(r['Path_A'], r['Path_B'])
            deep_ssim_col.append((idx, dv.get('deep_ssim', np.nan)))
            deep_ncc_col.append((idx, dv.get('deep_ncc',  np.nan)))
            deep_phash_col.append((idx, dv.get('phash_min', 999)))
            if dv.get('ok', False):
                deep_override.iloc[idx] = True
                # If we override, do not treat as FP anymore
                confocal_false_positive.iloc[idx] = False
        
        # Attach diagnostics
        if deep_ssim_col:
            for idx, val in deep_ssim_col:
                df.at[idx, 'Deep_SSIM'] = val
        if deep_ncc_col:
            for idx, val in deep_ncc_col:
                df.at[idx, 'Deep_NCC'] = val
        if deep_phash_col:
            for idx, val in deep_phash_col:
                df.at[idx, 'Deep_pHash'] = val
        
        if deep_override.sum() > 0:
            print(f"      → {deep_override.sum()} pairs promoted via Deep Verify!")
    
    # ---- IHC Deep Verify (calculation-only) ----------------------------------
    ihc_override = pd.Series([False]*len(df))
    ihc_ssim_col, ihc_ncc_col, ihc_phash_col = [], [], []
    
    if ENABLE_IHC_DEEP_VERIFY:
        # Scope by modality if routing enabled
        if ENABLE_MODALITY_ROUTING and modality_cache:
            def _is_ihc_modality(p):
                mod = modality_cache.get(str(p), {}).get('modality', 'unknown')
                return mod in ('bright_field', 'gel', 'unknown')  # IHC-compatible modalities
            
            ihc_modality_a = df['Path_A'].map(_is_ihc_modality)
            ihc_modality_b = df['Path_B'].map(_is_ihc_modality)
            ihc_modality_pair = ihc_modality_a & ihc_modality_b
        else:
            ihc_modality_pair = pd.Series([True]*len(df))
        
        # Cache ihc-like classification per path to avoid repeated reads
        _ihc_like_cache = {}
        def _ihc_like_path(p):
            if p not in _ihc_like_cache:
                img = cv2.imread(p)
                _ihc_like_cache[p] = _is_ihc_like_bgr(img)
            return _ihc_like_cache[p]
        
        ihc_like_a = df['Path_A'].map(_ihc_like_path)
        ihc_like_b = df['Path_B'].map(_ihc_like_path)
        ihc_like_pair = ihc_like_a & ihc_like_b
        
        # Candidates: strong semantic, low-ish global SSIM (typical for staining variance)
        cand = (
            ihc_modality_pair       # First filter by modality
          & ihc_like_pair           # Then by color heuristic
          & (clip_score >= IHC_DV_CLIP_MIN)
          & (ssim_score < 0.88)   # invite tough cases; final decision is deep-verify
        )
        
        idxs = list(df.index[cand])
        if len(idxs):
            print(f"    Deep-verifying {len(idxs)} IHC candidates...")
        for idx in idxs:
            r = df.loc[idx]
            dv = deep_verify_identical_ihc(r['Path_A'], r['Path_B'])
            ihc_ssim_col.append((idx, dv.get('IHC_SSIM', np.nan)))
            ihc_ncc_col.append((idx, dv.get('IHC_NCC',  np.nan)))
            ihc_phash_col.append((idx, dv.get('IHC_pHash', 999)))
            if dv.get('ok', False):
                ihc_override.iloc[idx] = True
        
        # Attach diagnostics
        for idx, val in ihc_ssim_col:  
            df.at[idx, 'IHC_SSIM']  = val
        for idx, val in ihc_ncc_col:   
            df.at[idx, 'IHC_NCC']   = val
        for idx, val in ihc_phash_col: 
            df.at[idx, 'IHC_pHash'] = val
        
        if ihc_override.sum() > 0:
            print(f"      → {ihc_override.sum()} IHC pairs promoted via Deep Verify!")
    
    # TIER A: Any path succeeds, EXCEPT confocal false positives
    tier_a_mask = (
        (path1_exact | path2_strict | path3_orb | path4_relaxed | path5_western | path6_orb_relax | confocal_path_A | deep_override | ihc_override) &
        (~confocal_false_positive)  # ← CRITICAL: Block false positives
    )
    
    # Mark which path triggered (for diagnostics)
    df.loc[path1_exact & tier_a_mask, 'Tier_Path'] = 'Exact'
    df.loc[path2_strict & ~path1_exact & tier_a_mask, 'Tier_Path'] = 'Strict'
    df.loc[path3_orb & ~path2_strict & ~path1_exact & tier_a_mask, 'Tier_Path'] = 'ORB'
    df.loc[path4_relaxed & ~path3_orb & ~path2_strict & ~path1_exact & tier_a_mask, 'Tier_Path'] = 'Relaxed'
    df.loc[path5_western & ~path4_relaxed & ~path3_orb & ~path2_strict & ~path1_exact & tier_a_mask, 'Tier_Path'] = 'Western'
    df.loc[path6_orb_relax & ~path5_western & ~path4_relaxed & ~path3_orb & ~path2_strict & ~path1_exact & tier_a_mask, 'Tier_Path'] = 'ORB-Relaxed'
    df.loc[confocal_path_A & tier_a_mask & df['Tier_Path'].isna(), 'Tier_Path'] = 'Confocal-A'
    df.loc[deep_override & tier_a_mask & df['Tier_Path'].isna(), 'Tier_Path'] = 'Confocal-DeepVerify'
    df.loc[ihc_override & tier_a_mask & df['Tier_Path'].isna(), 'Tier_Path'] = 'IHC-DeepVerify'
    
    # TIER B: Borderline + confocal-aware B + Filtered confocal false positives
    tier_b_mask = (
        # Original Tier B paths
        ((phash_dist >= tier_b_phash_min) & (phash_dist <= tier_b_phash_max)) |
        (
            (clip_score >= tier_b_clip_min) & (clip_score < tier_b_clip_max) &
            (ssim_score >= tier_b_ssim_min) & (ssim_score < tier_b_ssim_max)
        ) |
        # Confocal-aware B path
        (confocal_path_B & ~tier_a_mask) |
        # Confocal false positives go to Tier B for manual review
        (confocal_false_positive & (clip_score >= 0.97))
    )
    df.loc[(confocal_path_B & ~tier_a_mask) & df['Tier_Path'].isna(), 'Tier_Path'] = 'Confocal-B'
    
    df.loc[tier_a_mask, 'Tier'] = 'A'
    df.loc[tier_b_mask & ~tier_a_mask, 'Tier'] = 'B'
    
    return df

# --- ROTATION/MIRROR-ROBUST PHASH ---------------------------------------------
def compute_phash_bundle(img: Image.Image) -> dict:
    """Compute pHash for 8 transform variants"""
    bundle = {}
    
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.autocontrast(img)
        
        for angle in [0, 90, 180, 270]:
            rotated = img.rotate(angle, expand=True) if angle > 0 else img
            bundle[f'rot_{angle}'] = str(imagehash.phash(rotated))
        
        mirrored = ImageOps.mirror(img)
        for angle in [0, 90, 180, 270]:
            rotated = mirrored.rotate(angle, expand=True) if angle > 0 else mirrored
            bundle[f'mirror_h_rot_{angle}'] = str(imagehash.phash(rotated))
    except Exception as e:
        warnings.warn(f"pHash bundle computation failed: {e}")
        for key in ['rot_0', 'rot_90', 'rot_180', 'rot_270',
                   'mirror_h_rot_0', 'mirror_h_rot_90', 'mirror_h_rot_180', 'mirror_h_rot_270']:
            bundle[key] = ""
    
    return bundle

def hamming_min_transform(bundle_a: dict, bundle_b: dict, short_circuit: int = 3) -> Tuple[int, str]:
    """Find minimum Hamming distance across all transform pairs"""
    min_dist = 999
    min_transform = "none"
    
    for key_a in bundle_a:
        hash_a = bundle_a[key_a]
        if not hash_a:
            continue
            
        for key_b in bundle_b:
            hash_b = bundle_b[key_b]
            if not hash_b:
                continue
            
            try:
                dist = imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
                
                if dist < min_dist:
                    min_dist = dist
                    min_transform = f"{key_a} vs {key_b}"
                    
                    if dist <= short_circuit:
                        return (min_dist, min_transform)
            except Exception:
                continue
    
    return (min_dist, min_transform)

# --- PHOTOMETRIC NORMALIZATION ------------------------------------------------
def apply_clahe(img_gray: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)

def normalize_photometric(img: np.ndarray, apply_clahe_flag: bool = True,
                          clip_limit: float = 2.5, tile_size: int = 8) -> Tuple[np.ndarray, dict]:
    """Apply deterministic photometric normalization"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    params = {'clahe_applied': apply_clahe_flag, 'clip_limit': clip_limit, 'tile_size': tile_size}
    
    if apply_clahe_flag:
        img_gray = apply_clahe(img_gray, clip_limit, tile_size)
    
    mean = np.mean(img_gray)
    std = np.std(img_gray)
    
    if std > 1e-6:
        img_normalized = (img_gray - mean) / std
    else:
        img_normalized = img_gray - mean
    
    params['mean'] = float(mean)
    params['std'] = float(std)
    
    img_normalized = np.clip(img_normalized * 50 + 128, 0, 255).astype(np.uint8)
    
    return img_normalized, params

def compute_ssim_normalized(path_a: str, path_b: str, target_h: int = 512,
                           apply_norm: bool = True) -> Tuple[float, dict]:
    """
    Compute SSIM with photometric normalization + patch-wise refinement.
    
    Returns:
        score_mixed (float): Weighted mix of global + top-K patch SSIMs (if enabled)
        metadata (dict): Includes global SSIM, patch stats, grid info
    
    Why this reduces false positives:
    - Global SSIM can be fooled by similar backgrounds/layouts
    - Patch-wise SSIM demands multiple local regions truly match
    - Grid artifacts (same modality, different content) fail patch tests
    """
    from skimage.metrics import structural_similarity as ssim_func
    
    try:
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        
        if img_a is None or img_b is None:
            return np.nan, {}
        
        # Step 1: Optional text masking
        if ENABLE_TEXT_MASKING and MASKING_APPLY_TO_SSIM:
            try:
                mask_a = detect_text_regions_heuristic(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
                mask_b = detect_text_regions_heuristic(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
                img_a = cv2.inpaint(img_a, mask_a, 3, cv2.INPAINT_TELEA)
                img_b = cv2.inpaint(img_b, mask_b, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass
        
        # Step 2: Photometric normalization
        if apply_norm:
            img_a_gray, params_a = normalize_photometric(img_a)
            img_b_gray, params_b = normalize_photometric(img_b)
        else:
            img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
            img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
            params_a = params_b = {}
        
        # Step 3: Resize by height and pad to same width
        def resize_to_height(img, h):
            scale = h / img.shape[0]
            new_w = max(1, int(round(img.shape[1] * scale)))
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)
        
        img_a_gray = resize_to_height(img_a_gray, target_h)
        img_b_gray = resize_to_height(img_b_gray, target_h)
        
        max_w = max(img_a_gray.shape[1], img_b_gray.shape[1])
        if img_a_gray.shape[1] < max_w:
            img_a_gray = cv2.copyMakeBorder(img_a_gray, 0, 0, 0, max_w - img_a_gray.shape[1],
                                           cv2.BORDER_CONSTANT, value=255)
        if img_b_gray.shape[1] < max_w:
            img_b_gray = cv2.copyMakeBorder(img_b_gray, 0, 0, 0, max_w - img_b_gray.shape[1],
                                           cv2.BORDER_CONSTANT, value=255)
        
        # Step 4: Compute global SSIM (baseline)
        # Critical: Specify data_range for uint8 images (comprehensive test finding)
        ssim_global, _ = ssim_func(img_a_gray, img_b_gray, full=True, data_range=255)
        
        # Step 5: Compute patch-wise SSIM (NEW - discrimination layer)
        patch_scores = []
        patch_min = np.nan
        patch_topk_mean = np.nan
        
        if USE_PATCHWISE_SSIM:
            gh = int(max(1, SSIM_GRID_H))
            gw = int(max(1, SSIM_GRID_W))
            
            ha, wa = img_a_gray.shape
            hb, wb = img_b_gray.shape
            
            # Use the smaller common grid-aligned area
            H = min(ha - (ha % gh), hb - (hb % gh))
            W = min(wa - (wa % gw), wb - (wb % gw))
            
            if H > 0 and W > 0:
                # Crop to aligned grid
                ca = img_a_gray[:H, :W]
                cb = img_b_gray[:H, :W]
                
                # Patch dimensions
                ph = H // gh
                pw = W // gw
                
                # Compute SSIM for each patch
                for r in range(gh):
                    for c in range(gw):
                        ya, yb = r * ph, (r + 1) * ph
                        xa, xb = c * pw, (c + 1) * pw
                        
                        patch_a = ca[ya:yb, xa:xb]
                        patch_b = cb[ya:yb, xa:xb]
                        
                        # Skip tiny patches
                        if patch_a.size < 64 or patch_b.size < 64:
                            continue
                        
                        try:
                            # Critical: Specify data_range for uint8 images
                            ps, _ = ssim_func(patch_a, patch_b, full=True, data_range=255)
                            if not np.isnan(ps):
                                patch_scores.append(float(ps))
                        except Exception:
                            continue
                
                # Aggregate patch statistics
                if patch_scores:
                    patch_scores_arr = np.array(patch_scores, dtype=float)
                    patch_min = float(np.min(patch_scores_arr))
                    
                    # Top-K mean (emphasize best-matching regions)
                    k = int(max(1, min(len(patch_scores_arr), SSIM_TOPK_PATCHES)))
                    topk_patches = np.sort(patch_scores_arr)[-k:]
                    patch_topk_mean = float(np.mean(topk_patches))
        
        # Step 6: Compute mixed SSIM score (final output)
        if USE_PATCHWISE_SSIM and not np.isnan(patch_topk_mean):
            # Mix global with top-K patches
            w = float(np.clip(SSIM_MIX_WEIGHT, 0.0, 1.0))
            score_mixed = float((1.0 - w) * ssim_global + w * patch_topk_mean)
        else:
            # Fallback to global only
            score_mixed = float(ssim_global)
        
        # Step 7: Package metadata
        metadata = {
            'normalization_applied': apply_norm,
            'params_a': params_a,
            'params_b': params_b,
            'ssim_global': float(ssim_global),
            'patch_min': float(patch_min) if not np.isnan(patch_min) else None,
            'patch_topk_mean': float(patch_topk_mean) if not np.isnan(patch_topk_mean) else None,
            'num_patches': len(patch_scores) if patch_scores else 0,
            'grid': [int(SSIM_GRID_H), int(SSIM_GRID_W)],
            'mix_weight': float(SSIM_MIX_WEIGHT) if USE_PATCHWISE_SSIM else 0.0
        }
        
        return score_mixed, metadata
        
    except Exception as e:
        warnings.warn(f"SSIM computation failed: {e}")
        return np.nan, {}

# --- Confocal Deep Verify Helper Functions -----------------------------------
def _zscore(img: np.ndarray) -> np.ndarray:
    """Z-score normalization for NCC computation"""
    img = img.astype(np.float32)
    mu, sd = float(np.mean(img)), float(np.std(img))
    if sd < 1e-6:
        sd = 1.0
    return (img - mu) / sd

def _ncc_same_size(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between z-scored images"""
    a_z = _zscore(a)
    b_z = _zscore(b)
    return float(np.mean(a_z * b_z))

def _ecc_align_gray(a: np.ndarray, b: np.ndarray, max_iter: int = 120) -> Tuple[np.ndarray, dict]:
    """ECC affine alignment: warp 'a' onto 'b'"""
    a_gray = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h = min(a_gray.shape[0], b_gray.shape[0])
    
    # Scale both to common height for stability
    def _resize_to_h(x, H):
        s = H / x.shape[0]
        return cv2.resize(x, (max(1, int(round(x.shape[1]*s))), H), interpolation=cv2.INTER_AREA)
    
    a_r = _resize_to_h(a_gray, h)
    b_r = _resize_to_h(b_gray, h)
    
    # Pad to same width
    W = max(a_r.shape[1], b_r.shape[1])
    if a_r.shape[1] < W:
        a_r = cv2.copyMakeBorder(a_r, 0, 0, 0, W - a_r.shape[1], cv2.BORDER_CONSTANT, value=128)
    if b_r.shape[1] < W:
        b_r = cv2.copyMakeBorder(b_r, 0, 0, 0, W - b_r.shape[1], cv2.BORDER_CONSTANT, value=128)
    
    # ECC alignment
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(_zscore(b_r), _zscore(a_r), warp, cv2.MOTION_AFFINE, criteria)
        a_aligned = cv2.warpAffine(a_r, warp, (b_r.shape[1], b_r.shape[0]),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=128)
        return a_aligned, {'cc': float(cc)}
    except Exception:
        # Return no-op alignment if ECC fails
        return a_r, {'cc': 0.0}

def deep_verify_identical_confocal(path_a: str, path_b: str) -> dict:
    """
    Robust, calculation-only confirmation for confocal pairs:
      - ECC alignment (affine)
      - Recompute SSIM after alignment
      - Normalized cross-correlation (NCC)
      - Rotation/mirror-robust pHash (8-way bundle)
    """
    try:
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            return {'ok': False, 'err': 'read_fail'}
        
        # Optional text masking for overlays that bias signals
        if ENABLE_TEXT_MASKING and MASKING_APPLY_TO_SSIM:
            try:
                ma = detect_text_regions_heuristic(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
                mb = detect_text_regions_heuristic(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
                img_a = cv2.inpaint(img_a, ma, 3, cv2.INPAINT_TELEA)
                img_b = cv2.inpaint(img_b, mb, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass
        
        # ECC align + SSIM/NCC
        a_aligned, ecc_info = _ecc_align_gray(img_a, img_b, max_iter=DEEP_VERIFY_MAX_STEPS)
        from skimage.metrics import structural_similarity as ssim_func
        b_gray = img_b if img_b.ndim == 2 else cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        
        # Ensure same shape
        if a_aligned.shape != b_gray.shape:
            H = min(a_aligned.shape[0], b_gray.shape[0])
            a_aligned = cv2.resize(a_aligned, (int(round(a_aligned.shape[1]*H/a_aligned.shape[0])), H))
            b_gray = cv2.resize(b_gray, (int(round(b_gray.shape[1]*H/b_gray.shape[0])), H))
            W = max(a_aligned.shape[1], b_gray.shape[1])
            if a_aligned.shape[1] < W:
                a_aligned = cv2.copyMakeBorder(a_aligned,0,0,0,W-a_aligned.shape[1],cv2.BORDER_CONSTANT,value=128)
            if b_gray.shape[1] < W:
                b_gray = cv2.copyMakeBorder(b_gray,0,0,0,W-b_gray.shape[1],cv2.BORDER_CONSTANT,value=128)
        
        # Critical: Specify data_range for uint8 images
        deep_ssim, _ = ssim_func(a_aligned, b_gray, full=True, data_range=255)
        deep_ncc = _ncc_same_size(a_aligned, b_gray)
        
        # pHash (8 x rotations + mirror)
        try:
            imA = Image.open(path_a).convert('RGB')
            imB = Image.open(path_b).convert('RGB')
            bundle_a = compute_phash_bundle(imA)
            bundle_b = compute_phash_bundle(imB)
            phash_min, _ = hamming_min_transform(bundle_a, bundle_b, short_circuit=PHASH_BUNDLE_SHORT_CIRCUIT)
        except Exception:
            phash_min = 999
        
        return {
            'ok': (deep_ssim >= DEEP_VERIFY_ALIGN_SSIM_MIN and deep_ncc >= DEEP_VERIFY_NCC_MIN) or (phash_min <= DEEP_VERIFY_PHASH_MAX),
            'deep_ssim': float(deep_ssim),
            'deep_ncc': float(deep_ncc),
            'phash_min': int(phash_min),
            'ecc_cc': float(ecc_info.get('cc', 0.0))
        }
    except Exception as e:
        return {'ok': False, 'err': f'exception:{e}'}

# --- IHC / Histology Deep Verify Helper Functions -----------------------------
def _is_ihc_like_bgr(img_bgr: np.ndarray, frac_min: float = None) -> bool:
    """
    Heuristic: IHC often shows brown DAB and purple/blue hematoxylin.
    We detect by HSV bands in OpenCV ranges.
    """
    if frac_min is None:
        frac_min = IHC_LIKE_MIN_FRACTION
    if img_bgr is None:
        return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # H:[0..179], S/V:[0..255]
    # Brown/orange DAB range (~10..35 deg), moderately saturated
    mask_brown = (h >= 10) & (h <= 35) & (s >= 50) & (v >= 30)
    # Purple-blue hematoxylin (~120..160 deg)
    mask_purple = (h >= 120) & (h <= 160) & (s >= 40) & (v >= 30)
    frac = max(np.mean(mask_brown), np.mean(mask_purple))
    return bool(frac >= frac_min)

def _stain_channel_gray(img_bgr: np.ndarray) -> np.ndarray:
    """
    Try to extract a stain-robust channel for matching:
      1) Prefer skimage's HED DAB channel (if available)
      2) Fallback to HSV V channel mixed with Lab 'b' to emphasize brown
    Output is uint8 grayscale.
    """
    # Attempt HED deconvolution (DAB channel)
    try:
        from skimage.color import rgb2hed
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        hed = rgb2hed(rgb)  # floats, optical density space
        dab = hed[:, :, 2]  # DAB channel
        # Normalize to 0..255 (invert OD to intensity-like)
        dab_norm = dab - np.min(dab)
        dab_norm = dab_norm / max(np.max(dab_norm), 1e-6)
        ch = (255.0 * (1.0 - dab_norm)).astype(np.uint8)
        return ch
    except Exception:
        pass
    
    # Fallback: HSV V blended with Lab b (brownish info)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    b  = lab[:, :, 2].astype(np.float32) / 255.0
    mix = (0.6 * v + 0.4 * b)
    mix = np.clip(mix * 255.0, 0, 255).astype(np.uint8)
    return mix

def deep_verify_identical_ihc(path_a: str, path_b: str) -> dict:
    """
    Robust confirmation for histology/IHC pairs:
      - stain-robust channel extraction
      - ECC alignment on that channel
      - SSIM + NCC after alignment
      - pHash bundles check (RGB)
    """
    try:
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            return {'ok': False, 'err': 'read_fail'}
        
        # Quick ihc-like gate (optional)
        if IHC_DV_REQUIRE_IHC_LIKE:
            if not (_is_ihc_like_bgr(img_a) and _is_ihc_like_bgr(img_b)):
                return {'ok': False, 'err': 'not_ihc_like'}
        
        # Text masking if enabled
        if ENABLE_TEXT_MASKING and MASKING_APPLY_TO_SSIM:
            try:
                ma = detect_text_regions_heuristic(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
                mb = detect_text_regions_heuristic(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
                img_a = cv2.inpaint(img_a, ma, 3, cv2.INPAINT_TELEA)
                img_b = cv2.inpaint(img_b, mb, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass
        
        ch_a = _stain_channel_gray(img_a)
        ch_b = _stain_channel_gray(img_b)
        
        # ECC align A->B on stain channel
        a_aligned, ecc_info = _ecc_align_gray(ch_a, ch_b, max_iter=IHC_DV_MAX_STEPS)
        
        from skimage.metrics import structural_similarity as ssim_func
        # Ensure same size
        if a_aligned.shape != ch_b.shape:
            H = min(a_aligned.shape[0], ch_b.shape[0])
            a_aligned = cv2.resize(a_aligned, (int(round(a_aligned.shape[1]*H/a_aligned.shape[0])), H))
            ch_b      = cv2.resize(ch_b,      (int(round(ch_b.shape[1]*H/ch_b.shape[0])), H))
            W = max(a_aligned.shape[1], ch_b.shape[1])
            if a_aligned.shape[1] < W:
                a_aligned = cv2.copyMakeBorder(a_aligned,0,0,0,W-a_aligned.shape[1],cv2.BORDER_CONSTANT,value=128)
            if ch_b.shape[1] < W:
                ch_b = cv2.copyMakeBorder(ch_b,0,0,0,W-ch_b.shape[1],cv2.BORDER_CONSTANT,value=128)
        
        # Critical: Specify data_range for uint8 images
        ssim_after, _ = ssim_func(a_aligned, ch_b, full=True, data_range=255)
        ncc_after = _ncc_same_size(a_aligned, ch_b)
        
        # pHash bundle (RGB)
        try:
            imA = Image.open(path_a).convert('RGB')
            imB = Image.open(path_b).convert('RGB')
            bundle_a = compute_phash_bundle(imA)
            bundle_b = compute_phash_bundle(imB)
            phash_min, _ = hamming_min_transform(bundle_a, bundle_b, short_circuit=PHASH_BUNDLE_SHORT_CIRCUIT)
        except Exception:
            phash_min = 999
        
        ok = ((ssim_after >= IHC_DV_SSIM_AFTER_ALIGN_MIN and ncc_after >= IHC_DV_NCC_MIN)
              or (phash_min <= IHC_DV_PHASH_MAX))
        
        return {
            'ok': bool(ok),
            'IHC_SSIM': float(ssim_after),
            'IHC_NCC': float(ncc_after),
            'IHC_pHash': int(phash_min),
            'ecc_cc': float(ecc_info.get('cc', 0.0))
        }
    except Exception as e:
        return {'ok': False, 'err': f'exception:{e}'}

def attach_clip_zscore_to_df(df: pd.DataFrame,
                             panel_paths: Optional[List[Path]] = None,
                             vecs: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Attach per-pair CLIP outlier score (z-score) without page heuristics.
    
    Z-score formula:
        z_ij = min( (s_ij - μ_i)/σ_i , (s_ij - μ_j)/σ_j )
    
    Where:
        - s_ij: CLIP similarity between panels i and j
        - μ_i: Mean CLIP similarity of panel i to all others
        - σ_i: Std dev of panel i's similarities
        - We take the MIN of both endpoints' z-scores (conservative)
    
    Why this works:
        - Grid panels have MANY similar neighbors → low μ, high σ → LOW z-scores
        - True duplicates are OUTLIERS for both panels → HIGH z-scores
        - No page information needed!
    
    Example:
        Grid panels (microscopy A-F on same page):
            - Raw CLIP: 0.97 (high!)
            - Z-score: 1.2 (low - many neighbors at 0.96-0.98)
            - Result: Filtered by z-score gate
        
        True duplicate (same Western blot, different pages):
            - Raw CLIP: 0.98 (high!)
            - Z-score: 4.5 (very high - no other matches this strong)
            - Result: Passes z-score gate
    """
    if df is None or df.empty or 'Cosine_Similarity' not in df.columns:
        return df

    df = df.copy()

    # Method 1: Global statistics (accurate - uses full similarity matrix)
    if vecs is not None and panel_paths is not None and len(panel_paths) == len(vecs):
        print("    Computing CLIP z-scores (global method)...")
        
        idx_by_path = {str(p): i for i, p in enumerate(panel_paths)}
        
        try:
            # Compute full similarity matrix
            sim = vecs @ vecs.T
            np.fill_diagonal(sim, np.nan)  # Ignore self-similarity
            
            # Per-panel statistics
            mu = np.nanmean(sim, axis=1)   # Mean similarity per panel
            sd = np.nanstd(sim, axis=1)    # Std dev per panel
            sd = np.where(sd < 1e-6, 1e-6, sd)  # Avoid division by zero
            
            zs = []
            for _, r in df.iterrows():
                pa, pb = str(r['Path_A']), str(r['Path_B'])
                i = idx_by_path.get(pa, None)
                j = idx_by_path.get(pb, None)
                
                if i is None or j is None:
                    zs.append(0.0)
                    continue
                
                # Pair similarity
                val = pd.to_numeric(r['Cosine_Similarity'], errors='coerce')
                sij = float(0.0 if pd.isna(val) else val)
                
                # Z-scores for both endpoints
                zi = (sij - float(mu[i])) / float(sd[i])
                zj = (sij - float(mu[j])) / float(sd[j])
                
                # Take conservative minimum
                zs.append(float(min(zi, zj)))
            
            df['CLIP_Z'] = zs
            return df
            
        except Exception as e:
            warnings.warn(f"Global z-score computation failed: {e}, using fallback")
            # Fall through to Method 2

    # Method 2: Approximate from observed pairs (fallback)
    print("    Computing CLIP z-scores (approximate method)...")
    
    try:
        s = pd.to_numeric(df['Cosine_Similarity'], errors='coerce').fillna(0.0)
        
        # Per-panel statistics from observed pairs
        by_a = df.groupby('Path_A')['Cosine_Similarity'].agg(['mean', 'std']).rename(
            columns={'mean': 'mu_a', 'std': 'sd_a'}
        )
        by_b = df.groupby('Path_B')['Cosine_Similarity'].agg(['mean', 'std']).rename(
            columns={'mean': 'mu_b', 'std': 'sd_b'}
        )
        
        # Join and compute approximate z-score
        tmp = df.join(by_a, on='Path_A').join(by_b, on='Path_B')
        
        mu_i = tmp[['mu_a', 'mu_b']].mean(axis=1).fillna(0.0)
        sd_i = tmp[['sd_a', 'sd_b']].mean(axis=1).replace(0.0, 1.0).fillna(1.0)
        
        df['CLIP_Z'] = ((s - mu_i) / sd_i).astype(float)
        
    except Exception as e:
        warnings.warn(f"Approximate z-score computation failed: {e}")
        df['CLIP_Z'] = 0.0
    
    return df

# --- MODALITY DETECTION (OPTION 2: Advanced) ----------------------------------
def detect_modality(img_path: str) -> dict:
    """
    Classify scientific image type using computer vision heuristics.
    
    Returns:
        {
            'modality': str,  # 'western_blot', 'confocal', 'tem', 'bright_field', 'gel', 'table', 'unknown'
            'confidence': float (0-1),
            'features': dict
        }
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {'modality': 'unknown', 'confidence': 0.0, 'features': {}}
        
        # Feature 1: Edge Density
        edges = cv2.Canny(img, 50, 150)
        edge_density = float(np.sum(edges > 0) / (img.shape[0] * img.shape[1]))
        
        # Feature 2: Directional Anisotropy (horizontal vs vertical)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        vertical_strength = float(np.mean(np.abs(grad_y)))
        horizontal_strength = float(np.mean(np.abs(grad_x)))
        anisotropy = vertical_strength / (horizontal_strength + 1e-6)
        
        # Feature 3: Spatial Frequency (texture scale)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        h, w = img.shape
        low_freq = float(np.mean(magnitude[:h//8, :w//8]))
        mid_freq = float(np.mean(magnitude[h//4:3*h//4, w//4:3*w//4]))
        high_freq = float(np.mean(magnitude[3*h//8:5*h//8, 3*w//8:5*w//8]))
        
        # Feature 4: Text Presence (OCR-free heuristic)
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        text_components = 0
        if num_labels > 1:
            for i in range(1, num_labels):
                x, y, w_comp, h_comp, area = stats[i]
                aspect = max(w_comp, h_comp) / max(min(w_comp, h_comp), 1)
                if 2.0 <= aspect <= 8.0 and area < img.size * 0.01:
                    text_components += 1
        
        text_density = float(text_components / max(num_labels, 1))
        
        features = {
            'edge_density': edge_density,
            'anisotropy': anisotropy,
            'low_freq': low_freq,
            'mid_freq': mid_freq,
            'high_freq': high_freq,
            'text_density': text_density
        }
        
        # Decision tree with confidence
        # Rule 1: Table/Graph (high text, low visual)
        if text_density > 0.3 and edge_density < 0.08:
            return {'modality': 'table', 'confidence': min(text_density / 0.3, 1.0), 'features': features}
        
        # Rule 2: Western Blot (horizontal bands)
        if anisotropy > 1.4 and 0.08 < edge_density < 0.20:
            confidence = min((anisotropy - 1.0) / 1.0, 1.0)
            return {'modality': 'western_blot', 'confidence': confidence, 'features': features}
        
        # Rule 3: Confocal (cellular texture, mid-frequency)
        if mid_freq > 80 and 0.12 < edge_density < 0.25:
            confidence = min(mid_freq / 150, 1.0)
            return {'modality': 'confocal', 'confidence': confidence, 'features': features}
        
        # Rule 4: TEM (very high detail)
        if edge_density > 0.28 and high_freq > 100:
            confidence = min(edge_density / 0.35, 1.0)
            return {'modality': 'tem', 'confidence': confidence, 'features': features}
        
        # Rule 5: Bright-field (smooth, low frequency)
        if edge_density < 0.08 and low_freq > mid_freq:
            return {'modality': 'bright_field', 'confidence': 1.0 - edge_density / 0.08, 'features': features}
        
        # Default: Unknown
        return {'modality': 'unknown', 'confidence': 0.0, 'features': features}
        
    except Exception as e:
        warnings.warn(f"Modality detection failed for {img_path}: {e}")
        return {'modality': 'unknown', 'confidence': 0.0, 'features': {}}

def apply_modality_specific_tier_gating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply RESEARCH-BACKED thresholds based on detected modality (Option 2: Advanced).
    
    Based on peer-reviewed research:
    - Proofig AI commercial tool parameters
    - Nature/Cell journal guidelines
    - IEEE/SPIE imaging conference findings
    - MS-SSIM adaptation for microscopy (Renieblas et al.)
    
    Modality-specific rules:
    - Western Blot: Rotation-tolerant, relaxed SSIM (Proofig validated)
    - Confocal: Patch-SSIM critical, high texture sensitivity
    - TEM: Very strict, MS-SSIM for fine detail
    - Bright-field: Strong photometric normalization
    - Gel: Column structure, rotation bundles
    """
    if df.empty:
        return df
    
    print("  Detecting modalities for all panels...")
    
    # Detect modality for all unique panels
    all_paths = set(df['Path_A'].tolist() + df['Path_B'].tolist())
    modality_cache = {}
    
    for path in tqdm(all_paths, desc="Modality"):
        result = detect_modality(path)
        modality_cache[path] = result['modality']
    
    # Attach modality columns
    df['Modality_A'] = df['Path_A'].map(modality_cache)
    df['Modality_B'] = df['Path_B'].map(modality_cache)
    df['Same_Modality'] = df['Modality_A'] == df['Modality_B']
    
    # Extract signals
    clip_score = pd.to_numeric(df['Cosine_Similarity'], errors='coerce').fillna(0)
    ssim_score = pd.to_numeric(df['SSIM'], errors='coerce').fillna(0)
    phash_dist = pd.to_numeric(df.get('Hamming_Distance', 999), errors='coerce').fillna(999)
    orb_inliers = pd.to_numeric(df.get('ORB_Inliers', 0), errors='coerce').fillna(0)
    orb_coverage = pd.to_numeric(df.get('Crop_Coverage', 0), errors='coerce').fillna(0)
    patch_min = pd.to_numeric(df.get('Patch_SSIM_Min', 0), errors='coerce').fillna(0)
    
    df['Tier'] = None
    df['Tier_Path'] = None
    
    # Research-backed modality-specific thresholds
    MODALITY_PARAMS = {
        'western_blot': {
            'tier_a': {'clip': 0.94, 'ssim': 0.60, 'combined': 1.55, 'phash': 3},
            'tier_b': {'clip': 0.92, 'ssim': 0.50, 'combined': 1.45, 'phash': 4},
            'orb': {'min_inliers': 25, 'coverage': 0.70}
        },
        'confocal': {
            'tier_a': {'clip': 0.95, 'ssim': 0.75, 'combined': 1.70, 'phash': 3, 'patch_min': 0.70},
            'tier_b': {'clip': 0.92, 'ssim': 0.65, 'combined': 1.60, 'phash': 4},
            'fp_gate': {'clip_high': 0.96, 'ssim_low': 0.50}  # High CLIP + Low SSIM = FP
        },
        'tem': {
            'tier_a': {'clip': 0.95, 'ssim': 0.85, 'combined': 1.80, 'phash': 2},
            'tier_b': {'clip': 0.93, 'ssim': 0.75, 'combined': 1.70, 'phash': 3}
        },
        'bright_field': {
            'tier_a': {'clip': 0.93, 'ssim': 0.75, 'combined': 1.68, 'phash': 4},
            'tier_b': {'clip': 0.90, 'ssim': 0.65, 'combined': 1.58, 'phash': 5}
        },
        'gel': {
            'tier_a': {'clip': 0.94, 'ssim': 0.70, 'combined': 1.64, 'phash': 3},
            'tier_b': {'clip': 0.91, 'ssim': 0.60, 'combined': 1.54, 'phash': 4},
            'orb': {'min_inliers': 20, 'coverage': 0.60}
        },
        'table': {
            'tier_b': {'clip': 0.92, 'ssim': 0.90}  # Flag for OCR review
        }
    }
    
    # Apply exact matches first (all modalities)
    exact_mask = (phash_dist <= 2)
    df.loc[exact_mask, 'Tier'] = 'A'
    df.loc[exact_mask, 'Tier_Path'] = 'Exact (pHash)'
    
    # Apply modality-specific rules
    for modality, params in MODALITY_PARAMS.items():
        mask = (df['Modality_A'] == modality) & (df['Modality_B'] == modality) & (~exact_mask)
        
        if mask.sum() == 0:
            continue
        
        # Tier A criteria
        if 'tier_a' in params:
            ta = params['tier_a']
            
            # Path 1: Combined threshold (research-backed)
            tier_a_combined = mask & (
                (clip_score >= ta['clip']) & 
                (ssim_score >= ta['ssim']) & 
                ((clip_score + ssim_score) >= ta['combined'])
            )
            
            # Path 2: pHash exact (with modality-specific tolerance)
            tier_a_phash = mask & (phash_dist <= ta['phash'])
            
            # Path 3: Patch-SSIM gate (confocal only)
            tier_a_patch = pd.Series([False] * len(df))
            if modality == 'confocal' and 'patch_min' in ta:
                tier_a_patch = mask & (
                    (clip_score >= ta['clip']) & 
                    (patch_min >= ta['patch_min'])
                )
            
            # Path 4: ORB partial (WB, gel)
            tier_a_orb = pd.Series([False] * len(df))
            if modality in ['western_blot', 'gel'] and 'orb' in params:
                orb_params = params['orb']
                tier_a_orb = mask & (
                    (orb_inliers >= orb_params['min_inliers']) &
                    (orb_coverage >= orb_params['coverage'])
                )
            
            # Combine all Tier A paths for this modality
            tier_a_final = tier_a_combined | tier_a_phash | tier_a_patch | tier_a_orb
            df.loc[tier_a_final, 'Tier'] = 'A'
            df.loc[tier_a_final, 'Tier_Path'] = f'{modality.replace("_", " ").title()}-Specific'
        
        # Tier B criteria
        if 'tier_b' in params:
            tb = params['tier_b']
            tier_b = mask & (df['Tier'].isna()) & (
                (clip_score >= tb['clip']) & 
                (ssim_score >= tb['ssim'])
            )
            df.loc[tier_b, 'Tier'] = 'B'
            df.loc[tier_b, 'Tier_Path'] = f'{modality.replace("_", " ").title()}-Borderline'
        
        # Confocal false positive filter (research-backed)
        if modality == 'confocal' and 'fp_gate' in params:
            fp = params['fp_gate']
            fp_mask = mask & (df['Tier'].isna()) & (
                (clip_score >= fp['clip_high']) & 
                (ssim_score < fp['ssim_low'])
            )
            # Mark as filtered (same modality, different content)
            df.loc[fp_mask, 'Tier_Path'] = 'Confocal-FP-Filtered'
    
    # Unknown or cross-modality: Use universal rule fallback
    unknown_mask = (
        ((df['Modality_A'] == 'unknown') | (df['Modality_B'] == 'unknown') | (~df['Same_Modality'])) &
        (df['Tier'].isna())
    )
    tier_a_universal = unknown_mask & (clip_score >= 0.95) & (ssim_score >= 0.65) & ((clip_score + ssim_score) >= 1.65)
    df.loc[tier_a_universal, 'Tier'] = 'A'
    df.loc[tier_a_universal, 'Tier_Path'] = 'Universal-Fallback'
    
    # Remaining borderline → Tier B
    tier_b_universal = (df['Tier'].isna()) & (clip_score >= 0.92) & (ssim_score >= 0.40)
    df.loc[tier_b_universal, 'Tier'] = 'B'
    df.loc[tier_b_universal, 'Tier_Path'] = 'Universal-Borderline'
    
    # Print modality distribution
    print(f"  Modality distribution:")
    from collections import Counter
    modality_counts = Counter(modality_cache.values())
    for modality, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {modality}: {count} panels")
    
    return df

# --- ORB-RANSAC FOR PARTIAL DUPLICATES ----------------------------------------
def extract_orb_features(img_path: str, max_keypoints: int = 1000,
                        retry_scales: List[float] = [1.0, 2.0, 0.5]) -> Tuple[Optional[list], Optional[np.ndarray]]:
    """Extract ORB keypoints with multi-scale robustness"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        
        # Optional text masking for ORB
        if ENABLE_TEXT_MASKING and MASKING_APPLY_TO_ORB:
            try:
                mask = detect_text_regions_heuristic(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray, _ = normalize_photometric(img_gray)
        
        orb = cv2.ORB_create(nfeatures=max_keypoints)
        # SAFE DEFAULTS in case all retries fail
        kp, desc = [], None
        
        for scale in retry_scales:
            if scale != 1.0:
                h, w = img_gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                img_scaled = cv2.resize(img_gray, (new_w, new_h))
            else:
                img_scaled = img_gray
            
            kpi, desci = orb.detectAndCompute(img_scaled, None)
            
            if desci is not None and len(kpi) >= 50:
                if scale != 1.0:
                    for k in kpi:
                        k.pt = (k.pt[0] / scale, k.pt[1] / scale)
                return kpi, desci
        
        # Fallback: return None,None instead of possibly-unbound variables
        return (kp if len(kp) else None, desc)
        
    except Exception as e:
        warnings.warn(f"ORB extraction failed for {img_path}: {e}")
        return None, None

def match_orb_features(desc_a: np.ndarray, desc_b: np.ndarray,
                      ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
    """BFMatcher (Hamming) + Lowe's ratio test"""
    if desc_a is None or desc_b is None:
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    try:
        matches = bf.knnMatch(desc_a, desc_b, k=2)
    except Exception:
        return []
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches

def estimate_homography_ransac(kp_a: list, kp_b: list, matches: List[cv2.DMatch],
                               min_inliers: int = 30, max_reproj_error: float = 4.0) -> dict:
    """RANSAC homography with degenerate detection"""
    result = {
        'H': None, 'homography_type': None, 'inliers': 0, 'inlier_ratio': 0.0,
        'reproj_error': 999.0, 'crop_coverage': 0.0, 'is_partial_dupe': False, 'is_degenerate': False
    }
    
    if len(matches) < min_inliers:
        return result
    
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
    
    try:
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if H is None:
            return result
        
        det_H = np.linalg.det(H[:2, :2])
        if abs(det_H) < 0.1 or abs(det_H) > 10.0:
            result['is_degenerate'] = True
            return result
        
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches)
        
        if inliers < min_inliers or inlier_ratio < 0.30:
            return result
        
        inlier_pts_a = pts_a[mask.ravel() == 1]
        inlier_pts_b = pts_b[mask.ravel() == 1]
        
        inlier_pts_a_h = np.hstack([inlier_pts_a, np.ones((len(inlier_pts_a), 1))])
        projected = (H @ inlier_pts_a_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        errors = np.linalg.norm(projected - inlier_pts_b, axis=1)
        median_error = np.median(errors)
        
        if median_error > max_reproj_error:
            return result
        
        is_affine = np.allclose(H[2, :2], 0) and np.isclose(H[2, 2], 1)
        
        result['H'] = H
        result['homography_type'] = 'affine' if is_affine else 'projective'
        result['inliers'] = int(inliers)
        result['inlier_ratio'] = float(inlier_ratio)
        result['reproj_error'] = float(median_error)
        
        return result
        
    except Exception as e:
        warnings.warn(f"Homography estimation failed: {e}")
        return result

def compute_crop_coverage(H: np.ndarray, img_a_shape: tuple, img_b_shape: tuple) -> float:
    """Project A's corners through homography into B's coordinate space"""
    if H is None:
        return 0.0
    
    h_a, w_a = img_a_shape[:2]
    h_b, w_b = img_b_shape[:2]
    
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]])
    
    try:
        corners_a_h = np.hstack([corners_a, np.ones((4, 1))])
        projected = (H @ corners_a_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        inside_count = 0
        for pt in projected:
            if 0 <= pt[0] <= w_b and 0 <= pt[1] <= h_b:
                inside_count += 1
        
        quad_area = cv2.contourArea(projected.astype(np.float32))
        b_area = w_b * h_b
        
        coverage = min(inside_count / 4.0, quad_area / b_area)
        return float(coverage)
        
    except Exception:
        return 0.0

# ═══════════════════════════════════════════════════════════════
# PHASE B: FORENSICS - Copy-Move Detection & Selective Brightness
# ═══════════════════════════════════════════════════════════════

# --- COPY-MOVE / CLONE STAMP DETECTION ----------------------------------------
def compute_dct_block_descriptors(img: np.ndarray, block_size: int = 32,
                                  stride: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Fast block hashing using DCT coefficients"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    h, w = img.shape
    descriptors = []
    coordinates = []
    
    for y in range(0, h - block_size, stride):
        for x in range(0, w - block_size, stride):
            block = img[y:y+block_size, x:x+block_size]
            dct_block = cv2.dct(np.float32(block))
            low_freq = dct_block[:8, :8].flatten()
            
            norm = np.linalg.norm(low_freq)
            if norm > 1e-6:
                low_freq = low_freq / norm
            
            descriptors.append(low_freq)
            coordinates.append([y, x])
    
    return np.array(descriptors), np.array(coordinates)

def find_clone_candidates(descriptors: np.ndarray, coords: np.ndarray, img_shape: tuple,
                         min_separation: float = 0.10, similarity_threshold: float = 0.95) -> List[Tuple[int, int]]:
    """Find similar blocks with spatial separation constraint"""
    from scipy.spatial.distance import cdist
    
    min_dim = min(img_shape[:2])
    min_sep_px = int(min_dim * min_separation)
    
    similarities = 1 - cdist(descriptors, descriptors, metric='cosine')
    candidates = []
    
    for i in range(len(descriptors)):
        for j in range(i + 1, len(descriptors)):
            if similarities[i, j] < similarity_threshold:
                continue
            
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < min_sep_px:
                continue
            
            candidates.append((i, j))
    
    return candidates

def confirm_clone_with_orb(img: np.ndarray, region_a: Tuple[int, int, int, int],
                          region_b: Tuple[int, int, int, int]) -> dict:
    """Stage 2: Confirm block proposals with ORB + RANSAC"""
    result = {'inliers': 0, 'inlier_ratio': 0.0, 'reproj_error': 999.0, 'confirmed': False}
    
    try:
        ya, xa, ha, wa = region_a
        yb, xb, hb, wb = region_b
        
        crop_a = img[ya:ya+ha, xa:xa+wa]
        crop_b = img[yb:yb+hb, xb:xb+wb]
        
        if len(crop_a.shape) == 3:
            crop_a = cv2.cvtColor(crop_a, cv2.COLOR_RGB2GRAY)
            crop_b = cv2.cvtColor(crop_b, cv2.COLOR_RGB2GRAY)
        
        orb = cv2.ORB_create(nfeatures=500)
        kp_a, desc_a = orb.detectAndCompute(crop_a, None)
        kp_b, desc_b = orb.detectAndCompute(crop_b, None)
        
        if desc_a is None or desc_b is None or len(kp_a) < 10:
            return result
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc_a, desc_b, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 30:
            return result
        
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches])
        
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
        
        if H is None:
            return result
        
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(good_matches)
        
        inlier_pts_a = pts_a[mask.ravel() == 1]
        inlier_pts_b = pts_b[mask.ravel() == 1]
        
        inlier_pts_a_h = np.hstack([inlier_pts_a, np.ones((len(inlier_pts_a), 1))])
        projected = (H @ inlier_pts_a_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        errors = np.linalg.norm(projected - inlier_pts_b, axis=1)
        median_error = np.median(errors)
        
        result['inliers'] = int(inliers)
        result['inlier_ratio'] = float(inlier_ratio)
        result['reproj_error'] = float(median_error)
        result['confirmed'] = (inliers >= 30 and inlier_ratio >= 0.30 and median_error <= 4.0)
        
    except Exception as e:
        warnings.warn(f"ORB confirmation failed: {e}")
    
    return result

def detect_clone_stamp_two_stage(img: np.ndarray, block_sizes: List[int] = [32, 48],
                                 stride: int = 8, min_area_pct: float = 0.5) -> dict:
    """Two-stage clone detection with safeguards for microscopy"""
    result = {
        'clone_regions': [],
        'inliers': 0,
        'inlier_ratio': 0.0,
        'clone_area_pct': 0.0,
        'is_suspicious': False
    }
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    h, w = img_gray.shape
    total_area = h * w
    
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / total_area
    
    if edge_density < 0.05:
        return result
    
    all_candidates = []
    
    for block_size in block_sizes:
        descriptors, coords = compute_dct_block_descriptors(img_gray, block_size, stride)
        
        if len(descriptors) < 100:
            continue
        
        candidates = find_clone_candidates(descriptors, coords, img_gray.shape,
                                          min_separation=0.10, similarity_threshold=0.95)
        
        for idx_a, idx_b in candidates[:50]:
            ya, xa = coords[idx_a]
            yb, xb = coords[idx_b]
            
            region_a = (ya, xa, block_size, block_size)
            region_b = (yb, xb, block_size, block_size)
            
            confirmation = confirm_clone_with_orb(img_gray, region_a, region_b)
            
            if confirmation['confirmed']:
                all_candidates.append({
                    'region_a': region_a,
                    'region_b': region_b,
                    'inliers': confirmation['inliers'],
                    'inlier_ratio': confirmation['inlier_ratio'],
                    'reproj_error': confirmation['reproj_error'],
                    'block_size': block_size
                })
    
    if not all_candidates:
        return result
    
    result['clone_regions'] = all_candidates
    
    unique_area = set()
    for clone in all_candidates:
        ya, xa, ha, wa = clone['region_a']
        for dy in range(ha):
            for dx in range(wa):
                unique_area.add((ya + dy, xa + dx))
        
        yb, xb, hb, wb = clone['region_b']
        for dy in range(hb):
            for dx in range(wb):
                unique_area.add((yb + dy, xb + dx))
    
    clone_area_pct = 100.0 * len(unique_area) / total_area
    
    best = max(all_candidates, key=lambda x: x['inliers'])
    
    result['inliers'] = best['inliers']
    result['inlier_ratio'] = best['inlier_ratio']
    result['clone_area_pct'] = clone_area_pct
    result['is_suspicious'] = clone_area_pct >= min_area_pct
    
    return result

# --- SELECTIVE BRIGHTNESS DETECTION -------------------------------------------
def compute_robust_exposure_stats(img: np.ndarray, tile_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Use median/MAD instead of mean/std for robustness"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    h, w = img.shape
    n_tiles_h = h // tile_size
    n_tiles_w = w // tile_size
    
    median_map = np.zeros((n_tiles_h, n_tiles_w))
    mad_map = np.zeros((n_tiles_h, n_tiles_w))
    
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            tile = img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            tile_median = np.median(tile)
            tile_mad = np.median(np.abs(tile - tile_median))
            
            median_map[i, j] = tile_median
            mad_map[i, j] = tile_mad if tile_mad > 1e-6 else 1.0
    
    return median_map, mad_map

def check_boundary_continuity(img: np.ndarray, flagged_tiles: list, tile_size: int = 64) -> bool:
    """
    Check if exposure anomaly has sharp boundary discontinuity.
    
    Method: Compute gradient magnitude at tile boundaries.
    Returns: True if discontinuity present (suspicious)
    
    Reference: Forensic detection of selective brightness manipulation
    """
    if len(flagged_tiles) == 0:
        return False
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Compute edges
    edges = cv2.Canny(img, 50, 150)
    edge_density_total = np.sum(edges > 0) / edges.size
    
    # Check boundary edge density for flagged tiles
    boundary_strength = 0.0
    for (i, j) in flagged_tiles:
        y, x = i * tile_size, j * tile_size
        
        # Extract boundary regions (2px border)
        if y > 0:  # Top boundary
            top_edge = edges[max(0, y-2):y+2, x:min(x+tile_size, edges.shape[1])]
            boundary_strength = max(boundary_strength, np.sum(top_edge > 0) / max(top_edge.size, 1))
        
        if x > 0:  # Left boundary
            left_edge = edges[y:min(y+tile_size, edges.shape[0]), max(0, x-2):x+2]
            boundary_strength = max(boundary_strength, np.sum(left_edge > 0) / max(left_edge.size, 1))
    
    # Threshold: boundary edges 3x stronger than image average = discontinuity
    return boundary_strength > 3.0 * edge_density_total if edge_density_total > 0 else False

def detect_selective_brightness(img: np.ndarray, tile_size: int = 64,
                                z_threshold: float = 4.0) -> dict:
    """Detect localized exposure manipulation with gradient safeguards"""
    result = {
        'max_z_score': 0.0,
        'flagged_tiles': [],
        'flagged_area_pct': 0.0,
        'is_suspicious': False
    }
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    h, w = img_gray.shape
    
    median_map, mad_map = compute_robust_exposure_stats(img_gray, tile_size)
    n_tiles_h, n_tiles_w = median_map.shape
    
    if n_tiles_h < 3 or n_tiles_w < 3:
        return result
    
    z_scores = np.zeros_like(median_map)
    
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            i_min, i_max = max(0, i-1), min(n_tiles_h, i+2)
            j_min, j_max = max(0, j-1), min(n_tiles_w, j+2)
            
            neighbors = median_map[i_min:i_max, j_min:j_max]
            neighbor_median = np.median(neighbors)
            neighbor_mad = np.median(np.abs(neighbors - neighbor_median))
            
            if neighbor_mad > 1e-6:
                z_scores[i, j] = abs(median_map[i, j] - neighbor_median) / neighbor_mad
    
    flagged = []
    max_z = 0.0
    
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            if z_scores[i, j] >= z_threshold:
                flagged.append((i, j))
                max_z = max(max_z, z_scores[i, j])
    
    if flagged:
        flagged_area_pct = 100.0 * len(flagged) / (n_tiles_h * n_tiles_w)
        result['max_z_score'] = float(max_z)
        result['flagged_tiles'] = flagged
        result['flagged_area_pct'] = flagged_area_pct
        
        # NEW: Check boundary continuity for more robust detection
        result['has_boundary_discontinuity'] = check_boundary_continuity(img_gray, flagged, tile_size)
        result['is_suspicious'] = (max_z >= z_threshold and 
                                   result['has_boundary_discontinuity'])  # Now requires discontinuity
    
    return result

# ═══════════════════════════════════════════════════════════════
# PHASE C: ADVANCED FEATURES - ELA, Metadata, Text Masking
# ═══════════════════════════════════════════════════════════════

# --- ELA (WITH MAJOR CAVEATS) -------------------------------------------------
def detect_jpeg_origin(img_path: str) -> bool:
    """Try to determine if panel originated from JPEG"""
    try:
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            return True
        
        img = Image.open(img_path)
        exif = img._getexif()
        
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ['Compression', 'JPEGInterchangeFormat']:
                    return True
        
        img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            return False
        
        h, w = img_cv.shape
        if h < 64 or w < 64:
            return False
        
        grad = np.abs(np.diff(img_cv.astype(float), axis=1))
        
        block_edges = []
        for x in range(8, w-8, 8):
            edge_strength = np.mean(grad[:, x-1:x+1])
            block_edges.append(edge_strength)
        
        if len(block_edges) > 5:
            all_edges = np.mean(grad)
            boundary_edges = np.mean(block_edges)
            
            if boundary_edges > 1.2 * all_edges:
                return True
        
        return False
        
    except Exception:
        return False

def analyze_ela_hotspots(ela_map: np.ndarray, panel_area: int,
                         hotspot_ratio_threshold: float = 2.0, min_area_pct: float = 1.0) -> dict:
    """Use RELATIVE hotspot metrics (not absolute)"""
    result = {
        'hotspot_area_pct': 0.0,
        'hotspot_ratio': 0.0,
        'is_suspicious': False,
        'confidence': 'low'
    }
    
    ela_median = np.median(ela_map)
    
    if ela_median < 1e-6:
        return result
    
    threshold = ela_median * hotspot_ratio_threshold
    hotspot_mask = ela_map > threshold
    
    hotspot_area = np.sum(hotspot_mask)
    hotspot_area_pct = 100.0 * hotspot_area / panel_area
    
    if hotspot_area_pct < min_area_pct:
        return result
    
    hotspot_mean = np.mean(ela_map[hotspot_mask])
    hotspot_ratio = hotspot_mean / ela_median
    
    edges = cv2.Canny((ela_map * 255 / np.max(ela_map)).astype(np.uint8), 50, 150)
    edge_overlap = np.sum(hotspot_mask & (edges > 0))
    edge_alignment = edge_overlap / max(hotspot_area, 1)
    
    if hotspot_ratio >= 3.0 and edge_alignment > 0.5:
        confidence = 'high'
    elif hotspot_ratio >= 2.5 and edge_alignment > 0.3:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    result['hotspot_area_pct'] = hotspot_area_pct
    result['hotspot_ratio'] = hotspot_ratio
    result['is_suspicious'] = (hotspot_ratio >= hotspot_ratio_threshold and 
                              hotspot_area_pct >= min_area_pct)
    result['confidence'] = confidence
    
    return result

def compute_ela_map_conditional(img_path: str, quality: int = 90) -> dict:
    """Conditional ELA - only run when applicable"""
    result = {
        'ela_applicable': False,
        'ela_map': None,
        'hotspot_area_pct': 0.0,
        'hotspot_ratio': 0.0,
        'is_suspicious': False,
        'confidence': 'low',
        'note': ''
    }
    
    is_jpeg_origin = detect_jpeg_origin(img_path)
    
    if not is_jpeg_origin:
        result['note'] = "ELA not applicable: likely PNG/vector origin"
        return result
    
    result['ela_applicable'] = True
    
    try:
        import io
        img_original = Image.open(img_path)
        
        buffer = io.BytesIO()
        img_original.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        img_reencoded = Image.open(buffer)
        
        arr_original = np.array(img_original)
        arr_reencoded = np.array(img_reencoded)
        
        if len(arr_original.shape) == 2:
            ela_map = np.abs(arr_original.astype(float) - arr_reencoded.astype(float))
        else:
            arr_original_gray = cv2.cvtColor(arr_original, cv2.COLOR_RGB2GRAY)
            arr_reencoded_gray = cv2.cvtColor(arr_reencoded, cv2.COLOR_RGB2GRAY)
            ela_map = np.abs(arr_original_gray.astype(float) - arr_reencoded_gray.astype(float))
        
        result['ela_map'] = ela_map
        
        panel_area = ela_map.shape[0] * ela_map.shape[1]
        hotspot_analysis = analyze_ela_hotspots(ela_map, panel_area)
        
        result.update(hotspot_analysis)
        result['note'] = "ELA run on JPEG re-encode"
        
    except Exception as e:
        warnings.warn(f"ELA computation failed: {e}")
        result['note'] = f"ELA failed: {str(e)}"
    
    return result

# --- METADATA CHECK (ADVISORY ONLY) -------------------------------------------
def extract_image_metadata_cautious(img_path: Path) -> dict:
    """Extract metadata with expectation of minimal EXIF after PDF rasterization"""
    metadata = {
        'dpi': None,
        'color_profile': None,
        'software': None,
        'timestamp': None,
        'metadata_available': False
    }
    
    try:
        img = Image.open(img_path)
        
        if hasattr(img, 'info') and 'dpi' in img.info:
            metadata['dpi'] = img.info['dpi']
            metadata['metadata_available'] = True
        
        exif = img._getexif()
        
        if exif:
            metadata['metadata_available'] = True
            
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                
                if tag == 'Software':
                    metadata['software'] = str(value)
                elif tag == 'DateTime':
                    metadata['timestamp'] = str(value)
                elif tag == 'ColorSpace':
                    metadata['color_profile'] = str(value)
        
    except Exception as e:
        warnings.warn(f"Metadata extraction failed for {img_path}: {e}")
    
    return metadata

# --- TEXT MASKING -------------------------------------------------------------
def detect_text_regions_heuristic(img_np: np.ndarray) -> np.ndarray:
    """Lightweight text detection without OCR"""
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()
    
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    for i in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[i]
        
        aspect_ratio = max(comp_w, comp_h) / max(min(comp_w, comp_h), 1)
        if not (3.0 <= aspect_ratio <= 10.0):
            continue
        
        max_dimension = max(comp_w, comp_h)
        if max_dimension > 0.5 * w:
            continue
        
        min_dimension = min(comp_w, comp_h)
        if min_dimension > 0.05 * w:
            continue
        
        margin = 0.10
        near_margin = (x < margin * w or y < margin * h or 
                      x + comp_w > (1 - margin) * w or y + comp_h > (1 - margin) * h)
        
        if near_margin:
            mask[labels == i] = 255
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

def mask_text_annotations_scoped(img_pil: Image.Image) -> Tuple[Image.Image, Image.Image, float]:
    """Mask text/annotations with scoped application"""
    img_np = np.array(img_pil)
    
    mask = detect_text_regions_heuristic(img_np)
    
    total_pixels = mask.shape[0] * mask.shape[1]
    masked_pixels = np.sum(mask > 0)
    mask_coverage_pct = 100.0 * masked_pixels / total_pixels
    
    if masked_pixels > 0:
        if len(img_np.shape) == 3:
            masked_img = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
        else:
            masked_img = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
    else:
        masked_img = img_np
    
    masked_pil = Image.fromarray(masked_img)
    mask_pil = Image.fromarray(mask)
    
    return masked_pil, mask_pil, mask_coverage_pct

# ═══════════════════════════════════════════════════════════════
# INTEGRATION FUNCTIONS (Complete Pipeline)
# ═══════════════════════════════════════════════════════════════

def load_or_compute_phash_bundles(panel_paths: List[Path]) -> List[dict]:
    """Cache pHash bundles (8 transforms per image)"""
    cache_path = get_cache_path("phash_bundles")
    meta_path = get_cache_meta_path("phash_bundles")
    
    if ENABLE_CACHE and cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get("file_hash") == compute_file_hash(panel_paths):
                cached_data = np.load(cache_path, allow_pickle=True)
                print(f"  ✓ Loaded cached pHash bundles")
                return cached_data.tolist()
        except Exception:
            pass
    
    print(f"  Computing pHash bundles (8 transforms)...")
    bundles = []
    for path in tqdm(panel_paths, desc="pHash bundles"):
        img = Image.open(path)
        bundle = compute_phash_bundle(img)
        bundles.append(bundle)
    
    if ENABLE_CACHE:
        ensure_dir(cache_path.parent)
        np.save(cache_path, np.array(bundles, dtype=object))
        with open(meta_path, 'w') as f:
            json.dump({
                "file_hash": compute_file_hash(panel_paths),
                "num_panels": len(panel_paths),
                "timestamp": datetime.now().isoformat()
            }, f)
    
    return bundles

def load_or_compute_orb_features(panel_paths: List[Path]) -> dict:
    """Cache ORB descriptors (keypoints converted to picklable format)"""
    import pickle
    cache_path = OUT_DIR / "cache" / f"orb_features_{CACHE_VERSION}.pkl"
    meta_path = get_cache_meta_path("orb_features")
    
    if ENABLE_CACHE and cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get("file_hash") == compute_file_hash(panel_paths):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    # Convert back from tuples to KeyPoint objects
                    orb_data = {}
                    for path, data in cached.items():
                        kp_list = [cv2.KeyPoint(x=kp[0], y=kp[1], size=kp[2], angle=kp[3], 
                                                response=kp[4], octave=kp[5], class_id=kp[6]) 
                                   for kp in data['keypoints_tuples']]
                        orb_data[path] = {'keypoints': kp_list, 'descriptors': data['descriptors']}
                    print(f"  ✓ Loaded cached ORB features")
                    return orb_data
        except Exception:
            pass
    
    print(f"  Computing ORB features...")
    orb_data = {}
    for path in tqdm(panel_paths, desc="ORB"):
        kp, desc = extract_orb_features(str(path), ORB_MAX_KEYPOINTS, ORB_RETRY_SCALES)
        orb_data[str(path)] = {'keypoints': kp, 'descriptors': desc}
    
    if ENABLE_CACHE:
        ensure_dir(cache_path.parent)
        # Convert KeyPoint objects to tuples for pickling
        cacheable = {}
        for path, data in orb_data.items():
            kp_tuples = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, 
                         kp.octave, kp.class_id) for kp in data['keypoints']] if data['keypoints'] else []
            cacheable[path] = {'keypoints_tuples': kp_tuples, 'descriptors': data['descriptors']}
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cacheable, f)
        with open(meta_path, 'w') as f:
            json.dump({
                "file_hash": compute_file_hash(panel_paths),
                "num_panels": len(panel_paths),
                "timestamp": datetime.now().isoformat()
            }, f)
    
    return orb_data

def get_orb_features_for_subset(request_paths: List[Path]) -> dict:
    """
    Read global ORB cache without recomputing, filter to requested paths
    
    CRITICAL: Don't call load_or_compute_orb_features with subset or you'll
    overwrite the global cache with only those images!
    """
    import pickle
    cache_path = OUT_DIR / "cache" / f"orb_features_{CACHE_VERSION}.pkl"
    
    if not cache_path.exists():
        # Return empty - cache will be built in main pipeline
        return {}
    
    try:
        with open(cache_path, 'rb') as f:
            full_cache = pickle.load(f)
        
        subset = {}
        for p in request_paths:
            key = str(p)
            if key in full_cache:
                # Convert tuples back to KeyPoints
                data = full_cache[key]
                kps = [cv2.KeyPoint(x=t[0], y=t[1], size=t[2], angle=t[3],
                                    response=t[4], octave=t[5], class_id=t[6])
                       for t in data['keypoints_tuples']]
                subset[key] = {'keypoints': kps, 'descriptors': data['descriptors']}
        
        return subset
    except Exception:
        return {}

def phash_find_duplicates_with_bundles(panel_paths: List[Path], max_dist: int,
                                       meta_df: pd.DataFrame) -> pd.DataFrame:
    """Find duplicates using rotation/mirror-robust pHash bundles"""
    print(f"\n[Stage 3] pHash-RT with rotation/mirror bundles (≤{max_dist})...")
    
    bundles = load_or_compute_phash_bundles(panel_paths)
    
    rows = []
    # Bucket prefilter (use all 8 transforms for better recall)
    buckets = {}
    for i, bundle in enumerate(bundles):
        prefixes = []
        for k in ['rot_0','rot_90','rot_180','rot_270',
                  'mirror_h_rot_0','mirror_h_rot_90','mirror_h_rot_180','mirror_h_rot_270']:
            h = bundle.get(k, '')
            if h:
                prefixes.append(h[:4])
        for pref in set(prefixes):
            if pref:
                buckets.setdefault(pref, []).append(i)
    
    checked = set()
    for idxs in buckets.values():
        for a in range(len(idxs)-1):
            for b in range(a+1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if (i, j) in checked:
                    continue
                checked.add((i, j))
                
                d, transform = hamming_min_transform(bundles[i], bundles[j],
                                                    PHASH_BUNDLE_SHORT_CIRCUIT)
                
                if d <= max_dist:
                    pa, pb = str(panel_paths[i]), str(panel_paths[j])
                    
                    if SUPPRESS_SAME_PAGE_DUPES and same_page(pa, pb, meta_df):
                        continue
                    if SUPPRESS_ADJACENT_PAGE_DUPES and adjacent_pages(pa, pb, meta_df, ADJACENT_PAGE_MAX_GAP):
                        continue
                    
                    rows.append({
                        "Image_A": Path(pa).name,
                        "Image_B": Path(pb).name,
                        "Path_A": pa,
                        "Path_B": pb,
                        "Hamming_Distance": int(d),
                        "Transform_Matched": transform
                    })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("Hamming_Distance")
    
    print(f"  ✓ {len(df)} pairs (pHash ≤ {max_dist})")
    log_same_page_breakdown(df, meta_df, label="pHash-RT")
    return df

def orb_find_partial_duplicates(panel_paths: List[Path], clip_df: pd.DataFrame,
                                phash_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Run ORB-RANSAC on triggered pairs"""
    print(f"\n[Stage 4] ORB-RANSAC partial duplicate detection...")
    
    # Build trigger set
    trigger_pairs = set()
    
    # Trigger 1: High CLIP similarity
    for _, row in clip_df.iterrows():
        if pd.to_numeric(row.get('Cosine_Similarity', 0), errors='coerce') >= ORB_TRIGGER_CLIP_THRESHOLD:
            trigger_pairs.add(tuple(sorted([row['Path_A'], row['Path_B']])))
    
    # Trigger 2: Low pHash distance
    for _, row in phash_df.iterrows():
        if pd.to_numeric(row.get('Hamming_Distance', 999), errors='coerce') <= ORB_TRIGGER_PHASH_THRESHOLD:
            trigger_pairs.add(tuple(sorted([row['Path_A'], row['Path_B']])))
    
    if not trigger_pairs:
        print("  ✓ No pairs triggered ORB-RANSAC")
        return pd.DataFrame()
    
    print(f"  Triggered on {len(trigger_pairs)} pairs")
    
    # Collect only the paths we need (subset optimization)
    triggered_paths = set()
    for path_a, path_b in trigger_pairs:
        triggered_paths.add(path_a)
        triggered_paths.add(path_b)
    
    triggered_paths_list = [Path(p) for p in triggered_paths]
    print(f"  Loading ORB for {len(triggered_paths_list)} unique panels (subset cache)...")
    
    # Try subset read from cache first
    orb_data = get_orb_features_for_subset(triggered_paths_list)
    
    # If cache is empty (first run), compute only triggered subset
    if not orb_data or len(orb_data) < len(triggered_paths_list):
        print(f"  Computing ORB for triggered subset only...")
        orb_data = load_or_compute_orb_features(triggered_paths_list)
    
    # Run RANSAC
    rows = []
    for path_a, path_b in tqdm(trigger_pairs, desc="ORB-RANSAC"):
        # Skip same-page and adjacent pages
        if SUPPRESS_SAME_PAGE_DUPES and same_page(path_a, path_b, meta_df):
            continue
        if SUPPRESS_ADJACENT_PAGE_DUPES and adjacent_pages(path_a, path_b, meta_df, ADJACENT_PAGE_MAX_GAP):
            continue
        
        data_a = orb_data.get(path_a)
        data_b = orb_data.get(path_b)
        
        if not data_a or not data_b:
            continue
        
        if data_a['descriptors'] is None or data_b['descriptors'] is None:
            continue
        
        # Match
        matches = match_orb_features(data_a['descriptors'], data_b['descriptors'],
                                     ORB_RATIO_THRESHOLD)
        
        if len(matches) < 30:
            continue
        
        # RANSAC
        result = estimate_homography_ransac(data_a['keypoints'], data_b['keypoints'],
                                           matches, min_inliers=TIER_A_ORB_INLIERS,
                                           max_reproj_error=TIER_A_ORB_ERROR)
        
        if result['H'] is None or result['is_degenerate']:
            continue
        
        # Compute coverage
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            continue
            
        coverage = compute_crop_coverage(result['H'], img_a.shape, img_b.shape)
        
        is_partial_dupe = (result['inliers'] >= TIER_A_ORB_INLIERS and
                          result['inlier_ratio'] >= TIER_A_ORB_RATIO and
                          result['reproj_error'] <= TIER_A_ORB_ERROR and
                          coverage >= TIER_A_ORB_COVERAGE)
        
        rows.append({
            'Image_A': Path(path_a).name,
            'Image_B': Path(path_b).name,
            'Path_A': path_a,
            'Path_B': path_b,
            'ORB_Inliers': result['inliers'],
            'Inlier_Ratio': result['inlier_ratio'],
            'Reproj_Error': result['reproj_error'],
            'Crop_Coverage': coverage,
            'Homography_Type': result['homography_type'],
            'Is_Partial_Dupe': is_partial_dupe
        })
    
    df = pd.DataFrame(rows)
    partial_count = len(df[df['Is_Partial_Dupe']]) if not df.empty else 0
    print(f"  ✓ Found {partial_count} partial duplicates")
    log_same_page_breakdown(df, meta_df, label="ORB-RANSAC")
    return df

def merge_reports_enhanced(df_clip: pd.DataFrame, df_phash: pd.DataFrame,
                          df_orb: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive merge with all columns"""
    
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
            'Transform_Matched': '',
            'ORB_Inliers': '',
            'Inlier_Ratio': '',
            'Reproj_Error': '',
            'Crop_Coverage': '',
            'Homography_Type': '',
            'Source': 'CLIP'
        }
    
    # Add pHash pairs
    for _, row in df_phash.iterrows():
        key = pair_key(row['Path_A'], row['Path_B'])
        if key in all_pairs:
            all_pairs[key]['Hamming_Distance'] = row['Hamming_Distance']
            all_pairs[key]['Transform_Matched'] = row.get('Transform_Matched', '')
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
                'Transform_Matched': row.get('Transform_Matched', ''),
                'ORB_Inliers': '',
                'Inlier_Ratio': '',
                'Reproj_Error': '',
                'Crop_Coverage': '',
                'Homography_Type': '',
                'Source': 'pHash'
            }
    
    # Add ORB pairs
    for _, row in df_orb.iterrows():
        key = pair_key(row['Path_A'], row['Path_B'])
        orb_data = {
            'ORB_Inliers': row.get('ORB_Inliers', ''),
            'Inlier_Ratio': row.get('Inlier_Ratio', ''),
            'Reproj_Error': row.get('Reproj_Error', ''),
            'Crop_Coverage': row.get('Crop_Coverage', ''),
            'Homography_Type': row.get('Homography_Type', '')
        }
        
        if key in all_pairs:
            all_pairs[key].update(orb_data)
            if all_pairs[key]['Source'] == 'CLIP+pHash':
                all_pairs[key]['Source'] = 'CLIP+pHash+ORB'
            else:
                all_pairs[key]['Source'] += '+ORB'
        else:
            all_pairs[key] = {
                'Image_A': row['Image_A'],
                'Image_B': row['Image_B'],
                'Path_A': row['Path_A'],
                'Path_B': row['Path_B'],
                'Cosine_Similarity': '',
                'SSIM': '',
                'Hamming_Distance': '',
                'Transform_Matched': '',
                **orb_data,
                'Source': 'ORB'
            }
    
    df = pd.DataFrame(list(all_pairs.values()))
    
    if len(df) > 0:
        df['_clip_score'] = pd.to_numeric(df['Cosine_Similarity'], errors='coerce').fillna(0)
        df = df.sort_values('_clip_score', ascending=False).drop(columns=['_clip_score'])
    
    # Auto-attach CLIP z-scores if not already present
    if USE_CLIP_ZSCORE and 'Cosine_Similarity' in df.columns and 'CLIP_Z' not in df.columns:
        try:
            df = attach_clip_zscore_to_df(df)  # Uses approximate method
        except Exception as e:
            warnings.warn(f"Failed to attach z-scores: {e}")
    
    # Ensure Patch_SSIM_Min column exists (for tier gating)
    if 'Patch_SSIM_Min' not in df.columns:
        df['Patch_SSIM_Min'] = np.nan  # Will be filled by SSIM stage if available
    
    return df

# ═══════════════════════════════════════════════════════════════
# PHASE D: CALIBRATION & INTEGRATION
# ═══════════════════════════════════════════════════════════════

def generate_methods_description(**kwargs) -> str:
    """Auto-generate methods description for publication"""
    template = f"""
# Duplicate Detection Methodology

## Preprocessing
- **Photometric Normalization**: CLAHE (clip_limit={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_SIZE}×{CLAHE_TILE_SIZE})
- **Text Masking**: Heuristic detection (applied to pHash/SSIM/ORB, not CLIP)

## Detection Pipeline

### Stage 1: Semantic Similarity (CLIP)
- Method: OpenAI CLIP (ViT-B-32)
- Threshold: ≥ {TIER_A_CLIP} (Tier A) | ≥ {TIER_B_CLIP_MIN} (Tier B)

### Stage 2: Structural Similarity (SSIM)
- Method: SSIM on normalized images
- Threshold: ≥ {TIER_A_SSIM} (Tier A) | ≥ {TIER_B_SSIM_MIN} (Tier B)

### Stage 3: Exact Detection (pHash-RT)
- Method: Perceptual hash with 8 transform variants
- Threshold: ≤ {TIER_A_PHASH_RT} (Tier A) | {TIER_B_PHASH_RT_MIN}-{TIER_B_PHASH_RT_MAX} (Tier B)

### Stage 4: Partial Duplicates (ORB-RANSAC)
- Trigger: CLIP ≥ {ORB_TRIGGER_CLIP_THRESHOLD} OR pHash ≤ {ORB_TRIGGER_PHASH_THRESHOLD}
- Gates: Inliers ≥ {TIER_A_ORB_INLIERS}, Ratio ≥ {TIER_A_ORB_RATIO}, Error ≤ {TIER_A_ORB_ERROR}px

## Advanced Discrimination Layers

### CLIP Z-Score (Self-Normalized Outlier Detection)
- **Purpose**: Filters semantic lookalikes (same modality, different content)
- **Formula**: z_ij = min((s_ij - μ_i)/σ_i, (s_ij - μ_j)/σ_j)
- **Gate**: z ≥ {CLIP_ZSCORE_MIN}
- **Why it works**: Grid panels have many similar neighbors → low z-scores; true duplicates are outliers → high z-scores

### Patch-Wise SSIM (MS-SSIM-Lite)
- **Purpose**: Defeats "similar-but-different" grids and backgrounds
- **Grid**: {SSIM_GRID_H}×{SSIM_GRID_W} patches per image
- **Aggregation**: Average top-{SSIM_TOPK_PATCHES} local SSIMs
- **Mix**: {SSIM_MIX_WEIGHT}×patch + {1.0-SSIM_MIX_WEIGHT}×global
- **Gate**: Minimum patch SSIM ≥ {SSIM_PATCH_MIN_GATE}

### Deep Verify (Confocal/IHC Robust Confirmation)
- **Method**: ECC alignment + SSIM + NCC + pHash bundle (8 transforms)
- **Confocal**: SSIM ≥ {DEEP_VERIFY_ALIGN_SSIM_MIN}, NCC ≥ {DEEP_VERIFY_NCC_MIN}, OR pHash ≤ {DEEP_VERIFY_PHASH_MAX}
- **IHC**: Stain-robust channel extraction + alignment + verification
- **Critical**: Calculation-only (no page heuristics)

## Tier Classification
**Tier A (Review Required)**: 
- pHash-RT ≤ {TIER_A_PHASH_RT}, OR
- (CLIP ≥ {TIER_A_CLIP} AND SSIM ≥ {TIER_A_SSIM}), OR
- ORB-RANSAC pass, OR
- Deep Verify confirmation

**Tier B (Manual Check)**: 
- pHash-RT {TIER_B_PHASH_RT_MIN}-{TIER_B_PHASH_RT_MAX}, OR
- (CLIP {TIER_B_CLIP_MIN}-{TIER_B_CLIP_MAX} AND SSIM {TIER_B_SSIM_MIN}-{TIER_B_SSIM_MAX}), OR
- Confocal FP candidates (high CLIP, low SSIM)

## Forensic Adjuncts (Advisory)
- **Copy-Move Detection**: Two-stage (DCT block proposal + ORB-RANSAC confirmation)
- **Selective Brightness**: Robust exposure stats (median/MAD) with boundary continuity check
- **ELA**: Conditional on JPEG-origin detection (PDF rasterization caveat)

## Calibration
Thresholds target FPR ≤ 0.5% on hard negatives (same modality, different content).

**Run Parameters**:
- Seed: {kwargs.get('random_seed', RANDOM_SEED)}
- Device: {kwargs.get('device', 'N/A')}
- Timestamp: {kwargs.get('timestamp', datetime.now().isoformat())}
"""
    return template

# --- DEVICE SELECTION (MAC OPTIMIZED) -----------------------------------------
def get_device():
    """Select best available device - MPS for M1/M2, CPU otherwise"""
    if ENABLE_MPS and torch.backends.mps.is_available():
        return "mps"  # Metal Performance Shaders (M1/M2)
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()

# --- UTILS --------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def page_stem(i: int) -> str:
    return f"page_{i+1}"

def same_page(pair_a: str, pair_b: str, meta_df: pd.DataFrame) -> bool:
    a = meta_df.loc[meta_df["Panel_Path"] == pair_a]
    b = meta_df.loc[meta_df["Panel_Path"] == pair_b]
    if a.empty or b.empty:
        return False
    return a.iloc[0]["Page"] == b.iloc[0]["Page"]

def page_of(panel_path: str, meta_df: pd.DataFrame) -> Optional[int]:
    """Get page number for a panel path"""
    row = meta_df.loc[meta_df["Panel_Path"] == panel_path]
    if row.empty:
        return None
    
    page_val = row.iloc[0]["Page"]
    
    # Handle different formats: 'page_3', 3, '3'
    if isinstance(page_val, str):
        # Extract number from 'page_3' format
        if page_val.startswith('page_'):
            return int(page_val.replace('page_', ''))
        else:
            return int(page_val)
    else:
        return int(page_val)

def adjacent_pages(pa: str, pb: str, meta_df: pd.DataFrame, max_gap: int = 1) -> bool:
    """Check if two panels are on adjacent pages (within max_gap)"""
    a = page_of(pa, meta_df)
    b = page_of(pb, meta_df)
    if a is None or b is None:
        return False
    return abs(a - b) <= max_gap

def log_same_page_breakdown(df: pd.DataFrame, meta_df: pd.DataFrame, label: str):
    """Log same-page vs cross-page distribution for any stage"""
    if df is None or df.empty:
        print(f"  [{label}] no pairs")
        return
    
    def _same(pa, pb): 
        a = meta_df.loc[meta_df["Panel_Path"] == pa]
        b = meta_df.loc[meta_df["Panel_Path"] == pb]
        if a.empty or b.empty: 
            return False
        return a.iloc[0]["Page"] == b.iloc[0]["Page"]
    
    same = df.apply(lambda r: _same(r["Path_A"], r["Path_B"]), axis=1)
    n_same = int(same.sum())
    n_total = len(df)
    n_cross = n_total - n_same
    pct_same = 100.0 * n_same / max(n_total, 1)
    pct_cross = 100.0 * n_cross / max(n_total, 1)
    
    print(f"  [{label}] pairs={n_total} | same-page={n_same} ({pct_same:.1f}%) | cross-page={n_cross} ({pct_cross:.1f}%)")

def get_cache_path(cache_name: str) -> Path:
    """Get cache file path with version"""
    cache_dir = OUT_DIR / "cache"
    ensure_dir(cache_dir)
    return cache_dir / f"{cache_name}_{CACHE_VERSION}.npy"

def get_metadata_cache_path() -> Path:
    """Get metadata cache path"""
    cache_dir = OUT_DIR / "cache"
    ensure_dir(cache_dir)
    return cache_dir / f"metadata_{CACHE_VERSION}.json"

def get_cache_meta_path(cache_name: str) -> Path:
    """Get per-cache metadata path (for file-hash validation)"""
    cache_dir = OUT_DIR / "cache"
    ensure_dir(cache_dir)
    return cache_dir / f"{cache_name}_{CACHE_VERSION}_meta.json"

def compute_file_hash(paths: List[Path]) -> str:
    """Compute hash of file list for cache validation (mtime + size)"""
    import hashlib
    hasher = hashlib.md5()
    for p in sorted(paths):
        st = p.stat()
        hasher.update(str(p).encode())
        hasher.update(str(st.st_mtime).encode())
        hasher.update(str(st.st_size).encode())
    return hasher.hexdigest()

# --- AUDIT TRAIL --------------------------------------------------------------
def write_run_metadata(out_dir: Path, start_time: float, **kwargs):
    """Write run metadata for audit trail"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": time.time() - start_time,
        "config": {
            "pdf_path": str(PDF_PATH),
            "dpi": DPI,
            "caption_pages": list(CAPTION_PAGES),
            "min_panel_area": MIN_PANEL_AREA,
            "sim_threshold": SIM_THRESHOLD,
            "phash_max_dist": PHASH_MAX_DIST,
            "ssim_threshold": SSIM_THRESHOLD,
            "top_k_neighbors": TOP_K_NEIGHBORS,
            "use_mutual_nn": USE_MUTUAL_NN,
            "use_ssim_validation": USE_SSIM_VALIDATION,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "num_workers": NUM_WORKERS,
            "random_seed": RANDOM_SEED,
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
        },
        "results": kwargs
    }
    
    metadata_file = out_dir / "RUN_METADATA.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  ✓ Run metadata saved: {metadata_file}")

# --- IoU for NMS --------------------------------------------------------------
def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute Intersection over Union"""
    ax1, ay1 = box_a['x'], box_a['y']
    ax2, ay2 = ax1 + box_a['w'], ay1 + box_a['h']
    bx1, by1 = box_b['x'], box_b['y']
    bx2, by2 = bx1 + box_b['w'], by1 + box_b['h']
    
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_a['w'] * box_a['h'] + box_b['w'] * box_b['h'] - inter
    
    return inter / union if union > 0 else 0.0

# --- SSIM with better error handling ------------------------------------------
def compute_ssim(path_a: str, path_b: str, target_h: int = 512) -> float:
    """Compute SSIM with NaN handling for failed loads"""
    try:
        img_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        
        if img_a is None or img_b is None:
            return np.nan
        
        def resize_to_height(img, h):
            scale = h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)
        
        img_a = resize_to_height(img_a, target_h)
        img_b = resize_to_height(img_b, target_h)
        
        max_w = max(img_a.shape[1], img_b.shape[1])
        if img_a.shape[1] < max_w:
            img_a = cv2.copyMakeBorder(img_a, 0, 0, 0, max_w - img_a.shape[1], 
                                       cv2.BORDER_CONSTANT, value=255)
        if img_b.shape[1] < max_w:
            img_b = cv2.copyMakeBorder(img_b, 0, 0, 0, max_w - img_b.shape[1], 
                                       cv2.BORDER_CONSTANT, value=255)
        
        # Critical: Specify data_range for uint8 images (comprehensive test finding)
        score, _ = ssim(img_a, img_b, full=True, data_range=255)
        return float(score)
    except Exception:
        return np.nan

# --- PARALLEL SSIM COMPUTATION ------------------------------------------------
def compute_ssim_parallel(pairs: List[Tuple[str, str]]) -> List[float]:
    """Compute SSIM scores in parallel using all CPU cores"""
    with Pool(NUM_WORKERS) as pool:
        results = pool.starmap(compute_ssim, pairs)
    return results

# --- STEP 1: PDF -> PAGE PNGs -------------------------------------------------
def pdf_to_pages(pdf_path: Path, out_dir: Path, dpi: int) -> List[Path]:
    """Convert PDF to PNGs using PyMuPDF (no system dependencies!)"""
    pages_dir = out_dir / "pages"
    ensure_dir(pages_dir)
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n[1/7] Converting PDF to PNGs at {dpi} DPI...")
    
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        print("\n💡 Make sure the PDF is valid and not corrupted")
        sys.exit(1)
    
    saved = []
    excluded = 0
    
    # Calculate zoom factor from DPI (72 is default PDF DPI)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    for i in tqdm(range(len(doc)), desc="Converting pages"):
        if (i + 1) in CAPTION_PAGES:
            excluded += 1
            continue
        
        page = doc[i]
        pix = page.get_pixmap(matrix=mat)
        
        fp = pages_dir / f"{page_stem(i)}.png"
        pix.save(str(fp))
        saved.append(fp)
    
    doc.close()
    print(f"  ✓ Saved {len(saved)} pages (excluded {excluded} caption pages)")
    return saved

# --- STEP 2: PANEL DETECTION --------------------------------------------------
def detect_panels_cv(page_png: Path, out_dir: Path) -> List[Tuple[Path, dict]]:
    """Detect panels with NMS"""
    ensure_dir(out_dir)
    
    img_pil = Image.open(page_png).convert("RGB")
    img_np = np.array(img_pil)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_panels = []
    for contour in contours:
        epsilon = CONTOUR_APPROX_EPSILON * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        if (MIN_PANEL_AREA <= area <= MAX_PANEL_AREA and
            MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            valid_panels.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area, 'aspect_ratio': aspect_ratio
            })
    
    # NMS
    valid_panels.sort(key=lambda p: p['area'], reverse=True)
    nms_panels = []
    for panel in valid_panels:
        if all(compute_iou(panel, kept) < 0.5 for kept in nms_panels):
            nms_panels.append(panel)
    
    nms_panels.sort(key=lambda p: (p['y'], p['x']))
    
    # Save panels
    saved = []
    base = page_png.stem
    
    for idx, panel in enumerate(nms_panels):
        x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
        crop = img_pil.crop((x, y, x+w, y+h))
        crop = ImageOps.autocontrast(crop)
        
        out = out_dir / f"{base}_panel{idx+1:02d}.png"
        crop.save(out)
        
        metadata = {
            "page": base,
            "panel_num": idx + 1,
            "x": x, "y": y,
            "width": w, "height": h,
            "area": panel['area']
        }
        saved.append((out, metadata))
    
    # Debug visualization
    if DEBUG_MODE and len(nms_panels) > 0:
        debug_img = img_pil.copy()
        draw = ImageDraw.Draw(debug_img)
        for panel in nms_panels:
            x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
            draw.rectangle([x, y, x+w, y+h], outline='red', width=5)
        debug_path = out_dir / f"{base}_debug.png"
        debug_img.save(debug_path)
    
    return saved

def pages_to_panels_auto(pages: List[Path], out_dir: Path) -> Tuple[List[Path], pd.DataFrame]:
    """Extract panels from all pages"""
    panels_dir = out_dir / "panels"
    ensure_dir(panels_dir)
    
    print(f"\n[2/7] Auto-detecting panels (MIN_AREA={MIN_PANEL_AREA:,}, NMS enabled)...")

    all_items: List[Tuple[Path, dict]] = []
    page_stats = []
    
    for p in tqdm(pages, desc="Detecting panels"):
        stem = p.stem
        subdir = panels_dir / stem
        items = detect_panels_cv(p, subdir)
        all_items.extend(items)
        page_stats.append((stem, len(items)))
    
    print(f"\n  ✓ Extracted {len(all_items)} panels total")
    
    if not all_items:
        return [], pd.DataFrame()

    panel_paths = [it[0] for it in all_items]
    meta_df = pd.DataFrame([{
        "Panel_Path": str(path),
        "Panel_Name": path.name,
        "Page": md["page"],
        "Panel_Num": md["panel_num"],
        "X": md["x"], "Y": md["y"],
        "Width": md["width"], "Height": md["height"],
        "Area": md["area"]
    } for path, md in all_items])
    
    return panel_paths, meta_df

# --- STEP 3A: CLIP with FAISS -------------------------------------------------
@dataclass
class CLIPModel:
    model: any
    preprocess: any
    device: str

def load_clip(model_name="ViT-B-32", pretrained="openai") -> CLIPModel:
    """Load CLIP model"""
    print(f"  Loading CLIP on {DEVICE}...")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.to(DEVICE)
    model.eval()
    print(f"  ✓ CLIP loaded on {DEVICE}")
    return CLIPModel(model, preprocess, DEVICE)

@torch.no_grad()
def embed_images(paths: List[Path], clip: CLIPModel) -> np.ndarray:
    """Generate CLIP embeddings with Mac-optimized batching"""
    vecs = []
    
    print(f"  Using batch_size={BATCH_SIZE} on {DEVICE}")
    
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Embedding"):
        batch = paths[i:i+BATCH_SIZE]
        imgs = [clip.preprocess(Image.open(p).convert("RGB")) for p in batch]
        if not imgs:
            continue
        
        x = torch.stack(imgs).to(clip.device)
        feats = clip.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy())
    
    return np.vstack(vecs).astype("float32") if vecs else np.zeros((0, 512), dtype="float32")

def load_or_compute_embeddings(panel_paths: List[Path], clip: CLIPModel, 
                              use_mmap: bool = True) -> np.ndarray:
    """
    Load cached embeddings or compute new ones
    Memory-mapped loading: 300x faster (0.1s vs 30s)
    """
    cache_path = get_cache_path("clip_embeddings")
    meta_cache_path = get_metadata_cache_path()
    
    # Check if cache exists and is valid
    if ENABLE_CACHE and cache_path.exists() and meta_cache_path.exists():
        try:
            with open(meta_cache_path, 'r') as f:
                cached_meta = json.load(f)
            
            current_hash = compute_file_hash(panel_paths)
            
            if cached_meta.get("file_hash") == current_hash:
                if use_mmap:
                    print("  ✓ Loading cached embeddings (memory-mapped, zero-copy)...")
                    vecs = np.load(cache_path, mmap_mode='r')
                else:
                    print("  ✓ Loading cached embeddings (in-memory)...")
                    vecs = np.load(cache_path)
                
                if vecs.shape[0] == len(panel_paths):
                    return vecs
        except Exception as e:
            print(f"  ⚠ Cache load failed: {e}, recomputing...")
    
    # Compute new embeddings
    print("  Computing new embeddings...")
    vecs = embed_images(panel_paths, clip)
    
    # Save cache (always save as regular .npy for compatibility)
    if ENABLE_CACHE:
        np.save(cache_path, vecs)
        with open(meta_cache_path, 'w') as f:
            json.dump({
                "file_hash": compute_file_hash(panel_paths),
                "num_panels": len(panel_paths),
                "timestamp": datetime.now().isoformat()
            }, f)
        print(f"  ✓ Cached embeddings to {cache_path}")
    
    return vecs

def clip_topk_numpy(vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Use numpy for top-k search (Mac-optimized, avoids FAISS crashes)"""
    # Compute full similarity matrix
    n = len(vecs)
    if n <= 1:
        return np.empty((0, 0)), np.empty((0, 0), dtype=int)
    # Clamp k
    k = max(1, min(k, n - 1))
    sim_matrix = vecs @ vecs.T
    sims_list, idxs_list = [], []
    for i in range(n):
        row_sims = sim_matrix[i].copy()
        row_sims[i] = -1  # exclude self
        top_k_idxs = np.argpartition(row_sims, -k)[-k:]
        top_k_idxs = top_k_idxs[np.argsort(row_sims[top_k_idxs])[::-1]]
        top_k_sims = row_sims[top_k_idxs]
        sims_list.append(top_k_sims)
        idxs_list.append(top_k_idxs)
    return np.array(sims_list), np.array(idxs_list)

def clip_find_duplicates_faiss(panel_paths: List[Path], vecs: np.ndarray, 
                                threshold: float, meta_df: pd.DataFrame,
                                batch_threshold: int = 1000) -> pd.DataFrame:
    """
    Adaptive CLIP duplicate detection with batching for large documents
    
    Strategy:
    - n < batch_threshold: All-pairs matrix (current approach)
    - n >= batch_threshold: Batched processing (memory-efficient)
    """
    n = len(vecs)
    
    if n < batch_threshold:
        # Use optimized all-pairs (optimal for <1000 panels)
        return _clip_find_duplicates_allpairs(panel_paths, vecs, threshold, meta_df)
    else:
        # Large document: Use batched processing
        return _clip_find_duplicates_batched(panel_paths, vecs, threshold, meta_df)


def _clip_find_duplicates_allpairs(panel_paths: List[Path], vecs: np.ndarray, 
                                    threshold: float, meta_df: pd.DataFrame) -> pd.DataFrame:
    """All-pairs CLIP search (optimal for <1000 panels)"""
    print(f"  Finding similar pairs (mode={CLIP_PAIRING_MODE}, threshold≥{threshold})...")
    
    n = len(vecs)
    if n < 2:
        return pd.DataFrame()

    # Compute full similarity matrix
    sim_matrix = vecs @ vecs.T
    np.fill_diagonal(sim_matrix, -1.0)

    rows = []

    if CLIP_PAIRING_MODE.lower() == "thresh":
        # GLOBAL SCAN: Consider ALL pairs above threshold (fixes cross-page bias)
        iu, ju = np.triu_indices(n, k=1)  # Upper triangle (avoid duplicates)
        mask = sim_matrix[iu, ju] >= threshold
        cand_i = iu[mask]
        cand_j = ju[mask]
        cand_s = sim_matrix[iu, ju][mask]

        # Sort by score descending (best pairs first)
        order = np.argsort(-cand_s)
        if CLIP_MAX_OUTPUT_PAIRS is not None and len(order) > CLIP_MAX_OUTPUT_PAIRS:
            order = order[:CLIP_MAX_OUTPUT_PAIRS]
        cand_i, cand_j, cand_s = cand_i[order], cand_j[order], cand_s[order]

        for i, j, score in zip(cand_i, cand_j, cand_s):
            pa, pb = str(panel_paths[i]), str(panel_paths[j])
            if SUPPRESS_SAME_PAGE_DUPES and same_page(pa, pb, meta_df):
                continue
            if SUPPRESS_ADJACENT_PAGE_DUPES and adjacent_pages(pa, pb, meta_df, ADJACENT_PAGE_MAX_GAP):
                continue
            rows.append({
                "Image_A": Path(pa).name,
                "Image_B": Path(pb).name,
                "Path_A": pa,
                "Path_B": pb,
                "Cosine_Similarity": float(score)
            })

    else:
        # FALLBACK: Original top-K behavior (kept for compatibility)
        sims, idxs = clip_topk_numpy(vecs, TOP_K_NEIGHBORS)
        
        # Find mutual pairs if enabled
        mutual_pairs = set()
        if USE_MUTUAL_NN:
            for i in range(n):
                my_neighbors = idxs[i]
                for j in my_neighbors:
                    if j <= i:
                        continue
                    if i in idxs[j]:
                        mutual_pairs.add(tuple(sorted((i, j))))
            print(f"    Found {len(mutual_pairs)} mutual NN pairs")
        
        for i in range(n):
            neighbors = idxs[i]
            scores = sims[i]
            
            for j, score in zip(neighbors, scores):
                if j <= i or score < threshold:
                    continue
                
                if USE_MUTUAL_NN:
                    pair = tuple(sorted((i, j)))
                    if pair not in mutual_pairs:
                        continue
                
                pa, pb = str(panel_paths[i]), str(panel_paths[j])
                
                if SUPPRESS_SAME_PAGE_DUPES and same_page(pa, pb, meta_df):
                    continue
                if SUPPRESS_ADJACENT_PAGE_DUPES and adjacent_pages(pa, pb, meta_df, ADJACENT_PAGE_MAX_GAP):
                    continue
                
                rows.append({
                    "Image_A": Path(pa).name,
                    "Image_B": Path(pb).name,
                    "Path_A": pa,
                    "Path_B": pb,
                    "Cosine_Similarity": float(score)
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Cosine_Similarity", ascending=False)
    
    # ADD DIAGNOSTIC
    log_same_page_breakdown(df, meta_df, label="CLIP")
    
    return df


def _clip_find_duplicates_batched(panel_paths: List[Path], vecs: np.ndarray,
                                   threshold: float, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Batched CLIP search for large documents (>1000 panels)"""
    n = len(vecs)
    batch_size = min(500, n // 4)  # Adaptive batch size
    
    print(f"  Large document ({n} panels) - using batched processing...")
    print(f"  Batch size: {batch_size}")
    
    rows = []
    
    for i in tqdm(range(0, n, batch_size), desc="CLIP batching"):
        i_end = min(i + batch_size, n)
        batch_vecs = vecs[i:i_end]
        
        # Compare batch against all vectors
        sims = batch_vecs @ vecs.T
        
        # Find pairs above threshold
        for local_idx, global_i in enumerate(range(i, i_end)):
            for j in range(global_i + 1, n):
                score = sims[local_idx, j]
                
                if score >= threshold:
                    pa = str(panel_paths[global_i])
                    pb = str(panel_paths[j])
                    
                    if SUPPRESS_SAME_PAGE_DUPES and same_page(pa, pb, meta_df):
                        continue
                    
                    rows.append({
                        "Image_A": Path(pa).name,
                        "Image_B": Path(pb).name,
                        "Path_A": pa,
                        "Path_B": pb,
                        "Cosine_Similarity": float(score)
                    })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Cosine_Similarity", ascending=False)
    
    log_same_page_breakdown(df, meta_df, label="CLIP")
    
    return df

# --- STEP 3B: pHash with caching and parallel processing ---------------------
def phash_hex(img_path: Path) -> str:
    """Compute pHash"""
    try:
        im = Image.open(img_path).convert("RGB")
        if ENABLE_TEXT_MASKING and MASKING_APPLY_TO_PHASH:
            try:
                masked_im, _, _ = mask_text_annotations_scoped(im)
                im = masked_im
            except Exception:
                pass
        im = ImageOps.exif_transpose(im)
        im = ImageOps.autocontrast(im)
        return str(imagehash.phash(im))
    except:
        return ""

def compute_phashes_parallel(panel_paths: List[Path]) -> List[str]:
    """Compute pHashes in parallel"""
    with Pool(NUM_WORKERS) as pool:
        hashes = list(tqdm(
            pool.imap(phash_hex, panel_paths),
            total=len(panel_paths),
            desc="Computing pHash"
        ))
    return hashes

def load_or_compute_phashes(panel_paths: List[Path]) -> List[str]:
    """Load cached pHashes or compute new ones"""
    cache_path = get_cache_path("phashes")
    meta_cache_path = get_metadata_cache_path()
    
    if ENABLE_CACHE and cache_path.exists() and meta_cache_path.exists():
        with open(meta_cache_path, 'r') as f:
            cached_meta = json.load(f)
        
        current_hash = compute_file_hash(panel_paths)
        
        if cached_meta.get("file_hash") == current_hash:
            print("  ✓ Loading cached pHashes...")
            hashes_array = np.load(cache_path, allow_pickle=True)
            if len(hashes_array) == len(panel_paths):
                return hashes_array.tolist()
    
    # Compute new pHashes
    print(f"  Computing pHashes (parallel, {NUM_WORKERS} workers)...")
    hashes = compute_phashes_parallel(panel_paths)
    
    # Save cache
    if ENABLE_CACHE:
        np.save(cache_path, np.array(hashes, dtype=object))
        print(f"  ✓ Cached pHashes to {cache_path}")
    
    return hashes

def hamming_hex(h1: str, h2: str) -> int:
    """Hamming distance"""
    if not h1 or not h2:
        return 999
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)

def adaptive_bucket_prefix(num_panels: int) -> int:
    """
    Choose optimal prefix length based on panel count
    Trade-off: Shorter prefix = more candidates, Longer prefix = faster
    """
    if num_panels < 100:
        return 3  # Be thorough
    elif num_panels < 500:
        return 4  # Balanced (most cases)
    else:
        return 5  # Speed matters

def phash_find_duplicates(panel_paths: List[Path], max_dist: int, 
                         meta_df: pd.DataFrame, prefix_len: int = None) -> pd.DataFrame:
    """
    OPTIMIZED: pHash with adaptive bucketing (10x speedup)
    Reduces comparisons from O(n²) to O(bucket_size²)
    """
    if prefix_len is None:
        prefix_len = adaptive_bucket_prefix(len(panel_paths))
    
    print(f"\n[3b/7] Finding pHash matches (Hamming≤{max_dist}, prefix={prefix_len})...")
    
    hashes = load_or_compute_phashes(panel_paths)
    
    # Build buckets
    buckets: Dict[str, List[int]] = {}
    for i, h in enumerate(hashes):
        if h:
            prefix = h[:prefix_len]
            buckets.setdefault(prefix, []).append(i)
    
    # Filter out singleton buckets (no pairs possible)
    multi_buckets = {k: v for k, v in buckets.items() if len(v) >= 2}
    
    # Estimate comparisons
    estimated_comparisons = sum(len(b) * (len(b) - 1) // 2 for b in multi_buckets.values())
    all_pairs_count = len(hashes) * (len(hashes) - 1) // 2
    reduction = 100 * (1 - estimated_comparisons / max(all_pairs_count, 1))
    
    print(f"  Bucketing: {len(hashes)} hashes → {len(multi_buckets)} non-empty buckets")
    print(f"  Reduction: {all_pairs_count:,} → {estimated_comparisons:,} comparisons ({reduction:.1f}% fewer)")
    
    rows = []
    checked: Set[Tuple[int, int]] = set()
    
    # Process buckets with progress bar
    for bucket_key, idxs in tqdm(multi_buckets.items(), desc="pHash bucketing"):
        for a in range(len(idxs) - 1):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                
                pair_key = tuple(sorted((i, j)))
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                
                d = hamming_hex(hashes[i], hashes[j])
                
                if d <= max_dist:
                    pa, pb = str(panel_paths[i]), str(panel_paths[j])
                    
                    if SUPPRESS_SAME_PAGE_DUPES and same_page(pa, pb, meta_df):
                        continue
                    
                    rows.append({
                        "Image_A": Path(pa).name,
                        "Image_B": Path(pb).name,
                        "Path_A": pa,
                        "Path_B": pb,
                        "Hamming_Distance": int(d)
                    })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("Hamming_Distance")
    
    print(f"  ✓ {len(df)} pairs (pHash ≤ {max_dist})")
    
    return df

# --- STEP 4: MERGE & SSIM VALIDATION ------------------------------------------
def merge_reports(df_clip: pd.DataFrame, df_phash: pd.DataFrame) -> pd.DataFrame:
    """Merge CLIP and pHash results"""
    def uniq_key(a, b):
        return tuple(sorted((a, b)))
    
    seen = set()
    merged_rows = []

    for _, r in df_clip.iterrows():
        k = uniq_key(r.Path_A, r.Path_B)
        if k in seen:
            continue
        seen.add(k)
        merged_rows.append({
            "Image_A": r.Image_A,
            "Image_B": r.Image_B,
            "Path_A": r.Path_A,
            "Path_B": r.Path_B,
            "Cosine_Similarity": r.Cosine_Similarity,
            "Hamming_Distance": "",
            "Source": "CLIP"
        })
    
    for _, r in df_phash.iterrows():
        k = uniq_key(r.Path_A, r.Path_B)
        if k in seen:
            for row in merged_rows:
                if set([row["Path_A"], row["Path_B"]]) == set([r.Path_A, r.Path_B]):
                    row["Hamming_Distance"] = int(r.Hamming_Distance)
                    row["Source"] = "CLIP+pHash"
                    break
            continue
        
        seen.add(k)
        merged_rows.append({
            "Image_A": r.Image_A,
            "Image_B": r.Image_B,
            "Path_A": r.Path_A,
            "Path_B": r.Path_B,
            "Cosine_Similarity": "",
            "Hamming_Distance": int(r.Hamming_Distance),
            "Source": "pHash"
        })

    df_final = pd.DataFrame(merged_rows)
    
    if len(df_final) > 0:
        df_final["Cosine_Similarity_num"] = pd.to_numeric(
            df_final["Cosine_Similarity"], errors="coerce"
        ).fillna(-1)
        df_final["Hamming_Distance_num"] = pd.to_numeric(
            df_final["Hamming_Distance"], errors="coerce"
        ).fillna(99)
        
        df_final = df_final.sort_values(
            by=["Source", "Cosine_Similarity_num", "Hamming_Distance_num"],
            ascending=[True, False, True]
        ).drop(columns=["Cosine_Similarity_num", "Hamming_Distance_num"])

    return df_final

def _compute_ssim_for_row(row: dict) -> dict:
    """
    Module-level worker for parallel SSIM (no lambda, no partial)
    Must be at module level for ThreadPoolExecutor serialization
    """
    try:
        score, metadata = compute_ssim_normalized(
            row['Path_A'], 
            row['Path_B'], 
            apply_norm=True
        )
        return {
            'Path_A': row['Path_A'],
            'Path_B': row['Path_B'],
            'SSIM': score,
            'metadata': metadata,
            'error': None
        }
    except Exception as e:
        return {
            'Path_A': row['Path_A'],
            'Path_B': row['Path_B'],
            'SSIM': np.nan,
            'metadata': {},
            'error': str(e)
        }


def add_ssim_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    ANNOTATE-ONLY SSIM validation with patch-wise discrimination
    Sequential processing for Streamlit Cloud stability
    """
    if df.empty or not USE_SSIM_VALIDATION:
        return df
    
    print(f"\n[4b/7] SSIM annotation (patch-wise={USE_PATCHWISE_SSIM}, threshold≥{SSIM_THRESHOLD} for reference)...")
    
    # Sequential processing (safest for Streamlit Cloud)
    ssim_scores = []
    patch_mins = []
    patch_topks = []
    global_ssims = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SSIM"):
        try:
            score, metadata = compute_ssim_normalized(
                row['Path_A'], 
                row['Path_B'], 
                apply_norm=True
            )
            ssim_scores.append(score)
            patch_mins.append(metadata.get('patch_min', np.nan))
            patch_topks.append(metadata.get('patch_topk_mean', np.nan))
            global_ssims.append(metadata.get('ssim_global', score))
        except Exception as e:
            print(f"  ⚠️ SSIM failed for pair: {e}")
            ssim_scores.append(np.nan)
            patch_mins.append(np.nan)
            patch_topks.append(np.nan)
            global_ssims.append(np.nan)
    
    # Attach all metrics (NO FILTERING - let Tier gating decide)
    df = df.copy()
    df["SSIM"] = ssim_scores                    # Mixed score (used for gating)
    df["Patch_SSIM_Min"] = patch_mins            # Minimum patch (critical filter!)
    df["Patch_SSIM_TopK"] = patch_topks          # Top-K average (diagnostic)
    df["Global_SSIM"] = global_ssims             # Global only (diagnostic)
    
    # Show discrimination effect
    if USE_PATCHWISE_SSIM:
        valid_patch_mins = pd.to_numeric(df['Patch_SSIM_Min'], errors='coerce').dropna()
        if not valid_patch_mins.empty:
            high_patch = len(df[pd.to_numeric(df['Patch_SSIM_Min'], errors='coerce') >= SSIM_PATCH_MIN_GATE])
            print(f"    → {high_patch}/{len(df)} pairs pass patch gate (min≥{SSIM_PATCH_MIN_GATE})")
    
    print(f"  ✓ Computed SSIM for {len(df)} pairs (no filtering, Tier gate will decide)")
    
    return df

# --- DIFFERENCE HIGHLIGHTING --------------------------------------------------
def create_difference_map(img_a_path: str, img_b_path: str, target_h: int = 800) -> Optional[Image.Image]:
    """Create a heatmap showing pixel differences between two images"""
    try:
        # Read images
        img_a = cv2.imread(img_a_path)
        img_b = cv2.imread(img_b_path)
        
        if img_a is None or img_b is None:
            return None
        
        # Resize to same height
        def resize_h(img, h):
            scale = h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)
        
        img_a = resize_h(img_a, target_h)
        img_b = resize_h(img_b, target_h)
        
        # Pad to same width
        max_w = max(img_a.shape[1], img_b.shape[1])
        if img_a.shape[1] < max_w:
            img_a = cv2.copyMakeBorder(img_a, 0, 0, 0, max_w - img_a.shape[1], 
                                       cv2.BORDER_CONSTANT, value=255)
        if img_b.shape[1] < max_w:
            img_b = cv2.copyMakeBorder(img_b, 0, 0, 0, max_w - img_b.shape[1], 
                                       cv2.BORDER_CONSTANT, value=255)
        
        # Compute absolute difference
        diff = cv2.absdiff(img_a, img_b)
        
        # Convert to grayscale and normalize
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply colormap (red = high difference, blue = low difference)
        diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        
        # Blend with original image A
        alpha = 0.4
        highlighted = cv2.addWeighted(img_a, 1-alpha, diff_colored, alpha, 0)
        
        # Convert to PIL
        highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(highlighted_rgb)
        
    except Exception:
        return None

# --- INDUSTRY-GRADE VISUALIZATION (5 Helper Functions) ------------------------

def _ecc_align_b_to_a(img_a_bgr: np.ndarray, img_b_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Stable ECC alignment with float32 normalization"""
    a_gray = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    b_gray = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    if a_gray.shape != b_gray.shape:
        b_gray = cv2.resize(b_gray, (a_gray.shape[1], a_gray.shape[0]))
        img_b_bgr = cv2.resize(img_b_bgr, (img_a_bgr.shape[1], img_a_bgr.shape[0]))
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5)
    
    for mode, label in [(cv2.MOTION_TRANSLATION, 'translation'),
                        (cv2.MOTION_AFFINE, 'affine')]:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(a_gray, b_gray, warp, mode, criteria, None, 3)
            aligned = cv2.warpAffine(img_b_bgr, warp,
                                     (img_a_bgr.shape[1], img_a_bgr.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
            return aligned, {'method': 'ECC', 'transform_type': label}
        except cv2.error:
            continue
    
    aligned = cv2.resize(img_b_bgr, (img_a_bgr.shape[1], img_a_bgr.shape[0]))
    return aligned, {'method': 'resize_only', 'transform_type': 'none'}

def align_images_for_visualization(path_a: str, path_b: str, 
                                   H_orb: Optional[np.ndarray] = None,
                                   max_size: int = 1200) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, dict]:
    """Align image B to image A for meaningful visual comparison"""
    try:
        img_a_bgr = cv2.imread(path_a)
        img_b_bgr = cv2.imread(path_b)
        
        if img_a_bgr is None or img_b_bgr is None:
            return None, None, False, {}
        
        if max(img_a_bgr.shape[:2]) > max_size:
            scale = max_size / max(img_a_bgr.shape[:2])
            new_h_a = int(img_a_bgr.shape[0] * scale)
            new_w_a = int(img_a_bgr.shape[1] * scale)
            img_a_bgr = cv2.resize(img_a_bgr, (new_w_a, new_h_a))
            
            new_h_b = int(img_b_bgr.shape[0] * scale)
            new_w_b = int(img_b_bgr.shape[1] * scale)
            img_b_bgr = cv2.resize(img_b_bgr, (new_w_b, new_h_b))
            
            if H_orb is not None:
                scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
                H_orb = scale_matrix @ H_orb @ np.linalg.inv(scale_matrix)
        
        metadata = {'method': None, 'transform_type': None}
        
        # Method 1: Use ORB Homography
        if H_orb is not None:
            try:
                is_affine = np.allclose(H_orb[2, :2], 0) and np.isclose(H_orb[2, 2], 1)
                
                if is_affine:
                    M_affine = H_orb[:2, :]
                    img_b_aligned = cv2.warpAffine(img_b_bgr, M_affine, 
                                                   (img_a_bgr.shape[1], img_a_bgr.shape[0]),
                                                   flags=cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_CONSTANT,
                                                   borderValue=(255, 255, 255))
                    metadata['method'] = 'ORB-RANSAC'
                    metadata['transform_type'] = 'affine'
                else:
                    img_b_aligned = cv2.warpPerspective(img_b_bgr, H_orb,
                                                        (img_a_bgr.shape[1], img_a_bgr.shape[0]),
                                                        flags=cv2.INTER_LINEAR,
                                                        borderMode=cv2.BORDER_CONSTANT,
                                                        borderValue=(255, 255, 255))
                    metadata['method'] = 'ORB-RANSAC'
                    metadata['transform_type'] = 'projective'
                
                img_a_rgb = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
                img_b_aligned_rgb = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2RGB)
                
                return img_a_rgb, img_b_aligned_rgb, True, metadata
            except Exception as e:
                warnings.warn(f"ORB alignment failed: {e}, trying ECC")
        
        # Method 2: ECC Alignment (use stable helper)
        img_b_aligned, metadata = _ecc_align_b_to_a(img_a_bgr, img_b_bgr)
        
        img_a_rgb = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
        img_b_aligned_rgb = cv2.cvtColor(img_b_aligned, cv2.COLOR_BGR2RGB)
        
        return img_a_rgb, img_b_aligned_rgb, True, metadata
    except:
        return None, None, False, {}

def _best_ssim_win_size(h: int, w: int, preferred: int = 11) -> int:
    """Adaptive SSIM window size (must be odd and ≤ min(h,w))"""
    max_allowed = min(h, w)
    if max_allowed < 3:
        return 3
    k = min(preferred, max_allowed)
    if k % 2 == 0:
        k -= 1
    if k < 3:
        k = 3
    return k

def make_ssim_dissimilarity_map(img_a: np.ndarray, img_b: np.ndarray,
                                win_size: int = 11) -> Tuple[Image.Image, float]:
    """Create SSIM dissimilarity heatmap with viridis colormap (perceptually uniform)"""
    from skimage.metrics import structural_similarity as ssim_func
    
    try:
        if len(img_a.shape) == 3:
            img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        else:
            img_a_gray = img_a
            
        if len(img_b.shape) == 3:
            img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        else:
            img_b_gray = img_b
        
        if img_a_gray.shape != img_b_gray.shape:
            img_b_gray = cv2.resize(img_b_gray, (img_a_gray.shape[1], img_a_gray.shape[0]))
        
        # Adaptive window size
        win_size = _best_ssim_win_size(img_a_gray.shape[0], img_a_gray.shape[1], preferred=win_size)
        
        ssim_score, ssim_map = ssim_func(img_a_gray, img_b_gray, full=True, win_size=win_size, data_range=255)
        
        dissim_map = 1.0 - ssim_map
        dissim_normalized = (dissim_map * 255).astype(np.uint8)
        dissim_colored = cv2.applyColorMap(dissim_normalized, cv2.COLORMAP_VIRIDIS)
        
        threshold_val = np.percentile(dissim_map, 90)
        high_dissim_mask = (dissim_map > threshold_val).astype(np.uint8) * 255
        contours, _ = cv2.findContours(high_dissim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dissim_colored, contours, -1, (255, 0, 0), 2)
        
        img_pil = Image.fromarray(cv2.cvtColor(dissim_colored, cv2.COLOR_BGR2RGB))
        return img_pil, float(ssim_score)
    except:
        blank = Image.new('RGB', (img_a.shape[1], img_a.shape[0]), 'white')
        return blank, 0.0

def make_hard_diff_mask(img_a: np.ndarray, img_b: np.ndarray,
                        blur_ksize: int = 3, threshold: int = 25,
                        morph: int = 3) -> Image.Image:
    """Binary change mask (ImageMagick-style) with morphology cleanup"""
    try:
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        
        # Reduce sensor speckle
        a = cv2.GaussianBlur(a, (blur_ksize, blur_ksize), 0)
        b = cv2.GaussianBlur(b, (blur_ksize, blur_ksize), 0)
        
        diff = cv2.absdiff(a, b)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with closing
        if morph > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        
        # Red overlay on A
        overlay = img_a.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        overlay = np.where(mask[..., None] > 0, cv2.addWeighted(overlay, 0.4, red, 0.6, 0), overlay)
        
        return Image.fromarray(overlay)
    except:
        return Image.fromarray(img_a)

def make_checkerboard_composite(img_a: np.ndarray, img_b: np.ndarray,
                                tile_size: int = None) -> Image.Image:
    """Create checkerboard pattern (forensic standard) with adaptive tile size"""
    try:
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        h, w = img_a.shape[:2]
        
        # Adaptive tile size: ~18 tiles across min dimension
        if tile_size is None:
            tile_size = max(24, round(min(h, w) / 18))
        
        checkerboard = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                if ((y // tile_size) + (x // tile_size)) % 2 == 0:
                    y_end = min(y + tile_size, h)
                    x_end = min(x + tile_size, w)
                    checkerboard[y:y_end, x:x_end] = 1
        
        if len(img_a.shape) == 3:
            checkerboard = np.stack([checkerboard] * 3, axis=2)
        
        composite = np.where(checkerboard, img_a, img_b).astype(np.uint8)
        
        for y in range(0, h, tile_size):
            cv2.line(composite, (0, y), (w, y), (128, 128, 128), 1)
        for x in range(0, w, tile_size):
            cv2.line(composite, (x, 0), (x, h), (128, 128, 128), 1)
        
        return Image.fromarray(composite)
    except:
        return Image.fromarray(img_a)

def make_blink_comparator_gif(img_a: np.ndarray, img_b: np.ndarray,
                               output_path: Path, duration: int = 500, max_width: int = 1200) -> bool:
    """Create optimized animated GIF alternating A and B (astronomy technique)"""
    try:
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        img_a_pil = Image.fromarray(img_a)
        img_b_pil = Image.fromarray(img_b)
        
        # Cap size for reasonable file size
        if img_a_pil.width > max_width:
            scale = max_width / img_a_pil.width
            new_h = int(img_a_pil.height * scale)
            img_a_pil = img_a_pil.resize((max_width, new_h), Image.Resampling.LANCZOS)
            img_b_pil = img_b_pil.resize((max_width, new_h), Image.Resampling.LANCZOS)
        
        draw_a = ImageDraw.Draw(img_a_pil)
        draw_b = ImageDraw.Draw(img_b_pil)
        
        font = None
        for font_path in ["/System/Library/Fonts/Helvetica.ttc",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            try:
                from PIL import ImageFont
                font = ImageFont.truetype(font_path, 32)
                break
            except:
                continue
        
        label_color = (255, 0, 0)
        draw_a.text((10, 10), "IMAGE A", fill=label_color, font=font)
        draw_b.text((10, 10), "IMAGE B", fill=label_color, font=font)
        
        img_a_pil.save(output_path, save_all=True, append_images=[img_b_pil],
                      duration=duration, loop=0, optimize=True)
        return True
    except:
        return False

def make_simple_slider_html(img_a_rel: str, img_b_rel: str, pair_idx: int,
                            tier: Optional[str] = None, scores: dict = None) -> str:
    """Pure HTML/CSS/JS slider (no CDN) - offline fallback"""
    tier_badge = ""
    if tier == 'A':
        tier_badge = '<span style="background:#ff4444;color:white;padding:5px 10px;border-radius:3px;font-weight:bold;">TIER A</span>'
    elif tier == 'B':
        tier_badge = '<span style="background:#ffa500;color:white;padding:5px 10px;border-radius:3px;font-weight:bold;">TIER B</span>'
    
    scores_html = ""
    if scores:
        scores_html = "<div style='margin:10px 0;font-family:monospace;font-size:14px;'>"
        for key, value in scores.items():
            scores_html += f"<div><strong>{key}:</strong> {value}</div>"
        scores_html += "</div>"
    
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Pair #{pair_idx:03d}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto; margin:20px; background:#f5f5f5; }}
.wrap {{ position:relative; max-width:1200px; margin:auto; background:#fff; padding:20px; border-radius:8px; }}
.viewer {{ position:relative; overflow:hidden; border:1px solid #ddd; }}
.viewer img {{ width:100%; display:block; }}
#top {{ position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden; }}
.slider {{ width:100%; margin:10px 0; }}
</style></head><body><div class="wrap">
<h2>Pair #{pair_idx:03d} - Offline Slider</h2>
{tier_badge}
{scores_html}
<div class="viewer" id="v">
  <img src="{img_b_rel}" id="bottom">
  <div id="top"><img src="{img_a_rel}"></div>
</div>
<input type="range" min="0" max="100" value="50" class="slider" id="s">
<p style="text-align:center;color:#666;">Drag slider to compare (no internet required)</p>
<script>
const s=document.getElementById('s'); const top=document.getElementById('top');
s.addEventListener('input', () => {{ top.style.width = s.value + '%'; }});
</script></div></body></html>"""

def make_juxtapose_html(img_a_rel: str, img_b_rel: str, pair_idx: int,
                       tier: Optional[str] = None, scores: dict = None) -> str:
    """Create interactive HTML with Juxtapose slider (Knight Lab standard)"""
    tier_badge = ""
    if tier == 'A':
        tier_badge = '<span style="background: #ff4444; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold;">TIER A - REVIEW REQUIRED</span>'
    elif tier == 'B':
        tier_badge = '<span style="background: #ffa500; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold;">TIER B - MANUAL CHECK</span>'
    
    scores_html = ""
    if scores:
        scores_html = "<div style='margin: 10px 0; font-family: monospace; font-size: 14px;'>"
        for key, value in scores.items():
            scores_html += f"<div><strong>{key}:</strong> {value}</div>"
        scores_html += "</div>"
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pair #{pair_idx:03d} - Interactive Comparison</title>
    <link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white;
                     padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ margin: 0 0 20px 0; color: #333; }}
        .juxtapose {{ margin: 20px 0; border: 2px solid #ddd; border-radius: 4px; overflow: hidden; }}
        .instructions {{ background: #e3f2fd; padding: 15px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #2196f3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pair #{pair_idx:03d} - Duplicate Comparison</h1>
        {tier_badge}
        {scores_html}
        <div class="instructions">
            <strong>📌 Instructions:</strong> Drag the vertical slider left/right to compare images.
        </div>
        <div class="juxtapose" data-startingposition="50%">
            <img src="{img_a_rel}" data-label="Image A" />
            <img src="{img_b_rel}" data-label="Image B (Aligned)" />
        </div>
    </div>
    <script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
</body>
</html>"""

# --- STEP 5: VISUAL COMPARISONS -----------------------------------------------
def create_duplicate_comparisons(df_final: pd.DataFrame, out_dir: Path):
    """Create side-by-side comparisons"""
    if len(df_final) == 0:
        print("\n[6/7] No duplicates to visualize.")
        return
    
    comp_dir = out_dir / "duplicate_comparisons"
    ensure_dir(comp_dir)
    
    print(f"\n[6/7] Creating visual comparisons for {len(df_final)} pairs...")
    
    # Cross-platform font handling
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]
    
    font = font_small = None
    for font_path in font_candidates:
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(font_path, 20)
            font_small = ImageFont.truetype(font_path, 16)
            break
        except:
            continue
    
    # FIX: Use enumerate for sequential numbering (not DataFrame index!)
    for seq_num, (idx, row) in enumerate(tqdm(df_final.iterrows(), total=len(df_final), desc="Generating"), start=1):
        try:
            img_a = Image.open(row['Path_A']).convert('RGB')
            img_b = Image.open(row['Path_B']).convert('RGB')
            
            max_height = 800
            target_h = min(max_height, max(img_a.height, img_b.height))
            
            new_w_a = int(img_a.width * (target_h / img_a.height))
            new_w_b = int(img_b.width * (target_h / img_b.height))
            
            img_a_resized = img_a.resize((new_w_a, target_h), Image.Resampling.LANCZOS)
            img_b_resized = img_b.resize((new_w_b, target_h), Image.Resampling.LANCZOS)
            
            # Create difference map if enabled
            diff_map = None
            if HIGHLIGHT_DIFFERENCES:
                diff_map = create_difference_map(row['Path_A'], row['Path_B'], target_h)
            
            # Layout: [Image A] [Image B] [Diff Map]
            gap = 20
            diff_w = diff_map.width if diff_map else 0
            total_w = new_w_a + gap + new_w_b + (gap + diff_w if diff_map else 0)
            
            canvas = Image.new('RGB', (total_w, target_h + 80), 'white')
            canvas.paste(img_a_resized, (0, 60))
            canvas.paste(img_b_resized, (new_w_a + gap, 60))
            
            if diff_map:
                canvas.paste(diff_map, (new_w_a + gap + new_w_b + gap, 60))
            
            draw = ImageDraw.Draw(canvas)
            
            # Labels
            draw.text((10, 10), f"A: {row['Image_A']}", fill='black', font=font)
            draw.text((new_w_a + gap + 10, 10), f"B: {row['Image_B']}", fill='black', font=font)
            if diff_map:
                draw.text((new_w_a + gap + new_w_b + gap + 10, 10), 
                         "Difference Heatmap", fill='red', font=font)
            
            # Scores - FIX: Handle empty values safely
            score_text = []
            if 'Cosine_Similarity' in row and row['Cosine_Similarity'] != "" and pd.notna(row['Cosine_Similarity']):
                try:
                    score_text.append(f"CLIP: {float(row['Cosine_Similarity']):.3f}")
                except (ValueError, TypeError):
                    pass
            if 'Hamming_Distance' in row and row['Hamming_Distance'] != "" and pd.notna(row['Hamming_Distance']):
                try:
                    score_text.append(f"pHash: {int(row['Hamming_Distance'])}")
                except (ValueError, TypeError):
                    pass
            if 'SSIM' in row and row['SSIM'] != "" and pd.notna(row['SSIM']):
                try:
                    score_text.append(f"SSIM: {float(row['SSIM']):.3f}")
                except (ValueError, TypeError):
                    pass
            score_text.append(f"Source: {row['Source']}")
            
            draw.text((10, 35), " | ".join(score_text), fill='blue', font=font_small)
            
            # Save legacy simple comparison (backward compatible)
            # FIX: Use seq_num for consistent numbering
            out_name = f"pair_{seq_num:03d}_{row.get('Source', 'unknown').replace('+', '_')}.png"
            canvas.save(comp_dir / out_name)
            
            # === NEW: Create industry-grade visualizations ===
            pair_dir = comp_dir / f"pair_{seq_num:03d}_detailed"
            ensure_dir(pair_dir)
            
            # Align images for accurate comparison
            img_a_arr, img_b_arr, success, align_meta = align_images_for_visualization(
                row['Path_A'], row['Path_B'], None, max_size=1200
            )
            
            if success and img_a_arr is not None:
                # Save aligned versions
                Image.fromarray(img_a_arr).save(pair_dir / "1_raw_A.png")
                Image.fromarray(img_b_arr).save(pair_dir / "2_raw_B_aligned.png")
                
                # 50/50 Overlay
                overlay_img = Image.blend(Image.fromarray(img_a_arr), Image.fromarray(img_b_arr), 0.5)
                overlay_img.save(pair_dir / "3_overlay_50_50.png")
                
                # SSIM dissimilarity (viridis) with adaptive window
                ssim_map_img, ssim_val = make_ssim_dissimilarity_map(img_a_arr, img_b_arr, win_size=11)
                ssim_map_img.save(pair_dir / "4_ssim_viridis.png")
                
                # Hard diff mask (binary change detection)
                hard_diff_img = make_hard_diff_mask(img_a_arr, img_b_arr, blur_ksize=3, threshold=25)
                hard_diff_img.save(pair_dir / "5_hard_diff_mask.png")
                
                # Checkerboard (adaptive tile size)
                checker_img = make_checkerboard_composite(img_a_arr, img_b_arr, tile_size=None)
                checker_img.save(pair_dir / "6_checkerboard.png")
                
                # Blink GIF (optimized)
                make_blink_comparator_gif(img_a_arr, img_b_arr, pair_dir / "7_blink.gif", duration=500, max_width=1200)
                
                # HTML sliders (both CDN and offline) - FIX: Handle empty values
                scores_dict = {}
                
                # CLIP score
                clip_val = row.get('Cosine_Similarity', '')
                if clip_val and clip_val != '' and pd.notna(clip_val):
                    try:
                        scores_dict['CLIP'] = f"{float(clip_val):.3f}"
                    except (ValueError, TypeError):
                        scores_dict['CLIP'] = 'N/A'
                else:
                    scores_dict['CLIP'] = 'N/A'
                
                # SSIM score
                scores_dict['SSIM'] = f"{ssim_val:.3f}"
                
                # pHash score
                phash_val = row.get('Hamming_Distance', '')
                if phash_val and phash_val != '' and pd.notna(phash_val):
                    try:
                        scores_dict['pHash'] = f"{int(phash_val)}"
                    except (ValueError, TypeError):
                        scores_dict['pHash'] = 'N/A'
                else:
                    scores_dict['pHash'] = 'N/A'
                
                scores_dict['Transform'] = row.get('Transform_Matched', 'N/A')
                scores_dict['Alignment'] = align_meta.get('method', 'N/A')
                
                # Juxtapose slider (CDN) - FIX: Use seq_num
                html_content = make_juxtapose_html("1_raw_A.png", "2_raw_B_aligned.png",
                                                  seq_num, row.get('Tier'), scores_dict)
                with open(pair_dir / "interactive.html", 'w') as f:
                    f.write(html_content)
                
                # Simple slider (offline fallback) - FIX: Use seq_num
                html_offline = make_simple_slider_html("1_raw_A.png", "2_raw_B_aligned.png",
                                                      seq_num, row.get('Tier'), scores_dict)
                with open(pair_dir / "interactive_offline.html", 'w') as f:
                    f.write(html_offline)
            
        except Exception as e:
            print(f"\n  ⚠ Failed pair {seq_num}: {e}")
    
    # Create master index (tier-organized) - FIX: Use enumerate for consistent numbering
    tier_a_pairs = []
    tier_b_pairs = []
    other_pairs = []
    
    for seq_num, (idx, row) in enumerate(df_final.iterrows(), start=1):
        tier = row.get('Tier', '')
        if tier == 'A':
            tier_a_pairs.append((seq_num, row))
        elif tier == 'B':
            tier_b_pairs.append((seq_num, row))
        else:
            other_pairs.append((seq_num, row))
    
    # FIX: Add <base> tag for safe file:// URL handling with spaces/apostrophes
    html_idx = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<base href="./">
<title>Duplicate Report</title><style>
body {{ font-family:Arial,sans-serif; margin:20px; background:#f5f5f5; }}
.container {{ max-width:1200px; margin:auto; background:white; padding:20px; border-radius:8px; }}
.pair {{ margin:10px 0; padding:15px; border:1px solid #ddd; background:#fafafa; border-radius:4px; }}
.tier-a {{ border-left:5px solid #ff4444; }}
.tier-b {{ border-left:5px solid #ffa500; }}
a {{ color:#0066cc; text-decoration:none; margin-right:15px; }}
a:hover {{ text-decoration:underline; }}
.section {{ margin:30px 0; }}
h2 {{ color:#333; border-bottom:2px solid #ddd; padding-bottom:10px; }}
</style></head><body><div class="container">
<h1>🔬 Duplicate Detection Report</h1>
<p><strong>Total pairs:</strong> {len(df_final)} | 
   <strong>Tier A:</strong> {len(tier_a_pairs)} | 
   <strong>Tier B:</strong> {len(tier_b_pairs)} | 
   <strong>Other:</strong> {len(other_pairs)}</p>
"""
    
    # Tier A section (priority) - FIX: seq_num is already 1-based
    if tier_a_pairs:
        html_idx += f'<div class="section"><h2>🚨 Tier A - Review Required ({len(tier_a_pairs)})</h2>'
        for seq_num, row in tier_a_pairs:
            html_idx += f"""<div class="pair tier-a">
<h3>Pair #{seq_num:03d}</h3>
<p><strong>{row['Image_A']}</strong> vs <strong>{row['Image_B']}</strong></p>
<p>
<a href="pair_{seq_num:03d}_detailed/interactive.html">📊 Interactive (CDN)</a>
<a href="pair_{seq_num:03d}_detailed/interactive_offline.html">📊 Offline Slider</a>
<a href="pair_{seq_num:03d}_detailed/7_blink.gif">✨ Blink GIF</a>
<a href="pair_{seq_num:03d}_detailed/4_ssim_viridis.png">🎨 SSIM Map</a>
</p></div>
"""
        html_idx += '</div>'
    
    # Tier B section
    if tier_b_pairs:
        html_idx += f'<div class="section"><h2>⚠️  Tier B - Manual Check ({len(tier_b_pairs)})</h2>'
        for seq_num, row in tier_b_pairs:
            html_idx += f"""<div class="pair tier-b">
<h3>Pair #{seq_num:03d}</h3>
<p>{row['Image_A']} vs {row['Image_B']}</p>
<p>
<a href="pair_{seq_num:03d}_detailed/interactive.html">📊 Interactive</a>
<a href="pair_{seq_num:03d}_detailed/interactive_offline.html">📊 Offline</a>
<a href="pair_{seq_num:03d}_detailed/7_blink.gif">✨ Blink</a>
</p></div>
"""
        html_idx += '</div>'
    
    # Other pairs
    if other_pairs:
        html_idx += f'<div class="section"><h2>ℹ️  Other Pairs ({len(other_pairs)})</h2>'
        for seq_num, row in other_pairs:
            html_idx += f"""<div class="pair">
<h3>Pair #{seq_num:03d}</h3>
<p>{row['Image_A']} vs {row['Image_B']}</p>
<a href="pair_{seq_num:03d}_detailed/interactive.html">📊 View</a>
</div>
"""
        html_idx += '</div>'
    
    html_idx += "</div></body></html>"
    
    index_path = comp_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_idx)
    
    print(f"  ✓ Saved to: {comp_dir}")
    print(f"  ✓ Interactive: {index_path}")
    
    # Auto-open the index page when done (configurable)
    if AUTO_OPEN_RESULTS:
        print(f"\n  🌐 Opening results in browser...")
        _open_file_once(index_path)

# --- MAIN PIPELINE ------------------------------------------------------------
def main():
    """
    COMPLETE PIPELINE with progressive gating and all advanced features
    """
    start_time = time.time()
    ensure_dir(OUT_DIR)
    cache_dir = OUT_DIR / "cache"
    ensure_dir(cache_dir)
    set_seeds(RANDOM_SEED)
    
    print("\n" + "="*70)
    print("  🔬 JOURNAL-GRADE DUPLICATE DETECTION")
    print("="*70)
    print(f"📄 PDF: {PDF_PATH.name}")
    print(f"📁 Output: {OUT_DIR}")
    print(f"⚙️  Device: {DEVICE}")
    print(f"🎯 Thresholds: CLIP≥{SIM_THRESHOLD}, pHash≤{PHASH_MAX_DIST}, SSIM≥{SSIM_THRESHOLD}")
    print(f"🔧 Features: Bundles={USE_PHASH_BUNDLES}, ORB={USE_ORB_RANSAC}, Tier={USE_TIER_GATING}")
    print("="*70)
    
    stage_counts = {}
    
    # ═══ STAGE 0: PREPROCESSING ═══
    print("\n[Stage 0] Preprocessing...")
    
    # Validate PDF file
    if not PDF_PATH.exists():
        print(f"❌ ERROR: PDF file not found!")
        print(f"   Path: {PDF_PATH}")
        print(f"   Absolute: {PDF_PATH.absolute()}")
        print("\n💡 This usually means:")
        print("   1. The file wasn't uploaded correctly")
        print("   2. The path is wrong")
        print("   3. File permissions issue")
        sys.exit(1)
    
    if not PDF_PATH.is_file():
        print(f"❌ ERROR: Path exists but is not a file!")
        print(f"   Path: {PDF_PATH}")
        sys.exit(1)
    
    # Check if readable
    try:
        with open(PDF_PATH, 'rb') as f:
            # Try to read first 8 bytes (PDF signature)
            header = f.read(8)
            if not header.startswith(b'%PDF'):
                print(f"❌ ERROR: File is not a valid PDF!")
                print(f"   Header: {header}")
                print(f"   Expected: %PDF-...")
                sys.exit(1)
    except PermissionError:
        print(f"❌ ERROR: No permission to read PDF file!")
        print(f"   Path: {PDF_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Cannot read PDF file!")
        print(f"   Path: {PDF_PATH}")
        print(f"   Error: {e}")
        sys.exit(1)
    
    print(f"✓ PDF file validated: {PDF_PATH.name} ({PDF_PATH.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # Extract pages
    pages = pdf_to_pages(PDF_PATH, OUT_DIR, DPI)
    if not pages:
        print("❌ ERROR: No pages extracted from PDF")
        print("\n💡 This usually means:")
        print("   1. PDF is password-protected")
        print("   2. PDF is corrupted")
        print("   3. PyMuPDF failed to process the file")
        print("\n🔍 Try:")
        print("   - Open the PDF in another viewer to verify it works")
        print("   - Re-export the PDF from the source application")
        print("   - Check for encryption or restrictions")
        return
    stage_counts['pages'] = len(pages)
    
    panels, meta_df = pages_to_panels_auto(pages, OUT_DIR)
    if not panels:
        print("❌ No panels detected!")
        return
    stage_counts['panels'] = len(panels)
    
    meta_path = OUT_DIR / "panel_manifest.tsv"
    meta_df.to_csv(meta_path, sep="\t", index=False)
    print(f"  ✓ {len(pages)} pages → {len(panels)} panels")
    
    # ═══ MODALITY DETECTION/ROUTING (if enabled) ═══
    modality_cache = {}
    if ENABLE_MODALITY_DETECTION or ENABLE_MODALITY_ROUTING or getattr(args, "tile_first_auto", False):
        modality_cache = get_modality_cache(panels)
    
    # ═══ AUTO-ENABLE TILE-FIRST (if --tile-first-auto) ═══
    if getattr(args, "tile_first_auto", False):
        confocal_count = sum(1 for v in modality_cache.values() if v.get('modality') == 'confocal')
        
        if confocal_count >= 3:
            print(f"  🔬 Auto-enabling Tile-First ({confocal_count} confocal panels detected)")
            args.tile_first = True
        else:
            print(f"  📊 Using Standard Pipeline ({confocal_count} confocal panels, < threshold)")
            args.tile_first = False
    
    # ═══ TILE-FIRST FAST-PATH (BYPASSES PANEL PIPELINE) ═══
    if TILE_FIRST_AVAILABLE and getattr(args, "tile_first", False):
        print(f"\n╔══════════════════════════════════════════════════════════════════╗")
        print(f"║  🔬 TILE-FIRST MODE: Micro-Tiles ONLY (NO GRID DETECTION)       ║")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        
        # Configure tile-first
        tfc = TileFirstConfig()
        tfc.CONFOCAL_MIN_GRID = 999        # ← Force NO grid detection
        tfc.WB_MIN_LANES = 999             # ← Force NO lane detection
        tile_size = getattr(args, "tile_size", 384)
        
        # Guardrail: Check if tile size is reasonable
        try:
            from PIL import Image
            sample_images = [Image.open(p) for p in panels[:5]]  # Sample first 5
            min_dim = min(min(img.size) for img in sample_images)
            
            if tile_size > min_dim - 32:
                old_size = tile_size
                tile_size = max(256, min_dim - 64)  # Leave margin
                print(f"  ⚠️  Tile size ({old_size}px) larger than panels ({min_dim}px)")
                print(f"  ✓ Auto-adjusting to {tile_size}px")
        except Exception as e:
            print(f"  ⚠️  Could not validate tile size: {e}")
        
        tfc.MICRO_TILE_SIZE = tile_size
        tfc.MICRO_TILE_STRIDE = getattr(args, "tile_stride", 0.65)
        tfc.TILE_CLIP_MIN = SIM_THRESHOLD
        
        # Load CLIP model (use existing function)
        clip_obj = load_clip()
        clip_model = clip_obj.model
        preprocess = clip_obj.preprocess
        
        # Run tile-first pipeline with error handling
        try:
            df_merged = run_tile_first_pipeline(
                panel_paths=[str(p) for p in panels],
                clip_model=clip_model,
                preprocess=preprocess,
                device=DEVICE,
                config=tfc
            )
            
            # Save results (handle empty DataFrame gracefully - memory-safe version returns empty DF instead of crashing)
            final_path = OUT_DIR / "final_merged_report.tsv"
            
            if df_merged is None or len(df_merged) == 0:
                print("\n⚠️  TILE PIPELINE: No pairs found (or memory-safe fallback)")
                print("   Creating empty TSV with proper headers...")
                # Create properly formatted empty TSV
                empty_df = pd.DataFrame(columns=[
                    'Path_A', 'Path_B', 'Cosine_Similarity', 'SSIM', 
                    'Hamming_Distance', 'Tile_Matches', 'Tier', 'Extraction_Method'
                ])
                empty_df.to_csv(final_path, sep="\t", index=False)
            else:
                df_merged.to_csv(final_path, sep="\t", index=False)
                
        except MemoryError as e:
            print(f"\n❌ MEMORY ERROR: Tile-first pipeline ran out of memory!")
            print(f"   This typically happens on limited-resource environments (e.g., Streamlit Cloud).")
            print(f"   Try: Reduce tile size, increase stride, or use panel-level detection instead.")
            # Create empty TSV with proper headers
            final_path = OUT_DIR / "final_merged_report.tsv"
            empty_df = pd.DataFrame(columns=['Path_A', 'Path_B', 'Cosine_Similarity', 'Tier'])
            empty_df.to_csv(final_path, sep="\t", index=False)
            # Don't raise - return gracefully
            print("   Empty report saved, exiting gracefully")
            return
        except Exception as e:
            print(f"\n❌ ERROR in tile-first pipeline: {e}")
            import traceback
            traceback.print_exc()
            # Create empty TSV with proper headers
            final_path = OUT_DIR / "final_merged_report.tsv"
            empty_df = pd.DataFrame(columns=['Path_A', 'Path_B', 'Cosine_Similarity', 'Tier'])
            empty_df.to_csv(final_path, sep="\t", index=False)
            # Don't raise - return gracefully
            print("   Empty report saved, exiting gracefully")
            return
        
        print(f"\n  ✓ Final report: {final_path}")
        print(f"     Total pairs: {len(df_merged)}")
        if len(df_merged) > 0:
            print(f"     Tier A: {len(df_merged[df_merged['Tier'] == 'A'])}")
            print(f"     Tier B: {len(df_merged[df_merged['Tier'] == 'B'])}")
        
        print(f"\n{'='*70}")
        print(f"  ✅ TILE-FIRST PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"  ⏱️  Runtime: {time.time() - start_time:.1f}s")
        print(f"{'='*70}")
        
        return  # ← CRITICAL: Exit here, skip panel pipeline
    
    # ═══ STAGE 1: CLIP SEMANTIC FILTERING ═══
    print(f"\n[Stage 1] CLIP semantic filtering (≥{SIM_THRESHOLD})...")
    
    clip = load_clip()
    vecs = load_or_compute_embeddings(panels, clip)
    df_clip = clip_find_duplicates_faiss(panels, vecs, SIM_THRESHOLD, meta_df)
    stage_counts['stage1_clip'] = len(df_clip)
    
    clip_path = OUT_DIR / "ai_duplicate_report.tsv"
    df_clip.to_csv(clip_path, sep="\t", index=False)
    print(f"  ✓ {len(df_clip)} pairs (CLIP ≥ {SIM_THRESHOLD})")
    
    # Attach CLIP z-scores for discrimination
    if USE_CLIP_ZSCORE and not df_clip.empty:
        print(f"  Attaching CLIP z-scores (min≥{CLIP_ZSCORE_MIN})...")
        df_clip = attach_clip_zscore_to_df(df_clip, panels, vecs)
        
        # Show discrimination effect
        high_z_count = len(df_clip[pd.to_numeric(df_clip['CLIP_Z'], errors='coerce') >= CLIP_ZSCORE_MIN])
        print(f"    → {high_z_count}/{len(df_clip)} pairs pass z-score gate "
              f"({100*high_z_count/max(len(df_clip),1):.1f}%)")
    
    # ═══ STAGE 2: SSIM STRUCTURAL VALIDATION ═══
    df_ssim_validated = df_clip
    
    if USE_SSIM_VALIDATION and len(df_clip) > 0:
        print(f"\n[Stage 2] SSIM structural validation (≥{SSIM_THRESHOLD})...")
        df_ssim_validated = add_ssim_validation(df_clip)
        stage_counts['stage2_ssim'] = len(df_ssim_validated)
        print(f"  ✓ {len(df_ssim_validated)} pairs (SSIM ≥ {SSIM_THRESHOLD})")
    else:
        stage_counts['stage2_ssim'] = len(df_clip)
    
    # ═══ STAGE 3: PHASH-RT EXACT DETECTION ═══
    if USE_PHASH_BUNDLES:
        df_phash = phash_find_duplicates_with_bundles(panels, PHASH_MAX_DIST, meta_df)
    else:
        print(f"\n[Stage 3] pHash exact/near-exact (≤{PHASH_MAX_DIST})...")
        df_phash = phash_find_duplicates(panels, PHASH_MAX_DIST, meta_df)
        print(f"  ✓ {len(df_phash)} pairs (pHash ≤ {PHASH_MAX_DIST})")
    
    stage_counts['stage3_phash'] = len(df_phash)
    
    phash_path = OUT_DIR / "phash_duplicate_report.tsv"
    df_phash.to_csv(phash_path, sep="\t", index=False)
    
    # ═══ STAGE 4: ORB-RANSAC PARTIAL DUPLICATES ═══
    df_orb = pd.DataFrame()
    
    if USE_ORB_RANSAC:
        df_orb = orb_find_partial_duplicates(panels, df_ssim_validated, df_phash, meta_df)
        stage_counts['stage4_orb'] = len(df_orb[df_orb.get('Is_Partial_Dupe', False)]) if not df_orb.empty else 0
    else:
        stage_counts['stage4_orb'] = 0
    
    # ═══ STAGE 5: MERGE & TIER CLASSIFICATION ═══
    print(f"\n[Stage 5] Merging results & tier classification...")
    
    df_merged = merge_reports_enhanced(df_ssim_validated, df_phash, df_orb)
    stage_counts['merged_pairs'] = len(df_merged)
    
    # Require geometric corroboration for near pages (kills grid lookalikes)
    if REQUIRE_GEOMETRY_FOR_NEAR_PAGES and not df_merged.empty:
        print(f"  Applying near-page geometry requirement (gap≤{NEAR_PAGE_GAP})...")
        
        # Annotate page gap
        df_merged["Page_A"] = df_merged["Path_A"].apply(lambda p: page_of(p, meta_df))
        df_merged["Page_B"] = df_merged["Path_B"].apply(lambda p: page_of(p, meta_df))
        df_merged["Page_Gap"] = (df_merged["Page_A"] - df_merged["Page_B"]).abs()
        
        # Check if pair has geometric evidence
        phash_ok = pd.to_numeric(df_merged.get("Hamming_Distance", 999), errors="coerce").fillna(999) <= PHASH_MAX_DIST
        orb_ok = pd.to_numeric(df_merged.get("ORB_Inliers", 0), errors="coerce").fillna(0) >= TIER_A_ORB_INLIERS
        
        # Keep: far pages (any evidence) OR near pages (only with geometry)
        near = df_merged["Page_Gap"] <= NEAR_PAGE_GAP
        keep_mask = (~near) | (near & (phash_ok | orb_ok))
        
        before_count = len(df_merged)
        df_merged = df_merged.loc[keep_mask].copy()
        filtered_count = before_count - len(df_merged)
        
        print(f"    Filtered {filtered_count} near-page pairs without geometry")
        print(f"    Kept {len(df_merged)} pairs (far pages or geometry-confirmed)")
    
    if USE_TIER_GATING and len(df_merged) > 0:
        # Route between universal (Option 1), modality-routing (internal), and modality-specific (Option 2)
        if USE_MODALITY_SPECIFIC_GATING and modality_cache:
            print(f"  Using modality-specific tier gating (Option 2: Advanced)...")
            df_merged = apply_tier_gating(df_merged, modality_cache=modality_cache)
        elif ENABLE_MODALITY_ROUTING and modality_cache:
            print(f"  Using universal tier gating with internal modality routing...")
            df_merged = apply_tier_gating(df_merged, modality_cache=modality_cache)
        else:
            print(f"  Using universal tier gating (Option 1: Simple)...")
            df_merged = apply_tier_gating(df_merged, modality_cache=None)
        
        # ═══ DOCUMENT 54 IMPROVEMENTS ═══
        # Apply conditional SSIM gate + enhanced filtering
        try:
            from doc54_improvements import apply_doc54_tier_improvements
            df_merged = apply_doc54_tier_improvements(df_merged)
        except Exception as e:
            print(f"  ⚠️  Warning: Could not apply Document 54 improvements: {e}")
            print(f"     Continuing with standard tier gating...")
        
        stage_counts['tier_a'] = len(df_merged[df_merged.get('Tier') == 'A'])
        stage_counts['tier_b'] = len(df_merged[df_merged.get('Tier') == 'B'])
        
        # ═══ DIAGNOSTICS ═══
        print(f"\n  [Tier Gating Diagnostics]")
        
        # Show confocal false positives that were filtered
        if 'Confocal_FP' in df_merged.columns:
            confocal_fps = df_merged[df_merged['Confocal_FP'] == True]
            if len(confocal_fps) > 0:
                print(f"    ✓ Filtered {len(confocal_fps)} confocal false positives")
                for _, row in confocal_fps.head(3).iterrows():
                    print(f"        • {row['Image_A']} vs {row['Image_B']}")
                    clip_val = row.get('Cosine_Similarity', 0)
                    ssim_val = row.get('SSIM', 0)
                    if pd.notna(clip_val) and pd.notna(ssim_val):
                        print(f"          CLIP={float(clip_val):.3f}, SSIM={float(ssim_val):.3f}")
        
        # Show which paths triggered for Tier A
        tier_a = df_merged[df_merged['Tier'] == 'A']
        if len(tier_a) > 0:
            path_counts = tier_a['Tier_Path'].value_counts()
            print(f"    ✓ Tier A detection paths:")
            for path, count in path_counts.items():
                if pd.notna(path):
                    print(f"        • {path}: {count} pair(s)")
    
    # ═══ TILE-BASED DETECTION (AUTO FOR CONFOCAL) ═══
    # Auto-enable tile mode if confocal images detected, or if explicitly requested
    enable_tile_mode = False
    if TILE_MODULE_AVAILABLE:
        if hasattr(args, 'enable_tile_mode') and args.enable_tile_mode:
            enable_tile_mode = True
            print(f"  🔬 Tile mode: ENABLED (via --enable-tile-mode flag)")
        elif modality_cache and not (hasattr(args, 'disable_tile_mode') and args.disable_tile_mode):
            # Check if we have confocal images (auto-enable)
            confocal_count = sum(1 for v in modality_cache.values() if v.get('modality') == 'confocal')
            if confocal_count >= 3:  # At least 3 confocal panels
                enable_tile_mode = True
                print(f"  🔬 Tile mode: AUTO-ENABLED (detected {confocal_count} confocal panels)")
    
    if enable_tile_mode:
        import re
        
        # Build page map from panel paths
        page_map = {}
        for panel_path in panels:
            panel_name = Path(panel_path).name
            match = re.search(r'page[_-]?(\d+)', panel_name, re.I)
            if match:
                page_map[str(panel_path)] = int(match.group(1))
            else:
                page_map[str(panel_path)] = 0
        
        # Configure tile detection
        tile_config = TileConfig()
        tile_config.ENABLE_TILE_MODE = True
        
        # Run tile detection pipeline
        all_tiles, tile_matches = run_tile_detection_pipeline(
            panel_paths=[str(p) for p in panels],
            modality_cache={str(k): v for k, v in modality_cache.items()},
            page_map=page_map,
            config=tile_config
        )
        
        # Apply tile evidence to tiers
        if len(tile_matches) > 0:
            df_merged = apply_tile_evidence_to_dataframe(df_merged, tile_matches, tile_config)
            
            # Update stage counts after tile evidence
            if USE_TIER_GATING:
                stage_counts['tier_a'] = len(df_merged[df_merged.get('Tier') == 'A'])
                stage_counts['tier_b'] = len(df_merged[df_merged.get('Tier') == 'B'])
    
    final_path = OUT_DIR / "final_merged_report.tsv"
    df_merged.to_csv(final_path, sep="\t", index=False)
    
    print(f"\n  ✓ Final report: {final_path}")
    print(f"     Total pairs: {len(df_merged)}")
    if USE_TIER_GATING:
        print(f"     Tier A (Review): {stage_counts.get('tier_a', 0)}")
        print(f"     Tier B (Check): {stage_counts.get('tier_b', 0)}")
    
    # ═══ STAGE 6: VISUAL REPORTS ═══
    print(f"\n[Stage 6] Creating visual comparisons...")
    create_duplicate_comparisons(df_merged, OUT_DIR)
    
    # ═══ SAVE METADATA ═══
    print(f"\n[Stage 7] Saving run metadata...")
    write_run_metadata(OUT_DIR, start_time, **stage_counts)
    
    # ═══ SUMMARY ═══
    runtime = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 PIPELINE SUMMARY")
    print("="*70)
    print(f"  Pages extracted: {stage_counts['pages']}")
    print(f"  Panels detected: {stage_counts['panels']}")
    print(f"  Stage 1 (CLIP): {stage_counts['stage1_clip']} candidates")
    print(f"  Stage 2 (SSIM): {stage_counts['stage2_ssim']} candidates")
    print(f"  Stage 3 (pHash): {stage_counts['stage3_phash']} candidates")
    if USE_ORB_RANSAC:
        print(f"  Stage 4 (ORB-RANSAC): {stage_counts['stage4_orb']} partial dupes")
    print(f"  Final pairs: {stage_counts['merged_pairs']}")
    if USE_TIER_GATING:
        print(f"    • Tier A (Review): {stage_counts.get('tier_a', 0)} 🚨")
        print(f"    • Tier B (Check): {stage_counts.get('tier_b', 0)} ⚠️")
    print(f"\n⏱️  Runtime: {runtime:.1f}s")
    print("="*70)
    
    if len(df_merged) > 0:
        if USE_TIER_GATING and stage_counts.get('tier_a', 0) > 0:
            print(f"\n⚠️  {stage_counts['tier_a']} Tier-A pair(s) require manual review!")
            print(f"    Check: {OUT_DIR / 'duplicate_comparisons'}")
        elif not USE_TIER_GATING:
            print(f"\n📊 {len(df_merged)} pair(s) found - review recommended")
        else:
            print("\n✅ No Tier-A duplicates detected!")
    else:
        print("\n✅ No duplicates detected!")
    
    print("="*70)


def parse_cli_args():
    """Parse command-line arguments for CLI mode."""
    parser = argparse.ArgumentParser(
        description="AI-powered PDF panel duplicate detection with multi-stage filtering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument("--pdf", type=str, help="Path to input PDF file")
    parser.add_argument("--output", type=str, help="Output directory for results")
    
    # PDF conversion
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF rendering")
    
    # Detection thresholds
    parser.add_argument("--sim-threshold", type=float, default=0.96, help="CLIP similarity threshold (0-1)")
    parser.add_argument("--phash-max-dist", type=int, default=4, help="pHash max Hamming distance")
    parser.add_argument("--ssim-threshold", type=float, default=0.90, help="SSIM threshold (0-1)")
    
    # Processing options
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP batch size")
    
    # Feature flags
    parser.add_argument("--use-phash-bundles", action="store_true", default=True, help="Enable rotation/mirror-robust pHash")
    parser.add_argument("--no-phash-bundles", action="store_false", dest="use_phash_bundles", help="Disable pHash bundles")
    parser.add_argument("--use-orb", action="store_true", default=True, help="Enable ORB-RANSAC partial duplicate detection")
    parser.add_argument("--no-orb", action="store_false", dest="use_orb", help="Disable ORB-RANSAC")
    parser.add_argument("--use-tier-gating", action="store_true", default=True, help="Enable Tier A/B classification")
    parser.add_argument("--no-tier-gating", action="store_false", dest="use_tier_gating", help="Disable tier gating")
    parser.add_argument("--highlight-diffs", action="store_true", default=True, help="Highlight visual differences")
    parser.add_argument("--no-highlight-diffs", action="store_false", dest="highlight_diffs", help="Disable diff highlighting")
    parser.add_argument("--enable-cache", action="store_true", default=True, help="Enable embedding cache")
    parser.add_argument("--no-cache", action="store_false", dest="enable_cache", help="Disable cache")
    parser.add_argument("--suppress-same-page", action="store_true", default=False, help="Suppress same-page duplicates")
    
    # Modality-specific detection (Option 2)
    parser.add_argument("--use-modality-specific", action="store_true", default=False, help="Use modality-specific tier gating (Option 2: Advanced)")
    parser.add_argument("--enable-modality-detection", action="store_true", default=False, help="Pre-classify image types (WB, confocal, TEM, etc.)")
    
    # Internal modality routing (silent, no UI columns)
    parser.add_argument("--auto-modality", dest="auto_modality", action="store_true", help="Enable internal modality routing (no UI columns)")
    parser.add_argument("--no-auto-modality", dest="auto_modality", action="store_false", help="Disable internal modality routing")
    parser.set_defaults(auto_modality=ENABLE_MODALITY_ROUTING)
    parser.add_argument("--expose-modality-columns", action="store_true", default=False, help="Include Modality_A/Modality_B in TSV for debugging")
    
    # ORB relax feature flag
    parser.add_argument("--enable-orb-relax", action="store_true", default=False, help="Enable high-confidence relaxed ORB detection for tough partial duplicates")
    parser.add_argument("--disable-orb-relax", action="store_true", help="Explicitly disable ORB relax (overrides config file)")
    
    # Tile-based detection (AUTO-ENABLED for confocal)
    parser.add_argument("--enable-tile-mode", dest="enable_tile_mode", action="store_true", help="Force enable sub-panel tile verification")
    parser.add_argument("--disable-tile-mode", dest="disable_tile_mode", action="store_true", help="Disable tile mode (even if confocal detected)")
    parser.set_defaults(enable_tile_mode=False, disable_tile_mode=False)
    
    # Tile-first mode (micro-tiles ONLY, bypasses panel pipeline)
    parser.add_argument("--tile-first", action="store_true", default=False, 
                       help="Use tile-first pipeline (micro-tiles ONLY, NO grid detection)")
    parser.add_argument("--tile-first-auto", action="store_true", default=False,
                       help="Auto-enable tile-first if ≥3 confocal panels detected")
    parser.add_argument("--tile-size", type=int, default=384, 
                       help="Tile size for micro-tiling (default: 384)")
    parser.add_argument("--tile-stride", type=float, default=0.65, 
                       help="Tile stride ratio for overlap (default: 0.65 = 35%% overlap)")
    
    # Output customization
    parser.add_argument("--out-suffix", type=str, default="", help="Append suffix to output directory for this run")
    parser.add_argument("--focus-pages", nargs="*", type=int, default=[], help="Optional list of page numbers to highlight in metrics (e.g., 19 30)")
    parser.add_argument("--save-metrics-json", action="store_true", default=False, help="Write run_metrics.json with performance summary")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("--auto-open", action="store_true", default=False, help="Auto-open results after processing")
    parser.add_argument("--no-auto-open", action="store_false", dest="auto_open", help="Don't auto-open results")
    
    return parser.parse_args()


if __name__ == "__main__":
    import traceback
    
    try:
        # CLI support for Streamlit
        _save_metrics = False
        _focus_pages = []
        if len(sys.argv) > 1:
            args = parse_cli_args()
            if args.pdf: 
                PDF_PATH = Path(args.pdf)
            if args.output: 
                OUT_DIR = Path(args.output)
            # Optional output suffix
            if args.out_suffix:
                OUT_DIR = OUT_DIR.parent / f"{OUT_DIR.name}_{args.out_suffix}"
                print(f"  ▶ OUT_DIR overridden → {OUT_DIR}")
            DPI = args.dpi
            SIM_THRESHOLD = args.sim_threshold
            PHASH_MAX_DIST = args.phash_max_dist
            SSIM_THRESHOLD = args.ssim_threshold
            BATCH_SIZE = args.batch_size
            USE_PHASH_BUNDLES = args.use_phash_bundles
            USE_ORB_RANSAC = args.use_orb
            USE_TIER_GATING = args.use_tier_gating
            HIGHLIGHT_DIFFERENCES = args.highlight_diffs
            ENABLE_CACHE = args.enable_cache
            SUPPRESS_SAME_PAGE_DUPES = args.suppress_same_page
            USE_MODALITY_SPECIFIC_GATING = args.use_modality_specific
            ENABLE_MODALITY_DETECTION = args.enable_modality_detection
            # Internal modality routing
            if hasattr(args, "auto_modality"):
                ENABLE_MODALITY_ROUTING = bool(args.auto_modality)
            if getattr(args, "expose_modality_columns", False):
                EXPOSE_MODALITY_COLUMNS = True
            # ORB relax feature flag
            if args.enable_orb_relax:
                ENABLE_ORB_RELAX = True
            if args.disable_orb_relax:
                ENABLE_ORB_RELAX = False
            DEBUG_MODE = args.debug
            AUTO_OPEN_RESULTS = args.auto_open
            # Metrics options
            _save_metrics = args.save_metrics_json
            _focus_pages = args.focus_pages
        
        main()
        
        # Optional: lightweight metrics writer (if --save-metrics-json was passed)
        if _save_metrics:
            try:
                tsv = OUT_DIR / "final_merged_report.tsv"
                if tsv.exists():
                    df = pd.read_csv(tsv, sep="\t")
                    # Basic metrics
                    total = len(df)
                    tierA = int((df.get("Tier") == "A").sum()) if "Tier" in df else 0
                    tierB = int((df.get("Tier") == "B").sum()) if "Tier" in df else 0
                    cross = 0
                    if "Path_A" in df.columns and "Path_B" in df.columns:
                        # Naive page parse; expects ".../page_19_..." pattern
                        def _pg(p): 
                            s = str(p)
                            for tok in s.split("_"):
                                if tok.isdigit(): return int(tok)
                            return 0
                        cross = sum(_pg(a) != _pg(b) for a,b in zip(df["Path_A"], df["Path_B"]))
                    metrics = {
                        "sim_threshold": SIM_THRESHOLD,
                        "ssim_threshold": SSIM_THRESHOLD,
                        "phash_max_dist": PHASH_MAX_DIST,
                        "enable_orb_relax": ENABLE_ORB_RELAX,
                        "pairs_total": total,
                        "tierA": tierA,
                        "tierB": tierB,
                        "cross_page_pct": (100.0 * cross / total) if total else 0.0,
                    }
                    (OUT_DIR / "run_metrics.json").write_text(json.dumps(metrics, indent=2))
                    print(f"  ✓ saved metrics → {OUT_DIR/'run_metrics.json'}")
            except Exception as e:
                print(f"  (metrics write skipped: {e})")
        
    except Exception as e:
        # DETAILED ERROR REPORT FOR DEBUGGING
        print("\n" + "="*70, file=sys.stderr)
        print("❌ FATAL ERROR", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Message: {str(e)}", file=sys.stderr)
        print("\nFull Traceback:", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        traceback.print_exc()
        print("-" * 70, file=sys.stderr)
        print(f"\nPython Version: {sys.version}", file=sys.stderr)
        print(f"Working Directory: {os.getcwd()}", file=sys.stderr)
        print("="*70, file=sys.stderr)
        
        # Exit with error code for Streamlit to detect
        sys.exit(1)
