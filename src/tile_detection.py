#!/usr/bin/env python3
"""
Tile-Based Duplicate Detection Module
Plugs into existing ai_pdf_panel_duplicate_check_AUTO.py pipeline
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import imagehash
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class TileConfig:
    """Centralized tile detection configuration"""
    
    # Feature flags
    ENABLE_TILE_MODE = False  # Toggle in main script
    ENABLE_TILE_DEBUG = False
    ALLOW_SAME_PAGE_TILES = False  # Allow tile verification for same-page pairs (for figure grids)
    
    # Tile extraction (✅ OPTIMIZED FOR SMALL CONFOCAL PANELS)
    TILE_MIN_GRID_CELLS = 2          # Lowered from 4
    TILE_MAX_GRID_CELLS = 30         # Increased from 20
    TILE_PROJECTION_VALLEY_DEPTH = 10  # Relaxed from 18
    TILE_MIN_SEPARATOR_WIDTH = 5
    TILE_SIZE = 256                  # ✅ Reduced from 384 to fit 271px panels
    TILE_STRIDE_RATIO = 0.70         # More overlap (was 0.65)
    TILE_MIN_AREA = 9000
    TILE_MAX_TILES_PER_PANEL = 20
    
    # NEW: Force micro-tiles for confocal (bypass grid detection)
    FORCE_MICRO_TILES_FOR_CONFOCAL = True
    
    # Candidate generation
    TILE_CLIP_MIN = 0.96
    TILE_TOPK = 50
    
    # Verification thresholds (✅ RELAXED FOR CONFOCAL)
    TILE_CONFOCAL_SSIM_MIN = 0.88     # Was 0.92
    TILE_CONFOCAL_NCC_MIN = 0.985     # Was 0.990
    TILE_CONFOCAL_PHASH_MAX = 6       # Was 5 (allow slight rotation tolerance)
    
    TILE_IHC_SSIM_MIN = 0.90
    TILE_IHC_NCC_MIN = 0.985
    TILE_IHC_PHASH_MAX = 6
    
    TILE_WB_PHASH_MAX = 5
    
    TILE_ECC_MAX_STEPS = 120
    
    # Integration policy (✅ REQUIRE MULTIPLE TILES FOR HIGH CONFIDENCE)
    REQUIRE_TILE_EVIDENCE_FOR_CONFOCAL = True
    MIN_VERIFIED_TILES_FOR_TIER_A = 2  # Increased from 1
    DEMOTE_GRID_WITHOUT_TILE = True

# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Tile:
    """Represents a single tile extracted from a panel"""
    tile_id: str
    panel_path: str
    page: int
    row: int  # -1 for micro-tiles
    col: int  # -1 for micro-tiles
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    modality: str
    
    def get_image(self) -> np.ndarray:
        """Lazy load tile image"""
        panel_img = cv2.imread(self.panel_path)
        if panel_img is None:
            return None
        x, y, w, h = self.bbox
        return panel_img[y:y+h, x:x+w]

@dataclass
class TileMatch:
    """Represents a verified tile-level match"""
    tile_a: Tile
    tile_b: Tile
    method: str
    confidence: float
    metrics: Dict

# ═══════════════════════════════════════════════════════════════
# TILE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _find_valleys_1d(projection: np.ndarray, min_depth: int, min_width: int) -> List[int]:
    """Find valley centers in a 1D projection profile"""
    try:
        proj = projection.astype(np.float32).reshape(-1, 1)
        proj = cv2.GaussianBlur(proj, (5, 1), 0).flatten()
    except Exception:
        proj = projection
    baseline = float(np.median(proj))
    valleys = []
    in_valley = False
    start = 0
    for i in range(1, len(proj) - 1):
        if proj[i] < (baseline - min_depth):
            if not in_valley:
                start = i
                in_valley = True
        elif in_valley:
            width = i - start
            if width >= min_width:
                valleys.append((start + i) // 2)
            in_valley = False
    return valleys

def _detect_grid_cells(img_bgr: np.ndarray, config: TileConfig) -> Optional[List[Tuple[int,int,int,int]]]:
    """Try to detect an r×c grid by projection profiles. Returns list of (x,y,w,h) or None."""
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
    h, w = gray.shape[:2]
    h_proj = np.mean(gray, axis=1)
    v_proj = np.mean(gray, axis=0)
    rows = _find_valleys_1d(h_proj, config.TILE_PROJECTION_VALLEY_DEPTH, config.TILE_MIN_SEPARATOR_WIDTH)
    cols = _find_valleys_1d(v_proj, config.TILE_PROJECTION_VALLEY_DEPTH, config.TILE_MIN_SEPARATOR_WIDTH)
    num_rows = len(rows) + 1
    num_cols = len(cols) + 1
    total = num_rows * num_cols
    if total < config.TILE_MIN_GRID_CELLS or total > config.TILE_MAX_GRID_CELLS:
        return None
    row_bounds = [0] + rows + [h]
    col_bounds = [0] + cols + [w]
    cells = []
    for ri in range(len(row_bounds) - 1):
        y1, y2 = row_bounds[ri], row_bounds[ri+1]
        for ci in range(len(col_bounds) - 1):
            x1, x2 = col_bounds[ci], col_bounds[ci+1]
            ww, hh = (x2 - x1), (y2 - y1)
            if ww * hh < config.TILE_MIN_AREA:
                continue
            cells.append((x1, y1, ww, hh))
    return cells if len(cells) >= config.TILE_MIN_GRID_CELLS else None

def _micro_tiles(img_bgr: np.ndarray, config: TileConfig) -> List[Tuple[int,int,int,int]]:
    """Create overlapping square tiles with optional edge coverage"""
    h, w = img_bgr.shape[:2]
    size = config.TILE_SIZE
    
    # ✅ Adapt tile size if image is too small
    if h < size or w < size:
        size = min(h, w, 256)  # Use smaller tile (min 256px)
        if size < 128:  # Too small to be useful
            return []
        print(f"  [Adapt] Tile size reduced to {size}px to fit {h}×{w} panel")
    
    stride = max(8, int(size * config.TILE_STRIDE_RATIO))
    tiles = []
    for y in range(0, max(1, h - size + 1), stride):
        for x in range(0, max(1, w - size + 1), stride):
            tiles.append((x, y, size, size))
            if len(tiles) >= config.TILE_MAX_TILES_PER_PANEL:
                return tiles
    # Ensure bottom/right edges covered
    if h >= size:
        for x in range(0, max(1, w - size + 1), stride):
            tiles.append((x, h - size, size, size))
            if len(tiles) >= config.TILE_MAX_TILES_PER_PANEL:
                return tiles
    if w >= size:
        for y in range(0, max(1, h - size + 1), stride):
            tiles.append((w - size, y, size, size))
            if len(tiles) >= config.TILE_MAX_TILES_PER_PANEL:
                return tiles
    return tiles[:config.TILE_MAX_TILES_PER_PANEL]

def extract_tiles_from_panel(panel_path: str, page: int, modality: str, config: TileConfig) -> List[Tile]:
    """Extract tiles from a single panel"""
    img = cv2.imread(panel_path)
    if img is None:
        return []
    
    panel_id = Path(panel_path).stem
    
    # ✅ NEW: Force micro-tiles for confocal (skip grid detection)
    if modality == 'confocal' and config.FORCE_MICRO_TILES_FOR_CONFOCAL:
        if config.ENABLE_TILE_DEBUG:
            print(f"  [Confocal] Forcing micro-tiles (bypassing grid detection)")
        micro = _micro_tiles(img, config)
        tiles = []
        for idx, (x, y, w, h) in enumerate(micro):
            tile = Tile(
                tile_id=f"{panel_id}_micro_{idx}",
                panel_path=panel_path,
                page=page,
                row=-1,  # -1 indicates micro-tile
                col=-1,
                bbox=(x, y, w, h),
                modality=modality
            )
            tiles.append(tile)
        return tiles
    
    # Original grid detection logic for non-confocal...
    grid_cells = _detect_grid_cells(img, config)
    tiles = []
    
    if grid_cells is not None:
        # Grid mode
        for idx, (x, y, w, h) in enumerate(grid_cells):
            row = idx // int(np.sqrt(len(grid_cells)) + 0.5)
            col = idx % int(np.sqrt(len(grid_cells)) + 0.5)
            tile = Tile(
                tile_id=f"{panel_id}_r{row}_c{col}",
                panel_path=panel_path,
                page=page,
                row=row,
                col=col,
                bbox=(x, y, w, h),
                modality=modality
            )
            tiles.append(tile)
    else:
        # Micro-tiling fallback
        micro = _micro_tiles(img, config)
        for idx, (x, y, w, h) in enumerate(micro):
            tile = Tile(
                tile_id=f"{panel_id}_t{idx}",
                panel_path=panel_path,
                page=page,
                row=-1,
                col=-1,
                bbox=(x, y, w, h),
                modality=modality
            )
            tiles.append(tile)
    
    return tiles

# ═══════════════════════════════════════════════════════════════
# TILE VERIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════

def _zscore(img: np.ndarray) -> np.ndarray:
    """Z-score normalization"""
    img = img.astype(np.float32)
    mu, sd = float(np.mean(img)), float(np.std(img))
    if sd < 1e-6:
        sd = 1.0
    return (img - mu) / sd

def _ncc_same_size(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation - resizes to common size if needed"""
    # Ensure same size
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h))
        b = cv2.resize(b, (w, h))
    
    a_z = _zscore(a)
    b_z = _zscore(b)
    return float(np.mean(a_z * b_z))

def _ecc_align_gray(a: np.ndarray, b: np.ndarray, max_iter: int = 120) -> Tuple[np.ndarray, dict]:
    """ECC affine alignment"""
    a_gray = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h = min(a_gray.shape[0], b_gray.shape[0])
    
    def _resize_to_h(x, H):
        s = H / x.shape[0]
        return cv2.resize(x, (max(1, int(round(x.shape[1]*s))), H), interpolation=cv2.INTER_AREA)
    
    a_r = _resize_to_h(a_gray, h)
    b_r = _resize_to_h(b_gray, h)
    
    W = max(a_r.shape[1], b_r.shape[1])
    if a_r.shape[1] < W:
        a_r = cv2.copyMakeBorder(a_r, 0, 0, 0, W - a_r.shape[1], cv2.BORDER_CONSTANT, value=128)
    if b_r.shape[1] < W:
        b_r = cv2.copyMakeBorder(b_r, 0, 0, 0, W - b_r.shape[1], cv2.BORDER_CONSTANT, value=128)
    
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(_zscore(b_r), _zscore(a_r), warp, cv2.MOTION_AFFINE, criteria)
        a_aligned = cv2.warpAffine(a_r, warp, (b_r.shape[1], b_r.shape[0]),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=128)
        return a_aligned, {'cc': float(cc)}
    except Exception:
        return a_r, {'cc': 0.0}

def _compute_phash_bundle_min(img_a: np.ndarray, img_b: np.ndarray) -> int:
    """pHash distance across 8 rotation/mirror transforms"""
    try:
        pil_a = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
        pil_b = Image.fromarray(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
        
        hash_a = imagehash.phash(pil_a)
        
        transforms = [
            pil_b,
            pil_b.rotate(90, expand=True),
            pil_b.rotate(180, expand=True),
            pil_b.rotate(270, expand=True),
            pil_b.transpose(Image.FLIP_LEFT_RIGHT),
            pil_b.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True),
            pil_b.transpose(Image.FLIP_LEFT_RIGHT).rotate(180, expand=True),
            pil_b.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True),
        ]
        
        min_dist = 999
        for t_img in transforms:
            hash_t = imagehash.phash(t_img)
            dist = hash_a - hash_t
            if dist < min_dist:
                min_dist = dist
                if dist <= 3:  # Short-circuit
                    break
        
        return int(min_dist)
    except Exception:
        return 999

# ═══════════════════════════════════════════════════════════════
# TILE-LEVEL VERIFICATION
# ═══════════════════════════════════════════════════════════════

def verify_tile_pair_confocal(tile_a: Tile, tile_b: Tile, config: TileConfig) -> Optional[TileMatch]:
    """Confocal-specific tile verification"""
    img_a = tile_a.get_image()
    img_b = tile_b.get_image()
    
    if img_a is None or img_b is None:
        return None
    
    # Fast path: pHash
    phash_dist = _compute_phash_bundle_min(img_a, img_b)
    if phash_dist <= config.TILE_CONFOCAL_PHASH_MAX:
        return TileMatch(tile_a, tile_b, "Tile-Exact", 1.0, {'phash': phash_dist})
    
    # ECC + SSIM + NCC
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
    
    aligned_b, ecc_info = _ecc_align_gray(gray_a, gray_b, max_iter=config.TILE_ECC_MAX_STEPS)
    
    try:
        ssim_val, _ = ssim_func(gray_a, aligned_b, full=True)
    except Exception:
        ssim_val = 0.0
    
    ncc_val = _ncc_same_size(gray_a, aligned_b)
    
    metrics = {
        'ssim': float(ssim_val),
        'ncc': float(ncc_val),
        'phash': phash_dist,
        'ecc_converged': ecc_info['cc'] > 0.5
    }
    
    if ssim_val >= config.TILE_CONFOCAL_SSIM_MIN and ncc_val >= config.TILE_CONFOCAL_NCC_MIN:
        confidence = (ssim_val + min((ncc_val - 0.99) * 10, 1.0)) / 2
        return TileMatch(tile_a, tile_b, "Tile-DeepVerify-Confocal", confidence, metrics)
    
    return None

def verify_tile_pair(tile_a: Tile, tile_b: Tile, config: TileConfig) -> Optional[TileMatch]:
    """Route to modality-specific verification"""
    modality = tile_a.modality if tile_a.modality == tile_b.modality else 'unknown'
    
    if modality == 'confocal':
        return verify_tile_pair_confocal(tile_a, tile_b, config)
    else:
        # Generic fallback (pHash only for now)
        img_a = tile_a.get_image()
        img_b = tile_b.get_image()
        if img_a is None or img_b is None:
            return None
        phash_dist = _compute_phash_bundle_min(img_a, img_b)
        if phash_dist <= 5:
            return TileMatch(tile_a, tile_b, "Tile-Exact", 1.0, {'phash': phash_dist})
    
    return None

# ═══════════════════════════════════════════════════════════════
# MAIN TILE PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_tile_detection_pipeline(panel_paths: List[str], 
                                modality_cache: dict,
                                page_map: dict,
                                config: TileConfig = None) -> Tuple[List[Tile], List[TileMatch]]:
    """
    Main tile detection pipeline
    
    Args:
        panel_paths: List of panel image paths
        modality_cache: Dict mapping panel_path -> {'modality': str, 'confidence': float}
        page_map: Dict mapping panel_path -> page number
        config: TileConfig instance (uses defaults if None)
    
    Returns:
        (all_tiles, verified_matches)
    """
    if config is None:
        config = TileConfig()
    
    if not config.ENABLE_TILE_MODE:
        print("  Tile mode disabled, skipping...")
        return [], []
    
    print("\n" + "="*70)
    print("🔬 TILE-BASED DETECTION")
    print("="*70)
    
    # Phase 1: Extract tiles
    all_tiles = []
    for panel_path in tqdm(panel_paths, desc="Extracting tiles"):
        page = page_map.get(panel_path, 0)
        mod_info = modality_cache.get(panel_path, {'modality': 'unknown', 'confidence': 0.0})
        modality = mod_info['modality']
        
        tiles = extract_tiles_from_panel(panel_path, page, modality, config)
        all_tiles.extend(tiles)
    
    grid_tiles = sum(1 for t in all_tiles if t.row >= 0)
    micro_tiles = sum(1 for t in all_tiles if t.row == -1)
    
    print(f"  ✓ Extracted {len(all_tiles)} tiles")
    print(f"    • Grid-based: {grid_tiles}")
    print(f"    • Micro-tiles: {micro_tiles}")
    
    if len(all_tiles) == 0:
        return [], []
    
    # Phase 2: Group tiles by panel pair (from existing panel-level candidates)
    # For now, we'll do a simple same-modality check
    print("\n[Tile Verification]")
    verified_matches = []
    
    # Build panel->tiles index
    panel_tiles = {}
    for tile in all_tiles:
        if tile.panel_path not in panel_tiles:
            panel_tiles[tile.panel_path] = []
        panel_tiles[tile.panel_path].append(tile)
    
    # For each panel pair, verify tiles
    panel_pairs = []
    for i, pa in enumerate(panel_paths):
        for pb in panel_paths[i+1:]:
            mod_a = modality_cache.get(pa, {}).get('modality', 'unknown')
            mod_b = modality_cache.get(pb, {}).get('modality', 'unknown')
            page_a = page_map.get(pa, 0)
            page_b = page_map.get(pb, 0)
            
            # Skip same page unless explicitly allowed (for figure grids)
            if page_a == page_b and not config.ALLOW_SAME_PAGE_TILES:
                continue
            
            # Only check same modality
            if mod_a != mod_b:
                continue
            
            # Only check confocal for now (can expand later)
            if mod_a != 'confocal':
                continue
            
            panel_pairs.append((pa, pb))
    
    if config.ALLOW_SAME_PAGE_TILES:
        print(f"  Checking {len(panel_pairs)} panel pairs for tile matches (including same-page)...")
    else:
        print(f"  Checking {len(panel_pairs)} panel pairs for tile matches (cross-page only)...")
    
    # ✅ Check ALL pairs (removed [:100] limit)
    for pa, pb in tqdm(panel_pairs, desc="Tile verification"):
        tiles_a = panel_tiles.get(pa, [])
        tiles_b = panel_tiles.get(pb, [])
        
        # ✅ Count ALL matching tiles (no short-circuit)
        pair_matches = []
        for tile_a in tiles_a:
            for tile_b in tiles_b:
                match = verify_tile_pair(tile_a, tile_b, config)
                if match is not None:
                    pair_matches.append(match)
        
        # Add all matches for this pair
        verified_matches.extend(pair_matches)
        
        # Optional: Log multi-tile evidence
        if len(pair_matches) > 0 and config.ENABLE_TILE_DEBUG:
            print(f"    {Path(pa).name} ↔ {Path(pb).name}: {len(pair_matches)} tile matches")
    
    print(f"  ✓ Found {len(verified_matches)} tile matches")
    
    return all_tiles, verified_matches

# ═══════════════════════════════════════════════════════════════
# INTEGRATION WITH PANEL-LEVEL RESULTS
# ═══════════════════════════════════════════════════════════════

def apply_tile_evidence_to_dataframe(df, tile_matches: List[TileMatch], config: TileConfig):
    """Apply tile evidence to panel-level tier assignments"""
    import pandas as pd
    
    if len(tile_matches) == 0:
        return df
    
    print("\n[Applying Tile Evidence]")
    
    # Aggregate by panel pair
    panel_matches = {}
    for match in tile_matches:
        pair = tuple(sorted([match.tile_a.panel_path, match.tile_b.panel_path]))
        if pair not in panel_matches:
            panel_matches[pair] = []
        panel_matches[pair].append(match)
    
    # Add tile columns
    df['Tile_Evidence'] = False
    df['Tile_Evidence_Count'] = 0
    df['Tile_Best_Path'] = None
    
    for idx, row in df.iterrows():
        pair = tuple(sorted([row['Path_A'], row['Path_B']]))
        
        if pair in panel_matches:
            matches = panel_matches[pair]
            df.at[idx, 'Tile_Evidence'] = True
            df.at[idx, 'Tile_Evidence_Count'] = len(matches)
            
            best = max(matches, key=lambda m: m.confidence)
            df.at[idx, 'Tile_Best_Path'] = best.method
    
    # Apply promotion/demotion (✅ REQUIRE MULTIPLE TILES FOR HIGH CONFIDENCE)
    original_tier_a = len(df[df['Tier'] == 'A'])
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('Tier')):
            continue
        
        mod_a = row.get('Modality_A', 'unknown')
        mod_b = row.get('Modality_B', 'unknown')
        tile_count = row.get('Tile_Evidence_Count', 0)
        
        # Require tile evidence for confocal
        if config.REQUIRE_TILE_EVIDENCE_FOR_CONFOCAL and mod_a == 'confocal' and mod_b == 'confocal':
            # ✅ Promote to Tier-A if ≥MIN_VERIFIED_TILES_FOR_TIER_A tiles match
            if tile_count >= config.MIN_VERIFIED_TILES_FOR_TIER_A and row.get('Tier') != 'A':
                df.at[idx, 'Tier'] = 'A'
                df.at[idx, 'Tier_Path'] = f'Multi-Tile-Confirmed-{tile_count}'
            
            # ✅ Demote from Tier-A if insufficient tile evidence
            if row['Tier'] == 'A' and tile_count < config.MIN_VERIFIED_TILES_FOR_TIER_A:
                # Check if it's already protected by exact/ORB/DeepVerify
                tier_path = row.get('Tier_Path', '')
                if tier_path not in ['Exact', 'ORB', 'Confocal-DeepVerify', 'IHC-DeepVerify']:
                    df.at[idx, 'Tier'] = 'B'
                    df.at[idx, 'Tier_Path'] = f'Confocal-NeedsTileEvidence-{tile_count}'
    
    new_tier_a = len(df[df['Tier'] == 'A'])
    promoted = len(df[df['Tier_Path'].astype(str).str.contains('Multi-Tile-Confirmed', na=False)])
    print(f"  ✓ Tier A: {original_tier_a} → {new_tier_a} ({new_tier_a - original_tier_a:+d})")
    if promoted > 0:
        print(f"    ↑ {promoted} pairs promoted via multi-tile evidence")
    
    return df

