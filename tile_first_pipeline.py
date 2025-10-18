#!/usr/bin/env python3
"""
Tile-First Pipeline: Micro-Tiles ONLY (NO GRID DETECTION)

This module provides pure micro-tile extraction with NO grid or lane detection.
All images are uniformly tiled into 384×384 overlapping tiles.

Configuration:
- CONFOCAL_MIN_GRID = 999  (impossible → never detects grids)
- WB_MIN_LANES = 999       (impossible → never detects lanes)
- MICRO_TILE_SIZE = 384    (uniform tile size for all images)
"""

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

class TileFirstConfig:
    """Configuration for pure micro-tile extraction (NO GRID)"""
    
    # FORCE MICRO-TILES ONLY (NO GRID DETECTION)
    CONFOCAL_MIN_GRID = 999        # Impossible → never detects grids
    WB_MIN_LANES = 999             # Impossible → never detects lanes
    
    # Micro-tile parameters
    MICRO_TILE_SIZE = 384          # 384×384 tiles (uniform for all images)
    MICRO_TILE_STRIDE = 0.65       # 35% overlap between tiles
    
    # Candidate generation
    TILE_CLIP_MIN = 0.96           # CLIP similarity threshold
    TILE_TOPK = 50                 # Max candidates per tile
    
    # Tile verification
    TILE_SSIM_MIN = 0.95           # SSIM threshold for tile match
    TILE_PHASH_MAX = 3             # pHash distance threshold
    
    # Tier gating
    TIER_A_MIN_TILES = 2           # Min matching tiles for Tier A
    TIER_A_SSIM = 0.95             # SSIM for Tier A tile
    TIER_A_PHASH = 3               # pHash for Tier A tile


def extract_micro_tiles(image_path: str, config: TileFirstConfig) -> List[Dict]:
    """
    Extract uniform micro-tiles from an image (NO GRID DETECTION).
    
    Returns list of dicts with:
    - tile_id: Unique identifier (e.g., "page_1_panel01_t5")
    - image_path: Path to source image
    - bbox: (x, y, w, h) tile location
    - tile_data: np.ndarray of tile pixels
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    h, w = img.shape[:2]
    tile_size = config.MICRO_TILE_SIZE
    stride = int(tile_size * config.MICRO_TILE_STRIDE)
    
    tiles = []
    panel_id = Path(image_path).stem
    tile_idx = 0
    
    # Create overlapping tiles
    for y in range(0, max(1, h - tile_size + 1), stride):
        for x in range(0, max(1, w - tile_size + 1), stride):
            tile_id = f"{panel_id}_t{tile_idx}"
            bbox = (x, y, tile_size, tile_size)
            tile_data = img[y:y+tile_size, x:x+tile_size].copy()
            
            tiles.append({
                'tile_id': tile_id,
                'image_path': image_path,
                'bbox': bbox,
                'tile_data': tile_data
            })
            tile_idx += 1
    
    # Add edge tiles if needed
    if h > tile_size and (h - tile_size) % stride != 0:
        for x in range(0, max(1, w - tile_size + 1), stride):
            tile_id = f"{panel_id}_t{tile_idx}"
            bbox = (x, h - tile_size, tile_size, tile_size)
            tile_data = img[h-tile_size:h, x:x+tile_size].copy()
            tiles.append({'tile_id': tile_id, 'image_path': image_path, 'bbox': bbox, 'tile_data': tile_data})
            tile_idx += 1
    
    if w > tile_size and (w - tile_size) % stride != 0:
        for y in range(0, max(1, h - tile_size + 1), stride):
            tile_id = f"{panel_id}_t{tile_idx}"
            bbox = (w - tile_size, y, tile_size, tile_size)
            tile_data = img[y:y+tile_size, w-tile_size:w].copy()
            tiles.append({'tile_id': tile_id, 'image_path': image_path, 'bbox': bbox, 'tile_data': tile_data})
            tile_idx += 1
    
    return tiles


def compute_tile_embeddings(tiles: List[Dict], clip_model, preprocess, device) -> np.ndarray:
    """Compute CLIP embeddings for all tiles"""
    import torch
    from PIL import Image
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(tiles), batch_size), desc="Computing tile CLIP"):
        batch_tiles = tiles[i:i+batch_size]
        batch_images = []
        
        for tile in batch_tiles:
            tile_bgr = tile['tile_data']
            tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(tile_rgb)
            batch_images.append(preprocess(pil_img))
        
        with torch.no_grad():
            batch_tensor = torch.stack(batch_images).to(device)
            batch_features = clip_model.encode_image(batch_tensor)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            embeddings.append(batch_features.cpu().numpy())
    
    return np.vstack(embeddings)


def find_tile_candidates(tiles: List[Dict], embeddings: np.ndarray, config: TileFirstConfig) -> List[Tuple]:
    """
    Find candidate tile pairs using CLIP similarity.
    
    Returns list of (tile_idx_a, tile_idx_b, clip_similarity) tuples.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n[Tile Candidate Generation]")
    sim_matrix = cosine_similarity(embeddings)
    candidates = []
    
    # Build panel index
    panel_for_tile = {}
    for idx, tile in enumerate(tiles):
        panel_for_tile[idx] = tile['image_path']
    
    for i in range(len(tiles)):
        similarities = sim_matrix[i]
        top_indices = np.argsort(similarities)[::-1]
        
        count = 0
        for j in top_indices:
            if i == j:
                continue
            if similarities[j] < config.TILE_CLIP_MIN:
                break
            # Skip same-panel comparisons
            if panel_for_tile[i] == panel_for_tile[j]:
                continue
            
            candidates.append((i, j, similarities[j]))
            count += 1
            if count >= config.TILE_TOPK:
                break
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for i, j, sim in candidates:
        pair = tuple(sorted([i, j]))
        if pair not in seen:
            seen.add(pair)
            unique_candidates.append((i, j, sim))
    
    print(f"  ✓ Generated {len(unique_candidates)} tile candidate pairs (CLIP ≥ {config.TILE_CLIP_MIN})")
    return unique_candidates


def verify_tile_pair(tile_a: Dict, tile_b: Dict, config: TileFirstConfig) -> Dict:
    """
    Verify if two tiles match using SSIM + pHash.
    
    Returns dict with:
    - match: bool (True if verified match)
    - ssim: float
    - phash_dist: int
    - method: str ("Tile-Exact" or "Tile-SSIM")
    """
    from skimage.metrics import structural_similarity as ssim_func
    from PIL import Image
    import imagehash
    
    img_a = tile_a['tile_data']
    img_b = tile_b['tile_data']
    
    # Convert to PIL for pHash
    pil_a = Image.fromarray(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    pil_b = Image.fromarray(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    
    hash_a = imagehash.phash(pil_a)
    hash_b = imagehash.phash(pil_b)
    phash_dist = hash_a - hash_b
    
    # Fast path: exact pHash match
    if phash_dist <= config.TILE_PHASH_MAX:
        return {
            'match': True,
            'ssim': 1.0,
            'phash_dist': int(phash_dist),
            'method': 'Tile-Exact'
        }
    
    # Verify with SSIM
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    
    # Resize if needed
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    
    ssim_val = ssim_func(gray_a, gray_b)
    
    if ssim_val >= config.TILE_SSIM_MIN:
        return {
            'match': True,
            'ssim': float(ssim_val),
            'phash_dist': int(phash_dist),
            'method': 'Tile-SSIM'
        }
    
    return {
        'match': False,
        'ssim': float(ssim_val),
        'phash_dist': int(phash_dist),
        'method': None
    }


def aggregate_tile_matches(tiles: List[Dict], verified_pairs: List[Dict], config: TileFirstConfig) -> pd.DataFrame:
    """
    Aggregate tile-level matches to panel-level pairs.
    
    Returns DataFrame with columns:
    - Image_A, Image_B: Panel paths
    - Matched_Tiles: Number of matching tiles
    - Tier: 'A', 'B', or None
    - Tier_A_Tiles: List of Tier-A quality tiles
    - Matched_Positions: List of tile IDs that matched
    - Best_SSIM, Best_pHash: Best match quality
    - Extraction_Method: Always "micro"
    """
    # Group by panel pair
    panel_matches = {}
    
    for pair in verified_pairs:
        tile_a = tiles[pair['tile_idx_a']]
        tile_b = tiles[pair['tile_idx_b']]
        panel_a = tile_a['image_path']
        panel_b = tile_b['image_path']
        pair_key = tuple(sorted([panel_a, panel_b]))
        
        if pair_key not in panel_matches:
            panel_matches[pair_key] = []
        
        panel_matches[pair_key].append({
            'tile_id_a': tile_a['tile_id'],
            'tile_id_b': tile_b['tile_id'],
            'ssim': pair['ssim'],
            'phash_dist': pair['phash_dist'],
            'method': pair['method']
        })
    
    # Build DataFrame
    rows = []
    for (panel_a, panel_b), matches in panel_matches.items():
        # Count Tier-A quality tiles
        tier_a_tiles = [m for m in matches 
                       if m['ssim'] >= config.TIER_A_SSIM and m['phash_dist'] <= config.TIER_A_PHASH]
        
        # Determine tier
        if len(tier_a_tiles) >= config.TIER_A_MIN_TILES:
            tier = 'A'
        elif len(matches) >= 1:
            tier = 'B'
        else:
            tier = None
        
        # Best match quality
        best_ssim = max(m['ssim'] for m in matches)
        best_phash = min(m['phash_dist'] for m in matches)
        
        # Matched positions (tile IDs)
        matched_positions = ', '.join([m['tile_id_a'].split('_')[-1] for m in tier_a_tiles[:5]])
        
        rows.append({
            'Image_A': Path(panel_a).name,
            'Image_B': Path(panel_b).name,
            'Path_A': panel_a,
            'Path_B': panel_b,
            'Matched_Tiles': len(matches),
            'Tier_A_Tiles': len(tier_a_tiles),
            'Tier': tier,
            'Best_SSIM': best_ssim,
            'Best_pHash': best_phash,
            'Matched_Positions': matched_positions,
            'Extraction_Method': 'micro',  # Always micro (NO GRID)
            'Tier_Path': 'Tile-Micro'
        })
    
    return pd.DataFrame(rows)


def run_tile_first_pipeline(panel_paths: List[str], clip_model, preprocess, device, config: TileFirstConfig = None) -> pd.DataFrame:
    """
    Main tile-first pipeline: micro-tiles ONLY (NO GRID).
    
    1. Extract micro-tiles from all panels
    2. Compute CLIP embeddings per tile
    3. Find candidate tile pairs
    4. Verify tile pairs (SSIM + pHash)
    5. Aggregate to panel-level pairs
    
    Returns DataFrame with panel-level results.
    """
    if config is None:
        config = TileFirstConfig()
    
    print("\n" + "="*70)
    print("🔬 TILE-FIRST PIPELINE: MICRO-TILES ONLY (NO GRID DETECTION)")
    print("="*70)
    print(f"  Tile size: {config.MICRO_TILE_SIZE}×{config.MICRO_TILE_SIZE}")
    print(f"  Tile overlap: {int((1-config.MICRO_TILE_STRIDE)*100)}%")
    print(f"  CLIP threshold: {config.TILE_CLIP_MIN}")
    print(f"  SSIM threshold: {config.TILE_SSIM_MIN}")
    print(f"  pHash max distance: {config.TILE_PHASH_MAX}")
    print("="*70)
    
    # Phase 1: Extract micro-tiles
    print(f"\n[Phase 1] Extracting micro-tiles from {len(panel_paths)} panels...")
    all_tiles = []
    for panel_path in tqdm(panel_paths, desc="Extracting tiles"):
        tiles = extract_micro_tiles(panel_path, config)
        all_tiles.extend(tiles)
    
    print(f"  ✓ Extracted {len(all_tiles)} tiles from {len(panel_paths)} panels")
    print(f"  ✓ Avg tiles per panel: {len(all_tiles) / len(panel_paths):.1f}")
    
    if len(all_tiles) == 0:
        print("  ⚠️  No tiles extracted!")
        return pd.DataFrame()
    
    # Phase 2: Compute CLIP embeddings
    print(f"\n[Phase 2] Computing CLIP embeddings...")
    tile_embeddings = compute_tile_embeddings(all_tiles, clip_model, preprocess, device)
    print(f"  ✓ Computed {len(tile_embeddings)} embeddings")
    
    # Phase 3: Find candidates
    print(f"\n[Phase 3] Finding candidate tile pairs...")
    candidates = find_tile_candidates(all_tiles, tile_embeddings, config)
    
    if len(candidates) == 0:
        print("  ⚠️  No candidates found!")
        return pd.DataFrame()
    
    # Phase 4: Verify tile pairs
    print(f"\n[Phase 4] Verifying {len(candidates)} tile pairs...")
    verified_pairs = []
    for tile_idx_a, tile_idx_b, clip_sim in tqdm(candidates, desc="Verifying tiles"):
        result = verify_tile_pair(all_tiles[tile_idx_a], all_tiles[tile_idx_b], config)
        if result['match']:
            verified_pairs.append({
                'tile_idx_a': tile_idx_a,
                'tile_idx_b': tile_idx_b,
                'clip_sim': clip_sim,
                **result
            })
    
    print(f"  ✓ Verified {len(verified_pairs)} tile matches")
    
    # Phase 5: Aggregate to panel pairs
    print(f"\n[Phase 5] Aggregating to panel-level pairs...")
    df_results = aggregate_tile_matches(all_tiles, verified_pairs, config)
    
    if len(df_results) > 0:
        print(f"  ✓ Found {len(df_results)} panel pairs")
        print(f"    • Tier A: {len(df_results[df_results['Tier'] == 'A'])}")
        print(f"    • Tier B: {len(df_results[df_results['Tier'] == 'B'])}")
    
    return df_results

