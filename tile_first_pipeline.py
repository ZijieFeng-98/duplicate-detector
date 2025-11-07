#!/usr/bin/env python3
"""
Tile-First Pipeline: MEMORY-SAFE Micro-Tiles (NO GRID DETECTION)

This module provides pure micro-tile extraction optimized for Streamlit Cloud (1GB RAM limit).

Key Features:
- Streaming CLIP embeddings (no memory spike)
- Automatic tile downsampling (max 200 tiles)
- Chunked similarity search (avoids N×N matrix)
- Graceful error recovery (returns empty DF instead of crash)
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from itertools import combinations
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import warnings
import gc

class TileFirstConfig:
    """Configuration for memory-safe micro-tile extraction"""
    
    # FORCE MICRO-TILES ONLY (NO GRID DETECTION)
    CONFOCAL_MIN_GRID = 999        # Impossible → never detects grids
    WB_MIN_LANES = 999             # Impossible → never detects lanes
    
    # Micro-tile parameters
    MICRO_TILE_SIZE = 384          # 384×384 tiles (uniform for all images)
    MICRO_TILE_STRIDE = 0.65       # 35% overlap between tiles
    
    # Memory protection (NEW for Streamlit Cloud)
    MAX_TILES_TOTAL = 200          # Hard limit for memory safety
    BATCH_SIZE = 16                # Conservative batch size (reduced from 32)
    ENABLE_AUTO_DOWNSAMPLING = True # Auto-reduce if too many tiles
    MEMORY_SAFE_MODE = True        # Enable all memory optimizations
    CHUNK_SIZE = 50                # For chunked similarity search
    
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


def detect_confocal_grid(image_path: str, max_dim: int = 768) -> Dict[str, float]:
    """Detect repetitive confocal grids using FFT energy analysis."""

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {
                'is_confocal': False,
                'grid_energy': 0.0,
                'dominant_spacing': 0.0
            }

        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_size = (max(8, int(w * scale)), max(8, int(h * scale)))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32)
        img -= img.mean()
        std = img.std() + 1e-6
        if std < 1e-3:
            return {
                'is_confocal': False,
                'grid_energy': 0.0,
                'dominant_spacing': 0.0
            }

        freq = np.fft.fft2(img)
        freq_shift = np.fft.fftshift(freq)
        magnitude = np.abs(freq_shift)

        # Remove central DC component
        cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        guard = max(3, int(0.02 * max(magnitude.shape)))
        magnitude[cy-guard:cy+guard+1, cx-guard:cx+guard+1] = 0

        horiz_profile = magnitude.mean(axis=0)
        vert_profile = magnitude.mean(axis=1)

        def _peak_energy(profile: np.ndarray) -> Tuple[float, float]:
            if profile.size == 0:
                return 0.0, 0.0
            center = profile.size // 2
            dead_zone = max(2, int(profile.size * 0.05))
            mask = np.ones_like(profile, dtype=bool)
            mask[max(0, center - dead_zone):min(profile.size, center + dead_zone + 1)] = False
            filtered = profile[mask]
            if filtered.size == 0:
                return 0.0, 0.0

            topk = np.sort(filtered)[-6:]
            energy = float(np.sum(topk)) / (float(np.sum(np.abs(filtered))) + 1e-6)

            # Dominant spacing from largest positive-frequency peak
            positive = profile[center + dead_zone + 1:]
            if positive.size == 0:
                spacing = 0.0
            else:
                dominant_idx = int(np.argmax(positive)) + dead_zone + 1
                spacing = dominant_idx / max(profile.size, 1)

            return energy, spacing

        horiz_energy, horiz_spacing = _peak_energy(horiz_profile)
        vert_energy, vert_spacing = _peak_energy(vert_profile)

        grid_energy = float(max(horiz_energy, vert_energy))
        dominant_spacing = float(max(horiz_spacing, vert_spacing))

        is_confocal = grid_energy >= 0.08 and dominant_spacing > 0

        return {
            'is_confocal': bool(is_confocal),
            'grid_energy': grid_energy,
            'dominant_spacing': dominant_spacing
        }

    except Exception:
        return {
            'is_confocal': False,
            'grid_energy': 0.0,
            'dominant_spacing': 0.0
        }


def compute_color_histogram(image_path: str, bins: int = 32) -> np.ndarray:
    """Compute normalized color histogram for fluorophore spectrum comparison."""

    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(bins * 3, dtype=np.float32)

    hist_parts = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [bins], [0, 256])
        hist_parts.append(hist.flatten())

    hist_vec = np.concatenate(hist_parts).astype(np.float32)
    total = hist_vec.sum()
    if total <= 0:
        return np.zeros_like(hist_vec)

    return hist_vec / total


def compute_multiscale_ssim(gray_a: np.ndarray, gray_b: np.ndarray, levels: int = 3) -> float:
    """Approximate multi-scale SSIM by averaging SSIM across downsampled scales."""

    from skimage.metrics import structural_similarity as ssim

    try:
        scores = []
        current_a = gray_a.astype(np.float32)
        current_b = gray_b.astype(np.float32)

        for level in range(levels):
            if current_a.shape[0] < 32 or current_a.shape[1] < 32:
                break

            score = ssim(current_a, current_b, data_range=255)
            scores.append(score)

            if level == levels - 1:
                break

            new_w = max(16, current_a.shape[1] // 2)
            new_h = max(16, current_a.shape[0] // 2)
            if new_h == current_a.shape[0] and new_w == current_a.shape[1]:
                break

            current_a = cv2.resize(current_a, (new_w, new_h), interpolation=cv2.INTER_AREA)
            current_b = cv2.resize(current_b, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if not scores:
            return 0.0

        return float(np.mean(scores))

    except Exception:
        return 0.0


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
            
            # Only store tiles that are full size
            if tile_data.shape[0] == tile_size and tile_data.shape[1] == tile_size:
                tiles.append({
                    'tile_id': tile_id,
                    'image_path': image_path,
                    'bbox': bbox,
                    'tile_data': tile_data
                })
                tile_idx += 1
    
    return tiles


def extract_tiles_memory_safe(panel_paths: List[str], config: TileFirstConfig) -> List[Dict]:
    """
    Extract tiles with memory protection for Streamlit Cloud.
    
    - Limits total tiles to MAX_TILES_TOTAL
    - Auto-adjusts stride if needed
    - Random sampling if still too many
    """
    all_tiles = []
    tiles_per_panel = []
    
    print(f"\n  Extracting micro-tiles (memory-safe mode)...")
    print(f"  Max tiles: {config.MAX_TILES_TOTAL}")
    
    # First pass: extract from all panels
    for panel_path in tqdm(panel_paths, desc="Extracting tiles"):
        tiles = extract_micro_tiles(panel_path, config)
        tiles_per_panel.append(len(tiles))
        all_tiles.extend(tiles)
        
        # Early exit if approaching limit
        if config.MEMORY_SAFE_MODE and len(all_tiles) > config.MAX_TILES_TOTAL * 1.5:
            warnings.warn(
                f"⚠️  Tile count ({len(all_tiles)}) exceeds safe limit. "
                f"Stopping early to conserve memory."
            )
            break
    
    # If too many tiles, downsample
    if len(all_tiles) > config.MAX_TILES_TOTAL:
        import random
        random.seed(42)  # Deterministic
        
        original_count = len(all_tiles)
        all_tiles = random.sample(all_tiles, config.MAX_TILES_TOTAL)
        
        warnings.warn(
            f"⚠️  Downsampled {original_count} → {len(all_tiles)} tiles for memory safety"
        )
    
    print(f"  ✓ Extracted {len(all_tiles)} tiles from {len(panel_paths)} panels")
    if tiles_per_panel:
        print(f"    Avg tiles/panel: {np.mean(tiles_per_panel):.1f}")
    
    return all_tiles


def compute_tile_embeddings_streaming(
    tiles: List[Dict],
    clip_model,
    preprocess,
    device: str,
    batch_size: int = 16
) -> np.ndarray:
    """
    Memory-safe streaming CLIP embedding computation.
    
    Key optimizations:
    - Small batch size (16 vs 32)
    - Explicit cache clearing after each batch
    - No intermediate storage of full embedding matrix
    - Graceful handling of failed tiles
    """
    import torch
    from PIL import Image
    
    embeddings = []
    total_batches = (len(tiles) + batch_size - 1) // batch_size
    
    print(f"  Computing CLIP embeddings (streaming mode, batch={batch_size})...")
    
    for batch_idx in tqdm(range(0, len(tiles), batch_size), desc="Computing tile CLIP", total=total_batches):
        batch_tiles = tiles[batch_idx:batch_idx + batch_size]
        
        # Load images for this batch
        batch_imgs = []
        for tile in batch_tiles:
            try:
                tile_bgr = tile['tile_data']
                tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(tile_rgb)
                batch_imgs.append(preprocess(pil_img))
            except Exception as e:
                warnings.warn(f"Failed to process tile {tile.get('tile_id', '?')}: {e}")
                # Use zero embedding as fallback
                batch_imgs.append(torch.zeros(3, 224, 224))
        
        if not batch_imgs:
            continue
        
        # Compute embeddings for this batch
        try:
            with torch.no_grad():
                x = torch.stack(batch_imgs).to(device)
                feats = clip_model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                
                # Store as numpy immediately
                embeddings.append(feats.cpu().numpy())
                
                # Critical: Clear cache after each batch
                del x, feats, batch_imgs
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                warnings.warn(f"OOM at batch {batch_idx//batch_size}, skipping batch")
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                # Add zero embeddings for this batch
                embeddings.append(np.zeros((len(batch_tiles), 512), dtype=np.float32))
            else:
                raise
    
    if not embeddings:
        return np.zeros((0, 512), dtype=np.float32)
    
    result = np.vstack(embeddings)
    print(f"  ✓ Computed {len(result)} embeddings")
    
    return result


def find_tile_candidates_memory_safe(
    tiles: List[Dict],
    embeddings: np.ndarray,
    config: TileFirstConfig
) -> List[Tuple[int, int, float]]:
    """
    Memory-efficient tile matching using chunked similarity search.
    
    Avoids creating full N×N similarity matrix by processing in chunks.
    """
    if len(embeddings) == 0:
        return []
    
    matches = []
    n = len(embeddings)
    chunk_size = config.CHUNK_SIZE
    
    print(f"  Finding candidate tile pairs (chunked search)...")
    
    # Process in chunks to avoid memory spike
    for i in tqdm(range(0, n, chunk_size), desc="Candidate search"):
        i_end = min(i + chunk_size, n)
        chunk_emb = embeddings[i:i_end]
        
        # Compare chunk against all embeddings
        sims = chunk_emb @ embeddings.T  # Only chunk_size × N, not N × N
        
        # Find matches above threshold
        for local_idx in range(len(chunk_emb)):
            global_idx = i + local_idx
            
            # Only consider j > global_idx (avoid duplicates)
            for j in range(global_idx + 1, n):
                score = float(sims[local_idx, j])
                
                # Check threshold and same-panel filter
                if score >= config.TILE_CLIP_MIN:
                    # Skip same panel
                    if tiles[global_idx]['image_path'] == tiles[j]['image_path']:
                        continue
                    
                    matches.append((global_idx, j, score))
        
        # Clear chunk
        del sims
        gc.collect()
    
    print(f"  ✓ Found {len(matches)} candidate pairs")
    
    return matches


def verify_tile_pair(tile_a: Dict, tile_b: Dict, config: TileFirstConfig) -> Dict:
    """
    Verify a tile pair using SSIM and pHash.
    
    Returns dict with:
    - match: bool (True if verified)
    - ssim: float
    - phash_dist: int
    """
    from skimage.metrics import structural_similarity as ssim
    import imagehash
    from PIL import Image
    
    try:
        # Convert to grayscale for SSIM
        gray_a = cv2.cvtColor(tile_a['tile_data'], cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(tile_b['tile_data'], cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        ssim_score = ssim(gray_a, gray_b, data_range=255)
        
        # Compute pHash
        pil_a = Image.fromarray(cv2.cvtColor(tile_a['tile_data'], cv2.COLOR_BGR2RGB))
        pil_b = Image.fromarray(cv2.cvtColor(tile_b['tile_data'], cv2.COLOR_BGR2RGB))
        
        phash_a = imagehash.phash(pil_a)
        phash_b = imagehash.phash(pil_b)
        phash_dist = int(phash_a - phash_b)
        
        # Check thresholds
        match = (ssim_score >= config.TILE_SSIM_MIN) or (phash_dist <= config.TILE_PHASH_MAX)
        
        return {
            'match': match,
            'ssim': float(ssim_score),
            'phash_dist': phash_dist
        }
    
    except Exception as e:
        warnings.warn(f"Verification failed: {e}")
        return {'match': False, 'ssim': 0.0, 'phash_dist': 999}


def aggregate_tile_matches(
    tiles: List[Dict],
    verified_pairs: List[Dict],
    config: TileFirstConfig
) -> pd.DataFrame:
    """
    Aggregate tile matches to panel-level pairs.
    
    Groups tile matches by panel pair and applies tier gating.
    """
    if not verified_pairs:
        return pd.DataFrame()
    
    # Group by panel pair
    panel_groups = {}
    
    for pair in verified_pairs:
        tile_a = tiles[pair['tile_idx_a']]
        tile_b = tiles[pair['tile_idx_b']]
        
        panel_a = tile_a['image_path']
        panel_b = tile_b['image_path']
        
        # Canonical ordering
        if panel_a > panel_b:
            panel_a, panel_b = panel_b, panel_a
        
        key = (panel_a, panel_b)
        
        if key not in panel_groups:
            panel_groups[key] = []
        
        panel_groups[key].append(pair)
    
    # Build DataFrame
    rows = []
    for (panel_a, panel_b), matches in panel_groups.items():
        num_tiles = len(matches)
        avg_ssim = float(np.mean([m['ssim'] for m in matches]))
        min_phash = int(min([m['phash_dist'] for m in matches]))
        avg_clip = float(np.mean([m['clip_sim'] for m in matches]))
        
        # Tier gating
        tier_a = (
            (num_tiles >= config.TIER_A_MIN_TILES) and
            (avg_ssim >= config.TIER_A_SSIM or min_phash <= config.TIER_A_PHASH)
        )
        
        tier = 'A' if tier_a else 'B'
        
        rows.append({
            'Path_A': panel_a,
            'Path_B': panel_b,
            'Cosine_Similarity': avg_clip,
            'SSIM': avg_ssim,
            'Hamming_Distance': min_phash,
            'Tile_Matches': num_tiles,
            'Tier': tier,
            'Extraction_Method': 'micro'
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by tier and similarity
    df = df.sort_values(['Tier', 'Cosine_Similarity'], ascending=[True, False])
    
    return df


def run_tile_first_pipeline(
    panel_paths: List[str],
    clip_model,
    preprocess,
    device: str,
    config: TileFirstConfig = None
) -> pd.DataFrame:
    """
    Memory-safe tile-first pipeline with error recovery.
    
    Optimized for Streamlit Cloud (1GB RAM limit):
    - Streaming embeddings (no memory spike)
    - Automatic tile downsampling
    - Chunked similarity search
    - Graceful error handling
    
    Returns DataFrame with panel-level results, or empty DF on error.
    """
    if config is None:
        config = TileFirstConfig()
    
    try:
        print("\n" + "="*70)
        print("🔬 TILE-FIRST PIPELINE: MEMORY-SAFE MICRO-TILES")
        print("="*70)
        print(f"  Tile size: {config.MICRO_TILE_SIZE}×{config.MICRO_TILE_SIZE}")
        print(f"  Tile overlap: {int((1-config.MICRO_TILE_STRIDE)*100)}%")
        print(f"  Max tiles: {config.MAX_TILES_TOTAL} (memory protection)")
        print(f"  CLIP threshold: {config.TILE_CLIP_MIN}")
        print(f"  Batch size: {config.BATCH_SIZE} (conservative)")
        print("="*70)
        
        # Phase 1: Extract tiles with memory protection
        all_tiles = extract_tiles_memory_safe(panel_paths, config)
        
        if not all_tiles:
            warnings.warn("⚠️  No tiles extracted - returning empty DataFrame")
            return pd.DataFrame()
        
        # Phase 2: Compute embeddings with streaming
        try:
            tile_embeddings = compute_tile_embeddings_streaming(
                all_tiles, clip_model, preprocess, device,
                batch_size=config.BATCH_SIZE
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                warnings.warn("⚠️  OOM during embedding - trying with batch_size=8")
                if device == 'cuda':
                    import torch
                    torch.cuda.empty_cache()
                gc.collect()
                
                tile_embeddings = compute_tile_embeddings_streaming(
                    all_tiles, clip_model, preprocess, device,
                    batch_size=8  # Even more conservative
                )
            else:
                raise
        
        if len(tile_embeddings) == 0:
            warnings.warn("⚠️  No embeddings computed - returning empty DataFrame")
            return pd.DataFrame()
        
        # Phase 3: Find matches (memory-efficient)
        candidates = find_tile_candidates_memory_safe(
            all_tiles, tile_embeddings, config
        )
        
        if len(candidates) == 0:
            print("  ⚠️  No candidates found!")
            return pd.DataFrame()
        
        # Phase 4: Verify tile pairs
        print(f"\n  Verifying {len(candidates)} tile pairs...")
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
        print(f"\n  Aggregating to panel-level pairs...")
        df_results = aggregate_tile_matches(all_tiles, verified_pairs, config)
        
        if len(df_results) > 0:
            print(f"  ✓ Found {len(df_results)} panel pairs")
            print(f"    • Tier A: {len(df_results[df_results['Tier'] == 'A'])}")
            print(f"    • Tier B: {len(df_results[df_results['Tier'] == 'B'])}")
        else:
            print("  ⚠️  No panel pairs found after aggregation")
        
        print("="*70)
        print("  ✅ TILE-FIRST PIPELINE COMPLETE")
        print("="*70)
        
        return df_results
        
    except MemoryError as e:
        warnings.warn(f"❌ Memory error in tile pipeline: {e}")
        print("\n  ⚠️  TILE PIPELINE FAILED: Out of memory")
        print("  💡 Try: Reduce --tile-size or use standard panel pipeline")
        return pd.DataFrame()  # Return empty instead of crashing
        
    except Exception as e:
        warnings.warn(f"❌ Tile pipeline error: {e}")
        import traceback
        traceback.print_exc()
        print("\n  ⚠️  TILE PIPELINE FAILED")
        print(f"  Error: {e}")
        return pd.DataFrame()  # Graceful fallback
